"""Frozen OpenVLA (LIBERO fine-tuned) with the two calls G0 needs.

- act(img):            greedy 7-token action, exactly the LIBERO eval path of openvla/openvla
- logprobs(img, tok):  teacher-forced log-probs over the 256 action bins for given action tokens,
                       so every condition is scored on the *teacher's* token prefix

Preprocessing copies experiments/robot/libero (rotate 180, JPEG round-trip, lanczos resize to 224,
center crop 0.9 area). TensorFlow is used when importable so the pixels match the eval path;
otherwise a PIL approximation is used and `self.preproc` says so (record it with the results).
"""
import json
import os

import numpy as np
from PIL import Image

N_BINS = 256
ACTION_DIM = 7
EMPTY_TOKEN = 29871  # appended after "Out:" at training time (see predict_action)


def _tf_or_none():
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")  # preprocessing only; keep TF off the GPU
        return tf
    except Exception:
        return None


def preprocess_image(img_obs, center_crop=True, tf=None):
    """img_obs: HxWx3 uint8 in the env obs['agentview_image'] convention -> PIL image the processor gets.
    Op-for-op the openvla path (get_libero_image -> resize_image, then get_vla_action's center crop), including
    doing the crop-box arithmetic in tf float32: computing it in float64 changed pixels by 1 (selftest.py)."""
    img = img_obs[::-1, ::-1]  # "rotate 180 degrees to match train preprocessing"
    if tf is not None:
        x = tf.image.encode_jpeg(img)
        x = tf.io.decode_image(x, expand_animations=False, dtype=tf.uint8)
        x = tf.image.resize(x, (224, 224), method="lanczos3", antialias=True)
        x = tf.cast(tf.clip_by_value(tf.round(x), 0, 255), tf.uint8).numpy()
        image = Image.fromarray(x).convert("RGB")
        if center_crop:
            image = tf.convert_to_tensor(np.array(image))
            orig_dtype = image.dtype
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.expand_dims(image, axis=0)
            h = tf.reshape(tf.clip_by_value(tf.sqrt(0.9), 0, 1), shape=(1,))
            w = tf.reshape(tf.clip_by_value(tf.sqrt(0.9), 0, 1), shape=(1,))
            boxes = tf.stack([(1 - h) / 2, (1 - w) / 2, (1 - h) / 2 + h, (1 - w) / 2 + w], axis=1)
            image = tf.image.crop_and_resize(image, boxes, tf.range(1), (224, 224))[0]
            image = tf.clip_by_value(image, 0, 1)
            image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
            image = Image.fromarray(image.numpy()).convert("RGB")
        return image
    img = np.ascontiguousarray(img)
    buf = __import__("io").BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=95)
    x = Image.open(buf).convert("RGB").resize((224, 224), Image.LANCZOS)
    if center_crop:
        s = np.sqrt(0.9) * 224
        o = (224 - s) / 2
        x = x.crop((o, o, o + s, o + s)).resize((224, 224), Image.BILINEAR)
    return x


class OpenVLAPolicy:
    def __init__(self, ckpt, suite, device="cuda:0", attn="sdpa", center_crop=True):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch, self.device, self.center_crop = torch, device, center_crop
        if str(device).startswith("cuda"):
            # Pick the card by index instead of CUDA_VISIBLE_DEVICES: robosuite cross-checks that variable against
            # MUJOCO_EGL_DEVICE_ID. set_device keeps stray CUDA contexts off GPU 0.
            torch.cuda.set_device(device)
        self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            ckpt, attn_implementation=attn, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True,
        ).to(device).eval()
        stats = os.path.join(ckpt, "dataset_statistics.json")
        if os.path.isfile(stats):
            self.vla.norm_stats = json.load(open(stats))
        key = suite
        if key not in self.vla.norm_stats and f"{key}_no_noops" in self.vla.norm_stats:
            key = f"{key}_no_noops"
        assert key in self.vla.norm_stats, f"unnorm key {suite} not in {list(self.vla.norm_stats)}"
        self.unnorm_key = key
        self.vocab = int(self.vla.vocab_size)
        self.bin_lo = self.vocab - N_BINS  # action tokens are the last 256 ids of the base vocab
        self.tf = _tf_or_none()
        self.preproc = "tf" if self.tf is not None else "pil-approx"
        self.name = ckpt

    # ------------------------------------------------------------------ preprocessing
    def preprocess(self, img_obs):
        return preprocess_image(img_obs, self.center_crop, self.tf)

    def _inputs(self, img_obs, language):
        prompt = f"In: What action should the robot take to {language.lower()}?\nOut:"
        inputs = self.processor(prompt, self.preprocess(img_obs)).to(self.device, dtype=self.torch.bfloat16)
        ids = inputs["input_ids"]
        if not bool((ids[:, -1] == EMPTY_TOKEN).all()):
            ids = self.torch.cat([ids, self.torch.full((ids.shape[0], 1), EMPTY_TOKEN, device=ids.device)], 1)
        return ids, inputs["pixel_values"]

    # ------------------------------------------------------------------ decoding
    def tokens_to_action(self, tokens):
        disc = np.clip(self.vocab - np.asarray(tokens) - 1, 0, self.vla.bin_centers.shape[0] - 1)
        norm = self.vla.bin_centers[disc]
        st = self.vla.get_action_stats(self.unnorm_key)
        mask = np.asarray(st.get("mask", np.ones_like(st["q01"], dtype=bool)))
        hi, lo = np.asarray(st["q99"]), np.asarray(st["q01"])
        return np.where(mask, 0.5 * (norm + 1) * (hi - lo) + lo, norm)

    # ------------------------------------------------------------------ calls
    def act(self, img_obs, language):
        with self.torch.inference_mode():
            ids, pv = self._inputs(img_obs, language)
            out = self.vla.generate(ids, attention_mask=self.torch.ones_like(ids), pixel_values=pv,
                                    max_new_tokens=ACTION_DIM, do_sample=False)
        tokens = out[0, -ACTION_DIM:].cpu().numpy()
        return tokens, self.tokens_to_action(tokens)

    def logprobs(self, img_obs, language, tokens):
        """(7, 256) log-probs of each action dim given the prefix `tokens[:k]` (teacher forcing)."""
        torch = self.torch
        with torch.inference_mode():
            ids, pv = self._inputs(img_obs, language)
            tok = torch.as_tensor(np.asarray(tokens), device=ids.device).long()[None]
            full = torch.cat([ids, tok], 1)
            logits = self.vla(input_ids=full, attention_mask=torch.ones_like(full), pixel_values=pv).logits
            # The last ACTION_DIM+1 positions are [EMPTY, a1..a7]; position k predicts a_{k+1}.
            lg = logits[0, -(ACTION_DIM + 1):-1, self.bin_lo:self.vocab].float()
            return torch.log_softmax(lg, -1).cpu().numpy()

    def bin_of(self, tokens):
        return np.asarray(tokens) - self.bin_lo


class DryPolicy:
    """No model. Lets the scene / rendering / logging pipeline run on a machine without the 7B model."""

    preproc, unnorm_key, name = "none", "dry", "dry"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.bin_lo = 0

    def act(self, img_obs, language):
        tokens = self.rng.integers(0, N_BINS, ACTION_DIM)
        a = np.zeros(ACTION_DIM)
        a[-1] = 1.0  # stays open after normalize+invert
        return tokens, a

    def logprobs(self, img_obs, language, tokens):
        return np.full((ACTION_DIM, N_BINS), -np.log(N_BINS))

    def bin_of(self, tokens):
        return np.asarray(tokens)


def env_action(action):
    """openvla LIBERO post-processing: gripper [0,1] -> [-1,1], binarize, then flip sign."""
    a = np.array(action, dtype=float)
    a[-1] = np.sign(2 * a[-1] - 1)
    a[-1] = -a[-1]
    return a


def tie_gap(logp, bins):
    """Largest log-prob gap (nats) between the argmax bin and the generated bin over the 7 dims: 0 when they agree.
    A mismatch with a gap of ~1e-2 or less is a bf16 near-tie between the KV-cached generate path and the
    full forward pass, not a misaligned position (a misalignment gives gaps of several nats)."""
    lp = np.asarray(logp)
    return float((lp.max(-1) - lp[np.arange(len(bins)), np.asarray(bins)]).max())


def kl_per_dim(logp_teacher, logp_other):
    p = np.exp(logp_teacher)
    return (p * (logp_teacher - logp_other)).sum(-1)
