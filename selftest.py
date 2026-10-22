"""Equivalence test: our G0 pipeline vs the official openvla LIBERO evaluation, stage by stage.

The `ref_*` functions are copied from openvla/openvla (MIT) experiments/robot/{libero/libero_utils.py,
openvla_utils.py, robot_utils.py} and must not be edited. Every stage is compared on the same input and the
maximum difference is printed. The rollout is driven by the REFERENCE action, so a pass also reproduces the
official evaluation path end to end (success count at the bottom).

Stages
    language    ours (scene.language_from_filename) vs libero.libero.benchmark.grab_language_from_filename
    init        ours (numpy-only unpickler) vs torch.load(weights_only=False)
    image       ours (scene.to_obs(scene.render(ego))) vs env obs["agentview_image"], every step
    preprocess  ours (policy.preprocess_image) vs ref get_libero_image + get_vla_action crop          [needs tf]
    action      ours (OpenVLAPolicy.act) vs ref vla.predict_action(**processor(prompt, image))        [needs model]
    teacher     argmax of ours teacher-forced logprobs == our generated tokens                     [needs model]
    env_action  ours (policy.env_action) vs ref normalize_gripper_action + invert_gripper_action

Usage (server): python selftest.py --occ_root ... --libero_root ... --device cuda:3
Usage (no GPU): python selftest.py --occ_root ... --libero_root ... --no_model
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image

from policy import EMPTY_TOKEN, env_action, preprocess_image
from scene import MAX_STEPS, Scene, dummy_action, language_from_filename, list_tasks, load_init_states, make_env


# ----------------------------------------------------------------------------- reference (openvla, verbatim logic)

def ref_resize_image(img, resize_size, tf):
    assert isinstance(resize_size, tuple)
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()


def ref_get_libero_image(obs, resize_size, tf):
    img = obs["agentview_image"]
    img = img[::-1, ::-1]
    return ref_resize_image(img, (resize_size, resize_size), tf)


def ref_crop_and_resize(image, crop_scale, batch_size, tf):
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack([height_offsets, width_offsets, height_offsets + new_heights,
                               width_offsets + new_widths], axis=1)
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))
    if expanded_dims:
        image = image[0]
    return image


def ref_vla_image(full_image, tf):
    """get_vla_action up to the processor call (center_crop=True)."""
    image = Image.fromarray(full_image).convert("RGB")
    image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = ref_crop_and_resize(image, 0.9, 1, tf)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
    return Image.fromarray(image.numpy()).convert("RGB")


def ref_postprocess(action):
    action = np.array(action, dtype=float)
    action[..., -1] = 2 * (action[..., -1] - 0.0) / (1.0 - 0.0) - 1
    action[..., -1] = np.sign(action[..., -1])
    action[..., -1] = action[..., -1] * -1.0
    return action


def ref_grab_language_from_filename(x):
    if x[0].isupper():
        if "SCENE10" in x:
            language = " ".join(x[x.find("SCENE") + 8:].split("_"))
        else:
            language = " ".join(x[x.find("SCENE") + 7:].split("_"))
    else:
        language = " ".join(x.split("_"))
    en = language.find(".bddl")
    return language[:en]


# ----------------------------------------------------------------------------- checks

class Report:
    def __init__(self):
        self.rows = []

    def add(self, stage, value, ok, note=""):
        self.rows.append((stage, value, ok, note))
        print(f"  [{'PASS' if ok else 'FAIL'}] {stage:12s} {value}  {note}", flush=True)

    def summary(self):
        bad = [r for r in self.rows if not r[2]]
        print("\n==== selftest summary ====")
        for stage in dict.fromkeys(r[0] for r in self.rows):
            rs = [r for r in self.rows if r[0] == stage]
            print(f"  {stage:12s} {sum(r[2] for r in rs)}/{len(rs)} pass")
        print("ALL PASS" if not bad else f"{len(bad)} FAIL")
        return not bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--occ_root", required=True)
    p.add_argument("--libero_root", required=True)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--n_tasks", type=int, default=2)
    p.add_argument("--n_init", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=0, help="0 = suite max (220 for spatial)")
    p.add_argument("--check_every", type=int, default=10, help="model checks every k steps (image check every step)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--no_model", action="store_true")
    a = p.parse_args()
    rep = Report()

    # language: every task file of every suite, ours vs reference (and vs LIBERO's own if importable)
    try:
        from libero.libero.benchmark import grab_language_from_filename as libero_lang
    except Exception:
        libero_lang = None
    names = [os.path.basename(f) for f in glob.glob(os.path.join(a.occ_root, "benchmark_assets", "bddl_files", "*", "*.bddl"))]
    bad = [n for n in names if language_from_filename(n) != ref_grab_language_from_filename(n)
           or (libero_lang and language_from_filename(n) != libero_lang(n))]
    rep.add("language", f"{len(names) - len(bad)}/{len(names)} equal", not bad,
            f"(vs LIBERO import: {'yes' if libero_lang else 'no, copy only'}) {bad[:2]}")

    tasks = [t for t in list_tasks(a.occ_root, a.libero_root, [a.suite], original=True) if t["skip_reason"] is None]
    tasks = tasks[: a.n_tasks]

    # init states: ours vs torch.load
    try:
        import torch
        for t in tasks:
            ours = load_init_states(t["init"])
            ref = np.asarray(torch.load(t["init"], weights_only=False))
            rep.add("init", f"shape {ours.shape} maxdiff {np.abs(ours - ref).max():.1e}",
                    ours.shape == ref.shape and np.array_equal(ours, ref), t["name"][:40])
    except ImportError:
        print("  [skip] init: torch not installed")

    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        tf = None
        print("  [skip] preprocess: tensorflow not installed")

    policy = None
    if not a.no_model:
        from policy import OpenVLAPolicy
        import torch
        policy = OpenVLAPolicy(f"openvla/openvla-7b-finetuned-libero-{a.suite.split('_')[1]}", a.suite, a.device, a.attn)
        rep.add("unnorm_key", policy.unnorm_key, True, f"preproc={policy.preproc}")

    successes = 0
    n_ep = 0
    for t in tasks:
        env = make_env(t["bddl"], 256)
        inits = load_init_states(t["init"])
        scene = Scene(env, [], t["targets"], 256)
        print(f"[{t['name'][:60]}] language=\"{t['language']}\"", flush=True)
        for ep in range(min(a.n_init, len(inits))):
            scene.close()
            env.reset()
            scene = Scene(env, [], t["targets"], 256)
            obs = env.set_init_state(inits[ep])
            scene.calibrate_orientation(obs["agentview_image"])
            for _ in range(10):
                obs, _, _, _ = env.step(dummy_action())
            img_diffs, img_mean, done = [], [], False
            for step in range(a.max_steps or MAX_STEPS[a.suite]):
                ours_img = scene.to_obs(scene.render(scene.ego))
                dd = np.abs(ours_img.astype(int) - obs["agentview_image"].astype(int))
                img_diffs.append(int(dd.max()))
                img_mean.append(float(dd.mean()))
                check = step % a.check_every == 0
                ref_full = ref_get_libero_image(obs, 224, tf) if tf is not None else None
                if tf is not None and check:
                    ref_pil = ref_vla_image(ref_full, tf)
                    our_pil = preprocess_image(obs["agentview_image"], True, tf)
                    d = int(np.abs(np.asarray(ref_pil, int) - np.asarray(our_pil, int)).max())
                    rep.add("preprocess", f"maxdiff {d}", d == 0, f"ep{ep} t={step}")
                if policy is not None:
                    prompt = f"In: What action should the robot take to {t['language'].lower()}?\nOut:"
                    inputs = policy.processor(prompt, ref_vla_image(ref_full, tf)).to(a.device, dtype=torch.bfloat16)
                    with torch.inference_mode():
                        ref_act = policy.vla.predict_action(**inputs, unnorm_key=policy.unnorm_key, do_sample=False)
                    if check:
                        tok, our_act = policy.act(obs["agentview_image"], t["language"])
                        d = float(np.abs(our_act - ref_act).max())
                        rep.add("action", f"maxdiff {d:.2e}", d < 1e-6, f"ep{ep} t={step}")
                        lp = policy.logprobs(obs["agentview_image"], t["language"], tok)
                        ok = bool((lp.argmax(-1) == policy.bin_of(tok)).all())
                        rep.add("teacher", f"argmax==tokens {ok}", ok, f"ep{ep} t={step}")
                else:
                    ref_act = np.zeros(7)
                    ref_act[-1] = 0.0  # open
                e_ours, e_ref = env_action(ref_act), ref_postprocess(ref_act)
                if check:
                    rep.add("env_action", f"maxdiff {np.abs(e_ours - e_ref).max():.1e}", np.array_equal(e_ours, e_ref),
                            f"ep{ep} t={step}")
                obs, _, done, _ = env.step(e_ref.tolist())
                if done:
                    break
            # Not required to be 0: robosuite samples the camera inside the control step, we render after it, so a
            # moving arm differs by a few pixels. The policy-driving path uses the env image (probe --drive occ).
            mean_d = float(np.mean(img_mean))
            rep.add("image", f"mean abs diff {mean_d:.3f}, max {max(img_diffs)} over {len(img_diffs)} steps",
                    mean_d < 0.5, f"ep{ep} (info: timing, see comment)")
            successes += bool(done)
            n_ep += 1
            print(f"  ep{ep} success={done} steps={step + 1}", flush=True)
        scene.close()
        env.close()
    if policy is not None:
        print(f"\nreference-path success {successes}/{n_ep} (openvla reports ~85% on libero_spatial)")
    rep.summary()


if __name__ == "__main__":
    main()
