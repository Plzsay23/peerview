"""LIBERO-Occ scene tools: occluder discovery, env construction, camera control, rendering.

폐색 물체는 LIBERO-Occ BDDL 을 원본 LIBERO BDDL 과 비교해 "추가된 물체"로 찾는다.
LIBERO-Occ 는 생성 코드를 공개하지 않았고 언어 지시는 원본과 같다(2026-10-16 33개 전수 확인).

All rendering here goes through one `mujoco.Renderer` so that the four G0 conditions
(occluded / peer / reprojected / clean) share a single image convention. That convention
is mapped onto the env's own `agentview_image` convention by `calibrate_orientation`,
which is checked at runtime instead of assumed.
"""
import io
import os
import pickle
import re
import zipfile

import mujoco
import numpy as np

# Longest training demo + margin, copied from openvla/experiments/robot/libero/run_libero_eval.py.
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
HIDDEN_GROUP = 5  # MjvOption.geomgroup[5] is 0 by default -> geoms moved here are not drawn


# ----------------------------------------------------------------------------- BDDL

def _section(text, name):
    """Body of a top-level `(:name ...)` block, found by paren matching."""
    i = text.find("(:" + name)
    if i < 0:
        return ""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[i + len(name) + 2 : j]
    raise ValueError(f"unbalanced ({name} block")


def bddl_instances(text):
    """Instance names declared in :fixtures and :objects (`a_1 a_2 - type` lines)."""
    names = set()
    for sec in ("fixtures", "objects"):
        for line in _section(text, sec).splitlines():
            if " - " in line:
                names.update(line.split(" - ")[0].split())
    return names


def bddl_language(text):
    return re.search(r"\(:language (.*?)\)\s*$", text, re.M).group(1).strip()


def bddl_obj_of_interest(text):
    return _section(text, "obj_of_interest").split()


def list_tasks(occ_root, libero_root, suites=None, require_init=True, original=False):
    """Every LIBERO-Occ task with its original BDDL, init file and occluder set.

    original=True: same task list, but each task points at the ORIGINAL LIBERO bddl + init file and has no
    occluders. Sanity check of the policy pipeline (OpenVLA reports ~85% on libero_spatial).

    occ_root:    LIBERO-Occ checkout (benchmark_assets/ is read in place, no install step).
    libero_root: upstream LIBERO checkout (for the original BDDL files).
    """
    bddl_dir = os.path.join(occ_root, "benchmark_assets", "bddl_files")
    init_dir = os.path.join(occ_root, "benchmark_assets", "init_files")
    orig_dir = os.path.join(libero_root, "libero", "libero", "bddl_files")
    tasks = []
    for occ_suite in sorted(os.listdir(bddl_dir)):
        suite = occ_suite.replace("_occluded", "")
        if suites and suite not in suites:
            continue
        for fname in sorted(f for f in os.listdir(os.path.join(bddl_dir, occ_suite)) if f.endswith(".bddl")):
            stem = fname[:-5]
            occ_path = os.path.join(bddl_dir, occ_suite, fname)
            orig_path = os.path.join(orig_dir, suite, fname)
            init_path = os.path.join(init_dir, occ_suite, stem + ".pruned_init")
            occ_txt = open(occ_path, encoding="utf-8").read()
            orig_txt = open(orig_path, encoding="utf-8").read()
            t = dict(
                suite=suite,
                name=stem,
                bddl=occ_path,
                init=init_path if os.path.exists(init_path) else None,
                language=bddl_language(occ_txt),
                language_same=bddl_language(occ_txt) == bddl_language(orig_txt),
                occluders=sorted(bddl_instances(occ_txt) - bddl_instances(orig_txt)),
                targets=bddl_obj_of_interest(occ_txt),
            )
            # No added object = occlusion made by moving existing objects; we cannot tell which.
            if original:
                oi = os.path.join(libero_root, "libero", "libero", "init_files", suite, stem + ".pruned_init")
                t.update(bddl=orig_path, init=oi if os.path.exists(oi) else None, occluders=[], original=True)
                t["skip_reason"] = "no init file" if (require_init and t["init"] is None) else None
                tasks.append(t)
                continue
            t["skip_reason"] = (
                "no added occluder" if not t["occluders"]
                else "no init file" if (require_init and t["init"] is None)
                else None
            )
            tasks.append(t)
    return tasks


# ----------------------------------------------------------------------------- init states

_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("_codecs", "encode"),
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """The .pruned_init files are torch.save'd numpy arrays. Load them without torch and
    without letting the pickle import anything but numpy (checked: only these 4 globals)."""

    def find_class(self, module, name):
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"refusing global {module}.{name}")
        if module.startswith("numpy") and module.endswith("multiarray"):
            module = "numpy._core.multiarray" if hasattr(np, "_core") else "numpy.core.multiarray"
        return super().find_class(module, name)


def load_init_states(path):
    with zipfile.ZipFile(path) as z:
        pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
        return np.asarray(_NumpyOnlyUnpickler(io.BytesIO(z.read(pkl))).load())


# ----------------------------------------------------------------------------- env

def make_env(bddl_path, resolution=256, seed=0):
    # Imported lazily: `libero.libero` prompts on stdin when ~/.libero/config.yaml is missing
    # (setup_server.sh writes it).
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)  # openvla: "seed seems to affect object positions even when using fixed initial state"
    return env


def dummy_action():
    return [0, 0, 0, 0, 0, 0, -1]


# ----------------------------------------------------------------------------- geometry helpers

def look_at(pos, target, up=(0.0, 0.0, 1.0)):
    """MuJoCo camera frame (x right, y up, looks along -z) as a world rotation matrix."""
    z = np.asarray(pos, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross((0.0, 1.0, 0.0), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def spherical(center, az_deg, el_deg, dist):
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.asarray(center) + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def to_spherical(pos, center):
    v = np.asarray(pos) - np.asarray(center)
    dist = float(np.linalg.norm(v))
    return float(np.rad2deg(np.arctan2(v[1], v[0]))), float(np.rad2deg(np.arcsin(v[2] / dist))), dist


ORIENTATIONS = {
    "identity": lambda im: im,
    "flipud": lambda im: im[::-1],
    "fliplr": lambda im: im[:, ::-1],
    "rot180": lambda im: im[::-1, ::-1],
}


# ----------------------------------------------------------------------------- scene

class Scene:
    """Rendering access to a LIBERO env's MuJoCo model, with one hijacked spare camera as the peer."""

    def __init__(self, env, occluders, targets, res=256, ego_cam="agentview"):
        sim = env.sim
        self.m = getattr(sim.model, "_model", sim.model)
        self.d = getattr(sim.data, "_data", sim.data)
        self.res = res
        if self.m.vis.global_.offwidth < res or self.m.vis.global_.offheight < res:
            self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, res)
            self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, res)
        self.r = mujoco.Renderer(self.m, height=res, width=res)
        # Match robosuite's offscreen context: collision meshes (group 0) off, visual (group 1) on.
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[0] = 0
        self.opt.geomgroup[1] = 1

        self.ego = self._cam_id(ego_cam)
        self.peer = self._spare_camera()
        self._peer_backup = (
            self.m.cam_pos[self.peer].copy(), self.m.cam_quat[self.peer].copy(),
            float(self.m.cam_fovy[self.peer]), int(self.m.cam_mode[self.peer]),
        )
        self.occ_geoms = self._geoms_of(occluders)
        self.tgt_geoms = self._geoms_of(targets)
        self.occ_ids = np.concatenate([g for g in self.occ_geoms.values()] or [np.zeros(0, int)])
        self.tgt_ids = np.concatenate([g for g in self.tgt_geoms.values()] or [np.zeros(0, int)])
        self.to_obs = ORIENTATIONS["identity"]
        self.orientation = None

    # -- ids
    def _cam_id(self, name):
        i = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if i < 0:
            raise KeyError(f"camera {name!r} not in model")
        return i

    def camera_names(self):
        return [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(self.m.ncam)]

    def _spare_camera(self):
        # Cameras cannot be added to a compiled model, so a world-attached camera the policy never
        # sees is repurposed. Its pose is written to both model and data and restored afterwards.
        for name in ("sideview", "birdview", "frontview"):
            i = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if i >= 0 and i != self.ego and self.m.cam_bodyid[i] == 0:
                return i
        raise RuntimeError(f"no world-attached spare camera among {self.camera_names()}")

    def _geoms_of(self, instances):
        """geom ids per BDDL instance: a geom belongs to `x_1` if it or any ancestor body is named `x_1_*`."""
        out = {}
        for inst in instances:
            ids = []
            for g in range(self.m.ngeom):
                gname = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                hit = gname.startswith(inst + "_")
                b = self.m.geom_bodyid[g]
                while not hit and b > 0:
                    bname = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                    hit = bname == inst or bname.startswith(inst + "_")
                    b = self.m.body_parentid[b]
                if hit:
                    ids.append(g)
            out[inst] = np.array(ids, dtype=int)
        return out

    # -- poses
    def cam_pose(self, cam):
        return self.d.cam_xpos[cam].copy(), self.d.cam_xmat[cam].reshape(3, 3).copy(), float(self.m.cam_fovy[cam])

    def body_pos(self, instance):
        for b in range(self.m.nbody):
            n = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if n == instance or n.startswith(instance + "_"):
                return self.d.xpos[b].copy()
        raise KeyError(instance)

    def set_peer(self, pos, target, fovy=None):
        R = look_at(pos, target)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        p = self.peer
        self.m.cam_mode[p] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        self.m.cam_pos[p] = pos
        self.m.cam_quat[p] = q
        self.m.cam_fovy[p] = self.m.cam_fovy[self.ego] if fovy is None else fovy
        # Parent is the world body, so the global pose equals the local one; writing data directly
        # avoids an mj_forward in the middle of a rollout.
        self.d.cam_xpos[p] = pos
        self.d.cam_xmat[p] = R.flatten()

    def restore_peer(self):
        pos, quat, fovy, mode = self._peer_backup
        p = self.peer
        self.m.cam_pos[p], self.m.cam_quat[p], self.m.cam_fovy[p], self.m.cam_mode[p] = pos, quat, fovy, mode
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        self.d.cam_xpos[p] = pos
        self.d.cam_xmat[p] = R

    # -- rendering
    def render(self, cam, kind="rgb", hide=None):
        hide = np.zeros(0, int) if hide is None else np.asarray(hide, int)
        old = self.m.geom_group[hide].copy()
        self.m.geom_group[hide] = HIDDEN_GROUP
        try:
            if kind == "depth":
                self.r.enable_depth_rendering()
            elif kind == "seg":
                self.r.enable_segmentation_rendering()
            self.r.update_scene(self.d, camera=cam, scene_option=self.opt)
            out = self.r.render().copy()
        finally:
            self.r.disable_depth_rendering()
            self.r.disable_segmentation_rendering()
            self.m.geom_group[hide] = old
        if kind == "seg":  # -> geom id per pixel, -1 for non-geom / background
            out = np.where(out[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, out[..., 0], -1)
        return out

    def calibrate_orientation(self, obs_img):
        """Pick the flip that maps our render onto env obs['agentview_image']; returns (name, mean abs err)."""
        ours = self.render(self.ego)
        errs = {k: float(np.abs(f(ours).astype(float) - obs_img.astype(float)).mean()) for k, f in ORIENTATIONS.items()}
        best = min(errs, key=errs.get)
        self.orientation, self.to_obs = best, ORIENTATIONS[best]
        return best, errs

    def close(self):
        self.restore_peer()
        self.r.close()
