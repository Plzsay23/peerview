"""The four G0 conditions for one simulator state, plus the diagnostics that make them interpretable.

    1 occ     폐색된 자기 시점 (바닥값)
    2 peer    peer 카메라 원본을 자기 시점 자리에 교체 투입
    3 reproj  peer RGB-D 를 GT 기하로 자기 시점에 재투영한 see-through
    4 clean   폐색 물체만 지운 자기 시점 (천장값 = privileged distillation 의 교사)

Images are returned in the env obs['agentview_image'] convention (scene.to_obs applied), which is
what the policy wrapper expects.
"""
import numpy as np

import reproject
from scene import spherical, to_spherical


def hidden_region(scene, min_px=30):
    """What the occluder takes away from the ego camera, from the clean (occluder-hidden) render.

    Returns (center, n_hidden_target_px, n_occ_px). center is the median 3D point of the hidden target
    pixels if at least min_px of them exist, else of everything behind the occluder; None if the occluder
    is not in view. The peer camera is aimed here: aiming at the first object of interest (v1) often
    pointed peers at an object that was not occluded at all (G0 run 2026-10-16, recovery median 0.06).
    """
    ego_seg = scene.render(scene.ego, "seg")
    mask = np.isin(ego_seg, scene.occ_ids)
    if not mask.any():
        return None, 0, 0
    clean_seg = scene.render(scene.ego, "seg", hide=scene.occ_ids)
    clean_depth = scene.render(scene.ego, "depth", hide=scene.occ_ids)
    hidden_tgt = mask & np.isin(clean_seg, scene.tgt_ids)
    region = hidden_tgt if hidden_tgt.sum() >= min_px else mask & (clean_seg >= 0)
    if not region.any():
        return None, int(hidden_tgt.sum()), int(mask.sum())
    pts, _, _ = reproject.unproject(clean_depth, scene.cam_pose(scene.ego), region)
    return np.median(pts, axis=0), int(hidden_tgt.sum()), int(mask.sum())


class PeerSampler:
    """Peer camera poses on a sphere around the look-at point, as offsets from the ego camera.

    aim="hidden" (default): look at hidden_region(), accept a pose if the see-through would bring back at
    least min_fill of the hidden target pixels (or of the occluder area when no target is hidden).
    aim="target": v1 behaviour, look at the first object of interest, accept if min_target_px of it is seen.
    """

    def __init__(self, rng, az_min=30, az_max=150, el_min=15, el_max=60, dist_lo=0.8, dist_hi=1.2,
                 tries=24, min_target_px=30, aim="hidden", min_fill=0.5):
        self.rng = rng
        self.az = (az_min, az_max)
        self.el = (el_min, el_max)
        self.dist = (dist_lo, dist_hi)
        self.tries, self.min_target_px = tries, min_target_px
        self.aim, self.min_fill = aim, min_fill

    def center(self, scene):
        if self.aim == "hidden":
            c, _, _ = hidden_region(scene, self.min_target_px)
            if c is not None:
                return c
        return look_center(scene)

    def _fill(self, scene, ego_seg, ego_pose, hidden_tgt_px):
        pdep = scene.render(scene.peer, "depth")
        pseg = scene.render(scene.peer, "seg")
        _, _, _, st = reproject.see_through(None, ego_seg, None, pdep, pseg, ego_pose, scene.cam_pose(scene.peer),
                                            scene.occ_ids, scene.tgt_ids, hole_iters=0)
        if hidden_tgt_px >= self.min_target_px:
            return st["reproj_target_px"] / hidden_tgt_px, int(np.isin(pseg, scene.tgt_ids).sum())
        return (st["filled_px"] / st["occ_px"] if st["occ_px"] else 0.0), int(np.isin(pseg, scene.tgt_ids).sum())

    def sample(self, scene, center, n):
        """Up to n accepted poses. Returns (list of dicts, number of candidates tried)."""
        ego_pos, _, _ = scene.cam_pose(scene.ego)
        az0, _, d0 = to_spherical(ego_pos, center)
        if self.aim == "hidden":
            ego_seg = scene.render(scene.ego, "seg")
            ego_pose = scene.cam_pose(scene.ego)
            _, hidden_tgt_px, _ = hidden_region(scene, self.min_target_px)
        found, tried = [], 0
        while len(found) < n and tried < self.tries:
            tried += 1
            daz = self.rng.uniform(*self.az) * self.rng.choice([-1, 1])
            el = self.rng.uniform(*self.el)
            dist = d0 * self.rng.uniform(*self.dist)
            pos = spherical(center, az0 + daz, el, dist)
            scene.set_peer(pos, center)
            if self.aim == "hidden":
                fill, tpx = self._fill(scene, ego_seg, ego_pose, hidden_tgt_px)
                ok = fill >= self.min_fill
            else:
                seg = scene.render(scene.peer, "seg")
                tpx, fill = int(np.isin(seg, scene.tgt_ids).sum()), None
                ok = tpx >= self.min_target_px
            if ok:
                found.append(dict(pos=pos, center=np.asarray(center, float), daz=float(daz), el=float(el),
                                  dist=float(dist), dist_ratio=float(dist / d0), target_px_peer=tpx, pred_fill=fill))
        return found, tried


def look_center(scene):
    """Look-at point: the first object of interest that has a body in the model."""
    pts = []
    for name in scene.tgt_geoms:
        try:
            pts.append(scene.body_pos(name))
        except KeyError:
            pass
    return pts[0] if pts else None


def render_conditions(scene, peers, hole_iters=2):
    """peers: list from PeerSampler.sample. Returns (images, per-condition diagnostics)."""
    to_obs = scene.to_obs
    ego_pose = scene.cam_pose(scene.ego)
    occ = scene.render(scene.ego)
    clean = scene.render(scene.ego, hide=scene.occ_ids)
    ego_seg = scene.render(scene.ego, "seg")
    clean_seg = scene.render(scene.ego, "seg", hide=scene.occ_ids)
    clean_depth = scene.render(scene.ego, "depth", hide=scene.occ_ids)
    base = dict(
        occ_px=int(np.isin(ego_seg, scene.occ_ids).sum()),
        target_px_occ=int(np.isin(ego_seg, scene.tgt_ids).sum()),
        target_px_clean=int(np.isin(clean_seg, scene.tgt_ids).sum()),
    )
    images = {("occ", -1): to_obs(occ), ("clean", -1): to_obs(clean)}
    diags = {("occ", -1): dict(base), ("clean", -1): dict(base)}
    for k, p in enumerate(peers):
        scene.set_peer(p["pos"], p["center"])
        peer_pose = scene.cam_pose(scene.peer)
        prgb = scene.render(scene.peer)
        pdep = scene.render(scene.peer, "depth")
        pseg = scene.render(scene.peer, "seg")
        rimg, zbuf, filled, st = reproject.see_through(
            occ, ego_seg, prgb, pdep, pseg, ego_pose, peer_pose, scene.occ_ids, scene.tgt_ids, hole_iters)
        st["geo_agree"] = reproject.geometry_agreement(zbuf, filled, clean_depth)
        peer_info = {k2: v for k2, v in p.items() if k2 not in ("pos", "center")}
        images[("peer", k)] = to_obs(prgb)
        images[("reproj", k)] = to_obs(rimg)
        diags[("peer", k)] = dict(base, **peer_info)
        diags[("reproj", k)] = dict(base, **peer_info, **st)
    return images, diags


def grid_png(images, path, scale=1):
    """Save the condition images side by side, rotated 180 degrees: i.e. as the policy sees them after
    its own preprocessing rotation (before JPEG / resize / crop)."""
    from PIL import Image, ImageDraw

    keys = list(images)
    h, w = images[keys[0]].shape[:2]
    canvas = Image.new("RGB", (w * len(keys), h + 14), "white")
    draw = ImageDraw.Draw(canvas)
    for i, k in enumerate(keys):
        canvas.paste(Image.fromarray(np.ascontiguousarray(images[k][::-1, ::-1])), (i * w, 14))
        draw.text((i * w + 3, 1), f"{k[0]}{'' if k[1] < 0 else k[1]}", fill="black")
    if scale != 1:
        canvas = canvas.resize((canvas.width * scale, canvas.height * scale))
    canvas.save(path)
