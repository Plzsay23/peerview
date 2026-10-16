"""Peer RGB-D -> ego see-through view (condition 3, the hand-designed codec / probe).

정직한 z-buffer 로 그리면 폐색 물체가 ego 시점에서 여전히 앞에 있어서 전부 버려진다.
그래서 ego 에서 폐색 물체가 보이는 픽셀(mask)에 한해 깊이 테스트를 의도적으로 위반하고
peer 가 본 "폐색 물체 뒤"를 덮어 그린다(see-through). 폐색 물체 자신의 peer 픽셀은 쓰지 않는다.

Pinhole model for a MuJoCo camera: square pixels, vertical fovy, principal point at the image
center, camera looks along -z with y up. Depth from mujoco.Renderer is linear z-distance (m).
"""
import numpy as np


def focal(fovy_deg, H):
    return 0.5 * H / np.tan(np.deg2rad(fovy_deg) / 2)


def unproject(depth, pose, mask):
    pos, R, fovy = pose
    H, W = depth.shape
    f = focal(fovy, H)
    v, u = np.nonzero(mask)
    d = depth[v, u]
    pc = np.stack([(u + 0.5 - W / 2) / f * d, -(v + 0.5 - H / 2) / f * d, -d], axis=1)
    return pos + pc @ R.T, v, u


def project(pts, pose, H, W):
    pos, R, fovy = pose
    f = focal(fovy, H)
    pc = (pts - pos) @ R  # world -> camera
    d = -pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = W / 2 + f * pc[:, 0] / d - 0.5
        v = H / 2 - f * pc[:, 1] / d - 0.5
    return u, v, d


def _fill_holes(img, zbuf, lab, filled, region, iters):
    """Grow filled pixels into unfilled ones inside `region` (3x3 neighbourhood, nearest depth wins)."""
    H, W = filled.shape
    for _ in range(iters):
        todo = region & ~filled
        if not todo.any():
            break
        best_z = np.full((H, W), np.inf)
        best_img, best_lab = img.copy(), lab.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                zs = np.full((H, W), np.inf)
                ys, yd = slice(max(dy, 0), H + min(dy, 0)), slice(max(-dy, 0), H + min(-dy, 0))
                xs, xd = slice(max(dx, 0), W + min(dx, 0)), slice(max(-dx, 0), W + min(-dx, 0))
                src_ok = np.zeros((H, W), bool)
                src_ok[yd, xd] = filled[ys, xs]
                zs[yd, xd] = np.where(filled[ys, xs], zbuf[ys, xs], np.inf)
                take = todo & src_ok & (zs < best_z)
                best_z[take] = zs[take]
                shifted = np.zeros_like(img)
                shifted[yd, xd] = img[ys, xs]
                shifted_lab = np.full((H, W), -1)
                shifted_lab[yd, xd] = lab[ys, xs]
                best_img[take], best_lab[take] = shifted[take], shifted_lab[take]
        grow = todo & np.isfinite(best_z)
        img[grow], zbuf[grow], lab[grow] = best_img[grow], best_z[grow], best_lab[grow]
        filled = filled | grow
    return img, zbuf, lab, filled


def see_through(ego_rgb, ego_seg, peer_rgb, peer_depth, peer_seg, ego_pose, peer_pose,
                occ_ids, tgt_ids, hole_iters=2, max_depth=10.0):
    """Returns (image, zbuf, filled, stats). Only pixels where the ego camera sees an occluder are replaced.
    ego_rgb / peer_rgb may be None to get coverage only (used when choosing peer poses)."""
    H, W = ego_seg.shape
    mask = np.isin(ego_seg, occ_ids)
    src = (peer_seg >= 0) & ~np.isin(peer_seg, occ_ids) & (peer_depth < max_depth)
    pts, pv, pu = unproject(peer_depth, peer_pose, src)
    u, v, d = project(pts, ego_pose, H, W)
    ui, vi = np.round(u), np.round(v)
    ok = np.isfinite(ui) & (d > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, d, pv, pu = ui[ok].astype(int), vi[ok].astype(int), d[ok], pv[ok], pu[ok]
    ok = mask[vi, ui]
    ui, vi, d, pv, pu = ui[ok], vi[ok], d[ok], pv[ok], pu[ok]

    colour = ego_rgb is not None and peer_rgb is not None
    out = ego_rgb.copy() if colour else np.zeros((H, W, 3), np.uint8)
    zbuf = np.full((H, W), np.inf)
    lab = np.full((H, W), -1)
    order = np.argsort(-d)  # far first, so the nearest point is written last and wins
    if colour:
        out[vi[order], ui[order]] = peer_rgb[pv[order], pu[order]]
    zbuf[vi[order], ui[order]] = d[order]
    lab[vi[order], ui[order]] = peer_seg[pv[order], pu[order]]
    filled = np.isfinite(zbuf)
    n_splat = int(filled.sum())
    out, zbuf, lab, filled = _fill_holes(out, zbuf, lab, filled, mask, hole_iters)

    occ_px = int(mask.sum())
    stats = dict(
        occ_px=occ_px,
        splat_px=n_splat,
        filled_px=int(filled.sum()),
        hole_ratio=float(1 - filled.sum() / occ_px) if occ_px else None,
        reproj_target_px=int((filled & np.isin(lab, tgt_ids)).sum()),
    )
    return (out if colour else None), zbuf, filled, stats


def geometry_agreement(zbuf, filled, clean_depth, tol=0.01):
    """Fraction of see-through pixels whose reprojected depth matches the clean ego depth within tol (m).
    With GT depth and GT poses this should be close to 1; it is the check that the warp is right."""
    if not filled.any():
        return None
    return float((np.abs(zbuf[filled] - clean_depth[filled]) < tol).mean())


def self_reprojection_error(depth, pose):
    """Unproject a camera's own depth and project it back: max pixel error (should be ~0)."""
    H, W = depth.shape
    mask = depth < 10.0
    pts, v, u = unproject(depth, pose, mask)
    uu, vv, _ = project(pts, pose, H, W)
    return float(max(np.abs(uu - u).max(), np.abs(vv - v).max())) if len(u) else None
