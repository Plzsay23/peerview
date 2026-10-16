"""No-policy check of the G0 scene pipeline: occluder discovery, orientation calibration, peer sampling,
see-through reprojection. Writes one condition grid + one diagnostics row per task.

What to look for in diag.jsonl:
    orientation_err[orientation] small (our render == env obs)       -> image convention is right
    self_reproj_px ~ 0                                                -> pinhole model is right
    geo_agree close to 1 (GT depth, GT pose)                          -> warp is right
    occ_px > 0 and target_px_occ < target_px_clean                    -> the occluder actually occludes the target
    n_peer_found > 0                                                  -> a peer can see the target (C2)
"""
import argparse
import json
import os

import numpy as np

import reproject
from conditions import PeerSampler, grid_png, look_center, render_conditions
from scene import MAX_STEPS, Scene, dummy_action, list_tasks, load_init_states, make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--occ_root", required=True)
    p.add_argument("--libero_root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--suites", nargs="+", default=list(MAX_STEPS))
    p.add_argument("--task_filter", default="")
    p.add_argument("--init_idx", type=int, default=0)
    p.add_argument("--n_peer", type=int, default=3)
    p.add_argument("--min_target_px", type=int, default=30)
    p.add_argument("--peer_aim", default="hidden", choices=["hidden", "target"])
    p.add_argument("--min_fill", type=float, default=0.5)
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--list_only", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tasks = list_tasks(a.occ_root, a.libero_root, a.suites)
    if a.task_filter:
        tasks = [t for t in tasks if a.task_filter in t["name"]]
    for t in tasks:
        print(f"{t['suite']:15s} {t['name'][:70]:70s} occ={t['occluders']} tgt={t['targets']} "
              f"init={'Y' if t['init'] else '-'} lang_same={t['language_same']} skip={t['skip_reason']}")
    if a.list_only:
        return

    rng = np.random.default_rng(a.seed)
    f = open(os.path.join(a.out, "diag.jsonl"), "w")
    for t in tasks:
        if t["skip_reason"]:
            continue
        env = make_env(t["bddl"], a.res)
        inits = load_init_states(t["init"])
        scene = Scene(env, t["occluders"], t["targets"], a.res)
        obs = env.set_init_state(inits[min(a.init_idx, len(inits) - 1)])
        for _ in range(10):
            obs, _, _, _ = env.step(dummy_action())
        ori, errs = scene.calibrate_orientation(obs["agentview_image"])
        ego_depth = scene.render(scene.ego, "depth")
        self_err = reproject.self_reprojection_error(ego_depth, scene.cam_pose(scene.ego))
        sampler = PeerSampler(rng, tries=24, min_target_px=a.min_target_px, aim=a.peer_aim, min_fill=a.min_fill)
        peers, tried = sampler.sample(scene, sampler.center(scene), a.n_peer)
        images, diags = render_conditions(scene, peers)
        grid_png(images, os.path.join(a.out, f"{t['suite']}__{t['name'][:60]}.png"))
        row = dict(suite=t["suite"], task=t["name"], occluders=t["occluders"], targets=t["targets"],
                   n_init_states=len(inits), state_dim=int(inits.shape[-1]),
                   occ_geoms={k: len(v) for k, v in scene.occ_geoms.items()},
                   tgt_geoms={k: len(v) for k, v in scene.tgt_geoms.items()},
                   peer_camera=scene.camera_names()[scene.peer], cameras=scene.camera_names(),
                   orientation=ori, orientation_err=errs, self_reproj_px=self_err,
                   n_peer_found=len(peers), peer_tried=tried,
                   occ=diags[("occ", -1)], reproj=[diags[("reproj", k)] for k in range(len(peers))])
        f.write(json.dumps(row, default=float) + "\n")
        f.flush()
        r0 = row["reproj"][0] if row["reproj"] else {}
        print(f"  ori={ori}({errs[ori]:.2f}) self={self_err} occ_px={row['occ']['occ_px']} "
              f"tgt occ/clean={row['occ']['target_px_occ']}/{row['occ']['target_px_clean']} "
              f"peers={len(peers)}/{tried} hole={r0.get('hole_ratio')} geo={r0.get('geo_agree')} "
              f"reproj_tgt={r0.get('reproj_target_px')}", flush=True)
        scene.close()
        env.close()


if __name__ == "__main__":
    main()
