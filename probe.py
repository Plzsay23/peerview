"""G0 gate: does a frozen VLA use an external (peer) observation? Open-loop probe + closed-loop success.

One pass per episode: the policy drives the robot with one condition (--drive), and every
--probe_every steps the current state is scored under all four conditions against the teacher
(the clean, occluder-removed view). --probe_every 0 gives plain success-rate runs (C1 replication).

Outputs in --out:
    config.json      arguments, versions, preprocessing path
    tasks.json       task list with occluders / skip reasons / geom counts / orientation calibration
    episodes.jsonl   one row per episode (drive condition, success, steps)
    probe.jsonl      one row per (probed state, condition, peer index)
    png/             condition grids for the first --save_png probes of each task
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

from conditions import PeerSampler, grid_png, look_center, render_conditions
from policy import DryPolicy, env_action, kl_per_dim
from scene import MAX_STEPS, Scene, dummy_action, list_tasks, load_init_states, make_env

CKPT_SUFFIX = {"libero_spatial": "spatial", "libero_object": "object", "libero_goal": "goal", "libero_10": "10"}


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--occ_root", required=True)
    p.add_argument("--libero_root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--suites", nargs="+", default=list(MAX_STEPS))
    p.add_argument("--task_filter", default="")
    p.add_argument("--max_tasks", type=int, default=0)
    p.add_argument("--shard", default="0/1", help="i/n: run every n-th task starting at i (one process per GPU)")
    p.add_argument("--n_init", type=int, default=10)
    p.add_argument("--drive", default="clean", choices=["clean", "occ", "peer", "reproj"])
    p.add_argument("--probe_every", type=int, default=10)
    p.add_argument("--max_probes", type=int, default=0)
    p.add_argument("--n_peer", type=int, default=3)
    p.add_argument("--peer_tries", type=int, default=24)
    p.add_argument("--min_target_px", type=int, default=30)
    p.add_argument("--peer_aim", default="hidden", choices=["hidden", "target"])
    p.add_argument("--min_fill", type=float, default=0.5)
    p.add_argument("--az_min", type=float, default=30)
    p.add_argument("--az_max", type=float, default=150)
    p.add_argument("--el_min", type=float, default=15)
    p.add_argument("--el_max", type=float, default=60)
    p.add_argument("--dist_lo", type=float, default=0.8)
    p.add_argument("--dist_hi", type=float, default=1.2)
    p.add_argument("--hole_iters", type=int, default=2)
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ckpt_template", default="openvla/openvla-7b-finetuned-libero-{suffix}")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--no_center_crop", action="store_true")
    p.add_argument("--dry_policy", action="store_true")
    p.add_argument("--original", action="store_true", help="original LIBERO scenes (no occluder): pipeline sanity check")
    p.add_argument("--save_png", type=int, default=3)
    return p.parse_args()


def jdump(f, row):
    f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    f.flush()


def score(policy, language, images, diags, meta):
    """Teacher = clean view. Every condition: its own greedy action, and teacher-forced log-probs on
    the teacher's tokens (KL per action dim)."""
    t_tok, t_act = policy.act(images[("clean", -1)], language)
    t_lp = policy.logprobs(images[("clean", -1)], language, t_tok)
    rows = []
    for key, img in images.items():
        if key == ("clean", -1):
            tok, act, lp = t_tok, t_act, t_lp
        else:
            tok, act = policy.act(img, language)
            lp = policy.logprobs(img, language, t_tok)
        kl = kl_per_dim(t_lp, lp)
        rows.append(dict(
            meta, cond=key[0], peer=key[1],
            kl=float(kl.sum()), kl_dims=kl, tok_agree=float((tok == t_tok).mean()),
            l1_pos=float(np.abs(act[:3] - t_act[:3]).sum()), l1_rot=float(np.abs(act[3:6] - t_act[3:6]).sum()),
            grip_agree=bool(np.sign(act[-1] - 0.5) == np.sign(t_act[-1] - 0.5)),
            action=act, teacher_action=t_act,
            teacher_argmax_ok=bool((t_lp.argmax(-1) == policy.bin_of(t_tok)).all()),
            **diags[key],
        ))
    return rows


def main():
    a = parse()
    os.makedirs(os.path.join(a.out, "png"), exist_ok=True)
    rng = np.random.default_rng(a.seed)
    tasks = list_tasks(a.occ_root, a.libero_root, a.suites, original=a.original)
    if a.task_filter:
        tasks = [t for t in tasks if a.task_filter in t["name"]]
    run_tasks = [t for t in tasks if t["skip_reason"] is None]
    if a.max_tasks:
        run_tasks = run_tasks[: a.max_tasks]
    si, sn = (int(x) for x in a.shard.split("/"))
    run_tasks = run_tasks[si::sn]
    print(f"shard {a.shard}: {len(run_tasks)} tasks", flush=True)

    import mujoco
    cfg = dict(vars(a), python=sys.version, platform=platform.platform(), mujoco=mujoco.__version__,
               started=time.strftime("%Y-%m-%d %H:%M:%S"),
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), mujoco_gl=os.environ.get("MUJOCO_GL"))
    f_ep = open(os.path.join(a.out, "episodes.jsonl"), "a")
    f_pr = open(os.path.join(a.out, "probe.jsonl"), "a")
    task_log, policy, policy_suite = [], None, None

    for t in run_tasks:
        if policy_suite != t["suite"]:
            policy = None
            if a.dry_policy:
                policy = DryPolicy(a.seed)
            else:
                from policy import OpenVLAPolicy
                ckpt = a.ckpt_template.format(suffix=CKPT_SUFFIX[t["suite"]])
                print(f"[load] {ckpt} -> {a.device} (first run downloads ~15GB into HF_HOME={os.environ.get('HF_HOME')})", flush=True)
                t0 = time.time()
                policy = OpenVLAPolicy(ckpt, t["suite"], a.device, a.attn, center_crop=not a.no_center_crop)
                import torch
                print(f"[load] done in {time.time() - t0:.0f}s, unnorm_key={policy.unnorm_key}, preproc={policy.preproc}, "
                      f"gpu_mem={torch.cuda.memory_allocated(a.device) / 2**30:.1f}GiB", flush=True)
            policy_suite = t["suite"]
            cfg.setdefault("policies", {})[t["suite"]] = dict(name=policy.name, unnorm_key=policy.unnorm_key,
                                                             preproc=policy.preproc)
            json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1, default=str)

        env = make_env(t["bddl"], a.res)
        inits = load_init_states(t["init"])
        scene = Scene(env, t["occluders"], t["targets"], a.res)
        sampler = PeerSampler(rng, a.az_min, a.az_max, a.el_min, a.el_max, a.dist_lo, a.dist_hi,
                              a.peer_tries, a.min_target_px, a.peer_aim, a.min_fill)
        max_steps = a.max_steps or MAX_STEPS[t["suite"]]
        info = dict(t, n_init_states=len(inits), peer_camera=scene.camera_names()[scene.peer],
                    occ_geoms={k: len(v) for k, v in scene.occ_geoms.items()},
                    tgt_geoms={k: len(v) for k, v in scene.tgt_geoms.items()})
        n_png = 0
        for ep in range(min(a.n_init, len(inits))):
            obs = env.set_init_state(inits[ep])
            if ep == 0:
                info["orientation"], info["orientation_err"] = scene.calibrate_orientation(obs["agentview_image"])
                print(f"[{t['suite']}] {t['name']}: orientation={info['orientation']} "
                      f"err={info['orientation_err'][info['orientation']]:.2f} occ_geoms={info['occ_geoms']}", flush=True)
            for _ in range(a.num_steps_wait):
                obs, _, _, _ = env.step(dummy_action())

            ep_peer = None
            if a.drive in ("peer", "reproj"):
                c = sampler.center(scene)
                found, _ = sampler.sample(scene, c, 1)
                ep_peer = found[:1]
            success, steps, n_probe = False, 0, 0
            t_ep = time.time()
            for step in range(max_steps):
                meta = dict(suite=t["suite"], task=t["name"], ep=ep, step=step, drive=a.drive)
                do_probe = a.probe_every and step % a.probe_every == 0 and (not a.max_probes or n_probe < a.max_probes)
                if do_probe:
                    t_pr = time.time()
                    c = sampler.center(scene)
                    peers, tried = sampler.sample(scene, c, a.n_peer)
                    images, diags = render_conditions(scene, peers, a.hole_iters)
                    meta.update(n_peer_found=len(peers), peer_tried=tried)
                    rows = score(policy, t["language"], images, diags, meta)
                    for row in rows:
                        jdump(f_pr, row)
                    kl = {c2: np.mean([r["kl"] for r in rows if r["cond"] == c2]) for c2 in ("occ", "peer", "reproj")}
                    print(f"  ep{ep} t={step:3d} probe peers={len(peers)}/{tried} KL occ={kl['occ']:.2f} "
                          f"peer={kl['peer']:.2f} reproj={kl['reproj']:.2f} "
                          f"teacher_ok={rows[0]['teacher_argmax_ok']} ({time.time() - t_pr:.1f}s)", flush=True)
                    if n_png < a.save_png:
                        grid_png(images, os.path.join(a.out, "png", f"{t['suite']}__{t['name'][:60]}__ep{ep}_t{step}.png"))
                        n_png += 1
                    n_probe += 1

                if a.drive in ("occ", "clean"):
                    img = scene.to_obs(scene.render(scene.ego, hide=scene.occ_ids if a.drive == "clean" else None))
                elif ep_peer:
                    images, _ = render_conditions(scene, ep_peer, a.hole_iters)
                    img = images[(a.drive, 0)]
                else:  # no visible peer pose this episode: fall back to the occluded view, and say so
                    img = scene.to_obs(scene.render(scene.ego))
                _, act = policy.act(img, t["language"])
                obs, _, done, _ = env.step(env_action(act).tolist())
                steps = step + 1
                if step and step % 50 == 0:
                    print(f"  ep{ep} t={step:3d} ({time.time() - t_ep:.0f}s)", flush=True)
                if done:
                    success = True
                    break
            jdump(f_ep, dict(suite=t["suite"], task=t["name"], ep=ep, drive=a.drive, success=success, steps=steps,
                             n_probe=n_probe, peer_found=None if ep_peer is None else bool(ep_peer),
                             peer=None if not ep_peer else {k: v for k, v in ep_peer[0].items() if k not in ("pos", "center")}))
            print(f"  ep{ep} success={success} steps={steps} probes={n_probe} ({time.time() - t_ep:.0f}s)", flush=True)
        task_log.append(info)
        json.dump(task_log + [dict(t, run=False) for t in tasks if t["skip_reason"]],
                  open(os.path.join(a.out, "tasks.json"), "w"), indent=1, default=str)
        scene.close()
        env.close()
    print("done", a.out)


if __name__ == "__main__":
    main()
