"""Break the G0 result down: does the frozen policy ignore the peer information, or did the see-through
image fail to deliver it? Reads probe.jsonl (+ episodes.jsonl) from one or more run dirs.

Per-state reproj recovery is 1 - KL(reproj)/KL(occ). "best peer" takes the best of the peer poses tried
at that state (who-to-ask is out of scope, so this is the fair upper bound for a given state).
fill = see-through target pixels / target pixels lost to the occluder (how much of the hidden target came back).
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return "      -"
    return f"{np.median(x):6.2f} [{np.percentile(x, 25):5.2f},{np.percentile(x, 75):5.2f}] n={len(x)}"


def table(title, groups):
    print(f"\n{title}")
    for k in groups:
        print(f"  {str(k):28s} {q(groups[k])}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="+")
    a = p.parse_args()
    rows = [json.loads(l) for d in a.run_dir for l in open(os.path.join(d, "probe.jsonl"))]
    eps = [json.loads(l) for d in a.run_dir if os.path.exists(os.path.join(d, "episodes.jsonl"))
           for l in open(os.path.join(d, "episodes.jsonl"))]

    states = defaultdict(dict)
    for r in rows:
        states[(r["task"], r["ep"], r["step"])][(r["cond"], r["peer"])] = r
    S = list(states.values())
    ok = [s[("clean", -1)]["teacher_argmax_ok"] for s in S]
    print(f"states={len(S)} teacher_argmax_ok={np.mean(ok):.3f}  episodes={len(eps)} "
          f"success(drive={eps[0]['drive'] if eps else '-'})={np.mean([e['success'] for e in eps]) if eps else float('nan'):.2f}")

    rec, best, sign, l1rec = [], [], [], []
    by_fill, by_hole, by_step, by_task, by_ok, by_hidden = (defaultdict(list) for _ in range(6))
    for (task, ep, step), s in states.items():
        occ = s[("occ", -1)]
        lost = max(occ["target_px_clean"] - occ["target_px_occ"], 1)
        hidden = 1 - occ["target_px_occ"] / max(occ["target_px_clean"], 1)
        reps = [r for (c, _), r in s.items() if c == "reproj"]
        if not reps:
            continue
        rs = [1 - r["kl"] / max(occ["kl"], 1e-9) for r in reps]
        best.append(max(rs))
        sign.append(np.mean([r["kl"] < occ["kl"] for r in reps]))
        for r, rv in zip(reps, rs):
            rec.append(rv)
            if occ["l1_pos"] > 0.05:
                l1rec.append(1 - r["l1_pos"] / occ["l1_pos"])
            fill = r["reproj_target_px"] / lost
            by_fill["<0.25" if fill < .25 else "0.25-0.5" if fill < .5 else "0.5-0.75" if fill < .75 else ">=0.75"].append(rv)
            h = r["hole_ratio"] if r["hole_ratio"] is not None else np.nan
            by_hole["hole<0.3" if h < .3 else "hole 0.3-0.6" if h < .6 else "hole>=0.6"].append(rv)
            by_step["t=0" if step == 0 else "t=20-40" if step <= 40 else "t=60-100" if step <= 100 else "t>100"].append(rv)
            by_task[task.replace("pick_up_the_black_bowl_", "")[:28]].append(rv)
            by_ok[f"teacher_ok={s[('clean', -1)]['teacher_argmax_ok']}"].append(rv)
            by_hidden["hidden<0.5" if hidden < .5 else "hidden 0.5-0.8" if hidden < .8 else "hidden>=0.8"].append(rv)

    print(f"\nreproj recovery, all (state,peer)   {q(rec)}")
    print(f"reproj recovery, best peer/state    {q(best)}")
    print(f"share of peers with KL(reproj)<KL(occ) {np.mean(sign):.2f}")
    print(f"reproj recovery on action L1 (pos)  {q(l1rec)}")
    for c in ("occ", "peer", "reproj"):
        rr = [r for r in rows if r["cond"] == c]
        print(f"  {c:7s} mean L1 pos={np.mean([r['l1_pos'] for r in rr]):.3f} rot={np.mean([r['l1_rot'] for r in rr]):.3f} "
              f"grip_agree={np.mean([r['grip_agree'] for r in rr]):.2f}")
    table("by how much of the hidden target the see-through brought back (fill)", by_fill)
    table("by hole ratio of the see-through", by_hole)
    table("by how much of the target the occluder hides", by_hidden)
    table("by step", by_step)
    table("by teacher_argmax_ok", by_ok)
    table("by task", by_task)


if __name__ == "__main__":
    main()
