"""probe.jsonl -> the G0 bar chart and a one-screen verdict table.

Recovery of condition c at a state = 1 - KL(teacher||c) / KL(teacher||occ): 0 = no better than the
occluded view, 1 = as good as the clean view. Only states where occlusion actually moves the policy
(KL_occ > --min_kl) and actually hides the target (target_px_occ < --vis_ratio * target_px_clean) count.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--min_kl", type=float, default=0.5)
    p.add_argument("--vis_ratio", type=float, default=0.7)
    a = p.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(a.run_dir, "probe.jsonl"))]
    states = defaultdict(dict)
    for r in rows:
        states[(r["task"], r["ep"], r["step"])][(r["cond"], r["peer"])] = r
    n_all = len(states)
    print(f"states={n_all}  teacher_argmax_ok={np.mean([r['teacher_argmax_ok'] for r in rows if r['cond'] == 'clean']):.3f}")

    rec, kls, toks = defaultdict(list), defaultdict(list), defaultdict(list)
    n_used = 0
    for s in states.values():
        occ = s[("occ", -1)]
        if occ["kl"] <= a.min_kl or occ["target_px_occ"] >= a.vis_ratio * max(occ["target_px_clean"], 1):
            continue
        n_used += 1
        for (cond, _), r in s.items():
            kls[cond].append(r["kl"])
            toks[cond].append(r["tok_agree"])
            rec[cond].append(1 - r["kl"] / max(occ["kl"], 1e-9))
    print(f"states used={n_used}/{n_all} (occlusion moves the policy and hides the target)")
    order = ["occ", "peer", "reproj", "clean"]
    print(f"{'cond':8s} {'n':>5s} {'KL':>8s} {'recovery':>9s} {'tok_agree':>9s}")
    for c in order:
        if kls[c]:
            print(f"{c:8s} {len(kls[c]):5d} {np.mean(kls[c]):8.3f} {np.median(rec[c]):9.3f} {np.mean(toks[c]):9.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["(1) occluded", "(2) peer raw", "(3) reprojected", "(4) clean"]
    vals = [np.median(rec[c]) if rec[c] else np.nan for c in order]
    errs = [[np.median(rec[c]) - np.percentile(rec[c], 25) if rec[c] else 0 for c in order],
            [np.percentile(rec[c], 75) - np.median(rec[c]) if rec[c] else 0 for c in order]]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(labels, vals, yerr=errs, color=["#888", "#c97", "#59c", "#4a4"], capsize=4)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("recovery (median, IQR)")
    ax.set_title(f"G0: frozen OpenVLA, {n_used} occluded states")
    fig.tight_layout()
    fig.savefig(os.path.join(a.run_dir, "g0_bars.png"), dpi=150)
    print("wrote", os.path.join(a.run_dir, "g0_bars.png"))


if __name__ == "__main__":
    main()
