# peerview

G0 gate for proposal A ("VLA with Noise Observation"). When a robot's own camera is occluded, can a frozen VLA use a peer robot's observation? No training here - only a quick check of whether this looks possible.

- Policy: OpenVLA fine-tuned on LIBERO (frozen, weights untouched)
- Scenes: LIBERO-Occ (arXiv 2606.10862), occluded variants of LIBERO tasks with the same language instructions
- How to run: [run.md](run.md) (Korean)

## Four conditions

Same simulator state, same policy, only the input image changes.

| # | Condition | Meaning |
|---|---|---|
| 1 | occ | Own view with the occluder. Floor |
| 2 | peer | Peer camera image fed in place of the own view |
| 3 | reproj | Peer RGB-D warped into the own view with GT geometry (see-through) |
| 4 | clean | Own view with only the occluder hidden. Ceiling, used as the teacher |

Scoring: for each condition, KL between its action-token distribution and the teacher's (condition 4), teacher-forced on the teacher's tokens. Recovery = 1 - KL(c) / KL(occ). 0 = no better than the occluded view, 1 = as good as the clean view.

## How the scene is built

- Occluder = objects added in the LIBERO-Occ BDDL relative to the original LIBERO BDDL. Found automatically for 39 of 40 tasks (one task occludes by moving existing objects and is skipped).
- Hiding the occluder = moving its geoms to an invisible geom group. Rendering only, physics unchanged.
- Peer camera = a world-attached camera of the model (`sideview`) re-posed on a sphere around the target. The policy camera (`agentview`) is never touched.
- See-through = peer depth unprojected to 3D, projected into the own camera, written only where the own camera sees the occluder (depth test intentionally violated there).

## Pipeline check (local WSL, no policy, 39 tasks)

| Check | Result |
|---|---|
| Our render vs env observation | Identical after a vertical flip (mean abs error 0.00) |
| Own-depth reprojection error | ~1e-13 px |
| Tasks where the occluder hides >30% of the target | 25 / 39 (libero_spatial: 10 / 10) |
| Hidden target pixels recovered by see-through (median, those 25) | 0.64 |
| Depth agreement of see-through pixels with the clean view (median) | 0.83 |

Not yet run: the OpenVLA scoring path (needs a server GPU; 7B in bf16 does not fit an 8 GB laptop GPU).

## Files

| File | Role |
|---|---|
| `scene.py` | Task list, occluder discovery, rendering, peer camera |
| `reproject.py` | See-through reprojection and geometry checks |
| `conditions.py` | The four condition images and diagnostics for one state |
| `policy.py` | OpenVLA wrapper (greedy action, teacher-forced log-probs), model-free `DryPolicy` |
| `probe.py` | Rollout + periodic four-condition scoring |
| `render_check.py` | Rendering pipeline check without a policy |
| `summarize.py` | Table and bar chart |
| `setup_server.sh` | One-time conda environment for the lab server |
