# Attempt 07: whole-update change scores from our field rank the stale Gaussians
2026-11-06 | root: /home/thanhnc/gauss_ws/src/uags | git: master@84afeb1 +dirty (registration, committed next) | [all reports](../INDEX.md)
Follows: [Stale answers by query type; attempt 06](../2026-11-06-stale-split-and-cusum/README.md)
Verdict: IN PROGRESS -- registration and CPU tests done; validation waits for the GPU (a USGLoc job holds 9.2 GB)
Key result: n/a
Attempts: 1 (running) | Time: 2026-11-06 10:33 -> (running) JST (`date` stamps)

## 0 Context
- Ask (user, 2026-11-06): "now you can run the improvement idea, try to improve the current results".
- Why the ranking: on validation, stale answers have no headroom (ours 96.3 / 2.0 / 0.0 / 0.8% on 001-004, about
  0.7 point at most); the ranking has (Table III test G AP: ours 53.4, NormLift's drop 63.8).
- Success test: the registered rules of [configs/whole_update_prereg.json](../../../configs/whole_update_prereg.json):
  accept for the test if the best variant's mean validation G AP is >= ours + 3.0 points; adopt if the test mean is
  >= ours + 3.0 points and higher in >= 7 of 10 scenes.
- Splits: NeRF-Object-Removal validation 001-004 (selection), test 005-011, 013-015 (once), as every earlier attempt.
- Machine: RTX 4070 Ti (12 GB) shared with USGLoc, through scripts/gpu_gate.sh 6500 MiB; venv ~/venvs/uags (torch 2.11 cu128).

## 3 Attempts
| # | Idea | Result | Critic | Link |
|---|---|---|---|---|
| 07 | whole-update scores dir / agr / kish from our field's sums (+ the same mean aggregation for the residual baselines) | running | - | - |

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| CPU tests (hand examples, toy scene reproduction, independent recomputation, refusals, selection, gates, adoption) | `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_whole_update.py` | 7 passed in 529 s ([log](logs/test_whole_update.txt)) | PASS |

## 7 Open issues
- Validation not run yet.

## 8 Files
| File | Change |
|---|---|
| [configs/whole_update_prereg.json](../../../configs/whole_update_prereg.json), [usplat/whole_update.py](../../../usplat/whole_update.py), [scripts/whole_update_nor.py](../../../scripts/whole_update_nor.py), [scripts/whole_update_select.py](../../../scripts/whole_update_select.py), [scripts/whole_update.sh](../../../scripts/whole_update.sh), [tests/test_whole_update.py](../../../tests/test_whole_update.py) | new |
| [attempt 06 critic](../2026-11-06-stale-split-and-cusum/attempts/06-cusum.md) | dated correction: its 2.0-point margin was unreachable |
