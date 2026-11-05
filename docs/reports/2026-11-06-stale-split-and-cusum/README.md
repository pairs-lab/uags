# Stale answers by query type, and attempt 06: a sequential change test
2026-11-06 | root: /home/thanhnc/gauss_ws/src/uags | git: master@5f87c81 +dirty (paper edit, committed with this report) | [all reports](../INDEX.md)
Follows: [Teaser and best-case comparison](../2026-11-02-teaser-and-best-case/README.md)
Verdict: NEGATIVE -- the registered CUSUM test left more stale answers on validation (25.4 vs 24.8%), so it stopped before the test; the paper now also reports the object-query split (post hoc, disclosed)
Key result: object-query test scenes, stale answers: averaging 19.5% -> ours 4.8%, lower in 8 of 8 ([stale_by_query](logs/stale_by_query.txt)); attempt 06: n/a (stopped at validation, [06](attempts/06-cusum.md))
Attempts: 1 (0 Good / 1 Bad) | Time: 2026-11-05 23:50 -> 2026-11-06 01:12 JST (`date` stamps)

## 0 Context
- Ask (user, 2026-11-05/06): "could we can improve the results more"; then chose (AskUserQuestion): report the
  object-query split, disclosed as post hoc; run registered attempt 06 (CUSUM change test); delete the video caches
  and the duplicate proxyclip copy. Standing rule: clean the folders after the work.
- Success test: the registered rules of [configs/cusum_reset_prereg.json](../../../configs/cusum_reset_prereg.json)
  (accept for the test: >= 2.0 points fewer stale answers than ours on validation).
- Splits: NeRF-Object-Removal validation 001-004 (selection), test 005-011, 013-015 (once), as all earlier attempts.
- Machine: RTX 4070 Ti (12 GB) shared with a USGLoc training job, through scripts/gpu_gate.sh 6500 MiB; venv ~/venvs/uags (torch 2.11 cu128).

## 1 Findings
| Finding | Evidence |
|---|---|
| Our stale-answer mean (test 20.7%) comes from 3 scenes: 009 (100%), 013 (68.5%), 005 (32.5%); 7 of 10 test scenes are at 0-6% | [stale_by_query](logs/stale_by_query.txt) |
| The three 'wall'-query scenes (001, 009, 013): LSeg labelled the removed object 'wall' (56-82% of its pixels before removal); after removal every method answers 'wall' on 86-100% of the pixels, where the background is partly wall, partly floor or carpet: the metric cannot separate stale from correct answers there | [stale_by_query](logs/stale_by_query.txt), [wall_scenes](figures/wall_scenes.jpg) |
| Object-query test scenes (8): before 43.2%, test off 19.4%, averaging 19.5%, ours 4.8%, ours lower in 8/8; validation (3): 23.7% -> 0.9% | [stale_by_query](logs/stale_by_query.txt) |
| CUSUM halves false alarms (2.8 vs 5.8%) and ranks better on the hardest scene (001: G AP 62.2 vs 22.7), but resets fewer stale Gaussians where stale answers can move (002, 004) | [val_rows](logs/val_rows.md), [06](attempts/06-cusum.md) |

## 2 Baseline
| Baseline | Stale % | FA % | G AP | Check |
|---|---|---|---|---|
| ours, Table III records, reproduced on the full validation set in every attempt 06 run | 24.8 | 5.8 | 60.3 | AP/AUROC within 7.1e-8, N2 delta 0 on 4/4 scenes ([val_rows](logs/val_rows.md)) |
| ours, attempt 03 (Appendix D, copied from that report, not re-checked) | 24.8 | - | 60.3 | same |

## 3 Attempts
| # | Idea | Result (validation, mean of 4) | Critic | Link |
|---|---|---|---|---|
| 06 | CUSUM over our c_p fires the reset (C90 / C97 / C99) | stale 25.4 / 25.4 / 25.8 vs 24.8; C90: FA 2.8 vs 5.8, G AP 65.8 vs 60.3 | Bad: needs 22.8 or less; no test run. Summing helps ranking, not the outcome | [06](attempts/06-cusum.md) |

## 4 Simplify
not done: no change was kept.

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| CPU tests of attempt 06 (hand examples, toy scene, independent delta/threshold, refusals, selection, gates, adoption) | `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_cusum_reset.py` | 6 passed in 174 s ([log](logs/test_cusum_reset.txt)) | PASS |
| I1 registration committed before any real-data run | `git log configs/cusum_reset_prereg.json` | 5f87c81 at 00:16; first scene 00:16:57 ([driver](logs/val_driver.txt)) | PASS |
| I1 reproduction stop rule (Table III and N2 dump) | `STAGE=val scripts/cusum_reset.sh` | 4/4 scenes pass ([val_rows](logs/val_rows.md)) | PASS |
| I2 eval not changed | Table III and dump code used as is | hash flags False only by the path lines of the 2026-10-31 move (recorded, not required by the prereg); the reproduction above is the equivalence check | PASS |
| Independent check: selection re-derived and test gate | `scripts/cusum_reset_select.py check-test ...` | re-selection C90 / not accepted; test refused (exit 1) ([selection](logs/val_selection.txt)) | PASS |
| I4 paper numbers match the logs | Sec. IV-D and Table XI (Appendix D) vs [val_rows](logs/val_rows.md) | 25.4, 25.4, 25.8 vs 24.8; FA 2.8 vs 5.8; AP 65.8 vs 60.3 | PASS |
| Paper builds | `latexmk -pdf main.tex` | 24 pages, 0 warnings, 0 undefined | PASS |
| Tests that read main.tex | `pytest tests/test_make_tables_tab3.py tests/test_make_tables_tab2.py tests/test_video.py` | 24 passed | PASS |
| I3 references | - | no new references | NOT RUN |
| Test split of attempt 06 | - | not allowed: validation did not accept | NOT RUN |

## 6 Cost & time
| Step | Wall | Runs | GPU |
|---|---|---|---|
| CPU tests | 3 min | 1 | 0 |
| Validation, 4 scenes x 3 variants | 50 min (34 min waiting for USGLoc to free memory) | 4 | 0.23 GPU-h (812 s compute) |
| Paper build and tests | 1 min | 2 | 0 |

## 7 Open issues
- The stale-answer measure is flat on 2 of 4 validation scenes (001 saturated at 96.3%, 003 at 0%): the decision rests on 002 and 004.
- The object-query split is post hoc (disclosed in the paper text).
- Untested idea: fire when either test fires (ours OR CUSUM); needs a new registration.
- Still open from earlier runs: the video zip shows the old comparison scene and the v2 method figure; DINOtxt path mismatch; the user picks the supplement items.

## 8 Files
| File | Change |
|---|---|
| [papers/main.tex](../../../papers/main.tex) | Sec. IV-D (object-query split, ab9ba47; the two firing-rule attempts); Appendix D: six attempts, CUSUM row in Table XI |
| papers/main.pdf | rebuilt |
| [configs/cusum_reset_prereg.json](../../../configs/cusum_reset_prereg.json), [usplat/cusum_reset.py](../../../usplat/cusum_reset.py), [scripts/cusum_reset_nor.py](../../../scripts/cusum_reset_nor.py), [scripts/cusum_reset_select.py](../../../scripts/cusum_reset_select.py), [scripts/cusum_reset.sh](../../../scripts/cusum_reset.sh), [tests/test_cusum_reset.py](../../../tests/test_cusum_reset.py) | new (5f87c81) |
| /home/thanhnc/gauss_ws/src/uags/results/cusum_reset (val/*.json, selection.json, logs; 0.2 MB, git-ignored) | new |
| [attempts/06-cusum.md](attempts/06-cusum.md), [logs/](logs/), [figures/wall_scenes.jpg](figures/wall_scenes.jpg) | this report |

Removed in this run:
| Removed | Size |
|---|---|
| results/video/{clips, stills, contact, submission} (regenerable; zip and master kept) | 84 MB |
| /home/thanhnc/gauss_ws/src/third_party/proxyclip (byte-identical to the loaded gauss_ws/third_party/proxyclip) | 13 MB |
| session scratchpad, /tmp/pytest-of-thanhnc, repo __pycache__ and .pytest_cache | 77 MB |
| results/cusum_reset/text_cache_lseg_nor.pt (byte-identical copy of results/qual/text_cache_lseg_nor.pt); LaTeX aux files; scratchpad | < 1 MB |
