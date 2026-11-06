# Attempt 07: whole-update change scores from our field rank the stale Gaussians
2026-11-06 | root: /home/thanhnc/gauss_ws/src/uags | git: master@81ae01f (registration efb6b8d) +dirty (paper, tables) | [all reports](../INDEX.md)
Follows: [Stale answers by query type; attempt 06](../2026-11-06-stale-split-and-cusum/README.md)
Verdict: PARTIAL -- adopted by its registered rule (beats our test in 10/10 test scenes), but it only ties the same-aggregation feature residual and trails it fused with depth
Key result: test G AP (higher is better): ours 53.4 -> agr 64.6 (+11.2, 10/10 scenes); fair baseline feat_mean 64.7; fused 81.0 vs 84.9 ([test rows](logs/test_rows.md), [attempt 07](attempts/07-whole-update-score.md))
Attempts: 1 (1 Good vs ours / Bad vs the fair baseline) | Time: 2026-11-06 10:33 -> 13:25 JST (`date` stamps; last stamp 13:24:40)

## 0 Context
- Ask (user, 2026-11-06): "now you can run the improvement idea, try to improve the current results".
- Why the ranking: stale answers on validation have no headroom (ours 96.3 / 2.0 / 0.0 / 0.8% on 001-004, at most
  about 0.7 point to gain). The ranking has room: Table III test G AP is 53.4 for ours, 63.8 for NormLift's drop.
- Success test: the registered rules of [configs/whole_update_prereg.json](../../../configs/whole_update_prereg.json).
  - Accept for the test: the best variant's mean validation G AP >= ours + 3.0 points.
  - Adopt: test mean >= ours + 3.0 points, and higher in >= 7 of 10 scenes.
  - Effect if adopted: new rows in Table III and Table IV, sign tests with Holm. The method, the reset, the stale
    answers and Table I stay unchanged.
- Splits: NeRF-Object-Removal validation 001-004 (selection) and test 005-011, 013-015 (run once), as in every
  earlier attempt.
- Machine: RTX 4070 Ti (12 GB), shared with USGLoc through scripts/gpu_gate.sh 6500 MiB; venv ~/venvs/uags (torch
  2.11 cu128). The CPU was shared with another project (load 25-32 of 24 cores), so a scene took 7-12 min.

## 1 Findings
| Finding | Evidence |
|---|---|
| agr beats our test in every test scene: G AP 64.6 vs 53.4 (10/10), G AUROC 91.4 vs 87.2 (10/10) | [compare](logs/tables/compare.md) |
| It ties the fair baseline: the feature residual averaged over the update gets 64.7 (agr higher in 6/10) | [compare](logs/tables/compare.md) |
| Fused with depth it trails that baseline: 81.0 vs 84.9 (agr higher in 2/10; Holm p 1.000). It ties feat + depth (81.1) and NormLift + depth (79.9) | [tab4 tests](logs/tables/tab4_tests.json) |
| The gain is the aggregation, not our field: averaging raises the feature residual 56.6 -> 64.7 (8/10); it does not help NCC (30.4 -> 27.2) or depth (57.4 -> 55.8) | [compare](logs/tables/compare.md) |
| Pixel level is worse: agr P AUROC 71.7 vs ours 77.8 (lower in 8/10); P AP 59.8 vs NormLift 65.8 | [compare](logs/tables/compare.md) |
| No history gate: NaN on 1.1% of the observed test Gaussians (ours 7.0%, dispersion 88.5%) | [test rows](logs/test_rows.md) |

## 2 Baseline
| Baseline | Val G AP | Test G AP | Check |
|---|---|---|---|
| ours (Table III records), reproduced on the full sets in this run | 60.3 | 53.4 | every Table III AP / AUROC within 6.4e-8 (val, 4/4) and 3.6e-7 (test, 10/10); FA / Rec. exact ([val](logs/val_rows.md), [test](logs/test_rows.md)) |
| fair baseline feat_mean (registered with the attempt) | 67.0 | 64.7 | same runs, same records |

## 3 Attempts
| # | Idea | Result | Critic | Link |
|---|---|---|---|---|
| 07 | whole-update scores from our field's sums: dir / agr / kish | val G AP 60.9 / 67.4 / 66.0 vs ours 60.3: agr accepted. Test agr 64.6 vs 53.4, 10/10: ADOPT | Good vs ours. Bad vs the fair baseline: feat_mean ties (64.7) and wins fused (84.9 vs 81.0) | [07](attempts/07-whole-update-score.md) |

## 4 Simplify
not done: agr is a single statistic (one part). dir and kish were alternatives, and the registered selection dropped them.

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| CPU tests of the attempt (toy reproduction, independent recomputation, refusals, selection, gates, adoption) | `CUDA_VISIBLE_DEVICES= python -m pytest tests/test_whole_update.py` | 7 passed ([log](logs/test_whole_update.txt)) | PASS |
| CPU tests of the paper tables (generator, Table III/IV layout and cells, video numbers, fused report) | `python -m pytest tests/test_whole_update_tables.py tests/test_make_tables_tab2.py tests/test_make_tables_tab3.py tests/test_make_tables_marks.py tests/test_video.py tests/test_plot_nor_qual.py` | 37 passed ([log](logs/test_paper_tables.txt)) | PASS |
| The layout test catches a changed cell (one cell changed in a copy of each generated file) | scratch run on copies | 3/3 caught | PASS |
| I1 registration committed before any real-data run | `git log configs/whole_update_prereg.json` | efb6b8d at 11:10; first scene at 11:11:15 ([selection log](logs/val_selection.txt)) | PASS |
| I1 Table III reproduces (stop rule) | `STAGE=val` / `STAGE=test scripts/whole_update.sh` | val 4/4, test 10/10; capture and residual identities hold ([test run](logs/test_run.txt)) | PASS |
| I1 the test ran once, the adoption is the registered rule | `STAGE=table scripts/whole_update.sh` | check-test PASS before the test; Verdict: ADOPT (above ours by 11.2 points, higher in 10/10) ([table](logs/test_table.md)) | PASS |
| I1 paper cells come from the records | `python3 scripts/whole_update_tables.py --prereg configs/whole_update_prereg.json --test-dir results/whole_update/test --table results/whole_update/table/whole_update_test.json --tab3-record results/nor/tab2_relmax/change_test.json --out-dir docs/reports/2026-11-06-whole-update-score/logs/tables` | [tables/](logs/tables/). The 8 earlier Table IV tests reproduce exactly (x/10, median, sign p); only Holm changes (family 8 -> 24) | PASS |
| I2 eval not changed | Table III code imported, never edited | the reproduction is the equivalence check | PASS |
| I2 test changed: `test_rows_match_the_paper` | [tests/test_make_tables_tab2.py](../../../tests/test_make_tables_tab2.py) | the registered adoption changes Table III's layout. The test now checks the new layout, the new cells against their tracked file, and the whole body against the generated one: stricter than before | PASS |
| I3 references | no new reference | - | NOT RUN (none needed) |
| I4 text matches the records | recompute every number of the new text | all match ([text_numbers](logs/text_numbers.txt)) | PASS |
| I4 build: pages, warnings, abstract length | `latexmk -pdf main.tex`; pdftotext page 1 | 24 pages, 0 warnings, abstract 249 words (limit 250) ([build](logs/build_checks.txt)) | PASS |
| I4 fair baseline | feat_mean registered with the attempt, same runs | reported in Table III/IV, the text, the abstract and Appendix D | PASS |

## 6 Cost & time
| Step | Wall time | Runs | GPU |
|---|---|---|---|
| validation (4 scenes, 3 variants + 3 baseline rows) | 11:11 -> 11:57 (46 min; 2726 s in the scenes) | 4 | peak 4.6 GiB |
| test (10 scenes, agr only) | 11:58 -> 13:03 (65 min; 3854 s) | 10 | peak 4.6 GiB |
| tables, paper, report | 13:04 -> 13:25 (last `date` 13:24:40) | - | CPU |

## 7 Open issues
- No claim that our field ranks better than the baselines: the same-aggregation feature residual ties agr unfused
  and beats it fused with depth (84.9 vs 81.0). The paper says so in the abstract, Sec. IV-D, Table IV and Appendix D.
- Holm family: I read "Holm over all the table's tests" as each of our 2 fused rows against each of the 6 fused
  baselines at 2 levels (24 tests). The narrow reading is 16 tests; 24 is more conservative.
- The abstract is at 249 of 250 words.
- Not updated: the video (it shows the old Table III rows; all its numbers are unchanged and still true), and the
  8-page submission version (bb8dc0c).

## 8 Files
| File | Change |
|---|---|
| [configs/whole_update_prereg.json](../../../configs/whole_update_prereg.json), [usplat/whole_update.py](../../../usplat/whole_update.py), [scripts/whole_update_nor.py](../../../scripts/whole_update_nor.py), [scripts/whole_update_select.py](../../../scripts/whole_update_select.py), [scripts/whole_update.sh](../../../scripts/whole_update.sh), [tests/test_whole_update.py](../../../tests/test_whole_update.py) | new (efb6b8d) |
| [scripts/whole_update_tables.py](../../../scripts/whole_update_tables.py), [tests/test_whole_update_tables.py](../../../tests/test_whole_update_tables.py) | new: the adopted rows of Tables III and IV |
| [tests/test_make_tables_tab2.py](../../../tests/test_make_tables_tab2.py) | the paper layout test follows the new Table III; a new check for Table IV |
| [papers/main.tex](../../../papers/main.tex) | Tables III and IV (rows, captions), Setup (definitions), Sec. IV-D, the fused-baselines paragraph, Appendix D (seven attempts, row 07), abstract, introduction, conclusion |
| [attempt 06 critic](../2026-11-06-stale-split-and-cusum/attempts/06-cusum.md) | dated correction: its 2.0-point margin was unreachable (81ae01f) |
| /home/thanhnc/gauss_ws/src/uags/results/whole_update (val/, test/, selection.json, table/, logs/; 380 KB; git-ignored) | new |
| [logs/](logs/), [logs/tables/](logs/tables/), [attempts/07](attempts/07-whole-update-score.md) | this report |
