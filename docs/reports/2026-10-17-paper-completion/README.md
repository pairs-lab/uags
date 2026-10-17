# Complete the UAGS paper
2026-10-17 | root: /home/thanhnc/gauss_ws/src/uags | git: master@9358007 (+ uncommitted NOR loader work) | [all reports](../INDEX.md)
Follows: [Replica Table I](../2026-09-29-replica-tab1/README.md)
Verdict: IN PROGRESS -- Table I marked, views figure and Replica ablations in the paper; NOR benchmark being built; improvement plan being designed
Key result: views figure (Replica test, N=6): u_epi ranks reducible errors better than u_ale at M=10 (0.35 vs 0.90 x random, 6/6 scenes); u_ale ranks irreducible errors better than u_epi at M=90 (0.48 vs 0.83, 6/6) ([log](logs/views_test.txt), [wins](logs/views_test_wins.txt))
Attempts: n/a | Time: started 2026-10-17 10:44 JST

## 0 Context
- Ask (user, 2026-10-17): "now we need to complete the paper". Target: RA-L submission 2026-11-27 (8 pages, IEEEtran journal).
- Done before this run: Table I (Replica test, N=6, one shot) is in papers/main.tex; views figure and Table III validated.
- Machine changed: RTX 4070 Ti 12 GB (was RTX 5070 Ti 16 GB). GitHub push needs credentials on this machine; commits stay local.
- Second ask (user, 2026-10-17): "use bold for best and underline for second best in table ... please improve and make it
  dominate". Rule kept: improvements are chosen on validation scenes only, baselines are not weakened, the first test run stays on record.
- RIO10 never arrived. User decision (2026-10-17): use the NeRF-Object-Removal dataset (Weder et al., CVPR 2023; 16 real
  iPhone scenes, object present then physically removed) for the stale-content table. Protocol fixed before any run:
  validation scenes 001-004, test 005-016 (once); map on frames before `split_start`, update on frames from `split_end`.

## 1 Findings
| Finding | Evidence | Effect on the paper |
|---|---|---|
| Raw AUSE grows with the error share, which differs by target, scene and M (e_irr share 20 +- 7%); raw error bars covered most of the axis | random levels in [views_test.txt](logs/views_test.txt) (second and third tables) | figure plots AUSE / random-ranking AUSE per scene (1 = random); every curve ordering is the same on the raw scale (both tables in the log) |
| Channels separate: u_epi < u_ale on e_red at M=10 (6/6 scenes), M=30 (5/6); u_ale < u_epi on e_irr at M=90 (6/6), full map (5/6) | [views_test_wins.txt](logs/views_test_wins.txt) | stated in Sec. V-C with the per-scene counts |
| 1 - max p of averaging ranks e_red better than both our channels in every scene and M | [views_test_wins.txt](logs/views_test_wins.txt) | stated: our channels split the error kinds, they do not replace a total-uncertainty ranking |
| The bias correction of rho_hat changes no Table III cell by more than 0.05 (r_bar, r_tilde rows; val and test) | [tab3_rows.tex](logs/tab3_rows.tex); exact values in this README, 5 Verification | stated in the ablation text; the abstract / contribution wording "bias-corrected" is not backed by a measured gain (open issue) |
| Prior-shrunk agreement hurts the aleatoric ranking (AUSE ale 14.8 vs 8.5 test, 7.3 vs 5.0 val) | [tab3_rows.tex](logs/tab3_rows.tex) | supports "agreement that does not shrink with evidence" |
| Centred features lower AUSE ale a little on both splits (8.3 vs 8.5 test, 4.7 vs 5.0 val) | [tab3_rows.tex](logs/tab3_rows.tex) | stated; the method was frozen before test, so not adopted |
| Table I marks over ALL rows (user, 2026-10-17: CSS must be marked where it is best): CSS best in mIoU, Acc, ECE, Brier, AURC; ours best in NLL (tie with averaging) and AUSE, second in mIoU, ECE, Brier, AURC | `scripts/make_tables.py tab1 ...` (default `--rank all`; `--rank ov` would rank the open-vocabulary rows only) | caption: bold best, underline second best; CSS is told the classes at mapping time |
| Our total uncertainty (1 - max p of P2, a row measured in the frozen views run) ties averaging on e_red (0.19 vs 0.19 at M=10) and beats it on e_irr in 6/6 scenes at M=90 and 180 (0.44 vs 0.52) | [views_test.txt](logs/views_test.txt), [views_test_wins.txt](logs/views_test_wins.txt) | added to the figure as "Ours P2, 1 - max p"; shown next to averaging's |

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| Environment on the new machine | CPU suite and GPU tests (see earlier rows of this run) | 145 passed / 10 skipped; GPU 24 passed after a stale test was updated (ee7b4fe) | PASS |
| I1 cross-machine: views room_0 on the RTX 4070 Ti vs the RTX 5070 Ti file | [repro_room_0.log](logs/repro_room_0.log) | 114 AUSE values, largest change 0.0021 (x100); mIoU equal to 3 decimals ([json](logs/views_test_room_0_4070ti.json)) | PASS |
| Views figure: same full map as Table I | `python3 scripts/plot_views.py --stage test` | mIoU diff 0 over 30 pairs; covered pixels equal in 90/90 (scene, M, row); temperatures equal in 114/114 (part, row) ([log](logs/views_test.txt)) | PASS |
| Views figure: random level and drawing | `CUDA_VISIBLE_DEVICES="" python3 -m pytest tests/test_plot_views.py` | 11 passed: random level = sparsification_curve with constant u; drawn y, x, error bars, colours, markers, ticks; refusal on a failed check | PASS |
| Table III: full method equals Table I P2 | `python3 scripts/make_tables.py tab3 tab3/eval_test.json --val tab3/eval_val.json --tab1 tab1/eval_test.json` | "same printed cells" ([tab3_rows.tex](logs/tab3_rows.tex)) | PASS |
| Table III exact values (r_bar vs full, test) | python read of tab3/eval_test.json summary | ECE 8.648 vs 8.683, NLL 0.9945 vs 0.9955, AUSE ale 8.450 vs 8.498 | PASS (backs the "0.05" sentence) |
| I2 eval unchanged | -- | no eval code changed; plot_views only reads the JSON files | PASS |
| Table marks | `CUDA_VISIBLE_DEVICES="" python3 -m pytest tests/test_make_tables_marks.py tests/test_make_tables_tab3.py` | 16 passed; paper rows equal the old rows after removing the marks (checked in the edit script) | PASS |
| Paper compiles, page count | pdflatex + bibtex on a copy of papers/ | 9 pages (8 at 2d980dc): references [30]-[40] spill to page 9 | FAIL (over the 8-page limit) |

## 7 Open issues
- Page limit: 9 pages now, and Table II / NOR setup / timing / qualitative figure still to fill. A cut pass is needed.
- Claim wording: "bias-corrected agreement" is in the abstract, contributions and related work, but Table III shows no
  measured gain from the correction. Needs an author decision (keep as a few-view property, or drop the claim).
- Stale-content benchmark on NOR: loader, driver, pipeline being built (workflow wf_2faa1219-1f0); then GPU runs.
- Improvement plan (workflow wf_067f2900-c19, CPU only): ideas for the predictive, the channels and the change test, judged for legitimacy;
  experiments then run on validation scenes only. CSS (closed vocabulary) will likely stay ahead on calibration.
- Push: no GitHub credentials on this machine.

## 8 Files
- Code: [scripts/plot_views.py](../../../scripts/plot_views.py), [tests/test_plot_views.py](../../../tests/test_plot_views.py),
  [scripts/make_tables.py](../../../scripts/make_tables.py) (tab3), requirements.txt (matplotlib) -- commits 03bbd81, bc90c09.
- Paper: papers/main.tex (views figure, metrics sentence, Table III Replica block), papers/figs/views.pdf -- commit f88229c;
  Table I marks (623af34, also scripts/make_tables.py, tests/test_make_tables_marks.py); total-uncertainty curve (9358007).
- Logs: [views_test.txt](logs/views_test.txt), [views_val.txt](logs/views_val.txt), [views_test_defined.txt](logs/views_test_defined.txt),
  [views_val_defined.txt](logs/views_val_defined.txt), [views_test_wins.txt](logs/views_test_wins.txt), [tab3_rows.tex](logs/tab3_rows.tex),
  [repro_room_0.log](logs/repro_room_0.log), [views_test_room_0_4070ti.json](logs/views_test_room_0_4070ti.json).
- Data (not copied): /home/thanhnc/gauss_ws/data/uags_results/{views,tab3,tab1}/ (eval JSON files, ~1 MB each).
