# Complete the UAGS paper
2026-10-17 | root: /home/thanhnc/gauss_ws/src/uags | git: master@9358007 (+ uncommitted NOR loader work) | [all reports](../INDEX.md)
Follows: [Replica Table I](../2026-09-29-replica-tab1/README.md)
Verdict: IN PROGRESS -- NOR change benchmark built (commit 25ccc6c); validation run started 2026-10-18 10:53
Key result: views figure (Replica test, N=6): u_epi ranks reducible errors better than u_ale at M=10 (0.35 vs 0.90 x random, 6/6 scenes); u_ale ranks irreducible errors better than u_epi at M=90 (0.48 vs 0.83, 6/6) ([log](logs/views_test.txt), [wins](logs/views_test_wins.txt))
Attempts: n/a | Time: started 2026-10-17 10:44 JST
Running: `STAGE=val scripts/nor_tab2.sh` | PID 627181 | /home/thanhnc/gauss_ws/data/uags_results/nor/tab2/run_val.20261018.log | 2026-10-18 10:53

## 0 Context
- Ask (user, 2026-10-17): "now we need to complete the paper". Target: RA-L submission 2026-11-27 (8 pages, IEEEtran journal).
- Done before this run: Table I (Replica test, N=6, one shot) is in papers/main.tex; views figure and Table III validated.
- Machine changed: RTX 4070 Ti 12 GB (was RTX 5070 Ti 16 GB). GitHub push needs credentials on this machine; commits stay local.
- Second ask (user, 2026-10-17): "use bold for best and underline for second best in table ... please improve and make it
  dominate". Rule kept: improvements are chosen on validation scenes only, baselines are not weakened, the first test run stays on record.
- Third ask (user, 2026-10-17): give our method the class list at mapping time like CSS, "same with baselines". Plan: new rows
  "Ours, closed set" (view-level Dirichlet votes) and CSS with soft votes (the same trick for the baseline); hard/soft and alpha0 chosen
  by validation NLL; a sidecar pass on the saved maps so existing rows do not change; test run once for the new rows (workflow
  wf_bebbfcb1-3f4).
- Fourth decision (user, 2026-10-17): the paper stays purely open-vocabulary. The closed-set row was cancelled (workflow stopped
  before it changed any file). CSS stays in Table I as an unranked reference row ("shown, not compared"); the CSS-based change rows
  (CSS variance, Yager conflict) are dropped (the removal scenes also have no class labels).
- NOR protocol fixed before any method ran (lead, 2026-10-17, from the loader checks [check_nor.log](logs/check_nor.log)):

| # | Decision | Why |
|---|---|---|
| D1 | pixel truth = `masks_real` on the update frames | `masks_synthetic` is a different, virtual (ShapeNet) object: IoU with the box 0.14-0.59 |
| D2 | stale = centre in `bbox.txt` + 2 cm AND > 2 cm from every high-confidence after-removal depth point | 12-83% of the in-box surface is still there after removal (walls, desk under books) |
| D3 | one rigid ICP transform per scene (update part to map part), on depth outside the box, for all scenes and methods | before-to-after depth error 3.54 cm (012) and 2.50 cm (016); 0.58 / 0.89 cm after a rigid fit |
| D4 | keyframe rule = run_mapping defaults (5 cm / 5 deg), as Replica | keeps 27-85 keyframes per scene |
| D5 | rows: NCC, depth residual, feature residual k=2, NormLift drop, dispersion, ours, ours fused with depth | no class labels for CSS; user decision on CSS |
| D6 | validation 001-004, test 005-016 (once) | fixed before any run |
| D7 | (2026-10-18, after val scene 001 with the default field, before any other variant result) change-stream weight chosen on validation: default "view" (w = min(1, m/a0HW)) vs "relmax" (w = m / running max of m, fast_min = slow_min = 0.5, merged code 26ff3cf). Rule: the higher mean validation PIXEL-level AP of "Ours" (pixel set = covered pixels of the update keyframes, the same for both variants); tie (< 0.005): pixel AUROC; still tied: default | default field scores only 0.36-0.47% of the 1M Gaussians (a full view needs >= 9.7 px of splat mass), so ours rarely tests: Kept 0.99 (001), 0.66 (002) |
| D3' | (2026-10-18, still before any method ran) no alignment; a scene whose before-to-after median depth error is > 1 cm is left out: 012 (3.54 cm), 016 (2.50 cm). Test N = 10 | the one-transform ICP lowered photometric agreement in 6 well-aligned scenes (013: 0.63 -> 0.40) and left 1.34 cm in 012 (driver report) |
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

## 2 Baseline (object removal, validation 001-004, default change weight "view")
`STAGE=val scripts/nor_tab2.sh` at commit 25ccc6c; rows [nor_tab2_val_default.tex](logs/nor_tab2_val_default.tex), per scene [summary](logs/nor_val_default_summary.txt). In %, mean over 4 scenes.

| Row | G AP | G AUROC | P AP | P AUROC | FA | Rec |
|---|---|---|---|---|---|---|
| NCC | 21.3 | 79.3 | 27.1 | 40.4 | 18.9 | 58.3 |
| Depth residual | 39.1 | 83.0 | 28.9 | 39.8 | 0.9 | 5.8 |
| Feature residual k=2 | 61.8 | 85.6 | 30.0 | 39.8 | 17.3 | 79.6 |
| NormLift drop | 28.4 | 88.4 | 28.7 | 40.6 | 0.4 | 0.0 |
| Dispersion (ours) | 56.9 | 92.5 | 29.6 | 39.5 | 0.7 | 29.6 |
| Ours | 47.6 | 86.9 | 30.3 | 40.0 | 5.4 | 41.0 |
| Ours + depth | 61.3 | 87.2 | 30.4 | 39.6 | -- | -- |

Problem: only 0.5% of the Gaussians are scored (a full view needs >= 9.7 px of splat mass; the maps hit the 1M cap), and
70% of the covered pixels are never scored, so every pixel AUROC is below 50. This is why D7 compares the "relmax" weight.

## 3 Attempts (object removal, validation only)
| # | Idea | Result (val, N=4) | Critic |
|---|---|---|---|
| 01 | default change weight "view" | ours P AP 30.3; 0.5% of Gaussians scored; P AUROC < 50 for every row | Bad: the test reaches almost no Gaussian of a 1M map |
| 02 | change weight "relmax", fast_min = slow_min = 0.5 (D7) | ours P AP **68.3**; 61% of Gaussians scored; Kept 10.3% (was 59.0%) ([rows](logs/nor_tab2_val_relmax.tex), [summary](logs/nor_val_relmax_summary.txt)) | Good: chosen by the D7 rule (higher pixel AP of ours) |

Relmax validation rows (%): G AP / G AUROC / P AP / P AUROC / FA / Rec. NCC 36.3/82.4/58.2/78.4/10.1/49.3; depth
80.1/92.8/65.0/73.7/0.4/48.0; feature 62.8/89.5/69.3/79.2/10.5/59.7; NormLift drop 61.1/92.3/71.8/85.9/1.7/36.5;
dispersion 23.9/56.5/56.2/63.4/0.1/4.3; ours 60.3/90.1/68.3/78.7/5.8/53.2; ours + depth 84.0/93.7/73.5/78.7/-/-.
Where a baseline wins on val: depth residual beats ours alone in G AP (80.1 vs 60.3); NormLift drop has the best P AUROC.

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
| Paper compiles, page count | pdflatex + bibtex on a copy of papers/ | 8 pages after the object-removal text replaced RIO10 and the CSS change rows went; 1 undefined citation (weder2023removing, bib entry pending from the NOR workflow) | PASS (8 pages) / FAIL (citation) |

## 7 Open issues
- Page limit: 8 pages now; Table II numbers, the qualitative figure and the timing sentence still to fill.
- Timing run deferred: the desktop's graphics load the GPU at 11-76% (no compute jobs); measure when the screen is idle.
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
