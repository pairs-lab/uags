# Complete the UAGS paper
2026-10-17 | root: /home/thanhnc/gauss_ws/src/uags | git: master@9358007 (+ uncommitted NOR loader work) | [all reports](../INDEX.md)
Follows: [Replica Table I](../2026-09-29-replica-tab1/README.md)
Verdict: IN PROGRESS -- every table and figure is filled; ScanNet held-out test added to Table I and checked independently; paper 8 pages; open: push
Key result: views figure (Replica test, N=6): u_epi ranks reducible errors better than u_ale at M=10 (0.35 vs 0.90 x random, 6/6 scenes); u_ale ranks irreducible errors better than u_epi at M=90 (0.48 vs 0.83, 6/6) ([log](logs/views_test.txt), [wins](logs/views_test_wins.txt))
Attempts: n/a | Time: started 2026-10-17 10:44 JST

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
- Fifth ask (user, 2026-10-19): "we win but only slightly ... make it dominate" (Table I). Path taken: the Table I v2
  pre-registration from the merged origin work (configs/tab1_v2_prereg.json, registered 2026-10-10 before any v2 run):
  read-out-only predictives on the existing v1 fields (P1 prior grid, P2 on P1, P4 open-vocabulary vote predictive),
  chosen by pooled validation NLL against v1's P2; the test split then gets a disclosed second look. Validation stage
  started 2026-10-19 02:33 (`STAGE=val scripts/replica_tab1_v2.sh`, EVAL_NEED 7500 for this 12 GB card).
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

## Table II, object removal test (N=10, run once, relmax; [rows](logs/nor_tab2_test.tex))
| Row | G AP | G AUROC | P AP | P AUROC | FA | Rec |
|---|---|---|---|---|---|---|
| NCC | 30.4 | 75.3 | 48.0 | 72.5 | 12.4 | 46.3 |
| Depth residual | 57.4 | 89.3 | 39.7 | 67.5 | 0.6 | 22.8 |
| Feature residual k=2 | 56.6 | 87.6 | 57.8 | 78.6 | 13.7 | 75.6 |
| NormLift drop | 63.8 | 91.6 | **65.8** | **82.0** | 4.1 | 41.0 |
| Dispersion (ours) | 18.4 | 53.2 | 45.8 | 66.4 | 0.5 | 8.0 |
| Ours | 53.4 | 87.2 | 57.2 | 77.8 | 6.2 | 56.8 |
| Ours + depth | **75.0** | **92.3** | 61.8 | 78.2 | -- | -- |

Baseline wins: NormLift drop beats ours alone in G AP in 7/10 scenes and is best at the pixel level. Kept 4.8%.
Command: `VARIANT=relmax FIELD_CONFIG='{"change_weight": "relmax", "fast_min": 0.5, "slow_min": 0.5}' STAGE=test scripts/nor_tab2.sh`
(commit 2a9d578; first attempt stopped at scene 006 by a 3.5 GB mapping cap, restarted with 5.5 GB; 005 reused).

## 3 Attempts (object removal, validation only)
| # | Idea | Result (val, N=4) | Critic |
|---|---|---|---|
| 01 | default change weight "view" | ours P AP 30.3; 0.5% of Gaussians scored; P AUROC < 50 for every row | Bad: the test reaches almost no Gaussian of a 1M map |
| 03 | Table I v2 (pre-registered 2026-10-10): P4 vote predictive, P1 prior grid, P2 on P1, read-outs of the v1 fields | pooled val NLL: P2 0.800 (kept), P2-on-P1 0.809, P1 c0=0.01 0.811, P4 best 0.854 ([selection](logs/tab1_v2_selection_val.json), [rows](logs/tab1_v2_val_rows.txt)) | Bad: no v2 row beats P2 on validation; Table I unchanged. P4 lowers ECE on room_2 (2.6 vs 2.7) but costs NLL |
| 03b | Table I v2, test (second look, selection fixed: P2) | not selected rows on test: P4 vote predictive mIoU 28.2, ECE 6.3 (P2: 8.7), NLL 1.06 (P2: 1.00), AURC 16.4 (P2: 12.2); P2 on P1 ECE 8.5, NLL 1.01 ([rows](logs/tab1_v2_test_rows.tex)) | P4 is better calibrated on test but lost on validation NLL, the pre-registered rule; Table I keeps P2 (test numbers never change a choice) |
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
| Independent number check of the paper (subagent, read-only) | recomputed every cell and count from the raw JSONs; re-ran make_tables tab1/tab2/tab3 (byte-identical rows) | ~375 numbers: 2 wrong ("16 real scenes" -> 14 used; 0.35 -> 0.34), 5 wording/unit fixes (nearly unchanged, 27% is a per-scene mean, GiB, persistence among the change-test parts, 101 classes); all fixed 2026-10-19 | PASS after fixes |
| Paper compiles, page count | pdflatex + bibtex on a copy of papers/ | 8 pages after the object-removal text replaced RIO10 and the CSS change rows went; 1 undefined citation (weder2023removing, bib entry pending from the NOR workflow) | PASS (8 pages) / FAIL (citation) |
| Paper compiles, recheck 2026-10-19 12:50 (commit 72be4ba) | pdflatex, bibtex, pdflatex x2 on a copy of papers/ | 8 pages; 0 warnings in main.log; no undefined citation or reference | PASS |
| ScanNet edit, independent check (subagent, read-only, 2026-10-20) | recomputed every Table I cell and mark from the JSONs, every ScanNet count, protocol facts (OpenGaussian scene list and 19 classes from its eval_scannet.py; 30 Hz from .sens timestamps), each shortened sentence | 112/112 cells and all marks match; counts exact; 2 FAIL fixed: abstract tied "best" to 14 scenes (test N=10), Fig. 4 text "the two counts" had no antecedent and listed 1.00/1.19 in reverse order (N_eff 1.002, view count 1.191, [views_test.txt](logs/views_test.txt)); notes applied: "mean and, where shown, std"; ScanNet keyframes kept 70-94% (map logs); two of the five mIoU wins are < 0.01 (now stated) | PASS after fixes |
| Paper compiles after the ScanNet edit (commit after the check fixes) | pdflatex, bibtex, pdflatex x2 on a copy of papers/ (fresh ieeectl.bib) | 8 pages; 0 warnings; bibtex 0 warnings | PASS |

## Table III, object-removal block (test, N=10, ablations of the final method; [log](logs/nor_tab3_test.txt))
| Variant | G AP | P AP | FA | Rec | Kept |
|---|---|---|---|---|---|
| Full method | 53.4 | 57.2 | 6.2 | 56.8 | 4.8 |
| No persistence (k=1) | 47.5 | 51.3 | 7.3 | 56.1 | 5.3 |
| Pooled null | 53.6 | 56.9 | 4.6 | 46.8 | 6.5 |
| Per-Gaussian max null | 53.2 | 56.5 | 2.6 | 32.6 | 8.6 |
| Re-admission without depth | 53.4 | 57.2 | 6.2 | 56.8 | 9.5 |
| Centred features | 48.2 | 51.1 | 4.4 | 43.5 | 4.6 |

Two buckets (lambda = 0.5): not scored, its recent bucket breaks the shared coverage rule (eval refused). Centred features
needed EVAL_NEED 8500 (first try ran out of memory). Run time: [time_office_0.log](logs/time_office_0.log), 109.9 ms per
keyframe on the RTX 4070 Ti; the NOR timing failed (time_field.py passes no dataset part) and is not in the paper.

## Per-class IoU table and uncertainty figure (user request 2026-10-19)
- Table II of the paper (tab:perclass), CSS-paper layout: classes present in >= 5 of the 6 Replica test scenes (rule fixed
  before reading cells); vent and wall-plug are 0 for every row and become a caption note; IoU 0 is never marked; P1 row
  dropped (negative predictive, stays in Table I). `scripts/make_tables.py perclass eval_test_224fe1e.json --val ...`
  ([rows](logs/perclass_rows.tex)); tests/test_make_tables_perclass.py. Ours leads on ceiling (80.9) and chair (70.7) and mIoU;
  others lead elsewhere by 0.1-2.1 points.
- Fig. 3 (fig:qual): scripts/render_qual.py on room_0 (test), frame 29 = most e_red pixels by `--list-frames`
  ([frames](logs/qual_frames_room_0.json), [meta](logs/qual_room_0_f29.json)); 30-frame field mapped as replica_views.sh.
- Space: Algorithm 1 removed; related work, setup, metrics, results and ablation text tightened (all numbers kept); 8 pages.
- Cleanup (user-approved): 55 files, 46 GB of NOR ablation and default-variant fields deleted; relmax fields kept.

## ScanNet held-out set (Table I, pre-registered 2026-10-10, never run before)
- /home/thanhnc/hesfm_dataset/ScanNet holds 10 other scenes (HESFM subset), none of the 10 the UAGS prereg fixed
  (configs/scannet_{val,test}.json), so the registered scenes were downloaded to /home/thanhnc/gauss_ws/data/scannet
  (.sens, 2D label zip, .txt; curl with resume, as fetch_subset.sh).
- SensReader is Python 2 only and this machine's Python 2 lacks numpy/cv2/imageio: scripts/scannet_sens_export.py is a
  Python 3 port (streams frames; keeps the stored JPEG bytes). scene0000_00: 5578 frames in 48 s, 1296x968 colour,
  640x480 depth. tests/test_scannet_sens_export.py.
- Chain started 2026-10-19 ~12:40: download -> export + unzip -> `V2_ROWS=1 STAGE=val scripts/scannet_tab1.sh`
  (MAP_NEED 7000, EVAL_NEED 7500, DISTILL_NEED 7000 for the 12 GB card). Log: /home/thanhnc/gauss_ws/data/uags_results/scannet/chain.log.
- Download done 12:43:49 (14 GB); every export has one label image per frame (scene0000_00: 5578, scene0062_00: 730, ...).
- After validation exits 0, a second chain builds the test maps and fields only (`EVAL=0 STAGE=test`; no distillation,
  no evaluation). The single test evaluation starts by hand after the validation check. Log: .../scannet/chain2.log.
- Frozen until the test evaluation: scripts/ usplat/ configs/ (the test stage refuses code that differs from the
  validation commit). Paper and report edits are allowed.
- Validation stage 12:46-17:29 (commit 71b163b, clean). Fits: held-out PSNR 19.4 (scene0000_00), 20.5 (scene0062_00);
  every row covers 99.9% of the labeled pixels ([chain lines](logs/scannet_val_chain.txt)).
- Selection by the fixed rules ([selection](logs/scannet_selection_val.json)): LEGaussians lambda_u = 1, CSS alpha0 = 0.1;
  v2 rules keep ours P2 (no v2 predictive has a lower pooled validation NLL).
- Validation means, N=2 ([rows](logs/scannet_val_rows.txt), `eval_val.json` summary): mIoU averaging 71.3, ours P2 71.1;
  ECE P2 2.74 vs averaging 3.56; NLL 0.410 vs 0.409; AURC 3.20 vs 2.93; AUSE u_ale 3.83 vs NormLift 1-C_inter 3.39.
  So on validation ours wins ECE only; averaging and NormLift lead the other columns by small margins.
- Sanity check before the test (by hand, 17:35): no broken scene, all rows in a plausible range. Test queued:
  chain2 (maps and fields, EVAL=0) -> chain3 (`STAGE=test`, once; then `STAGE=table`). Logs: .../scannet/chain{2,3}.log.

## ScanNet test (N=8, run once 2026-10-19 18:24 - 2026-10-20 00:22)
Command: `STAGE=test scripts/scannet_tab1.sh` (chain3, after `EVAL=0 STAGE=test` maps; one attempt in look_test.json,
selection 697a4ac8). Fits: held-out PSNR 17.5-30.4. Rows: [v1 rows](logs/scannet_tab1_rows.tex) (paper), [v2 rows](logs/scannet_tab1_v2_rows.tex),
[per scene](logs/scannet_test_scene_rows.txt), [wins](logs/scannet_test_wins.txt). Results file:
/home/thanhnc/gauss_ws/data/uags_results/scannet_tab1/eval_test.json (2.2 MB). Mean over 8 scenes, %; NLL in nats:

| Row | mIoU | Acc | ECE | NLL | Brier | AURC | AUSE_u |
|---|---|---|---|---|---|---|---|
| 2D features | 62.5 | 83.7 | 8.8 | 0.64 | 27.4 | 7.2 | -- |
| Averaging | 67.7 | 86.3 | 6.2 | **0.51** | 22.3 | **4.4** | -- |
| NormLift 1-C_inter / 1-R | = averaging | | | | | | 8.3 / 11.4 |
| Feature 3DGS | 65.0 | 85.1 | 7.4 | 0.57 | 24.6 | 5.7 | -- |
| LEGaussians (l_u = 1) | 62.3 | 83.6 | 8.1 | 0.63 | 26.8 | 6.3 | 10.8 |
| Ours P0 | **68.0** | **86.5** | 6.4 | **0.51** | **22.1** | **4.4** | **8.2** |
| Ours P1 | 67.9 | 86.5 | 7.0 | 0.52 | 22.5 | 4.8 | 8.2 |
| Ours P2 | as P0 | | **6.1** | **0.51** | **22.1** | 5.0 | **8.2** |
| CSS (closed set, not ranked) | 67.3 | 86.3 | 7.0 | 0.60 | 22.9 | 8.9 | 9.4 |

- Ours best or tied in every column of the mean, but small margins: against averaging, P0 mIoU higher in 5/8 scenes,
  P2 ECE lower in 3/8, NLL 4/8, Brier 4/8; u_ale AUSE below NormLift 1-C_inter in 3/8 (8.23 vs 8.26).
- Baseline wins: P2 has a higher AURC than averaging in 8/8 scenes (5.0 vs 4.4; P0 ties at 4.39). On Replica P2 lowered
  AURC in 6/6: this effect does not transfer.
- v2 rows (not selected, second look): P4 vote predictive mIoU 68.1, ECE 6.7, NLL 0.59, AURC 9.5.
- CSS (closed set) is behind ours in every column here (on Replica it led most columns).

## Paper: ScanNet in Table I (2026-10-20)
| Change | Why | Evidence |
|---|---|---|
| Table I = Replica and ScanNet side by side, means only (`make_tables.py tab1pair`, new, marks per dataset) | two stacked blocks with std made the paper 9 pages (66 lines over); side by side adds no height | [rows](logs/tab1pair_rows.tex); tab1 output byte-identical on 3 files after the refactor; tests/test_make_tables_pair.py |
| std moved to the caption (mIoU std over scenes 2.3-4.9 Replica, 9.7-10.5 ScanNet) and per-scene win counts in the text | scenes differ more than rows; paired counts say more than std | [wins](logs/scannet_test_wins.txt) |
| ScanNet setup paragraph, results paragraph, abstract, intro, limitations ("margins smaller than the spread between scenes; evaluation frames close to mapping frames"), conclusion | report the held-out result honestly, incl. P2's AURC loss in 8/8 scenes | paper commits 189b4d4, b1dfcec |
| Removed "(fixed-threshold row of Table IV)" | Table IV has no such row since the NOR block replaced it (error found while cutting) | papers/main.tex threshold paragraph |
| `\bibliography{ieeectl,ref}` (was `{ref}`), dash for repeated names off | the IEEE "more than six authors -> et al." control entry was written but never loaded (Replica ref listed 25 authors) | bibtex: 0 warnings |
| Text tightened (captions, roadmap, related work, method prose, ablation, run time, limitations, conclusion); no number removed | back to 8 pages | build: 8 pages, 0 warnings |

## Paper: ScanNet per-class table (user request 2026-10-20)
| Change | Evidence |
|---|---|
| Table II = per-class IoU, Replica (9 classes) and ScanNet (7 classes) side by side; classes in >= 5 test scenes (the existing rule, unchanged); one colour per class in both groups (`make_tables.py perclasspair`, new) | [rows](logs/perclasspair_rows.tex); old `perclass` and `tab1` outputs byte-identical (cmp with the committed script); tests/test_make_tables_pair.py (27 table tests pass) |
| ScanNet per class: ours leads cabinet 72.0, chair 70.4, bed 90.1, ties window 57.0; averaging leads floor, wall, door by 0.2-0.4 | [rows](logs/perclasspair_rows.tex) |
| Space: the taller Table II pushed Table III off page 6 (IEEEtran full-width float limit 0.8 of the page) and the chain made 9 pages; fixed by "mIoU" header in the pair table (the tallest rotated cell) and a shorter Table I caption | build: Tables I-III on page 6, IV on page 7 |
| References: `\bibliography{ref}` came back from the editor (outside my edits); user chose to keep it and put the IEEE control entry (more than six authors -> et al., no dash) at the top of ref.bib; the preamble writer of ieeectl.bib removed | build: 8 pages, 0 LaTeX / 0 BibTeX warnings, 13 refs with et al., no generated file |

## 7 Open issues
- Fit time in the run-time sentence is from the first machine (RTX 5070 Ti); the paper names that card.
- ScanNet (Table I): ours best or tied on every mean, but wins only 3-5 of 8 scenes against averaging; P2 loses AURC in
  8/8 scenes. Stated in the text and limitations.
- Table I on Replica: ours best in 6/7 open-vocabulary columns, by small margins; the pre-registered v2 predictives
  did not win on validation (attempt 03). No further Replica change is planned (test already seen).
- NOR timing not in the paper (time_field.py passes no dataset part; the file is frozen for the NOR evals).
- Push: no GitHub credentials on this machine; the user pushes.
- Resolved 2026-10-19: "bias-corrected" claim reworded to what is measured (d8a34f8); qualitative uncertainty figure
  added (166f2f2); NOR benchmark filled (Table II, Table III block).

## 8 Files
- Code: [scripts/plot_views.py](../../../scripts/plot_views.py), [tests/test_plot_views.py](../../../tests/test_plot_views.py),
  [scripts/make_tables.py](../../../scripts/make_tables.py) (tab3), requirements.txt (matplotlib) -- commits 03bbd81, bc90c09.
- Paper: papers/main.tex (views figure, metrics sentence, Table III Replica block), papers/figs/views.pdf -- commit f88229c;
  Table I marks (623af34, also scripts/make_tables.py, tests/test_make_tables_marks.py); total-uncertainty curve (9358007).
- Logs: [views_test.txt](logs/views_test.txt), [views_val.txt](logs/views_val.txt), [views_test_defined.txt](logs/views_test_defined.txt),
  [views_val_defined.txt](logs/views_val_defined.txt), [views_test_wins.txt](logs/views_test_wins.txt), [tab3_rows.tex](logs/tab3_rows.tex),
  [repro_room_0.log](logs/repro_room_0.log), [views_test_room_0_4070ti.json](logs/views_test_room_0_4070ti.json).
- Data (not copied): /home/thanhnc/gauss_ws/data/uags_results/{views,tab3,tab1}/ (eval JSON files, ~1 MB each).
