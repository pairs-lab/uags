# Quantitative statistics, package 1 (CPU): summary
2026-10-21 | no GPU job | rules: [configs/quant_prereg.json](../../../../configs/quant_prereg.json) (R0-R2, family F1) | spec: [design/quant_review.md](design/quant_review.md)

Every number below comes from a JSON in [quant/](quant/), cited as `file#key`. The JSON was computed from the saved records; no number here was typed by hand.
`$RES` = `<repo>/../../data/uags_results`, `$NORD` = `<repo>/../../data/nerf_object_removal/nerf-object-removal`.
Every test split here was already seen, so all p-values are descriptive. Only a Holm p < 0.05 within F1 may be called "significant".

## Commands
```
cd uags && export PATH=$HOME/venvs/uags/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=""
L=docs/reports/2026-10-20-figures-video/logs/quant
python3 scripts/quant_stats.py seg --replica $RES/tab1/eval_test.json --scannet $RES/scannet_tab1/eval_test.json --legs legs_l4,legs_l1 --prereg configs/quant_prereg.json --json $L/seg.json --tex $L/seg.tex
python3 scripts/quant_stats.py nor --test $RES/nor/tab2_relmax/change_test.json --val $RES/nor/tab2_relmax/change_val.json --per-scene $RES/nor/tab2_relmax --boxes $NORD --json $L/nor.json --tex $L/nor.tex
python3 scripts/quant_stats.py views --views $RES/views/views_test.json --tab1 $RES/tab1/eval_test.json --scannet-tab1 $RES/scannet_tab1/eval_test.json --json $L/views.json --tex $L/views.tex
python3 scripts/quant_stats.py cost --logs $RES/scannet_tab1 --legs legs_l1 --json $L/cost.json --tex $L/cost.tex
python3 scripts/quant_stats.py betas --val $RES/tab1/eval_val.json $RES/scannet_tab1/eval_val.json --test $RES/tab1/eval_test.json $RES/scannet_tab1/eval_test.json --json $L/betas.json --tex $L/betas.tex
python3 scripts/quant_stats.py v2grid --replica $RES/tab1_v2/eval_test.json --scannet $RES/scannet_tab1/eval_test.json --prereg configs/tab1_v2_prereg.json configs/scannet_v2_prereg.json --json $L/v2grid.json --tex $L/v2grid.tex
python3 scripts/plot_calibration.py --replica $RES/tab1/eval_test.json --scannet $RES/scannet_tab1/eval_test.json --rows ours_P2,ours_P0,averaging,f3dgs,@legs,2d_features --out papers/figs/calibration.pdf --json $L/calibration.json
python3 -m pytest tests/test_quant_stats.py -q -p no:cacheprovider        # 26 passed in 22.2 s
```

## Outputs
| File | LaTeX blocks (`%% block:` names) |
|---|---|
| [quant/seg.tex](quant/seg.tex), [seg.json](quant/seg.json) | `f1` (Table 1a: 20 tests, x/N per dataset, median diff, sign p, Holm, Wilcoxon), `calibration` (both aggregations), `per_scene_{replica,scannet}_{miou,acc,ece,nll,brier,aurc,ause}` |
| [quant/nor.tex](quant/nor.tex), [nor.json](quant/nor.json) | `r_static`, `table3_{test,val}_<col>` (per scene, incl. `*_common`), `table3_*_r_static`, `common_share_*`, `objects_{test,val}`, `paired` |
| [quant/views.tex](quant/views.tex), [views.json](quant/views.json) | `views_replica` (M = 10, 30, 90, full), `views_scannet` (full map) |
| [quant/cost.tex](quant/cost.tex), [cost.json](quant/cost.json) | `cost`, `cost_total`, `cost_per_scene` |
| [quant/betas.tex](quant/betas.tex), [betas.json](quant/betas.json) | `p2_coefficients` |
| [quant/v2grid.tex](quant/v2grid.tex), [v2grid.json](quant/v2grid.json) | `v2grid_replica`, `v2grid_scannet` (18 candidates each) |
| [papers/figs/calibration.pdf](../../../../papers/figs/calibration.pdf) (+ .png), [quant/calibration.json](quant/calibration.json) | (a) per-scene paired ECE / NLL dots, with the scene mean and the pooled value; (b) pooled reliability diagrams with pixel-share strips and both ECEs |

## Checks
| Check | Result | Status |
|---|---|---|
| Sign test = `scipy.stats.binomtest`, every k <= n <= 25, ties dropped | equal to 1e-12 | PASS |
| Holm on toy families | hand values; monotone; capped at 1 | PASS |
| ECE from bins = usplat `expected_calibration_error` (toy data) | equal to 1e-12 | PASS |
| ECE from bins = stored ECE, every row, scene and variant of both test records | max err 5.6e-17 over 204 (Replica) + 544 (ScanNet) entries (`seg.json#calibration/ece_bins_check`) | PASS |
| Per-scene mean / std = the record's summary (Table I, Table III) | `seg` and `nor` stop otherwise (1e-9) | PASS |
| Every LaTeX cell traces to a JSON key (fresh runs and saved files, 6 subcommands) | text = rendered key values; label cells hold no numbers | PASS |
| Independent check: F1, calibration, r_static, views, cost, betas = the review's own scripts (other code) | equal at the review's printed precision | PASS |
| v2 selection recomputed from validation = saved `selection_val.json` (both datasets) | `v2grid.json#datasets/*/saved_selection_matches` | PASS |
| Figure: no Type 3 fonts, no paths or names | pdffonts: CID TrueType only; grep of PDF / LaTeX / JSON: no `/home` | PASS |
| No existing file edited | `git status`: only new files | PASS |
| Gaussian-level PR curves per scene (R9) | need the per-Gaussian GPU dump (B1) | NOT RUN |

## Paper edits the numbers require
Locations are in papers/main.tex. The lead's commit 6029f04 already rewrote "we run them once" and added the ScanNet P2 sentence in Sec. V-B.

| # | Where | Change | Numbers (source `file#key`) |
|---|---|---|---|
| 1 | Abstract; Conclusion | Replace "our predictive is the best calibrated open-vocabulary map, on ScanNet by a small margin" (Conclusion: "the adaptive predictive has the lowest ECE of the open-vocabulary rows"). New claim: ours ties averaging; training-free lifting (averaging and ours) is better calibrated than the trained maps. | P2 vs averaging: Holm p 1 on all 5 metrics (`seg.json#f1/tests/0..4/pooled/holm_p`), ECE 6/14 (`#f1/tests/1/pooled/wins`). NLL lower than Feature 3DGS in 13/14, Holm p 0.033 (`#f1/tests/7/pooled/wins`, `/holm_p`). Averaging vs Feature 3DGS NLL also 13/14 (`#descriptive/averaging_vs_trained/2/pooled/wins`). P0 vs Feature 3DGS NLL also 13/14 (`#descriptive/p0_in_place_of_p2/7/pooled/wins`). |
| 2 | Abstract last sentence; Intro "(Gaussian-level AP 75.0 vs 63.8)"; Conclusion | Keep "fused with depth ranks stale Gaussians best" only if R3 / F2 passes (B1, GPU). Until then give x/10. | Fused vs NormLift G-AP 75.0 vs 63.8, 8/10, sign p 0.11, median +14.7 (`nor.json#paired/comparisons/0`). On the common set: 73.0 vs 75.0, 6/10 (`#paired/comparisons/24`). |
| 3 | Sec. V-B, Replica paragraph: "It has the lowest ECE, Brier and AURC of the open-vocabulary rows" | ECE is lowest only as a scene mean. Pooled over pixels, P2 has the highest ECE of these rows. R2: no ECE claim on Replica. Brier holds under both aggregations. | ECE scene mean / pooled: P2 8.70 / 7.12, P0 9.86 / 5.62, averaging 9.62 / 5.70, Feature 3DGS 10.01 / 6.29, LEGaussians 10.59 / 6.20, 2D 11.78 / 7.05 (`seg.json#calibration/replica/<row>/ece_scene_mean`, `/ece_pooled`). R2 verdicts: `#r2/checks/0, 9, 12, 15` (no claim). Brier P2 36.20 / 35.92, the lowest under both (`#calibration/replica/*/brier_*`). |
| 4 | Sec. V-B, ScanNet paragraph: "our rows are best or tied in every column ... P2 lowers ECE from 6.2 to 6.1 and NLL from 0.512 to 0.509" | Write "ties averaging" (R2: no claim either way). Give both aggregations and the CI, not a gain. | ECE 6.06 vs 6.17 (scene mean), 2.62 vs 3.13 (pooled), 3/8. NLL 0.509 vs 0.512 (scene mean), 0.506 vs 0.499 (pooled: averaging lower), 4/8. Brier pooled 21.35 vs 21.14. (`seg.json#calibration/scannet/{ours_P2,averaging}/*`, `#r2/checks/33-35`). Bootstrap 95% CI of the mean gain: ECE [-0.354, +0.680] points, NLL [-0.009, +0.015] (`#f1/tests/1/scannet/boot95_mean_diff`, `#f1/tests/2/...`). |
| 5 | Sec. V-B, after the ScanNet paragraph (new sentence) | The calibration lead over the trained rows comes from training-free lifting, not from P2's extra parameters. P0 (one temperature) and averaging hold it under both aggregations on both datasets. P2 does not on Replica ECE. | `seg.json#r2/checks/18-26, 51-59` (P0 vs Feature 3DGS, LEGaussians, 2D: holds), `#r2/checks/27-32, 60-65` (averaging: holds), `#r2/checks/9, 12, 15` (P2, Replica ECE: no claim). P2 vs P0 ECE: 3/6 and 3/8 (`#descriptive/p2_vs_p0/0`). |
| 6 | Wording of "significant" anywhere | Only the 9 F1 cells with Holm p < 0.05: Feature 3DGS NLL; LEGaussians mIoU, ECE, NLL, Brier, AURC; 2D features ECE, NLL, Brier. The two ECE cells hold under R2 only on ScanNet. | `seg.json#f1/tests/*/pooled/significant`; Replica pooled ECE of P2 7.12 vs LEGaussians 6.20 and 2D 7.05 (`#r2/checks/12, 15`). |
| 7 | Table III (tab:change) new column; Sec. V-D text; Sec. metrics "targets a nominal per-view false-alarm rate of 10^-3" | Add r_static (x10^-3, nominal 1, median (min-max)). Do not call it FA. | Ours 4.0 (0.5-23.4), pooled 3.6; within 2x of nominal in 1/10 test and 2/4 val scenes. Closer to nominal than: feature residual 10/10, NCC 9/10, dispersion 9/10, NormLift 8/10, depth residual 3/10. Row medians: NCC 22.2, depth 1.1, feature 52.8, NormLift 4.1, dispersion 8.3 (`nor.json#r_static/test/rows/<row>/{median,min,max,pooled,within_2x,ours_closer/wins}`, `#r_static/val/rows/ours/within_2x`). LaTeX: `nor.tex` block `r_static`. |
| 8 | Sec. limitations: "the false-alarm rate on a new sequence can exceed the nominal rate" | State that it does: the measured median rate. | as #7 |
| 9 | Sec. V-D / Table III caption | Add AP on the common set (change_baselines.py "Report both"). | Common set = 3.0-18.4% of the observed Gaussians (`nor.json#table3/test/common_share/{min,max}`). G-AP fused 73.0, NormLift 75.0, ours 55.1, depth 54.4, feature 59.4 (`#table3/test/cols/g_ap_common/<row>/mean`). |
| 10 | Sec. V-C views paragraph: "on e_irr ours is lower at every M (... all six scenes at M = 90 and 180)" | Give x/6 at every M, including M = 30. | P2 1-max p vs averaging on e_irr: 6/6, 5/6, 6/6, 6/6 at M = 10, 30, 90, full (`views.json#replica/points/5, 13, 21, 25`). |
| 11 | Sec. V-C (new sentence) | Add the ScanNet full-map point. P2's total-uncertainty advantage does not transfer. | u_ale beats u_epi 8/8, 0.586 vs 0.866 (`views.json#scannet/points/0`). P2 1-max p vs averaging 1/8, 0.315 vs 0.259 (`#scannet/points/1`). u_ale vs NormLift 1-C_inter 3/8, 0.586 vs 0.574 (`#scannet/points/3`). |
| 12 | Sec. limitations; appendix | ScanNet P2 sign (V-B sentence done in 6029f04; values match). Add it to Limitations and add the full coefficient table. | Replica beta_0 45.6, gamma_e +0.36, gamma_a +11.83, gamma_u +0.055. ScanNet 34.6, -0.80, -2.06, +0.142 (`betas.json#datasets/{0,1}/P2/*`). LaTeX: `betas.tex`. |
| 13 | Run time paragraph (or a new cost table) | Add the ScanNet build cost with the shared-GPU caveat. Credit training-free lifting (ours, averaging and NormLift are one pass). Say "semantic stage only" if quoting the ~130-225x ratio. | Field update 5.6 s (3.0-27.4); LSeg 13.5 s; Feature 3DGS training 1267.5 s (740-2037); LEGaussians 730 s (385-1375) (`cost.json#aggregate/all/{ours_s,lseg_s,f3dgs_s,legs_s}`). Ratio of medians 225x and 130x (`#aggregate/all/*_ratio_of_medians`). End to end, with the shared 3DGS fit (324.3 s): 345.6 s vs 1614.1 s and 1077.6 s; per-scene ratio 4.0x (3.0-6.6) and 2.9x (1.8-5.3) (`#aggregate/all/total_*`). Contention: ms/iter 36.8-89.4 and 15.3-45.8. |
| 14 | Table I caption | Point to Table 1a; give per-dataset x/N in the text. | `seg.tex` block `f1` |
| 15 | Appendix (new) | Add Table 1a; both calibration aggregations; per-scene Table I (6 + 8 scenes) and Table III (10 scenes, with Kept, r_static, common set); the per-object NOR table (table only, no trend); the v2 grid (every candidate labelled "not selected", ours_P2 selected on both datasets); the P2 coefficients; the calibration figure; the cost tables. | `seg.tex`, `nor.tex` (`objects_test`), `v2grid.tex` (`v2grid.json#datasets/*/chosen`), `betas.tex`, `cost.tex`, papers/figs/calibration.pdf |
| 16 | Setup (6029f04 text) | Optional: the list of test looks omits the crashed NOR lambda = 0.5 ablation pass (scene 005, review 1h). | review 1h (not recomputed) |

## Differences from the review's text (small)
| Review says | Computed | Source |
|---|---|---|
| "NLL vs Feature 3DGS survives, as do 9 other cells" | 9 cells in total (8 others) | `seg.json#f1/tests/*/pooled/significant`; the review's own table1a.out also has 9 |
| Replica pooled NLL, P0 0.982 | 0.9815, prints 0.981 (P2 0.982) | `seg.json#calibration/replica/ours_P0/nll_pooled` |
| u_epi on e_red at M = 10: 0.35 | 0.3447 (the paper's "0.34" is right) | `views.json#replica/points/0` |
| Bootstrap CI of fused vs NormLift G-AP: [-0.3, +20.8] | [-0.5, +20.9] (fresh seed-0 stream per comparison) | `nor.json#paired/comparisons/0/boot95_mean_diff` |
| Box 007 thinnest side 0.13 m | 0.14 m (usplat.box3d.box_extents) | `nor.json#objects/test/2/sides_m` |

## Open
- F2 (fused baselines, R3), the fixed threshold (R4), PR curves and B7 need the GPU stages. Not run here.
- The per-object table is descriptive only. No size-vs-AP trend is claimed (R9).
- The JSONs in quant/ hold no absolute paths. The source runlogs do, so they must not be shipped (R10).
