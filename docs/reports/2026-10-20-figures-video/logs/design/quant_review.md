VERDICT revise
ISSUES:
- [selection, post hoc primary] R1 names 'the primary test' (P2 vs Feature 3DGS NLL over 14 scenes) after ~200 paired p-values had already been computed and read (S/quant_stats.out, S/quant_extra.out). It is the smallest p in that list, so calling it 'significant' is HARKing. Fix: no primary test on data already seen. Use Holm over the full declared family instead (4 baselines x 5 metrics = 20 sign tests on the pooled 14 scenes). NLL vs Feature 3DGS still survives (Holm p 0.033), as do 9 other cells (S/review/quant/holm.out, table1a.out).
- [misleading credit] The calibration and cost advantages over the trained baselines also belong to the averaging baseline. Averaging beats Feature 3DGS and LEGaussians on NLL in 13/14 and 13/14 scenes, and on ECE in 13/14 and 13/14 (S/review/quant/review_checks.out E). Its build cost is the same pass as ours (paper: 'averaging and NormLift reuse our sums and add 2.3 ms'). So the summary line 'UAGS is better calibrated than the trained baselines ... 130-225x lower cost' and the proposed paper sentence 'significantly better calibrated than the trained baselines' credit the vMF field with a property of training-free lifting. Ours P0 (one temperature) gives the same result, so P2's extra parameters are not the cause either. Write 'training-free lifting (averaging and ours)', and put averaging/NormLift rows in the cost table.
- [misleading aggregate] Table 1a headlines the MEAN per-scene relative gain, which outliers drive. ECE vs averaging reads +11.4% as a mean, but the median is -3.0% and ours wins only 6/14. Brier vs Feature 3DGS: +6.9% mean vs +2.3% median (S/review/quant/relgain.out). Report the median paired absolute difference plus per-dataset x/N.
- [inconsistent test choice] Cells switch between the sign test and the Wilcoxon p, taking whichever is smaller: '0.057 (Wilcoxon 0.009)' for ECE vs F3DGS, and 'fused vs depth ... Wilcoxon p 0.049' where the sign p is 0.109. Use one test everywhere (exact two-sided sign test). Show Wilcoxon only as a complete secondary column.
- [aggregation cherry-pick, false alarms] Section 1d and the summary compare per-view static false-alarm rates by their MEAN: 'ours 6.3 vs 42 to 86 for NormLift and the feature residual'. This happens right after the design warns that the ECE claim depends on aggregation. Medians: ours 3.97e-3, NormLift 4.05e-3, feature 52.8e-3, NCC 22.2e-3, dispersion 8.3e-3, depth 1.09e-3. Pooled: ours 3.6e-3 vs NormLift 50.7e-3. Use median, range and the per-scene |log10(rate/1e-3)|. On that metric ours is closer to nominal than feature 10/10, NCC 9/10, dispersion 9/10, NormLift 8/10, but depth 3/10 (S/review/quant/review_checks.out A).
- [selective reporting, views] Section 1f says 'P2's 1 - max p ranks e_irr better than averaging at M = 10, 90 and 180 (6/6)'. It leaves out M = 30 (5/6). It also leaves out ScanNet's full-map point, which is reversed: 0.315 vs 0.259, P2 better in 1/8 (S/review/quant/scannet_fullmap.out). The cited evidence file only holds ours_P0, not P2.
- [unsupported claim] Section 1b says that within a scene P2 spreads confidence too much (bins 0.3-0.4 under-confident by 14-16 points, 0.75-0.85 over-confident by 13-22). Those are pooled bins. Per scene, P0 and averaging show the same pattern, e.g. office_1 averaging +31.7 points at confidence 0.27-0.33 and office_4 averaging -29.3 at 0.80-0.87. Drop the sentence or show per-scene bins for every row. Pooled ECE lets one scene's over-confidence cancel another's under-confidence.
- [number error] 'On the set where every row is defined (2-17% of the observed Gaussians)' is wrong. It is 3.0-18.4% (gaussian.common.n / counts.n_observed, NOR test JSONs).
- [bug] nor_dump_scores step 5 builds the theta_fix histogram from c_ring.min(1) after update_view. vmf_field.py:535 sets c_ring[fire] = NaN, so every Gaussian that fired is dropped and the 0.999 quantile is biased low. Use c_last[last_tested] (set at vmf_field.py:519 and 528). This is exactly what ChangeTally.step does (scripts/run_mapping.py:150-158): reuse it.
- [check design] The '--check every Table III AP within 0.002' step has two problems. (1) ranking_table and pixel_level build the common set from the rows they are given (EC:620-628, 682), so adding fused rows changes the common-set values. Check only Table III's cells (NaN-last G/P AP and AUROC, FA, Rec) from a call with EC.ROWS alone, before any rows are added. (2) The 0.002 tolerance is untested: no NOR evaluation was ever repeated. Measure run-to-run noise first by repeating EC on one validation scene into a new output.
- [provenance] 'Never edit EC' is too narrow. EC.code_state hashes 9 files (EC:191-193): eval_change_nor.py, time_field.py, eval_seg.py, usplat/change_baselines.py, evaluation.py, seg_eval.py, rows.py, nor_align.py, box3d.py. The fields' provenance hashes MAPPING_CODE (run_mapping.py:378-381: vmf_field.py, baselines.py, datasets.py, ...). None of these may be edited.
- [test-look accounting] 'NOR test look 2' and 'Replica test look 3' are wrong. NOR test has already had Table III plus 5 Table IV ablation passes (RES/nor/tab2_abl_{k1,pooled,maxnull,nodepth,centre}) and 1 crashed pass (tab2_abl_lam05, scene 005). Replica test has had Table I v1, the views test (plus the room_0 repeat in views_repro), the Table IV ablations (RES/tab3, 2026-10-17), Table I v2 (RES/tab1_v2) and the qual render. The paper's sentence 'We freeze every choice before the test scenes run, and we run them once' is therefore false. configs/tab1_v2_prereg.json reporting rule 5 requires the paper to state the second look, and it does not.
- [missing disclosure, no GPU] On ScanNet the fitted P2 coefficients have the opposite sign: gamma_e = -0.80, gamma_a = -2.06, against Replica +0.36 and +11.8 (betas.ours_P2 in scannet_tab1/eval_val.json; code a, b map to paper gamma_e, gamma_a; usplat/predictive.py:212-216). P2 therefore sharpens the uncertain pixels on ScanNet, the opposite of the rationale in Sec. III-H. This fits P2's AURC loss in 8/8 ScanNet scenes. Neither the paper nor the design reports it; add it as an A-item.
- [missed A-item, no GPU] The full-map point of the views test already exists for ScanNet, from the Table I JSON. u_ale ranks the full-map error better than u_epi in 8/8 test scenes (0.586 vs 0.866). u_ale vs NormLift 1-C_inter is 3/8 (a tie). P2's 1 - max p loses to averaging 1/8. The design proposes a 2 h GPU run (B4) without using this.
- [A4 inaccurate] A4 says the v2 rows are 'not done yet'. Per-family v2 representatives already exist in docs/reports/2026-10-17-paper-completion/logs/tab1_v2_test_rows.tex and scannet_tab1_v2_rows.tex. Missing are every individual grid row (as the prereg reporting rules require) and the paper's statement of the Replica second look. make_tables.py tab1 --v2 prints only the representatives, so the full grid needs new code.
- [B7 scope] The pre-registered reducibility criterion is validation-only: stage 'val', office_0 and room_2, M = 10 (eval_views.py:42-53, 143-144), with a CPU verdict command at eval_views.py:377-415. The '(+70 min test)' is not part of it and would be an extra test look. The check is registered, cheap (~20 min GPU) and binding. Run it on validation once the user agrees; otherwise the report and paper must say it was registered and not run. Skipping it because of its likely outcome is selective reporting.
- [B6 infeasible as written] B6 says to run scripts/replica_static_fa.sh 'with OUT and FROZEN overrides'. FROZEN is hard-coded at replica_static_fa.sh:50 (FROZEN=configs/static_fa.json), the CONFIGS list comes from that file's ablations, and MAP_NEED=10000 needs about 12 GB free with the gate's headroom. A relmax run needs a new parametrised script and new frozen configs (change_weight relmax, fast_min = slow_min = 0.5, as the Table III fields).
- [GPU feasibility] gpu_gate.sh waits for need + 2048 MiB free. A 6500 gate needs 8.5 GB free; 3.9 GB is in use now (8.4 GB free). The MAP_NEED defaults in replica_views.sh and scannet_tab1.sh (9000) need 11 GB free, which is rare on this shared card. Measured peaks: ScanNet mapping alloc 2.3-2.8 GB, reserved 6.28 GB (timing lines of RES/scannet_tab1/map_scene*.log). Replica k-fields: alloc 2.2 GB, reserved 6.27 GB (RES/repro.log). Set gates from measured peaks. The '40 min' is compute time, not wall time.
- [R3 gameable detail] 'p against the strongest fused baseline' picks the comparator on test. Report all four fused baselines, each with x/10, median difference and sign p, with Holm over the declared 8-test family (4 rows x G/P AP).
- [R5 incomplete] R5 states a rule only for the e_red claim. Add the e_irr rule: the full map is already computed (8/8); also M = 30. State the M values explicitly.
- [R6 ambiguity] The B3 gate 'the mapping-only map marks >= 20% of the mask pixels' does not say which map. Ours P0 and averaging differ before the update. Use a method-independent gate: LSeg 2D relevancy on held-out MAP frames. Also, 'ours' flag counts equal the saved counts' demands exact equality despite GPU noise; use the measured tolerance.
- [post hoc power] '~460 scenes for 80% power' takes the observed effect size as the true one. Post hoc power is uninformative because it is a function of the p-value. Replace it with the CI of the paired difference (e.g. ScanNet ECE gain [-0.35, +0.68] points) and say that no difference is claimed.
- [claims vs paper] The abstract and conclusion say 'our predictive is the best calibrated open-vocabulary map'. That depends on aggregation and on the baseline chosen. Replica pooled ECE: P2 7.12 vs averaging 5.70. ScanNet pooled NLL: P2 0.506 vs averaging 0.499. Per-scene ECE wins vs averaging: 3/6 and 3/8. The design's paper edits only add 'on par with averaging'; they must also replace the 'best calibrated' wording, and the intro's 'AP 75.0 vs 63.8' wording must wait for B1.
- [visual risks] (1) A6 pooled reliability diagrams hide per-scene miscalibration: pooled ECE favours P0 and averaging on Replica and P2 on ScanNet. Show per-scene paired ECE/NLL as well, with identical bins, axes and rows. (2) A7 is a size-vs-AP scatter with N = 10, Spearman 0.60 and p = 0.07, and the size measure (volume vs thinnest side) was picked after seeing both. Use a table, or show both measures with no trend line. (3) PR curves must be per scene, because AP is a mean over scenes.
- [anonymity] Runlog JSONs record env.host = the user's login name, /home/<user> paths and the git repo path (fields cwd, executable, git, env; e.g. RES/nor/tab2_relmax/test/005.json). R9 ('every number links to its JSON or log') must mean LaTeX %% comments only, never visible links. Never ship these records, or the new dump/stats JSONs, in an anonymous supplement without scrubbing them.
- [naming and notation] The new column must not be called a 'false-alarm rate'/FA, because Table III's FA is the share of static Gaussians flagged at least once. Call it r_static (x10^-3, nominal 1). Paper-facing text must use the paper's symbols gamma_e, gamma_a, gamma_u, beta_0, theta(h), q, lambda, not the code names a, b, g.
- [minor line refs] SegAccumulator is at seg_eval.py:405-433 (not 408-436), and summarize at :366-402.
- [effort] '3 h CPU' for package 1 leaves out writing and testing 4 new scripts and 2 tests. Realistic: about 1 day of coding, ~1 h CPU, and 45-60 min of GPU compute, with gate waiting on top.

REVISED SPEC:
The spec below corrects the 2026-10-20 design. Every number in it comes from a saved JSON or log; I checked them in the read-only phase.

# Quantitative results: design v2 (review-corrected, 2026-10-21, read-only phase)

## Short names
| Name | Path or meaning |
|---|---|
| R | /home/thanhnc/gauss_ws/src/uags (repo) |
| RES | /home/thanhnc/gauss_ws/data/uags_results |
| NORD | /home/thanhnc/gauss_ws/data/nerf_object_removal/nerf-object-removal |
| EC | R/scripts/eval_change_nor.py |
| DSF | usplat.vmf_field.DirectionalSemanticField |
| NOR | NeRF-Object-Removal |
| S | /tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design |
| Q | S/review/quant |
| L | R/docs/reports/2026-10-20-figures-video/logs (the folder exists) |

- Tables are numbered as in papers/main.tex: I = tab:replica, II = tab:perclass, III = tab:change, IV = tab:ablation.
- Paper symbols map to code names as follows: P2 coefficients gamma_e, gamma_a, gamma_u = code a, b, g. Change threshold theta(h), quantile q = 0.999, retention lambda.

### Evidence
All evidence was computed on the CPU only; nothing in R or the data folders changed.
- From the original design: S/quant_stats.py and S/quant_extra.py, each with its .out file.
- From the review, in Q: table1a.py, holm.py, relgain.py, nor1c.py, review_checks.py and scannet_fullmap.py, each with its .out file.
- To rerun any of them: `~/venvs/uags/bin/python <f>.py > <f>.out`.

All test splits below have already been seen. So every number in section 1 is descriptive and post hoc.

## 0 Bottom line
1. **Ours ties averaging on every Table I metric.**
   - Pooled over the 14 test scenes, ours wins 6 to 10 of 14, Holm p = 1 everywhere, and the median paired differences are about 0.
   - Claim no difference.
2. **Training-free lifting (averaging and ours alike) is better calibrated than our trained re-implementations and than per-frame 2D features.**
   - Example: ours P2 vs Feature 3DGS NLL is lower in 13/14 scenes (Holm p 0.033). Averaging vs Feature 3DGS is also 13/14.
   - Ours P0, with one temperature, gives the same result.
   - Write this as a property of training-free lifting, not of the vMF field.
3. **New and robust, with no GPU run:**
   - The measured per-view static false-alarm rate is a median of 4.0e-3 against the nominal 1e-3. Only 1 of 10 test scenes is within a factor of 2.
   - Ours is closer to nominal than the feature residual (10/10), NCC (9/10), the dispersion (9/10) and NormLift (8/10), but not the depth residual (3/10).
   - On the ScanNet full map, u_ale ranks errors better than u_epi in 8/8 test scenes.
4. **Must be fixed before claiming more:**
   - (a) The headline NOR result (ours fused with depth, G-AP 75.0) has no fused baselines.
   - (b) "Best calibrated" depends on how ECE is aggregated.
   - (c) The ScanNet P2 coefficients have the opposite sign.
   - (d) The paper says the test scenes ran once, which is false for Replica and NOR.
5. **No method change has validation support** (section 3C).

## 1 What the saved results say

### 1a Table I, paired over test scenes
Source: Q/table1a.out.
- mIoU uses ours_P0 (P0 and P2 share labels). The other columns use ours_P2.
- Median paired difference: positive = ours better. In points for mIoU, ECE, Brier and AURC; in nats for NLL.
- Tests: exact two-sided sign test over 14 scenes, then Holm over these 20 tests.

| Baseline, metric | Replica x/6 | ScanNet x/8 | Pooled x/14 | Median diff | Sign p | Holm |
|---|---|---|---|---|---|---|
| averaging mIoU | 3 | 5 | 8 | +0.003 | 0.79 | 1 |
| averaging ECE | 3 | 3 | 6 | -0.176 | 0.79 | 1 |
| averaging NLL | 5 | 4 | 9 | +0.006 | 0.42 | 1 |
| averaging Brier | 6 | 4 | 10 | +0.240 | 0.18 | 1 |
| averaging AURC | 6 | 0 | 6 | -0.089 | 0.79 | 1 |
| Feature 3DGS mIoU | 2 | 7 | 9 | +0.49 | 0.42 | 1 |
| Feature 3DGS ECE | 4 | 7 | 11 | +0.81 | 0.057 | 0.46 |
| Feature 3DGS NLL | 6 | 7 | 13 | +0.038 | 0.0018 | 0.033 |
| Feature 3DGS Brier | 6 | 6 | 12 | +0.80 | 0.013 | 0.14 |
| Feature 3DGS AURC | 6 | 6 | 12 | +0.69 | 0.013 | 0.14 |
| LEGaussians mIoU | 6 | 8 | 14 | +3.52 | 1.2e-4 | 0.0024 |
| LEGaussians ECE | 6 | 8 | 14 | +1.26 | 1.2e-4 | 0.0024 |
| LEGaussians NLL / Brier / AURC | 6 | 7 | 13 | +0.108 / +3.24 / +1.62 | 0.0018 | 0.033 |
| 2D features mIoU | 4 | 7 | 11 | +3.45 | 0.057 | 0.46 |
| 2D features ECE / NLL / Brier | 5-6 | 7-8 | 13 | +3.21 / +0.100 / +3.56 | 0.0018 | 0.033 |
| 2D features AURC | 6 | 6 | 12 | +2.01 | 0.013 | 0.14 |

- Averaging also beats the trained baselines: NLL 13/14 vs both, ECE 13/14 vs both (Q/review_checks.out E).
- With P0 in place of P2, NLL vs Feature 3DGS is still 13/14, Holm 0.033 (Q/holm.out).
- Per-dataset counts are the main description. The pooled 14-scene test mixes a synthetic and a real dataset; label it a post hoc summary.

### 1b Calibration depends on how it is aggregated
Pooled ECE is computed from the stored 15-bin reliability bins.
- Check: recomputing each scene's ECE from its bins gives the stored ECE for every row and scene of both test files (max error 6e-17).

| Test set | Metric | P2 | P0 | Averaging |
|---|---|---|---|---|
| Replica | ECE, mean of scenes | 8.70 | 9.86 | 9.62 |
| Replica | ECE, pooled over pixels | 7.12 | 5.62 | 5.70 |
| Replica | NLL, pooled over pixels | 0.982 | 0.982 | 0.989 |
| ScanNet | ECE, mean of scenes | 6.06 | 6.36 | 6.17 |
| ScanNet | ECE, pooled over pixels | 2.62 | 3.16 | 3.13 |
| ScanNet | NLL, pooled over pixels | 0.506 | 0.502 | 0.499 |

- Per scene, P0 and averaging show the same pattern as P2: under-confident middle bins and over-confident high bins. Example: office_1 averaging is +31.7 points at confidence 0.27-0.33; office_4 averaging is -29.3 at 0.80-0.87.
- Pooling lets opposite miscalibration in different scenes cancel. Report both aggregations and never claim one alone.

### 1c Object removal (NOR) test, N = 10
Source: Q/nor1c.out. One test (sign test); median paired differences.

| Comparison | Level | Means | Ours-side better | Median diff | Sign p |
|---|---|---|---|---|---|
| fused vs NormLift | G-AP | 75.0 vs 63.8 | 8/10 | +14.7 | 0.109 |
| fused vs depth | G-AP | 75.0 vs 57.4 | 8/10 | +10.9 | 0.109 |
| fused vs feature residual | G-AP | 75.0 vs 56.6 | 9/10 | +24.0 | 0.021 |
| ours vs NormLift | G-AP | 53.4 vs 63.8 | 3/10 | -6.4 | 0.344 |
| ours vs NormLift | P-AP | 57.2 vs 65.8 | 1/10 | -7.7 | 0.021 |
| fused vs NormLift | P-AP | 61.8 vs 65.8 | 2/10 | -4.8 | 0.109 |
| fused vs NormLift, common set | G-AP | 73.0 vs 75.0 | 6/10 | +1.5 | 0.754 |
| ours vs feature residual | FA | 6.2 vs 13.7 | 9/10 | +7.0 | 0.021 |
| ours vs feature residual | Rec | 56.8 vs 75.6 | 1/10 | -14.0 | 0.021 |

- The common set (Gaussians every row scores) is 3.0-18.4% of the observed Gaussians.
- The bootstrap 95% CI of the mean G-AP gain of fused vs NormLift is [-0.3, +20.8] (S/quant_stats.out).
- The fused row is the only row that combines depth with features (decision D5 fixed it before any run). Its baselines were never fused with depth.

### 1d Measured per-view static rate r_static
- Definition: fires / tests on the static observed Gaussians during the update. Code: EC:644-655; field operating_point[r].rate_static of RES/nor/tab2_relmax/{val,test}/0XX.json.
- Units: x1e-3; nominal 1.
- Source: Q/review_checks.out A.

| Row | Test median | Test min-max | Test pooled | Val median | median abs(log10(r/1e-3)) | Ours closer to nominal |
|---|---|---|---|---|---|---|
| NCC | 22.2 | 1.1-204.9 | 32.2 | 32.9 | 1.35 | 9/10 |
| Depth residual | 1.1 | 0.3-9.1 | 2.6 | 1.6 | 0.40 | 3/10 |
| Feature residual | 52.8 | 4.3-277.8 | 71.0 | 68.0 | 1.72 | 10/10 |
| NormLift drop | 4.1 | 0.1-191.0 | 50.7 | 6.6 | 0.90 | 8/10 |
| Dispersion | 8.3 | 3.0-68.4 | 28.9 | 2.3 | 0.91 | 9/10 |
| Ours | 4.0 | 0.5-23.4 | 3.6 | 4.8 | 0.60 | - |

- Ours lies within [0.5x, 2x] of nominal in 1/10 test scenes and 2/4 validation scenes.
- The paper's contribution says "we measure the actual rate", but no number is in the paper yet.

### 1e Build cost
- Scope: ScanNet, median over the 10 scenes, shared RTX 4070 Ti, production logs.
- Sources: the `timing:` lines of RES/scannet_tab1/map_*.log (run_mapping.py:698-709) and the `saved ... train N s` lines of distill_*.log (distill_features.py:153).

| Method | Build time per scene |
|---|---|
| Ours (field update plus the view-hook baselines) | 5.6 s |
| Averaging and NormLift | the same pass: they are the view hooks (paper, controlled timing: +2.3 ms per keyframe) |
| LSeg extraction (every method) | 13.5 s |
| Feature 3DGS, our re-implementation (features only, fixed map) | 1268 s of training |
| LEGaussians, our re-implementation (features only, fixed map) | 730 s of training |

- Contention on the shared GPU is visible in the ms per iteration: Feature 3DGS 37-89, LEGaussians 15-46.
- Claim only that training-free lifting (averaging, NormLift, ours) builds about two orders of magnitude faster than our trained re-implementations.

### 1f Uncertainty ranking (relative AUSE; 1 = random)
Replica test (views_test.json and the Table I JSON, via plot_views.py:169-205):
- u_epi beats u_ale on e_red at M = 10: 6/6 (0.35 vs 0.90).
- u_ale beats u_epi on e_irr at M = 90: 6/6 (0.48 vs 0.83). On the full map: 5/6 (0.42 vs 0.82).
- P2's 1 - max p beats averaging's on e_irr at every M: 6/6 (M = 10), 5/6 (30), 6/6 (90), 6/6 (full map).

ScanNet full map (no new run needed: scannet_tab1/eval_test.json; Q/scannet_fullmap.out):
- u_ale beats u_epi: 8/8 (0.586 vs 0.866).
- u_ale vs NormLift 1 - C_inter: 3/8 (0.586 vs 0.574), a tie.
- P2's 1 - max p vs averaging's: 1/8 (0.315 vs 0.259). The Replica advantage does not transfer.

### 1g Fitted P2 coefficients
Validation fits, read from betas.ours_P2. The formula is usplat/predictive.py:212-216.

| Dataset | beta_0 | gamma_e | gamma_a | gamma_u |
|---|---|---|---|---|
| Replica | 45.6 | +0.36 | +11.8 | +0.055 |
| ScanNet | 34.6 | -0.80 | -2.06 | +0.14 |

- On ScanNet the fit sharpens uncertain pixels. This is the opposite of the intent stated in Sec. III-H, and it fits P2's AURC loss in 8/8 ScanNet scenes.
- The paper does not report these coefficients.

### 1h Test-look ledger (to be disclosed)
| Split | Evaluations so far | Folders |
|---|---|---|
| Replica test | Table I v1; views test (plus the room_0 repeat); Table IV ablations (2026-10-17); Table I v2, the second look (2026-10-19); qual render (room_0) | RES/tab1, views, views_repro, tab3, tab1_v2, qual |
| ScanNet test | Table I v1 and v2 rows in one run (2026-10-19/20) | RES/scannet_tab1 |
| NOR test | Table III; 5 Table IV ablation passes (k1, pooled, maxnull, nodepth, centre); 1 crashed pass (lam05, scene 005) | RES/nor/tab2_relmax, tab2_abl_* |

The reports state that no choice followed a test run. Still, the paper's sentence "we run them once" is false and must change (section 9).

## 2 Statistics and reporting rules
Commit these as configs/quant_prereg.json before any new run.
- **R0. Report everything.** Every analysis that runs is reported, in the main text or the appendix, whatever its outcome.
- **R1. Tests and effect sizes.**
  - The unit is the scene. Use one paired test everywhere: the exact two-sided sign test, ties dropped.
  - Wilcoxon appears only as a complete secondary column, never picked per cell.
  - The effect size is the median paired difference in the metric's own unit. Relative gains appear only as medians.
  - No "primary" test is declared on results already seen.
  - "Significant" means Holm-adjusted p < 0.05 within a declared family:
    - F1: the 20 tests of 1a.
    - F2: ours+depth vs each of the 4 new fused rows, at G-AP and P-AP (8 tests).
    - F3: package-2 claims, each declared in its own prereg before it runs.
  - Always print the per-dataset x/N.
- **R2. Aggregation.** Show the per-scene mean and the pixel-pooled value side by side. Claim a calibration difference only if it holds under both and in x/N.
- **R3. Fused baselines on NOR.**
  - For X in {ncc, feat, normlift, dispersion}: X+depth = fuse_rank([s_X, s_depth], mask=observed). This is the same equal-weight rank average as the paper's fused row (change_baselines.py:522-538; NaN counts as rank 0, :496-518; EC:798).
  - Run validation, then test once. Report every fused row with x/10, the median difference, sign p, and Holm over F2.
  - "Ours fused with depth ranks best" may be written only if its test mean G-AP is the highest of the 5 fused rows. Even then, the sentence carries the x/10 against each fused row.
  - Otherwise, replace the abstract and intro claim "75.0 vs 63.8" with the measured comparison.
- **R4. Fixed threshold on NOR.**
  - theta_fix is the upper edge of the bin where the CDF reaches 0.999 (4000 bins on [0, 2], the rule of vmf_field.py:597-601).
  - The histogram counts the per-view scores c_last[last_tested] of the static (D2) observed Gaussians in the calibrated update passes of the 4 validation scenes, pooled. This is ChangeTally.score_hist (run_mapping.py:150-158) restricted to static Gaussians. Never read c_ring: it is NaN after a fire (vmf_field.py:535).
  - theta_fix uses validation labels (D2); state this.
  - Secondary threshold: the median of the 4 validation pooled calibrated thresholds, 0.1755 (0.167, 0.184, 0.146, 0.208).
  - Test once. Primary metric: the median over test scenes of abs(log10(r_static / 1e-3)), comparing ours at theta_fix with ours at its label-free theta(h). Also report x/10 closer to nominal, FA, Rec, G-AP and Kept.
  - Report the result whichever way it goes.
- **R5. ScanNet views.**
  - M in {10, 30} mapping keyframes, built with run_mapping.py --max-keyframes M --pca-from <full field> --field-config '{"change_test": false}'.
  - Betas from scannet_tab1/eval_val.json. Rows: ours_P0, ours_P2, averaging, normlift, viewcount.
  - The validation scenes only check the pipeline; then run test once.
  - Write "u_epi ranks e_red better than u_ale" only if it holds at M = 10 in at least 7/8 test scenes.
  - Write "u_ale ranks e_irr better than u_epi" at M = 30 only if at least 7/8. The full map is already 8/8.
  - Otherwise write "not replicated".
- **R6. Stale query on NOR (package 2).**
  - Prompts: 001 space heater, 002 office chair, 003 teddy bear, 004 armchair, 005 book, 006 cardboard box, 007 jacket, 008 backpack, 009 laundry basket, 010 space heater, 011 backpack, 013 paper bag, 014 bicycle, 015 bicycle. Use the 5-template LSeg ensemble.
  - Score each pixel by LERF relevancy against the canonical phrases object, things, stuff, texture. A pixel is positive if its relevancy is above 0.5.
  - Held-out frames: update frames that are not update keyframes and have a non-empty masks_real, evenly spaced, at most 20 per scene.
  - Gate, method-independent: LSeg's own 2D relevancy on held-out MAP frames marks at least 20% of the mask pixels.
  - Primary: the stale-query rate of ours after the update vs averaging after the update, x/N and sign p over the gated test scenes. Report every scene.
- **R7. Method changes.** Only through a committed pre-registration, with selection on validation and every candidate reported on test. None is recommended now.
- **R8. Honour the existing pre-registrations:**
  - (a) Report every v2 candidate of both prereg files as "not selected".
  - (b) The paper states the Replica second look (tab1_v2_prereg.json, reporting rule 5).
  - (c) The reducibility criterion (eval_views.py:42-53, 143-144) is validation-only and binding once it runs.
  - (d) configs/static_fa.json tests the outdated default change weight "view" (vmf_field.py:147). A relmax run needs a newly frozen config. Its consequence clause (report the measured rate, not the nominal one) is honoured through 1d.
- **R9. Figures.**
  - Compared panels share axes, bins, colour scales and row sets. Show all scenes.
  - Show per-scene views beside any pooled view.
  - No fitted trend line with N <= 10 unless a CI is shown.
  - PR curves are per scene.
  - Every number is traced in LaTeX %% comments only, never as a visible link.
- **R10. Anonymity.**
  - Runlog JSONs record the host and user name, /home paths and git paths. Never ship or link them; scrub any supplement.
  - Figures and videos carry no names or paths.

## 3 Options

### A. No new runs
| # | Item | Notes |
|---|---|---|
| A1 | Table 1a in the appendix; per-dataset x/N in the text; keep the std sentence in the Table I caption | Holm over F1 |
| A2 | r_static column in Table III (x10^-3, nominal 1) and the text of 1d | Do not call it FA |
| A3 | Cost table: ours / averaging / NormLift (one pass), LSeg, Feature 3DGS and LEGaussians re-implementations | Shared-GPU caveat; never mix with the 5070 Ti Replica fit times |
| A4 | Every v2 grid candidate, labelled "not selected" | Representatives exist in docs/reports/2026-10-17-paper-completion/logs/{tab1_v2_test_rows,scannet_tab1_v2_rows}.tex; the full grid needs quant_stats.py v2grid |
| A5 | Per-scene appendix tables (Table I: 6 + 8 scenes; Table III: 10 scenes) | |
| A6 | Calibration figure: per-scene paired ECE and NLL dots, plus per-dataset pooled reliability diagrams with bin-count histograms and both ECEs | Appendix |
| A7 | NOR per-object table: box dimensions and the G-AP of every row | No scatter, no trend claim |
| A8 | Table III AP on the common set as well | change_baselines.py:480-483 says "Report both" |
| A9 | P2 coefficients (1g) and one sentence on the ScanNet sign; also in Limitations | New |
| A10 | ScanNet full-map point of the views test (1f), in the text or the figure | New |
| A11 | Paper integrity edits (section 9) | New |

### B. New evaluations with a protocol fixed in advance
| # | Item | GPU (measured basis) | Notes |
|---|---|---|---|
| B1 | NOR per-Gaussian dump and the fused baselines (R3) | ~25 min: stage timings of the 14 scenes sum to 23.5 min | One more NOR test evaluation; disclose it |
| B2 | Fixed threshold (R4) | +3 min: the update pass took 1.7-32 s per scene | Same pass as B1 |
| B3 | Stale query (R6) | ~30 min, plus 0.5-1 day of code | Package 2 |
| B4 | ScanNet views (R5) | 2-3 h | Package 2 |
| B5 | Risk-coverage and sparsification curves; must reproduce the saved AURC, ECE and NLL | 20-30 min | Package 2 |
| B6 | Replica static false alarms with a relmax config (new script and configs) | 1-1.5 h | Package 3 |
| B7 | Reducibility verdict, validation only | ~20 min | Package 1 with the user's go-ahead, else disclose that it was not run |
| B8 | Controlled timing | 45 min, idle GPU only | Package 3 |

### C. Method changes
None has validation support:
- Weighted or three-way fusion: credit shared with a baseline, and another NOR test look.
- Constraining gamma >= 0: the unconstrained fit is the validation-NLL optimum, so the pre-registered rule would never select the constrained one.
- lambda > 0: the evaluation refuses it (EC:504-505, run_abl_lam05.log).
- A stronger extractor: lifts every row alike.
- Visibility-aware weights: < 0.3 mIoU expected.
- Swapping in P4: post hoc.

## 4 Package 1 (do now)

**Step 0: pre-register.** Commit configs/quant_prereg.json (R0-R2 and families F1, F2) and configs/nor_fused_prereg.json (R3, R4) before any new run.

**Step 1: CPU, about 1 day of code and about 1 h of runs.**
```
cd /home/thanhnc/gauss_ws/src/uags
export PATH=$HOME/venvs/uags/bin:$HOME/.local/cuda-12.8/bin:$PATH CUDA_HOME=$HOME/.local/cuda-12.8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
L=docs/reports/2026-10-20-figures-video/logs; F=docs/reports/2026-10-20-figures-video/figures
RES=/home/thanhnc/gauss_ws/data/uags_results; NORD=/home/thanhnc/gauss_ws/data/nerf_object_removal/nerf-object-removal
python3 scripts/quant_stats.py seg --replica $RES/tab1/eval_test.json --scannet $RES/scannet_tab1/eval_test.json --legs legs_l4,legs_l1 --prereg configs/quant_prereg.json --json $L/stats_seg.json --tex $L/stats_seg.tex
python3 scripts/quant_stats.py nor --test $RES/nor/tab2_relmax/change_test.json --val $RES/nor/tab2_relmax/change_val.json --per-scene $RES/nor/tab2_relmax --boxes $NORD --json $L/stats_nor.json --tex $L/rstatic.tex
python3 scripts/quant_stats.py views --views $RES/views/views_test.json --tab1 $RES/tab1/eval_test.json --scannet-tab1 $RES/scannet_tab1/eval_test.json --json $L/stats_views.json
python3 scripts/quant_stats.py cost --logs $RES/scannet_tab1 --json $L/cost_scannet.json --tex $L/cost_scannet.tex
python3 scripts/quant_stats.py betas --val $RES/tab1/eval_val.json $RES/scannet_tab1/eval_val.json --tex $L/p2_coeffs.tex
python3 scripts/quant_stats.py v2grid --replica $RES/tab1_v2/eval_test.json --scannet $RES/scannet_tab1/eval_test.json --prereg configs/tab1_v2_prereg.json configs/scannet_v2_prereg.json --tex $L/v2_grid.tex
python3 scripts/plot_calibration.py --replica $RES/tab1/eval_test.json --scannet $RES/scannet_tab1/eval_test.json --rows ours_P2,ours_P0,averaging,f3dgs,@legs,2d_features --out $F/calibration.pdf --json $L/calibration.json
```
Expected: the numbers of section 1, identical to Q/*.out and S/quant_*.out.

Checks (tests/test_quant_stats.py):
- The sign test equals scipy.stats.binomtest; Holm on a toy example.
- Per-scene ECE recomputed from the bins equals the stored ECE for every row of both eval_test.json files.
- Every LaTeX cell traces to a JSON key.

**Step 2: GPU validation stage, gate 6500 MiB (needs 8.5 GB free).**
```
# noise floor: EC unchanged, new output (never overwrite Table III records)
scripts/gpu_gate.sh 6500 python3 scripts/eval_change_nor.py --field /home/thanhnc/gauss_ws/data/uags_fields/nor_001_lseg_relmax.pt --align none --update-stride 2 --tau-cov 0.5 --out $RES/nor/dump/noise/001_repeat.json
python3 scripts/nor_fused_report.py noise --a $RES/nor/tab2_relmax/val/001.json --b $RES/nor/dump/noise/001_repeat.json --out $RES/nor/dump/noise/tolerance.json
python3 scripts/nor_tab2.py scenes $RES/nor/tab2_relmax/config_val.json | while read -r s data st ust al ply field; do
  scripts/gpu_gate.sh 6500 python3 scripts/nor_dump_scores.py --stage val --field "$field" --align "$al" --update-stride "$ust" --tau-cov 0.5 \
    --check $RES/nor/tab2_relmax/val/$s.json --tolerance $RES/nor/dump/noise/tolerance.json --dump $RES/nor/dump/val/$s.npz --out $RES/nor/dump/val/$s.json; done
python3 scripts/nor_fused_report.py theta --dir $RES/nor/dump/val --prereg configs/nor_fused_prereg.json --out $RES/nor/dump/theta_fix.json
```
- The tolerance is max(0.002, 3 x the largest difference of any Table III cell between the two 001 records).

**Step 3: GPU test stage, run once.**
```
python3 scripts/nor_tab2.py scenes $RES/nor/tab2_relmax/config_test.json | while read -r s data st ust al ply field; do
  scripts/gpu_gate.sh 6500 python3 scripts/nor_dump_scores.py --stage test --field "$field" --align "$al" --update-stride "$ust" --tau-cov 0.5 \
    --theta-fix $RES/nor/dump/theta_fix.json --check $RES/nor/tab2_relmax/test/$s.json --tolerance $RES/nor/dump/noise/tolerance.json \
    --dump $RES/nor/dump/test/$s.npz --out $RES/nor/dump/test/$s.json; done
python3 scripts/nor_fused_report.py table --dir $RES/nor/dump/test --val-dir $RES/nor/dump/val --prereg configs/nor_fused_prereg.json --json $L/nor_fused_test.json --tex $L/nor_fused_test.tex
```
Expected outputs:
1. Table III cells reproduced per scene within the tolerance (G and P AP/AUROC with NaN last, FA, Rec, for EC.ROWS). Stop if any cell fails.
2. The 4 new fused rows: G and P AP/AUROC, per scene and as mean ± std, x/10 vs ours+depth, median difference, sign p, Holm over F2, and the R3 claim verdict.
3. Common-set AP for each new fused row paired with ours+depth.
4. R4 table: r_static, abs(log10) distance to nominal, FA, Rec, G-AP and Kept of ours at theta_fix, at 0.1755 and at theta(h).
5. Per-scene Gaussian-level PR curves (appendix).

**Step 4: GPU, B7, validation only (only after the user's go-ahead; its verdict binds).**
```
OUT=$RES/views_reduc STAGE=val KS=10 MAP_NEED=6500 EVAL_NEED=5000 scripts/replica_views.sh
python3 scripts/eval_views.py verdict $RES/views_reduc/views_val_office_0.json $RES/views_reduc/views_val_room_2.json --out $RES/views_reduc/reducibility_val.json
```
- MAP_NEED: the measured k-field peak is 2.2 GB allocated and 6.27 GB reserved (RES/repro.log).
- EVAL_NEED 5000: record the actual peak; on an out-of-memory error, rerun at 7000.
- If the user declines, the report and the paper say the criterion was registered on 2026-10-10 and not run.

**Step 5: decide with the user.** Apply R3, R4 and the B7 verdict to the text; choose package 2.

## 5 Later packages
**Package 2** (user decides; about 3-4 h of GPU wall time):
- B4 via scripts/scannet_views.sh.
- B5 via scripts/dump_curves.py.
- B3 via scripts/nor_stale_query.py plus configs/nor_query_prereg.json.

**Package 3:**
- B6 via a new scripts/static_fa_relmax.sh and configs/static_fa_relmax*.json. These are new files: replica_static_fa.sh hard-codes FROZEN at line 50 and uses MAP_NEED 10000. Config: change_weight relmax, fast_min = slow_min = 0.5, frozen before the first run.
- B8: controlled timing, on an idle GPU only.

## 6 Code plan

### Do-not-edit list
- The 9 files hashed by EC.code_state (EC:191-193, 241-248).
- MAPPING_CODE (run_mapping.py:378-381), which includes usplat/vmf_field.py and baselines.py.
- New scripts import these files; they never change them.

### scripts/quant_stats.py (CPU)
Subcommands: seg | nor | views | cost | betas | v2grid.

| Input | Where it comes from |
|---|---|
| Table I metrics | metrics.scenes[s].rows[r].all, with keys miou, ece, nll, brier, aurc, n_pixels, reliability{count,conf,acc} and ranking[sig].ause (seg_eval.py:318-328, 366-402) |
| NOR per-scene values | RES/nor/tab2_relmax/{val,test}/0XX.json, metrics.operating_point[r].{rate_static, tests_static, fires_static, fa, rec}; the change_*.json merges give scenes[s].rows[r].{g_ap, p_ap, g_ap_common, ...} |
| Views | plot_views.py: random_level :129-136, views_relative :169-186, full_relative :189-205 |
| Timing | run_mapping.py:698-709 and the distill log lines (distill_features.py:153) |
| Boxes | usplat/box3d.py read_box_corners |

- Statistics: scipy.stats.binom for the sign test, Holm, and a scene bootstrap (20,000 resamples, seed 0) for the CIs only.
- Output: JSON and LaTeX rows.

### scripts/plot_calibration.py (CPU)
- Matplotlib, Agg backend, the serif font settings of render_qual.py.
- Panel 1: per-scene paired dots of ECE and NLL.
- Panel 2: pooled reliability diagrams, identical 15 bins, with count histograms and both ECEs in the legend.

### scripts/nor_dump_scores.py (GPU)
Arguments: EC.build_parser() (EC:890-920) plus --stage, --check, --tolerance, --theta-fix and --dump.

Steps, in order:
1. `gpu_reserve.apply(dev)`, as EC:768-770 does.
2. `ctx = EC.load_context(args)` (EC:697-742).
3. Mapping replay:
   - `k = int(args.k or ckpt["field"]["config"]["change_persist"])` (EC:775).
   - `EC.replay_mapping(ctx, ctx["map_views"], k, len(ctx["upd_views"]))` (EC:423-452).
   - `EC.replay_check` (EC:455-469), with the same stop and warn rules as EC:780-783.
   - `thr = {r: nl.threshold(q)}` (EC:392-410).
4. `gt, gt_any, shell, info = EC.ground_truth(ctx, args)` (EC:745-761), run before the update so the tally can split static from stale.
5. Tally, inside this process only:
   - Wrap DSF.update_view. After the original call, if `self.cfg.change_test and (self.thr_by_hist is not None or math.isfinite(self.cfg.change_thr))`, then:
     - bincount `(c_last[last_tested & static]*2000).long().clamp(max=3999)` into hist_static, and the same for stale;
     - record the first-fire view where `last_fired` holds.
   - This mirrors ChangeTally.step (run_mapping.py:150-158).
6. `upd = EC.run_update(ctx, ctx["upd_views"], k, thr, q)` (EC:472-547), unchanged.
7. Table III check: repeat EC:797-830 with EC.ROWS only:
   - `fused = fuse_rank([ours, depth], mask=pop)`;
   - `ranking_table` (EC:620-628);
   - `operating_points` and `realised_rates` (EC:635-655);
   - `pixel_level` (EC:659-684).
   - Compare with the saved record (gaussian.rows[r].ap/auroc, pixel.rows[r].ap/auroc, operating_point[r].fa/rec) within the tolerance. Exit on any mismatch.
8. New rows: X+depth via fuse_rank (change_baselines.py:522-538). Use separate ranking_table and pixel_level calls on (fused plus the 4 new rows), so the original common sets stay untouched.
9. Test stage with --theta-fix only:
   - patch `DSF.calibrate_change_threshold = lambda self, q=None: self.set_change_threshold(theta)` (vmf_field.py:626-634);
   - rerun EC.run_update;
   - then operating_points, realised_rates, ours' G and P AP, and Kept;
   - restore the method.
10. Save an npz: scores of every row (float32), flags, tests, fires, observed, seen, stale, in_box, n_views, changed_end, robust_end, first_fire, hist_static and hist_stale (for both passes).
11. Save a runlog record: EC.code_state(), RM.mapping_code_state() (run_mapping.py:384-399), the sha256 of the field and the map, settings, timing and peak memory.
12. Guards on --stage test. Refuse if:
    - `git status --porcelain scripts usplat configs` is non-empty;
    - configs/nor_fused_prereg.json is uncommitted;
    - theta_fix.json is missing;
    - a finished test record exists (run once).

### scripts/nor_fused_report.py (CPU)
- noise: the largest difference of any Table III cell between two records, written as the tolerance.
- theta: sums the validation hist_static and applies R4.
- table: per scene and mean ± std, x/10, median differences, sign p, Holm over F2, PR curves and the R3/R4 verdicts.

### Package 2-3 scripts
**scripts/scannet_views.sh** is a copy of replica_views.sh :61-95:
- dataset scannet, --stride 10 for mapping;
- scene entries from configs/scannet_{val,test}.json (data uags_f2/<scene>, stride 20, offset 5; eval_seg.py:190 reads the offset);
- configs/vocab_scannet.json and betas from scannet_tab1/eval_val.json;
- MAP_NEED 6500 (measured: 2.84 GB allocated, 6.28 GB reserved), EVAL_NEED 7000.
- eval_views.py:436 reads integer ks.

**scripts/dump_curves.py** imports eval_seg and accumulates with SegAccumulator (seg_eval.py:405-433). Curves come from evaluation.py sparsification_curve (:51-81) and aurc (:150-157). Check: the recomputed AURC, ECE and NLL equal the Table I JSON.

**scripts/nor_stale_query.py** follows R6:
- load with EC.load_context;
- ours via DSF.from_state plus calibration; NormLift and averaging via EC.make_normlift (EC:349-358) and the averaging feature (baselines.py:206-210);
- text via get_extractor("lseg").text_embedding (extractors/lseg.py:282-290) and pca.project_text (feature_pca.py:59-66);
- render with render_row (seg_eval.py:109);
- held-out frames chosen with time_field.evenly (:157).
- Flag counts must match the saved ones within the measured tolerance, not exactly.

### Smoke tests (CPU)
- tests/test_nor_dump.py: on the mock-backend toy of tests/test_eval_change.py, the 7 dumped rows equal EC.evaluate's, and the tally histogram equals ChangeTally's.
- tests/test_quant_stats.py.

## 7 GPU estimate
gpu_gate.sh waits until (need + 2048 MiB) is free; 3.9 GB of 12.3 GB was in use at the check.

| Package | Item | GPU compute | Gate (MiB) |
|---|---|---|---|
| 1 | Noise repeat | ~3 min (val 001: 163 s) | 6500 |
| 1 | B1 + B2, validation | ~10 min (609 s summed) | 6500 |
| 1 | B1 + B2, test | ~15 min (800 s summed, plus one update pass) | 6500 |
| 1 | B7 (optional) | ~20 min | 6500 / 5000 |
| 2 | B4 | 2-3 h | 6500 / 7000 |
| 2 | B5 | 20-30 min | 7000 |
| 2 | B3 | ~30 min | 6500 |
| 3 | B6 | 1-1.5 h | from measured peak |
| 3 | B8 | 45 min | idle GPU only |

Package 1 is about 30-50 min of compute; wall time depends on the gate.

## 8 Risks
1. **Headline NOR claim.** NormLift+depth may reach or pass 75.0. The R3 rewrite is then mandatory.
2. **B7 verdict.** It may drop the u_epi claim, and it binds once run.
3. **Paper edits.** The edits in section 9 weaken the abstract's calibration claim and the "run once" wording, but they are required.
4. **GPU noise.** A Table III cell may fail the check. Investigate before the test stage; never relax the tolerance after looking at test.
5. **Monkeypatching** affects only the dump process. The checks at EC:504-505 and 536-537 still run.
6. **Multiplicity.** Only Holm-adjusted family tests may be called significant.
7. **Timing** comes from a shared GPU. Give medians and the caveat, or run B8.
8. **ScanNet views (B4)** may not replicate on e_red. The P2 coefficients are negative there.
9. **Disk.** About 1 GB of npz dumps; 428 GB are free.
10. **Anonymity.** Records carry host and user paths; R10 applies.

## 9 Paper edits (after package 1)
- **Abstract and conclusion.** Replace "our predictive is the best calibrated open-vocabulary map, on ScanNet by a small margin" with: as well calibrated as averaging, and better calibrated than trained open-vocabulary maps (NLL lower than our Feature 3DGS re-implementation in 13 of 14 test scenes, Holm-adjusted p 0.03). Training-free lifting shares this.
- **Intro.** The fused claim "(Gaussian-level AP 75.0 vs 63.8)" stays only under R3.
- **Setup.** Replace "we run them once" with a statement of the test looks in 1h, including the Replica second look of the pre-registered v2 predictives and the NOR ablation passes. No choice changed after a test run.
- **Sec. V-D.** Add r_static: median 4.0x10^-3 (0.5-23x10^-3) at q = 0.999; within a factor of 2 of nominal in 1 of 10 test scenes; closer to nominal than the feature residual (10/10) and NormLift (8/10), less close than the depth residual (3/10). Add the Table III column.
- **Sec. III-H or V-C, and Limitations.** Report the fitted gamma_e, gamma_a, gamma_u per dataset, and that the ScanNet fit has negative gamma_e and gamma_a.
- **Sec. V-C.** Add the ScanNet full-map ranking (u_ale beats u_epi 8/8; P2's 1 - max p loses to averaging 1/8). Fig. views states that its curves are Replica only.
- **Appendix.** Table 1a, per-scene tables, the v2 grid ("not selected"), the calibration figure, the per-object NOR table, and PR curves.
