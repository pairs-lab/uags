# Quantitative-results design for the UAGS paper, in the spec field below. It contains: (1) numbers computed now on the CPU from the saved result JSONs and logs, (2) options ranked in three classes (A: no new runs, B: new evaluations with a fixed protocol, C: method changes), (3) a first package to run now, with scripts, commands, checks and expected outputs, (4) later packages, (5) the code plan with every reused code path cited as file:line, (6) a GPU estimate and (7) risks. The evidence is in scratch files only; nothing in the repo or data folders was changed and no GPU job ran. Scratch scripts and their outputs: /tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/quant_stats.py with quant_stats.out (paired per-scene tests for Table I, NOR and the views experiment), and /tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design/quant_extra.py with quant_extra.out (pixel-pooled calibration, measured false-alarm rates, pooled 14-scene tests, per-object NOR table, build-time table, power analysis).

The saved results already contain stronger and more honest quantitative claims than the paper makes. Across the 14 Replica and ScanNet test scenes, UAGS is better calibrated than the trained baselines: NLL is lower than Feature 3DGS in 13 of 14 scenes (sign test p = 0.002) and every metric beats LEGaussians in at least 13 of 14 scenes. Building the field takes about 5.6 s per ScanNet scene, against 21 and 12 minutes of training for those two baselines. On static content our label-free threshold fires on 6.3 per 1000 tests (6x the nominal 1 per 1000), against 42 to 86 for NormLift and the feature residual. Against averaging, which uses the same LSeg features, UAGS only ties, and two current claims are weak. First, the paper's best change-detection result (ours fused with depth, AP 75.0 vs 63.8) compares a depth-plus-features row with feature-only rows; NormLift and the feature residual were never fused with the same depth score. Second, the Replica ECE gain reverses when ECE is pooled over all pixels instead of averaged per scene (P2 7.1 vs P0 5.6). First package: CPU-only statistics, tables and figures, plus one 40-minute GPU run on the object-removal scenes that adds the missing fused baselines and tests a fixed threshold carried over from validation. No method change is recommended, because no candidate has validation evidence for a real gain.

# Quantitative results: design (2026-10-20, read-only phase)

Short names used below:
- **R** = /home/thanhnc/gauss_ws/src/uags (repo)
- **RES** = /home/thanhnc/gauss_ws/data/uags_results
- **EC** = scripts/eval_change_nor.py (the object-removal evaluation)
- **NOR** = the NeRF-Object-Removal scenes
- **S** = /tmp/claude-1000/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/design

Every number below was computed on the CPU from the saved JSONs and logs, with these commands:
```
~/venvs/uags/bin/python S/quant_stats.py > S/quant_stats.out
~/venvs/uags/bin/python S/quant_extra.py > S/quant_extra.out
```
Throughout, "x/N" means our row is better in x of N scenes, and "p" is the exact two-sided sign test over scenes unless marked otherwise.

## 0 Bottom line
1. **Accuracy cannot move much.** Every lifting row uses the same LSeg features. Ours vs averaging on mIoU: +0.3% relative, 8/14 scenes, p = 0.79.
   - Adding scenes will not help. The ScanNet ECE and NLL gaps have a paired effect size of dz = 0.13. Reaching 80% power would need about 460 scenes.
2. **Robust today, with no new run:**
   - better calibration than the trained baselines and the 2D features;
   - a build cost 130-225x lower;
   - a measured false-alarm rate far closer to nominal than the other feature rows;
   - P2's total uncertainty ranks irreducible errors better than averaging's (6/6 Replica scenes).
3. **Must be fixed before claiming more:**
   - The headline NOR result compares our score fused with depth against feature-only rows. Fusing NormLift or the feature residual with the same depth score was never run.
   - The Replica ECE claim depends on how ECE is aggregated (per-scene mean vs pooled over pixels).
4. **No method change has validation support** (section 2C).

## 1 What the saved results already say

### 1a Ours vs each baseline, 14 test scenes (Replica 6 + ScanNet 8)
Mean per-scene relative gain (positive = ours better), x/14 scenes, sign test p.

| Baseline | mIoU (P0) | ECE (P2) | NLL (P2) | Brier (P2) | AURC (P2) |
|---|---|---|---|---|---|
| Feature 3DGS | +2.9%, 9/14, 0.42 | +24.5%, 11/14, 0.057 (Wilcoxon 0.009) | **+9.6%, 13/14, 0.002** | +6.9%, 12/14, 0.013 | +14.7%, 12/14, 0.013 |
| LEGaussians | +12.5%, 14/14, 1e-4 | +29.2%, 14/14, 1e-4 | +17.9%, 13/14, 0.002 | +14.7%, 13/14, 0.002 | +22.3%, 13/14, 0.002 |
| 2D features | +8.7%, 11/14, 0.057 | +34.6%, 13/14, 0.002 | +17.1%, 13/14, 0.002 | +15.4%, 13/14, 0.002 | +28.4%, 12/14, 0.013 |
| Averaging | +0.3%, 8/14, 0.79 | +11.4%, 6/14, 0.79 | +0.8%, 9/14, 0.42 | +0.9%, 10/14, 0.18 | −4.5%, 6/14, 0.79 |

Per dataset, ours vs averaging:
- **Replica:** P2 has lower Brier and AURC in 6/6 scenes (p = 0.031, the smallest p possible with 6 scenes). P2 NLL 5/6, P2 ECE 3/6 (p = 1.0). P0 has lower NLL and Brier in 6/6.
- **ScanNet:** P2 ECE 3/8, NLL 4/8, Brier 4/8; P0 mIoU 5/8. P2's AURC is worse than both P0 and averaging in all 8 scenes (p = 0.008).

Aleatoric uncertainty (AUSE of u_ale) on Replica and ScanNet:
- vs NormLift 1−C_inter: 3/6 and 3/8 (a tie);
- vs LEGaussians u: 6/6 and 7/8;
- vs NormLift 1−R: 6/6 and 8/8.

On ScanNet, u_ale ranks errors worse than our own 1 − max p (AUSE 8.2 vs 3.8).

### 1b The ECE claim depends on aggregation
Pixel-pooled ECE comes from the stored reliability bins. Recomputing each scene's ECE from those bins gives exactly the stored per-scene ECE, so the bins are read correctly.

| Test set | Metric | P2 | P0 | Averaging |
|---|---|---|---|---|
| Replica | ECE, mean of scenes | 8.70 | 9.86 | 9.62 |
| Replica | ECE, pooled over pixels | 7.12 | 5.62 | 5.70 |
| Replica | NLL, pooled over pixels | 0.982 | 0.982 | 0.989 |
| ScanNet | ECE, mean of scenes | 6.06 | 6.36 | 6.17 |
| ScanNet | ECE, pooled over pixels | 2.62 | 3.16 | 3.13 |
| ScanNet | NLL, pooled over pixels | 0.506 | 0.502 | 0.499 |

On ScanNet, pixel-pooled accuracy is 86.64 for ours vs 86.75 for averaging.

Why P2 differs: its per-pixel temperature fixes per-scene over- and under-confidence. But within a scene it spreads the confidence too much: confidence bins 0.3–0.4 are under-confident by 14–16 points, and bins 0.75–0.85 are over-confident by 13–22 points.

### 1c Object removal (NOR) test, N = 10
- **Fused (ours + depth) vs NormLift, G-AP 75.0 vs 63.8:** 8/10, sign p 0.11, Wilcoxon p 0.105, bootstrap 95% CI of the mean gain [−0.3, +20.8]. Not significant.
- **Fused vs depth alone:** 8/10, Wilcoxon p 0.049.
- **Fused vs feature residual:** 9/10, p 0.021.
- **Ours vs NormLift, pixel AP 57.2 vs 65.8:** NormLift is better (ours 1/10, p 0.021).
- **Ours vs feature residual at the label-free threshold:** false-alarm rate (FA) lower in 9/10 (6.2 vs 13.7%, p 0.021), recall lower in 9/10 (56.8 vs 75.6%).
- **On the set where every row is defined** (2–17% of the observed Gaussians), the fused row's G-AP is 73.0 vs NormLift 75.0.

### 1d Measured per-view static false-alarm rate (nominal 1 per 1000)
The paper says "we measure the actual rate" but does not report it. Values are fires per 1000 tests on static Gaussians.

| Row | Test mean / median / max | Test scenes within [0.5x, 2x] of nominal | Val mean |
|---|---|---|---|
| Ours | 6.3 / 4.0 / 23.4 | 1/10 | 6.1 |
| Depth residual | 2.1 / 1.1 / 9.1 | 3/10 | 1.6 |
| Dispersion 1−ρ̂ | 21.2 / 8.3 / 68.4 | 0/10 | 2.6 |
| NormLift drop | 42.0 / 4.1 / 191 | 0/10 | 15.9 |
| NCC | 58.4 / 22.2 / 205 | 1/10 | 31.0 |
| Feature residual | 85.8 / 52.8 / 278 | 0/10 | 55.4 |

### 1e Build cost per ScanNet scene (median of 10)
RTX 4070 Ti, shared GPU, taken from the production logs:
- **Ours:** 5.6 s for the field update, plus 13.5 s of LSeg extraction that every method needs.
- **Feature 3DGS:** 1268 s of training (225x ours).
- **LEGaussians:** 730 s of training (130x ours).

Sources: `timing:` lines of RES/scannet_tab1/map_*.log (written at scripts/run_mapping.py:698-709) and `train N s` lines of the distill_*.log files.

### 1f Other robust points
- **Views experiment (Replica test):** P2's 1 − max p ranks irreducible errors (e_irr) better than averaging's at M = 10, 90 and 180 (6/6, p 0.031). u_epi beats u_ale on reducible errors (e_red) at M = 10 (6/6). u_ale beats u_epi on e_irr at M = 90 (6/6).
- **NOR, per object:** the depth residual's G-AP rises with object volume (Spearman +0.60, p 0.07). On the thin book (005), depth scores 23.2 vs 92.8 for the feature residual, so the two signals complement each other. Fusion still fails there (53.5).

## 2 Options, ranked

### A. No new runs
| # | Item | Value | Effort |
|---|---|---|---|
| A1 | Paired comparison table (1a), one per dataset plus pooled. One pre-declared primary test per claim; every other p shown as descriptive. Replaces the "std exceeds the gaps" caption | high | 1 h |
| A2 | Measured false-alarm column (1d) in Table III; reword "targets 10^-3" as "measured 6.3×10^-3 (nominal 10^-3)" | high | 30 min |
| A3 | Cost table (1e): ours online and training-free vs Feature 3DGS and LEGaussians training, same GPU. State the caveat: shared GPU, production logs | high | 30 min |
| A4 | Appendix table of every pre-registered v2 candidate, labelled "not selected", for both datasets. This is required by the "reporting" rule at configs/scannet_v2_prereg.json:48 and is not done yet | medium (credibility) | 15 min |
| A5 | Per-scene appendix tables: Table I for 6 + 8 scenes, Table III for 10 scenes | medium | 1 h |
| A6 | Reliability diagrams, pooled over pixels, with both ECEs in the legend and the text of 1b | medium (risky, see 1b) | 1 h |
| A7 | NOR per-object panel (1f): object size vs AP for depth and feature rows | medium | 1 h |
| A8 | Report Table III AP on the common set too, as EC's own docstring says (change_baselines.py:480-491) | low | 15 min |

### B. New evaluations with a protocol fixed in advance
None of these selects or tunes anything. Each new look at a test split is disclosed.

| # | Item | Why it is not tuning | GPU | Likely outcome |
|---|---|---|---|---|
| **B1** | Save per-Gaussian scores for NOR; add fused baselines "X + depth" for X in {NCC, feature residual, NormLift drop, dispersion}, using the same equal-weight rank average as the paper's fused row (change_baselines.py:522-538, EC:798) | Required fair baselines (strongest fair baseline rule); nothing is fitted; second look at NOR test | 30 min for 14 scenes (the original runs took 26.4 min) | NormLift + depth could match or beat 75.0; the headline claim may change |
| **B2** | Fixed-threshold test on NOR: one threshold θ_fix carried over from validation vs our mapping-pass rule | Directly tests the threshold contribution; θ_fix comes from validation only | +10 min | Uncertain: per-scene thresholds already range 0.10–0.36 on test |
| B3 | Downstream stale-query test on NOR held-out update frames: does the map still answer the removed object's prompt? Compares the map before the update, averaging after the update, ours after the update, our robust set, an oracle reset and 2D features | Frozen method; prompts and metric fixed now | 30 min + 0.5–1 day of coding | Directly tests the paper's motivation; likely favours reset over no reset |
| B4 | Views experiment on ScanNet, M ∈ {10, 30}, as replica_views.sh does | No selection; second look at ScanNet test | about 2 h (eval_views took 423 s per 90 Replica frames; ScanNet has 840 test and 316 val frames) | May not replicate: ScanNet's fitted P2 coefficients are negative (a = −0.80, b = −2.06) |
| B5 | Save risk-coverage and sparsification curves for Table I's pixels; add risk at 50/80/90% coverage | Evaluation only; must reproduce the saved AURC | 20 min | P2 loses on ScanNet (0/8); show it as is |
| B6 | Replica static false-alarm test: Sequence_2 has no change, so every fire is a false alarm. Needs a new frozen config with the "relmax" change weight, because configs/static_fa.json (frozen 2026-10-10) predates that decision and would test the old default "view" weight (vmf_field.py:147) | Measures false alarms only | about 1 h | Probably several times nominal, as on NOR |
| B7 | Run the reducibility check pre-registered on 2026-10-10 and never run (eval_views.py:41-53, 142-144): validation first, test only if the claim is kept | Pre-registered; its verdict binds once it runs | 25 min (+70 min test) | May drop the "u_epi ranks reducible error" claim. Decide with the user |
| B8 | Controlled timing benchmark on one validation scene with no other GPU job | Timing only | 45 min | Only if reviewers ask |

### C. Method changes
Each would need a committed pre-registration file, selection on validation only, and a disclosed extra test look.

| # | Idea | Validation evidence on disk | Expected gain | Integrity cost | Verdict |
|---|---|---|---|---|---|
| C1 | Weighted or three-way fusion (ours + depth + NormLift) | None until the B1 dumps; validation has only 4 scenes | 0 to +3 G-AP | Third NOR test look; credit shared with a baseline | no |
| C2 | Constrain P2 coefficients to γ_e, γ_a ≥ 0 | ScanNet validation fit gave a = −0.80, b = −2.06 (Replica: +0.36, +11.8) | ScanNet AURC 5.0 → about 4.4, ECE 6.06 → about 6.3: net zero | The ScanNet test gates are already spent ("no rerun") | no |
| C3 | Recent-bucket retention λ > 0 | The NOR evaluation refuses it: the coverage check at EC:504-505 raised an error for λ = 0.5 (run_abl_lam05.log), so the evaluation would have to change | unknown | Changes the evaluation | no |
| C4 | Stronger feature extractor | Lifts every row the same way (shared features) | no change in ranking | 15–20 h GPU | not now |
| C5 | Visibility-aware evidence weights | Pixels-vs-views ablation moved mIoU by only 0.1 | < +0.3 mIoU | New test looks on both datasets | no |
| C6 | Use the vote predictive P4 (best test ECE) | Lost on validation NLL under the pre-registered rule | n/a | Post-hoc swap, forbidden | no |

## 3 First package (do now)

**Step 0 — pre-register.** Commit configs/nor_fused_prereg.json with rules R3 and R4 from the selection_rules field, before any run.

**Step 1 — CPU work (about 3 h).** Set up:
```
cd /home/thanhnc/gauss_ws/src/uags
export PATH=$HOME/venvs/uags/bin:$HOME/.local/cuda-12.8/bin:$PATH CUDA_HOME=$HOME/.local/cuda-12.8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
L=docs/reports/2026-10-20-figures-video/logs
RES=/home/thanhnc/gauss_ws/data/uags_results
```
Then:
```
python3 scripts/quant_stats.py seg --replica $RES/tab1/eval_test.json --scannet $RES/scannet_tab1/eval_test.json --legs legs_l4,legs_l1 --primary nll:f3dgs --json $L/stats_seg.json --tex $L/stats_seg.tex
python3 scripts/quant_stats.py nor --test $RES/nor/tab2_relmax/change_test.json --val $RES/nor/tab2_relmax/change_val.json --boxes /home/thanhnc/gauss_ws/data/nerf_object_removal/nerf-object-removal --json $L/stats_nor.json --tex $L/fa_realised.tex
python3 scripts/quant_stats.py views --views $RES/views/views_test.json --tab1 $RES/tab1/eval_test.json --json $L/stats_views.json
python3 scripts/quant_stats.py cost --logs $RES/scannet_tab1 --json $L/cost_scannet.json --tex $L/cost_scannet.tex
python3 scripts/plot_reliability.py --replica $RES/tab1/eval_test.json --scannet $RES/scannet_tab1/eval_test.json --rows ours_P2,ours_P0,averaging,f3dgs,@legs --out papers/figs/reliability.pdf --json $L/reliability.json
python3 scripts/make_tables.py tab1 $RES/scannet_tab1/eval_test.json --val $RES/scannet_tab1/eval_val.json --v2 --prereg configs/scannet_v2_prereg.json > $L/scannet_v2_rows.tex
python3 scripts/make_tables.py tab1 $RES/tab1_v2/eval_test.json --val $RES/tab1_v2/eval_val.json --v2 > $L/replica_v2_rows.tex
```
Expected: the numbers of section 1, identical to S/quant_stats.out and S/quant_extra.out.

Checks:
- tests/test_quant_stats.py: the exact sign-test p on toy data equals scipy's binomial test; per-scene ECE recomputed from the bins equals the stored ECE (this already holds for all 6 Replica scenes).
- Every LaTeX cell traces back to a JSON key.

**Step 2 — GPU work (about 40 min, gate 6500 MiB).**

Validation stage:
```
python3 scripts/nor_tab2.py scenes $RES/nor/tab2_relmax/config_val.json | while read -r s data st ust al ply field; do
  scripts/gpu_gate.sh 6500 python3 scripts/nor_dump_scores.py --field "$field" --align "$al" --update-stride "$ust" --tau-cov 0.5 \
    --check $RES/nor/tab2_relmax/val/$s.json --out $RES/nor/dump/val/$s.npz --record $RES/nor/dump/val/$s.json; done
python3 scripts/nor_fused_report.py theta --dir $RES/nor/dump/val --prereg configs/nor_fused_prereg.json --out $RES/nor/dump/theta_fix.json
```
Test stage (run once):
```
python3 scripts/nor_tab2.py scenes $RES/nor/tab2_relmax/config_test.json | while read -r s data st ust al ply field; do
  scripts/gpu_gate.sh 6500 python3 scripts/nor_dump_scores.py --field "$field" --align "$al" --update-stride "$ust" --tau-cov 0.5 \
    --theta-fix $RES/nor/dump/theta_fix.json --check $RES/nor/tab2_relmax/test/$s.json --out $RES/nor/dump/test/$s.npz --record $RES/nor/dump/test/$s.json; done
python3 scripts/nor_fused_report.py table --dir $RES/nor/dump/test --val-dir $RES/nor/dump/val --prereg configs/nor_fused_prereg.json --json $L/nor_fused_test.json --tex $L/nor_fused_test.tex
```
Expected outputs:
1. The 7 Table III rows reproduced per scene, |ΔAP| ≤ 0.002. If not, stop.
2. G and P AP / AUROC for ours + depth (75.0 / 61.8, a reproduction check) and for the four new fused rows, each with x/10 wins vs ours + depth and p.
3. AP on the set where each pair of rows is defined.
4. FA, recall and measured static rate of ours at θ_fix vs at the mapping-pass threshold.
5. Gaussian-level PR curves.
6. Optional: views until first detection.

**Step 3 — decision point with the user.** Rewrite the NOR claims under rule R3 (selection_rules). Then choose package 2.

**Paper edits from package 1:**
- Table I: add a paired panel (or a new table) from A1.
- Table III: add the measured false-alarm column and the fused baseline rows.
- New cost table.
- Appendix: per-scene tables, v2 candidate rows, reliability figure.
- Text: "significantly better calibrated than the trained baselines (NLL lower than Feature 3DGS in 13/14 test scenes, p = 0.002)" and "on par with averaging".

## 4 Later packages
- **Package 2 (about 3 h GPU):** B3 (stale-query test), B4 (ScanNet views), B5 (curves).
- **Package 3 (decide with the user):** B6 (Replica static false alarms, new frozen config), B7 (reducibility verdict), B8 (controlled timing).

## 5 Code plan
Rule for all new code: never edit EC. Its code_state() (EC:191-193, 241-248) hashes the file into Table III's provenance. New scripts import EC instead.

### scripts/quant_stats.py (CPU)
- Subcommands: seg | nor | views | cost.
- Reads metrics.scenes[s].rows[r].all and its reliability and ranking entries (written at seg_eval.py:318-328 and 366-408).
- Reads NOR per-scene metrics: operating_point[r].rate_static, gaussian.rows, pixel.rows.
- Computes the relative AUSE of the views experiment by importing scripts/plot_views.py (random_level at :129-137, views_relative at :169-187, full_relative at :189-205).
- Reads run_mapping timing (scripts/run_mapping.py:698-709) and distill logs (scripts/distill_features.py:104-143).
- Uses the box helper usplat/box3d.py (read_box_corners).
- Statistics: scipy.stats.binom and wilcoxon (method="exact"), scene bootstrap with 20,000 resamples, seed 0.
- Output: JSON plus LaTeX rows.

### scripts/plot_reliability.py (CPU)
- Pools the stored bins over scenes (counts, mean confidence, accuracy).
- Matplotlib with the Agg backend; serif fonts as render_qual.py sets them.
- Legend shows both ECEs.

### scripts/nor_dump_scores.py (GPU)
Takes EC's parser arguments plus --check, --theta-fix, --out and --record. Steps, in order:
1. `held = gpu_reserve.apply(dev)`, as EC does at EC:768-770.
2. `ctx = EC.load_context(args)` (EC:697-742).
3. `gt = EC.ground_truth(ctx, args)` (EC:745-761), computed first because the instrumentation needs the static mask.
4. Mapping replay: `EC.replay_mapping` (EC:423-452), then `EC.replay_check` (EC:455-469), then thresholds from `Null.threshold(q)` (EC:392-410).
5. Instrumentation, inside this process only: a class-level wrapper on `DirectionalSemanticField.update_view`. For the calibrated update copy (cfg.change_test and a set threshold) it records:
   - a 4000-bin histogram of the persistent score `c_ring.min(1)` over `last_tested` static and stale Gaussians (the quantities set at vmf_field.py:516-528);
   - the view of the first fire of each Gaussian.
6. `upd = EC.run_update(ctx, views, k, thr, q)` (EC:472-547), unchanged.
7. New fused rows: `fuse_rank([s[X], s["depth"]], mask=observed)` (change_baselines.py:522-538, NaN counted as rank 0 at :497-519).
8. Scores: `EC.ranking_table` (EC:620-628) and `EC.pixel_level` with the extended row list (EC:659-684).
9. With --theta-fix:
   - temporarily replace `DirectionalSemanticField.calibrate_change_threshold` by `set_change_threshold(θ_fix)` (vmf_field.py:626-634; calibration at :581-624);
   - run `EC.run_update` again (EC:479 calls calibrate; the checks at EC:504-505 and 536-537 still hold);
   - then `EC.operating_points` and `EC.realised_rates` (EC:635-655).
10. Save a compressed npz: scores and flags of all rows, tests, fires, observed, seen, stale, in_box, n_views, changed_end, robust_end, first_fire, the two histograms.
11. Save a runlog record with EC.code_state() and the field and map sha256.
12. --check: assert every Table III AP is within 0.002 of the saved per-scene JSON.

### scripts/nor_fused_report.py (CPU)
- theta: θ_fix by rule R4 (selection_rules), from the validation histograms.
- table: merges the scenes and prints means, std, wins and p (from quant_stats); also PR curves and the R3 claim verdict.

### scripts/nor_stale_query.py (package 2)
- Load: EC.load_context.
- Update replay: ours (`DirectionalSemanticField.from_state` + calibrate) and NormLift (`EC.make_normlift`, EC:349-358; the averaging feature is baselines.py:206-210).
- Held-out frames: update frames that are not update keyframes, chosen with time_field.py:157 `evenly`.
- Text: `get_extractor("lseg").text_embedding` (extractors/lseg.py:282-290), then `pca.project_text` (feature_pca.py:59-66).
- Render: `render_row` (seg_eval.py:109).
- Checks: ours' flag counts equal the saved n_flagged_static and n_flagged_stale.

### scripts/scannet_views.sh (package 2)
- A copy of replica_views.sh: the k-field mapping at :70-76 with --dataset scannet --stride 10 --downscale 2, configs/vocab_scannet.json and KS="10 30".
- eval_views.py reads integer ks (:436) and the ScanNet offset (eval_seg.py:190).

### scripts/dump_curves.py (package 2)
- Imports eval_seg and runs one scene as eval_seg does, through SegAccumulator (seg_eval.py:408-436).
- Curves from evaluation.py sparsification_curve (:51-79) and aurc (:150-157).
- Check: recomputed AURC, ECE and NLL equal the Table I JSON.

### Smoke tests (CPU)
- tests/test_nor_dump.py: on the mock-backend toy of tests/test_eval_change.py, the dump's 7 rows equal EC.evaluate's.
- tests/test_quant_stats.py.

## 6 GPU estimate
| Package | Items | GPU time |
|---|---|---|
| 1 | B1 + B2 | about 40 min (the 14 NOR evaluations took 26.4 min; θ_fix adds one update pass of 2–32 s per scene; peak 4.6 GB, gate 6500 MiB) |
| 2 | B3 + B4 + B5 | about 30 + 110 + 20 = 160 min |
| 3 | B6 + B7 + B8 | about 60 + 25 (+70) + 45 min |

## 7 Risks
The full list is in the risks field.

RULES:["R1 (statistics): the scene is the unit, never the pixel. Each claim has ONE primary test, declared before writing. For segmentation calibration the primary test is NLL, ours P2 vs the stronger trained baseline (Feature 3DGS), over the 14 test scenes. Every other comparison is descriptive: x/N wins, exact two-sided sign test p, exact Wilcoxon p, and a scene bootstrap 95% CI (20,000 resamples, seed 0). Smallest possible p: 0.031 (N=6), 0.008 (N=8), 0.002 (N=10), 1.2e-4 (N=14). Use 'significant' only for the primary tests with p < 0.05.", "R2 (aggregation): show the pixel-pooled ECE (from the stored reliability bins) and the pixel-weighted NLL, Brier and accuracy next to the per-scene means, never in place of them. Write 'lower ECE' only with the aggregation named, or when it holds under both.", "R3 (fused baselines, NOR): for each single-input row X in {ncc, feat, normlift, dispersion}, the row 'X+depth' = usplat.change_baselines.fuse_rank([X, depth], mask=observed): the same equal-weight rank average as the paper's fused row, nothing fitted. Run validation, then test once; report every row. 'Ours fused with depth is best' may be written only if its test mean G-AP is the highest of all fused rows. Always print x/10 wins and p against the strongest fused baseline. If a fused baseline is higher, the abstract and intro claim (75.0 vs 63.8) is replaced by the measured comparison.", "R4 (fixed threshold, NOR): θ_fix = the 0.999-quantile of ours' per-view persistent score over the tested STATIC (D2) Gaussians in the update passes of the 4 validation scenes, pooled (upper edge of the q bin, 4000 bins on [0, 2]; the fixed_rule of configs/static_fa.json). Secondary θ: the median of the validation scenes' own pooled calibrated thresholds (0.175). On test, compare ours at θ_fix with ours at its mapping-pass threshold. Primary metric: the median over scenes of |log10(measured static per-view rate / 1e-3)|. Also report FA and recall. Report the result whichever way it goes.", 'R5 (ScanNet views): M in {10, 30} mapping keyframes, evenly spaced; run_mapping.py --max-keyframes M --pca-from <full field> --field-config \'{"change_test": false}\'. Betas from scannet_tab1/eval_val.json. Rows: ours_P0, ours_P2, averaging, normlift, viewcount. Run the validation scenes (0000_00, 0062_00) first, only to check the pipeline (no choice is made), then test once. Write \'u_epi ranks e_red better than u_ale on ScanNet\' only if it holds in at least 7/8 test scenes; otherwise write \'not replicated\'.', 'R6 (stale query, NOR), fixed before any run. Object prompts: 001 space heater, 002 office chair, 003 teddy bear, 004 armchair, 005 book, 006 cardboard box, 007 jacket, 008 backpack, 009 laundry basket, 010 space heater, 011 backpack, 013 paper bag, 014 bicycle, 015 bicycle; the 5-template LSeg ensemble. LERF relevancy against the canonical phrases object, things, stuff, texture; a pixel is positive if relevancy > 0.5. Held-out frames: update frames that are not update keyframes and have a non-empty masks_real, evenly spaced, at most 20 per scene. Gate: a scene enters the primary mean only if the mapping-only map marks at least 20% of the mask pixels of held-out MAP frames as the object. Primary: stale-query rate of ours after the update vs averaging after the update, x/N wins and sign test over the gated test scenes. Every scene is reported.', 'R7 (method changes): only through a committed pre-registration JSON (candidates, validation metric, tie rule, test-look counter), as configs/tab1_v2_prereg.json does. Select on validation only. Report every candidate on test, labelled selected or not selected. None is recommended now.', "R8 (honour existing pre-registrations): report the ScanNet and Replica v2 rows as 'not selected' (configs/scannet_v2_prereg.json 'reporting' rule). The reducibility criterion at eval_views.py:142-144 binds once it is run. The 2026-10-10 static false-alarm config is outdated (no relmax); a new static run needs a newly frozen config, labelled as such.", 'R9 (figures): per-scene plots show every scene, with no scene dropped. Reliability diagrams show both ECEs. Every number in a caption links to its JSON or log.']
SCRIPTS:['scripts/quant_stats.py (CPU): subcommands seg | nor | views | cost. Paired scene statistics, pixel-pooled calibration, measured false-alarm table, per-object NOR table, cost table, JSON and LaTeX output.', 'scripts/plot_reliability.py (CPU): reliability diagrams pooled over pixels, from the stored bins, with scene-mean and pooled ECE.', 'configs/nor_fused_prereg.json: pre-registration of rules R3 and R4, committed before any run.', 'scripts/nor_dump_scores.py (GPU, imports scripts/eval_change_nor.py without editing it): per-Gaussian npz of every row; fused baselines X+depth; pixel AP; update-pass score histograms; a second update pass at θ_fix; --check against the saved Table III per-scene JSON.', "scripts/nor_fused_report.py (CPU): 'theta' selects θ_fix on validation; 'table' gives the fused-baseline table, AP where each pair of rows is defined, PR curves, wins/p and the R3 claim verdict.", 'tests/test_quant_stats.py and tests/test_nor_dump.py (CPU; the mock-backend toy of tests/test_eval_change.py).', 'Package 2: scripts/nor_stale_query.py + configs/nor_query_prereg.json (downstream stale-query test, GPU).', 'Package 2: scripts/scannet_views.sh (ScanNet k-view fields + eval_views.py, M = 10, 30, GPU).', "Package 2: scripts/dump_curves.py (risk-coverage and sparsification curves for Table I's pixels, GPU, must reproduce the saved AURC).", 'Package 3: configs/static_fa_relmax.json (a newly frozen Replica static false-alarm config, run by the existing scripts/replica_static_fa.sh with OUT and FROZEN overrides).']
GPU:40
RISKS:["Headline NOR claim. The paper's 'better than every baseline (G-AP 75.0 vs 63.8)' compares a depth-plus-features row with feature-only rows. Alone, NormLift's drop beats our score in 7/10 test scenes (63.8 vs 53.4). NormLift fused with depth may therefore reach or pass 75.0, and the claim would have to be rewritten under R3. I estimate this at roughly even odds or higher.", "Replica ECE claim depends on aggregation. Pooled over pixels, P2's ECE is 7.12 vs P0 5.62 and averaging 5.70, while the per-scene mean favours P2 (8.70 vs 9.86 and 9.62). Reliability diagrams (A6) will show this, so the text must explain it.", "Paired tests weaken some current wording. ScanNet 'best or tied' is a tie with averaging: ECE 3/8, NLL 4/8, and pixel-pooled NLL, Brier and accuracy slightly favour averaging. P2's AURC is worse than P0 and averaging in 8/8 ScanNet scenes (p = 0.008).", "Measured false alarms. Ours fires 6.3 per 1000 static tests on NOR test (1/10 scenes within [0.5x, 2x] of nominal). The 'nominal 10^-3' wording must become the measured rate. The fixed-threshold test (B2) may show no advantage for the label-free rule.", "ScanNet views (B4) may not replicate the Replica separation. ScanNet's fitted P2 coefficients are negative (a = −0.80, b = −2.06), and there u_ale's AUSE is 2.2x that of our own 1 − max p. A negative result must be reported.", "Running the pre-registered reducibility check (B7) can drop the u_epi claim, and its verdict binds once run, so the user should decide first. The 2026-10-10 static false-alarm config tests an outdated change weight ('view' instead of 'relmax'); running it as frozen would measure the wrong method.", 'Extra test looks: NOR test look 2 (B1, B2, B3), ScanNet test look 2 (B4), Replica test look 3 (B5, B7). Each must be disclosed, and no result may feed back into a method choice.', "Code. nor_dump_scores.py replaces class methods with wrappers in its own process. It must reproduce every Table III AP within 0.002 (--check) or stop. Editing eval_change_nor.py would change the code hash in Table III's provenance.", 'Timing (A3) comes from production logs on a shared GPU: LEGaussians ran at 15–46 ms per iteration across scenes. Report medians and the caveat, or run B8. Never compare against Replica timings taken on the RTX 5070 Ti.', 'Multiplicity. There are dozens of descriptive p-values; only the pre-declared primary tests may be called significant. Pixel-level tests would overstate significance, because the scene is the unit.', "No method change has validation support. The user's wish for 'better numbers' can only be met by stronger analysis and new evaluation axes, not by tuning. A post-hoc swap (for example P4 for its better test ECE) is forbidden."]