# Replica Table I: open-vocabulary segmentation and calibration
2026-09-29 | root: /home/thanhnc/gauss_ws/src/uags | git: master@778bb64 | [all reports](../INDEX.md)
Follows: [Data, baselines and evaluation infrastructure](../2026-09-29-data-and-baselines/README.md)
Verdict: IN PROGRESS -- Table I test done and in the paper (d287321); views figure and Table III running
Key result: test (N=6, one shot): AUSE of our u_ale 8.50 vs NormLift 1-C_inter 8.76, CSS variance 11.20, LEGaussians 19.73 (best uncertainty); tuned CSS best calibrated (ECE 5.05 vs ours P2 8.70) ([table](#test-table-i-one-shot-run-code-224fe1e))
Attempts: 1 (0 Good / 1 Bad: attempt 06 P3) | Time: started 2026-09-29 13:49 +07; final val 2026-09-30 06:50

Running: [chain3.sh](logs/chain3.sh): Table I val eval -> Table I test (once) -> views + Table III val/test | /home/thanhnc/gauss_ws/data/uags_results/chain3.log | started 2026-09-30 ~08:55

## 0 Context
- Ask: "continue to complete our paper" -> fill Table I (plan section 3) with measured numbers.
- Protocol (plan): Replica, Semantic-NeRF renders; map on Sequence_1 every 5th frame (180 frames), evaluate on
  Sequence_2 every 10th frame (90); validation scenes office_0, room_2 fit one temperature per row (P2: its 4
  parameters); the 6 test scenes (room_0, room_1, office_1-4) run ONCE with those temperatures.
- Features: LSeg (chosen on validation, [attempt 05](../2026-09-29-data-and-baselines/attempts/05-extractor-study.md)),
  PCA-64 fit on 20 mapping frames, 5-template ensemble; vocabulary = classes present in each scene's evaluation frames.
- Rows: ours P0/P1/P2 (+ P3, attempt 06), ours-scalar, 2D features per frame, averaging (GradBP), NormLift
  (Sec. 3.3; 1-R, epi, 1-C_inter), view count, CSS (Dirichlet over the present classes, given at mapping time;
  prior alpha0 chosen on val from {0.01, 0.1, 1, 10}), and since 2026-09-29 23:50 the trained SOTA baselines
  Feature 3DGS (CVPR 2024) and LEGaussians (CVPR 2024), re-implemented on the same map, frames, features and PCA
  ([usplat/distill.py](../../../usplat/distill.py); user: "I do not see any SOTA baseline").
- 3DGS: scripts/fit_gs.py, 30k iterations, max 1M Gaussians, 640x480; office_0: held-out PSNR 41.5 dB,
  SSIM 0.989, 620,290 Gaussians, 205 s (map meta in /home/thanhnc/gauss_ws/data/uags_maps/office_0_seq1.pt).
- Pipeline: [scripts/replica_tab1.sh](../../../scripts/replica_tab1.sh) (STAGE=val, then STAGE=test once).
- Machine: RTX 5070 Ti 16 GB (shared; GPU steps through scripts/gpu_gate.sh).

## 1 Findings
| Finding | Reason | Evidence |
|---|---|---|
| Mapping cannot run beside a USGLoc stage | LSeg (fp32, 640x480, one frame) needs > 3.3 GB: a 3-keyframe probe capped at 3500 MiB ran out of memory in the DPT head at the first PCA frame (14:20); USGLoc stages hold 6.5-9.2 GB and a policy server 3.8 GB of the 16 GB | [logs/memprobe_lseg_3p5gb.log](logs/memprobe_lseg_3p5gb.log) |

So the gate stays at 6000 MiB (+2048 headroom) and the stage waits for gaps between USGLoc runs.

| Mapping OOM at 15:13 (cap 6000 MiB) | LSeg activations + field + baselines + renders peaked together in the first mapping frame (5.6 GB allocated) | [logs/oom_mapping_office_0_1513.log](logs/oom_mapping_office_0_1513.log) |
| Fix: extract all tokens first (CPU, fp32, same values), free LSeg, then build the field (commit 5b21781) | 3-keyframe smoke test: extraction peak 4.9 GB, mapping peak 4.8 GB (was their sum) | [logs/smoke_mapping_3kf.log](logs/smoke_mapping_3kf.log) |

| Change before the first run | Reason | Commit |
|---|---|---|
| First queued run (13:48) stopped before it did any work; relaunched | add the "2D features, per frame" row of Table I (LSeg on each evaluation frame, same token -> upsample -> PCA path as mapping, own temperature, covers every pixel) so all rows come from one evaluation | 01f39a7 |
| All-zero statistics saved as None | disk has 25 GB free (98% used); a field with its centred channel and change buckets is ~1 GB | 01f39a7 |
| NormLift 1 - C_inter channel, rbar estimator, flat pooled null | paper review (Table I / III rows) | 4829617 |

| Mapping OOM for room_2 (954k Gaussians) at the 5.7 GB cap | mapping memory grows with N (office_0 620k: 4.8 GB) | [logs/oom_mapping_room_2_1516.log](logs/oom_mapping_room_2_1516.log); fix: caps MAP_NEED / EVAL_NEED = 9000 (5e8c51e); room_2 then peaked at 5.7 GB ([logs/map_room_2.log](logs/map_room_2.log)) |
| Eval OOM at 15:18 | a new USGLoc batch (batch_seeds12: 2 seeds x 7 scenes, ~30 min each) started in the same gap; also the eval kept the field and every row's buffer on the GPU next to LSeg | [logs/oom_eval_val_1518.log](logs/oom_eval_val_1518.log); fix: rows frozen on the CPU, identical buffers stored once, 2D tokens extracted before the rows (7dd4bfe); the gate now never starts beside a USGLoc job (its notes: MASt3R fails with launch timeouts when sharing) |
| u_ale is defined for few Gaussians but most pixel mass | tau_min = 3 views; most Gaussians are tiny (median pixel mass over 179 views: 4.2 px office_0, 1.6 px room_2; a full view is a0*H*W = 7.7 px) | [logs/support_stats.log](logs/support_stats.log): defined for 22.4% / 12.2% of Gaussians but 84.2% / 81.8% of pixel mass; u_ale mean 0.020 / 0.007 (LSeg features agree across views) |

| Feature 3DGS training on office_0 died at 05:08 with cudaErrorLaunchTimeout after 4k clean iterations (loss 0.0224 -> 0.0081) | this GPU also drives the display (gnome-shell), so a kernel that runs past the watchdog kills the process; 30k-iteration trainings launch ~100x more kernels than a mapping pass | [logs/cuda_timeout_distill_office_0_0508.log](logs/cuda_timeout_distill_office_0_0508.log); fix: the training state is checkpointed every 2k iterations and resumed (a resumed run equals an uninterrupted one: tests/test_distill.py), the pipeline retries a failed GPU step up to 3 times, and the LEGaussians decoder init follows the seed (d359bc4) |

| First test evaluation (08:37) refused by the field-identity check, before any metric was printed or saved | the check hashed the main field AND every auxiliary file into one identity; the validation scenes also hold the LEGaussians lambda_u = 1, 2 files, so the hashes differed although every file matched; it also checked only after a whole scene was evaluated | [logs/refused_test_eval_0837.log](logs/refused_test_eval_0837.log) (no test numbers: status "running", no metrics, no partial file); fix 224fe1e: main field identity exact, auxiliary files a subset of validation's, checked right after loading; the validation eval is re-run to record both sets (same betas) |

## 2 Baseline
3DGS maps (Sequence_1, every 5th frame = 180 frames, 157 train / 23 held out, i % 8 == 0; 30k iterations; 640x480):

| Scene | Held-out PSNR | SSIM | Gaussians | Train time | Peak GPU | Source |
|---|---|---|---|---|---|---|
| office_0 | 41.53 dB | 0.989 | 620,290 | 205 s | 756 MiB | fitter agent run (copied from its report, not re-checked; map meta in /home/thanhnc/gauss_ws/data/uags_maps/office_0_seq1.pt) |
| room_2 | 37.05 dB | 0.977 | 954,119 | 268 s | 900 MiB | [logs/fit_room_2.log](logs/fit_room_2.log) |
| room_0 (test) | 34.19 dB | 0.955 | 999,855 (cap hit at step 3,100) | -- | 894 MiB | [log](logs/prep_fit_room_0.20260929-233230.log) |
| room_1 (test) | 36.19 dB | 0.969 | 999,794 (cap hit at step 4,300) | -- | 910 MiB | [log](logs/prep_fit_room_1.20260929-233230.log) |
| office_1 (test) | 40.46 dB | 0.978 | 459,077 | -- | 664 MiB | [log](logs/prep_fit_office_1.20260929-233230.log) |
| office_2 (test) | 37.25 dB | 0.983 | 999,776 (cap hit at step 11,200) | -- | 880 MiB | [log](logs/prep_fit_office_2.20260929-233230.log) |
| office_3 (test) | 36.77 dB | 0.980 | 915,539 | -- | 876 MiB | [log](logs/prep_fit_office_3.20260929-233230.log) |
| office_4 (test) | 29.62 dB | 0.860 | 515,855 | -- | 698 MiB | [log](logs/prep_fit_office_4.20260929-233230.log) |

Test maps were fit 2026-09-29 23:32 - 2026-09-30 05:06 in gaps between USGLoc jobs (worktree at 4c268ad); their
fields and the rebuilt validation fields all carry mapping code b7b80b92d565
([mapping logs](logs/prep_map_office_1.20260929-233230.log)). office_4 is the weakest map (29.6 dB); all rows share it.

Held-out frames sit between training frames of the same trajectory: the PSNR measures interpolation, not new viewpoints.
The semantic baselines (averaging, NormLift, view count, CSS, 2D per frame) are rows of Table I, below.

## 3 Attempts
| # | Idea | Result | Critic | Link |
|---|---|---|---|---|
| 06 | P3: render the per-Gaussian class predictive q_n(c) = softmax(beta_g A_d(kappa) mu*.t_c) as a mixture (like CSS, but open-vocabulary: computed at query time); motivated by CSS ECE 4.7 vs ours 7.1 on office_0 (partial val, N=1, [log](logs/eval_val_partial_2013.log)) | val (N=2): beta_g = 40 chosen by NLL; P3 NLL 0.847 / ECE 5.01 vs P2 0.824 / 4.88 vs P0 0.833 / 6.51 ([json](logs/eval_val_final_bfcab98.json)) | Bad (NEGATIVE): the rule fixed before the run (P3 enters the paper only if its val NLL beats P2) says no; mixing per-Gaussian predictives helps ECE over P0 but not over the adaptive temperature | -- |

(Attempts 01-05 are in the [data-and-baselines run](../2026-09-29-data-and-baselines/README.md).)

### Test Table I (one-shot run, code 224fe1e)
Mean +- population std over the 6 test scenes (room_0, room_1, office_1-4), all pixels (coverage rule), with the
temperatures and row choices frozen on validation (CSS alpha0 = 0.01, LEGaussians lambda_u = 4; P3 excluded by the
rule). Command: [chain3.sh](logs/chain3.sh) -> `STAGE=test scripts/replica_tab1.sh` in a clean worktree at 224fe1e,
2026-09-30 08:56-09:20. Source: [logs/eval_test_224fe1e.json](logs/eval_test_224fe1e.json),
[log](logs/eval_test_224fe1e.log); LaTeX rows: [logs/table1_rows.tex](logs/table1_rows.tex) (make_tables.py, now in
papers/main.tex, commit d287321). The test split was evaluated once; an earlier attempt (08:37) was refused by the
identity check before producing any number ([log](logs/refused_test_eval_0837.log)).

| Row | mIoU | Acc | ECE | NLL | Brier | AURC | AUSE own u | AUSE MSP |
|---|---|---|---|---|---|---|---|---|
| 2D features, per frame | 26.34 +- 4.87 | 74.86 | 11.78 | 1.12 | 40.46 | 16.33 | -- | 12.10 |
| Averaging (GradBP) | 27.91 +- 3.44 | 76.66 | 9.62 | 1.00 | 36.65 | 14.00 | -- | 10.54 |
| NormLift 1-R / 1-C_inter / epi | same predictive | | | | | | 14.52 / 8.76 / 19.30 | |
| View count | same predictive | | | | | | 22.43 | |
| Feature 3DGS | 27.85 +- 2.92 | 76.96 | 10.01 | 1.04 | 37.00 | 14.56 | -- | 10.97 |
| LEGaussians (lambda_u = 4) | 23.97 +- 2.29 | 74.87 | 10.59 | 1.14 | 39.97 | 15.85 | 19.73 | 11.88 |
| CSS (alpha0 = 0.01) | 28.54 +- 3.33 | 77.17 | 5.05 | 1.01 | 35.05 | 11.54 | 11.20 | 8.13 |
| CSS (alpha0 = 1, default; not chosen) | 28.35 | 77.01 | 6.55 | 1.07 | 36.65 | 14.39 | 15.60 | 10.81 |
| Ours P0 | 27.96 +- 3.62 | 76.77 | 9.86 | 1.00 | 36.40 | 13.78 | 8.50 (u_ale), 16.38 (u_epi) | 10.36 |
| Ours P1 | 27.86 | 76.70 | 10.21 | 1.04 | 37.50 | 12.89 | 8.50 | 9.42 |
| Ours P2 | 27.96 | 76.77 | 8.70 +- 5.56 | 1.00 | 36.20 | 12.24 | 8.50 | 8.82 |
| Ours P3 (beta_g = 40; not in the paper) | 27.93 | 76.73 | 8.98 | 1.03 | 37.07 | 13.00 | 8.51 | 9.46 |

Per scene (ours P2 / CSS mIoU, ECE; AUSE of u_ale / NormLift 1-C_inter / CSS var_top / LEGaussians u):

| Scene | mIoU P2 / CSS | ECE P2 / CSS | AUSE ours / NormLift / CSS / LEGaussians |
|---|---|---|---|
| room_0 | 26.9 / 28.5 | 6.9 / 4.9 | 9.42 / 7.77 / 11.50 / 24.86 |
| room_1 | 26.8 / 27.3 | 15.5 / 6.7 | 10.26 / 11.00 / 10.25 / 21.77 |
| office_1 | 25.0 / 26.1 | 16.9 / 7.3 | 19.23 / 21.58 / 32.75 / 27.42 |
| office_2 | 29.4 / 28.7 | 2.8 / 4.5 | 3.69 / 4.43 / 2.78 / 12.74 |
| office_3 | 24.4 / 25.2 | 3.2 / 3.7 | 3.47 / 3.19 / 4.38 / 13.71 |
| office_4 | 35.2 / 35.5 | 6.8 / 3.2 | 4.91 / 4.57 / 5.55 / 17.88 |

Our u_ale has a lower AUSE than NormLift's 1-C_inter on 3 of 6 test scenes. Where the baseline wins: tuned CSS
has the best mIoU, ECE, Brier, AURC and MSP AUSE (it is told the scene's classes at mapping time); NLL ties
(1.00 vs 1.01). Claims allowed by these numbers: u_ale is the best own-uncertainty ranking of the 0/1 error; P2 is the
best open-vocabulary predictive; "calibrated" is NOT used (P2 wins ECE but ties NLL, the rule in the draft).

### Validation Table I (final code bfcab98)
Mean over office_0, room_2 (N=2); all pixels (coverage rule); lower is better except mIoU/Acc. Command: the chain
([logs/chain.sh](logs/chain.sh)) ran `STAGE=val scripts/replica_tab1.sh` in a clean worktree at bfcab98
(2026-09-30 05:31-06:50: baseline training, then eval). Source: [logs/eval_val_final_bfcab98.json](logs/eval_val_final_bfcab98.json),
[log](logs/eval_val_final_bfcab98.log), trainings [logs/distill_*](logs/distill_office_0_f3dgs.20260930-053133.log).
Arrows mark the rows chosen on validation by the rules fixed beforehand (CSS prior and P3 beta_g by NLL, LEGaussians
lambda_u by the AUSE of its own u).

| Row | mIoU | Acc | ECE | NLL | Brier | AURC | AUSE own u | AUSE MSP |
|---|---|---|---|---|---|---|---|---|
| 2D features, per frame | 30.76 | 80.97 | 8.70 | 0.940 | 32.71 | 9.97 | -- | 7.79 |
| Averaging (GradBP) | 31.42 | 81.08 | 6.59 | 0.833 | 29.27 | 6.59 | -- | 4.65 |
| NormLift 1-R / 1-C_inter / epi | same predictive | | | | | | 7.24 / 5.07 / 11.52 | |
| View count | same predictive | | | | | | 11.23 | |
| Feature 3DGS | 30.77 | 81.01 | 6.75 | 0.868 | 30.23 | 7.44 | -- | 5.32 |
| LEGaussians lambda_u=1 (paper) | 28.20 | 79.08 | 6.52 | 0.945 | 32.94 | 8.14 | 19.46 | 5.71 |
| LEGaussians lambda_u=2 | 27.82 | 78.62 | 6.66 | 0.963 | 33.66 | 8.56 | 19.88 | 6.03 |
| -> LEGaussians lambda_u=4 | 27.93 | 78.84 | 6.49 | 0.955 | 33.30 | 8.38 | 17.32 | 5.90 |
| CSS alpha0=1 (default) | 31.83 | 82.23 | 4.85 | 0.804 | 29.26 | 6.46 | 7.91 | 4.75 |
| -> CSS alpha0=0.01 | 31.70 | 82.16 | 4.70 | 0.774 | 28.41 | 5.62 | 4.63 | 3.89 |
| CSS alpha0=0.1 / 10 | 31.75 / 31.84 | | 4.39 / 8.23 | 0.776 / 0.890 | | 5.85 / 7.51 | 5.82 / 21.37 | |
| Ours P0 | 31.50 | 81.07 | 6.51 | 0.833 | 29.25 | 6.60 | 5.03 (u_ale), 8.81 (u_epi) | 4.66 |
| Ours P1 | 31.53 | 80.99 | 6.82 | 0.857 | 30.01 | 6.46 | 5.02 | 4.51 |
| Ours P2 | 31.50 | 81.07 | 4.88 | 0.824 | 29.45 | 6.15 | 5.03 | 4.21 |
| -> Ours P3 beta_g=40 | 31.58 | 81.04 | 5.01 | 0.847 | 29.84 | 6.52 | 5.03 | 4.57 |

Reading (validation only; the test decides): the tuned CSS (closed set, told the scene's classes at mapping time) is
the best calibrated row (NLL 0.774 vs ours P2 0.824, AURC 5.62 vs 6.15); keep it. Ours P2 is the best open-vocabulary
predictive. The trained baselines are below fusion: Feature 3DGS 30.8 mIoU; LEGaussians loses ~3.5 mIoU to its
128-code quantization and its learned u ranks errors worse than a random order at lambda_u 1 and 2 (AUSE 19.5 / 19.9);
lambda_u = 4 wins only because its u collapses to 0 (all ties: random order apart from the coverage tier). Our u_epi
ranks the 0/1 error better than NormLift's epistemic score and the view count (8.81 vs 11.52 / 11.23); u_ale ties
NormLift's 1 - C_inter (5.03 vs 5.07). The field-identity check passed: one identity for all val fields (2a6e51e7a3f1).

### Validation Table I (pre-review code a2ed2d9)
Mean over the 2 validation scenes (N=2, population std in the JSON); all pixels (coverage rule); lower is better
except mIoU/Acc. Command: `OUT=.../tab1_pre STAGE=val scripts/replica_tab1.sh` in a clean worktree at a2ed2d9
(23:13-23:19). Source: [logs/eval_val_pre_a2ed2d9.json](logs/eval_val_pre_a2ed2d9.json),
[log](logs/eval_val_pre_a2ed2d9.log). These fields had no seed and no code identity; they are rebuilt for the final run.

| Row | mIoU | Acc | ECE | NLL | Brier | AURC | AUSE own u | AUSE MSP |
|---|---|---|---|---|---|---|---|---|
| 2D features, per frame | 30.76 | 80.97 | 8.69 | 0.939 | 32.71 | 9.96 | -- | 7.78 |
| Averaging (GradBP) | 31.42 | 81.07 | 6.61 | 0.833 | 29.27 | 6.59 | -- | 4.65 |
| NormLift 1-R / 1-C_inter / epi | same predictive | | | | | | 7.24 / 5.07 / 11.52 | |
| View count | same predictive | | | | | | 11.23 | |
| CSS (alpha0 = 1) | 31.83 | 82.22 | 4.85 | 0.804 | 29.26 | 6.46 | 7.91 (var_top) | 4.74 |
| Ours P0 | 31.49 | 81.06 | 6.53 | 0.833 | 29.25 | 6.60 | 5.03 (u_ale), 8.81 (u_epi) | 4.66 |
| Ours P1 | 31.53 | 80.99 | 6.82 | 0.857 | 30.01 | 6.46 | 5.02 | 4.51 |
| Ours P2 | 31.49 | 81.06 | 4.89 | 0.824 | 29.45 | 6.15 | 5.03 | 4.21 |

Per scene: CSS is better calibrated on office_0 (ECE 4.65 vs P2 7.05), P2 on room_2 (2.73 vs 5.04). u_ale is
undefined on 9.6% (office_0) and 15.8% (room_2) of the evaluated pixels (ranked most uncertain, three-tier rule).
AUSE here is against the 0/1 error; the e_red / e_irr split is the views figure (not run yet).

### Pre-test review (workflow wf_ce10d633-f61)
6 reviewers -> 43 raw findings -> 30 merged ([summary](logs/pretest_review/summary.md),
[merged](logs/pretest_review/merged_findings.json), [votes](logs/pretest_review/verification_votes.json)).
Verification stopped at the subagent session limit (45 of 100 agents failed); the fixer, P3 and critic agents did
not run, so I applied the fixes myself (f7fa9a7, 4c268ad, eeb6e87):

| Finding | Fix | Commit |
|---|---|---|
| F01 Tab III variants fit their own PCA basis | run_mapping --pca-from <main>; ablation_rows refuses another basis | f7fa9a7 |
| F02 no pipeline for the Tab III e_red/e_irr columns | caption changed: Tab III AUSE is against the 0/1 error; the split stays in the views figure | eeb6e87 |
| F03 undefined-u tier not stated | stated in the metrics text (rule unchanged, fixed before any val run) | eeb6e87 |
| F04 CSS vocabulary misdescribed | docstring + paper: CSS gets the classes present in the eval frames | f7fa9a7, eeb6e87 |
| F05 CSS prior untuned | read-outs css_a{0.01,0.1,10} next to alpha0=1; lowest val NLL picks the Table I row | f7fa9a7 |
| F06/F08/F26 test-once guards | STAGE validated; test refuses a failed/betas-less val record, an existing eval_test.json, a dirty tree; logs timestamped | f7fa9a7 |
| F07 no partial results | <out>.partial.json after every scene | f7fa9a7 |
| F10 3 GB pinned by the PCA sample | samples cloned, buffers freed | f7fa9a7 |
| F11/F27 reservation skewed peaks, could crash | peak counters reset after the reservation; halving retry | f7fa9a7 |
| F13 no code identity | fields carry a mapping-code hash; eval_seg --betas refuses test fields built differently | f7fa9a7 |
| F16 wrong templates recorded | records the LSeg 5-template ensemble | f7fa9a7 |
| F18 no seed | --seed 0 | f7fa9a7 |
| F19 table generator layout | midrule, P2 "as P0", val-chosen CSS/P3 rows, status check | 8035304 |
| F28 NaN channels not shared | NaN-aware sharing | f7fa9a7 |
| F30 tiny-evidence mu* not unit, support 0 | exact normalisation, Kish with W2 > 0 | f7fa9a7 |
| F09/F14/F15/F22/F29 paper wording | frame counts, population std, tau_cov fixed in advance, 2D row covers every pixel, stale comment | eeb6e87, 778bb64 |
| F12 timing fields vs paper breakdown; F17 policy; F20/F21/F23/F24/F25 | F17/F24/F25 fixed; F12/F20/F21/F23 open (section 7) | f7fa9a7 |

### Trained SOTA baselines (added 23:50 at the user's request)
Feature 3DGS (L1 feature distillation) and LEGaussians (128-code quantization, 8-D codes + MLP decoder, learned
u in [0,1] with its Eq. 10-12, lambda_u = 1), both on our map, frames, LSeg PCA-64 features, Adam 1e-3, up to 30k
iterations with a stopping rule fixed in advance (loss gain < 1% over 5k, min 5k). Paper details checked in
arXiv 2311.18482 (LEGaussians; codebook N=128, u initialized to 0, all lambdas 1, lr 1e-3, 30k iterations).
Tests: [tests/test_distill.py](../../../tests/test_distill.py) (5 CPU tests incl. exact resume).

First Replica runs (office_0, code d359bc4, discarded after the lambda_u change): Feature 3DGS converged at 16k
iterations (L1 0.0077, 437 s, 27.3 ms/iter); LEGaussians with its lambda_u = 1 converged at 27k (loss 0.930, 357 s)
but its u was bimodal: 67.6% of Gaussians at u > 0.99, 23.8% below 0.01 (u saturates where the per-pixel
cross-entropy of our 128 LSeg codes stays above lambda_u). Decision, fixed before any LEGaussians evaluation:
lambda_u in {1, 2, 4} is trained on the validation scenes and the lowest validation AUSE of its own u picks the
test value (3c1e955; the strongest fair baseline). The paper states it.

## 5 Verification
(filled at the end)

## 7 Open issues
- The test split has not run. The chain runs it once, after the final validation evaluation, with the frozen
  temperatures and val-chosen rows (CSS prior, P3 beta_g by NLL; LEGaussians uses its published lambda_u = 1).
- CSS beats ours P2 on validation NLL (0.804 vs 0.824) and ties on ECE (4.85 vs 4.89); ours P2 is better on AURC and
  MSP AUSE. Keep these rows even where the baseline wins.
- Feature 3DGS / LEGaussians are re-implementations on shared features (no SAM/CLIP+DINO pipelines, no LEGaussians
  smoothing MLP); their original pipelines are not run.
- Open review items: F12 (timing fields: field-only time needs a no-baselines run), F20 (Tab III full-method
  temperatures differ slightly from Tab I because the fitter's pixel subsets change with the rows), F21 (ablation rows
  exist for P0/P2 only), F23 (views figure M=180 point).
- GPU: USGLoc batches take the card back to back; our jobs start only in gaps (never beside a USGLoc job).
- Disk: 99% full (18 GB free after deleting the obsolete pre-review fields office_0/room_2_lseg_pre.pt, 2 GB).

## 8 Files
- Code (this run): scripts/replica_tab1.sh, scripts/replica_tab3.sh, scripts/replica_views.sh, scripts/eval_seg.py,
  scripts/eval_views.py, scripts/run_mapping.py, scripts/make_tables.py, scripts/distill_features.py,
  usplat/distill.py, usplat/rows.py, usplat/seg_eval.py, usplat/baselines.py, usplat/ablation_rows.py,
  usplat/vmf_field.py, usplat/gpu_reserve.py, scripts/gpu_gate.sh, tests/ (149 CPU tests pass at 5eb9f9a).
- Paper: papers/main.tex (journal class; review fixes; new baseline rows; 8 pages at 778bb64).
- This folder: logs/ (memory probes, OOM logs, mapping logs, support stats, partial and pre-review validation
  results, pretest_review/, workflow_pretest_review.js, chain.sh).
- Data (not in git): /home/thanhnc/gauss_ws/data/uags_maps/*_seq1.pt (35-55 MB each),
  /home/thanhnc/gauss_ws/data/uags_fields/*_lseg*.pt (0.8-1.3 GB each), /home/thanhnc/gauss_ws/data/uags_results/.
