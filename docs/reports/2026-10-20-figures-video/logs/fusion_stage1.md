# Fusion with the current view: stage 1

2026-10-24 | root: /home/thanhnc/gauss_ws/src/uags | git: master@2dbd2a1, +6 untracked new files (nothing committed) | part of [Figures, video and stronger results](../README.md)

**Verdict: stage 1 DONE.** The code, the CPU tests and the draft pre-registration are ready. No fused row has run on any real scene, validation or test.

**Next step:** the lead decides Q1-Q5 (section 7), sets `registered` in the prereg, and commits the prereg and the code. Only then can `STAGE=val` run. Until then the driver refuses every stage (section 5, row V9).

**Update 2026-10-24 01:06:** Q1-Q5 were decided as drafted (Q5: Table I only if C1 holds, else the appendix). The prereg and code were committed at 9671f02. The validation stage then ran on both datasets (section 9). No test stage has run.

## 1 What it is

| Item | Definition |
|---|---|
| Protocol fact | Table I's "2D features, per frame" row runs LSeg on the evaluation frame itself (`eval_seg.py --with-2d`). So the current image is available at query time. |
| Idea | Fuse the map predictive with that frame's 2D predictive, per pixel. Our uncertainty sets how much each pixel trusts the 2D logits. |
| Logits | `L_map = β_map · F_p·t_c` (the map row's own fitted temperature: a scalar for P0 and averaging, P2's per-pixel β_p for P2). `L_2d = β_2d · z_p·t_c` (the 2D row's own fitted temperature). |
| Pool (one form, fixed now) | `z_p = (1 − w_p)·L_map + w_p·L_2d`. This is the normalised log-linear pool `P ∝ P_map^(1−w) · P_2d^w`. |
| Why this form | (1) w = 0 and w = 1 give the two single-source rows exactly, so the family nests both. (2) The weight acts on logits, like the temperatures. (3) It multiplies the sources' probabilities, as independent evidence combines. (4) One form means no extra selection. The linear pool is not registered. |
| Weight | `w_p = sigmoid(θ0 + Σ θ_j x_pj)` on map-covered pixels. |
| Coverage | Uncovered pixels (α < τ_cov = 0.5) get `w = 1`: the 2D prediction decides there. Map rows count them as wrong. Every fused row fills holes the same way, so fused rows differ only on map-covered pixels. Every row is also reported on map-covered pixels only (variant `map_covered`). |
| Fit, stage 1 | The single-source temperatures come from eval_seg's `BetaFitter`, replayed in Table I's order and seed. They equal eval_seg's own fit exactly on the CPU toy (V4). |
| Fit, stage 2 | θ of each form is fit by NLL on covered labelled validation pixels (≤ 2000 per view, own generator, seed 1), with the logits fixed. L-BFGS in float64, starting at θ = 0. |

## 2 Rows

| Row | Base logits | Weight features x | Role |
|---|---|---|---|
| ours_P0, ours_P2, averaging, 2d_features | — | — | reference (Table I rows, recomputed in the same run) |
| fuse_avg_const | averaging | none (constant w) | fair baseline (binding) |
| fuse_avg_vc | averaging | 1/(1+n_vis), 1−α | fair baseline (binding) |
| fuse_avg_nl | averaging | NormLift 1/(1+N_eff), 1−C_inter, 1−α | fair baseline (binding; Q1) |
| fuse_p0_const / fuse_p2_const | ours P0 / P2 | none | fair baseline (binding: the one on @ug's base) |
| fuse_p0_ug / fuse_p2_ug | ours P0 / P2 | u_epi, u_ale (0 if undefined), 1[u_ale undefined], 1−α | **OURS-UG** (@ug picks one per dataset) |

All fused rows get the same 2D input (same tokens, PCA and temperature as the Table I row).

## 3 Pre-registration draft: [configs/fusion_prereg.json](../../../../configs/fusion_prereg.json) (uncommitted, sha256 3cef91bd45d9)

| Rule | Content |
|---|---|
| Fit | Per dataset, on its existing validation scenes only (Replica office_0, room_2; ScanNet scene0000_00, scene0062_00). |
| Selection @ug | The lowest pixel-pooled validation NLL among [fuse_p0_ug, fuse_p2_ug]. Tie tolerance 1e-4 nats; a tie goes to P0 (the simpler form). The paired constant-weight row follows the base. |
| Reproduction stop | The reference rows must match Table I's validation record within 0.01 nats of NLL and 0.5 points of mIoU per scene. Otherwise `check-test` refuses, and the gap is recorded and brought to the lead. |
| Validation gate | Off in this draft (Q3). If on: no test look unless @ug beats every binding baseline on validation NLL. |
| C1 (gain) | @ug beats EACH binding baseline (AVG+2D const, OURS+2D const on the same base, AVG+2D view-count, AVG+2D NormLift) on the test mean **and** in ≥ 4/6 Replica and ≥ 5/8 ScanNet scenes (a tie is not a win). The headline is mIoU; the same rule applies to ECE, NLL, Brier and AURC. |
| R2 | A calibration claim also needs the same sign pixel-pooled (`writable` in the table record). |
| C2 ("significant") | Sign test over the 14 pooled test scenes. Holm over family F3f: 4 baselines × 5 metrics = 20 tests, p < 0.05. C1 (+R2) must also hold on both datasets. |
| Report | Every row, every metric (mIoU, Acc, ECE, NLL, Brier, AURC), in both variants (all pixels, map-covered), on validation and test. Also: fitted θ, mean 2D weight, the reproduction check and the gate. Comparisons with map-only rows are descriptive only. |
| Test once | Test runs only after validation was made at one clean commit equal to the tree and the selection reproduces. A finished test record is never rerun; `look_test.json` logs every attempt. |
| Disclosure | A later attempt after Table I. Replica: a further test look. ScanNet: the third test look (after Table I v1+v2 and smoothing). |

## 4 Files (all new; no existing file edited)

| File | Role |
|---|---|
| [usplat/view_fusion.py](../../../../usplat/view_fusion.py) | `FusionForm`, `FORMS` (7), `pixel_bundle`, `base_logits` (op for op as `evaluate_view`), `design`, `weights`, `fused_outputs`, `fit_weights`, `FusionFitData`, `check_fusion_record` |
| [scripts/eval_fusion.py](../../../../scripts/eval_fusion.py) | Imports `eval_seg` and `seg_eval` without editing them. Fit: BetaFitter in Table I order, then the fusion pass. Evaluation: reference rows through `evaluate_view`, fused rows, variants `all` and `map_covered`. Row options are refused. |
| [scripts/fusion_select.py](../../../../scripts/fusion_select.py) | `select` (choose_v2, reproduction, gate), `check-test`, `look`, `table` (C1 + R2), `claim` (F3f, sign test + Holm) |
| [scripts/fusion_tab1.sh](../../../../scripts/fusion_tab1.sh) | `DATASET=replica\|scannet STAGE=val\|test\|table`, `DRY_RUN`. The gates are listed in section 5. |
| [configs/fusion_prereg.json](../../../../configs/fusion_prereg.json) | Section 3 |
| [tests/test_view_fusion.py](../../../../tests/test_view_fusion.py) | 6 CPU tests (section 5) |

Do-not-edit list respected: `git status` shows only these 6 untracked files plus this note. `git diff --stat` is empty: eval_seg.py, eval_change_nor.py, run_mapping.py, seg_eval.py, rows.py, predictive.py, vmf_field.py, baselines.py, make_tables.py, main.tex and the existing configs are unchanged.

## 5 Verification

Command: `cd uags && PATH=$HOME/venvs/uags/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -q -p no:cacheprovider tests/test_view_fusion.py` gave **6 passed in 40.3 s** (2026-10-24 00:10).

| # | Check | Test | Result |
|---|---|---|---|
| V1 | Code forms = prereg forms (names, bases, features, order); the choices and baselines resolve | test_forms_match_prereg | PASS |
| V2 | **w = 0 reproduces the map row exactly**: pred, conf, entropy, NLL, Brier, correct and tied match `evaluate_view` bit for bit on 240 covered pixels, and match the 2D row on 16 uncovered pixels (P2's NaN β there is never read). **w = 1 reproduces the 2D row exactly** on all pixels. Checked for all 7 forms (bases P0, P2, averaging). A non-finite feature and a wrong θ length are refused. | test_endpoints_reproduce_rows_exactly | PASS |
| V3 | **The fit recovers known coefficients.** UG θ = [0.526, −2.018, 1.476, 0.983, −0.981] vs the true [0.5, −2.0, 1.5, 1.0, −1.0] (max error 0.026, 60k toy pixels). A constant w is recovered as 0.300 vs 0.3. Map-only data push w below 0.02. The fit NLL is ≤ both endpoints. | test_fit_recovers_known_coefficients | PASS |
| V4 | Mock CLI on a toy with a hole (159 uncovered pixels), NormLift and view-count sums, and a fake 2D extractor. The single-source betas **equal eval_seg.py's own fit** (dict equality). The reference rows' metrics equal eval_seg's. The map rows' `map_covered` equals eval_seg's `covered`. A `--betas` rerun reproduces every metric. θ = ∓1e300 makes the fused rows equal the map rows (map-covered pixels) and the 2D row (all pixels) in every metric. Refused: `--rows`, a changed spec, a missing fusion record, and `--val-scenes` given together with `--betas`. | test_eval_fusion_cli_mock | PASS |
| V5 | Validation gate (incl. within-tolerance = not beaten); @ug tie rule (P2 must win by > 1e-4); F3f = 20 tests over 14 scenes, Holm (14/14 wins significant, 4/6 + 5/8 not); claim refuses a missing dataset | test_select_gate_and_claim | PASS |
| V6 | Driver end to end in a scratch git repo (mock). Refused: test before val; test scenes given as val; OUT = a Table I dir and OUT = smooth/scannet; `EVAL_EXTRA --rows`; `UAGS_LSEG_FLIP` set; uncommitted prereg; dirty tree; code committed after val; a Table I record that no longer reproduces. Then: val → selection (reproduction pass) → test once → table (R2 `writable` checked) + claim. A second test and a val after test are both refused. | test_fusion_driver | PASS |
| V7 | Independent check: the existing eval tests still pass, unchanged | `pytest tests/test_seg_eval.py tests/test_smoothing.py` → 36 passed in 67.6 s | PASS |
| V8 | Only new files; no tracked file changed | `git status --short`, `git diff --stat` | PASS |
| V9 | The real driver refuses while the prereg is uncommitted (replica val/test/table, scannet val; `DRY_RUN=1`). No output dir was created. | `DATASET=… STAGE=… DRY_RUN=1 bash scripts/fusion_tab1.sh` → "refused: configs/fusion_prereg.json is not committed", exit 1 | PASS |
| — | Real validation / test scenes | — | NOT RUN (forbidden in stage 1) |

## 6 GPU and time estimate (not measured on real data)

**LSeg tokens (checked).** eval_seg caches the tokens **in memory only, per scene load**: `load_scene` extracts `f["tok2d"]` for every frame, then frees LSeg. There is no on-disk cache. eval_fusion reuses these tokens for all its passes within one load: BetaFitter 3D, BetaFitter 2D, the fusion pass and the evaluation. As in Table I, the fit and the evaluation each load a validation scene, so LSeg runs twice per validation scene. LSeg costs 0.09–0.11 s per frame on this 4070 Ti (ScanNet mapping, [quant/cost.json](quant/cost.json): 48.4 s for 521 frames; median 13.5 s). In total that is 2372 frames ≈ 4 min of LSeg. A disk cache would save less than 2 min and add a new path to the 2D row, so it is not built.

Synthetic CPU benchmarks (no real data; the scripts stayed in the session scratchpad):

| Item | Measured (2026-10-24, load 3.5 from other jobs) |
|---|---|
| Metrics of one row (all + map_covered), 6.8 M px | 22–24 s |
| Metrics of one row, 18.6 M px | 76–79 s (4.1–4.2 s per M px) |
| `fused_outputs`, 7 forms, one 76.8k-px view, CPU | 0.11–0.17 s |
| Fit of 7 forms, 316 views × 2000 px, K = 14 | 30 s |

Each run has 11 rows (4 reference + 7 fused). The metrics rate per M px per row: the low end comes from the Table I walls, which include the loads (Replica test: 1462 s for 17 rows × 38.5 M px, so ≤ 2.2 and about 1.8 after loads; ScanNet test on this machine: 5942 s for 34 rows × 54.4 M px, so ≤ 3.2). The high end is 4.2, from the benchmark above. Per view: 7 renders at 0.006–0.013 s each (Table I eval_s: 9–20 s for 90 views × 17 rows) plus the fused rows, about 0.1–0.4 s. Pixel counts come from the Table I records. Scene loading takes 30–90 s on the CPU (copied from smooth_stage1.md, not re-checked).

| Stage | Loads | LSeg frames | Labelled px | Metrics | Wall (est.) |
|---|---|---|---|---|---|
| Replica val | 4 | 360 | 12.3 M | 4–10 min | 8–22 min |
| ScanNet val | 4 | 632 | 21.2 M | 12–16 min | 18–31 min |
| Replica test | 6 | 540 | 38.5 M | 13–30 min | 19–44 min |
| ScanNet test | 8 | 840 | 54.4 M | 32–42 min | 40–61 min |
| **Total** | 22 | 2372 | 126 M | | **1.4–2.6 h** |

- The GPU computes only for about 5–15 min of this (LSeg ≈ 4 min, plus 7 renders per view over ≈ 3400 view passes). But `gpu_gate.sh` holds the lock and the reserved fraction for the whole wall time.
- GPU cap: `EVAL_NEED` = 7000 MiB, Table I's cap with LSeg in the same process. Table I's measured peak allocation was 2.27–3.65 GiB with 17–36 frozen rows; this run freezes 6 rows sharing 2 buffers.
- The gate needs 9048 MiB free. At 00:18, 3258 of 12282 MiB were in use by other jobs, so it would wait.
- RAM is estimated at ≤ 20 GB peak, driven by ScanNet scene0000_00 (18.6 M px × 11 rows × ~36 B, plus 2.8 GB of tokens). Table I's ScanNet val run peaked at 36.7 GiB with 36 rows. The machine has 94 GB.

## 7 Open decisions (for the lead)

| # | Question | Draft |
|---|---|---|
| Q1 | Should `fuse_avg_nl` (AVG+2D with NormLift's 1/(1+N_eff), 1−C_inter, 1−α) be a binding baseline? NormLift's terms are Table I's comparators of our two channels, so this is the strongest fair uncertainty-guided averaging. It goes beyond the three baselines of the plan. | binding |
| Q2 | OURS-UG on P0 or P2 logits? | both registered; validation NLL picks; a tie goes to P0 |
| Q3 | Validation gate (skip the test look if @ug does not beat every baseline on validation)? | off, as smoothing D3 (both datasets are scored on test) |
| Q4 | Reproduction tolerances (0.01 nats, 0.5 mIoU points per scene), and should a failure block the test stage? The Replica Table I record was made on an RTX 5070 Ti, so the 2D row (LSeg on GPU) may differ more than the map rows did in smoothing (1.8e-4 nats). | blocks the test stage (`required: true`) |
| Q5 | Placement: a new block under Table I, or the appendix? Also the extra test looks (Replica: further, ScanNet: third). | block in Table I; both datasets |

Risks (not decisions):
- On ScanNet, P2's γ_e and γ_a are negative (it sharpens with uncertainty), so u's meaning for the weight may differ between the datasets. The fit is per dataset.
- The fused rows have no AUSE_u (`--` in the table).

## 8 Files of this note

This note is the only docs file written. The two benchmark scripts and their output stayed in the session scratchpad and are not linked; their numbers are copied into section 6. The new code files are listed in section 4.

## 9 Validation (2026-10-24, code 9671f02, prereg sha256 626ce55f7ee3)

**Result: validation done on both datasets; no test stage ran.** @ug = `fuse_p2_ug` on both datasets. The reference rows reproduce Table I. **Replica:** OURS-UG has the lowest pooled validation NLL of all rows. **ScanNet:** OURS-UG beats its paired constant-weight row but loses to all three AVG+2D rows on pooled validation NLL. The validation gate is recorded as failed there; it is disabled (Q3), so it blocks nothing.

Commands: `DATASET=replica STAGE=val scripts/fusion_tab1.sh`, then `DATASET=scannet STAGE=val scripts/fusion_tab1.sh`. Both ran through `gpu_gate.sh 7000`, with a clean tree at 9671f02.
Records are in /home/thanhnc/gauss_ws/data/uags_results/fusion/<dataset>/: `eval_val.json` (status ok, `dirty_tracked` false), `selection_val.json`, `driver_val.log`, `eval_val.<ts>.log`.
Tables below: means over the 2 validation scenes (same units as Table I); "NLL pool" is pixel-weighted. Read-out: a JSON-only script in the session scratchpad (not kept).

### 9a Run cost

| Dataset | Driver wall | eval_fusion wall | Peak GPU alloc | Peak RSS | Gate |
|---|---|---|---|---|---|
| Replica | 21.9 min (00:24:35–00:46:26) | 21.7 min | 2.15 GiB (reserved 6251 MiB) | 9.7 GiB | started at once (9077 MiB free) |
| ScanNet | 19.5 min (00:46:26–01:05:57) | 19.4 min | 2.38 GiB | 17.4 GiB | started at once (9432 MiB free) |

Both runs fall inside the section 6 estimates (Replica 8–22 min, ScanNet 18–31 min).

### 9b Reproduction of Table I (independent check, `selection_val.json` → reproduction)

| Dataset | Table I record | ours_P0 / ours_P2 / averaging: max abs diff (β, NLL, mIoU) | 2d_features: max abs diff (β, NLL, mIoU) | Status |
|---|---|---|---|---|
| Replica | eval_val_224fe1e.json (RTX 5070 Ti) | ≤ 4.6e-6, ≤ 2.7e-7, ≤ 8.7e-7 | 1.3e-4, 6.1e-6, 2.0e-6 | PASS |
| ScanNet | scannet_tab1/eval_val.json (this GPU) | 0, 0, 0 | 0, 0, 0 | PASS |

### 9c Replica validation (office_0, room_2; 12.3 M labelled px, 31,288 not map-covered)

| Row | mIoU | Acc | ECE | NLL | Brier | AURC | NLL pool | mIoU cov | Acc cov | ECE cov | NLL cov | Brier cov | AURC cov | mean w (office_0/room_2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_P0 | 31.5 | 81.1 | 6.51 | 0.8328 | 29.25 | 6.60 | 0.8092 | 31.7 | 81.3 | 6.51 | 0.8269 | 29.08 | 6.56 | -- |
| ours_P2 | 31.5 | 81.1 | 4.88 | 0.8238 | 29.45 | 6.15 | 0.8003 | 31.7 | 81.3 | 4.88 | 0.8180 | 29.28 | 6.12 | -- |
| averaging | 31.4 | 81.1 | 6.59 | 0.8331 | 29.27 | 6.59 | 0.8095 | 31.6 | 81.3 | 6.60 | 0.8273 | 29.10 | 6.56 | -- |
| 2d_features | 30.8 | 81.0 | 8.70 | 0.9396 | 32.71 | 9.97 | 0.9111 | 30.7 | 81.0 | 8.72 | 0.9401 | 32.71 | 9.98 | -- |
| fuse_avg_const | 32.9 | 82.7 | 7.27 | 0.8160 | 28.52 | 6.78 | 0.7922 | 32.9 | 82.7 | 7.29 | 0.8161 | 28.51 | 6.78 | 0.288/0.288 |
| fuse_avg_vc | 32.7 | 82.6 | 6.89 | 0.8145 | 28.46 | 6.70 | 0.7908 | 32.7 | 82.6 | 6.91 | 0.8146 | 28.45 | 6.69 | 0.277/0.251 |
| fuse_avg_nl | 32.9 | 82.7 | 7.01 | 0.8138 | 28.44 | 6.60 | 0.7903 | 32.9 | 82.7 | 7.03 | 0.8139 | 28.43 | 6.60 | 0.258/0.196 |
| fuse_p0_const | 32.9 | 82.7 | 7.23 | 0.8158 | 28.51 | 6.79 | 0.7920 | 32.9 | 82.7 | 7.25 | 0.8159 | 28.50 | 6.79 | 0.287/0.287 |
| fuse_p0_ug | 32.9 | 82.8 | 7.10 | 0.8075 | 28.11 | 6.39 | 0.7838 | 33.0 | 82.8 | 7.12 | 0.8076 | 28.09 | 6.39 | 0.224/0.197 |
| fuse_p2_const | 32.9 | 82.7 | 6.42 | 0.8042 | 28.56 | 6.12 | 0.7806 | 33.0 | 82.7 | 6.44 | 0.8043 | 28.55 | 6.11 | 0.278/0.278 |
| **fuse_p2_ug (@ug)** | 32.8 | 82.7 | 5.23 | **0.7930** | **28.05** | **5.83** | **0.7702** | 32.9 | 82.8 | 5.24 | 0.7931 | 28.03 | 5.83 | 0.193/0.183 |

### 9d ScanNet validation (scene0000_00, scene0062_00; 21.2 M labelled px, 17,555 not map-covered)

| Row | mIoU | Acc | ECE | NLL | Brier | AURC | NLL pool | mIoU cov | Acc cov | ECE cov | NLL cov | Brier cov | AURC cov | mean w (0000/0062) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_P0 | 71.1 | 89.4 | 3.41 | 0.4113 | 16.58 | 2.97 | 0.5488 | 71.1 | 89.5 | 3.41 | 0.4097 | 16.52 | 2.97 | -- |
| ours_P2 | 71.1 | 89.4 | 2.74 | 0.4104 | 16.64 | 3.20 | 0.5473 | 71.1 | 89.5 | 2.74 | 0.4088 | 16.58 | 3.19 | -- |
| averaging | 71.3 | 89.6 | 3.56 | 0.4093 | 16.42 | 2.93 | 0.5444 | 71.3 | 89.7 | 3.56 | 0.4077 | 16.36 | 2.93 | -- |
| 2d_features | 67.5 | 87.7 | 5.71 | 0.5151 | 20.12 | 4.46 | 0.6820 | 67.6 | 87.7 | 5.71 | 0.5149 | 20.12 | 4.46 | -- |
| fuse_avg_const | 71.3 | 89.7 | 3.64 | 0.4080 | 16.36 | 2.92 | 0.5427 | 71.4 | 89.7 | 3.64 | 0.4078 | 16.35 | 2.92 | 0.066/0.066 |
| fuse_avg_vc | 71.4 | 89.7 | 3.50 | 0.4072 | 16.35 | 2.92 | 0.5420 | 71.4 | 89.7 | 3.49 | 0.4070 | 16.35 | 2.92 | 0.062/0.003 |
| fuse_avg_nl | 71.3 | 89.7 | 3.52 | **0.4064** | **16.35** | **2.86** | **0.5405** | 71.4 | 89.7 | 3.51 | 0.4061 | 16.34 | 2.86 | 0.059/0.000 |
| fuse_p0_const | 71.1 | 89.5 | 3.57 | 0.4097 | 16.49 | 2.95 | 0.5460 | 71.2 | 89.5 | 3.56 | 0.4095 | 16.48 | 2.94 | 0.104/0.104 |
| fuse_p0_ug | 71.2 | 89.6 | 3.48 | 0.4080 | 16.43 | 2.95 | 0.5446 | 71.2 | 89.6 | 3.47 | 0.4077 | 16.42 | 2.95 | 0.018/0.012 |
| fuse_p2_const | 71.1 | 89.5 | 2.95 | 0.4086 | 16.54 | 3.15 | 0.5445 | 71.2 | 89.5 | 2.94 | 0.4084 | 16.53 | 3.15 | 0.103/0.103 |
| fuse_p2_ug (@ug) | 71.2 | 89.6 | 2.91 | 0.4070 | 16.47 | 3.14 | 0.5429 | 71.2 | 89.6 | 2.90 | 0.4067 | 16.46 | 3.14 | 0.026/0.032 |

### 9e Fitted weights of OURS-UG and the fit (`eval_val.json` → fusion)

w = sigmoid(a0 + a_e·u_epi + a_a·u_ale + a_u·1[u_ale undefined] + a_c·(1−α)). A positive coefficient puts more weight on the 2D frame.

| Dataset | Form | a0 | a_e | a_a | a_u | a_c | Fit NLL (map w=0 / 2D w=1) | Iterations, converged |
|---|---|---|---|---|---|---|---|---|
| Replica | fuse_p0_ug | −3.11 | +6.74 | +35.6 | +0.36 | +6.39 | 0.8918 (0.9164 / 1.0125) | 34, yes |
| Replica | **fuse_p2_ug** | −3.48 | +7.32 | +61.7 | +0.74 | **−163.4** | 0.8711 (0.8986 / 1.0125) | 44, yes |
| ScanNet | fuse_p0_ug | −7.90 | +11.5 | +73.7 | −2.42 | +15.0 | 0.5565 (0.5594 / 0.6972) | 45, yes |
| ScanNet | **fuse_p2_ug** | −6.58 | +10.9 | +56.4 | −1.73 | +12.7 | 0.5547 (0.5577 / 0.6972) | 37, yes |

Constant weights: Replica w = 0.288 (avg), 0.287 (P0), 0.278 (P2); ScanNet w = 0.066 (avg), 0.104 (P0), 0.103 (P2).

Baseline weights:

| Dataset | Form | Coefficients |
|---|---|---|
| Replica | AVG+2D view-count | a0 −1.52, viewcount +23.9, 1−α +3.96 |
| Replica | AVG+2D NormLift | a0 −1.87, nl_epi +6.14, nl_cinter +29.0, 1−α +5.57 |
| ScanNet | AVG+2D view-count | a0 −1.52, viewcount **−70.6**, 1−α +11.4 |
| ScanNet | AVG+2D NormLift | a0 **+9.76**, nl_epi **−390**, nl_cinter **−113**, 1−α **−271** |

Every fit converged (gradient ≤ 5.1e-8) on 360,000 (Replica) and 632,000 (ScanNet) fit pixels.

### 9f Selection and gate (`selection_val.json`)

| Dataset | @ug (pooled val NLL) | Paired const | Gate (recorded only): @ug lower than the baseline by (nats) | Gate |
|---|---|---|---|---|
| Replica | fuse_p2_ug 0.7702 (fuse_p0_ug 0.7838) | fuse_p2_const | avg_const +0.0220, p2_const +0.0104, avg_vc +0.0206, avg_nl +0.0201 | passed |
| ScanNet | fuse_p2_ug 0.5429 (fuse_p0_ug 0.5446) | fuse_p2_const | avg_const **−0.0002**, p2_const +0.0016, avg_vc **−0.0009**, avg_nl **−0.0024** | **failed** (disabled, so the test is not blocked) |

### 9g What validation shows (descriptive, validation only, not a claim)

| Finding | Evidence |
|---|---|
| Fusion with the frame raises Replica mIoU by 1.2–1.5 points over the map rows. Every fused row does this (32.7–33.0 vs 31.4–31.5); UG adds no mIoU over the constant weight (fuse_p2_ug 32.8 vs fuse_p2_const 32.9). | 9c |
| On Replica the gain of UG is in NLL, Brier, AURC and ECE vs its paired const: NLL pool 0.7702 vs 0.7806, ECE 5.23 vs 6.42, AURC 5.83 vs 6.12. Its ECE stays above ours_P2 alone (5.23 vs 4.88). | 9c |
| On ScanNet the frame adds little. The fitted weights are small (mean w 0.000–0.104 per scene), mIoU changes by ≤ 0.3 points, and NLL pool falls from 0.5473 (P2) to 0.5429 (@ug). AVG+2D NormLift and AVG+2D view-count beat @ug on validation NLL. | 9d, 9f |
| Holes are not the source of any gain: only 0.25% (Replica) and 0.08% (ScanNet) of labelled pixels are not map-covered, and the covered-only columns rank the rows the same way. | 9c, 9d |
| Our channels enter with the expected sign on both datasets: a_e and a_a are positive (more 2D weight where the map is uncertain). | 9e |

### 9h Open points for the lead (before any test stage)

1. **ScanNet: the gate failed (it is disabled).** By Q3 the test runs anyway. On validation, C1 against `fuse_avg_nl`, `fuse_avg_vc` and `fuse_avg_const` does not hold for NLL on ScanNet. A test look is likely to report "on par / below the fair fusion baselines" there.
2. **Extreme coefficients:** Replica fuse_p2_ug a_c = −163.4; ScanNet fuse_avg_nl (−390, −113, −271, a0 +9.76) and fuse_avg_vc (viewcount −70.6). These fits converged on validation pixels and are what the prereg fixes. I did not inspect the feature distributions (e.g. how many covered pixels have 1−α far from 0), so how these weights act on test pixels is unverified.
3. **Stale prereg wording:** the committed `status` field still reads "DRAFT … uncommitted"; `registered` is correct. Editing the prereg now changes its sha256, so `check-test` would refuse ("changed after the selection"). A correction would need a new validation selection, or a note elsewhere (e.g. in this report).
4. Q5: Table I only if C1 holds, else the appendix (lead decision).

- Correction (lead, 2026-10-24): configs/fusion_prereg.json "status" still reads "DRAFT ... uncommitted"; the file was registered and committed in 9671f02 before any real-data run ("registered" field). Left unedited so its sha256 stays the one the selection records.
