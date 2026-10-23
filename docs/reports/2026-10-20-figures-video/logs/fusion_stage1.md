# Fusion with the current view: stage 1

2026-10-24 | root: /home/thanhnc/gauss_ws/src/uags | git: master@2dbd2a1, +6 untracked new files (nothing committed) | part of [Figures, video and stronger results](../README.md)

**Verdict: stage 1 DONE.** The code, the CPU tests and the draft pre-registration are ready. No fused row has run on any real scene, validation or test.

**Next step:** the lead decides Q1-Q5 (section 7), sets `registered` in the prereg, and commits the prereg and the code. Only then can `STAGE=val` run. Until then the driver refuses every stage (section 5, row V9).

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
