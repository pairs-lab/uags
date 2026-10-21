# Spatial smoothing (BKI spatial kernel): stage 1

2026-10-21 | root: /home/thanhnc/gauss_ws/src/uags | git: master@fdc5458, +5 untracked new files (nothing committed) | part of [Figures, video and stronger results](../README.md)

**Verdict: stage 1 DONE.** The code, the CPU tests and the pre-registration are ready. No smoothed row has run on any real scene, validation or test.

**Update 2026-10-21 (lead decisions D1-D5, before any real-data run):** @avg_own is binding in C1; the gate g = 0.8 stays as a candidate; both datasets run; h per scene; the scale grid is extended to s ∈ {1, 2, 4, 8} (24 candidates, 4 batches). The prereg now reads `"registered": "2026-10-21, before any real-data run"`; it is still uncommitted.

**Next step:** the lead commits the prereg. Only then may `STAGE=val` run.

## 1 What it is

| Item | Definition |
|---|---|
| Gap it fills | BKI pools observations over space and over views. Our field pools over the views of one Gaussian only. |
| Kernel | `k_nm = exp(-‖x_n - x_m‖² / (2σ²)) · gate_nm` over the k nearest other centres. Self weight = 1. `σ = s·h`, with h = the scene's median nearest-neighbour distance. |
| Gate | `cos(d_n, d_m) ≥ g`, where d is the row's own unsmoothed direction (ours: μ*; averaging: its feature). A pair passes if either side has no evidence. |
| Our sums | `R' = R + Σk R_m`, `W1' = W1 + Σk W1_m`, `Wr' = Wr + Σk Wr_m`, `W2' = W2 + Σk² W2_m`, `Q' = Q + Σk² Q_m`. Each neighbour view enters with weight k·ω. |
| Read-outs | The field's own code runs on the pooled sums (a shallow copy with the raw channel replaced). The saved field is never changed. |
| u_epi, u_ale of the smoothed rows | **Recomputed** from the pooled sums: `u_epi' = 1/(1+W1')` and `u_ale' = 1 - ρ̂(R', W1', W2', Q')`, defined once W1' ≥ τ_min = 3. |
| AUSE in Table I | The AUSE_u column of a smoothed row uses its own recomputed u_ale' (ranking signal `u_ale`). The unsmoothed P2 keeps its own u_ale. |
| Rows per kernel | `ours_P0s_<tag>` (one temperature). `ours_P2s_<tag>` (P2 refit: β0, γ_e, γ_a, γ_u). `averaging_s_<tag>` (the averaging F and S1 pooled with the same kernel, one temperature). |
| Known bias | A neighbour seen in the same view is a correlated observation, but BKI-style pooling counts it as independent. So u_epi' and u_ale' come out too confident. P2 rescales them; validation decides. |

## 2 Grid (fixed from first principles; no scene was read)

Grid: s ∈ {1, 2, 4, 8} × k ∈ {8, 16, 32} × g ∈ {none, 0.8} = **24 candidates per dataset**, the same on both datasets, plus the unsmoothed `ours_P2`. The lead added s = 8 (D5) before any data.

| Axis | Reason |
|---|---|
| unit h | The kernel follows the map density, which differs between Replica and ScanNet maps of the same 10⁶ budget. |
| s | On a uniform surface the kernel pools `2 ln2 · s²` neighbours: 1.4, 5.5, 22 and 89. s = 0.5 (0.35 neighbours) is almost the identity, which ours_P2 already is. |
| s ≤ 8 | One LSeg patch (16 px) is about 10 cm at 2 m in Replica. h is about 0.5 cm for 10⁶ Gaussians (0.58 cm on a synthetic 6×5×3 m room). So σ = 8h ≈ 4 cm stays inside one feature cell. |
| k | The k-th neighbour is at about h·√(k/ln2): 3.4h, 4.8h, 6.8h (5.8h measured at k = 32). At s ≤ 2, k = 32 leaves the kernel untruncated; s = 4 is truncated (weight 0.24 at the 32nd); at s = 8 every k truncates (weight 0.70), so s = 8 is close to a uniform kNN mean of size k. |
| g | none = plain BKI. 0.8 blocks pairs more than 37° apart. Not grounded in data: see open question Q2. |
| order | Least smoothing first (s, then k, then gated before ungated). Ties go to the first listed, and `ours_P2` is first. |

Kernel mass measured on a synthetic 10⁶-Gaussian room (mean neighbours pooled, self excluded):

| s | k = 8 | k = 16 | k = 32 |
|---|---|---|---|
| 1 | 1.59 | 1.65 | 1.66 |
| 2 | 4.47 | 6.10 | 6.99 |
| 4 | 6.78 | 11.9 | 18.8 |
| 8 | not measured (added after the benchmark) | | |

Expected: at s = 1 the three k values are near-duplicates, so the tie rule keeps the smallest k.

## 3 Pre-registration: `configs/smooth_prereg.json` (registered 2026-10-21, before any real-data run; uncommitted)

| Rule | Content |
|---|---|
| Selection (per dataset) | `@smooth`: the lowest pixel-pooled validation NLL among `ours_P2` and the 24 `ours_P2s` (make_tables.choose_v2; tie tol 1e-4 nats, first listed wins). |
| Paired baseline | `averaging_s` with the SAME kernel as @smooth (never chosen on averaging's own numbers). |
| Extra baseline (binding, D1) | `@avg_own`: averaging with its own validation-best kernel (among averaging and the 24 averaging_s), the strongest fair smoothed averaging. |
| Scenes | Replica: val office_0, room_2; test 6 scenes. ScanNet: val scene0000_00, scene0062_00; test 8 scenes. These are the existing Table I fields (read-out only). |
| Batches | One eval_seg run per σ (s1, s2, s4, s8). Each run holds the 6 kernels plus the reference rows ours_P0, ours_P2 and averaging, which must agree across batches within 1e-6. |
| Report | Every candidate on validation and on test. Rows not chosen are labelled "not selected". Also: ours_P2s vs ours_P2 (descriptive), h per scene, gate-blocked share. |
| C1 (gain claim) | Ours_s beats averaging, averaging_s (same kernel) and @avg_own (binding) on the test mean **and** in more than half of that dataset's test scenes (≥4/6, ≥5/8). R2 of quant_prereg (both aggregations) applies to calibration. |
| C2 ("significant") | Sign test over the pooled test scenes of the datasets where smoothing was selected, Holm over family F3s (3 baselines × 5 metrics = 15 tests). One dataset alone cannot reach Holm p < 0.05. |
| Disclosure | A later attempt after Table I was final; kernel chosen on validation among 24 fixed candidates; test scored once; Replica = a further test look, ScanNet = the second look; every candidate in the appendix. |
| Test once | check-test refuses unless validation is finished at one clean commit equal to the tree and the selection reproduces. Finished test batches are never rerun; look_test.json records each attempt. |

## 4 Files (all new; no existing file was edited)

| File | Role |
|---|---|
| [usplat/smoothing.py](../../../../usplat/smoothing.py) | SmoothSpec, GRID (24), kNN (kd-tree; torch chunked reference), pool, smoothed_field / smoothed_averaging, lazy rows, `row_factory` (map from the field checkpoint, checked by sha256 and N; aux identity for eval_seg). |
| [scripts/smooth_select.py](../../../../scripts/smooth_select.py) | `batches`, `select` (merge batches, consistency check, choose_v2, Table I independent check), `check-test`, `look`, `table` (C1), `claim` (F3s, sign test + Holm from quant_stats.py). |
| [scripts/smooth_tab1.sh](../../../../scripts/smooth_tab1.sh) | `DATASET=replica\|scannet STAGE=val\|test\|table`. Gates: prereg committed, code tracked, clean tree, prereg scenes, OUT never a Table I directory, field built on the scene's map, text-cache COPY in OUT, `gpu_gate.sh` at the prereg's cap, DRY_RUN. |
| [configs/smooth_prereg.json](../../../../configs/smooth_prereg.json) | The draft of section 3. |
| [tests/test_smoothing.py](../../../../tests/test_smoothing.py) | 9 CPU tests (section 5). |

Do-not-edit list respected: no change to usplat/*.py (existing), eval_seg.py, eval_change_nor.py, run_mapping.py, make_tables.py, or any file hashed by EC.code_state / MAPPING_CODE (`git status`: only the 5 untracked files above).

## 5 Verification

Command: `cd uags && PATH=$HOME/venvs/uags/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -q -p no:cacheprovider tests/test_smoothing.py` → **9 passed in 82 s** (2026-10-21, after the 24-candidate update; 9 passed in 77 s before it).

| Check | Test | Result |
|---|---|---|
| Grid and batches of the code = prereg (24 candidates, batches s1, s2, s4, s8) | test_grid_matches_prereg | PASS |
| kNN: kd-tree and torch = float64 brute force; self dropped with duplicate centres | test_knn_backends_equal_bruteforce | PASS |
| Pooled sums and μ*, u_epi, u_ale = brute-force reference (gate off / on); averaging too | test_pool_equals_bruteforce[None, 0.3] | PASS |
| s = 0 (and underflowing s) gives the unsmoothed P0, P2 and averaging bit for bit | test_sigma_zero_reproduces_unsmoothed | PASS |
| Gate blocks the other object (R'·e1 = 0 exactly; ungated > 0.1); unseen Gaussians borrow | test_gate_blocks_disagreeing_neighbours | PASS |
| Refusals: centred channel, no map, changed map sha, wrong N, no averaging sums | test_refusals | PASS |
| eval_seg mock pipeline: s = 0 rows score exactly as ours_P0/P2/averaging (same betas, metrics); P2s has 4 P2 params and u_ale ranking; the registered grid runs; batches give identical reference betas | test_row_factory_eval_seg_mock | PASS |
| Driver end to end (scratch git repo, mock): refusals (test before val, test scenes as val, Table I OUT, EVAL_EXTRA --rows, uncommitted prereg, dirty tree, code changed after val); val → select → test once → table + claim; second test and val after test refused | test_smooth_driver | PASS |
| Table I independent-check parser on the saved Table I val records (self-comparison = 0) | inline, JSON only | PASS |
| Real validation / test scenes | — | NOT RUN (forbidden in stage 1) |

Synthetic CPU benchmark (N = 10⁶, d = 64, a 6×5×3 m room surface; no real data; script kept in the session scratchpad only):

| Step | Time / memory |
|---|---|
| cKDTree build + query k = 33 | 0.6 + 1.1 s, RSS 1.1 GB |
| `knn()` total (sort, sha256 cache key) | 3.9–5.1 s |
| pool one kernel (ours or averaging) | 1–7 s (k = 8 → 32; the gate doubles it) |
| freeze one batch (6 kernels × 3 rows) | 41–45 s (measured for s ≤ 4; the cost depends on k, not s, so s8 should be the same) |
| peak RSS, whole benchmark | 8.4 GB |
| GPU memory used by kNN / pooling | 0 (CPU) |

## 6 GPU estimate (not measured on real data)

Per batch (21 rows). Render cost per row and view is 0.013–0.036 s, taken from the measured Table I eval_s (Replica v1 9–20 s for 90 views × 17 rows; ScanNet v2 scene0000_00 364 s for 279 views × 36 rows). CPU per scene load is 30–90 s (field, frames, baselines) plus 50 s of kNN + pooling.

| Stage | Loads × views | Wall per batch | × 4 batches | GPU render part (4 batches) |
|---|---|---|---|---|
| Replica val (2 scenes, fit + eval) | 4 × 90 | 7–12 min | 28–48 min | 7–18 min |
| ScanNet val (279 + 37 views) | 4 loads, 632 view-passes | 9–17 min | 36–68 min | 12–32 min |
| **Validation total** | | | **1.1–1.9 h** | 18–50 min |
| Replica test (6 scenes) | 6 × 90 | 10–18 min | 40–72 min | 10–27 min |
| ScanNet test (8 scenes, 840 views) | 8 loads | 16–29 min | 64–116 min | 15–42 min |
| **Test total** | | | **1.7–3.1 h** | 25–69 min |

- Cap: EVAL_NEED 6000 MiB per batch. Estimated peak ≈ 4.0 GB: 0.5 GB of reference buffers + 6 × 0.5 GB of candidates + 0.5 GB of render workspace. The v1 measured peak was 2.51 GiB with about 2 GB of buffers.
- The gate needs 8.0 GB free. At the time of writing, 3.8 of 12.3 GB were in use.
- The driver prints each batch's measured peak. Check the first batch before the next ones.

## 7 Decisions and open questions

| # | Question | Decision / status |
|---|---|---|
| Q1 | Is `@avg_own` binding in C1? | D1: binding |
| Q2 | g = 0.8 has no data behind it (on the CPU toy, noisy few-view μ* made the gate block 94% of pairs). | D2: kept as a candidate next to none; selection decides; the factory logs the blocked share |
| Q3 | Another Replica test look? | D3: both datasets; the paper discloses the extra looks |
| Q7 | h per scene or per dataset? | D4: per scene |
| — | Scale grid | D5: s ∈ {1, 2, 4, 8}, 24 candidates, 4 batches |
| Q4 | The kernel treats neighbour views as independent, so u_epi' and u_ale' are too confident; the AUSE_u of ours_P2s may change for that reason alone. | Open: reported as is |
| Q5 | The kernel uses centre distance only (no Gaussian scale or shape). | Open: out of scope |
| Q6 | Commit the prereg; the paper's "we run them once" sentence must cover this look. | Lead commits |

## 8 Files of this note

This note is the only docs file written. The synthetic benchmark script and its log stayed in the session scratchpad and are not linked (their numbers are copied into section 5). The new code files are listed in section 4.

## 9 Validation attempt 1 (2026-10-21): crashed, fixed, waiting for a commit

| Step | Command / source | Result |
|---|---|---|
| Dry run, both datasets | `DATASET=replica\|scannet STAGE=val DRY_RUN=1 scripts/smooth_tab1.sh` at cafb61d | PASS: prereg scenes; fields built on their own maps; averaging sums present; 4 batches each |
| Replica val, batch s1 | `DATASET=replica STAGE=val scripts/smooth_tab1.sh` (17:53–17:57) | **FAILED** 3/3 attempts. `eval_seg`: `row 'averaging_s_k8_s1_g0.8': buffer has non-finite values on 3 of 620290 Gaussians`. No batch finished and no metric was produced or read. Record: `data/uags_results/smooth/replica/eval_val_s1.json` (status failed) |
| Root cause | CPU debug on office_0 | 3 isolated, unseen Gaussians (S1 = 0, F = 0) whose 8 neighbours are far away: w ≈ 1e-43 (denormal). S1' was denormal but > 0, and F' was denormal. `Averaging.feature`: F'/(S1' + 1e-12) ≈ 1e-30, whose norm underflows to 0, so the feature became inf |
| Fix | `usplat/smoothing.py`: `W_MIN = 1e-12`; kernel weights below it are set to 0 (neighbours beyond 7.4σ) | Each pooled sum changes by less than 1e-12 of a neighbour's evidence. s = 0 is still exact |
| Prereg | `configs/smooth_prereg.json`: new key `amended` plus one clause in `method.kernel` | Nothing else changed |
| New test | `tests/test_smoothing.py::test_underflow_guard_keeps_features_finite` | FAILS with the guard off (`W_MIN = 0`), PASSES with it. Whole file: **10 passed in 56 s** |
| CPU check, all 24 kernels × 3 rows on the 4 validation fields (buffers finite; u finite or NaN; no metric computed) | a session-scratchpad script (not kept): it builds each field's SmoothingContext and freezes every row | PASS on all 4 fields (table below) |

Results of the CPU check on the validation fields:

| Field | N | h (m) | Non-finite buffers | Max gate-blocked share (ours / averaging) | Time |
|---|---|---|---|---|---|
| office_0 | 620,290 | 0.00491 | 0 | 0.01% / 0.01% | 84 s |
| room_2 | 954,119 | 0.00391 | 0 | 0.00% / 0.00% | 147 s |
| scene0000_00 | 999,222 | 0.00751 | 0 | 0.09% / 0.09% | 141 s |
| scene0062_00 | 998,600 | 0.00290 | 0 | 0.03% / 0.03% | 156 s |

What this shows:
- The gate g = 0.8 blocks at most 0.09% of the pairs on real LSeg fields, so gated and ungated kernels are nearly the same rows. Under the tie rule the gated one (listed first) wins a tie.
- h (2.9–7.5 mm) matches the stage-1 estimate of about 0.5 cm.

Status: the fix is uncommitted (3 files modified). The driver refuses a dirty tree, so validation reruns after the lead commits.

## 10 Validation (attempt 2, code b4d6205, prereg sha256 832fc2f42f82)

Command: `DATASET=replica STAGE=val scripts/smooth_tab1.sh`, then `DATASET=scannet STAGE=val`.
Records: `/home/thanhnc/gauss_ws/data/uags_results/smooth/<dataset>/eval_val_s{1,2,4,8}.json` and `selection_val.json`.
Every batch was made at b4d6205 with `dirty_tracked` false.

### 10a Replica (18:10–22:14, 4.1 h wall; the CPU was shared with another project's 22-worker job, load 28–38)

**Selection: `@smooth` = `ours_P2` (smoothing NOT selected); `@avg_own` = `averaging` (no kernel helps averaging either).**

Pooled validation NLL in nats; lower is better; ours_P2 = 0.80034, averaging = 0.80959:

| Kernel (gated = ungated to 1e-5) | ours_P2s | Δ vs ours_P2 | ours_P0s | averaging_s |
|---|---|---|---|---|
| k8_s1 | 0.80059 | +0.00025 | 0.80948 | 0.80979 |
| k16_s1 | 0.80074 | +0.00040 | 0.80952 | 0.80983 |
| k32_s1 | 0.80086 | +0.00052 | 0.80954 | 0.80985 |
| k8_s2 | 0.80216 | +0.00182 | 0.80992 | 0.81020 |
| k16_s2 | 0.80232 | +0.00198 | 0.80999 | 0.81027 |
| k32_s2 | 0.80231 | +0.00197 | 0.81006 | 0.81033 |
| k8_s4 | 0.80355 | +0.00321 | 0.81095 | 0.81114 |
| k16_s4 | 0.80379 | +0.00345 | 0.81127 | 0.81143 |
| k32_s4 | 0.80392 | +0.00358 | 0.81149 | 0.81167 |
| k8_s8 | 0.80428 | +0.00394 | 0.81163 | 0.81172 |
| k16_s8 | 0.80505 | +0.00470 | 0.81225 | 0.81231 |
| k32_s8 | 0.80589 | +0.00555 | 0.81299 | 0.81302 |

Every one of the 24 kernels is worse than the unsmoothed row, for ours and for averaging, and NLL grows steadily with the kernel size.

Validation metrics. mIoU, ECE and AURC are the mean of the 2 scenes, in %. NLL is shown as both the scene mean and the pixel-pooled value:

| Row | mIoU | ECE | NLL mean / pooled | AURC | AUSE u_ale |
|---|---|---|---|---|---|
| ours_P2 (kept) | 31.50 | 4.91 | 0.8238 / 0.8003 | 6.15 | 5.03 |
| ours_P0 | 31.50 | 6.51 | 0.8328 / 0.8093 | 6.60 | 5.03 |
| averaging (= @avg_own) | 31.42 | 6.59 | 0.8332 / 0.8096 | 6.59 | – |
| averaging_s (same kernel) | n/a: no kernel chosen | | | | |

Checks:

| Check | Result |
|---|---|
| Reference rows (ours_P0, ours_P2, averaging) across the 4 batches | identical: max \|Δβ\| = 0, max \|Δmetric\| = 0 |
| Independent check against Table I v1 validation (`eval_val_224fe1e.json`, RTX 5070 Ti, code 224fe1e) | max \|Δβ\| 0.085 (P2 β0 45.47 vs 45.56), max \|Δ val NLL\| 1.7e-4: another GPU and code version; same predictive |
| Gate blocked share (logged, rounded to 0.1%) | 0.0% for every gated kernel. The CPU check gave 0.01% on office_0 and 0.00% on room_2 |
| Kernel mass, neighbours pooled (office_0 / room_2) | s1: 2.6–6.2; s2: 4.3–11.9; s4: 6.0–18.9; s8: 7.2–25.7 |
| h | office_0 4.91 mm, room_2 3.91 mm |
| Peak GPU memory, every batch | allocated 3.87 GiB, reserved 5.18 GiB (cap 6000 MiB) |
| Peak RSS | 13.6–14.4 GiB |
| Wall per batch | s1 33 min, s2 58 min, s4 72 min, s8 74 min. The stage-1 estimate was 7–12 min; CPU contention and the eval_seg metric code (CPU) dominate |

### 10b ScanNet (2026-10-21 22:14 to 2026-10-22 00:23, 2.1 h wall)

**Selection: `@smooth` = `ours_P2s_k32_s4_g0.8`** (pooled val NLL 0.54243 vs ours_P2 0.54725, Δ −0.00482).
- Tie rule: k32_s4 (0.54246) and k32_s8_g0.8 (0.54249) are within the 1e-4 tolerance, so the one listed first (k32_s4_g0.8) wins. The kernel is the same with or without the gate.
- **`@avg_own` = `averaging_s_k32_s8_g0.8`** (0.54358).
- **Paired baseline: `averaging_s_k32_s4_g0.8`** (0.54384).

Pooled validation NLL in nats; lower is better; ours_P2 = 0.54725, averaging = 0.54442. Gated and ungated values differ by at most 1.5e-4; the table shows the gated value:

| Kernel | ours_P2s | Δ vs ours_P2 | ours_P0s | averaging_s |
|---|---|---|---|---|
| k8_s1 | 0.54604 | −0.00121 | 0.54875 | 0.54448 |
| k16_s1 | 0.54592 | −0.00133 | 0.54872 | 0.54447 |
| k32_s1 | 0.54585 | −0.00140 | 0.54867 | 0.54441 |
| k8_s2 | 0.54496 | −0.00229 | 0.54826 | 0.54437 |
| k16_s2 | 0.54455 | −0.00270 | 0.54818 | 0.54451 |
| k32_s2 | 0.54424 | −0.00301 | 0.54806 | 0.54414 |
| k8_s4 | 0.54417 | −0.00308 | 0.54704 | 0.54398 |
| k16_s4 | 0.54328 | −0.00397 | 0.54667 | 0.54416 |
| **k32_s4** | **0.54243** | **−0.00482** | 0.54645 | 0.54384 |
| k8_s8 | 0.54412 | −0.00313 | 0.54633 | 0.54377 |
| k16_s8 | 0.54313 | −0.00412 | 0.54557 | 0.54380 |
| k32_s8 | 0.54249 | −0.00476 | 0.54544 | 0.54358 |

Validation metrics. mIoU, ECE and AURC are the mean of the 2 scenes, in %. NLL is shown as both the scene mean and the pixel-pooled value. These are validation numbers, already used for the selection: descriptive only.

| Row | mIoU | ECE | NLL mean / pooled | AURC | AUSE u_ale |
|---|---|---|---|---|---|
| **ours_P2s_k32_s4_g0.8 (chosen)** | 71.08 | **2.35** | **0.4031 / 0.5424** | 3.17 | **3.31** |
| ours_P0s_k32_s4_g0.8 (same buffer, mIoU row) | 71.08 | 3.52 | 0.4088 / 0.5465 | 2.92 | 3.31 |
| ours_P2 (Table I) | 71.07 | 2.73 | 0.4104 / 0.5472 | 3.20 | 3.83 |
| ours_P0 | 71.07 | 3.41 | 0.4112 / 0.5488 | 2.97 | 3.83 |
| averaging | **71.27** | 3.56 | 0.4092 / 0.5444 | 2.93 | – |
| averaging_s_k32_s4_g0.8 (same kernel) | 71.19 | 3.67 | 0.4079 / 0.5438 | 2.90 | – |
| averaging_s_k32_s8_g0.8 (@avg_own) | 71.13 | 3.69 | 0.4073 / 0.5436 | **2.89** | – |

How to read it:
- On validation, smoothing moves ours' NLL from worse than averaging (0.5472 vs 0.5444) to better than all three averaging rows (0.5424 vs 0.5436–0.5444). ECE also improves (2.35 vs 3.56–3.69).
- mIoU and AURC still favour the averaging rows.
- The P2 refit on the smoothed channels has more negative coefficients: γ_e −3.04 and γ_a −2.53, against −0.82 and −2.07 for ours_P2, and γ_u is 0.63. P2 therefore sharpens even more where uncertainty is high, the sign issue already reported for ScanNet.
- On Replica the same 24 kernels all lose (section 10a). The gain is therefore dataset-specific; the test stage decides whether it holds on the 8 ScanNet test scenes.

Checks:

| Check | Result |
|---|---|
| Reference rows across the 4 batches | identical: max \|Δβ\| = 0, max \|Δmetric\| = 0 |
| Independent check against Table I validation (`scannet_tab1/eval_val.json`, same GPU, earlier code) | max \|Δβ\| 0.053, max \|Δ val NLL\| 1.0e-4 |
| Gate blocked share (logged, rounded to 0.1%) | ≤ 0.1% for every gated kernel |
| Kernel mass of the chosen kernel k32_s4 | 18.6–18.9 neighbours (scene0000_00 / scene0062_00) |
| h | scene0000_00 7.51 mm, scene0062_00 2.90 mm |
| Peak GPU memory, every batch | allocated 3.92 GiB, reserved 5.18 GiB (cap 6000 MiB) |
| Peak RSS | 22.0–22.4 GiB |
| Wall per batch | 31–33 min |

### 10c Status

| Dataset | Selected | Next |
|---|---|---|
| Replica | ours_P2 (smoothing not selected; NEGATIVE on validation) | The test stage would only report every candidate as "not selected"; C1 does not apply |
| ScanNet | ours_P2s_k32_s4_g0.8; paired averaging_s_k32_s4_g0.8; @avg_own averaging_s_k32_s8_g0.8 | The test stage (once) decides C1/C2. **Not run: stopped as instructed** |

Nothing was committed. The validation records and selections are under `/home/thanhnc/gauss_ws/data/uags_results/smooth/{replica,scannet}/`.

## 11 Test, run once (code c36abde, prereg sha256 832fc2f42f82)

| Look | Command | Time | look_test.json |
|---|---|---|---|
| ScanNet test, 2nd look at its split | `DATASET=scannet STAGE=test scripts/smooth_tab1.sh`, then `STAGE=table` | 2026-10-22 00:24–05:14 (4.8 h; batches 66–80 min) | attempt 1, commit c36abde |
| Replica test, a further look | `DATASET=replica STAGE=test scripts/smooth_tab1.sh` (its table stage also ran the claim) | 05:14–08:26 (3.2 h; batches 42–51 min) | attempt 1, commit c36abde |

- Every test batch record has `dirty_tracked` false.
- Peak GPU allocation: 3.98 GiB (ScanNet), 4.07 GiB (Replica); cap 6000 MiB.
- Peak RSS: 18.9–19.2 GiB (ScanNet), 15.3–15.9 GiB (Replica).
- Nothing was changed after the test numbers were read.
- Records: `/home/thanhnc/gauss_ws/data/uags_results/smooth/{scannet,replica}/eval_test_s{1,2,4,8}.json`, `table_smooth.{json,tex}` and `claim_smooth.json`.

**Verdict: NEGATIVE.** On ScanNet the kernel chosen on validation (ours_P2s_k32_s4_g0.8) is worse than ours_P2 and than all three averaging rows on NLL, ECE, Brier and AURC. Claim rule C1 fails on every metric, so Table I keeps ours_P2, and the paper may say only that the spatial kernel did not help.

### 11a ScanNet test (N = 8, mean ± population std; % except NLL in nats)

| Row | mIoU | ECE | NLL | Brier | AURC | AUSE u_ale |
|---|---|---|---|---|---|---|
| **ours_P2s_k32_s4_g0.8 (chosen; mIoU from ours_P0s)** | 67.91 ± 10.20 | 6.61 ± 6.32 | 0.535 ± 0.264 | 22.42 ± 10.41 | 5.35 ± 2.41 | 8.01 ± 7.25 |
| ours_P2 (Table I) | 67.99 ± 10.39 | 6.06 ± 5.76 | 0.509 ± 0.232 | 22.14 ± 10.07 | 5.03 ± 2.25 | 8.23 ± 8.06 |
| averaging | 67.68 ± 10.36 | 6.17 ± 5.40 | 0.512 ± 0.232 | 22.29 ± 10.01 | 4.39 ± 1.89 | – |
| averaging_s_k32_s4_g0.8 (same kernel) | 67.58 ± 10.16 | 6.20 ± 5.43 | 0.512 ± 0.232 | 22.31 ± 9.98 | 4.40 ± 1.89 | – |
| averaging_s_k32_s8_g0.8 (@avg_own) | 67.57 ± 10.17 | 6.21 ± 5.43 | 0.512 ± 0.233 | 22.30 ± 9.97 | 4.40 ± 1.89 | – |

Scene-by-scene comparisons of the chosen row: x/8 = scenes where it is better; median difference in display units, positive = chosen row better; exact sign test p (unadjusted):

| Metric | vs ours_P2 (P0 for mIoU) | vs averaging | vs averaging_s same kernel | vs @avg_own |
|---|---|---|---|---|
| mIoU | 4/8, −0.01, p 1 | 3/8, −0.27, p 0.73 | 4/8, −0.03, p 1 | 4/8, +0.07, p 1 |
| ECE | 3/8, −0.25, p 0.73 | 3/8, −0.27, p 0.73 | 3/8, −0.28, p 0.73 | 3/8, −0.28, p 0.73 |
| NLL | 1/8, −0.007, p 0.070 | 2/8, −0.011, p 0.29 | 2/8, −0.009, p 0.29 | 2/8, −0.008, p 0.29 |
| Brier | 3/8, −0.11, p 0.73 | 2/8, −0.47, p 0.29 | 2/8, −0.45, p 0.29 | 2/8, −0.36, p 0.29 |
| AURC | 2/8, −0.27, p 0.29 | 0/8, −0.70, p 0.0078 | 0/8, −0.74, p 0.0078 | 0/8, −0.75, p 0.0078 |
| AUSE u_ale | 4/8, +0.06, p 1 | – | – | – |

Claim rules:

| Rule | Result |
|---|---|
| C1, per metric (better mean AND x/8 ≥ 5 against each of the 3 averaging rows) | **FAIL on all 5** (mIoU, ECE, NLL, Brier, AURC) |
| F3s (15 sign tests over 8 scenes, Holm) | No test favours ours. Smallest Holm p = 0.117: AURC 0/8 against each averaging row, i.e. **ours worse**. All others are 1.0. Nothing is significant in either direction |

- **Validation did not transfer.** On ScanNet validation all 24 kernels beat ours_P2 on NLL. On ScanNet test all 12 gated kernels are worse than ours_P2 (0.5144–0.5353 vs 0.5093), and the chosen kernel is the worst of them.
- **Likely cause (not tested):** the P2 refit on the smoothed channels (γ_e −3.04, γ_a −2.53, against −0.82 and −2.07 for ours_P2) fits the 2 validation scans and does not carry over.

### 11b Replica test (N = 6): smoothing not selected; every candidate "not selected"

Reference rows reproduce Table I:

| Row | mIoU | ECE | NLL | Brier | AURC | AUSE u_ale |
|---|---|---|---|---|---|---|
| ours_P2 | 27.96 ± 3.62 | 8.68 ± 5.57 | 0.996 ± 0.430 | 36.20 ± 13.31 | 12.24 ± 8.03 | 8.50 ± 5.48 |
| averaging | 27.91 ± 3.44 | 9.63 ± 3.94 | 1.004 ± 0.438 | 36.64 ± 13.40 | 14.00 ± 9.20 | – |

Every candidate on test, means (gated kernels; ungated differ by ≤ 2e-5 nats on Replica and ≤ 6e-5 on ScanNet). All rows are "not selected" except ScanNet k32_s4_g0.8:

| Kernel | Replica P2s NLL | Replica P2s ECE | Replica avg_s NLL | ScanNet P2s NLL | ScanNet P2s ECE | ScanNet avg_s NLL |
|---|---|---|---|---|---|---|
| reference ours_P2 / averaging | 0.9955 / 1.0036 | 8.68 / 9.63 | – | 0.5093 / 0.5116 | 6.06 / 6.17 | – |
| k8_s1 | 1.0003 | 8.53 | 1.0038 | 0.5157 | 6.16 | 0.5118 |
| k16_s1 | 1.0004 | 8.50 | 1.0038 | 0.5161 | 6.18 | 0.5118 |
| k32_s1 | 1.0003 | 8.47 | 1.0038 | 0.5162 | 6.18 | 0.5119 |
| k8_s2 | 0.9982 | 8.28 | 1.0043 | 0.5213 | 6.31 | 0.5118 |
| k16_s2 | 0.9983 | 8.21 | 1.0044 | 0.5239 | 6.40 | 0.5120 |
| k32_s2 | 0.9986 | 8.17 | 1.0043 | 0.5255 | 6.45 | 0.5121 |
| k8_s4 | 0.9971 | 8.14 | 1.0050 | 0.5202 | 6.30 | 0.5116 |
| k16_s4 | 0.9958 | 8.06 | 1.0052 | 0.5266 | 6.44 | 0.5118 |
| k32_s4 | 0.9942 | 8.07 | 1.0052 | **0.5353 (chosen)** | 6.61 | 0.5122 |
| k8_s8 | 0.9983 | 8.13 | 1.0053 | 0.5144 | 6.19 | 0.5114 |
| k16_s8 | 0.9984 | 8.06 | 1.0058 | 0.5183 | 6.27 | 0.5116 |
| k32_s8 | 0.9981 | 8.07 | 1.0062 | 0.5260 | 6.40 | 0.5122 |

- On Replica test one candidate (k32_s4, 0.9942) is below ours_P2 (0.9955). It was worse on validation, so it was not selected; this is a post hoc observation, and no choice changes.
- The validation ranking does not predict the test ranking on either dataset.

### 11c LaTeX rows (generated from the test records by a scratchpad script, `test_report.py`; ScanNet columns only)

These are the prereg's Table I rows for ScanNet. C1 fails, so they belong in the appendix with "did not help"; the main table keeps ours_P2.

```latex
%% generated from uags_results/smooth/scannet/eval_test_s*.json (ScanNet: mIoU Acc ECE NLL Brier AURC AUSE_u; means over 8 test scenes)
Ours P2, spatial kernel & 67.9 & 86.5 & 6.6 & 0.54 & 22.4 & 5.4 & 8.0 \\
Averaging, same kernel & 67.6 & 86.3 & 6.2 & 0.51 & 22.3 & 4.4 & -- \\
Averaging, own kernel & 67.6 & 86.3 & 6.2 & 0.51 & 22.3 & 4.4 & -- \\
```

The pipeline's own candidate tables are in `uags_results/smooth/{scannet,replica}/table_smooth.tex`, with every row, marked `%% not selected`.

Proposed disclosure sentence, per the prereg: "After Table I was final we tried a spatial kernel that pools each Gaussian's statistics with its k nearest neighbours, as in BKI. Validation chose a kernel on ScanNet and none on Replica among 24 candidates fixed in advance. On the ScanNet test scenes (scored once) it did not beat P2 or averaging (NLL 0.535 vs 0.509 and 0.512; appendix)."
