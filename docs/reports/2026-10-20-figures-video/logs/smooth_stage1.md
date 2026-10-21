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
