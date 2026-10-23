# Fused-detector reset: stage 1

2026-10-23 | root: /home/thanhnc/gauss_ws/src/uags | git: master@914ace7, +6 untracked new files (nothing committed) | part of [Figures, video and stronger results](../README.md)

**Verdict: stage 1 DONE.** The code, the CPU tests and the draft pre-registration are ready. No variant has run on any real scene, validation or test.

**Next step:** the lead reviews the open decisions (section 6) and commits the prereg and the code. Only then will `STAGE=val` run (the driver refuses before that, see section 5).

## 1 What it is

| Item | Definition |
|---|---|
| Idea | A stronger detector decides **when** our field resets a Gaussian. Everything else stays ours: the reset itself, depth re-admission, the robust set and the coverage. |
| V1 | feat + depth (feature residual + rendered-depth residual) |
| V2 | ours + feat + depth (our persistent score c_p of the variant field + both residuals) |
| V3 | NormLift drop + depth |
| Component values | The per-view test statistic of each Table III row, computed as `eval_change_nor.py` computes it (k = 2 min, frozen map, shared coverage). |
| Fusion | `rank_X(x)` = mid-rank of x in component X's **mapping-pass** per-view null (pooled; NaN → 0). `s_V` = equal-weight mean of the ranks. It is defined where at least one component is finite. Nothing is fitted. |
| Why not `fuse_rank` | tab:fused ranks against the update population, so it has no per-view scale and cannot fire online. This is the per-view form of the same equal-weight rank average. |
| Threshold | `thr_V` = the 0.999 quantile of `s_V` over the mapping replay's covered Gaussian-views. This is the label-free rule every Table III row uses. V3 uses NormLift's own null segment. |
| Firing | Fire when the Gaussian is covered, `s_V > thr_V`, and it is not currently `changed`. Our own test is off (+inf thresholds). The reset is ours: `ExternalResetField.external_reset`. |
| Firing score (for AP) | ours: Table III's c_max. V: the max of `s_V` over the update views. |

## 2 Metrics per row (ours, V1, V2, V3)

| Metric | Rule | Better |
|---|---|---|
| stale-answer | N2, `dump_nor_qual.contrast_and_fa`: share of covered `masks_real` pixels whose top class is the scene's query | lower |
| static relabel | share of static observed Gaussians whose top class changed vs the mapping μ* | lower (cost) |
| G-AP / P-AP | firing score; `EC.ranking_table` / `EC.pixel_level` | higher |
| FA, Rec | reset at least once; `EC.operating_points`; plus realised fires/tests | FA lower, Rec higher |
| Kept | stale observed Gaussians in the robust set at the end | lower |

Context rows (no reset decision): before, change test off, averaging (stale-answer and static relabel).

## 3 Pre-registration: [configs/reset_fused_prereg.json](../../../../configs/reset_fused_prereg.json) (DRAFT, uncommitted)

| Rule | Content |
|---|---|
| Selection (validation 001-004) | Candidates are [ours, V1, V2, V3]. Pick the lowest mean stale-answer among the rows whose mean static relabel is ≤ ours' + 2 points. Ours always qualifies. An exact tie goes to the first listed row (ours). If ours is lowest, keep ours. |
| Claim (test, 10 scenes) | Only if the chosen variant (not ours) is strictly lower than ours in **≥ 7/10** scenes **and** has a lower test mean. The claim always carries x/10, both means, FA and static relabel. |
| Report | Every row on validation and test, per scene and as mean ± std. Rows not chosen are labelled "not selected". |
| Reproduction stop (exit 2) | Table III via `dump_nor_qual.reproduction` (tol 0.002, FA/Rec 0.5 pt), and ours / change-off / averaging / before N2 rates within 0.5 pt of the scene's dump. |
| Test once | `check-test` must pass first. A finished test record is never overwritten. |
| Disclosure | This is a further look at the NOR test split, made after Table III and tab:fused were final. |

## 4 Files (all new; no existing file edited)

| File | Role |
|---|---|
| [usplat/reset_fused.py](../../../../usplat/reset_fused.py) | `VARIANTS`, `MappingRanks`, `fused_stat`, `field_score`, `ExternalResetField` (`disable_own_test`, `external_reset`) |
| [scripts/reset_fused_nor.py](../../../../scripts/reset_fused_nor.py) | One scene: Table III replay with capture of the mapping components, ours via `EC.run_update` (captured), the reproduction stop, 3 variant passes (one at a time), metrics, N2, one JSON |
| [scripts/reset_fused_select.py](../../../../scripts/reset_fused_select.py) | `select`, `check-test`, `table` (claim + verdict) |
| [scripts/reset_fused.sh](../../../../scripts/reset_fused.sh) | `STAGE=val\|test\|table`, `DRY_RUN=1`. Gates: prereg and code tracked and clean; OUT is not a Table III or dump dir; the text cache is a COPY in OUT; test needs check-test. GPU through `gpu_gate.sh 6500`, headroom 1024. |
| [configs/reset_fused_prereg.json](../../../../configs/reset_fused_prereg.json) | Section 3 |
| [tests/test_reset_fused.py](../../../../tests/test_reset_fused.py) | 7 CPU tests (section 5) |

Monkeypatches (only inside the process, all undone in `finally` and asserted): `EC.DirectionalSemanticField`, `EC.residuals_view` and `EC.make_normlift` during the mapping replay; `dump_nor_qual.capture_fields` during ours' update; `DQ.SETS` during the N2 call; `EC.RULES` during `operating_points`.

## 5 Verification

Env as in the task. Tests: `python3 -m pytest -q -p no:cacheprovider tests/test_reset_fused.py` → **7 passed in 38.7 s** (2026-10-23 13:26-13:27).

| Check | Test | Result |
|---|---|---|
| external_reset = native reset, bit for bit, every state tensor after every view (default config **and** relmax / fast_min 0.5 as in the NOR fields); the field did reset; refuses when its own test fired | `test_external_reset_equals_the_native_reset[False/True]` | PASS |
| `run_variant` with our own rule equals `EC.run_update`'s ours (flags, resets, tests, fires, robust, changed, score, μ*) | `test_native_rule_through_run_variant_equals_run_update` | PASS |
| Toy scene: reproduces the record (Δ 0) and the dump's N2 (Δ 0); captured mapping samples = replay nulls; patches undone; the text cache unchanged; ours row = Table III ours; variant thresholds = independent recomputation (ranks by counting); variants reset the object (Rec > 0.5); coverage identical; refusals (finished test, wrong variants) | `test_scene_run` | PASS |
| Ranks / fusion on hand examples | `test_ranks_and_fused_stat` | PASS |
| Selection rule: budget, ties to ours, keep ours; refusals (unfinished, other prereg, mixed code) | `test_select_rule` | PASS |
| check-test (untracked / dirty prereg, missing or changed selection) in a temporary git repo; claim 7/10 allowed, 6/10 refused | `test_check_test_and_claim` | PASS |
| Regression: `tests/test_eval_change.py tests/test_dump_nor_qual.py` | existing | **14 passed** in 38 s |
| Driver refuses an untracked prereg | `STAGE=val DRY_RUN=1 scripts/reset_fused.sh` | REFUSED, exit 3 (as intended) |
| Frozen files untouched | `git status --short` | only the 6 new untracked files |
| Real data | none | NOT RUN (by the task) |

Toy sanity run (not evidence for the paper): all rows reach stale-answer 0 on the toy. The variants fire more than ours there (FA 1.0-4.1% vs 0.6%), so the toy cannot rank the rows.

## 6 Open decisions for the lead (fixed in the draft; change before commit if needed)

| # | Decision in the draft | Alternative |
|---|---|---|
| R1 | Rank scale = the mapping ECDF per component, pooled over history buckets | Per history bucket, as ours' and NormLift's own thresholds |
| R2 | No re-fire while `changed` (frozen-map residuals of a removed object stay high, so re-firing would drop the new evidence at every view) | Fire whenever `s_V > thr_V` |
| R3 | Ours is in the selection set and wins exact ties | Pick the best variant even when it is worse than ours |
| R4 | The static-relabel budget is +2 points on the validation mean (from the task). Note: ours' static relabel was 33% on 002 in the teaser curve, so it is a large baseline. | — |
| R5 | V2's "ours" component is the variant field's own c_p (after its own resets) | c_p of a parallel native ours field |
| R6 | GPU coverage tolerance: a variant's observed set may differ from ours' by ≤ 1e-4 (atomic backward sums); CPU must be exact | Exact on GPU too (risky stop) |

## 7 GPU estimate (not measured on real data)

| Item | Value |
|---|---|
| Gate | `gpu_gate.sh 6500`, headroom 1024 MiB |
| Peak | ≈ Table III's 4.56 GiB at N = 1e6. Variant passes run one at a time, each with 1 field copy (~1 GiB) + ChangeBaselines (+ NormLift copy for V3). The nulls and ranks stay on the CPU (~0.5 GB RAM). |
| Time | ≈ 1.3-1.6 × the N2 dump's wall time per scene (dumps: 71-968 s; 3645 s for 14 scenes). Estimate: ~25-40 min for validation and ~25-35 min for test, plus waiting at the gate. |

## 8 Open issues

- The prereg is a draft and is not committed. The lead must commit it together with the 5 code files, or the driver refuses to run.
- The test split has been seen before (Table III, Table IV, tab:fused, N2 dumps). This run is a further look and must be disclosed.
- The N2 query is generic in 4 test scenes ('wall' 009/013, 'phone' 010, 'handbag' 007/011; see [nor_summary.md](nor_summary.md) section 8). The stale-answer rate inherits this.
- `--dump` cross-checks need the existing dumps in `uags_results/qual/nor/<scene>`. They exist for all 14 scenes.

## 9 Validation attempt 1 (2026-10-23): STOPPED on a GPU OOM, fix pending a commit

| Step | Command | Result |
|---|---|---|
| Run | `STAGE=val scripts/reset_fused.sh` at afd693b (13:30) | 001 done (164 s, peak 4567 MiB). 002 **failed** with CUDA OOM inside the V1 pass (5.23 GiB allocated against the 6.13 GiB cap). The driver stopped. Logs: `uags_results/reset_fused/driver_val.log`, `logs/val_00{1,2}.20261023-133001.log` |
| Diagnosis | the same 002 scene through a wrapper that prints memory and calls `gc.collect()` before each variant (`logs/diag/diag.py`; output in scratch, **not a record**) | 2736 MiB were still allocated after ours' pass, and they were freed only by `gc.collect()`. The cause is a reference cycle: field → `capture_fields`' `Captured` class → its `from_state` closure → the `made` list → field. With the collection, 002 finished at peak 4533 MiB. |
| Fix | `scripts/reset_fused_nor.py`: `free_gpu()` = `gc.collect()` only, after ours' fields are deleted and before each variant pass. Per the lead and the gpu_reserve rule there is no `empty_cache()`; the `empty_cache()` after each variant pass that afd693b had is removed too. No computed number changes. | `tests/test_reset_fused.py`: 7 passed in 41 s (13:39). **Uncommitted**, so the driver refuses until it is committed. |
| Records | 001.json (status done, code afd693b), 002.json (status running) | Once the fix is committed, all 4 validation scenes must be rerun: `select` refuses records made with different code. |

The preliminary numbers below are not used for selection: 001 is from afd693b, and 002 is from the diagnostic wrapper.

| Scene | Row | stale-answer | static relabel | FA | Rec | Kept | G-AP | P-AP |
|---|---|---|---|---|---|---|---|---|
| 001 | ours | 96.3 | 8.48 | 1.62 | 20.3 | 23.0 | 22.7 | 56.4 |
| 001 | V1 | 96.4 | 8.47 | 0.30 | 79.2 | 4.5 | 81.7 | 68.2 |
| 001 | V2 | 96.3 | 8.43 | 0.16 | 62.0 | 8.4 | 78.2 | 68.4 |
| 001 | V3 | 96.3 | 8.41 | 0.15 | 59.6 | 7.9 | 74.4 | 69.4 |
| 002 | ours | 2.04 | 33.3 | 14.0 | 53.6 | 9.0 | 64.0 | 78.8 |
| 002 | V1 | 0.19 | 19.9 | 0.37 | 79.1 | 5.0 | 92.6 | 78.1 |
| 002 | V2 | 0.17 | 22.3 | 4.84 | 88.0 | 2.8 | 91.4 | 80.9 |
| 002 | V3 | 1.50 | 19.8 | 0.04 | 49.1 | 10.8 | 92.0 | 84.7 |

All values are %. On 001 the query is 'wall', so every row answers "wall" on about 96% of the object's pixels, and the scene cannot separate the rows.

## 10 Validation (2026-10-23, f0fc8ee): DONE, selection = V2 (by 0.008 points)

Command: `STAGE=val scripts/reset_fused.sh` at f0fc8ee, 13:40:11-13:56:38 (16.5 min including the selection). Records are in `/home/thanhnc/gauss_ws/data/uags_results/reset_fused/val/00{1..4}.json` and the selection in `.../reset_fused/selection.json` (git_head f0fc8ee, prereg sha 3f74b2f42a51). Driver log: `.../reset_fused/driver_val2.log`. The section 9 records were overwritten by this run.

**Validation means ± std over the 4 scenes (%; N = 4; lower is better except Rec, G-AP and P-AP)**

| Row | stale-answer | static relabel | FA | Rec | Kept | G-AP | P-AP | static rate | resets |
|---|---|---|---|---|---|---|---|---|---|
| ours | 24.77 ± 47.69 | 30.54 ± 16.96 | 5.78 ± 5.85 | 53.2 ± 24.6 | 10.3 ± 9.3 | 60.3 ± 27.0 | 68.3 ± 10.7 | 0.61 | 75,950 |
| V1 | 24.15 ± 48.15 | 26.89 ± 17.26 | 1.49 ± 1.72 | 78.6 ± 4.1 | 4.1 ± 2.3 | 86.9 ± 4.4 | 72.5 ± 8.1 | 0.19 | 64,314 |
| **V2 (chosen)** | **24.14 ± 48.13** | 27.76 ± 17.19 | 3.72 ± 3.07 | 78.7 ± 13.1 | 4.3 ± 3.5 | 85.7 ± 5.5 | 73.4 ± 9.0 | 0.40 | 78,561 |
| V3 | 24.52 ± 47.88 | 26.71 ± 17.16 | 0.78 ± 1.37 | 63.3 ± 13.4 | 6.9 ± 4.3 | 84.8 ± 7.8 | 74.9 ± 9.1 | 0.09 | 50,988 |

Selection (`reset_fused_select.py select`): the static-relabel limit is 30.54 + 2 = 32.54. All 4 rows qualify. The lowest mean stale-answer is V2 at 24.142, against V1 24.150, V3 24.516 and ours 24.771.

**Per scene stale-answer % (static relabel %)**

| Scene (query) | ours | V1 | V2 | V3 | change off | averaging | before |
|---|---|---|---|---|---|---|---|
| 001 (wall) | 96.30 (8.48) | 96.37 (8.47) | 96.34 (8.43) | 96.33 (8.41) | 96.27 | 96.28 | 60.56 |
| 002 (chair) | 2.04 (33.29) | 0.19 (19.88) | 0.17 (22.34) | 1.50 (19.82) | 20.56 | 19.96 | 33.19 |
| 003 (book) | 0.00 (30.65) | 0.00 (30.07) | 0.00 (30.72) | 0.00 (29.71) | 0.00 | 0.00 | 51.88 |
| 004 (chair) | 0.75 (49.73) | 0.05 (49.15) | 0.06 (49.56) | 0.23 (48.89) | 51.20 | 51.16 | 56.95 |

**Checks per scene**

| Scene | Table III repro: max AP/AUROC Δ, FA/Rec Δ, frames | N2 vs dump (max Δ) | null identity | coverage diff (V1/V2/V3) | replay rel dW1 | wall s | peak MiB |
|---|---|---|---|---|---|---|---|
| 001 | 1.7e-8, 0, equal | 0 | PASS | 0/0/0 | 6e-8 | 136 | 4567 |
| 002 | 1.7e-8, 0, equal | 0 | PASS | 0/0/0 | 7e-8 | 279 | 4566 |
| 003 | 1.2e-7, 0, equal | 0 | PASS | 0/0/0 | 9e-8 | 197 | 4564 |
| 004 | 1.8e-8, 1.6e-5, equal | 0 | PASS | 0/0/0 | 8e-8 | 364 | 4567 |

Timing (s): variant passes 6-23 each. The N2 step was the slowest variable part (16-177). The rest was load 19-27, replay 24-35 and Table III 27-35.

**Reading (for the lead, before test):**
- The choice is decided by noise. V2 beats V1 by 0.008 points on the mean. 001 ('wall' query, about 96% for every row, including change off) dominates both the mean and the std, and 003 is 0 for every row. Only 002 and 004 separate the rows: there all three variants are lower than ours (002: 0.2-1.5 vs 2.0; 004: 0.05-0.23 vs 0.75).
- The variants are better than ours on the cost side as well: static relabel −2.8 to −3.8 points, FA 0.8-3.7% vs 5.8%, Rec 63-79% vs 53%, and G-AP 85-87 vs 60.
- Under the claim rule (≥ 7/10 strictly lower and a lower mean), scenes where every row is 0, such as 003, count as non-wins. Several test scenes may be ties of this kind.
- STOPPED before test, as instructed. `check-test` has not been run.
