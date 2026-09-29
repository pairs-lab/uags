# Method completion: fixes F1-F9
2026-09-28 | root: /home/thanhnc/gauss_ws/src/uags | git: master@ab28fb4 +dirty (uncommitted; [logs/code.patch](logs/code.patch)) | [all reports](../INDEX.md)
Follows: [UAGS paper completion plan](../2026-09-28-paper-completion-plan/README.md)
Verdict: DONE -- all plan fixes F1-F9 implemented behind switches, 41 CPU + 1 GPU test pass, 39/40 review findings fixed; change detection on REAL rearrangements not tested yet (gate G3)
Key result: change-score AUROC, real CLIP, simulated change, unseen scene (office): old 1 - r̄ 0.739 -> ours c_last 0.824 (higher is better; chess, the tuning scene: 0.704 -> 0.750) ([logs/g1_office_postreview.json](logs/g1_office_postreview.json), [logs/g1_chess_postreview.json](logs/g1_chess_postreview.json))
CORRECTION 2026-09-29: every real-CLIP number in this run (attempts 01-04, finding rows 4-7, the key result) was measured with a BROKEN extractor: usplat/clip_dense.py fed open_clip's batch-first transformer a sequence-first tensor, so no attention across tokens happened (zero-shot pixel accuracy ~1% on Replica). The numbers are kept as measured; they describe near-context-free patch features, not CLIP. Re-measured with the fixed extractor in [2026-09-29-data-and-baselines](../2026-09-29-data-and-baselines/README.md).
Attempts: 4 (2 Good / 2 Bad) | Time: 2026-09-28 23:35 -> 2026-09-29 00:47 +07 (`date` stamps)

## 0 Context
- Ask: "complete our method" = implement the plan's method fixes (plan §2, F1-F9) before any dataset run.
- Success test: every fix behind a config switch (old behaviour reproducible for the ablations), unit-tested,
  the gradient identity checked per Gaussian on real gsplat (GPU), and the change signal checked on REAL
  CLIP features before RIO10 arrives.
- Real-feature check (gate G1; internal, not a paper result): 7-Scenes seq-01 (usgloc data, read-only),
  80 frames, CLIP ViT-B/16 laion2b at full resolution, PCA-64 and centre from the first half only.
  Simulated change = a point's second-half observations swapped with a partner > 0.5 m away.
  Split: chess = tuning scene, office = unseen scene, run once per frozen config. One seed.
- Machine: RTX 5070 Ti 16 GB shared with USGLoc jobs (GPU used only via [scripts/gpu_gate.sh](../../../scripts/gpu_gate.sh));
  replay on CPU. torch 2.11+cu128, gsplat 1.5.3, open_clip 3.3.0, numpy 1.26.4 (system).

## 1 Findings
| # | Finding | Evidence |
|---|---|---|
| 1 | Evidence now counts views (F1): W1 = k views at any resolution; the old pixel count grew 9.0x for 3x resolution and called 100% of Gaussians stable after ONE view | [tests/test_view_count.py](../../../tests/test_view_count.py), [logs/cpu_tests_run2.log](logs/cpu_tests_run2.log) |
| 2 | The joint-prior agreement ‖R‖/(W1+c0) mixes the amount of evidence into u_ale; the Kish estimator is unbiased for ρ² without depending on n (Monte-Carlo: 0.1576 vs 0.1586; r̄² 0.1974). Kish is the aleatoric default; the joint prior serves κ̂ and the predictive. NormLift's R(j) has the joint-prior form, so the aleatoric channel is now distinct from it. | [tests/test_estimators.py](../../../tests/test_estimators.py) |
| 3 | A slow channel that absorbs the newest views hides the change (c peaks at 0.48 by hand); fixed with disjoint recent/older buckets | [tests/test_change.py](../../../tests/test_change.py) |
| 4 | On real CLIP, the plan defaults (β=0.5, thr 0.5) fire on 90% of static points; β=0.9 ranks best | [attempts/01](attempts/01-default-change-test.md), [attempts/02](attempts/02-beta-and-threshold.md) |
| 5 | A fixed threshold on c does not transfer between scenes (static q95 0.71 chess vs 0.47 office); the per-scene quantile of the mapping pass does (27.2/6.0% vs 27.4/5.7% at q=0.99) | [attempts/02](attempts/02-beta-and-threshold.md), [attempts/04](attempts/04-selfcalibrated-threshold.md) |
| 6 | Normalising c by an i.i.d. no-change null fails on real data even after the review's fix (0.750 -> 0.623 chess): consecutive views are correlated | [attempts/03](attempts/03-normalized-score.md) |
| 7 | Centring helps the change test: raw-channel c_max 0.667/0.679 vs centred 0.750/0.772 (chess/office) | [logs/g1_chess_postreview.json](logs/g1_chess_postreview.json), [logs/g1_office_postreview.json](logs/g1_office_postreview.json) |
| 8 | Adversarial code review: 40 confirmed defects (2 rejected), incl. "change test never fires at the defaults", a missing null term, NaN scores silently dropped, no re-admission without depth; 39 fixed | [logs/code_review.md](logs/code_review.md) |
| 9 | Side effect: my `pip install plyfile` put numpy 2.5.3 in ~/.local over system numpy 1.26.4 (shared with USGLoc); reverted within ~1 min | §5 |
| 10 | The repo tracks `usplat/__pycache__/*.pyc`; a test run rewrites them. Restored to HEAD; suggest `.gitignore` | `git status` |

## 2 Baseline
The pre-fix field, reproduced exactly by `FieldConfig.legacy()`: n_eff, R, μ*, κ*, epistemic, aleatoric, r̄ and one
forgetting step match the code at git ab28fb4 loaded from git ([tests/test_legacy.py](../../../tests/test_legacy.py)).
As a change score on the same replay observations its 1 - r̄ gives AUROC 0.704 (chess) / 0.739 (office).

## 3 Attempts
| # | idea | result | critic | link |
|---|---|---|---|---|
| 01 | plan defaults: centred c, β=0.5, thr=0.5 | chess c_max 0.614 < old 0.704; 90% of static fire | Bad: short memory, threshold below noise | [01](attempts/01-default-change-test.md) |
| 02 | β grid on chess, thr = static q95; office once | office c_last 0.824, c_max 0.772 vs old 0.739, resid 0.692 | Good ranking; Bad operating point (does not transfer) | [02](attempts/02-beta-and-threshold.md) |
| 03 | normalise c by an i.i.d. no-change null (retried after the review fixed the null) | chess 0.623, office 0.644 vs raw 0.750, 0.772 | Bad: views are correlated; ablation-only | [03](attempts/03-normalized-score.md) |
| 04 | threshold = q-quantile of the scene's own mapping-pass scores | 27.2/6.0% vs 27.4/5.7% at q=0.99 | Good: transfers; FA above nominal; modest power on the proxy | [04](attempts/04-selfcalibrated-threshold.md) |

## 4 Simplify
not done: the ablation table on real data (plan Tab III) decides which parts stay. Already dropped from the defaults
on evidence: β=0.5, fixed thresholds, the normalised score, the joint prior for u_ale.

## 5 Verification
| Check | Command | Result | Status |
|---|---|---|---|
| CPU tests (41) | `PYTHONPATH=. python3 -m pytest -q tests --ignore=tests/test_gsplat_gpu.py` | 41 passed | PASS [logs/cpu_tests_run3.log](logs/cpu_tests_run3.log) |
| GPU: linearity, mass, per-Gaussian identity, invariants, change | `scripts/gpu_gate.sh 1500 python3 tests/test_gsplat_gpu.py` | identity err 2.4e-6 (R), 1.5e-5 (mass, max 241.7) | PASS [logs/gpu_test_run3.log](logs/gpu_test_run3.log); first run failed on the uncalibrated default, fixed ([log](logs/gpu_test_run3a_failed_uncalibrated.log)) |
| Mock end to end (defaults + calibration) | `python3 scripts/run_mapping.py --backend mock` | AUROC 1.000; flagged 100% changed / 0% unchanged | PASS [logs/cpu_tests_run3.log](logs/cpu_tests_run3.log) |
| Legacy preset = pre-fix code | `pytest tests/test_legacy.py` | all quantities equal | PASS |
| I1 reproducible | commands above; uncommitted code in [logs/code.patch](logs/code.patch) | patch saved (3338 lines) | PASS |
| I2 tests not weakened to pass | every changed assertion has a stated reason (footprint ground truth; orthogonal content; mechanics tests pin β/threshold; calibrated default flow now tested); the review flagged weak tests and they were strengthened | [logs/code_review.md](logs/code_review.md) #8, #25-39 | PASS |
| I4 independent check | office run once per frozen config; code checked by an adversarial review | done | PASS |
| Env restored after the pip side effect | `python3 -c "import numpy; print(numpy.__version__)"` | 1.26.4 from /usr/lib/python3/dist-packages | PASS |
| Change detection on real rearrangements | RIO10 | no data yet | NOT RUN: needs the RIO10 access form (plan D2) |

## 7 Open issues
- Power on the proxy is modest (27% detected at 6% false alarms); only real RIO10 changes can judge it (gate G3).
- Static false alarms above nominal when the update window is longer than the mapping window; match windows.
- One seed, simulated change, AUROC standard error ~0.016: differences below ~0.03 are not resolved.
- Memory not optimised (review #14): 4 (N,d) tensors + ~1.8 GB transient per view at N=1e6, d=64; the plan budget must use these.
- The depth re-admission (F5) is unit-tested only; no real depth run yet.
- `usplat/__pycache__/*.pyc` are tracked in git; add `__pycache__/` to `.gitignore` (not done: repo policy is yours).

## 8 Files
- Changed: usplat/{vmf_field, bki_accumulate, consolidation, efe_planner, feature_pca, clip_dense, gsplat_adapter, evaluation, datasets, __init__}.py,
  scripts/run_mapping.py, tests/{test_pipeline_dryrun, test_gsplat_gpu}.py.
- New: usplat/{vmf_math, predictive}.py, scripts/{gpu_gate.sh, cpu_check_change.py},
  tests/{test_view_count, test_estimators, test_change, test_metrics, test_legacy, test_state}.py.
- This folder: [attempts/](attempts/) 01-04; [logs/code.patch](logs/code.patch); [logs/code_review.md](logs/code_review.md)
  (+ full JSON); test logs; replay outputs logs/g1_*.json.
- Token caches for the replay (not copied, regenerable in ~40 s/scene): scratchpad tok_chess.pt, tok_office.pt (~100 MB each).
