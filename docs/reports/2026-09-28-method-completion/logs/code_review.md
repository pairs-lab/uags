# Adversarial code review (workflow wf_fde25f68-a03): 40 confirmed, 2 rejected

4 finders (math, numerics, pipeline, tests) + 1 skeptic per finding; the skeptic had to reproduce it.

| # | lens | verified severity | file:line | finding | status |
|---|---|---|---|---|---|
| 0 | math | major | usplat/vmf_field.py:87 | With the default config the change test never fires on a clean content swap, and every detection test overrides the default | FIXED: change_thr default inf + calibrate flow in run_mapping (map -> calibrate -> --update-data); tests: test_default_config_detects_after_calibration, dryrun on defaults |
| 1 | math | minor | usplat/vmf_field.py:392 | The normalized change score's null E0[1-cos(S,F)] leaves out the shared-view term, so it is not scale-free and its bias depends on how many views a Gaussian has | FIXED: shared-view term added to the null; test tightened to [0.85,1.15] over beta 0.5/0.8/0.9, 10/30 views |
| 2 | math | minor | usplat/vmf_field.py:470 | The default epistemic score 1/(1+Kish) ignores view weight, so a Gaussian only grazed in 5 views gets the same u_epi as one fully seen in 5 views | FIXED: epistemic default = 1/(1+support) |
| 3 | math | minor | usplat/vmf_field.py:52 | The documented legacy switches do not reproduce the field at git HEAD (epistemic, kappa_post, mu_post with a text prior, aleatoric gating, Consolidator defaults) | FIXED: FieldConfig.legacy() + tests/test_legacy.py (matches git ab28fb4) |
| 4 | math | major | usplat/vmf_field.py:422 | evidence='pixels' does not ablate F1: gating still counts views through the F2 support, and view-unit constants turn into pixel units | FIXED: pixel mode gates on W1; change/re-admission bookkeeping always in view units |
| 5 | math | minor | usplat/vmf_field.py:19 | The Kish agreement is documented as an unbiased estimate of ||E u|| that does not depend on n; only rho_hat^2 is unbiased | FIXED: docstring: unbiased for rho^2 |
| 6 | math | minor | usplat/predictive.py:40 | P2 treats an undefined aleatoric value (NaN) as 0, the most confident value, including for Gaussians just reset by the change test | FIXED: P2 gets an undefined-indicator term g; test_p2_recovers_slopes |
| 7 | numerics | minor | usplat/evaluation.py:66 | AUROC/AP/AUSE silently drop NaN scores, positives included, so a method is graded on the subset it chose to score (AP = 1.0 with 3 of 4 positives unscored) | FIXED: undefined='last' policy; test_undefined_policy |
| 8 | numerics | minor | tests/test_view_count.py:27 | No test protects F1: the whole CPU suite still passes with view evidence replaced by pixel mass | FIXED: W1 = k asserted; partial view test |
| 9 | numerics | minor | usplat/vmf_field.py:471 | Default epistemic 1/(1+n_K) ignores view weight: 10 grazing views (w=1e-3) score the same as 10 full views, and it jumps from 1.0 to 0.5 at w=1e-6 | FIXED: same as 2 |
| 10 | numerics | minor | usplat/vmf_field.py:359 | Without depth input, a changed Gaussian is never re-admitted under the default readmit_depth=0.05 (the default run_mapping path) | FIXED: views-only fallback when no depth stream (warned); fail closed once a depth stream exists |
| 11 | numerics | minor | usplat/vmf_field.py:239 | One NaN or inf depth pixel permanently poisons depth_abs for every Gaussian covering it (NaN*0 in the masked target); re-admission is then blocked forever | FIXED: torch.where masking in update_view and bki groups; NaN/inf test |
| 12 | numerics | minor | usplat/vmf_field.py:305 | centre=False silently turns off the change test; an unset zbar silently makes the 'centred' channel the raw channel (dry-run, mock and GPU tests never centre) | FIXED: change test runs on the raw channel when centre=False; warning when zbar unset |
| 13 | numerics | minor | usplat/vmf_field.py:383 | Normalized change score is wrong for slow_decay<1: the slow-bucket rho^2 collapses and n_S drops below 1, so E0 goes to 1 and normalization switches itself off | FIXED: validate() raises for normalized + slow_decay<1 |
| 14 | numerics | minor | usplat/vmf_field.py:186 | Memory at N=1e6, d=64: 4 persistent (N,d) tensors (1.02 GB, one of them redundant) plus about 1.8 GB transient per update | OPEN: not optimised; plan budget must count 4 (N,d) tensors (+1.8 GB transient at N=1e6, d=64) |
| 15 | numerics | minor | usplat/vmf_field.py:532 | from_state drops the text prior (prior_mu), so a reloaded field gives different mu_post | FIXED: prior_mu + dtype saved; test_state.py |
| 16 | numerics | minor | usplat/vmf_field.py:241 | update_view works under torch.no_grad but fails under torch.inference_mode with an unclear autograd error | FIXED: clear RuntimeError; test |
| 17 | pipeline | major | scripts/gpu_gate.sh:23 | gpu_gate.sh guards only a one-time free-memory reading: no pgrep check, no MASt3R headroom and no per-process memory cap, all of which plan K1 requires | FIXED: gate: USGLoc-aware headroom, re-check under lock, memory fraction export |
| 18 | pipeline | major | usplat/vmf_field.py:333 | The `changed` flag never clears when no depth is fed, which is the default CLI run and the mock backend: false alarms are permanent and the planner keeps changed Gaussians gated forever | FIXED: same as 10 |
| 19 | pipeline | minor | usplat/vmf_field.py:460 | state_dict/from_state loses the text prior (prior_mu) and the dtype, so a reloaded field gives different mu_post and kappa_post | FIXED: same as 15 |
| 20 | pipeline | minor | usplat/efe_planner.py:85 | EFEPlanner still gates aleatoric at 5.0, but that number now counts views, not pixels: Gaussians with 3-4 views and high aleatoric are not discounted | FIXED: EFE uses cfg.tau_min |
| 21 | pipeline | minor | scripts/gpu_gate.sh:22 | gpu_gate.sh: one transient nvidia-smi failure ends a waiting gate under set -euo pipefail; the job never runs and its exit code looks like a job failure | FIXED: robust numeric parsing; no set -e |
| 22 | pipeline | minor | scripts/run_mapping.py:214 | --backend mock ignores --field-config and all other CLI options | FIXED: mock honours --field-config |
| 23 | pipeline | minor | usplat/feature_pca.py:69 | The saved output mixes devices: the PCA basis is stored as a CUDA tensor while the field state is on CPU | FIXED: basis saved on CPU |
| 24 | pipeline | minor | scripts/run_mapping.py:160 | --use-depth with --downscale 1 crashes on datasets whose depth size differs from the colour size (ScanNet 480x640 depth vs 968x1296 colour) | FIXED: depth resized to the colour grid |
| 25 | pipeline | minor | tests/test_pipeline_dryrun.py:13 | The dry-run test docstring says 'most observably changed Gaussians are flagged', but recall is 0.50 and the test never checks recall | FIXED: dryrun docstring + recall assert on the calibrated default flow |
| 26 | tests | minor | usplat/vmf_field.py:87 | The change test never fires at the defaults (fast_memory=0.9, change_thr=0.5), and every test pins fast_memory=0.5, so the default is untested | FIXED: same as 0 |
| 27 | tests | minor | tests/test_view_count.py:27 | The F1 view-count tests pass when view counting is removed | FIXED: same as 8 |
| 28 | tests | minor | usplat/vmf_field.py:231 | F3 centring on the real update path (update_view) is never tested: every update_view test runs with zbar=None | FIXED: test_centred_update_view_path |
| 29 | tests | minor | usplat/vmf_field.py:159 | The Kish estimator is only tested with one pixel per view and unit weights, so Q==W2==W1 and the key terms are unchecked | FIXED: test_kish_multi_pixel_fractional (Monte-Carlo) |
| 30 | tests | minor | tests/test_change.py:88 | F5 re-admission is barely tested: the readmit_views gate and the depth reset on fire can be deleted, and the no-depth test passes even when nothing fires | FIXED: test_readmit_timing_and_partition; n_resets asserted |
| 31 | tests | minor | tests/test_pipeline_dryrun.py:13 | The pipeline test claims 'most changed Gaussians are flagged' but never asserts recall; measured recall is 0.50 | FIXED: same as 25 |
| 32 | tests | minor | tests/test_gsplat_gpu.py:123 | The GPU change check measures one yes/no event that depends on the random z1, and pytest never runs the GPU file | FIXED: orthogonal z1; pytest entry with skip |
| 33 | tests | minor | tests/test_metrics.py:77 | The F6 predictives are only loosely tested: P1 passes with the wrong concentration, and the P2 slopes are fitted on all-zero inputs | FIXED: P1 vs A_d(kappa_hat ||R||), |E mu|->1; P2 slopes recovered |
| 34 | tests | minor | usplat/vmf_field.py:533 | state_dict/from_state drops the text prior: a restored field with prior_kappa>0 gives different posteriors and query labels | FIXED: same as 15 |
| 35 | tests | minor | usplat/vmf_field.py:359 | Default readmit_depth=0.05 without depth input makes a reset permanent; no test covers it | FIXED: same as 10 |
| 36 | tests | minor | usplat/feature_pca.py:66 | F7 (text projected without renormalisation) has no test; the renormalising mutation passes and flips the query label | FIXED: test_query_decode_first |
| 37 | tests | minor | usplat/evaluation.py:131 | AURC, NLL and Brier are effectively untested (the only check is aurc(0,1)==1) | FIXED: test_calibration_metrics_bruteforce |
| 38 | tests | minor | tests/test_estimators.py:87 | The A_d round-trip and monotonicity checks cannot fail, so the continued-fraction branch for d>=64 is unanchored | FIXED: test_a_d_against_mpmath |
| 39 | tests | minor | tests/test_change.py:143 | test_normalized_score_is_scale_free feeds a malformed ViewStats (R is not m*u), which cancels the 'varying weight' | FIXED: ViewStats fixed in the normalized test |

Rejected by the skeptics:
- math: With fast_memory=0.5 (used by all detection tests), the raw change score flags static content whose centred observations are dispersed
- numerics: Change score's null depends on per-Gaussian dispersion: with the per-scene calibrated threshold, false alarms fall only on ambiguous static Gaussians (20% vs 0%)
