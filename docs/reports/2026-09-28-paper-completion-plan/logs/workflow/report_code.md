# UAGS code audit (label: code)

Scratch folder: `S=/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/wf/code/`. It holds every script and log cited below. No project file was changed. All Python runs used `PYTHONDONTWRITEBYTECODE=1`, so the 13 git-tracked `.pyc` files were not touched. I ran one GPU test, which used about 1.5 s.

## Module table

| Module | What it implements | Test | Paper line ↔ code line | Status |
|---|---|---|---|---|
| usplat/vmf_field.py | R, n_eff, η, μ*, κ*, angular variance, r̄, r̃, κ̂, u_epi, u_ale (NaN below min_obs), query, optional text prior | Only indirect: test_pipeline_dryrun (mock), test_gsplat_gpu (μ*, r̄). r̃, κ̂ and the prior have no test. | η l.249-250 ↔ :59-63; κ*, variance l.254-256 ↔ :65-68; u_epi l.324 ↔ :82; r̃ l.332 ↔ :74-76; τ_min l.336 ↔ :84-87; κ̂ l.342 ↔ :78-80 | Formulas match. Their meaning does not (see mismatches 1, 2, 6). |
| usplat/bki_accumulate.py | Gradient trick with an appended ones-channel, plus a brute-force reference | test_accumulate (mock); test_gsplat_gpu (linearity, mass) | l.278-297 ↔ :23-46 | Matches. |
| usplat/mock_rasterizer.py | Dense P×N alpha compositing on CPU | used by the tests | not in paper | OK |
| usplat/gsplat_adapter.py | gsplat feature raster in 32-channel chunks, geometry detached | test_gsplat_gpu | l.290-295 ↔ :62-69 | Correct. Slow: calls the full `rasterization()` once per chunk (:64-69). |
| usplat/clip_dense.py | MaskCLIP value-path trick (keeps the residual and FFN, :106-107), plus a mock | test_clip_dense: shapes only, random ViT-B-32 | Implementation paragraph TODO l.415 | No quality check. |
| usplat/feature_pca.py | Uncentered streaming PCA; text is projected with the image basis | test_feature_pca: image-image cosine on synthetic data | not in paper | Text-image behaviour is untested (see risk 2). |
| usplat/consolidation.py | Decays R and n_eff when r̄ ≤ r_unstable; robust mask = n_eff ≥ min_obs and r̄ ≥ r_stable | none | l.357-369 ↔ :29-47 | See mismatch 4. |
| usplat/efe_planner.py | u_epi·discount·ν + λ·ν·max(0, cos) | none | l.378-384 ↔ :33-63 | See mismatch 5. |
| usplat/evaluation.py | Sparsification/AUSE (trapz over [0, 0.95]), 10-bin ECE, 1−cos error | none | l.408-413 (no formula in paper) | No script calls ECE. |
| usplat/localization.py | se3_exp, pose_errors, photometric L1 refine_pose on the robust subset | test_localization covers se3/pose_errors only | tab:loc l.456-480 | refine_pose has no test. The README claim "11 cm → 0.5 cm" has no script behind it. |
| usplat/gs_io.py | INRIA .ply and gsplat .pt loaders | test_gs_io **FAILS**: `plyfile` is not installed | none | .ply loses view-dependent colour: only the DC colour term is kept (:39-42). |
| usplat/datasets.py | Replica, ScanNet and 7-Scenes loaders (RGB, depth, c2w, K) | none | l.395-398 | No GT labels, no train/test splits. 7-Scenes uses DSLAM poses with fx=585 (:104-127). |
| scripts/run_mapping.py | Mock and gsplat mapping loops, PCA, consolidation, query demo | mock run in (d) | whole method | No per-pixel render, no eval, no timing. |
| scripts/make_synthetic_scene.py | Synthetic 3DGS scene plus Replica-format frames | not run (GPU) | none | OK |

## Paper-vs-code mismatches

The formulas are coded as the paper writes them. The problems are in what the formulas do.

1. **Small-sample shrinkage (high).** The paper motivates r̃ = r̄ − (1−r̄)/(n+1) (l.332, 334-336) with the case "a single observation gives r̄ = 1". But when r̄ = 1 the formula returns r̃ = 1, so it corrects nothing (vmf_field.py:74-76). `formula_checks.log` shows r̃ = [1.0, 1.0] at n_eff = 1.0 and 0.3. Only the τ_min gate does any work. The formula is not a standard estimator. The usual one for unit vectors is ρ̂² = (n r̄² − 1)/(n − 1), with Kish n = (Σk)²/Σk² when observations are weighted. This undercuts the "explicit small-sample correction" contribution (l.121-122).
2. **n_eff counts pixel mass, not views (high).** The paper says small n_eff means few observations and "additional viewpoints reduce it" (l.252, 320-321). In the code, n_eff is Σ over pixels, so it scales with resolution and with each Gaussian's screen footprint. In test_gsplat_gpu, one 160×120 view gives Σ n_eff = 5596.1 over 300 Gaussians, and 287 of them already have n_eff > 1. So τ_min = 5 is often passed after a single view. Pixels inside one 16-px CLIP patch are near-duplicates (bilinear upsampling, clip_dense.py:113-116), so the i.i.d. vMF likelihood over-counts evidence and κ* is overconfident.
3. **Consolidation uses raw r̄, not u_ale (low-medium).** consolidation.py:33 and :46 use r̄. The paper rule also uses r̄ (l.361-366), but the text says "the aleatoric signal identifies…" (l.359), and u_ale is defined through r̃. The stable set also needs n_eff ≥ min_obs (:36, :47), which the paper text omits.
4. **Decay only shrinks n_eff (medium).** Scaling both R and n_eff by λ leaves r̄ and μ* unchanged. In `formula_checks.log`, r̄ stays 0.200 and μ* moves by 4.5e-08. After 22 steps n_eff drops below 5, the Gaussian leaves the unstable set, u_ale becomes NaN and u_epi rises from 0.02 to 0.169. Default thresholds also disagree across files: r_unstable is 0.5 in consolidation.py:21, 0.55 in run_mapping.py:200, and the mock uses 0.55/0.8 (run_mapping.py:83).
5. **EFE (medium).** The paper term is Σ ν·u_epi·(1−u_ale) (l.379-381). The code adds two rules the paper does not state: a discount of 1 where u_ale is undefined, and a clamp of [0.05, 1] (efe_planner.py:40-43). Combined with mismatch 4, a forgotten dynamic Gaussian gets value 0.169 again, where the paper formula gives 0.0. The planner is therefore pulled back to the very regions l.384-386 says it avoids. No visibility model ν exists; the mock uses `torch.rand` (run_mapping.py:90). λ_p is still a hand-set weight, the same thing the paper criticises in VISTA (l.185).
6. **κ has no effect when κ0 = 0 (medium).** κ is listed as a hyperparameter (l.416). Changing it from 10 to 1000 moves μ* by at most 4.5e-08 and leaves r̄ exactly equal (`formula_checks.log`). So μ*, r̄, u_epi, u_ale, consolidation, EFE and mIoU all ignore κ. Also, μ* = R/‖R‖ is just the kernel-weighted mean CLIP feature. "Ours" mIoU therefore equals a deterministic averaging baseline.
7. **Text prior (low).** l.229-230 describe a text-embedding prior. The code supports it (vmf_field.py:28-33) but no script uses it.
8. **Novel-view semantic and uncertainty rendering "as in CSS" (l.263-265) is not implemented (high).** Tab:seg and fig:qual both need it.
9. **Identity check (low).** l.296-297 say the identity was "verified against an explicit weighted-sum reference". That is true only on the mock (test_accumulate.py). On gsplat the test checks only linearity and total mass (test_gsplat_gpu.py:53-75).
10. **AUSE (medium).** The paper gives no formula, and the error targets "reconstruction-improvable" and "dynamic/boundary" (l.424-426) have no code. evaluation.py:15-31 uses trapz over [0, 0.95]; usgloc/metrics.py:79-109 uses the mean gap. That is two definitions in one group. The unit (per-Gaussian or per-pixel) is not defined.
11. **ECE (medium-high).** The paper does not say what "confidence" means for a vMF posterior. evaluation.py:36 suggests "r_bar or softmax max", but r̄ is not a probability of being correct. No script calls ECE.
12. **Dense CLIP (medium).** Features are extracted from the *downscaled* frame (run_mapping.py:127, 141, 146; default downscale 4 → 160×120 → 10×8 patch grid). The default `--clip-pretrained openai` (:194) is not cached; only laion2b is.
13. **PCA-64 is not in the paper (high).** Text is projected through the image PCA basis (run_mapping.py:167). See risk 2 for the measured effect.
14. **Localization (medium).** The paper says "matching" (l.457-459). The code does photometric L1 refinement only, from a given initial pose, with DC colour (localization.py:60-106).
15. **Datasets (medium).** The paper lists 7-Scenes, ScanNet and dynamic data. The loaders have no labels and no splits. 7-Scenes uses DSLAM poses with fx = 585, while the usgloc maps use SfM pGT with f = 525–528.

## Missing pieces per deliverable (person-days, pd)

| Deliverable | Exists | Missing | pd |
|---|---|---|---|
| tab:seg | vMF field, gradient trick, MaskCLIP, gsplat adapter | Semantic-GT dataset and loader (ScanNet or Replica-SemanticNeRF; download size not checked) 1.5. 3DGS training on it (adapt usgloc fit_gaussians) 1.5 + GPU time (not measured). Dense CLIP: full-res extraction 0.25, FastSAM-mask option with per-pixel validity weights 1, val comparison 1. Text/PCA fix 0.75. Per-pixel render of μ*/u_epi/u_ale and mIoU/Acc harness with prompt protocol and ignore labels 2. ECE/AUSE definitions, code and tests 1.25. CSS-Dirichlet adapter (one-hot from CLIP argmax, same kernel) 1.5. Feature3DGS-style optimised deterministic field in gsplat 1.5 (official Feature 3DGS: 3–5) | ~12 |
| tab:loc | refine_pose, robust mask, pose_errors | Dynamic dataset: none on disk. A public rearrangement set such as RIO10/3RScan (not checked) or our own capture: 3–10. Multi-session protocol and scene-change handling 3 (geometry is fixed, run_mapping.py:115; moved objects have no Gaussians at their new place, so background gets flagged). Coarse init via retrieval or ACE 1.5. Real-data eval (median ATE/ARE, SR 5cm/5°, 2cm/2°, SH colour, failure count) 1.5. ACE and Marepo reruns 2–3. Chu & Kim: code not checked, reimplementation 5+ | ~11–19, plus Chu & Kim |
| tab:abl | `--consolidate-every 0`, EFE class | Categorical row (reuse CSS adapter) 0.5. Scalar-uncertainty variant and AUSE_ale target 0.5. w/o consolidation 0.25 (needs the loc pipeline). EFE vs heuristic view score: pool-based next-best-view using a ones-channel render as ν, a VISTA-style baseline, and a fix for the NaN discount 3–4 (a simulator would be 8–10) | ~4.5–5.5 |
| fig:qual | nothing | RGB, μ* query map, u_epi and u_ale renders on dynamic data | 1–1.5 after the pipelines |
| Implementation paragraph | defaults at run_mapping.py:193-205 | Timing and GPU-memory instrumentation, hyperparameter table, note that κ has no effect | 0.5–1 |
| Hygiene | 6 test files | Tests for consolidation, EFE, evaluation, datasets and refine_pose; install `plyfile`; untrack `__pycache__` | 1 |
| **Total** | | | **~30–45 pd** |

## Reusable usgloc parts (read-only)

| Part | Path | Use in UAGS | Adapter | Concern |
|---|---|---|---|---|
| Trained 3DGS maps, 55 GB | usgloc/maps/*.pt | Map geometry for 7-/12-Scenes. Sizes: heads 246,331; chess 688,973; apt1_kitchen 1,039,668; office1_gates381 1,099,705 Gaussians (mmap load, see command below) | Maps store `covs`, not quats/scales. Use `_cov_to_quat_scale` (rendering.py:148-173) or gsplat 1.5.3's `covars` argument. 0.5 pd | Maps are in the SfM-pGT frame with per-scene focal. The UAGS 7-Scenes loader (DSLAM, fx=585) must not be used with them. Pick one variant (_ft/_r1/_sel). A descriptor field is map-specific. |
| Map training | usgloc/mapping.py:74, :164; scripts/train_map.py | Base 3DGS on new data (ScanNet, dynamic set) | New dataset class with the SevenScenes interface | GPU time on a shared card |
| Loaders | usgloc/datasets.py:127 (SevenScenes, splits + SfM pGT), :297 (TwelveScenes) | Replace usplat/datasets.py for 7-/12-Scenes | ~0.5 pd | Both datasets are static; no scene change. |
| Mask-CLIP | mapping.py:560-623 MaskClipEncoder (FastSAM + ViT-B-16 laion2b), :44-51 vocabulary, :617 prompts | Alternative z_obs source: paint mask features into pixels, add a validity mask | 1 pd | `distill_clip_features` (:625) assigns by centre projection, not the T·α kernel. Use it only as a baseline. |
| Metrics | usgloc/metrics.py pose_errors, success_rate, ause :79, coverage_ece :70 | tab:loc metrics | none | The AUSE definition differs from uags evaluation.py. Pick one. |
| Renderer | rendering.py:135 GsplatRenderer (extra channels) | Per-pixel μ*/uncertainty rendering | small | none |
| Weighted refinement | refinement.py (per-pixel exp(−γU)) | none | none | This is the same idea as the UAGS robust subset. Overlap. |
| ACE/Marepo | scripts/analysis/run_marepo_baseline.py, make_ace_dataset.py, results/ace_baseline/ | tab:loc baselines | per dataset | The code and models live in **another session's /tmp scratchpad** (run_marepo_baseline.py:29-48) and could be deleted. |
| Calibration | calibration.py (pose-covariance temperature) | not relevant (no pose covariance in UAGS) | none | none |

**Overlap concern (high).** The two papers share the group, venue, datasets, maps, CLIP backbone and baselines. The USGLoc draft already claims that "rendering-based refinement discounts moved and ambiguous content" and that "no stage requires dynamic-object heuristics or a language model" (usgloc_ral_draft.tex abstract and l.88). The UAGS abstract (l.66-68) says nearly the same. USGLoc's limitation 1 (l.471) names "a temporal conflict model" as future work, and that is what UAGS consolidation is. UAGS needs to cite USGLoc, state the difference clearly, and not report the same 7-Scenes localization numbers.

## Test outputs

Before the GPU test, `nvidia-smi` showed 9254 of 16303 MiB used, so about 7 GB was free.

`export CUDA_HOME=$HOME/.local/cuda-12.8; export PATH=$CUDA_HOME/bin:$PATH; PYTHONPATH=. timeout 300 python3 tests/test_gsplat_gpu.py` → `S/test_gsplat_gpu.log`:
```
PASS linearity: max |render(a*b1+b2) - a*render(b1)-render(b2)| = 1.91e-06
PASS mass: sum n_eff = 5596.1 vs alpha mass 5596.1 (rel gap 0.00e+00)
PASS constant-direction: min cos(mu_post, z) = 1.00000, min r_bar = 1.00000 on 287 observed Gaussians
PASS conflict: mean aleatoric under flipping semantics = 1.000 (> 0.5)
ALL GPU CHECKS PASSED   (real 1.46 s, exit 0)
```
`PYTHONPATH=. timeout 300 python3 scripts/run_mapping.py --backend mock` → `S/run_mapping_mock.log`:
```
gaussians: 200 | mean n_eff: 91.4
mean epistemic: 0.327 | mean aleatoric (valid): 0.336
aleatoric  dynamic=0.460  static=0.289   (dynamic should be higher)
robust(stable) Gaussians: 17 | flagged unstable: 28
EFE per candidate view: [-54.218, -49.143, -51.882, -56.131, -55.725] | best=v3
AUSE(aleatoric -> semantic error): 0.0275  (lower=better)
done.   (exit 0)
```
The CPU tests (`S/cpu_tests.log`) give the same 5/6 as before. test_gs_io exits 1 with `ModuleNotFoundError: No module named 'plyfile'`. The others pass: accumulate R diff 0.00e+00 and n_eff diff 9.54e-07; dryrun dynamic 0.353 > static 0.225; PCA 0.999; se3; clip shapes. Caveat: the mock prototypes are random 32-d vectors, nearly orthogonal. Real CLIP features are not (next item).

**Extra CPU check on real CLIP features (not tuned):** `S/clip_cone_check.py` and `S/clip_static_vs_moved.py`, OpenCLIPDense ViT-B-16 laion2b on 7-Scenes chess, office and redkitchen, full 640×480:
- Two random pixels from *different scenes* have cosine p5/p50/p95 = 0.478 / 0.646 / 0.798.
- A 50/50 mix of two different contents gives r̄ = 0.86 / 0.907 / 0.948. 0.0% of mixes reach r̄ ≤ 0.55 and 97.5% are ≥ 0.85.
- Pooling every pixel of one frame still gives r̄ ≈ 0.80.
- Static (same pixel, frame i vs i+3) vs changed content: r̄ p50 0.94 vs 0.912. AUROC is 0.725 in raw 512-d and 0.745 in PCA-64.
- PCA 512→64: image features keep 0.897 of their energy, text embeddings keep only **0.109** (min 0.077). Per-pixel zero-shot argmax agrees with 512-d on only 0.707 of pixels.
- Labels from full-res extraction vs the default downscale-4 extraction agree on only 0.55–0.71 of pixels.
- CPU extraction time: 1.81 s per frame at 640×480, 0.12 s at 160×120.

## Scalability

The memory model comes from the code (bki_accumulate.py:36-45, vmf_field.py:26-27, 55-56):
- persistent: N(d+1)·4 B
- transient per update: 3·N(d+1)·4 (buffer, grad, R clone) + 2·P(d+1)·4 (image and its grad) + z_obs + the 512-d CLIP map.

A CPU proxy (`S/scale_cpu.py`: sparse P×N matrix standing in for the rasterizer, real uags update path, K=16 contributors per pixel; `S/scale_cpu.log`) matches the model:

| Config (P = 640×480 unless noted) | Measured transient | Model | CPU time/frame |
|---|---|---|---|
| N=1M, d=64 | 0.97 GB | 0.94 GB | 0.83 s |
| N=1M, d=64, 160×120 | 0.75 GB | – | 0.23 s |
| N=250k, d=512 | 2.74 GB | 2.8 GB | 2.97 s |

Extrapolated GPU memory for N=1M at 640×480:

| | d=64 (PCA) | d=512 |
|---|---|---|
| Persistent field | 0.26 GB | 2.05 GB |
| Transient | ~0.94 GB | ~7.4 GB |
| + CLIP map, z, fp32 ViT-B-16 | ~1.9 GB | ~1.9 GB |
| **Total** (gsplat intersection buffers not measured) | **~2.5–3 GB** | **~11 GB** |

- d=512 does not fit the roughly 3.5–7 GB free on the shared card.
- Saved fields (R and μ* stored, run_mapping.py:174-176) take about 4.1 GB per scene at d=512 and about 0.5 GB at d=64. That matters with 48 GB of disk free.
- Raster passes per frame are ⌈(d+1)/32⌉: 3 for d=64, 17 for d=512. Each pass redoes projection and tile sorting (gsplat_adapter.py:64-69). gsplat 1.5.3's `rasterization` has its own `channel_chunk` and `covars` arguments; passing all channels in one call shares the projection.
- GPU time was not measured, because only one GPU test was allowed.
- The memory cuts are: drop the `.clone()`, accumulate R in place, and run the backward pass per channel chunk.

Command for the Gaussian counts: `python3 -c "torch.load('maps/<m>.pt', mmap=True, weights_only=True)"` in usgloc/.

## Top 5 code risks

1. **The consolidation thresholds cannot fire on real CLIP features.** Measured r̄ for changed content sits at about 0.86–0.95, so r_unstable = 0.55 is never reached and changed content looks "stable". Static vs changed separates only at AUROC about 0.73. The mock hides this because its prototypes are nearly orthogonal. Fix: center or whiten features before normalizing, and calibrate thresholds on a validation split.
2. **PCA-64 breaks open-vocabulary queries.** Text keeps about 11% of its energy in the image PCA basis, and 29% of per-pixel labels change compared with 512-d. At d=512 the memory does not fit the shared GPU. Fix: fit the basis on image and text together, center it, or query in 512-d.
3. **Dense CLIP runs at 160×120 by default (10×8 patches).** Labels agree with full-res extraction on only 55–71% of pixels. The default openai weights are not cached, and fetching them breaks the 50 MB download rule.
4. **The uncertainty meaning is fragile.** n_eff counts correlated pixels, so it depends on resolution and footprint. The shrinkage is a no-op at r̄ = 1. κ has no effect. Decay followed by the NaN discount sends the EFE planner back to forgotten dynamic Gaussians. The mIoU of "Ours" equals naive averaging.
5. **There is no real-data evaluation path.** Missing: labels, per-pixel renders, mIoU/ECE harness, baselines, pose init, dynamic dataset and a scene-change protocol. The only localization number (in the README) has no script. On top of this, the method and 7-Scenes setup overlap heavily with USGLoc.