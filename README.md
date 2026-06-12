# usplat — Uncertainty-Aware Open-Vocabulary Semantic 3DGS

A research scaffold for **directional (von Mises–Fisher) semantic uncertainty**
on a 3D Gaussian Splatting map, with a **disentangled epistemic / aleatoric**
signal that drives **active-inference exploration** and **uncertainty-driven
consolidation** for dynamic scenes.

It is the continuous generalization of CSS's BKI–Dirichlet semantic update:
swap the categorical one-hot label for a unit CLIP direction and accumulate the
kernel-weighted resultant. See the derivation notes accompanying this package.

---

## ⚠️ What is tested here vs. what needs your GPU

This package is split so the **research logic is fully runnable and unit-tested
on CPU**, while the heavy 3DGS rasterization is cleanly adaptered.

| Component | Status |
|---|---|
| vMF field + posterior + epistemic/aleatoric (`vmf_field.py`) | ✅ tested |
| Gradient-trick BKI accumulation (`bki_accumulate.py`) | ✅ **proven exact** vs brute force (`tests/test_accumulate.py`) |
| EFE planner (`efe_planner.py`) | ✅ runs |
| Consolidation / forgetting (`consolidation.py`) | ✅ runs |
| Calibration: sparsification / AUSE / ECE (`evaluation.py`) | ✅ runs |
| End-to-end mapping loop (`scripts/run_mapping.py --backend mock`) | ✅ runs on CPU |
| **gsplat feature rasterizer** (`gsplat_adapter.py`, channel-chunked) | ✅ **GPU-validated** (`tests/test_gsplat_gpu.py`: linearity exact, n_eff ≡ alpha mass, vMF invariants) |
| Dense CLIP extractor (`clip_dense.py` → `OpenCLIPDense`) | ✅ MaskCLIP-style value-projection trick on any open_clip ViT; shape-tested |
| PCA feature compression (`feature_pca.py`) | ✅ tested (queries map into the same compressed space) |
| 3DGS checkpoint loading (`gs_io.py`: INRIA .ply / gsplat .pt) | ✅ tested |
| Datasets (`datasets.py`: Replica / ScanNet / 7-Scenes) | ✅ Replica-format E2E-tested on a synthetic scene |
| Render-based localization w/ robust mask (`localization.py`) | ✅ GPU-validated (11 cm → 0.5 cm on synthetic) |
| Full real pipeline (`run_mapping.py --backend gsplat`) | ✅ runs E2E on GPU (synthetic scene; real scenes need a trained .ply + data) |

Run the tests / demo:

```bash
pip install -r requirements.txt
PYTHONPATH=. python tests/test_accumulate.py        # exactness of accumulation
PYTHONPATH=. python tests/test_pipeline_dryrun.py   # dynamic-region separation
PYTHONPATH=. python tests/test_gs_io.py tests/test_feature_pca.py
PYTHONPATH=. python tests/test_localization.py tests/test_clip_dense.py
PYTHONPATH=. python scripts/run_mapping.py --backend mock

# GPU (needs gsplat — see CUDA toolchain note below):
PYTHONPATH=. python tests/test_gsplat_gpu.py        # gradient trick on REAL gsplat
PYTHONPATH=. python scripts/make_synthetic_scene.py --out /tmp/synth_scene
PYTHONPATH=. python scripts/run_mapping.py --backend gsplat \
    --ply /tmp/synth_scene/scene.pt --data /tmp/synth_scene --dataset replica \
    --clip-pretrained none --feat-dim 16 --downscale 4 --out /tmp/field.pt

# Real scene (pre-trained 3DGS .ply + Replica/ScanNet/7-Scenes data):
PYTHONPATH=. python scripts/run_mapping.py --backend gsplat \
    --ply path/to/point_cloud.ply --data path/to/scene --dataset replica \
    --feat-dim 64 --stride 5 --query "a sofa,a television" --out outputs/field.pt
```

### CUDA toolchain (no system CUDA install needed)

gsplat JIT-compiles its CUDA extension and needs `nvcc`. Without a system
CUDA toolkit, run `scripts/setup_cuda_toolchain.sh` (fetches NVIDIA's
redistributable tarballs into `~/.local/cuda-12.8`, no sudo) and:

```bash
export CUDA_HOME=$HOME/.local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
```

The build is cached in `~/.cache/torch_extensions` after the first import.

---

## The one integration point you must wire

You need, per Gaussian, the kernel-weighted CLIP resultant `R_n = Σ_p w_{p,n} z_p`
and mass `n_eff_n = Σ_p w_{p,n}`. **Do not dump per-pixel contributor lists.**
Because alpha-compositing is linear in per-Gaussian features, both are exact
**gradients**:

```
F_p = Σ_n w_{p,n} f_n   ⇒   ∂/∂f_n [ Σ_p z_p·F_p ] = Σ_p w_{p,n} z_p = R_n
```

So with any differentiable feature rasterizer:

```python
field.update_from_rasterizer(rasterizer, z_obs)   # one render + one backward
```

where `rasterizer(buffer:(N, d+1)) -> image:(P, d+1)` renders a feature buffer
(geometry detached). `gsplat_adapter.GSplatFeatureRasterizer` implements this
against gsplat; `mock_rasterizer.MockFeatureRasterizer` is the CPU stand-in used
to prove the trick. **This identity is verified to zero error in the tests.**

### Bring-up on a real codebase (recommended order)

1. Fork a language-embedded 3DGS repo (Feature-3DGS / LangSplat / OpenGaussian).
2. Train (or load) the geometry as usual; attach a `(N, d+1)` feature buffer.
3. Replace `MockDenseCLIP` with your dense CLIP head (per-pixel or SAM-mask+CLIP).
4. Per keyframe: get the dense CLIP map `z_obs (P,d)`, build a
   `GSplatFeatureRasterizer` for that pose, call `update_from_rasterizer`.
5. Read `field.epistemic()`, `field.aleatoric()`, `field.mu_post()`,
   `field.query(text)`.

---

## First experiment (the figure that sells "uncertainty-aware")

On a few ScanNet scenes, render the posterior field to **held-out** views and
compute, per query-pixel, `error = 1 - cos(mu_post, GT-label text embedding)`.
Then run `evaluation.sparsification_curve` **separately** for each channel:

- **epistemic** should predict reconstruction-improvable regions (low coverage),
- **aleatoric** should predict dynamic / boundary / ambiguous regions.

Report AUSE per channel + ECE. The mock already reproduces the expected shape
(AUSE ≈ 0.03 for aleatoric→error, and dynamic > static aleatoric).

### Ablations that isolate each contribution
- vMF directional vs. CSS-style categorical (open-vocab vs. closed-set).
- scalar uncertainty vs. the 2-D `(n_eff, r̄)` split (the small-`n_eff` r̄ guard).
- EFE planner vs. VISTA-style `c·G_geom + G_sem` (does aleatoric-gating help?).
- consolidation on/off (localization error under scene change vs. SDD-SLAM).

---

## Watch-outs (already handled, keep handling)
- `r̄` is biased high at small `n_eff` (one obs ⇒ r̄=1). `aleatoric()` returns
  `NaN` below `min_obs`; use the credible interval for a paper.
- Fixed `kappa_lik` keeps the posterior closed-form; learning it breaks
  conjugacy (use `kappa_hat_banerjee` if you want a data-driven per-Gaussian κ).
- Thresholds in `Consolidator` are scene/noise dependent — calibrate on a val split.

## Layout
```
usplat/
  vmf_field.py        directional semantic field + uncertainty
  bki_accumulate.py   gradient-trick accumulation (+ brute-force reference)
  mock_rasterizer.py  pure-torch differentiable rasterizer (tests/CPU)
  gsplat_adapter.py   real gsplat feature rasterizer (GPU)
  clip_dense.py       dense CLIP interface + mock
  efe_planner.py      expected-free-energy next-best-view
  consolidation.py    forgetting/consolidation for dynamics
  evaluation.py       sparsification / AUSE / ECE
scripts/run_mapping.py
tests/{test_accumulate,test_pipeline_dryrun}.py
```
