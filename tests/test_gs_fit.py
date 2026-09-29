"""gs_fit (plan K9): CPU unit tests of the fitter pieces + gated GPU fits.

CPU: hold-out split, downscale factor, SSIM against scikit-image, the loss weights, the
learning rates and the means-lr decay, depth back-projection (round trip through the gsplat
pixel convention, K_depth), voxel averaging, frustum sampling (inside the frusta, colour
from the pixel each point came from, every frame used), the Gaussian cap of
CappedDefaultStrategy, the opacity reset (runs inside the refine window; off = gsplat 1.5.3),
pre-activation storage of the exported / saved parameters, the .pt round trip through
usplat.gs_io, and the CLI (boolean flags, git dirty flag with untracked files, code hashes).
The whole fit loop also runs on the CPU with a stand-in renderer (_FakeRender): it checks
that only training frames are rendered with gradients, that densification and the opacity
reset are wired into the loop, float64 K, and non-finite poses (error in fit_gaussians,
dropped and listed by fit_dataset).

GPU (skipped without CUDA, and skipped unless run under scripts/gpu_gate.sh, which exports
UAGS_GPU_MEM_FRACTION, or with UAGS_GPU_TESTS=1): fit the synthetic scene of
scripts/make_synthetic_scene.py (plus depth PNGs rendered from its scene.pt, so the depth
init runs) through the fit_gs.py CLI for 400 iterations; check the held-out PSNR (absolute
and gain), re-render the raw checkpoint with the documented activations and through gs_io
and compare with the reported PSNR, and check the memory fields; a second, random-init fit
checks the Gaussian cap and the opacity reset on the GPU.

  scripts/gpu_gate.sh 2000 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
      python3 -m pytest -q -p no:cacheprovider tests/test_gs_fit.py
"""
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pytest
import torch

from usplat import gs_fit
from usplat.gs_fit import (FitConfig, split_holdout, downscale_factor, ssim,
                           backproject_depth, voxel_downsample, random_frustum_points,
                           save_gaussians_pt, photometric_loss)
from usplat.gs_io import load_gaussians_pt, SH_C0

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _gpu_ok():
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    if not (os.environ.get("UAGS_GPU_MEM_FRACTION") or os.environ.get("UAGS_GPU_TESTS")):
        return False, "GPU is shared: run under scripts/gpu_gate.sh (or UAGS_GPU_TESTS=1)"
    try:
        from gsplat.cuda._backend import _C
        assert _C is not None
    except Exception:
        return False, "gsplat CUDA backend not built"
    return True, ""


# synthetic CLI fit, 400 iterations, held-out PSNR: 36.1 dB before the opacity-reset fix,
# 37.2 dB after it (N=1 each). The floor catches a fit that stalls.
PSNR_MIN_SYNTHETIC = 30.0

_ok, _why = _gpu_ok()
gpu = pytest.mark.skipif(not _ok, reason=_why)
needs_gsplat = pytest.mark.skipif(not gs_fit._HAS_GSPLAT, reason="gsplat not installed")


def _look_at(eye, target=(0.0, 0.0, 0.0)):
    """OpenCV c2w looking from eye to target (y down)."""
    eye = np.asarray(eye, float)
    fwd = np.asarray(target, float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross([0.0, 1.0, 0.0], fwd)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, eye
    return torch.from_numpy(c2w).float()


# ----------------------------------------------------------------------------- CPU
def test_split_holdout_every_8th():
    tr, te = split_holdout(20, 8)
    assert te == [0, 8, 16]
    assert sorted(tr + te) == list(range(20)) and not set(tr) & set(te)
    assert split_holdout(5, 0) == ([0, 1, 2, 3, 4], [])
    assert split_holdout(1, 8) == ([0], [])


def test_downscale_factor_caps_long_side():
    assert downscale_factor(480, 640, 640) == 1
    assert downscale_factor(680, 1200, 640) == 2          # 600 x 340
    assert downscale_factor(968, 1296, 640) == 3          # 432 x 322
    assert downscale_factor(680, 1200, 0) == 1
    for h, w, m in [(680, 1200, 300), (540, 960, 640), (1080, 1920, 512)]:
        assert max(h, w) // downscale_factor(h, w, m) <= m


def test_ssim_matches_skimage():
    skm = pytest.importorskip("skimage.metrics")
    g = torch.Generator().manual_seed(0)
    x = torch.rand(3, 40, 50, generator=g, dtype=torch.float64)
    y = (x + 0.1 * torch.randn(3, 40, 50, generator=g, dtype=torch.float64)).clamp(0, 1)
    ours = float(ssim(x[None], y[None]))
    ref = skm.structural_similarity(x.permute(1, 2, 0).numpy(), y.permute(1, 2, 0).numpy(),
                                    channel_axis=2, data_range=1.0, gaussian_weights=True,
                                    sigma=1.5, use_sample_covariance=False)
    assert abs(ours - ref) < 1e-6, (ours, ref)
    assert abs(float(ssim(x[None], x[None])) - 1.0) < 1e-9


def test_backproject_depth_round_trip():
    H, W = 30, 40
    K = torch.tensor([[50.0, 0, 19.5], [0, 55.0, 14.5], [0, 0, 1]])
    c2w = _look_at([0.3, -0.2, -2.0])
    g = torch.Generator().manual_seed(1)
    depth = 1.0 + torch.rand(H, W, generator=g)
    depth[0, 0] = 0.0                                          # invalid pixel is dropped
    img = torch.randint(0, 256, (H, W, 3), generator=g, dtype=torch.uint8)
    fr = dict(image=img, depth=depth, c2w=c2w, K=K)
    pts, rgb = backproject_depth(fr, pixel_step=1, depth_min=0.05, depth_max=10.0)
    assert pts.shape == (H * W - 1, 3) and rgb.shape == (H * W - 1, 3)
    # project with the gsplat convention: pixel (i, j) covers [j, j+1) x [i, i+1)
    w2c = torch.linalg.inv(c2w.double())
    pc = pts.double() @ w2c[:3, :3].T + w2c[:3, 3]
    u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
    v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
    ii, jj = torch.floor(v).long(), torch.floor(u).long()
    vv, uu = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    keep = depth.reshape(-1) > 0
    assert torch.equal(ii, vv.reshape(-1)[keep]) and torch.equal(jj, uu.reshape(-1)[keep])
    assert torch.allclose(pc[:, 2].float(), depth.reshape(-1)[keep], atol=1e-5)
    assert torch.allclose(rgb, img.reshape(-1, 3)[keep].float() / 255)


def test_voxel_downsample_averages_per_voxel():
    p = torch.tensor([[0.01, 0.01, 0.01], [0.03, 0.03, 0.03], [0.51, 0.0, 0.0]])
    c = torch.tensor([[1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]])
    q, d = voxel_downsample(p, c, 0.1)
    assert q.shape == (2, 3)
    i = int(torch.argmin(q[:, 0]))
    assert torch.allclose(q[i], torch.tensor([0.02, 0.02, 0.02]))
    assert torch.allclose(d[i], torch.tensor([0.5, 0.5, 0.0]))
    q0, _ = voxel_downsample(p, c, 0.0)
    assert torch.equal(q0, p)


def test_random_frustum_points_inside_frusta():
    H, W = 20, 30
    K = torch.tensor([[25.0, 0, 15.0], [0, 25.0, 10.0], [0, 0, 1]])
    frames = [dict(image=torch.full((H, W, 3), 10 * k, dtype=torch.uint8), depth=None,
                   c2w=_look_at([2.0 * math.sin(a), 0.0, -2.0 * math.cos(a)]), K=K)
              for k, a in enumerate([-0.3, 0.0, 0.3])]
    pts, cols = random_frustum_points(frames, 3000, 0.5, 3.0, torch.Generator().manual_seed(0))
    inside = torch.zeros(len(pts), dtype=torch.bool)
    for fr in frames:
        w2c = torch.linalg.inv(fr["c2w"].double())
        pc = pts.double() @ w2c[:3, :3].T + w2c[:3, 3]
        u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
        v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
        inside |= ((pc[:, 2] >= 0.5 - 1e-5) & (pc[:, 2] <= 3.0 + 1e-5)
                   & (u >= -1e-4) & (u <= W + 1e-4) & (v >= -1e-4) & (v <= H + 1e-4))
    assert bool(inside.all())
    assert set(torch.unique(cols[:, 0] * 255).round().tolist()) <= {0.0, 10.0, 20.0}


def test_random_frustum_points_colour_from_its_pixel():
    """Each point's colour is the pixel it was sampled through, in the frame it came from
    (R = column, G = row, B = 100 + frame id), and every frame contributes points."""
    H, W = 20, 30
    K = torch.tensor([[25.0, 0, 15.0], [0, 25.0, 10.0], [0, 0, 1]])
    frames = []
    for k, a in enumerate([-0.3, 0.0, 0.3]):
        img = torch.zeros(H, W, 3, dtype=torch.uint8)
        img[..., 0] = torch.arange(W, dtype=torch.uint8)[None, :]
        img[..., 1] = torch.arange(H, dtype=torch.uint8)[:, None]
        img[..., 2] = 100 + k
        frames.append(dict(image=img, depth=None, K=K,
                           c2w=_look_at([2.0 * math.sin(a), 0.1 * k, -2.0 * math.cos(a)])))
    n = 3000
    pts, cols = random_frustum_points(frames, n, 0.5, 3.0, torch.Generator().manual_seed(0))
    c = (cols * 255).round().long()
    fid = c[:, 2] - 100
    counts = torch.bincount(fid, minlength=3)
    assert len(counts) == 3 and bool((counts > 0.8 * n / 3).all()), counts
    for k, fr in enumerate(frames):
        sel = fid == k
        w2c = torch.linalg.inv(fr["c2w"].double())
        pc = pts[sel].double() @ w2c[:3, :3].T + w2c[:3, 3]
        u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
        v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
        col, row = c[sel, 0].double(), c[sel, 1].double()
        assert bool(((u > col - 1e-3) & (u < col + 1 + 1e-3)).all())
        assert bool(((v > row - 1e-3) & (v < row + 1 + 1e-3)).all())


def test_photometric_loss_is_08_l1_plus_02_dssim():
    g = torch.Generator().manual_seed(0)
    gt = torch.rand(24, 30, 3, generator=g)
    img = (gt + 0.2 * torch.randn(24, 30, 3, generator=g)).clamp(0, 1)
    l1 = float((img - gt).abs().mean())
    s = float(ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
    assert abs(l1 - (1 - s)) > 0.01          # swapped weights would move the loss by > 6e-3
    assert FitConfig().ssim_lambda == 0.2
    assert abs(float(photometric_loss(img, gt, 0.2)) - (0.8 * l1 + 0.2 * (1 - s))) < 1e-6
    assert float(photometric_loss(gt, gt, 0.2)) < 1e-6


def test_optimizer_lrs_and_means_decay():
    N = 8
    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.zeros(N, 3)),
        "scales": torch.nn.Parameter(torch.zeros(N, 3)),
        "quats": torch.nn.Parameter(torch.ones(N, 4)),
        "opacities": torch.nn.Parameter(torch.zeros(N)),
        "sh0": torch.nn.Parameter(torch.zeros(N, 1, 3))})
    opts = gs_fit.make_optimizers(splats, FitConfig(), 2.5)
    lr = {k: o.param_groups[0]["lr"] for k, o in opts.items()}
    want = dict(means=1.6e-4 * 2.5, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3)
    assert lr == pytest.approx(want, rel=1e-12)
    assert all(o.defaults["eps"] == 1e-15 for o in opts.values())
    sch = gs_fit.make_means_scheduler(opts["means"], 200)
    for _ in range(200):
        splats["means"].grad = torch.zeros_like(splats["means"])
        opts["means"].step()
        sch.step()
    assert opts["means"].param_groups[0]["lr"] == pytest.approx(0.01 * want["means"], rel=1e-6)


def test_resolve_schedule_scales_with_iters():
    assert gs_fit.resolve_schedule(FitConfig()) == (500, 15_000, 3000)       # gsplat values
    assert gs_fit.resolve_schedule(FitConfig(iters=3000)) == (500, 1500, 300)
    assert gs_fit.resolve_schedule(FitConfig(iters=400)) == (66, 200, 100)
    assert gs_fit.resolve_schedule(FitConfig(iters=3000, reset_every=1000))[2] == 1000
    assert gs_fit.resolve_schedule(FitConfig(iters=1))[2] == 100             # never 0


def test_backproject_uses_K_depth_when_present():
    H, W = 12, 16
    Kc = torch.tensor([[20.0, 0, 8.0], [0, 20.0, 6.0], [0, 0, 1]])
    Kd = torch.tensor([[23.0, 0, 7.5], [0, 22.0, 6.5], [0, 0, 1]])
    c2w = _look_at([0.1, 0.2, -1.5])
    depth = torch.full((H, W), 1.3)
    img = torch.zeros(H, W, 3, dtype=torch.uint8)
    img[..., 0] = torch.arange(W, dtype=torch.uint8)[None, :]          # R = column
    img[..., 1] = torch.arange(H, dtype=torch.uint8)[:, None]          # G = row
    base = dict(image=img, depth=depth, c2w=c2w, K=Kc)
    p_c, _ = backproject_depth(base, 1)
    p_n, _ = backproject_depth(dict(base, K_depth=None), 1)
    p_d, rgb = backproject_depth(dict(base, K_depth=Kd), 1)
    assert torch.equal(p_c, p_n) and not torch.allclose(p_c, p_d)
    # Kd has the longer focal: every depth pixel lands inside the colour image
    assert len(p_d) == H * W
    w2c = torch.linalg.inv(c2w.double())
    pc = p_d.double() @ w2c[:3, :3].T + w2c[:3, 3]
    u = Kd[0, 0] * pc[:, 0] / pc[:, 2] + Kd[0, 2]
    vv, uu = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    assert torch.allclose(u, uu.reshape(-1).double() + 0.5, atol=1e-4)
    # colour = the colour-camera pixel (K) the point projects to
    uc = Kc[0, 0] * pc[:, 0] / pc[:, 2] + Kc[0, 2]
    vc = Kc[1, 1] * pc[:, 1] / pc[:, 2] + Kc[1, 2]
    col, row = (rgb[:, 0] * 255).round().double(), (rgb[:, 1] * 255).round().double()
    assert bool(((uc > col - 1e-3) & (uc < col + 1 + 1e-3)).all())
    assert bool(((vc > row - 1e-3) & (vc < row + 1 + 1e-3)).all())
    assert not torch.equal(col, uu.reshape(-1).double())           # not the depth pixel


class _ListDataset:
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return dict(self.frames[i])


def test_load_frames_scales_K_depth_from_the_raw_frame():
    H, W = 40, 60
    K = torch.tensor([[50.0, 0, 30.0], [0, 50.0, 20.0], [0, 0, 1]], dtype=torch.float64)
    Kd = torch.tensor([[58.5, 0, 31.0], [0, 58.5, 19.0], [0, 0, 1]], dtype=torch.float64)
    fr = dict(image=torch.zeros(H, W, 3, dtype=torch.uint8), depth=torch.ones(H, W),
              c2w=torch.eye(4, dtype=torch.float64), K=K, K_depth=Kd)
    frames, f = gs_fit.load_frames(_ListDataset([fr]), max_res=30)
    assert f == 2 and frames[0]["depth"].shape == (20, 30)
    want = Kd.float().clone()
    want[:2] *= 0.5
    assert frames[0]["K_depth"].dtype == torch.float32
    assert torch.allclose(frames[0]["K_depth"], want)
    assert torch.allclose(frames[0]["K"][:2], K.float()[:2] * 0.5)
    frames1, _ = gs_fit.load_frames(_ListDataset([dict(fr, K_depth=None)]), max_res=30)
    assert "K_depth" not in frames1[0]


def test_export_and_checkpoint_hold_pre_activation_values(tmp_path):
    """The fitter's export stores log scales, logit opacities, sh0 and unit quats; gs_io
    gives back the physical init values (kNN scale, opacity 0.1, the point colours)."""
    N = 50
    g = torch.Generator().manual_seed(0)
    pts, rgb = torch.rand(N, 3, generator=g), torch.rand(N, 3, generator=g)
    out = gs_fit.export_splats(gs_fit.init_splats(pts, rgb, FitConfig(), "cpu"))
    p = tmp_path / "init.pt"
    save_gaussians_pt(p, out, {})
    ck = torch.load(str(p), weights_only=True)
    d2 = torch.cdist(pts.double(), pts.double()) ** 2
    knn = d2.sort(dim=1).values[:, 1:4].mean(1).sqrt().float()       # 3 nearest others
    assert torch.allclose(ck["scales"], knn.log()[:, None].expand(N, 3), atol=1e-5)
    assert torch.allclose(ck["opacities"], torch.full((N,), math.log(0.1 / 0.9)), atol=1e-6)
    assert torch.allclose(SH_C0 * ck["sh0"][:, 0] + 0.5, rgb, atol=1e-6)
    assert torch.allclose(ck["quats"].norm(dim=-1), torch.ones(N), atol=1e-6)
    assert bool((ck["scales"] < 0).all()) and bool((ck["opacities"] < 0).all())
    gio = load_gaussians_pt(p)
    assert torch.allclose(gio["scales"], knn[:, None].expand(N, 3), atol=1e-6)
    assert torch.allclose(gio["opacities"], torch.full((N,), 0.1), atol=1e-6)
    assert torch.allclose(gio["colors"], rgb, atol=1e-6)


def _toy_params(N, seed=0):
    g = torch.Generator().manual_seed(seed)
    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.randn(N, 3, generator=g)),
        "scales": torch.nn.Parameter(torch.log(0.05 * torch.ones(N, 3))),
        "quats": torch.nn.Parameter(torch.randn(N, 4, generator=g)),
        "opacities": torch.nn.Parameter(torch.zeros(N)),
        "sh0": torch.nn.Parameter(torch.zeros(N, 1, 3)),
    })
    opts = gs_fit.make_optimizers(splats, FitConfig(), 1.0)
    for k, opt in opts.items():                                # populate the Adam state
        splats[k].grad = torch.zeros_like(splats[k])
        opt.step()
        opt.zero_grad(set_to_none=True)
    return splats, opts


@needs_gsplat
def test_capped_strategy_never_exceeds_cap():
    N = 20
    grads = torch.linspace(1e-3, 2e-3, N)                      # all above grow_grad2d
    grads[:5] = 0.0                                            # 15 candidates
    # no cap binding: every candidate grows (clone or split adds one Gaussian each)
    splats, opts = _toy_params(N)
    st = gs_fit.CappedDefaultStrategy(max_gaussians=10 ** 6)
    state = dict(st.initialize_state(1.0), grad2d=grads.clone(), count=torch.ones(N),
                 cap_hit_step=-1)
    nd, ns = st._grow_gs(splats, opts, state, 600)
    assert nd + ns == 15 and len(splats["means"]) == 35 and state["cap_hit_step"] == -1
    # cap 26: only the 6 largest-gradient candidates grow
    splats, opts = _toy_params(N)
    st = gs_fit.CappedDefaultStrategy(max_gaussians=26)
    state = dict(st.initialize_state(1.0), grad2d=grads.clone(), count=torch.ones(N),
                 cap_hit_step=-1)
    means0 = splats["means"].detach().clone()
    nd, ns = st._grow_gs(splats, opts, state, 700)
    assert nd + ns == 6 and len(splats["means"]) == 26 and state["cap_hit_step"] == 700
    # the 6 largest gradients are the last 6 Gaussians; they are the ones cloned or split,
    # so the first 14 survive unchanged at the front (split moves its parents to the end)
    assert torch.equal(splats["means"].detach()[:14], means0[:14])
    assert len(opts["means"].state[splats["means"]]["exp_avg"]) == 26
    # at the cap: no growth at all
    state["grad2d"] = torch.full((26,), 1e-2)
    state["count"] = torch.ones(26)
    assert st._grow_gs(splats, opts, state, 800) == (0, 0)
    assert len(splats["means"]) == 26


def _toy_info(splats):
    """Minimal gsplat info for DefaultStrategy._update_state (zero screen gradients)."""
    N = len(splats["means"])
    m2d = torch.zeros(1, N, 2, requires_grad=True)
    m2d.grad = torch.zeros(1, N, 2)
    return dict(width=10, height=10, n_cameras=1, gaussian_ids=None, means2d=m2d,
                radii=torch.ones(1, N, 2, dtype=torch.int32))


@needs_gsplat
def test_opacity_reset_runs_inside_refine_window():
    """reset_every=4, refine window [0, 10): resets at steps 4 and 8, none at 12; off gives
    the gsplat 1.5.3 behaviour (never). If the 'off' arm fails, the installed gsplat resets
    by itself: re-check the docstring and the default."""
    logit_hi = 2.0                                             # opacity 0.88
    for on in (True, False):
        splats, opts = _toy_params(20)
        st = gs_fit.CappedDefaultStrategy(refine_start_iter=0, refine_stop_iter=10,
                                          refine_every=1000, reset_every=4, opacity_reset=on)
        state = dict(st.initialize_state(1.0), n_opacity_resets=0, cap_hit_step=-1)
        top = {}
        for step in range(14):
            if step in (0, 5, 11):
                splats["opacities"].data.fill_(logit_hi)
            st.step_post_backward(splats, opts, state, step, _toy_info(splats))
            top[step] = float(torch.sigmoid(splats["opacities"].detach()).max())
        if on:
            assert state["n_opacity_resets"] == 2
            assert top[3] > 0.8 and top[4] <= 0.01 + 1e-6 and top[8] <= 0.01 + 1e-6
            assert top[7] > 0.8 and top[12] > 0.8             # step 12 >= refine_stop
            ea = opts["opacities"].state[splats["opacities"]]["exp_avg"]
            assert float(ea.abs().max()) == 0.0
        else:
            assert state["n_opacity_resets"] == 0 and min(top.values()) > 0.8


class _FakeRender:
    """CPU stand-in for gs_fit.render: an image that depends on every parameter group and
    on means2d (so every Gaussian gets a screen-space gradient), gsplat's info keys, and a
    log of the camera (cx = 10 + frame id) of every render made with gradients enabled."""

    def __init__(self):
        self.train_ids = []

    def __call__(self, splats, viewmat, K, width, height, packed=False):
        assert K.dtype == torch.float32 and viewmat.dtype == torch.float32   # as gsplat
        if torch.is_grad_enabled():
            self.train_ids.append(round(float(K[0, 2])) - 10)
        m = splats["means"]
        pc = m @ viewmat[:3, :3].T + viewmat[:3, 3]
        means2d = (pc[:, :2] / pc[:, 2:3].abs().clamp_min(0.1))[None]         # (1,N,2)
        a = torch.sigmoid(splats["opacities"]) * torch.exp(splats["scales"]).mean(-1)
        a = a * torch.nn.functional.normalize(splats["quats"], dim=-1)[:, 0].abs()
        col = (a[:, None] * (SH_C0 * splats["sh0"][:, 0] + 0.5)).sum(0) / a.sum()
        ramp = torch.linspace(0, 1, width)[None, :, None] * torch.tanh(means2d.mean())
        img = (col[None, None] + 0.1 * ramp).expand(height, width, 3)
        info = dict(width=width, height=height, n_cameras=1, gaussian_ids=None,
                    radii=torch.ones(1, len(m), 2, dtype=torch.int32), means2d=means2d)
        return img, torch.ones(height, width), info


def _fake_frames(n, bad=(), H=16, W=20):
    """n frames on a circle around the origin, K float64 with cx = 10 + frame id."""
    g = torch.Generator().manual_seed(3)
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        c2w = _look_at([2.0 * math.sin(a), 0.2, -2.0 * math.cos(a)]).double()
        if k in bad:
            c2w[:3, 3] = -float("inf")                        # ScanNet: lost tracking
        K = torch.tensor([[20.0, 0, 10.0 + k], [0, 20.0, 8.0], [0, 0, 1]], dtype=torch.float64)
        out.append(dict(image=torch.randint(0, 256, (H, W, 3), generator=g, dtype=torch.uint8),
                        depth=None, c2w=c2w, K=K))
    return out


def _fake_cfg(**kw):
    # init_scale 0.05: 40 random points are sparse, and kNN-sized Gaussians would all be
    # larger than the scale-prune limit 0.1 * scene_scale
    base = dict(iters=60, init="random", random_points=40, init_scale=0.05, max_gaussians=150,
                grow_grad2d=1e-20, refine_every=5, log_every=0, eval_train_frames=4)
    base.update(kw)
    return FitConfig(**base)


@pytest.fixture
def one_thread():
    """Tiny CPU convolutions are ~100x slower with many threads on a loaded machine."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@needs_gsplat
def test_fit_dataset_cpu_loop_split_drop_and_wiring(monkeypatch, one_thread):
    fake = _FakeRender()
    monkeypatch.setattr(gs_fit, "render", fake)
    lams = []
    real_loss = gs_fit.photometric_loss
    monkeypatch.setattr(gs_fit, "photometric_loss",
                        lambda i, g, lam: (lams.append(lam), real_loss(i, g, lam))[1])
    cfg = _fake_cfg()
    splats, meta = gs_fit.fit_dataset(_ListDataset(_fake_frames(17, bad=(5, 8))), cfg, "cpu")
    train = [i for i in range(17) if i % 8 != 0 and i != 5]
    assert meta["dropped_frames"] == [5, 8] and meta["n_dropped"] == 2
    assert meta["test_frames"] == [0, 16]
    assert meta["train_frames"] == train and meta["n_train"] == 13 and meta["n_test"] == 2
    # only training frames are rendered with gradients, and every one of them is
    assert set(fake.train_ids) == set(train) and len(fake.train_ids) == cfg.iters
    assert lams == [0.2] * cfg.iters
    # densification and the opacity reset run inside the loop
    start, stop, reset = gs_fit.resolve_schedule(cfg)
    assert (start, stop, reset) == (10, 30, 6)
    assert meta["n_opacity_resets"] == len(range(reset, stop, reset)) == 4
    assert meta["n_init"] == 40 < meta["n_gaussians"] <= 150 and meta["cap_hit_step"] > 0
    assert splats["means"].shape == (meta["n_gaussians"], 3)
    assert meta["psnr"] is not None and meta["peak_mem_mib"] is None
    assert meta["proc_mem_mib"] is None and meta["proc_peak_mib"] is None
    txt = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))
    assert meta["config_sha256"] == hashlib.sha256(txt.encode()).hexdigest()
    assert gs_fit.config_sha256(_fake_cfg(seed=1)) != meta["config_sha256"]
    # opacity_reset=False: the library behaviour, no reset
    _, meta0 = gs_fit.fit_dataset(_ListDataset(_fake_frames(9)), _fake_cfg(opacity_reset=False),
                                  "cpu")
    assert meta0["n_opacity_resets"] == 0 and meta0["dropped_frames"] == []
    assert meta0["n_dropped"] == 0


@needs_gsplat
def test_fit_gaussians_float64_K_and_nonfinite_pose(monkeypatch, one_thread):
    monkeypatch.setattr(gs_fit, "render", _FakeRender())
    fr = _fake_frames(6)
    assert fr[0]["K"].dtype == torch.float64
    _, meta = gs_fit.fit_gaussians(fr[1:], fr[:1], _fake_cfg(iters=12), "cpu")
    assert meta["psnr"] is not None
    sp = gs_fit.init_splats(torch.randn(10, 3), torch.rand(10, 3), FitConfig(), "cpu")
    assert gs_fit.evaluate(sp, fr[:2], "cpu")[0] is not None           # evaluate casts K
    bad = _fake_frames(6, bad=(3,))
    with pytest.raises(ValueError, match="train frame 2 has a non-finite c2w"):
        gs_fit.fit_gaussians(bad[1:], bad[:1], _fake_cfg(iters=5), "cpu")
    with pytest.raises(ValueError, match="test frame 0 has a non-finite c2w"):
        gs_fit.fit_gaussians(bad[:3], bad[3:4], _fake_cfg(iters=5), "cpu")
    with pytest.raises(ValueError, match="no training frame with a finite pose"):
        gs_fit.fit_dataset(_ListDataset(_fake_frames(4, bad=(1, 2, 3))), _fake_cfg(), "cpu")


def _cli():
    spec = importlib.util.spec_from_file_location("fit_gs_cli",
                                                  os.path.join(ROOT, "scripts", "fit_gs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_boolean_flags():
    ap = _cli().build_parser()
    req = ["--dataset", "replica", "--data", "x", "--out", "y.pt"]
    a = ap.parse_args(req)
    assert a.opacity_reset is True and a.reset_every == -1
    assert ap.parse_args(req + ["--opacity-reset", "0"]).opacity_reset is False
    assert ap.parse_args(req + ["--opacity-reset", "true"]).opacity_reset is True
    with pytest.raises(SystemExit):
        ap.parse_args(req + ["--opacity-reset", "maybe"])


def test_cli_git_rev_counts_untracked_files_and_hashes_code(tmp_path):
    cli = _cli()
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
           "-c", "commit.gpgsign=false"]
    subprocess.run(git + ["init", "-q"], check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    subprocess.run(git + ["add", "a.py"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "a"], check=True)
    rev = cli._git_rev(str(tmp_path))
    assert rev and not rev.endswith("+dirty")
    (tmp_path / "b.py").write_text("y = 2\n")                 # untracked new code
    assert cli._git_rev(str(tmp_path)) == rev + "+dirty"
    h = cli.code_sha256(str(tmp_path), ("a.py", "missing.py"))
    assert h == {"a.py": hashlib.sha256(b"x = 1\n").hexdigest(), "missing.py": None}
    assert set(cli.code_sha256()) == set(cli.CODE_FILES) and None not in cli.code_sha256().values()


def test_save_load_roundtrip(tmp_path):
    N = 64
    g = torch.Generator().manual_seed(0)
    rgb = torch.rand(N, 3, generator=g)
    sp = dict(means=torch.randn(N, 3, generator=g),
              quats=torch.nn.functional.normalize(torch.randn(N, 4, generator=g), dim=-1),
              scales=torch.log(0.01 + 0.05 * torch.rand(N, 3, generator=g)),
              opacities=torch.logit(0.02 + 0.96 * torch.rand(N, generator=g)),
              sh0=((rgb - 0.5) / SH_C0)[:, None, :])
    meta = dict(psnr=25.0, n_gaussians=N, config=dict(iters=10), train_frames=[1, 2],
                torch=torch.__version__, t=(1, 2))                 # str subclass, tuple
    p = tmp_path / "gs.pt"
    save_gaussians_pt(p, sp, meta)
    g2 = load_gaussians_pt(p)
    assert torch.allclose(g2["scales"], sp["scales"].exp())
    assert torch.allclose(g2["opacities"], torch.sigmoid(sp["opacities"]))
    assert torch.allclose(g2["colors"], rgb, atol=1e-6)
    ck = torch.load(str(p), weights_only=True)
    assert ck["meta"]["psnr"] == 25.0 and ck["sh0"].shape == (N, 1, 3)
    assert ck["meta"]["torch"] == str(torch.__version__) and ck["meta"]["t"] == [1, 2]


# ----------------------------------------------------------------------------- GPU
def _env():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _write_synthetic_depth(scene):
    """Render the expected depth of scene.pt for every traj.txt pose into Replica depth
    PNGs (uint16, metres * 6553.5), so the Replica loader serves depth for the init."""
    import cv2
    from gsplat import rasterization
    from usplat.datasets import ReplicaDataset as R
    sp = {k: v.cuda() for k, v in torch.load(str(scene / "scene.pt"))["splats"].items()}
    K = torch.tensor([[R.FX, 0, R.CX], [0, R.FY, R.CY], [0, 0, 1.0]], device="cuda")
    poses = np.loadtxt(str(scene / "traj.txt")).reshape(-1, 4, 4)
    for i, c2w in enumerate(poses):
        vm = torch.linalg.inv(torch.from_numpy(c2w).float().cuda())
        with torch.no_grad():
            out, _, _ = rasterization(sp["means"], sp["quats"], sp["scales"], sp["opacities"],
                                      sp["colors"], vm[None], K[None], R.W, R.H,
                                      render_mode="RGB+ED")
        d = (out[0, ..., 3].clamp(0, 10.0) * R.DEPTH_SCALE).round().cpu().numpy()
        cv2.imwrite(str(scene / "results" / f"depth{i:06d}.png"), d.astype(np.uint16))


@gpu
def test_fit_synthetic_scene_cli(tmp_path):
    frac = os.environ.get("UAGS_GPU_MEM_FRACTION")
    if frac:
        torch.cuda.set_per_process_memory_fraction(float(frac))
    scene = tmp_path / "scene"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_synthetic_scene.py"),
                    "--out", str(scene), "--frames", "16"], check=True, env=_env(), cwd=ROOT)
    _write_synthetic_depth(scene)
    out = tmp_path / "fit.pt"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "fit_gs.py"),
                        "--dataset", "replica", "--data", str(scene), "--stride", "1",
                        "--out", str(out), "--iters", "400", "--max-res", "300",
                        "--init-pixel-step", "2", "--init-voxel", "0.01",
                        "--log-every", "100", "--seed", "0"],
                       check=True, env=_env(), cwd=ROOT, capture_output=True, text=True)
    m = json.loads(r.stdout.strip().splitlines()[-1])
    print(r.stderr[-2000:], m)
    assert m["n_test"] == 2 and m["n_train"] == 14 and m["init_method"] == "depth"
    assert (m["height"], m["width"]) == (170, 300) and m["downscale"] == 4
    assert m["psnr"] > PSNR_MIN_SYNTHETIC and m["psnr"] > m["psnr_init"] + 10.0, m
    assert m["n_opacity_resets"] == 1 and m["reset_every"] == 100      # 400 iters: [66, 200)
    ck = torch.load(str(out), weights_only=True)
    meta = ck["meta"]
    assert meta["psnr"] == m["psnr"] and meta["test_frames"] == [0, 8]
    assert meta["git"] and set(meta["code_sha256"]) >= {"usplat/gs_fit.py", "scripts/fit_gs.py"}
    # memory: the process holds the CUDA context on top of the allocator's reservation
    assert meta["peak_reserved_mib"] >= meta["peak_mem_mib"] > 0
    if importlib.util.find_spec("pynvml") is not None:
        assert meta["proc_mem_mib"] is not None, "NVML found no entry for this process"
        assert meta["proc_mem_mib"] > meta["peak_mem_mib"]
        assert meta["proc_peak_mib"] >= max(meta["proc_mem_mib"], meta["peak_reserved_mib"])
    # re-score the held-out frames from the file: raw parameters with the documented
    # activations (not gs_fit.render, not gs_io), then through gs_io (colours clamped)
    from gsplat import rasterization
    from usplat.datasets import make_dataset
    test, _ = gs_fit.load_frames(make_dataset("replica", str(scene)), 300,
                                 indices=meta["test_frames"])
    raw = dict(means=ck["means"], quats=torch.nn.functional.normalize(ck["quats"], dim=-1),
               scales=ck["scales"].exp(), opacities=torch.sigmoid(ck["opacities"]),
               colors=ck["sh0"])
    gio = load_gaussians_pt(out)
    ps_raw, ps_io = [], []
    for fr in test:
        H, W = fr["image"].shape[:2]
        vm = torch.linalg.inv(fr["c2w"].double()).float()[None].cuda()
        K = fr["K"][None].cuda()
        gt = fr["image"].cuda().float() / 255
        for g, deg, acc in ((raw, 0, ps_raw), (gio, None, ps_io)):
            with torch.no_grad():
                img, _, _ = rasterization(g["means"].cuda(), g["quats"].cuda(),
                                          g["scales"].cuda(), g["opacities"].cuda(),
                                          g["colors"].cuda(), vm, K, W, H, sh_degree=deg)
            acc.append(-10 * math.log10(float(((img[0].clamp(0, 1) - gt) ** 2).mean())))
    print("re-scored", np.mean(ps_raw), np.mean(ps_io), "reported", m["psnr"])
    assert abs(np.mean(ps_raw) - m["psnr"]) < 0.05
    assert abs(np.mean(ps_io) - m["psnr"]) < 0.5


@gpu
def test_fit_respects_gaussian_cap():
    frac = os.environ.get("UAGS_GPU_MEM_FRACTION")
    if frac:
        torch.cuda.set_per_process_memory_fraction(float(frac))
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from make_synthetic_scene import make_splats, orbit_poses
    from gsplat import rasterization
    dev = "cuda"
    sc = make_splats(1500, device=dev)
    W, H = 160, 120
    K = torch.tensor([[150.0, 0, 80.0], [0, 150.0, 60.0], [0, 0, 1]])
    frames = []
    for c2w in orbit_poses(10):
        c2w = torch.from_numpy(c2w).float()
        with torch.no_grad():
            img, _, _ = rasterization(sc["means"], sc["quats"], sc["scales"], sc["opacities"],
                                      sc["colors"], torch.linalg.inv(c2w)[None].to(dev),
                                      K[None].to(dev), W, H)
        frames.append(dict(image=(img[0].clamp(0, 1) * 255).byte().cpu(), depth=None,
                           c2w=c2w, K=K))
    cfg = FitConfig(iters=300, random_points=3000, max_gaussians=3050, refine_start=50,
                    refine_every=25, log_every=0, eval_train_frames=0)
    # "test" views are training views here: this checks the fit and the cap, not
    # generalisation (random init generalises poorly, see the module docstring)
    splats, meta = gs_fit.fit_gaussians(frames, frames[::3], cfg, dev)
    assert meta["n_init"] == 3000
    assert meta["n_gaussians"] <= 3050
    assert meta["cap_hit_step"] >= 50, meta
    assert meta["psnr"] > meta["psnr_init"]
    assert meta["reset_every"] == 30 and meta["n_opacity_resets"] == 4   # 30, 60, 90, 120
