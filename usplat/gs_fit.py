"""
gs_fit.py — fit a 3D Gaussian Splatting map on gsplat 1.5.3 (plan K9).

Recipe (the gsplat `simple_trainer.py` defaults, SH degree 0 only, batch size 1):

  parameters per Gaussian (stored pre-activation, as gsplat checkpoints do)
      means (3), quats (4, wxyz, unnormalised), scales (3, log), opacities (1, logit),
      sh0 (1,3) with colour c = SH_C0 * sh0 + 0.5   (the DC term the feature adapter uses)
  render    I = rasterization(means, quats, exp(scales), sigmoid(opacities), sh0,
                              sh_degree=0), black background
  loss      L = 0.8 |I - I_gt|_1 + 0.2 (1 - SSIM(I, I_gt))
            SSIM: 11x11 Gaussian window, sigma 1.5, C1 = 0.01^2, C2 = 0.03^2, mean over the
            valid region (the `padding="valid"` of fused-ssim used by the gsplat example)
  Adam      lr: means 1.6e-4 * scene_scale, scales 5e-3, quats 1e-3, opacities 5e-2,
            sh0 2.5e-3; eps 1e-15; the means lr decays exponentially to 1% at the last step
  scene_scale = 1.1 * max_i ||c_i - mean(c)|| over the training camera centres c_i
  densify   gsplat.strategy.DefaultStrategy (clone small / split large Gaussians whose mean
            screen-space gradient > 2e-4, prune opacity < 0.005, and after step reset_every
            also prune Gaussians larger than 0.1 * scene_scale), wrapped so the number of
            Gaussians never exceeds `max_gaussians`: when a refine step would pass the cap,
            only the candidates with the largest gradients grow, and growth stops once the
            cap is reached (pruning continues).
  schedule  refine window [refine_start, refine_stop) = [min(500, iters/6),
            min(15000, iters/2)), refine every 100 steps; opacity reset every
            reset_every = max(100, refine_stop/5) steps inside the window (3000 at the
            default 30k iterations, the gsplat value; short runs keep the same four resets
            per window and the scale prune)
  reset     opacities are clamped to 2 * prune_opa = 0.01 and their Adam moments zeroed.

Opacity reset: gsplat 1.5.3 `DefaultStrategy.step_post_backward` tests
`step % reset_every == 0 & step > 0`, which Python parses as a chained comparison that
is always False, so the library never resets (nor does the gsplat example).
CappedDefaultStrategy runs the intended reset itself (opacity_reset=True, the default);
opacity_reset=False keeps the library behaviour. Measured held-out PSNR, library -> default
(the old reset_every=3000 without reset -> 4 resets per refine window):
  synthetic scene, random init, 3000 its, seeds 0-2   14.90 +- 0.62 -> 28.32 +- 3.21 dB
                                                      (one reset at step 1000: 17.75, seed 0)
  7-Scenes chess seq-01, depth init, 3000 its, N=1    19.56 -> 19.73 dB, 242k -> 181k Gaussians
  same, 30k its                                       20.58 -> 20.60 dB (review's runs, N=1),
                                                      20.64 / 20.67 dB (this code, N=2); the
                                                      1M cap no longer binds (999k -> 953k-961k)

Init: back-project the depth of a subset of training frames (every `init_pixel_step`-th
pixel, voxel-downsampled, colours from the image) when depth exists; otherwise sample
points uniformly in the training camera frusta (depth uniform in [near, far]). Initial
scale = mean distance to the 3 nearest neighbours, opacity 0.1, random quaternions.
The random init overfits the training views: on the synthetic scene
(scripts/make_synthetic_scene.py, 16 views, 3000 iterations) the training PSNR passes 38 dB,
while the held-out PSNR is 15 dB without the opacity reset and 25-30 dB with it (numbers
above); the depth init reached 40 dB held-out in 1000 iterations (measured before the
reset fix). Use depth whenever the dataset has it.

Pixel convention: gsplat puts the centre of pixel (i, j) at (j + 0.5, i + 0.5); the
back-projection uses the same convention, so at downscale factor 1 an init point projects
to its own pixel. At factor f > 1, usplat.datasets.downscale resizes depth with
cv2.INTER_NEAREST, which reads the source pixel whose centre sits (0.5 - 0.5/f) downscaled
pixels up and left of the target centre (0.25 px at f = 2, 0.375 px at f = 4); the init
points inherit that sub-pixel shift (init only: training moves the means; not measured).

Frames are dataset dicts (usplat.datasets): image (H,W,3) uint8, depth (H,W) m or None,
c2w (4,4) OpenCV camera-to-world, K (3,3), and optionally K_depth (3,3), the intrinsics of
an unregistered depth image (e.g. 7-Scenes: depth camera f = 585, colour camera f ~ 525),
used only by the back-projection. K and c2w are cast to float32. A non-finite c2w (ScanNet
marks lost tracking with -inf) is an error in fit_gaussians; fit_dataset drops such frames
after the hold-out split and lists them in meta["dropped_frames"]. Images stay on the CPU
as uint8; one frame is uploaded per step.

Memory (meta): peak_mem_mib and peak_reserved_mib are torch allocator peaks
(max_memory_allocated / _reserved); they exclude the CUDA context and allocations outside
the caching allocator. proc_mem_mib is the largest GPU memory of this process that NVML
reported at the samples (every 100 steps, end of training, end of evaluation), and
proc_peak_mib = peak_reserved_mib + the largest (NVML process memory - torch reserved) seen
at a sample: the process peak to budget a shared GPU with (None without NVML).
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as Fn

try:
    from gsplat.strategy import DefaultStrategy
    from gsplat.strategy.ops import reset_opa
    _HAS_GSPLAT = True
except Exception:                                   # CPU-only machine without gsplat
    DefaultStrategy = object
    _HAS_GSPLAT = False

SH_C0 = 0.28209479177387814


@dataclass
class FitConfig:
    iters: int = 30_000
    max_gaussians: int = 1_000_000
    max_res: int = 640               # downscale (integer factor) so the long side <= this
    seed: int = 0
    holdout_every: int = 8           # frames i % holdout_every == 0 are held out (0: none)
    ssim_lambda: float = 0.2
    # init
    init: str = "auto"               # auto (depth if any frame has depth) | depth | random
    init_frames: int = 32            # training frames back-projected (evenly spaced)
    init_pixel_step: int = 4
    init_voxel: float = 0.02         # voxel size [m] for the depth init (0: off)
    init_max_points: int = 300_000
    random_points: int = 100_000
    rand_near: float = -1.0          # random init depth range; <0: 0.1 / 3.0 * scene_scale
    rand_far: float = -1.0
    depth_min: float = 0.05
    depth_max: float = 10.0
    init_opacity: float = 0.1
    init_scale: float = 1.0
    scene_scale: float = -1.0        # <0: 1.1 * camera-centre extent
    # Adam (gsplat example)
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    # densification (DefaultStrategy); <0: min(500, iters // 6) and min(15000, iters // 2)
    refine_start: int = -1
    refine_stop: int = -1
    refine_every: int = 100
    reset_every: int = -1            # <=0: max(refine_every, refine_stop // 5) (3000 at 30k)
    opacity_reset: bool = True       # False: gsplat 1.5.3 behaviour (the reset never runs)
    grow_grad2d: float = 2e-4
    prune_opa: float = 0.005
    # logging / eval
    log_every: int = 500
    eval_train_frames: int = 16      # training frames scored for psnr_train (0: skip)


# ----------------------------------------------------------------------------- frames
def split_holdout(n: int, every: int = 8):
    """Indices (train, test): frame i is held out iff every > 0 and i % every == 0."""
    if every <= 0 or n < 2:
        return list(range(n)), []
    test = [i for i in range(n) if i % every == 0]
    train = [i for i in range(n) if i % every != 0]
    return train, test


def downscale_factor(h: int, w: int, max_res: int) -> int:
    """Smallest integer factor f with max(h, w) // f <= max_res (1 if max_res <= 0)."""
    if max_res <= 0:
        return 1
    return max(1, math.ceil(max(h, w) / max_res))


def load_frames(ds, max_res: int = 640, indices: Optional[Sequence[int]] = None):
    """Read dataset frames, downscaled with usplat.datasets.downscale. Returns (frames, f).

    A K_depth of the raw frame is rescaled here to the output depth grid (downscale knows
    only K), from the raw frame, so a later downscale that also scales it is not doubled."""
    from usplat.datasets import downscale
    idx = range(len(ds)) if indices is None else indices
    frames, f = [], None
    for i in idx:
        raw = ds[i]
        if f is None:
            h, w = raw["image"].shape[:2]
            f = downscale_factor(h, w, max_res)
        fr = downscale(raw, f)
        d = fr.get("depth")
        out = dict(image=fr["image"].contiguous(), depth=None if d is None else d.float(),
                   c2w=fr["c2w"].float(), K=fr["K"].float())
        if raw.get("K_depth") is not None and d is not None:
            Kd = raw["K_depth"].clone().float()
            (Hd, Wd), (hd, wd) = raw["depth"].shape[:2], d.shape[:2]
            Kd[0] *= wd / Wd
            Kd[1] *= hd / Hd
            out["K_depth"] = Kd
        frames.append(out)
    return frames, (f or 1)


def valid_pose(c2w) -> bool:
    """False for a pose with inf / nan entries (ScanNet marks lost tracking with -inf)."""
    return bool(torch.isfinite(torch.as_tensor(c2w)).all())


def _prep_frames(frames, what):
    """Shallow copies with float32 K / K_depth / c2w; ValueError on a non-finite pose."""
    out = []
    for i, f in enumerate(frames):
        if not valid_pose(f["c2w"]):
            raise ValueError(f"{what} frame {i} has a non-finite c2w (lost tracking?); drop "
                             "it before fitting (fit_dataset does)")
        g = dict(f, c2w=torch.as_tensor(f["c2w"]).float(), K=torch.as_tensor(f["K"]).float())
        if g.get("K_depth") is not None:
            g["K_depth"] = torch.as_tensor(g["K_depth"]).float()
        out.append(g)
    return out


def _evenly(seq, m):
    """min(m, len(seq)) evenly spaced items of seq (first and last included)."""
    m = min(m, len(seq))
    return [seq[int(round(x))] for x in np.linspace(0, len(seq) - 1, m)] if m > 0 else []


def camera_scene_scale(c2ws: torch.Tensor) -> float:
    """gsplat convention: 1.1 * max distance of the camera centres from their mean."""
    c = c2ws[:, :3, 3].double()
    return 1.1 * float((c - c.mean(0)).norm(dim=-1).max()) if len(c) else 0.0


# ----------------------------------------------------------------------------- init
def backproject_depth(frame, pixel_step=4, depth_min=0.05, depth_max=10.0):
    """World points (M,3) and colours (M,3) in [0,1] from every pixel_step-th pixel.

    With frame["K_depth"] (unregistered depth) the depth pixels are back-projected with
    K_depth and each point takes the colour of the pixel it projects to with K (the two
    cameras are assumed to share the pose); points outside the colour image are dropped."""
    Kd = frame.get("K_depth")
    K = (Kd if Kd is not None else frame["K"]).double()
    d, img, c2w = frame["depth"], frame["image"], frame["c2w"].double()
    H, W = d.shape
    v, u = torch.meshgrid(torch.arange(0, H, pixel_step), torch.arange(0, W, pixel_step),
                          indexing="ij")
    v, u = v.reshape(-1), u.reshape(-1)
    z = d[v, u].double()
    ok = torch.isfinite(z) & (z > depth_min) & (z < depth_max)
    v, u, z = v[ok], u[ok], z[ok]
    x = (u.double() + 0.5 - K[0, 2]) / K[0, 0] * z          # gsplat pixel centres
    y = (v.double() + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = torch.stack([x, y, z], -1)
    if Kd is not None:
        Kc = frame["K"].double()
        u = torch.floor(Kc[0, 0] * x / z + Kc[0, 2]).long()
        v = torch.floor(Kc[1, 1] * y / z + Kc[1, 2]).long()
        Hc, Wc = img.shape[:2]
        inside = (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc)
        pc, u, v = pc[inside], u[inside], v[inside]
    pw = pc @ c2w[:3, :3].T + c2w[:3, 3]
    rgb = img[v, u].float() / 255.0
    return pw.float(), rgb


def voxel_downsample(points: torch.Tensor, colors: torch.Tensor, voxel: float):
    """Average points and colours per voxel of side `voxel` (no-op if voxel <= 0)."""
    if voxel <= 0 or len(points) == 0:
        return points, colors
    key = torch.floor(points.double() / voxel).long()
    uniq, inv = torch.unique(key, dim=0, return_inverse=True)
    acc = torch.zeros(len(uniq), 7, dtype=torch.float64)       # sum p, sum c, count
    ones = torch.ones(len(inv), 1, dtype=torch.float64)
    acc.index_add_(0, inv, torch.cat([points.double(), colors.double(), ones], 1))
    acc = acc[:, :6] / acc[:, 6:]
    return acc[:, :3].float(), acc[:, 3:].float()


def random_frustum_points(frames, n, near, far, generator=None):
    """n points uniform in pixel and in depth [near, far] of random training cameras."""
    g = generator
    fi = torch.randint(len(frames), (n,), generator=g)
    pts, cols = torch.empty(n, 3), torch.empty(n, 3)
    for j in fi.unique().tolist():
        sel = torch.where(fi == j)[0]
        fr = frames[j]
        H, W = fr["image"].shape[:2]
        K, c2w = fr["K"].double(), fr["c2w"].double()
        m = len(sel)
        u = torch.rand(m, generator=g, dtype=torch.float64) * W
        v = torch.rand(m, generator=g, dtype=torch.float64) * H
        z = near + (far - near) * torch.rand(m, generator=g, dtype=torch.float64)
        pc = torch.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
        pts[sel] = (pc @ c2w[:3, :3].T + c2w[:3, 3]).float()
        ui = u.long().clamp(0, W - 1)
        vi = v.long().clamp(0, H - 1)
        cols[sel] = fr["image"][vi, ui].float() / 255.0
    return pts, cols


def knn_mean_dist(points: torch.Tensor, k: int = 3) -> torch.Tensor:
    """sqrt(mean squared distance to the k nearest other points) (the gsplat init scale)."""
    from scipy.spatial import cKDTree
    P = points.double().numpy()
    kk = min(k + 1, len(P))
    if kk < 2:
        return torch.full((len(P),), 0.01)
    d, _ = cKDTree(P).query(P, k=kk)
    d2 = (d[:, 1:] ** 2).mean(-1)
    return torch.from_numpy(np.sqrt(d2)).float().clamp_min(1e-6)


def init_points(train_frames, cfg: FitConfig, scene_scale: float, gen: torch.Generator):
    """Initial points and colours; returns (points, colors, method)."""
    has_depth = any(f.get("depth") is not None for f in train_frames)
    method = cfg.init
    if method == "auto":
        method = "depth" if has_depth else "random"
    if method == "depth":
        if not has_depth:
            raise ValueError("init=depth but no training frame has depth")
        cand = [i for i, f in enumerate(train_frames) if f.get("depth") is not None]
        pick = _evenly(cand, cfg.init_frames if cfg.init_frames > 0 else len(cand))
        P, C = zip(*[backproject_depth(train_frames[i], cfg.init_pixel_step,
                                       cfg.depth_min, cfg.depth_max) for i in pick])
        pts, cols = voxel_downsample(torch.cat(P), torch.cat(C), cfg.init_voxel)
        cap = min(cfg.init_max_points, cfg.max_gaussians)
    elif method == "random":
        near = cfg.rand_near if cfg.rand_near > 0 else 0.1 * scene_scale
        far = cfg.rand_far if cfg.rand_far > 0 else 3.0 * scene_scale
        n = min(cfg.random_points, cfg.max_gaussians)
        pts, cols = random_frustum_points(train_frames, n, near, far, gen)
        cap = n
    else:
        raise ValueError(f"unknown init '{cfg.init}'")
    if len(pts) == 0:
        raise ValueError("init produced no points (check depth range)")
    if len(pts) > cap:
        keep = torch.randperm(len(pts), generator=gen)[:cap]
        pts, cols = pts[keep], cols[keep]
    return pts, cols, method


def init_splats(points, rgbs, cfg: FitConfig, device="cuda") -> torch.nn.ParameterDict:
    N = len(points)
    scales = torch.log(knn_mean_dist(points) * cfg.init_scale)[:, None].repeat(1, 3)
    quats = torch.rand(N, 4)                                  # gsplat example: random
    opac = torch.logit(torch.full((N,), cfg.init_opacity))
    sh0 = ((rgbs - 0.5) / SH_C0)[:, None, :]                  # (N,1,3)
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(points.float()),
        "scales": torch.nn.Parameter(scales.float()),
        "quats": torch.nn.Parameter(quats.float()),
        "opacities": torch.nn.Parameter(opac.float()),
        "sh0": torch.nn.Parameter(sh0.float()),
    }).to(device)


def make_optimizers(splats, cfg: FitConfig, scene_scale: float):
    lrs = dict(means=cfg.lr_means * scene_scale, scales=cfg.lr_scales, quats=cfg.lr_quats,
               opacities=cfg.lr_opacities, sh0=cfg.lr_sh0)
    return {k: torch.optim.Adam([{"params": splats[k], "lr": lr, "name": k}], eps=1e-15)
            for k, lr in lrs.items()}


def make_means_scheduler(optimizer, iters: int):
    """Exponential decay of the means lr to 1% after `iters` scheduler steps."""
    return torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                  gamma=0.01 ** (1.0 / max(iters, 1)))


def resolve_schedule(cfg: FitConfig):
    """(refine_start, refine_stop, reset_every) with the auto values of FitConfig."""
    start = cfg.refine_start if cfg.refine_start >= 0 else min(500, cfg.iters // 6)
    stop = cfg.refine_stop if cfg.refine_stop >= 0 else min(15_000, cfg.iters // 2)
    reset = cfg.reset_every if cfg.reset_every > 0 else max(cfg.refine_every, stop // 5)
    return start, stop, reset


def export_splats(splats) -> dict:
    """CPU float32 copy of the pre-activation parameters, quaternions normalised."""
    out = {k: v.detach().float().cpu().contiguous() for k, v in splats.items()}
    out["quats"] = Fn.normalize(out["quats"], dim=-1)
    return out


# ----------------------------------------------------------------------------- strategy
if _HAS_GSPLAT:
    @dataclass
    class CappedDefaultStrategy(DefaultStrategy):
        """DefaultStrategy whose growth never takes the count above max_gaussians.

        A clone or a split adds exactly one Gaussian, so a refine step grows by the number
        of candidates (mean 2D gradient > grow_grad2d). If that exceeds the budget, the
        candidates outside the `budget` largest gradients get their accumulated gradient
        zeroed before the library's grow step (the running stats are reset right after).
        The step at which the cap was first binding is kept in state["cap_hit_step"].

        With opacity_reset, the opacity reset that gsplat 1.5.3 never runs (see the module
        docstring) runs here after the library step, at 0 < step < refine_stop_iter with
        step % reset_every == 0; it is idempotent, so a fixed library would do no harm.
        The resets are counted in state["n_opacity_resets"].
        """
        max_gaussians: int = 1_000_000
        opacity_reset: bool = True

        def step_post_backward(self, params, optimizers, state, step, info, packed=False):
            super().step_post_backward(params, optimizers, state, step, info, packed=packed)
            if (self.opacity_reset and 0 < step < self.refine_stop_iter
                    and step % self.reset_every == 0):
                reset_opa(params=params, optimizers=optimizers, state=state,
                          value=self.prune_opa * 2.0)
                state["n_opacity_resets"] = state.get("n_opacity_resets", 0) + 1

        @torch.no_grad()
        def _grow_gs(self, params, optimizers, state, step):
            assert self.refine_scale2d_stop_iter == 0, "2D-scale splitting is not capped"
            budget = self.max_gaussians - len(params["means"])
            grads = state["grad2d"] / state["count"].clamp_min(1)
            cand = grads > self.grow_grad2d
            n_cand = int(cand.sum())
            if n_cand > max(budget, 0) and state.get("cap_hit_step", -1) < 0:
                state["cap_hit_step"] = int(step)
            if budget <= 0:
                return 0, 0
            if n_cand > budget:
                idx = torch.where(cand)[0]
                top = torch.topk(grads[idx], budget).indices
                drop = torch.ones(len(idx), dtype=torch.bool, device=idx.device)
                drop[top] = False
                state["grad2d"][idx[drop]] = 0.0
            return super()._grow_gs(params, optimizers, state, step)
else:
    CappedDefaultStrategy = None


# ----------------------------------------------------------------------------- losses
def _gauss_1d(size, sigma, device, dtype):
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def ssim(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5,
         c1: float = 0.01 ** 2, c2: float = 0.03 ** 2) -> torch.Tensor:
    """Mean SSIM of (B,C,H,W) images in [0,1] over the valid region (no padding).

    The five local moments are blurred in one separable grouped convolution."""
    B, C = x.shape[:2]
    g = _gauss_1d(window, sigma, x.device, x.dtype)
    st = torch.cat([x, y, x * x, y * y, x * y], 1)            # (B,5C,H,W)
    n = st.shape[1]
    st = Fn.conv2d(st, g.view(1, 1, 1, -1).expand(n, 1, 1, window), groups=n)
    st = Fn.conv2d(st, g.view(1, 1, -1, 1).expand(n, 1, window, 1), groups=n)
    mx, my, sxx, syy, sxy = st.split(C, 1)
    vx, vy, cxy = sxx - mx * mx, syy - my * my, sxy - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return s.mean()


def photometric_loss(img: torch.Tensor, gt: torch.Tensor, ssim_lambda: float = 0.2):
    """(1 - lambda) |img - gt|_1 + lambda (1 - SSIM) of (H,W,3) images."""
    l1 = (img - gt).abs().mean()
    s = ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
    return (1.0 - ssim_lambda) * l1 + ssim_lambda * (1.0 - s)


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = float(((x - y) ** 2).mean())
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


# ----------------------------------------------------------------------------- render
def render(splats, viewmat, K, width, height, packed=False):
    """(H,W,3) image, (H,W) alpha and the gsplat info dict for one camera."""
    from gsplat import rasterization
    img, alpha, info = rasterization(
        means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]), colors=splats["sh0"],
        viewmats=viewmat[None], Ks=K[None], width=width, height=height,
        sh_degree=0, packed=packed, near_plane=0.01, far_plane=1e10)
    return img[0], alpha[0, ..., 0], info


def _viewmat(fr):
    return torch.linalg.inv(fr["c2w"].double()).float()


@torch.no_grad()
def evaluate(splats, frames, device="cuda"):
    """Mean per-frame PSNR and SSIM (renders clamped to [0,1]); None if no frames."""
    if not frames:
        return None, None
    ps, ss = [], []
    for fr in frames:
        H, W = fr["image"].shape[:2]
        gt = fr["image"].to(device).float() / 255.0
        img, _, _ = render(splats, _viewmat(fr).to(device), fr["K"].float().to(device), W, H)
        img = img.clamp(0, 1)
        ps.append(psnr(img, gt))
        ss.append(float(ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])))
    return float(np.mean(ps)), float(np.mean(ss))


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class ProcessGPUMemory:
    """Samples the GPU memory NVML reports for this process (CUDA context, library
    workspaces and every allocation, unlike torch's allocator counters).

    max_mib: largest sampled value; max_overhead_mib: largest (NVML - torch reserved) at
    a sample, so peak_reserved + max_overhead_mib estimates the process peak between
    samples. Both stay None on a CPU device or without pynvml / a process entry."""

    def __init__(self, device):
        self.dev = torch.device(device)
        self.max_mib = self.max_overhead_mib = None
        self._nv = None
        if self.dev.type != "cuda":
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nv = pynvml
        except Exception:
            self._nv = None

    def sample(self):
        if self._nv is None:
            return
        nv, pid, used = self._nv, os.getpid(), None
        try:
            for i in range(nv.nvmlDeviceGetCount()):
                for p in nv.nvmlDeviceGetComputeRunningProcesses(nv.nvmlDeviceGetHandleByIndex(i)):
                    if p.pid == pid and p.usedGpuMemory is not None:
                        used = (used or 0) + p.usedGpuMemory
        except Exception:
            return
        if used is None:
            return
        used /= 2 ** 20
        over = used - torch.cuda.memory_reserved(self.dev) / 2 ** 20
        self.max_mib = used if self.max_mib is None else max(self.max_mib, used)
        self.max_overhead_mib = (over if self.max_overhead_mib is None
                                 else max(self.max_overhead_mib, over))

    def close(self):
        if self._nv is not None:
            try:
                self._nv.nvmlShutdown()
            except Exception:
                pass
            self._nv = None


def config_sha256(cfg: FitConfig) -> str:
    """sha256 of the canonical JSON of the config (sorted keys, no spaces)."""
    txt = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                     allow_nan=False)
    return hashlib.sha256(txt.encode("ascii")).hexdigest()


# ----------------------------------------------------------------------------- fit
def fit_gaussians(train_frames, test_frames=(), cfg: Optional[FitConfig] = None,
                  device="cuda", log: Optional[Callable[[str], None]] = None):
    """Fit a 3DGS map to posed frames. Returns (splats: dict of CPU tensors, meta: dict).

    splats holds the gsplat pre-activation parameters (means, quats, log scales, logit
    opacities, sh0 (N,1,3)); meta holds the metrics (held-out PSNR before/after, SSIM,
    Gaussian count, GPU memory (see the module docstring), times), the config and its
    sha256. Every frame needs a finite c2w (ValueError otherwise)."""
    if not _HAS_GSPLAT:
        raise ImportError("gsplat is required for fitting")
    cfg = cfg or FitConfig()
    log = log or (lambda s: None)
    if not train_frames:
        raise ValueError("no training frames")
    train_frames = _prep_frames(train_frames, "train")
    test_frames = _prep_frames(test_frames, "test")
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    pm = ProcessGPUMemory(device)
    try:
        t0 = time.perf_counter()

        # ---- scene scale + init --------------------------------------------------------
        c2ws = torch.stack([f["c2w"] for f in train_frames])
        scene_scale = cfg.scene_scale if cfg.scene_scale > 0 else camera_scene_scale(c2ws)
        if scene_scale < 1e-6:
            scene_scale = 1.0                                 # single camera position
        pts, cols, method = init_points(train_frames, cfg, scene_scale, gen)
        splats = init_splats(pts, cols, cfg, device)
        n_init = len(pts)
        optimizers = make_optimizers(splats, cfg, scene_scale)
        sched = make_means_scheduler(optimizers["means"], cfg.iters)
        refine_start, refine_stop, reset_every = resolve_schedule(cfg)
        strategy = CappedDefaultStrategy(
            prune_opa=cfg.prune_opa, grow_grad2d=cfg.grow_grad2d,
            refine_start_iter=refine_start, refine_stop_iter=refine_stop,
            refine_every=cfg.refine_every, reset_every=reset_every,
            max_gaussians=cfg.max_gaussians, opacity_reset=cfg.opacity_reset)
        strategy.check_sanity(splats, optimizers)
        state = strategy.initialize_state(scene_scale=scene_scale)
        state["cap_hit_step"] = -1
        state["n_opacity_resets"] = 0
        log(f"[fit] init={method} N={n_init} scene_scale={scene_scale:.3f} "
            f"train={len(train_frames)} test={len(test_frames)} "
            f"refine=[{refine_start},{refine_stop}) reset_every={reset_every}"
            f"{'' if cfg.opacity_reset else ' (off)'}")

        _sync(device)
        t_init = time.perf_counter() - t0
        psnr_init, ssim_init = evaluate(splats, test_frames, device)

        # ---- optimisation --------------------------------------------------------------
        views = [(_viewmat(f).to(device), f["K"].to(device)) for f in train_frames]
        order: list[int] = []
        t1 = time.perf_counter()
        loss_v = float("nan")
        for step in range(cfg.iters):
            if not order:
                order = torch.randperm(len(train_frames), generator=gen).tolist()
            j = order.pop()
            fr = train_frames[j]
            H, W = fr["image"].shape[:2]
            gt = fr["image"].to(device, non_blocking=True).float() / 255.0
            img, _, info = render(splats, views[j][0], views[j][1], W, H)
            strategy.step_pre_backward(splats, optimizers, state, step, info)
            loss = photometric_loss(img, gt, cfg.ssim_lambda)
            loss.backward()
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            sched.step()
            strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)
            last = step + 1 == cfg.iters
            if step % 100 == 0 or last:
                pm.sample()
            if last or (cfg.log_every > 0 and (step + 1) % cfg.log_every == 0):
                loss_v = float(loss.detach())
                log(f"[fit] step {step + 1}/{cfg.iters} loss {loss_v:.4f} "
                    f"N {len(splats['means'])} {time.perf_counter() - t1:.1f}s")
        _sync(device)
        t_train = time.perf_counter() - t1

        # ---- evaluation ----------------------------------------------------------------
        t2 = time.perf_counter()
        psnr_test, ssim_test = evaluate(splats, test_frames, device)
        psnr_train, _ = evaluate(splats, _evenly(train_frames, cfg.eval_train_frames), device)
        _sync(device)
        pm.sample()
        t_eval = time.perf_counter() - t2
    finally:
        pm.close()

    out = export_splats(splats)
    H, W = train_frames[0]["image"].shape[:2]
    peak_res = (torch.cuda.max_memory_reserved(device) / 2 ** 20) if cuda else None
    meta = dict(
        iters=int(cfg.iters), n_gaussians=int(len(out["means"])), n_init=int(n_init),
        init_method=method, psnr=psnr_test, ssim=ssim_test, psnr_init=psnr_init,
        ssim_init=ssim_init, psnr_train=psnr_train, final_loss=loss_v,
        n_train=len(train_frames), n_test=len(test_frames), height=int(H), width=int(W),
        scene_scale=float(scene_scale), refine_start=int(refine_start),
        refine_stop=int(refine_stop), reset_every=int(reset_every),
        n_opacity_resets=int(state.get("n_opacity_resets", 0)),
        cap_hit_step=int(state.get("cap_hit_step", -1)),
        peak_mem_mib=(torch.cuda.max_memory_allocated(device) / 2 ** 20) if cuda else None,
        peak_reserved_mib=peak_res,
        proc_mem_mib=pm.max_mib,
        proc_peak_mib=(None if pm.max_overhead_mib is None or peak_res is None
                       else peak_res + pm.max_overhead_mib),
        memory_note="peak_mem_mib / peak_reserved_mib: torch allocator peaks, no CUDA context; "
                    "proc_mem_mib: max sampled NVML process memory; proc_peak_mib: "
                    "peak_reserved_mib + max(NVML process - torch reserved) at the samples",
        time_init_s=t_init, time_train_s=t_train, time_eval_s=t_eval,
        wall_time_s=time.perf_counter() - t0,
        config=asdict(cfg), config_sha256=config_sha256(cfg))
    return out, meta


def fit_dataset(ds, cfg: Optional[FitConfig] = None, device="cuda", log=None):
    """Load a usplat.datasets dataset, hold out every cfg.holdout_every-th frame, fit.

    Frames with a non-finite pose are dropped after the split (so they do not move other
    frames between train and test), counted in meta["n_dropped"] and listed in
    meta["dropped_frames"]; meta train_frames / test_frames are dataset indices."""
    cfg = cfg or FitConfig()
    t0 = time.perf_counter()
    frames, factor = load_frames(ds, cfg.max_res)
    tr_idx, te_idx = split_holdout(len(frames), cfg.holdout_every)
    bad = [i for i, f in enumerate(frames) if not valid_pose(f["c2w"])]
    if bad:
        if log:
            log(f"[fit] dropped {len(bad)}/{len(frames)} frames with a non-finite pose: "
                f"{bad[:20]}{' ...' if len(bad) > 20 else ''}")
        drop = set(bad)
        tr_idx = [i for i in tr_idx if i not in drop]
        te_idx = [i for i in te_idx if i not in drop]
    if not tr_idx:
        raise ValueError(f"no training frame with a finite pose ({len(bad)} of "
                         f"{len(frames)} frames dropped)")
    t_load = time.perf_counter() - t0
    splats, meta = fit_gaussians([frames[i] for i in tr_idx], [frames[i] for i in te_idx],
                                 cfg, device, log)
    meta.update(downscale=int(factor), time_load_s=t_load,
                wall_time_s=meta["wall_time_s"] + t_load,
                train_frames=tr_idx, test_frames=te_idx, n_dropped=len(bad),
                dropped_frames=bad)
    return splats, meta


def save_gaussians_pt(path, splats: dict, meta: dict):
    """Flat dict {means, quats, scales (log), opacities (logit), sh0 (N,1,3), meta}; loadable
    with usplat.gs_io.load_gaussians_pt (weights_only=True, so meta holds plain types)."""
    # plain Python types only (e.g. torch.__version__ is a str subclass that the
    # weights_only unpickler rejects); non-JSON values become strings
    meta = json.loads(json.dumps(meta, default=str))
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    torch.save(dict(**{k: splats[k] for k in ("means", "quats", "scales", "opacities", "sh0")},
                    meta=meta), str(path))
