"""
localization.py — render-based pose refinement against the 3DGS map, using the
consolidation module's ROBUST subset of Gaussians.

This is the harness for the "localization under scene change" experiment:
a coarse pose (e.g. from retrieval or odometry) is refined by minimizing a
photometric loss between the splat render and the query image. Restricting the
render to the robust (low-aleatoric, well-observed) subset removes moved /
unstable content from the objective, which is exactly where render-based
refinement otherwise breaks under rearrangement.

gsplat's rasterizer is differentiable w.r.t. the camera (viewmats), so the
refinement is a direct first-order optimization of an SE(3) increment.
"""
from __future__ import annotations
import math
import torch


def se3_exp(xi):
    """xi: (6,) = (rho, phi) -> (4,4) SE(3) matrix, differentiable."""
    rho, phi = xi[:3], xi[3:]
    theta = phi.norm()
    K = torch.zeros(3, 3, device=xi.device, dtype=xi.dtype)
    K[0, 1], K[0, 2] = -phi[2], phi[1]
    K[1, 0], K[1, 2] = phi[2], -phi[0]
    K[2, 0], K[2, 1] = -phi[1], phi[0]
    eye = torch.eye(3, device=xi.device, dtype=xi.dtype)
    if theta < 1e-8:
        R = eye + K
        V = eye + 0.5 * K
    else:
        a = torch.sin(theta) / theta
        b = (1 - torch.cos(theta)) / theta ** 2
        c = (1 - a) / theta ** 2
        R = eye + a * K + b * (K @ K)
        V = eye + b * K + c * (K @ K)
    T = torch.eye(4, device=xi.device, dtype=xi.dtype)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def pose_errors(c2w_est, c2w_gt):
    """-> (translation error [m], rotation error [deg])."""
    dt = (c2w_est[:3, 3] - c2w_gt[:3, 3]).norm().item()
    Rrel = c2w_est[:3, :3].t() @ c2w_gt[:3, :3]
    cos = ((Rrel.trace() - 1.0) / 2.0).clamp(-1.0, 1.0)
    return dt, math.degrees(math.acos(cos.item()))


@torch.no_grad()
def _subset_splats(splats, mask):
    if mask is None:
        return splats
    return {k: (v[mask] if torch.is_tensor(v) and v.shape[:1] == mask.shape else v)
            for k, v in splats.items()}


def refine_pose(splats, image, K, init_c2w, robust_mask=None,
                iters=100, lr=3e-3, downscale=4, device="cuda"):
    """Refine init_c2w by photometric alignment of the (robust-subset) render.

    splats     : dict from gs_io.load_gaussians (must include 'colors')
    image      : (H, W, 3) uint8 query image
    K          : (3, 3) intrinsics matching `image`
    robust_mask: (N,) bool — Consolidator.robust_mask(); None = all Gaussians
    returns    : refined c2w (4,4), final loss (float)
    """
    from gsplat import rasterization
    sub = _subset_splats(splats, robust_mask)
    means = sub["means"].to(device)
    quats = sub["quats"].to(device)
    scales = sub["scales"].to(device)
    opac = sub["opacities"].to(device).reshape(-1)
    colors = sub["colors"].to(device)

    H, W = image.shape[:2]
    h, w = H // downscale, W // downscale
    tgt = torch.as_tensor(image, device=device).float() / 255.0
    tgt = torch.nn.functional.interpolate(
        tgt.permute(2, 0, 1)[None], size=(h, w), mode="bilinear",
        align_corners=False)[0].permute(1, 2, 0)
    Ks = K.clone().to(device)
    Ks[0] *= w / W
    Ks[1] *= h / H

    w2c0 = torch.linalg.inv(init_c2w.to(device).double()).float()
    xi = torch.zeros(6, device=device, requires_grad=True)
    opt = torch.optim.Adam([xi], lr=lr)
    loss = torch.tensor(float("nan"))
    for _ in range(iters):
        opt.zero_grad()
        viewmat = se3_exp(xi) @ w2c0
        render, alpha, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac,
            colors=colors, viewmats=viewmat[None], Ks=Ks[None],
            width=w, height=h, render_mode="RGB")
        a = alpha[0, ..., 0].detach()
        loss = ((render[0] - tgt).abs().mean(-1) * a).sum() / (a.sum() + 1e-8)
        loss.backward()
        opt.step()
    with torch.no_grad():
        w2c = se3_exp(xi.detach()) @ w2c0
        c2w = torch.linalg.inv(w2c.double()).float()
    return c2w.to(init_c2w.device), float(loss.item())
