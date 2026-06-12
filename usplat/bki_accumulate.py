"""
bki_accumulate.py — kernel-weighted vMF sufficient statistics via autograd.

The CSS-style BKI update needs, per Gaussian n:
    R_n     = sum_p  w_{p,n} z_p        (kernel-weighted resultant of CLIP dirs)
    n_eff_n = sum_p  w_{p,n}            (effective observation mass)
where w_{p,n} = T_{p,n} * alpha_{p,n} is the splat blending weight (the valid BKI
kernel, per CSS) and z_p is the observed CLIP direction at pixel p.

Key identity (alpha-compositing is LINEAR in per-Gaussian features):
    F_p = sum_n w_{p,n} f_n
    => d/df_n [ sum_p ( z_p . F_p ) ] = sum_p w_{p,n} z_p = R_n
We fuse R and n_eff into ONE render/backward by appending a constant ones-channel:
    buffer has d+1 channels; channel<d recovers R, channel d recovers n_eff.

Works with ANY differentiable feature rasterizer whose output is linear in the
per-Gaussian feature buffer (gsplat, Feature-3DGS, the mock). No CUDA edits.
"""
from __future__ import annotations
import torch


def bki_grad_accumulate(rasterize_fn, z_obs: torch.Tensor,
                        num_gaussians: int, device=None, dtype=torch.float32):
    """
    rasterize_fn : callable (buffer:(N, d+1)) -> image:(P, d+1), differentiable
                   w.r.t. buffer, with geometry/opacity detached.
    z_obs        : (P, d) observed per-pixel CLIP directions (should be unit norm).
    returns      : R (N, d),  n_eff (N,)
    """
    P, d = z_obs.shape
    device = device or z_obs.device
    z_obs = z_obs.to(device, dtype)

    # buffer value is irrelevant (render is linear in it) -> zeros is fine.
    buf = torch.zeros(num_gaussians, d + 1, device=device, dtype=dtype,
                      requires_grad=True)
    img = rasterize_fn(buf)                       # (P, d+1)
    # loss = sum_p [ z_p . F_p(:d) + F_p(d) ]
    loss = (img[:, :d] * z_obs).sum() + img[:, d].sum()
    loss.backward()

    grad = buf.grad                                # (N, d+1)
    R = grad[:, :d].detach().clone()               # = sum_p w_{p,n} z_p
    n_eff = grad[:, d].detach().clone()            # = sum_p w_{p,n}
    return R, n_eff


@torch.no_grad()
def bki_bruteforce_accumulate(W: torch.Tensor, z_obs: torch.Tensor):
    """Reference accumulation from an explicit (P, N) weight matrix. For tests."""
    R = W.t() @ z_obs            # (N, d) = sum_p w_{p,n} z_p
    n_eff = W.sum(dim=0)         # (N,)   = sum_p w_{p,n}
    return R, n_eff
