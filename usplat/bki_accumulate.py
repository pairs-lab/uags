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

Works with ANY rasterizer whose output is affine in the per-Gaussian feature
buffer (gsplat without SH, Feature-3DGS without a decoder, the mock). No CUDA edits.
The same identity with an arbitrary per-pixel target Y (P, C) gives
    d/dbuf [ sum_p Y_p . F_p ] = sum_p w_{p,n} Y_p ,
so several channel groups (raw features, centred features, depth, each with its own
pixel mask and its own mass channel) come out of ONE render and ONE backward pass.
"""
from __future__ import annotations
import torch


def bki_grad_accumulate_targets(rasterize_fn, targets: torch.Tensor,
                                num_gaussians: int, device=None,
                                dtype=torch.float32):
    """
    rasterize_fn : callable (buffer:(N, C)) -> image:(P, C), affine in buffer,
                   with geometry/opacity detached.
    targets      : (P, C) per-pixel targets Y.
    returns      : (N, C) = sum_p w_{p,n} Y_p
    """
    device = device or targets.device
    Y = targets.to(device, dtype)
    # buffer value is irrelevant (render is affine in it) -> zeros is fine.
    buf = torch.zeros(num_gaussians, Y.shape[1], device=device, dtype=dtype,
                      requires_grad=True)
    img = rasterize_fn(buf)                        # (P, C)
    (img * Y).sum().backward()
    return buf.grad.detach()


def bki_grad_accumulate_groups(rasterize_fn, groups, num_gaussians,
                               device=None, dtype=torch.float32):
    """Accumulate several channel groups in one pass.

    groups : list of (target (P, C_g) or None, pixel_weight (P,) or None).
             Each group gets C_g weighted-target channels plus one mass channel:
             S_g = sum_p w_{p,n} a_p Y_p,  m_g = sum_p w_{p,n} a_p.
    returns: list of (S_g (N, C_g) or None, m_g (N,))
    """
    cols, spans = [], []
    P = None
    for target, weight in groups:
        if target is not None:
            P = target.shape[0]
        elif weight is not None:
            P = weight.shape[0]
    assert P is not None, "at least one group needs a target or a pixel weight"
    for target, weight in groups:
        dev = device or (target.device if target is not None else weight.device)
        a = (torch.ones(P, device=dev, dtype=dtype) if weight is None
             else weight.to(dev, dtype))
        c = 0
        if target is not None:
            tg = target.to(dev, dtype)
            # masked pixels contribute exactly 0 even if their target is NaN/inf
            cols.append(torch.where(a[:, None] > 0, tg, torch.zeros_like(tg)) * a[:, None])
            c = target.shape[1]
        cols.append(a[:, None])
        spans.append(c)
    S = bki_grad_accumulate_targets(rasterize_fn, torch.cat(cols, dim=1),
                                    num_gaussians, device=device, dtype=dtype)
    out, i = [], 0
    for c in spans:
        out.append((S[:, i:i + c] if c else None, S[:, i + c]))
        i += c + 1
    return out


def bki_grad_accumulate(rasterize_fn, z_obs: torch.Tensor,
                        num_gaussians: int, device=None, dtype=torch.float32):
    """
    rasterize_fn : callable (buffer:(N, d+1)) -> image:(P, d+1), affine in the
                   buffer, with geometry/opacity detached.
    z_obs        : (P, d) observed per-pixel CLIP directions (should be unit norm).
    returns      : R (N, d),  n_eff (N,)  (pixel mass: sum_p w_{p,n})
    """
    (R, n_eff), = bki_grad_accumulate_groups(rasterize_fn, [(z_obs, None)],
                                             num_gaussians, device, dtype)
    return R, n_eff


@torch.no_grad()
def bki_bruteforce_accumulate(W: torch.Tensor, z_obs: torch.Tensor):
    """Reference accumulation from an explicit (P, N) weight matrix. For tests."""
    R = W.t() @ z_obs            # (N, d) = sum_p w_{p,n} z_p
    n_eff = W.sum(dim=0)         # (N,)   = sum_p w_{p,n}
    return R, n_eff
