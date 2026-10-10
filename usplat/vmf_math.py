"""
vmf_math.py — the von Mises–Fisher mean-resultant function and its inverse.

    A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa)  =  E_{x ~ vMF(mu, kappa)}[mu^T x]

It is the one special function the field needs:
  * the joint conjugate prior p(mu, kappa) ∝ C_d(kappa)^{c0} exp(kappa m0^T mu) has the
    MAP concentration A_d(kappa_hat) = ||R + m0|| / (n + c0);
  * the posterior mean of the direction is E[mu] = A_d(kappa_post) mu*, which the
    mean-field predictive uses to shrink poorly-supported Gaussians;
  * 1 - A_d(kappa) is the exact expected angular spread E[1 - cos].
  * A_d(kappa)/kappa is the variance of a vMF sample along each tangent direction
    (tangent_var), which the vote predictive P4 moment-matches (usplat.predictive).

The Bessel ratio is evaluated once per dimension d on a log-spaced kappa grid (float64)
and then interpolated in torch, so the per-Gaussian calls stay on the GPU.
Small kappa (kappa < d/2) uses Perron's continued fraction, which converges fast there
and avoids the underflow of I_nu(kappa) for large orders; larger kappa uses scipy's
exponentially scaled Bessel functions.
"""
from __future__ import annotations
import math
import numpy as np
import torch

_KAPPA_MIN, _KAPPA_MAX, _GRID = 1e-6, 1e7, 6000
_TABLES: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _ratio_cf(nu: float, x: float, terms: int = 400) -> float:
    """I_{nu+1}(x)/I_nu(x) by Perron's continued fraction, evaluated bottom-up:
    x / (2(nu+1) + x^2 / (2(nu+2) + x^2 / (2(nu+3) + ...)))."""
    x2 = x * x
    tail = 2.0 * (nu + terms)
    for k in range(terms - 1, 0, -1):
        tail = 2.0 * (nu + k) + x2 / tail
    return x / tail


def a_d_scalar(d: int, kappa: float) -> float:
    """Exact A_d(kappa) for one value (float64)."""
    nu = d / 2.0 - 1.0
    if kappa <= 0:
        return 0.0
    if kappa < max(nu, 1.0):
        return _ratio_cf(nu, kappa)
    from scipy.special import ive
    return float(ive(nu + 1.0, kappa) / ive(nu, kappa))


def _table(d: int):
    if d not in _TABLES:
        k = np.logspace(math.log10(_KAPPA_MIN), math.log10(_KAPPA_MAX), _GRID)
        a = np.array([a_d_scalar(d, float(x)) for x in k])
        a = np.maximum.accumulate(a)          # enforce monotone (guards rounding)
        _TABLES[d] = (np.log(k), a)
    return _TABLES[d]


def a_d(d: int, kappa: torch.Tensor) -> torch.Tensor:
    """A_d(kappa), elementwise; kappa >= 0. Returns a tensor like kappa."""
    logk, a = _table(d)
    kap = kappa.detach().to(torch.float64)
    x = torch.log(kap.clamp(min=_KAPPA_MIN, max=_KAPPA_MAX))
    out = _interp(x, torch.as_tensor(logk, device=kap.device),
                  torch.as_tensor(a, device=kap.device))
    out = torch.where(kap <= 0, torch.zeros_like(out), out)
    # beyond the grid A_d -> 1 - (d-1)/(2 kappa)
    big = kap > _KAPPA_MAX
    out = torch.where(big, 1.0 - (d - 1) / (2.0 * kap.clamp(min=1.0)), out)
    return out.to(kappa.dtype)


def a_d_inv(d: int, r: torch.Tensor) -> torch.Tensor:
    """kappa such that A_d(kappa) = r, elementwise; r in [0, 1). NaN stays NaN."""
    logk, a = _table(d)
    rr = r.detach().to(torch.float64)
    rc = rr.clamp(min=float(a[0]), max=float(a[-1]))
    x = _interp(rc, torch.as_tensor(a, device=rr.device),
                torch.as_tensor(logk, device=rr.device))
    out = torch.exp(x)
    out = torch.where(rr <= 0, torch.zeros_like(out), out)
    out = torch.where(torch.isnan(rr), rr, out)
    return out.to(r.dtype)


def tangent_var(d: int, kappa: torch.Tensor) -> torch.Tensor:
    """A_d(kappa)/kappa, elementwise: the variance of x ~ vMF(mu, kappa) along any direction
    orthogonal to mu, from E[x x^T] = (A/kappa) I + (1 - d A/kappa) mu mu^T (the trace is 1;
    mu^T E[x x^T] mu = 1 - (d-1) A/kappa = E[(mu^T x)^2]). 1/d at kappa = 0 (uniform), -> 0
    as kappa -> inf. Used by the tangent-Gaussian vote sampler (usplat.predictive)."""
    a = a_d(d, kappa).to(torch.float64)
    kap = kappa.detach().to(torch.float64)
    out = torch.where(kap > 0, a / kap.clamp_min(_KAPPA_MIN), torch.full_like(a, 1.0 / d))
    return out.to(kappa.dtype)


def _interp(x, xp, fp):
    """Piecewise-linear interpolation (torch), xp increasing; clamps outside."""
    idx = torch.searchsorted(xp, x.contiguous()).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    t = ((x - x0) / (x1 - x0).clamp(min=1e-300)).clamp(0.0, 1.0)
    return f0 + t * (f1 - f0)
