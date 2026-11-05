"""
cusum_reset.py — attempt 06 (configs/cusum_reset_prereg.json): a sequential change test, Page's
CUSUM over our own per-view change score, decides WHEN our field resets a Gaussian. Everything
else of our update stays (usplat/reset_fused.ExternalResetField: the reset, the depth
re-admission and the robust set).

Our test fires at a view when the persistent score c_p (min of the last k = 2 defined
c = 1 - cos(R_old, R_new)) exceeds a per-view threshold. CUSUM accumulates the excess of c_p over
a reference level delta,
    S <- max(0, S + c_p - delta)        at every view where c_p is defined (else S is kept),
so a moderate change seen over many views adds up, while static content (c_p mostly below delta)
keeps S near 0. delta of variant Cxx is the xx quantile of the pooled mapping-replay c_p values.
The threshold theta_S is the 0.999 quantile of S over the tested Gaussian-views of the static
mapping replay (pooled, numpy linear), the rule of every Table III row. A firing sets S to 0.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

VARIANTS: Dict[str, float] = {"C90": 0.90, "C97": 0.97, "C99": 0.99}


def mapping_delta(views: Sequence[dict], q_delta: float) -> float:
    """The q_delta quantile (numpy linear, float64) of the finite mapping-replay c_p values."""
    vals = [v["ours"].double().numpy() for v in views]
    a = np.concatenate(vals) if vals else np.zeros(0)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, q_delta)) if a.size else float("nan")


def cusum_null(views: Sequence[dict], n_gaussians: int, delta: float) -> np.ndarray:
    """S over the mapping replay (static: no firing, no reset), in view order, starting at 0.
    views: [{'idx': (M,) long, 'ours': (M,) c_p of the covered Gaussians idx, NaN = untested}].
    Returns S at every tested covered Gaussian-view (float64), the null of theta_S."""
    S = torch.zeros(n_gaussians, dtype=torch.float64)
    out = []
    for v in views:
        c = v["ours"].double()
        ok = torch.isfinite(c)
        i = v["idx"][ok].long()
        S[i] = torch.clamp(S[i] + c[ok] - delta, min=0.0)
        out.append(S[i].numpy().copy())
    return np.concatenate(out) if out else np.zeros(0)


def cusum_threshold(null: np.ndarray, q: float) -> float:
    return float(np.quantile(null, q)) if null.size else float("nan")


def cusum_decider(delta: float, thr: float, n_gaussians: int, device):
    """decide(field, comps, cov) for reset_fused_nor.run_variant: S on the covered Gaussians
    where c_p (comps['ours']) is defined; fire where S > thr and the Gaussian is not changed;
    a firing restarts S. Returns (fire, S where tested, NaN elsewhere)."""
    S = torch.zeros(n_gaussians, dtype=torch.float64, device=device)
    nan = torch.tensor(float("nan"), dtype=torch.float64, device=device)

    def decide(f, comps, cov):
        c = comps["ours"].to(device).double()
        tested = torch.isfinite(c) & cov
        S.copy_(torch.where(tested, torch.clamp(S + torch.nan_to_num(c) - delta, min=0.0), S))
        stat = torch.where(tested, S, nan)
        fire = tested & (S > thr) & ~f.changed
        S.masked_fill_(fire, 0.0)
        return fire, stat
    decide.state = S
    return decide
