"""
whole_update.py — attempt 07 (configs/whole_update_prereg.json): change scores that compare a
Gaussian's evidence over the whole update with its evidence after the mapping pass, read from the
raw-channel sums of our field (vmf_field._Channel: R = sum_v w_v u_v, W1 = sum_v w_v,
W2 = sum_v w_v^2, Q = sum_v w_v^2 |u_v|^2), and the whole-update mean of a per-view residual
(the same aggregation for the per-view baselines).

  M  the raw channel of the saved field after the mapping pass (Table III's field)
  E  the raw channel of Table III's copy with the change test off after the update (no reset),
     so E = M + U, U the sums of the update's views

Variants (higher = more likely stale; NaN = undefined, ranked last by the evaluation):
  dir   1 - cos(R_M, R_U), R_U = R_E - R_M     our test's cosine, the update's views pooled
  agr   |R_M| / W1_M - |R_E| / W1_E           drop of the mean resultant length
  kish  rho_M - rho_E                         drop of the Kish agreement
                                              (_Channel.agreement('kish'), in float64)
"""
from __future__ import annotations

import torch

from usplat.vmf_field import _Channel

VARIANTS = ("dir", "agr", "kish")
# baseline row -> the per-view residual it averages (eval_change_nor.RESIDUALS' names)
RESIDUALS = dict(ncc_mean="ncc", feat_mean="feat", depth_mean="depth_render")
SUMS = ("R", "W1", "W2", "Q")


def channel_sums(ch):
    """{R, W1, W2, Q} of a vmf_field._Channel as float64 CPU tensors."""
    return {k: getattr(ch, k).detach().double().cpu() for k in SUMS}


def as_channel(s):
    """A float64 CPU _Channel holding the sums s (Wr is not needed by agreement())."""
    ch = _Channel(0, 0, "cpu", torch.float64)          # no (N, d) allocation: set below
    for k in SUMS:
        setattr(ch, k, s[k].double())
    return ch


def variant_scores(M, E, names=VARIANTS):
    """{variant: (N,) float64 score} from the float64 sums M (after mapping) and E (after the
    update, no reset); NaN where undefined (module docstring)."""
    tiny = torch.finfo(torch.float64).tiny
    nan = torch.full_like(M["W1"], float("nan"))
    out = {}
    for v in names:
        if v == "dir":
            RU = E["R"] - M["R"]
            nm, nu = M["R"].norm(dim=-1), RU.norm(dim=-1)
            ok = (M["W1"] > 0) & (nm > 0) & (nu > 0)
            cos = (M["R"] * RU).sum(-1) / (nm * nu).clamp_min(tiny)
            out[v] = torch.where(ok, 1.0 - cos.clamp(-1.0, 1.0), nan)
        elif v == "agr":
            ok = (M["W1"] > 0) & (E["W1"] > 0)
            rm = M["R"].norm(dim=-1) / M["W1"].clamp_min(tiny)
            re = E["R"].norm(dim=-1) / E["W1"].clamp_min(tiny)
            out[v] = torch.where(ok, rm - re, nan)
        elif v == "kish":                       # NaN where either agreement is undefined
            out[v] = as_channel(M).agreement("kish", 0.0) - as_channel(E).agreement("kish", 0.0)
        else:
            raise ValueError(f"unknown variant {v!r} (not in {VARIANTS})")
    return out


class ResidualMeans:
    """Per Gaussian: the sum, count and max of the finite per-view residuals of each name, over
    the views passed to add() (float64, on the residuals' device)."""

    def __init__(self, N, names=tuple(RESIDUALS.values())):
        self.N, self.names, self.n_views = int(N), tuple(names), 0
        self.sum, self.count, self.max = {}, {}, {}

    @torch.no_grad()
    def add(self, per_view):
        """per_view: {name: (N,) residual of one view, NaN where not observed}."""
        self.n_views += 1
        for name in self.names:
            r = per_view.get(name)
            if r is None:
                continue
            r = r.detach().double()
            ok = torch.isfinite(r)
            if name not in self.sum:
                self.sum[name] = torch.zeros(self.N, dtype=torch.float64, device=r.device)
                self.count[name] = torch.zeros(self.N, dtype=torch.long, device=r.device)
                self.max[name] = torch.full((self.N,), float("nan"), dtype=torch.float64,
                                            device=r.device)
            self.sum[name] += torch.where(ok, r, torch.zeros_like(r))
            self.count[name] += ok.long()
            self.max[name] = torch.where(ok, torch.fmax(self.max[name], r), self.max[name])

    def mean(self, name):
        """(N,) float64 CPU: the mean of the finite residuals, NaN without any."""
        if name not in self.sum:
            return torch.full((self.N,), float("nan"), dtype=torch.float64)
        c = self.count[name].cpu()
        return torch.where(c > 0, self.sum[name].cpu() / c.clamp_min(1).double(),
                           torch.full((self.N,), float("nan"), dtype=torch.float64))

    def scores(self):
        """{baseline row: (N,) mean residual} (RESIDUALS)."""
        return {row: self.mean(name) for row, name in RESIDUALS.items()}

    def identity(self, cb):
        """{name: True if the finite counts and maxima equal the ChangeBaselines series' count
        and max filter} (the residuals seen here are the ones its filters saw)."""
        out = {}
        for name in self.names:
            ts = cb.series.get(name)
            if ts is None or name not in self.sum:
                out[name] = ts is None and name not in self.sum
                continue
            mx = self.max[name].to(ts.max.dtype).cpu()
            same_max = bool(((mx == ts.max.cpu()) | (torch.isnan(mx) & torch.isnan(ts.max.cpu()))).all())
            out[name] = bool(torch.equal(self.count[name].cpu(), ts.count.cpu()) and same_max)
        return out
