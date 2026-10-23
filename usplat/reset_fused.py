"""
reset_fused.py — let a fused change detector decide WHEN our field resets a Gaussian
(configs/reset_fused_prereg.json). Everything else of our update stays: the reset itself
(F5: the Gaussian's evidence is dropped and restarted from the current view), the depth
re-admission and the robust set. Only the firing decision changes.

Variants (fixed before any data):
    V1  feat + depth            (the feature residual fused with the depth residual)
    V2  ours + feat + depth     (our change score, the feature residual and the depth residual)
    V3  normlift + depth        (NormLift's reliability drop fused with the depth residual)
Components are the per-view test statistics of Table III's rows (scripts/eval_change_nor.py):
    ours      the field's persistent score c_p (min of the last k = 2 defined c) at this view
    feat      the min of the last k = 2 covered feature residuals against the frozen map
    depth     the same for the rendered-depth residual (Table III's "depth" row)
    normlift  R_ref - R_t, NormLift's reliability drop since the reference
Rank fusion with a mapping-pass scale. Table III's fused rows (usplat.change_baselines.
fuse_rank) rank against the update population, which has no per-view threshold. Here every
component value x is ranked against that component's OWN mapping-pass per-view null
(the finite per-view values of the mapping replay, pooled over views and Gaussians):
    rank_X(x) = mid-rank empirical CDF in [0, 1]  (NaN -> 0, the lowest rank, as fuse_rank)
    s_V = mean over V's components of rank_X        (equal weights, nothing fitted)
s_V is defined on a covered Gaussian where at least one component is finite (as fuse_rank:
NaN only when every input is NaN). Its threshold is the q = 0.999 empirical quantile of s_V
over the mapping replay's covered Gaussian-views (pooled; numpy's linear quantile in
float64, as eval_change_nor.Null.threshold), the rule
every Table III row uses. For V3 the mapping views are those of NormLift's own null segment
(the last L mapping keyframes), where its component exists.
Firing: covered, s_V defined, s_V > thr_V, and the Gaussian is not currently `changed`
(a Gaussian reset by the variant is not reset again before it is re-admitted: the frozen-map
residuals of a removed object stay high for ever, and re-firing would drop the new evidence
at every view).

ExternalResetField.external_reset(stats, fire, gauss_depth) applies the field's own reset
AFTER update_view, so that the state equals the one update_view gives when its own test
fires on exactly `fire` (the evidence of the current view becomes the new start). The field
runs with its own test switched off (thr = +inf in every history bucket after calibration),
so its own scores are still computed (V2 needs them). tests/test_reset_fused.py checks the
identity against the native reset bit for bit.
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch

from .vmf_field import DirectionalSemanticField, _EPS

VARIANTS: Dict[str, tuple] = {"V1": ("feat", "depth"),
                              "V2": ("ours", "feat", "depth"),
                              "V3": ("normlift", "depth")}
COMPONENTS = ("ours", "feat", "depth", "normlift")


class MappingRanks:
    """Per component, the sorted finite mapping-pass per-view values; rank() is the mid-rank
    empirical CDF in [0, 1] of a value among them (NaN -> 0; no samples -> 0)."""

    def __init__(self, samples: Dict[str, torch.Tensor]):
        self.sorted = {}
        for name, x in samples.items():
            x = x.detach().double().cpu().reshape(-1)
            self.sorted[name] = torch.sort(x[torch.isfinite(x)]).values

    def n(self, name):
        return int(self.sorted[name].numel())

    @torch.no_grad()
    def rank(self, name, x):
        """(N,) float64 on x's device; computed on the CPU (the null stays off the GPU)."""
        s = self.sorted[name]
        xd = x.detach().double().cpu()
        ok = torch.isfinite(xd)
        out = torch.zeros_like(xd)
        if s.numel() > 0 and bool(ok.any()):
            v = xd[ok].contiguous()
            lo = torch.searchsorted(s, v, right=False).double()
            hi = torch.searchsorted(s, v, right=True).double()
            out[ok] = (lo + 0.5 * (hi - lo)) / s.numel()
        return out.to(x.device)


@torch.no_grad()
def fused_stat(ranks: MappingRanks, comps: Dict[str, torch.Tensor], names: Sequence[str],
               mask=None):
    """(N,) float64 s_V = mean of the components' mapping ranks; NaN where every component is
    NaN or outside `mask` (bool (N,))."""
    xs = [comps[n] for n in names]
    r = torch.stack([ranks.rank(n, x) for n, x in zip(names, xs)], 1).mean(1)
    any_def = torch.stack([torch.isfinite(x) for x in xs], 1).any(1)
    keep = any_def if mask is None else any_def & mask.to(any_def.device)
    return torch.where(keep, r, torch.full_like(r, float("nan")))


def field_score(field):
    """(N,) the field's persistent change score of the LAST view (c_p where it was tested at
    that view, NaN elsewhere): update_view sets c_last = c_p exactly where last_tested."""
    return torch.where(field.last_tested, field.c_last, torch.full_like(field.c_last, float("nan")))


class ExternalResetField(DirectionalSemanticField):
    """The field with an externally decided reset (module docstring)."""

    @torch.no_grad()
    def disable_own_test(self):
        """After calibrate_change_threshold: thresholds +inf in every history bucket. The test
        still computes c (c_ring, c_last, c_max, last_tested, n_tests); it never fires, and
        no null is recorded (thr_by_hist is set)."""
        if self.thr_by_hist is None:
            raise RuntimeError("calibrate_change_threshold first (the field must be calibrated)")
        self.thr_by_hist = torch.full_like(self.thr_by_hist, float("inf"))
        self.cfg.change_thr = float("inf")

    @torch.no_grad()
    def external_reset(self, s, fire, gauss_depth=None):
        """Reset `fire` (bool (N,)) as _ingest does when its own test fires on these
        Gaussians at the view s (the ViewStats update_view just returned): the view's
        evidence becomes the new start of every reset statistic, the Gaussian is flagged
        changed (with the re-admission rule re-applied to it) and counted in n_resets /
        n_fires / last_fired. Call it right after update_view(s), before the next view."""
        cfg = self.cfg
        if not cfg.change_test:
            raise RuntimeError("external_reset needs change_test = True")
        fire = fire.to(self.device).bool()
        if bool((self.last_fired & ~fire).any()):
            raise RuntimeError("the field's own test fired: call disable_own_test()")
        hb = torch.bucketize(self.ns, self._hist_edges)        # as _ingest, before the reset
        self.last_fired = fire.clone()
        self.n_fires += torch.bincount(hb[fire], minlength=self.n_fires.numel())
        if not bool(fire.any()):
            return
        w, seen = self._weights(s.m, s.ref)
        u = s.R / s.m.clamp_min(_EPS)[:, None]
        wv, _ = self._change_weights(s.m, s)
        if cfg.centre:
            wc, _ = self._weights(s.mc, s.ref)
            uc = s.Rc / s.mc.clamp_min(_EPS)[:, None]
            wt, _ = self._change_weights(s.mc, s)
            ut = uc
        else:
            wt, ut = wv, u
        un2 = (ut * ut).sum(-1)
        w2 = wt * wt
        zero = lambda x: torch.where(fire, x, torch.zeros_like(x))
        zero2 = lambda x: torch.where(fire[:, None], x, torch.zeros_like(x))
        # posterior channels: reset, then the view (adding exact zeros elsewhere)
        self.raw.reset(fire)
        self.raw.add(zero(w), zero2(u))
        if cfg.centre:
            self.cen.reset(fire)
            self.cen.add(zero(wc), zero2(uc))
        self.views_since_reset[fire] = 0
        self.views_since_reset += zero(wv)
        # change-test buckets, exactly as _ingest's reset branch (+ its T2 / TQ update)
        self.c_ring[fire] = float("nan")
        self.Rs[fire] = 0; self.ns[fire] = 0
        self.Rf[fire] = wt[fire, None] * ut[fire]
        self.nf[fire] = wt[fire]
        self.fB2[fire] = w2[fire]; self.fC[fire] = w2[fire]
        self.fCq[fire] = (w2 * un2)[fire]; self.fBq2[fire] = (w2 * un2)[fire]
        self.T2[fire] = w2[fire]
        self.TQ[fire] = (w2 * un2)[fire]
        self.changed |= fire
        self.n_resets += fire.to(torch.int32)
        # depth residual since the reset: this view only
        self.depth_abs[fire] = 0; self.depth_w[fire] = 0
        if s.Ds is not None and gauss_depth is not None:
            wd, seen_d = self._view_weights(s.Dm, s.ref)
            res = s.Ds / s.Dm.clamp_min(_EPS) - gauss_depth.to(self.device, self.dtype)
            res = torch.where(seen_d, res, torch.zeros_like(res))
            self.depth_abs += zero(wd * res.abs())
            self.depth_w += zero(wd)
        # re-admission of the reset Gaussians, as at the end of _ingest
        enough = self.views_since_reset >= cfg.readmit_views
        if cfg.readmit_depth is None or not self.depth_seen:
            ok = enough
        else:
            dres = self.depth_abs / self.depth_w.clamp_min(_EPS)
            ok = enough & (self.depth_w > 0) & (dres <= cfg.readmit_depth)
        self.changed[fire] = ~ok[fire]
