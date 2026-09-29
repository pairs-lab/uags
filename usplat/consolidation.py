"""
consolidation.py — which Gaussians form the robust (long-term) map.

The mechanism itself lives in DirectionalSemanticField._ingest, because the change test
must see the slow statistics BEFORE a new view is added:
  * the centred evidence is split into an older (slow) and a recent (fast) bucket;
    c = 1 - cos(slow, fast); c > threshold => the Gaussian's semantics changed:
    its evidence is reset (forget) and it is flagged `changed`;
  * a changed Gaussian is re-admitted only after new views AND a small observed-depth
    residual, so content that moved away cannot return with stale geometry;
  * robust = enough (Kish) views, high agreement on the centred channel, not changed.

This module exposes those masks, plus the OLD rule (decay R and n by lambda when r_bar is
low) for the ablation "temporal change test vs dispersion threshold". Note what the old
rule does: scaling R and n by the same lambda leaves r_bar and mu* unchanged; it only
lowers n until the Gaussian drops below min_obs.
"""
from __future__ import annotations
import torch


class Consolidator:
    def __init__(self, field, min_obs=None, r_unstable=None, r_stable=None,
                 forget_lambda=0.9, legacy=False):
        self.field = field
        self.min_obs = field.cfg.tau_min if min_obs is None else min_obs
        # legacy defaults are those of git ab28fb4 (0.5 / 0.85)
        self.r_unstable = (0.5 if legacy else 0.55) if r_unstable is None else r_unstable
        self.r_stable = ((0.85 if legacy else field.cfg.r_stable) if r_stable is None
                         else r_stable)
        self.forget_lambda = forget_lambda
        self.legacy = legacy

    @torch.no_grad()
    def step(self):
        """Legacy mode: one r_bar-threshold forgetting step. Otherwise a no-op that
        reports the masks (the change test already ran inside every view update)."""
        f = self.field
        if self.legacy:
            rbar = f.mean_resultant_length("raw")
            enough = f.raw.W1 >= self.min_obs
            unstable = enough & (rbar <= self.r_unstable)
            f.raw.scale(unstable, self.forget_lambda)
            if f.cfg.centre:
                f.cen.scale(unstable, self.forget_lambda)
            stable = enough & (rbar >= self.r_stable)
            return {"unstable": unstable, "stable": stable}
        return {"unstable": f.changed.clone(), "stable": self.robust_mask()}

    @torch.no_grad()
    def robust_mask(self):
        f = self.field
        if self.legacy:
            rbar = f.mean_resultant_length("raw")
            return (f.raw.W1 >= self.min_obs) & (rbar >= self.r_stable)
        return f.robust_mask()

    @torch.no_grad()
    def changed_mask(self):
        return self.field.changed.clone()
