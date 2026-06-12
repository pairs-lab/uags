"""
consolidation.py — uncertainty-driven consolidation vs. forgetting.

Ties the dynamics problem (SDD-SLAM thresholds; Chu & Kim's LLM-picks-stable-
objects) back to the uncertainty signal instead of heuristics:

  high n_eff & LOW  r_bar  -> persistently CONFLICTING observations across time
                              => the thing moved / is unstable => FORGET (decay).
  high n_eff & HIGH r_bar  -> stable, agreed-upon                => CONSOLIDATE.

Forgetting is exponential decay of the sufficient statistics (an Ebbinghaus-style
forgetting curve on the accumulated evidence). Consolidated Gaussians are
optionally frozen / promoted to a long-term map and define the "robust map" used
for localization, without any LLM in the loop.
"""
from __future__ import annotations
import torch


class Consolidator:
    def __init__(self, field, min_obs=5.0, r_unstable=0.5, r_stable=0.85,
                 forget_lambda=0.9):
        self.field = field
        self.min_obs = min_obs
        self.r_unstable = r_unstable
        self.r_stable = r_stable
        self.forget_lambda = forget_lambda

    @torch.no_grad()
    def step(self):
        """Apply one consolidation/forgetting step. Returns boolean masks."""
        f = self.field
        rbar = f.mean_resultant_length()
        enough = f.n_eff >= self.min_obs
        unstable = enough & (rbar <= self.r_unstable)     # dynamic / ambiguous
        stable = enough & (rbar >= self.r_stable)         # robust

        # forget the unstable ones: decay their evidence
        f.R[unstable] *= self.forget_lambda
        f.n_eff[unstable] *= self.forget_lambda
        return {"unstable": unstable, "stable": stable}

    @torch.no_grad()
    def robust_mask(self):
        """The set of Gaussians defining the robust (long-term) map."""
        rbar = self.field.mean_resultant_length()
        return (self.field.n_eff >= self.min_obs) & (rbar >= self.r_stable)
