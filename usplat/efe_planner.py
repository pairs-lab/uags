"""
efe_planner.py — active inference next-best-view selection.

Expected Free Energy of a candidate viewpoint v:
    EFE(v) = - [ epistemic_value(v) + lambda * pragmatic_value(v) ]
We MINIMIZE EFE, i.e. maximize (epistemic + pragmatic):
    epistemic_value(v) = expected reduction of EPISTEMIC uncertainty made visible
                         from v   (chase reducible 1/n_eff only -- NOT aleatoric).
    pragmatic_value(v) = expected semantic relevance of what v reveals to a query.

This is the principled generalization of VISTA's hand-weighted
G = c*G_geometric + G_semantic: VISTA is the special case where epistemic value
is a view-diversity heuristic and aleatoric/epistemic are not separated. Here we
deliberately gate out aleatoric regions so the agent never wastes views on
genuinely-ambiguous / dynamic content.

`visibility(v)` is the geometry hook: a (N,) soft mask of how strongly each
Gaussian is expected to be seen from candidate v (predicted blending mass).
Provide it from a cheap forward splat / frustum+occlusion check.
"""
from __future__ import annotations
import torch
import torch.nn.functional as Fn


class EFEPlanner:
    def __init__(self, field, lam=1.0, aleatoric_min_obs=None):
        self.field = field
        self.lam = float(lam)
        self.aleatoric_min_obs = aleatoric_min_obs

    @torch.no_grad()
    def epistemic_value(self, visibility):
        """visibility: (V, N) per-candidate per-Gaussian expected blending mass.
        Returns (V,) expected reducible-uncertainty gain. Aleatoric-dominated
        Gaussians are down-weighted: more views won't help them."""
        epi = self.field.epistemic()                            # (N,) reducible
        ale, valid = self.field.aleatoric(self.aleatoric_min_obs)
        # where aleatoric is known & high, discount (irreducible). Unknown -> keep.
        discount = torch.ones_like(epi)
        known = valid
        discount[known] = (1.0 - ale[known]).clamp(0.05, 1.0)
        # content flagged as changed stays gated even though its evidence was reset
        # (a reset makes u_epi high and u_ale undefined, which would re-attract views)
        discount[self.field.changed] = 0.05
        gain_per_gauss = epi * discount                         # (N,)
        return visibility @ gain_per_gauss                      # (V,)

    @torch.no_grad()
    def pragmatic_value(self, visibility, query_embed):
        """Expected semantic relevance of revealed content to the task query.
        query_embed: (d,) unit text embedding."""
        sims = self.field.query(query_embed.unsqueeze(0)).squeeze(-1)  # (N,)
        sims = sims.clamp(min=0.0)
        return visibility @ sims                                # (V,)

    @torch.no_grad()
    def score(self, visibility, query_embed=None):
        """Returns (efe (V,), best_index). Lower EFE = better; we return argmin."""
        epi = self.epistemic_value(visibility)
        if query_embed is not None:
            prag = self.pragmatic_value(visibility, query_embed)
        else:
            prag = torch.zeros_like(epi)
        value = epi + self.lam * prag                           # maximize value
        efe = -value                                            # minimize EFE
        return efe, int(torch.argmin(efe).item())
