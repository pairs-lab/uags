"""
evaluation.py — calibration metrics that make "uncertainty-aware" credible.

The headline experiment: for held-out views, compute per-element semantic error
and each uncertainty channel, then show
  - epistemic predicts reconstruction-improvable regions, and
  - aleatoric predicts dynamic / boundary / ambiguous regions,
via sparsification curves + AUSE, plus ECE for raw calibration.
"""
from __future__ import annotations
import torch


@torch.no_grad()
def sparsification_curve(errors, uncertainty, num_bins=20):
    e = torch.as_tensor(errors, dtype=torch.float32).flatten()
    u = torch.as_tensor(uncertainty, dtype=torch.float32).flatten()
    finite = torch.isfinite(e) & torch.isfinite(u)
    e, u = e[finite], u[finite]
    M = e.numel()
    e_model = e[torch.argsort(u, descending=True)]    # drop most-uncertain first
    e_oracle = e[torch.argsort(e, descending=True)]    # drop highest-error first
    fracs = torch.linspace(0, 1, num_bins + 1)[:-1]
    model = torch.tensor([e_model[int(f*M):].mean().item() if int(f*M) < M else 0.0
                          for f in fracs])
    oracle = torch.tensor([e_oracle[int(f*M):].mean().item() if int(f*M) < M else 0.0
                           for f in fracs])
    norm = e.mean().item() + 1e-12
    ause = torch.trapz((model - oracle) / norm, fracs).item()
    return dict(fractions=fracs, model_curve=model, oracle_curve=oracle,
                random_curve=torch.full_like(model, e.mean().item()), ause=ause)


@torch.no_grad()
def expected_calibration_error(confidence, correct, num_bins=10):
    """confidence in [0,1] (e.g. r_bar or softmax max), correct in {0,1}."""
    c = torch.as_tensor(confidence, dtype=torch.float32).flatten()
    y = torch.as_tensor(correct, dtype=torch.float32).flatten()
    bins = torch.linspace(0, 1, num_bins + 1)
    ece, M = 0.0, c.numel()
    for i in range(num_bins):
        m = (c > bins[i]) & (c <= bins[i+1]) if i > 0 else (c >= bins[i]) & (c <= bins[i+1])
        if m.any():
            ece += (m.float().mean() * (y[m].mean() - c[m].mean()).abs()).item()
    return ece


@torch.no_grad()
def semantic_error_vs_text(mu_post, gt_text_dirs):
    """Per-element semantic error = 1 - cos(mu_post, ground-truth text dir).
    mu_post, gt_text_dirs: (M, d) unit vectors. returns (M,)."""
    return 1.0 - (mu_post * gt_text_dirs).sum(-1)
