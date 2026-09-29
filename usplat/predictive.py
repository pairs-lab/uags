"""
predictive.py — per-pixel class probabilities from the field (the predictive, F6).

A novel view is rendered with the SAME splat weights as the update, m_p = sum_g w_{p,g} f_g,
for a per-Gaussian buffer f:
  P0 plug-in   f_g = mu*_g                         p_p(c) = softmax_c(beta m_p . t_c)
  P1 posterior f_g = E[mu_g] = A_d(kappa_post,g) mu*_g   (mean-field over the vMF
               posterior of each Gaussian: weakly supported Gaussians pull m_p toward 0,
               which flattens the softmax where the map knows little)
  P2 adaptive  beta_p = beta0 * exp(-a u_epi,p - b u_ale,p - g 1[u_ale undefined])
               with composited uncertainties; an undefined u_ale (too few views, or just
               reset by the change test) gets its own fitted term instead of counting as 0,
               the most confident value
The un-normalised render also lowers confidence where the map has little opacity.
Every row of a comparison fits its own temperature on a validation split only
(`fit_temperature`, `fit_adaptive`); texts are mapped into the field space without
renormalisation (StreamingPCA.project_text).
"""
from __future__ import annotations
import torch


def gaussian_buffer(field, mode="P0"):
    """(N, d) per-Gaussian feature to render for predictive `mode`."""
    if mode == "P0":
        return field.mu_post()
    if mode == "P1":
        return field.mean_direction()
    raise ValueError(mode)


def logits(m, text, beta=1.0):
    """m (P, d) rendered features, text (K, d) -> (P, K) logits beta * m . t.
    beta: scalar, or (P,) per-pixel (P2)."""
    s = m @ text.t()
    if torch.is_tensor(beta) and beta.dim() == 1:
        return beta[:, None] * s
    return beta * s


def adaptive_beta(beta0, a, b, u_epi, u_ale, g=0.0):
    """beta_p = beta0 * exp(-a u_epi - b u_ale - g * [u_ale undefined])."""
    undef = ~torch.isfinite(u_ale)
    ua = torch.where(undef, torch.zeros_like(u_ale), u_ale)
    return beta0 * torch.exp(-a * u_epi - b * ua - g * undef.to(u_epi.dtype))


def _nll(z, y):
    return torch.nn.functional.cross_entropy(z, y)


def fit_temperature(scores, labels, init=10.0, iters=200):
    """Fit beta in softmax(beta * scores) by NLL. scores (M, K) cosines, labels (M,)."""
    s = scores.detach().double()
    y = labels.long()
    logb = torch.tensor(float(torch.log(torch.tensor(init))), dtype=torch.float64,
                        requires_grad=True)
    opt = torch.optim.LBFGS([logb], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = _nll(torch.exp(logb) * s, y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(logb).detach())


def fit_adaptive(scores, labels, u_epi, u_ale, init_beta=10.0, iters=300):
    """Fit (beta0, a, b, g) of P2 by NLL on a validation split."""
    s = scores.detach().double()
    y = labels.long()
    ue = u_epi.detach().double()
    ua_raw = u_ale.detach().double()
    undef = (~torch.isfinite(ua_raw)).double()
    ua = torch.where(undef > 0, torch.zeros_like(ua_raw), ua_raw)
    p = torch.tensor([float(torch.log(torch.tensor(init_beta))), 0.0, 0.0, 0.0],
                     dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([p], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        beta = torch.exp(p[0] - p[1] * ue - p[2] * ua - p[3] * undef)
        loss = _nll(beta[:, None] * s, y)
        loss.backward()
        return loss
    opt.step(closure)
    q = p.detach()
    return dict(beta0=float(torch.exp(q[0])), a=float(q[1]), b=float(q[2]), g=float(q[3]))
