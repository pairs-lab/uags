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
  P1(c0)       P1 with the joint prior's pseudo-count c0 in place of FieldConfig.c0:
               A_d(kappa_hat) = ||R||/(W1 + c0), eta = kappa_hat R + kappa0 mu0 (posterior_mean);
               a read-out of the saved statistics, field.cfg is not touched
The class predictive per Gaussian, rendered as a mixture (probs rows, usplat.rows):
  P4 votes     the directional counterpart of CSS's Dirichlet vote counts. A pixel of
               Gaussian n is z ~ vMF(mu*_n, kappa_n), kappa_n = A_d^{-1}(rho_n),
               rho_n = ||R_n||/W1_n (raw channel: the pixel-level mean resultant length, which
               already holds within-view coherence and across-view agreement), clamped to
               [0, 1 - 1e-6]. Its vote fractions over the QUERIED texts t (K, d)
                   v_n(c) = P[argmax_k z . t_k = c]                       (vote_fractions)
               enter a Dirichlet posterior predictive with W1_n expected votes
                   q_n(c) = (alpha0 + W1_n v_n(c)) / (K alpha0 + W1_n)    (vote_predictive)
               (uniform where W1_n = 0). CSS counts the per-pixel argmax votes at mapping
               time over classes fixed then; P4 derives the votes at query time from the
               directional statistics, so it stays open-vocabulary.
The un-normalised render also lowers confidence where the map has little opacity.
Every row of a comparison fits its own temperature on a validation split only
(`fit_temperature`, `fit_adaptive`); texts are mapped into the field space without
renormalisation (StreamingPCA.project_text).
"""
from __future__ import annotations
import torch

from .vmf_math import a_d, a_d_inv, tangent_var


def gaussian_buffer(field, mode="P0", c0=None):
    """(N, d) per-Gaussian feature to render for predictive `mode`; c0 (optional) reads the
    posterior with that joint-prior pseudo-count instead of field.cfg.c0 (posterior_read_out)."""
    if mode not in ("P0", "P1"):
        raise ValueError(mode)
    if c0 is not None:
        mu, kp = posterior_read_out(field, c0)
        return mu if mode == "P0" else a_d(field.d, kp)[:, None] * mu
    if mode == "P0":
        return field.mu_post()
    return field.mean_direction()


def posterior_read_out(field, c0=None):
    """(mu* (N, d), kappa_post (N,)) of a DirectionalSemanticField with the joint prior's
    pseudo-count c0 (None: field.cfg.c0), op for op as field.mu_post / field.kappa_post:
        kappa_hat = A_d^{-1}(min(||R||/(W1 + c0), 1 - 1e-9))   (kappa_mode "map"; else kappa_lik)
        eta = kappa_hat R + kappa0 mu0,   kappa_post = ||eta||,   mu* = eta/||eta|| (0 if eta = 0)
    c0 only moves kappa_hat (and so mu* only through the text prior kappa0 mu0). Reads
    field.raw and field.prior_mu; field.cfg is not modified (no re-mapping, no copy)."""
    cfg = field.cfg
    c0 = cfg.c0 if c0 is None else float(c0)
    if cfg.kappa_mode == "map":
        r = field.raw.agreement("joint", c0).clamp(max=1 - 1e-9)
        kappa = a_d_inv(field.d, r)
    else:                                # kappa_lik: c0 has no effect
        kappa = torch.full((field.N,), cfg.kappa_lik, device=field.device, dtype=field.dtype)
    e = kappa[:, None] * field.raw.R
    if field.prior_mu is not None:
        e = e + cfg.prior_kappa * field.prior_mu
    n = e.norm(dim=-1, keepdim=True)
    mu = torch.where(n > 0, e / n.clamp_min(torch.finfo(e.dtype).tiny), torch.zeros_like(e))
    return mu, e.norm(dim=-1)


def posterior_mean(field, c0=None):
    """P1 at pseudo-count c0: E[mu] = A_d(kappa_post) mu*. c0 = field.cfg.c0 gives
    field.mean_direction() exactly."""
    return gaussian_buffer(field, "P1", field.cfg.c0 if c0 is None else c0)


# ---- P4: vote fractions of a vMF pixel model and their Dirichlet posterior predictive
RHO_MAX = 1.0 - 1e-6                 # rho clamp of the vote concentration (kappa finite)


def vote_kappa(field):
    """kappa_n = A_d^{-1}(rho_n), rho_n = ||R_n||/W1_n of the raw channel clamped to
    [0, 1 - 1e-6] (0 for an unseen Gaussian): the concentration of ONE pixel observation."""
    rho = field.mean_resultant_length("raw").clamp(0.0, RHO_MAX)
    return a_d_inv(field.d, rho)


def tangent_noise(d, n_samples, seed=0):
    """(S, d) float32 N(0, I) draws in antithetic pairs (eps_s, -eps_s), from a CPU generator
    with a fixed seed: the same samples for every Gaussian, every call and every device."""
    if n_samples < 2 or n_samples % 2:
        raise ValueError(f"n_samples must be even and >= 2 (antithetic pairs), got {n_samples}")
    g = torch.Generator().manual_seed(int(seed))
    e = torch.randn(n_samples // 2, d, generator=g, dtype=torch.float32)
    return torch.cat([e, -e])


@torch.no_grad()
def vote_fractions(mu, kappa, text, lam=1.0, n_samples=256, seed=0, chunk=4096):
    """v (N, K): v_nc = P[argmax_k z . t_k = c] for z ~ vMF(mu_n, kappa_n), by a deterministic
    tangent-Gaussian Monte Carlo with S = n_samples shared antithetic draws eps_s:
        z_ns = A_n mu_n + sigma_n (I - mu_n mu_n^T) eps_s,   A_n = A_d(kappa_n),
        sigma_n = lam sqrt(A_n/kappa_n)       (lam = 1 matches the vMF tangent variance,
                                               vmf_math.tangent_var; the radial part is fixed
                                               at its mean A_n: in d = 64 the vote fractions
                                               stay within ~0.02 of an exact vMF sampler,
                                               tests/test_vote_predictive.py, below the MC
                                               error ~0.03 of S = 256)
    With m_nk = mu_n . t_k, ET = E T^T (S, K) and e_ns = mu_n . eps_s,
        z_ns . t_k = A_n m_nk + sigma_n (ET_sk - e_ns m_nk),
    and since sigma_n > 0 the argmax is that of (A_n/sigma_n - e_ns) m_nk + ET_sk: one fused
    multiply-add per (n, s, k). mu: (N, d) unit rows (or 0); kappa (N,) >= 0; text (K, d) in
    the field space. float32 on mu's device, chunks of `chunk` Gaussians, so the result does
    not depend on the chunk size or the device's random stream.
    Cost for N = 1e6, K = 30, S = 256, d = 64: m and e take N d (K + S) = 1.8e10 MACs, the
    scores N S K = 7.7e9 FMAs (+ argmax); memory per chunk of 4096: scores (B, S, K) 126 MB,
    e and the argmax index 12 MB, the output (N, K) 120 MB. The run is memory bound (the scores
    are written and read once: ~61 GB of traffic), so on a 16 GB GPU it is estimated well
    under 1 s; on the CPU it measured 17 s (N = 2e5 scaled) on 4 threads. It runs once per lam
    (usplat.rows.VoteCache shares it across the alpha0 grid)."""
    if not lam > 0:
        raise ValueError(f"lam must be > 0 (got {lam}); lam = 0 is argmax_votes")
    N, d = mu.shape
    K = text.shape[0]
    dev, dt = mu.device, torch.float32
    mu = mu.to(dt)
    t = text.to(dev, dt)
    E = tangent_noise(d, n_samples, seed).to(dev)
    ET = E @ t.t()                                             # (S, K)
    kap = kappa.to(dev, torch.float64).reshape(-1)
    sigma = lam * tangent_var(d, kap).sqrt()                   # > 0 (1/sqrt(d) at kappa = 0)
    coef0 = (a_d(d, kap) / sigma).to(dt)                       # A_n / sigma_n
    out = torch.empty(N, K, device=dev, dtype=dt)
    for i in range(0, N, chunk):
        mc = mu[i:i + chunk]
        m = mc @ t.t()                                         # (B, K)
        coef = coef0[i:i + chunk, None] - mc @ E.t()           # (B, S): A/sigma - e_ns
        s = torch.addcmul(ET[None], coef[:, :, None], m[:, None, :])   # (B, S, K) = z.t/sigma
        idx = s.argmax(-1)                                     # (B, S)
        v = torch.zeros(mc.shape[0], K, device=dev, dtype=dt)
        v.scatter_add_(1, idx, torch.ones_like(idx, dtype=dt))  # integer counts: exact on GPU too
        out[i:i + chunk] = v / n_samples
    return out


@torch.no_grad()
def argmax_votes(mu, text):
    """The zero-noise limit (kappa -> inf) of vote_fractions: one-hot argmax_k mu . t_k, an
    exact tie split evenly (so the votes do not depend on the order of the texts)."""
    s = mu @ text.to(mu.device, mu.dtype).t()
    hit = (s == s.amax(-1, keepdim=True)).to(mu.dtype)
    return hit / hit.sum(-1, keepdim=True)


def vote_predictive(votes, mass, alpha0):
    """Dirichlet posterior predictive with `mass` expected votes split as `votes` (N, K):
    q_nc = (alpha0 + mass_n v_nc) / (K alpha0 + mass_n); uniform where mass_n = 0."""
    K = votes.shape[1]
    w = mass.to(votes.device, votes.dtype).reshape(-1, 1)
    q = (alpha0 + w * votes) / (K * alpha0 + w).clamp_min(torch.finfo(votes.dtype).tiny)
    return torch.where(w > 0, q, torch.full_like(q, 1.0 / K))


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
