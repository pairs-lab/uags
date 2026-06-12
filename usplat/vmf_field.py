"""
vmf_field.py — per-Gaussian directional (von Mises-Fisher) semantic field.

Continuous generalization of CSS's BKI-Dirichlet update:
    eta        = kappa0 * mu0 + kappa_lik * R          (natural parameter)
    R          = sum_i k_i z_i,   n_eff = sum_i k_i     (sufficient statistics)
    mu_post    = eta / ||eta||,   kappa_post = ||eta||  (posterior vMF)
    r_bar      = ||R|| / n_eff in [0,1]                 (data concentration)
    epistemic  ~ 1 / n_eff   (reducible)
    aleatoric  ~ 1 - r_bar   (irreducible; trust only once n_eff >= min_obs)
"""
from __future__ import annotations
import torch
import torch.nn.functional as Fn

from .bki_accumulate import bki_grad_accumulate


class DirectionalSemanticField:
    def __init__(self, num_gaussians, feat_dim, kappa_lik=10.0,
                 prior_mu=None, prior_kappa=0.0, device="cpu", dtype=torch.float32):
        self.N, self.d = int(num_gaussians), int(feat_dim)
        self.kappa_lik = float(kappa_lik)
        self.prior_kappa = float(prior_kappa)
        self.device, self.dtype = torch.device(device), dtype
        self.R = torch.zeros(self.N, self.d, device=self.device, dtype=dtype)
        self.n_eff = torch.zeros(self.N, device=self.device, dtype=dtype)
        if prior_mu is not None and prior_kappa > 0:
            pm = prior_mu.to(self.device, dtype)
            pm = pm.expand(self.N, self.d) if pm.dim() == 1 else pm
            self.prior_mu = Fn.normalize(pm, dim=-1)
        else:
            self.prior_mu = None

    # -------- updates -------------------------------------------------------
    @torch.no_grad()
    def update_sparse(self, gauss_ids, weights, z, normalize_z=True):
        """Update from explicit (gaussian_id, weight, clip_vector) triples."""
        gauss_ids = gauss_ids.to(self.device).long()
        weights = weights.to(self.device, self.dtype)
        z = z.to(self.device, self.dtype)
        if normalize_z:
            z = Fn.normalize(z, dim=-1)
        self.R.index_add_(0, gauss_ids, weights.unsqueeze(-1) * z)
        self.n_eff.index_add_(0, gauss_ids, weights)

    def update_from_rasterizer(self, rasterize_fn, z_obs):
        """Update from a differentiable feature rasterizer via the gradient trick.
        rasterize_fn: (buf:(N,d+1))->(P,d+1); z_obs: (P,d) per-pixel CLIP dirs.
        (Not no_grad: the accumulation IS an autograd backward pass.)"""
        with torch.enable_grad():
            R, n_eff = bki_grad_accumulate(rasterize_fn, z_obs, self.N,
                                           device=self.device, dtype=self.dtype)
        with torch.no_grad():
            self.R += R
            self.n_eff += n_eff

    # -------- posterior reads ----------------------------------------------
    def eta(self):
        e = self.kappa_lik * self.R
        if self.prior_mu is not None:
            e = e + self.prior_kappa * self.prior_mu
        return e

    def kappa_post(self):           return self.eta().norm(dim=-1)
    def mu_post(self):              return Fn.normalize(self.eta(), dim=-1)
    def angular_variance(self, eps=1e-8):
        return (self.d - 1) / (self.kappa_post() + eps)

    # -------- uncertainty split --------------------------------------------
    def mean_resultant_length(self, eps=1e-8):
        return self.R.norm(dim=-1) / (self.n_eff + eps)

    def r_bar_corrected(self, eps=1e-8):
        r = self.mean_resultant_length(eps)
        return (r - (1.0 - r) / (self.n_eff + 1.0)).clamp_(0.0, 1.0)

    def kappa_hat_banerjee(self, eps=1e-8):
        r = self.mean_resultant_length(eps).clamp(max=1.0 - 1e-4)
        return r * (self.d - r * r) / (1.0 - r * r + eps)

    def epistemic(self):            return 1.0 / (1.0 + self.n_eff)

    def aleatoric(self, min_obs=5.0):
        a = 1.0 - self.r_bar_corrected()
        valid = self.n_eff >= min_obs
        return torch.where(valid, a, torch.full_like(a, float("nan"))), valid

    # -------- query / housekeeping -----------------------------------------
    @torch.no_grad()
    def query(self, text_embeds):
        t = Fn.normalize(text_embeds.to(self.device, self.dtype), dim=-1)
        return self.mu_post() @ t.t()          # (N, K) cosine similarity

    @torch.no_grad()
    def reset(self):                self.R.zero_(); self.n_eff.zero_()
