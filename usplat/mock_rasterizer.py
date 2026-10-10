"""
mock_rasterizer.py — a tiny, pure-PyTorch, differentiable feature rasterizer.

It is NOT meant to be fast or photorealistic. Its only jobs are:
  (1) let the full pipeline run end-to-end on CPU with no CUDA / gsplat, and
  (2) provide a ground-truth weight matrix W so we can numerically verify the
      gradient-trick accumulation (see usplat.bki_accumulate).

It exposes the SAME call signature the real adapter promises:
    rasterizer(feature_buffer: (N, k))  ->  feature_image: (P, k)
linear in `feature_buffer` (exactly as real alpha-compositing is).
"""
from __future__ import annotations
import math
import torch


class MockFeatureRasterizer:
    def __init__(self, means2d, sigmas, opacities, depths, pixels, dtype=torch.float32):
        """
        means2d   : (N, 2) screen-space Gaussian centers
        sigmas    : (N,)   isotropic screen-space std (mock covariance)
        opacities : (N,)   in (0, 1]
        depths    : (N,)   for front-to-back ordering
        pixels    : (P, 2) pixel coordinates to evaluate
        """
        self.means2d = means2d.to(dtype)
        self.sigmas = sigmas.to(dtype)
        self.opacities = opacities.to(dtype)
        self.depths = depths.to(dtype)
        self.pixels = pixels.to(dtype)
        self.N = means2d.shape[0]
        self.num_gaussians = self.N
        self.dtype = dtype
        self._W = None  # cached (P, N) blending weights

    @torch.no_grad()
    def weights(self) -> torch.Tensor:
        """Explicit (P, N) alpha-compositing weights w_{p,n}=T_{p,n}*alpha_{p,n}.
        Geometry-only (independent of any feature buffer) — used for verification."""
        if self._W is not None:
            return self._W
        P = self.pixels.shape[0]
        # per-pixel, per-gaussian alpha
        diff = self.pixels[:, None, :] - self.means2d[None, :, :]      # (P,N,2)
        d2 = (diff * diff).sum(-1)                                      # (P,N)
        alpha = self.opacities[None, :] * torch.exp(
            -0.5 * d2 / (self.sigmas[None, :] ** 2 + 1e-9))             # (P,N)
        alpha = alpha.clamp(0.0, 0.999)
        # front-to-back compositing in depth order
        order = torch.argsort(self.depths)                              # near->far
        W = torch.zeros(P, self.N, dtype=self.dtype)
        T = torch.ones(P, dtype=self.dtype)
        for n in order.tolist():
            a = alpha[:, n]
            W[:, n] = T * a
            T = T * (1.0 - a)
        self._W = W
        return W

    @torch.no_grad()
    def unoccluded_mass(self) -> torch.Tensor:
        """(N,) splat mass of each Gaussian without occlusion: the integral of its alpha over
        the image plane, o 2 pi sigma^2 (isotropic conic I / sigma^2, i.e. o 2 pi /
        sqrt(det conic), the formula of the gsplat adapter). Like the adapter it ignores the
        0.999 alpha clamp and the pixel grid's border (FieldConfig.change_weight="visibility")."""
        return self.opacities * (2.0 * math.pi) * (self.sigmas ** 2 + 1e-9)

    def __call__(self, feature_buffer: torch.Tensor) -> torch.Tensor:
        """Render (P, k) feature image = W @ feature_buffer. Differentiable w.r.t.
        feature_buffer; W is treated as constant (detached) — matching real
        feature rasterizers where colors enter the composite linearly."""
        W = self.weights().detach()                                     # (P,N) const
        return W @ feature_buffer                                       # (P,k)
