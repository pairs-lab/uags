"""
gsplat_adapter.py — REAL feature rasterizer via gsplat.

  pip install gsplat            (needs CUDA toolkit + GPU)

Exposes the same callable interface as MockFeatureRasterizer:
    rasterizer(feature_buffer:(N, k)) -> feature_image:(P, k)
so usplat.bki_accumulate.bki_grad_accumulate works unchanged. The gradient trick
is valid here because gsplat composites `colors` linearly and is differentiable
w.r.t. them; geometry/opacity are detached so the blending weights are constant.

Large feature dims are rasterized in channel chunks (alpha-compositing is
channel-separable, so chunking is exact); autograd accumulates gradients across
chunks transparently.
"""
from __future__ import annotations
import torch

try:
    from gsplat import rasterization  # noqa: F401
    _HAS_GSPLAT = True
except Exception:
    _HAS_GSPLAT = False


class GSplatFeatureRasterizer:
    def __init__(self, means, quats, scales, opacities, viewmat, K, width, height,
                 near_plane=0.01, far_plane=1e10, channel_chunk=32):
        if not _HAS_GSPLAT:
            raise ImportError("gsplat not installed. `pip install gsplat` (GPU).")
        # detach geometry: blending weights must be constant w.r.t. the feature buf
        self.means = means.detach()
        self.quats = quats.detach()
        self.scales = scales.detach()
        self.opacities = opacities.detach().reshape(-1)
        self.viewmat = viewmat.detach()        # (4,4) world-to-camera
        self.K = K.detach()                    # (3,3)
        self.width, self.height = int(width), int(height)
        self.near_plane, self.far_plane = near_plane, far_plane
        self.channel_chunk = int(channel_chunk)
        self.num_gaussians = means.shape[0]

    @classmethod
    def from_splats(cls, splats, viewmat, K, width, height, **kw):
        """splats: dict from usplat.gs_io.load_gaussians (post-activation)."""
        return cls(splats["means"], splats["quats"], splats["scales"],
                   splats["opacities"], viewmat, K, width, height, **kw)

    def _rasterize(self, colors):
        from gsplat import rasterization
        render, alpha, _meta = rasterization(
            means=self.means, quats=self.quats, scales=self.scales,
            opacities=self.opacities, colors=colors,
            viewmats=self.viewmat[None], Ks=self.K[None],
            width=self.width, height=self.height,
            near_plane=self.near_plane, far_plane=self.far_plane,
            render_mode="RGB",                 # composite features like colors
            packed=True,
        )                                       # render: (1, H, W, k)
        return render, alpha

    def __call__(self, feature_buffer: torch.Tensor) -> torch.Tensor:
        k = feature_buffer.shape[-1]
        outs = []
        for c0 in range(0, k, self.channel_chunk):
            chunk = feature_buffer[:, c0:c0 + self.channel_chunk]
            render, _ = self._rasterize(chunk)
            outs.append(render.reshape(-1, chunk.shape[-1]))   # (P, kc)
        return torch.cat(outs, dim=-1)                          # (P=H*W, k)

    @torch.no_grad()
    def render_image(self, colors):
        """Convenience: render an (H, W, k) image + (H, W) alpha from colors (N, k)."""
        render, alpha = self._rasterize(colors)
        return render[0], alpha[0, ..., 0]
