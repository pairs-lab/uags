"""
gsplat_adapter.py — REAL feature rasterizer via gsplat.

  pip install gsplat            (needs CUDA toolkit + GPU)

Exposes the same callable interface as MockFeatureRasterizer:
    rasterizer(feature_buffer:(N, k)) -> feature_image:(P, k)
so usplat.bki_accumulate.bki_grad_accumulate works unchanged. The gradient trick
is valid here because gsplat composites `colors` linearly and is differentiable
w.r.t. them; geometry/opacity are detached so the blending weights are constant.

Large feature dims are rasterized in channel chunks (alpha-compositing is
channel-separable, so chunking is exact). The chunking is gsplat's own
(`rasterization(..., channel_chunk=...)`), so the Gaussians are projected ONCE per
view and only the per-pixel compositing is repeated per chunk.

The identity needs a rasterizer that is affine in the per-Gaussian buffer: plain
`colors` (sh_degree=None), no alpha normalisation, no decoder.

unoccluded_mass() gives the splat mass each Gaussian would have without occlusion, from
gsplat's projection meta (FieldConfig.change_weight="visibility"). GPU-UNVALIDATED: the meta
layout is read from gsplat 1.5.3 (rendering.py, ProjectionEWA3DGSPacked.cu) and the formula is
checked on the CPU with synthetic metas only (tests/test_change_coverage.py).
"""
from __future__ import annotations
import math
import torch

try:
    from gsplat import rasterization  # noqa: F401
    _HAS_GSPLAT = True
except Exception:
    _HAS_GSPLAT = False


def unoccluded_mass_from_meta(meta, num_gaussians, opacities=None):
    """(N,) unoccluded splat mass of every Gaussian from gsplat's projection meta
    (rasterization(...)[2]):
        m_unocc_n = o_n 2 pi / sqrt(det conic_n) = integral over the image plane of
                    o_n exp(-1/2 d^T conic_n d)
    conic = (a, b, c) is the inverse 2D covariance [[a, b], [b, c]] after gsplat's eps2d blur
    and meta["opacities"] the opacity after the antialiasing compensation: what the rasterizer
    composites (alpha = min(0.999, o exp(-sigma))). Not modelled: the 0.999 clamp, the 1/255
    cut-off and the image border (a Gaussian cut by the border gets m_v / m_unocc < 1, which is
    what "visibility" should say). Culled Gaussians get 0. Packed meta: gaussian_ids (nnz,),
    conics (nnz, 3), opacities (nnz,); unpacked (gaussian_ids None, one camera): conics
    (C, N, 3), opacities (C, N), radii (C, N) or (C, N, 2), radii > 0 = projected.
    GPU-UNVALIDATED (see the module docstring)."""
    conics = meta["conics"]
    op = meta.get("opacities")
    ids = meta.get("gaussian_ids")
    out = torch.zeros(num_gaussians, device=conics.device, dtype=conics.dtype)
    if ids is None:                                   # unpacked: first camera
        conics = conics.reshape(-1, num_gaussians, 3)[0]
        op = opacities if op is None else op.reshape(-1, num_gaussians)[0]
        valid = torch.ones(num_gaussians, dtype=torch.bool, device=conics.device)
        radii = meta.get("radii")
        if radii is not None:
            per_axis = radii.dim() >= 3 and radii.shape[-1] == 2   # gsplat >= 1.5: (C, N, 2)
            if per_axis:
                valid = (radii.reshape(-1, num_gaussians, 2)[0] > 0).all(-1)
            else:
                valid = radii.reshape(-1, num_gaussians)[0] > 0
        ids = torch.nonzero(valid).squeeze(1)
        conics, op = conics[ids], op.reshape(-1)[ids]
    elif op is None:
        op = opacities.reshape(-1)[ids]
    a, b, c = conics.unbind(-1)
    det = (a * c - b * b).clamp_min(torch.finfo(conics.dtype).tiny)
    out[ids.long()] = op.to(conics.dtype) * (2.0 * math.pi) / det.sqrt()
    return out


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

    def _rasterize(self, colors, render_mode="RGB", with_meta=False):
        from gsplat import rasterization
        render, alpha, meta = rasterization(
            means=self.means, quats=self.quats, scales=self.scales,
            opacities=self.opacities, colors=colors,
            viewmats=self.viewmat[None], Ks=self.K[None],
            width=self.width, height=self.height,
            near_plane=self.near_plane, far_plane=self.far_plane,
            render_mode=render_mode,           # "RGB": composite features like colors
            packed=True, channel_chunk=self.channel_chunk,
        )                                       # render: (1, H, W, k)
        if with_meta:
            return render, alpha, meta
        return render, alpha

    def __call__(self, feature_buffer: torch.Tensor) -> torch.Tensor:
        render, _ = self._rasterize(feature_buffer)
        return render.reshape(-1, feature_buffer.shape[-1])     # (P=H*W, k)

    @torch.no_grad()
    def render_image(self, colors):
        """Convenience: render an (H, W, k) image + (H, W) alpha from colors (N, k)."""
        render, alpha = self._rasterize(colors)
        return render[0], alpha[0, ..., 0]

    @torch.no_grad()
    def render_depth(self):
        """Expected depth (H, W) = sum_n w_{p,n} z_n / alpha_p (0 where alpha ~ 0)."""
        dummy = torch.zeros(self.num_gaussians, 1, device=self.means.device,
                            dtype=self.means.dtype)
        render, alpha = self._rasterize(dummy, render_mode="ED")
        return render[0, ..., -1], alpha[0, ..., 0]

    @torch.no_grad()
    def unoccluded_mass(self):
        """(N,) splat mass of every Gaussian without occlusion in this view (one extra
        projection with a 1-channel buffer; see unoccluded_mass_from_meta). GPU-UNVALIDATED."""
        dummy = torch.zeros(self.num_gaussians, 1, device=self.means.device,
                            dtype=self.means.dtype)
        _, _, meta = self._rasterize(dummy, with_meta=True)
        return unoccluded_mass_from_meta(meta, self.num_gaussians, self.opacities)

    @torch.no_grad()
    def gaussian_depths(self):
        """(N,) camera-frame depth of each Gaussian centre in this view."""
        R, t = self.viewmat[:3, :3], self.viewmat[:3, 3]
        return self.means @ R[2] + t[2]
