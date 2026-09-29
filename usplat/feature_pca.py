"""
feature_pca.py — linear compression of CLIP directions for splat-side storage.

Rasterizing 512-D feature buffers is memory-hungry (image buffer is H*W*d).
Standard practice in language-embedded 3DGS is to compress (LangSplat: trained
autoencoder to 3-D; Feature-3DGS: 128-D distillation). We use a *linear* PCA
projection so that text queries can be mapped into the SAME compressed space at
query time — no decoder in the loop, and cosine similarity is approximately
preserved on the retained subspace.

The basis is estimated streaming from the first frames' dense features
(accumulating the second-moment matrix), then frozen. Projected vectors are
re-normalized to the unit sphere, so the vMF machinery applies unchanged in
the compressed space.
"""
from __future__ import annotations
import torch
import torch.nn.functional as Fn


class StreamingPCA:
    def __init__(self, in_dim, out_dim, device="cpu", dtype=torch.float32):
        self.in_dim, self.out_dim = int(in_dim), int(out_dim)
        self.device, self.dtype = torch.device(device), dtype
        self.M = torch.zeros(in_dim, in_dim, device=device, dtype=torch.float64)
        self.count = 0
        self.basis = None        # (in_dim, out_dim), frozen after fit()

    @torch.no_grad()
    def accumulate(self, z, max_samples=20000):
        """z: (..., in_dim) unit features. Subsamples rows for the moment matrix."""
        z = z.reshape(-1, self.in_dim)
        if z.shape[0] > max_samples:
            idx = torch.randperm(z.shape[0], device=z.device)[:max_samples]
            z = z[idx]
        z64 = z.to(self.device, torch.float64)
        self.M += z64.t() @ z64
        self.count += z.shape[0]

    @torch.no_grad()
    def fit(self):
        assert self.count > 0, "accumulate() some features first"
        # second moment (not centered): directions live on the sphere, and we
        # want the projection to preserve the directions themselves.
        evals, evecs = torch.linalg.eigh(self.M / self.count)
        order = torch.argsort(evals, descending=True)[: self.out_dim]
        self.basis = evecs[:, order].to(self.dtype)              # (D, k)
        explained = (evals[order].sum() / evals.sum()).item()
        return explained

    @torch.no_grad()
    def project(self, z, renormalize=True):
        """z: (..., in_dim) -> (..., out_dim), re-normalized to the sphere."""
        assert self.basis is not None, "call fit() first"
        out = z.to(self.basis.device, self.dtype) @ self.basis
        return Fn.normalize(out, dim=-1) if renormalize else out

    @torch.no_grad()
    def project_text(self, t):
        """Map unit text embeddings (..., in_dim) into the compressed space WITHOUT
        renormalising ("decode first"). For a unit field direction mu in the subspace,
        mu . (B^T t) = (B mu) . t is the cosine in the FULL space between the decoded
        feature and the text. Renormalising B^T t instead inflates texts that keep
        little energy in the image subspace and changes the argmax label."""
        assert self.basis is not None, "call fit() first"
        return t.to(self.basis.device, self.dtype) @ self.basis

    def state_dict(self):
        return dict(basis=None if self.basis is None else self.basis.cpu(),
                    in_dim=self.in_dim, out_dim=self.out_dim)

    @classmethod
    def from_state(cls, state, device="cpu"):
        p = cls(state["in_dim"], state["out_dim"], device=device)
        p.basis = state["basis"].to(device)
        return p


class RunningMean:
    """Mean feature of the mapping frames (the centre zbar of the centred channel)."""

    def __init__(self, dim, device="cpu"):
        self.s = torch.zeros(dim, device=device, dtype=torch.float64)
        self.n = 0

    @torch.no_grad()
    def add(self, z):
        z = z.reshape(-1, self.s.shape[0])
        self.s += z.to(self.s.device, torch.float64).sum(0)
        self.n += z.shape[0]

    def mean(self, dtype=torch.float32):
        assert self.n > 0, "add() some features first"
        return (self.s / self.n).to(dtype)
