"""
distill.py — optimization-based feature lifting baselines for Table I (re-implemented on our
geometry and features, so every row shares the 3D-GS map, the LSeg features, the PCA basis
and the evaluation):

(f) Feature 3DGS (Zhou et al., CVPR 2024, arXiv 2312.03203): per-Gaussian features f_n
    trained so that the alpha-composited feature image matches the 2D feature map, with its
    L1 feature loss,
        L = mean_p |F_p - z_p|,   F_p = sum_n w_{p,n} f_n                 (row "f3dgs")
    Adam, lr 1e-3 (the setting of both papers).
    Geometry stays fixed (every method shares one map); no speed-up decoder (the features are
    already 64-d after the shared PCA). No uncertainty of its own (the evaluator adds MSP).
(g) LEGaussians (Shi et al., CVPR 2024, arXiv 2311.18482), as its paper defines it, on the
    shared features: the mapping frames' features are quantized into a codebook S of N = 128
    unit vectors (spherical k-means; LEGaussians quantizes CLIP+DINO features, we quantize the
    shared LSeg PCA-64 features), each pixel's target is its nearest code M_p; every Gaussian
    holds an 8-d semantic code s_n and an uncertainty u_n in [0, 1] (initialized to 0, kept in
    range by projection), both alpha-composited (R_s, U); an MLP decodes R_s into a softmax
    over the N codes, trained with its Eq. 10-12 (lambda_CE = lambda_u = 1):
        L = mean_p CE(softmax(MLP(R_s,p)), M_p) (1 - U_p) + lambda_u mean_p U_p
    Adam, lr 1e-3 (its setting). At query time the decoded feature is softmax(MLP(R_s)) S
    (its query rule), a render-then-decode row (usplat.rows.Row.decoder); row "legs",
    channel "legs_u".
    Its adaptive smoothing MLP (Eq. 13-14) is not used.
Both are trained on random mapping views, on pixels whose rendered opacity is at least
alpha_min (the map's covered pixels). A Gaussian that no view renders keeps its initial value.

scripts/distill_features.py trains them; row_factory() below reads the files next to the
main field (<main>_f3dgs.pt, <main>_legs.pt) for scripts/eval_seg.py.
"""
from __future__ import annotations
import glob
import hashlib
import math
import os
import re
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as Fn

from .rows import TensorRow

METHODS = ("f3dgs", "legs")
LEGS_CODES, LEGS_DIM, LEGS_HIDDEN = 128, 8, 64


def basis_sha256(pca_state):
    """Identity of a PCA basis (a distilled field is only valid in the basis it was fit in)."""
    b = pca_state["basis"].detach().cpu().contiguous().float().numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


def spherical_kmeans(x, k, iters=30, seed=0):
    """(M, d) unit rows -> (k, d) unit centres maximizing the summed cosine to the nearest
    centre; an empty cluster is re-seeded with the point farthest from its centre."""
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(x.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        sim = x @ c.t()
        a = sim.argmax(1)
        new = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=k)
        worst = sim.max(1).values.argsort()
        j = 0
        for i in torch.nonzero(cnt == 0).flatten().tolist():
            new[i] = x[worst[j]]
            j += 1
        c = Fn.normalize(new, dim=-1)
    return c


class LEGDecoder(torch.nn.Module):
    """softmax(MLP(R_s)) over the codebook; decode() gives the query feature (probs @ S)."""

    def __init__(self, codebook, dim=LEGS_DIM, hidden=LEGS_HIDDEN):
        super().__init__()
        self.register_buffer("codebook", codebook.clone())
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden, codebook.shape[0]))

    def logits(self, r):
        return self.mlp(r)

    @torch.no_grad()
    def forward(self, r):                               # (P, dim) rendered code -> (P, d)
        return torch.softmax(self.mlp(r.to(self.codebook.dtype)), -1) @ self.codebook


def converged(history, window=5, tol=0.01):
    """Stopping rule: the mean loss of the last `window` log points improved by less than
    `tol` (relative) on the `window` points before them."""
    if len(history) < 2 * window:
        return False
    recent = sum(h[1] for h in history[-window:]) / window
    prev = sum(h[1] for h in history[-2 * window:-window]) / window
    return prev <= 0 or (prev - recent) / prev < tol


def train_features(views: Sequence, N: int, d: int, method="f3dgs", iters=30000, lr=1e-3,
                   lam_u=1.0, alpha_min=0.5, seed=0, device="cpu", n_codes=LEGS_CODES,
                   code_samples=2000, log_every=1000, min_iters=5000, window=5, tol=0.01,
                   log=print):
    """views: list of (rasterizer, target_fn): rasterizer(buffer (N, C)) -> (P, C), linear
    and differentiable in the buffer; target_fn() -> (P, d) unit feature map of that view.
    Trains for at most `iters` iterations (the papers' 30k) and stops earlier, after
    `min_iters`, once the loss improved by less than `tol` over the last window x log_every
    iterations (converged()). f3dgs: returns dict(features (N, d), history, iters_run).
    legs: dict(features = codes (N, 8), u (N,), decoder (LEGDecoder, cpu), history, iters_run)."""
    if method not in METHODS:
        raise ValueError(f"method {method!r} not in {METHODS}")
    g = torch.Generator().manual_seed(seed)
    ones = torch.ones(N, 1, device=device)
    history, run = [], []
    if method == "f3dgs":
        f = torch.zeros(N, d, device=device, requires_grad=True)
        params, dec, u = [f], None, None
    else:
        # codebook of the mapping views' features (LEGaussians' discrete feature space)
        samp = []
        for _, target in views:
            z = target()
            samp.append(z[torch.randperm(z.shape[0], generator=g)[:code_samples].to(z.device)])
        S = spherical_kmeans(torch.cat(samp).float(), n_codes, seed=seed)
        f = (0.01 * torch.randn(N, LEGS_DIM, generator=g)).to(device).requires_grad_(True)
        u = torch.zeros(N, device=device, requires_grad=True)          # "set to zero"
        dec = LEGDecoder(S).to(device)
        params = [f, u] + list(dec.mlp.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    for it in range(1, iters + 1):
        ras, target = views[int(torch.randint(len(views), (1,), generator=g))]
        cols = [f, ones] if u is None else [f, u[:, None], ones]
        img = ras(torch.cat(cols, dim=1))
        alpha = img[:, -1]
        cov = alpha >= alpha_min
        if not bool(cov.any()):
            continue
        z = target()
        if u is None:
            loss = (img[:, :d][cov] - z[cov]).abs().mean()                 # Feature 3DGS: L1
        else:
            idx = (z[cov].float() @ dec.codebook.t()).argmax(1)             # M_p: nearest code
            ce = Fn.cross_entropy(dec.logits(img[:, :LEGS_DIM][cov]), idx, reduction="none")
            U = img[:, LEGS_DIM][cov]
            loss = (ce * (1.0 - U)).mean() + lam_u * U.mean()               # its Eq. 10-12
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if u is not None:
            with torch.no_grad():
                u.clamp_(0.0, 1.0)                                          # u in [0, 1]
        run.append(loss.item())
        if it % log_every == 0 or it == iters:
            history.append((it, sum(run) / len(run)))
            log(f"[distill {method}] iter {it}/{iters} loss {history[-1][1]:.4f}")
            run = []
            if it >= min_iters and converged(history, window, tol):
                log(f"[distill {method}] converged at iter {it}")
                break
    out = dict(features=f.detach(), history=history, u=None, decoder=None, iters_run=it)
    if u is not None:
        out.update(u=u.detach(), decoder=dec.cpu().eval())
    return out


def variant_name(method, lam_u=None):
    return method


def out_path(main_path, name):
    stem = main_path[:-3] if main_path.endswith(".pt") else main_path
    return f"{stem}_{name}.pt"


def rows_from_file(path, device="cpu"):
    d = torch.load(path, map_location="cpu", weights_only=False)
    name = d["name"]
    unc, dec = {}, None
    if d.get("u") is not None:
        unc = {"legs_u": d["u"].to(device)}
        dec = LEGDecoder(d["decoder"]["codebook"], dim=d["features"].shape[1],
                         hidden=d.get("decoder_hidden", LEGS_HIDDEN))
        dec.load_state_dict(d["decoder"])
        dec = dec.to(device).eval()
    return TensorRow(name, "feature", buf=d["features"].to(device), unc=unc, decoder=dec), d


def row_factory(ckpt, field=None, device="cpu", class_ids=None, field_path=None):
    """scripts/eval_seg.py --row-factory usplat.distill:row_factory: the distilled rows of the
    files next to the main field. Each must be fit in the main field's PCA basis (checked);
    its identity is appended to ckpt["_aux_identities"] (eval_seg folds it into the scene's
    field identity, so validation and test must use the same distillation code/config)."""
    main = field_path or (ckpt.get("config") or {}).get("out")
    if not main:
        return []
    stem = main[:-3] if main.endswith(".pt") else main
    paths = sorted(p for m in METHODS for p in glob.glob(f"{glob.escape(stem)}_{m}.pt"))
    want = basis_sha256(ckpt["pca"]) if "pca" in ckpt else None
    rows = []
    for p in paths:
        row, d = rows_from_file(p, device)
        if want is not None and d.get("pca_basis_sha256") != want:
            raise ValueError(f"{p}: distilled in another PCA basis than {main}")
        if field is not None and row.buf.shape[0] != field.N:
            raise ValueError(f"{p}: {row.buf.shape[0]} Gaussians, the field has {field.N}")
        ckpt.setdefault("_aux_identities", []).append(d.get("identity", "unknown"))
        rows.append(row)
    return rows
