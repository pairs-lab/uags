"""
smoothing.py — 3D spatial smoothing of the per-Gaussian statistics: the spatial kernel of
Bayesian Kernel Inference (BKI), as a READ-OUT of a saved field (no re-mapping, nothing
saved is changed). Rows for scripts/eval_seg.py --row-factory usplat.smoothing:row_factory;
the rules (candidates, selection on validation, reporting) are configs/smooth_prereg.json and
the driver is scripts/smooth_tab1.sh.

Our field pools evidence over the VIEWS of one Gaussian (the splat weight is the kernel inside
a view). BKI also pools over SPACE: an observation of a point x_m enters the posterior of x_n
with a kernel weight k(x_n, x_m). Here the observations of Gaussian m are its view means
u_{v,m} with view weights omega_{v,m}; Gaussian n receives them with weight k_nm omega_{v,m}:
    k_nm = exp(-||x_n - x_m||^2 / (2 sigma^2)) * gate_nm,   m in kNN_k(n),   k_nn = 1
    sigma = s * h,   h = median over the scene's Gaussians of the distance from a centre to
                     its nearest other centre (numpy median of the float32 kNN distances), so
                     the kernel scale follows the map's density, not metres
    gate_nm = 1 without a gate; with g: 1[cos(d_n, d_m) >= g], d = the unsmoothed direction
              (ours: mu*; averaging: its unit feature). A pair where either Gaussian has no
              evidence (W1 = 0, resp. S1 = 0) passes: no direction to disagree with, and an
              unseen neighbour adds nothing anyway.
The sufficient statistics of the field (Sec. III, R = S w u, W1 = S w, W2 = S w^2,
Q = S w^2 ||u||^2, Wr = S w ||u||) of view weights k_nm omega are then
    R'_n  = R_n  + S_m k_nm R_m      W1'_n = W1_n + S_m k_nm W1_m      Wr'_n = Wr_n + S_m k_nm Wr_m
    W2'_n = W2_n + S_m k_nm^2 W2_m   Q'_n  = Q_n  + S_m k_nm^2 Q_m
(first-order sums pool with k, second-order sums with k^2). Every read-out is the field's own
code on the pooled sums (a shallow copy of the field with the raw channel replaced):
    mu*' = R'/||R'||;  u_epi' = 1/(1 + support'), support' = min(W1'^2/W2', W1') = W1' (k omega <= 1)
    u_ale' = 1 - rho', rho'^2 = (||R'||^2 - Q')/(W1'^2 - W2') clamped, defined once support' >=
             tau_min (the field's estimator and threshold, unchanged)
So u_epi and u_ale of the smoothed rows are RECOMPUTED from the pooled sums: the pooled
support counts the neighbours' kernel-weighted views. The Kish agreement assumes independent
observations; neighbours seen in the same view are positively correlated, so u_ale' is biased
toward agreement and u_epi' toward certainty, as in any BKI kernel that treats neighbouring
observations as independent. The P2 temperature (beta0, gamma_e, gamma_a, gamma_u) is refit on
validation for every smoothed row, like every row.
Averaging (usplat.baselines.Averaging: F = S_v R_v, S1 = S_v m_v, pixel masses) is smoothed
with the SAME kernel: F' = F + S k F_m, S1' = S1 + S k S1_m, feature F'/(S1' + 1e-12)
normalised (Averaging.feature on the pooled sums); its gate uses its own unsmoothed feature.
Rows (one triple per SmoothSpec, tag = k<k>_s<s>[_g<g>]):
    ours_P0s_<tag>      kind feature, buffer mu*' (P0 on the pooled sums), channels u_epi, u_ale
    ours_P2s_<tag>      the same buffer, adaptive beta on (u_epi', u_ale') (P2)
    averaging_s_<tag>   kind feature, the smoothed averaging feature, no own uncertainty
Table I's AUSE_u column reads the row's own "u_ale" (ranking signal u_ale): for ours_P2s that
is the RECOMPUTED u_ale'. s = 0 (or a kernel whose weights underflow) gives exactly the
unsmoothed rows (tests/test_smoothing.py).

kNN (k_max = the largest k of the grid; the k nearest OTHER centres, sorted by distance, then
index): backend "kdtree" (default) is scipy.spatial.cKDTree on the CPU in float64: exact, no
GPU memory at all. Measured 2026-10-21 on this machine (24 cores) on a synthetic 1M-point
room surface: cKDTree build 0.6 s + query (k = 33) 1.1 s, peak RSS 1.1 GB; the whole knn()
call (self removal, sort, the cache's sha256) 3.9-5.1 s. The result is int32 indices and
float32 distances, 256 MB for N = 1e6, k = 32. Backend "torch" (chunked
torch.cdist(compute_mode="donot_use_mm_for_euclid_dist") + topk, CPU or CUDA) is the
brute-force reference (O(N^2): for tests, or a GPU run under scripts/gpu_gate.sh): its peak
device memory is the (B, N) float32 distance block + its topk + the points (12 N bytes),
B = budget // (4 N) rows per chunk (budget_mib, default 512 MiB: B = 134 for N = 1e6), so
it stays near budget_mib + 12 N bytes whatever N is; the outputs go to the CPU per chunk.
Pooling is chunked over Gaussians (chunk B: a gather of (B, k, d) float32 and one batched
matrix product; 8192 x 32 x 64: 67 MB), cast back to the field dtype. Measured on the same
synthetic 1M-Gaussian field (d = 64): pooling one kernel 1-7 s per method (k = 8 to 32, the
gate doubles it); freezing one batch (6 kernels x ours_P0s, ours_P2s, averaging_s) 41-45 s,
peak RSS 8.4 GB for the whole benchmark.
Memory per frozen candidate in eval_seg (N = 1e6, d = 64, float32): ours_P0s/ours_P2s share one
(N, 64) buffer (256 MB; eval_seg.frozen_on stores equal tensors once) plus u_epi', u_ale'
(8 MB); averaging_s another 256 MB. The pooled field of one candidate is computed once and
kept until the next candidate's (one-entry cache), the kNN once per scene and process
(two-entry cache keyed by the centres' sha256).
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .rows import Row, _field_channel

# ---- the candidate grid of configs/smooth_prereg.json (tests check that they agree)
GRID_S = (1.0, 2.0, 4.0, 8.0)   # sigma / h (s = 8 added 2026-10-21 by the lead, before any data)
GRID_K = (8, 16, 32)            # kNN truncation
GRID_G = (0.8, None)            # gate on the cosine of the unsmoothed directions; None = off
KNN_BACKENDS = ("kdtree", "torch")
W_MIN = 1e-12                   # kernel weights below this are 0 (pool: float32 underflow guard)


@dataclass(frozen=True)
class SmoothSpec:
    """One kernel: k nearest other centres, sigma = s * h, optional direction gate g."""
    k: int
    s: float
    g: Optional[float] = None

    def __post_init__(self):
        if int(self.k) != self.k or self.k < 1:
            raise ValueError(f"k must be a positive integer (got {self.k})")
        if not (math.isfinite(self.s) and self.s >= 0):
            raise ValueError(f"s must be finite and >= 0 (got {self.s})")
        if self.g is not None and not (-1.0 <= self.g <= 1.0):
            raise ValueError(f"g must be None or in [-1, 1] (got {self.g})")

    def tag(self) -> str:
        return f"k{self.k}_s{self.s:g}" + ("" if self.g is None else f"_g{self.g:g}")


# list order = tie order (configs/smooth_prereg.json): least smoothing first
GRID = tuple(SmoothSpec(k, s, g) for s in GRID_S for k in GRID_K for g in GRID_G)


def row_names(spec: SmoothSpec, prefix="ours", avg="averaging"):
    """(ours_P0s_<tag>, ours_P2s_<tag>, averaging_s_<tag>)."""
    t = spec.tag()
    return f"{prefix}_P0s_{t}", f"{prefix}_P2s_{t}", f"{avg}_s_{t}"


def batches(grid=GRID):
    """{"s<s>": [spec, ...]}: one eval_seg run per sigma (memory, scripts/smooth_tab1.sh)."""
    out: Dict[str, list] = OrderedDict()
    for sp in grid:
        out.setdefault(f"s{sp.s:g}", []).append(sp)
    return out


# ------------------------------------------------------------------------------- kNN
def _sorted_without_self(idx: np.ndarray, dist: np.ndarray, k: int):
    """Rows of k + 1 candidates -> the k nearest OTHER points: drop the point itself where it
    appears (else, with more than k + 1 duplicates at distance 0, the last column), then sort
    each row by (distance, index) so ties do not depend on the backend."""
    N = idx.shape[0]
    selfcol = idx == np.arange(N)[:, None]
    has = selfcol.any(1)
    drop = selfcol.copy()
    drop[~has, -1] = True
    first = drop.cumsum(1) == 1                       # drop exactly one column per row
    keep = ~(drop & first)
    idx = idx[keep].reshape(N, k)
    dist = dist[keep].reshape(N, k)
    o = np.lexsort((idx, dist), axis=1)
    return np.take_along_axis(idx, o, 1), np.take_along_axis(dist, o, 1)


def _knn_kdtree(P: np.ndarray, k: int, workers=-1):
    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    d, i = tree.query(P, k=k + 1, workers=workers)
    return i.astype(np.int64), d


@torch.no_grad()
def _knn_torch(P: np.ndarray, k: int, device="cpu", budget_mib=512):
    pts = torch.as_tensor(P, dtype=torch.float32, device=device)
    N = pts.shape[0]
    B = max(1, int(budget_mib * 2 ** 20 // (4 * N)))
    I = np.empty((N, k + 1), np.int64)
    D = np.empty((N, k + 1), np.float64)
    for a in range(0, N, B):
        dd = torch.cdist(pts[a:a + B], pts, compute_mode="donot_use_mm_for_euclid_dist")
        v, j = torch.topk(dd, k + 1, dim=1, largest=False, sorted=True)
        I[a:a + B] = j.cpu().numpy()
        D[a:a + B] = v.double().cpu().numpy()
        del dd
    return I, D


def knn(points, k: int, backend="kdtree", device="cpu", workers=-1, budget_mib=512):
    """(idx (N, k) int32, dist (N, k) float32) torch tensors on the CPU: the k nearest OTHER
    centres of every point, sorted by (distance, index). Needs N > k."""
    if backend not in KNN_BACKENDS:
        raise ValueError(f"kNN backend {backend!r} not in {KNN_BACKENDS}")
    P = torch.as_tensor(points).detach().cpu().double().numpy()
    N = P.shape[0]
    if N <= k:
        raise ValueError(f"kNN with k = {k} needs more than {k} points (got {N})")
    if backend == "kdtree":
        I, D = _knn_kdtree(P, k, workers)
    else:
        I, D = _knn_torch(P, k, device, budget_mib)
    I, D = _sorted_without_self(I, D, k)
    return torch.from_numpy(I.astype(np.int32)), torch.from_numpy(D.astype(np.float32))


def median_nn_distance(dist) -> float:
    """h: the median over points of the distance to the nearest other centre."""
    h = float(np.median(torch.as_tensor(dist)[:, 0].double().numpy()))
    if not (math.isfinite(h) and h > 0):
        raise ValueError(f"median nearest-neighbour distance h = {h}: half the centres or more "
                         f"coincide; sigma = s h is undefined")
    return h


# --------------------------------------------------------------------------- pooling
def _gate_pass(dirs, evidence, ic, rows, g):
    """(B, k) bool: cos(d_n, d_m) >= g (float32 batched dot products), or either side has no
    evidence."""
    dn = dirs[rows].float()
    dm = dirs[ic].float()                                       # (B, k, d)
    cos = torch.bmm(dm, dn[:, :, None])[:, :, 0]
    return (cos >= g) | ~evidence[rows][:, None] | ~evidence[ic]


@torch.no_grad()
def pool(first: Dict[str, torch.Tensor], second: Dict[str, torch.Tensor], idx, dist,
         spec: SmoothSpec, h: float, dirs=None, evidence=None, chunk=8192, stats=None):
    """Kernel-pooled per-Gaussian sums: x'_n = x_n + S_j w_nj x_{m_j} for `first` and
    x_n + S_j w_nj^2 x_{m_j} for `second`, over the first spec.k columns of idx/dist (the k
    nearest other centres), w = exp(-d^2/(2 (s h)^2)) times the gate (spec.g: needs dirs (N, d)
    unit or zero rows and evidence (N,) bool). Weights below W_MIN = 1e-12 (a neighbour beyond
    7.4 sigma) are set to 0: such a neighbour adds less than 1e-12 of its evidence (nothing at
    float32 precision), and its denormal products made the averaging feature's norm underflow
    to 0 (feature = inf) on 3 of 620,290 Gaussians of Replica office_0 (isolated, unseen
    Gaussians; validation attempt of 2026-10-21, before any number). s = 0: w = 0, the sums come back unchanged
    (bit for bit: the neighbour sum is exactly 0). The neighbour sum is a float32 batched
    matrix product (k <= 32 terms), added to x_n in float64; the result has each input's
    dtype and device.
    stats (a dict, optional) receives kernel_mass = mean_n S_j w_nj (neighbours pooled, self
    excluded) and gate_blocked = the share of the N k pairs the gate set to 0 (0 without g)."""
    k = spec.k
    if idx.shape[1] < k:
        raise ValueError(f"kNN has {idx.shape[1]} neighbours, the kernel needs {k}")
    if spec.g is not None and (dirs is None or evidence is None):
        raise ValueError("a gated kernel needs dirs and evidence")
    sigma = spec.s * h
    out = {n: torch.empty_like(x) for n, x in {**first, **second}.items()}
    N = idx.shape[0]
    mass = blocked = 0.0
    for a in range(0, N, chunk):
        rows = torch.arange(a, min(a + chunk, N))
        ic = idx[a:a + chunk, :k].long()
        dc = dist[a:a + chunk, :k].double()
        if sigma > 0:
            w = torch.exp(-dc * dc / (2.0 * sigma * sigma))
            w = torch.where(w >= W_MIN, w, torch.zeros_like(w))     # underflow guard
        else:
            w = torch.zeros_like(dc)
        if spec.g is not None:
            ok = _gate_pass(dirs, evidence, ic, rows, float(spec.g))
            blocked += float((~ok).sum())
            w = w * ok.double()
        mass += float(w.sum())
        for sums, ww in ((first, w), (second, w * w)):
            if not sums:
                continue
            wf = ww.float()[:, None, :]                             # (B, 1, k)
            for n, x in sums.items():
                G = x[ic].float()                                   # (B, k) or (B, k, d)
                G = G[:, :, None] if G.dim() == 2 else G
                s = torch.bmm(wf, G)[:, 0]                          # (B,) or (B, d)
                s = s[:, 0] if x.dim() == 1 else s
                out[n][a:a + chunk] = (x[a:a + chunk].double() + s.double()).to(x.dtype)
    if stats is not None:
        stats.update(kernel_mass=mass / max(N, 1), gate_blocked=blocked / max(N * k, 1))
    return out


def _check_field(field):
    """The smoothed read-outs replace the RAW channel only: refuse a field whose rows read
    other state (the centred channel as the dispersion channel, pixel-mass evidence)."""
    c = field.cfg
    if c.centre:
        raise ValueError("smoothing reads the raw channel; centre=True (centred dispersion "
                         "channel) is not supported")
    if c.evidence != "views" or c.epistemic == "pixels":
        raise ValueError(f"smoothing needs view evidence (evidence={c.evidence!r}, "
                         f"epistemic={c.epistemic!r})")


@torch.no_grad()
def smoothed_field(field, idx, dist, spec, h, chunk=8192, stats=None):
    """A shallow copy of the field whose raw channel holds the kernel-pooled sums; every
    read-out (mu_post, epistemic, aleatoric, ...) is the field's own code on them. The field
    itself is not modified."""
    from .vmf_field import _Channel
    _check_field(field)
    ch = field.raw
    dirs = ev = None
    if spec.g is not None:
        dirs, ev = field.mu_post(), ch.W1 > 0
    P = pool(dict(R=ch.R, W1=ch.W1, Wr=ch.Wr), dict(W2=ch.W2, Q=ch.Q), idx, dist, spec, h,
             dirs, ev, chunk, stats)
    new = _Channel.__new__(_Channel)
    for n, x in P.items():
        setattr(new, n, x)
    sf = copy.copy(field)
    sf.raw = new
    sf.view_hooks = []                   # never feed the copy (and never the original's hooks)
    return sf


@torch.no_grad()
def smoothed_averaging(avg, idx, dist, spec, h, chunk=8192, stats=None):
    """An Averaging accumulator (usplat.baselines) with the kernel-pooled F and S1; its
    feature() is the smoothed averaging feature. The gate uses avg's own unsmoothed feature."""
    from .baselines import Averaging
    dirs = ev = None
    if spec.g is not None:
        dirs, ev = avg.feature(), avg.S1 > 0
    P = pool(dict(F=avg.F, S1=avg.S1), {}, idx, dist, spec, h, dirs, ev, chunk, stats)
    out = Averaging.__new__(Averaging)
    out.N, out.d, out.dtype = avg.N, avg.d, avg.dtype
    out.F, out.S1 = P["F"], P["S1"]
    return out


# --------------------------------------------------------------------------- context
_KNN_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_KNN_CACHE_SIZE = 2                      # eval_seg loads each validation scene twice (fit, eval)


def _fingerprint(points, k, backend):
    p = torch.as_tensor(points).detach().cpu().float().contiguous().numpy()
    return f"{hashlib.sha256(p.tobytes()).hexdigest()}|{p.shape}|{k}|{backend}"


class SmoothingContext:
    """Per scene: the kNN of the centres (k_max neighbours, cached per process), h, and the
    pooled field / averaging of the last candidate asked for (one-entry caches: eval_seg
    freezes the rows of one candidate one after the other)."""

    def __init__(self, field, means, averaging=None, k_max=None, knn_backend="kdtree",
                 chunk=8192, verbose=True):
        self.field, self.averaging = field, averaging
        self.means = torch.as_tensor(means).detach().cpu()
        if self.means.shape != (field.N, 3):
            raise ValueError(f"centres {tuple(self.means.shape)} do not match the field's "
                             f"{field.N} Gaussians")
        if averaging is not None and averaging.N != field.N:
            raise ValueError(f"averaging covers {averaging.N} Gaussians, the field {field.N}")
        self.k_max = int(k_max or max(sp.k for sp in GRID))
        self.backend, self.chunk, self.verbose = knn_backend, int(chunk), verbose
        self._nbr = None
        self._h = None
        self._last_field = (None, None)
        self._last_avg = (None, None)
        self.knn_s = None
        self.log = []                    # one entry per pooled candidate (kernel mass, gate)

    def neighbours(self):
        if self._nbr is None:
            key = _fingerprint(self.means, self.k_max, self.backend)
            t0 = time.perf_counter()
            if key in _KNN_CACHE:
                _KNN_CACHE.move_to_end(key)
                self._nbr, cached = _KNN_CACHE[key], True
            else:
                self._nbr, cached = knn(self.means, self.k_max, self.backend), False
                _KNN_CACHE[key] = self._nbr
                while len(_KNN_CACHE) > _KNN_CACHE_SIZE:
                    _KNN_CACHE.popitem(last=False)
            self.knn_s = time.perf_counter() - t0
            self._h = median_nn_distance(self._nbr[1])
            if self.verbose:
                print(f"[smoothing] N {self.field.N}, kNN k {self.k_max} ({self.backend}"
                      f"{', cached' if cached else ''}) {self.knn_s:.1f} s, h (median NN "
                      f"distance) {self._h:.5g}", flush=True)
        return self._nbr

    @property
    def h(self):
        self.neighbours()
        return self._h

    def field_for(self, spec):
        if self._last_field[0] != spec:
            idx, dist = self.neighbours()
            self._last_field = (None, None)              # free the previous one first
            st, t0 = {}, time.perf_counter()
            sf = smoothed_field(self.field, idx, dist, spec, self.h, self.chunk, st)
            self._last_field = (spec, sf)
            self._log("ours", spec, st, time.perf_counter() - t0)
        return self._last_field[1]

    def averaging_for(self, spec):
        if self.averaging is None:
            raise ValueError("no averaging accumulator to smooth")
        if self._last_avg[0] != spec:
            idx, dist = self.neighbours()
            self._last_avg = (None, None)
            st, t0 = {}, time.perf_counter()
            sa = smoothed_averaging(self.averaging, idx, dist, spec, self.h, self.chunk, st)
            self._last_avg = (spec, sa)
            self._log("averaging", spec, st, time.perf_counter() - t0)
        return self._last_avg[1]

    def _log(self, what, spec, st, sec):
        self.log.append(dict(rows=what, spec=spec.tag(), seconds=sec, **st))
        if self.verbose:
            print(f"[smoothing] {what} {spec.tag()}: kernel mass {st['kernel_mass']:.3g} "
                  f"neighbours, gate blocked {100 * st['gate_blocked']:.1f}% of pairs, "
                  f"{sec:.1f} s", flush=True)


# ------------------------------------------------------------------------------ rows
@dataclass
class SmoothedFieldRow(Row):
    """ours_P0s / ours_P2s (adaptive=True): P0 on the kernel-pooled sums of spec, with the
    recomputed channels."""
    ctx: Optional[SmoothingContext] = None
    spec: Optional[SmoothSpec] = None
    channels: Sequence[str] = ("u_epi", "u_ale")

    def __post_init__(self):
        super().__post_init__()
        if self.ctx is None or self.spec is None:
            raise ValueError(f"row {self.name!r}: needs ctx and spec")
        if self.adaptive and any(c not in self.channels for c in self.p2_channels):
            raise ValueError(f"row {self.name!r}: P2 needs channels {self.p2_channels}")

    def buffer(self):
        from .predictive import gaussian_buffer
        return gaussian_buffer(self.ctx.field_for(self.spec), "P0")

    def uncertainties(self):
        sf = self.ctx.field_for(self.spec)
        return {c: _field_channel(sf, c) for c in self.channels}


@dataclass
class SmoothedAveragingRow(Row):
    """averaging_s: the averaging feature on the kernel-pooled F, S1 (no own uncertainty)."""
    ctx: Optional[SmoothingContext] = None
    spec: Optional[SmoothSpec] = None

    def __post_init__(self):
        super().__post_init__()
        if self.ctx is None or self.spec is None:
            raise ValueError(f"row {self.name!r}: needs ctx and spec")

    def buffer(self):
        return self.ctx.averaging_for(self.spec).feature()


def smoothed_rows(ctx, grid=GRID, prefix="ours", averaging=True):
    """Per spec, in grid order: ours_P0s_<tag>, ours_P2s_<tag>[, averaging_s_<tag>]."""
    rows = []
    for sp in grid:
        p0, p2, av = row_names(sp, prefix)
        rows.append(SmoothedFieldRow(p0, "feature", ctx=ctx, spec=sp))
        rows.append(SmoothedFieldRow(p2, "feature", adaptive=True, ctx=ctx, spec=sp))
        if averaging:
            rows.append(SmoothedAveragingRow(av, "feature", ctx=ctx, spec=sp))
    return rows


# ------------------------------------------------------------------------ row factory
def _sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def map_path(ckpt):
    """The 3DGS map the field was built on: ckpt["inputs"]["ply"]["path"] (fields saved since
    2026-10-18, with its sha256) or ckpt["config"]["ply"] (run_mapping.py --ply)."""
    entry = (ckpt.get("inputs") or {}).get("ply") or {}
    path = entry.get("path") or (ckpt.get("config") or {}).get("ply")
    if not path:
        raise KeyError("the field checkpoint names no 3DGS map (inputs.ply or config.ply)")
    if not os.path.exists(path):
        raise FileNotFoundError(f"3DGS map {path} of the field checkpoint is missing")
    return path, entry.get("sha256")


def identity(grid=GRID, knn_backend="kdtree"):
    """sha256 of what decides the smoothed rows besides the field and the map: the grid, the
    kNN backend and this file. eval_seg --betas refuses test scenes whose identity was not
    read on validation (check_field_identity, aux identities)."""
    with open(os.path.abspath(__file__), "rb") as f:
        src = hashlib.sha256(f.read()).hexdigest()
    key = dict(grid=[sp.tag() for sp in grid], knn=knn_backend, source=src)
    return "smoothing:" + hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def _averaging_from(ckpt, device="cpu"):
    """The first saved Averaging / NormLift accumulator (NormLift's F, S1 are the averaging
    sums), loaded alone: the other baselines (CSS counts) are not read."""
    from .baselines import load_baselines
    for e in ckpt.get("baselines") or []:
        if e.get("type") in ("Averaging", "NormLift"):
            return load_baselines([e], device=device)[0]
    return None


def row_factory(ckpt, field=None, device="cpu", grid=GRID, knn_backend="kdtree"):
    """scripts/eval_seg.py --row-factory usplat.smoothing:row_factory: the smoothed rows of
    every candidate of `grid` (ours_P0s_<tag>, ours_P2s_<tag>, averaging_s_<tag>), lazily (a
    row is computed only when eval_seg freezes it, so --rows picks a batch for free). The
    centres come from the field checkpoint's own map (checked against its sha256 when
    recorded, and against field.N); the averaging sums from ckpt["baselines"]. Use it next to
    usplat.baselines:row_factory (the unsmoothed averaging row)."""
    if field is None:
        raise ValueError("usplat.smoothing:row_factory needs the field")
    from .gs_io import load_gaussians
    path, sha = map_path(ckpt)
    if sha is not None and _sha256_file(path) != sha:
        raise ValueError(f"{path} changed since the field was built (sha256 differs)")
    means = load_gaussians(path, device="cpu")["means"]
    if means.shape[0] != field.N:
        raise ValueError(f"{path}: {means.shape[0]} Gaussians, the field has {field.N}")
    avg = _averaging_from(ckpt, device=field.device)
    if avg is None:
        raise KeyError("the field checkpoint holds no averaging sums (ckpt['baselines']): "
                       "averaging_s cannot be built")
    ctx = SmoothingContext(field, means, avg, k_max=max(sp.k for sp in grid),
                           knn_backend=knn_backend)
    ckpt.setdefault("_aux_identities", []).append(identity(grid, knn_backend))
    return smoothed_rows(ctx, grid)
