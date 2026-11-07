"""
baselines.py — per-Gaussian baselines for the segmentation and change tables (plan K11).

Every baseline is a training-free accumulator over the same splat weights w_{p,n} = T*alpha
as the field, and exposes its read-outs as Row objects (the shared interface of rows.py):
    name, kind ("feature" | "probs"), buffer() -> (N, d) | (N, K), uncertainties() -> {str: (N,)}
with higher uncertainty = less certain. Baselines that need only the per-view sums subscribe to
field.view_hooks and add no render or backward pass: for view v the hook gets
    R_{v,n} = sum_p w_{p,n} z_p ,   m_{v,n} = sum_p w_{p,n}          (ViewStats, raw channel)
CSS needs one-hot label targets and runs its own bki_grad_accumulate_groups call per view.

(a) Averaging = Gradient-Weighted Feature Back-Projection (Joseph, Amrutur, Bhatnagar,
    SIGGRAPH Asia 2025, doi:10.1145/3757377.3763926): the pixel-weighted mean feature
        f_n = sum_v R_{v,n} / (1e-12 + sum_v m_{v,n}),     row feature f_n/||f_n||
    The official code (create_feature_field_lseg) gets both sums from backward passes of
    sum_p render.feats and sum_p render; the 1e-12 and NaN -> 0 (unseen Gaussians) are
    copied from it. No uncertainty of its own (the evaluator adds MSP). Always the RAW
    channel, also when the field runs its centred channel (centre=True).
(b) NormLift (arXiv 2609.18898, Sec. 3.3), for Gaussian j and view v:
        W_j^(v) = sum_{i in v} A_ij = m_v ,  f_j^(v) = sum_{i in v} A_ij B_i / W_j^(v) = R_v/m_v  (Eq. 6)
        f_j = sum_v W f^(v) / sum_v W                 (the averaging feature, its Eq. 3)
        C_intra = sum_v W ||f^(v)|| / sum_v W ,  C_inter = ||f_j|| / C_intra                (Eq. 7)
        N_eff = (sum_v W)^2 / sum_v W^2 ,  R(j) = ||f_j|| N_eff / (N_eff + beta), beta = 1   (Eq. 8)
    Sufficient statistics: S1 = sum_v m_v, S2 = sum_v m_v^2, F = sum_v R_v and
    Sn = sum_v ||R_v|| (W ||f^(v)|| = ||R_v||). Unseen Gaussians get R = 0 and N_eff = 0 (the
    paper's zero-evidence case). The KNN mode-voting refinement (its Sec. 3.4) is not part of
    this row. Uncertainties: "normlift_1-R" = 1 - R(j), "normlift_epi" = 1/(1 + N_eff), and
    "normlift_1-Cinter" = 1 - C_inter (its cross-view consistency alone: the comparator of our
    aleatoric channel; like r_bar it is 1 at one view, so it is biased toward certainty at
    few views). These scores are built from NormLift's quantities; NormLift defines R only.
(c) Visible-view count: n_vis = #views with m_v > 0 and n_full = #views with m_v >= a0*H*W
    (the field's full-view mass, ViewStats.ref). Uncertainty 1/(1 + n); averaging feature.
(d) CSS, Continuous Semantic Splatting (Wilson et al., ICRA 2025,
    doi:10.1109/icra55743.2025.11128341, arXiv 2411.02547): a closed-set Dirichlet per
    Gaussian over K classes FIXED at mapping time. CSS receives, at mapping time, the
    classes PRESENT in the scene's evaluation frames (scripts/run_mapping.py css_vocabulary:
    the same class set every row is queried with, queries="present"), with their global ids
    `class_ids`. This is prior knowledge the open-vocabulary rows do not get at mapping time;
    the paper states it.
    Per pixel y_p = argmax_c z_p . t_c (t in the field space, StreamingPCA.project_text) and
        alpha_{n,c} = alpha0 + sum_v sum_p w_{p,n} 1[y_p = c]                     (CSS Eq. 12)
    with an uninformative prior alpha0 > 0. Pixels without a label (non-finite z, z = 0, or
    all K scores tied) are masked: argmax would turn them into class-0 votes (the field
    likewise gives z = 0 no direction).
    With S = sum_c alpha_c and pbar = alpha/S (the row buffer, kind "probs"):
        css_var_top     = pbar_c* (1 - pbar_c*)/(S + 1),  c* = argmax_c alpha_c
                          (CSS Eq. 10 at the most likely class: the per-Gaussian form of the
                          variance CSS reads out, its Eq. 17)
        css_var         = sum_c Var[theta_c] = (1 - sum_c pbar_c^2)/(S + 1)
                          (total variance: OUR extension of CSS Eq. 10, summed over classes)
        css_exp_entropy = E[H(theta)] = psi(S + 1) - sum_c pbar_c psi(alpha_c + 1)    (aleatoric)
        css_mi          = H(pbar) - E[H(theta)]                                      (epistemic)
    CSS's expectation measure (its Eq. 18, 1 - E[theta_c*]) is the evaluator's "msp" on the
    css row. CSS itself renders the pixel variance with SQUARED weights, sum_n w^2 Var_n (its
    Eq. 15), plus a background Dirichlet (Eq. 16). A linear rasterizer only gives
    sum_n w x_n, so these rows use the common composite sum_n w u_n / alpha instead.
    Query sets: rows(class_ids) keeps the columns of the queried global ids, in the query
    order (eval_seg's View.class_ids); seg_eval.class_scores renormalises them per pixel.
    The uncertainty channels stay over the full mapped vocabulary (for CSS the queried and
    the mapped classes are the same; css_full below reads them on the queried columns).
    "yager_conflict": per view, one subjective-logic opinion per Gaussian from its one-hot counts
    r_c (prior weight K): b_c = r_c/S_v, u = K/S_v, S_v = sum_c r_c + K. Views are fused in
    order with Yager's rule (Yager, Inf. Sci. 41(2):93-137, 1987,
    doi:10.1016/0020-0255(87)90007-7), which moves the conflict mass
        k = (sum_c b_c)(sum_c b'_c) - sum_c b_c b'_c
    to ignorance instead of renormalising it away (Dempster):
        b_c <- b_c b'_c + b_c u' + u b'_c ,   u <- u u' + k
    The row reports the accumulated conflict sum_v k_v. Evidential BKI for Gaussian maps:
    E2-BKI (Kim et al., RA-L 11(3) 2026, doi:10.1109/lra.2026.3653367).
    Full-vocabulary control "css_full" (full_vocab_css; scripts/run_mapping.py
    --css-full-vocab): a second CSS over EVERY class of --vocab (101 in
    configs/vocab_replica.json), fed the same pixels, so it is NOT told which classes the
    scene holds: y_p is the argmax over all K classes. Its rows "css_full" and
    "css_full_a<alpha0>" (CSS_ALPHA0_GRID) are queried like CSS: rows(class_ids) keeps the
    present columns and seg_eval.class_scores renormalises them per pixel, so the mass of a
    vote for an absent class drops out. Its Dirichlet channels are read out on the same
    columns (query_uncertainty): theta restricted to the queried classes and renormalised is
    exactly Dirichlet(alpha_cols) (aggregation and neutrality), the law whose mean is the
    renormalised predictive; over all K columns a Gaussian whose votes went to absent classes
    would get a flat present-class predictive and a confident u. With the vocabulary equal to
    the present classes the labels, counts, rows and channels are CSS's exactly. No Yager
    opinions (another N x K tensor). VALIDATION-ONLY control (CSS_FULL_RULE): no test number.
    Memory: counts N x K fp32 (404 MB for 1M Gaussians at K = 101), plus the one-hot pass,
    which holds an (N, K + 1) buffer and its gradient. Disk: storage="sparse" saves each
    (N, K) state tensor row-sparse and exact (sparse_rows: per Gaussian its int16 number of
    nonzeros, per nonzero an int16 column and the fp32 value), 2N + 6 nnz bytes instead of
    4NK, or dense when that is not smaller (compact_rows: above about 2K/3 nonzeros per
    Gaussian); nnz counts the (Gaussian, class) pairs that got a vote, at most the number of
    classes a Gaussian's pixels were labelled with.
(e) Ours-scalar (plan Tab III h): the field's P0 feature with the single scalar
    1 - A_d(kappa_post) (the exact posterior spread, rows.py channel "spread"), no
    epistemic/aleatoric split.

make_rows(field, baselines) returns the field's rows from rows.field_rows (P0: mu*, P1:
A_d(kappa_post) mu*, P2: P0 with the adaptive beta; each with u_epi and u_ale, NaN where u_ale
is undefined), then Ours-scalar, then every baseline's rows. Rows read their accumulator when
asked, so they can be built before the updates; seg_eval.freeze_rows snapshots them.

Checkpoints: default_baselines(field, text, class_ids) builds and attaches the K11 set
(NormLift with the averaging row, the view count on its feature, CSS; full_vocab_css builds
the control css_full); update_all runs one
keyframe for all of them; baselines_state(bs) is saved as ckpt["baselines"]; row_factory is
scripts/eval_seg.py --row-factory usplat.baselines:row_factory and returns the baseline rows
only (eval_seg builds the ours_* rows itself).
"""
from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import Callable, Optional, Sequence
import math
import torch
import torch.nn.functional as Fn

from .bki_accumulate import bki_grad_accumulate_groups
from .rows import Row, FieldRow, field_rows

_EPS = 1e-12
_GRADBP_EPS = 1e-12                     # gradbp_backproject.py:63, initial denominator


@dataclass
class LazyRow(Row):
    """A Row whose buffer and uncertainties are read from an accumulator when asked, so
    rows can be built before the updates (seg_eval.freeze_rows snapshots them)."""
    buffer_fn: Optional[Callable[[], torch.Tensor]] = dc_field(default=None, repr=False)
    uncertainties_fn: Optional[Callable[[], dict]] = dc_field(default=None, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.buffer_fn is None:
            raise ValueError(f"row {self.name!r}: LazyRow needs buffer_fn")

    def buffer(self) -> torch.Tensor:
        return self.buffer_fn()

    def uncertainties(self) -> dict:
        return {} if self.uncertainties_fn is None else dict(self.uncertainties_fn())


def _row(name, kind, buffer_fn, uncertainties_fn=None):
    return LazyRow(name, kind, buffer_fn=buffer_fn, uncertainties_fn=uncertainties_fn)


def _dtype_name(dt):
    return str(dt).replace("torch.", "")


def _load_tensors(obj, s):
    """Copy the _state tensors of s into obj (shapes must match: copy_ would broadcast)."""
    missing = [k for k in obj._state if k not in s]
    if missing:
        raise KeyError(f"{type(obj).__name__} state lacks {missing}")
    for k in obj._state:
        dst, src = getattr(obj, k), torch.as_tensor(s[k])
        if tuple(src.shape) != tuple(dst.shape):
            raise ValueError(f"{type(obj).__name__}.{k}: saved shape {tuple(src.shape)} != "
                             f"{tuple(dst.shape)}")
        dst.copy_(src)
    return obj


class _Accumulator:
    """View-hook accumulator: attach() subscribes on_view to field.view_hooks."""
    _state: tuple = ()

    def attach(self, field):
        field.view_hooks.append(self.on_view)
        return self

    def detach(self, field):
        field.view_hooks.remove(self.on_view)

    def state_dict(self):
        return {k: getattr(self, k).cpu() for k in self._state}

    def load_state_dict(self, s):
        return _load_tensors(self, s)


def _zeros(device, dtype):
    return lambda *s: torch.zeros(*s, device=device, dtype=dtype)


# ----------------------------------------------------------------------- (a) averaging
class Averaging(_Accumulator):
    """Gradient-Weighted Feature Back-Projection (training-free averaging)."""
    _state = ("F", "S1")

    def __init__(self, num_gaussians, feat_dim, device="cpu", dtype=torch.float32):
        self.N, self.d, self.dtype = int(num_gaussians), int(feat_dim), dtype
        z = _zeros(device, dtype)
        self.F, self.S1 = z(self.N, self.d), z(self.N)      # sum_v R_v, sum_v m_v

    @classmethod
    def for_field(cls, field, **kw):
        return cls(field.N, field.d, device=field.device, dtype=field.dtype, **kw).attach(field)

    def config(self):
        return dict(num_gaussians=self.N, feat_dim=self.d, dtype=_dtype_name(self.dtype))

    @torch.no_grad()
    def on_view(self, s):
        self.F += s.R                                       # raw channel, never s.Rc
        self.S1 += s.m

    @torch.no_grad()
    def feature(self):
        """f/||f|| as gradbp_backproject.py:166-169 (0 for unseen Gaussians)."""
        f = self.F / (self.S1 + _GRADBP_EPS)[:, None]
        f = f / f.norm(dim=-1, keepdim=True)
        return torch.where(torch.isnan(f), torch.zeros_like(f), f)

    def rows(self):
        return [_row("averaging", "feature", self.feature)]


# ------------------------------------------------------------------------ (b) NormLift
class NormLift(Averaging):
    """NormLift reliability (Sec. 3.3, Eq. 6-8). Its feature equals averaging's, so one
    NormLift accumulator can serve both rows: with_averaging=True (or rows(True))."""
    _state = ("F", "S1", "S2", "Sn")

    def __init__(self, num_gaussians, feat_dim, beta=1.0, device="cpu", dtype=torch.float32,
                 with_averaging=False):
        super().__init__(num_gaussians, feat_dim, device, dtype)
        self.beta = float(beta)
        self.with_averaging = bool(with_averaging)
        z = _zeros(device, dtype)
        self.S2, self.Sn = z(self.N), z(self.N)             # sum_v W^2, sum_v W ||f^(v)||

    def config(self):
        return dict(super().config(), beta=self.beta, with_averaging=self.with_averaging)

    @torch.no_grad()
    def on_view(self, s):
        super().on_view(s)
        self.S2 += s.m * s.m
        self.Sn += s.R.norm(dim=-1)                         # W^(v) ||f^(v)|| = ||R_v||

    @staticmethod
    def _div(a, b):
        """a/b, 0 where b = 0 (zero evidence)."""
        return torch.where(b > 0, a / torch.where(b > 0, b, torch.ones_like(b)),
                           torch.zeros_like(a))

    @torch.no_grad()
    def norm(self):
        """||f_j|| in [0, 1] (0 for unseen)."""
        return self._div(self.F.norm(dim=-1), self.S1)

    @torch.no_grad()
    def c_intra(self):
        return self._div(self.Sn, self.S1)

    @torch.no_grad()
    def c_inter(self):
        return self._div(self.F.norm(dim=-1), self.Sn)

    @torch.no_grad()
    def n_eff(self):
        """Kish effective views over the per-view weights W^(v) (0 for unseen)."""
        return self._div(self.S1 * self.S1, self.S2)

    @torch.no_grad()
    def reliability(self):
        ne = self.n_eff()
        return self.norm() * ne / (ne + self.beta)

    def uncertainties(self):
        return {"normlift_1-R": 1.0 - self.reliability(),
                "normlift_epi": 1.0 / (1.0 + self.n_eff()),
                "normlift_1-Cinter": 1.0 - self.c_inter()}

    def rows(self, with_averaging=None):
        avg = self.with_averaging if with_averaging is None else with_averaging
        rows = [_row("averaging", "feature", self.feature)] if avg else []
        return rows + [_row("normlift", "feature", self.feature, self.uncertainties)]


# ------------------------------------------------------- (b') NormLift + KNN mode voting
# NormLift as published (arXiv 2609.18898, Sec. 3.4, Eqs. 9-10): a one-pass, reliability-guided
# KNN mode voting on the lifted features, ported from the official code (kaust-violab/
# NormLift_release @ e0982e2, pipeline/knn_vote.py, MIT): every Gaussian i is a target; with the
# extended neighbourhood E(i) = {i} + its K nearest Gaussians (self excluded from the K),
#     S_ij = R(j) * sum_{k in E(i)} R(k) d_ik g_jk,  d_ik = exp(-||mu_i - mu_k||^2 / (2 sigma_d^2)),
#     g_jk = sigmoid((cos(f_j, f_k) - tau) / gamma),  d_ii = 1,
# j* = argmax over the K neighbours, and f_i <- f_{j*} (the unnormalized source feature) iff
# S_ij* > S_ii + delta. All reads come from the original features (simultaneous update), and
# R, the reliability of Eq. 8, is not changed. Hyperparameters of the official code; no tuning.
# Deviations: our 64-D field features (as every row); an exact scipy cKDTree KNN with ties
# ordered by (distance, index) and self removed explicitly (faiss IndexFlatL2 drops its first
# column instead); the Table I 2D-render evaluation instead of NormLift's 3D point labels.
NORMLIFT_VOTE = dict(k=36, sigma_d=0.3, tau=0.55, gamma=0.1, delta=0.05, chunk=4096)


def knn_indices(means, k, workers=8):
    """Exact k nearest neighbours of every point among the others: (idx (N, k) int64, squared
    distances (N, k) float32), each row sorted by (distance, index). Self is removed
    explicitly; a point with k or more exact duplicates loses its farthest candidate instead."""
    import numpy as np
    from scipy.spatial import cKDTree
    pts = means.detach().cpu().double().numpy()
    n = pts.shape[0]
    if n <= k:
        raise ValueError(f"knn_indices: {n} points for k = {k}")
    d, idx = cKDTree(pts).query(pts, k=k + 1, workers=workers)
    self_hit = idx == np.arange(n)[:, None]
    keep = ~self_hit
    keep[~self_hit.any(1), -1] = False                 # self not returned: drop the farthest
    idx, d = idx[keep].reshape(n, k), d[keep].reshape(n, k)
    order = np.lexsort((idx, d), axis=-1)              # by distance, then by index
    idx, d = np.take_along_axis(idx, order, -1), np.take_along_axis(d, order, -1)
    return torch.from_numpy(idx.astype(np.int64)), torch.from_numpy((d * d).astype(np.float32))


@torch.no_grad()
def knn_mode_vote(feats, rel, knn_idx, knn_dsq, sigma_d=0.3, tau=0.55, gamma=0.1, delta=0.05,
                  chunk=4096):
    """The chunk loop of the official knn_vote.py (Eqs. 9-10). feats (N, D) unnormalized lifted
    features (0 for unseen), rel (N,) reliability R, knn_idx / knn_dsq (N, K) from knn_indices.
    Returns (new_feats (N, D), replaced (N,) bool, source (N,) int64: the copied index or -1)."""
    N = feats.shape[0]
    feats = feats.float()
    feats_n = Fn.normalize(feats, dim=1, p=2)
    rel = rel.float()
    dw = torch.exp(-knn_dsq.float() / (2 * sigma_d ** 2))
    new = feats.clone()
    replaced = torch.zeros(N, dtype=torch.bool)
    source = torch.full((N,), -1, dtype=torch.long)
    for start in range(0, N, chunk):
        tgt = torch.arange(start, min(start + chunk, N))
        ext = torch.cat([tgt[:, None], knn_idx[tgt]], 1)                  # (M, K+1): self first
        f = feats_n[ext]                                                  # (M, K+1, D)
        r = rel[ext]                                                      # (M, K+1)
        w_d = torch.cat([torch.ones(len(tgt), 1, dtype=dw.dtype), dw[tgt]], 1)
        g = torch.sigmoid((torch.bmm(f, f.transpose(1, 2)) - tau) / gamma)  # g_jk (M, K+1, K+1)
        S = r * (g * (r * w_d)[:, None, :]).sum(-1)                        # S_ij (M, K+1)
        mx, am = S[:, 1:].max(1)
        src = ext.gather(1, (am + 1)[:, None]).squeeze(1)
        ok = mx > S[:, 0] + delta
        if bool(ok.any()):
            new[tgt[ok]] = feats[src[ok]]
            replaced[tgt[ok]] = True
            source[tgt[ok]] = src[ok]
    return new, replaced, source


class NormLiftVote:
    """The NormLift row with KNN mode voting: the voted lifted features (normalized for the row
    buffer, as averaging's and NormLift's) and NormLift's own uncertainties (R unchanged)."""

    def __init__(self, normlift, means, params=None):
        if means.shape[0] != normlift.N:
            raise ValueError(f"normlift_vote: {means.shape[0]} means for {normlift.N} Gaussians")
        self.nl, self.means = normlift, means
        self.params = dict(NORMLIFT_VOTE, **(params or {}))
        self._f, self.stats = None, None

    @torch.no_grad()
    def lifted(self):
        """f_j = F / S1 (0 where S1 = 0): NormLift's lifted feature (its Eq. 3), unnormalized."""
        return self.nl._div(self.nl.F, self.nl.S1[:, None].expand_as(self.nl.F))

    @torch.no_grad()
    def feature(self):
        if self._f is None:
            p = self.params
            idx, dsq = knn_indices(self.means, p["k"])
            new, rep, _ = knn_mode_vote(self.lifted().cpu(), self.nl.reliability().cpu(), idx, dsq,
                                        p["sigma_d"], p["tau"], p["gamma"], p["delta"], p["chunk"])
            f = new / new.norm(dim=-1, keepdim=True)
            f = torch.where(torch.isnan(f), torch.zeros_like(f), f)
            self._f = f.to(device=self.nl.F.device, dtype=self.nl.F.dtype)
            self.stats = dict(n=int(rep.numel()), replaced=int(rep.sum()),
                              share=float(rep.float().mean()), **{k: p[k] for k in p})
            print(f"normlift_vote: replaced {self.stats['replaced']} of {self.stats['n']} Gaussians "
                  f"({100 * self.stats['share']:.2f}%), K={p['k']} sigma_d={p['sigma_d']} "
                  f"tau={p['tau']} gamma={p['gamma']} delta={p['delta']}", flush=True)
        return self._f

    def rows(self):
        return [_row("normlift_vote", "feature", self.feature, self.nl.uncertainties)]


# ------------------------------------------------------------------- (c) view count
class ViewCount(_Accumulator):
    """Visible-view count n_vis (m > 0) and full-view count n_full (m >= a0*H*W)."""
    _state = ("n_vis", "n_full")

    def __init__(self, num_gaussians, feature: Optional[Callable[[], torch.Tensor]] = None,
                 device="cpu", dtype=torch.float32):
        self.N, self.dtype = int(num_gaussians), dtype
        z = _zeros(device, dtype)
        self.n_vis, self.n_full = z(self.N), z(self.N)
        self._feature = feature                # row buffer, e.g. Averaging.feature

    @classmethod
    def for_field(cls, field, feature=None, **kw):
        return cls(field.N, feature, device=field.device, dtype=field.dtype, **kw).attach(field)

    def config(self):
        return dict(num_gaussians=self.N, dtype=_dtype_name(self.dtype))

    @torch.no_grad()
    def on_view(self, s):
        self.n_vis += (s.m > 0).to(self.n_vis.dtype)
        self.n_full += (s.m >= s.ref).to(self.n_full.dtype)

    def uncertainties(self):
        return {"viewcount": 1.0 / (1.0 + self.n_vis),
                "viewcount_a0": 1.0 / (1.0 + self.n_full)}

    def rows(self):
        if self._feature is None:
            raise ValueError("ViewCount needs a feature source for its row "
                             "(e.g. ViewCount(N, feature=averaging.feature))")
        return [_row("viewcount", "feature", self._feature, self.uncertainties)]


# ------------------------------------------------------------------------ (d) CSS
def opinion_from_counts(r: torch.Tensor):
    """Subjective-logic opinion from evidence counts r (N, K), prior weight K:
    b = r/S, u = K/S, S = sum r + K."""
    K = r.shape[-1]
    S = r.sum(-1) + K
    return r / S[:, None], K / S


def yager_fuse(b1, u1, b2, u2):
    """Yager's rule on singleton + full-frame masses. Returns (b, u, conflict k)."""
    k = (b1.sum(-1) * b2.sum(-1) - (b1 * b2).sum(-1)).clamp_min(0.0)
    b = b1 * b2 + b1 * u2[:, None] + u1[:, None] * b2
    return b, u1 * u2 + k, k


DIRICHLET_KEYS = ("css_var_top", "css_var", "css_mi", "css_exp_entropy")


@torch.no_grad()
def dirichlet_uncertainties(alpha: torch.Tensor, chunk: int = 1 << 18):
    """(N, K) concentrations (all > 0) -> {DIRICHLET_KEYS} (N,) each.
    Evaluated in float64 (MI is a difference of two nearly equal terms at large S)."""
    if not bool((alpha > 0).all()):
        raise ValueError("Dirichlet concentrations must be > 0 (alpha0 > 0)")
    N = alpha.shape[0]
    out = {k: torch.empty(N, device=alpha.device, dtype=alpha.dtype) for k in DIRICHLET_KEYS}
    for i in range(0, N, chunk):
        a = alpha[i:i + chunk].double()
        S = a.sum(-1)
        p = a / S[:, None]
        top = p.amax(-1)                                    # pbar at c* = argmax alpha
        exp_ent = torch.digamma(S + 1) - (p * torch.digamma(a + 1)).sum(-1)
        ent_mean = -torch.special.xlogy(p, p).sum(-1)
        out["css_var_top"][i:i + chunk] = top * (1.0 - top) / (S + 1)
        out["css_var"][i:i + chunk] = (1.0 - (p * p).sum(-1)) / (S + 1)
        out["css_exp_entropy"][i:i + chunk] = exp_ent
        out["css_mi"][i:i + chunk] = (ent_mean - exp_ent).clamp_min(0.0)
    return out


CSS_STORAGE = ("dense", "sparse")
_ROW_SPARSE = "row_sparse_v1"
_INT16_MAX = 32767


@torch.no_grad()
def sparse_rows(x: torch.Tensor):
    """(N, K) -> exact row-sparse form: nnz (N,) int16 nonzeros per row, col (nnz,) int16 and
    val (nnz,) (x's dtype) in row-major order. 2N + (2 + itemsize) nnz bytes; dense_rows
    inverts it bit for bit (only +0.0 entries are dropped)."""
    if x.dim() != 2 or x.shape[1] > _INT16_MAX:
        raise ValueError(f"sparse_rows needs (N, K <= {_INT16_MAX}), got {tuple(x.shape)}")
    nz = (x != 0) | torch.signbit(x)                        # NaN and -0.0 are kept as values
    r, c = nz.nonzero(as_tuple=True)
    return dict(format=_ROW_SPARSE, shape=[int(s) for s in x.shape],
                nnz=nz.sum(1).to(torch.int16), col=c.to(torch.int16), val=x[r, c].clone())


def is_sparse_rows(s):
    return isinstance(s, dict) and s.get("format") == _ROW_SPARSE


def sparse_nbytes(s):
    return s["nnz"].nbytes + s["col"].nbytes + s["val"].nbytes


@torch.no_grad()
def compact_rows(x: torch.Tensor):
    """sparse_rows(x) if it is smaller than x, 2N + (2 + itemsize) nnz < itemsize N K bytes
    (fp32: below 2K/3 nonzeros per row), else x itself (dense); load_state_dict takes both."""
    s = sparse_rows(x)
    return s if sparse_nbytes(s) < x.numel() * x.element_size() else x


@torch.no_grad()
def dense_rows(s, device="cpu"):
    """Inverse of sparse_rows."""
    N, K = (int(v) for v in s["shape"])
    nnz, col, val = s["nnz"].long(), s["col"].long(), s["val"]
    if nnz.numel() != N or int(nnz.sum()) != col.numel() or col.numel() != val.numel():
        raise ValueError(f"corrupt row-sparse state: {nnz.numel()} rows for N = {N}, "
                         f"{int(nnz.sum())} nonzeros, {col.numel()} columns, {val.numel()} values")
    x = torch.zeros(N, K, dtype=val.dtype)
    x[torch.repeat_interleave(torch.arange(N), nnz), col] = val
    return x.to(device)


class CSS:
    """Closed-set Dirichlet per Gaussian (CSS) with a Yager-conflict read-out."""
    _state = ("counts", "yb", "yu", "yk")

    def __init__(self, num_gaussians, text: torch.Tensor, alpha0=1.0, yager=True,
                 device="cpu", dtype=torch.float32, class_ids: Optional[Sequence[int]] = None,
                 name="css", storage="dense", query_uncertainty=False):
        """text: (K, d) class embeddings in the field space (StreamingPCA.project_text) of the
        frozen vocabulary. class_ids: the K global class ids of the text rows (vocabulary
        ids; rows(class_ids) selects columns by them). alpha0 > 0: uninformative Dirichlet
        prior (1 = flat; the default, not tuned at mapping time: scripts/eval_seg.py also scores
        the read-outs of CSS_ALPHA0_GRID and the lowest validation NLL picks the Table I row).
        name: the row name ("css"; the read-outs are "<name>_a<alpha0>"). storage: how
        state_dict saves the (N, K) tensors, "dense" or "sparse" (compact_rows, exact).
        query_uncertainty: rows(class_ids) reads the Dirichlet channels out on the queried
        columns, Dirichlet(alpha[:, cols]) (css_full), instead of on all K."""
        if storage not in CSS_STORAGE:
            raise ValueError(f"storage {storage!r} not in {CSS_STORAGE}")
        if storage == "sparse" and int(text.shape[0]) > _INT16_MAX:
            raise ValueError(f"storage 'sparse' keeps int16 columns: K <= {_INT16_MAX}")
        self.name, self.storage = str(name), storage
        self.query_uncertainty = bool(query_uncertainty)
        self.N, self.K = int(num_gaussians), int(text.shape[0])
        self.device, self.dtype = torch.device(device), dtype
        self.text = text.to(self.device, dtype)
        self.alpha0 = float(alpha0)
        if not self.alpha0 > 0:                              # also rejects NaN
            raise ValueError(f"alpha0 must be > 0 (got {alpha0}): an unseen Gaussian's "
                             "Dirichlet would be improper and its uncertainties NaN")
        self.class_ids = None if class_ids is None else [int(c) for c in class_ids]
        if self.class_ids is not None:
            if len(self.class_ids) != self.K:
                raise ValueError(f"{len(self.class_ids)} class ids for {self.K} texts")
            if len(set(self.class_ids)) != self.K:
                raise ValueError(f"duplicate class ids {self.class_ids}")
        z = _zeros(self.device, dtype)
        self.counts = z(self.N, self.K)                      # sum_p w 1[y_p = c]
        self.yager = bool(yager)
        if self.yager:
            self.yb, self.yu, self.yk = z(self.N, self.K), z(self.N) + 1.0, z(self.N)
        else:
            self._state = ("counts",)

    @classmethod
    def for_field(cls, field, text, **kw):
        return cls(field.N, text, device=field.device, dtype=field.dtype, **kw)

    def config(self):
        cfg = dict(num_gaussians=self.N, text=self.text.cpu(), alpha0=self.alpha0,
                   yager=self.yager, class_ids=self.class_ids, dtype=_dtype_name(self.dtype))
        if self.name != "css":                       # a default CSS saves the v1 config
            cfg["name"] = self.name
        if self.storage != "dense":
            cfg["storage"] = self.storage
        if self.query_uncertainty:
            cfg["query_uncertainty"] = True
        return cfg

    # ------------------------------------------------------------ updates
    @torch.no_grad()
    def labels_and_valid(self, z):
        """y_p = argmax_c z_p . t_c, and the pixels that carry a direction: finite z, z != 0
        and not all K scores tied (argmax would send those to class 0)."""
        z = z.to(self.device, self.dtype)
        ok = torch.isfinite(z).all(-1)
        z = torch.where(ok[:, None], z, torch.zeros_like(z))
        s = z @ self.text.t()
        ok = ok & (z.norm(dim=-1) > 0)
        if self.K > 1:
            ok = ok & (s.amax(-1) > s.amin(-1))
        return s.argmax(-1), ok

    def labels(self, z):
        """y_p = argmax_c z_p . t_c (masked pixels: see labels_and_valid)."""
        return self.labels_and_valid(z)[0]

    def update_view(self, rasterize_fn, z_obs, pixel_valid=None):
        """One render + backward with K one-hot channels: C_v = sum_p w a_p onehot(y_p),
        a_p = pixel_valid * (pixel carries a direction)."""
        if torch.is_inference_mode_enabled():
            raise RuntimeError("CSS.update_view runs a backward pass; call it outside "
                               "torch.inference_mode()")
        y, ok = self.labels_and_valid(z_obs)
        a = ok.to(self.dtype)
        if pixel_valid is not None:
            a = a * pixel_valid.to(self.device, self.dtype).reshape(-1)
        onehot = Fn.one_hot(y, self.K).to(self.dtype)
        with torch.enable_grad():
            (C_v, _m), = bki_grad_accumulate_groups(rasterize_fn, [(onehot, a)], self.N,
                                                    device=self.device, dtype=self.dtype)
        self.update_counts(C_v)
        return C_v

    @torch.no_grad()
    def update_sparse(self, gauss_ids, weights, z, normalize_z=True):
        """One view from explicit (gaussian_id, weight, feature) triples."""
        z = z.to(self.device, self.dtype)
        if normalize_z:
            z = Fn.normalize(z, dim=-1)
        y, ok = self.labels_and_valid(z)
        onehot = Fn.one_hot(y, self.K).to(self.dtype)
        w = weights.to(self.device, self.dtype) * ok.to(self.dtype)
        C_v = torch.zeros(self.N, self.K, device=self.device, dtype=self.dtype)
        C_v.index_add_(0, gauss_ids.to(self.device).long(), w[:, None] * onehot)
        self.update_counts(C_v)
        return C_v

    @torch.no_grad()
    def update_counts(self, C_v):
        """Add one view's (N, K) one-hot counts (Dirichlet and Yager)."""
        self.counts += C_v
        if self.yager:
            b2, u2 = opinion_from_counts(C_v)
            self.yb, self.yu, k = yager_fuse(self.yb, self.yu, b2, u2)
            self.yk += k

    # ----------------------------------------------------------- read-outs
    def alpha(self):
        return self.counts + self.alpha0

    @torch.no_grad()
    def mean(self):
        """Dirichlet mean alpha/S (N, K): the row buffer."""
        a = self.alpha()
        return a / a.sum(-1, keepdim=True).clamp_min(_EPS)

    def yager_conflict(self):
        if not self.yager:
            raise ValueError("CSS(yager=False) keeps no opinions")
        return self.yk

    def uncertainties(self, cols=None):
        """Dirichlet channels over all K classes, or with cols over Dirichlet(alpha[:, cols]):
        the exact law of theta restricted to those classes and renormalised. The channels are
        symmetric in the classes, so cols are read in mapping order (all K in any order give
        the K-class values bit for bit)."""
        a = self.alpha()
        out = dirichlet_uncertainties(a if cols is None else a[:, torch.sort(cols).values])
        if self.yager:
            out["yager_conflict"] = self.yk.clone()
        return out

    def columns(self, class_ids):
        """Column index of each queried global class id, in the query order."""
        if self.class_ids is None:
            raise ValueError("CSS was built without class_ids: pass CSS(..., class_ids=...) "
                             "(the vocabulary ids of the text rows) to select query columns")
        ids = [int(c) for c in class_ids]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate queried class ids {ids}")
        pos = {c: j for j, c in enumerate(self.class_ids)}
        miss = [c for c in ids if c not in pos]
        if miss:
            raise ValueError(f"queried class ids {miss} are not in the CSS vocabulary "
                             f"({self.K} classes fixed at mapping time)")
        return torch.tensor([pos[c] for c in ids], device=self.device, dtype=torch.long)

    def rows(self, class_ids=None, name=None):
        """class_ids=None: all K columns in mapping order. Else the columns of the queried
        global ids in the query order (View.class_ids); the evaluator renormalises them.
        name: the row name (default self.name). query_uncertainty: the Dirichlet channels on
        the queried columns too."""
        name = self.name if name is None else name
        if class_ids is None:
            return [_row(name, "probs", self.mean, self.uncertainties)]
        cols = self.columns(class_ids)
        unc = (lambda: self.uncertainties(cols)) if self.query_uncertainty else self.uncertainties
        return [_row(name, "probs", lambda: self.mean()[:, cols], unc)]

    def with_alpha0(self, alpha0):
        """The same counts read out with another prior alpha0 (a shallow copy: alpha0 enters
        only at read-out, so no second mapping pass)."""
        import copy
        if not float(alpha0) > 0:
            raise ValueError(f"alpha0 must be > 0 (got {alpha0})")
        v = copy.copy(self)
        v.alpha0 = float(alpha0)
        return v

    def state_dict(self):
        s = {k: getattr(self, k).cpu() for k in self._state}
        if self.storage == "sparse":                 # (N, K) tensors row-sparse if smaller
            s = {k: compact_rows(v) if v.dim() == 2 else v for k, v in s.items()}
        s["class_ids"] = None if self.class_ids is None else list(self.class_ids)
        return s

    def load_state_dict(self, s):
        if self.yager and "yb" not in s:
            raise KeyError("CSS state has no Yager opinions (saved with yager=False)")
        ids = s.get("class_ids")
        if ids is not None:
            ids = [int(c) for c in ids]
            if self.class_ids is not None and ids != self.class_ids:
                raise ValueError(f"saved class ids {ids} != this CSS's {self.class_ids}")
            self.class_ids = ids
        s = {k: dense_rows(v) if is_sparse_rows(v) else v for k, v in s.items()}
        return _load_tensors(self, s)


# ----------------------------------------------------------------- (e) ours rows
def ours_scalar_row(field, name="ours_scalar"):
    """Tab III (h): the P0 feature with the single scalar 1 - A_d(kappa_post) (rows.py
    channel "spread"), no epistemic/aleatoric split."""
    return FieldRow(name, "feature", field=field, mode="P0", channels=("spread",))


def make_rows(field, baselines=(), modes=("P0", "P1", "P2"), scalar=True, class_ids=None):
    """Ours rows (rows.field_rows: P0 mu*, P1 A_d(kappa_post) mu*, P2 adaptive beta, each with
    u_epi and u_ale), Ours-scalar, then every baseline's rows. field=None: baselines only.
    class_ids: the queried global ids (View.class_ids); they select the CSS columns."""
    rows = []
    if field is not None:
        rows += field_rows(field, "ours", modes=modes)
        rows += [ours_scalar_row(field)] if scalar else []
    for b in baselines:
        rows += b.rows(class_ids) if isinstance(b, CSS) else b.rows()
    names = [r.name for r in rows]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"duplicate row names {dup}")
    return rows


def update_all(field, rasterize_fn, z_obs, css=(), **kw):
    """One keyframe for the field (and its hooked baselines) plus the CSS passes. `css` may
    hold any baselines: only CSS runs its own pass, the others ride on field.view_hooks."""
    stats = field.update_view(rasterize_fn, z_obs, **kw)
    for c in css:
        if isinstance(c, CSS):
            c.update_view(rasterize_fn, z_obs, pixel_valid=kw.get("pixel_valid"))
    return stats


# ----------------------------------------------------------------- checkpoints / eval
_TYPES = {"Averaging": Averaging, "NormLift": NormLift, "ViewCount": ViewCount, "CSS": CSS}


def default_baselines(field, text=None, class_ids=None, alpha0=1.0, beta=1.0, yager=True):
    """The K11 set on a field: NormLift (+ the averaging row from the same sums), the view
    count on its feature, and CSS if text (K, d) is given. Hooks are attached; pass the list
    to update_all(..., css=bs) for every keyframe."""
    nl = NormLift.for_field(field, beta=beta, with_averaging=True)
    bs = [nl, ViewCount.for_field(field, feature=nl.feature)]
    if text is not None:
        bs.append(CSS.for_field(field, text, alpha0=alpha0, yager=yager, class_ids=class_ids))
    return bs


CSS_FULL = "css_full"


def full_vocab_css(field, text, class_ids, alpha0=1.0):
    """The control for "CSS was told the scene's classes" (run_mapping.py --css-full-vocab):
    CSS over every vocabulary class, text (K, d) with its K global ids, rows css_full*, its
    Dirichlet channels on the queried columns. No Yager opinions; row-sparse on disk. Append
    it to the baselines list: update_all and run_mapping.py run its own one-hot pass like
    CSS's. Validation only (CSS_FULL_RULE)."""
    return CSS.for_field(field, text, alpha0=alpha0, yager=False, class_ids=class_ids,
                         name=CSS_FULL, storage="sparse", query_uncertainty=True)


def baselines_state(baselines):
    """ckpt["baselines"]: [{type, config, state}] in list order. A ViewCount keeps the list
    index of the accumulator whose feature() it renders ("feature_from")."""
    out = []
    for b in baselines:
        name = type(b).__name__
        if name not in _TYPES:
            raise TypeError(f"cannot save baseline of type {name}")
        cfg = b.config()
        if isinstance(b, ViewCount):
            src = getattr(b._feature, "__self__", None)
            idx = next((i for i, o in enumerate(baselines) if o is src), None)
            if b._feature is not None and idx is None:
                raise ValueError("ViewCount's feature source is not in the saved list")
            cfg["feature_from"] = idx
        out.append(dict(type=name, config=cfg, state=b.state_dict()))
    return out


def load_baselines(saved, device="cpu"):
    """Inverse of baselines_state (hooks are not attached: the loaded sums are final)."""
    bs, links = [], []
    for e in saved:
        cfg = dict(e["config"])
        dtype = getattr(torch, cfg.pop("dtype", "float32"))
        src = cfg.pop("feature_from", None)
        b = _TYPES[e["type"]](**cfg, device=device, dtype=dtype)
        b.load_state_dict(e["state"])
        if src is not None:
            links.append((b, src))
        bs.append(b)
    for b, i in links:                                     # rewire ViewCount's feature
        b._feature = bs[i].feature
    return bs


# CSS prior read-outs evaluated next to the mapped alpha0 (row "css", alpha0 = 1): rows
# "css_a<alpha0>". The Table I CSS row is the one with the lowest validation NLL (the
# strongest CSS on the validation scenes); the counts are pixel masses at the mapping grid.
# The full-vocabulary control gets the same grid ("css_full_a<alpha0>") and the same rule.
CSS_ALPHA0_GRID = (0.01, 0.1, 10.0)

# css_full is a VALIDATION-ONLY control (fixed 2026-10-10, before any run with it): the v1 test
# fields hold no css_full state, and no guarded stage maps the test scenes with
# --css-full-vocab (scripts/replica_tab1_v2.sh evaluates the existing fields and never maps;
# scripts/scannet_tab1.sh refuses it in MAP_EXTRA, as its prereg has no css_full choice), so
# it has no test number. Its read-out follows CSS's rule (scripts/make_tables.py choose --key
# css), machine-readable here: the row of "rows" with the lowest mean validation NLL over the
# validation scenes (eval_seg summary, variant "all"); ties go to the first listed (the mapped
# alpha0 = 1); choose_css_full applies it. A test look would need a guarded mapping + test
# stage pre-registered with this rule, and v2 validation temperatures (the mapping-code hash
# changes).
CSS_FULL_RULE = dict(rows=(CSS_FULL,) + tuple(f"{CSS_FULL}_a{a:g}" for a in CSS_ALPHA0_GRID),
                     criterion="nll", stat="mean", variant="all", stage="val", test=False)


def choose_css_full(val_summary, rule=CSS_FULL_RULE):
    """CSS_FULL_RULE on an eval_seg validation summary {row: {variant: {metric: {stat}}}}: the
    row with the lowest finite criterion, None if no css_full row is in it."""
    cand = {}
    for r in rule["rows"]:
        try:
            v = float(val_summary[r][rule["variant"]][rule["criterion"]][rule["stat"]])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(v):
            cand[r] = v
    return min(cand, key=cand.get) if cand else None


def normlift_vote_factory(ckpt, field=None, device="cpu", class_ids=None, means=None):
    """scripts/eval_seg.py --row-factory usplat.baselines:normlift_vote_factory: the row
    "normlift_vote", NormLift with its Sec. 3.4 KNN mode voting (NORMLIFT_VOTE). It needs the
    map's Gaussian centres, which eval_seg passes as `means`; registered in
    configs/normlift_vote_prereg.json."""
    if means is None:
        raise ValueError("normlift_vote_factory needs the Gaussian means (eval_seg passes them)")
    if "baselines" not in ckpt:
        raise KeyError("checkpoint has no 'baselines'")
    nls = [b for b in load_baselines(ckpt["baselines"], device=device) if isinstance(b, NormLift)]
    if len(nls) != 1:
        raise ValueError(f"expected one NormLift accumulator in ckpt['baselines'], found {len(nls)}")
    if field is not None and nls[0].N != field.N:
        raise ValueError(f"saved NormLift covers {nls[0].N} Gaussians, the field {field.N}")
    return NormLiftVote(nls[0], means).rows()


def row_factory(ckpt, field=None, device="cpu", class_ids=None):
    """scripts/eval_seg.py --row-factory usplat.baselines:row_factory. Rebuilds the baselines
    saved in ckpt["baselines"] and returns their rows only (eval_seg adds ours_* itself).
    class_ids: the scene's queried global ids (View.class_ids), which select the CSS
    columns; None keeps all mapped classes (eval_seg --queries all)."""
    if "baselines" not in ckpt:
        raise KeyError("checkpoint has no 'baselines' (map with the K11 accumulators and save "
                       "baselines_state(bs) as ckpt['baselines'])")
    bs = load_baselines(ckpt["baselines"], device=device)
    if field is not None and any(b.N != field.N for b in bs):
        raise ValueError(f"saved baselines cover {[b.N for b in bs]} Gaussians, the field "
                         f"{field.N}")
    rows = make_rows(None, bs, class_ids=class_ids)
    for b in bs:
        if isinstance(b, CSS):
            for a0 in CSS_ALPHA0_GRID:
                rows += b.with_alpha0(a0).rows(class_ids, name=f"{b.name}_a{a0:g}")
    return rows
