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
    this row. Uncertainties: "normlift_1-R" = 1 - R(j), "normlift_epi" = 1/(1 + N_eff).
(c) Visible-view count: n_vis = #views with m_v > 0 and n_full = #views with m_v >= a0*H*W
    (the field's full-view mass, ViewStats.ref). Uncertainty 1/(1 + n); averaging feature.
(d) CSS, Continuous Semantic Splatting (Wilson et al., ICRA 2025,
    doi:10.1109/icra55743.2025.11128341, arXiv 2411.02547): a closed-set Dirichlet per
    Gaussian over K classes FIXED at mapping time. CSS receives the frozen evaluation
    vocabulary (configs/vocab_replica.json, all classes, with their global ids `class_ids`),
    never the classes present in a test scene.
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
    The uncertainty channels stay over the full mapped vocabulary.
    "yager_conflict": per view, one subjective-logic opinion per Gaussian from its one-hot counts
    r_c (prior weight K): b_c = r_c/S_v, u = K/S_v, S_v = sum_c r_c + K. Views are fused in
    order with Yager's rule (Yager, Inf. Sci. 41(2):93-137, 1987,
    doi:10.1016/0020-0255(87)90007-7), which moves the conflict mass
        k = (sum_c b_c)(sum_c b'_c) - sum_c b_c b'_c
    to ignorance instead of renormalising it away (Dempster):
        b_c <- b_c b'_c + b_c u' + u b'_c ,   u <- u u' + k
    The row reports the accumulated conflict sum_v k_v. Evidential BKI for Gaussian maps:
    E2-BKI (Kim et al., RA-L 11(3) 2026, doi:10.1109/lra.2026.3653367).
(e) Ours-scalar (plan Tab III h): the field's P0 feature with the single scalar
    1 - A_d(kappa_post) (the exact posterior spread, rows.py channel "spread"), no
    epistemic/aleatoric split.

make_rows(field, baselines) returns the field's rows from rows.field_rows (P0: mu*, P1:
A_d(kappa_post) mu*, P2: P0 with the adaptive beta; each with u_epi and u_ale, NaN where u_ale
is undefined), then Ours-scalar, then every baseline's rows. Rows read their accumulator when
asked, so they can be built before the updates; seg_eval.freeze_rows snapshots them.

Checkpoints: default_baselines(field, text, class_ids) builds and attaches the K11 set
(NormLift with the averaging row, the view count on its feature, CSS); update_all runs one
keyframe for all of them; baselines_state(bs) is saved as ckpt["baselines"]; row_factory is
scripts/eval_seg.py --row-factory usplat.baselines:row_factory and returns the baseline rows
only (eval_seg builds the ours_* rows itself).
"""
from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import Callable, Optional, Sequence
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
                "normlift_epi": 1.0 / (1.0 + self.n_eff())}

    def rows(self, with_averaging=None):
        avg = self.with_averaging if with_averaging is None else with_averaging
        rows = [_row("averaging", "feature", self.feature)] if avg else []
        return rows + [_row("normlift", "feature", self.feature, self.uncertainties)]


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


class CSS:
    """Closed-set Dirichlet per Gaussian (CSS) with a Yager-conflict read-out."""
    _state = ("counts", "yb", "yu", "yk")

    def __init__(self, num_gaussians, text: torch.Tensor, alpha0=1.0, yager=True,
                 device="cpu", dtype=torch.float32, class_ids: Optional[Sequence[int]] = None):
        """text: (K, d) class embeddings in the field space (StreamingPCA.project_text) of the
        frozen vocabulary. class_ids: the K global class ids of the text rows (vocabulary
        ids; rows(class_ids) selects columns by them). alpha0 > 0: uninformative Dirichlet
        prior (1 = flat) [A: fixed on val]."""
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
        return dict(num_gaussians=self.N, text=self.text.cpu(), alpha0=self.alpha0,
                    yager=self.yager, class_ids=self.class_ids, dtype=_dtype_name(self.dtype))

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

    def uncertainties(self):
        out = dirichlet_uncertainties(self.alpha())
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

    def rows(self, class_ids=None):
        """class_ids=None: all K columns in mapping order. Else the columns of the queried
        global ids in the query order (View.class_ids); the evaluator renormalises them."""
        if class_ids is None:
            return [_row("css", "probs", self.mean, self.uncertainties)]
        cols = self.columns(class_ids)
        return [_row("css", "probs", lambda: self.mean()[:, cols], self.uncertainties)]

    def state_dict(self):
        s = {k: getattr(self, k).cpu() for k in self._state}
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
    return make_rows(None, bs, class_ids=class_ids)
