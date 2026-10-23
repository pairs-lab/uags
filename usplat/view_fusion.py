"""
view_fusion.py — per-pixel fusion of a map predictive with the predictive of the evaluation
frame's own 2D features: the Table I block "with the current view". Rules (rows, fit,
selection, claims): configs/fusion_prereg.json; evaluation: scripts/eval_fusion.py (imports
scripts/eval_seg.py and usplat.seg_eval, edits neither); driver: scripts/fusion_tab1.sh.

Protocol fact this uses: Table I's "2D features, per frame" row applies the extractor to the
evaluation frame itself (eval_seg.py --with-2d, PixelSource, frame_features_2d), so the
current image is available at query time in this protocol. Every fused row gets the SAME 2D
input; only the map logits and the weight differ between them.

Per labelled pixel p and queried text t_c (the scene's present classes, field space):
    L_map,p(c) = beta_map,p F_p . t_c   the map row's own logits with its own fitted temperature
                                        (a scalar beta, or the P2 per-pixel beta_p; F_p is not
                                        divided by alpha_p, as in usplat.seg_eval)
    L_2d,p(c)  = beta_2d z_p . t_c      the 2D row's own logits with its own fitted temperature
    z_p        = (1 - w_p) L_map,p + w_p L_2d,p                     (fused logits)
               = the normalised log-linear pool  P(c) ~ P_map(c)^(1-w_p) P_2d(c)^(w_p)
    w_p        = sigmoid(theta_0 + sum_j theta_j x_pj)   on map-covered pixels (alpha_p >= tau_cov
                                                         and alpha_p > alpha_eps, seg_eval's rule)
    w_p        = 1                                       on uncovered pixels: the 2D prediction
                                                         decides there (a map row counts them
                                                         as wrong with the uniform predictive)
Weight features x (FEATURES; every value is a render with the splat weights, sum w u / alpha):
    u_epi            our rendered epistemic channel
    u_ale            our rendered aleatoric channel, 0 where undefined
    u_ale_undef      1[u_ale undefined] (more than undef_frac of alpha from Gaussians without it)
    one_minus_alpha  1 - alpha_p
    viewcount        rendered 1/(1 + n_vis) of the view-count baseline (usplat.baselines.ViewCount)
    nl_epi           rendered 1/(1 + N_eff) of NormLift (channel "normlift_epi")
    nl_cinter        rendered 1 - C_inter of NormLift (channel "normlift_1-Cinter")
Forms (FORMS, in the prereg's order; base = the map row whose logits are fused):
    fuse_avg_const  averaging + 2D, one constant weight               (fair baseline)
    fuse_avg_vc     averaging + 2D, w from viewcount, 1 - alpha       (fair baseline)
    fuse_avg_nl     averaging + 2D, w from nl_epi, nl_cinter, 1 - alpha (fair baseline)
    fuse_p0_const   ours P0 + 2D, one constant weight                 (fair baseline)
    fuse_p0_ug      ours P0 + 2D, w from u_epi, u_ale, u_ale_undef, 1 - alpha   (OURS-UG)
    fuse_p2_const   ours P2 + 2D, one constant weight                 (fair baseline)
    fuse_p2_ug      ours P2 + 2D, the OURS-UG weight                  (OURS-UG)
Exactness: w = 0 gives the map row's logits bit for bit on covered pixels (1.0 * L + 0.0 *
L_2d), w = 1 the 2D row's logits everywhere (0.0 * L_map + 1.0 * L_2d; L_map is set to 0 on
uncovered pixels, where P2's beta may be NaN); the convex form is written out so that both
endpoints are exact (tests/test_view_fusion.py).
Fit (fit_weights): two stages. (1) The single-source temperatures are eval_seg's BetaFitter
on Table I's pixel subsets (scripts/eval_fusion.py replays its order), so they reproduce
Table I's validation betas. (2) theta of each form by NLL on a joint subset of covered labelled
validation pixels (at most max_pixels_per_view per view, drawn with its own generator), with
the single-source logits FIXED (stacking): L-BFGS (strong Wolfe), float64, theta = 0 (w = 1/2)
at the start. Uncovered pixels have w = 1 whatever theta, so they do not enter the fit.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as Fn

from .predictive import adaptive_beta
from .seg_eval import IGNORE, TIE_TOL, class_scores, _coverage

FEATURES = ("u_epi", "u_ale", "u_ale_undef", "one_minus_alpha", "viewcount", "nl_epi",
            "nl_cinter")
BASES = ("averaging", "ours_P0", "ours_P2")
UG = ("u_epi", "u_ale", "u_ale_undef", "one_minus_alpha")
# rendered channel of the bundle -> (render, channel name in that render)
CHANNELS = {"u_epi": ("ours", "u_epi"), "u_ale": ("ours", "u_ale"),
            "viewcount": ("avg", "viewcount"), "nl_epi": ("avg", "normlift_epi"),
            "nl_cinter": ("avg", "normlift_1-Cinter")}
ROW_2D = "2d_features"
FIT_SEED_OFFSET = 1          # the fusion subset's generator: seed + 1 (BetaFitter keeps seed)
FIT_ITERS = 500


@dataclass(frozen=True)
class FusionForm:
    """One fused row: the map row `base` whose logits are fused with the 2D row's, and the
    weight features (empty: one constant weight)."""
    name: str
    base: str
    features: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.base not in BASES:
            raise ValueError(f"form {self.name!r}: base {self.base!r} not in {BASES}")
        bad = [f for f in self.features if f not in FEATURES]
        if bad or len(set(self.features)) != len(self.features):
            raise ValueError(f"form {self.name!r}: features {list(self.features)} (unknown "
                             f"{bad} or repeated); allowed {FEATURES}")

    def params(self):
        """Names of theta, intercept first."""
        return ["w0"] + [f"w_{f}" for f in self.features]

    def spec(self):
        return dict(name=self.name, base=self.base, features=list(self.features))


# the prereg's rows, in its order (tests/test_view_fusion.py checks that they agree)
FORMS = (
    FusionForm("fuse_avg_const", "averaging"),
    FusionForm("fuse_avg_vc", "averaging", ("viewcount", "one_minus_alpha")),
    FusionForm("fuse_avg_nl", "averaging", ("nl_epi", "nl_cinter", "one_minus_alpha")),
    FusionForm("fuse_p0_const", "ours_P0"),
    FusionForm("fuse_p0_ug", "ours_P0", UG),
    FusionForm("fuse_p2_const", "ours_P2"),
    FusionForm("fuse_p2_ug", "ours_P2", UG),
)
FORMS_BY_NAME = {f.name: f for f in FORMS}


def forms_by_names(names):
    bad = [n for n in names if n not in FORMS_BY_NAME]
    if bad:
        raise ValueError(f"unknown fused rows {bad}; registered {list(FORMS_BY_NAME)}")
    return [FORMS_BY_NAME[n] for n in names]


# ------------------------------------------------------------------------- pixel bundle
@dataclass
class Bundle:
    """The labelled pixels of one view with everything the fused rows read: scores F . t of
    the ours and averaging renders and z . t of the 2D row (float32, the evaluator's
    class_scores), the rendered channels, alpha and the map coverage."""
    pix: torch.Tensor                    # (M,) flat pixel index (labelled pixels)
    y: torch.Tensor                      # (M,) long, query index
    cov: torch.Tensor                    # (M,) bool, map-covered
    alpha: torch.Tensor                  # (M,) map alpha
    s: Dict[str, torch.Tensor]           # "ours" | "avg" | "2d" -> (M, K)
    chan: Dict[str, torch.Tensor]        # CHANNELS -> (M,) (u_ale NaN where undefined)

    @property
    def K(self):
        return self.s["2d"].shape[1]

    def take(self, idx):
        """The sub-bundle of positions idx (a LongTensor on the bundle's device)."""
        return Bundle(self.pix[idx.to(self.pix.device)], self.y[idx], self.cov[idx],
                      self.alpha[idx], {k: v[idx] for k, v in self.s.items()},
                      {k: v[idx] for k, v in self.chan.items()})

    def to(self, device):
        return Bundle(self.pix, self.y.to(device), self.cov.to(device), self.alpha.to(device),
                      {k: v.to(device) for k, v in self.s.items()},
                      {k: v.to(device) for k, v in self.chan.items()})


@torch.no_grad()
def pixel_bundle(r_ours, r_avg, r_2d, text, gt_labels, tau_cov, alpha_eps=1e-6,
                 ignore_index=IGNORE):
    """Bundle of one view from three seg_eval.RowRender: r_ours (our mu* buffer with channels
    u_epi, u_ale), r_avg (the averaging buffer with the channels viewcount, normlift_epi,
    normlift_1-Cinter) and r_2d (the 2D row through eval_seg.PixelSource). The map coverage
    is seg_eval's rule on r_ours.alpha (every map row shares the geometry, hence alpha)."""
    gt = torch.as_tensor(gt_labels).reshape(-1).long()
    lab = gt != ignore_index
    if bool((gt[lab] < 0).any()):
        raise ValueError(f"negative label {int(gt[lab].min())} that is not the ignore index")
    pix = torch.nonzero(lab).squeeze(1)
    dev = r_ours.feat.device
    idx = pix.to(dev)
    for rr in (r_avg, r_2d):
        if rr.alpha.shape[0] != gt.numel():
            raise ValueError(f"{rr.name!r}: render has {rr.alpha.shape[0]} pixels, labels "
                             f"have {gt.numel()}")
    if r_ours.alpha.shape[0] != gt.numel():
        raise ValueError(f"{r_ours.name!r}: render has {r_ours.alpha.shape[0]} pixels, labels "
                         f"have {gt.numel()}")
    a_o, a_a = r_ours.alpha[idx], r_avg.alpha[idx].to(dev)
    if not torch.allclose(a_o.float(), a_a.float(), rtol=0, atol=1e-5):
        raise ValueError("the ours and averaging renders have different alpha: not the same "
                         "map geometry")
    y = gt.to(dev)[idx]
    renders = {"ours": r_ours, "avg": r_avg}
    chan = {}
    for c, (src, name) in CHANNELS.items():
        u = renders[src].unc.get(name)
        if u is None:
            raise ValueError(f"render {renders[src].name!r} has no channel {name!r} "
                             f"(needed for {c!r}); it has {sorted(renders[src].unc)}")
        chan[c] = u[idx].to(dev)
    s = {"ours": class_scores(r_ours, text, alpha_eps)[idx],
         "avg": class_scores(r_avg, text, alpha_eps)[idx].to(dev),
         "2d": class_scores(r_2d, text, alpha_eps)[idx].to(dev)}
    K = s["2d"].shape[1]
    if y.numel() and int(y.max()) >= K:
        raise ValueError(f"label {int(y.max())} >= number of classes {K}")
    return Bundle(pix, y, _coverage(a_o, tau_cov, alpha_eps), a_o, s, chan)


# ------------------------------------------------------------------------------ logits
def _scalar(betas, row):
    b = (betas or {}).get(row)
    if b is None or isinstance(b, dict):
        raise ValueError(f"fusion needs the fitted scalar beta of {row!r} (got {b!r})")
    return float(b)


def base_logits(b: Bundle, base, betas, p2_channels=("u_epi", "u_ale")):
    """(M, K) float64 map logits of `base`, op for op as seg_eval.evaluate_view, and 0 on
    uncovered pixels (where P2's beta may be undefined; the 2D row decides there)."""
    if base == "ours_P0":
        L = _scalar(betas, "ours_P0") * b.s["ours"].double()
    elif base == "averaging":
        L = _scalar(betas, "averaging") * b.s["avg"].double()
    elif base == "ours_P2":
        p = (betas or {}).get("ours_P2")
        if not isinstance(p, dict):
            raise ValueError(f"fusion on P2 needs the P2 parameters of 'ours_P2' (got {p!r})")
        bp = adaptive_beta(p["beta0"], p["a"], p["b"], b.chan[p2_channels[0]],
                           b.chan[p2_channels[1]], p.get("g", 0.0))
        if not torch.isfinite(bp[b.cov]).all():
            raise ValueError("non-finite P2 beta on covered pixels (is u_epi finite?)")
        bp = torch.where(b.cov, bp, torch.ones_like(bp))
        L = bp[:, None].double() * b.s["ours"].double()
    else:
        raise ValueError(f"base {base!r} not in {BASES}")
    return torch.where(b.cov[:, None], L, torch.zeros_like(L))


def logits_2d(b: Bundle, betas):
    """(M, K) float64 logits of the 2D row (its own fitted temperature)."""
    return _scalar(betas, ROW_2D) * b.s["2d"].double()


def design(b: Bundle, features):
    """(M, F) float64 weight features; 0 on uncovered pixels (not used there). Refuses a
    non-finite feature on a covered pixel."""
    cols = []
    for f in features:
        if f == "u_ale":
            u = b.chan["u_ale"]
            x = torch.where(torch.isfinite(u), u, torch.zeros_like(u))
        elif f == "u_ale_undef":
            x = (~torch.isfinite(b.chan["u_ale"])).to(torch.float64)
        elif f == "one_minus_alpha":
            x = 1.0 - b.alpha.double()
        elif f in CHANNELS:
            x = b.chan[f]
        else:
            raise ValueError(f"unknown weight feature {f!r}")
        cols.append(x.double())
    M = b.y.numel()
    X = torch.stack(cols, 1) if cols else torch.zeros(M, 0, dtype=torch.float64,
                                                       device=b.y.device)
    bad = b.cov & ~torch.isfinite(X).all(1)
    if bool(bad.any()):
        raise ValueError(f"weight features {list(features)} non-finite on {int(bad.sum())} "
                         f"covered pixels")
    return torch.where(b.cov[:, None], X, torch.zeros_like(X))


def weights(theta, X, cov):
    """w = sigmoid(theta_0 + X theta_1:) on covered pixels, 1 elsewhere. theta may hold
    +-inf (w = 1 or 0 exactly: the endpoint rows)."""
    th = torch.as_tensor(theta, dtype=torch.float64, device=X.device)
    if th.numel() != X.shape[1] + 1:
        raise ValueError(f"theta has {th.numel()} values, the form needs {X.shape[1] + 1}")
    w = torch.sigmoid(th[0] + X @ th[1:])
    return torch.where(cov, w, torch.ones_like(w))


def fuse(Lm, L2, w):
    """Convex logit pool, written so that w = 0 and w = 1 are exact."""
    return (1.0 - w)[:, None] * Lm + w[:, None] * L2


@torch.no_grad()
def fused_outputs(form: FusionForm, theta, b: Bundle, betas):
    """Per-labelled-pixel arrays of one fused row in seg_eval.evaluate_view's format (pix,
    gt, pred, conf, entropy, nll, brier, correct, covered, tied, alpha, u, K). The fused row
    decides on every labelled pixel (covered = True; the map coverage is kept by the caller as
    'map_covered'); a tied maximum is no decision (pred -1), as in seg_eval. Its only channel
    is the fusion weight u = {'w2d': w}."""
    Lm = base_logits(b, form.base, betas)
    L2 = logits_2d(b, betas)
    w = weights(theta, design(b, form.features), b.cov)
    p = torch.softmax(fuse(Lm, L2, w), dim=1)
    y = b.y
    conf, pred = p.max(dim=1)
    tied = (p >= conf[:, None] - TIE_TOL).sum(1) > 1
    pred = torch.where(~tied, pred, torch.full_like(pred, -1))
    py = p.gather(1, y[:, None]).squeeze(1)
    return dict(pix=b.pix.cpu(), gt=y.cpu(), pred=pred.cpu(), conf=conf.float().cpu(),
                entropy=(-torch.special.xlogy(p, p).sum(1)).float().cpu(),
                nll=(-torch.log(py.clamp_min(1e-12))).float().cpu(),
                brier=((p * p).sum(1) - 2.0 * py + 1.0).float().cpu(),
                correct=(pred == y).cpu(), covered=torch.ones_like(y, dtype=torch.bool).cpu(),
                tied=tied.cpu(), alpha=b.alpha.float().cpu(), u={"w2d": w.float().cpu()},
                K=p.shape[1])


# --------------------------------------------------------------------------------- fit
@dataclass
class Chunk:
    Lm: torch.Tensor          # (n, K) float64 map logits (covered pixels)
    L2: torch.Tensor          # (n, K) float64 2D logits
    X: torch.Tensor           # (n, F) float64 weight features
    y: torch.Tensor           # (n,) long


def _nll_sum(theta, c: Chunk):
    w = torch.sigmoid(theta[0] + c.X @ theta[1:])             # X (n, 0): w = sigmoid(theta_0)
    return Fn.cross_entropy(fuse(c.Lm, c.L2, w), c.y, reduction="sum")


def chunk_nll(theta, chunks: Sequence[Chunk]):
    """Mean NLL (nats) of the fused logits over the chunks' pixels at theta."""
    th = torch.as_tensor(theta, dtype=torch.float64)
    n = sum(int(c.y.numel()) for c in chunks)
    with torch.no_grad():
        return float(sum(_nll_sum(th, c) for c in chunks) / max(n, 1))


def fit_weights(chunks: Sequence[Chunk], n_features, iters=FIT_ITERS, init=None):
    """theta (F + 1 values, intercept first) minimising the mean fused NLL over the chunks,
    by L-BFGS (strong Wolfe) in float64 from init (default 0: w = 1/2). Returns (theta list,
    info: nll, nll at w = 0 (map) and w = 1 (2D) on the same pixels, n_pixels, n_iter,
    grad_norm, converged = grad_norm < 1e-5)."""
    chunks = [c for c in chunks if c.y.numel()]
    n = sum(int(c.y.numel()) for c in chunks)
    if n == 0:
        raise RuntimeError("fusion fit: no covered labelled validation pixels")
    for c in chunks:
        if c.X.shape[1] != n_features:
            raise ValueError(f"chunk has {c.X.shape[1]} features, the form {n_features}")
    th0 = torch.zeros(n_features + 1, dtype=torch.float64) if init is None else \
        torch.as_tensor(init, dtype=torch.float64).clone()
    theta = th0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=iters, history_size=20,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = sum(_nll_sum(theta, c) for c in chunks) / n
        loss.backward()
        return loss
    opt.step(closure)
    th = theta.detach()
    th_g = th.clone().requires_grad_(True)
    loss = sum(_nll_sum(th_g, c) for c in chunks) / n
    loss.backward()
    g = float(th_g.grad.norm())
    st = opt.state[theta]
    zero = torch.full((n_features + 1,), 0.0, dtype=torch.float64)
    w0, w1 = zero.clone(), zero.clone()
    w0[0], w1[0] = -math.inf, math.inf
    info = dict(nll=float(loss.detach()), nll_map=chunk_nll(w0, chunks),
                nll_2d=chunk_nll(w1, chunks), n_pixels=n, n_iter=int(st.get("n_iter", 0)),
                grad_norm=g, converged=bool(g < 1e-5))
    return [float(x) for x in th], info


class FusionFitData:
    """Collects the fusion fit pixels of the validation views: per view, the covered labelled
    pixels, at most `cap` drawn with this collector's own generator (seed), kept on the CPU."""

    def __init__(self, cap=None, seed=0):
        self.cap, self.seed = cap, int(seed)
        self.gen = torch.Generator().manual_seed(self.seed)
        self.bundles: List[Bundle] = []

    @torch.no_grad()
    def add(self, b: Bundle):
        idx = torch.nonzero(b.cov).squeeze(1).cpu()
        if self.cap is not None and idx.numel() > self.cap:
            idx = idx[torch.randperm(idx.numel(), generator=self.gen)[:self.cap]]
        self.bundles.append(b.take(idx.to(b.y.device)).to("cpu"))
        return self

    def chunks(self, form: FusionForm, betas):
        return [Chunk(base_logits(b, form.base, betas), logits_2d(b, betas),
                      design(b, form.features), b.y) for b in self.bundles]

    def fit(self, forms: Sequence[FusionForm], betas, iters=FIT_ITERS):
        """{form name: dict(spec, params, theta, fit info)}."""
        out = {}
        for f in forms:
            theta, info = fit_weights(self.chunks(f, betas), len(f.features), iters)
            out[f.name] = dict(spec=f.spec(), params=f.params(), theta=theta, fit=info,
                               n_views=len(self.bundles))
        return out


def check_fusion_record(fusion, forms):
    """Refuse fitted parameters that do not belong to the code's forms (another spec, a
    missing row, a theta of the wrong length)."""
    if not isinstance(fusion, dict):
        raise ValueError("the record holds no fitted fusion parameters ('fusion')")
    for f in forms:
        e = fusion.get(f.name)
        if e is None:
            raise ValueError(f"no fitted fusion parameters for {f.name!r}; the record has "
                             f"{sorted(fusion)}")
        if e.get("spec") != f.spec():
            raise ValueError(f"{f.name!r}: the record's spec {e.get('spec')} differs from the "
                             f"code's {f.spec()}")
        if len(e.get("theta") or []) != len(f.params()):
            raise ValueError(f"{f.name!r}: theta has {len(e.get('theta') or [])} values, the form "
                             f"needs {len(f.params())}")
