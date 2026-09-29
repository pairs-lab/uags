"""
seg_eval.py — segmentation + calibration evaluation of rendered rows (plan K10b, §3).

Per view and row, ONE render of the row's buffer with a ones channel and its uncertainty
channels appended (the rasterizer is affine in the buffer, see usplat.rows):
    F_p     = sum_n w_{p,n} f_n            feature / probability channels
    alpha_p = sum_n w_{p,n} 1              coverage (the ones channel)
    U_p     = sum_n w_{p,n} u_n / alpha_p  each uncertainty channel
A channel with undefined (NaN) Gaussians is rendered from its defined part only, plus a
defined-mass channel D_p = sum_n w_{p,n} 1[u_n defined]:
    U_p = sum_{n defined} w u_n / D_p,     undefined (NaN) where D_p < (1 - undef_frac) alpha_p
Buffers must be finite and uncertainties finite or NaN: +-inf cannot be averaged, so it
raises (use NaN for "undefined" and a finite value for "very uncertain").
Per-pixel class probabilities:
    feature rows  p_p = softmax_c(beta_p F_p . t_c)     (F_p not divided by alpha)
    probs rows    p_p = softmax_c(beta_p log q_p),  q_p = F_p / alpha_p (uniform if alpha ~ 0);
                  beta = 1 gives q_p itself; a fitted beta is temperature scaling of log q
    beta_p = beta (P0/P1 and every baseline, fit on val) or the P2 form
             beta0 exp(-a U_epi - b U_ale - g 1[U_ale undefined])   (usplat.predictive)
Every row needs its own beta (plan: "every row gets its own beta fit on val"); a missing
beta raises. A probs row is evaluated with its raw probabilities (beta 1) only when the
caller names it in `raw_probs`.

DECISIONS. pred_p = argmax_c p_p(c) when the maximum is unique. A TIED maximum (e.g. every
Gaussian on the pixel unseen, so F_p = 0; or a CSS Gaussian still at its Dirichlet prior)
is no decision: pred = -1, counted wrong like an uncovered pixel, and counted (n_tied). So
no metric depends on the order of the text queries.

COVERAGE RULE (plan §3, fixed before any val run). A labelled pixel with
alpha_p < tau_cov is not covered by the map. In the "all" variant it counts:
  * as WRONG (pred = -1: a false negative of its GT class, a false positive of none),
  * with the uniform predictive (conf 1/K, entropy log K, NLL log K, Brier 1 - 1/K),
  * as the MOST uncertain pixel of every ranking signal: u = 1 for MSP (above any covered
    1 - max p <= 1 - 1/K), entropy log K (its maximum; only an exactly uniform covered
    pixel ties with it, and that pixel is wrong too), and u = +inf for every own channel.
    Own channels rank in three tiers: uncovered (+inf) > covered with undefined u (NaN,
    ranked above every finite value) > covered finite u. So on uncovered pixels (always
    wrong) an own channel is scored like MSP, and it is not penalised for its NaN values.
The "covered" variant keeps only covered pixels. Pixels with alpha below alpha_eps are never
covered (so tau_cov = 0 still excludes empty pixels). Counts n_uncovered, n_undefined (per
own channel, covered pixels) and n_tied are reported next to the metrics.

Metrics (SegAccumulator.metrics), per row and variant:
  mIoU over the classes present in the GT of the variant's pixels, pixel accuracy,
  ECE (equal-width bins on the max probability, 15), NLL, Brier, reliability bins,
  and for every ranking signal (own channels, "msp" = 1 - max p, "entropy"):
  AUSE against the 0/1 error (tie-aware, usplat.evaluation.sparsification_curve) and AURC
  (risk-coverage), both on the three-tier key above.
Error targets for the views figure (plan §3): with the same pixels at a k-view map and at the
full map, e_red = 1[wrong at k AND right at full], e_irr = 1[wrong at k AND wrong at full].
All k-view fields share the full geometry, so a hole is uncovered at every k; by default the
targets are on pixels covered in both maps (views_cross_ause; holes counted as n_hole).
Temperatures: fit_betas / BetaFitter on a validation split only (fit_temperature for fixed
beta, fit_adaptive for P2), on covered labelled pixels (the uniform predictive of uncovered
pixels does not depend on beta). Tied pixels stay in the fit: an all-class tie has the same
NLL (log K) for every beta, so it does not move the fit.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union
import math
import torch

from .rows import Row, freeze
from .evaluation import sparsification_curve, aurc, expected_calibration_error
from .predictive import adaptive_beta, fit_temperature, fit_adaptive

IGNORE = -1
ABSENT_SCORE = -1e4       # score of a class a view does not query (joint beta fits)
TIE_TOL = 1e-12           # probabilities within this of the maximum tie with it


# ------------------------------------------------------------------------------- render
@dataclass
class RowRender:
    name: str
    kind: str
    feat: torch.Tensor                 # (P, C) composited buffer (not divided by alpha)
    alpha: torch.Tensor                # (P,)   sum_n w_{p,n}
    unc: Dict[str, torch.Tensor]       # name -> (P,) rendered uncertainty (NaN = undefined)


@dataclass
class View:
    """One evaluation frame: rasterizer (buf (N, C) -> (P, C)), GT (H, W) or (P,) long in
    the index space of `text` rows (ignore = -1), text (K, d) in the field space (not
    renormalised), class_ids (K,) global ids of the text rows (only needed when views with
    different class sets are pooled in one beta fit)."""
    rasterizer: Callable
    gt: torch.Tensor
    text: Optional[torch.Tensor] = None
    class_ids: Optional[Sequence[int]] = None
    name: str = ""


def _check_names(rows: Sequence[Row]):
    names = [r.name for r in rows]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"duplicate row names {dup}: results are keyed by name")


def _render_dtype(rasterizer):
    dt = getattr(rasterizer, "dtype", None)          # the mock rasterizer carries one
    return dt if isinstance(dt, torch.dtype) and dt.is_floating_point else torch.float32


@torch.no_grad()
def render_row(row: Row, rasterizer, alpha_eps=1e-6, undef_frac=0.5) -> RowRender:
    """ONE rasterizer call: [buffer | 1 | u_1 (| defined_1) | u_2 ...]."""
    buf = row.buffer().detach()
    if buf.dim() != 2:
        raise ValueError(f"row {row.name!r}: buffer must be (N, C), got {tuple(buf.shape)}")
    N, C0 = buf.shape
    dt = _render_dtype(rasterizer)
    dev = buf.device
    buf = buf.to(dt)
    bad = ~torch.isfinite(buf).all(1)
    if bool(bad.any()):
        raise ValueError(f"row {row.name!r}: buffer has non-finite values on "
                         f"{int(bad.sum())} of {N} Gaussians")
    cols = [buf, torch.ones(N, 1, device=dev, dtype=dt)]
    layout = []
    for name, u in row.uncertainties().items():
        u = u.detach().to(dev, dt).reshape(-1)
        if u.numel() != N:
            raise ValueError(f"row {row.name!r}: channel {name!r} has {u.numel()} values, "
                             f"buffer has {N} Gaussians")
        if bool(torch.isinf(u).any()):
            raise ValueError(f"row {row.name!r}: channel {name!r} is +-inf on "
                             f"{int(torch.isinf(u).sum())} Gaussians; use NaN for an undefined "
                             f"value (inf cannot be averaged over a pixel)")
        fin = torch.isfinite(u)
        if bool(fin.all()):
            cols.append(u[:, None])
            layout.append((name, 1))
        else:                              # render the defined part and its mass
            cols += [torch.where(fin, u, torch.zeros_like(u))[:, None], fin.to(dt)[:, None]]
            layout.append((name, 2))
    img = rasterizer(torch.cat(cols, dim=1))
    feat, alpha = img[:, :C0], img[:, C0]
    if getattr(row, "decoder", None) is not None:      # render-then-decode rows (LEGaussians)
        if row.kind != "feature":
            raise ValueError(f"row {row.name!r}: a decoder needs a feature row")
        feat = row.decoder(feat)
    seen = alpha > alpha_eps
    nan = torch.full_like(alpha, float("nan"))
    unc, i = {}, C0 + 1
    for name, c in layout:
        if c == 1:
            u = torch.where(seen, img[:, i] / alpha.clamp_min(alpha_eps), nan)
        else:
            dm = img[:, i + 1]
            ok = seen & (dm >= (1.0 - undef_frac) * alpha) & (dm > alpha_eps)
            u = torch.where(ok, img[:, i] / dm.clamp_min(alpha_eps), nan)
        unc[name] = u
        i += c
    return RowRender(row.name, row.kind, feat, alpha, unc)


def class_scores(rr: RowRender, text=None, alpha_eps=1e-6):
    """(P, K) scores that beta multiplies: F . t (feature rows) or log q (probs rows)."""
    if rr.kind == "feature":
        if text is None:
            raise ValueError(f"row {rr.name!r}: feature rows need text (K, d)")
        t = text.to(rr.feat.device, rr.feat.dtype)
        if t.shape[1] != rr.feat.shape[1]:
            raise ValueError(f"row {rr.name!r}: text dim {t.shape[1]} != buffer dim "
                             f"{rr.feat.shape[1]}")
        return rr.feat @ t.t()
    K = rr.feat.shape[1]
    if text is not None and text.shape[0] != K:
        raise ValueError(f"row {rr.name!r}: {K} class probabilities but {text.shape[0]} texts")
    q = (rr.feat / rr.alpha.clamp_min(alpha_eps)[:, None]).clamp_min(0.0)
    s = q.sum(1, keepdim=True)
    ok = (rr.alpha > alpha_eps)[:, None] & (s > 0)
    q = torch.where(ok, q / s.clamp_min(1e-30), torch.full_like(q, 1.0 / K))
    return torch.log(q.clamp_min(1e-12))


def _is_p2(beta):
    return isinstance(beta, dict)


def pixel_beta(row: Row, beta, unc: Dict[str, torch.Tensor]):
    """Scalar beta, or the per-pixel P2 beta from the rendered (u_epi, u_ale)."""
    if not _is_p2(beta):
        return float(beta)
    ce, ca = row.p2_channels
    if ce not in unc or ca not in unc:
        raise ValueError(f"row {row.name!r}: P2 needs rendered channels {ce!r}, {ca!r}")
    return adaptive_beta(beta["beta0"], beta["a"], beta["b"], unc[ce], unc[ca],
                         beta.get("g", 0.0))


def row_beta(row: Row, betas, raw_probs=()):
    """The beta a row is evaluated with. Every row needs a fitted beta; a probs row named
    in raw_probs (and absent from betas) uses its raw probabilities (beta 1)."""
    b = (betas or {}).get(row.name)
    if row.name in raw_probs:
        if row.kind != "probs":
            raise ValueError(f"row {row.name!r} is a feature row: raw_probs is for probs rows")
        if b is not None:
            raise ValueError(f"row {row.name!r} has a fitted beta and is listed in raw_probs")
        return 1.0
    if b is None:
        hint = " or list it in raw_probs" if row.kind == "probs" else ""
        raise ValueError(f"no beta for {row.kind} row {row.name!r}: fit it on val "
                         f"(fit_betas){hint}")
    if row.adaptive and not _is_p2(b):
        raise ValueError(f"row {row.name!r} is adaptive (P2) but got a scalar beta")
    if not row.adaptive and _is_p2(b):
        raise ValueError(f"row {row.name!r} is not adaptive but got P2 parameters {b}")
    return b


def _coverage(alpha, tau_cov, alpha_eps):
    return (alpha >= float(tau_cov)) & (alpha > float(alpha_eps))


# ---------------------------------------------------------------------------- one view
@torch.no_grad()
def evaluate_view(rows: Sequence[Row], rasterizer, text, gt_labels, betas, tau_cov,
                  ignore_index=IGNORE, alpha_eps=1e-6, undef_frac=0.5, raw_probs=()):
    """One render per row. gt_labels (H, W) or (P,) long in the index space of `text`
    rows (pixel order = the rasterizer's pixel order, row-major for gsplat).
    Returns {row name: per-labelled-pixel arrays on CPU}:
        pix (flat pixel index), gt, pred (-1 = no decision: not covered or tied), conf,
        entropy, nll, brier, correct (coverage and tie rules applied), covered, tied
        (covered with a tied maximum), alpha, u {channel: rendered, NaN undefined},
        K (number of classes)."""
    _check_names(rows)
    gt = torch.as_tensor(gt_labels).reshape(-1).long()
    lab = gt != ignore_index
    if bool((gt[lab] < 0).any()):
        raise ValueError(f"negative label {int(gt[lab].min())} that is not the ignore index "
                         f"{ignore_index}")
    pix = torch.nonzero(lab).squeeze(1)
    out = {}
    for row in rows:
        b0 = row_beta(row, betas, raw_probs)             # before rendering: fail fast
        rr = render_row(row, rasterizer, alpha_eps, undef_frac)
        P = rr.alpha.shape[0]
        if P != gt.numel():
            raise ValueError(f"row {row.name!r}: render has {P} pixels, labels have {gt.numel()}")
        dev = rr.feat.device
        idx = pix.to(dev)
        y = gt.to(dev)[idx]
        s = class_scores(rr, text, alpha_eps)[idx].double()
        K = s.shape[1]
        if y.numel() and int(y.max()) >= K:
            raise ValueError(f"label {int(y.max())} >= number of classes {K}")
        b = pixel_beta(row, b0, {k: v[idx] for k, v in rr.unc.items()})
        cov = _coverage(rr.alpha[idx], tau_cov, alpha_eps)
        if torch.is_tensor(b):
            if not torch.isfinite(b[cov]).all():
                raise ValueError(f"row {row.name!r}: non-finite P2 beta on covered pixels "
                                 f"(is {row.p2_channels[0]!r} finite?)")
            b = torch.where(cov, b, torch.ones_like(b))          # uncovered: uniform below
        logits = (b[:, None].double() * s) if torch.is_tensor(b) else b * s
        p = torch.softmax(logits, dim=1)
        p = torch.where(cov[:, None], p, torch.full_like(p, 1.0 / K))
        conf, pred = p.max(dim=1)
        tied = (p >= conf[:, None] - TIE_TOL).sum(1) > 1     # argmax would pick class 0
        pred = torch.where(cov & ~tied, pred, torch.full_like(pred, -1))
        py = p.gather(1, y[:, None]).squeeze(1)
        out[row.name] = dict(
            pix=pix.cpu(), gt=y.cpu(), pred=pred.cpu(),
            conf=conf.float().cpu(),
            entropy=(-torch.special.xlogy(p, p).sum(1)).float().cpu(),
            nll=(-torch.log(py.clamp_min(1e-12))).float().cpu(),
            brier=((p * p).sum(1) - 2.0 * py + 1.0).float().cpu(),
            correct=(pred == y).cpu(), covered=cov.cpu(), tied=(cov & tied).cpu(),
            alpha=rr.alpha[idx].float().cpu(),
            u={k: v[idx].float().cpu() for k, v in rr.unc.items()}, K=K)
    return out


def freeze_rows(rows: Sequence[Row]):
    _check_names(rows)
    return [freeze(r) for r in rows]


def evaluate_views(rows, views: Iterable[View], betas, tau_cov, accumulator=None, **kw):
    """Snapshot the rows once, evaluate every view, accumulate. Returns the accumulator."""
    fr = freeze_rows(rows)
    acc = accumulator or SegAccumulator()
    for v in views:
        acc.add(evaluate_view(fr, v.rasterizer, v.text, v.gt, betas, tau_cov, **kw))
    return acc


# ------------------------------------------------------------------------- aggregation
def _cat(chunks, key):
    return torch.cat([c[key] for c in chunks])


def confusion_iou(gt, pred, num_classes=None):
    """Per-class IoU = TP / (GT + PRED - TP); pred = -1 is a miss (no false positive).
    Returns (iou (K,) NaN for classes absent from gt, gt_count (K,))."""
    gt, pred = gt.long(), pred.long()
    K = int(num_classes or (max(int(gt.max()), int(pred.max())) + 1 if gt.numel() else 0))
    tp = torch.bincount(gt[pred == gt], minlength=K)[:K].double()
    gc = torch.bincount(gt, minlength=K)[:K].double()
    pc = torch.bincount(pred[pred >= 0], minlength=K)[:K].double()
    union = gc + pc - tp
    iou = torch.where(gc > 0, tp / union.clamp_min(1), torch.full_like(tp, float("nan")))
    return iou, gc


def class_accuracy(gt, correct, gt_count):
    """Per-class recall TP / GT (NaN for absent classes)."""
    tp = torch.bincount(gt.long()[correct], minlength=gt_count.numel())[:gt_count.numel()]
    return torch.where(gt_count > 0, tp.double() / gt_count.clamp_min(1),
                       torch.full_like(gt_count, float("nan")))


def reliability(conf, correct, num_bins=15):
    """Equal-width bins on conf: per bin count, mean confidence, accuracy."""
    c, y = conf.double(), correct.double()
    idx = torch.clamp((c * num_bins).long(), max=num_bins - 1)
    n = torch.bincount(idx, minlength=num_bins).double()
    sc = torch.zeros(num_bins, dtype=torch.float64).index_add_(0, idx, c)
    sy = torch.zeros(num_bins, dtype=torch.float64).index_add_(0, idx, y)
    ok = n > 0
    return dict(count=n.long().tolist(),
                conf=torch.where(ok, sc / n.clamp_min(1), torch.nan).tolist(),
                acc=torch.where(ok, sy / n.clamp_min(1), torch.nan).tolist())


def ranking_metrics(error, key, ause_bins=20, n_undefined=0, n_uncovered=0):
    """AUSE (tie-aware) and AURC of a ranking key (ranking_signals: uncovered +inf,
    undefined above every finite value). n_undefined / n_uncovered are passed through."""
    sp = sparsification_curve(error, key, num_bins=ause_bins, undefined="last")
    return dict(ause=sp["ause"], random_gap=sp["random_gap"], n_undefined=int(n_undefined),
                n_uncovered=int(n_uncovered), aurc=aurc(key, error))


def undefined_key(u):
    """A finite value above every finite value of u (the tier of undefined u)."""
    fin = torch.isfinite(u)
    top = float(u[fin].max()) if bool(fin.any()) else 0.0
    return top + 1.0 + abs(top)


def ranking_signals(d):
    """{signal: (M,) float64 ranking key} with the coverage rule applied (the "all"
    variant): msp = 1 - conf (1 where not covered), entropy (log K there: uniform
    predictive), and every own channel in three tiers: +inf where not covered, then
    undefined (NaN) u at undefined_key(u), then the finite u itself. On covered pixels the
    order is the raw order (NaN ranked most uncertain), so the covered variant is a subset."""
    cov = d["covered"]
    clash = {"msp", "entropy"} & set(d["u"])
    if clash:
        raise ValueError(f"own channels {sorted(clash)} clash with the built-in signals")
    sig = {"msp": torch.where(cov, 1.0 - d["conf"].double(),
                              torch.ones_like(d["conf"], dtype=torch.float64)),
           "entropy": d["entropy"].double()}
    for k, u in d["u"].items():
        u = u.double()
        key = torch.where(torch.isfinite(u), u, torch.full_like(u, undefined_key(u)))
        sig[k] = torch.where(cov, key, torch.full_like(u, float("inf")))
    return sig


def summarize(d, class_names=None, num_bins=15, ause_bins=20):
    """Metrics of one row from concatenated per-pixel arrays (see SegAccumulator)."""
    out = {}
    cov = d["covered"]
    tied = d.get("tied", torch.zeros_like(cov))
    signals = ranking_signals(d)
    for variant in ("all", "covered"):
        sel = torch.ones_like(cov) if variant == "all" else cov
        M = int(sel.sum())
        if M == 0:
            out[variant] = dict(n_pixels=0)
            continue
        gt, pred, correct = d["gt"][sel], d["pred"][sel], d["correct"][sel]
        conf, c = d["conf"][sel], cov[sel]
        n_unc = int((~c).sum())
        iou, gc = confusion_iou(gt, pred)
        present = gc > 0
        names = (class_names if class_names is not None
                 else [str(i) for i in range(iou.numel())])
        err = 1.0 - correct.double()
        n_undef = {k: int((c & ~torch.isfinite(u[sel])).sum()) for k, u in d["u"].items()}
        ranking = {k: ranking_metrics(err, v[sel], ause_bins, n_undef.get(k, 0), n_unc)
                   for k, v in signals.items()}
        out[variant] = dict(
            n_pixels=M, coverage=float(c.double().mean()), n_uncovered=n_unc,
            n_tied=int(tied[sel].sum()),
            miou=float(iou[present].mean()), acc=float(correct.double().mean()),
            macc=float(class_accuracy(gt, correct, gc)[present].mean()),
            n_classes=int(present.sum()),
            iou={names[i]: float(iou[i]) for i in torch.nonzero(present).squeeze(1).tolist()},
            ece=expected_calibration_error(conf, correct, num_bins),
            nll=float(d["nll"][sel].double().mean()),
            brier=float(d["brier"][sel].double().mean()),
            reliability=reliability(conf, correct, num_bins),
            ranking=ranking)
        out[variant]["aurc"] = out[variant]["ranking"]["msp"]["aurc"]
    return out


class SegAccumulator:
    """Collects evaluate_view outputs over many views (one scene: one text index space).
    Stores ~(30 + 4 x channels) bytes per labelled pixel and row."""

    KEYS = ("gt", "pred", "conf", "entropy", "nll", "brier", "correct", "covered", "tied",
            "alpha")

    def __init__(self, class_names=None):
        self.class_names = class_names
        self.chunks: Dict[str, List[dict]] = {}
        self.n_views = 0

    def add(self, view_out):
        for name, d in view_out.items():
            slim = {k: v for k, v in d.items() if k not in ("pix", "K")}
            slim["gt"], slim["pred"] = d["gt"].int(), d["pred"].int()
            self.chunks.setdefault(name, []).append(slim)
        self.n_views += 1

    def arrays(self, name):
        ch = self.chunks[name]
        d = {k: _cat(ch, k) for k in self.KEYS}
        chans = list(ch[0]["u"].keys())
        d["u"] = {k: torch.cat([c["u"][k] for c in ch]) for k in chans}
        return d

    def metrics(self, num_bins=15, ause_bins=20):
        return {name: summarize(self.arrays(name), self.class_names, num_bins, ause_bins)
                for name in self.chunks}


# -------------------------------------------------------------------- temperature fits
class BetaFitter:
    """Collects (scores, labels, P2 channels) on covered labelled VAL pixels, possibly from
    several scenes (each with its own rows objects; rows are matched by name), then fits
    one beta (or one P2 parameter set) per row name. `settings` records what the fit
    depends on (tau_cov, undef_frac, ...); evaluate with the same values."""

    def __init__(self, tau_cov, max_pixels_per_view=None, seed=0, fit_probs=True,
                 alpha_eps=1e-6, undef_frac=0.5, init=10.0):
        self.tau_cov, self.cap, self.fit_probs = tau_cov, max_pixels_per_view, fit_probs
        self.alpha_eps, self.undef_frac, self.init = alpha_eps, undef_frac, init
        self.settings = dict(tau_cov=float(tau_cov), undef_frac=float(undef_frac),
                             alpha_eps=float(alpha_eps), max_pixels_per_view=max_pixels_per_view,
                             seed=int(seed), fit_probs=bool(fit_probs))
        self.gen = torch.Generator().manual_seed(seed)
        self.data: Dict[str, dict] = {}
        self.info: Dict[str, dict] = {}

    @torch.no_grad()
    def add_view(self, rows, rasterizer, text, gt_labels, class_ids=None,
                 ignore_index=IGNORE):
        _check_names(rows)
        gt = torch.as_tensor(gt_labels).reshape(-1).long()
        sub = None
        for row in rows:
            if row.kind == "probs" and not self.fit_probs:
                continue
            rr = render_row(row, rasterizer, self.alpha_eps, self.undef_frac)
            dev = rr.feat.device
            if rr.alpha.shape[0] != gt.numel():
                raise ValueError(f"row {row.name!r}: render has {rr.alpha.shape[0]} pixels, "
                                 f"labels have {gt.numel()}")
            ok = (gt.to(dev) != ignore_index) & _coverage(rr.alpha, self.tau_cov, self.alpha_eps)
            idx = torch.nonzero(ok).squeeze(1)
            if sub is None:                     # same pixel subset for every row of a view
                sub = idx.cpu()
                if self.cap is not None and sub.numel() > self.cap:
                    sub = sub[torch.randperm(sub.numel(), generator=self.gen)[:self.cap]]
            idx = sub.to(dev)
            s = class_scores(rr, text, self.alpha_eps)[idx].float().cpu()
            ids = (list(range(s.shape[1])) if class_ids is None
                   else [int(c) for c in class_ids])
            e = self.data.setdefault(row.name, dict(kind=row.kind, adaptive=row.adaptive,
                                                    chunks=[]))
            if (e["kind"], e["adaptive"]) != (row.kind, row.adaptive):
                raise ValueError(f"row {row.name!r}: kind/adaptive differ between views")
            ch = dict(s=s, y=gt[idx.cpu()], ids=ids)
            if row.adaptive:
                ce, ca = row.p2_channels
                ch["ue"], ch["ua"] = rr.unc[ce][idx].cpu(), rr.unc[ca][idx].cpu()
                if not torch.isfinite(ch["ue"]).all():
                    raise ValueError(f"row {row.name!r}: {ce!r} must be finite on covered pixels")
            e["chunks"].append(ch)
        return self

    def add(self, rows, views: Iterable[View], ignore_index=IGNORE):
        fr = freeze_rows(rows)
        for v in views:
            self.add_view(fr, v.rasterizer, v.text, v.gt, v.class_ids, ignore_index)
        return self

    @staticmethod
    def _stack(chunks):
        """Scores in one class space; classes a view did not query get ABSENT_SCORE."""
        same = all(c["ids"] == chunks[0]["ids"] for c in chunks)
        if same:
            return torch.cat([c["s"] for c in chunks]), torch.cat([c["y"] for c in chunks])
        union = sorted({i for c in chunks for i in c["ids"]})
        col = {g: j for j, g in enumerate(union)}
        S, Y = [], []
        for c in chunks:
            m = torch.full((c["s"].shape[0], len(union)), ABSENT_SCORE)
            cols = torch.tensor([col[g] for g in c["ids"]])
            m[:, cols] = c["s"]
            S.append(m)
            Y.append(cols[c["y"]])
        return torch.cat(S), torch.cat(Y)

    def fit(self):
        betas = {}
        for name, e in self.data.items():
            s, y = self._stack(e["chunks"])
            if y.numel() == 0:
                raise RuntimeError(f"row {name!r}: no covered labelled val pixels")
            init = 1.0 if e["kind"] == "probs" else self.init
            if e["adaptive"]:
                ue = torch.cat([c["ue"] for c in e["chunks"]])
                ua = torch.cat([c["ua"] for c in e["chunks"]])
                betas[name] = fit_adaptive(s, y, ue, ua, init_beta=init)
            else:
                betas[name] = fit_temperature(s, y, init=init)
            self.info[name] = dict(n_pixels=int(y.numel()), n_views=len(e["chunks"]),
                                   n_classes=int(s.shape[1]))
        return betas


def fit_betas(rows, val_views: Iterable[View], tau_cov, **kw):
    """One beta (P2: dict beta0, a, b, g) per row, fit by NLL on the val views."""
    return BetaFitter(tau_cov, **kw).add(rows, val_views).fit()


# ----------------------------------------------------------------------- error targets
def error_targets(correct_k, correct_full):
    """Same pixels at a k-view map and at the full map (plan §3):
    e_red = wrong at k AND right at full (reducible by more views),
    e_irr = wrong at k AND wrong at full."""
    ck = torch.as_tensor(correct_k).bool()
    cf = torch.as_tensor(correct_full).bool()
    if ck.shape != cf.shape:
        raise ValueError(f"shape mismatch {tuple(ck.shape)} vs {tuple(cf.shape)}")
    return dict(e_red=(~ck & cf).double(), e_irr=(~ck & ~cf).double())


def _aligned_keep(out_k, out_full, covered_only):
    if "pix" in out_k and "pix" in out_full and not torch.equal(out_k["pix"], out_full["pix"]):
        raise ValueError("the two outputs cover different pixels")
    if not torch.equal(out_k["gt"], out_full["gt"]):
        raise ValueError("the two outputs have different GT")
    both = out_k["covered"].bool() & out_full["covered"].bool()
    keep = both if covered_only else torch.ones_like(both)
    return keep, int((~both).sum())


def aligned_error_targets(out_k, out_full, covered_only=True, return_mask=False):
    """error_targets from two evaluate_view (or SegAccumulator.arrays) outputs of one row;
    checks that both cover the same pixels with the same GT. covered_only (default) keeps
    the pixels covered in both maps: a hole of the shared geometry is uncovered at every k,
    so it would be e_irr at every k whatever the semantics. The targets then have one entry
    per kept pixel; return_mask=True also returns the (M,) keep mask to subset u with."""
    keep, _ = _aligned_keep(out_k, out_full, covered_only)
    t = error_targets(out_k["correct"][keep], out_full["correct"][keep])
    return (t, keep) if return_mask else t


def cross_ause(uncertainties: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
               ause_bins=20):
    """{u name: {target name: AUSE}} (the 2x2 views figure: (u_epi, u_ale) x (e_red, e_irr)).
    Inputs must be aligned (same pixels); NaN u ranks as most uncertain. For evaluate_view
    outputs use views_cross_ause, which applies the coverage rule."""
    for un, u in uncertainties.items():
        for tn, t in targets.items():
            if torch.as_tensor(u).numel() != torch.as_tensor(t).numel():
                raise ValueError(f"{un!r} has {torch.as_tensor(u).numel()} values, target {tn!r} "
                                 f"{torch.as_tensor(t).numel()} (covered-only targets: subset u "
                                 f"with aligned_error_targets(..., return_mask=True))")
    return {un: {tn: sparsification_curve(t, u, num_bins=ause_bins, undefined="last")["ause"]
                 for tn, t in targets.items()}
            for un, u in uncertainties.items()}


def views_cross_ause(out_k, out_full, signals=None, covered_only=True, ause_bins=20):
    """The views figure for one row: AUSE of the k-view map's ranking signals (own channels,
    msp, entropy; ranking_signals, so the coverage rule applies) against e_red / e_irr.
    covered_only (default): pixels covered in both maps; holes are counted (n_hole), not
    scored. Returns dict(ause={signal: {e_red, e_irr}}, n_pixels, n_hole, n_e_red, n_e_irr,
    n_undefined={own channel: covered pixels with undefined u})."""
    keep, n_hole = _aligned_keep(out_k, out_full, covered_only)
    t = error_targets(out_k["correct"][keep], out_full["correct"][keep])
    sig = ranking_signals(out_k)
    names = list(sig) if signals is None else list(signals)
    miss = [s for s in names if s not in sig]
    if miss:
        raise ValueError(f"unknown signals {miss}; available {list(sig)}")
    cov = out_k["covered"].bool()[keep]
    return dict(ause=cross_ause({s: sig[s][keep] for s in names}, t, ause_bins),
                n_pixels=int(keep.sum()), n_hole=n_hole,
                n_e_red=int(t["e_red"].sum()), n_e_irr=int(t["e_irr"].sum()),
                n_undefined={k: int((cov & ~torch.isfinite(u[keep])).sum())
                             for k, u in out_k["u"].items() if k in names})


def boundary_band(gt, width=3):
    """(H, W) bool: pixels within `width` px (Chebyshev) of a label change. Ignore (-1)
    counts as its own label (a void/object edge is a boundary)."""
    g = torch.as_tensor(gt).double()[None, None]
    k = 2 * int(width) + 1
    mx = torch.nn.functional.max_pool2d(g, k, stride=1, padding=width)
    mn = -torch.nn.functional.max_pool2d(-g, k, stride=1, padding=width)
    return (mx != mn)[0, 0]


def scene_stats(values: Sequence[float]):
    """mean, std (population, N = number of scenes) and N; NaN entries are counted apart."""
    v = torch.tensor([x for x in values if x is not None and math.isfinite(x)],
                     dtype=torch.float64)
    n_nan = len(values) - v.numel()
    if v.numel() == 0:
        return dict(mean=float("nan"), std=float("nan"), n=0, n_nan=n_nan)
    return dict(mean=float(v.mean()), std=float(v.std(unbiased=False)), n=int(v.numel()),
                n_nan=n_nan)
