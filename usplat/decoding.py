"""
decoding.py — label-free decision rules for every row of Table I (attempt 08,
configs/decoding_prereg.json).

A rule turns a row's class scores s (P, K) — F . t for feature rows, log q for probs rows, as
usplat.seg_eval.class_scores — and its calibrated predictive p = softmax(beta s) into labels. Its
statistics come from the same scene, without labels, from the covered pixels of the scene's
evaluation frames (a first pass over the frames; scripts/decode_tab1.py). The predictive itself is
not changed: NLL and Brier are those of the paper's rule; the label, its stated confidence
p(label), and so mIoU, accuracy, per-class IoU, ECE and AURC follow the rule.

  base      argmax_c s_pc                                        the paper's rule
  centre    argmax_c s_pc - lam a_p m_c,  m_c = sum_p s_pc / sum_p a_p
            class-mean centring: a class's scene-wide score offset (the 'wall' hub of LSeg
            features) is removed; a_p = alpha_p for feature rows (F_p is not divided by alpha),
            1 for probs rows and the 2D row (alpha = 1)
  prior     argmax_c log p_pc - tau log pi_c,  pi_c = mean_p p_pc
            label-free prior balancing (logit adjustment with the scene's own predicted prior)
  eiou      argmax_c log p_pc + b_c, with b maximising the expected mean IoU of the scene under p
            (Nowozin, "Optimal decisions from probabilistic models: the intersection-over-union
            case", CVPR 2014): coordinate ascent over a fixed grid on a fixed pixel subsample,
            on the run's device (scripts/decode_tab1.py passes its GPU)

A tie of the top two decision scores within TIE_TOL is no decision (-1), as in seg_eval.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch

TIE_TOL = 1e-12
RULES = ("base", "centre", "prior", "eiou")
EIOU_GRID = tuple(x / 2.0 for x in range(-8, 9))          # b_c in {-4, -3.5, ..., 4}
EIOU_SWEEPS = 3
EIOU_SUBSAMPLE = 100_000


class SceneStats:
    """Label-free per-row statistics of one scene over its covered evaluation pixels."""

    def __init__(self, K: int, n_frames: int, subsample: int = EIOU_SUBSAMPLE, seed: int = 0):
        self.K = int(K)
        self.sum_s = torch.zeros(self.K, dtype=torch.float64)
        self.sum_a = 0.0
        self.sum_p = torch.zeros(self.K, dtype=torch.float64)
        self.n = 0
        self.per_frame = max(1, int(math.ceil(subsample / max(1, int(n_frames)))))
        self.gen = torch.Generator().manual_seed(int(seed))
        self.sub = []

    @torch.no_grad()
    def add(self, s, p, a):
        """s, p (M, K) and a (M,) of the covered pixels of one frame."""
        if s.shape[0] == 0:
            return
        s, p, a = s.double().cpu(), p.double().cpu(), a.double().cpu()
        self.sum_s += s.sum(0)
        self.sum_a += float(a.sum())
        self.sum_p += p.sum(0)
        self.n += int(s.shape[0])
        k = min(self.per_frame, int(s.shape[0]))
        idx = torch.randperm(int(s.shape[0]), generator=self.gen)[:k]
        self.sub.append(p[idx].float())

    def finalize(self):
        if self.n == 0:
            raise ValueError("no covered pixel in the scene")
        sub = torch.cat(self.sub) if self.sub else torch.zeros(0, self.K)
        return dict(m=self.sum_s / max(self.sum_a, 1e-30), pi=self.sum_p / self.n, sub=sub,
                    n_pixels=self.n, n_subsample=int(sub.shape[0]))


def expected_miou(logp, b, p=None):
    """Expected mean IoU over the K classes of the decisions argmax(logp + b) under p = exp(logp)
    (Nowozin 2014): E[TP_c] / (E[G_c] + |D_c| - E[TP_c]). p: exp(logp), if already computed."""
    p = logp.exp() if p is None else p
    lab = (logp + b).argmax(1)
    K = logp.shape[1]
    onehot = torch.zeros_like(p).scatter_(1, lab[:, None], 1.0)
    tp = (p * onehot).sum(0)
    g = p.sum(0)
    d = onehot.sum(0)
    iou = tp / (g + d - tp).clamp_min(1e-12)
    return float(iou.sum() / K)


@torch.no_grad()
def eiou_offsets(sub_p, grid=EIOU_GRID, sweeps=EIOU_SWEEPS, device=None):
    """b (K,) on the CPU by coordinate ascent on the fixed grid (start 0; a value replaces the
    current one only if it strictly raises the expected mIoU; classes in index order). device: where
    the search runs (default: that of sub_p); a GPU is about 100x faster for K = 101."""
    dev = sub_p.device if device is None else torch.device(device)
    sub_p = sub_p.to(dev, torch.float64)
    K = sub_p.shape[1]
    b = torch.zeros(K, dtype=torch.float64, device=dev)
    if sub_p.shape[0] == 0:
        return b.cpu(), float("nan")
    logp = sub_p.clamp_min(1e-12).log()
    p = logp.exp()
    best = expected_miou(logp, b, p)
    for _ in range(int(sweeps)):
        moved = False
        for c in range(K):
            keep = float(b[c])
            for v in grid:
                if v == keep:
                    continue
                b[c] = v
                e = expected_miou(logp, b, p)
                if e > best + 1e-12:
                    best, keep, moved = e, v, True
            b[c] = keep
        if not moved:
            break
    return b.cpu(), best


def scene_parameters(stats: SceneStats, rule: str, value=None, device=None):
    """The rule's per-scene parameter vector (K,) from the scene statistics (device: where the
    expected-IoU search runs)."""
    st = stats if isinstance(stats, dict) else stats.finalize()
    if rule == "base":
        return None
    if rule == "centre":
        return float(value) * st["m"]
    if rule == "prior":
        return float(value) * st["pi"].clamp_min(1e-12).log()
    if rule == "eiou":
        return eiou_offsets(st["sub"], device=device)[0]
    raise ValueError(f"unknown rule {rule!r}")


@torch.no_grad()
def decide(rule, s, p, covered, a=None, param=None):
    """(labels (M,) long, -1 = no decision; stated confidence p(label) (M,); tied (M,) bool) of
    the M pixels. s, p (M, K); covered (M,) bool; a (M,) the alpha weights of 'centre'."""
    K = s.shape[1]
    covered = covered.to(s.device)                      # every input on the scores' device
    p = p.to(s.device)
    if rule == "base":
        score = s.double()
    elif rule == "centre":
        aw = torch.ones(s.shape[0], dtype=torch.float64) if a is None else a.double().cpu()
        score = s.double() - aw.to(s.device)[:, None] * param.to(s.device, torch.float64)[None]
    elif rule in ("prior", "eiou"):
        lp = p.double().clamp_min(1e-12).log()
        score = lp - param.to(p.device, torch.float64)[None] if rule == "prior" else \
            lp + param.to(p.device, torch.float64)[None]
    else:
        raise ValueError(f"unknown rule {rule!r}")
    top = score.max(1)
    tied = (score >= top.values[:, None] - TIE_TOL).sum(1) > 1
    lab = torch.where(covered & ~tied, top.indices, torch.full_like(top.indices, -1))
    # no decision: a tie keeps max p (as evaluate_view's conf), an uncovered pixel 1/K (uniform p)
    conf = torch.where(lab >= 0, p.gather(1, lab.clamp_min(0)[:, None]).squeeze(1),
                       torch.where(covered, p.max(1).values, torch.full_like(p[:, 0], 1.0 / K)))
    return lab, conf.float(), covered & tied


def decision_metrics(gt, lab, conf, covered, u: Optional[Dict[str, torch.Tensor]] = None,
                     num_bins=15, ause_bins=20, class_names=None):
    """The label metrics of one row and rule over a scene's labelled pixels, in the layout of
    usplat.seg_eval.summarize ('all' and 'covered'): mIoU, acc, macc, per-class IoU, ECE and the
    reliability of p(label), AURC of 1 - p(label), AUSE of the row's own channels u against the
    rule's 0/1 error. NLL and Brier are not here: they belong to the predictive."""
    from usplat.evaluation import expected_calibration_error
    from usplat.seg_eval import (class_accuracy, confusion_iou, ranking_metrics, reliability,
                                 undefined_key)
    correct = (lab == gt) & covered
    out = {}
    for variant in ("all", "covered"):
        sel = torch.ones_like(covered) if variant == "all" else covered
        M = int(sel.sum())
        if M == 0:
            out[variant] = dict(n_pixels=0)
            continue
        g, l, c, cv = gt[sel], lab[sel], correct[sel], covered[sel]
        cf = conf[sel]
        iou, gc = confusion_iou(g, l)
        present = gc > 0
        names = class_names if class_names is not None else [str(i) for i in range(iou.numel())]
        err = 1.0 - c.double()
        msp = torch.where(cv, 1.0 - cf.double(), torch.ones_like(cf, dtype=torch.float64))
        ranking = {"msp": ranking_metrics(err, msp, ause_bins, 0, int((~cv).sum()))}
        for k, uu in (u or {}).items():
            uu = uu[sel].double()
            key = torch.where(torch.isfinite(uu), uu, torch.full_like(uu, undefined_key(uu)))
            key = torch.where(cv, key, torch.full_like(uu, float("inf")))
            ranking[k] = ranking_metrics(err, key, ause_bins, int((cv & ~torch.isfinite(uu)).sum()),
                                         int((~cv).sum()))
        out[variant] = dict(
            n_pixels=M, coverage=float(cv.double().mean()), n_uncovered=int((~cv).sum()),
            n_tied=int(((l < 0) & cv).sum()), miou=float(iou[present].mean()),
            acc=float(c.double().mean()), macc=float(class_accuracy(g, c, gc)[present].mean()),
            n_classes=int(present.sum()),
            iou={names[i]: float(iou[i]) for i in torch.nonzero(present).squeeze(1).tolist()},
            ece=expected_calibration_error(cf, c, num_bins), reliability=reliability(cf, c, num_bins),
            ranking=ranking, aurc=ranking["msp"]["aurc"])
    return out
