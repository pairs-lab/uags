"""
evaluation.py — metrics for uncertainty ranking, change detection and calibration.

Uncertainty RANKING (not calibration):
  sparsification curve: drop the most-uncertain fraction f of elements, report the mean
  error of the rest; oracle = drop by the error itself.
  AUSE = mean over f in {0, 1/B, ..., (B-1)/B} of (model(f) - oracle(f)), in error units
  (mean-gap definition; `normalize=True` divides by the mean error).
  TIES: view counts give many equal uncertainties. Within a block of equal uncertainty the
  errors are replaced by the block mean, which equals the expectation over a random order
  of the tied elements. Sorting would otherwise break ties by array position.

Change DETECTION: AUROC (tie-aware Mann-Whitney) and average precision (step-wise over
distinct thresholds, the scikit-learn definition).

UNDEFINED values are never dropped silently (a method must not be graded only on what it
chose to score): undefined="last" ranks a NaN score as the least changed (-inf) and a NaN
uncertainty as the most uncertain (+inf, removed first); undefined="drop" removes them and
is only for diagnostics. sparsification_curve returns n_undefined; callers of auroc /
average_precision must report the count of NaN scores next to the value.

CALIBRATION of a K-way predictive p (P, K) against labels: ECE (equal-width bins on the
max probability), NLL, Brier, accuracy, and the area under the risk-coverage curve (AURC).
"""
from __future__ import annotations
import torch


def _flat(x):
    return torch.as_tensor(x, dtype=torch.float64).flatten()


def _tie_mean(sorted_key, values):
    """Replace values by their mean within runs of equal sorted_key."""
    _, inv, counts = torch.unique_consecutive(sorted_key, return_inverse=True,
                                              return_counts=True)
    sums = torch.zeros(counts.numel(), dtype=values.dtype).index_add_(0, inv, values)
    return (sums / counts.to(values.dtype))[inv]


def _undef_scores(s, fill, undefined):
    bad = ~torch.isfinite(s)
    if undefined == "drop":
        return s, ~bad, int(bad.sum())
    if undefined == "last":
        return torch.where(bad, torch.full_like(s, fill), s), torch.ones_like(bad), int(bad.sum())
    raise ValueError(f"undefined must be 'last' or 'drop', got {undefined!r}")


@torch.no_grad()
def sparsification_curve(errors, uncertainty, num_bins=20, normalize=False,
                         undefined="last"):
    e, u = _flat(errors), _flat(uncertainty)
    if not torch.isfinite(e).all():
        raise ValueError("errors must be finite")
    u, ok, n_undef = _undef_scores(u, float("inf"), undefined)
    e, u = e[ok], u[ok]
    M = e.numel()
    if M == 0:
        raise ValueError("no finite (error, uncertainty) pairs")
    order = torch.argsort(u, descending=True, stable=True)
    e_model = _tie_mean(u[order], e[order])            # drop most-uncertain first
    e_oracle = torch.sort(e, descending=True).values   # drop highest-error first

    def curve(es):
        # mean of the remaining elements after dropping the first k: suffix means
        suffix = torch.flip(torch.cumsum(torch.flip(es, [0]), 0), [0])
        ks = [int(i * M / num_bins) for i in range(num_bins)]
        return torch.stack([suffix[k] / (M - k) for k in ks])

    fracs = torch.arange(num_bins, dtype=torch.float64) / num_bins
    model, oracle = curve(e_model), curve(e_oracle)
    norm = e.mean().item() if normalize else 1.0
    ause = ((model - oracle).mean() / (norm + 1e-12)).item()
    return dict(fractions=fracs, model_curve=model, oracle_curve=oracle,
                random_curve=torch.full_like(model, e.mean().item()), ause=ause,
                random_gap=((e.mean() - oracle).mean() / (norm + 1e-12)).item(),
                n_undefined=n_undef)


@torch.no_grad()
def auroc(scores, labels, undefined="last"):
    """P(score_pos > score_neg) + 0.5 P(tie); labels in {0,1}. NaN scores rank last."""
    s, y = _flat(scores), _flat(labels)
    s, ok, _ = _undef_scores(s, float("-inf"), undefined)
    s, y = s[ok], y[ok]
    order = torch.argsort(s, stable=True)
    ranks_sorted = torch.arange(1, s.numel() + 1, dtype=torch.float64)
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = _tie_mean(s[order], ranks_sorted)
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


@torch.no_grad()
def average_precision(scores, labels, undefined="last"):
    """sum_k (R_k - R_{k-1}) P_k over distinct score thresholds (descending)."""
    s, y = _flat(scores), _flat(labels)
    s, ok, _ = _undef_scores(s, float("-inf"), undefined)
    s, y = s[ok], y[ok]
    npos = y.sum()
    if npos == 0:
        return float("nan")
    order = torch.argsort(s, descending=True, stable=True)
    s, y = s[order], y[order]
    tp, fp = torch.cumsum(y, 0), torch.cumsum(1 - y, 0)
    last = torch.ones_like(s, dtype=torch.bool)       # last index of each tie block
    last[:-1] = s[1:] != s[:-1]
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp)
    rec = tp / npos
    rec_prev = torch.cat([torch.zeros(1, dtype=rec.dtype), rec[:-1]])
    return float(((rec - rec_prev) * prec).sum())


@torch.no_grad()
def expected_calibration_error(confidence, correct, num_bins=15):
    """Binary-style ECE of a confidence in [0,1] against 0/1 correctness."""
    c, y = _flat(confidence), _flat(correct)
    idx = torch.clamp((c * num_bins).long(), max=num_bins - 1)
    ece = 0.0
    for b in range(num_bins):
        m = idx == b
        if m.any():
            ece += m.double().mean().item() * abs(y[m].mean().item() - c[m].mean().item())
    return ece


@torch.no_grad()
def multiclass_calibration(probs, labels, num_bins=15):
    """probs (M, K) rows summing to 1, labels (M,) int. -> dict of metrics."""
    p = torch.as_tensor(probs, dtype=torch.float64)
    y = torch.as_tensor(labels).long()
    conf, pred = p.max(dim=1)
    correct = (pred == y).double()
    py = p.gather(1, y[:, None]).squeeze(1)
    onehot = torch.zeros_like(p).scatter_(1, y[:, None], 1.0)
    return dict(
        acc=correct.mean().item(),
        ece=expected_calibration_error(conf, correct, num_bins),
        nll=(-torch.log(py.clamp_min(1e-12))).mean().item(),
        brier=((p - onehot) ** 2).sum(1).mean().item(),
        aurc=aurc(1.0 - conf, 1.0 - correct),
    )


@torch.no_grad()
def aurc(uncertainty, errors):
    """Area under the risk-coverage curve: keep the most-certain fraction, risk = mean
    error of what is kept; averaged over all coverages (ties averaged)."""
    e, u = _flat(errors), _flat(uncertainty)
    order = torch.argsort(u, stable=True)               # most certain first
    es = _tie_mean(u[order], e[order])
    risk = torch.cumsum(es, 0) / torch.arange(1, es.numel() + 1, dtype=torch.float64)
    return risk.mean().item()


@torch.no_grad()
def semantic_error_vs_text(mu_post, gt_text_dirs):
    """Per-element semantic error = 1 - cos(mu_post, ground-truth text dir).
    mu_post, gt_text_dirs: (M, d) unit vectors. returns (M,)."""
    return 1.0 - (mu_post * gt_text_dirs).sum(-1)
