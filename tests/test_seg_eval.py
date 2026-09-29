"""K10b: segmentation + calibration evaluation (usplat.seg_eval, usplat.rows, eval_seg.py).

Synthetic scene: one Gaussian per pixel centre (sigma 0.3 px, opacity 0.95) on a 16x16 grid,
four classes in quadrants, orthonormal texts; the mock rasterizer gives the exact weights.
 * perfect features: mIoU 1, accuracy 1 and ECE ~ 0 at a large beta; one render per row;
 * coverage rule: pixels under a hole (alpha < tau_cov) are wrong, get the uniform
   predictive, MSP u = 1 and u = +inf for own channels; the covered variant excludes them;
 * AUSE of an oracle uncertainty is 0 (and below MSP's);
 * undefined (NaN) Gaussian uncertainties render as undefined where they dominate;
 * e_red / e_irr on a toy case, and cross-AUSE of an oracle is 0;
 * the P2 path (fit_betas with fit_adaptive on a real field) runs and does not lose to P0
   on the fitting split (P2 nests P0);
 * the CLI runs end to end on CPU (mock backend, 7-Scenes layout, vocabulary with ignore
   and merges, text cache) and writes a run record.
Regression tests of the review fixes (fix-segeval):
 * each row is scored with its own beta, P2 with its own channels (brute force);
 * F is not divided by alpha; alpha = tau_cov is covered, alpha < alpha_eps never;
 * a tied maximum (zero evidence) is no decision, and metrics do not depend on text order;
 * ranking tiers: uncovered > undefined > finite (hole + NaN channel);
 * ECE / NLL / Brier / accuracy of both variants against brute force;
 * the beta fit uses covered pixels only;
 * input validation (duplicate names, P2 params on a plain row, NaN buffer, inf u, missing
   beta of a probs row, negative labels);
 * views cross-AUSE on covered pixels by default (holes counted apart);
 * the CLI passes --undef-frac to the fit, records it, refuses a mismatching --betas file,
   and refuses rows without a fitted beta.
"""
import importlib.util
import json
import math
import os
import pathlib

import pytest

import numpy as np
import torch
import torch.nn.functional as Fn

from usplat.mock_rasterizer import MockFeatureRasterizer
from usplat.rows import TensorRow, field_rows, freeze
import usplat.seg_eval as seg_eval
from usplat.seg_eval import (View, evaluate_view, evaluate_views, SegAccumulator, fit_betas,
                             error_targets, aligned_error_targets, cross_ause, boundary_band,
                             render_row, confusion_iou, ranking_signals, views_cross_ause,
                             BetaFitter)
from usplat.vmf_field import DirectionalSemanticField, FieldConfig

REPO = pathlib.Path(__file__).resolve().parents[1]


def _scene(H=16, W=16, K=4, d=8, hole=None, seed=0):
    """Returns rasterizer, per-Gaussian labels, per-pixel GT (H, W), text (K, d)."""
    g = torch.Generator().manual_seed(seed)
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    cen = torch.stack([xs, ys], -1).reshape(-1, 2).float()
    quad = lambda x, y: (y >= H // 2).long() * 2 + (x >= W // 2).long()
    keep = torch.ones(H * W, dtype=torch.bool)
    if hole is not None:                               # (x0, y0): no Gaussians beyond
        keep = ~((cen[:, 0] >= hole[0]) & (cen[:, 1] >= hole[1]))
    means = cen[keep]
    N = means.shape[0]
    lab_g = quad(means[:, 0], means[:, 1])
    pix = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy"),
                      -1).reshape(-1, 2).float()      # row-major (H, W) pixel order
    ras = MockFeatureRasterizer(means, torch.full((N,), 0.3), torch.full((N,), 0.95),
                                torch.rand(N, generator=g), pix)
    gt = quad(pix[:, 0], pix[:, 1]).reshape(H, W)
    text = torch.eye(d)[:K]
    return ras, lab_g, gt, text


class _FixedW:
    """Explicit weights W (P, N): render = W @ buf, alpha_p = sum_n W[p, n]."""
    dtype = torch.float32

    def __init__(self, W):
        self.W = W.float()

    def __call__(self, buf):
        return self.W @ buf


class _Count:
    def __init__(self, ras):
        self.ras, self.calls = ras, 0

    def __call__(self, buf):
        self.calls += 1
        return self.ras(buf)


def test_perfect_features_miou_ece():
    ras, lab, gt, text = _scene()
    K = text.shape[0]
    rows = [TensorRow("perfect", "feature", buf=text[lab]),
            TensorRow("onehot", "probs", buf=Fn.one_hot(lab, K).float())]
    cnt = _Count(ras)
    out = evaluate_view(rows, cnt, text, gt, {"perfect": 200.0}, tau_cov=0.5,
                        raw_probs={"onehot"})
    assert cnt.calls == len(rows), cnt.calls                  # one render per row
    acc = SegAccumulator(class_names=["a", "b", "c", "d"])
    acc.add(out)
    m = acc.metrics()
    for r in ("perfect", "onehot"):
        a = m[r]["all"]
        assert a["miou"] == 1.0 and a["acc"] == 1.0 and a["coverage"] == 1.0, a
        assert a["n_classes"] == 4 and set(a["iou"]) == {"a", "b", "c", "d"}
    assert m["perfect"]["all"]["ece"] < 1e-6, m["perfect"]["all"]["ece"]
    assert m["perfect"]["all"]["nll"] < 1e-6
    print(f"PASS perfect features: mIoU 1, ECE {m['perfect']['all']['ece']:.2e} at beta 200; "
          f"probs row mIoU 1 (ECE {m['onehot']['all']['ece']:.4f})")


def test_coverage_rule():
    ras, lab, gt, text = _scene(hole=(12, 12))
    K = text.shape[0]
    torch.manual_seed(0)
    u = 0.5 * torch.rand(lab.numel())
    rows = [TensorRow("perfect", "feature", buf=text[lab], unc={"u_rand": u}),
            TensorRow("onehot", "probs", buf=Fn.one_hot(lab, K).float())]
    out = evaluate_view(rows, ras, text, gt, {"perfect": 200.0, "onehot": 1.0}, tau_cov=0.5)
    hole = torch.zeros(16, 16, dtype=torch.bool)
    hole[12:, 12:] = True
    hole = hole.reshape(-1)
    for r in ("perfect", "onehot"):
        o = out[r]
        assert torch.equal(o["covered"], ~hole)                # the 4x4 hole only
        assert not o["correct"][hole].any() and (o["pred"][hole] == -1).all()
        assert o["correct"][~hole].all()
        assert torch.allclose(o["entropy"][hole], torch.full((16,), math.log(K)), atol=1e-6)
        assert torch.allclose(o["nll"][hole], torch.full((16,), math.log(K)), atol=1e-6)
    acc = SegAccumulator()
    acc.add(out)
    m = acc.metrics()["perfect"]
    assert abs(m["all"]["acc"] - 240 / 256) < 1e-12 and m["all"]["miou"] < 1.0
    assert m["covered"]["acc"] == 1.0 and m["covered"]["miou"] == 1.0
    assert m["covered"]["n_pixels"] == 240 and abs(m["all"]["coverage"] - 240 / 256) < 1e-12
    sig = ranking_signals(acc.arrays("perfect"))
    ur = acc.arrays("perfect")["u"]["u_rand"]
    top = ur[torch.isfinite(ur)].max()
    assert (sig["msp"][hole] == 1.0).all()                     # rows without a channel: u = 1
    assert torch.isinf(sig["u_rand"][hole]).all()              # own channel: ranked first
    assert torch.equal(sig["u_rand"][~hole], ur[~hole].double())
    # the only errors are the uncovered pixels and MSP puts them first
    assert abs(m["all"]["ranking"]["msp"]["ause"]) < 1e-12
    assert abs(m["all"]["ranking"]["u_rand"]["ause"]) < 1e-12  # own channel: uncovered first too
    assert m["all"]["n_uncovered"] == 16 and m["covered"]["n_uncovered"] == 0
    print(f"PASS coverage: 16 uncovered pixels wrong (acc {m['all']['acc']:.4f}, covered-only "
          f"1.0), uniform predictive, MSP u = 1, own u = +inf (finite max {float(top):.3f})")


def test_oracle_ause_zero():
    ras, lab, gt, text = _scene()
    wrong = torch.zeros(lab.numel(), dtype=torch.bool)
    wrong[torch.randperm(lab.numel(), generator=torch.Generator().manual_seed(3))[:40]] = True
    feat_lab = torch.where(wrong, (lab + 1) % 4, lab)
    rows = [TensorRow("noisy", "feature", buf=text[feat_lab],
                      unc={"oracle": wrong.float(), "const": torch.zeros(lab.numel())})]
    acc = evaluate_views(rows, [View(ras, gt, text)], {"noisy": 50.0}, tau_cov=0.5)
    m = acc.metrics()["noisy"]["all"]
    n_err = 256 - round(m["acc"] * 256)
    assert n_err == 40, n_err
    rk = m["ranking"]
    assert abs(rk["oracle"]["ause"]) < 1e-12, rk["oracle"]
    assert rk["msp"]["ause"] > 0.05                            # confidently wrong: MSP fails
    assert abs(rk["const"]["ause"] - rk["const"]["random_gap"]) < 1e-12
    print(f"PASS AUSE: oracle {rk['oracle']['ause']:.1e}, MSP {rk['msp']['ause']:.4f}, "
          f"constant = random gap {rk['const']['random_gap']:.4f}")


def test_undefined_channel_rendering():
    ras, lab, gt, text = _scene()
    u = torch.full((lab.numel(),), 0.3)
    u[lab == 0] = float("nan")                               # quadrant 0 undefined
    rr = render_row(TensorRow("r", "feature", buf=text[lab], unc={"u": u}), ras)
    q0 = (gt.reshape(-1) == 0)
    assert torch.isnan(rr.unc["u"][q0]).all()
    assert torch.allclose(rr.unc["u"][~q0], torch.full(((~q0).sum(),), 0.3), atol=1e-5)
    acc = evaluate_views([TensorRow("r", "feature", buf=text[lab], unc={"u": u})],
                         [View(ras, gt, text)], {"r": 50.0}, tau_cov=0.5)
    assert acc.metrics()["r"]["all"]["ranking"]["u"]["n_undefined"] == int(q0.sum())
    print("PASS undefined channel: NaN where undefined Gaussians dominate, counted in AUSE")


def test_error_targets_and_cross_ause():
    ck = torch.tensor([True, False, False, True, False])
    cf = torch.tensor([True, True, False, False, True])
    t = error_targets(ck, cf)
    assert t["e_red"].tolist() == [0, 1, 0, 0, 1]
    assert t["e_irr"].tolist() == [0, 0, 1, 0, 0]
    ca = cross_ause({"u_red": t["e_red"], "u_irr": t["e_irr"]}, t)
    assert abs(ca["u_red"]["e_red"]) < 1e-12 and abs(ca["u_irr"]["e_irr"]) < 1e-12
    assert ca["u_red"]["e_irr"] > 0 and ca["u_irr"]["e_red"] > 0
    # aligned outputs of one row: k-view map (8 wrong Gaussians) vs full map (4 of them)
    ras, lab, gt, text = _scene()
    wk = torch.zeros(lab.numel(), dtype=torch.bool); wk[:8] = True
    wf = torch.zeros(lab.numel(), dtype=torch.bool); wf[:4] = True
    ok = evaluate_view([TensorRow("r", "feature", buf=text[torch.where(wk, (lab + 1) % 4, lab)])],
                       ras, text, gt, {"r": 50.0}, 0.5)["r"]
    of = evaluate_view([TensorRow("r", "feature", buf=text[torch.where(wf, (lab + 1) % 4, lab)])],
                       ras, text, gt, {"r": 50.0}, 0.5)["r"]
    t2 = aligned_error_targets(ok, of)
    assert int(t2["e_red"].sum()) == 4 and int(t2["e_irr"].sum()) == 4
    band = boundary_band(gt, width=1)
    assert band[0, 0].item() is False and band[7, 7].item() is True and band[8, 8].item() is True
    print("PASS e_red/e_irr toy case; oracle cross-AUSE 0; boundary band")


def test_p2_path_runs():
    torch.manual_seed(0)
    ras, lab, gt, text = _scene(d=8)
    N, P = lab.numel(), gt.numel()
    f = DirectionalSemanticField(N, 8, FieldConfig(change_test=False, view_mass_frac=1e-3))
    f.set_centre(text.mean(0))
    gpix = gt.reshape(-1)
    left = torch.arange(P) % 16 < 8
    for v in range(6):
        flip = torch.rand(P) < 0.3                            # 30% of pixels see another class
        cls = torch.where(flip, torch.randint(0, 4, (P,)), gpix)
        z = Fn.normalize(text[cls] + 0.4 * torch.randn(P, 8), dim=-1)
        valid = torch.ones(P) if v < 2 else (~left).float()   # left half: 2 views only
        f.update_view(ras, z, pixel_valid=valid)
    rows = field_rows(f)
    assert [r.name for r in rows] == ["ours_P0", "ours_P1", "ours_P2"]
    view = View(ras, gt, text)
    betas = fit_betas(rows, [view], tau_cov=0.5)
    assert isinstance(betas["ours_P0"], float) and isinstance(betas["ours_P1"], float)
    assert set(betas["ours_P2"]) == {"beta0", "a", "b", "g"}
    acc = evaluate_views(rows, [view], betas, tau_cov=0.5)
    m = acc.metrics()
    u_ale = acc.arrays("ours_P2")["u"]["u_ale"]
    assert torch.isnan(u_ale).any() and torch.isfinite(u_ale).any()   # g term exercised
    for r in ("ours_P0", "ours_P1", "ours_P2"):
        for k in ("miou", "acc", "ece", "nll", "brier", "aurc"):
            assert math.isfinite(m[r]["covered"][k]), (r, k)
    nll0, nll2 = m["ours_P0"]["covered"]["nll"], m["ours_P2"]["covered"]["nll"]
    assert nll2 <= nll0 + 1e-4, (nll0, nll2)                  # P2 nests P0 on the fit split
    fr = freeze(rows[2])
    assert fr.adaptive and set(fr.unc) == {"u_epi", "u_ale"}
    print(f"PASS P2: betas P0 {betas['ours_P0']:.2f}, P2 {betas['ours_P2']}; val NLL "
          f"P0 {nll0:.4f} >= P2 {nll2:.4f}")


def test_confusion_iou_miss_is_not_false_positive():
    gt = torch.tensor([0, 0, 1, 1])
    pred = torch.tensor([0, -1, 1, 0])
    iou, gc = confusion_iou(gt, pred)
    # class 0: TP 1, GT 2, PRED 2 (pixels 0, 3) -> 1/3; class 1: TP 1, GT 2, PRED 1 -> 1/2.
    # The uncovered pixel (-1) is a miss of class 0 and a false positive of no class.
    assert iou.tolist() == [1 / 3, 0.5] and gc.tolist() == [2, 2]


# ------------------------------------------------------- regression tests (review fixes)
def _render_u(W, u, undef_frac=0.5):
    """Brute-force rendered u (defined part only; NaN where undefined dominates)."""
    fin = torch.isfinite(u)
    alpha = W.sum(1).double()
    dm = W.double() @ fin.double()
    num = W.double() @ torch.where(fin, u, torch.zeros_like(u)).double()
    ok = (alpha > 1e-6) & (dm >= (1 - undef_frac) * alpha) & (dm > 1e-6)
    return torch.where(ok, num / dm.clamp_min(1e-6), torch.full_like(num, float("nan")))


def test_each_row_uses_its_own_beta():
    """Fairness: two feature rows with the same buffer and different betas are each scored
    with their own beta; the P2 row uses beta0 exp(-a u_epi - b u_ale - g 1[undef]) with
    the right channels (brute force from the explicit weights)."""
    ras, lab, gt, text = _scene()
    N, K = lab.numel(), text.shape[0]
    g = torch.Generator().manual_seed(1)
    noisy = torch.where(torch.rand(N, generator=g) < 0.2, (lab + 1) % 4, lab)
    buf = 0.5 * text[noisy]
    ue, ua = torch.rand(N, generator=g), torch.rand(N, generator=g)
    ua[lab == 0] = float("nan")                                # u_ale undefined: g term
    rows = [TensorRow("a", "feature", buf=buf), TensorRow("b", "feature", buf=buf),
            TensorRow("c", "feature", adaptive=True, buf=buf, unc={"u_epi": ue, "u_ale": ua})]
    p2 = dict(beta0=20.0, a=3.0, b=1.0, g=0.5)
    out = evaluate_view(rows, ras, text, gt, {"a": 1.0, "b": 30.0, "c": p2}, tau_cov=0.5)
    W = ras.weights()
    s = ((W @ buf) @ text.t()).double()
    y = gt.reshape(-1)
    rue, rua = _render_u(W, ue), _render_u(W, ua)
    undef = torch.isnan(rua)
    assert undef.any() and (~undef).any()

    def p2beta(e, a):
        return 20.0 * torch.exp(-3.0 * e - 1.0 * torch.nan_to_num(a) - 0.5 * undef.double())
    for name, b in (("a", 1.0), ("b", 30.0), ("c", p2beta(rue, rua)[:, None])):
        p = torch.softmax(b * s, dim=1)
        assert torch.allclose(out[name]["conf"].double(), p.max(1).values, atol=1e-5), name
        assert torch.allclose(out[name]["nll"].double(), -torch.log(p.gather(1, y[:, None])[:, 0]),
                              atol=1e-4), name
    swapped = torch.softmax(p2beta(rua, rue)[:, None] * s, dim=1).max(1).values
    assert not torch.allclose(out["c"]["conf"].double(), swapped, atol=1e-3)
    assert not torch.allclose(out["a"]["conf"], out["b"]["conf"], atol=1e-3)
    print("PASS own beta per row: a (1.0), b (30.0), P2 (brute force, channels not swapped)")


def test_feature_not_divided_by_alpha_and_coverage_thresholds():
    """F_p is not divided by alpha (weak opacity flattens the softmax); alpha = tau_cov is
    covered (>=); alpha in (alpha_eps, tau_cov) is not; alpha <= alpha_eps never is."""
    alpha = torch.tensor([1.0, 0.5, 0.3, 1e-8, 0.0])
    ras = _FixedW(torch.diag(alpha))
    text = torch.eye(4)[:2]
    buf = text[torch.zeros(5, dtype=torch.long)]                # every Gaussian: class 0
    gt = torch.zeros(5, dtype=torch.long)
    row = [TensorRow("r", "feature", buf=buf)]
    o = evaluate_view(row, ras, text, gt, {"r": 4.0}, tau_cov=0.5)["r"]
    assert o["covered"].tolist() == [True, True, False, False, False]
    assert o["correct"].tolist() == [True, True, False, False, False]
    conf = o["conf"].double()
    assert conf[1] < conf[0]                                    # alpha 0.5: flatter
    assert abs(conf[0] - torch.sigmoid(torch.tensor(4.0 * 1.0, dtype=torch.float64))) < 1e-6
    assert abs(conf[1] - torch.sigmoid(torch.tensor(4.0 * 0.5, dtype=torch.float64))) < 1e-6
    o0 = evaluate_view(row, ras, text, gt, {"r": 4.0}, tau_cov=0.0)["r"]
    assert o0["covered"].tolist() == [True, True, True, False, False]   # alpha_eps floor
    print(f"PASS alpha: conf {conf[0]:.4f} at alpha 1 > {conf[1]:.4f} at alpha 0.5; "
          f"alpha = tau_cov covered; alpha < alpha_eps never")


def test_ties_are_no_decision_and_order_invariant():
    """Zero evidence (F_p = 0, uniform probabilities) ties every class: no decision (pred -1,
    wrong, counted), so the metrics do not change when the text queries are permuted."""
    ras, lab, gt, text = _scene()
    N, K = lab.numel(), text.shape[0]
    right = (lab == 1) | (lab == 3)                             # right half never observed
    half = torch.where(right[:, None], torch.zeros(N, 8), text[lab])
    probs_half = torch.where(right[:, None], torch.full((N, K), 1.0 / K),
                             Fn.one_hot(lab, K).float())

    def run(perm):
        inv = perm.argsort()
        rows = [TensorRow("zero", "feature", buf=torch.zeros(N, 8)),
                TensorRow("half", "feature", buf=half),
                TensorRow("uniform", "probs", buf=torch.full((N, K), 1.0 / K)),
                TensorRow("css_half", "probs", buf=probs_half[:, perm])]
        betas = {"zero": 10.0, "half": 10.0, "uniform": 1.0, "css_half": 1.0}
        acc = evaluate_views(rows, [View(ras, inv[gt], text[perm])], betas, tau_cov=0.5)
        return acc.metrics()
    m1 = run(torch.arange(K))
    m2 = run(torch.tensor([1, 0, 3, 2]))
    for r in ("zero", "uniform"):
        a = m1[r]["all"]
        assert a["acc"] == 0.0 and a["miou"] == 0.0 and a["n_tied"] == 256, (r, a["n_tied"])
    # the unobserved right half (128 px) ties, except where the left half leaks in (col 8)
    for r in ("half", "css_half"):
        assert 96 <= m1[r]["all"]["n_tied"] <= 128, (r, m1[r]["all"]["n_tied"])
    for r in m1:
        for k in ("acc", "miou", "macc", "ece", "nll", "brier", "n_tied"):
            assert abs(m1[r]["all"][k] - m2[r]["all"][k]) < 1e-9, (r, k, m1[r]["all"][k],
                                                                    m2[r]["all"][k])
    print(f"PASS ties: zero row acc 0 (256 tied); half row acc {m1['half']['all']['acc']:.4f} "
          f"({m1['half']['all']['n_tied']} tied) in both text orders")


def test_ranking_tiers_hole_and_undefined():
    """Own channels rank uncovered (always wrong) > covered undefined (NaN) > finite, so an
    own channel is scored like MSP on uncovered pixels and not penalised for NaN."""
    ras, lab, gt, text = _scene(hole=(12, 12))
    u = torch.zeros(lab.numel())
    u[lab == 0] = float("nan")                                 # a correct quadrant undefined
    rows = [TensorRow("perfect", "feature", buf=text[lab], unc={"u": u})]
    acc = evaluate_views(rows, [View(ras, gt, text)], {"perfect": 200.0}, tau_cov=0.5)
    d = acc.arrays("perfect")
    sig = ranking_signals(d)["u"]
    unc = ~d["covered"]
    und = d["covered"] & torch.isnan(d["u"]["u"])
    fin = d["covered"] & ~und
    assert int(unc.sum()) == 16 and int(und.sum()) == 64
    assert torch.isinf(sig[unc]).all() and torch.isfinite(sig[~unc]).all()
    assert sig[und].min() > sig[fin].max()
    m = acc.metrics()["perfect"]
    r = m["all"]["ranking"]["u"]
    assert abs(r["ause"]) < 1e-12 and abs(m["all"]["ranking"]["msp"]["ause"]) < 1e-12, r
    assert r["n_undefined"] == 64 and r["n_uncovered"] == 16
    rc = m["covered"]["ranking"]["u"]
    assert rc["n_undefined"] == 64 and rc["n_uncovered"] == 0
    print(f"PASS tiers: own-channel AUSE {r['ause']:.1e} = MSP (16 uncovered, 64 undefined)")


def _ece_bf(conf, correct, bins=15):
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf >= lo) & ((conf < hi) if b < bins - 1 else (conf <= 1.0))
        if m.any():
            e += m.double().mean() * abs(correct[m].double().mean() - conf[m].double().mean())
    return float(e)


def test_calibration_metrics_brute_force():
    """ECE-15, NLL, Brier and accuracy of both variants against brute force from p."""
    ras, lab, gt, text = _scene(hole=(12, 12))
    N, K = lab.numel(), text.shape[0]
    g = torch.Generator().manual_seed(2)
    noisy = torch.where(torch.rand(N, generator=g) < 0.25, (lab + 1) % 4, lab)
    buf = 0.8 * text[noisy] + 0.1 * torch.rand(N, 8, generator=g)
    acc = evaluate_views([TensorRow("r", "feature", buf=buf)], [View(ras, gt, text)],
                         {"r": 5.0}, tau_cov=0.5)
    m = acc.metrics()["r"]
    W = ras.weights().double()
    s = (W @ buf.double()) @ text.double().t()
    cov = W.sum(1) >= 0.5
    p = torch.where(cov[:, None], torch.softmax(5.0 * s, 1), torch.full((256, K), 1.0 / K))
    y = gt.reshape(-1)
    conf, pred = p.max(1)
    correct = (pred == y) & cov
    brier = ((p - Fn.one_hot(y, K).double()) ** 2).sum(1)
    nll = -torch.log(p.gather(1, y[:, None])[:, 0])
    for var, sel in (("all", torch.ones(256, dtype=torch.bool)), ("covered", cov)):
        v = m[var]
        assert abs(v["acc"] - correct[sel].double().mean()) < 1e-12, var
        assert abs(v["brier"] - brier[sel].mean()) < 1e-5, (var, v["brier"])
        assert abs(v["nll"] - nll[sel].mean()) < 1e-5, (var, v["nll"])
        assert abs(v["ece"] - _ece_bf(conf[sel], correct[sel])) < 1e-5, (var, v["ece"])
    assert abs(m["all"]["ece"] - m["covered"]["ece"]) > 1e-3     # the variants differ
    print(f"PASS calibration brute force: ECE all {m['all']['ece']:.4f} / covered "
          f"{m['covered']['ece']:.4f}, Brier {m['all']['brier']:.4f}, NLL {m['all']['nll']:.4f}")


def test_beta_fit_uses_covered_pixels_only():
    """The fit equals the fit with every uncovered pixel set to ignore, and the uncovered
    pixels would move it if they were used (tau_cov = 0)."""
    g = torch.Generator().manual_seed(4)
    P = 300
    alpha = torch.cat([torch.ones(200), torch.full((100,), 0.3)])
    ras = _FixedW(torch.diag(alpha))
    text = torch.eye(4)
    gt = torch.randint(0, 4, (P,), generator=g)
    lab = torch.where(torch.rand(P, generator=g) < 0.2, (gt + 1) % 4, gt)
    lab[200:] = (gt[200:] + 2) % 4                             # uncovered: all wrong
    rows = [TensorRow("r", "feature", buf=text[lab])]
    b_cov = fit_betas(rows, [View(ras, gt, text)], tau_cov=0.5)["r"]
    masked = torch.where(alpha >= 0.5, gt, torch.full_like(gt, -1))
    b_mask = fit_betas(rows, [View(ras, masked, text)], tau_cov=0.5)["r"]
    b_all = fit_betas(rows, [View(ras, gt, text)], tau_cov=0.0)["r"]
    assert b_cov == b_mask, (b_cov, b_mask)
    assert abs(b_all - b_cov) > 0.1 * b_cov, (b_all, b_cov)
    print(f"PASS fit on covered pixels: beta {b_cov:.3f} (with uncovered pixels: {b_all:.3f})")


def test_input_validation():
    ras, lab, gt, text = _scene()
    N, K = lab.numel(), text.shape[0]
    ok = TensorRow("r", "feature", buf=text[lab])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_view([ok, TensorRow("r", "feature", buf=torch.zeros(N, 8))], ras, text, gt,
                      {"r": 50.0}, 0.5)
    with pytest.raises(ValueError, match="duplicate"):
        BetaFitter(0.5).add_view([ok, ok], ras, text, gt)
    with pytest.raises(ValueError, match="not adaptive"):
        evaluate_view([ok], ras, text, gt, {"r": dict(beta0=1.0, a=0.0, b=0.0)}, 0.5)
    bad = text[lab].clone()
    bad[5] = float("nan")
    with pytest.raises(ValueError, match="'nanrow'.*non-finite"):
        evaluate_view([TensorRow("nanrow", "feature", buf=bad)], ras, text, gt,
                      {"nanrow": 50.0}, 0.5)
    uinf = torch.full((N,), 0.1)
    uinf[0] = float("inf")
    with pytest.raises(ValueError, match="inf"):
        render_row(TensorRow("r", "feature", buf=text[lab], unc={"u": uinf}), ras)
    css = TensorRow("css", "probs", buf=Fn.one_hot(lab, K).float())
    with pytest.raises(ValueError, match="no beta for probs row 'css'"):
        evaluate_view([ok, css], ras, text, gt, {"r": 50.0}, 0.5)
    o = evaluate_view([ok, css], ras, text, gt, {"r": 50.0}, 0.5, raw_probs={"css"})
    assert o["css"]["correct"].all()
    with pytest.raises(ValueError, match="raw_probs"):
        evaluate_view([css], ras, text, gt, {"css": 0.8}, 0.5, raw_probs={"css"})
    with pytest.raises(ValueError, match="raw_probs is for probs rows"):
        evaluate_view([ok], ras, text, gt, {}, 0.5, raw_probs={"r"})
    with pytest.raises(ValueError, match="negative label"):
        evaluate_view([ok], ras, text, torch.where(gt == 0, torch.full_like(gt, -2), gt),
                      {"r": 50.0}, 0.5)
    o64 = evaluate_view([TensorRow("r", "feature", buf=text[lab].double())], ras, text, gt,
                        {"r": 50.0}, 0.5)                    # float64 buffer: rendered float32
    assert o64["r"]["correct"].all()
    acc = SegAccumulator()
    clash = TensorRow("r", "feature", buf=text[lab], unc={"msp": torch.zeros(N)})
    acc.add(evaluate_view([clash], ras, text, gt, {"r": 50.0}, 0.5))
    with pytest.raises(ValueError, match="clash"):
        acc.metrics()
    print("PASS validation: duplicate names, P2 params on a plain row, NaN buffer, inf u, "
          "missing probs beta, raw_probs misuse, negative label, signal name clash")


def test_views_cross_ause_covered_only():
    """Holes of the shared geometry are uncovered at every k: by default they are left out
    of e_red / e_irr (counted as n_hole); with covered_only=False they are e_irr and every
    signal ranks them first (coverage rule), so no channel gets credit for its NaN there."""
    ras, lab, gt, text = _scene(hole=(12, 12))
    N = lab.numel()
    wk = torch.zeros(N, dtype=torch.bool); wk[:8] = True       # wrong at k views
    wf = torch.zeros(N, dtype=torch.bool); wf[:4] = True       # wrong at full (subset)
    g = torch.Generator().manual_seed(5)
    unc = {"u_irr": wf.float(), "u_rand": torch.rand(N, generator=g)}
    ok = evaluate_view([TensorRow("r", "feature", buf=text[torch.where(wk, (lab + 1) % 4, lab)],
                                  unc=unc)], ras, text, gt, {"r": 50.0}, 0.5)["r"]
    of = evaluate_view([TensorRow("r", "feature", buf=text[torch.where(wf, (lab + 1) % 4, lab)],
                                  unc=unc)], ras, text, gt, {"r": 50.0}, 0.5)["r"]
    t, keep = aligned_error_targets(ok, of, return_mask=True)
    assert int(keep.sum()) == 240 and int(t["e_irr"].sum()) == 4 and int(t["e_red"].sum()) == 4
    t_all = aligned_error_targets(ok, of, covered_only=False)
    assert int(t_all["e_irr"].sum()) == 4 + 16                 # the hole is e_irr at every k
    v = views_cross_ause(ok, of)
    assert v["n_hole"] == 16 and v["n_pixels"] == 240 and v["n_e_irr"] == 4
    assert abs(v["ause"]["u_irr"]["e_irr"]) < 1e-12
    va = views_cross_ause(ok, of, covered_only=False)
    assert va["n_e_irr"] == 20 and abs(va["ause"]["u_irr"]["e_irr"]) < 1e-12
    print(f"PASS views cross-AUSE: covered-only (16 hole pixels apart); u_rand on e_irr "
          f"{v['ause']['u_rand']['e_irr']:.4f}")


# --------------------------------------------------------------------------- CLI (CPU)
def _load_cli():
    spec = importlib.util.spec_from_file_location("eval_seg", REPO / "scripts" / "eval_seg.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_7scenes(root, labels_fn, n=2):
    import cv2
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        base = root / f"frame-{i:06d}"
        cv2.imwrite(str(base) + ".color.png", np.zeros((480, 640, 3), np.uint8))
        cv2.imwrite(str(base) + ".depth.png", np.full((480, 640), 2000, np.uint16))
        c2w = np.eye(4)
        c2w[0, 3] = 0.05 * i
        np.savetxt(str(base) + ".pose.txt", c2w)
        cv2.imwrite(str(base) + ".label.png", labels_fn().astype(np.uint16))


def _cli_toy(tmp_path):
    """Files of a toy 7-Scenes scene with a saved field; returns the common CLI argv."""
    from usplat.feature_pca import StreamingPCA
    # raw ids: 0 void (ignored), 3 chair (left), 7 table (right), 8 merged into 7 (a strip)
    def labels():
        lab = np.zeros((480, 640), np.int64)
        lab[40:, :320] = 3
        lab[40:, 320:] = 7
        lab[40:80, 320:] = 8
        return lab
    _fake_7scenes(tmp_path / "data", labels)
    D, d = 16, 8
    pca = StreamingPCA(D, d)
    pca.basis = torch.eye(D)[:, :d]
    t_chair, t_table, t_sofa = torch.eye(D)[0], torch.eye(D)[1], torch.eye(D)[2]
    # Gaussians on the plane z = 2 m (7-Scenes K: fx 585 at 640x480)
    xs = torch.arange(-1.15, 1.2, 0.04)
    ys = torch.arange(-0.9, 0.95, 0.04)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    means = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.full((gx.numel(),), 2.0)], -1)
    N = means.shape[0]
    splats = dict(means=means, quats=torch.tensor([1.0, 0, 0, 0]).expand(N, 4).clone(),
                  scales=torch.full((N, 3), 0.03), opacities=torch.full((N,), 0.9))
    torch.save(splats, tmp_path / "gs.pt")
    right = means[:, 0] > 0                                   # table on the right half
    feat = torch.where(right[:, None], pca.project(t_table[None]), pca.project(t_chair[None]))
    f = DirectionalSemanticField(N, d, FieldConfig(change_test=False))
    f.set_centre(torch.zeros(d))
    for _ in range(4):
        f.update_sparse(torch.arange(N), torch.ones(N), feat, num_pixels=100)
    torch.save(dict(field=f.state_dict(), pca=pca.state_dict()), tmp_path / "field.pt")
    vocab = dict(templates=["a photo of a {}."], ignore=[0], merges={8: 7},
                 classes={3: {"name": "chair"}, 7: {"name": "table", "prompt": "table"},
                          9: "sofa"})
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    key = lambda p: f"m|p|a photo of a {{}}.|{p}"
    torch.save({key("chair"): t_chair, key("table"): t_table, key("sofa"): t_sofa},
               tmp_path / "text.pt")
    scenes = [dict(name="toy", field=str(tmp_path / "field.pt"), ply=str(tmp_path / "gs.pt"),
                   data=str(tmp_path / "data"), dataset="7scenes", stride=1,
                   label_pattern="{root}/frame-{idx:06d}.label.png")]
    (tmp_path / "scenes.json").write_text(json.dumps(scenes))
    return ["--vocab", str(tmp_path / "vocab.json"), "--backend", "mock", "--device", "cpu",
            "--downscale", "16", "--tau-cov", "0.5", "--extractor", "clearclip", "--clip-model", "m",
            "--clip-pretrained", "p", "--text-cache", str(tmp_path / "text.pt")]


def test_cli_mock_backend(tmp_path):
    cli = _load_cli()
    base = _cli_toy(tmp_path)
    scenes = str(tmp_path / "scenes.json")
    out_path = tmp_path / "out" / "eval.json"
    cli.main(base + ["--scenes", scenes, "--val-scenes", scenes, "--out", str(out_path)])
    rec = json.loads(out_path.read_text())
    res = rec.get("metrics", rec)
    sc = res["scenes"]["toy"]
    assert sc["classes"] == ["chair", "table"]                # sofa absent: not queried
    assert sc["n_unknown_label_pixels"] == 0 and sc["n_views"] == 2
    p0 = sc["rows"]["ours_P0"]["all"]
    assert p0["miou"] > 0.9 and p0["n_classes"] == 2, p0
    assert set(res["betas"]["ours_P2"]) == {"beta0", "a", "b", "g"}
    assert res["summary"]["ours_P0"]["all"]["miou"]["n"] == 1
    # the betas of this run can be read back for a test-only run, with their settings
    betas = cli.load_betas(str(out_path))
    assert set(betas) == {"ours_P0", "ours_P1", "ours_P2", "ours_scalar"}
    assert sc["betas_used"]["ours_P0"] == betas["ours_P0"]
    st = cli.load_fit_settings(str(out_path))
    assert st["undef_frac"] == 0.5 and st["tau_cov"] == 0.5
    print(f"PASS CLI (mock backend): mIoU {p0['miou']:.3f}, coverage {p0['coverage']:.3f}, "
          f"record status {rec.get('status', 'plain json')}")


def test_cli_undef_frac_reaches_fit_and_betas_are_checked(tmp_path, monkeypatch):
    """--undef-frac reaches the val fit (not only the test evaluation), is recorded with the
    betas, and a --betas file fit with another value is refused; rows without a fitted beta
    are refused before any render; --val-scenes with --betas is refused."""
    cli = _load_cli()
    base = _cli_toy(tmp_path)
    scenes = str(tmp_path / "scenes.json")
    seen = []
    orig = seg_eval.render_row

    def spy(row, rasterizer, alpha_eps=1e-6, undef_frac=0.5):
        seen.append(undef_frac)
        return orig(row, rasterizer, alpha_eps, undef_frac)
    monkeypatch.setattr(seg_eval, "render_row", spy)
    fit = tmp_path / "fit.json"
    cli.main(base + ["--val-scenes", scenes, "--undef-frac", "0.2", "--plain-json",
                     "--out", str(fit)])
    assert seen and set(seen) == {0.2}, sorted(set(seen))      # the fit renders with 0.2
    assert cli.load_fit_settings(str(fit))["undef_frac"] == 0.2
    seen.clear()
    ev = tmp_path / "ev.json"
    out = cli.main(base + ["--scenes", scenes, "--betas", str(fit), "--undef-frac", "0.2",
                           "--plain-json", "--out", str(ev)])
    assert set(seen) == {0.2}
    used = out["scenes"]["toy"]["betas_used"]
    assert used == cli.load_betas(str(fit)), used
    with pytest.raises(ValueError, match="undef_frac"):
        cli.main(base + ["--scenes", scenes, "--betas", str(fit), "--undef-frac", "0.5",
                         "--plain-json", "--out", str(tmp_path / "bad.json")])
    with pytest.raises(ValueError, match="not both"):
        cli.main(base + ["--scenes", scenes, "--val-scenes", scenes, "--betas", str(fit),
                         "--undef-frac", "0.2", "--plain-json", "--out", str(tmp_path / "b.json")])
    part = dict(betas={k: v for k, v in cli.load_betas(str(fit)).items() if k != "ours_P1"},
                beta_fit_settings=cli.load_fit_settings(str(fit)))
    (tmp_path / "part.json").write_text(json.dumps(part))
    seen.clear()
    with pytest.raises(ValueError, match="'ours_P1'.*no usable fitted beta"):
        cli.main(base + ["--scenes", scenes, "--betas", str(tmp_path / "part.json"),
                         "--undef-frac", "0.2", "--plain-json", "--out", str(tmp_path / "c.json")])
    assert not seen                                            # refused before rendering
    print("PASS CLI: --undef-frac 0.2 reaches the fit and is recorded; mismatch refused; "
          "missing beta refused")


def test_vocab_file_of_k8():
    """The frozen Replica vocabulary (configs/vocab_replica.json, K8) parses: ignore_ids and
    a single template are read; labels map to vocabulary indices."""
    import pytest
    path = REPO / "configs" / "vocab_replica.json"
    if not path.exists():
        pytest.skip("configs/vocab_replica.json not present")
    cli = _load_cli()
    v = cli.load_vocab(str(path))
    raw_file = json.loads(path.read_text())
    assert v["ignore"] == {int(i) for i in raw_file["ignore_ids"]}
    assert v["templates"] == [raw_file["template"]]
    assert len(v["ids"]) == len(set(v["ids"])) and 0 not in v["ids"]
    raw = torch.tensor([[0, v["ids"][0]], [v["ids"][-1], 0]])
    lab, unk = cli.map_labels(raw, v)
    assert lab.tolist() == [[-1, 0], [len(v["ids"]) - 1, -1]] and unk == 0
    print(f"PASS vocabulary: {len(v['ids'])} classes, ignore {sorted(v['ignore'])}")
