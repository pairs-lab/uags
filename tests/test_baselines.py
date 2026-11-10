"""K11 baselines on the CPU mock rasterizer.

 * Averaging equals the official Gradient-Weighted Back-Projection code
   (docs/reports/2026-09-28-paper-completion-plan/logs/wf/sota/gradbp_backproject.py,
   create_feature_field_lseg, transcribed below with gsplat replaced by the mock), also when
   the field runs its centred channel (averaging stays on the raw channel);
 * NormLift ||f||, C_intra, C_inter, N_eff and R(j) equal Eq. 6-8 computed by hand on a toy
   Gaussian, and a float64 per-view brute force from the explicit weights with beta = 2.5,
   a pixel mask and centre=True;
 * view counts (m > 0 and m >= a0*H*W, including m exactly a0*H*W);
 * CSS one-hot counts equal brute force W^T onehot(argmax z.t) with a pixel mask; its
   uncertainties use alpha0; directionless pixels (z = 0, NaN, all scores tied) are no vote;
   alpha0 <= 0 is rejected;
 * CSS query columns follow the queried global class ids, in the evaluator's order;
 * Dirichlet total and top-class variance, expected entropy and MI equal Monte-Carlo
   estimates; chunking does not change them;
 * Yager's rule: a two-view toy case by hand, and a brute-force enumeration of focal sets;
 * make_rows: names, kinds, shapes and values equal the field's read-outs; every row is a
   rows.Row that seg_eval.render_row and rows.freeze accept;
 * checkpoints: baselines_state -> torch.save -> row_factory gives the same rows (baseline rows
   only), and scripts/eval_seg.py --row-factory usplat.baselines:row_factory runs (mock backend).
"""
import importlib.util
import io
import json
import math
import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as Fn
from usplat.vmf_field import DirectionalSemanticField, FieldConfig
from usplat.mock_rasterizer import MockFeatureRasterizer
from usplat.vmf_math import a_d
from usplat.rows import Row, freeze, field_rows
from usplat.seg_eval import render_row, evaluate_view
import usplat.baselines as B
from usplat.baselines import (CSS_ALPHA0_GRID, Averaging, NormLift, ViewCount, CSS, make_rows, update_all,
                              opinion_from_counts, yager_fuse, default_baselines,
                              baselines_state, load_baselines, row_factory, ours_scalar_row)

REPO = pathlib.Path(__file__).resolve().parents[1]


def _views(V, N=30, S=24, seed=0, dead=(0,)):
    """V mock cameras over the same N Gaussians (shifted centres, new depth order).
    Gaussians in `dead` have opacity 0: no view sees them."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(N, 2, generator=g) * S
    sig = 1.5 + 2.0 * torch.rand(N, generator=g)
    opac = 0.4 + 0.5 * torch.rand(N, generator=g)
    opac[list(dead)] = 0.0
    grid = torch.stack(torch.meshgrid(torch.arange(S), torch.arange(S), indexing="xy"), -1)
    pix = grid.reshape(-1, 2).float() + 0.5
    out = []
    for _ in range(V):
        shift = (torch.rand(1, 2, generator=g) - 0.5) * 8
        out.append(MockFeatureRasterizer(base + shift, sig, opac,
                                         torch.rand(N, generator=g), pix))
    return out, S * S


def _gradbp_official(rasterizers, feats_per_view, N, d):
    """create_feature_field_lseg (gradbp_backproject.py:25-171), one image per batch, with
    rasterization(...) replaced by the mock render (P, C) = W @ colors."""
    gaussian_features = torch.zeros(N, d)
    gaussian_denoms = torch.ones(N) * 1e-12                              # l.63
    colors_feats = torch.zeros(N, d, requires_grad=True)
    colors_feats_0 = torch.zeros(N, 3, requires_grad=True)
    for ras, feats in zip(rasterizers, feats_per_view):
        output_for_grad = ras(colors_feats)
        target = (output_for_grad * feats).sum()                        # l.127
        target.backward()
        colors_feats_copy = colors_feats.grad.clone()
        colors_feats.grad.zero_()
        output_for_grad = ras(colors_feats_0)
        target_0 = output_for_grad.sum()                                # l.145
        target_0.backward()
        gaussian_features += colors_feats_copy                          # l.149
        gaussian_denoms += colors_feats_0.grad[:, 0]                    # l.150
        colors_feats_0.grad.zero_()
    gaussian_features = gaussian_features / gaussian_denoms[..., None]  # l.166
    gaussian_features = gaussian_features / gaussian_features.norm(dim=-1, keepdim=True)
    gaussian_features[torch.isnan(gaussian_features)] = 0               # l.169
    return gaussian_features


def _field(N, d, **kw):
    f = DirectionalSemanticField(N, d, FieldConfig(**kw))
    f.set_centre(Fn.normalize(torch.ones(d), dim=0) * 0.3)
    return f


def test_averaging_equals_official_gradbp():
    """centre=True: the hooks also carry the centred channel; averaging must use the raw one."""
    torch.manual_seed(0)
    V, d = 4, 16
    ras, P = _views(V)
    N = ras[0].N
    feats = [Fn.normalize(torch.randn(P, d), dim=-1) for _ in range(V)]
    field = _field(N, d, centre=True)
    avg = Averaging.for_field(field)
    for r, z in zip(ras, feats):
        s = field.update_view(r, z, num_pixels=P)
    assert s.Rc is not None                                              # centred channel on
    ref = _gradbp_official(ras, feats, N, d)
    ours = avg.feature()
    err = (ours - ref).abs().max().item()
    assert err < 1e-4, err
    assert (ours[0] == 0).all() and (ref[0] == 0).all()                  # unseen -> 0
    seen = ours[1:].norm(dim=-1)
    assert torch.allclose(seen, torch.ones_like(seen), atol=1e-5)
    # the pixel-weighted mean feature, from the explicit weights
    Rbf = sum(r.weights().t() @ z for r, z in zip(ras, feats))
    bf = Fn.normalize(Rbf, dim=-1)
    assert (ours[1:] - bf[1:]).abs().max() < 1e-5
    print(f"PASS averaging (centre=True field): max |ours - official GradBP| = {err:.2e} over "
          f"{N} Gaussians x {d} dims, {V} views; unseen Gaussian -> 0 in both")


def test_normlift_eq6_eq8_by_hand():
    """Gaussian 0: view 1 has two pixels (w 1, 1) on e1 and e2, view 2 one pixel (w 3) on e1.
       W = (2, 3), f^(1) = (1/2, 1/2), f^(2) = e1, f = (4/5, 1/5), ||f|| = sqrt(17)/5,
       C_intra = (2 sqrt(1/2) + 3)/5, C_inter = ||f||/C_intra, N_eff = 25/13,
       R = ||f|| (25/13)/(25/13 + 1) = ||f|| 25/38.  Gaussian 1 is never seen."""
    d = 3
    e1, e2 = torch.eye(d)[0], torch.eye(d)[1]
    field = _field(2, d, change_test=False)
    nl = NormLift.for_field(field)
    field.update_sparse(torch.tensor([0, 0]), torch.tensor([1.0, 1.0]),
                        torch.stack([e1, e2]), num_pixels=100)
    field.update_sparse(torch.tensor([0]), torch.tensor([3.0]), e1[None], num_pixels=100)
    fn = math.sqrt(17) / 5
    ci = (2 * math.sqrt(0.5) + 3) / 5
    exp = dict(norm=fn, c_intra=ci, c_inter=fn / ci, n_eff=25 / 13, reliability=fn * 25 / 38)
    for k, v in exp.items():
        got = getattr(nl, k)()
        assert abs(got[0].item() - v) < 1e-6, (k, got[0].item(), v)
        assert got[1].item() == 0.0, (k, got[1])                         # zero evidence
    assert abs(nl.c_intra()[0] * nl.c_inter()[0] - nl.norm()[0]) < 1e-6   # Eq. 7 identity
    f0 = nl.feature()[0]
    assert torch.allclose(f0, Fn.normalize(torch.tensor([0.8, 0.2, 0.0]), dim=0), atol=1e-6)
    u = nl.uncertainties()
    assert abs(u["normlift_1-R"][0] - (1 - fn * 25 / 38)) < 1e-6
    assert abs(u["normlift_epi"][0] - 1 / (1 + 25 / 13)) < 1e-6
    assert abs(u["normlift_1-Cinter"][0] - (1 - fn / ci)) < 1e-6
    assert u["normlift_1-R"][1] == 1.0 and u["normlift_epi"][1] == 1.0
    assert u["normlift_1-Cinter"][1] == 1.0                             # unseen: most uncertain
    print(f"PASS NormLift toy: ||f|| {fn:.6f}, C_intra {ci:.6f}, C_inter {fn / ci:.6f}, "
          f"N_eff {25 / 13:.6f}, R {fn * 25 / 38:.6f} (Eq. 6-8 by hand)")

    # mock rasterizer: per-view W^(v) and f^(v) from the hooks equal the explicit weights
    torch.manual_seed(1)
    V, d = 3, 8
    ras, P = _views(V, seed=3)
    N = ras[0].N
    feats = [Fn.normalize(torch.randn(P, d), dim=-1) for _ in range(V)]
    field = _field(N, d)
    nl = NormLift.for_field(field)
    for r, z in zip(ras, feats):
        field.update_view(r, z, num_pixels=P)
    Wv = torch.stack([r.weights().sum(0) for r in ras])                  # (V, N)
    Fv = torch.stack([r.weights().t() @ z for r, z in zip(ras, feats)])  # (V, N, d) = W f^(v)
    S1 = Wv.sum(0)
    f = Fv.sum(0) / S1.clamp_min(1e-12)[:, None]
    neff = S1 ** 2 / (Wv ** 2).sum(0).clamp_min(1e-30)
    Rj = f.norm(dim=-1) * neff / (neff + 1.0)
    ci = Fv.norm(dim=-1).sum(0) / S1.clamp_min(1e-12)
    s = slice(1, None)
    assert torch.allclose(nl.n_eff()[s], neff[s], rtol=1e-4)
    assert torch.allclose(nl.reliability()[s], Rj[s], atol=1e-5)
    assert torch.allclose(nl.c_intra()[s], ci[s], atol=1e-5)
    assert nl.reliability()[0] == 0 and nl.n_eff()[0] == 0
    print("PASS NormLift mock: N_eff, R(j), C_intra from hooks equal the explicit weights")


def test_normlift_beta_mask_centre_bruteforce():
    """Regression (review: beta hard-coded to 1 passed every test). beta = 2.5, a pixel mask
    and a centre=True field, against a float64 per-view brute force of Eq. 6-8."""
    torch.manual_seed(0)
    V, d, beta = 5, 12, 2.5
    ras, P = _views(V, N=40, S=20, seed=11)
    N = ras[0].N
    feats = [Fn.normalize(torch.randn(P, d), dim=-1) for _ in range(V)]
    masks = [(torch.rand(P) > 0.3).float() for _ in range(V)]
    field = _field(N, d, centre=True)
    nl = NormLift.for_field(field, beta=beta)
    avg = Averaging.for_field(field)
    for r, z, a in zip(ras, feats, masks):
        field.update_view(r, z, pixel_valid=a, num_pixels=P)
    W = [r.weights().double() * a.double()[:, None] for r, a in zip(ras, masks)]  # (P, N)
    Wv = torch.stack([w.sum(0) for w in W])                              # (V, N)
    fv = [w.t() @ z.double() for w, z in zip(W, feats)]                   # W f^(v), (N, d)
    S1 = Wv.sum(0)
    seen = S1 > 0
    f = sum(fv) / S1.clamp_min(1e-300)[:, None]
    nrm = f.norm(dim=-1)
    cintra = torch.stack([x.norm(dim=-1) for x in fv]).sum(0) / S1.clamp_min(1e-300)
    neff = S1 ** 2 / (Wv ** 2).sum(0).clamp_min(1e-300)
    ref = dict(norm=nrm, c_intra=cintra, c_inter=nrm / cintra.clamp_min(1e-300), n_eff=neff,
               reliability=nrm * neff / (neff + beta))
    errs = {}
    for k, v in ref.items():
        got = getattr(nl, k)().double()
        errs[k] = (got[seen] - v[seen]).abs().max().item()
        assert errs[k] < (1e-5 if k == "n_eff" else 1e-6), (k, errs[k])
        assert (got[~seen] == 0).all(), k
    u = nl.uncertainties()["normlift_1-R"].double()
    assert (u[seen] - (1 - ref["reliability"][seen])).abs().max() < 1e-6
    # beta enters R(j): beta = 1 would differ by far more than the tolerance
    assert (nrm * neff / (neff + 1.0) - ref["reliability"])[seen].abs().max() > 1e-2
    bf = Fn.normalize(sum(fv), dim=-1)                                    # raw, not centred
    assert (avg.feature()[seen].double() - bf[seen]).abs().max() < 1e-6
    assert torch.equal(avg.feature(), nl.feature())
    print("PASS NormLift beta=2.5, mask, centre=True vs float64 brute force: max err "
          + ", ".join(f"{k} {v:.1e}" for k, v in errs.items()))


def test_view_count():
    d = 4
    field = _field(3, d, view_mass_frac=0.1, change_test=False)     # ref = 0.1 * 100 = 10
    avg = Averaging.for_field(field)
    vc = ViewCount.for_field(field, feature=avg.feature)
    z = Fn.normalize(torch.randn(2, d), dim=-1)
    for masses in ([5.0, 20.0], [5.0, 20.0], [5.0, 0.0]):            # Gaussian 2 never seen
        field.update_sparse(torch.tensor([0, 1]), torch.tensor(masses), z, num_pixels=100)
    assert vc.n_vis.tolist() == [3.0, 2.0, 0.0], vc.n_vis
    assert vc.n_full.tolist() == [0.0, 2.0, 0.0], vc.n_full
    u = vc.uncertainties()
    assert torch.allclose(u["viewcount"], torch.tensor([1 / 4, 1 / 3, 1.0]))
    assert torch.allclose(u["viewcount_a0"], torch.tensor([1.0, 1 / 3, 1.0]))
    (row,) = vc.rows()
    assert row.kind == "feature" and torch.equal(row.buffer(), avg.feature())
    # regression (review: `>` instead of `>=` passed): m exactly a0*H*W is a full view
    f2 = _field(2, d, view_mass_frac=0.25, change_test=False)       # ref = 25 (exact)
    vc2 = ViewCount.for_field(f2)
    f2.update_sparse(torch.tensor([0, 1]), torch.tensor([25.0, 24.999]), z, num_pixels=100)
    assert vc2.n_full.tolist() == [1.0, 0.0] and vc2.n_vis.tolist() == [1.0, 1.0]
    print("PASS view count: n_vis (3, 2, 0), n_full (0, 2, 0), u = 1/(1+n); m = a0*H*W is full")


def test_css_counts_bruteforce():
    torch.manual_seed(2)
    V, d, K = 3, 8, 5
    ras, P = _views(V, seed=5)
    N = ras[0].N
    text = Fn.normalize(torch.randn(K, d), dim=-1) * (0.5 + torch.rand(K, 1))  # unnormalised
    feats = [Fn.normalize(torch.randn(P, d), dim=-1) for _ in range(V)]
    masks = [(torch.rand(P) > 0.25).float() for _ in range(V)]
    field = _field(N, d)
    css = CSS.for_field(field, text, alpha0=0.5)
    per_view = []
    for r, z, a in zip(ras, feats, masks):
        update_all(field, r, z, css=[css], pixel_valid=a, num_pixels=P)
        y = (z @ text.t()).argmax(-1)
        per_view.append(r.weights().t() @ (Fn.one_hot(y, K).float() * a[:, None]))
    bf = sum(per_view)
    err = (css.counts - bf).abs().max().item()
    assert err < 1e-4, err
    assert torch.allclose(css.alpha(), bf + 0.5, atol=1e-4)
    m = css.mean()
    assert torch.allclose(m.sum(-1), torch.ones(N), atol=1e-5)
    assert torch.allclose(m[0], torch.full((K,), 1.0 / K))              # unseen: uniform
    # Yager state from the brute-force per-view counts
    b, u, k = torch.zeros(N, K), torch.ones(N), torch.zeros(N)
    for C in per_view:
        b2, u2 = opinion_from_counts(C)
        b, u, kk = yager_fuse(b, u, b2, u2)
        k += kk
    assert torch.allclose(css.yager_conflict(), k, atol=1e-5)
    (row,) = css.rows()
    unc = row.uncertainties()
    assert row.kind == "probs" and row.buffer().shape == (N, K)
    assert set(unc) == {"css_var_top", "css_var", "css_mi", "css_exp_entropy", "yager_conflict"}
    # regression (review: alpha0 ignored in the uncertainties passed): Dirichlet(counts + 0.5)
    ref = B.dirichlet_uncertainties(bf + 0.5)
    for key, v in ref.items():
        assert torch.allclose(unc[key], v, rtol=1e-4, atol=1e-6), key
    flat = B.dirichlet_uncertainties(bf + 1.0)
    assert (unc["css_var"] - flat["css_var"]).abs()[0] > 1e-3            # alpha0 matters
    print(f"PASS CSS: counts = brute force (max err {err:.1e}) with a pixel mask, "
          f"{V} views, K = {K}; Yager state matches; uncertainties use alpha0 = 0.5")


def test_css_directionless_pixels_are_no_vote():
    """Regression (review: z = 0 / tied pixels became class-0 votes). Non-finite z, z = 0 and
    all-tied scores carry no direction: no count, as in the field."""
    g = torch.Generator().manual_seed(0)
    K, d = 4, 3
    text = Fn.normalize(torch.randn(K, d, generator=g), dim=-1)
    css = CSS(1, text, yager=False)
    z = torch.zeros(10, d)
    z[3] = float("nan")
    z[4, 0] = float("inf")
    C = css.update_sparse(torch.zeros(10, dtype=torch.long), torch.ones(10), z)
    assert C.abs().sum() == 0 and css.counts.abs().sum() == 0, C
    tied = CSS(1, torch.ones(3, d), yager=False)                          # identical texts
    C = tied.update_sparse(torch.zeros(5, dtype=torch.long), torch.ones(5),
                           torch.randn(5, d, generator=g))
    assert C.abs().sum() == 0
    one = CSS(1, text[:1], yager=False)                                   # K = 1: all vote
    C = one.update_sparse(torch.zeros(5, dtype=torch.long), torch.ones(5),
                          torch.randn(5, d, generator=g))
    assert C.tolist() == [[5.0]]
    # render path: brute force with the directionless pixels masked out
    torch.manual_seed(6)
    ras, P = _views(2, seed=7)
    N = ras[0].N
    css2 = CSS(N, text)
    bf = torch.zeros(N, K)
    for r in ras:
        z = Fn.normalize(torch.randn(P, d), dim=-1)
        z[:40] = 0.0
        z[40:50] = float("nan")
        a = (torch.rand(P) > 0.2).float()
        css2.update_view(r, z, pixel_valid=a)
        keep = a * (torch.arange(P) >= 50).float()
        y = (torch.nan_to_num(z) @ text.t()).argmax(-1)
        bf += r.weights().t() @ (Fn.one_hot(y, K).float() * keep[:, None])
    assert torch.isfinite(css2.counts).all()
    assert (css2.counts - bf).abs().max() < 1e-4
    print("PASS CSS: z = 0, NaN/inf z and tied scores are no vote (sparse and render paths)")


def test_css_rejects_nonpositive_alpha0():
    """Regression (review: alpha0 = 0 gave NaN uncertainties for unseen Gaussians)."""
    for a0 in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            CSS(2, torch.eye(3), alpha0=a0)
    with pytest.raises(ValueError):
        B.dirichlet_uncertainties(torch.zeros(2, 3))
    u = CSS(2, torch.eye(3), alpha0=1e-3).uncertainties()                 # small prior is fine
    assert all(torch.isfinite(v).all() for v in u.values())
    print("PASS CSS: alpha0 in (0, -1, nan) rejected; alpha0 = 1e-3 gives finite channels")


def _css_query_scene():
    """A frozen 5-class vocabulary with global ids; the GT uses 4 of them (probe3 setup)."""
    g = torch.Generator().manual_seed(0)
    N, d, S = 80, 12, 20
    base = torch.rand(N, 2, generator=g) * S
    grid = torch.stack(torch.meshgrid(torch.arange(S), torch.arange(S), indexing="xy"), -1)
    pix = grid.reshape(-1, 2).float() + 0.5
    ras = MockFeatureRasterizer(base, torch.full((N,), 1.5), torch.full((N,), 0.8),
                                torch.rand(N, generator=g), pix)
    vocab = Fn.normalize(torch.randn(5, d, generator=g), dim=-1)
    gt_voc = (pix[:, 0] > S / 2).long() * 3 + (pix[:, 1] > S / 2).long()  # indices {0,1,3,4}
    z = Fn.normalize(vocab[gt_voc] + 0.3 * torch.randn(S * S, d, generator=g), dim=-1)
    return ras, vocab, gt_voc, z, N, d


def test_css_query_columns_follow_class_ids():
    """Regression (review, major: CSS kept no class ids). CSS maps the frozen vocabulary;
    the evaluator queries only the classes present, in its own order. rows(class_ids) must
    select those columns in that order: the prediction is the argmax of the full-vocabulary
    render over the queried columns, whatever the query order."""
    ras, vocab, gt_voc, z, N, d = _css_query_scene()
    ids = [10, 11, 13, 14, 20]                             # global ids of the vocabulary rows
    field = DirectionalSemanticField(N, d, FieldConfig(change_test=False))
    css = CSS.for_field(field, vocab, class_ids=ids)
    update_all(field, ras, z, css=[css], num_pixels=z.shape[0])
    present = torch.unique(gt_voc)
    full = render_row(css.rows()[0], ras)                   # all 5 columns
    res = {}
    for order in ("sorted", "reversed"):
        q = present if order == "sorted" else present.flip(0)
        local = torch.full((5,), -1)
        local[q] = torch.arange(len(q))
        gt = local[gt_voc]
        qids = [ids[i] for i in q.tolist()]
        (row,) = make_rows(None, [css], class_ids=qids)
        assert torch.equal(row.buffer(), css.mean()[:, q])
        out = evaluate_view([freeze(row)], ras, vocab[q], gt, {}, tau_cov=0.3,
                            raw_probs=("css",))["css"]
        dec = out["covered"] & (out["pred"] >= 0)          # seg_eval: a tied max is no decision
        assert dec.float().mean() > 0.9
        want = full.feat[out["pix"]][:, q].argmax(-1)
        assert torch.equal(out["pred"][dec], want[dec]), order
        res[order] = out
    acc = res["sorted"]["correct"][res["sorted"]["covered"]].float().mean().item()
    assert acc > 0.9, acc
    assert torch.equal(res["sorted"]["correct"], res["reversed"]["correct"])
    # the full-vocabulary row against a present-class query fails loudly, never silently
    with pytest.raises(ValueError, match="class probabilities"):
        evaluate_view(css.rows(), ras, vocab[present], torch.zeros(z.shape[0], dtype=torch.long),
                      {}, tau_cov=0.3, raw_probs=("css",))
    with pytest.raises(ValueError, match="not in the CSS vocabulary"):
        css.rows(class_ids=[10, 99])
    with pytest.raises(ValueError, match="without class_ids"):
        CSS(N, vocab).rows(class_ids=[10])
    with pytest.raises(ValueError):
        CSS(N, vocab, class_ids=[1, 2, 3])                  # 3 ids for 5 texts
    with pytest.raises(ValueError):
        CSS(N, vocab, class_ids=[1, 1, 2, 3, 4])             # duplicate id
    print(f"PASS CSS query columns: acc on covered {acc:.3f}, identical for sorted and "
          f"reversed query orders; unknown ids / missing ids raise")


def test_dirichlet_vs_monte_carlo():
    alpha = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.5, 2.0, 0.3, 4.0], [20.0, 1.0, 1.0, 1.0],
                          [3.0, 3.0, 0.2, 0.2], [100.0, 90.0, 5.0, 1.0]], dtype=torch.float64)
    out = B.dirichlet_uncertainties(alpha)
    M = 400_000
    torch.manual_seed(4)
    th = torch.distributions.Dirichlet(alpha).sample((M,)).clamp_min(1e-300)   # (M, N, K)
    H = -(th * th.log()).sum(-1)                                                # (M, N)
    exp_ent = H.mean(0)
    pbar = th.mean(0)
    mi = -(pbar * pbar.log()).sum(-1) - exp_ent
    D = ((th - pbar) ** 2).sum(-1)                                              # (M, N)
    var = D.mean(0)
    top = alpha.argmax(-1)                                                      # c* per row
    Dt = (th.gather(2, top[None, :, None].expand(M, -1, 1)).squeeze(2)
          - pbar.gather(1, top[:, None]).squeeze(1)) ** 2                       # (M, N)
    se, se_var, se_top = (H.std(0) / math.sqrt(M), D.std(0) / math.sqrt(M),
                          Dt.std(0) / math.sqrt(M))
    for name, mc, tol in (("css_exp_entropy", exp_ent, 6 * se), ("css_mi", mi, 6 * se + 1e-4),
                          ("css_var", var, 6 * se_var), ("css_var_top", Dt.mean(0), 6 * se_top)):
        gap = (out[name] - mc).abs()
        assert (gap < tol).all(), (name, out[name], mc)
    # closed forms: total variance (sum of CSS Eq. 10) and CSS Eq. 10 at the top class
    S = alpha.sum(-1)
    p = alpha / S[:, None]
    assert torch.allclose(out["css_var"], (p * (1 - p) / (S[:, None] + 1)).sum(-1))
    pt = p.gather(1, top[:, None]).squeeze(1)
    assert torch.allclose(out["css_var_top"], pt * (1 - pt) / (S + 1))
    # regression (review: N > chunk was never exercised): chunk = 5 over N = 37
    a = torch.rand(37, 6, generator=torch.Generator().manual_seed(1)) * 50 + 0.1
    one, many = B.dirichlet_uncertainties(a), B.dirichlet_uncertainties(a, chunk=5)
    assert set(one) == set(many) == set(B.DIRICHLET_KEYS)
    for k in one:
        assert torch.equal(one[k], many[k]), k
    print("PASS Dirichlet: total and top-class variance, expected entropy and MI match Monte "
          f"Carlo ({M} samples; max MI gap {(out['css_mi'] - mi).abs().max():.1e}); "
          "chunk = 5 equals one chunk")


def _yager_bruteforce(ops):
    """Enumerate focal-set products: {c} & {c} = {c}, {c} & {c'} = empty (-> ignorance),
    {c} & Theta = {c}, Theta & Theta = Theta. ops: list of (b list, u)."""
    b, u, conflict = list(ops[0][0]), ops[0][1], 0.0
    for b2, u2 in ops[1:]:
        K = len(b)
        m = {c: 0.0 for c in range(K)}
        m["T"], empty = 0.0, 0.0
        focal1 = [(c, b[c]) for c in range(K)] + [("T", u)]
        focal2 = [(c, b2[c]) for c in range(K)] + [("T", u2)]
        for s1, x in focal1:
            for s2, y in focal2:
                if s1 == "T":
                    m[s2] += x * y
                elif s2 == "T" or s1 == s2:
                    m[s1] += x * y
                else:
                    empty += x * y
        b, u = [m[c] for c in range(K)], m["T"] + empty
        conflict += empty
    return b, u, conflict


def test_yager_two_views():
    K = 3
    css = CSS(3, torch.eye(K), alpha0=1.0)
    # Gaussian 0: views agree (class 0 twice); 1: views disagree (0 then 1); 2: never seen
    css.update_counts(torch.tensor([[3.0, 0, 0], [3.0, 0, 0], [0, 0, 0]]))
    css.update_counts(torch.tensor([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 0]]))
    # one view with counts (3,0,0): S = 6, b = (1/2, 0, 0), u = 1/2
    # agree:    b0 = 1/4 + 1/4 + 1/4 = 3/4, u = 1/4, conflict 0
    # disagree: conflict = (1/2)(1/2) - 0 = 1/4, b = (1/4, 1/4, 0), u = 1/4 + 1/4 = 1/2
    assert torch.allclose(css.yb[0], torch.tensor([0.75, 0, 0])) and abs(css.yu[0] - 0.25) < 1e-6
    assert torch.allclose(css.yb[1], torch.tensor([0.25, 0.25, 0])) and abs(css.yu[1] - 0.5) < 1e-6
    assert torch.allclose(css.yager_conflict(), torch.tensor([0.0, 0.25, 0.0]))
    assert torch.allclose(css.yu[2], torch.tensor(1.0))                    # vacuous stays vacuous
    assert torch.allclose(css.yb.sum(-1) + css.yu, torch.ones(3))          # masses sum to 1
    # a random 4-view sequence against the focal-set enumeration
    g = torch.Generator().manual_seed(7)
    C = [torch.rand(1, 4, generator=g) * 5 for _ in range(4)]
    ops = [opinion_from_counts(c) for c in C]
    b, u, k = torch.zeros(1, 4), torch.ones(1), torch.zeros(1)
    for b2, u2 in ops:
        b, u, kk = yager_fuse(b, u, b2, u2)
        k += kk
    bb, ub, kb = _yager_bruteforce([(o[0][0].tolist(), float(o[1][0])) for o in ops])
    assert torch.allclose(b[0], torch.tensor(bb), atol=1e-6) and abs(u[0] - ub) < 1e-6
    assert abs(k[0] - kb) < 1e-6, (k, kb)
    print(f"PASS Yager: two-view toy (conflict 0 / 0.25 / 0) and 4-view enumeration "
          f"(conflict {kb:.4f})")


def test_make_rows():
    torch.manual_seed(3)
    V, d, K = 3, 8, 4
    ras, P = _views(V, seed=9)
    N = ras[0].N
    field = _field(N, d, tau_min=2.0)
    nl = NormLift.for_field(field)
    vc = ViewCount.for_field(field, feature=nl.feature)
    css = CSS.for_field(field, Fn.normalize(torch.randn(K, d), dim=-1))
    rows = make_rows(field, [nl, vc, css])                   # built BEFORE the updates
    z0 = Fn.normalize(torch.randn(d), dim=0)
    for r in ras:
        z = Fn.normalize(z0 + 0.3 * torch.randn(P, d), dim=-1)
        update_all(field, r, z, css=[css], num_pixels=P)
    names = [r.name for r in rows]
    assert names == ["ours_P0", "ours_P1", "ours_P2", "ours_scalar", "normlift", "viewcount",
                     "css"], names
    assert all(isinstance(r, Row) for r in rows)
    assert [r.name for r in make_rows(None, [NormLift(N, d)])] == ["normlift"]
    assert [r.name for r in NormLift(N, d).rows(with_averaging=True)] == ["averaging", "normlift"]
    assert [r.name for r in NormLift(N, d, with_averaging=True).rows()] == ["averaging",
                                                                            "normlift"]
    by = {r.name: r for r in rows}
    assert torch.equal(by["ours_P0"].buffer(), field.mu_post())
    assert torch.equal(by["ours_P1"].buffer(), field.mean_direction())
    assert by["ours_P2"].adaptive and not by["ours_P0"].adaptive
    u = by["ours_P0"].uncertainties()
    assert set(u) == {"u_epi", "u_ale"}
    assert torch.equal(u["u_epi"], field.epistemic())
    ua = field.aleatoric()[0]
    assert torch.equal(torch.isnan(u["u_ale"]), torch.isnan(ua))
    assert torch.isfinite(ua).any() and torch.isnan(ua[0])           # unseen: undefined
    us = by["ours_scalar"].uncertainties()
    assert set(us) == {"spread"} and torch.equal(by["ours_scalar"].buffer(), field.mu_post())
    assert torch.allclose(us["spread"], 1.0 - a_d(d, field.kappa_post()))
    assert torch.equal(by["normlift"].buffer(), nl.feature())
    assert torch.allclose(by["normlift"].uncertainties()["normlift_1-R"], 1 - nl.reliability())
    for r in rows:
        buf = r.buffer()
        assert buf.shape == ((N, K) if r.kind == "probs" else (N, d)), (r.name, buf.shape)
        for k, v in r.uncertainties().items():
            assert v.shape == (N,), (r.name, k)
            assert torch.isfinite(v).all() or k == "u_ale", (r.name, k)
        rr = render_row(r, ras[0])                                  # the evaluator's render
        assert rr.feat.shape == (P, buf.shape[1]) and set(rr.unc) == set(r.uncertainties())
        fr = freeze(r)
        assert torch.equal(fr.buffer(), buf) and fr.kind == r.kind
    try:
        make_rows(field, [nl, NormLift(N, d)])
        raise AssertionError("duplicate names must raise")
    except ValueError:
        pass
    print(f"PASS make_rows: {names}; seg_eval.render_row and rows.freeze accept every row")


def test_checkpoint_row_factory():
    """Regression (review: no way to feed the baselines to eval_seg). default_baselines ->
    update_all -> baselines_state -> torch.save/load -> row_factory: the same rows, baseline
    rows only (no clash with eval_seg's ours_* rows), ViewCount rewired to the NormLift
    feature, CSS columns chosen by class id."""
    torch.manual_seed(5)
    V, d, K = 3, 8, 4
    ras, P = _views(V, seed=13)
    N = ras[0].N
    field = _field(N, d)
    ids = [7, 3, 12, 5]
    bs = default_baselines(field, Fn.normalize(torch.randn(K, d), dim=-1), class_ids=ids,
                           alpha0=0.5, beta=2.0)
    for r in ras:
        update_all(field, r, Fn.normalize(torch.randn(P, d), dim=-1), css=bs, num_pixels=P)
    buf = io.BytesIO()
    torch.save(dict(baselines=baselines_state(bs)), buf)
    buf.seek(0)
    ckpt = torch.load(buf, map_location="cpu", weights_only=False)
    rows = row_factory(ckpt, field, "cpu")
    names = [r.name for r in rows]
    grid = [f"css_a{a:g}" for a in CSS_ALPHA0_GRID]
    assert names == ["averaging", "normlift", "viewcount", "css"] + grid, names
    orig = {r.name: r for r in make_rows(None, bs)}
    # the prior read-outs: same counts, alpha0 swapped (Dirichlet mean (counts + a0) / S)
    by = {r.name: r for r in rows}
    for a in CSS_ALPHA0_GRID:
        cnt = bs[2].counts
        want = (cnt + a) / (cnt + a).sum(-1, keepdim=True)
        assert torch.allclose(by[f"css_a{a:g}"].buffer(), want), a
    assert bs[2].alpha0 == 0.5                                  # the copies leave CSS alone
    for r in rows:
        if r.name.startswith("css_a"):
            continue
        o = orig[r.name]
        assert r.kind == o.kind and torch.equal(r.buffer(), o.buffer()), r.name
        ur, uo = r.uncertainties(), o.uncertainties()
        assert set(ur) == set(uo) and all(torch.equal(ur[k], uo[k]) for k in ur), r.name
    loaded = load_baselines(ckpt["baselines"])
    assert loaded[1]._feature.__self__ is loaded[0]                      # rewired to NormLift
    assert loaded[0].beta == 2.0 and loaded[2].alpha0 == 0.5 and loaded[2].class_ids == ids
    (css_q,) = [r for r in row_factory(ckpt, field, "cpu", class_ids=[12, 7]) if r.name == "css"]
    assert torch.equal(css_q.buffer(), bs[2].mean()[:, [2, 0]])
    ours = field_rows(field, "ours") + [ours_scalar_row(field)]          # eval_seg's own rows
    all_names = [r.name for r in ours] + names
    assert len(set(all_names)) == len(all_names)
    with pytest.raises(KeyError):
        row_factory({}, field, "cpu")
    with pytest.raises(ValueError):
        row_factory(ckpt, _field(N + 1, d), "cpu")
    with pytest.raises(ValueError):
        baselines_state([ViewCount(N, feature=NormLift(N, d).feature)])  # source not saved
    with pytest.raises(KeyError):
        CSS(N, torch.eye(K, d)).load_state_dict(CSS(N, torch.eye(K, d), yager=False).state_dict())
    with pytest.raises(ValueError):
        CSS(N, torch.eye(K, d), class_ids=[1, 2, 3, 4]).load_state_dict(bs[2].state_dict())
    print(f"PASS checkpoint: row_factory gives {names} equal to the live rows; CSS columns by "
          f"class id; no clash with the ours_* rows")


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


def test_eval_seg_cli_row_factory(tmp_path):
    """scripts/eval_seg.py --row-factory usplat.baselines:row_factory on the mock backend
    (the toy scene of tests/test_seg_eval.py). --queries all: the CSS vocabulary equals the
    query set, so the unpatched CLI (no class_ids to the factory) can score CSS."""
    pytest.importorskip("cv2")
    from usplat.feature_pca import StreamingPCA
    spec = importlib.util.spec_from_file_location("eval_seg", REPO / "scripts" / "eval_seg.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    def labels():
        lab = np.zeros((480, 640), np.int64)
        lab[40:, :320] = 3
        lab[40:, 320:] = 7
        return lab
    _fake_7scenes(tmp_path / "data", labels)
    D, d = 16, 8
    pca = StreamingPCA(D, d)
    pca.basis = torch.eye(D)[:, :d]
    t = {"chair": torch.eye(D)[0], "table": torch.eye(D)[1], "sofa": torch.eye(D)[2]}
    xs = torch.arange(-1.15, 1.2, 0.04)
    ys = torch.arange(-0.9, 0.95, 0.04)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    means = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.full((gx.numel(),), 2.0)], -1)
    N = means.shape[0]
    splats = dict(means=means, quats=torch.tensor([1.0, 0, 0, 0]).expand(N, 4).clone(),
                  scales=torch.full((N, 3), 0.03), opacities=torch.full((N,), 0.9))
    torch.save(splats, tmp_path / "gs.pt")
    right = means[:, 0] > 0
    feat = torch.where(right[:, None], pca.project(t["table"][None]),
                       pca.project(t["chair"][None]))
    f = DirectionalSemanticField(N, d, FieldConfig(change_test=False))
    f.set_centre(torch.zeros(d))
    text = pca.project_text(torch.stack([t["chair"], t["table"], t["sofa"]]))
    bs = default_baselines(f, text, class_ids=[3, 7, 9])      # vocabulary ids, file order
    for _ in range(4):
        f.update_sparse(torch.arange(N), torch.ones(N), feat, num_pixels=100)
        bs[2].update_sparse(torch.arange(N), torch.ones(N), feat)
    torch.save(dict(field=f.state_dict(), pca=pca.state_dict(), baselines=baselines_state(bs)),
               tmp_path / "field.pt")
    vocab = dict(templates=["a photo of a {}."], ignore=[0],
                 classes={3: {"name": "chair"}, 7: {"name": "table"}, 9: "sofa"})
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    key = lambda p: f"m|p|a photo of a {{}}.|{p}"
    torch.save({key(k): v for k, v in t.items()}, tmp_path / "text.pt")
    scenes = [dict(name="toy", field=str(tmp_path / "field.pt"), ply=str(tmp_path / "gs.pt"),
                   data=str(tmp_path / "data"), dataset="7scenes", stride=1,
                   label_pattern="{root}/frame-{idx:06d}.label.png")]
    (tmp_path / "scenes.json").write_text(json.dumps(scenes))
    out_path = tmp_path / "eval.json"
    cli.main(["--vocab", str(tmp_path / "vocab.json"), "--scenes", str(tmp_path / "scenes.json"),
              "--val-scenes", str(tmp_path / "scenes.json"), "--backend", "mock",
              "--device", "cpu", "--downscale", "16", "--tau-cov", "0.5", "--queries", "all",
              "--row-factory", "usplat.baselines:row_factory",
              "--extractor", "clearclip", "--clip-model", "m", "--clip-pretrained", "p", "--text-cache",
              str(tmp_path / "text.pt"), "--plain-json", "--out", str(out_path)])
    res = json.loads(out_path.read_text())
    rows = res["scenes"]["toy"]["rows"]
    assert {"ours_P0", "averaging", "normlift", "viewcount", "css"} <= set(rows), sorted(rows)
    assert set(res["betas"]) >= {"averaging", "normlift", "viewcount", "css"}
    for name in ("averaging", "normlift", "viewcount", "css"):
        assert rows[name]["all"]["miou"] > 0.9, (name, rows[name]["all"])
    print("PASS eval_seg --row-factory: mIoU (all) "
          + ", ".join(f"{n} {rows[n]['all']['miou']:.3f}"
                      for n in ("ours_P0", "averaging", "normlift", "viewcount", "css")))


if __name__ == "__main__":
    import tempfile
    test_averaging_equals_official_gradbp()
    test_normlift_eq6_eq8_by_hand()
    test_normlift_beta_mask_centre_bruteforce()
    test_view_count()
    test_css_counts_bruteforce()
    test_css_directionless_pixels_are_no_vote()
    test_css_rejects_nonpositive_alpha0()
    test_css_query_columns_follow_class_ids()
    test_dirichlet_vs_monte_carlo()
    test_yager_two_views()
    test_make_rows()
    test_checkpoint_row_factory()
    with tempfile.TemporaryDirectory() as tmp:
        test_eval_seg_cli_row_factory(pathlib.Path(tmp))


# ----------------------------------------------------------- attempt 09: RobustLift
def test_robustlift_without_weighting_equals_field_direction():
    """c0 = -1: every view keeps its weight, so the direction is the field's R / ||R|| (a mapping
    pass: no change test, no resets)."""
    torch.manual_seed(0)
    N, d, V = 30, 8, 6
    ras, P = _views(V, N=N)
    f = _field(N, d, change_test=False)
    rl = B.RobustLift.for_field(f, c0=-1.0)
    for r in ras:
        f.update_view(r, Fn.normalize(torch.randn(P, d), dim=-1), num_pixels=P)
    seen = f.raw.W1 > 0
    want = Fn.normalize(f.raw.R[seen], dim=-1)
    assert torch.allclose(rl.feature()[seen], want, atol=1e-5)
    assert torch.allclose(rl.W, f.raw.W1, atol=1e-5)


def test_robustlift_drops_a_disagreeing_view():
    """Three views of e1, then a view of e2 (cos 0 < c0 = 0.5): its weight is 0, the direction stays
    e1 and the evidence count still grows."""
    from usplat.vmf_field import ViewStats
    d = 4
    rl = B.RobustLift(1, d, c0=0.5, tau=3.0)
    e1, e2 = torch.eye(d)[0], torch.eye(d)[1]
    for u in (e1, e1, e1, e2):
        rl.on_view(ViewStats(R=(2.0 * u)[None], m=torch.tensor([2.0]), Rc=None, mc=None, Ds=None, Dm=None,
                             ref=1.0))
    assert torch.allclose(rl.feature()[0], e1)
    assert rl.W.item() == 4.0
    rl2 = B.RobustLift(1, d, c0=0.5, tau=5.0)          # below tau: no weighting yet
    for u in (e1, e1, e1, e2):
        rl2.on_view(ViewStats(R=(2.0 * u)[None], m=torch.tensor([2.0]), Rc=None, mc=None, Ds=None, Dm=None,
                              ref=1.0))
    assert rl2.F[0, 1].item() == 1.0


def test_robustlift_checkpoint_roundtrip():
    N, d = 12, 6
    ras, P = _views(3, N=N)
    f = _field(N, d)
    bs = [B.RobustLift.for_field(f, c0=0.7)]
    for r in ras:
        f.update_view(r, Fn.normalize(torch.randn(P, d), dim=-1), num_pixels=P)
    buf = io.BytesIO(); torch.save({"baselines": baselines_state(bs)}, buf); buf.seek(0)
    rows = row_factory(torch.load(buf, weights_only=False))
    assert [r.name for r in rows] == ["robust_c0.7"]
    assert torch.allclose(load_baselines(baselines_state(bs))[0].feature(), bs[0].feature())
