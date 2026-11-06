"""CSS over the full vocabulary (the control "CSS was told the scene's classes").

 * scripts/run_mapping.py css_vocabulary(full=True) gives every class of the vocabulary file
   (101 for configs/vocab_replica.json) without reading evaluation frames; full=False gives the
   present classes; with every class present the two are equal;
 * vocabulary == present classes: css_full is css exactly (counts, rows, uncertainties,
   every alpha0 read-out, the evaluator's outputs), also through the checkpoint (compact
   state) and usplat.baselines.row_factory, for a query of every class in any order; on a
   query subset css_full's Dirichlet channels are those of Dirichlet(alpha_cols);
 * css_full's Dirichlet channels describe the renormalised present-class predictive: a
   Gaussian whose votes all went to an absent class is the most uncertain, and the channels
   equal a Monte Carlo of the full Dirichlet restricted to the queried classes and
   renormalised (aggregation + neutrality); a default CSS keeps its all-K channels;
 * extra absent classes: the evaluator's present-column renormalisation equals a float64
   brute force (W^T onehot(argmax over ALL classes) -> Dirichlet mean -> render the present
   columns -> divide by alpha -> renormalise -> temperature), and css_full's present counts
   are css's minus exactly the votes of pixels whose full-vocabulary argmax is absent;
 * row-sparse storage is exact (bit for bit, -0.0 / tiny / NaN values), needs
   2N + 6 nnz bytes, falls back to dense when that is not smaller, loads into dense or sparse
   CSS, and rejects corrupt states;
 * the validation-only rule CSS_FULL_RULE / choose_css_full (lowest mean validation NLL among
   css_full and its alpha0 read-outs, ties to the first listed, css never chosen);
 * scripts/eval_seg.py --row-factory usplat.baselines:row_factory scores css_full and its
   alpha0 read-outs with no change to eval_seg (mock backend).
"""
import importlib.util
import io
import json
import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as Fn

import usplat.baselines as B
from usplat.baselines import (CSS, CSS_ALPHA0_GRID, DIRICHLET_KEYS, baselines_state,
                              default_baselines, dense_rows, full_vocab_css, load_baselines,
                              make_rows, row_factory, sparse_rows, update_all)
from usplat.mock_rasterizer import MockFeatureRasterizer
from usplat.rows import freeze
from usplat.seg_eval import class_scores, evaluate_view, render_row
from usplat.vmf_field import DirectionalSemanticField, FieldConfig

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _views(V, N=40, S=20, seed=0):
    """V mock cameras over the same N Gaussians (Gaussian 0 has opacity 0: never seen)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(N, 2, generator=g) * S
    sig = 1.0 + 1.5 * torch.rand(N, generator=g)
    opac = 0.4 + 0.5 * torch.rand(N, generator=g)
    opac[0] = 0.0
    grid = torch.stack(torch.meshgrid(torch.arange(S), torch.arange(S), indexing="xy"), -1)
    pix = grid.reshape(-1, 2).float() + 0.5
    out = []
    for _ in range(V):
        shift = (torch.rand(1, 2, generator=g) - 0.5) * 6
        out.append(MockFeatureRasterizer(base + shift, sig, opac, torch.rand(N, generator=g),
                                         pix))
    return out, S * S


def _field(N, d):
    f = DirectionalSemanticField(N, d, FieldConfig(change_test=False))
    f.set_centre(torch.zeros(d))
    return f


def _roundtrip(bs):
    buf = io.BytesIO()
    torch.save(dict(baselines=baselines_state(bs)), buf)
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=False)


# ------------------------------------------------------------------ the vocabulary
class _Args:
    def __init__(self, vocab, extractor="lseg"):
        self.vocab, self.extractor = str(vocab), extractor
        self.dataset, self.css_eval_data, self.css_eval_stride = "replica_semnerf", "eval", 10


def test_css_vocabulary_full_and_present(tmp_path, monkeypatch):
    rm = _load_script("run_mapping")
    import usplat.datasets as D
    frames = [dict(labels=torch.tensor([[0, 3], [7, 7]])), dict(labels=torch.tensor([[3, 0]]))]
    calls = []

    def fake_dataset(name, root, stride=1):
        calls.append((name, root, stride))
        return frames
    monkeypatch.setattr(D, "make_dataset", fake_dataset)
    vocab = dict(ignore_ids=[0], template="a photo of a {}.",
                 templates_by_extractor={"lseg": ["a {} here."]},
                 classes={"3": {"name": "chair"}, "7": {"name": "table", "prompt": "a table"},
                          "9": {"name": "sofa"}})
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(vocab))
    ids, prompts, tpl = rm.css_vocabulary(_Args(path))
    assert (ids, prompts, tpl) == ([3, 7], ["chair", "a table"], ["a {} here."])
    assert calls == [("replica_semnerf", "eval", 10)]
    calls.clear()
    fids, fprompts, ftpl = rm.css_vocabulary(_Args(path), full=True)
    assert (fids, fprompts, ftpl) == ([3, 7, 9], ["chair", "a table", "sofa"], ["a {} here."])
    assert calls == []                                        # no evaluation frame read
    # every class present: the full vocabulary IS the present vocabulary
    del vocab["classes"]["9"]
    path.write_text(json.dumps(vocab))
    assert rm.css_vocabulary(_Args(path)) == rm.css_vocabulary(_Args(path), full=True)
    rep = REPO / "configs" / "vocab_replica.json"
    if rep.exists():
        v = json.loads(rep.read_text())
        ids, prompts, tpl = rm.css_vocabulary(_Args(rep), full=True)
        assert len(ids) == len(v["classes"]) == 101 and ids == [int(k) for k in v["classes"]]
        assert tpl == v["templates_by_extractor"]["lseg"]
    print("PASS css_vocabulary: present [3, 7], full [3, 7, 9] (no frames read), equal when "
          "every class is present; Replica: 101 classes")


# ----------------------------------------------------- vocabulary == present classes
def test_css_full_equals_css_when_vocab_is_present():
    torch.manual_seed(0)
    V, d, K = 3, 8, 5
    ras, P = _views(V, seed=1)
    N = ras[0].N
    ids = [12, 4, 30, 7, 9]
    text = Fn.normalize(torch.randn(K, d), dim=-1) * (0.5 + torch.rand(K, 1))
    field = _field(N, d)
    bs = default_baselines(field, text, class_ids=ids)
    bs.append(full_vocab_css(field, text.clone(), list(ids)))
    css, full = bs[2], bs[3]
    for r in ras:
        z = Fn.normalize(torch.randn(P, d), dim=-1)
        z[:10] = 0.0                                          # directionless: no vote in both
        update_all(field, r, z, css=bs, num_pixels=P)
    assert torch.equal(css.counts, full.counts) and css.counts.abs().sum() > 0
    assert full.name == "css_full" and not full.yager and full.storage == "sparse"
    live = {r.name: r for r in make_rows(None, bs)}
    loaded = row_factory(_roundtrip(bs), field, "cpu")
    names = [r.name for r in loaded]
    assert names == (["averaging", "normlift", "viewcount", "css", "css_full"]
                     + [f"css_a{a:g}" for a in CSS_ALPHA0_GRID]
                     + [f"css_full_a{a:g}" for a in CSS_ALPHA0_GRID]), names
    # all columns / a query subset / every class in another order (the evaluated case)
    for q in (None, [7, 12, 9], [9, 4, 12, 30, 7]):
        rows = {r.name: r for r in row_factory(_roundtrip(bs), field, "cpu", class_ids=q)}
        for suffix in [""] + [f"_a{a:g}" for a in CSS_ALPHA0_GRID]:
            a, b = rows["css" + suffix], rows["css_full" + suffix]
            assert a.kind == b.kind == "probs"
            assert torch.equal(a.buffer(), b.buffer()), suffix
            ua, ub = a.uncertainties(), b.uncertainties()
            assert set(ub) == set(DIRICHLET_KEYS) and set(ua) == set(DIRICHLET_KEYS) | {"yager_conflict"}
            if q is None or len(q) == len(ids):               # the same classes: bit for bit
                assert all(torch.equal(ua[k], ub[k]) for k in DIRICHLET_KEYS), suffix
            else:                                             # css_full: Dirichlet(alpha_cols)
                a0 = 1.0 if not suffix else float(suffix[2:])
                cols = sorted(ids.index(c) for c in q)
                want = B.dirichlet_uncertainties(full.counts[:, cols] + a0)
                assert all(torch.equal(ub[k], want[k]) for k in DIRICHLET_KEYS), suffix
                assert not torch.equal(ua["css_var"], ub["css_var"])
    assert torch.equal(live["css_full"].buffer(), live["css"].buffer())
    # the evaluator's outputs are equal too (render, renormalise, temperature)
    gt = torch.randint(0, 3, (P,))
    rr = {n: freeze(r) for n, r in rows.items() if n.startswith("css")}
    out = evaluate_view(list(rr.values()), ras[0], None, gt,
                        {n: 2.0 for n in rr}, tau_cov=0.1)
    for suffix in [""] + [f"_a{a:g}" for a in CSS_ALPHA0_GRID]:
        o1, o2 = out["css" + suffix], out["css_full" + suffix]
        for k in ("pred", "correct", "covered"):                     # discrete: bit for bit
            assert torch.equal(o1[k], o2[k]), (suffix, k)
        for k in ("conf", "nll", "brier"):          # float32 roundoff: the classes are renormalised
            assert torch.allclose(o1[k], o2[k], rtol=0, atol=1e-6), (suffix, k)   # in another order
        assert all(torch.allclose(o1["u"][k], o2["u"][k], rtol=0, atol=1e-6, equal_nan=True)
                   for k in DIRICHLET_KEYS), suffix                  # float32 roundoff, as above
    print(f"PASS css_full == css with the vocabulary equal to the present classes: counts, "
          f"{len(names)} rows through the checkpoint, evaluator outputs")


# --------------------------------------------------- extra absent vocabulary classes
def test_css_full_present_column_renormalisation_exact():
    """K_full = 7 vocabulary classes, the scene holds 4: features near the absent classes
    send votes there; the query keeps the present columns (in a scrambled order) and the
    evaluator renormalises them per pixel."""
    g = torch.Generator().manual_seed(3)
    V, d, Kf = 3, 10, 7
    ras, P = _views(V, N=60, S=18, seed=4)
    N = ras[0].N
    ids = [21, 5, 13, 2, 40, 8, 17]                         # global ids of the 7 texts
    text = Fn.normalize(torch.randn(Kf, d, generator=g), dim=-1)
    present_cols = [4, 0, 5, 2]                              # query order != vocabulary order
    qids = [ids[c] for c in present_cols]
    field = _field(N, d)
    css = CSS.for_field(field, text[sorted(present_cols)], class_ids=[ids[c] for c in sorted(present_cols)])
    full = full_vocab_css(field, text, ids, alpha0=0.7)
    bf_full = torch.zeros(N, Kf, dtype=torch.float64)
    bf_missing = torch.zeros(N, len(present_cols), dtype=torch.float64)
    sp = sorted(present_cols)
    for r in ras:
        own = torch.randint(0, Kf, (P,), generator=g)        # 3 of 7 pixel classes absent
        z = Fn.normalize(text[own] + 0.6 * torch.randn(P, d, generator=g), dim=-1)
        update_all(field, r, z, css=[css, full], num_pixels=P)
        W = r.weights().double()
        s = z.double() @ text.double().t()
        y = s.argmax(-1)
        bf_full += W.t() @ Fn.one_hot(y, Kf).double()
        absent = ~torch.isin(y, torch.tensor(sp))            # full argmax on an absent class
        yp = s[:, sp].argmax(-1)
        bf_missing += W.t() @ (Fn.one_hot(yp, len(sp)).double() * absent[:, None])
    assert (full.counts.double() - bf_full).abs().max() < 1e-4
    assert bf_full[:, [c for c in range(Kf) if c not in sp]].sum() > 1.0   # absent votes exist
    # css_full's present counts = css's counts minus the votes it gave to absent classes
    assert (css.counts.double() - full.counts[:, sp].double() - bf_missing).abs().max() < 1e-4
    (row,) = full.rows(qids)
    assert row.name == "css_full"
    eval_ras = ras[1]
    rr = render_row(freeze(row), eval_ras)
    # brute force in float64
    alpha = bf_full + 0.7
    pbar = alpha / alpha.sum(-1, keepdim=True)
    W = eval_ras.weights().double()
    F = W @ pbar[:, present_cols]
    a = W.sum(1)
    q = F / a[:, None]
    q = q / q.sum(1, keepdim=True)
    seen = a > 1e-6
    got = class_scores(rr).double().exp()
    assert (got[seen] - q[seen]).abs().max() < 1e-5
    gt = torch.randint(0, len(qids), (P,), generator=g)
    for beta in (1.0, 2.5):
        betas = {} if beta == 1.0 else {"css_full": beta}
        raw = ("css_full",) if beta == 1.0 else ()
        o = evaluate_view([freeze(row)], eval_ras, None, gt, betas, tau_cov=0.3,
                          raw_probs=raw)["css_full"]
        p = torch.softmax(beta * torch.log(q.clamp_min(1e-12)), dim=1)
        cov = (a >= 0.3) & seen
        p = torch.where(cov[:, None], p, torch.full_like(p, 1.0 / len(qids)))
        assert torch.equal(o["covered"], cov)
        assert (o["conf"].double() - p.max(1).values).abs().max() < 1e-5
        nll = -torch.log(p.gather(1, gt[:, None]).squeeze(1))
        assert (o["nll"].double() - nll).abs().max() < 1e-4
        dec = o["pred"] >= 0
        assert torch.equal(o["pred"][dec], p.argmax(1)[dec]) and dec.float().mean() > 0.5
    # it is NOT the closed-set CSS over the present classes (the control differs)
    (crow,) = css.rows(qids)
    rc = class_scores(render_row(freeze(crow), eval_ras)).exp()
    assert (rc[seen] - got[seen].float()).abs().max() > 1e-3
    print(f"PASS css_full present-column renormalisation = float64 brute force (beta 1, 2.5); "
          f"absent-class votes {bf_full[:, [c for c in range(Kf) if c not in sp]].sum():.1f} px "
          f"drop out; css - css_full[present] = those pixels' present votes")


def test_css_full_uncertainty_on_queried_columns():
    """Vocabulary ids [1..5], the scene holds [1, 2]. G0 voted only for the absent id 5, G1
    split 5 px between 1 and 2: both have a flat present-class predictive, and G0 (no present
    evidence) must be the more uncertain. Over all 5 columns its channels call it confident."""
    text = torch.eye(5, 3)
    counts = torch.tensor([[0, 0, 0, 0, 5.0], [2.5, 2.5, 0, 0, 0], [4.0, 1.0, 0, 2.0, 0.5]])
    full = CSS(3, text, yager=False, class_ids=[1, 2, 3, 4, 5], name="css_full",
               storage="sparse", query_uncertainty=True)
    plain = CSS(3, text, yager=False, class_ids=[1, 2, 3, 4, 5])        # the default: all K
    full.counts.copy_(counts)
    plain.counts.copy_(counts)
    (r,) = full.rows([2, 1])
    (rp,) = plain.rows([2, 1])
    assert torch.equal(r.buffer(), rp.buffer())
    u, up = r.uncertainties(), rp.uncertainties()
    assert torch.equal(up["css_var_top"], plain.uncertainties()["css_var_top"])  # unchanged
    assert up["css_var_top"][0] < 1.1 * up["css_var_top"][1]  # all-K: G0 as sure as G1
    # closed form on alpha_cols: G0 alpha (1, 1), G1 alpha (3.5, 3.5)
    assert abs(float(u["css_var_top"][0]) - 0.25 / 3) < 1e-7
    assert abs(float(u["css_var_top"][1]) - 0.25 / 8) < 1e-7
    assert u["css_var_top"][0] > 2.5 * u["css_var_top"][1]
    # aggregation + neutrality: theta[:, cols] / sum(theta[:, cols]) ~ Dirichlet(alpha_cols)
    torch.manual_seed(0)
    alpha = (counts + 1.0).double()
    cols = [0, 1]
    for n in range(3):
        th = torch.distributions.Dirichlet(alpha[n]).sample((400000,))[:, cols]
        th = th / th.sum(1, keepdim=True)
        c = int(alpha[n, cols].argmax())
        mc = dict(css_var_top=float(th[:, c].var()), css_var=float(th.var(0).sum()),
                  css_exp_entropy=float((-torch.special.xlogy(th, th).sum(1)).mean()))
        for k, v in mc.items():
            assert abs(float(u[k][n]) - v) < 2e-3 * max(1.0, v), (n, k, float(u[k][n]), v)
    # through the checkpoint and the alpha0 read-outs
    lb = load_baselines(_roundtrip([full])["baselines"])
    assert lb[0].query_uncertainty and lb[0].config()["query_uncertainty"] is True
    (r2,) = lb[0].with_alpha0(0.1).rows([1, 2])
    want = B.dirichlet_uncertainties(counts[:, cols] + 0.1)
    assert all(torch.equal(r2.uncertainties()[k], want[k]) for k in DIRICHLET_KEYS)
    assert "query_uncertainty" not in plain.config()
    print(f"PASS css_full channels on the queried columns: var_top G0 "
          f"{float(u['css_var_top'][0]):.4f} > G1 {float(u['css_var_top'][1]):.4f} (all-K "
          f"{float(up['css_var_top'][0]):.4f} ~ {float(up['css_var_top'][1]):.4f}); = Monte Carlo")


# ---------------------------------------------------------------------- storage
def test_sparse_rows_exact_and_compact():
    g = torch.Generator().manual_seed(7)
    x = torch.rand(500, 101, generator=g)
    x[torch.rand(500, 101, generator=g) > 0.04] = 0.0         # ~4 nonzeros per Gaussian
    x[3] = 0.0                                                 # an unseen Gaussian
    x[5, 7] = 1e-38                                            # tiny / subnormal values
    x[5, 8] = 1e-45
    x[6, 100] = 3.0e5
    x[7, 3] = -0.0                                             # kept: bit for bit
    s = sparse_rows(x)
    assert s["nnz"].dtype == torch.int16 and s["col"].dtype == torch.int16
    y = dense_rows(s)
    assert torch.equal(x.view(torch.int32), y.view(torch.int32)) and y.dtype == x.dtype
    nnz = int(((x != 0) | torch.signbit(x)).sum())
    assert s["nnz"].nbytes + s["col"].nbytes + s["val"].nbytes == 2 * 500 + 6 * nnz
    # NaN is kept as a value (dense_rows restores it in place)
    xn = x.clone()
    xn[9, 1] = float("nan")
    yn = dense_rows(sparse_rows(xn))
    assert torch.isnan(yn[9, 1]) and torch.equal(torch.nan_to_num(yn), torch.nan_to_num(xn))
    # through CSS state_dict -> torch.save -> a dense or a sparse CSS
    text = torch.eye(101, 4)
    c = CSS(500, text, yager=False, storage="sparse", name="css_full")
    c.counts.copy_(x)
    buf = io.BytesIO()
    torch.save(c.state_dict(), buf)
    n_sparse = buf.tell()
    dense = CSS(500, text, yager=False)
    dense.counts.copy_(x)
    buf2 = io.BytesIO()
    torch.save(dense.state_dict(), buf2)
    assert n_sparse < buf2.tell() / 5, (n_sparse, buf2.tell())
    buf.seek(0)
    st = torch.load(buf, weights_only=False)
    for tgt in (CSS(500, text, yager=False, storage="sparse"), CSS(500, text, yager=False)):
        tgt.load_state_dict(st)
        assert torch.equal(tgt.counts, x)
    # with Yager opinions every (N, K) tensor is row-sparse
    cy = CSS(500, text, yager=True, storage="sparse")
    cy.update_counts(x)
    sy = cy.state_dict()
    assert B.is_sparse_rows(sy["counts"]) and B.is_sparse_rows(sy["yb"]) and sy["yu"].dim() == 1
    cy2 = CSS(500, text, yager=True)
    cy2.load_state_dict(sy)
    assert torch.equal(cy2.yb, cy.yb) and torch.equal(cy2.counts, cy.counts)
    bad = dict(s, nnz=s["nnz"][:-1])
    with pytest.raises(ValueError, match="corrupt"):
        dense_rows(bad)
    with pytest.raises(ValueError):
        CSS(2, text, storage="fp16")
    with pytest.raises(ValueError, match="int16"):
        CSS(1, torch.zeros(40000, 2), storage="sparse")
    with pytest.raises(ValueError):
        sparse_rows(torch.zeros(3))
    # dense fallback: row-sparse only while 2N + 6 nnz < 4NK (here N = K = 3: nnz < 5)
    for k_nz in (4, 5, 9):
        xs = torch.zeros(3, 3)
        xs.view(-1)[:k_nz] = torch.arange(1, k_nz + 1).float()
        cs = CSS(3, torch.eye(3, 2), yager=False, storage="sparse")
        cs.counts.copy_(xs)
        st = cs.state_dict()["counts"]
        assert B.is_sparse_rows(st) == (k_nz < 5), k_nz
        assert B.is_sparse_rows(B.compact_rows(xs)) == (k_nz < 5)
        back = CSS(3, torch.eye(3, 2), yager=False)
        back.load_state_dict(cs.state_dict())
        assert torch.equal(back.counts, xs)
    print(f"PASS row-sparse storage: exact (-0.0, tiny, subnormal, NaN values), {n_sparse} B vs "
          f"dense {buf2.tell()} B for 500 x 101 at {nnz / 500:.1f} nonzeros per Gaussian; "
          f"dense when not smaller")


def test_full_vocab_css_checkpoint_config():
    """A default CSS saves the v1 config (no new keys); css_full records its name, storage
    and no Yager state; load_baselines rebuilds both."""
    torch.manual_seed(2)
    ras, P = _views(2, seed=9)
    N, d = ras[0].N, 6
    field = _field(N, d)
    text = Fn.normalize(torch.randn(6, d), dim=-1)
    bs = default_baselines(field, text[:3], class_ids=[1, 2, 3])
    bs.append(full_vocab_css(field, text, [1, 2, 3, 4, 5, 6]))
    for r in ras:
        update_all(field, r, Fn.normalize(torch.randn(P, d), dim=-1), css=bs, num_pixels=P)
    saved = baselines_state(bs)
    cfg_css, cfg_full = saved[2]["config"], saved[3]["config"]
    assert set(cfg_css) == {"num_gaussians", "text", "alpha0", "yager", "class_ids", "dtype"}
    assert cfg_full["name"] == "css_full" and cfg_full["storage"] == "sparse"
    assert cfg_full["query_uncertainty"] is True
    assert cfg_full["yager"] is False and set(saved[3]["state"]) == {"counts", "class_ids"}
    nnz = int((bs[3].counts != 0).sum())                      # row-sparse only if smaller
    assert B.is_sparse_rows(saved[3]["state"]["counts"]) == (2 * N + 6 * nnz < 4 * N * 6)
    assert torch.is_tensor(saved[2]["state"]["counts"])       # css: dense, as in v1
    lb = load_baselines(_roundtrip(bs)["baselines"])
    assert lb[3].name == "css_full" and lb[3].storage == "sparse" and lb[3].class_ids == [1, 2, 3, 4, 5, 6]
    assert lb[3].query_uncertainty and not lb[2].query_uncertainty
    assert torch.equal(lb[3].counts, bs[3].counts) and torch.equal(lb[2].counts, bs[2].counts)
    (q,) = [r for r in row_factory(_roundtrip(bs), field, "cpu", class_ids=[3, 1])
            if r.name == "css_full"]
    assert torch.equal(q.buffer(), bs[3].mean()[:, [2, 0]])
    with pytest.raises(ValueError, match="not in the CSS vocabulary"):
        bs[3].rows(class_ids=[7])
    print("PASS checkpoint: v1 CSS config unchanged; css_full name/storage/query_uncertainty "
          "recorded, no Yager")


# ------------------------------------------------------------- validation-only rule
def test_css_full_rule_is_validation_only():
    r = B.CSS_FULL_RULE
    assert r == dict(rows=("css_full", "css_full_a0.01", "css_full_a0.1", "css_full_a10"),
                     criterion="nll", stat="mean", variant="all", stage="val", test=False)
    for s in ("VALIDATION-ONLY", "CSS_FULL_RULE"):
        assert s in B.__doc__, s
    nll = lambda v: {"all": {"nll": {"mean": v}}}
    summ = {"css": nll(0.1), "css_a0.01": nll(0.05), "css_full": nll(0.9),
            "css_full_a0.1": nll(0.5), "css_full_a0.01": nll(0.5), "css_full_a10": nll("nan"),
            "ours_P2": nll(0.01)}
    assert B.choose_css_full(summ) == "css_full_a0.01"        # tie: the first listed
    assert B.choose_css_full(dict(summ, css_full=nll(0.4))) == "css_full"
    assert B.choose_css_full({"css": nll(0.1)}) is None
    assert B.choose_css_full({"css_full": {"covered": {}}}) is None
    print("PASS CSS_FULL_RULE: validation only, lowest mean NLL among css_full*, NaN skipped")


# ------------------------------------------------------------------- eval_seg CLI
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


def test_eval_seg_cli_scores_css_full(tmp_path):
    """scripts/eval_seg.py --row-factory usplat.baselines:row_factory --queries present: CSS
    mapped on the present classes [3, 7], css_full on the whole vocabulary [3, 7, 9]; both
    are queried on [3, 7], get their own betas and score the toy scene."""
    pytest.importorskip("cv2")
    from usplat.feature_pca import StreamingPCA
    cli = _load_script("eval_seg")

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
    bs = default_baselines(f, text[:2], class_ids=[3, 7])     # present classes
    bs.append(full_vocab_css(f, text, [3, 7, 9]))             # the whole vocabulary
    for _ in range(4):
        f.update_sparse(torch.arange(N), torch.ones(N), feat, num_pixels=100)
        bs[2].update_sparse(torch.arange(N), torch.ones(N), feat)
        bs[3].update_sparse(torch.arange(N), torch.ones(N), feat)
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
              "--device", "cpu", "--downscale", "16", "--tau-cov", "0.5", "--queries", "present",
              "--row-factory", "usplat.baselines:row_factory", "--no-p3",
              "--extractor", "clearclip", "--clip-model", "m", "--clip-pretrained", "p",
              "--text-cache", str(tmp_path / "text.pt"), "--plain-json", "--out", str(out_path)])
    res = json.loads(out_path.read_text())
    rows = res["scenes"]["toy"]["rows"]
    full_rows = ["css_full"] + [f"css_full_a{a:g}" for a in CSS_ALPHA0_GRID]
    assert set(full_rows) <= set(rows) and set(full_rows) <= set(res["betas"]), sorted(rows)
    assert res["scenes"]["toy"]["class_ids"] == [3, 7]
    for name in ["css"] + full_rows:
        assert rows[name]["all"]["miou"] > 0.9, (name, rows[name]["all"])
        assert set(rows[name]["all"]["ranking"]) >= set(DIRICHLET_KEYS)
    assert "yager_conflict" not in rows["css_full"]["all"]["ranking"]
    assert B.choose_css_full(res["summary"]) in B.CSS_FULL_RULE["rows"]
    print("PASS eval_seg --row-factory: css_full* scored on the present classes, mIoU "
          + ", ".join(f"{n} {rows[n]['all']['miou']:.3f}" for n in ["css"] + full_rows))


if __name__ == "__main__":
    import tempfile
    test_css_full_equals_css_when_vocab_is_present()
    test_css_full_present_column_renormalisation_exact()
    test_css_full_uncertainty_on_queried_columns()
    test_sparse_rows_exact_and_compact()
    test_full_vocab_css_checkpoint_config()
    test_css_full_rule_is_validation_only()
    with tempfile.TemporaryDirectory() as tmp:
        test_eval_seg_cli_scores_css_full(pathlib.Path(tmp))
