"""P4 vote predictive and the P1 prior grid (usplat.predictive, usplat.rows; Table I v2).

 * tangent_var = A_d/kappa is the vMF tangent variance (identity A' = 1 - A^2 - (d-1) A/kappa),
   1/d at kappa = 0;
 * (i) the tangent-Gaussian Monte Carlo vote fractions match an exact vMF sampler (Wood's
   rejection algorithm) on 2- and 3-class geometries in d = 64 for several kappa, within
   the Monte Carlo error, and on non-unit texts sharing a component within the documented
   error of its fixed radial part (0.075 at kappa <= 10 ... 0.008 at kappa >= 100);
 * the vote concentration: the kish kappa is unbiased at 2..16 views of one vMF pixel each
   (the plug-in rbar kappa, the ablation P4_rbar, is biased upward); a Gaussian with one
   effective view takes the median kappa of the defined ones;
 * (ii) kappa -> inf (rho -> 1) gives the one-hot argmax (= the zero-noise row P4_inf);
   rho -> 0 gives near-equal votes for texts placed symmetrically around mu*;
 * (iii) the same output twice, the same output for any chunk size, and permuting the
   texts permutes the votes;
 * (iv) alpha0 -> 0 gives the votes, alpha0 -> inf the uniform distribution, an unseen
   Gaussian is uniform for every alpha0;
 * rows: names of the grids, one Monte Carlo run per lam (VoteCache) that a field update
   invalidates, the evaluator renders the mixture of q_n, uncertainty channels as the
   other field rows;
 * P1 c0 grid: c0 = FieldConfig.c0 reproduces field.mean_direction() bit for bit (also with
   a text prior and with kappa_mode "fixed"), smaller c0 lengthens E[mu], field.cfg is
   never modified; v2_rows builds ours_P2p1 as field_rows' P2 on the P1 base.
"""
import math

import pytest
import torch
import torch.nn.functional as Fn

import usplat.seg_eval as seg_eval
from usplat.predictive import (argmax_votes, posterior_mean, tangent_noise, vote_fractions,
                               vote_kappa, vote_predictive, gaussian_buffer)
from usplat.rows import (FieldRow, FieldVoteRow, VoteCache, freeze, p1_prior_rows, p4_rows,
                         v2_rows, P1_C0S, P4_ALPHA0S, P4_LAMBDAS, P4_INF_ALPHA0, P4_RBAR)
from usplat.vmf_field import DirectionalSemanticField, FieldConfig
from usplat.vmf_math import a_d_inv, a_d_scalar, tangent_var


def wood_vmf(mu, kappa, n):
    """Exact samples of vMF(mu, kappa) on S^{d-1} (Wood 1994, Commun. Stat. Simul. 23(1)):
    w = mu^T x by rejection from a Beta proposal, x = w mu + sqrt(1 - w^2) v, v uniform
    on the tangent sphere. float64; uses the global RNG (seeded by the caller)."""
    d = mu.numel()
    m1 = d - 1.0
    b = (-2.0 * kappa + math.sqrt(4.0 * kappa ** 2 + m1 ** 2)) / m1
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + m1 * math.log(1.0 - x0 ** 2)
    beta = torch.distributions.Beta(torch.tensor(m1 / 2, dtype=torch.float64),
                                    torch.tensor(m1 / 2, dtype=torch.float64))
    got, ws = 0, []
    while got < n:
        k = 2 * (n - got) + 100
        z = beta.sample((k,))
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = torch.rand(k, dtype=torch.float64)
        ok = kappa * w + m1 * torch.log(1.0 - x0 * w) - c >= torch.log(u)
        ws.append(w[ok])
        got += int(ok.sum())
    w = torch.cat(ws)[:n]
    m = mu.double() / mu.double().norm()
    v = torch.randn(n, d, dtype=torch.float64)
    v = Fn.normalize(v - (v @ m)[:, None] * m, dim=-1)
    return w[:, None] * m + (1.0 - w * w).clamp_min(0).sqrt()[:, None] * v


def _texts(d, angles):
    """Unit texts at the given (theta, phi) around mu = e_0: t = cos th e0 + sin th (cos phi
    e1 + sin phi e2)."""
    t = torch.zeros(len(angles), d)
    for k, (th, ph) in enumerate(angles):
        t[k, 0] = math.cos(th)
        t[k, 1] = math.sin(th) * math.cos(ph)
        t[k, 2] = math.sin(th) * math.sin(ph)
    return t


def _field(N=40, d=16, K=4, seed=0, **cfg):
    g = torch.Generator().manual_seed(seed)
    f = DirectionalSemanticField(N, d, FieldConfig(change_test=False, **cfg))
    text = Fn.normalize(torch.randn(K, d, generator=g), dim=-1)
    lab = torch.arange(N) % K
    for _ in range(5):
        z = Fn.normalize(text[lab] + 0.6 * torch.randn(N, d, generator=g), dim=-1)
        f.update_sparse(torch.arange(N - 4), torch.rand(N - 4, generator=g) + 0.1, z[:N - 4],
                        num_pixels=100)                          # the last 4 stay unseen
    return f, text


# ------------------------------------------------------------------------------- math
def test_tangent_var_is_the_vmf_tangent_variance():
    """A_d(kappa)/kappa = (1 - E[(mu^T x)^2]) / (d - 1) with E[(mu^T x)^2] = A^2 + A'(kappa)
    (t = mu^T x is the sufficient statistic, so Var t = dA/dkappa)."""
    for d in (3, 16, 64):
        for kap in (0.5, 5.0, 40.0, 300.0):
            h = 1e-4 * kap
            A = a_d_scalar(d, kap)
            dA = (a_d_scalar(d, kap + h) - a_d_scalar(d, kap - h)) / (2 * h)
            want = (1.0 - A * A - dA) / (d - 1)
            got = float(tangent_var(d, torch.tensor([kap], dtype=torch.float64)))
            assert abs(got - want) < 1e-4 * want + 1e-9, (d, kap, got, want)
        assert float(tangent_var(d, torch.zeros(1))) == pytest.approx(1.0 / d)
    print("PASS tangent_var = A/kappa = vMF tangent variance; 1/d at kappa = 0")


@pytest.mark.parametrize("name,angles", [
    ("2 classes", [(0.25, 0.0), (0.55, math.pi)]),
    ("3 classes", [(0.0, 0.0), (0.45, 0.0), (0.45, 2.2)]),
])
def test_votes_match_exact_vmf_sampler(name, angles):
    """(i) Monte Carlo vote fractions (S = 8192, lam = 1) against Wood's exact sampler
    (n = 100000 per kappa) in d = 64: within 4 binomial standard errors of the difference
    plus 0.01 for the fixed radial part of the tangent approximation."""
    torch.manual_seed(0)
    d = 64
    mu = torch.zeros(d)
    mu[0] = 1.0
    t = _texts(d, angles)
    kappas = torch.tensor([5.0, 20.0, 60.0, 200.0])
    S, n = 8192, 100_000
    v = vote_fractions(mu.expand(len(kappas), d), kappas, t, lam=1.0, n_samples=S)
    worst = 0.0
    for i, kap in enumerate(kappas.tolist()):
        z = wood_vmf(mu, kap, n)
        ve = torch.bincount((z @ t.double().t()).argmax(-1), minlength=len(angles)).double() / n
        p = ve.clamp(1e-3, 1 - 1e-3)
        tol = 4.0 * torch.sqrt(p * (1 - p) * (1.0 / S + 1.0 / n)) + 0.01
        diff = (v[i].double() - ve).abs()
        assert (diff <= tol).all(), (name, kap, v[i].tolist(), ve.tolist(), tol.tolist())
        worst = max(worst, float(diff.max()))
    assert v.sum(-1).sub(1).abs().max() < 1e-6
    print(f"PASS votes vs exact vMF ({name}): max |MC - Wood| {worst:.4f} over kappa "
          f"{kappas.tolist()}")


def test_votes_within_the_documented_error_on_shared_texts():
    """(i) realistic texts: norms 0.4-0.9, a common component (pairwise cosine ~ 0.8 of the
    directions), mu near one class: the tangent approximation (S = 8192) against Wood's
    sampler (n = 100000) is within 4 standard errors plus the error documented for its fixed
    radial part in vote_fractions (0.075 at kappa <= 10, 0.04 at 20, 0.02 at 60, 0.008 at 200)."""
    torch.manual_seed(3)
    g = torch.Generator().manual_seed(7)
    d, K = 64, 3
    c = Fn.normalize(torch.randn(d, generator=g), dim=0)
    dl = torch.randn(K, d, generator=g)
    dl = Fn.normalize(dl - (dl @ c)[:, None] * c, dim=-1)
    th = Fn.normalize(math.sqrt(0.8) * c + math.sqrt(0.2) * dl, dim=-1)
    t = (0.4 + 0.5 * torch.rand(K, generator=g))[:, None] * th
    xi = torch.randn(d, generator=g)
    xi = Fn.normalize(xi - (xi @ th[0]) * th[0], dim=0)
    mu = Fn.normalize(0.75 * th[0] + 0.66 * xi, dim=0)
    bound = {5.0: 0.075, 20.0: 0.04, 60.0: 0.02, 200.0: 0.008}
    kappas = torch.tensor(list(bound))
    S, n = 8192, 100_000
    v = vote_fractions(mu.expand(len(kappas), d), kappas, t, lam=1.0, n_samples=S)
    worst = {}
    for i, kap in enumerate(kappas.tolist()):
        z = wood_vmf(mu, kap, n)
        ve = torch.bincount((z @ t.double().t()).argmax(-1), minlength=K).double() / n
        p = ve.clamp(1e-3, 1 - 1e-3)
        tol = 4.0 * torch.sqrt(p * (1 - p) * (1.0 / S + 1.0 / n)) + bound[kap]
        diff = (v[i].double() - ve).abs()
        assert (diff <= tol).all(), (kap, v[i].tolist(), ve.tolist(), tol.tolist())
        worst[kap] = round(float(diff.max()), 4)
    print(f"PASS votes vs exact vMF on shared non-unit texts: max |MC - Wood| per kappa {worst}")


def test_vote_kappa_unbiased_at_few_views():
    """The kish kappa of P4 (vote_kappa default) through the field: Gaussians seen in n views,
    one pixel z ~ vMF(mu, kappa) per view, 3 px of mass per view at 320 x 240: its median is
    within 10% of kappa at n = 2, 4, 16, where the plug-in rbar kappa (P4_rbar) is 1.3-5x too
    large; one view leaves kish undefined and those Gaussians get the median kappa of the
    defined ones, not rbar's 1e7 (one-hot votes)."""
    d, G, ns = 64, 400, (1, 2, 4, 16)
    mu = torch.zeros(d)
    mu[0] = 1.0
    for kt in (20.0, 200.0):
        torch.manual_seed(int(kt))
        f = DirectionalSemanticField(G * len(ns), d, FieldConfig(change_test=False))
        for v in range(max(ns)):
            ids = torch.cat([torch.arange(i * G, (i + 1) * G) for i, n in enumerate(ns) if v < n])
            z = wood_vmf(mu, kt, ids.numel()).float()
            f.update_sparse(ids, torch.full((ids.numel(),), 3.0), z, num_pixels=320 * 240)
        kk, kr = vote_kappa(f), vote_kappa(f, "rbar")
        assert torch.equal(vote_kappa(f, "kish"), kk)
        defined = torch.isfinite(f.raw.agreement("kish", f.cfg.c0))
        for i, n in enumerate(ns):
            sl = slice(i * G, (i + 1) * G)
            if n == 1:
                assert not defined[sl].any()
                assert torch.equal(kk[sl], torch.full((G,), float(kk[defined].median())))
                assert float(kr[sl].median()) > 1e6                 # rbar: one-hot votes
                continue
            assert defined[sl].all()
            med = float(kk[sl].median())
            assert abs(med / kt - 1) < 0.1, (kt, n, med)
            if n <= 4:
                assert float(kr[sl].median()) > 1.3 * kt, (kt, n, float(kr[sl].median()))
    empty = DirectionalSemanticField(3, d, FieldConfig(change_test=False))
    assert torch.equal(vote_kappa(empty), torch.zeros(3))      # nothing defined: kappa 0
    with pytest.raises(ValueError, match="kappa estimator"):
        vote_kappa(empty, "plugin")
    print("PASS kish vote kappa unbiased at 2..16 views; one view: the field's median kappa")


def test_kappa_limits():
    """(ii) rho -> 1: kappa = A_d^{-1}(1 - 1e-6) gives the one-hot argmax; rho = 0
    (kappa = 0): near-equal votes for texts placed symmetrically around mu."""
    torch.manual_seed(1)
    d, K, N = 64, 5, 200
    mu = Fn.normalize(torch.randn(N, d), dim=-1)
    t = Fn.normalize(torch.randn(K, d), dim=-1)
    kap = a_d_inv(d, torch.full((N,), 1.0 - 1e-6))
    v = vote_fractions(mu, kap, t, lam=1.0)
    s = mu @ t.t()
    top2 = s.topk(2, dim=-1).values
    clear = (top2[:, 0] - top2[:, 1]) > 1e-2                    # not a near tie
    assert clear.sum() > 150
    assert torch.equal(v[clear], argmax_votes(mu, t)[clear])
    # symmetric texts: the same angle to mu, orthogonal tangent directions
    mu0 = torch.zeros(1, d)
    mu0[0, 0] = 1.0
    tt = torch.zeros(4, d)
    tt[:, 0] = math.cos(0.7)
    tt[torch.arange(4), 1 + torch.arange(4)] = math.sin(0.7)
    v0 = vote_fractions(mu0, torch.zeros(1), tt, lam=1.0, n_samples=4096)
    assert (v0 - 0.25).abs().max() < 0.03, v0
    # through a field: two opposite observations give rho = 0, hence kappa = 0
    f = DirectionalSemanticField(1, d, FieldConfig(change_test=False))
    f.update_sparse(torch.zeros(2, dtype=torch.long), torch.ones(2),
                    torch.stack([mu0[0], -mu0[0]]), num_pixels=100)
    assert float(vote_kappa(f)) == 0.0
    print(f"PASS limits: rho -> 1 one-hot on {int(clear.sum())}/{N} clear Gaussians; "
          f"rho = 0 symmetric votes {v0[0].tolist()}")


def test_determinism_chunks_and_text_permutation():
    """(iii) same output twice; chunk sizes 1, 7, 4096 give identical votes; permuting the
    texts permutes the vote columns."""
    torch.manual_seed(2)
    d, K, N = 32, 6, 50
    mu = Fn.normalize(torch.randn(N, d), dim=-1)
    mu[3] = 0                                                   # an unseen Gaussian
    t = 0.7 * Fn.normalize(torch.randn(K, d), dim=-1)           # texts need not be unit
    kap = torch.rand(N) * 80
    v = vote_fractions(mu, kap, t, lam=1.0, n_samples=128)
    assert torch.equal(v, vote_fractions(mu, kap, t, lam=1.0, n_samples=128))
    for c in (1, 7, 4096):
        assert torch.equal(v, vote_fractions(mu, kap, t, lam=1.0, n_samples=128, chunk=c)), c
    perm = torch.randperm(K)
    vp = vote_fractions(mu, kap, t[perm], lam=1.0, n_samples=128)
    assert torch.equal(vp, v[:, perm])
    assert torch.equal(argmax_votes(mu, t[perm]), argmax_votes(mu, t)[:, perm])
    assert not torch.equal(v, vote_fractions(mu, kap, t, lam=1.0, n_samples=128, seed=1))
    e = tangent_noise(d, 8)
    assert torch.equal(e[:4], -e[4:])                           # antithetic pairs
    with pytest.raises(ValueError, match="even"):
        tangent_noise(d, 7)
    print("PASS determinism, chunk-size invariance, text-permutation equivariance")


def test_alpha0_limits():
    """(iv) q = (alpha0 + W1 v)/(K alpha0 + W1): alpha0 -> 0 gives v, alpha0 -> inf the
    uniform; W1 = 0 is uniform for every alpha0; rows sum to 1."""
    v = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.25, 0.25], [0.2, 0.3, 0.5]])
    w = torch.tensor([3.0, 0.5, 0.0])
    assert torch.allclose(vote_predictive(v, w, 1e-9)[:2], v[:2], atol=1e-6)
    assert torch.allclose(vote_predictive(v, w, 1e9), torch.full((3, 3), 1 / 3), atol=1e-6)
    for a in (1e-9, 0.1, 1.0, 1e9):
        q = vote_predictive(v, w, a)
        assert torch.allclose(q[2], torch.full((3,), 1 / 3))
        assert torch.allclose(q.sum(-1), torch.ones(3))
    q = vote_predictive(v, w, 0.1)
    assert torch.allclose(q[0], torch.tensor([3.1, 0.1, 0.1]) / 3.3)
    assert (q > 0).all()
    f, text = _field()
    with pytest.raises(ValueError, match="alpha0"):
        FieldVoteRow("x", "probs", field=f, text=text, alpha0=0.0)
    with pytest.raises(ValueError, match="probs"):
        FieldVoteRow("x", "feature", field=f, text=text)
    with pytest.raises(ValueError, match="lam"):
        FieldVoteRow("x", "probs", field=f, text=text, lam=-1.0)
    with pytest.raises(ValueError, match="kappa_estimator"):
        FieldVoteRow("x", "probs", field=f, text=text, kappa_estimator="plugin")
    print("PASS alpha0 limits and unseen Gaussians")


# ------------------------------------------------------------------------------- rows
def test_p4_rows_cache_and_mixture():
    f, text = _field()
    rows = p4_rows(f, text)
    want = [f"ours_P4_a{a:g}_s{lam:g}" for lam in P4_LAMBDAS for a in P4_ALPHA0S] + [
        "ours_P4_inf", "ours_P4_rbar"]
    assert [r.name for r in rows] == want
    assert rows[-1].cache is None and rows[-1].kappa_estimator == "rbar"
    cache = rows[0].cache
    fr = {r.name: freeze(r) for r in rows}
    assert cache.runs == len(P4_LAMBDAS)                        # one Monte Carlo per lam
    W1 = f.n_eff
    for r in rows[:-2]:                                         # the row is the formula
        v = vote_fractions(f.mu_post(), vote_kappa(f), text, r.lam)
        assert torch.equal(fr[r.name].buf, vote_predictive(v, W1, r.alpha0)), r.name
    vr = vote_fractions(f.mu_post(), vote_kappa(f, "rbar"), text, P4_RBAR[1])
    assert torch.equal(fr["ours_P4_rbar"].buf, vote_predictive(vr, W1, P4_RBAR[0]))
    inf = fr["ours_P4_inf"].buf
    assert torch.equal(inf, vote_predictive(argmax_votes(f.mu_post(), text), W1, P4_INF_ALPHA0))
    unseen = W1 == 0
    assert unseen.sum() == 4
    for r in fr.values():
        assert torch.allclose(r.buf[unseen], torch.full((4, text.shape[0]), 0.25))
        assert torch.allclose(r.buf.sum(-1), torch.ones(f.N), atol=1e-6)
        assert set(r.unc) == {"u_epi", "u_ale"}
        assert torch.equal(r.unc["u_epi"], f.epistemic())
    # a seen Gaussian votes mostly for its own class (the toy's kish kappa is ~4-12 in d = 16:
    # a wide spread, so the top vote is not always the mode's class, which P4_inf votes for)
    lab = torch.arange(f.N) % text.shape[0]
    acc = lambda q: float((q[~unseen].argmax(-1) == lab[~unseen]).float().mean())
    assert acc(fr["ours_P4_a0.01_s1"].buf) > 0.8 and acc(fr["ours_P4_inf"].buf) > 0.9
    # an update of the field invalidates the shared votes
    f.update_sparse(torch.arange(3), torch.ones(3), Fn.normalize(torch.randn(3, f.d), dim=-1),
                    num_pixels=100)
    rows[0].buffer()
    assert cache.runs == len(P4_LAMBDAS) + 1
    v = vote_fractions(f.mu_post(), vote_kappa(f), text, rows[0].lam)
    assert torch.equal(rows[0].buffer(), vote_predictive(v, f.n_eff, rows[0].alpha0))
    # the evaluator mixes the per-Gaussian predictives: q_p = sum_n w q_n / alpha_p
    w = torch.zeros(1, f.N)
    w[0, :3] = torch.tensor([0.5, 0.25, 0.1])
    r0 = freeze(rows[4])
    rr = seg_eval.render_row(r0, lambda buf: w @ buf)
    assert torch.allclose(rr.feat / rr.alpha[:, None], (w @ r0.buf) / w.sum(), atol=1e-6)
    print(f"PASS P4 rows {want[0]} .. {want[-1]}: {cache.runs} MC runs, mixture render")


def test_vote_cache_reuses_only_identical_inputs():
    c = VoteCache()
    a = torch.ones(3)
    calls = []
    run = lambda: calls.append(1) or torch.zeros(1)
    c.get((a,), "k", run)
    c.get((a.clone(),), "k", run)                               # equal content: reused
    assert len(calls) == 1
    c.get((a.double(),), "k", run)                              # another dtype: recomputed
    c.get((a + 1e-7,), "k", run)
    assert len(calls) == 3 and c.runs == 3


def test_p1_prior_grid_reproduces_mean_direction():
    """c0 = FieldConfig.c0 reproduces mean_direction exactly (plain, with a text prior, with
    kappa_mode 'fixed'); smaller c0 gives a longer E[mu]; field.cfg is untouched."""
    for cfg in (dict(), dict(prior_kappa=2.0), dict(kappa_mode="fixed")):
        f, text = _field(**cfg)
        if cfg.get("prior_kappa"):
            f.prior_mu = Fn.normalize(torch.randn(f.N, f.d), dim=-1)
        rows = {r.name: r for r in p1_prior_rows(f)}
        assert list(rows) == [f"ours_P1_c{c:g}" for c in P1_C0S]
        assert torch.equal(rows["ours_P1_c1"].buffer(), f.mean_direction()), cfg
        assert torch.equal(posterior_mean(f), f.mean_direction())
        assert torch.equal(gaussian_buffer(f, "P0", f.cfg.c0), f.mu_post())
        assert f.cfg.c0 == 1.0
        n = {c: rows[f"ours_P1_c{c:g}"].buffer().norm(dim=-1) for c in P1_C0S}
        seen = f.n_eff > 0
        if cfg.get("kappa_mode") == "fixed":                    # kappa_lik: c0 has no effect
            assert all(torch.equal(n[c], n[1.0]) for c in P1_C0S)
        else:
            for lo, hi in zip(P1_C0S[:-1], P1_C0S[1:]):
                assert (n[lo][seen] >= n[hi][seen]).all() and (n[lo][seen] > n[hi][seen]).any()
        if cfg.get("prior_kappa"):                              # c0 moves mu* through the prior
            assert not torch.allclose(gaussian_buffer(f, "P0", 0.01), f.mu_post())
    with pytest.raises(ValueError, match="c0"):
        FieldRow("x", "feature", field=f, mode="P1", c0=0.0)
    print("PASS P1 c0 grid: c0 = 1 is mean_direction bit for bit; monotone in c0")


def test_v2_rows():
    f, text = _field()
    rows = v2_rows(f, text)
    names = [r.name for r in rows]
    assert names[0] == "ours_P2p1" and len(names) == len(set(names)) == 1 + 5 + 9 + 1 + 1
    p2 = rows[0]
    assert p2.adaptive and p2.mode == "P1" and p2.kind == "feature"
    assert torch.equal(p2.buffer(), f.mean_direction())
    assert [r.name for r in v2_rows(f)] == names[:6]            # no text: no P4 rows
    print(f"PASS v2_rows: {len(names)} rows")
