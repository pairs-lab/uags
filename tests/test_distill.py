"""usplat.distill: the Feature 3DGS and LEGaussians re-implementations (CPU, linear toy
rasterizers) and their row factory."""
import os
import torch
import torch.nn.functional as Fn
from usplat import distill as D


def _toy(P=200, N=40, d=8, V=6, seed=0, conflict=None):
    """V views, each a fixed (P, N) splat-weight matrix W_v (rows sum to 0.9); targets are
    renders of true features, except that `conflict` Gaussians get a new random feature in
    every view (multi-view inconsistency)."""
    g = torch.Generator().manual_seed(seed)
    true = Fn.normalize(torch.randn(N, d, generator=g), dim=-1)
    other = Fn.normalize(torch.randn(N, d, generator=g), dim=-1)
    views = []
    for v in range(V):
        W = torch.rand(P, N, generator=g) ** 8
        W = 0.9 * W / W.sum(1, keepdim=True)
        feat = true.clone()
        if conflict is not None:
            feat[conflict] = Fn.normalize(torch.randn(int(conflict.sum()), d, generator=g), dim=-1)
        z = Fn.normalize(W @ feat, dim=-1)
        views.append(((lambda buf, W=W: W @ buf), (lambda z=z: z)))
    return views, true, N, d


def test_f3dgs_fits_consistent_views():
    views, true, N, d = _toy()
    res = D.train_features(views, N, d, method="f3dgs", iters=1500, lr=2e-2, log_every=100,
                           log=lambda s: None)
    h = res["history"]
    assert h[-1][1] < 0.5 * h[0][1], h                       # the L1 loss drops
    cos = Fn.cosine_similarity(res["features"], true, dim=-1)
    assert cos.median() > 0.9, cos
    assert res["u"] is None
    print(f"PASS Feature 3DGS toy: loss {h[0][1]:.3f} -> {h[-1][1]:.3f}, median cos {cos.median():.3f}")


def test_legs_uncertainty_marks_conflicting_gaussians():
    conflict = torch.arange(40) < 10                          # 10 Gaussians change between views
    views, true, N, d = _toy(conflict=conflict, V=30)         # too many views to memorize
    res = D.train_features(views, N, d, method="legs", iters=3000, lr=1e-2, lam_u=1.0,
                           n_codes=16, code_samples=200, log=lambda s: None)
    u = res["u"]
    assert u.shape == (N,) and bool(((u >= 0) & (u <= 1)).all())
    assert res["features"].shape == (N, D.LEGS_DIM)
    assert u[conflict].mean() > u[~conflict].mean() + 0.05, (u[conflict].mean(), u[~conflict].mean())
    dec = res["decoder"]
    out = dec(torch.randn(5, D.LEGS_DIM))                      # decoded query feature
    assert out.shape == (5, d) and bool((out.norm(dim=-1) <= 1 + 1e-5).all())
    print(f"PASS LEGaussians toy: u conflicting {u[conflict].mean():.3f} vs consistent "
          f"{u[~conflict].mean():.3f}")


def test_row_factory_reads_files_and_checks_basis(tmp_path):
    import pytest
    main = str(tmp_path / "scene_lseg.pt")
    basis = torch.linalg.qr(torch.randn(16, 8))[0]
    pca = dict(basis=basis)
    dec = D.LEGDecoder(Fn.normalize(torch.randn(4, 8), dim=-1))
    files = (("f3dgs", torch.randn(5, 8), None, None),
             ("legs_l2", torch.randn(5, D.LEGS_DIM), torch.rand(5), dec.state_dict()))
    for name, f, u, dstate in files:
        torch.save(dict(name=name, features=f, u=u, decoder=dstate, identity=f"id-{name}",
                        pca_basis_sha256=D.basis_sha256(pca)), D.out_path(main, name))
    ck = dict(config=dict(out=main), pca=pca)
    rows = D.row_factory(ck, None, "cpu")
    assert [r.name for r in rows] == ["f3dgs", "legs_l2"], [r.name for r in rows]
    by = {r.name: r for r in rows}
    assert by["f3dgs"].uncertainties() == {} and by["f3dgs"].decoder is None
    assert set(by["legs_l2"].uncertainties()) == {"legs_u"}
    r = torch.randn(3, D.LEGS_DIM)                              # the decoder round-trips
    assert torch.allclose(by["legs_l2"].decoder(r), dec(r))
    assert sorted(ck["_aux_identities"]) == ["id-f3dgs", "id-legs_l2"]
    open(D.out_path(main, "legs_l2") + ".ckpt", "w").close()     # a training state is no row
    assert [r.name for r in D.row_factory(dict(config=dict(out=main), pca=pca), None, "cpu")] \
        == ["f3dgs", "legs_l2"]
    assert D.variant_name("legs", 1.0) == "legs_l1" and D.variant_name("f3dgs") == "f3dgs"
    with pytest.raises(ValueError, match="PCA basis"):
        D.row_factory(dict(config=dict(out=main), pca=dict(basis=basis.flip(1))), None, "cpu")
    assert D.row_factory(dict(config=dict(out=str(tmp_path / "other.pt")), pca=pca), None, "cpu") == []
    # a field built before the 2026-10-31 move stores its old path: the files are found where
    # usplat.paths.moved puts that path
    import usplat.paths as P
    old = P.OLD_ROOT + "/uags_fields/scene_lseg.pt"
    saved = P._RENAMED["uags_fields"]
    P._RENAMED["uags_fields"] = str(tmp_path)
    try:
        assert [r.name for r in D.row_factory(dict(config=dict(out=old), pca=pca), None, "cpu")] \
            == ["f3dgs", "legs_l2"]
    finally:
        P._RENAMED["uags_fields"] = saved
    print("PASS distill row factory: rows, legs_u channel, basis check, aux identities, moved paths")


def test_stopping_rule():
    """Training stops once the loss improves by less than tol over the last window."""
    assert not D.converged([(i, 1.0) for i in range(9)])                 # too short
    flat = [(i, 1.0 - 0.1 * min(i, 5) / 5) for i in range(20)]
    assert D.converged(flat)                                              # plateau
    falling = [(i, 1.0 / (1 + i)) for i in range(10)]
    assert not D.converged(falling)
    views, true, N, d = _toy()
    res = D.train_features(views, N, d, method="f3dgs", iters=20000, lr=2e-2, log_every=100,
                           min_iters=500, log=lambda s: None)
    assert res["iters_run"] < 20000, res["iters_run"]
    print(f"PASS stopping rule: toy converged at {res['iters_run']} iterations")


def test_resume_matches_uninterrupted(tmp_path):
    """A run stopped after 400 iterations and resumed from its checkpoint gives the same result
    as an uninterrupted 600-iteration run (same views drawn, same Adam state)."""
    for method, kw in (("f3dgs", {}), ("legs", dict(n_codes=8, code_samples=100))):
        views, true, N, d = _toy(V=10)
        full = D.train_features(views, N, d, method=method, iters=600, lr=1e-2, log_every=100,
                                min_iters=10 ** 6, log=lambda s: None, **kw)
        ck = str(tmp_path / f"{method}.ckpt")
        D.train_features(views, N, d, method=method, iters=400, lr=1e-2, log_every=100,
                         min_iters=10 ** 6, ckpt_path=ck, ckpt_every=200, log=lambda s: None, **kw)
        res = D.train_features(views, N, d, method=method, iters=600, lr=1e-2, log_every=100,
                               min_iters=10 ** 6, ckpt_path=ck, ckpt_every=200,
                               log=lambda s: None, **kw)
        assert torch.allclose(res["features"], full["features"], atol=1e-5), method
        if method == "legs":
            assert torch.allclose(res["u"], full["u"], atol=1e-5)
        assert [h[0] for h in res["history"]] == [h[0] for h in full["history"]]
    print("PASS resume: identical to an uninterrupted run (f3dgs, legs)")
