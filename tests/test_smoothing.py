"""Spatial smoothing read-out (usplat/smoothing.py, scripts/smooth_select.py,
scripts/smooth_tab1.sh, configs/smooth_prereg.json) on CPU toy fields:
 * the grid and batches of the code equal the prereg's candidates and batches;
 * kNN: the kd-tree and the chunked torch backend equal a float64 brute force (self dropped,
   also with duplicated centres);
 * pooling equals a brute-force reference (sums and every read-out), gated or not, for ours
   and for averaging;
 * s = 0 (and a kernel whose weights underflow) reproduces the unsmoothed rows bit for bit;
 * the gate blocks pooling across disagreeing neighbours, and lets unseen Gaussians borrow;
 * refusals (centred channel, missing map, wrong N, no averaging sums);
 * the row factory works in scripts/eval_seg.py's mock pipeline (s = 0 rows score exactly as
   the unsmoothed rows, the default grid runs, the aux identity is recorded);
 * the driver end to end in a scratch git repository (mock backend): its gates, the
   selection, the once-only test stage, the table and the claim.
"""
import functools
import importlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pytest
import torch

from usplat import smoothing as S
from usplat.baselines import NormLift, baselines_state, default_baselines
from usplat.rows import field_rows, freeze
from usplat.vmf_field import DirectionalSemanticField, FieldConfig

REPO = pathlib.Path(__file__).resolve().parents[1]
PREREG = REPO / "configs" / "smooth_prereg.json"


# ------------------------------------------------------------------------------ toys
def _toy(N=160, d=8, V=6, seed=0, unseen=0.15, centres=None):
    """A field (+ NormLift with the averaging sums) from V random views: each view sees a random
    subset of the Gaussians with random pixel masses and noisy features around one of 3 class
    directions picked by position; a share `unseen` is never observed."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(N, 3, generator=g) * torch.tensor([2.0, 1.5, 0.2]) if centres is None \
        else centres
    cls = (x[:, 0] > 1.0).long() + (x[:, 1] > 0.9).long()
    base = torch.nn.functional.normalize(torch.randn(3, d, generator=g), dim=-1)
    f = DirectionalSemanticField(N, d, FieldConfig(change_test=False))
    nl = NormLift.for_field(f, with_averaging=True)
    seen = torch.rand(N, generator=g) > unseen
    for _ in range(V):
        ids = torch.nonzero(seen & (torch.rand(N, generator=g) > 0.3)).squeeze(1)
        reps = torch.randint(1, 4, (ids.numel(),), generator=g)
        gid = ids.repeat_interleave(reps)
        w = torch.rand(gid.numel(), generator=g) * 3.0
        z = base[cls[gid]] + 0.6 * torch.randn(gid.numel(), d, generator=g)
        f.update_sparse(gid, w, z, num_pixels=1000)
    return f, nl, x


def _brute(points, k):
    P = np.asarray(points, np.float64)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    N = len(P)
    I = np.empty((N, k), np.int64)
    for n in range(N):
        cand = [m for m in range(N) if m != n]
        cand.sort(key=lambda m: (D[n, m], m))
        I[n] = cand[:k]
    return I, np.take_along_axis(D, I, 1), D


def _ref_pool(first, second, points, spec, dirs=None, evidence=None):
    """Brute-force pooled sums in float64 (explicit loops)."""
    I, Dk, D = _brute(points, spec.k)
    h = float(np.median(np.sort(D + np.diag(np.full(len(D), np.inf)), 1)[:, 0]))
    sig = spec.s * h
    out = {}
    N = len(points)
    for sums, p in ((first, 1), (second, 2)):
        for name, x in sums.items():
            x = x.double().numpy()
            y = x.copy()
            for n in range(N):
                for j, m in enumerate(I[n]):
                    w = np.exp(-Dk[n, j] ** 2 / (2 * sig * sig)) if sig > 0 else 0.0
                    if spec.g is not None and evidence[n] and evidence[m]:
                        if float(np.dot(dirs[n], dirs[m])) < spec.g:
                            w = 0.0
                    y[n] = y[n] + (w ** p) * x[m]
            out[name] = y
    return out, h


# ------------------------------------------------------------------------------ tests
def test_grid_matches_prereg():
    p = json.loads(PREREG.read_text())
    assert p["candidates"] == [sp.tag() for sp in S.GRID]
    assert p["batches"] == {b: [sp.tag() for sp in v] for b, v in S.batches().items()}
    assert p["grid"]["s"] == list(S.GRID_S) and p["grid"]["k"] == list(S.GRID_K)
    assert p["grid"]["g"] == list(S.GRID_G) and len(S.GRID) == p["grid"]["n_candidates"] == 24
    assert p["choices"]["@smooth"]["rows"] == ["ours_P2"] + [S.row_names(sp)[1] for sp in S.GRID]
    assert p["choices"]["@avg_own"]["rows"] == ["averaging"] + [S.row_names(sp)[2]
                                                                for sp in S.GRID]
    sel = _load("smooth_select_t", REPO / "scripts" / "smooth_select.py")
    for sp in S.GRID:
        assert sel.names(sp.tag()) == S.row_names(sp)
    assert p["pipeline"].startswith("scripts/smooth_tab1.sh")
    for d in p["datasets"].values():                    # the copied Table I settings
        assert d["eval"]["tau_cov"] == 0.5 and d["eval"]["downscale"] == 2
    print(f"PASS grid: {len(S.GRID)} candidates, batches {list(p['batches'])}")


def test_knn_backends_equal_bruteforce():
    g = torch.Generator().manual_seed(1)
    P = torch.rand(300, 3, generator=g)
    I, Dk, _ = _brute(P.numpy(), 8)
    for be in S.KNN_BACKENDS:
        idx, dist = S.knn(P, 8, backend=be, budget_mib=1)       # tiny budget: many chunks
        assert idx.dtype == torch.int32 and dist.dtype == torch.float32
        assert np.array_equal(idx.numpy(), I), be
        assert np.allclose(dist.numpy(), Dk, rtol=1e-6, atol=1e-7), be
    Q = torch.cat([P[:50], P[:3]])                               # duplicated centres
    for be in S.KNN_BACKENDS:
        idx, dist = S.knn(Q, 4, backend=be)
        assert not (idx.numpy() == np.arange(53)[:, None]).any()   # never itself
        for a, b in ((0, 50), (1, 51), (2, 52)):                 # its twin at distance 0
            assert int(idx[a, 0]) == b and int(idx[b, 0]) == a and float(dist[a, 0]) == 0.0
    with pytest.raises(ValueError, match="more than"):
        S.knn(P[:5], 8)
    print("PASS kNN: kdtree and torch = brute force; duplicates handled")


@pytest.mark.parametrize("g", [None, 0.3])
def test_pool_equals_bruteforce(g):
    f, nl, x = _toy()
    spec = S.SmoothSpec(8, 1.5, g)
    idx, dist = S.knn(x, 8)
    h = S.median_nn_distance(dist)
    sf = S.smoothed_field(f, idx, dist, spec, h)
    ch = f.raw
    mu = f.mu_post().double().numpy()
    ref, h_ref = _ref_pool(dict(R=ch.R, W1=ch.W1, Wr=ch.Wr), dict(W2=ch.W2, Q=ch.Q), x.numpy(),
                           spec, mu, (ch.W1 > 0).numpy())
    assert h == pytest.approx(h_ref, rel=1e-6)
    for n, y in ref.items():
        assert np.allclose(getattr(sf.raw, n).double().numpy(), y, rtol=1e-5, atol=1e-6), n
    assert torch.equal(f.raw.R, ch.R) and f.raw is ch               # the field is not modified
    # every read-out = the field's own code on the reference sums
    st = f.state_dict()
    for n, y in ref.items():
        st[f"raw_{n}"] = torch.from_numpy(y).float()
    fr = DirectionalSemanticField.from_state(st)
    assert torch.allclose(sf.mu_post(), fr.mu_post(), atol=1e-5)
    assert torch.allclose(sf.epistemic(), fr.epistemic(), atol=1e-6)
    a1, a2 = sf.aleatoric()[0], fr.aleatoric()[0]
    assert torch.equal(torch.isnan(a1), torch.isnan(a2))
    assert torch.allclose(a1.nan_to_num(), a2.nan_to_num(), atol=1e-5)
    assert (sf.raw.W1 >= f.raw.W1).all() and (sf.raw.W1 > f.raw.W1).any()
    # averaging: same kernel on F, S1 (its own gate directions)
    sa = S.smoothed_averaging(nl, idx, dist, spec, h)
    refa, _ = _ref_pool(dict(F=nl.F, S1=nl.S1), {}, x.numpy(), spec,
                        nl.feature().double().numpy(), (nl.S1 > 0).numpy())
    assert np.allclose(sa.F.double().numpy(), refa["F"], rtol=1e-5, atol=1e-5)
    assert np.allclose(sa.S1.double().numpy(), refa["S1"], rtol=1e-5, atol=1e-5)
    Fr = torch.from_numpy(refa["F"])
    fa = torch.nn.functional.normalize(Fr / (torch.from_numpy(refa["S1"]) + 1e-12)[:, None], dim=-1)
    assert torch.allclose(sa.feature().double(), fa, atol=1e-5)
    print(f"PASS pooling (g={g}) = brute force: sums, mu*, u_epi, u_ale, averaging")


def test_sigma_zero_reproduces_unsmoothed():
    f, nl, x = _toy(seed=2)
    ctx = S.SmoothingContext(f, x, nl, k_max=8)
    for spec in (S.SmoothSpec(8, 0.0), S.SmoothSpec(8, 0.0, 0.8), S.SmoothSpec(4, 1e-4)):
        rows = [freeze(r) for r in S.smoothed_rows(ctx, [spec])]
        ref = [freeze(r) for r in field_rows(f, modes=("P0", "P2"))]
        for got, want in zip(rows[:2], ref):
            assert got.adaptive == want.adaptive
            assert torch.equal(got.buf, want.buf), spec
            assert got.unc.keys() == want.unc.keys()
            for c in want.unc:
                a, b = got.unc[c], want.unc[c]
                assert torch.equal(torch.isnan(a), torch.isnan(b))
                assert torch.equal(a[~torch.isnan(a)], b[~torch.isnan(b)]), (spec, c)
        assert torch.equal(rows[2].buf, nl.feature()), spec
    print("PASS s -> 0: ours_P0s/P2s/averaging_s equal the unsmoothed rows bit for bit")


def test_gate_blocks_disagreeing_neighbours():
    N, d = 40, 4
    x = torch.stack([torch.arange(N).float(), torch.zeros(N), torch.zeros(N)], 1)
    e0, e1 = torch.eye(d)[0], torch.eye(d)[1]
    f = DirectionalSemanticField(N, d, FieldConfig(change_test=False))
    left = torch.arange(N) < N // 2
    seen = torch.ones(N, dtype=torch.bool)
    seen[N // 2 + 1] = False                           # one unseen Gaussian on the right
    ids = torch.nonzero(seen).squeeze(1)
    for _ in range(3):
        z = torch.where(left[ids, None], e0.expand(len(ids), d), e1.expand(len(ids), d))
        f.update_sparse(ids, torch.full((len(ids),), 50.0), z, num_pixels=1000)
    idx, dist = S.knn(x, 4)
    h = S.median_nn_distance(dist)
    assert h == pytest.approx(1.0)
    b = N // 2 - 1                                     # the last Gaussian of the left object
    gated = S.smoothed_field(f, idx, dist, S.SmoothSpec(4, 2.0, 0.8), h)
    plain = S.smoothed_field(f, idx, dist, S.SmoothSpec(4, 2.0), h)
    assert float(gated.raw.R[b] @ e1) == 0.0           # nothing from the other object
    assert float(plain.raw.R[b] @ e1) > 0.1            # the ungated kernel mixes them
    w = lambda r: float(np.exp(-r * r / 8.0))           # sigma = 2h
    want = f.raw.W1[b] + f.raw.W1[b - 1] * w(1) + f.raw.W1[b - 2] * w(2)
    assert float(gated.raw.W1[b]) == pytest.approx(float(want), rel=1e-6)
    u = N // 2 + 1                                     # unseen: borrows from both sides
    assert float(f.raw.W1[u]) == 0.0
    assert float(gated.raw.R[u] @ e1) > 0 and float(gated.raw.R[u] @ e0) > 0
    st = {}
    S.pool(dict(W1=f.raw.W1), {}, idx, dist, S.SmoothSpec(4, 2.0, 0.8), h,
           f.mu_post(), f.raw.W1 > 0, stats=st)
    assert 0 < st["gate_blocked"] < 0.2 and st["kernel_mass"] > 0
    print(f"PASS gate: no pooling across the boundary (blocked {100 * st['gate_blocked']:.1f}% "
          f"of pairs), unseen Gaussians borrow")


def test_refusals(tmp_path):
    f, nl, x = _toy(N=60)
    with pytest.raises(ValueError, match="centre"):
        fc = DirectionalSemanticField(60, 8, FieldConfig(change_test=False, centre=True))
        S.smoothed_field(fc, *S.knn(x, 4), S.SmoothSpec(4, 1.0), 0.1)
    with pytest.raises(ValueError, match="do not match"):
        S.SmoothingContext(f, x[:10], nl)
    with pytest.raises(ValueError, match="k must"):
        S.SmoothSpec(0, 1.0)
    with pytest.raises(ValueError, match="dirs"):
        S.pool(dict(W1=f.raw.W1), {}, *S.knn(x, 4), S.SmoothSpec(4, 1.0, 0.5), 0.1)
    ck = dict(field=f.state_dict(), baselines=baselines_state([nl]))
    with pytest.raises(KeyError, match="names no 3DGS map"):
        S.row_factory(ck, f)
    splats = dict(means=x, quats=torch.tensor([1.0, 0, 0, 0]).expand(60, 4).clone(),
                  scales=torch.full((60, 3), 0.01), opacities=torch.full((60,), 0.9))
    torch.save(splats, tmp_path / "gs.pt")
    ck["config"] = dict(ply=str(tmp_path / "gs.pt"))
    assert len(S.row_factory(dict(ck), f)) == 3 * len(S.GRID)
    with pytest.raises(ValueError, match="changed since"):
        S.row_factory(dict(ck, inputs=dict(ply=dict(path=str(tmp_path / "gs.pt"),
                                                    sha256="0" * 64))), f)
    torch.save(dict(splats, means=torch.cat([x, x[:1]])), tmp_path / "gs2.pt")
    with pytest.raises(ValueError, match="the field has 60"):
        S.row_factory(dict(ck, config=dict(ply=str(tmp_path / "gs2.pt"))), f)
    with pytest.raises(KeyError, match="no averaging"):
        S.row_factory(dict(ck, baselines=[]), f)
    print("PASS refusals")


# ------------------------------------------------------------------- eval_seg mock pipeline
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cli_toy(tmp_path):
    """tests/test_seg_eval.py's toy scene (two objects on a plane, 2 frames, mock backend) with
    noisy, partial views, the NormLift/averaging sums and the map path in the checkpoint."""
    t = _load("test_seg_eval_smooth", REPO / "tests" / "test_seg_eval.py")
    from usplat.feature_pca import StreamingPCA

    def labels():
        lab = np.zeros((480, 640), np.int64)
        lab[40:, :320] = 3
        lab[40:, 320:] = 7
        return lab
    t._fake_7scenes(tmp_path / "data", labels)
    D, d = 16, 8
    pca = StreamingPCA(D, d)
    pca.basis = torch.eye(D)[:, :d]
    tx = {"chair": torch.eye(D)[0], "table": torch.eye(D)[1], "sofa": torch.eye(D)[2]}
    xs = torch.arange(-1.15, 1.2, 0.04)
    ys = torch.arange(-0.9, 0.95, 0.04)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    means = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.full((gx.numel(),), 2.0)], -1)
    N = means.shape[0]
    splats = dict(means=means, quats=torch.tensor([1.0, 0, 0, 0]).expand(N, 4).clone(),
                  scales=torch.full((N, 3), 0.03), opacities=torch.full((N,), 0.9))
    torch.save(splats, tmp_path / "gs.pt")
    right = means[:, 0] > 0
    feat = torch.where(right[:, None], pca.project(tx["table"][None]),
                       pca.project(tx["chair"][None]))
    f = DirectionalSemanticField(N, d, FieldConfig(change_test=False))
    f.set_centre(torch.zeros(d))
    bs = default_baselines(f)                                  # NormLift (averaging), views
    g = torch.Generator().manual_seed(0)
    for _ in range(4):
        sub = torch.rand(N, generator=g) > 0.4                # partial views
        ids = torch.nonzero(sub).squeeze(1)
        z = feat[ids] + 0.9 * torch.randn(len(ids), d, generator=g)
        f.update_sparse(ids, torch.rand(len(ids), generator=g) * 0.05, z, num_pixels=100)
    torch.save(dict(field=f.state_dict(), pca=pca.state_dict(), baselines=baselines_state(bs),
                    config=dict(ply=str(tmp_path / "gs.pt"))), tmp_path / "field.pt")
    vocab = dict(templates=["a photo of a {}."], ignore=[0],
                 classes={3: {"name": "chair"}, 7: {"name": "table"}, 9: "sofa"})
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    key = lambda p: f"m|p|a photo of a {{}}.|{p}"
    torch.save({key(k): v for k, v in tx.items()}, tmp_path / "text.pt")
    scenes = [dict(name="toy", field=str(tmp_path / "field.pt"), ply=str(tmp_path / "gs.pt"),
                   data=str(tmp_path / "data"), dataset="7scenes", stride=1,
                   label_pattern="{root}/frame-{idx:06d}.label.png")]
    (tmp_path / "scenes.json").write_text(json.dumps(scenes))
    return ["--vocab", str(tmp_path / "vocab.json"), "--backend", "mock", "--device", "cpu",
            "--downscale", "16", "--tau-cov", "0.5", "--extractor", "clearclip",
            "--clip-model", "m", "--clip-pretrained", "p", "--text-cache", str(tmp_path / "text.pt"),
            "--plain-json"]


def test_row_factory_eval_seg_mock(tmp_path):
    pytest.importorskip("cv2")
    cli = _load("eval_seg_smooth", REPO / "scripts" / "eval_seg.py")
    base = _cli_toy(tmp_path)
    scenes = str(tmp_path / "scenes.json")
    grid = [S.SmoothSpec(4, 0.0), S.SmoothSpec(4, 2.0), S.SmoothSpec(8, 2.0, 0.8)]
    mod = f"smoothtoy_{uuid.uuid4().hex[:8]}"                 # a factory with a toy grid
    (tmp_path / f"{mod}.py").write_text(
        "import functools\nfrom usplat import smoothing as S\n"
        f"row_factory = functools.partial(S.row_factory, grid={grid!r})\n"
        "SmoothSpec = S.SmoothSpec\n".replace("SmoothSpec(", "S.SmoothSpec("))
    sys.path.insert(0, str(tmp_path))
    try:
        rows = ["ours_P0", "ours_P2", "averaging"] + [n for sp in grid for n in S.row_names(sp)]
        out = cli.main(base + ["--val-scenes", scenes, "--scenes", scenes, "--rows", ",".join(rows),
                               "--row-factory", f"usplat.baselines:row_factory,{mod}:row_factory",
                               "--out", str(tmp_path / "eval.json")])
    finally:
        sys.path.remove(str(tmp_path))
    assert set(out["betas"]) == set(rows)
    r = out["scenes"]["toy"]["rows"]
    for sp in grid:
        p0, p2, av = S.row_names(sp)
        assert set(out["betas"][p2]) == {"beta0", "a", "b", "g"}
        assert {"u_epi", "u_ale"} <= set(r[p2]["all"]["ranking"]), r[p2]["all"]["ranking"].keys()
        assert np.isfinite(r[p2]["all"]["nll"]) and np.isfinite(r[av]["all"]["nll"])
    p0, p2, av = S.row_names(grid[0])                          # s = 0: the unsmoothed rows
    assert out["betas"][p0] == out["betas"]["ours_P0"]
    assert out["betas"][p2] == out["betas"]["ours_P2"]
    assert out["betas"][av] == out["betas"]["averaging"]
    for a, b in ((p0, "ours_P0"), (p2, "ours_P2"), (av, "averaging")):
        for k in ("miou", "nll", "ece", "brier", "aurc"):
            assert r[a]["all"][k] == r[b]["all"][k], (a, k)
    p0s, p2s, _ = S.row_names(grid[1])                         # s = 2 changes the field rows
    assert r[p2s]["all"]["nll"] != r["ours_P2"]["all"]["nll"]
    ids = out["beta_fit_settings"]["aux_identities"]
    assert any(i.startswith("smoothing:") for i in ids)
    # the registered factory: one batch of the real grid on the same toy
    sel = _load("smooth_select_t2", REPO / "scripts" / "smooth_select.py")
    p = json.loads(PREREG.read_text())
    b = next(iter(p["batches"]))
    out2 = cli.main(base + ["--val-scenes", scenes, "--scenes", scenes, "--rows",
                            ",".join(sel.batch_rows(p, b)), "--row-factory",
                            "usplat.baselines:row_factory,usplat.smoothing:row_factory",
                            "--out", str(tmp_path / "eval2.json")])
    assert set(out2["betas"]) == set(sel.batch_rows(p, b))
    assert out2["betas"]["ours_P2"] == out["betas"]["ours_P2"]   # batches are one evaluation
    print(f"PASS eval_seg mock: {len(rows)} rows (s = 0 equals the unsmoothed rows); batch {b} "
          f"of the registered grid: {len(out2['betas'])} rows")


# ---------------------------------------------------------------------------- driver
def _git(root, *a):
    subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True,
                   env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                            GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"))


def test_smooth_driver(tmp_path):
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("needs git and bash")
    pytest.importorskip("cv2")
    toy = tmp_path / "toy"
    toy.mkdir()
    _cli_toy(toy)
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "configs").mkdir()
    for f in ("smooth_tab1.sh", "smooth_select.py", "make_tables.py", "quant_stats.py",
              "eval_seg.py"):
        shutil.copy2(REPO / "scripts" / f, repo / "scripts" / f)
    (repo / "scripts" / "gpu_gate.sh").write_text("#!/usr/bin/env bash\nshift\nexec \"$@\"\n")
    (repo / "scripts" / "gpu_gate.sh").chmod(0o755)
    shutil.copytree(REPO / "usplat", repo / "usplat", ignore=shutil.ignore_patterns("__pycache__"))
    sc = json.loads((toy / "scenes.json").read_text())
    (repo / "configs" / "toy_val.json").write_text(json.dumps(sc))
    (repo / "configs" / "toy_test.json").write_text(json.dumps([dict(sc[0], name="toy_test")]))
    out = tmp_path / "out"
    p = json.loads(PREREG.read_text())
    p["datasets"] = {"toy": dict(scenes=dict(validation="configs/toy_val.json",
                                             test="configs/toy_test.json"),
                                 eval=dict(vocab=str(toy / "vocab.json"), dataset="7scenes",
                                           stride=1, downscale=16, tau_cov=0.5,
                                           extractor="clearclip"),
                                 text_cache_source=str(toy / "text.pt"), out=str(out),
                                 test_look="toy test look")}
    pr = "configs/smooth_prereg_toy.json"
    (repo / pr).write_text(json.dumps(p, indent=1))
    git = lambda *a: _git(repo, *a)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "init")
    env = dict(os.environ, PREREG=pr, DATASET="toy", BACKEND="mock", DEVICE="cpu",
               EVAL_EXTRA="--clip-model m --clip-pretrained p")
    env.pop("OUT", None)

    def run(stage, **kw):
        return subprocess.run(["bash", str(repo / "scripts" / "smooth_tab1.sh")],
                              env=dict(env, STAGE=stage, **kw), capture_output=True, text=True,
                              timeout=1200)

    def refused(r, msg):
        return r.returncode == 1 and msg in r.stdout + r.stderr

    r = run("test")
    assert refused(r, "is not a finished validation record"), r.stdout + r.stderr
    r = run("val", SCENES="configs/toy_test.json")
    assert refused(r, "validation scenes are ['toy']"), r.stdout + r.stderr
    r = run("val", OUT="/home/thanhnc/gauss_ws/data/uags_results/tab1")
    assert refused(r, "never touched"), r.stdout + r.stderr
    r = run("val", EVAL_EXTRA="--clip-model m --rows ours_P2")
    assert refused(r, "EVAL_EXTRA option '--rows'"), r.stdout + r.stderr
    (repo / pr).write_text((repo / pr).read_text().replace('"consistency_tol": 1e-06',
                                                           '"consistency_tol": 1e-05'))
    r = run("val")
    assert refused(r, "is not committed"), r.stdout + r.stderr
    git("checkout", "--", pr)
    pp = repo / "usplat" / "smoothing.py"
    pp.write_text(pp.read_text() + "\n# local edit\n")
    r = run("val")
    assert refused(r, "uncommitted changes to tracked files"), r.stdout + r.stderr
    git("checkout", "--", "usplat/smoothing.py")
    r = run("val", DRY_RUN="1")
    assert r.returncode == 0 and r.stdout.count("[dry-run]") == len(p["batches"]) == 4, r.stdout + r.stderr
    r = run("val")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    sel = json.loads((out / "selection_val.json").read_text())
    assert sel["status"] == "ok"
    ch = sel["metrics"]["selection"]["chosen"]
    assert set(ch) == {"@smooth", "@avg_own"}
    assert sel["metrics"]["selection"]["table1_check"]["status"] == "none named in the prereg"
    for b in p["batches"]:
        v = json.loads((out / f"eval_val_{b}.json").read_text())
        assert v["status"] == "ok" and v["git"]["dirty_tracked"] is False
    assert (out / "text_cache_clearclip.pt").exists()
    # test: refuses code committed after validation, then runs once
    rp = repo / "usplat" / "smoothing.py"
    rp.write_text(rp.read_text() + "\n# a later commit\n")
    git("commit", "-qam", "later")
    r = run("test")
    assert refused(r, "differ from the validation commit"), r.stdout + r.stderr
    git("revert", "--no-edit", "HEAD")
    r = run("test")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    look = json.loads((out / "look_test.json").read_text())
    assert len(look["attempts"]) == 1 and look["chosen"] == ch
    tab = json.loads((out / "table_smooth.json").read_text())
    assert len(tab["rows"]) == 3 + 3 * len(S.GRID) and tab["scenes"] == ["toy_test"]
    assert (out / "claim_smooth.json").exists() and (out / "table_smooth.tex").exists()
    r = run("test")
    assert refused(r, "every test batch"), r.stdout + r.stderr
    r = run("val")
    assert refused(r, "the test stage has started"), r.stdout + r.stderr
    r = run("table")
    assert r.returncode == 0, r.stdout + r.stderr
    print(f"PASS driver: val -> selection {ch} -> test once -> table + claim; gates refuse")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
