"""Coverage of the change test (scripts/audit_change_coverage.py) and the opt-in change-stream
weight (FieldConfig.change_weight).

Audit: a mock scene whose Gaussians range from sub-pixel to large footprints is mapped with
the mock rasterizer; the field is saved as run_mapping.py saves it and audited on the CPU.
The shares, deciles, null buckets and thresholds must equal direct computations on the field,
the audited field must not change, and the pre-registered decision must follow its rule: a
small full-view mass a0 makes every observed Gaussian testable (PASS), a large one leaves the
small Gaussians untested (FAIL).
"""
import importlib.util
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as Fn

from usplat.vmf_field import DirectionalSemanticField, FieldConfig
from usplat.mock_rasterizer import MockFeatureRasterizer
from usplat.clip_dense import MockDenseCLIP

REPO = Path(__file__).resolve().parents[1]


def _load_audit():
    spec = importlib.util.spec_from_file_location("audit_change_coverage",
                                                  REPO / "scripts" / "audit_change_coverage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scene(N=160, H=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    means2d = torch.rand(N, 2, generator=g) * H
    sigmas = 0.3 + 2.7 * torch.rand(N, generator=g) ** 2      # many tiny footprints
    opac = 0.3 + 0.6 * torch.rand(N, generator=g)
    depths = torch.rand(N, generator=g)
    grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(H), indexing="xy"), -1)
    ras = MockFeatureRasterizer(means2d, sigmas, opac, depths, grid.reshape(-1, 2).float())
    proto = (means2d[:, 0] / H * 5).long().clamp(0, 4)
    return ras, proto, H


def _map_field(a0, views=12, seed=0, **cfg):
    torch.manual_seed(seed)
    ras, proto, H = _scene(seed=seed)
    clip = MockDenseCLIP(feat_dim=16, num_protos=5, noise=0.2, seed=seed)
    f = DirectionalSemanticField(ras.N, 16, FieldConfig(view_mass_frac=a0, **cfg))
    dom = ras.weights().argmax(1)
    for _ in range(views):
        f.update_view(ras, clip.extract_from_labels(proto[dom].reshape(H, H)).reshape(-1, 16))
    return f, ras


def _save(f, path):
    torch.save(dict(field=f.state_dict(), config={}, field_config=f.cfg.to_dict()), path)


def test_audit_matches_field_and_decides(tmp_path):
    A = _load_audit()
    out = {}
    for name, a0 in (("small_a0", 1e-4), ("large_a0", 5e-3)):
        f, _ = _map_field(a0)
        p = tmp_path / f"{name}.pt"
        _save(f, p)
        rep_path = tmp_path / f"{name}.json"
        res = A.main(["--field", str(p), "--out", str(rep_path)])
        rec = json.loads(rep_path.read_text())
        assert rec["status"] == "ok"
        r = rec["metrics"]["fields"][str(p)]
        assert r == json.loads(json.dumps(res["fields"][str(p)]))   # record = returned report
        pm = f.pixel_mass.double()
        obs = pm > 0
        dfd = torch.isfinite(f.c_last) & obs
        assert r["observed"] == int(obs.sum()) and r["defined"] == int(dfd.sum())
        assert math.isclose(r["defined_share_gaussians"], float(dfd[obs].double().mean()))
        assert math.isclose(r["defined_share_pixel_mass"],
                            float(pm[dfd].sum() / pm[obs].sum()), rel_tol=1e-9)
        # deciles: equal-count groups by pixel mass, smallest first
        assert sum(d["n"] for d in r["deciles"]) == int(obs.sum())
        po, do = pm[obs], dfd[obs]
        order = torch.argsort(po, stable=True)
        low = order[: sum(d["n"] for d in r["deciles"][:5])]
        assert math.isclose(r["lower_defined_share"], float(do[low].double().mean()))
        assert all(a["pixel_mass_max"] <= b["pixel_mass_min"]
                   for a, b in zip(r["deciles"], r["deciles"][1:]))
        # null buckets and the borrowing rule of calibrate_change_threshold
        counts = f.c_hist.sum(1)
        assert [n["samples"] for n in r["null"]] == [int(c) for c in counts]
        assert [n["own"] for n in r["null"]] == [bool(c >= f.MIN_NULL) for c in counts]
        # thresholds = calibrate on a fresh copy; the audited field is untouched
        for q in ("0.99", "0.999"):
            g = DirectionalSemanticField.from_state(f.state_dict())
            if counts.sum() == 0:
                assert "error" in r["thresholds"][q]
                continue
            t = g.calibrate_change_threshold(float(q))
            assert math.isclose(r["thresholds"][q]["change_thr"], t)
            assert r["thresholds"][q]["by_hist"] == [float(x) for x in g.thr_by_hist]
        dcs = r["decision"]
        assert dcs["passed"] == (r["defined_share_pixel_mass"] >= 0.5
                                 and r["lower_defined_share"] >= 0.5)
        assert rec["metrics"]["passed"][str(p)] == dcs["passed"]
        out[name] = r
    assert out["small_a0"]["decision"]["passed"]
    assert not out["large_a0"]["decision"]["passed"]
    assert out["large_a0"]["deciles"][0]["defined_share"] < 0.5
    print(f"PASS audit: a0=1e-4 defined {out['small_a0']['defined_share_gaussians']:.2f} "
          f"(PASS); a0=5e-3 lower deciles {out['large_a0']['lower_defined_share']:.2f} (FAIL)")


def test_audit_does_not_change_the_field(tmp_path):
    A = _load_audit()
    f, _ = _map_field(1e-4)
    c_last, c_max = f.c_last.clone(), f.c_max.clone()
    thr0 = f.cfg.change_thr
    r = A.audit(f)
    assert torch.equal(f.c_last.nan_to_num(-1), c_last.nan_to_num(-1))
    assert torch.equal(f.c_max, c_max) and f.cfg.change_thr == thr0 and f.thr_by_hist is None
    assert not r["calibrated_at_save"]
    f.calibrate_change_threshold()
    assert A.audit(f)["calibrated_at_save"]
    print("PASS audit leaves the field unchanged and flags a calibrated field")


def test_audit_stream_support_is_necessary():
    """On a record-only field every Gaussian with a defined score has change-stream support
    min(n, n^2/T2) >= tau_min (n = ns + nf), in every change_weight mode, and the audit reports
    that share; the posterior W1 is no such condition under "visibility" (a partly drawn
    Gaussian counts as a full view in the stream, as a fraction of one in W1)."""
    A = _load_audit()
    for mode in ("view", "visibility", "relmax"):
        kw = {} if mode == "view" else dict(fast_min=0.5, slow_min=0.5)
        f, _ = _map_field(2e-3, change_weight=mode, **kw)
        r = A.audit(f)
        obs = f.pixel_mass > 0
        n = (f.ns + f.nf).double()
        sup = torch.minimum(n, n * n / f.T2.double().clamp_min(1e-12))
        dfd = torch.isfinite(f.c_last) & obs
        assert bool(dfd.any()) and bool((sup[dfd] >= f.cfg.tau_min).all()), mode
        assert math.isclose(r["stream_support_ge_tau_share"],
                            float((sup[obs] >= f.cfg.tau_min).double().mean())), mode
        if mode == "visibility":
            assert bool((dfd & (f.raw.W1 < f.cfg.tau_min)).any())
    print("PASS audit: stream support >= tau_min for every defined score (all modes)")


# ---------------------------------------------------------------- change-stream weight
def test_posterior_bit_identical_across_change_weight():
    """change_weight touches only the change-test stream and views_since_reset: in the
    record-only mapping pass the posterior statistics are bit-identical for "view",
    "visibility" and "relmax" while the change streams differ; after a calibrated update with
    a content change, the Gaussians that no setting reset keep bit-identical statistics.
    Every setting uses fast_min = slow_min = 0.5 (the opt-in modes refuse fast_min = 1 with
    fast_memory = 0), so the configs differ in change_weight only."""
    ras, proto, H = _scene(seed=1)
    clip = MockDenseCLIP(feat_dim=16, num_protos=5, noise=0.2, seed=1)
    dom = ras.weights().argmax(1)
    torch.manual_seed(1)
    moved = torch.where(torch.arange(ras.N) % 2 == 0, (proto + 1) % 5, proto)  # half change
    zs = [clip.extract_from_labels((proto if v < 8 else moved)[dom].reshape(H, H))
          .reshape(-1, 16) for v in range(14)]
    fields = {}
    for mode in ("view", "visibility", "relmax"):
        f = DirectionalSemanticField(ras.N, 16, FieldConfig(
            view_mass_frac=2e-3, change_weight=mode, fast_min=0.5, slow_min=0.5,
            readmit_depth=None))
        for v, z in enumerate(zs):
            if v == 8:
                snap = {n: getattr(f.raw, n).clone() for n in ("R", "W1", "W2", "Q", "Wr")}
                snap["ns"] = f.ns.clone()
                f.calibrate_change_threshold()
            f.update_view(ras, z)
        fields[mode] = (f, snap)
    ref, rsnap = fields["view"]
    never = torch.ones(ras.N, dtype=torch.bool)
    for f, _ in fields.values():
        never &= f.n_resets == 0
    assert 0 < int(never.sum()) < ras.N
    for mode, (f, snap) in fields.items():
        for name in ("R", "W1", "W2", "Q", "Wr"):
            assert torch.equal(snap[name], rsnap[name]), (mode, name)      # mapping pass
            assert torch.equal(getattr(f.raw, name)[never], getattr(ref.raw, name)[never]), \
                (mode, name)
        assert torch.equal(f.pixel_mass, ref.pixel_mass)
        if mode != "view":
            assert not torch.equal(snap["ns"], rsnap["ns"]), mode           # stream differs
    print(f"PASS change_weight: posterior bit-identical (mapping; {int(never.sum())} "
          "never-reset Gaussians after the update), change streams differ")


def test_unoccluded_mass_mock_and_gsplat_meta():
    """Mock: an unoccluded Gaussian inside the pixel grid has m_v = its unoccluded mass (the
    pixel sum approximates the integral), an occluded one less: w = m_v / m_unocc is its
    visible share. gsplat meta: o 2 pi / sqrt(det conic) equals the numerical integral of
    o exp(-1/2 d^T conic d) for anisotropic conics, packed and unpacked; culled Gaussians 0."""
    from usplat.gsplat_adapter import unoccluded_mass_from_meta
    grid = torch.stack(torch.meshgrid(torch.arange(40), torch.arange(40), indexing="xy"),
                       -1).reshape(-1, 2).float()
    ras = MockFeatureRasterizer(torch.tensor([[20.0, 20.0], [20.0, 21.0], [8.0, 30.0]]),
                                torch.tensor([2.0, 3.0, 1.5]), torch.tensor([0.5, 0.6, 0.4]),
                                torch.tensor([1.0, 0.0, 0.5]), grid)
    vis = ras.weights().sum(0) / ras.unoccluded_mass()
    assert abs(float(vis[2]) - 1.0) < 1e-3 and abs(float(vis[1]) - 1.0) < 1e-3
    assert 0.3 < float(vis[0]) < 0.75                       # behind Gaussian 1
    # gsplat meta (synthetic): anisotropic covariances, conic = their inverse
    g = torch.Generator().manual_seed(0)
    N = 5
    A = torch.randn(N, 2, 2, generator=g, dtype=torch.float64)
    cov = A @ A.transpose(1, 2) + 0.5 * torch.eye(2, dtype=torch.float64)
    inv = torch.linalg.inv(cov)
    conic = torch.stack([inv[:, 0, 0], inv[:, 0, 1], inv[:, 1, 1]], -1)
    op = 0.2 + 0.7 * torch.rand(N, generator=g, dtype=torch.float64)
    xs = torch.arange(-12.0, 12.0, 0.02, dtype=torch.float64)
    X, Y = torch.meshgrid(xs, xs, indexing="xy")
    ref = torch.stack([op[n] * torch.exp(-0.5 * (inv[n, 0, 0] * X * X + 2 * inv[n, 0, 1] * X * Y
                                                 + inv[n, 1, 1] * Y * Y)).sum() * 0.02 ** 2
                       for n in range(N)])
    ids = torch.tensor([0, 2, 3])
    packed = unoccluded_mass_from_meta(dict(gaussian_ids=ids, conics=conic[ids],
                                            opacities=op[ids]), N)
    assert torch.allclose(packed[ids], ref[ids], rtol=1e-4) and packed[1] == 0 == packed[4]
    radii = torch.zeros(1, N, 2, dtype=torch.int32)
    radii[0, ids] = 3
    unpacked = unoccluded_mass_from_meta(dict(gaussian_ids=None, conics=conic[None],
                                              opacities=None, radii=radii), N, opacities=op)
    assert torch.equal(unpacked, packed)
    print(f"PASS unoccluded mass: mock visible shares {[round(float(v), 3) for v in vis]}; "
          "gsplat meta formula = numerical integral (packed and unpacked)")


def test_visibility_weight_makes_small_gaussians_testable():
    """A Gaussian of 2 px (a quarter of a full view at a0 H W = 7.68 px) that is fully
    visible counts as one view per view with change_weight="visibility" (w = m / m_unocc), so
    its change score is defined after the persistence window; with view weights it never is
    in 8 views. A half-occluded one (w = 0.5) stays untested while fast_min = 0.75
    (fast_memory = 0: the recent bucket is the current view alone) and is tested with
    fast_min = 0.5. views_since_reset and the stream mass follow the visibility weights. The
    option needs the unoccluded mass (ValueError otherwise); relmax keeps m_max in its state.
    Both opt-in modes refuse fast_min >= 1 with fast_memory = 0 (relmax: w = 1 only at a view
    that sets a new running max, so the test would run only at those views)."""
    import pytest
    torch.manual_seed(0)
    N, d, P = 4, 8, 76800                                   # 320 x 240: a0 H W = 7.68 px
    m = torch.tensor([2.0, 2.0, 1.0, 7.68])
    m_un = torch.tensor([2.0, 4.0, 1.0, 7.68])              # Gaussian 1: half occluded
    mu = Fn.normalize(torch.randn(N, d), dim=-1)
    zs = [Fn.normalize(mu + 0.2 * torch.randn(N, d), dim=-1) for _ in range(8)]
    out = {}
    for mode, fast_min in (("view", 1.0), ("visibility", 0.75), ("visibility", 0.5)):
        f = DirectionalSemanticField(N, d, FieldConfig(change_weight=mode, fast_min=fast_min,
                                                       slow_min=fast_min, readmit_depth=None))
        for z in zs:
            f.update_sparse(torch.arange(N), m, z, num_pixels=P, m_unocc=m_un)
        out[mode, fast_min] = f
    v, s, s5 = out["view", 1.0], out["visibility", 0.75], out["visibility", 0.5]
    tested = torch.isfinite(s.c_last)
    assert tested.tolist() == [True, False, True, True]     # w = 0.5 < fast_min = 0.75
    assert torch.isfinite(s5.c_last).all()
    assert torch.isnan(v.c_last[:3]).all() and torch.isfinite(v.c_last[3])
    assert torch.allclose(s.views_since_reset, 8 * (m / m_un).clamp(max=1))
    assert torch.allclose(s.ns + s.nf, 8 * (m / m_un).clamp(max=1))
    assert torch.allclose(v.views_since_reset, 8 * (m / 7.68).clamp(max=1))
    assert torch.equal(v.raw.W1, s.raw.W1) and torch.equal(v.raw.R, s.raw.R)
    for mode in ("visibility", "relmax"):
        with pytest.raises(ValueError, match="fast_min < 1"):
            DirectionalSemanticField(N, d, FieldConfig(change_weight=mode))
        FieldConfig(change_weight=mode, fast_memory=0.5).validate()     # nf accumulates
    f = DirectionalSemanticField(N, d, FieldConfig(change_weight="visibility", fast_min=0.5))
    with pytest.raises(ValueError, match="unoccluded"):
        f.update_sparse(torch.arange(N), m, mu, num_pixels=P)
    W = torch.rand(P, N)
    with pytest.raises(ValueError, match="unoccluded"):
        f.update_view(lambda buf: W @ buf, Fn.normalize(torch.randn(P, d), dim=-1))
    r = DirectionalSemanticField(N, d, FieldConfig(change_weight="relmax", fast_min=0.5,
                                                   slow_min=0.5, readmit_depth=None))
    for scale in (0.5, 1.0, 0.25):
        r.update_sparse(torch.arange(N), m * scale, mu, num_pixels=P)
    assert torch.allclose(r.views_since_reset, torch.full((N,), 1.0 + 1.0 + 0.25))
    g = DirectionalSemanticField.from_state(r.state_dict())
    assert torch.equal(g.m_max, r.m_max) and g.cfg.change_weight == "relmax"
    print("PASS visibility weight: 2 px Gaussians tested (views: never); relmax state kept")
