"""Static false-alarm pipeline on the CPU (run_mapping.py --summary-json, the residual
baselines' label-free null, scripts/static_fa_report.py, scripts/replica_static_fa.sh).

 * ChangeTally (the field's per-view tests and fires) and ResidualNull (the baselines' null,
   thresholds, tests and fires) are exact against brute force on toy sequences.
 * On a static mock scene with per-view noise injected through --feature-transform, the
   realised per-view rate is finite and the summary record is complete; the transform is
   called on the update sequence only, with the documented arguments.
 * The validation selection (fixed threshold) and the test criterion follow their written
   rules on mock runs and refuse unfinished or mismatched runs. The criterion step refuses on
   its own a second test record (unless --force), a missing selection, a dirty tree and a
   config changed after validation; select refuses once validation is closed.
 * replica_static_fa.sh refuses a test run without a finished validation selection, with a
   dirty tree or over a finished test record, and closes validation once test records exist.
"""
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as Fn

from usplat.vmf_field import DirectionalSemanticField, FieldConfig
from usplat.mock_rasterizer import MockFeatureRasterizer
from usplat.change_baselines import ChangeBaselines, ResidualNull

REPO = Path(__file__).resolve().parents[1]


def _load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(name, rel):
    return _load_path(name, REPO / rel)


RM = _load("run_mapping_cli", "scripts/run_mapping.py")
TC = _load("test_change_helpers", "tests/test_change.py")
SR = _load("static_fa_report_cli", "scripts/static_fa_report.py")


# ---------------------------------------------------------------- brute force
def test_tally_matches_brute_force():
    """Field: per-view tests / fires and their per-bucket sums of the update pass equal the
    brute-force change test; the reset share and the score histogram are consistent."""
    torch.manual_seed(1)
    N, d, half, V = 250, 8, 14, 30
    mu = Fn.normalize(torch.randn(N, d), dim=-1)
    views = []
    for v in range(V):
        ids = torch.nonzero(torch.rand(N) < 0.6).squeeze(1)
        views.append((ids, Fn.normalize(mu[ids] + 0.4 * torch.randn(len(ids), d), dim=-1)))
    f = DirectionalSemanticField(N, d, FieldConfig(view_mass_frac=0.01, readmit_depth=None))
    tally = None
    for v, (ids, U) in enumerate(views):
        if v == half:
            calib = RM.calibrate_field(f, 0.9)
            tally = RM.ChangeTally(f)
        TC._view(f, ids, U)
        if tally is not None:
            tally.step(v - half)
    thr = calib["thr_by_hist"]
    ref = TC._brute_tests(views, lambda v, b: float("inf") if v < half else thr[b])[half:]
    s = tally.summary()
    assert s["views"] == V - half
    for row, (t_ids, f_ids, tb, fb) in zip(s["per_view"], ref):
        assert row["tests"] == len(t_ids) and row["fires"] == len(f_ids)
        assert row["tests_by_hist"] == torch.bincount(torch.tensor(tb, dtype=torch.long),
                                                      minlength=6).tolist()
        assert row["fires_by_hist"] == torch.bincount(torch.tensor(fb, dtype=torch.long),
                                                      minlength=6).tolist()
    T = sum(len(r[0]) for r in ref)
    F = sum(len(r[1]) for r in ref)
    assert s["tests"] == T and s["fires"] == F > 0 and math.isclose(s["rate"], F / T)
    assert sum(b["tests"] for b in s["by_hist"]) == T and sum(s["score_hist"]) == T
    fired = sorted({n for r in ref for n in r[1]})
    tested = {n for r in ref for n in r[0]}
    pm = f.pixel_mass.double()
    seen_upd = {int(n) for ids, _ in views[half:] for n in ids}
    assert s["gaussians_reset"] == len(fired) and s["gaussians_tested"] == len(tested)
    assert s["gaussians_observed_update"] == len(seen_upd) and set(fired) <= tested <= seen_upd
    assert math.isclose(s["gaussian_fa"], len(fired) / len(seen_upd))      # the paper's FA
    assert math.isclose(s["reset_share_of_tested"], len(fired) / len(tested))
    assert math.isclose(s["reset_pixel_mass_share"], float(pm[fired].sum() / pm[pm > 0].sum()))
    # calibration record: borrowing (history rule only) and the null's own exceedance
    counts = f.c_hist.sum(1)
    centres = (torch.arange(4000, dtype=torch.float64) + 0.5) / 2000
    for b in range(6):
        own = bool(counts[b] >= f.MIN_NULL)
        assert (calib["borrows_from"][b] is None) == own
        if counts[b] == 0:
            assert calib["null_exceedance"][b] is None
            continue
        above = float(f.c_hist[b][centres > thr[b]].sum() / counts[b])
        assert math.isclose(calib["null_exceedance"][b], above, abs_tol=1e-12)
        if own:
            assert calib["null_exceedance"][b] <= 1 - 0.9 + 1e-12      # q-quantile, own null
    print(f"PASS tally = brute force: {T} tests, {F} fires, rate {F / T:.4f}")


def _mock_scene(N=40, H=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(H), indexing="xy"), -1)
    return MockFeatureRasterizer(torch.rand(N, 2, generator=g) * H,
                                 1.0 + 2.0 * torch.rand(N, generator=g),
                                 0.4 + 0.5 * torch.rand(N, generator=g),
                                 torch.rand(N, generator=g), grid.reshape(-1, 2).float()), H


def test_residual_null_matches_brute_force():
    """ResidualNull: the per-view persistent residual (min of the last k observed), its
    mapping-pass histogram threshold and the update-pass tests / fires equal a brute force
    over the per-view residuals that ChangeBaselines.update returns."""
    torch.manual_seed(0)
    ras, H = _mock_scene()
    N, P, d, k, q = ras.N, H * H, 6, 2, 0.9
    cb, null = ChangeBaselines(N, k=k, view_mass_frac=1e-3), ResidualNull()
    gd = 1.0 + torch.rand(N)
    W = ras.weights()
    ren_d = W @ gd / W.sum(1).clamp_min(1e-12)
    hist, rings, vals, got = {}, {}, {}, []
    for v in range(20):
        obs = torch.rand(P) < 0.8                            # partial pixel masks per view
        pv = cb.update(ras, P, observed_feat=Fn.normalize(torch.randn(P, d), dim=-1),
                       rendered_feat=Fn.normalize(torch.randn(P, d), dim=-1),
                       depth_obs=torch.where(obs, ren_d + 0.01 * torch.randn(P), torch.zeros(P)),
                       gauss_depth=gd, rendered_depth=ren_d)
        if v < 10:
            null.record(cb, pv)
        else:
            if v == 10:
                null.calibrate(q)
            got.append(null.test(cb, pv))
        ref = {}
        for name, r in pv.items():                           # brute force persistence
            for n in range(N):
                if not math.isfinite(float(r[n])):
                    continue
                ring = rings.setdefault((name, n), [])
                ring.append(float(r[n]))
                if len(ring) >= k:
                    ref.setdefault(name, []).append(min(ring[-k:]))
        for name, xs in ref.items():
            (vals if v < 10 else hist).setdefault(name, []).append(xs)
    for name, per_view in vals.items():
        xs = torch.tensor([x for vv in per_view for x in vv], dtype=torch.float64)
        hi = null.hi(name)
        b = (xs.clamp(0, hi) * (null.bins / hi)).long().clamp(max=null.bins - 1)
        cdf = torch.cumsum(torch.bincount(b, minlength=null.bins).double(), 0) / len(xs)
        i = int(torch.nonzero(cdf >= q)[0])
        assert math.isclose(null.thr[name], (i + 1) * hi / null.bins), name
        assert null.summary()[name]["null_samples"] == len(xs)
    for t, row in enumerate(got):
        for name, (tests, fires) in row.items():
            xs = hist.get(name, [[]] * len(got))[t]
            assert tests == len(xs), (name, t)
            assert fires == sum(x > null.thr[name] for x in xs), (name, t)
    assert sum(r["feat"][1] for r in got) > 0
    print(f"PASS residual null = brute force: thresholds {null.thr}")


# ---------------------------------------------------------------- mock CLI
XFORM = '''
import torch
import torch.nn.functional as Fn
CALLS = []

def noise(arg, pca, device):
    g = torch.Generator().manual_seed(7)
    s = float(arg)
    def xform(z, fr, i, hw):
        CALLS.append((i, tuple(fr["labels"].shape), tuple(hw), tuple(z.shape)))
        return Fn.normalize(z + s * torch.randn(z.shape, generator=g), dim=-1)
    return xform
'''


def _cfg_args(name, selection=None):
    return SR.run_args(SR.load_config(REPO / "configs" / f"{name}.json"), selection)


def test_mock_static_summary_complete(tmp_path, monkeypatch):
    (tmp_path / "fa_xform.py").write_text(XFORM)
    monkeypatch.syspath_prepend(str(tmp_path))
    out = tmp_path / "runs" / "static_fa" / "toy.json"
    RM.main(["--backend", "mock", "--mock-static", "--mock-frames", "32",
             "--feature-transform", "fa_xform:noise", "--feature-transform-arg", "0.05",
             "--summary-json", str(out)] + _cfg_args("static_fa"))
    import fa_xform
    assert [c[0] for c in fa_xform.CALLS] == list(range(16))     # update views only
    assert all(c[1:] == ((48, 48), (48, 48), (48 * 48, 32)) for c in fa_xform.CALLS)
    rec = json.loads(out.read_text())
    assert rec["status"] == "ok"
    assert rec["config"]["feature_transform"] == "fa_xform:noise"
    assert rec["config"]["feature_transform_arg"] == "0.05"
    m = rec["metrics"]
    u = m["update"]
    assert u["views"] == 16 and len(u["per_view"]) == 16
    assert u["tests"] == sum(r["tests"] for r in u["per_view"]) > 0
    assert u["fires"] == sum(r["fires"] for r in u["per_view"])
    assert isinstance(u["rate"], float) and 0.0 <= u["rate"] <= 1.0
    assert [b["bucket"] for b in u["by_hist"]] == RM._bucket_labels(
        DirectionalSemanticField.HIST_EDGES)
    assert sum(u["score_hist"]) == u["tests"] and len(u["score_hist"]) == 4000
    assert 0.0 <= u["reset_pixel_mass_share"] <= 1.0
    for key in ("gaussians_observed", "gaussians_reset", "reset_update_mass_share"):
        assert key in u
    c = m["calibration"]
    assert c["rule"] == "history" and c["quantile"] == 0.999 and len(c["thr_by_hist"]) == 6
    assert math.isfinite(c["change_thr"]) and len(c["null_samples"]) == 6
    assert m["config"]["k"] == 2 and m["config"]["calib_mode"] == "history"
    assert m["field_config"]["readmit_depth"] == 0.05
    assert set(u["baselines"]) == {"ncc", "feat", "depth_gauss", "depth_render"}
    for name, b in u["baselines"].items():
        assert b["tests"] > 0 and 0.0 <= b["rate"] <= 1.0, name
        assert math.isfinite(m["baselines_null"][name]["threshold"]), name
    print(f"PASS mock static summary: rate {u['rate']:.4f} ({u['fires']}/{u['tests']}); "
          f"baselines {({n: round(b['rate'], 4) for n, b in u['baselines'].items()})}")


def test_no_save_field_leaves_existing_field(tmp_path):
    """--no-save-field (save_field of run_gsplat): an existing field file stays byte-identical
    and the 1 GB record is never built; without it the record replaces the file atomically."""
    out = tmp_path / "fields" / "office_0_lseg.pt"
    out.parent.mkdir()
    torch.save(dict(field="v1"), out)
    before = out.read_bytes()
    built = []
    build = lambda: built.append(1) or dict(field="new", code=dict(mapping_code_sha256="ab" * 32))
    assert RM.save_field(str(out), build, no_save=True) is False
    assert out.read_bytes() == before and not built
    assert RM.save_field(str(out), build) is True and built == [1]
    assert torch.load(out)["field"] == "new"
    assert sorted(p.name for p in out.parent.iterdir()) == [out.name]    # no temp file left


def test_feature_transform_loader(tmp_path, monkeypatch):
    (tmp_path / "fa_fac.py").write_text("def make(arg, pca, device):\n"
                                        "    return (arg, pca, device)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert RM.load_feature_transform("", "x", None, "cpu") is None
    assert RM.load_feature_transform("fa_fac:make", "a=1", "PCA", "cpu") == ("a=1", "PCA", "cpu")
    with pytest.raises(SystemExit):
        RM.load_feature_transform("fa_fac", "", None, "cpu")


def test_fixed_threshold_and_default_mock(tmp_path, capsys):
    """--fixed-change-thr skips the calibration (rule "fixed", no history thresholds); the
    default mock run prints exactly what run_mapping.py at git 3eb0e86 prints (no summary,
    same random numbers)."""
    out = tmp_path / "fixed.json"
    RM.main(["--backend", "mock", "--mock-static", "--mock-frames", "20", "--fixed-change-thr",
             "0.05", "--summary-json", str(out)])
    m = json.loads(out.read_text())["metrics"]
    assert m["calibration"]["rule"] == "fixed" and m["calibration"]["thr_by_hist"] is None
    assert m["calibration"]["change_thr"] == 0.05 and m["config"]["fixed_change_thr"] == 0.05
    assert m["calibration"]["borrows_from"] == [None] * 6           # nothing is borrowed
    assert len(m["calibration"]["null_exceedance"]) == 6
    r = subprocess.run(["git", "-C", str(REPO), "show", "3eb0e86:scripts/run_mapping.py"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("git history with 3eb0e86 not available")
    (tmp_path / "run_mapping_v1.py").write_text(r.stdout)
    v1 = _load_path("run_mapping_v1", tmp_path / "run_mapping_v1.py")
    capsys.readouterr()
    v1.run_mock()
    before = capsys.readouterr().out
    RM.main(["--backend", "mock"])
    assert capsys.readouterr().out == before and "calibrated change threshold" in before


# ---------------------------------------------------------------- validation / test rules
def _mock_run(results, cfg, scene, seed, selection=None, frames="24"):
    p = Path(results) / "runs" / cfg / f"{scene}.json"
    RM.main(["--backend", "mock", "--mock-static", "--mock-frames", frames, "--seed",
             str(seed), "--summary-json", str(p)] + _cfg_args(cfg, selection))
    return p


def test_select_and_criterion(tmp_path, monkeypatch):
    monkeypatch.setattr(SR, "_dirty", lambda *a, **k: [])     # the dev tree may be dirty
    res = tmp_path / "out"
    frozen = SR.load_config(REPO / "configs" / "static_fa.json")
    val, test = frozen["data"]["val_scenes"], frozen["data"]["test_scenes"]
    for i, s in enumerate(val):
        for c in ("static_fa", "static_fa_k1"):
            _mock_run(res, c, s, seed=i)
    with pytest.raises(SystemExit, match="from the validation selection"):
        _cfg_args("static_fa_fixed")
    sel = SR.main(["select", "--config", str(REPO / "configs" / "static_fa.json"),
                   "--results", str(res)])
    rec = json.loads((res / "val_selection.json").read_text())
    assert rec["status"] == "ok"
    pooled = [0] * 4000
    for s in val:
        h = json.loads((res / "runs" / "static_fa" / f"{s}.json").read_text())
        pooled = [a + b for a, b in zip(pooled, h["metrics"]["update"]["score_hist"])]
    f = DirectionalSemanticField(4, 2, FieldConfig(calib_mode="pooled"))
    f.c_hist[0] = torch.tensor(pooled, dtype=torch.float64)
    assert math.isclose(sel["fixed_change_thr"], f.calibrate_change_threshold(0.999))
    assert set(sel["rates"]) == {"static_fa", "static_fa_k1", "static_fa_pooled",
                                 "static_fa_max"}
    sel_path = str(res / "val_selection.json")
    assert _cfg_args("static_fa_fixed", sel_path)[-2:] == [
        "--fixed-change-thr", repr(float(sel["fixed_change_thr"]))]
    for i, s in enumerate(test):
        _mock_run(res, "static_fa", s, seed=10 + i)
    _mock_run(res, "static_fa_fixed", test[0], seed=10, selection=sel_path)
    crit = SR.main(["criterion", "--config", str(REPO / "configs" / "static_fa.json"),
                    "--results", str(res)])
    rates = {s: json.loads((res / "runs" / "static_fa" / f"{s}.json").read_text())
             ["metrics"]["update"]["rate"] for s in test}
    inside = sum(r is not None and 5e-4 <= r <= 2e-3 for r in rates.values())
    assert crit["scenes_inside"] == inside and crit["passed"] == (inside >= 5)
    assert crit["rate_per_scene"] == rates
    assert list(crit["rates"]["static_fa_fixed"]) == [test[0]]
    assert json.loads((res / "test_criterion.json").read_text())["status"] == "ok"
    # the Python step itself: no second test record, no re-selection after test
    crit_cli = ["criterion", "--config", str(REPO / "configs" / "static_fa.json"),
                "--results", str(res)]
    before = (res / "test_criterion.json").read_bytes()
    with pytest.raises(SystemExit, match="holds a finished test run"):
        SR.main(crit_cli)
    assert (res / "test_criterion.json").read_bytes() == before
    with pytest.raises(SystemExit, match="validation is closed"):
        SR.main(["select", "--config", str(REPO / "configs" / "static_fa.json"),
                 "--results", str(res)])
    assert SR.main(crit_cli + ["--force"])["passed"] == crit["passed"]
    # unfinished or mismatched frozen runs are refused
    p = res / "runs" / "static_fa" / f"{test[1]}.json"
    d = json.loads(p.read_text())
    p.write_text(json.dumps(dict(d, status="running")))
    with pytest.raises(SystemExit, match="not a finished"):
        SR.criterion(SR.build_parser().parse_args(
            ["criterion", "--config", str(REPO / "configs" / "static_fa.json"),
             "--results", str(res)]))
    bad = list(d["argv"])
    bad[bad.index("--change-quantile") + 1] = "0.99"
    p.write_text(json.dumps(dict(d, argv=bad)))
    with pytest.raises(SystemExit, match="not run with the arguments"):
        SR.criterion(SR.build_parser().parse_args(
            ["criterion", "--config", str(REPO / "configs" / "static_fa.json"),
             "--results", str(res)]))
    print(f"PASS select (fixed thr {sel['fixed_change_thr']:.4f}) and criterion "
          f"({inside}/6 inside -> {'PASS' if crit['passed'] else 'FAIL'}) on mock runs")


def test_one_code_version(tmp_path, monkeypatch):
    """select refuses validation runs of two mapping codes and records the code; criterion
    refuses test runs of two mapping codes or commits, or of another code than validation's."""
    monkeypatch.setattr(SR, "_dirty", lambda *a, **k: [])
    res = tmp_path / "out"
    cfg = str(REPO / "configs" / "static_fa.json")
    frozen = SR.load_config(cfg)
    val, test = frozen["data"]["val_scenes"], frozen["data"]["test_scenes"]

    def stamp(p, code, commit=None):
        d = json.loads(p.read_text())
        d["metrics"]["code"] = dict(mapping_code_sha256=code)
        if commit:
            d["git"] = dict(d.get("git") or {}, commit=commit)
        p.write_text(json.dumps(d))
    vals = [_mock_run(res, "static_fa", s, seed=i) for i, s in enumerate(val)]
    stamp(vals[0], "a" * 64)
    stamp(vals[1], "f" * 64)
    with pytest.raises(SystemExit, match="2 mapping codes"):
        SR.main(["select", "--config", cfg, "--results", str(res)])
    assert not SR._finished(res / "val_selection.json")
    stamp(vals[1], "a" * 64)
    sel = SR.main(["select", "--config", cfg, "--results", str(res)])
    assert sel["mapping_code_sha256"] == "a" * 64
    tests = [_mock_run(res, "static_fa", s, seed=10 + i) for i, s in enumerate(test)]
    crit = ["criterion", "--config", cfg, "--results", str(res)]
    for k, p in enumerate(tests):
        stamp(p, "a" * 64, "1" * 40 if k < 3 else "2" * 40)
    with pytest.raises(SystemExit, match="2 git commits"):
        SR.main(crit)
    for k, p in enumerate(tests):
        stamp(p, "a" * 64 if k < 3 else "f" * 64, "1" * 40)
    with pytest.raises(SystemExit, match="2 mapping codes"):
        SR.main(crit)
    for p in tests:
        stamp(p, "f" * 64, "1" * 40)
    with pytest.raises(SystemExit, match="validation selection"):
        SR.main(crit)
    assert not SR._finished(res / "test_criterion.json")
    for p in tests:
        stamp(p, "a" * 64, "1" * 40)
    out = SR.main(crit)
    assert out["mapping_code_sha256"] == "a" * 64 and out["git_commit"] == "1" * 40


def test_criterion_refusals(tmp_path, monkeypatch):
    """static_fa_report.py criterion refuses on its own, before writing anything: without a
    finished validation selection, with uncommitted code, and when a config file (the
    criterion bounds included) differs from the one the selection read."""
    monkeypatch.setattr(SR, "_dirty", lambda *a, **k: [])
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    for p in (REPO / "configs").glob("static_fa*.json"):
        shutil.copy2(p, cfg_dir / p.name)
    res = tmp_path / "out"
    cli = ["criterion", "--config", str(cfg_dir / "static_fa.json"), "--results", str(res)]
    out = res / "test_criterion.json"
    with pytest.raises(SystemExit, match="validation selection"):
        SR.main(cli)
    assert not out.exists()
    for s in SR.load_config(cfg_dir / "static_fa.json")["data"]["val_scenes"]:
        _mock_run(res, "static_fa", s, seed=0)
    SR.main(["select", "--config", str(cfg_dir / "static_fa.json"), "--results", str(res)])
    sel = json.loads((res / "val_selection.json").read_text())
    assert {Path(e["path"]).name for e in sel["inputs"]} == {
        p.name for p in cfg_dir.glob("static_fa*.json")}
    monkeypatch.setattr(SR, "_dirty", lambda *a, **k: [" M scripts/run_mapping.py"])
    with pytest.raises(SystemExit, match="uncommitted changes"):
        SR.main(cli)
    monkeypatch.setattr(SR, "_dirty", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="git cannot tell"):
        SR.main(cli)
    monkeypatch.setattr(SR, "_dirty", lambda *a, **k: [])
    c = json.loads((cfg_dir / "static_fa.json").read_text())
    c["criterion"]["high"] = 20.0                            # a looser criterion after val
    (cfg_dir / "static_fa.json").write_text(json.dumps(c, indent=1))
    with pytest.raises(SystemExit, match="changed after the validation selection"):
        SR.main(cli)
    assert not out.exists()
    print("PASS criterion refusals: no selection, dirty tree, unknown git state, config "
          "changed after validation")


def test_config_and_script_agree():
    frozen = SR.load_config(REPO / "configs" / "static_fa.json")
    c = frozen["criterion"]
    assert (c["nominal"], c["low"], c["high"], c["min_scenes"], c["num_scenes"]) == \
        (1e-3, 0.5, 2.0, 5, 6)
    assert frozen["change_quantile"] == 0.999 and frozen["field_config"] == dict(
        calib_mode="history", change_persist=2, change_quantile=0.999, readmit_depth=0.05)
    for name, change in (("static_fa_pooled", dict(calib_mode="pooled")),
                         ("static_fa_max", dict(calib_mode="max")),
                         ("static_fa_k1", dict(change_persist=1)), ("static_fa_fixed", {})):
        a = SR.load_config(REPO / "configs" / f"{name}.json")
        assert a["field_config"] == dict(frozen["field_config"], **change), name
        assert (a["change_quantile"], a["update_stride"], a["use_depth"]) == (0.999, 5, True)
    assert SR.load_config(REPO / "configs" / "static_fa_fixed.json")["fixed_change_thr"] == \
        "from_validation"
    sh = (REPO / "scripts" / "replica_static_fa.sh").read_text()
    dat = frozen["data"]
    for flag, v in (("--stride", dat["map_stride"]), ("--downscale", dat["downscale"]),
                    ("--pca-frames", dat["pca_frames"]), ("--feat-dim", dat["feat_dim"]),
                    ("--seed", dat["seed"]), ("--extractor", dat["extractor"])):
        assert f"{flag} {v} " in sh or f"{flag} {v}\n" in sh, flag
    assert "within [0.5x, 2x] of the nominal 1e-3 on >= 5 of the 6" in sh
    assert "scripts/gpu_gate.sh" in sh and "--no-save-field" in sh
    assert "/home/thanhnc/gauss_ws/data/uags_results/static_fa" in sh


# ---------------------------------------------------------------- shell refusal rules
def _sh_repo(tmp_path):
    """A committed copy of the files replica_static_fa.sh uses, with a gpu_gate.sh stub that
    records its call and fails (no run can start)."""
    root = tmp_path / "repo"
    for rel in ("scripts/replica_static_fa.sh", "scripts/static_fa_report.py",
                "scripts/audit_change_coverage.py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, root / rel)
    (root / "configs").mkdir()
    for p in (REPO / "configs").glob("static_fa*.json"):
        shutil.copy2(p, root / "configs" / p.name)
    (root / "usplat").mkdir()
    (root / "usplat" / "__init__.py").write_text("")
    gate = root / "scripts" / "gpu_gate.sh"
    gate.write_text("#!/usr/bin/env bash\necho called >> \"$GATE_LOG\"\nexit 3\n")
    gate.chmod(0o755)
    git = lambda *a: subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c",
                                     "user.email=t@t", *a], check=True, capture_output=True)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    return root


def _sh(root, out, stage, **env):
    e = dict(os.environ, OUT=str(out), STAGE=stage, GATE_LOG=str(out / "gate.log"), **env)
    e.pop("PYTHONPATH", None)
    return subprocess.run(["bash", str(root / "scripts" / "replica_static_fa.sh")], env=e,
                          capture_output=True, text=True, timeout=120)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_shell_refusals(tmp_path):
    root = _sh_repo(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    r = _sh(root, out, "test")                               # no validation selection
    assert r.returncode == 1 and "not a finished validation selection" in r.stdout, r.stdout
    (out / "val_selection.json").write_text(json.dumps(
        dict(status="running", metrics=dict(fixed_change_thr=0.1))))
    r = _sh(root, out, "test")                               # unfinished selection
    assert r.returncode == 1 and "not a finished validation selection" in r.stdout
    (out / "val_selection.json").write_text(json.dumps(
        dict(status="ok", metrics=dict(fixed_change_thr=0.1))))
    (out / "test_criterion.json").write_text(json.dumps(dict(status="ok")))
    r = _sh(root, out, "test")                               # finished test record
    assert r.returncode == 1 and "holds a finished test run" in r.stdout, r.stdout
    (out / "test_criterion.json").unlink()
    r = _sh(root, out, "test")                               # checks pass: maps or the gate
    assert r.returncode != 0
    assert "missing map" in r.stdout or (out / "gate.log").exists(), r.stdout + r.stderr
    crit = lambda: subprocess.run(
        [sys.executable, str(root / "scripts" / "static_fa_report.py"), "criterion",
         "--config", str(root / "configs" / "static_fa.json"), "--results", str(out)],
        cwd=root, capture_output=True, text=True, timeout=60)
    r = crit()                                               # clean tree: the config check
    assert r.returncode == 1 and "did not read static_fa.json" in r.stderr, r.stderr
    with open(root / "scripts" / "static_fa_report.py", "a") as f:
        f.write("# local edit\n")
    r = _sh(root, out, "test")                               # dirty tree
    assert r.returncode == 1 and "uncommitted changes" in r.stdout, r.stdout
    r = crit()                                               # the Python step refuses too
    assert r.returncode == 1 and "uncommitted changes" in r.stderr, r.stderr
    assert not (out / "test_criterion.json").exists()
    (out / "runs" / "static_fa").mkdir(parents=True)
    (out / "runs" / "static_fa" / "room_0.json").write_text("{}")
    r = _sh(root, out, "val")                                # validation closed
    assert r.returncode == 1 and "validation is closed" in r.stdout, r.stdout
    r = _sh(root, out, "nope")
    assert r.returncode == 2
    # a finished per-scene test record of another commit is refused, not kept
    subprocess.run(["git", "-C", str(root), "checkout", "-q", "--", "."], check=True)
    import shutil as sh
    sh.rmtree(out / "runs")
    maps = tmp_path / "maps"
    maps.mkdir()
    frozen = SR.load_config(REPO / "configs" / "static_fa.json")
    for s in frozen["data"]["test_scenes"]:
        (maps / f"{s}_seq1.pt").write_bytes(b"")
    room = out / "runs" / "static_fa" / f"{frozen['data']['test_scenes'][0]}.json"
    room.parent.mkdir(parents=True)
    room.write_text(json.dumps(dict(status="ok", git=dict(commit="0" * 40))))
    r = _sh(root, out, "test", MAPS=str(maps))
    assert r.returncode == 1 and "finished test record of another commit" in r.stdout, r.stdout
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    room.write_text(json.dumps(dict(status="ok", git=dict(commit=head))))
    r = _sh(root, out, "test", MAPS=str(maps))              # kept; the next run hits the gate
    assert "finished, kept" in r.stdout and (out / "gate.log").exists(), r.stdout + r.stderr
    print("PASS replica_static_fa.sh refusals: no selection, unfinished selection, finished "
          "test record, dirty tree (shell and criterion step), closed validation")
