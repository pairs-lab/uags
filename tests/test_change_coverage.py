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
