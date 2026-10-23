"""The fused-reset attempt (configs/reset_fused_prereg.json) on the CPU: usplat/reset_fused.py,
scripts/reset_fused_nor.py (mock backend, the toy NeRF-Object-Removal scene of
tests/test_eval_change.py) and scripts/reset_fused_select.py.
Checks:
  * external_reset equals the field's own reset bit for bit: two copies of the toy's saved
    field take the update views, one fires by its own test, the other with its test off and
    external_reset on the same Gaussians; every state tensor is identical after every view
    (default config and the relmax / fast_min 0.5 config of the NOR fields); the field did
    reset (the test is not vacuous); external_reset refuses a field whose own test fired
  * run_variant with our own rule (native_decider) gives run_update's ours exactly: flags,
    resets, robust set, changed set, score
  * the scene run reproduces the toy record and the toy dump (N2 delta 0), the captured
    mapping samples equal the replay's nulls, every patch is undone; the ours row equals
    Table III's ours; the variant thresholds equal an independent recomputation from the
    replay; the variants reset the removed object; a finished test record is refused
  * MappingRanks / fused_stat on hand examples (mid-ranks, NaN -> 0, all-NaN -> NaN)
  * select / check-test / table on hand-made records: the static-relabel budget, ties to
    ours, keep ours; refusals (unfinished, other prereg, untracked or dirty prereg, missing
    selection); the 7/10 + mean claim rule in both directions
"""
import importlib.util
import json
import math
import pathlib
import shutil
import subprocess

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TD = _load("toy_dump_nor_qual", REPO / "tests" / "test_dump_nor_qual.py")
RF = _load("reset_fused_nor", REPO / "scripts" / "reset_fused_nor.py")
SEL = _load("reset_fused_select", REPO / "scripts" / "reset_fused_select.py")
EC, DQ = RF.EC, RF.DQ
from usplat.reset_fused import (MappingRanks, fused_stat, ExternalResetField,  # noqa: E402
                                VARIANTS)
from usplat.vmf_field import DirectionalSemanticField, _STATE_TENSORS  # noqa: E402

PREREG = REPO / "configs" / "reset_fused_prereg.json"


def rf_argv(t, out, stage="toy", prereg=PREREG, dump=None):
    a = ["--table", str(t["rec"]), "--tolerance", str(t["tol"]), "--vocab", str(t["vocab"]),
         "--text-cache", str(t["cache"]), "--prereg", str(prereg), "--stage", stage,
         "--backend", "mock", "--device", "cpu", "--out", str(out)]
    return a + (["--dump", str(dump)] if dump else [])


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    t = TD.toy_inputs(tmp_path_factory.mktemp("reset_fused"))
    RF.EC.load_extractor = TD.EC.load_extractor
    assert RF.EC is TD.EC and RF.DQ.EC is EC
    DQ.main(TD.argv(t, t["tmp"] / "dump"))
    sha = DQ.sha256_file(t["cache"])
    out = t["tmp"] / "rf.json"
    res = RF.main(rf_argv(t, out, dump=t["tmp"] / "dump"))
    t.update(res=res, out=out, rf=json.loads(out.read_text()), cache_sha=sha,
             record=json.loads(t["rec"].read_text()))
    rec = t["record"]
    ns = DQ.ec_namespace(rec, DQ.build_parser().parse_args(TD.argv(t, t["tmp"] / "x")))
    t["ctx"] = EC.load_context(ns)
    t["k"] = int(rec["settings"]["k"])
    return t


def _state(f):
    s = {n: getattr(f, n) for n in _STATE_TENSORS}
    for n in ("R", "W1", "W2", "Q", "Wr"):
        s[f"raw_{n}"], s[f"cen_{n}"] = getattr(f.raw, n), getattr(f.cen, n)
    return s


@pytest.mark.parametrize("relmax", [False, True])
def test_external_reset_equals_the_native_reset(run, relmax):
    ctx = run["ctx"]
    st = dict(ctx["ckpt"]["field"])
    if relmax:
        st["config"] = dict(st["config"], change_weight="relmax", fast_min=0.5, slow_min=0.5)
    a = DirectionalSemanticField.from_state(st)
    b = ExternalResetField.from_state(st)
    a.calibrate_change_threshold(0.999)
    b.calibrate_change_threshold(0.999)
    thr = a.thr_by_hist.clone()
    b.disable_own_test()
    with torch.no_grad():
        for v in ctx["upd_views"]:
            z, ras, gz, kw = EC.view_inputs(ctx, v)
            a.update_view(ras, z, num_pixels=v["h"] * v["w"], **kw)
            sb = b.update_view(ras, z, num_pixels=v["h"] * v["w"], **kw)
            hb = torch.bucketize(b.ns, b._hist_edges)
            fire = b.last_tested & (b.c_last > thr[hb])
            assert torch.equal(fire, a.last_fired)
            b.external_reset(sb, fire, gauss_depth=gz)
            sa, sbb = _state(a), _state(b)
            for n in sa:
                assert DQ.nan_equal(sa[n], sbb[n]), n
    assert int(a.n_resets.sum()) > 0 and bool(a.changed.any())
    c = DirectionalSemanticField.from_state(st)              # own test on, never disabled
    with pytest.raises(RuntimeError, match="calibrate"):
        ExternalResetField.from_state(st).disable_own_test()
    c2 = ExternalResetField.from_state(st)
    c2.calibrate_change_threshold(0.999)
    with torch.no_grad():
        for v in ctx["upd_views"]:
            z, ras, gz, kw = EC.view_inputs(ctx, v)
            s = c2.update_view(ras, z, num_pixels=v["h"] * v["w"], **kw)
            if bool(c2.last_fired.any()):
                with pytest.raises(RuntimeError, match="own test fired"):
                    c2.external_reset(s, torch.zeros_like(c2.last_fired), gauss_depth=gz)
                break
        else:
            pytest.fail("the own test never fired on the toy")
    del c


def test_native_rule_through_run_variant_equals_run_update(run):
    ctx, k = run["ctx"], run["k"]
    replay, null, _ = EC.replay_mapping(ctx, ctx["map_views"], k, len(ctx["upd_views"]))
    thr = {r: nl.threshold(0.999) for r, nl in null.items()}
    with DQ.capture_fields(None) as made, torch.no_grad():
        upd = EC.run_update(ctx, ctx["upd_views"], k, thr, 0.999)
    ours = made[0]
    with torch.no_grad():
        r = RF.run_variant(ctx, ctx["upd_views"], k, 0.999, RF.native_decider(ours.thr_by_hist),
                           need_normlift=False)
    assert torch.equal(r["flags"], upd["flags"]["ours"])
    assert r["n_resets"] == upd["n_resets"] > 0
    assert torch.equal(r["robust_end"], upd["robust_end"])
    assert torch.equal(r["changed_end"], upd["changed_end"])
    assert torch.equal(r["observed"], upd["observed"])
    assert torch.equal(r["tests"], upd["tests"]["ours"]) and torch.equal(r["fires"], upd["fires"]["ours"])
    a, b = r["score"].double(), upd["scores"]["ours"].double()
    assert torch.equal(torch.isnan(a), torch.isnan(b))
    ok = ~torch.isnan(a)
    assert torch.allclose(a[ok].clamp_min(0), b[ok], atol=0, rtol=0)     # c_max starts at 0
    assert torch.allclose(r["mu"], ours.mu_post().float(), atol=0, rtol=0)


def test_scene_run(run, tmp_path):
    rf, rec = run["rf"], run["record"]
    assert rf["status"] == "done" and run["res"]["exit"] == 0
    assert rf["reproduction"]["ok"] and rf["reproduction"]["max_ap_auroc_delta"] == 0.0
    assert all(rf["null_identity"][c] for c in ("ours", "feat", "depth", "normlift"))
    assert rf["dump_check"]["ok"] and all(
        v["delta"] == 0.0 for s, v in rf["dump_check"].items() if s != "ok")
    # patches undone
    assert EC.DirectionalSemanticField is DirectionalSemanticField
    assert EC.residuals_view.__module__ == EC.__name__ and EC.make_normlift.__module__ == EC.__name__
    assert DQ.SETS == ("before", "nochange", "ours", "avg_before", "avg_after")
    assert set(EC.RULES) == set(EC.ROWS) | {"depth_gauss"}
    assert DQ.sha256_file(run["cache"]) == run["cache_sha"]
    # the ours row is Table III's ours
    o = rf["rows"]["ours"]
    assert o["gauss_ap"] == rec["gaussian"]["rows"]["ours"]["ap"]
    assert o["pixel_ap"] == pytest.approx(rec["pixel"]["rows"]["ours"]["ap"], abs=1e-12)
    assert o["fa"] == rec["operating_point"]["ours"]["fa"]
    assert o["rec"] == rec["operating_point"]["ours"]["rec"]
    assert o["kept"] == rec["ours"]["kept"]
    dump = json.loads((run["tmp"] / "dump" / "contrast.json").read_text())["sets"]
    assert o["stale_answer"] == dump["ours"]["stale_answer_rate"]
    assert rf["context"]["nochange"]["stale_answer"] == dump["nochange"]["stale_answer_rate"]
    # the variants fire on the removed object, thresholds are mapping quantiles in [0, 1]
    for v in VARIANTS:
        x = rf["rows"][v]
        assert 0 < float(x["threshold"]) <= 1 and x["n_null"] > 0
        assert float(x["rec"]) > 0.5 and x["n_resets"] > 0
        assert 0 <= float(x["static_relabel"]) <= 1 and 0 <= float(x["stale_answer"]) <= 1
    assert rf["rows"]["V3"]["null_views"] == "NormLift segment"
    assert all(c["n_differ"] == 0 for c in rf["coverage_check"].values())     # CPU: exact
    # independent recomputation of the thresholds from the replay's nulls + our capture
    ctx, k = run["ctx"], run["k"]
    seg = EC.normlift_segment(len(ctx["map_views"]), len(ctx["upd_views"]))
    with RF.capture_replay(len(ctx["upd_views"])) as (cap, state), torch.no_grad():
        state["seg"] = seg
        EC.replay_mapping(ctx, ctx["map_views"], k, len(ctx["upd_views"]))
    pooled = {c: torch.cat([w[c] for w in cap["views"]]) for c in ("ours", "feat", "depth")}
    pooled["normlift"] = torch.cat([w["normlift"] for w in cap["views"] if w["in_segment"]])
    ranks = MappingRanks(pooled)
    for v, names in VARIANTS.items():
        vals = []
        for w in cap["views"]:
            if "normlift" in names and not w["in_segment"]:
                continue
            # rank by counting, not by searchsorted
            r = []
            for n in names:
                srt = ranks.sorted[n]
                x = w[n].double()
                lo = (srt[None, :] < x[:, None]).sum(1).double()
                eq = (srt[None, :] == x[:, None]).sum(1).double()
                r.append(torch.where(torch.isfinite(x), (lo + eq / 2) / srt.numel(),
                                     torch.zeros_like(x)))
            s = torch.stack(r, 1).mean(1)
            defined = torch.stack([torch.isfinite(w[n]) for n in names], 1).any(1)
            vals.append(s[defined])
        import numpy as np
        want = float(np.quantile(torch.cat(vals).numpy(), 0.999))
        assert float(rf["rows"][v]["threshold"]) == pytest.approx(want, abs=1e-12), v
    # a finished test record is never redone; a prereg with other variants is refused
    done = tmp_path / "t.json"
    done.write_text(json.dumps(dict(status="done", stage="test")))
    with pytest.raises(SystemExit, match="run once"):
        RF.main(rf_argv(run, done, stage="test"))
    bad = tmp_path / "p.json"
    p = json.loads(PREREG.read_text())
    p["variants"]["V1"] = ["feat"]
    bad.write_text(json.dumps(p))
    with pytest.raises(SystemExit, match="variants differ"):
        RF.main(rf_argv(run, tmp_path / "o.json", prereg=bad))


def test_ranks_and_fused_stat():
    r = MappingRanks(dict(a=torch.tensor([1.0, 2.0, 2.0, 3.0, float("nan")]), b=torch.zeros(0)))
    got = r.rank("a", torch.tensor([2.0, 0.0, 4.0, float("nan"), 1.0]))
    assert torch.allclose(got, torch.tensor([0.5, 0.0, 1.0, 0.0, 0.125], dtype=torch.float64))
    assert torch.equal(r.rank("b", torch.tensor([1.0])), torch.zeros(1, dtype=torch.float64))
    comps = dict(a=torch.tensor([2.0, float("nan"), float("nan"), 3.0]),
                 c=torch.tensor([float("nan"), 5.0, float("nan"), 0.0]))
    r = MappingRanks(dict(a=torch.tensor([1.0, 2.0, 2.0, 3.0]), c=torch.tensor([4.0, 6.0])))
    s = fused_stat(r, comps, ("a", "c"), mask=torch.tensor([True, True, True, False]))
    assert s[0] == pytest.approx(0.25) and s[1] == pytest.approx(0.25)
    assert math.isnan(s[2]) and math.isnan(s[3])            # all NaN; outside the mask


# ------------------------------------------------------------------------ selection
def _rec(scene, stage, sha, rows, code="c"):
    full = {}
    for r, (sa, cost) in rows.items():
        full[r] = dict(stale_answer=sa, static_relabel=cost, fa=0.01, rec=0.5, kept=0.1,
                       gauss_ap=0.5, pixel_ap=0.5, rate_static=0.001, n_resets=3)
    return dict(status="done", stage=stage, scene=scene, smoke=0, prereg_sha256=sha,
                reproduction=dict(ok=True), code=code, rows=full)


def _write(d, recs):
    d.mkdir(parents=True, exist_ok=True)
    for r in recs:
        (d / f"{r['scene']}.json").write_text(json.dumps(r))


def test_select_rule(tmp_path):
    pr = tmp_path / "prereg.json"
    shutil.copy(PREREG, pr)
    sha = SEL.sha256_file(pr)
    val = json.loads(pr.read_text())["scenes"]["validation"]
    # V1 lowest stale-answer but over budget (0.30 > 0.20 + 0.02); V2 qualifies (0.21)
    rows = dict(ours=(0.30, 0.20), V1=(0.05, 0.30), V2=(0.10, 0.21), V3=(0.20, 0.19))
    _write(tmp_path / "val", [_rec(s, "val", sha, rows) for s in val])
    out = SEL.main(["select", "--prereg", str(pr), "--dir", str(tmp_path / "val"),
                    "--out", str(tmp_path / "sel.json")])
    assert out["chosen"] == "V2"
    assert [r["qualifies"] for r in out["table"]["rows"]] == [True, False, True, True]
    # a tie with ours keeps ours; nothing better keeps ours
    for rows, want in ((dict(ours=(0.1, 0.2), V1=(0.1, 0.2), V2=(0.2, 0.2), V3=(0.3, 0.2)), "ours"),
                       (dict(ours=(0.1, 0.2), V1=(0.2, 0.2), V2=(0.2, 0.2), V3=(0.3, 0.2)), "ours")):
        _write(tmp_path / "v2", [_rec(s, "val", sha, rows) for s in val])
        assert SEL.main(["select", "--prereg", str(pr), "--dir", str(tmp_path / "v2"),
                         "--out", str(tmp_path / "s2.json")])["chosen"] == want
    # refusals: an unfinished record, a record of another prereg, mixed code
    bad = [_rec(s, "val", sha, rows) for s in val]
    bad[0]["status"] = "running"
    bad[1]["prereg_sha256"] = "x"
    bad[2]["code"] = "other"
    _write(tmp_path / "v3", bad)
    with pytest.raises(SystemExit, match="status running.*another prereg.*different code"):
        SEL.main(["select", "--prereg", str(pr), "--dir", str(tmp_path / "v3"),
                  "--out", str(tmp_path / "s3.json")])


def test_check_test_and_claim(tmp_path):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    for rel in SEL.CODE:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, repo / rel)
    pr = repo / "configs" / "reset_fused_prereg.json"
    shutil.copy(PREREG, pr)
    g = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)
    g("init", "-q")
    g("-c", "user.email=t@t", "-c", "user.name=t", "add", *SEL.CODE)
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "code")
    sha = SEL.sha256_file(pr)
    prereg = json.loads(pr.read_text())
    rows = dict(ours=(0.30, 0.20), V1=(0.05, 0.30), V2=(0.10, 0.21), V3=(0.20, 0.19))
    _write(tmp_path / "val", [_rec(s, "val", sha, rows) for s in prereg["scenes"]["validation"]])
    sel = tmp_path / "sel.json"
    ct = ["check-test", "--prereg", str(pr), "--selection", str(sel), "--dir",
          str(tmp_path / "val"), "--repo", str(repo)]
    with pytest.raises(SystemExit, match="not tracked.*no validation selection"):
        SEL.main(ct)
    g("-c", "user.email=t@t", "-c", "user.name=t", "add", "configs/reset_fused_prereg.json")
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "prereg")
    SEL.main(["select", "--prereg", str(pr), "--dir", str(tmp_path / "val"), "--out", str(sel),
              "--repo", str(repo)])
    assert SEL.main(ct) == 0
    pr.write_text(pr.read_text() + " ")                         # dirty prereg
    with pytest.raises(SystemExit, match="uncommitted changes"):
        SEL.main(ct)
    g("checkout", "--", "configs/reset_fused_prereg.json")
    s = json.loads(sel.read_text())
    s["chosen"] = "V1"
    sel.write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="re-selection gives V2"):
        SEL.main(ct)
    s["chosen"] = "V2"
    sel.write_text(json.dumps(s))
    # claim: V2 lower in 7/10 and on the mean -> allowed; 6/10 -> not
    test = prereg["scenes"]["test"]
    for n_win, allowed in ((7, True), (6, False)):
        recs = []
        for i, sc in enumerate(test):
            v2 = 0.10 if i < n_win else 0.31
            recs.append(_rec(sc, "test", sha, dict(ours=(0.30, 0.2), V1=(0.2, 0.3),
                                                   V2=(v2, 0.21), V3=(0.2, 0.2))))
        _write(tmp_path / f"test{n_win}", recs)
        out = SEL.main(["table", "--prereg", str(pr), "--selection", str(sel), "--test-dir",
                        str(tmp_path / f"test{n_win}"), "--out-dir", str(tmp_path / f"tab{n_win}")])
        assert out["claim"]["allowed"] is allowed and out["claim"]["wins"] == n_win
        assert (tmp_path / f"tab{n_win}" / "reset_fused_test.md").exists()
