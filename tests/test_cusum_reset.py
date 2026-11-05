"""Attempt 06, the CUSUM change test (configs/cusum_reset_prereg.json), on the CPU:
usplat/cusum_reset.py, scripts/cusum_reset_nor.py (mock backend, the toy NeRF-Object-Removal
scene of tests/test_eval_change.py) and scripts/cusum_reset_select.py.
Checks:
  * mapping_delta / cusum_null / cusum_decider on hand examples: S <- max(0, S + c - delta) only
    where c is defined (and covered), S kept elsewhere, firing only where S > thr and the
    Gaussian is not changed, S restarts at 0 after a firing, the AP score is S before the restart
  * the scene run reproduces the toy record and the toy dump (N2 delta 0), the captured
    mapping samples equal the replay's nulls, every patch is undone, the ours row equals
    Table III's ours; each variant's delta, threshold and null size equal an independent
    recomputation (plain Python floats) from a separate capture of the mapping replay
  * one variant alone (the test stage) gives the same row as among all three; refusals: a
    finished test record, a test stage with several variants, unknown variants, a prereg with
    other variants or another quantile
  * select / check-test / table on hand-made records: the FA eligibility, the tie rule
    (measured from the lowest rate, not in loop order), the 2.0-point acceptance at its
    boundary, no eligible variant; the git and selection gates; the adoption rule in every
    direction; the object-query split
"""
import importlib.util
import json
import math
import pathlib
import shutil
import subprocess

import numpy as np
import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TD = _load("toy_dump_nor_qual", REPO / "tests" / "test_dump_nor_qual.py")
CR = _load("cusum_reset_nor", REPO / "scripts" / "cusum_reset_nor.py")
SEL = _load("cusum_reset_select", REPO / "scripts" / "cusum_reset_select.py")
RF, EC, DQ = CR.RF, CR.EC, CR.DQ
from usplat.cusum_reset import (VARIANTS, mapping_delta, cusum_null,  # noqa: E402
                                cusum_threshold, cusum_decider)
from usplat.vmf_field import DirectionalSemanticField  # noqa: E402

PREREG = REPO / "configs" / "cusum_reset_prereg.json"
NAN = float("nan")


def cr_argv(t, out, stage="toy", prereg=PREREG, dump=None, variants=None):
    a = ["--table", str(t["rec"]), "--tolerance", str(t["tol"]), "--vocab", str(t["vocab"]),
         "--text-cache", str(t["cache"]), "--prereg", str(prereg), "--stage", stage,
         "--backend", "mock", "--device", "cpu", "--out", str(out)]
    a += ["--dump", str(dump)] if dump else []
    return a + (["--variants", variants] if variants else [])


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    t = TD.toy_inputs(tmp_path_factory.mktemp("cusum_reset"))
    assert CR.EC is TD.EC and DQ.EC is EC
    DQ.main(TD.argv(t, t["tmp"] / "dump"))
    sha = DQ.sha256_file(t["cache"])
    out = t["tmp"] / "cr.json"
    res = CR.main(cr_argv(t, out, dump=t["tmp"] / "dump"))
    t.update(res=res, out=out, cr=json.loads(out.read_text()), cache_sha=sha,
             record=json.loads(t["rec"].read_text()))
    rec = t["record"]
    ns = DQ.ec_namespace(rec, DQ.build_parser().parse_args(TD.argv(t, t["tmp"] / "x")))
    t["ctx"] = EC.load_context(ns)
    t["k"] = int(rec["settings"]["k"])
    return t


# ------------------------------------------------------------------------- the statistic
def _views(*vs):
    return [dict(idx=torch.tensor(i, dtype=torch.int32), ours=torch.tensor(c, dtype=torch.float32))
            for i, c in vs]


def test_cusum_math():
    # dyadic values: every sum below is exact in float32 and float64
    views = _views(([0, 1], [0.75, NAN]), ([0, 2], [0.125, 0.5]), ([0, 1, 2], [0.0, 1.0, NAN]),
                   ([0], [0.0]))
    assert mapping_delta(views, 0.5) == float(np.quantile([0.75, 0.125, 0.5, 0.0, 1.0, 0.0], 0.5))
    assert math.isnan(mapping_delta(_views(([0], [NAN])), 0.9))
    null = cusum_null(views, 3, 0.25)
    # S0: 0.5, 0.375, 0.125, 0 | S2: 0.25 | S1: 0.75 (untested at view 1: no entry, S kept at 0)
    assert null.tolist() == [0.5, 0.375, 0.25, 0.125, 0.75, 0.0]
    assert cusum_threshold(null, 0.5) == float(np.quantile(null, 0.5))
    assert math.isnan(cusum_threshold(np.zeros(0), 0.999)) and cusum_null([], 3, 0.1).size == 0

    class F:                                     # the decider reads only field.changed
        changed = torch.zeros(3, dtype=torch.bool)
    f = F()
    d = cusum_decider(0.25, 0.6, 3, "cpu")
    T = lambda *x: torch.tensor(x)
    B = lambda *x: torch.tensor(x, dtype=torch.bool)

    def step(c, cov, changed):
        f.changed = B(*changed)
        fire, stat = d(f, dict(ours=T(*c)), B(*cov))
        return fire.tolist(), [None if math.isnan(x) else x for x in stat.tolist()], d.state.tolist()
    # view 1: S = 0.5, 0.25, untested 0; nothing above 0.6
    assert step((0.75, 0.5, NAN), (1, 1, 1), (0, 0, 0)) == ([False] * 3, [0.5, 0.25, None], [0.5, 0.25, 0.0])
    # view 2: G0 fires (0.75); G1 is above (1.0) but changed: kept; G2 defined but not covered: kept
    assert step((0.5, 1.0, 1.0), (1, 1, 0), (0, 1, 0)) == ([True, False, False], [0.75, 1.0, None],
                                                          [0.0, 1.0, 0.0])
    # view 3: G1 untested keeps S = 1.0 and cannot fire; S never goes below 0
    assert step((0.25, NAN, 0.0), (1, 1, 1), (0, 0, 0)) == ([False] * 3, [0.0, None, 0.0], [0.0, 1.0, 0.0])
    # view 4: G1 tested again, not changed: fires; the AP score is S before the restart
    assert step((0.0, 0.25, 0.0), (1, 1, 1), (0, 0, 0)) == ([False, True, False], [0.0, 1.0, 0.0],
                                                           [0.0, 0.0, 0.0])
    assert set(VARIANTS) == {"C90", "C97", "C99"}
    assert {k: float(v) for k, v in json.loads(PREREG.read_text())["variants"].items()} == VARIANTS


# ------------------------------------------------------------------------- the scene run
def _independent(ctx, k, q):
    """{variant: (delta, thr, n)} from a separate capture of the mapping replay: the field's
    c_p at every covered Gaussian of every view (read from the field itself), the CUSUM in
    plain Python floats."""
    seq = []
    seg = EC.normlift_segment(len(ctx["map_views"]), len(ctx["upd_views"]))
    with RF.capture_replay(0) as (cap, state), torch.no_grad():
        state["seg"] = seg
        inner = EC.residuals_view

        def spy(cb, ras, v, z, ren, gz, kw, observed):
            per = inner(cb, ras, v, z, ren, gz, kw, observed)
            fld = cap["field"][-1]
            c = torch.where(fld.last_tested, fld.c_last, torch.full_like(fld.c_last, NAN))
            seq.append([(int(i), float(c[i])) for i in torch.nonzero(observed).squeeze(1)])
            return per
        EC.residuals_view = spy
        try:
            EC.replay_mapping(ctx, ctx["map_views"], k, len(ctx["upd_views"]))
        finally:
            EC.residuals_view = inner
    vals = [c for view in seq for _, c in view if math.isfinite(c)]
    out = {}
    for name, qd in VARIANTS.items():
        delta = float(np.quantile(np.array(vals), qd))
        S, null = {}, []
        for view in seq:
            for i, c in view:
                if math.isfinite(c):
                    S[i] = max(0.0, S.get(i, 0.0) + c - delta)
                    null.append(S[i])
        out[name] = (delta, float(np.quantile(np.array(null), q)), len(null))
    return out, len(vals)


def test_scene_run(run):
    cr, rec = run["cr"], run["record"]
    assert cr["status"] == "done" and run["res"]["exit"] == 0 and cr["variants"] == list(VARIANTS)
    assert cr["reproduction"]["ok"] and cr["reproduction"]["max_ap_auroc_delta"] == 0.0
    assert all(cr["null_identity"][c] for c in ("ours", "feat", "depth", "normlift"))
    assert cr["dump_check"]["ok"] and all(
        v["delta"] == 0.0 for s, v in cr["dump_check"].items() if s != "ok")
    assert set(cr["frozen_code"]) == {"eval_code_equal", "mapping_code_equal", "note"}
    # patches undone
    assert EC.DirectionalSemanticField is DirectionalSemanticField
    assert EC.residuals_view.__module__ == EC.__name__ and EC.make_normlift.__module__ == EC.__name__
    assert DQ.SETS == ("before", "nochange", "ours", "avg_before", "avg_after")
    assert set(EC.RULES) == set(EC.ROWS) | {"depth_gauss"}
    assert DQ.sha256_file(run["cache"]) == run["cache_sha"]
    # the ours row is Table III's ours
    o = cr["rows"]["ours"]
    assert o["gauss_ap"] == rec["gaussian"]["rows"]["ours"]["ap"]
    assert o["pixel_ap"] == pytest.approx(rec["pixel"]["rows"]["ours"]["ap"], abs=1e-12)
    assert o["fa"] == rec["operating_point"]["ours"]["fa"]
    assert o["rec"] == rec["operating_point"]["ours"]["rec"]
    assert o["kept"] == rec["ours"]["kept"]
    dump = json.loads((run["tmp"] / "dump" / "contrast.json").read_text())["sets"]
    assert o["stale_answer"] == dump["ours"]["stale_answer_rate"]
    assert cr["context"]["nochange"]["stale_answer"] == dump["nochange"]["stale_answer_rate"]
    assert all(c["n_differ"] == 0 for c in cr["coverage_check"].values())     # CPU: exact
    # every variant: its CUSUM settings, bounded metrics, the fused row
    for v in VARIANTS:
        x = cr["rows"][v]
        assert {kk: x[kk] for kk in cr["cusum"][v]} == cr["cusum"][v]
        assert x["q_delta"] == VARIANTS[v] and x["n_null"] > 0 and 0 <= x["share_zero"] <= 1
        assert 0 <= float(x["stale_answer"]) <= 1 and 0 <= float(x["static_relabel"]) <= 1
        assert 0 <= float(x["fa"]) <= 1 and 0 <= float(x["rec"]) <= 1
        assert math.isfinite(float(x["fused_gauss_ap"])) and math.isfinite(float(x["gauss_ap"]))
    # a larger reference level gives a smaller S: deltas rise, thresholds do not
    d = [cr["rows"][v]["delta"] for v in ("C90", "C97", "C99")]
    th = [cr["rows"][v]["thr"] for v in ("C90", "C97", "C99")]
    assert d == sorted(d) and th == sorted(th, reverse=True)
    assert cr["rows"]["C90"]["n_resets"] > 0 and float(cr["rows"]["C90"]["rec"]) > 0
    # independent recomputation of delta, theta_S and the null size
    want, n_vals = _independent(run["ctx"], run["k"], 0.999)
    assert n_vals == cr["null_identity"]["n"]["ours"]
    for v, (delta, thr, n) in want.items():
        x = cr["rows"][v]
        assert x["delta"] == pytest.approx(delta, abs=1e-12), v
        assert x["thr"] == pytest.approx(thr, abs=1e-12), v
        assert x["n_null"] == n == n_vals, v


def test_one_variant_and_refusals(run, tmp_path):
    t = run
    one = tmp_path / "one.json"
    res = CR.main(cr_argv(t, one, stage="test", variants="C97", dump=t["tmp"] / "dump"))
    assert res["exit"] == 0
    r1 = json.loads(one.read_text())
    assert set(r1["rows"]) == {"ours", "C97"} and r1["variants"] == ["C97"]
    # CPU, same inputs: the variant's row does not depend on the other variants of the run
    dumps = lambda x: json.dumps(x, sort_keys=True)
    assert dumps(r1["rows"]["C97"]) == dumps(t["cr"]["rows"]["C97"])
    assert dumps(r1["rows"]["ours"]) == dumps(t["cr"]["rows"]["ours"])
    with pytest.raises(SystemExit, match="run once"):
        CR.main(cr_argv(t, one, stage="test", variants="C97"))
    with pytest.raises(SystemExit, match="accepted variant only"):
        CR.main(cr_argv(t, tmp_path / "a.json", stage="test"))
    with pytest.raises(SystemExit, match="not a subset"):
        CR.main(cr_argv(t, tmp_path / "b.json", variants="C97,C50"))
    p = json.loads(PREREG.read_text())
    p["variants"]["C95"] = 0.95
    bad = tmp_path / "p1.json"
    bad.write_text(json.dumps(p))
    with pytest.raises(SystemExit, match="variants differ"):
        CR.main(cr_argv(t, tmp_path / "c.json", prereg=bad))
    p = json.loads(PREREG.read_text())
    p["threshold"]["quantile"] = 0.99
    bad = tmp_path / "p2.json"
    bad.write_text(json.dumps(p))
    with pytest.raises(SystemExit, match="prereg quantile"):
        CR.main(cr_argv(t, tmp_path / "d.json", prereg=bad))
    assert not any((tmp_path / f"{n}.json").exists() for n in "abcd")   # refused before writing


# ---------------------------------------------------------------------------- selection
def _rec(scene, stage, sha, rows, query="chair", variants=None, code="c"):
    """rows: {row: (stale_answer, fa, gauss_ap)}."""
    full = {r: dict(stale_answer=sa, static_relabel=0.01, fa=fa, rec=0.5, kept=0.1, gauss_ap=ap,
                    pixel_ap=0.5, fused_gauss_ap=0.6, fused_pixel_ap=0.6, rate_static=0.001,
                    n_resets=3)
            for r, (sa, fa, ap) in rows.items()}
    return dict(status="done", stage=stage, scene=scene, smoke=0, prereg_sha256=sha,
                reproduction=dict(ok=True), code=code, rows=full, query=dict(name=query),
                variants=variants or [r for r in rows if r != "ours"])


def _write(d, recs):
    d.mkdir(parents=True, exist_ok=True)
    for r in recs:
        (d / f"{r['scene']}.json").write_text(json.dumps(r))


def _choose(rows, scenes=("001", "002")):
    recs = {s: _rec(s, "val", "x", rows) for s in scenes}
    return SEL.choose(SEL.summary(recs))


def test_choose_rule():
    # C90 lowest but FA over 2 x 0.05; C99 ties C97 (0.05 point) with the higher AP
    ch, acc, tab = _choose(dict(ours=(0.30, 0.05, 0.5), C90=(0.10, 0.12, 0.6), C97=(0.20, 0.08, 0.5),
                                C99=(0.2005, 0.06, 0.7)))
    assert (ch, acc) == ("C99", True)
    assert [r["eligible"] for r in tab["rows"]] == [False, True, True]
    assert [r["tied"] for r in tab["rows"]] == [False, True, True]
    # ties are measured from the lowest rate: C99 is 0.18 point above C90, not tied
    ch, acc, tab = _choose(dict(ours=(0.30, 0.05, 0.5), C90=(0.200, 0.05, 0.5), C97=(0.2009, 0.05, 0.6),
                                C99=(0.2018, 0.05, 0.9)))
    assert (ch, acc) == ("C97", True) and [r["tied"] for r in tab["rows"]] == [True, True, False]
    # 2.0 points below ours exactly is accepted; 1.0 point is not (the choice is still recorded)
    assert _choose(dict(ours=(0.30, 0.05, 0.5), C90=(0.28, 0.05, 0.5), C97=(0.29, 0.05, 0.5),
                        C99=(0.29, 0.05, 0.5)))[:2] == ("C90", True)
    assert _choose(dict(ours=(0.21, 0.05, 0.5), C90=(0.20, 0.05, 0.5), C97=(0.25, 0.05, 0.5),
                        C99=(0.25, 0.05, 0.5)))[:2] == ("C90", False)
    # nothing eligible; a NaN rate is never eligible
    assert _choose(dict(ours=(0.30, 0.05, 0.5), C90=(0.1, 0.2, 0.5), C97=(0.1, 0.2, 0.5),
                        C99=(NAN, 0.05, 0.5)))[:2] == (None, False)
    # the mean is over the scenes: (0.10 + 0.30) / 2 = 0.20 vs ours 0.25: 5 points below
    recs = {"001": _rec("001", "val", "x", dict(ours=(0.25, 0.05, 0.5), C90=(0.10, 0.05, 0.5),
                                                 C97=(0.3, 0.05, 0.5), C99=(0.3, 0.05, 0.5))),
            "002": _rec("002", "val", "x", dict(ours=(0.25, 0.05, 0.5), C90=(0.30, 0.05, 0.5),
                                                 C97=(0.3, 0.05, 0.5), C99=(0.3, 0.05, 0.5)))}
    s = SEL.summary(recs)
    assert s["C90"]["stale_answer"]["mean"] == pytest.approx(0.20)
    assert SEL.choose(s)[:2] == ("C90", True)


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    for rel in SEL.CODE:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, repo / rel)
    pr = repo / "configs" / "cusum_reset_prereg.json"
    shutil.copy(PREREG, pr)
    g = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)
    g("init", "-q")
    g("-c", "user.email=t@t", "-c", "user.name=t", "add", *SEL.CODE)
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "code")
    return repo, pr, g


GOOD = dict(ours=(0.30, 0.05, 0.5), C90=(0.10, 0.12, 0.6), C97=(0.20, 0.08, 0.5), C99=(0.2005, 0.06, 0.7))
NOT_ACCEPTED = dict(ours=(0.21, 0.05, 0.5), C90=(0.20, 0.05, 0.5), C97=(0.25, 0.05, 0.5),
                    C99=(0.25, 0.05, 0.5))


def test_check_test_gates(tmp_path):
    repo, pr, g = _git_repo(tmp_path)
    sha = SEL.RS.sha256_file(pr)
    val = json.loads(pr.read_text())["scenes"]["validation"]
    _write(tmp_path / "val", [_rec(s, "val", sha, GOOD) for s in val])
    sel = tmp_path / "sel.json"
    ct = ["check-test", "--prereg", str(pr), "--selection", str(sel), "--dir",
          str(tmp_path / "val"), "--repo", str(repo)]
    with pytest.raises(SystemExit, match="not tracked.*no validation selection"):
        SEL.main(ct)
    g("-c", "user.email=t@t", "-c", "user.name=t", "add", "configs/cusum_reset_prereg.json")
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "prereg")
    out = SEL.main(["select", "--prereg", str(pr), "--dir", str(tmp_path / "val"), "--out", str(sel),
                    "--repo", str(repo)])
    assert (out["chosen"], out["accepted"]) == ("C99", True) and out["scenes"] == sorted(val)
    assert SEL.main(ct) == 0
    pr.write_text(pr.read_text() + " ")                         # dirty prereg
    with pytest.raises(SystemExit, match="uncommitted changes"):
        SEL.main(ct)
    g("checkout", "--", "configs/cusum_reset_prereg.json")
    s = json.loads(sel.read_text())
    sel.write_text(json.dumps(dict(s, chosen="C97")))
    with pytest.raises(SystemExit, match="re-selection gives C99 / True"):
        SEL.main(ct)
    sel.write_text(json.dumps(s))
    assert SEL.main(ct) == 0
    # a validation that accepts nothing stops the attempt before the test
    _write(tmp_path / "val2", [_rec(sc, "val", sha, NOT_ACCEPTED) for sc in val])
    sel2 = tmp_path / "sel2.json"
    out = SEL.main(["select", "--prereg", str(pr), "--dir", str(tmp_path / "val2"), "--out", str(sel2),
                    "--repo", str(repo)])
    assert (out["chosen"], out["accepted"]) == ("C90", False)
    with pytest.raises(SystemExit, match="did not accept a variant"):
        SEL.main(["check-test", "--prereg", str(pr), "--selection", str(sel2), "--dir",
                  str(tmp_path / "val2"), "--repo", str(repo)])
    with pytest.raises(SystemExit, match="accepted no variant"):
        SEL.main(["table", "--prereg", str(pr), "--selection", str(sel2), "--test-dir",
                  str(tmp_path / "val2"), "--out-dir", str(tmp_path / "t2")])
    # a record of another stage or a smoke record is refused (reset_fused_select.load_stage)
    recs = [_rec(sc, "val", sha, GOOD) for sc in val]
    recs[0]["smoke"] = 4
    _write(tmp_path / "val3", recs)
    with pytest.raises(SystemExit, match="smoke run"):
        SEL.main(["select", "--prereg", str(pr), "--dir", str(tmp_path / "val3"), "--out",
                  str(tmp_path / "s3.json")])


def test_table_and_adoption(tmp_path):
    pr = tmp_path / "prereg.json"
    shutil.copy(PREREG, pr)
    sha = SEL.RS.sha256_file(pr)
    prereg = json.loads(pr.read_text())
    _write(tmp_path / "val", [_rec(s, "val", sha, GOOD) for s in prereg["scenes"]["validation"]])
    sel = tmp_path / "sel.json"
    assert SEL.main(["select", "--prereg", str(pr), "--dir", str(tmp_path / "val"), "--out",
                     str(sel)])["chosen"] == "C99"
    test = prereg["scenes"]["test"]
    wall = {test[0], test[4]}

    def table(name, per_scene, fa=0.06, variants=None):
        recs = [_rec(sc, "test", sha, dict(ours=(0.30, 0.05, 0.5), C99=(per_scene(i), fa, 0.6)),
                     query="wall" if sc in wall else "chair", variants=variants)
                for i, sc in enumerate(test)]
        _write(tmp_path / name, recs)
        return SEL.main(["table", "--prereg", str(pr), "--selection", str(sel), "--test-dir",
                         str(tmp_path / name), "--out-dir", str(tmp_path / f"tab_{name}")])
    # 2.0+ points lower on the mean, worse in 2 of 10, FA within 2 x ours': adopted
    out = table("a", lambda i: 0.25 if i < 8 else 0.35)
    ad = out["adoption"]
    assert ad["adopt"] and ad["worse"] == 2 and ad["lower_by"] == pytest.approx(0.03) and ad["fa_ok"]
    md = (tmp_path / "tab_a" / "cusum_reset_test.md").read_text()
    assert "Verdict: ADOPT" in md and (tmp_path / "tab_a" / "cusum_reset_test.json").exists()
    # the object-query split: test[0] and test[4] ask for 'wall'
    sp = out["split"]
    assert (sp["all"]["n"], sp["object"]["n"], sp["stuff"]["n"]) == (10, 8, 2)
    assert sp["stuff"]["C99"] == pytest.approx(0.25) and sp["object"]["C99"] == pytest.approx(0.275)
    # worse in 3 scenes (mean exactly 2.0 points lower): not adopted
    ad = table("b", lambda i: 0.25 if i < 7 else 0.35)["adoption"]
    assert not ad["adopt"] and ad["worse"] == 3 and ad["lower_by"] == pytest.approx(0.02)
    # 2.0 points exactly, worse nowhere: adopted (the boundary is inclusive)
    assert table("c", lambda i: 0.28)["adoption"]["adopt"]
    # 1.0 point lower: not adopted
    assert not table("d", lambda i: 0.29)["adoption"]["adopt"]
    # FA above 2 x ours': not adopted
    ad = table("e", lambda i: 0.10, fa=0.11)["adoption"]
    assert not ad["adopt"] and not ad["fa_ok"]
    assert "NOT adopted" in (tmp_path / "tab_e" / "cusum_reset_test.md").read_text()
    # test records must run the accepted variant only
    with pytest.raises(SystemExit, match="accepted variant C99 only"):
        table("f", lambda i: 0.2, variants=["C90", "C99"])
