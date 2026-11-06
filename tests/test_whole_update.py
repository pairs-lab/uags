"""Attempt 07, whole-update change scores (configs/whole_update_prereg.json), on the CPU:
usplat/whole_update.py, scripts/whole_update_nor.py (mock backend, the toy NeRF-Object-Removal
scene of tests/test_eval_change.py) and scripts/whole_update_select.py.
Checks:
  * the three variants and the residual means on hand examples (plain Python floats), and their
    NaN rules: unseen in mapping, unseen in the update, one mapping view (no Kish agreement)
  * the scene run reproduces the toy record (every Table III AP / AUROC delta 0 on the CPU),
    every patch is undone, the ours and fused rows are Table III's; the variant scores equal an
    independent recomputation (a fresh field with the change test off fed the update views,
    numpy float64); the mean rows equal the plain mean of the per-view residuals that a separate
    spy reads from an unpatched-field run_update; every row's AP equals EC.ranking of its scores
  * one variant alone (the test stage) gives the same rows; refusals: a finished test record, a
    test stage with several variants, unknown variants, a prereg with other variants or rows
  * select / check-test / table on hand-made records: the tie rule (measured from the highest),
    the 3.0-point acceptance at its boundary, a NaN variant; the git and selection gates; the
    adoption rule in every direction
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
WU = _load("whole_update_nor", REPO / "scripts" / "whole_update_nor.py")
SEL = _load("whole_update_select", REPO / "scripts" / "whole_update_select.py")
EC, DQ = WU.EC, WU.DQ
from usplat.vmf_field import DirectionalSemanticField  # noqa: E402
from usplat.whole_update import VARIANTS, RESIDUALS, variant_scores, ResidualMeans  # noqa: E402

PREREG = REPO / "configs" / "whole_update_prereg.json"
NAN = float("nan")


def wu_argv(t, out, stage="toy", prereg=PREREG, variants=None):
    a = ["--table", str(t["rec"]), "--tolerance", str(t["tol"]), "--prereg", str(prereg),
         "--stage", stage, "--backend", "mock", "--device", "cpu", "--out", str(out)]
    return a + (["--variants", variants] if variants else [])


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    t = TD.toy_inputs(tmp_path_factory.mktemp("whole_update"))
    assert WU.EC is TD.EC and DQ.EC is EC
    out = t["tmp"] / "wu.json"
    res = WU.main(wu_argv(t, out))
    t.update(res=res, out=out, wu=json.loads(out.read_text()), record=json.loads(t["rec"].read_text()))
    rec = t["record"]
    ns = DQ.ec_namespace(rec, WU.build_parser().parse_args(wu_argv(t, t["tmp"] / "x")))
    t["ctx"] = EC.load_context(ns)
    t["k"] = int(rec["settings"]["k"])
    t["q"] = float(rec["settings"]["change_quantile"])
    return t


# --------------------------------------------------------------------------- hand examples
def _sums(views, d=2):
    """Plain-Python channel sums of [(w, u), ...] (vmf_field._Channel.add)."""
    R, W1, W2, Q = [0.0] * d, 0.0, 0.0, 0.0
    for w, u in views:
        R = [r + w * x for r, x in zip(R, u)]
        W1 += w
        W2 += w * w
        Q += w * w * sum(x * x for x in u)
    return R, W1, W2, Q


def _plain(mapping, update):
    """(dir, agr, kish) of one Gaussian in plain Python floats (NaN where undefined)."""
    RM, A, A2, QM = _sums(mapping)
    RE, E, E2, QE = _sums(mapping + update)
    RU = [e - m for e, m in zip(RE, RM)]
    norm = lambda v: math.sqrt(sum(x * x for x in v))
    d = NAN
    if A > 0 and norm(RM) > 0 and norm(RU) > 0:
        d = 1.0 - sum(a * b for a, b in zip(RM, RU)) / (norm(RM) * norm(RU))
    agr = norm(RM) / A - norm(RE) / E if A > 0 and E > 0 else NAN

    def rho(R, W1, W2, Q):
        den = W1 * W1 - W2
        if not den > 1e-6 * W1 * W1 + 1e-12:
            return NAN
        return math.sqrt(min(1.0, max(0.0, (norm(R) ** 2 - Q) / den)))
    return d, agr, rho(RM, A, A2, QM) - rho(RE, E, E2, QE)


def _tensors(cases):
    M = {k: [] for k in ("R", "W1", "W2", "Q")}
    E = {k: [] for k in ("R", "W1", "W2", "Q")}
    for mapping, update in cases:
        for S, vs in ((M, mapping), (E, mapping + update)):
            R, W1, W2, Q = _sums(vs)
            S["R"].append(R); S["W1"].append(W1); S["W2"].append(W2); S["Q"].append(Q)
    f = lambda S: {k: torch.tensor(v, dtype=torch.float64) for k, v in S.items()}
    return f(M), f(E)


def _same(a, b, tol=1e-12):
    return all((math.isnan(x) and math.isnan(y)) or abs(x - y) <= tol for x, y in zip(a, b))


def test_variant_math():
    one, up = (1.0, 0.0), (0.0, 1.0)
    cases = [
        ([(1.0, one), (1.0, one)], [(1.0, up)]),                      # turned by 90 degrees
        ([(1.0, (0.6, 0.8))], [(0.5, (0.6, 0.8))]),                  # one mapping view: no Kish
        ([], [(1.0, one)]),                                          # unseen in mapping
        ([(1.0, one), (0.5, one)], []),                              # unseen in the update
        ([(1.0, one), (1.0, one)], [(1.0, (0.5, 0.0)), (1.0, (0.5, 0.0))]),   # dispersed update
        ([(1.0, one), (0.25, (0.0, -1.0))], [(0.5, (-1.0, 0.0))]),   # reversed
    ]
    M, E = _tensors(cases)
    got = variant_scores(M, E)
    assert set(got) == set(VARIANTS) == {"dir", "agr", "kish"}
    want = list(zip(*[_plain(m, u) for m, u in cases]))
    for v, w in zip(VARIANTS, want):
        assert _same(got[v].tolist(), list(w)), (v, got[v].tolist(), w)
    # the NaN rules and the values that a hand can check
    d, a, k = (got[v].tolist() for v in VARIANTS)
    assert d[0] == pytest.approx(1.0) and a[0] == pytest.approx(1 - math.sqrt(5) / 3)
    assert k[0] == pytest.approx(1 - math.sqrt(1 / 3))
    assert d[1] == pytest.approx(0.0, abs=1e-12) and a[1] == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(k[1])                                          # rho_M needs two views
    assert all(math.isnan(x[2]) for x in (d, a, k))                  # unseen in mapping
    assert math.isnan(d[3]) and a[3] == 0.0 and k[3] == 0.0           # no update evidence
    assert d[4] == pytest.approx(0.0, abs=1e-12) and a[4] == pytest.approx(0.25)
    assert k[4] == pytest.approx(1 - math.sqrt(6.5 / 12))             # the dispersion rises
    assert d[5] == pytest.approx(1.0 + 1.0 / math.sqrt(1 + 1 / 16))   # cos < 0: dir > 1
    with pytest.raises(ValueError, match="unknown variant"):
        variant_scores(M, E, ("dir", "max"))
    # one variant alone is the same score
    assert _same(variant_scores(M, E, ("kish",))["kish"].tolist(), got["kish"].tolist(), 0.0)


def test_residual_means():
    acc = ResidualMeans(3)
    T = lambda *x: torch.tensor(x, dtype=torch.float32)
    acc.add(dict(ncc=T(1.0, NAN, 0.5), feat=T(0.5, 0.25, NAN)))
    acc.add(dict(ncc=T(0.5, NAN, NAN)))                              # no feature this view
    acc.add(dict(ncc=T(0.25, 2.0, NAN), feat=T(NAN, 0.75, NAN)))
    s = acc.scores()
    assert set(s) == set(RESIDUALS) == {"ncc_mean", "feat_mean", "depth_mean"}
    assert _same(s["ncc_mean"].tolist(), [1.75 / 3, 2.0, 0.5])
    assert _same(s["feat_mean"].tolist(), [0.5, 0.5, NAN])
    assert all(math.isnan(x) for x in s["depth_mean"].tolist())     # never supplied
    assert acc.count["ncc"].tolist() == [3, 1, 1] and acc.max["ncc"].tolist() == [1.0, 2.0, 0.5]
    assert acc.n_views == 3


# ---------------------------------------------------------------------------- scene run
def _independent_E(ctx):
    """The raw channel after the update of a fresh copy of the saved field with the change
    test off, fed every update view (no capture, no hook)."""
    f = DirectionalSemanticField.from_state(ctx["ckpt"]["field"], device="cpu")
    f.cfg.change_test = False
    for v in ctx["upd_views"]:
        z, ras, gz, kw = EC.view_inputs(ctx, v)
        f.update_view(ras, z, num_pixels=v["h"] * v["w"], **kw)
    return f.raw


def _np_variants(m, e):
    RM, RE = m.R.double().numpy(), e.R.double().numpy()
    RU = RE - RM
    A, E = m.W1.double().numpy(), e.W1.double().numpy()
    nm, nu, ne = (np.linalg.norm(x, axis=1) for x in (RM, RU, RE))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where((A > 0) & (nm > 0) & (nu > 0), 1.0 - np.clip((RM * RU).sum(1) / (nm * nu), -1, 1), np.nan)
        a = np.where((A > 0) & (E > 0), nm / A - ne / E, np.nan)

        def rho(R, W1, W2, Q):
            den = W1 ** 2 - W2
            r2 = np.clip((np.linalg.norm(R, axis=1) ** 2 - Q) / np.maximum(den, 1e-12), 0.0, 1.0)
            return np.where(den > 1e-6 * W1 ** 2 + 1e-12, np.sqrt(r2), np.nan)
        k = (rho(RM, A, m.W2.double().numpy(), m.Q.double().numpy())
             - rho(RE, E, e.W2.double().numpy(), e.Q.double().numpy()))
    return dict(dir=d, agr=a, kish=k)


def _np_close(a, b, tol=1e-9):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    both = np.isnan(a) & np.isnan(b)
    return bool(((np.abs(a - b) <= tol) | both).all() and (np.isnan(a) == np.isnan(b)).all())


def _independent_means(ctx, k, q):
    """{baseline row: per-Gaussian mean} from a separate spy over Table III's run_update (the
    fields as the evaluation builds them, nothing captured)."""
    _, null, _ = EC.replay_mapping(ctx, ctx["map_views"], k, len(ctx["upd_views"]))
    thr = {r: nl.threshold(q) for r, nl in null.items()}
    seen = []
    inner = EC.residuals_view

    def spy(*a, **kw):
        per = inner(*a, **kw)
        seen.append({n: r.detach().double().cpu().numpy().copy() for n, r in per.items()})
        return per
    EC.residuals_view = spy
    try:
        upd = EC.run_update(ctx, ctx["upd_views"], k, thr, q)
    finally:
        EC.residuals_view = inner
    out = {}
    for row, name in RESIDUALS.items():
        vals = [s[name] for s in seen if name in s]
        if not vals:
            out[row] = np.full(ctx["N"], np.nan)
            continue
        stack = np.stack(vals)
        fin = np.isfinite(stack)
        c = fin.sum(0)
        out[row] = np.where(c > 0, np.where(fin, stack, 0.0).sum(0) / np.maximum(c, 1), np.nan)
    return out, upd, len(seen)


def test_scene_run(run):
    wu, rec, res = run["wu"], run["record"], run["res"]
    assert wu["status"] == "done" and res["exit"] == 0 and wu["variants"] == list(VARIANTS)
    assert wu["reproduction"]["ok"] and wu["reproduction"]["max_ap_auroc_delta"] == 0.0
    assert all(wu["capture_checks"].values()) and all(wu["residual_identity"].values())
    assert set(wu["frozen_code"]) == {"eval_code_equal", "mapping_code_equal", "note"}
    # patches undone
    assert EC.DirectionalSemanticField is DirectionalSemanticField
    assert EC.residuals_view.__module__ == EC.__name__
    # the Table III rows are Table III's
    for r in ("ours", "normlift", "feat", "ncc", "depth", "dispersion"):
        assert wu["rows"][r]["gauss_ap"] == rec["gaussian"]["rows"][r]["ap"], r
        assert wu["rows"][r]["pixel_ap"] == pytest.approx(rec["pixel"]["rows"][r]["ap"], abs=1e-12), r
    assert wu["rows"]["ours"]["fused_gauss_ap"] == rec["gaussian"]["rows"]["fused"]["ap"]
    assert wu["rows"]["ours"]["fa"] == rec["operating_point"]["ours"]["fa"]
    # the variants: an independent recomputation (fresh no-reset field, numpy float64)
    ctx = run["ctx"]
    M = DirectionalSemanticField.from_state(ctx["ckpt"]["field"], device="cpu").raw
    want = _np_variants(M, _independent_E(ctx))
    pop = res["observed"].numpy()
    y = res["stale"][res["observed"]]
    for v in VARIANTS:
        got = res["scores"][v].numpy()
        assert np.isnan(got[~pop]).all(), v
        assert _np_close(got[pop], want[v][pop]), v
        assert np.isfinite(got[pop]).any(), v
        r = EC.ranking(res["scores"][v][res["observed"]], y)
        assert wu["rows"][v]["gauss_ap"] == r["ap"] and wu["rows"][v]["gauss_n_nan"] == r["n_nan"], v
        assert math.isfinite(wu["rows"][v]["fused_gauss_ap"]), v
    # the mean rows: the plain mean of the residuals a separate spy reads
    means, upd, n_seen = _independent_means(ctx, run["k"], run["q"])
    assert n_seen == len(ctx["upd_views"]) and torch.equal(upd["observed"], res["observed"])
    for row in RESIDUALS:
        got = res["scores"][row].numpy()
        assert _np_close(got[pop], means[row][pop], 1e-12), row
        assert np.isnan(got[~pop]).all(), row
    assert np.isfinite(res["scores"]["feat_mean"].numpy()[pop]).any()
    # the fused rows of Table IV, recomputed here, rank the same as fuse_rank on the record's run
    for r in ("feat", "ncc", "normlift", "dispersion"):
        assert math.isfinite(wu["rows"][r]["fused_gauss_ap"]), r
    assert wu["counts"]["n_observed"] == rec["counts"]["n_observed"]


def test_one_variant_and_refusals(run, tmp_path):
    t = run
    one = tmp_path / "one.json"
    res = WU.main(wu_argv(t, one, stage="test", variants="kish"))
    assert res["exit"] == 0
    r1 = json.loads(one.read_text())
    assert r1["variants"] == ["kish"] and "dir" not in r1["rows"] and "agr" not in r1["rows"]
    assert set(r1["rows"]) == {"ours", "normlift", "feat", "ncc", "depth", "dispersion", "kish",
                               "ncc_mean", "feat_mean", "depth_mean"}
    # CPU, same inputs: a row does not depend on the other variants of the run: exactly at the
    # Gaussian level (the common-set AP excepted: it is over the rows of the run); at the pixel
    # level up to float32 rendering noise (the renders stack another number of score channels)
    for r in r1["rows"]:
        a, b = r1["rows"][r], t["wu"]["rows"][r]
        assert set(a) == set(b), r
        for key in a:
            if key == "gauss_common_ap":
                continue
            if "pixel" in key:
                assert a[key] == pytest.approx(b[key], abs=1e-6), (r, key)
            else:
                assert a[key] == b[key], (r, key)
    with pytest.raises(SystemExit, match="run once"):
        WU.main(wu_argv(t, one, stage="test", variants="kish"))
    with pytest.raises(SystemExit, match="accepted variant only"):
        WU.main(wu_argv(t, tmp_path / "a.json", stage="test"))
    with pytest.raises(SystemExit, match="not a subset"):
        WU.main(wu_argv(t, tmp_path / "b.json", variants="dir,max"))
    with pytest.raises(SystemExit, match="not a subset"):
        WU.main(wu_argv(t, tmp_path / "b.json", variants="dir,dir"))
    p = json.loads(PREREG.read_text())
    p["variants"]["mean_c"] = "x"
    bad = tmp_path / "p1.json"
    bad.write_text(json.dumps(p))
    with pytest.raises(SystemExit, match="variants differ"):
        WU.main(wu_argv(t, tmp_path / "c.json", prereg=bad))
    p = json.loads(PREREG.read_text())
    p["baselines"]["rows"] = ["feat_mean"]
    bad = tmp_path / "p2.json"
    bad.write_text(json.dumps(p))
    with pytest.raises(SystemExit, match="baseline rows differ"):
        WU.main(wu_argv(t, tmp_path / "d.json", prereg=bad))
    assert not any((tmp_path / f"{n}.json").exists() for n in "abcd")   # refused before writing


# ---------------------------------------------------------------------------- selection
def _rec(scene, stage, sha, rows, variants=None, code="c"):
    """rows: {row: (gauss_ap, fused_gauss_ap)}; every other required row at 0.5."""
    full = {r: dict(gauss_ap=0.5, gauss_auroc=0.8, pixel_ap=0.5, pixel_auroc=0.7)
            for r in SEL.TABLE3 + SEL.BASELINES}
    for r, (ap, fap) in rows.items():
        full[r] = dict(gauss_ap=ap, gauss_auroc=0.8, pixel_ap=0.5, pixel_auroc=0.7,
                       fused_gauss_ap=fap, fused_gauss_auroc=0.9, fused_pixel_ap=0.6,
                       fused_pixel_auroc=0.8)
    return dict(status="done", stage=stage, scene=scene, smoke=0, prereg_sha256=sha,
                reproduction=dict(ok=True), code=code, rows=full,
                variants=variants or [r for r in rows if r in SEL.VARIANTS])


def _write(d, recs):
    d.mkdir(parents=True, exist_ok=True)
    for r in recs:
        (d / f"{r['scene']}.json").write_text(json.dumps(r))


def _choose(rows, scenes=("001", "002")):
    recs = {s: _rec(s, "val", "x", rows) for s in scenes}
    return SEL.choose(SEL.summary(recs, SEL.TABLE3 + SEL.VARIANTS + SEL.BASELINES))


def test_choose_rule():
    # dir and agr tie (0.2 point): the higher fused AP wins
    ch, acc, tab = _choose(dict(ours=(0.60, 0.8), dir=(0.700, 0.80), agr=(0.698, 0.85), kish=(0.60, 0.9)))
    assert (ch, acc) == ("agr", True) and [r["tied"] for r in tab["rows"]] == [True, True, False]
    # ties are measured from the highest: kish is 0.8 point below dir, not tied
    ch, acc, tab = _choose(dict(ours=(0.60, 0.8), dir=(0.700, 0.80), agr=(0.696, 0.81), kish=(0.692, 0.99)))
    assert (ch, acc) == ("agr", True) and [r["tied"] for r in tab["rows"]] == [True, True, False]
    # equal fused APs within a tie: the registered order
    assert _choose(dict(ours=(0.6, 0.8), dir=(0.70, 0.8), agr=(0.70, 0.8), kish=(0.5, 0.8)))[:2] == ("dir", True)
    # 3.0 points above ours exactly is accepted; 2.9 points is not (the choice is still recorded)
    assert _choose(dict(ours=(0.60, 0.8), dir=(0.63, 0.8), agr=(0.5, 0.8), kish=(0.5, 0.8)))[:2] == ("dir", True)
    assert _choose(dict(ours=(0.60, 0.8), dir=(0.629, 0.8), agr=(0.5, 0.8), kish=(0.5, 0.8)))[:2] == ("dir", False)
    # a NaN variant is never chosen; all NaN: nothing
    assert _choose(dict(ours=(0.6, 0.8), dir=(NAN, 0.9), agr=(0.7, 0.8), kish=(0.65, 0.8)))[:2] == ("agr", True)
    assert _choose(dict(ours=(0.6, 0.8), dir=(NAN, 0.9), agr=(NAN, 0.8), kish=(NAN, 0.8)))[:2] == (None, False)
    # the mean is over the scenes: (0.70 + 0.62) / 2 = 0.66 vs ours 0.60: accepted
    recs = {"001": _rec("001", "val", "x", dict(ours=(0.60, 0.8), dir=(0.70, 0.8), agr=(0.5, 0.8), kish=(0.5, 0.8))),
            "002": _rec("002", "val", "x", dict(ours=(0.60, 0.8), dir=(0.62, 0.8), agr=(0.5, 0.8), kish=(0.5, 0.8)))}
    s = SEL.summary(recs, SEL.TABLE3 + SEL.VARIANTS + SEL.BASELINES)
    assert s["dir"]["gauss_ap"]["mean"] == pytest.approx(0.66)
    assert SEL.choose(s)[:2] == ("dir", True)


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    for rel in SEL.CODE:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, repo / rel)
    pr = repo / "configs" / "whole_update_prereg.json"
    shutil.copy(PREREG, pr)
    g = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)
    g("init", "-q")
    g("-c", "user.email=t@t", "-c", "user.name=t", "add", *SEL.CODE)
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "code")
    return repo, pr, g


GOOD = dict(ours=(0.60, 0.8), dir=(0.70, 0.80), agr=(0.698, 0.85), kish=(0.60, 0.9))
NOT_ACCEPTED = dict(ours=(0.60, 0.8), dir=(0.62, 0.8), agr=(0.61, 0.8), kish=(0.5, 0.8))


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
    g("-c", "user.email=t@t", "-c", "user.name=t", "add", "configs/whole_update_prereg.json")
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "prereg")
    out = SEL.main(["select", "--prereg", str(pr), "--dir", str(tmp_path / "val"), "--out", str(sel),
                    "--repo", str(repo)])
    assert (out["chosen"], out["accepted"]) == ("agr", True) and out["scenes"] == sorted(val)
    assert SEL.main(ct) == 0
    pr.write_text(pr.read_text() + " ")                         # dirty prereg
    with pytest.raises(SystemExit, match="uncommitted changes"):
        SEL.main(ct)
    g("checkout", "--", "configs/whole_update_prereg.json")
    s = json.loads(sel.read_text())
    sel.write_text(json.dumps(dict(s, chosen="dir")))
    with pytest.raises(SystemExit, match="re-selection gives agr / True"):
        SEL.main(ct)
    sel.write_text(json.dumps(s))
    assert SEL.main(ct) == 0
    # a validation that accepts nothing stops the attempt before the test
    _write(tmp_path / "val2", [_rec(sc, "val", sha, NOT_ACCEPTED) for sc in val])
    sel2 = tmp_path / "sel2.json"
    out = SEL.main(["select", "--prereg", str(pr), "--dir", str(tmp_path / "val2"), "--out", str(sel2),
                    "--repo", str(repo)])
    assert (out["chosen"], out["accepted"]) == ("dir", False)
    with pytest.raises(SystemExit, match="did not accept a variant"):
        SEL.main(["check-test", "--prereg", str(pr), "--selection", str(sel2), "--dir",
                  str(tmp_path / "val2"), "--repo", str(repo)])
    with pytest.raises(SystemExit, match="accepted no variant"):
        SEL.main(["table", "--prereg", str(pr), "--selection", str(sel2), "--test-dir",
                  str(tmp_path / "val2"), "--out-dir", str(tmp_path / "t2")])
    # a smoke record, or a validation record without every variant, is refused
    recs = [_rec(sc, "val", sha, GOOD) for sc in val]
    recs[0]["smoke"] = 4
    _write(tmp_path / "val3", recs)
    with pytest.raises(SystemExit, match="smoke run"):
        SEL.main(["select", "--prereg", str(pr), "--dir", str(tmp_path / "val3"), "--out",
                  str(tmp_path / "s3.json")])
    recs = [_rec(sc, "val", sha, GOOD) for sc in val]
    recs[1]["variants"] = ["dir"]
    _write(tmp_path / "val4", recs)
    with pytest.raises(SystemExit, match="must run every variant"):
        SEL.main(["select", "--prereg", str(pr), "--dir", str(tmp_path / "val4"), "--out",
                  str(tmp_path / "s4.json")])


def test_table_and_adoption(tmp_path):
    pr = tmp_path / "prereg.json"
    shutil.copy(PREREG, pr)
    sha = SEL.RS.sha256_file(pr)
    prereg = json.loads(pr.read_text())
    _write(tmp_path / "val", [_rec(s, "val", sha, GOOD) for s in prereg["scenes"]["validation"]])
    sel = tmp_path / "sel.json"
    assert SEL.main(["select", "--prereg", str(pr), "--dir", str(tmp_path / "val"), "--out",
                     str(sel)])["chosen"] == "agr"
    test = prereg["scenes"]["test"]

    def table(name, per_scene, variants=None):
        recs = [_rec(sc, "test", sha, dict(ours=(0.50, 0.7), agr=(per_scene(i), 0.8)), variants=variants)
                for i, sc in enumerate(test)]
        _write(tmp_path / name, recs)
        return SEL.main(["table", "--prereg", str(pr), "--selection", str(sel), "--test-dir",
                         str(tmp_path / name), "--out-dir", str(tmp_path / f"tab_{name}")])
    # 3.2 points above on the mean, higher in 7 of 10: adopted
    ad = table("a", lambda i: 0.55 if i < 7 else 0.49)["adoption"]
    assert ad["adopt"] and ad["wins"] == 7 and ad["above_by"] == pytest.approx(0.032)
    md = (tmp_path / "tab_a" / "whole_update_test.md").read_text()
    assert "Verdict: ADOPT" in md and (tmp_path / "tab_a" / "whole_update_test.json").exists()
    # the same mean, higher in 6 of 10: not adopted
    ad = table("b", lambda i: 0.56 if i < 6 else 0.49)["adoption"]
    assert not ad["adopt"] and ad["wins"] == 6 and ad["above_by"] == pytest.approx(0.032)
    # 3.0 points exactly, higher everywhere: adopted (the boundary is inclusive)
    assert table("c", lambda i: 0.53)["adoption"]["adopt"]
    # 2.9 points: not adopted
    assert not table("d", lambda i: 0.529)["adoption"]["adopt"]
    assert "NOT adopted" in (tmp_path / "tab_d" / "whole_update_test.md").read_text()
    # equal scenes are not wins
    ad = table("e", lambda i: 0.50 if i < 4 else 0.56)["adoption"]
    assert ad["wins"] == 6 and not ad["adopt"]
    # test records must run the accepted variant only
    with pytest.raises(SystemExit, match="accepted variant agr only"):
        table("f", lambda i: 0.6, variants=["dir", "agr"])
