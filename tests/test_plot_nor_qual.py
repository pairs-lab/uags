"""scripts/plot_nor_qual.py on the CPU.
Checks:
  * statistics: the sign test equals scipy's binomtest (ties dropped), Holm on a hand example,
    the population std of seg_eval.scene_stats
  * noise: the tolerance rule max(0.002, 3 x the largest Table III cell difference)
  * fused report: per-scene tables, F2 (8 tests, Holm), x/10, median differences, the claim
    verdict both ways, the LaTeX rows; a dump that did not reproduce Table III is refused
  * figures on the toy dump (tests/test_dump_nor_qual.py's scene, mock backend): N1, N2 and the
    teaser write a PDF and a PNG with TrueType fonts only (no Type 3) and no user name or home
    path; the N2 colour scale is one per row; captions quote the record's numbers
"""
import importlib.util
import json
import math
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PQ = _load("plot_nor_qual", REPO / "scripts" / "plot_nor_qual.py")


def test_sign_test_holm_and_std():
    from scipy.stats import binomtest
    d = [0.1, 0.2, -0.05, 0.0, 0.3, 0.1, 0.2, 0.0, 0.4, -0.1]
    pos, neg, p = PQ.sign_test(d)
    assert (pos, neg) == (6, 2) and p == pytest.approx(binomtest(6, 8, 0.5).pvalue)
    assert math.isnan(PQ.sign_test([0.0, 0.0])[2])
    adj = PQ.holm([0.01, 0.04, 0.03, 0.005])
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1 -> max(0.06, 0.04) = 0.06
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert PQ.holm([0.6, 0.9]) == pytest.approx([1.0, 1.0])
    ms = PQ.mean_std([1.0, 2.0, 3.0, float("nan")])
    assert ms["mean"] == 2.0 and ms["std"] == pytest.approx(np.std([1, 2, 3])) and ms["n"] == 3


def _record(ap=0.5, fa=0.1, frames=None):
    rows = {r: dict(ap=ap, auroc=0.8, common=dict(ap=ap, auroc=0.8)) for r in PQ.ROWS}
    return dict(scene="x", gaussian=dict(rows=rows), pixel=dict(rows=json.loads(json.dumps(rows)),
                                                                 frames=frames or []),
                operating_point={r: dict(fa=fa, rec=0.5) for r in PQ.TESTED},
                views=dict(mapping=[0, 1], update=[5, 6]), ours=dict(n_resets=3))


def test_noise_tolerance(tmp_path):
    fr = [dict(index=5, n_covered=10, n_pos=3, n_mask=4)]
    a, b = _record(frames=fr), _record(frames=fr)
    b["pixel"]["rows"]["ours"]["auroc"] += 0.0001
    (tmp_path / "a.json").write_text(json.dumps(a))
    (tmp_path / "b.json").write_text(json.dumps(dict(metrics=b)))      # runlog form
    r = PQ.noise(tmp_path / "a.json", tmp_path / "b.json")
    assert r["max_cell"] == "pixel/ours/auroc" and r["tolerance"] == 0.002 and r["frames_equal"]
    b["operating_point"]["ours"]["fa"] += 0.01
    (tmp_path / "b.json").write_text(json.dumps(b))
    r = PQ.noise(tmp_path / "a.json", tmp_path / "b.json")
    assert r["max_cell"] == "operating_point/ours/fa" and r["tolerance"] == pytest.approx(0.03)
    assert r["n_cells"] == 2 * 7 * 2 + 6 * 2


def _fake_dumps(tmp, scenes, gap, ok=True):
    """Dump folders with fused.json / meta.json: fused = 0.6 + gap[row] per scene."""
    recs = {}
    for i, s in enumerate(scenes):
        d = tmp / s
        d.mkdir()
        rows = {}
        for r in PQ.FUSED_ROWS:
            v = 0.6 + (0.0 if r == "fused" else -gap[r]) + 0.01 * i
            rows[r] = dict(ap=v, auroc=v + 0.1, common=dict(ap=v))
        (d / "fused.json").write_text(json.dumps(dict(
            gaussian=dict(rows=rows, common=dict(n=7)), pixel=dict(rows=rows))))
        (d / "meta.json").write_text(json.dumps(dict(status="done",
                                                     reproduction=dict(ok=ok))))
        rec = _record(ap=0.6 + 0.01 * i)
        (tmp / f"rec_{s}.json").write_text(json.dumps(rec))
        recs[s] = tmp / f"rec_{s}.json"
    return recs


def test_fused_report_and_verdict(tmp_path):
    test = list(PQ.TEST)
    gap = {"ncc+depth": 0.1, "feat+depth": 0.05, "normlift+depth": 0.02,
           "dispersion+depth": 0.2}
    recs = _fake_dumps(tmp_path, test, gap)
    pre = REPO / "configs" / "nor_fused_prereg.json"
    rep = PQ.fused_report(tmp_path, [], test, recs, pre)
    t = rep["test"]
    assert len(t["f2"]) == 8 and all(x["x_of_n"] == "10/10" for x in t["f2"])
    p10 = 2 * 0.5 ** 10
    assert all(x["sign_p"] == pytest.approx(p10) for x in t["f2"])
    assert all(x["holm_p"] == pytest.approx(min(1, 8 * p10)) for x in t["f2"])
    assert t["verdict"]["claim_may_stay"] and t["verdict"]["highest_mean_gaussian_ap"] == "fused"
    nl = next(x for x in t["f2"] if x["row"] == "normlift+depth" and x["metric"] == "gaussian_ap")
    assert nl["median_diff"] == pytest.approx(0.02)
    rows, stat = PQ.fused_latex(rep)
    assert rows.count("\\\\") == 5 and "NormLift reliability drop fused" in rows
    assert stat.count("\\\\") == 9
    md = PQ.fused_scene_md(rep)
    assert md.count("\n| 0") == 10
    # a fused baseline above ours+depth: the claim may not stay
    tmp2 = tmp_path / "b"
    tmp2.mkdir()
    gap["normlift+depth"] = -0.05
    recs = _fake_dumps(tmp2, test, gap)
    t = PQ.fused_report(tmp2, [], test, recs, pre)["test"]
    assert not t["verdict"]["claim_may_stay"]
    assert t["verdict"]["highest_mean_gaussian_ap"] == "normlift+depth"
    nl = next(x for x in t["f2"] if x["row"] == "normlift+depth" and x["metric"] == "gaussian_ap")
    assert nl["x_of_n"] == "0/10"
    # a dump that did not reproduce Table III is refused
    tmp3 = tmp_path / "c"
    tmp3.mkdir()
    recs = _fake_dumps(tmp3, test, gap, ok=False)
    with pytest.raises(SystemExit, match="did not reproduce"):
        PQ.fused_report(tmp3, [], test, recs, pre)


# ---------------------------------------------------------------- figures on the toy dump
@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    td = _load("toy_dump_test", REPO / "tests" / "test_dump_nor_qual.py")
    tmp = tmp_path_factory.mktemp("plot_nor_qual")
    t = td.toy_inputs(tmp)
    out = tmp / "dumps" / "005"
    td.DQ.main(td.argv(t, out, extra=["--curves"]))
    for s in ("013", "006"):
        shutil.copytree(out, tmp / "dumps" / s)
    rec = json.loads(t["rec"].read_text())
    recs = {}
    for s in PQ.TEST + PQ.VAL:
        recs[s] = tmp / f"rec_{s}.json"
        recs[s].write_text(json.dumps(rec))
    return dict(tmp=tmp, dumps=tmp / "dumps", recs=recs, rec=rec)


def _fonts_ok(pdf):
    out = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
    return "Type 3" not in out and "TrueType" in out


def test_change_and_query_figures(toy):
    scenes = ["005", "013", "006"]
    res = PQ.change_figure(toy["dumps"], scenes, toy["recs"], str(toy["tmp"] / "n1.pdf"))
    pdf, png = map(pathlib.Path, res["outputs"])
    assert pdf.exists() and png.exists() and _fonts_ok(pdf)
    assert PQ.pdf_check(str(pdf))["ok"]
    ap = 100 * toy["rec"]["pixel"]["rows"]["ours"]["ap"]
    assert res["info"]["005"]["pixel_ap"]["ours"] == pytest.approx(ap)
    cap = PQ.change_caption(res["info"], toy["recs"])
    fa = 100 * toy["rec"]["operating_point"]["ours"]["fa"]
    assert f"{fa:.1f}/" in cap and "geometry is fixed" in cap
    res = PQ.query_figure(toy["dumps"], scenes, toy["recs"], out=str(toy["tmp"] / "n2.pdf"))
    assert _fonts_ok(res["outputs"][0])
    for s in scenes:
        i = res["info"][s]
        assert i["query"] == "object" and i["vmin"] < i["vmax"]
    assert "never re-prompted" in PQ.query_caption(res["info"])


def test_teaser(toy):
    res = PQ.teaser_figure(toy["dumps"] / "005", toy["recs"]["002"], str(toy["tmp"] / "t.pdf"))
    assert _fonts_ok(res["outputs"][0]) and PQ.pdf_check(res["outputs"][0])["ok"]
    m = res["measured"]
    assert m["n_gt_pixels"] > 0 and 0 <= m["mean_p_ours_after"] <= 1
    cap = PQ.teaser_caption(res)
    rec = 100 * toy["rec"]["operating_point"]["ours"]["rec"]
    assert f"{rec:.1f}\\%" in cap and "'object'" in cap
    # caption words follow the measured values (C1-C3)
    r2 = json.loads(json.dumps(res))
    r2["measured"].update(mean_p_avg_after=0.2, mean_p_ours_after=0.1, mean_u_epi_after=0.0,
                          mean_u_epi_before=1.0)
    c2 = PQ.teaser_caption(r2)
    assert "still answers" not in c2 and "rises" not in c2
    r2["measured"].update(mean_p_avg_after=0.8, mean_u_epi_after=2.0)
    c3 = PQ.teaser_caption(r2)
    assert "still answers" in c3 and "ours does not" in c3 and "rises" in c3
