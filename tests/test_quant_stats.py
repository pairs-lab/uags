"""scripts/quant_stats.py and papers/figure_tools/plot_calibration.py (CPU statistics of the paper's
quantitative claims, configs/quant_prereg.json):
 * the exact sign test equals scipy.stats.binomtest (two-sided, p = 1/2) for every k <= n <= 25,
   and drops ties; Holm on a toy family (hand values, monotone, capped at 1);
 * the ECE recomputed from reliability bins equals usplat's expected_calibration_error on toy
   data, and the stored ECE for every row, scene and variant of both Table I test records;
 * the F1 guard refuses a prereg family that differs from the implemented one;
 * every LaTeX cell of every subcommand traces to a JSON key: the printed text equals the
   JSON value(s) at its keys rendered with its format, the .tex rows are exactly the JSON's
   rows, and a label cell holds no number (only method names and citations) — for fresh runs
   and for the saved outputs in docs/reports/2026-10-20-figures-video/logs/quant/;
 * independent check: key numbers equal those of the review's own scripts (quant_review.md,
   section 1; computed by other code on the same records);
 * plot_calibration.py writes a PDF without Type 3 fonts and its pooled ECE / NLL equal seg's.
The tests on the saved records skip when the results folder is absent.
"""
import json
import math
import os
import pathlib
import re
import sys

import numpy as np
import pytest
from scipy import stats

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "papers" / "figure_tools"))     # plot_calibration
sys.path.insert(0, str(REPO))
import quant_stats as QS                                                   # noqa: E402

RES = pathlib.Path(QS.RES)
SAVED = pathlib.Path(QS.OUT)
TEST_RECORDS = [RES / "tab1" / "eval_test.json", RES / "scannet_tab1" / "eval_test.json"]
HAVE_DATA = all(p.exists() for p in TEST_RECORDS) and (RES / "nor" / "tab2_relmax").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="saved result records not found")
SUBCOMMANDS = ("seg", "nor", "views", "cost", "betas", "v2grid")


# ------------------------------------------------------------------------------ statistics
def test_sign_test_equals_binomtest():
    for n in range(1, 26):
        for k in range(n + 1):
            d = [1.0] * k + [-2.5] * (n - k) + [0.0, 0.0]          # two ties, dropped
            st = QS.sign_test(d)
            assert st["n"] == n and st["k"] == k and st["ties"] == 2
            ref = stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue
            assert st["p"] == pytest.approx(ref, abs=1e-12), (k, n)
    assert math.isnan(QS.sign_test([0.0, 0.0])["p"])


def test_sign_test_smallest_p():
    # the smallest possible p for N = 6, 8, 10, 14 (all differences of one sign)
    for n, p in ((6, 0.03125), (8, 0.0078125), (10, 0.001953125), (14, 1.220703125e-4)):
        assert QS.sign_test([1.0] * n)["p"] == pytest.approx(p)


def test_holm_toy():
    assert QS.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert QS.holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])            # capped at 1
    adj = QS.holm([0.001, 0.2, 0.01, 0.03, 0.04])
    order = np.argsort([0.001, 0.2, 0.01, 0.03, 0.04])
    assert all(adj[order[i]] <= adj[order[i + 1]] for i in range(4))  # monotone
    assert adj == pytest.approx([0.005, 0.2, 0.04, 0.09, 0.09])
    with pytest.raises(SystemExit):
        QS.holm([0.01, float("nan")])


def test_summarize_counts_and_median():
    s = QS.summarize([1.0, -2.0, 0.0, 3.0, 0.5], boot=False)
    assert (s["n"], s["wins"], s["losses"], s["ties"]) == (5, 3, 1, 1)
    assert s["median_diff"] == 0.5 and s["mean_diff"] == pytest.approx(0.5)
    assert s["sign_p"] == pytest.approx(stats.binomtest(3, 4, 0.5).pvalue)


def test_fmt():
    assert QS.fmt(-0.001, "+.2f") == "0.00" and QS.fmt(0.004, "+.2f") == "0.00"
    assert QS.fmt(0.0183, "p") == "0.018" and QS.fmt(1.0, "p") == "1"
    assert QS.fmt(1.2e-6, "p") == "$1.2\\times10^{-6}$"
    assert QS.fmt(float("nan"), ".1f") == "--" and QS.fmt(None, ".1f") == "--"
    assert QS.fmt("ours_P1_c0.3", "tex") == "ours\\_P1\\_c0.3"


# ------------------------------------------------------------------------------ ECE from bins
def test_ece_from_bins_equals_usplat_toy():
    torch = pytest.importorskip("torch")
    from usplat.evaluation import expected_calibration_error
    from usplat.seg_eval import reliability
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        conf = torch.rand(5000, generator=g, dtype=torch.float64) ** 0.5
        conf[:7] = 1.0                                        # the top bin's closed edge
        correct = (torch.rand(5000, generator=g, dtype=torch.float64) < conf * 0.9).double()
        rel = reliability(conf, correct, 15)
        assert QS.bins_ece(rel) == pytest.approx(
            expected_calibration_error(conf, correct, 15), abs=1e-12)
    # pooled bins of two parts = the bins of the concatenation
    a, b = torch.rand(300, generator=g, dtype=torch.float64), torch.rand(500, generator=g,
                                                                          dtype=torch.float64)
    ya, yb = (a > 0.4).double(), (b > 0.6).double()
    pooled = QS.pool_bins([reliability(a, ya), reliability(b, yb)])
    assert QS.bins_ece(pooled) == pytest.approx(
        expected_calibration_error(torch.cat([a, b]), torch.cat([ya, yb]), 15), abs=1e-12)


@needs_data
@pytest.mark.parametrize("path", TEST_RECORDS, ids=["replica", "scannet"])
def test_ece_from_bins_equals_stored_every_row(path):
    m = QS.load_record(str(path))
    n = 0
    for s, sc in m["scenes"].items():
        for r, rr in sc["rows"].items():
            for var, v in rr.items():
                if isinstance(v, dict) and "reliability" in v:
                    assert QS.bins_ece(v["reliability"]) == pytest.approx(
                        QS.num(v["ece"]), abs=1e-12), (s, r, var)
                    assert sum(v["reliability"]["count"]) == int(v["n_pixels"])
                    n += 1
    assert n >= len(m["scenes"]) * 17                      # every row of every scene
    chk = QS.ece_check(m)
    assert chk["max_abs_err"] < 1e-12 and chk["n_checked"] == n


# ------------------------------------------------------------------------------ prereg guard
def test_prereg_f1_matches_and_guard_refuses_changes():
    p = json.load(open(REPO / "configs" / "quant_prereg.json"))
    f1 = QS.check_prereg_f1(p)
    assert len(f1["tests"]) == 20 and f1["adjustment"] == "Holm over these 20 tests"
    for change in ("drop", "swap", "ours"):
        q = json.loads(json.dumps(p))
        t = q["families"]["F1"]
        if change == "drop":
            t["tests"].pop()
        elif change == "swap":
            t["tests"][0], t["tests"][1] = t["tests"][1], t["tests"][0]
        else:
            t["ours_rows"]["ece"] = "ours_P0"
        with pytest.raises(SystemExit):
            QS.check_prereg_f1(q)


# ------------------------------------------------------------------------------ LaTeX trace
# label cells may hold digits only inside method names, citations and formulas
LABEL_OK = re.compile(r"\\cite\{[^}]*\}|3DGS|2D|P[0-4]|\$k=2\$|\$1-|1/\(1\+|1-max")


def trace(json_path, tex_path):
    """Every .tex row is the JSON's row, and every cell's text renders from its keys."""
    data = json.load(open(json_path))
    body = {k: v for k, v in data.items() if k != "latex"}
    lines = open(tex_path).read().splitlines()
    assert lines[0].startswith("%% generated by: python3 scripts/quant_stats.py ")
    assert lines[0] == f"%% generated by: {data['_command']}"
    printed = [ln for ln in lines if ln and not ln.startswith("%") and ln != "\\midrule"]
    rows = [r for b in data["latex"]["blocks"] for r in b["rows"] if "cells" in r]
    assert len(printed) == len(rows) and rows
    n_num = 0
    for ln, r in zip(printed, rows):
        assert ln == " & ".join(c["text"] for c in r["cells"]) + "\\\\"
        for c in r["cells"]:
            if c["keys"]:
                vals = [QS.get_key(body, k) for k in c["keys"]]
                assert QS.render(c["template"], vals, c["specs"]) == c["text"]
                n_num += 1
            else:
                assert not re.search(r"\d", LABEL_OK.sub("", c["text"])), c["text"]
    # no absolute path or user name in a printed row (anonymity)
    home = os.path.expanduser("~")
    assert not any("/home/" in ln or home in ln for ln in printed)
    return n_num


@needs_data
@pytest.mark.parametrize("cmd", SUBCOMMANDS)
def test_every_latex_cell_traces_fresh_run(cmd, tmp_path):
    js, tx = tmp_path / f"{cmd}.json", tmp_path / f"{cmd}.tex"
    QS.main([cmd, "--json", str(js), "--tex", str(tx)])
    assert trace(js, tx) > 0


@pytest.mark.parametrize("cmd", SUBCOMMANDS)
def test_every_latex_cell_traces_saved_output(cmd):
    js, tx = SAVED / f"{cmd}.json", SAVED / f"{cmd}.tex"
    if not (js.exists() and tx.exists()):
        pytest.skip("saved output not written yet")
    assert trace(js, tx) > 0


# ------------------------------------------------------------------------------ independent check
@pytest.fixture(scope="module")
def outs(tmp_path_factory):
    if not HAVE_DATA:
        pytest.skip("saved result records not found")
    d = tmp_path_factory.mktemp("quant")
    out = {}
    for cmd in SUBCOMMANDS:
        QS.main([cmd, "--json", str(d / f"{cmd}.json"), "--tex", str(d / f"{cmd}.tex")])
        out[cmd] = json.load(open(d / f"{cmd}.json"))
    return out


def test_seg_matches_review(outs):
    """quant_review.md 1a (Q/table1a.out, holm.out) and 1b (S/quant_extra.out)."""
    t = {(e["baseline"], e["metric"]): e for e in outs["seg"]["f1"]["tests"]}
    want = {  # (baseline, metric): (Replica x/6, ScanNet x/8, pooled x/14, median, sign p, Holm)
        ("averaging", "miou"): (3, 5, 8, 0.003, 0.7905, 1.0),
        ("averaging", "ece"): (3, 3, 6, -0.176, 0.7905, 1.0),
        ("averaging", "aurc"): (6, 0, 6, -0.089, 0.7905, 1.0),
        ("f3dgs", "ece"): (4, 7, 11, 0.809, 0.0574, 0.4590),
        ("f3dgs", "nll"): (6, 7, 13, 0.038, 0.0018, 0.0330),
        ("legs", "miou"): (6, 8, 14, 3.520, 0.0001, 0.0024),
        ("2d_features", "aurc"): (6, 6, 12, 2.012, 0.0129, 0.1423)}
    for k, (r, s, p, med, sp, h) in want.items():
        e = t[k]
        assert (e["replica"]["wins"], e["scannet"]["wins"], e["pooled"]["wins"]) == (r, s, p)
        assert e["pooled"]["median_diff"] == pytest.approx(med, abs=6e-4)
        assert e["pooled"]["sign_p"] == pytest.approx(sp, abs=6e-5)
        assert e["pooled"]["holm_p"] == pytest.approx(h, abs=6e-5)
    # Holm < 0.05: F3DGS NLL, LEGaussians x5, 2D features ECE/NLL/Brier (the review text says
    # "NLL ... survives, as do 9 other cells"; its own table1a.out has 9 in total)
    assert sum(e["pooled"]["significant"] for e in t.values()) == 9
    cal = outs["seg"]["calibration"]
    for d, r, em, ep, nll in (("replica", "ours_P2", 8.70, 7.12, 0.9818),
                              ("replica", "ours_P0", 9.86, 5.62, 0.9815),
                              ("replica", "averaging", 9.62, 5.70, 0.9888),
                              ("scannet", "ours_P2", 6.06, 2.62, 0.5056),
                              ("scannet", "ours_P0", 6.36, 3.16, 0.5018),
                              ("scannet", "averaging", 6.17, 3.13, 0.4993)):
        assert cal[d][r]["ece_scene_mean"] == pytest.approx(em, abs=0.006)
        assert cal[d][r]["ece_pooled"] == pytest.approx(ep, abs=0.006)
        assert cal[d][r]["nll_pooled"] == pytest.approx(nll, abs=6e-5)
    av = {e["label"]: e["pooled"]["wins"] for e in outs["seg"]["descriptive"]["averaging_vs_trained"]}
    assert av["averaging vs f3dgs nll"] == 13 and av["averaging vs legs ece"] == 13
    assert av["averaging vs legs aurc"] == 14


def test_nor_matches_review(outs):
    """quant_review.md 1c, 1d (Q/review_checks.out A, C)."""
    rs = outs["nor"]["r_static"]["test"]["rows"]
    want = {"ncc": (22.21, 32.15, 1.35, 9), "depth": (1.09, 2.63, 0.40, 3),
            "feat": (52.77, 70.98, 1.72, 10), "normlift": (4.05, 50.69, 0.90, 8),
            "dispersion": (8.31, 28.85, 0.91, 9)}
    for r, (med, pooled, dev, closer) in want.items():
        assert rs[r]["median"] == pytest.approx(med, abs=0.006)
        assert rs[r]["pooled"] == pytest.approx(pooled, abs=0.006)
        assert rs[r]["median_dev"] == pytest.approx(dev, abs=0.006)
        assert rs[r]["ours_closer"]["wins"] == closer
    assert rs["ours"]["median"] == pytest.approx(3.97, abs=0.006)
    assert rs["ours"]["within_2x"] == 1
    sh = outs["nor"]["table3"]["test"]["common_share"]
    assert (round(sh["min"], 1), round(sh["max"], 1)) == (3.0, 18.4)
    t3 = outs["nor"]["table3"]["test"]["cols"]
    assert t3["g_ap"]["fused"]["mean"] == pytest.approx(75.0, abs=0.05)
    assert t3["g_ap"]["normlift"]["mean"] == pytest.approx(63.8, abs=0.05)
    pc = {(e["metric"], e["first"], e["second"]): e for e in outs["nor"]["paired"]["comparisons"]}
    assert pc[("g_ap", "fused", "normlift")]["wins"] == 8
    assert pc[("g_ap", "fused", "normlift")]["sign_p"] == pytest.approx(0.109, abs=6e-4)
    assert pc[("p_ap", "ours", "normlift")]["wins"] == 1


def test_views_cost_betas_match_review(outs):
    """quant_review.md 1e, 1f, 1g (Q/scannet_fullmap.out, review_checks.out B, D;
    S/quant_extra.out 5)."""
    sc = {p["comparison"]: p for p in outs["views"]["scannet"]["points"]}
    assert (sc["u_ale vs u_epi"]["wins"], sc["u_ale vs u_epi"]["n"]) == (8, 8)
    assert sc["u_ale vs u_epi"]["mean_first"] == pytest.approx(0.586, abs=6e-4)
    assert sc["P2 1-max p vs averaging 1-max p"]["wins"] == 1
    assert sc["u_ale vs NormLift 1-C_inter"]["wins"] == 3
    rp = {(p["M"], p["target"], p["comparison"]): p for p in outs["views"]["replica"]["points"]}
    assert [rp[(m, "e_irr", "P2 1-max p vs averaging 1-max p")]["wins"]
            for m in (10, 30, 90, "full")] == [6, 5, 6, 6]
    assert rp[(10, "e_red", "u_epi vs u_ale")]["wins"] == 6
    assert rp[(90, "e_irr", "u_ale vs u_epi")]["wins"] == 6
    a = outs["cost"]["aggregate"]["all"]
    assert a["ours_s"]["median"] == pytest.approx(5.6, abs=0.05)
    assert a["lseg_s"]["median"] == pytest.approx(13.5, abs=0.05)
    assert a["f3dgs_s"]["median"] == pytest.approx(1268, abs=1)
    assert a["legs_s"]["median"] == pytest.approx(730, abs=1)
    b = {e["dataset"]: e["P2"] for e in outs["betas"]["datasets"]}
    assert b["ScanNet"]["gamma_e"] == pytest.approx(-0.80, abs=0.005)
    assert b["ScanNet"]["gamma_a"] == pytest.approx(-2.06, abs=0.005)
    assert b["Replica"]["gamma_a"] == pytest.approx(11.83, abs=0.005)
    assert all(e["test_record_uses_same_P2"] for e in outs["betas"]["datasets"])


def test_v2grid_reports_every_candidate(outs):
    for d, pre in (("replica", "tab1_v2_prereg.json"), ("scannet", "scannet_v2_prereg.json")):
        p = json.load(open(REPO / "configs" / pre))
        g = outs["v2grid"]["datasets"][d]
        rows = [e["row"] for e in g["rows"]]
        assert rows == QS.v2_order(p) and len(rows) == 18
        assert sorted(rows) == sorted(set(r for c in p["choices"].values() for r in c["rows"]
                                          if not r.startswith("@")) | set(p["ablations"]))
        assert [e["status"] for e in g["rows"]].count("selected") == 1
        assert g["chosen"]["@ours_v2"] == "ours_P2" and g["saved_selection_matches"]


# ------------------------------------------------------------------------------ figure
@needs_data
def test_plot_calibration(tmp_path, outs):
    import plot_calibration as PC
    pdf, js = tmp_path / "cal.pdf", tmp_path / "cal.json"
    PC.main(["--out", str(pdf), "--json", str(js)])
    assert pdf.exists() and (tmp_path / "cal.png").exists()
    assert b"/Type3" not in pdf.read_bytes()
    d = json.load(open(js))
    assert d["figure"]["no_type3_fonts"]
    for ds in ("replica", "scannet"):
        for key, r in d["rows"][ds].items():
            e, ref = d[ds]["rows"][r], outs["seg"]["calibration"][ds][r]
            assert e["ece_pooled"] == pytest.approx(ref["ece_pooled"], abs=1e-9)
            assert e["nll_pooled"] == pytest.approx(ref["nll_pooled"], abs=1e-9)
            assert e["ece_scene_mean"] == pytest.approx(ref["ece_scene_mean"], abs=1e-9)
            assert len(e["ece"]) == len(d[ds]["scenes"])               # every scene shown
        assert d["rows"][ds]["@legs"] == {"replica": "legs_l4", "scannet": "legs_l1"}[ds]
