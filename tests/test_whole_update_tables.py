"""scripts/whole_update_tables.py (the attempt-07 rows of Tables III and IV), on fake records; CPU:
 * a new row is mean {+- population std} over the scenes in %, the "+depth" key reads the fused
   values, FA / Rec. are "--"; a scene without a value is refused;
 * Table IV tests each of our 2 fused rows against each of the 6 fused baselines at 2 levels
   (24 tests), with the sign test and Holm of plot_nor_qual.py over all 24;
 * Table IV marks its fused rows among themselves; an attempt that was not adopted is refused.
"""
import importlib.util
import json
import math
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCENES = [f"{i:03d}" for i in range(10)]
ROWS = ("ours", "agr", "ncc", "ncc_mean", "feat", "feat_mean", "normlift", "dispersion")


def _wut():
    spec = importlib.util.spec_from_file_location("whole_update_tables",
                                                  REPO / "scripts" / "whole_update_tables.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _recs():
    """Each row r has value base_r + 0.01 * scene in every metric; agr's fused G AP is below
    feat_mean's in scenes 0-6 and above in 7-9."""
    recs = {}
    for i, s in enumerate(SCENES):
        rows = {}
        for j, r in enumerate(ROWS):
            v = 0.2 + 0.05 * j + 0.01 * i
            rows[r] = {k: v for k in ("gauss_ap", "gauss_auroc", "pixel_ap", "pixel_auroc", "fused_gauss_ap",
                                      "fused_gauss_auroc", "fused_pixel_ap", "fused_pixel_auroc")}
        rows["agr"]["fused_gauss_ap"] = rows["feat_mean"]["fused_gauss_ap"] + (0.02 if i >= 7 else -0.02)
        recs[s] = dict(rows=rows)
    return recs


def test_new_cells_population_std_and_refusal():
    m, recs = _wut(), _recs()
    v = [recs[s]["rows"]["agr"]["gauss_ap"] for s in SCENES]
    want = f"{100 * np.mean(v):.1f}{{\\scriptsize$\\,\\pm\\,${100 * np.std(v):.1f}}}"   # ddof=0
    cells = m.new_cells(recs, SCENES, "agr")
    assert cells[0] == want and cells[4:] == ["--", "--"] and len(cells) == 6
    f = [recs[s]["rows"]["agr"]["fused_gauss_ap"] for s in SCENES]
    assert m.new_cells(recs, SCENES, "agr+depth")[0].startswith(f"{100 * np.mean(f):.1f}")
    recs["003"]["rows"]["agr"]["pixel_ap"] = float("nan")
    with pytest.raises(SystemExit, match="finite in 9 of 10"):
        m.new_cells(recs, SCENES, "agr")


def test_sign_tests_family_and_holm():
    m, recs = _wut(), _recs()
    tests = m.sign_tests(recs, SCENES)
    assert len(tests) == 24 and {(t["ours"], t["baseline"]) for t in tests} == \
        {(o, b) for o in m.OURS_FUSED for b in m.BASE_FUSED}
    t = next(t for t in tests if (t["ours"], t["baseline"], t["metric"]) == ("agr", "feat_mean", "fused_gauss_ap"))
    assert t["x_of_n"] == "3/10" and math.isclose(t["median_diff"], -0.02)
    pq = m.PQ
    assert math.isclose(t["sign_p"], pq.sign_test(t["diffs"])[2])
    assert [x["holm_p"] for x in tests] == pq.holm([x["sign_p"] for x in tests])
    # every ours row is below every baseline row of a higher index in all scenes: 0/10
    t = next(t for t in tests if (t["ours"], t["baseline"], t["metric"]) == ("ours", "dispersion", "fused_pixel_ap"))
    assert t["x_of_n"] == "0/10" and math.isclose(t["holm_p"], min(1.0, 24 * t["sign_p"]))


def test_tab4_marks_among_fused_rows():
    m, recs = _wut(), _recs()
    plain = {t3: ("F+D", [f"{10 + i}.0", f"{50 - i}.0", "1.0", "1.0", "--", "--"])
             for i, (_, t3, _) in enumerate(m.TAB4)}
    lines = m.tab4(plain, m.sign_tests(recs, SCENES))
    rows = [ln for ln in lines if not ln.startswith("%")]
    assert len(rows) == len(m.TAB4) + len(m.BASE_FUSED) * 2
    assert "\\textbf{17.0}" in rows[7] and "\\underline{16.0}" in rows[6]      # G AP: higher is better
    assert "\\textbf{50.0}" in rows[0] and "\\underline{49.0}" in rows[1]      # G AUROC
    # P AP and P AUROC are equal in every row: every row shares the best mark in both columns
    assert all(r.count("\\textbf{1.0}") == 2 for r in rows[:len(m.TAB4)])
    assert all(ln.count("&") == 9 for ln in rows[len(m.TAB4):])               # 10 columns of tests


def test_refuses_an_attempt_not_adopted(tmp_path):
    m = _wut()
    t = tmp_path / "whole_update_test.json"
    t.write_text(json.dumps(dict(chosen="agr", adoption=dict(adopt=False))))
    with pytest.raises(SystemExit, match="not adopted"):
        m.load(str(REPO / "configs" / "whole_update_prereg.json"), str(tmp_path), str(t))
