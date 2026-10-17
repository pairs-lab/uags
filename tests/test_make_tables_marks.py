"""scripts/make_tables.py best / second-best marks (bold / underline) on the printed means:
 * marks(): the best and second-best distinct values, ties share a mark, None is not ranked;
 * mark(): wraps the mean only, never the std;
 * Table I: --rank all (default) ranks every row, --rank ov only the open-vocabulary rows (CSS
   shown, not ranked); P2's mIoU / accuracy ("as P0") and NormLift's predictive ("same as averaging") are
   not values; lower is better except mIoU and accuracy;
 * Table III: no marks by default, --marks ranks all rows.
"""
import importlib.util
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_tables", REPO / "scripts" / "make_tables.py")
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)


def _st(mean, std=0.01, n=6):
    return dict(mean=mean, std=std, n=n, n_nan=0)


def _row(miou, acc, ece, nll, brier, aurc, ause=None, sig="u_ale"):
    return {"all": dict(miou=_st(miou), acc=_st(acc), ece=_st(ece), nll=_st(nll),
                        brier=_st(brier), aurc=_st(aurc),
                        ause={sig: _st(ause)} if ause is not None else {})}


def _write(tmp_path):
    summary = {
        "2d_features": _row(0.263, 0.749, 0.118, 1.12, 0.405, 0.163),
        "averaging": _row(0.279, 0.767, 0.096, 1.00, 0.366, 0.140),
        "normlift": {"all": dict(_row(0.279, 0.767, 0.096, 1.00, 0.366, 0.140)["all"],
                                 ause={"normlift_1-R": _st(0.145),
                                       "normlift_1-Cinter": _st(0.088)})},
        "f3dgs": _row(0.279, 0.770, 0.100, 1.04, 0.370, 0.146),
        "legs_l1": _row(0.240, 0.749, 0.106, 1.14, 0.400, 0.158, 0.197, "legs_u"),
        "css": _row(0.285, 0.772, 0.051, 1.01, 0.351, 0.115, 0.112, "css_var_top"),
        "ours_P0": _row(0.280, 0.768, 0.099, 1.00, 0.364, 0.138, 0.085),
        "ours_P1": _row(0.279, 0.767, 0.102, 1.04, 0.375, 0.129, 0.085),
        "ours_P2": _row(0.280, 0.768, 0.087, 1.00, 0.362, 0.122, 0.085),
    }
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(dict(status="ok", metrics=dict(summary=summary))))
    return str(p)


def _line(tex, start):
    return next(ln for ln in tex.splitlines() if ln.startswith(start))


def test_marks_and_mark():
    assert mt.marks([3.0, 1.0, 2.0, 1.0, None], higher_better=False) == \
        ["second" if v == 2.0 else "best" if v == 1.0 else None for v in [3.0, 1.0, 2.0, 1.0, None]]
    assert mt.marks([28.0, 27.9, 28.0, 24.0], higher_better=True) == \
        ["best", "second", "best", None]
    assert mt.marks([None, None], True) == [None, None] and mt.marks([5.0], False) == ["best"]
    c = "8.7{\\scriptsize$\\,\\pm\\,$5.6}"
    assert mt.mark(c, "best") == "\\textbf{8.7}{\\scriptsize$\\,\\pm\\,$5.6}"
    assert mt.mark(c, "second") == "\\underline{8.7}{\\scriptsize$\\,\\pm\\,$5.6}"
    assert mt.mark("--", "best") == "--" and mt.mark(c, None) == c
    print("PASS marks: ties share, None not ranked; mark wraps the mean only")


def test_tab1_open_vocabulary_ranking(tmp_path):
    p = _write(tmp_path)
    tex = mt.tab1(p, rank="ov")
    assert "% bold = best, underline = second best per column, among the open-vocabulary" in tex
    css = _line(tex, "CSS")
    assert "\\textbf" not in css and "\\underline" not in css          # shown, not ranked
    p0, p2 = _line(tex, "Ours P0"), _line(tex, "Ours P2")
    assert p0.split(" & ")[2].startswith("\\textbf{28.0}")             # mIoU, higher is better
    assert "\\multicolumn{2}{c}{as P0}" in p2                          # not a value
    cells = p2.split(" & ")
    assert cells[3].startswith("\\textbf{8.7}")                        # ECE, lower is better
    assert cells[4].startswith("\\textbf{1.00}") and cells[5].startswith("\\textbf{36.2}")
    assert cells[6].startswith("\\textbf{12.2}") and cells[7].startswith("\\textbf{8.5}")
    avg = _line(tex, "Averaging").split(" & ")
    assert avg[4].startswith("\\underline{9.6}") and avg[5].startswith("\\textbf{1.00}")  # tie
    f3 = _line(tex, "Feature 3DGS").split(" & ")
    assert f3[3].startswith("\\textbf{77.0}")                          # accuracy: F3DGS best
    assert _line(tex, "Ours P0").split(" & ")[3].startswith("\\underline{76.8}")
    nl = _line(tex, "NormLift\\reimpl, $1-C")
    assert "same predictive as averaging" in nl and "\\underline{8.8}" in nl
    print("PASS Table I --rank ov: CSS unranked, P2 'as P0' and NormLift text not values")


def test_tab1_rank_all(tmp_path):
    tex = mt.tab1(_write(tmp_path))                       # --rank all (default)
    css = _line(tex, "CSS").split(" & ")
    assert css[2].startswith("\\textbf{28.5}") and css[4].startswith("\\textbf{5.1}")
    assert _line(tex, "Ours P2").split(" & ")[3].startswith("\\underline{8.7}")
    print("PASS Table I --rank all (default) ranks CSS")


def test_tab3_marks_only_on_request(tmp_path):
    full = {"all": dict(miou=_st(0.280), ece=_st(0.087), nll=_st(1.00),
                        ause={"u_epi": _st(0.164), "u_ale": _st(0.085)})}
    var = {"all": dict(miou=_st(0.280), ece=_st(0.081), nll=_st(1.01),
                       ause={"u_epi": _st(0.165), "u_ale": _st(0.088)})}
    summary = {"ours_P2": full, "abl_pixels_P2": var}
    p = tmp_path / "t3.json"
    p.write_text(json.dumps(dict(status="ok", metrics=dict(summary=summary))))
    plain = mt.tab3(str(p))
    assert "\\textbf{28.0}" not in plain and "\\underline" not in plain
    marked = mt.tab3(str(p), marks_on=True)
    pix = _line(marked, "Pixels instead of views").split(" & ")
    assert pix[2].startswith("\\textbf{8.1}") and pix[3].startswith("\\underline{1.01}")
    print("PASS Table III: marks only with --marks")
