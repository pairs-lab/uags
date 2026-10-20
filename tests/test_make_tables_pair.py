"""scripts/make_tables.py tab1pair (Table I with two datasets side by side, means only):
 * each group's cells are tab1's printed means with tab1's marks (computed per dataset);
 * each dataset takes its row choices (LEGaussians lambda_u, CSS alpha0) from its own --val file;
 * CSS stays the unranked last row; files with different Table I rows are refused;
 * tab1's own output is unchanged by the shared helper (same rows as before the refactor).
"""
import importlib.util
import json
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_tables", REPO / "scripts" / "make_tables.py")
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)


def _st(mean, std=0.01, n=6):
    return dict(mean=mean, std=std, n=n, n_nan=0)


def _row(miou, acc, ece, nll, brier, aurc, ause=None, sig="u_ale", n=6):
    return {"all": dict(miou=_st(miou, n=n), acc=_st(acc, n=n), ece=_st(ece, n=n),
                        nll=_st(nll, n=n), brier=_st(brier, n=n), aurc=_st(aurc, n=n),
                        ause={sig: _st(ause, n=n)} if ause is not None else {})}


def _summary(shift=0.0, n=6, legs_best="legs_l1"):
    s = {
        "2d_features": _row(0.263 + shift, 0.749, 0.118, 1.12, 0.405, 0.163, n=n),
        "averaging": _row(0.279 + shift, 0.767, 0.096, 1.00, 0.366, 0.140, n=n),
        "normlift": {"all": dict(_row(0.279 + shift, 0.767, 0.096, 1.00, 0.366, 0.140, n=n)["all"],
                                 ause={"normlift_1-R": _st(0.145, n=n),
                                       "normlift_1-Cinter": _st(0.088, n=n)})},
        "f3dgs": _row(0.279 + shift, 0.770, 0.100, 1.04, 0.370, 0.146, n=n),
        "css": _row(0.285 + shift, 0.772, 0.051, 1.01, 0.351, 0.115, 0.112, "css_var_top", n=n),
        "css_a0.1": _row(0.290 + shift, 0.772, 0.050, 0.99, 0.351, 0.115, 0.112, "css_var_top", n=n),
        "ours_P0": _row(0.280 + shift, 0.768, 0.099, 1.00, 0.364, 0.138, 0.085, n=n),
        "ours_P1": _row(0.279 + shift, 0.767, 0.102, 1.04, 0.375, 0.129, 0.085, n=n),
        "ours_P2": _row(0.280 + shift, 0.768, 0.087, 1.00, 0.362, 0.122, 0.085, n=n),
    }
    for lam, u in (("legs_l1", 0.20), ("legs_l4", 0.19)):     # validation picks the lower AUSE
        s[lam] = _row(0.240 + shift, 0.749, 0.106, 1.14, 0.400, 0.158,
                      u if lam != legs_best else 0.10, "legs_u", n=n)
    return s


def _write(path, summary):
    path.write_text(json.dumps(dict(status="ok", metrics=dict(summary=summary))))
    return str(path)


def _cells(line):
    """Printed cells of a row with each \\multicolumn{k} expanded to k None entries."""
    out = []
    for c in line.rstrip("\\").split(" & ")[1:]:
        m = re.match(r"\\multicolumn\{(\d+)\}", c)
        out += [None] * int(m.group(1)) if m else [c]
    return out


def test_pair_matches_tab1_per_dataset(tmp_path):
    a_test = _write(tmp_path / "a_test.json", _summary())
    a_val = _write(tmp_path / "a_val.json", _summary(legs_best="legs_l4"))
    b_test = _write(tmp_path / "b_test.json", _summary(shift=0.4, n=8))
    b_val = _write(tmp_path / "b_val.json", _summary(shift=0.4, n=8, legs_best="legs_l1"))
    pair = mt.tab1pair([a_test, b_test], [a_val, b_val])
    singles = [mt.tab1(a_test, val=a_val), mt.tab1(b_test, val=b_val)]
    assert "chosen on val: {'@css': 'css_a0.1', '@legs': 'legs_l4'}" in pair    # dataset A
    assert "chosen on val: {'@css': 'css_a0.1', '@legs': 'legs_l1'}" in pair    # dataset B
    rows = [ln for ln in pair.splitlines() if " & " in ln]
    single_rows = [[ln for ln in t.splitlines() if " & " in ln] for t in singles]
    assert len(rows) == len(single_rows[0]) == len(single_rows[1]) == 10
    for k, line in enumerate(rows):
        c = _cells(line)
        assert len(c) == 14                                       # 7 columns per dataset
        for g, s in enumerate(single_rows):
            want = [None if x is None else re.sub(r"\{\\scriptsize.*", "", x)
                    for x in _cells(s[k])]                       # the mean with its mark
            assert c[7 * g: 7 * g + 7] == want, (line, s[k])
    assert "\\pm" not in pair                                     # means only
    css = rows[-1]
    assert css.startswith("CSS") and "\\textbf" not in css and "\\underline" not in css
    print("PASS tab1pair: per-dataset choices, tab1's means and marks per group, CSS unranked")


def test_pair_refuses_other_rows(tmp_path):
    s = _summary()
    a_test = _write(tmp_path / "a.json", s)
    s2 = dict(s)
    del s2["f3dgs"]
    b_test = _write(tmp_path / "b.json", s2)
    with pytest.raises(SystemExit, match="same Table I rows"):
        mt.tab1pair([a_test, b_test], [a_test, a_test])
    with pytest.raises(SystemExit, match="two test files"):
        mt.tab1pair([a_test], [a_test])
    print("PASS tab1pair refusals: different rows, one dataset only")


def _pc_write(path, iou, bump_shift=0.0):
    """A per-class eval file: iou = {scene: {class: value}}; rows differ by a fixed bump."""
    rows = ["2d_features", "averaging", "f3dgs", "legs_l1", "ours_P0", "css"]
    bump = {"2d_features": -0.1, "averaging": 0.0, "f3dgs": 0.01, "legs_l1": -0.2,
            "ours_P0": 0.05 + bump_shift, "css": 0.2}
    scenes = {s: dict(rows={r: {"all": {"iou": {c: v + bump[r] for c, v in d.items()}}}
                            for r in rows}) for s, d in iou.items()}
    summary = {r: {"all": dict(miou=dict(mean=0.3 + bump[r], std=0.01, n=len(iou), n_nan=0))}
               for r in rows}
    path.write_text(json.dumps(dict(status="ok", metrics=dict(scenes=scenes, summary=summary))))
    return str(path)


def _rows(tex):
    return [ln for ln in tex.splitlines() if " & " in ln and not ln.startswith("Method")]


def test_perclasspair_groups_colours_and_marks(tmp_path):
    a = _pc_write(tmp_path / "a.json", {"s0": {"wall": 0.8, "door": 0.5}, "s1": {"wall": 0.6, "door": 0.7}})
    b = _pc_write(tmp_path / "b.json", {"s0": {"bed": 0.4, "wall": 0.7}, "s1": {"bed": 0.6, "wall": 0.5},
                                        "s2": {"bed": 0.5}}, bump_shift=-0.06)
    pair = mt.perclasspair([a, b], [None, None], min_scenes=2)   # no --val: default choices
    single = [mt.perclass(a, min_scenes=2), mt.perclass(b, min_scenes=2)]
    rows, srows = _rows(pair), [_rows(t) for t in single]
    assert len(rows) == len(srows[0]) == len(srows[1]) == 6
    for k, line in enumerate(rows):                        # group cells = perclass cells
        cells = line.rstrip("\\").split(" & ")
        want = srows[0][k].rstrip("\\").split(" & ")
        want2 = srows[1][k].rstrip("\\").split(" & ")
        assert cells[0] == want[0] == want2[0]
        assert cells[1:] == want[1:] + want2[1:], (line, srows[0][k], srows[1][k])
    head = next(ln for ln in pair.splitlines() if ln.startswith("Method"))
    col = dict((c, x) for x, c in re.findall(r"\\cellcolor\[HTML\]\{(\w+)\}\\rotatebox\{90\}"
                                             r"\{\\textcolor\{white\}\{([^}]*)\}\}", head))
    pairs = re.findall(r"\\cellcolor\[HTML\]\{(\w+)\}\\rotatebox\{90\}\{\\textcolor\{white\}\{([^}]*)\}\}", head)
    assert [c for _, c in pairs] == ["door", "wall", "bed", "wall"] or \
        [c for _, c in pairs] == ["wall", "door", "bed", "wall"]
    assert len({x for x, c in pairs if c == "wall"}) == 1            # one colour per class
    assert col["bed"] not in {col["wall"], col["door"]}               # a new class: a new colour
    assert head.count("mIoU") == 2
    lines = pair.splitlines()
    css = [i for i, ln in enumerate(lines) if ln.startswith("CSS")][0]
    assert lines[css - 1] == "\\midrule" and "\\textbf" not in lines[css] \
        and "\\underline" not in lines[css]
    ours = next(ln for ln in rows if ln.startswith("Ours")).rstrip("\\").split(" & ")
    assert ours[3] == "\\textbf{35.0}"              # dataset A mIoU: ours best
    assert ours[-1] == "29.0"                          # dataset B: third (31.0, 30.0), no mark
    print("PASS perclasspair: perclass cells per group, one colour per class, marks per dataset")


def test_perclasspair_refuses_other_rows(tmp_path):
    a = _pc_write(tmp_path / "a.json", {"s0": {"wall": 0.8}, "s1": {"wall": 0.6}})
    d = json.loads(pathlib.Path(a).read_text())
    del d["metrics"]["summary"]["f3dgs"]
    b = tmp_path / "b.json"
    b.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="same rows"):
        mt.perclasspair([a, str(b)], [None, None], min_scenes=2)
    print("PASS perclasspair refusal: different rows")
