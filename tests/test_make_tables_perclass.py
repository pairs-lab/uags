"""scripts/make_tables.py perclass: per-class IoU table (tab:perclass).
 * columns = classes present in >= PERCLASS_MIN_SCENES scenes, ordered by presence then name;
 * cell = mean IoU over the scenes where the class is present; last column = Table I mIoU;
 * a class every row prints as 0.0 becomes a comment, not a column; 0.0 is never marked;
 * CSS is the last row, after a midrule, and never marked.
"""
import importlib.util
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_tables", REPO / "scripts" / "make_tables.py")
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)


def _write(tmp_path, min_scenes=2):
    rows = ["2d_features", "averaging", "f3dgs", "legs_l1", "ours_P0", "ours_P1", "css"]
    # wall in 3 scenes, door in 2, lamp in 1 (dropped at min 2), vent 0 everywhere (dropped)
    iou = {"s0": {"wall": 0.8, "door": 0.5, "vent": 0.0},
           "s1": {"wall": 0.6, "door": 0.7, "lamp": 0.3, "vent": 0.0},
           "s2": {"wall": 0.7}}
    bump = {"2d_features": -0.1, "averaging": 0.0, "f3dgs": 0.01, "legs_l1": -0.2,
            "ours_P0": 0.05, "ours_P1": 0.04, "css": 0.2}
    scenes = {s: dict(rows={r: {"all": {"iou": {c: (0.0 if c == "vent" else v + bump[r])
                                                for c, v in d.items()}}} for r in rows})
              for s, d in iou.items()}
    summary = {r: {"all": dict(miou=dict(mean=0.3 + bump[r], std=0.01, n=3, n_nan=0))}
               for r in rows}
    p = tmp_path / "e.json"
    p.write_text(json.dumps(dict(status="ok", metrics=dict(scenes=scenes, summary=summary))))
    return str(p)


def test_perclass_columns_cells_and_marks(tmp_path):
    tex = mt.perclass(_write(tmp_path), min_scenes=2)
    head = next(l for l in tex.splitlines() if l.startswith("Method"))
    assert head.index("wall") < head.index("door") and "lamp" not in head and "vent" not in head
    assert "% zero for every row (no column): vent" in tex
    ours = next(l for l in tex.splitlines() if l.startswith("Ours (P0")).split(" & ")
    # wall: mean of 0.85, 0.65, 0.75 = 75.0 (best of the ranked rows); door: mean of .55, .75
    assert ours[1] == "\\textbf{75.0}" and ours[2] == "\\textbf{65.0}"
    assert ours[3].startswith("\\textbf{35.0}")                       # mIoU column
    css_i = [i for i, l in enumerate(tex.splitlines()) if l.startswith("CSS")][0]
    lines = tex.splitlines()
    assert lines[css_i - 1] == "\\midrule" and "\\textbf" not in lines[css_i]
    print("PASS perclass: columns, means, marks, CSS reference")


def test_zero_is_never_marked(tmp_path):
    p = _write(tmp_path)
    d = json.loads(pathlib.Path(p).read_text())
    for s in d["metrics"]["scenes"].values():               # door 0 except legs 0.1 in s0
        for r, v in s["rows"].items():
            if "door" in v["all"]["iou"]:
                v["all"]["iou"]["door"] = 0.001 if r == "legs_l1" else 0.0
    pathlib.Path(p).write_text(json.dumps(d))
    tex = mt.perclass(p, min_scenes=2)
    assert "\\underline{0.0}" not in tex and "\\textbf{0.0}" not in tex
    print("PASS 0.0 is never marked")
