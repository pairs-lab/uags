"""
whole_update_tables.py — the rows that attempt 07 (configs/whole_update_prereg.json, adopted by its
registered rule on 2026-11-06) adds to Table III (tab:change) and Table IV (tab:fused) of
papers/main.tex, from the test records of scripts/whole_update.sh. CPU only.

  python3 scripts/whole_update_tables.py --prereg P --test-dir <out>/test \
      --table <out>/table/whole_update_test.json --tab3-record <Table III merge record> --out-dir D

  D/tab3_new_rows.tex  the new rows, unmarked: mean {+- population std} over the test scenes, in %
                       (FA and Rec. "--": these rows rank and never fire)
  D/tab3_body.tex      Table III: the old rows keep their printed cells (make_tables.py tab2 of the
                       Table III merge record; the fused baselines of FUSED_ROWS), the new rows
                       above; bold / underline over all rows by make_tables.mark_columns
  D/tab4_body.tex      Table IV: its fused rows (marked among themselves), then each of our two
                       fused rows against each fused baseline row at the Gaussian and the pixel
                       AP: scenes where ours is higher, median difference (points), exact
                       two-sided sign test (ties dropped), Holm over all the table's tests
  D/tab4_tests.json    the tests with their per-scene differences
  D/compare.md         per-scene counts that the text of Sec. IV-D cites

Refuses unless the adoption record says ADOPT for the variant every test record ran, and the
records pass the checks of scripts/reset_fused_select.load_stage (finished, test stage, not smoke,
this prereg, Table III reproduced, one code state).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import make_tables as MT                                                   # noqa: E402
import plot_nor_qual as PQ                                                 # noqa: E402
import reset_fused_select as RS                                            # noqa: E402

FUSED_ROWS = os.path.join(REPO, "docs", "reports", "2026-10-20-figures-video", "logs", "nor_fused",
                          "nor_fused_test_rows.tex")
COLS = (("gauss_ap", True), ("gauss_auroc", True), ("pixel_ap", True), ("pixel_auroc", True))
TAB3_HIGHER = [True, True, True, True, False, True]          # G AP, G AUROC, P AP, P AUROC, FA, Rec

# Table III, in order: (label, input, source, key); None = \midrule. Sources: "tab2" = the merge
# record's row (make_tables.py tab2), "fusedfile" = the row of FUSED_ROWS with that index, "new" =
# the attempt-07 records (key = record row; "+depth" = its fused values).
TAB3 = [
    ("NCC photometric residual", "I", "tab2", "ncc"),
    ("Depth residual", "D", "tab2", "depth"),
    ("Feature residual, $k=2$~\\cite{ackermann2025clsplats, jiang2026mif}\\reimpl", "F", "tab2", "feat"),
    ("NormLift reliability drop~\\cite{zang2026normlift}\\reimpl", "F", "tab2", "normlift"),
    ("Dispersion $1-\\hat\\rho$ (ours)", "F", "tab2", "dispersion"),
    None,
    ("NCC photometric residual, update mean", "I", "new", "ncc_mean"),
    ("Depth residual, update mean", "D", "new", "depth_mean"),
    ("Feature residual, update mean", "F", "new", "feat_mean"),
    None,
    ("Ours: test, reset, re-admission", "F+D", "tab2", "ours"),
    ("Ours: whole-update score $\\Delta\\bar r$", "F", "new", "agr"),
    None,
    ("Ours fused with depth residual", "F+D", "tab2", "fused"),
    ("Ours: whole-update score + depth residual", "F+D", "new", "agr+depth"),
    ("NCC residual + depth residual", "I+D", "fusedfile", 1),
    ("NCC residual, update mean, + depth residual", "I+D", "new", "ncc_mean+depth"),
    ("Feature residual, $k=2$, + depth residual", "F+D", "fusedfile", 2),
    ("Feature residual, update mean, + depth residual", "F+D", "new", "feat_mean+depth"),
    ("NormLift reliability drop + depth residual", "F+D", "fusedfile", 3),
    ("Dispersion $1-\\hat\\rho$ + depth residual", "F+D", "fusedfile", 4),
]
# Table IV: the fused rows (label, Table III label) and the tests. Record row of each fused row.
TAB4 = [("Ours fused with depth residual", "Ours fused with depth residual", "ours"),
        ("Ours: whole-update score + depth", "Ours: whole-update score + depth residual", "agr"),
        ("NCC photometric residual + depth", "NCC residual + depth residual", "ncc"),
        ("NCC residual, update mean, + depth", "NCC residual, update mean, + depth residual", "ncc_mean"),
        ("Feature residual, $k=2$, + depth", "Feature residual, $k=2$, + depth residual", "feat"),
        ("Feature residual, update mean, + depth", "Feature residual, update mean, + depth residual",
         "feat_mean"),
        ("NormLift reliability drop + depth", "NormLift reliability drop + depth residual", "normlift"),
        ("Dispersion $1-\\hat\\rho$ + depth", "Dispersion $1-\\hat\\rho$ + depth residual", "dispersion")]
OURS_FUSED = ("ours", "agr")
BASE_FUSED = ("ncc", "ncc_mean", "feat", "feat_mean", "normlift", "dispersion")
BASE_LABEL = {"ncc": "NCC photometric residual", "ncc_mean": "NCC residual, update mean",
              "feat": "Feature residual, $k=2$", "feat_mean": "Feature residual, update mean",
              "normlift": "NormLift reliability drop", "dispersion": "Dispersion $1-\\hat\\rho$"}
LEVELS = (("fused_gauss_ap", "Gaussian AP"), ("fused_pixel_ap", "Pixel AP"))


def unmark(s):
    return re.sub(r"\\(textbf|underline)\{([^{}]*)\}", r"\2", s)


def table_rows(text):
    """{label: [input, 6 cells]} of the non-comment '&' lines of a LaTeX row block."""
    out = {}
    for ln in text.splitlines():
        if "&" not in ln or ln.lstrip().startswith("%"):
            continue
        c = [x.strip() for x in ln.strip().rstrip("\\").split("&")]
        out[c[0]] = [c[1]] + [unmark(x) for x in c[2:]]
    return out


def fused_file_rows(path=FUSED_ROWS):
    rows = [ln for ln in open(path).read().splitlines()
            if "&" in ln and not ln.lstrip().startswith("%")]
    return [[x.strip() for x in ln.strip().rstrip("\\").split("&")] for ln in rows]


def load(prereg_path, test_dir, table_path):
    prereg = json.load(open(prereg_path))
    adoption = json.load(open(table_path))
    v = adoption.get("chosen")
    if not (adoption.get("adoption") or {}).get("adopt"):
        raise SystemExit(f"{table_path}: the variant was not adopted; only Appendix D changes")
    recs = RS.load_stage(prereg, prereg_path, test_dir, "test")
    if any(r.get("variants") != [v] for r in recs.values()):
        raise SystemExit(f"every test record must run the adopted variant {v} only")
    bad = [s for s, r in recs.items()
           if not all(r["capture_checks"].values()) or not all(r["residual_identity"].values())]
    if bad:
        raise SystemExit(f"capture or residual identity checks failed in {bad}")
    return prereg, v, recs, list(prereg["scenes"]["test"])


def per_scene(recs, scenes, row, key):
    return [RS.num(recs[s]["rows"][row].get(key)) for s in scenes]


def new_cells(recs, scenes, key):
    """6 printed cells (unmarked) of a new row: 'agr' or 'agr+depth' style keys."""
    row, fused = (key[:-6], True) if key.endswith("+depth") else (key, False)
    cells = []
    for c, _ in COLS:
        v = per_scene(recs, scenes, row, ("fused_" + c) if fused else c)
        st = PQ.mean_std(v)
        if st["n"] != len(scenes):
            raise SystemExit(f"{key} {c}: finite in {st['n']} of {len(scenes)} scenes")
        cells.append(MT.cell(st, 100.0, 1))
    return cells + ["--", "--"]


def tab3(recs, scenes, tab2_record):
    old = table_rows(MT.tab2(tab2_record))
    ff = fused_file_rows()
    tab2_label = {e[2]: e[0] for e in MT.TAB2 if e}
    out, table, new = [], [], []
    for e in TAB3:
        if e is None:
            out.append(None)
            continue
        label, inp, src, key = e
        if src == "tab2":
            got = old[tab2_label[key]]
            if got[0] != inp:
                raise SystemExit(f"{label}: input {got[0]} in the merge record, {inp} here")
            cells = got[1:]
        elif src == "fusedfile":
            if ff[key][1] != inp:
                raise SystemExit(f"{label}: input {ff[key][1]} in {FUSED_ROWS}")
            cells = ff[key][2:]
        else:
            cells = new_cells(recs, scenes, key)
            new.append(f"{label} & {inp} & " + " & ".join(cells) + "\\\\")
        out.append((label, inp))
        table.append(list(cells))
    # the 'Ours fused' row of FUSED_ROWS must equal the merge record's fused row
    if ff[0][2:] != table_rows(MT.tab2(tab2_record))[tab2_label["fused"]][1:]:
        raise SystemExit("the fused row of FUSED_ROWS differs from the Table III merge record")
    # unmarked copies (mark_columns writes into the row lists): {label: (input, 6 cells)}
    plain = {o[0]: (o[1], list(c)) for o, c in zip([x for x in out if x], table)}
    MT.mark_columns(table, [True] * len(table), TAB3_HIGHER)
    rows, lines = iter(table), []
    for item in out:
        if item is None:
            lines.append("\\midrule")
            continue
        lines.append(f"{item[0]} & {item[1]} & " + " & ".join(next(rows)) + "\\\\")
    return lines, new, plain


def sign_tests(recs, scenes):
    tests = []
    for o in OURS_FUSED:
        for b in BASE_FUSED:
            for key, level in LEVELS:
                a, c = per_scene(recs, scenes, o, key), per_scene(recs, scenes, b, key)
                d = [x - y for x, y in zip(a, c)]
                if not all(math.isfinite(x) for x in d):
                    raise SystemExit(f"{o} vs {b} {key}: a scene without a value")
                pos, neg, p = PQ.sign_test(d)
                tests.append(dict(ours=o, baseline=b, metric=key, level=level, wins=pos,
                                  losses=neg, x_of_n=f"{pos}/{len(d)}",
                                  median_diff=float(np.median(d)), sign_p=p, diffs=d))
    for t, h in zip(tests, PQ.holm([t["sign_p"] for t in tests])):
        t["holm_p"] = h
    return tests


def tab4(plain, tests):
    table = [list(plain[t3][1][:4]) for _, t3, _ in TAB4]       # G AP, G AUROC, P AP, P AUROC
    MT.mark_columns(table, [True] * len(table), [True] * 4)
    lines = [f"{lab} & {plain[t3][0]} & " + " & ".join(c) + "\\\\"
             for (lab, t3, _), c in zip(TAB4, table)]
    lines.append("% tests: each of our fused rows vs each fused baseline row, exact two-sided sign "
                 f"test (ties dropped), Holm over the {len(tests)} tests")
    by = {(t["ours"], t["baseline"], t["metric"]): t for t in tests}
    for b in BASE_FUSED:
        for key, level in LEVELS:
            cells = []
            for o in OURS_FUSED:
                t = by[(o, b, key)]
                cells += [t["x_of_n"], f"{100 * t['median_diff']:+.1f}", f"{t['sign_p']:.3f}",
                          f"{t['holm_p']:.3f}"]
            lines.append(f"{BASE_LABEL[b]} & {level} & " + " & ".join(cells) + "\\\\")
    return lines


def compare(recs, scenes, v):
    pairs = [(v, "ours", "gauss_ap"), (v, "ours", "gauss_auroc"), (v, "ours", "pixel_ap"),
             (v, "ours", "pixel_auroc"), (v, "normlift", "gauss_ap"), (v, "normlift", "pixel_ap"),
             (v, "feat_mean", "gauss_ap"), (v, "feat_mean", "pixel_ap"), (v, "feat", "gauss_ap"),
             ("feat_mean", "feat", "gauss_ap"), ("ncc_mean", "ncc", "gauss_ap"),
             ("depth_mean", "depth", "gauss_ap"), ("feat_mean", "normlift", "gauss_ap"),
             (v, "ours", "fused_gauss_ap"), (v, "feat", "fused_gauss_ap"),
             (v, "normlift", "fused_gauss_ap"), (v, "feat_mean", "fused_gauss_ap"),
             ("feat_mean", "feat", "fused_gauss_ap")]
    lines = ["| A | B | metric | A higher | median A-B (points) | mean A | mean B |",
             "|---|---|---|---|---|---|---|"]
    for a, b, m in pairs:
        x, y = per_scene(recs, scenes, a, m), per_scene(recs, scenes, b, m)
        d = [p - q for p, q in zip(x, y)]
        lines.append(f"| {a} | {b} | {m} | {sum(t > 0 for t in d)}/{len(d)} | "
                     f"{100 * float(np.median(d)):+.1f} | {100 * np.mean(x):.1f} | "
                     f"{100 * np.mean(y):.1f} |")
    return lines


def run(prereg, test_dir, table, tab2_record, out_dir):
    _, v, recs, scenes = load(prereg, test_dir, table)
    if v != "agr":
        raise SystemExit(f"the table layout is written for the variant agr, not {v}")
    t3, new, plain = tab3(recs, scenes, tab2_record)
    tests = sign_tests(recs, scenes)
    t4 = tab4(plain, tests)
    head = (f"% generated by scripts/whole_update_tables.py from {test_dir} (attempt 07, adopted; "
            f"test scenes {', '.join(scenes)}); mean {{\\scriptsize +- population std}}, in %")
    os.makedirs(out_dir, exist_ok=True)
    files = {"tab3_new_rows.tex": [head] + new, "tab3_body.tex": [head] + t3,
             "tab4_body.tex": [head] + t4, "compare.md": compare(recs, scenes, v)}
    for name, lines in files.items():
        with open(os.path.join(out_dir, name), "w") as f:
            f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "tab4_tests.json"), "w") as f:         # one test per line
        f.write(f'{{"scenes": {json.dumps(scenes)}, "family": {len(tests)}, "tests": [\n'
                + ",\n".join(json.dumps(t) for t in tests) + "\n]}\n")
    return dict(tab3=t3, new=new, tab4=t4, tests=tests)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--table", required=True, help="whole_update_select.py table output (json)")
    ap.add_argument("--tab3-record", required=True, help="the Table III merge record (nor_tab2.py)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    r = run(a.prereg, a.test_dir, a.table, a.tab3_record, a.out_dir)
    print("\n".join(r["tab3"] + [""] + r["tab4"]))


if __name__ == "__main__":
    main()
