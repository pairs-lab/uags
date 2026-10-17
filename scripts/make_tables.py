"""
make_tables.py — LaTeX rows of the paper's Table I (tab:replica) and of the Replica block of
Table III (tab:ablation) from an eval_seg.py JSON, so every number in the paper comes from a
saved evaluation file (no hand copying).

    python3 scripts/make_tables.py tab1 <eval_test.json> --val <eval_val.json> > rows.tex
    python3 scripts/make_tables.py tab3 <eval_test.json> --val <eval_val.json> > rows.tex

Rows and columns follow papers/main.tex (tab:replica): mean {\\scriptsize +- std} over the
scenes of the file (population std, as usplat.seg_eval.scene_stats), N printed in a comment.
mIoU, Acc., ECE, AURC and AUSE in %, Brier x100, NLL in nats. Variant "all" applies the
coverage rule (uncovered pixels count as wrong).
AUSE_u = AUSE of the row's own uncertainty against the 0/1 error:
    NormLift rows  normlift_1-R / normlift_1-Cinter      CSS  css_var_top (its Eq. 10)
    LEGaussians    legs_u (its learned uncertainty)      ours u_ale
    2D, averaging, Feature 3DGS: none ("--")
Marks: in every number column, bold (best) and underline (second best) on the printed
mean (not its std); rows with the same printed value share the mark. Table I ranks every row
(--rank all, the default; --rank ov ranks the open-vocabulary rows only and leaves CSS, which is
told the class list at mapping time, unmarked). Table III (an ablation: each row is
read against the full method, and most cells tie) has marks only with --marks.
Choices made on the VALIDATION file only (--val): the CSS prior read-out (css or
css_a<alpha0>) and the P3 temperature (ours_P3_b<beta_g>) by the lowest validation NLL, and
LEGaussians' lambda_u (legs_l<lambda>) by the lowest validation AUSE of its own u. Without
--val the defaults css, ours_P3_b20 and legs_l1 are used and a comment says so.
    python3 scripts/make_tables.py choose <eval_val.json> --key legs    prints the chosen row

Table III, Replica block (rows of usplat/ablation_rows.py; columns mIoU, ECE, NLL, AUSE epi,
AUSE ale): means over the scenes of the file (the caption says "mean over N": no std), N
printed in a comment. mIoU, ECE and AUSE in %, NLL in nats.
    Full method  the ours predictive (P0, P1 or P2) with the lowest validation NLL (--val;
                 without it ours_P2 and a comment says so). The comment also says when the
                 --val file fit its betas on the same scenes it scores (in-sample)
    variants     abl_<variant>_P2 if the full method is P2, else abl_<variant>_P0 (the
                 ablations have no P1 row): the same predictive as the full method
    AUSE epi/ale the row's own u_epi / u_ale against the row's own 0/1 error on the map built
                 from all mapping frames, with the variant's pixels ("all": the coverage rule,
                 uncovered pixels count as wrong and as the most uncertain). Fig. views scores
                 covered pixels only, so its M=180 point is not this cell.
Comment lines flag a cell whose mean covers another number of scenes than the full method's
mIoU, and a variant that prints the same cells as the full method. At the end, per scene:
each variant's sign against the full method in every column, and each row's share of pixels
with an undefined u_ale (AUSE ale ranks them as the most uncertain, so that column also
measures how much of the map has u_ale defined).
    --digits D   decimals of the ECE and AUSE cells (default 1; mIoU keeps 1, NLL 2)
    --tab1 F     compare the full method's printed cells with the same row of the Table I
                 file F: each run fits its own betas, so ECE and NLL can differ a little
    python3 scripts/make_tables.py tab3 <eval_test.json> --val <eval_val.json> --tab1 <tab1.json>
"""
from __future__ import annotations
import argparse
import json
import math
import re

# (label, row name or a choice key, own-uncertainty signal or None, predictive cells)
# predictive cells: "all" | "same" (NormLift: same predictive as averaging) | "asP0" (P2)
TAB1 = [
    ("2D features, per frame", "2d_features", None, "all"),
    ("Averaging~\\cite{joseph2025gradbp}\\reimpl", "averaging", None, "all"),
    ("NormLift~\\cite{zang2026normlift}\\reimpl, $1-\\mathcal{R}$", "normlift", "normlift_1-R", "same"),
    ("NormLift\\reimpl, $1-C_{\\mathrm{inter}}$", "normlift", "normlift_1-Cinter", "same"),
    ("Feature 3DGS~\\cite{zhou2024feature}\\reimpl", "f3dgs", None, "all"),
    ("LEGaussians~\\cite{shi2024legaussians}\\reimpl", "@legs", "legs_u", "all"),
    ("CSS (Dirichlet)~\\cite{wilson2025css}\\reimpl", "@css", "css_var_top", "all"),
    None,                                                     # \midrule
    ("Ours P0 (plug-in)", "ours_P0", "u_ale", "all"),
    ("Ours P1 (posterior mean)", "ours_P1", "u_ale", "all"),
    ("Ours P2 (adaptive $\\beta$)", "ours_P2", "u_ale", "asP0"),
]
# attempt 06 (P3, "@p3") stays out of Table I: its validation NLL did not beat P2's (the
# rule fixed before the run); scripts/make_tables.py choose --key p3 still reports its choice
COLS = [("miou", 100.0), ("acc", 100.0), ("ece", 100.0), ("nll", 1.0), ("brier", 100.0),
        ("aurc", 100.0)]
# family: (row prefix, default row, criterion on the validation summary; lower is better)
CHOICES = {"@css": ("css", "css", "nll"), "@p3": ("ours_P3_b", "ours_P3_b20", "nll"),
           "@legs": ("legs_l", "legs_l1", "ause:legs_u")}


def body(path):
    d = json.load(open(path))
    if d.get("status", "ok") != "ok":
        raise SystemExit(f"{path}: status {d.get('status')!r}, not a finished evaluation")
    b = d["metrics"] if isinstance(d.get("metrics"), dict) else d
    if "summary" not in b:
        raise SystemExit(f"{path}: no summary (was --scenes given?)")
    return b


def _crit(s, crit, variant):
    if crit.startswith("ause:"):
        a = s[variant]["ause"].get(crit[5:])
        return a["mean"] if a else float("nan")
    return s[variant][crit]["mean"]


def choose(val_summary, key, variant="all"):
    """The row of the family `key` with the lowest mean validation criterion (NLL for the
    CSS prior and the P3 temperature, AUSE of its own u for LEGaussians' lambda_u)."""
    prefix, default, crit = CHOICES[key]
    cand = {r: _crit(s, crit, variant) for r, s in val_summary.items()
            if r == prefix or r.startswith(prefix)}
    if key == "@css":
        cand = {r: v for r, v in cand.items() if r == "css" or r.startswith("css_a")}
    cand = {r: v for r, v in cand.items() if math.isfinite(v)}
    return min(cand, key=cand.get) if cand else default


def cell(stat, scale, digits=1, std=True):
    if stat is None or stat.get("n", 0) == 0 or not math.isfinite(stat.get("mean", float("nan"))):
        return "--"
    m, s, n = stat["mean"] * scale, stat.get("std", float("nan")) * scale, stat["n"]
    if not std or n < 2 or not math.isfinite(s):
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f}{{\\scriptsize$\\,\\pm\\,${s:.{digits}f}}}"


def _mean(text):
    """The printed mean of a cell ('28.0{\\scriptsize...}' -> 28.0); None for '--' or text."""
    m = re.match(r"-?\d+(?:\.\d+)?", text or "")
    return float(m.group(0)) if m else None


def marks(values, higher_better):
    """'best' / 'second' / None for each printed value (None = not ranked): the best and the
    second-best distinct values; equal printed values share their mark."""
    distinct = sorted({v for v in values if v is not None}, reverse=higher_better)
    best, second = (distinct + [None, None])[:2]
    return ["best" if v is not None and v == best else
            "second" if v is not None and v == second else None for v in values]


def mark(text, how):
    """\\textbf (best) or \\underline (second best) around the printed mean of a cell."""
    m = re.match(r"(-?\d+(?:\.\d+)?)(.*)", text or "", re.S)
    if how is None or not m:
        return text
    return ("\\textbf{%s}" if how == "best" else "\\underline{%s}") % m.group(1) + m.group(2)


def mark_columns(table, ranked, higher):
    """Mark a table in place: table[i][j] is a printed cell or None (not a value, e.g. inside a
    \\multicolumn); only rows with ranked[i] take part; higher[j]: larger is better."""
    for j, hb in enumerate(higher):
        vals = [_mean(row[j]) if ranked[i] and row[j] is not None else None
                for i, row in enumerate(table)]
        for i, how in enumerate(marks(vals, hb)):
            if how:
                table[i][j] = mark(table[i][j], how)


TAB1_HIGHER = [True, True, False, False, False, False, False]      # COLS + AUSE_u


def tab1(path, variant="all", val=None, rank="all"):
    summ = body(path)["summary"]
    vs = body(val)["summary"] if val else None
    picked = {k: (choose(vs, k, variant) if vs else CHOICES[k][1]) for k in CHOICES}
    ns = sorted({r[variant]["miou"]["n"] for r in summ.values()})
    lines = [f"% generated by scripts/make_tables.py tab1 {path} --variant {variant}"
             + (f" --val {val}" if val else " (no --val: default CSS prior and P3 beta_g)")
             + f" --rank {rank}; scenes per row N = {ns}; chosen on val: {picked}",
             "% bold = best, underline = second best per column, among "
             + ("the open-vocabulary rows (CSS is not ranked)" if rank == "ov" else "all rows")]
    out, table, ranked = [], [], []           # out: ("line", text) or ("row", label, ov, pred)
    for entry in TAB1:
        if entry is None:
            out.append(("line", "\\midrule"))
            continue
        label, row, sig, pred = entry
        row = picked.get(row, row)
        if row not in summ:
            out.append(("line", f"% {label}: row {row!r} not in {path}"))
            continue
        m = summ[row][variant]
        cells = [cell(m.get(k), sc, 2 if k == "nll" else 1) for k, sc in COLS]
        if pred == "same":                     # NormLift: averaging's predictive, not repeated
            cells = [None] * len(COLS)
        elif pred == "asP0":                   # P2: P0's labels, so P0's mIoU and accuracy
            cells[:2] = [None, None]
        cells.append(cell(m["ause"].get(sig), 100.0) if sig else "--")
        closed = row.startswith("css")
        out.append(("row", label, "$\\times$" if closed else "\\checkmark", pred))
        table.append(cells)
        ranked.append(rank == "all" or not closed)
    mark_columns(table, ranked, TAB1_HIGHER)
    rows = iter(table)
    for item in out:
        if item[0] == "line":
            lines.append(item[1])
            continue
        _, label, ov, pred = item
        cells = next(rows)
        if pred == "same":
            body_cells = ["\\multicolumn{6}{c}{same predictive as averaging}", cells[-1]]
        elif pred == "asP0":
            body_cells = ["\\multicolumn{2}{c}{as P0}"] + cells[2:]
        else:
            body_cells = cells
        lines.append(f"{label} & {ov} & " + " & ".join(body_cells) + "\\\\")
    return "\n".join(lines)


# Table III (tab:ablation), Replica block: (label, ablation variant; None = the full method)
TAB3 = [
    ("\\textbf{Full method}", None),
    ("Pixels instead of views", "pixels"),
    ("Uncorrected mean length $\\bar r$", "est_rbar"),
    ("Prior-shrunk agreement", "est_joint"),
    ("Shrunk mean length $\\tilde r$", "est_legacy"),
    ("Centered features", "centre"),
    ("Pixel-level variant", "legacy"),
]
# Table III columns: (name in comments, stat key, scale, decimals; None = --digits).
# "ause:<signal>" = the AUSE of that signal. mIoU is better when higher, the rest when lower.
TAB3_COLS = [("mIoU", "miou", 100.0, 1), ("ECE", "ece", 100.0, None), ("NLL", "nll", 1.0, 2),
             ("AUSE epi", "ause:u_epi", 100.0, None), ("AUSE ale", "ause:u_ale", 100.0, None)]
HIGHER_BETTER = {"miou"}
FULL = ("ours_P0", "ours_P1", "ours_P2")                         # the full method's candidates
PIXELS = {"all": "with the coverage rule of Table I (uncovered pixels count as wrong and as "
                 "the most uncertain); Fig. views scores covered pixels only, so its M=180 "
                 "point is not this cell",
          "covered": "on covered pixels only (as Fig. views)"}


def full_method(val_summary, variant="all"):
    """The full method's row, the ours predictive (P0, P1 or P2) with the lowest mean
    validation NLL, and the reason for the comment line (ours_P2 when nothing to choose)."""
    if val_summary is None:
        return "ours_P2", "no --val: default, not chosen on validation"
    nll = {r: val_summary[r][variant]["nll"]["mean"] for r in FULL if r in val_summary}
    ok = {r: v for r, v in nll.items() if math.isfinite(v)}
    if not ok:
        return "ours_P2", f"no finite validation NLL of {', '.join(FULL)}: default"
    return min(ok, key=ok.get), "lowest validation NLL among " + ", ".join(
        f"{r} {v:.4f}" for r, v in nll.items())


def fit_note(path):
    """'; in-sample ...' when the file's betas were fit on the same scenes it scores (its
    eval_seg.py run had --val-scenes equal to --scenes and no --betas), else ''."""
    c = json.load(open(path)).get("config") or {}
    if c.get("val_scenes") and c.get("val_scenes") == c.get("scenes") and not c.get("betas"):
        return "; in-sample: the betas were fit on these same scenes"
    return ""


def _stat(m, key):
    """The summary stat {mean, std, n, n_nan} of one Table III column, or None."""
    if key.startswith("ause:"):
        return (m.get("ause") or {}).get(key[5:])
    return m.get(key)


def tab3_cells(m, digits=1):
    """The printed cells of one Table III row: means only."""
    return [cell(_stat(m, k), sc, digits if dg is None else dg, std=False)
            for _, k, sc, dg in TAB3_COLS]


def cell_n(m):
    """(column name, number of scenes behind its mean) for each cell that prints a mean."""
    out = []
    for name, k, _, _ in TAB3_COLS:
        st = _stat(m, k)
        if st and st.get("n", 0) > 0:
            out.append((name, st["n"]))
    return out


def n_notes(label, m, n_full):
    """'% <label>: N = ...' for the cells whose mean covers another number of scenes than the
    full method's mIoU (a mean over other scenes is not comparable). The column names are
    left out when every cell of the row has that N."""
    cols = {}
    for name, n in cell_n(m):
        if n != n_full:
            cols.setdefault(n, []).append(name)
    return [f"% {label}: N = {n} scenes"
            + ("" if len(names) == len(TAB3_COLS) else " for " + ", ".join(names))
            + f", the full method has {n_full}" for n, names in sorted(cols.items())]


def tab1_note(full, m, t1_path, variant="all", digits=1):
    """A comment that compares the full method's printed cells with the same row of a Table I
    file. Each run fits its own betas on random pixel subsets, so ECE and NLL can differ in
    the last digit; mIoU and AUSE do not depend on beta and should agree."""
    t1 = body(t1_path)["summary"]
    if full not in t1:
        return f"% full method vs Table I: row {full!r} not in {t1_path}"
    here, there = tab3_cells(m, digits), tab3_cells(t1[full].get(variant) or {}, digits)
    diff = [f"{name} {x} here, {y} in Table I"
            for (name, *_), x, y in zip(TAB3_COLS, here, there) if x != y]
    return (f"% full method vs Table I ({t1_path}, row {full}): "
            + ("differs in " + "; ".join(diff) if diff else "same printed cells"))


def _scene_value(m, key):
    """One scene's value of a Table III column (an AUSE comes from its "ranking"), or None."""
    if key.startswith("ause:"):
        return ((m.get("ranking") or {}).get(key[5:]) or {}).get("ause")
    return m.get(key)


def _undefined(m):
    """One scene's share of pixels with an undefined u_ale (covered pixels; AUSE ale ranks
    them above every finite u_ale), or None."""
    u = (m.get("ranking") or {}).get("u_ale") or {}
    if "n_undefined" not in u or not m.get("n_pixels"):
        return None
    return u["n_undefined"] / m["n_pixels"]


def _sign(v, f, higher_better):
    """'+' better than the full method's value f, '-' worse, '=' the same (up to float
    noise, relative 1e-9: a re-mapped field can repeat a value in its last bits), '?' missing."""
    if v is None or f is None or not (math.isfinite(v) and math.isfinite(f)):
        return "?"
    if math.isclose(v, f, rel_tol=1e-9, abs_tol=1e-12):
        return "="
    return "+" if (v > f) == higher_better else "-"


def _row_signs(vals, full_vals):
    """'mIoU --, ECE -+, ...': for each column, one sign per scene against the full method."""
    out = []
    for name, k, _, _ in TAB3_COLS:
        signs = [_sign(_scene_value(v, k), _scene_value(f, k), k in HIGHER_BETTER)
                 for v, f in zip(vals, full_vals)]
        out.append(f"{name} {''.join(signs)}")
    return ", ".join(out)


def scene_notes(b, rows, full, variant="all"):
    """Comment lines for reading the table at small N. Per scene (in file order): whether each
    variant is better (+), worse (-) or the same (=) as the full method in each column, and
    each row's share of pixels with an undefined u_ale."""
    scenes = b.get("scenes") or {}
    if not scenes:
        return ["% no per-scene results in the file: no per-scene signs"]
    per = {r: [((scenes[s].get("rows") or {}).get(r) or {}).get(variant) or {} for s in scenes]
           for _, r in rows}
    lines = [f"% per scene ({', '.join(scenes)}): + better, - worse, = same value (relative"
             " 1e-9) as the full method (mIoU: higher is better, the rest: lower); u_ale"
             " undefined = % of the pixels, ranked as the most uncertain in AUSE ale"]
    for label, r in rows:
        share = " ".join("?" if x is None else f"{100 * x:.1f}" for x in map(_undefined, per[r]))
        signs = "" if r == full else _row_signs(per[r], per[full]) + "; "
        lines.append(f"% {label}: {signs}u_ale undefined {share}")
    return lines


def tab3(path, variant="all", val=None, tab1=None, digits=1, marks_on=False):
    b = body(path)
    summ = b["summary"]
    full, why = full_method(body(val)["summary"] if val else None, variant)
    if val:
        why += fit_note(val)
    pred = "P2" if full == "ours_P2" else "P0"          # the ablation rows are P0 and P2 only
    rows = [(label, full if v is None else f"abl_{v}_{pred}") for label, v in TAB3]
    have = [(label, r) for label, r in rows if r in summ]
    ns = sorted({n for _, r in have for _, n in cell_n(summ[r][variant])})
    fm = summ[full][variant] if full in summ else None
    n_full = (_stat(fm, "miou") or {}).get("n") if fm else None
    lines = [f"% generated by scripts/make_tables.py tab3 {path} --variant {variant}"
             + (f" --val {val}" if val else "") + (f" --tab1 {tab1}" if tab1 else "")
             + (f" --digits {digits}" if digits != 1 else "") + f"; scenes per cell N = {ns}",
             f"% full method = {full} ({why}); variants: abl_<variant>_{pred}"
             + (" (the ablations have no P1 row)" if full == "ours_P1" else ""),
             "% means over scenes; AUSE epi / ale = each row's own u_epi / u_ale against the"
             " row's own 0/1 error on the map from all mapping frames, " + PIXELS[variant]]
    if tab1 and fm:
        lines.append(tab1_note(full, fm, tab1, variant, digits))
    table = [tab3_cells(summ[r][variant], digits) for _, r in have]
    if marks_on:
        lines.append("% bold = best, underline = second best per column, among all rows")
        mark_columns(table, [True] * len(table),
                     [k in HIGHER_BETTER for _, k, _, _ in TAB3_COLS])
    marked = dict(zip((r for _, r in have), table))
    for label, r in rows:
        if r not in summ:
            lines.append(f"% {label}: row {r!r} not in {path}")
            continue
        m = summ[r][variant]
        if n_full:
            lines += n_notes(label, m, n_full)
        cells = tab3_cells(m, digits)
        if fm and r != full and cells == tab3_cells(fm, digits):
            lines.append(f"% {label}: same printed cells as the full method")
        lines.append(f"{label} & " + " & ".join(marked[r]) + "\\\\")
    if fm:
        lines += scene_notes(b, have, full, variant)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", choices=["tab1", "tab3", "choose"])
    ap.add_argument("--key", default="legs", choices=["css", "p3", "legs"])
    ap.add_argument("path")
    ap.add_argument("--val", default="", help="validation eval_seg JSON (row choices)")
    ap.add_argument("--variant", default="all", choices=["all", "covered"])
    ap.add_argument("--tab1", default="", help="tab3: Table I eval_seg JSON; a comment compares "
                    "the full method's printed cells with its row")
    ap.add_argument("--rank", default="all", choices=["all", "ov"], help="tab1: rows that get "
                    "the best / second-best marks (all rows, or ov: open-vocabulary rows only, "
                    "CSS shown but not ranked)")
    ap.add_argument("--marks", action="store_true", help="tab3: bold / underline the best and "
                    "second-best cells of each column too")
    ap.add_argument("--digits", type=int, default=1,
                    help="tab3: decimals of the ECE and AUSE cells (mIoU keeps 1, NLL 2)")
    a = ap.parse_args()
    if a.table == "choose":                      # the validation choice of one row family
        print(choose(body(a.path)["summary"], "@" + a.key, a.variant))
    elif a.table == "tab3":
        print(tab3(a.path, a.variant, a.val or None, a.tab1 or None, a.digits, a.marks))
    else:
        print(tab1(a.path, a.variant, a.val or None, a.rank))


if __name__ == "__main__":
    main()
