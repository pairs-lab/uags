"""
make_tables.py — LaTeX rows of the paper's Table I (tab:replica) and of the Replica block of
Table III (tab:ablation) from an eval_seg.py JSON, and of Table II (tab:change) from a
scripts/nor_tab2.py merge, so every number in the paper comes from a saved evaluation file
(no hand copying).

    python3 scripts/make_tables.py tab1 <eval_test.json> --val <eval_val.json> > rows.tex
    python3 scripts/make_tables.py tab3 <eval_test.json> --val <eval_val.json> > rows.tex

Rows and columns follow papers/main.tex (tab:replica): mean {\\scriptsize +- std} over the
scenes of the file (population std, as usplat.seg_eval.scene_stats), N printed in a comment.
mIoU, Acc., ECE, AURC and AUSE in %, Brier x100, NLL in nats. Variant "all" applies the
coverage rule (uncovered pixels count as wrong).
AUSE_u = AUSE of the row's own uncertainty against the 0/1 error:
    NormLift rows  normlift_1-R / normlift_1-Cinter      LEGaussians  legs_u (its learned u)
    ours           u_ale                                 2D, averaging, Feature 3DGS: none ("--")
The ranked Table I rows are open-vocabulary. Closed-set CSS (told the class list at mapping
time) is printed below them as a reference (TAB1_REF) and never marked (user decision
2026-10-17: shown, not compared).
Marks: in every number column, bold (best) and underline (second best) on the printed
mean (not its std); rows with the same printed value share the mark. Table III (an ablation:
each row is read against the full method, and most cells tie) has marks only with --marks.
Choices made on the VALIDATION file only (--val): the P3 temperature (ours_P3_b<beta_g>) and
the CSS prior read-out (css or css_a<alpha0>, "choose --key css") by the lowest validation
NLL, and LEGaussians' lambda_u (legs_l<lambda>) by the lowest validation AUSE of its own u.
Without --val the defaults css, ours_P3_b20 and legs_l1 are used and a comment says so.
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

Table I v2 (read-out-only predictives, scripts/replica_tab1_v2.sh) follows the rules written
down before any v2 run in configs/tab1_v2_prereg.json (--prereg): '@p1' (the P1 c0 grid) and
'@p4' (the P4 vote grid) are the candidates with the lowest validation NLL POOLED over the
labelled pixels of the validation scenes, sum_s n_s NLL_s / sum_s n_s (not the scene mean of
the v1 keys); '@ours_v2' is the lowest of ours_P2, ours_P2p1, @p1, @p4. Values within the
prereg's tie tolerance go to the candidate listed first. All candidates are reported.
    python3 scripts/make_tables.py choose <eval_val.json> --key ours_v2
    python3 scripts/make_tables.py select <eval_val.json> --out selection_val.json
        the v2 selection as a run record (prereg and validation record sha256, every value)
    python3 scripts/make_tables.py check-scenes <scenes.json> --split validation
        exit 1 unless the scenes file holds exactly the prereg's scenes of that split
    python3 scripts/make_tables.py check-test <eval_val.json> --selection selection_val.json \\
        --test-record eval_test.json --look look_test.json   exit 1 unless the v2 test stage
        may run (the validation record is of the prereg's validation scenes and of the code
        now in scripts/ usplat/ configs/; the selection is that of any earlier test attempt)
    python3 scripts/make_tables.py tab1 <eval_test.json> --val <eval_val.json> --v2
        Table I with the v2 rows after ours P2

Table II (tab:change), stale content on NeRF-Object-Removal, from the merge of
scripts/nor_tab2.py (scripts/nor_tab2.sh writes it):
    python3 scripts/make_tables.py tab2 <change_test.json> > rows.tex
Rows (TAB2) follow papers/main.tex with their input column (I image, D LiDAR depth, F
features); columns Gaussian AP, AUROC, pixel AP, AUROC, FA, Rec., all in %, mean {\\scriptsize
+- std} over the scenes (population std). AP / AUROC rank a Gaussian (pixel) that a row never
scores last. FA / Rec. are at each row's own label-free operating point; the fused row has
none ("--"). Marks on the printed means of every column, all rows ranked: bold best,
underline second best; FA is better when lower, the rest when higher. --digits D sets the
decimals of FA and Rec. (AP and AUROC keep 1). Comment lines after the rows: the
unobserved Gaussians left out (mean per scene and pooled), the stale and box-only counts, each
row's never-scored share and its AP / AUROC on the set every row defines, ours' Kept, the
per-scene count of scenes where ours is higher than each row, the settings, and warnings (k
of the runs is not the label's k = 2, a mock backend, a mapping replay that was not exact,
a cell over fewer scenes).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys

# (label, row name or a choice key, own-uncertainty signal or None, predictive cells)
# predictive cells: "all" | "same" (NormLift: same predictive as averaging) | "asP0" (P2)
TAB1 = [
    ("2D features~\\cite{li2022lseg}, per frame", "2d_features", None, "all"),
    ("Averaging~\\cite{joseph2025gradbp}\\reimpl", "averaging", None, "all"),
    ("NormLift~\\cite{zang2026normlift}\\reimpl, $1-\\mathcal{R}$", "normlift", "normlift_1-R", "all"),
    ("Feature 3DGS~\\cite{zhou2024feature}\\reimpl", "f3dgs", None, "all"),
    ("LEGaussians~\\cite{shi2024legaussians}\\reimpl", "@legs", "legs_u", "all"),
    None,                                                     # \midrule
    ("Ours", "ours_P2", "u_ale", "all"),                    # P2 (adaptive beta) is our method
]
# reference rows: shown below the ranked rows, never marked (closed set: told the class list at
# mapping time, which the open-vocabulary rows do not get; user decision 2026-10-17)
TAB1_REF = [("CSS~\\cite{wilson2025css}\\reimpl, closed set", "@css", "css_var_top", "all")]
# attempt 06 (P3, "@p3") stays out of Table I: its validation NLL did not beat P2's (the
# rule fixed before the run); scripts/make_tables.py choose --key p3 still reports its choice
# Table I v2 (--v2): appended after ours P2; "asP1": P2 on the P1 base decides like P1
TAB1_V2 = [
    ("Ours P2 on P1 (adaptive $\\beta$)", "ours_P2p1", "u_ale", "asP1"),
    ("Ours P1, prior $c_0$ chosen on val", "@p1", "u_ale", "all"),
    ("Ours P4 (vote predictive)", "@p4", "u_ale", "all"),
    ("Ours P4, zero noise (argmax votes)", "ours_P4_inf", "u_ale", "all"),
    ("Ours P4, plug-in $\\kappa$ ($\\bar r$)", "ours_P4_rbar", "u_ale", "all"),
]
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


# ---- Table I v2: pre-registered selection on validation (configs/tab1_v2_prereg.json)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = os.path.join(ROOT, "configs", "tab1_v2_prereg.json")
V2_KEYS = ("@p1", "@p4", "@ours_v2")
# the code of a v2 evaluation (as the shell gates): the test stage runs the validation code
CODE_DIRS = ("scripts", "usplat", "configs")


def load_prereg(path=None):
    with open(path or PREREG) as f:
        return json.load(f)


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def prereg_rows(prereg):
    """Every row the v2 rules name (the candidates of all choices and the ablations)."""
    rows = [r for c in prereg["choices"].values() for r in c["rows"] if not r.startswith("@")]
    return sorted(set(rows) | set(prereg.get("ablations", [])))


def scene_names(path):
    """Sorted scene names of a scenes file (eval_seg.py: a JSON or YAML list of {name, ...})."""
    with open(path) as f:
        text = f.read()
    try:
        sc = json.loads(text)
    except json.JSONDecodeError:
        import yaml
        sc = yaml.safe_load(text)
    return sorted(s["name"] for s in sc)


def prereg_scenes(prereg, split, root=None):
    """Scene names of the prereg's `split` ('validation' | 'test'), read from the scene files
    it names (a relative path is from the repository root); the two splits are disjoint."""
    names = {}
    for k in ("validation", "test"):
        f = prereg["scenes"][k]
        names[k] = scene_names(f if os.path.isabs(f) else os.path.join(root or ROOT, f))
    both = sorted(set(names["validation"]) & set(names["test"]))
    if both:
        raise SystemExit(f"refused: the prereg's validation and test scenes overlap: {both}")
    return names[split]


def check_scenes(path, split, prereg, root=None):
    """Refuse a scenes file that does not hold exactly the prereg's `split` (by scene name)."""
    got, want = scene_names(path), prereg_scenes(prereg, split, root)
    if got != want:
        raise SystemExit(f"refused: {path} holds the scenes {got}; the prereg's {split} scenes "
                         f"are {want} (another dataset needs its own prereg)")
    return got


def check_validation_record(raw, val_body, prereg, path="", root=None):
    """Refuse a validation record not made on the prereg's validation scenes: it evaluates
    exactly those, with the betas fit on the same scene file (--val-scenes = --scenes, as
    STAGE=val runs eval_seg.py), so neither a temperature nor a choice saw a test label."""
    got, want = sorted(val_body.get("scenes") or {}), prereg_scenes(prereg, "validation", root)
    if got != want:
        raise SystemExit(f"refused: {path} evaluates the scenes {got}; the prereg's validation "
                         f"scenes are {want}")
    cfg = raw.get("config") or {}
    if not cfg.get("val_scenes") or cfg["val_scenes"] != cfg.get("scenes"):
        raise SystemExit(f"refused: {path}: the betas must be fit on the evaluated validation "
                         f"scenes (--val-scenes {cfg.get('val_scenes')!r}, --scenes "
                         f"{cfg.get('scenes')!r})")


def check_code(raw, path="", root=None):
    """Refuse unless the working tree runs the code of the validation run: every v2 row is a
    read-out at evaluation time, so a selection holds only for the code that made it. The
    record (usplat.runlog git state) names a commit and had no uncommitted tracked change
    (dirty_tracked), and `git diff <commit> -- scripts usplat configs` (bytecode aside) is
    empty, staged or not."""
    g = raw.get("git") or {}
    if g.get("error") or not g.get("commit"):
        raise SystemExit(f"refused: {path} records no git commit "
                         f"({g.get('error', 'no git state')})")
    if g.get("dirty_tracked", g.get("dirty", True)):
        raise SystemExit(f"refused: {path} was made with uncommitted changes (git diff sha256 "
                         f"{str(g.get('diff_sha256'))[:12]}); rerun STAGE=val on a committed tree")
    r = subprocess.run(["git", "-C", root or ROOT, "diff", "--quiet", g["commit"], "--",
                        *CODE_DIRS, ":(exclude,glob)**/*.pyc"], capture_output=True, text=True)
    if r.returncode == 1:
        raise SystemExit(f"refused: {' '.join(CODE_DIRS)} differ from the validation commit "
                         f"{g['commit'][:12]} of {path}; rerun STAGE=val with this code")
    if r.returncode != 0:
        raise SystemExit(f"refused: cannot compare the code with the validation commit "
                         f"{g['commit'][:12]}: {r.stderr.strip()}")


def pooled(val_body, row, crit="nll", variant="all"):
    """The validation criterion pooled over the labelled pixels of every validation scene:
    sum_s n_s c_s / sum_s n_s (n_s = n_pixels of the variant). Refuses a missing row."""
    scenes = val_body.get("scenes") or {}
    if not scenes:
        raise SystemExit("the validation record has no per-scene results (scenes)")
    num = den = 0.0
    for name, sc in scenes.items():
        m = (sc["rows"].get(row) or {}).get(variant)
        if not m or crit not in m:
            raise SystemExit(f"row {row!r} has no {variant}/{crit} in validation scene {name!r}")
        n = float(m.get("n_pixels", 0))
        num += float(m[crit]) * n
        den += n
    return num / den if den > 0 else float("nan")


def choose_v2(val_body, key, prereg=None):
    """(row, {candidate: [row, pooled value]}) of the v2 choice `key` by the prereg's rule:
    the lowest pooled validation criterion; values within tie.tol of it go to the candidate
    listed first; a candidate '@k' stands for the row chosen for key k. Missing or
    non-finite values refuse (a v2 selection needs every candidate scored on validation)."""
    p = prereg or load_prereg()
    met, tol = p["metric"], float(p["tie"]["tol"])
    if met.get("pooling") != "pixels":
        raise SystemExit(f"prereg pooling {met.get('pooling')!r}: only 'pixels' is implemented")
    vals = {}
    for c in p["choices"][key]["rows"]:
        r = choose_v2(val_body, c, p)[0] if c.startswith("@") else c
        vals[c] = [r, pooled(val_body, r, met["name"], met.get("variant", "all"))]
    bad = [c for c, (_, v) in vals.items() if not math.isfinite(v)]
    if bad:
        raise SystemExit(f"{key}: non-finite validation {met['name']} for {bad}")
    best = min(v for _, v in vals.values())
    first = next(c for c, (_, v) in vals.items() if v <= best + tol)
    return vals[first][0], vals


def selection(val_path, prereg_path=None, root=None):
    """The v2 selection and what it depends on (written by `select`, read by `check-test`);
    refuses a validation record not made on the prereg's validation scenes."""
    prereg_path = os.path.abspath(prereg_path or PREREG)
    p = load_prereg(prereg_path)
    raw = json.load(open(val_path))
    b = body(val_path)
    check_validation_record(raw, b, p, val_path, root)
    ch = {}
    for k in V2_KEYS:
        row, vals = choose_v2(b, k, p)
        ch[k] = dict(row=row, values=vals)
    return dict(prereg=dict(path=prereg_path, sha256=sha256_file(prereg_path),
                            registered=p.get("registered")),
                validation=dict(path=os.path.abspath(val_path), sha256=sha256_file(val_path),
                                run_id=raw.get("run_id"), git=raw.get("git"),
                                scenes=sorted(b["scenes"])),
                metric=p["metric"], tie=p["tie"], choices=ch,
                chosen={k: v["row"] for k, v in ch.items()})


def _finished(path):
    d = json.load(open(path))
    return d.get("status", "ok") == "ok", d


def check_test(val_path, sel_path, test_path, prereg_path=None, look_path=None, root=None):
    """Refuse (SystemExit "refused: ...") unless the v2 test stage may run: a finished
    validation record with a beta for every prereg row, of the prereg's validation scenes
    (check_validation_record) and of the code in the tree (check_code); a finished selection
    record of THIS validation record and THIS prereg file, which the rule reproduces; the
    selection of every earlier test attempt (look_path, written when the test stage starts:
    the selection is final from then on); and no finished test record (the test split is
    evaluated once per version). Returns the chosen rows."""
    prereg_path = os.path.abspath(prereg_path or PREREG)
    p = load_prereg(prereg_path)
    if not os.path.exists(val_path):
        raise SystemExit(f"refused: no validation record {val_path} (run STAGE=val first)")
    ok, d = _finished(val_path)
    b = d["metrics"] if isinstance(d.get("metrics"), dict) else d
    if not ok or not b.get("betas"):
        raise SystemExit(f"refused: {val_path} is not a finished validation fit with betas "
                         f"(status {d.get('status')})")
    miss = [r for r in prereg_rows(p) if r not in b["betas"]]
    if miss:
        raise SystemExit(f"refused: {val_path} has no fitted beta for the v2 rows {miss} "
                         f"(run eval_seg.py with --v2-rows)")
    check_validation_record(d, b, p, val_path, root)
    check_code(d, val_path, root)
    if not os.path.exists(sel_path):
        raise SystemExit(f"refused: no validation selection {sel_path}")
    ok, s = _finished(sel_path)
    sel = (s["metrics"] if isinstance(s.get("metrics"), dict) else s).get("selection")
    if not ok or not sel:
        raise SystemExit(f"refused: {sel_path} is not a finished selection record")
    if sel["validation"]["sha256"] != sha256_file(val_path):
        raise SystemExit(f"refused: {val_path} changed after the selection {sel_path}")
    if sel["prereg"]["sha256"] != sha256_file(prereg_path):
        raise SystemExit(f"refused: {prereg_path} changed after the selection {sel_path}")
    again = {k: choose_v2(b, k, p)[0] for k in V2_KEYS}
    if again != sel["chosen"]:
        raise SystemExit(f"refused: the rule gives {again}, the selection record {sel['chosen']}")
    if look_path and os.path.exists(look_path):  # an earlier test attempt (crashed or killed)
        lk = json.load(open(look_path))
        now = dict(chosen=sel["chosen"], validation_sha256=sel["validation"]["sha256"],
                   prereg_sha256=sel["prereg"]["sha256"])
        if {k: lk.get(k) for k in now} != now:
            raise SystemExit(f"refused: the test stage started before with another selection "
                             f"({look_path}: {len(lk.get('attempts') or [])} attempt(s), chosen "
                             f"{lk.get('chosen')}); the selection is final")
    if os.path.exists(test_path) and _finished(test_path)[0]:
        raise SystemExit(f"refused: {test_path} holds a finished v2 test run (the test split "
                         f"is evaluated once per version)")
    return sel["chosen"]


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


def tab1(path, variant="all", val=None, v2=False, prereg=None):
    summ = body(path)["summary"]
    vs = body(val)["summary"] if val else None
    picked = {k: (choose(vs, k, variant) if vs else CHOICES[k][1]) for k in CHOICES}
    if v2:                               # v2 choices only from validation (no defaults)
        if not val:
            raise SystemExit("tab1 --v2 needs --val (the v2 rows are chosen on validation)")
        p = load_prereg(prereg)
        picked.update({k: choose_v2(body(val), k, p)[0] for k in V2_KEYS})
    ns = sorted({r[variant]["miou"]["n"] for r in summ.values()})
    ranked_entries = TAB1 + (TAB1_V2 if v2 else [])
    lines = [f"% generated by scripts/make_tables.py tab1 {path} --variant {variant}"
             + (f" --val {val}" if val else " (no --val: default CSS prior and P3 beta_g)")
             + (" --v2" if v2 else "")
             + f"; scenes per row N = {ns}; chosen on val: "
             + str({k: v for k, v in picked.items()
                    if (v2 and k in V2_KEYS) or any(e and e[1] == k for e in ranked_entries + TAB1_REF)}),
             "% bold = best, underline = second best per column"]
    out, table = _tab1_rows(summ, picked, ranked_entries, variant, path)
    rows = iter(table)
    for item in out:
        if item[0] == "line":
            lines.append(item[1])
            continue
        _, label, pred = item
        lines.append(f"{label} & " + " & ".join(_tab1_body(next(rows), pred)) + "\\\\")
    return "\n".join(lines)


def _tab1_rows(summ, picked, ranked_entries, variant, path, std=True):
    """Table I cells of one evaluation, marked: (out, table). out lists ("line", text) and
    ("row", label, pred) items in print order; table holds one list of cells per row (None
    inside a \\multicolumn); the reference rows (CSS) are printed but never marked."""
    out, table, ranked = [], [], []
    ref = [None] + TAB1_REF                   # None: \midrule before the reference block
    for entry in ranked_entries + ref:
        if entry is None:
            out.append(("line", "\\midrule"))
            continue
        label, row, sig, pred = entry
        row = picked.get(row, row)
        if row not in summ:
            out.append(("line", f"% {label}: row {row!r} not in {path}"))
            continue
        m = summ[row][variant]
        cells = [cell(m.get(k), sc, 2 if k == "nll" else 1, std) for k, sc in COLS]
        if pred == "same":                     # NormLift: averaging's predictive, not repeated
            cells = [None] * len(COLS)
        elif pred in ("asP0", "asP1"):           # adaptive beta: argmax of its base
            cells[:2] = [None, None]
        cells.append(cell(m["ause"].get(sig), 100.0, 1, std) if sig else "--")
        out.append(("row", label, pred))
        table.append(cells)
        ranked.append(entry in ranked_entries)   # missing rows cannot shift CSS into the ranking
    mark_columns(table, ranked, TAB1_HIGHER)
    return out, table


def _tab1_body(cells, pred):
    """The printed cells of one Table I row (the \\multicolumn spans of NormLift and P2)."""
    if pred == "same":
        return ["\\multicolumn{6}{c}{same predictive as averaging}", cells[-1]]
    if pred in ("asP0", "asP1"):
        return [f"\\multicolumn{{2}}{{c}}{{as {pred[2:]}}}"] + cells[2:]
    return cells


def tab1pair(paths, vals, variant="all"):
    """Table I with two datasets side by side (e.g. Replica | ScanNet): the rows of tab1 (no v2
    rows), mean over the test scenes only (no std: two groups of seven columns with std do not
    fit the page width), each dataset's row choices from its own validation file, marks per
    dataset. paths, vals: one test and one validation eval_seg JSON per dataset, in column order."""
    if len(paths) != 2 or len(vals) != 2:
        raise SystemExit("tab1pair needs two test files and two --val files (one per dataset)")
    groups, notes = [], []
    for path, val in zip(paths, vals):
        summ, vs = body(path)["summary"], body(val)["summary"]
        picked = {k: choose(vs, k, variant) for k in CHOICES}
        ns = sorted({r[variant]["miou"]["n"] for r in summ.values()})
        used = {k: v for k, v in picked.items() if any(e and e[1] == k for e in TAB1 + TAB1_REF)}
        notes.append(f"% {path} --val {val}: scenes per row N = {ns}; chosen on val: {used}")
        groups.append(_tab1_rows(summ, picked, TAB1, variant, path, std=False))
    (out_a, tab_a), (out_b, tab_b) = groups
    if out_a != out_b:                       # same rows in the same order in both groups
        raise SystemExit(f"tab1pair: the two files do not hold the same Table I rows:\n"
                         f"{out_a}\n{out_b}")
    lines = [f"% generated by scripts/make_tables.py tab1pair {' '.join(paths)} --variant "
             f"{variant} --val {' --val '.join(vals)} (means only)"] + notes + \
            ["% bold = best, underline = second best per column, per dataset"]
    ra, rb = iter(tab_a), iter(tab_b)
    for item in out_a:
        if item[0] == "line":
            lines.append(item[1])
            continue
        _, label, pred = item
        cells = _tab1_body(next(ra), pred) + _tab1_body(next(rb), pred)
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    return "\n".join(lines)


# Table III (tab:ablation), Replica block: (label, ablation variant; None = the full method)
TAB3 = [
    ("\\textbf{Full method}", None),
    ("w/o view counting (pixels)", "pixels"),
    ("w/o $\\hat\\rho$ (uses $\\bar r$)", "est_rbar"),
    ("w/o $\\hat\\rho$ (uses $\\lVert\\R\\rVert/(W_1+c_0)$)", "est_joint"),
    ("w/o $\\hat\\rho$ (uses $\\tilde r$)", "est_legacy"),
    ("w/o raw features (centred)", "centre"),
    ("w/o all parts (pixel level)", "legacy"),
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


# Table II (tab:change): (label, input, row of scripts/eval_change_nor.py); None = \midrule
TAB2 = [
    ("NCC photometric residual", "I", "ncc"),
    ("Depth residual", "D", "depth"),
    ("Feature residual, $k=2$~\\cite{ackermann2025clsplats, jiang2026mif}\\reimpl", "F", "feat"),
    ("NormLift reliability drop~\\cite{zang2026normlift}\\reimpl", "F", "normlift"),
    ("Dispersion $1-\\hat\\rho$ (ours)", "F", "dispersion"),
    None,
    ("Ours: test, reset, re-admission", "F+D", "ours"),
    ("Ours fused with depth residual", "F+D", "fused"),
]
TAB2_K = 2                      # the persistence in the feature-residual label
# Table II columns: (name in comments, summary metric of the merge, larger is better), in %
TAB2_COLS = [("G AP", "g_ap", True), ("G AUROC", "g_auroc", True), ("P AP", "p_ap", True),
             ("P AUROC", "p_auroc", True), ("FA", "fa", False), ("Rec", "rec", True)]
TAB2_OP = ("fa", "rec")         # --digits sets their decimals; AP and AUROC keep 1


def _floats(x):
    """x with the "nan" / "inf" / "-inf" strings of usplat.runlog records back as floats."""
    if isinstance(x, dict):
        return {k: _floats(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_floats(v) for v in x]
    return float(x) if isinstance(x, str) and x in ("nan", "inf", "-inf") else x


def merged(path):
    """The metrics of a finished scripts/nor_tab2.py merge record."""
    d = json.load(open(path))
    if d.get("status", "ok") != "ok":
        raise SystemExit(f"{path}: status {d.get('status')!r}, not a finished merge")
    b = d["metrics"] if isinstance(d.get("metrics"), dict) else d
    if "summary" not in b or "scenes_order" not in b:
        raise SystemExit(f"{path}: no summary (not a scripts/nor_tab2.py merge)")
    return _floats(b)


def _pct(stat, digits=1):
    """The mean of a stat in % ('12.3'), or '--'."""
    if not stat or not stat.get("n") or not math.isfinite(stat.get("mean", float("nan"))):
        return "--"
    return f"{100 * stat['mean']:.{digits}f}"


def _pm(stat, digits=1):
    """'12.3 +- 4.5' in % (comments), or '--'."""
    if _pct(stat) == "--":
        return "--"
    return f"{100 * stat['mean']:.{digits}f} +- {100 * stat['std']:.{digits}f}"


def _finite(x):
    return x is not None and math.isfinite(x)


def tab2_wins(b, rows):
    """Per scene: in how many scenes ours is higher than each other row, per ranking column."""
    per, names = b.get("scenes") or {}, b["scenes_order"]
    if "ours" not in b["summary"] or not per:
        return []
    cols = TAB2_COLS[:4]
    lines = [f"% per scene ({', '.join(names)}): scenes where ours is higher than the row, "
             + ", ".join(name for name, _, _ in cols) + " (of the scenes where both are defined)"]
    for label, r in rows:
        if r == "ours":
            continue
        parts = []
        for _, k, _ in cols:
            pairs = [(per[s]["rows"]["ours"][k], per[s]["rows"][r][k]) for s in names
                     if r in per[s]["rows"]]
            pairs = [(x, y) for x, y in pairs if _finite(x) and _finite(y)]
            parts.append(f"{sum(x > y for x, y in pairs)}/{len(pairs)}")
        lines.append(f"% ours vs {label}: " + ", ".join(parts))
    return lines


def tab2_notes(b, rows, n):
    """Comment lines under Table II (see the module docstring)."""
    s, c, o = b["summary"], b["counts"], b.get("ours") or {}
    st, mp, fit = b.get("settings") or {}, b.get("maps") or {}, b.get("fit_config") or {}
    lines = [f"% unobserved Gaussians (no update view; left out of every Gaussian-level cell): "
             f"{_pm(c['unobserved_share'])}% per scene, {100 * c['pooled_unobserved_share']:.1f}% "
             f"of all Gaussians of the {n} scenes",
             f"% stale observed Gaussians (D2): {c['n_stale_observed']['mean']:.0f} per scene "
             f"({_pm(c['stale_share'], 2)}% of the observed); {_pct(c['box_static_share'])}% "
             f"of the observed Gaussians in the inflated box are static by D2 (within 2 cm of "
             f"the after-removal depth; the box alone would call them stale)",
             f"% pixels (D1: masks_real of the update keyframes, covered pixels, pooled per "
             f"scene): {_pct(c['p_pos_share'])}% positive; keyframes per scene: mapping "
             f"{c['n_mapping_views']['mean']:.1f}, update {c['n_update_views']['mean']:.1f}"]
    for label, r in rows:
        v = s[r]
        lines.append(f"% {label}: never scored (ranked last) G {_pct(v['g_unscored'])}% / P "
                     f"{_pct(v['p_unscored'])}%; on the set every row defines (G "
                     f"{_pct(c['g_common_share'])}%, P {_pct(c['p_common_share'])}% of the "
                     f"population): G AP {_pct(v['g_ap_common'])}, AUROC "
                     f"{_pct(v['g_auroc_common'])}; P AP {_pct(v['p_ap_common'])}, AUROC "
                     f"{_pct(v['p_auroc_common'])}")
    if o:
        lines.append(f"% ours: Kept (stale observed Gaussians in the robust set after the "
                     f"update) {_pm(o['kept'])}% (after the mapping pass {_pm(o['kept_after_mapping'])}%);"
                     f" end state after re-admission FA {_pm(o['changed_end_fa'], 2)}%, Rec "
                     f"{_pm(o['changed_end_rec'])}%; resets per scene "
                     f"{o['n_resets']['mean']:.0f}")
    lines += tab2_wins(b, rows)
    lines.append(f"% settings: q {st.get('change_quantile')}, k {st.get('k')}, tau_cov "
                 f"{st.get('tau_cov')}, D2 margins {st.get('inflate_m')} / {st.get('near_m')} m, "
                 f"keyframes {st.get('kf_trans')} m / {st.get('kf_rot')} deg, downscale "
                 f"{st.get('downscale')}, extractor {st.get('extractor')}, backend "
                 f"{st.get('backend')}; pipeline: {b.get('pipeline')}")
    if mp.get("n_gaussians", {}).get("n"):
        lines.append(f"% maps: {mp['n_gaussians']['mean']:.0f} Gaussians, held-out PSNR "
                     f"{mp['psnr']['mean']:.2f} dB (means over scenes); fit iters "
                     f"{fit.get('iters')}, max_gaussians {fit.get('max_gaussians')}, max_res "
                     f"{fit.get('max_res')}")
    if st.get("k") != TAB2_K:
        lines.append(f"% WARNING: the feature-residual label says k={TAB2_K}, the runs used "
                     f"k = {st.get('k')}")
    if st.get("backend") != "gsplat":
        lines.append(f"% WARNING: backend {st.get('backend')!r}, not a protocol run")
    if not (b.get("replay") or {}).get("all_ok", True):
        lines.append(f"% WARNING: the mapping replay was not exact in scenes "
                     f"{b['replay'].get('not_ok')} (max relative dW1 "
                     f"{b['replay'].get('max_rel_dW1'):.2e})")
    return lines


def tab2(path, digits=1):
    b = merged(path)
    summ, names = b["summary"], b["scenes_order"]
    n = len(names)
    out, table, have = [], [], []
    for entry in TAB2:
        if entry is None:
            out.append(("line", "\\midrule"))
            continue
        label, inp, row = entry
        if row not in summ:
            out.append(("line", f"% {label}: row {row!r} not in {path}"))
            continue
        cells = [cell(summ[row].get(k), 100.0, digits if k in TAB2_OP else 1)
                 for _, k, _ in TAB2_COLS]
        out.append(("row", label, inp))
        table.append(cells)
        have.append((label, row))
    ns = sorted({summ[r][k]["n"] for _, r in have for _, k, _ in TAB2_COLS
                 if summ[r].get(k, {}).get("n")})
    lines = [f"% generated by scripts/make_tables.py tab2 {path}"
             + (f" --digits {digits}" if digits != 1 else "")
             + f"; split {b.get('split')}, scenes {', '.join(names)}; scenes per cell N = {ns}",
             "% mean {\\scriptsize +- population std} over the scenes, in %; bold = best, "
             "underline = second best per column (FA: lower is better, the rest: higher); "
             "FA / Rec. at each row's own label-free operating point; fused: none"]
    for label, r in have:
        short = [name for name, k, _ in TAB2_COLS if summ[r].get(k, {}).get("n")
                 and summ[r][k]["n"] != n]
        if short:
            lines.append(f"% {label}: fewer than the {n} scenes in " + ", ".join(short))
    mark_columns(table, [True] * len(table), [hb for _, _, hb in TAB2_COLS])
    rows = iter(table)
    for item in out:
        if item[0] == "line":
            lines.append(item[1])
            continue
        _, label, inp = item
        lines.append(f"{label} & {inp} & " + " & ".join(next(rows)) + "\\\\")
    return "\n".join(lines + tab2_notes(b, have, n))


# Per-class IoU table (tab:perclass, the layout of CSS's Table I): one column per class that is
# present in at least PERCLASS_MIN_SCENES of the file's scenes (a rule fixed before any cell was
# looked at), each cell the mean IoU over the scenes where the class is present, then the
# Table I mIoU over all classes. A class that every row (CSS included) prints as 0.0 carries no
# comparison: it is listed in a comment (and the caption) instead of a column. Ranked rows get bold / underline per column; CSS is shown last
# and not ranked. Columns: by the number of scenes with the class, then by name.
PERCLASS_MIN_SCENES = 5
PERCLASS_ROWS = [("2D features~\\cite{li2022lseg}, per frame", "2d_features"),
                 ("Averaging~\\cite{joseph2025gradbp}\\reimpl", "averaging"),
                 ("Feature 3DGS~\\cite{zhou2024feature}\\reimpl", "f3dgs"),
                 ("LEGaussians~\\cite{shi2024legaussians}\\reimpl", "@legs"),
                 ("Ours", "ours_P2")]   # P1 (negative) is in Table I
PERCLASS_REF = [("CSS~\\cite{wilson2025css}\\reimpl, closed set", "@css")]
# header colours (white text on each): a fixed categorical palette, one per column in order
PERCLASS_COLORS = ["1F78B4", "B15928", "6A3D9A", "8C564B", "2F4F4F", "E31A1C", "33A02C",
                   "D95F02", "7570B3", "1B9E77", "B8860B", "666666", "A6761D", "E7298A"]


def perclass_classes(scenes, row="ours_P2", variant="all", min_scenes=PERCLASS_MIN_SCENES):
    """The table's classes: present (an IoU entry) in at least min_scenes scenes of `row`."""
    from collections import Counter
    n = Counter()
    for sc in scenes.values():
        n.update(((sc.get("rows") or {}).get(row, {}).get(variant) or {}).get("iou", {}).keys())
    keep = [c for c, k in n.items() if k >= min_scenes]
    return sorted(keep, key=lambda c: (-n[c], c)), n


def perclass(path, val=None, variant="all", min_scenes=PERCLASS_MIN_SCENES):
    lines, classes, labels, table, ranked = _perclass_table(path, val, variant, min_scenes)
    lines[0] = lines[0].replace("% ", "% generated by scripts/make_tables.py perclass ", 1)
    lines.append(f"% columns: {len(classes)} classes + mIoU; header row:")
    lines.append("Method & " + _perclass_head(classes, PERCLASS_COLORS) + "\\\\")
    for label, cells, rk in zip(labels, table, ranked):
        if not rk:
            lines.append("\\midrule")
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    return "\n".join(lines)


def _perclass_head(classes, colors, miou="mIoU (\\%)"):
    """Header cells: one coloured, rotated class name per column (colors: one per class, in
    order), then the rotated mIoU column (header text `miou`)."""
    return " & ".join(f"\\cellcolor[HTML]{{{colors[i % len(colors)]}}}"
                      f"\\rotatebox{{90}}{{\\textcolor{{white}}{{{c}}}}}"
                      for i, c in enumerate(classes)) + f" & \\rotatebox{{90}}{{\\textbf{{{miou}}}}}"


def perclasspair(paths, vals, variant="all", min_scenes=PERCLASS_MIN_SCENES, plain=False):
    """Per-class table with two datasets side by side (e.g. Replica | ScanNet): each group is
    perclass's columns and cells for its own test file (classes present in >= min_scenes of its
    scenes, row choices from its own --val file), marked per dataset. A class present in both
    groups keeps one header colour; the first group's colours are perclass's."""
    if len(paths) != 2 or len(vals) != 2:
        raise SystemExit("perclasspair needs two test files and two --val files (one per dataset)")
    groups = [_perclass_table(p, v, variant, min_scenes) for p, v in zip(paths, vals)]
    if groups[0][2] != groups[1][2] or groups[0][4] != groups[1][4]:
        raise SystemExit(f"perclasspair: the two files do not hold the same rows: "
                         f"{groups[0][2]} vs {groups[1][2]}")
    first = groups[0][1]
    color = {c: PERCLASS_COLORS[i % len(PERCLASS_COLORS)] for i, c in enumerate(first)}
    spare = [x for x in PERCLASS_COLORS if x not in color.values()]
    for c in groups[1][1]:
        if c not in color:
            color[c] = spare.pop(0) if spare else PERCLASS_COLORS[len(color) % len(PERCLASS_COLORS)]
    lines = [f"% generated by scripts/make_tables.py perclasspair {' '.join(paths)} --variant "
             f"{variant} " + " ".join(f"--val {v}" for v in vals if v)]
    lines += [ln for g in groups for ln in g[0]]
    n1, n2 = len(groups[0][1]), len(groups[1][1])
    lines.append(f"% columns: {n1} classes + mIoU | {n2} classes + mIoU "
                 f"(tabular: @{{}}l*{{{n1}}}{{c}}|c*{{{n2}}}{{c}}|c@{{}}); header row:")
    # "mIoU" without "(%)": the tallest rotated header sets the row height (caption gives the unit)
    if plain:          # plain class names, bold for the best only, the ranked rows only (user, 2026-11-06)
        lines.append("Method & " + " & ".join(" & ".join([f"{{\\scriptsize {c}}}" for c in g[1]] + ["mIoU"])
                                              for g in groups) + "\\\\")
    else:
        lines.append("Method & " + " & ".join(_perclass_head(g[1], [color[c] for c in g[1]], "mIoU")
                                              for g in groups) + "\\\\")
    for i, (label, rk) in enumerate(zip(groups[0][2], groups[0][4])):
        if plain and not rk:
            continue
        if not rk:
            lines.append("\\midrule")
        cells = groups[0][3][i] + groups[1][3][i]
        if plain:
            cells = [re.sub(r"\\underline\{([^}]*)\}", r"\1", c_) for c_ in cells]
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    return "\n".join(lines)


def _perclass_table(path, val, variant, min_scenes):
    """One dataset's per-class cells: (comment lines, classes, row labels, marked cells per
    row with the mIoU last, ranked flag per row)."""
    b = body(path)
    scenes, summ = b["scenes"], b["summary"]
    vs = body(val)["summary"] if val else None
    picked = {k: (choose(vs, k, variant) if vs else CHOICES[k][1]) for k in CHOICES}
    classes, n = perclass_classes(scenes, variant=variant, min_scenes=min_scenes)
    lines = [f"% {path}"
             + (f" --val {val}" if val else "") + f" --variant {variant}; classes present in "
             f">= {min_scenes} of {len(scenes)} scenes: "
             + ", ".join(f"{c} ({n[c]})" for c in classes),
             "% cell = mean IoU (%) over the scenes where the class is present; last column = "
             "Table I mIoU over all classes; bold / underline = best / second best of the "
             "ranked rows; CSS (closed set) not ranked"]
    table, labels, ranked = [], [], []
    for label, row in PERCLASS_ROWS + PERCLASS_REF:
        row = picked.get(row, row)
        if row not in summ:
            lines.append(f"% {label}: row {row!r} not in {path}")
            continue
        cells = []
        for c in classes:
            v = [((scenes[s]["rows"].get(row) or {}).get(variant) or {}).get("iou", {}).get(c)
                 for s in scenes]
            v = [x for x in v if x is not None and math.isfinite(x)]
            cells.append(f"{100 * sum(v) / len(v):.1f}" if v else "--")
        cells.append(cell(summ[row][variant].get("miou"), 100.0, 1, std=False))
        table.append(cells)
        labels.append(label)
        ranked.append(not row.startswith("css"))
    zero = [j for j in range(len(classes)) if all(r[j] in ("0.0", "--") for r in table)]
    if zero:
        lines.append("% zero for every row (no column): " + ", ".join(classes[j] for j in zero))
    classes = [c for j, c in enumerate(classes) if j not in zero]
    table = [[x for j, x in enumerate(r) if j not in zero] for r in table]
    mark_columns(table, ranked, [True] * (len(classes) + 1))
    table = [[x.replace("\\textbf{0.0}", "0.0").replace("\\underline{0.0}", "0.0") for x in r]
             for r in table]                      # an IoU of 0 is never "best" or "second"
    return lines, classes, labels, table, ranked


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("table", choices=["tab1", "tab1pair", "tab2", "tab3", "perclass",
                                      "perclasspair", "choose", "select", "check-test",
                                      "check-scenes"])
    ap.add_argument("--key", default="legs", choices=["css", "p3", "legs", "p1", "p4", "ours_v2"])
    ap.add_argument("path")
    ap.add_argument("--val", default="", help="validation eval_seg JSON (row choices)")
    ap.add_argument("--variant", default="all", choices=["all", "covered"])
    ap.add_argument("--plain", action="store_true", help="perclasspair: plain headers, bold best only, no CSS row")
    ap.add_argument("--v2", action="store_true", help="tab1: add the Table I v2 rows")
    ap.add_argument("--pair", nargs=2, metavar=("TEST2", "VAL2"), default=None,
                    help="tab1pair, perclasspair: test and validation eval_seg JSON of the "
                         "second dataset (the first: path and --val)")
    ap.add_argument("--prereg", default=PREREG, help="Table I v2 pre-registration (JSON)")
    ap.add_argument("--out", default="", help="select: write the selection as a run record")
    ap.add_argument("--selection", default="", help="check-test: the v2 selection record")
    ap.add_argument("--test-record", default="", help="check-test: the v2 test record path")
    ap.add_argument("--look", default="", help="check-test: the look record of the test stage")
    ap.add_argument("--split", default="validation", choices=["validation", "test"],
                    help="check-scenes: the prereg split the scenes file must hold")
    ap.add_argument("--tab1", default="", help="tab3: Table I eval_seg JSON; a comment compares "
                    "the full method's printed cells with its row")
    ap.add_argument("--marks", action="store_true", help="tab3: bold / underline the best and "
                    "second-best cells of each column too")
    ap.add_argument("--digits", type=int, default=1,
                    help="tab3: decimals of the ECE and AUSE cells (mIoU keeps 1, NLL 2); "
                         "tab2: of the FA and Rec. cells (AP and AUROC keep 1)")
    a = ap.parse_args(argv)
    if a.table == "choose":                      # the validation choice of one row family
        if "@" + a.key in V2_KEYS:
            print(choose_v2(body(a.path), "@" + a.key, load_prereg(a.prereg))[0])
        else:
            print(choose(body(a.path)["summary"], "@" + a.key, a.variant))
    elif a.table == "tab3":
        print(tab3(a.path, a.variant, a.val or None, a.tab1 or None, a.digits, a.marks))
    elif a.table == "tab2":
        print(tab2(a.path, a.digits))
    elif a.table == "perclass":
        print(perclass(a.path, a.val or None, a.variant))
    elif a.table == "perclasspair":
        if not (a.val and a.pair):
            raise SystemExit("perclasspair needs --val and --pair TEST2 VAL2")
        print(perclasspair([a.path, a.pair[0]], [a.val, a.pair[1]], a.variant, plain=a.plain))
    elif a.table == "tab1pair":
        if not (a.val and a.pair):
            raise SystemExit("tab1pair needs --val and --pair TEST2 VAL2")
        print(tab1pair([a.path, a.pair[0]], [a.val, a.pair[1]], a.variant))
    elif a.table == "select":                    # the v2 selection, recorded
        sel = selection(a.path, a.prereg)
        if a.out:
            sys.path.insert(0, ROOT)
            from usplat.runlog import RunRecord
            with RunRecord(a.out, config=vars(a), name="tab1_v2_select",
                           argv=[sys.argv[0]] + list(argv if argv is not None
                                                     else sys.argv[1:])) as rec:
                rec.add_input(a.path)
                rec.add_input(a.prereg)
                rec.add("selection", sel)
        print(json.dumps(sel["chosen"]))
    elif a.table == "check-test":                # gate of the v2 test stage
        if not (a.selection and a.test_record):
            raise SystemExit("check-test needs --selection and --test-record")
        chosen = check_test(a.path, a.selection, a.test_record, a.prereg, a.look or None)
        print(f"validation selection: {chosen}")
    elif a.table == "check-scenes":              # the scenes of a v2 stage: the prereg's
        print(f"{a.split} scenes: {check_scenes(a.path, a.split, load_prereg(a.prereg))}")
    else:
        print(tab1(a.path, a.variant, a.val or None, a.v2, a.prereg))


if __name__ == "__main__":
    main()
