"""
make_tables.py — LaTeX rows of the paper's Table I (tab:replica) from an eval_seg.py JSON, so
every number in the paper comes from a saved evaluation file (no hand copying).

    python3 scripts/make_tables.py tab1 <eval_test.json> --val <eval_val.json> > rows.tex

Rows and columns follow papers/main.tex (tab:replica): mean {\\scriptsize +- std} over the
scenes of the file (population std, as usplat.seg_eval.scene_stats), N printed in a comment.
mIoU, Acc., ECE, AURC and AUSE in %, Brier x100, NLL in nats. Variant "all" applies the
coverage rule (uncovered pixels count as wrong).
AUSE_u = AUSE of the row's own uncertainty against the 0/1 error:
    NormLift rows  normlift_1-R / normlift_1-Cinter      CSS  css_var_top (its Eq. 10)
    LEGaussians    legs_u (its learned uncertainty)      ours u_ale
    2D, averaging, Feature 3DGS: none ("--")
Choices made on the VALIDATION file only (--val): the CSS prior read-out (css or
css_a<alpha0>) and the P3 temperature (ours_P3_b<beta_g>) by the lowest validation NLL, and
LEGaussians' lambda_u (legs_l<lambda>) by the lowest validation AUSE of its own u. Without
--val the defaults css, ours_P3_b20 and legs_l1 are used and a comment says so.
    python3 scripts/make_tables.py choose <eval_val.json> --key legs    prints the chosen row

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
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

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


def cell(stat, scale, digits=1):
    if stat is None or stat.get("n", 0) == 0 or not math.isfinite(stat.get("mean", float("nan"))):
        return "--"
    m, s, n = stat["mean"] * scale, stat.get("std", float("nan")) * scale, stat["n"]
    if n < 2 or not math.isfinite(s):
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f}{{\\scriptsize$\\,\\pm\\,${s:.{digits}f}}}"


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
    lines = [f"% generated by scripts/make_tables.py tab1 {path} --variant {variant}"
             + (f" --val {val}" if val else " (no --val: default CSS prior and P3 beta_g)")
             + (" --v2" if v2 else "")
             + f"; scenes per row N = {ns}; chosen on val: {picked}"]
    for entry in TAB1 + (TAB1_V2 if v2 else []):
        if entry is None:
            lines.append("\\midrule")
            continue
        label, row, sig, pred = entry
        row = picked.get(row, row)
        if row not in summ:
            lines.append(f"% {label}: row {row!r} not in {path}")
            continue
        m = summ[row][variant]
        if pred == "same":
            cells = ["\\multicolumn{6}{c}{same predictive as averaging}"]
        else:
            cells = [cell(m.get(k), sc, 2 if k == "nll" else 1) for k, sc in COLS]
            if pred in ("asP0", "asP1"):           # adaptive beta: argmax of its base
                cells = [f"\\multicolumn{{2}}{{c}}{{as {pred[2:]}}}"] + cells[2:]
        ause = cell(m["ause"].get(sig), 100.0) if sig else "--"
        ov = "$\\times$" if row.startswith("css") else "\\checkmark"
        lines.append(f"{label} & {ov} & " + " & ".join(cells) + f" & {ause}\\\\")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("table", choices=["tab1", "choose", "select", "check-test", "check-scenes"])
    ap.add_argument("--key", default="legs", choices=["css", "p3", "legs", "p1", "p4", "ours_v2"])
    ap.add_argument("path")
    ap.add_argument("--val", default="", help="validation eval_seg JSON (row choices)")
    ap.add_argument("--variant", default="all", choices=["all", "covered"])
    ap.add_argument("--v2", action="store_true", help="tab1: add the Table I v2 rows")
    ap.add_argument("--prereg", default=PREREG, help="Table I v2 pre-registration (JSON)")
    ap.add_argument("--out", default="", help="select: write the selection as a run record")
    ap.add_argument("--selection", default="", help="check-test: the v2 selection record")
    ap.add_argument("--test-record", default="", help="check-test: the v2 test record path")
    ap.add_argument("--look", default="", help="check-test: the look record of the test stage")
    ap.add_argument("--split", default="validation", choices=["validation", "test"],
                    help="check-scenes: the prereg split the scenes file must hold")
    a = ap.parse_args(argv)
    if a.table == "choose":                      # the validation choice of one row family
        if "@" + a.key in V2_KEYS:
            print(choose_v2(body(a.path), "@" + a.key, load_prereg(a.prereg))[0])
        else:
            print(choose(body(a.path)["summary"], "@" + a.key, a.variant))
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
