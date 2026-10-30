"""
smooth_select.py — selection, test gate and reporting of the spatial-smoothing attempt
(configs/smooth_prereg.json, rows of usplat/smoothing.py, driver scripts/smooth_tab1.sh).
CPU only: it reads eval_seg.py records, never a scene.

Records of one dataset live in one directory OUT (the prereg's datasets.<d>.out):
    eval_val_<batch>.json, eval_test_<batch>.json   eval_seg.py, one per batch (one sigma)
    selection_val.json                              `select` (a usplat.runlog record)
    look_test.json                                  `look` (one entry per test attempt)
    table_smooth.json, table_smooth.tex             `table`
and `claim` reads the table records of both datasets -> claim_smooth.json.

    python3 scripts/smooth_select.py batches --prereg P                 batch name + its rows
    python3 scripts/smooth_select.py select --prereg P --dataset D --dir OUT [--out SEL]
    python3 scripts/smooth_select.py check-test --prereg P --dataset D --dir OUT
    python3 scripts/smooth_select.py look --prereg P --dataset D --dir OUT --ts TS --log LOG
    python3 scripts/smooth_select.py table --prereg P --dataset D --dir OUT
    python3 scripts/smooth_select.py claim --prereg P --tables T1.json [T2.json]

Rules (all from the prereg): the batches' validation records are merged into one evaluation
after checking that the reference rows (ours_P0, ours_P2, averaging) agree across batches
(consistency_tol: betas and every per-scene metric); `select` applies scripts/make_tables.py
choose_v2 (lowest pixel-pooled validation NLL, tie tol, first listed wins) to the prereg's
choices @smooth and @avg_own. `check-test` refuses unless every validation batch is a
finished fit of the prereg's validation scenes made on a clean tree at one commit whose
scripts/ usplat/ configs/ equal the working tree (make_tables.check_code), the selection was
made from exactly these records and this prereg file and the rule still gives it, an earlier
test attempt (look_test.json) used the same selection, and at least one test batch is not
finished (finished batches are never rerun). `table` applies claim rule C1 per dataset,
`claim` the sign tests + Holm of family F3s over the datasets where smoothing was selected.
`select` also records table1_check: the reference rows against the dataset's Table I
validation record (betas and per-scene NLL differences; reported, not a gate).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import make_tables as mt                                            # noqa: E402
from usplat.paths import moved_tree                                  # noqa: E402

REFERENCE = ("ours_P0", "ours_P2", "averaging")
CONSISTENCY_KEYS = ("miou", "acc", "ece", "nll", "brier", "aurc", "n_pixels")
TAB_COLS = ("miou", "acc", "ece", "nll", "brier", "aurc")          # Table I columns (+ AUSE_u)
CLAIM_METRICS = ("miou", "ece", "nll", "brier", "aurc")
TABLE_JSON, TABLE_TEX = "table_smooth.json", "table_smooth.tex"


def die(msg):
    raise SystemExit(msg)


# ------------------------------------------------------------------------------ prereg
def load(prereg):
    with open(prereg) as f:
        return moved_tree(json.load(f))      # frozen: paths from before the 2026-10-31 move


def view(p, dataset):
    """The prereg as make_tables sees one dataset (its 'scenes' entry)."""
    if dataset not in (p.get("datasets") or {}):
        die(f"refused: dataset {dataset!r} not in the prereg's datasets "
            f"{sorted(p.get('datasets') or {})}")
    return dict(p, scenes=p["datasets"][dataset]["scenes"])


def names(tag):
    return f"ours_P0s_{tag}", f"ours_P2s_{tag}", f"averaging_s_{tag}"


def batch_rows(p, batch):
    rows = list(REFERENCE)
    for tag in p["batches"][batch]:
        rows += names(tag)
    return rows


def record_path(d, stage, batch):
    return os.path.join(d, f"eval_{stage}_{batch}.json")


def _raw(path):
    with open(path) as f:
        return moved_tree(json.load(f))


def _body(raw):
    return raw["metrics"] if isinstance(raw.get("metrics"), dict) else raw


def finished(path):
    return os.path.exists(path) and _raw(path).get("status", "ok") == "ok"


# ------------------------------------------------------------------------------ merge
def _close(a, b, tol):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(x) and math.isnan(y):
        return True
    return abs(x - y) <= tol


def merged(p, d, stage, require_all=True):
    """(body, provenance): the batch records of `stage` merged into one eval_seg body
    (scenes[s].rows, betas, summary: union of the rows), after checking every record holds its
    batch's rows (and betas on validation) and the reference rows agree across batches."""
    tol = float(p.get("consistency_tol", 1e-6))
    body, prov, ref = None, {}, None
    for b in p["batches"]:
        path = record_path(d, stage, b)
        if not finished(path):
            if require_all:
                die(f"refused: {path} is not a finished {stage} record (run STAGE={stage})")
            continue
        raw = _raw(path)
        bb = mt.body(path)
        want = batch_rows(p, b)
        for s, sc in bb["scenes"].items():
            miss = [r for r in want if r not in sc["rows"]]
            if miss:
                die(f"refused: {path} scene {s} lacks the rows {miss}")
        if stage == "val":
            miss = [r for r in want if r not in (bb.get("betas") or {})]
            if miss:
                die(f"refused: {path} has no fitted beta for {miss}")
        prov[b] = dict(path=os.path.abspath(path), sha256=mt.sha256_file(path),
                       run_id=raw.get("run_id"), git=raw.get("git"),
                       memory=raw.get("memory"), wall_s=raw.get("wall_s"))
        if body is None:
            body = dict(scenes={s: dict(sc, rows=dict(sc["rows"])) for s, sc in bb["scenes"].items()},
                        betas=dict(bb.get("betas") or {}), summary=dict(bb.get("summary") or {}))
            ref = (b, bb)
            continue
        if sorted(bb["scenes"]) != sorted(body["scenes"]):
            die(f"refused: {path} evaluates {sorted(bb['scenes'])}, batch {ref[0]} "
                f"{sorted(body['scenes'])}")
        for r in REFERENCE:                      # the batches must be one evaluation
            if stage == "val" and not _close(bb["betas"].get(r), ref[1]["betas"].get(r), tol):
                die(f"refused: reference row {r}: betas differ between batches {ref[0]} and "
                    f"{b} ({ref[1]['betas'].get(r)} vs {bb['betas'].get(r)})")
            for s in body["scenes"]:
                x, y = ref[1]["scenes"][s]["rows"][r]["all"], bb["scenes"][s]["rows"][r]["all"]
                bad = [k for k in CONSISTENCY_KEYS if not _close(x.get(k), y.get(k), tol)]
                if bad:
                    die(f"refused: reference row {r}, scene {s}: {bad} differ between batches "
                        f"{ref[0]} and {b} by more than {tol}")
        for s, sc in bb["scenes"].items():
            body["scenes"][s]["rows"].update(sc["rows"])
        body["betas"].update(bb.get("betas") or {})
        body["summary"].update(bb.get("summary") or {})
    if body is None:
        die(f"refused: no finished {stage} record in {d}")
    return body, prov


# -------------------------------------------------------------------------- selection
def selection(p, prereg_path, dataset, d):
    v = view(p, dataset)
    for b in p["batches"]:                     # each record: the prereg's validation scenes
        path = record_path(d, "val", b)
        if finished(path):
            mt.check_validation_record(_raw(path), mt.body(path), v, path)
    body, prov = merged(p, d, "val")
    ch = {}
    for key in p["choices"]:
        row, vals = mt.choose_v2(body, key, v)
        ch[key] = dict(row=row, values=vals)
    best = ch["@smooth"]["row"]
    tag = best[len("ours_P2s_"):] if best.startswith("ours_P2s_") else None
    return dict(prereg=dict(path=os.path.abspath(prereg_path), sha256=mt.sha256_file(prereg_path),
                            registered=p.get("registered")),
                dataset=dataset, validation=prov, scenes=sorted(body["scenes"]),
                metric=p["metric"], tie=p["tie"], choices=ch,
                chosen={k: c["row"] for k, c in ch.items()}, kernel=tag,
                paired=None if tag is None else names(tag)[2],
                table1_check=table1_check(p, dataset, body))


def table1_check(p, dataset, body):
    """Independent check (reported, never a refusal): the reference rows of this run against
    the dataset's Table I validation record (prereg datasets.<d>.table1.val): the largest
    difference of the fitted betas and of the per-scene validation NLL. Another GPU or code
    version may differ in the last digits; a large gap means the read-out does not reproduce
    Table I and must be explained before any test run."""
    path = ((p["datasets"][dataset].get("table1") or {}).get("val")) or ""
    if not path:
        return dict(record=None, status="none named in the prereg")
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.isfile(path):
        return dict(record=path, status="missing")
    t1 = _body(_raw(path))
    out = dict(record=path, sha256=mt.sha256_file(path), rows={})
    for r in REFERENCE:
        b0, b1 = (t1.get("betas") or {}).get(r), body["betas"].get(r)
        if isinstance(b0, dict) and isinstance(b1, dict):
            db = max(abs(float(b0[k]) - float(b1[k])) for k in b0) if b0.keys() == b1.keys() \
                else float("nan")
        elif b0 is not None and b1 is not None:
            db = abs(float(b0) - float(b1))
        else:
            db = float("nan")
        dn = [abs(float(t1["scenes"][s]["rows"][r]["all"]["nll"]) -
                  float(body["scenes"][s]["rows"][r]["all"]["nll"]))
              for s in body["scenes"] if r in ((t1.get("scenes") or {}).get(s) or {}).get("rows", {})]
        out["rows"][r] = dict(max_abs_beta_diff=db,
                              max_abs_val_nll_diff=max(dn) if dn else float("nan"))
    return out


def cmd_select(a):
    p = load(a.prereg)
    sel = selection(p, a.prereg, a.dataset, a.dir)
    if a.out:
        from usplat.runlog import RunRecord
        with RunRecord(a.out, config=vars(a), name="smooth_select",
                       argv=[sys.argv[0]] + sys.argv[1:]) as rec:
            for b in p["batches"]:
                rec.add_input(record_path(a.dir, "val", b))
            rec.add_input(a.prereg)
            rec.add("selection", sel)
    print(json.dumps(dict(chosen=sel["chosen"], kernel=sel["kernel"], paired=sel["paired"])))


def _selection_record(path):
    if not os.path.exists(path):
        die(f"refused: no validation selection {path} (run STAGE=val)")
    raw = _raw(path)
    sel = _body(raw).get("selection")
    if raw.get("status", "ok") != "ok" or not sel:
        die(f"refused: {path} is not a finished selection record")
    return sel


def check_test(p, prereg_path, dataset, d):
    """The batches still to evaluate on test (refuses as described in the module doc)."""
    v = view(p, dataset)
    commits = set()
    for b in p["batches"]:
        path = record_path(d, "val", b)
        if not finished(path):
            die(f"refused: {path} is not a finished validation record")
        raw = _raw(path)
        mt.check_validation_record(raw, mt.body(path), v, path)
        mt.check_code(raw, path)
        commits.add((raw.get("git") or {}).get("commit"))
    if len(commits) != 1:
        die(f"refused: the validation batches were made at different commits {sorted(commits)}")
    sel_path = os.path.join(d, "selection_val.json")
    sel = _selection_record(sel_path)
    if sel["prereg"]["sha256"] != mt.sha256_file(prereg_path):
        die(f"refused: {prereg_path} changed after the selection {sel_path}")
    if sel.get("dataset") != dataset:
        die(f"refused: {sel_path} is a selection of {sel.get('dataset')!r}, not {dataset!r}")
    for b in p["batches"]:
        path = record_path(d, "val", b)
        if (sel["validation"].get(b) or {}).get("sha256") != mt.sha256_file(path):
            die(f"refused: {path} changed after the selection {sel_path}")
    again = selection(p, prereg_path, dataset, d)["chosen"]
    if again != sel["chosen"]:
        die(f"refused: the rule gives {again}, the selection record {sel['chosen']}")
    look = os.path.join(d, "look_test.json")
    if os.path.exists(look):
        lk = _raw(look)
        now = dict(chosen=sel["chosen"], prereg_sha256=sel["prereg"]["sha256"],
                   validation_sha256={b: e["sha256"] for b, e in sel["validation"].items()})
        if {k: lk.get(k) for k in now} != now:
            die(f"refused: the test stage started before with another selection ({look}); "
                f"the selection is final")
    todo = [b for b in p["batches"] if not finished(record_path(d, "test", b))]
    if not todo:
        die(f"refused: every test batch in {d} is finished (the test split is evaluated once)")
    return sel, todo


def cmd_check_test(a):
    p = load(a.prereg)
    sel, todo = check_test(p, a.prereg, a.dataset, a.dir)
    print(f"validation selection: {sel['chosen']} (kernel {sel['kernel']})", file=sys.stderr)
    for b in todo:
        print(f"batch {b}")


def cmd_look(a):
    p = load(a.prereg)
    sel = _selection_record(os.path.join(a.dir, "selection_val.json"))
    out = os.path.join(a.dir, "look_test.json")
    git = lambda *x: subprocess.run(["git", "-C", ROOT, *x], capture_output=True,
                                    text=True).stdout.strip()
    attempt = dict(started=a.ts, log=a.log, git=dict(commit=git("rev-parse", "HEAD"),
                                                       branch=git("rev-parse", "--abbrev-ref",
                                                                  "HEAD")))
    if os.path.exists(out):
        rec = _raw(out)
    else:
        rec = dict(dataset=a.dataset, test_look=p["datasets"][a.dataset].get("test_look"),
                   scenes=p["datasets"][a.dataset]["scenes"]["test"], chosen=sel["chosen"],
                   prereg_sha256=sel["prereg"]["sha256"],
                   validation_sha256={b: e["sha256"] for b, e in sel["validation"].items()},
                   attempts=[])
    rec.setdefault("attempts", []).append(attempt)
    with open(out + ".tmp", "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(out + ".tmp", out)
    print(f"look record {out}: attempt {len(rec['attempts'])}, chosen {sel['chosen']}")


# ------------------------------------------------------------------------------- table
def scene_values(body, row, key):
    out = {}
    for s, sc in sorted(body["scenes"].items()):
        r = sc["rows"][row]["all"]
        out[s] = float(r["ranking"]["u_ale"]["ause"]) if key == "ause_u" else float(r[key])
    return out


def _mean(v):
    return sum(v.values()) / len(v) if v else float("nan")


def comparison(body, ours_row, base_row, key, qs):
    """Per-scene signed differences (positive = ours better, display units), x/N, the test
    mean of both rows, and the sign test (descriptive here; Holm only in `claim`)."""
    a, b = scene_values(body, ours_row, key), scene_values(body, base_row, key)
    diff = qs.signed(a, b, key)
    st = qs.sign_test(list(diff.values()))
    n = len(diff)
    wins = sum(1 for x in diff.values() if x > 0)
    mean_better = (_mean(a) > _mean(b)) if qs.HIGHER[key] else (_mean(a) < _mean(b))
    return dict(ours=ours_row, baseline=base_row, metric=key, n=n, wins=wins,
                losses=sum(1 for x in diff.values() if x < 0), mean_ours=_mean(a),
                mean_baseline=_mean(b), mean_better=bool(mean_better),
                majority=wins * 2 > n, sign_p=st["p"], diffs=diff)


def table(p, prereg_path, dataset, d):
    import quant_stats as qs
    sel = _selection_record(os.path.join(d, "selection_val.json"))
    if sel["prereg"]["sha256"] != mt.sha256_file(prereg_path):
        die(f"refused: {prereg_path} changed after the selection")
    body, prov = merged(p, d, "test")
    want = sorted(mt.prereg_scenes(view(p, dataset), "test"))
    if sorted(body["scenes"]) != want:
        die(f"refused: the test records hold {sorted(body['scenes'])}, the prereg's test "
            f"scenes are {want}")
    rows = {}
    for r in REFERENCE + tuple(x for t in p["candidates"] for x in names(t)):
        cells = {k: _mean(scene_values(body, r, k)) for k in TAB_COLS}
        if not r.startswith("averaging"):
            cells["ause_u"] = _mean(scene_values(body, r, "ause_u"))
        rows[r] = cells
    chosen, kernel = sel["chosen"], sel["kernel"]
    out = dict(dataset=dataset, selection=dict(chosen=chosen, kernel=kernel,
                                                paired=sel["paired"]),
               test=prov, scenes=sorted(body["scenes"]), rows=rows,
               selected=sorted({chosen["@smooth"], chosen["@avg_own"]}
                               | ({names(kernel)[0], names(kernel)[2]} if kernel else set())))
    if kernel is None:
        out["C1"] = dict(applies=False, text="@smooth is the unsmoothed ours_P2: smoothing not "
                                             "selected, no gain claim on this dataset")
    else:
        p0, p2, av = names(kernel)
        bases = dict(averaging="averaging", averaging_same_kernel=av,
                     averaging_own_kernel=chosen["@avg_own"])
        comps = {}
        for m in CLAIM_METRICS:
            ours = p0 if m == "miou" else p2
            comps[m] = {k: comparison(body, ours, b, m, qs) for k, b in bases.items()}
        verdict = {m: all(c["mean_better"] and c["majority"] for c in comps[m].values())
                   for m in CLAIM_METRICS}
        desc = {m: comparison(body, p0 if m == "miou" else p2,
                              "ours_P0" if m == "miou" else "ours_P2", m, qs)
                for m in CLAIM_METRICS}
        out["C1"] = dict(applies=True, baselines=bases, comparisons=comps, verdict=verdict,
                         vs_unsmoothed_ours=desc,
                         text="C1: mean better than each baseline AND better in more than "
                              "half of the test scenes against each")
    return out


def tex(out):
    lab = {"miou": 100, "acc": 100, "ece": 100, "nll": 1, "brier": 100, "aurc": 100}
    lines = [f"%% generated by scripts/smooth_select.py table ({out['dataset']}); every "
             f"candidate on test, 'not selected' unless chosen on validation"]
    for r, c in out["rows"].items():
        cells = [f"{c[k] * lab[k]:.{3 if k == 'nll' else 1}f}" for k in TAB_COLS]
        cells.append(f"{c['ause_u'] * 100:.1f}" if "ause_u" in c else "--")
        note = "" if r in out["selected"] or r in REFERENCE else " %% not selected"
        lines.append(f"{r.replace('_', chr(92) + '_')} & " + " & ".join(cells) + r" \\" + note)
    return "\n".join(lines) + "\n"


def cmd_table(a):
    p = load(a.prereg)
    out = table(p, a.prereg, a.dataset, a.dir)
    js = os.path.join(a.dir, TABLE_JSON)
    with open(js + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(js + ".tmp", js)
    with open(os.path.join(a.dir, TABLE_TEX), "w") as f:
        f.write(tex(out))
    v = out["C1"].get("verdict")
    print(f"{a.dataset}: chosen {out['selection']['chosen']}; C1 {v if v else 'n/a'} -> {js}")


# ------------------------------------------------------------------------------- claim
def claim(p, tables):
    """Family F3s: ours vs each baseline x metric, sign test over the pooled test scenes of the
    datasets where @smooth selected a smoothed row, Holm over the family; C2 per test."""
    import quant_stats as qs
    used = [t for t in tables if t["C1"].get("applies")]
    if not used:
        return dict(datasets=[t["dataset"] for t in tables], applies=False,
                    text="smoothing selected on no dataset: no claim")
    bases = ("averaging", "averaging_same_kernel", "averaging_own_kernel")
    tests, ps = [], []
    for b in bases:
        for m in CLAIM_METRICS:
            diffs = []
            for t in used:
                diffs += list(t["C1"]["comparisons"][m][b]["diffs"].values())
            st = qs.sign_test(diffs)
            tests.append(dict(baseline=b, metric=m, n=len(diffs),
                              wins=sum(1 for x in diffs if x > 0), sign_p=st["p"],
                              c1=all(t["C1"]["verdict"][m] for t in used)))
            ps.append(st["p"] if math.isfinite(st["p"]) else 1.0)
    for t, h in zip(tests, qs.holm(ps)):
        t["holm_p"] = h
        t["significant"] = bool(h < 0.05 and t["c1"])
    return dict(datasets=[t["dataset"] for t in used], applies=True, family="F3s",
                n_tests=len(tests), tests=tests,
                gain_claims={m: all(t["c1"] for t in tests if t["metric"] == m)
                             for m in CLAIM_METRICS})


def cmd_claim(a):
    p = load(a.prereg)
    tables = [_raw(t) for t in a.tables]
    out = claim(p, tables)
    path = a.out or os.path.join(os.path.dirname(os.path.abspath(a.tables[0])), "..",
                                 "claim_smooth.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("datasets", "applies") if k in out}), "->", path)


def cmd_batches(a):
    p = load(a.prereg)
    for b in p["batches"]:
        print(b, ",".join(batch_rows(p, b)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["batches", "select", "check-test", "look", "table", "claim"])
    ap.add_argument("--prereg", default=os.path.join(ROOT, "configs", "smooth_prereg.json"))
    ap.add_argument("--dataset", default="")
    ap.add_argument("--dir", default="", help="the dataset's record directory (OUT)")
    ap.add_argument("--out", default="")
    ap.add_argument("--ts", default="")
    ap.add_argument("--log", default="")
    ap.add_argument("--tables", nargs="+", default=[])
    a = ap.parse_args(argv)
    if a.cmd in ("select", "check-test", "look", "table") and not (a.dataset and a.dir):
        die(f"{a.cmd} needs --dataset and --dir")
    {"batches": cmd_batches, "select": cmd_select, "check-test": cmd_check_test,
     "look": cmd_look, "table": cmd_table, "claim": cmd_claim}[a.cmd](a)


if __name__ == "__main__":
    main()
