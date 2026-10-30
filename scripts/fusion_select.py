"""
fusion_select.py — selection, test gate and reporting of the "with the current view" attempt
(configs/fusion_prereg.json, rows of usplat/view_fusion.py, evaluation scripts/eval_fusion.py,
driver scripts/fusion_tab1.sh). CPU only: it reads eval_fusion.py records, never a scene.

Records of one dataset live in one directory OUT (the prereg's datasets.<d>.out):
    eval_val.json, eval_test.json     eval_fusion.py (validation: fit + evaluation; test)
    selection_val.json                `select` (a usplat.runlog record)
    look_test.json                    `look` (one entry per test attempt)
    table_fusion.json, table_fusion.tex   `table`
and `claim` reads the table records of both datasets -> claim_fusion.json.

    python3 scripts/fusion_select.py select --prereg P --dataset D --dir OUT [--out SEL]
    python3 scripts/fusion_select.py check-test --prereg P --dataset D --dir OUT
    python3 scripts/fusion_select.py look --prereg P --dataset D --dir OUT --ts TS --log LOG
    python3 scripts/fusion_select.py table --prereg P --dataset D --dir OUT
    python3 scripts/fusion_select.py claim --prereg P --tables T1.json T2.json --out C.json

Rules (all from the prereg): `select` applies scripts/make_tables.py choose_v2 (lowest
pixel-pooled validation NLL, tie tol, first listed wins) to the prereg's choice @ug, names the
paired constant-weight row (same base) and the binding baselines, evaluates the validation gate
(recorded; binding only when validation_gate.enabled) and the reproduction check: the reference
rows against the dataset's Table I validation record (per-scene NLL and mIoU within the prereg's
tolerances; betas reported). `check-test` refuses unless the validation record is a finished fit
of the prereg's validation scenes made on a clean tree at a commit whose scripts/ usplat/
configs/ equal the working tree (make_tables.check_code), the selection was made from exactly
this record and this prereg file and the rule still gives it, the reference rows reproduce
Table I, the validation gate passed (when enabled), an earlier test attempt used the same
selection, and the test record is not finished. `table` applies claim rule C1 per metric,
`claim` the sign tests + Holm of family F3f over the pooled test scenes of both datasets.
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

REFERENCE = ("ours_P0", "ours_P2", "averaging", "2d_features")
TAB_COLS = ("miou", "acc", "ece", "nll", "brier", "aurc")
CLAIM_METRICS = ("miou", "ece", "nll", "brier", "aurc")
VARIANTS = ("all", "map_covered")
VAL, TEST = "eval_val.json", "eval_test.json"
TABLE_JSON, TABLE_TEX = "table_fusion.json", "table_fusion.tex"


def die(msg):
    raise SystemExit(msg)


# ------------------------------------------------------------------------------ prereg
def load(prereg):
    with open(prereg) as f:
        return moved_tree(json.load(f))      # frozen: paths from before the 2026-10-31 move


def view(p, dataset):
    if dataset not in (p.get("datasets") or {}):
        die(f"refused: dataset {dataset!r} not in the prereg's datasets "
            f"{sorted(p.get('datasets') or {})}")
    return dict(p, scenes=p["datasets"][dataset]["scenes"])


def prereg_rows(p):
    return list(p["rows"]["reference"]) + [f["name"] for f in p["forms"]]


def _raw(path):
    with open(path) as f:
        return moved_tree(json.load(f))


def _body(raw):
    return raw["metrics"] if isinstance(raw.get("metrics"), dict) else raw


def finished(path):
    return os.path.exists(path) and _raw(path).get("status", "ok") == "ok"


def check_forms(p):
    """The prereg's forms must be the code's (usplat.view_fusion.FORMS, in order)."""
    from usplat import view_fusion as vf
    code = [f.spec() for f in vf.FORMS]
    if p["forms"] != code:
        die(f"refused: the prereg's forms {p['forms']} differ from usplat.view_fusion.FORMS {code}")
    return vf


def _record(p, d, stage):
    path = os.path.join(d, VAL if stage == "val" else TEST)
    if not finished(path):
        die(f"refused: {path} is not a finished {stage} record (run STAGE={stage})")
    raw = _raw(path)
    body = mt.body(path)
    want = prereg_rows(p)
    for s, sc in body["scenes"].items():
        miss = [r for r in want if r not in sc["rows"]]
        if miss:
            die(f"refused: {path} scene {s} lacks the rows {miss}")
        for r in want:
            mv = [v for v in VARIANTS if v not in sc["rows"][r]]
            if mv:
                die(f"refused: {path} scene {s} row {r} lacks the variants {mv}")
    vf = check_forms(p)
    try:
        vf.check_fusion_record(body.get("fusion"), vf.FORMS)
    except ValueError as e:
        die(f"refused: {path}: {e}")
    return path, raw, body


# -------------------------------------------------------------------------- selection
def _resolve(p, chosen, name):
    """'@paired_const' -> the constant-weight row on the base of the chosen @ug row."""
    if name == "@paired_const":
        return p["paired_const"][chosen["@ug"]]
    if name.startswith("@"):
        return chosen[name]
    return name


def binding(p, chosen):
    return [_resolve(p, chosen, b) for b in p["claim"]["baselines"]]


def reproduction(p, dataset, body):
    """The reference rows of this validation run against the dataset's Table I validation
    record: per-scene |d NLL| and |d mIoU| (variant all) and |d beta|. status 'pass' when every
    difference is within the prereg's tolerances; 'fail' otherwise; 'none' without a record."""
    rp = p.get("reproduction") or {}
    path = ((p["datasets"][dataset].get("table1") or {}).get("val")) or ""
    if not path:
        return dict(record=None, status="none", text="no Table I record named in the prereg")
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.isfile(path):
        return dict(record=path, status="fail", text="the Table I validation record is missing")
    t1 = _body(_raw(path))
    out = dict(record=path, sha256=mt.sha256_file(path), rows={},
               tol=dict(nll=float(rp.get("nll_tol", 0.01)), miou=float(rp.get("miou_tol", 0.005))))
    ok = True
    for r in REFERENCE:
        b0, b1 = (t1.get("betas") or {}).get(r), (body.get("betas") or {}).get(r)
        if isinstance(b0, dict) and isinstance(b1, dict) and b0.keys() == b1.keys():
            db = max(abs(float(b0[k]) - float(b1[k])) for k in b0)
        elif b0 is not None and b1 is not None and not isinstance(b0, dict) \
                and not isinstance(b1, dict):
            db = abs(float(b0) - float(b1))
        else:
            db = float("nan")
        dn, dm = [], []
        for s in body["scenes"]:
            t = (((t1.get("scenes") or {}).get(s) or {}).get("rows") or {}).get(r)
            if t is None:
                dn.append(float("inf")); dm.append(float("inf"))
                continue
            a, c = t["all"], body["scenes"][s]["rows"][r]["all"]
            dn.append(abs(float(a["nll"]) - float(c["nll"])))
            dm.append(abs(float(a["miou"]) - float(c["miou"])))
        row_ok = max(dn) <= out["tol"]["nll"] and max(dm) <= out["tol"]["miou"]
        ok &= row_ok
        out["rows"][r] = dict(max_abs_beta_diff=db, max_abs_val_nll_diff=max(dn),
                              max_abs_val_miou_diff=max(dm), ok=bool(row_ok))
    out["status"] = "pass" if ok else "fail"
    return out


def validation_gate(p, body, chosen):
    """The chosen OURS-UG row's pooled validation NLL against each binding baseline's."""
    tol = float(p["tie"]["tol"])
    met = p["metric"]
    ug = chosen["@ug"]
    v_ug = mt.pooled(body, ug, met["name"], met.get("variant", "all"))
    comp = {}
    for b in binding(p, chosen):
        vb = mt.pooled(body, b, met["name"], met.get("variant", "all"))
        comp[b] = dict(value=vb, ug_lower_by=vb - v_ug, beats=bool(v_ug < vb - tol))
    g = p.get("validation_gate") or {}
    return dict(enabled=bool(g.get("enabled", False)), ug=ug, ug_value=v_ug,
                baselines=comp, passed=all(c["beats"] for c in comp.values()))


def selection(p, prereg_path, dataset, d):
    v = view(p, dataset)
    path, raw, body = _record(p, d, "val")
    mt.check_validation_record(raw, body, v, path)
    ch = {}
    for key in p["choices"]:
        row, vals = mt.choose_v2(body, key, v)
        ch[key] = dict(row=row, values=vals)
    chosen = {k: c["row"] for k, c in ch.items()}
    return dict(prereg=dict(path=os.path.abspath(prereg_path), sha256=mt.sha256_file(prereg_path),
                            registered=p.get("registered")),
                dataset=dataset,
                validation=dict(path=os.path.abspath(path), sha256=mt.sha256_file(path),
                                run_id=raw.get("run_id"), git=raw.get("git")),
                scenes=sorted(body["scenes"]), metric=p["metric"], tie=p["tie"], choices=ch,
                chosen=chosen, paired_const=p["paired_const"][chosen["@ug"]],
                binding=binding(p, chosen),
                validation_gate=validation_gate(p, body, chosen),
                fusion={k: dict(theta=e["theta"], params=e["params"], fit=e["fit"])
                        for k, e in body["fusion"].items()},
                reproduction=reproduction(p, dataset, body))


def cmd_select(a):
    p = load(a.prereg)
    sel = selection(p, a.prereg, a.dataset, a.dir)
    if a.out:
        from usplat.runlog import RunRecord
        with RunRecord(a.out, config=vars(a), name="fusion_select",
                       argv=[sys.argv[0]] + sys.argv[1:]) as rec:
            rec.add_input(os.path.join(a.dir, VAL))
            rec.add_input(a.prereg)
            rec.add("selection", sel)
    print(json.dumps(dict(chosen=sel["chosen"], paired_const=sel["paired_const"],
                          validation_gate=sel["validation_gate"]["passed"],
                          reproduction=sel["reproduction"]["status"])))


def _selection_record(path):
    if not os.path.exists(path):
        die(f"refused: no validation selection {path} (run STAGE=val)")
    raw = _raw(path)
    sel = _body(raw).get("selection")
    if raw.get("status", "ok") != "ok" or not sel:
        die(f"refused: {path} is not a finished selection record")
    return sel


def check_test(p, prereg_path, dataset, d):
    v = view(p, dataset)
    path, raw, body = _record(p, d, "val")
    mt.check_validation_record(raw, body, v, path)
    mt.check_code(raw, path)
    sel_path = os.path.join(d, "selection_val.json")
    sel = _selection_record(sel_path)
    if sel["prereg"]["sha256"] != mt.sha256_file(prereg_path):
        die(f"refused: {prereg_path} changed after the selection {sel_path}")
    if sel.get("dataset") != dataset:
        die(f"refused: {sel_path} is a selection of {sel.get('dataset')!r}, not {dataset!r}")
    if sel["validation"]["sha256"] != mt.sha256_file(path):
        die(f"refused: {path} changed after the selection {sel_path}")
    again = selection(p, prereg_path, dataset, d)
    if again["chosen"] != sel["chosen"]:
        die(f"refused: the rule gives {again['chosen']}, the selection record {sel['chosen']}")
    rep = again["reproduction"]
    if rep["status"] != "pass" and (p.get("reproduction") or {}).get("required", True):
        die(f"refused: the reference rows do not reproduce Table I's validation record "
            f"({rep.get('text') or rep.get('rows')}); record the gap and ask before any test run")
    gate = again["validation_gate"]
    if gate["enabled"] and not gate["passed"]:
        die(f"refused: validation gate: {gate['ug']} does not beat every binding baseline on "
            f"pooled validation NLL ({ {k: round(c['ug_lower_by'], 6) for k, c in gate['baselines'].items()} }); "
            f"the attempt ends on validation (no test look)")
    look = os.path.join(d, "look_test.json")
    if os.path.exists(look):
        lk = _raw(look)
        now = dict(chosen=sel["chosen"], prereg_sha256=sel["prereg"]["sha256"],
                   validation_sha256=sel["validation"]["sha256"])
        if {k: lk.get(k) for k in now} != now:
            die(f"refused: the test stage started before with another selection ({look}); "
                f"the selection is final")
    if finished(os.path.join(d, TEST)):
        die(f"refused: {os.path.join(d, TEST)} is a finished test record (the test split is "
            f"evaluated once)")
    return sel


def cmd_check_test(a):
    p = load(a.prereg)
    sel = check_test(p, a.prereg, a.dataset, a.dir)
    print(f"validation selection: {sel['chosen']} (paired {sel['paired_const']}); test may run")


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
                   validation_sha256=sel["validation"]["sha256"], attempts=[])
    rec.setdefault("attempts", []).append(attempt)
    with open(out + ".tmp", "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(out + ".tmp", out)
    print(f"look record {out}: attempt {len(rec['attempts'])}, chosen {sel['chosen']}")


# ------------------------------------------------------------------------------- table
def scene_values(body, row, key, variant="all"):
    return {s: float(sc["rows"][row][variant][key]) for s, sc in sorted(body["scenes"].items())}


def _mean(v):
    return sum(v.values()) / len(v) if v else float("nan")


def pooled_value(body, row, key, variant="all"):
    """Pixel-pooled value over the scenes: ECE from the summed reliability bins, NLL / Brier /
    accuracy pixel-weighted; NaN for the other metrics (mIoU, AURC are not pooled)."""
    import quant_stats as qs
    per = [sc["rows"][row][variant] for _, sc in sorted(body["scenes"].items())]
    if key == "ece":
        return qs.bins_ece(qs.pool_bins([m["reliability"] for m in per]))
    if key in ("nll", "brier", "acc"):
        n = sum(float(m["n_pixels"]) for m in per)
        return sum(float(m[key]) * float(m["n_pixels"]) for m in per) / n if n else float("nan")
    return float("nan")


def comparison(body, ours_row, base_row, key, qs, min_wins):
    """Per-scene signed differences (positive = ours better, display units), x/N, both test
    means (all pixels and map-covered pixels), and the sign test (descriptive here)."""
    a, b = scene_values(body, ours_row, key), scene_values(body, base_row, key)
    diff = qs.signed(a, b, key)
    st = qs.sign_test(list(diff.values()))
    n = len(diff)
    wins = sum(1 for x in diff.values() if x > 0)
    hb = qs.HIGHER[key]
    better = lambda x, y: (x > y) if hb else (x < y)
    ac, bc = (scene_values(body, ours_row, key, "map_covered"),
              scene_values(body, base_row, key, "map_covered"))
    out = dict(ours=ours_row, baseline=base_row, metric=key, n=n, wins=wins,
               losses=sum(1 for x in diff.values() if x < 0), mean_ours=_mean(a),
               mean_baseline=_mean(b), mean_better=bool(better(_mean(a), _mean(b))),
               enough_wins=bool(wins >= min_wins), sign_p=st["p"], diffs=diff,
               map_covered=dict(mean_ours=_mean(ac), mean_baseline=_mean(bc),
                                mean_better=bool(better(_mean(ac), _mean(bc)))))
    if key in ("ece", "nll", "brier"):                    # R2: both aggregations
        po, pb = pooled_value(body, ours_row, key), pooled_value(body, base_row, key)
        out["pooled"] = dict(ours=po, baseline=pb, better=bool(better(po, pb)))
    return out


def table(p, prereg_path, dataset, d):
    import quant_stats as qs
    sel = _selection_record(os.path.join(d, "selection_val.json"))
    if sel["prereg"]["sha256"] != mt.sha256_file(prereg_path):
        die(f"refused: {prereg_path} changed after the selection")
    path, raw, body = _record(p, d, "test")
    want = sorted(mt.prereg_scenes(view(p, dataset), "test"))
    if sorted(body["scenes"]) != want:
        die(f"refused: the test record holds {sorted(body['scenes'])}, the prereg's test scenes "
            f"are {want}")
    bp = (raw.get("config") or {}).get("betas") or ""
    if os.path.abspath(bp) != os.path.abspath(os.path.join(d, VAL)):
        die(f"refused: the test record read its parameters from {bp!r}, not {os.path.join(d, VAL)}")
    rows = {}
    for r in prereg_rows(p):
        rows[r] = {var: {k: _mean(scene_values(body, r, k, var)) for k in TAB_COLS}
                   for var in VARIANTS}
        rows[r]["pooled"] = {k: pooled_value(body, r, k) for k in ("acc", "ece", "nll", "brier")}
    chosen = sel["chosen"]
    ug = chosen["@ug"]
    min_wins = int(p["claim"]["min_wins"][dataset])
    comps = {m: {b: comparison(body, ug, b, m, qs, min_wins) for b in sel["binding"]}
             for m in CLAIM_METRICS}
    verdict = {m: all(c["mean_better"] and c["enough_wins"] for c in comps[m].values())
               for m in CLAIM_METRICS}
    # what the text may claim: C1, and for calibration also R2 (the same sign pixel-pooled)
    writable = {m: verdict[m] and all(c.get("pooled", {}).get("better", True)
                                      for c in comps[m].values()) for m in CLAIM_METRICS}
    base = next(f["base"] for f in p["forms"] if f["name"] == ug)
    desc_rows = dict(map_alone=base, two_d="2d_features", averaging="averaging")
    desc = {m: {k: comparison(body, ug, r, m, qs, min_wins) for k, r in desc_rows.items()}
            for m in CLAIM_METRICS}
    return dict(dataset=dataset, selection=dict(chosen=chosen, paired_const=sel["paired_const"],
                                                binding=sel["binding"]),
                test=dict(path=os.path.abspath(path), sha256=mt.sha256_file(path),
                          run_id=raw.get("run_id"), git=raw.get("git")),
                scenes=sorted(body["scenes"]), rows=rows,
                selected=sorted({ug} | set(sel["binding"])),
                C1=dict(ours=ug, baselines=sel["binding"], min_wins=min_wins, comparisons=comps,
                        verdict=verdict, writable=writable, headline_miou=verdict["miou"],
                        text="C1: test mean better than EACH binding baseline AND better in at "
                             "least min_wins test scenes against each (a tie is not a win); "
                             "writable: C1 and, for ECE/NLL/Brier, also better pixel-pooled "
                             "against each (quant_prereg R2)"),
                descriptive=desc)


def tex(out):
    lab = {"miou": 100, "acc": 100, "ece": 100, "nll": 1, "brier": 100, "aurc": 100}
    lines = [f"%% generated by scripts/fusion_select.py table ({out['dataset']}); every row on "
             f"test; columns per variant: all pixels | map-covered pixels"]
    for r, c in out["rows"].items():
        cells = []
        for var in VARIANTS:
            cells += [f"{c[var][k] * lab[k]:.{3 if k == 'nll' else 1}f}" for k in TAB_COLS]
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
    print(f"{a.dataset}: chosen {out['selection']['chosen']}; C1 {out['C1']['verdict']} -> {js}")


# ------------------------------------------------------------------------------- claim
def claim(p, tables):
    """Family F3f: OURS-UG vs each binding baseline x metric, sign test over the pooled test
    scenes of the prereg's datasets, Holm over the family; C2 per test."""
    import quant_stats as qs
    want = sorted(p["datasets"])
    got = sorted(t["dataset"] for t in tables)
    if got != want:
        die(f"refused: claim needs the tables of {want}, got {got}")
    keys = list(p["claim"]["baselines"])             # '@paired_const' resolves per dataset
    tests, ps = [], []
    for k in keys:
        for m in CLAIM_METRICS:
            diffs = []
            for t in tables:
                b = t["C1"]["baselines"][keys.index(k)]
                diffs += list(t["C1"]["comparisons"][m][b]["diffs"].values())
            st = qs.sign_test(diffs)
            tests.append(dict(baseline=k, metric=m, n=len(diffs),
                              wins=sum(1 for x in diffs if x > 0), sign_p=st["p"],
                              c1=all(t["C1"]["writable"][m] for t in tables)))
            ps.append(st["p"] if math.isfinite(st["p"]) else 1.0)
    for t, h in zip(tests, qs.holm(ps)):
        t["holm_p"] = h
        t["significant"] = bool(h < 0.05 and t["c1"])
    return dict(datasets=got, family="F3f", n_tests=len(tests), tests=tests,
                gain_claims={m: all(t["C1"]["writable"][m] for t in tables) for m in CLAIM_METRICS})


def cmd_claim(a):
    p = load(a.prereg)
    out = claim(p, [_raw(t) for t in a.tables])
    path = a.out or os.path.join(os.path.dirname(os.path.abspath(a.tables[0])), "..",
                                 "claim_fusion.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(dict(datasets=out["datasets"], gain_claims=out["gain_claims"])), "->", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["select", "check-test", "look", "table", "claim"])
    ap.add_argument("--prereg", default=os.path.join(ROOT, "configs", "fusion_prereg.json"))
    ap.add_argument("--dataset", default="")
    ap.add_argument("--dir", default="", help="the dataset's record directory (OUT)")
    ap.add_argument("--out", default="")
    ap.add_argument("--ts", default="")
    ap.add_argument("--log", default="")
    ap.add_argument("--tables", nargs="+", default=[])
    a = ap.parse_args(argv)
    if a.cmd in ("select", "check-test", "look", "table") and not (a.dataset and a.dir):
        die(f"{a.cmd} needs --dataset and --dir")
    {"select": cmd_select, "check-test": cmd_check_test, "look": cmd_look, "table": cmd_table,
     "claim": cmd_claim}[a.cmd](a)


if __name__ == "__main__":
    main()
