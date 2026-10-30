"""
extractor_select.py — the decisions of configs/extractor_prereg.json, as code.

  select     stage A + B: read the stage-A records OUT/a_<dataset>_<extractor>.json of the
             incumbent and every candidate on every dataset (scripts/extractor_2d.py), refuse
             unfinished, smoke, foreign-prereg or mixed-commit records, run the reproduction
             gate R0 (the incumbent's per-scene 2D mIoU must equal Table I's validation
             2d_features within stage_a.reproduction.tol), then the rule of stage_b: per dataset
             D, A_D(x) = mean over D's validation scenes of the PCA-64 2D mIoU, delta_D(x) =
             A_D(x) - A_D(incumbent); the winner maximises min_D delta_D (ties within tie_tol go
             to the earlier name in 'order'); the gate passes iff that minimum >= margin.
             -> OUT/selection_a.json
  winner     print the winner if a finished selection passed the gate and still reproduces from
             its inputs under this prereg; exit 1 otherwise (stage C refuses to start)
  select-c   stage C validation choices of one dataset (scripts/make_tables.py choose: CSS alpha0,
             P3 temperature, LEGaussians lambda_u, exactly Table I's) and the consistency check
             (the winner's 2D row in eval_val.json vs its stage-A record) -> <dir>/selection_val.json
  check-test refuse the stage C test evaluation unless the selection holds (see check_test)
  look       append a test attempt to <dir>/look_test.json (every attempt: one selection)
  claim      claim C1 per dataset: the new test record vs Table I's test record -> JSON (+ table)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from usplat.paths import moved_tree                                  # noqa: E402


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load(path):
    with open(path) as f:
        return moved_tree(json.load(f))      # prereg and records predate the 2026-10-31 move


def _abs(p, root=ROOT):
    return p if os.path.isabs(p) else os.path.join(root, p)


def record_body(raw):
    return raw["metrics"] if isinstance(raw.get("metrics"), dict) else raw


def names(prereg):
    return [prereg["incumbent"]] + list(prereg["order"])


def scene_names(path):
    import yaml
    with open(path) as f:
        return sorted(s["name"] for s in yaml.safe_load(f))


def git_head(root=ROOT):
    r = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


# ------------------------------------------------------------------------------- stage A/B
def check_a_record(raw, prereg_path, dataset, extractor, val_names, path):
    if raw.get("status") != "ok":
        raise SystemExit(f"refused: {path}: status {raw.get('status')!r}, not finished")
    b = record_body(raw)
    if b.get("smoke"):
        raise SystemExit(f"refused: {path} is a smoke run")
    if b.get("dataset") != dataset or b.get("extractor") != extractor:
        raise SystemExit(f"refused: {path} is {b.get('dataset')}/{b.get('extractor')}, "
                         f"expected {dataset}/{extractor}")
    if (b.get("prereg") or {}).get("sha256") != sha256_file(prereg_path):
        raise SystemExit(f"refused: {path} was made under another prereg (sha256 "
                         f"{str((b.get('prereg') or {}).get('sha256'))[:12]})")
    if sorted(b.get("scenes") or {}) != val_names:
        raise SystemExit(f"refused: {path} scores {sorted(b.get('scenes') or {})}, the "
                         f"validation scenes are {val_names}")
    g = raw.get("git") or {}
    if g.get("error") or not g.get("commit") or g.get("dirty_tracked", True):
        raise SystemExit(f"refused: {path} names no clean commit ({g.get('error') or g.get('commit')})")
    return b, g["commit"]


def reproduction(prereg, dataset, inc_body, root=ROOT):
    """R0: the incumbent's per-scene PCA-64 2D mIoU vs Table I's validation record."""
    spec = prereg["datasets"][dataset]
    tol = float(prereg["stage_a"]["reproduction"]["tol"])
    t1 = record_body(load(_abs(spec["table1"]["val"], root)))
    rows = {}
    for s, r in inc_body["scenes"].items():
        want = t1["scenes"][s]["rows"]["2d_features"]["all"]["miou"]
        got = r["pca"]["miou"]
        rows[s] = dict(stage_a=got, table1=want, diff=got - want, ok=abs(got - want) <= tol)
    return dict(tol=tol, scenes=rows, passed=all(v["ok"] for v in rows.values()))


def rule(prereg, table):
    """table[x][d] = A_D(x). -> (winner, deltas, min-delta per candidate)."""
    inc = prereg["incumbent"]
    datasets = list(prereg["datasets"])
    deltas = {x: {d: table[x][d] - table[inc][d] for d in datasets} for x in prereg["order"]}
    score = {x: min(v.values()) for x, v in deltas.items()}
    best = max(score.values())
    tol = float(prereg["stage_b"]["tie_tol"])
    winner = next(x for x in prereg["order"] if score[x] >= best - tol)
    return winner, deltas, score


def select(args):
    prereg = load(args.prereg)
    out = dict(status="ok", prereg=dict(path=os.path.abspath(args.prereg),
                                        sha256=sha256_file(args.prereg)),
               inputs={}, reproduction={}, table={}, per_scene={}, timing={})
    commits = set()
    bodies = {}
    for d in prereg["datasets"]:
        val_names = scene_names(_abs(prereg["datasets"][d]["scenes"]["validation"]))
        for x in names(prereg):
            p = os.path.join(args.dir, f"a_{d}_{x}.json")
            if not os.path.exists(p):
                raise SystemExit(f"refused: missing stage-A record {p}")
            raw = load(p)
            b, c = check_a_record(raw, args.prereg, d, x, val_names, p)
            commits.add(c)
            bodies[(d, x)] = b
            out["inputs"][p] = sha256_file(p)
            out["table"].setdefault(x, {})[d] = b["summary"]["pca"]["miou"]
            out["per_scene"].setdefault(x, {})[d] = {s: r["pca"]["miou"] for s, r in b["scenes"].items()}
            out["timing"].setdefault(x, {})[d] = dict(
                ms_per_frame=[r["ms_per_frame_extract"] for r in b["scenes"].values()],
                peak_alloc_mib=b.get("peak_alloc_mib"))
        out["reproduction"][d] = reproduction(prereg, d, bodies[(d, prereg["incumbent"])])
    if len(commits) != 1:
        raise SystemExit(f"refused: the stage-A records come from {len(commits)} commits {sorted(commits)}")
    out["git"] = commits.pop()
    r0 = all(v["passed"] for v in out["reproduction"].values())
    winner, deltas, score = rule(prereg, out["table"])
    margin = float(prereg["stage_b"]["margin"])
    out.update(deltas=deltas, min_delta=score, winner=winner, margin=margin,
               reproduction_passed=r0, gate_passed=bool(r0 and score[winner] >= margin))
    out["decision"] = ("stage C with " + winner) if out["gate_passed"] else (
        "attempt ends on validation: " + ("the incumbent does not reproduce Table I (R0)"
                                          if not r0 else
                                          f"best min-delta {score[winner] * 100:.2f} < margin "
                                          f"{margin * 100:.2f} points") + "; LSeg stays")
    with open(args.out + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(args.out + ".tmp", args.out)
    for x in names(prereg):
        print(f"{x:>12}: " + "  ".join(f"{d} {out['table'][x][d] * 100:6.2f}"
                                       + (f" ({deltas[x][d] * 100:+.2f})" if x in deltas else "")
                                       for d in prereg["datasets"]))
    print(f"reproduction R0: {'PASS' if r0 else 'FAIL'}; winner {winner} (min delta "
          f"{score[winner] * 100:+.2f} points, margin {margin * 100:.2f}); {out['decision']}")
    return out


def winner(args):
    prereg = load(args.prereg)
    p = os.path.join(args.dir, "selection_a.json")
    if not os.path.exists(p):
        raise SystemExit(f"refused: no stage-A selection {p}")
    sel = load(p)
    if sel.get("status") != "ok" or sel["prereg"]["sha256"] != sha256_file(args.prereg):
        raise SystemExit(f"refused: {p} is unfinished or was made under another prereg")
    for path, h in sel["inputs"].items():
        if not os.path.exists(path) or sha256_file(path) != h:
            raise SystemExit(f"refused: stage-A record {path} changed after the selection")
    w, _, score = rule(prereg, sel["table"])
    if w != sel["winner"]:
        raise SystemExit(f"refused: the rule gives {w}, the selection holds {sel['winner']}")
    if not sel.get("gate_passed"):
        raise SystemExit(f"refused: stage B gate not passed ({sel.get('decision')}); the attempt ended")
    print(w)
    return w


# ------------------------------------------------------------------------------- stage C
def select_c(args):
    """Table I's validation choices on the new evaluation + the 2D consistency check."""
    import make_tables as mt
    prereg = load(args.prereg)
    w = args.extractor
    raw = load(args.val)
    b = mt.body(args.val)
    if (raw.get("config") or {}).get("extractor") != w:
        raise SystemExit(f"refused: {args.val} was evaluated with {raw.get('config', {}).get('extractor')!r}, not {w!r}")
    g = raw.get("git") or {}
    if g.get("error") or not g.get("commit") or g.get("dirty_tracked", True):
        raise SystemExit(f"refused: {args.val} names no clean commit")
    a = record_body(load(os.path.join(args.adir, f"a_{args.dataset}_{w}.json")))
    tol = float(prereg["stage_c"]["consistency_tol"])
    cons = {}
    for s, r in b["scenes"].items():
        got = r["rows"]["2d_features"]["all"]["miou"]
        want = a["scenes"][s]["pca"]["miou"]
        cons[s] = dict(eval_val=got, stage_a=want, diff=got - want, ok=abs(got - want) <= tol)
    rec = dict(status="ok", extractor=w, dataset=args.dataset, eval_val=os.path.abspath(args.val),
               eval_val_sha256=sha256_file(args.val), eval_val_git=g.get("commit"),
               prereg_sha256=sha256_file(args.prereg),
               choices={k.lstrip("@"): mt.choose(b["summary"], k) for k in mt.CHOICES},
               consistency=dict(tol=tol, scenes=cons, passed=all(v["ok"] for v in cons.values())),
               betas_rows=sorted(b.get("betas") or {}))
    with open(args.out + ".tmp", "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(args.out + ".tmp", args.out)
    print(f"stage C validation selection -> {args.out}: {rec['choices']}; 2D consistency "
          f"{'PASS' if rec['consistency']['passed'] else 'FAIL'} "
          f"({', '.join(f'{s} {v['diff'] * 100:+.2f}' for s, v in cons.items())} points)")
    return rec


def check_test(args):
    """Refuse the stage C test evaluation unless: the selection is finished, made under this
    prereg, on this eval_val.json (sha256), its choices reproduce, the 2D consistency check
    passed, the validation evaluation names a clean commit and scripts/ usplat/ configs/ equal
    that commit, no finished test record exists."""
    import make_tables as mt
    sel = load(args.sel) if os.path.exists(args.sel) else None
    if not sel or sel.get("status") != "ok":
        raise SystemExit(f"refused: no finished stage C validation selection {args.sel}")
    if sel["prereg_sha256"] != sha256_file(args.prereg):
        raise SystemExit("refused: the prereg changed after the stage C validation selection")
    if sel["eval_val_sha256"] != sha256_file(args.val):
        raise SystemExit(f"refused: {args.val} changed after the selection")
    b = mt.body(args.val)
    again = {k.lstrip("@"): mt.choose(b["summary"], k) for k in mt.CHOICES}
    if again != sel["choices"]:
        raise SystemExit(f"refused: the rules give {again}, the selection holds {sel['choices']}")
    if not sel["consistency"]["passed"]:
        raise SystemExit("refused: the 2D row of the validation evaluation does not reproduce "
                         "stage A (consistency check failed); the lead decides")
    mt.check_code(load(args.val), args.val, root=ROOT)
    if os.path.exists(args.test):
        t = load(args.test)
        if t.get("status", "ok") == "ok":
            raise SystemExit(f"refused: {args.test} holds a finished test run (the test runs once)")
    print(f"stage C selection {args.sel}: choices {sel['choices']}, validation git "
          f"{sel['eval_val_git'][:12]}")


def look(args):
    sel_sha = sha256_file(args.sel)
    rec = load(args.look) if os.path.exists(args.look) else None
    if rec is not None and rec.get("selection_sha256") != sel_sha:
        raise SystemExit(f"refused: the test stage started before with another selection "
                         f"({args.look}); the selection is final")
    rec = rec or dict(selection=os.path.abspath(args.sel), selection_sha256=sel_sha, attempts=[])
    rec["attempts"].append(dict(started=args.ts, log=args.log, git=git_head()))
    with open(args.look + ".tmp", "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(args.look + ".tmp", args.look)
    print(f"look record {args.look}: attempt {len(rec['attempts'])}")


# ------------------------------------------------------------------------------- claim
METRICS = (("miou", True), ("acc", True), ("ece", False), ("nll", False), ("brier", False),
           ("aurc", False))


def paired(new_b, old_b, row_new, row_old, metric, higher):
    """Per test scene: (new, old, new better?) for a metric (variant all)."""
    out = {}
    for s in sorted(old_b["scenes"]):
        if s not in new_b["scenes"]:
            raise SystemExit(f"refused: test scene {s} missing from the new record")
        n = new_b["scenes"][s]["rows"][row_new]["all"][metric]
        o = old_b["scenes"][s]["rows"][row_old]["all"][metric]
        out[s] = dict(new=n, old=o, win=(n > o) if higher else (n < o))
    return out


def claim(args):
    prereg = load(args.prereg)
    c = prereg["claim"]
    res = dict(prereg_sha256=sha256_file(args.prereg), datasets={})
    for d, test in zip(args.datasets, args.tests):
        new_b = record_body(load(test))
        old_b = record_body(load(_abs(prereg["datasets"][d]["table1"]["test"])))
        need = int(c["min_wins"][d])
        rows = {}
        for metric, higher in METRICS:
            pr = paired(new_b, old_b, c["row"], c["row"], metric, higher)
            mean_new = sum(v["new"] for v in pr.values()) / len(pr)
            mean_old = sum(v["old"] for v in pr.values()) / len(pr)
            wins = sum(v["win"] for v in pr.values())
            better_mean = mean_new > mean_old if higher else mean_new < mean_old
            rows[metric] = dict(mean_new=mean_new, mean_old=mean_old, wins=wins, n=len(pr),
                                need=need, holds=bool(better_mean and wins >= need), scenes=pr)
        vs_avg = {}
        for metric, higher in METRICS:
            pr = paired(new_b, new_b, c["row"], "averaging", metric, higher)
            vs_avg[metric] = dict(ours=sum(v["new"] for v in pr.values()) / len(pr),
                                  averaging=sum(v["old"] for v in pr.values()) / len(pr),
                                  ours_better_in=sum(v["win"] for v in pr.values()), n=len(pr))
        res["datasets"][d] = dict(test=os.path.abspath(test), test_sha256=sha256_file(test),
                                  C1=rows, C1_headline=rows[c["headline"]]["holds"],
                                  vs_averaging_as_is=vs_avg)
        print(f"{d}: C1 ({c['row']}, {c['headline']}) "
              f"{'HOLDS' if rows[c['headline']]['holds'] else 'does not hold'}: "
              f"{rows[c['headline']]['mean_new'] * 100:.2f} vs Table I "
              f"{rows[c['headline']]['mean_old'] * 100:.2f}, better in "
              f"{rows[c['headline']]['wins']}/{rows[c['headline']]['n']} (need {need})")
    with open(args.out + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(args.out + ".tmp", args.out)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select"); s.add_argument("--prereg", required=True)
    s.add_argument("--dir", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("winner"); s.add_argument("--prereg", required=True)
    s.add_argument("--dir", required=True)
    s = sub.add_parser("select-c"); s.add_argument("--prereg", required=True)
    s.add_argument("--dataset", required=True); s.add_argument("--extractor", required=True)
    s.add_argument("--val", required=True); s.add_argument("--adir", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("check-test"); s.add_argument("--prereg", required=True)
    s.add_argument("--sel", required=True); s.add_argument("--val", required=True)
    s.add_argument("--test", required=True)
    s = sub.add_parser("look"); s.add_argument("--sel", required=True)
    s.add_argument("--look", required=True); s.add_argument("--ts", required=True)
    s.add_argument("--log", default="")
    s = sub.add_parser("claim"); s.add_argument("--prereg", required=True)
    s.add_argument("--datasets", nargs="+", required=True)
    s.add_argument("--tests", nargs="+", required=True); s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "claim" and len(a.datasets) != len(a.tests):
        raise SystemExit("give one --tests record per --datasets name")
    return dict(select=select, winner=winner, **{"select-c": select_c, "check-test": check_test},
                look=look, claim=claim)[a.cmd](a)


if __name__ == "__main__":
    main()
