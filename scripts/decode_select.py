"""
decode_select.py — selection, test gate and adoption rule of attempt 08
(configs/decoding_prereg.json; records of scripts/decode_tab1.py). CPU only.

  select      --prereg P --dir <out> --out <out>/selection.json
              S(rule) = mean over the two datasets of the mean over the selection rows of the
              mean validation mIoU (variant 'all'); chosen = the highest S among the registered
              rules (within TIE of the highest: the first in the registered order); accepted
              for the test if S(chosen) - S(base) >= MARGIN and no dataset's row mean is below base's
  check-test  --prereg P --selection S --dir <out> [--repo R]
              exit 0 only if the prereg and this attempt's code are tracked and committed
              unchanged, the selection is finished, made with this prereg, reproducible from the
              validation records and accepted, and no test record is finished yet
  table       --prereg P --selection S --dir <out> --out-dir D
              every selection row per test scene for base and the chosen rule, and the adoption
              rule: S_test(chosen) - S_test(base) >= ADOPT_MARGIN, ours_P2's mean test mIoU above
              base's on both datasets, and higher in >= MIN_WINS[dataset] of its test scenes
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
REPO = os.path.dirname(HERE)
CODE = ("scripts/decode_tab1.py", "scripts/decode_select.py", "scripts/decode_tab1.sh",
        "usplat/decoding.py")
DATASETS = ("replica", "scannet")
TIE, MARGIN, ADOPT_MARGIN = 0.0025, 0.010, 0.005
MIN_WINS = dict(replica=4, scannet=5)
OURS = "ours_P2"
EPS = 1e-12


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True)


def tracked_clean(repo, rel):
    if git(repo, "ls-files", "--error-unmatch", rel).returncode != 0:
        return f"{rel} is not tracked"
    if git(repo, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        return f"{rel} has uncommitted changes"
    return None


def rule_order(prereg):
    out = []
    for c in prereg["rules"]:
        for v in c.get("values") or [None]:
            out.append(c["rule"] if v is None else f"{c['rule']}_{v:g}")
    return out


def load_runs(prereg, prereg_path, d, stage):
    """{dataset: (run record, {rule: per-rule record body})}; refuses unfinished, foreign-prereg,
    unreproduced or incomplete runs."""
    sha = sha256_file(prereg_path)
    want = ["base"] + rule_order(prereg)
    out, bad = {}, []
    for ds in DATASETS:
        p = os.path.join(d, f"{ds}_{stage}.json")
        if not os.path.exists(p):
            bad.append(f"{ds}: {p} missing")
            continue
        r = json.load(open(p))
        if r.get("status") != "done":
            bad.append(f"{ds}: status {r.get('status')}")
        if r.get("stage") != stage:
            bad.append(f"{ds}: stage {r.get('stage')}")
        if r.get("prereg_sha256") != sha:
            bad.append(f"{ds}: made with another prereg")
        rep = r.get("reproduction") or {}
        if not rep or not all(v.get("ok") for v in rep.values()):
            bad.append(f"{ds}: the base rule did not reproduce the published record")
        if r.get("rules") != want:
            bad.append(f"{ds}: rules {r.get('rules')} != {want}")
        bodies = {}
        for k in want:
            q = (r.get("outputs") or {}).get(k)
            if not q or not os.path.exists(q):
                bad.append(f"{ds}: no record of rule {k}")
                continue
            bodies[k] = json.load(open(q))["metrics"]
        out[ds] = (r, bodies)
    if bad:
        raise SystemExit(f"{stage} runs not usable: " + "; ".join(bad))
    return out


def sel_rows(prereg, ds):
    return list(prereg["datasets"][ds]["selection_rows"])


def row_means(body, rows):
    """{row: mean over scenes of mIoU 'all'} and {row: {scene: mIoU}}."""
    per = {r: {s: float(v["rows"][r]["all"]["miou"]) for s, v in body["scenes"].items()} for r in rows}
    return {r: sum(x.values()) / len(x) for r, x in per.items()}, per


def scores(prereg, runs):
    """{rule: dict(S, by_dataset={ds: dict(mean, rows)})}."""
    out = {}
    for k in ["base"] + rule_order(prereg):
        byd = {}
        for ds in DATASETS:
            m, _ = row_means(runs[ds][1][k], sel_rows(prereg, ds))
            byd[ds] = dict(mean=sum(m.values()) / len(m), rows=m)
        out[k] = dict(S=sum(v["mean"] for v in byd.values()) / len(byd), by_dataset=byd)
    return out


def choose(prereg, sc):
    order = rule_order(prereg)
    hi = max(sc[k]["S"] for k in order)
    chosen = next(k for k in order if sc[k]["S"] >= hi - TIE - EPS)
    base = sc["base"]
    accepted = (sc[chosen]["S"] - base["S"] >= MARGIN - EPS and
                all(sc[chosen]["by_dataset"][ds]["mean"] >= base["by_dataset"][ds]["mean"] - EPS
                    for ds in DATASETS))
    return chosen, accepted


def cmd_select(a):
    prereg = json.load(open(a.prereg))
    runs = load_runs(prereg, a.prereg, a.dir, "val")
    sc = scores(prereg, runs)
    chosen, accepted = choose(prereg, sc)
    out = dict(status="done", chosen=chosen, accepted=accepted, rule=prereg["selection"], scores=sc,
               prereg_sha256=sha256_file(a.prereg), git_head=git(a.repo, "rev-parse", "HEAD").stdout.strip(),
               code={ds: runs[ds][0].get("code") for ds in DATASETS})
    tmp = f"{a.out}.tmp{os.getpid()}"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    for k, v in sc.items():
        per = ", ".join(f"{ds} {100 * v['by_dataset'][ds]['mean']:.2f}" for ds in DATASETS)
        print(f"{k:>10}: S {100 * v['S']:.2f} ({per}){'  <- chosen' if k == chosen else ''}")
    print(f"chosen: {chosen}; accepted for the test: {accepted} (needs S >= base + {100 * MARGIN:g} "
          f"points and no dataset below base); wrote {a.out}")
    return out


def cmd_check_test(a):
    bad = []
    prel = os.path.relpath(os.path.abspath(a.prereg), a.repo)
    for rel in (prel,) + CODE:
        if not os.path.exists(os.path.join(a.repo, rel)):
            bad.append(f"{rel} is missing")
            continue
        e = tracked_clean(a.repo, rel)
        if e:
            bad.append(e)
    if not os.path.exists(a.selection):
        bad.append(f"{a.selection}: no validation selection")
    else:
        sel = json.load(open(a.selection))
        if sel.get("status") != "done":
            bad.append("the selection is not finished")
        if sel.get("prereg_sha256") != sha256_file(a.prereg):
            bad.append("the selection was made with another prereg")
        try:
            prereg = json.load(open(a.prereg))
            again = choose(prereg, scores(prereg, load_runs(prereg, a.prereg, a.dir, "val")))
            if again != (sel.get("chosen"), sel.get("accepted")):
                bad.append(f"re-selection gives {again}, the file says {sel.get('chosen')} / {sel.get('accepted')}")
        except SystemExit as e:
            bad.append(str(e))
        if not sel.get("accepted"):
            bad.append("validation accepted no rule: the attempt stops before the test")
    for ds in DATASETS:
        p = os.path.join(a.dir, f"{ds}_test.json")
        if os.path.exists(p) and json.load(open(p)).get("status") == "done":
            bad.append(f"{p}: the test already ran (run once)")
    if bad:
        raise SystemExit("test refused: " + "; ".join(bad))
    print("check-test: PASS")
    return 0


def cmd_table(a):
    prereg = json.load(open(a.prereg))
    sel = json.load(open(a.selection))
    if sel.get("prereg_sha256") != sha256_file(a.prereg) or not sel.get("accepted"):
        raise SystemExit("no accepted selection made with this prereg: there is no test table")
    k = sel["chosen"]
    runs = load_runs(prereg, a.prereg, a.dir, "test")
    sc = scores(prereg, runs)
    lines = [f"Attempt 08 test (run once): chosen rule '{k}' against the paper's rule 'base'. mIoU (%), "
             f"variant all; higher is better.", ""]
    wins, ours_up = {}, {}
    for ds in DATASETS:
        rows = sel_rows(prereg, ds)
        mb, pb = row_means(runs[ds][1]["base"], rows)
        mk, pk = row_means(runs[ds][1][k], rows)
        scenes = sorted(pb[OURS])
        lines += [f"### {ds}", "| Row | " + " | ".join(scenes) + " | mean |", "|---|" + "---|" * (len(scenes) + 1)]
        for r in rows:
            for tag, per, m in (("base", pb, mb), (k, pk, mk)):
                lines.append(f"| {r} ({tag}) | " + " | ".join(f"{100 * per[r][s]:.1f}" for s in scenes)
                             + f" | {100 * m[r]:.1f} |")
        wins[ds] = sum(1 for s in scenes if pk[OURS][s] > pb[OURS][s])
        ours_up[ds] = mk[OURS] - mb[OURS]
        lines.append("")
    gain = sc[k]["S"] - sc["base"]["S"]
    adopt = (gain >= ADOPT_MARGIN - EPS and all(ours_up[ds] > EPS for ds in DATASETS)
             and all(wins[ds] >= MIN_WINS[ds] for ds in DATASETS))
    lines += [f"S_test: base {100 * sc['base']['S']:.2f}, {k} {100 * sc[k]['S']:.2f} (gain {100 * gain:+.2f} points)",
              "ours_P2: " + ", ".join(f"{ds} {100 * ours_up[ds]:+.2f} points, higher in {wins[ds]} scenes"
                                       for ds in DATASETS),
              f"Adoption rule: {prereg['adopt']}",
              f"Verdict: {'ADOPT' if adopt else 'NOT adopted (negative attempt)'}"]
    os.makedirs(a.out_dir, exist_ok=True)
    out = dict(chosen=k, scores=sc, wins=wins, ours_gain=ours_up, gain=gain, adopt=adopt,
               rule=prereg["adopt"])
    json.dump(out, open(os.path.join(a.out_dir, "decoding_test.json"), "w"), indent=1)
    open(os.path.join(a.out_dir, "decoding_test.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--prereg", required=True)
    s.add_argument("--dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--repo", default=REPO)
    c = sub.add_parser("check-test")
    c.add_argument("--prereg", required=True)
    c.add_argument("--selection", required=True)
    c.add_argument("--dir", required=True)
    c.add_argument("--repo", default=REPO)
    t = sub.add_parser("table")
    t.add_argument("--prereg", required=True)
    t.add_argument("--selection", required=True)
    t.add_argument("--dir", required=True)
    t.add_argument("--out-dir", required=True)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    return dict(select=cmd_select, **{"check-test": cmd_check_test}, table=cmd_table)[a.cmd](a)


if __name__ == "__main__":
    r = main()
    sys.exit(r if isinstance(r, int) else 0)
