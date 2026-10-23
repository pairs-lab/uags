"""
reset_fused_select.py — selection, test gate and claim of the fused-reset attempt
(configs/reset_fused_prereg.json; per-scene records from scripts/reset_fused_nor.py). CPU only.

  select      --prereg P --dir <out>/val --out <out>/selection.json
              the validation rule of the prereg: among ours, V1, V2, V3 (in this order), the
              row with the lowest mean (over the validation scenes) stale-answer rate whose
              mean static-relabel cost is not above ours' + budget (0.02); ours always
              qualifies; exact ties go to the row listed first (ours first)
  check-test  --prereg P --selection S --dir <out>/val [--repo R]
              exit 0 only if the prereg and this attempt's code are tracked and committed
              unchanged at HEAD, the selection is finished, was made with this prereg and
              re-selecting from the validation records gives the same choice
  table       --prereg P --selection S --test-dir <out>/test --out-dir D
              every row per test scene and as mean +- std, the claim rule, the verdict
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
ROWS = ("ours", "V1", "V2", "V3")
METRICS = ("stale_answer", "static_relabel", "fa", "rec", "kept", "gauss_ap", "pixel_ap",
           "rate_static", "n_resets")
CODE = ("scripts/reset_fused_nor.py", "scripts/reset_fused_select.py", "scripts/reset_fused.sh",
        "usplat/reset_fused.py", "scripts/dump_nor_qual.py")


def sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def num(x):
    return float("nan") if x is None else float(x)                # "nan" strings too


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def load_stage(prereg, prereg_path, d, stage):
    """{scene: record} of every prereg scene of the stage; refuses unfinished, smoke, foreign
    prereg or mixed-code records."""
    scenes = prereg["scenes"][stage]
    sha = sha256_file(prereg_path)
    out, codes, bad = {}, set(), []
    for s in scenes:
        p = os.path.join(d, f"{s}.json")
        if not os.path.exists(p):
            bad.append(f"{s}: missing")
            continue
        r = json.load(open(p))
        if r.get("status") != "done":
            bad.append(f"{s}: status {r.get('status')}")
        if r.get("stage") != dict(validation="val", test="test")[stage]:
            bad.append(f"{s}: stage {r.get('stage')}")
        if r.get("smoke"):
            bad.append(f"{s}: smoke run")
        if r.get("prereg_sha256") != sha:
            bad.append(f"{s}: made with another prereg")
        rep = r.get("reproduction")
        if not rep or not rep.get("ok"):
            bad.append(f"{s}: no Table III reproduction")
        codes.add(json.dumps(r.get("code"), sort_keys=True))
        out[s] = r
    if len(codes) > 1:
        bad.append("records made with different code")
    if bad:
        raise SystemExit(f"{stage} records not usable: " + "; ".join(bad))
    return out


def summary(recs):
    """{row: {metric: dict(mean, std, per_scene)}}."""
    out = {}
    for r in ROWS:
        out[r] = {}
        for met in METRICS:
            per = {s: num(rec["rows"][r][met]) for s, rec in recs.items()}
            vals = list(per.values())
            out[r][met] = dict(mean=mean(vals), std=std(vals), per_scene=per)
    return out


def choose(summ, budget):
    """The validation rule (module docstring). Returns (chosen, table)."""
    limit = summ["ours"]["static_relabel"]["mean"] + budget
    table, best = [], None
    for r in ROWS:
        sa, cost = summ[r]["stale_answer"]["mean"], summ[r]["static_relabel"]["mean"]
        ok = r == "ours" or (math.isfinite(cost) and cost <= limit)
        table.append(dict(row=r, stale_answer=sa, static_relabel=cost, qualifies=ok))
        if ok and math.isfinite(sa) and (best is None or sa < best[1]):
            best = (r, sa)
    return best[0], dict(rows=table, static_relabel_limit=limit, budget=budget)


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True)


def cmd_select(a):
    prereg = json.load(open(a.prereg))
    recs = load_stage(prereg, a.prereg, a.dir, "validation")
    summ = summary(recs)
    chosen, table = choose(summ, float(prereg["selection"]["static_relabel_budget"]))
    head = git(a.repo, "rev-parse", "HEAD").stdout.strip() or None
    out = dict(status="done", chosen=chosen, rule=prereg["selection"]["rule"], table=table,
               summary=summ, scenes=sorted(recs), prereg_sha256=sha256_file(a.prereg),
               code={s: r["code"] for s, r in recs.items()}, git_head=head)
    tmp = f"{a.out}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, a.out)
    for row in table["rows"]:
        print(f"{row['row']:>4}: stale-answer {row['stale_answer']:.4f} static relabel "
              f"{row['static_relabel']:.4f} {'qualifies' if row['qualifies'] else 'over budget'}")
    print(f"chosen: {chosen} (static-relabel limit {table['static_relabel_limit']:.4f}); "
          f"wrote {a.out}")
    return out


def tracked_clean(repo, rel):
    if git(repo, "ls-files", "--error-unmatch", rel).returncode != 0:
        return f"{rel} is not tracked"
    if git(repo, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        return f"{rel} has uncommitted changes"
    return None


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
            again, _ = choose(summary(load_stage(prereg, a.prereg, a.dir, "validation")),
                              float(prereg["selection"]["static_relabel_budget"]))
            if again != sel.get("chosen"):
                bad.append(f"re-selection gives {again}, the file says {sel.get('chosen')}")
        except SystemExit as e:
            bad.append(str(e))
    if bad:
        raise SystemExit("test refused: " + "; ".join(bad))
    print("check-test: PASS")
    return 0


def sign_wins(a_vals, b_vals):
    """Scenes where a is strictly lower than b."""
    return sum(1 for x, y in zip(a_vals, b_vals) if x < y)


def cmd_table(a):
    prereg = json.load(open(a.prereg))
    sel = json.load(open(a.selection))
    if sel.get("prereg_sha256") != sha256_file(a.prereg):
        raise SystemExit("the selection was made with another prereg")
    recs = load_stage(prereg, a.prereg, a.test_dir, "test")
    summ = summary(recs)
    scenes = list(prereg["scenes"]["test"])
    chosen = sel["chosen"]
    cr = prereg["claim"]
    out = dict(chosen=chosen, summary=summ, scenes=scenes, validation=sel["table"])
    if chosen == "ours":
        out["claim"] = dict(allowed=False, reason="validation kept ours: no variant selected")
    else:
        c = [summ[chosen]["stale_answer"]["per_scene"][s] for s in scenes]
        o = [summ["ours"]["stale_answer"]["per_scene"][s] for s in scenes]
        wins = sign_wins(c, o)
        lower = mean(c) < mean(o)
        allowed = wins >= int(cr["min_wins"]) and lower
        out["claim"] = dict(allowed=allowed, wins=wins, n=len(scenes), min_wins=cr["min_wins"],
                            mean_chosen=mean(c), mean_ours=mean(o), mean_lower=lower,
                            fa_chosen=summ[chosen]["fa"]["mean"], fa_ours=summ["ours"]["fa"]["mean"],
                            static_relabel_chosen=summ[chosen]["static_relabel"]["mean"],
                            static_relabel_ours=summ["ours"]["static_relabel"]["mean"])
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "reset_fused_test.json"), "w") as f:
        json.dump(out, f, indent=1)
    pct = lambda x: "-" if not math.isfinite(x) else f"{100 * x:.1f}"
    lines = ["| Scene | " + " | ".join(f"{r} stale-ans | {r} relabel | {r} FA" for r in ROWS) + " |",
             "|---|" + "---|" * (3 * len(ROWS))]
    for s in scenes:
        cells = []
        for r in ROWS:
            x = recs[s]["rows"][r]
            cells += [pct(num(x["stale_answer"])), pct(num(x["static_relabel"])), pct(num(x["fa"]))]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines += ["", "| Row | stale-answer % | static relabel % | FA % | Rec % | Kept % | G AP | P AP |",
              "|---|---|---|---|---|---|---|---|"]
    for r in ROWS:
        sm = summ[r]
        f = lambda m: f"{pct(sm[m]['mean'])} ± {pct(sm[m]['std'])}"
        lines.append(f"| {r}{' (chosen)' if r == chosen else ''} | {f('stale_answer')} | "
                     f"{f('static_relabel')} | {f('fa')} | {f('rec')} | {f('kept')} | "
                     f"{f('gauss_ap')} | {f('pixel_ap')} |")
    cl = out["claim"]
    lines += ["", f"Claim rule: {cr['rule']}",
              f"Verdict: {'claim ALLOWED' if cl['allowed'] else 'claim NOT allowed'} ({json.dumps(cl)})"]
    with open(os.path.join(a.out_dir, "reset_fused_test.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
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
    c.add_argument("--dir", required=True, help="the validation records")
    c.add_argument("--repo", default=REPO)
    t = sub.add_parser("table")
    t.add_argument("--prereg", required=True)
    t.add_argument("--selection", required=True)
    t.add_argument("--test-dir", required=True)
    t.add_argument("--out-dir", required=True)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    return dict(select=cmd_select, table=cmd_table)[a.cmd](a) if a.cmd != "check-test" \
        else cmd_check_test(a)


if __name__ == "__main__":
    main()
