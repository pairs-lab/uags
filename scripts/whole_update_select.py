"""
whole_update_select.py — selection, test gate and adoption rule of attempt 07
(configs/whole_update_prereg.json; per-scene records from scripts/whole_update_nor.py). CPU only.

  select      --prereg P --dir <out>/val --out <out>/selection.json
              chosen: the variant with the highest mean Gaussian AP (a tie within 0.5 point goes
              to the higher mean fused Gaussian AP, then to the order dir, agr, kish); accepted
              for the test only if that mean is >= 3.0 points above ours'
  check-test  --prereg P --selection S --dir <out>/val [--repo R]
              exit 0 only if the prereg and this attempt's code are tracked and committed
              unchanged, the selection is finished, made with this prereg, reproducible from the
              validation records and ACCEPTED
  table       --prereg P --selection S --test-dir <out>/test --out-dir D
              every row per test scene and as mean +- std, the adoption rule (>= 3.0 points
              above ours' mean Gaussian AP and higher in >= 7 of the 10 scenes)
The record checks (finished, not smoke, this prereg, Table III reproduced, one code state) are
scripts/reset_fused_select.load_stage (attempt 03), imported.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import reset_fused_select as RS                                            # noqa: E402

VARIANTS = ("dir", "agr", "kish")
BASELINES = ("ncc_mean", "feat_mean", "depth_mean")
TABLE3 = ("ours", "normlift", "feat", "ncc", "depth", "dispersion")
METRICS = ("gauss_ap", "gauss_auroc", "pixel_ap", "pixel_auroc", "fused_gauss_ap",
           "fused_gauss_auroc", "fused_pixel_ap", "fused_pixel_auroc")
CODE = ("scripts/whole_update_nor.py", "scripts/whole_update_select.py", "scripts/whole_update.sh",
        "usplat/whole_update.py", "scripts/dump_nor_qual.py")
TIE, MARGIN, MIN_WINS = 0.005, 0.03, 7
EPS = 1e-12                          # float slack at the 3.0-point boundaries


def summary(recs, rows):
    """{row: {metric: dict(mean, std, per_scene)}}; a metric a row lacks is NaN."""
    out = {}
    for r in rows:
        out[r] = {}
        for met in METRICS:
            per = {s: RS.num(rec["rows"][r].get(met)) for s, rec in recs.items()}
            vals = list(per.values())
            out[r][met] = dict(mean=RS.mean(vals), std=RS.std(vals), per_scene=per)
    return out


def choose(summ):
    """(chosen variant or None, accepted, table) by the prereg's selection rule: the variants
    within TIE of the highest mean Gaussian AP are tied; the tied one with the highest mean fused
    Gaussian AP is chosen (then the registered order)."""
    o = summ["ours"]["gauss_ap"]["mean"]
    table = [dict(row=v, gauss_ap=summ[v]["gauss_ap"]["mean"],
                  fused_gauss_ap=summ[v]["fused_gauss_ap"]["mean"]) for v in VARIANTS]
    ok = [r for r in table if math.isfinite(r["gauss_ap"])]
    best, hi = None, None
    if ok:
        hi = max(r["gauss_ap"] for r in ok)
        tied = [r for r in ok if r["gauss_ap"] >= hi - TIE - EPS]
        key = lambda r: r["fused_gauss_ap"] if math.isfinite(r["fused_gauss_ap"]) else -math.inf
        best = max(tied, key=key)                        # max keeps the first of equal keys
    accepted = best is not None and math.isfinite(o) and best["gauss_ap"] - o >= MARGIN - EPS
    for r in table:
        r["tied"] = hi is not None and math.isfinite(r["gauss_ap"]) and r["gauss_ap"] >= hi - TIE - EPS
        r["above_ours"] = r["gauss_ap"] - o
    return ((best["row"] if best else None), accepted,
            dict(rows=table, ours_gauss_ap=o, margin=MARGIN, tie=TIE))


def cmd_select(a):
    prereg = json.load(open(a.prereg))
    recs = RS.load_stage(prereg, a.prereg, a.dir, "validation")
    bad = [s for s, r in recs.items() if r.get("variants") != list(VARIANTS)]
    if bad:
        raise SystemExit(f"validation records must run every variant: {bad}")
    summ = summary(recs, TABLE3 + VARIANTS + BASELINES)
    chosen, accepted, table = choose(summ)
    head = RS.git(a.repo, "rev-parse", "HEAD").stdout.strip() or None
    out = dict(status="done", chosen=chosen, accepted=accepted, rule=prereg["selection"], table=table,
               summary=summ, scenes=sorted(recs), prereg_sha256=RS.sha256_file(a.prereg),
               code={s: r["code"] for s, r in recs.items()}, git_head=head)
    tmp = f"{a.out}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, a.out)
    print(f"ours: G AP {table['ours_gauss_ap']:.4f}")
    for row in table["rows"]:
        print(f"{row['row']:>5}: G AP {row['gauss_ap']:.4f} (+depth {row['fused_gauss_ap']:.4f}) "
              f"{'+' if row['above_ours'] >= 0 else ''}{100 * row['above_ours']:.1f} points vs ours"
              f"{' tied' if row['tied'] else ''}")
    for r in ("normlift", "feat") + BASELINES:
        print(f"{r:>10}: G AP {summ[r]['gauss_ap']['mean']:.4f}")
    print(f"chosen: {chosen}; accepted for the test: {accepted} (needs >= ours + {MARGIN:g}); wrote {a.out}")
    return out


def cmd_check_test(a):
    bad = []
    prel = os.path.relpath(os.path.abspath(a.prereg), a.repo)
    for rel in (prel,) + CODE:
        if not os.path.exists(os.path.join(a.repo, rel)):
            bad.append(f"{rel} is missing")
            continue
        e = RS.tracked_clean(a.repo, rel)
        if e:
            bad.append(e)
    if not os.path.exists(a.selection):
        bad.append(f"{a.selection}: no validation selection")
    else:
        sel = json.load(open(a.selection))
        if sel.get("status") != "done":
            bad.append("the selection is not finished")
        if sel.get("prereg_sha256") != RS.sha256_file(a.prereg):
            bad.append("the selection was made with another prereg")
        try:
            prereg = json.load(open(a.prereg))
            recs = RS.load_stage(prereg, a.prereg, a.dir, "validation")
            again, acc, _ = choose(summary(recs, TABLE3 + VARIANTS + BASELINES))
            if (again, acc) != (sel.get("chosen"), sel.get("accepted")):
                bad.append(f"re-selection gives {again} / {acc}, the file says "
                           f"{sel.get('chosen')} / {sel.get('accepted')}")
        except SystemExit as e:
            bad.append(str(e))
        if not sel.get("accepted"):
            bad.append("validation did not accept a variant: the attempt stops before the test")
    if bad:
        raise SystemExit("test refused: " + "; ".join(bad))
    print("check-test: PASS")
    return 0


def cmd_table(a):
    prereg = json.load(open(a.prereg))
    sel = json.load(open(a.selection))
    if sel.get("prereg_sha256") != RS.sha256_file(a.prereg):
        raise SystemExit("the selection was made with another prereg")
    if not sel.get("accepted") or not sel.get("chosen"):
        raise SystemExit("the validation selection accepted no variant: there is no test table")
    recs = RS.load_stage(prereg, a.prereg, a.test_dir, "test")
    v = sel["chosen"]
    if any(r.get("variants") != [v] for r in recs.values()):
        raise SystemExit(f"test records must run the accepted variant {v} only")
    rows = TABLE3 + (v,) + BASELINES
    summ = summary(recs, rows)
    scenes = list(prereg["scenes"]["test"])
    c = [summ[v]["gauss_ap"]["per_scene"][s] for s in scenes]
    o = [summ["ours"]["gauss_ap"]["per_scene"][s] for s in scenes]
    wins = sum(1 for x, y in zip(c, o) if x > y)
    above = RS.mean(c) - RS.mean(o)
    adopt = above >= MARGIN - EPS and wins >= MIN_WINS
    out = dict(chosen=v, summary=summ, scenes=scenes, validation=sel["table"],
               adoption=dict(adopt=adopt, rule=prereg["adopt"], above_by=above, wins=wins,
                             mean_chosen=RS.mean(c), mean_ours=RS.mean(o)))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "whole_update_test.json"), "w") as f:
        json.dump(out, f, indent=1)
    pct = lambda x: "-" if not math.isfinite(x) else f"{100 * x:.1f}"
    g = lambda r, s, m="gauss_ap": pct(RS.num(recs[s]["rows"][r].get(m)))
    lines = [f"| Scene | ours G AP | {v} G AP | NormLift G AP | feat_mean G AP | ours+depth | {v}+depth |",
             "|---|---|---|---|---|---|---|"]
    for s in scenes:
        lines.append(f"| {s} | {g('ours', s)} | {g(v, s)} | {g('normlift', s)} | {g('feat_mean', s)} | "
                     f"{g('ours', s, 'fused_gauss_ap')} | {g(v, s, 'fused_gauss_ap')} |")
    lines += ["", "| Row | G AP | G AUROC | P AP | P AUROC | G AP +depth | P AP +depth |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        sm = summ[r]
        f = lambda m: f"{pct(sm[m]['mean'])} ± {pct(sm[m]['std'])}"
        lines.append(f"| {r} | {f('gauss_ap')} | {f('gauss_auroc')} | {f('pixel_ap')} | {f('pixel_auroc')} | "
                     f"{f('fused_gauss_ap')} | {f('fused_pixel_ap')} |")
    lines += ["", f"Adoption rule: {prereg['adopt']}",
              f"Verdict: {'ADOPT' if adopt else 'NOT adopted (negative attempt)'} "
              f"(above ours by {100 * above:.1f} points, higher in {wins}/{len(scenes)})"]
    with open(os.path.join(a.out_dir, "whole_update_test.md"), "w") as f:
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
    if a.cmd == "check-test":
        return cmd_check_test(a)
    return dict(select=cmd_select, table=cmd_table)[a.cmd](a)


if __name__ == "__main__":
    main()
