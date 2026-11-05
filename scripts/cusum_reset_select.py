"""
cusum_reset_select.py — selection, test gate and adoption rule of attempt 06
(configs/cusum_reset_prereg.json; per-scene records from scripts/cusum_reset_nor.py). CPU only.

  select      --prereg P --dir <out>/val --out <out>/selection.json
              eligible: mean FA <= 2 x ours'; chosen: the eligible variant with the lowest mean
              stale-answer rate (a tie within 0.1 point goes to the higher mean firing Gaussian
              AP); accepted for the test only if that mean is >= 2.0 points below ours'
  check-test  --prereg P --selection S --dir <out>/val [--repo R]
              exit 0 only if the prereg and this attempt's code are tracked and committed
              unchanged, the selection is finished, made with this prereg, reproducible from the
              validation records and ACCEPTED
  table       --prereg P --selection S --test-dir <out>/test --out-dir D
              every row per test scene and as mean +- std, the object-query split, the adoption rule
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

VARIANTS = ("C90", "C97", "C99")
ROWS = ("ours",) + VARIANTS
METRICS = ("stale_answer", "static_relabel", "fa", "rec", "kept", "gauss_ap", "pixel_ap",
           "fused_gauss_ap", "fused_pixel_ap", "rate_static", "n_resets")
CODE = ("scripts/cusum_reset_nor.py", "scripts/cusum_reset_select.py", "scripts/cusum_reset.sh",
        "usplat/cusum_reset.py", "scripts/reset_fused_nor.py", "usplat/reset_fused.py",
        "scripts/dump_nor_qual.py")
STUFF = ("wall", "floor", "ceiling")              # object-query split: query not one of these
TIE, MARGIN, FA_FACTOR, MAX_WORSE = 0.001, 0.02, 2.0, 2
EPS = 1e-12                          # float slack at the 2.0-point boundaries


def summary(recs, rows=ROWS):
    out = {}
    for r in rows:
        out[r] = {}
        for met in METRICS:
            per = {s: RS.num(rec["rows"][r][met]) for s, rec in recs.items()}
            vals = list(per.values())
            out[r][met] = dict(mean=RS.mean(vals), std=RS.std(vals), per_scene=per)
    return out


def choose(summ):
    """(chosen variant or None, accepted, table) by the prereg's selection rule: among the
    eligible variants, those within TIE of the lowest mean stale-answer rate are tied; the tied
    one with the highest mean firing Gaussian AP is chosen (then the lower rate, then the order)."""
    o = summ["ours"]
    fa_limit = FA_FACTOR * o["fa"]["mean"]
    table = []
    for v in VARIANTS:
        sa, fa, ap = (summ[v][m]["mean"] for m in ("stale_answer", "fa", "gauss_ap"))
        ok = math.isfinite(fa) and fa <= fa_limit and math.isfinite(sa)
        table.append(dict(row=v, stale_answer=sa, fa=fa, gauss_ap=ap, eligible=ok))
    elig = [r for r in table if r["eligible"]]
    best = None
    if elig:
        lo = min(r["stale_answer"] for r in elig)
        tied = [r for r in elig if r["stale_answer"] <= lo + TIE + EPS]
        key = lambda r: (r["gauss_ap"] if math.isfinite(r["gauss_ap"]) else -math.inf, -r["stale_answer"])
        best = max(tied, key=key)                        # max keeps the first of equal keys
    ours_sa = o["stale_answer"]["mean"]
    accepted = best is not None and ours_sa - best["stale_answer"] >= MARGIN - EPS
    for r in table:
        r["tied"] = bool(elig) and r["eligible"] and r["stale_answer"] <= lo + TIE + EPS
    return ((best["row"] if best else None), accepted,
            dict(rows=table, fa_limit=fa_limit, ours_stale_answer=ours_sa, ours_fa=o["fa"]["mean"],
                 margin=MARGIN, tie=TIE))


def cmd_select(a):
    prereg = json.load(open(a.prereg))
    recs = RS.load_stage(prereg, a.prereg, a.dir, "validation")
    summ = summary(recs)
    chosen, accepted, table = choose(summ)
    head = RS.git(a.repo, "rev-parse", "HEAD").stdout.strip() or None
    out = dict(status="done", chosen=chosen, accepted=accepted, rule=prereg["selection"], table=table,
               summary=summ, scenes=sorted(recs), prereg_sha256=RS.sha256_file(a.prereg),
               code={s: r["code"] for s, r in recs.items()}, git_head=head)
    tmp = f"{a.out}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, a.out)
    print(f"ours: stale-answer {table['ours_stale_answer']:.4f} FA {table['ours_fa']:.4f}")
    for row in table["rows"]:
        print(f"{row['row']:>4}: stale-answer {row['stale_answer']:.4f} FA {row['fa']:.4f} "
              f"G AP {row['gauss_ap']:.4f} {'eligible' if row['eligible'] else 'FA over the limit'}")
    print(f"chosen: {chosen}; accepted for the test: {accepted} (needs <= ours - {MARGIN:g}); wrote {a.out}")
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
            again, acc, _ = choose(summary(RS.load_stage(prereg, a.prereg, a.dir, "validation")))
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


def split_means(recs, summ, rows, metric="stale_answer"):
    """{subset: {row: mean}} over all scenes, the object-query and the stuff-query scenes."""
    q = {s: r["query"]["name"] for s, r in recs.items()}
    subsets = dict(all=list(recs), object=[s for s in recs if q[s] not in STUFF],
                   stuff=[s for s in recs if q[s] in STUFF])
    return {k: dict(n=len(ss), **{r: RS.mean([summ[r][metric]["per_scene"][s] for s in ss]) for r in rows})
            for k, ss in subsets.items()}, q


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
    summ = summary(recs, ("ours", v))
    scenes = list(prereg["scenes"]["test"])
    c = [summ[v]["stale_answer"]["per_scene"][s] for s in scenes]
    o = [summ["ours"]["stale_answer"]["per_scene"][s] for s in scenes]
    worse = sum(1 for x, y in zip(c, o) if x > y)
    lower_by = RS.mean(o) - RS.mean(c)
    fa_ok = summ[v]["fa"]["mean"] <= FA_FACTOR * summ["ours"]["fa"]["mean"]
    adopt = lower_by >= MARGIN - EPS and worse <= MAX_WORSE and fa_ok
    split, q = split_means(recs, summ, ("ours", v))
    out = dict(chosen=v, summary=summ, scenes=scenes, validation=sel["table"], split=split, queries=q,
               adoption=dict(adopt=adopt, rule=prereg["adopt"], lower_by=lower_by, worse=worse,
                             fa_chosen=summ[v]["fa"]["mean"], fa_ours=summ["ours"]["fa"]["mean"],
                             fa_ok=fa_ok, mean_chosen=RS.mean(c), mean_ours=RS.mean(o)))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "cusum_reset_test.json"), "w") as f:
        json.dump(out, f, indent=1)
    pct = lambda x: "-" if not math.isfinite(x) else f"{100 * x:.1f}"
    lines = [f"| Scene | query | ours stale | {v} stale | ours FA | {v} FA | ours G AP | {v} G AP |",
             "|---|---|---|---|---|---|---|---|"]
    for s in scenes:
        x, y = recs[s]["rows"]["ours"], recs[s]["rows"][v]
        lines.append(f"| {s} | {q[s]} | {pct(RS.num(x['stale_answer']))} | {pct(RS.num(y['stale_answer']))} | "
                     f"{pct(RS.num(x['fa']))} | {pct(RS.num(y['fa']))} | {pct(RS.num(x['gauss_ap']))} | "
                     f"{pct(RS.num(y['gauss_ap']))} |")
    lines += ["", "| Row | stale-answer % | static relabel % | FA % | Rec % | Kept % | G AP | P AP | G AP +depth |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in ("ours", v):
        sm = summ[r]
        f = lambda m: f"{pct(sm[m]['mean'])} ± {pct(sm[m]['std'])}"
        lines.append(f"| {r} | {f('stale_answer')} | {f('static_relabel')} | {f('fa')} | {f('rec')} | "
                     f"{f('kept')} | {f('gauss_ap')} | {f('pixel_ap')} | {f('fused_gauss_ap')} |")
    lines += ["", "| Subset | N | ours stale % | " + f"{v} stale % |", "|---|---|---|---|"]
    for k, d in split.items():
        lines.append(f"| {k} | {d['n']} | {pct(d['ours'])} | {pct(d[v])} |")
    ad = out["adoption"]
    lines += ["", f"Adoption rule: {prereg['adopt']}",
              f"Verdict: {'ADOPT' if adopt else 'NOT adopted (negative attempt)'} "
              f"(lower by {100 * lower_by:.1f} points, worse in {worse}/{len(scenes)}, FA ok {fa_ok})"]
    with open(os.path.join(a.out_dir, "cusum_reset_test.md"), "w") as f:
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
