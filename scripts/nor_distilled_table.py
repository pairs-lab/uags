"""
nor_distilled_table.py — the CPU steps of configs/nor_distilled_prereg.json: select LEGaussians'
lambda_u on the validation scenes and merge the per-scene records of
scripts/eval_change_nor_distilled.py into Table III rows (mean and population std over the scenes,
usplat.seg_eval.scene_stats, as scripts/nor_tab2.py merge; cells as scripts/make_tables.py).

  python3 scripts/nor_distilled_table.py select --dir results/nor/distilled/val \
      --scenes 001,002,003,004 --rows legs_l1_feat,legs_l2_feat,legs_l4_feat --out <sel.json>
  python3 scripts/nor_distilled_table.py merge --dir results/nor/distilled/test \
      --scenes 005,...,015 --rows f3dgs_feat,legs_l2_feat \
      --labels "Feature 3DGS, feature residual;LEGaussians, feature residual" --out <m.json> --tex <rows.tex>

Refuses a scene record that is missing, unfinished, of another scene, with other settings than the
first scene's (change_quantile, k, tau_cov, undef_frac, inflate_m, near_m, backend; as scripts/nor_tab2.py
SAME), or with another update stride than its own Table III record (the stride is the scene's own: 1-3).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from usplat.seg_eval import scene_stats                                    # noqa: E402

METRICS = ("g_ap", "g_auroc", "p_ap", "p_auroc", "fa", "rec")
SAME = ("change_quantile", "k", "tau_cov", "undef_frac", "inflate_m", "near_m", "backend")


def load(dirpath, scenes):
    """{scene: metrics} of the finished records <dir>/<scene>.json, settings checked."""
    out, ref = {}, None
    for s in scenes:
        p = os.path.join(dirpath, f"{s}.json")
        if not os.path.exists(p):
            raise SystemExit(f"{p}: missing")
        r = json.load(open(p))
        if r.get("status", "ok") != "ok":
            raise SystemExit(f"{p}: status {r.get('status')}")
        m = r.get("metrics", r)
        if m["scene"] != s:
            raise SystemExit(f"{p}: scene {m['scene']}, expected {s}")
        st = {k: m["settings"][k] for k in SAME}
        cc = m.get("control_check") or {}
        refpath = (cc.get("reference") or {}).get("path")
        if refpath:                                     # the scene's own stride, as its Table III record
            want = json.load(open(refpath))["config"]["update_stride"]
            if int(m["settings"]["update_stride"]) != int(want):
                raise SystemExit(f"{p}: update stride {m['settings']['update_stride']}, its Table III record {want}")
        if ref is None:
            ref = st
        elif st != ref:
            raise SystemExit(f"{p}: settings {st} differ from {ref}")
        out[s] = dict(m, path=p)
    return out, ref


def stats(recs, row):
    """{metric: scene_stats over the scenes} of one row."""
    return {k: scene_stats([recs[s]["summary"][row][k] for s in recs]) for k in METRICS}


def cell(st, scale=100.0, digits=1):
    """scripts/make_tables.py cell(): mean{\\scriptsize$\\,\\pm\\,$std}, '--' without a mean."""
    if st is None or st.get("n", 0) == 0 or not math.isfinite(st.get("mean", float("nan"))):
        return "--"
    m, s, n = st["mean"] * scale, st.get("std", float("nan")) * scale, st["n"]
    if n < 2 or not math.isfinite(s):
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f}{{\\scriptsize$\\,\\pm\\,${s:.{digits}f}}}"


def lam(row):
    m = re.match(r"legs_l([0-9.]+)_feat$", row)
    if not m:
        raise SystemExit(f"{row}: not a legs_l<lambda>_feat row")
    return float(m.group(1))


def control_summary(recs):
    """Per scene: the control row's check against the scene's Table III record."""
    out = {}
    for s, m in recs.items():
        cc = m.get("control_check")
        out[s] = None if cc is None else dict(all_exact=cc["all_exact"], max_abs_diff=cc["max_abs_diff"],
                                              **{k: cc[k]["abs_diff"] for k in ("g_ap", "p_ap", "fa", "rec",
                                                                                "threshold", "n_observed")})
    return out


def cmd_select(a):
    rows = sorted([r for r in a.rows.split(",") if r], key=lam)
    recs, settings = load(a.dir, [s for s in a.scenes.split(",") if s])
    table = {r: stats(recs, r) for r in rows}
    for r in rows:
        if table[r]["g_ap"]["n"] != len(recs):
            raise SystemExit(f"{r}: Gaussian AP finite in {table[r]['g_ap']['n']} of {len(recs)} scenes")
    best = max(rows, key=lambda r: (round(table[r]["g_ap"]["mean"], 12), -lam(r)))   # ties: smaller
    out = dict(rule="highest mean Gaussian-level AP over the validation scenes; ties -> smaller lambda",
               scenes=list(recs), rows=rows, table=table, chosen=best, lambda_u=lam(best),
               settings=settings, control=control_summary(recs),
               records={s: m["path"] for s, m in recs.items()})
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for r in rows:
        t = table[r]
        print(f"{r:>14}: " + "  ".join(f"{k} {100 * t[k]['mean']:.1f}+-{100 * t[k]['std']:.1f}" for k in METRICS))
    print(f"chosen: {best} (lambda_u = {lam(best):g}); wrote {a.out}")
    return out


def cmd_merge(a):
    rows = [r for r in a.rows.split(",") if r]
    labels = [x for x in a.labels.split(";")] if a.labels else rows
    if len(labels) != len(rows):
        raise SystemExit("--labels: one label per row")
    recs, settings = load(a.dir, [s for s in a.scenes.split(",") if s])
    table = {r: stats(recs, r) for r in rows}
    lines = [f"% generated by scripts/nor_distilled_table.py merge --dir {a.dir} (configs/nor_distilled_prereg.json; "
             f"scenes {', '.join(recs)}); mean {{\\scriptsize +- population std}}, in %"]
    for r, lab in zip(rows, labels):
        lines.append(f"{lab} & F & " + " & ".join(cell(table[r][k]) for k in METRICS) + "\\\\")
    out = dict(scenes=list(recs), rows=rows, labels=labels, table=table, settings=settings,
               control=control_summary(recs), records={s: m["path"] for s, m in recs.items()},
               per_scene={r: {s: recs[s]["summary"][r] for s in recs} for r in rows})
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    if a.tex:
        with open(a.tex, "w") as f:
            f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("select", "merge"):
        p = sub.add_parser(name)
        p.add_argument("--dir", required=True)
        p.add_argument("--scenes", required=True, help="comma-separated scene names")
        p.add_argument("--rows", required=True, help="comma-separated rows")
        p.add_argument("--out", required=True)
        if name == "merge":
            p.add_argument("--labels", default="", help="';'-separated table labels, one per row")
            p.add_argument("--tex", default="")
    a = ap.parse_args(argv)
    return cmd_select(a) if a.cmd == "select" else cmd_merge(a)


if __name__ == "__main__":
    main()
