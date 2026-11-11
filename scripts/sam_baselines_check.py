"""
sam_baselines_check.py — the eval_seg control and the LangSplat level selection of
configs/sam_baselines_prereg.json (scripts/sam_baselines.sh calls both).

  control <record> <reference> --rows a,b,c [--tol 0.005] [--beta-rel 0.01] [--out <json>]
      The control rows of a new scripts/eval_seg.py record against the Table I record: every
      stored per-scene metric (each float under metrics.scenes[s].rows[row]: 'all' and
      'covered', rates as fractions, NLL in nats, per-class IoU, AUSE, reliability bins) within
      --tol absolute, and every fitted temperature (metrics.betas[row], P2 parameters included)
      within --beta-rel relative. Counts (integers) are reported, not tested. Prints and writes
      the max differences; exit 2 when a check fails, or when the scenes or rows differ.
  select <replica eval_val.json> <scannet eval_val.json> --out <json>
      Per dataset, the LangSplat level (s, p, w) whose row has the highest mean mIoU ('all'
      pixels) over the validation scenes; ties -> the earlier level. NOR uses Replica's level.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys

LANGSPLAT = ("langsplat_s", "langsplat_p", "langsplat_w")


def body(rec):
    return rec.get("metrics", rec)


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str) and v.lower() == "nan":
        return float("nan")
    return None


def leaves(obj, path=""):
    """(path, value) of every number in a JSON tree ('nan' strings as NaN)."""
    if isinstance(obj, dict):
        for k in sorted(obj):
            yield from leaves(obj[k], f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}[{i}]")
    else:
        v = _num(obj)
        if v is not None:
            yield path, v


def _diff(a, b):
    if isinstance(a, float) and math.isnan(a) or isinstance(b, float) and math.isnan(b):
        both = isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)
        return 0.0 if both else math.inf
    return abs(float(a) - float(b))


def control(rec, ref, rows, tol=0.005, beta_rel=0.01):
    """-> dict(ok, max_metric_diff, max_metric_path, max_count_diff, max_beta_rel,
    max_beta_path, n_metrics, failures)."""
    a, b = body(rec), body(ref)
    out = dict(rows=list(rows), tol=tol, beta_rel=beta_rel, max_metric_diff=0.0,
               max_metric_path=None, max_count_diff=0, max_beta_rel=0.0, max_beta_path=None,
               n_metrics=0, failures=[])
    sa, sb = set(a.get("scenes") or {}), set(b.get("scenes") or {})
    if sa != sb:
        out["failures"].append(f"scenes differ: {sorted(sa)} vs reference {sorted(sb)}")
    for s in sorted(sa & sb):
        for r in rows:
            ra = a["scenes"][s]["rows"].get(r)
            rb = b["scenes"][s]["rows"].get(r)
            if ra is None or rb is None:
                out["failures"].append(f"{s}: row {r} missing ({'here' if ra is None else 'reference'})")
                continue
            la, lb = dict(leaves(ra)), dict(leaves(rb))
            if set(la) != set(lb):
                out["failures"].append(f"{s}/{r}: metric keys differ "
                                       f"({sorted(set(la) ^ set(lb))[:5]})")
            for k in sorted(set(la) & set(lb)):
                x, y = la[k], lb[k]
                d = _diff(x, y)
                if isinstance(x, int) and isinstance(y, int):
                    out["max_count_diff"] = max(out["max_count_diff"], int(d))
                    continue
                out["n_metrics"] += 1
                if d > out["max_metric_diff"]:
                    out["max_metric_diff"], out["max_metric_path"] = d, f"{s}/{r}{k}"
                if d > tol:
                    out["failures"].append(f"{s}/{r}{k}: {x} vs {y} (|diff| {d:.3g} > {tol})")
    ba, bb = a.get("betas") or {}, b.get("betas") or {}
    for r in rows:
        if r not in ba and r not in bb:
            continue
        if r not in ba or r not in bb:
            out["failures"].append(f"beta of {r} missing ({'here' if r not in ba else 'reference'})")
            continue
        for (k, x), (_, y) in zip(sorted(leaves(ba[r])), sorted(leaves(bb[r]))):
            rel = _diff(x, y) / max(abs(float(y)), 1e-12)
            if rel > out["max_beta_rel"]:
                out["max_beta_rel"], out["max_beta_path"] = rel, f"{r}{k}"
            if rel > beta_rel:
                out["failures"].append(f"beta {r}{k}: {x} vs {y} (rel {rel:.3g} > {beta_rel})")
    out["ok"] = not out["failures"]
    out["failures"] = out["failures"][:50]
    return out


def mean_miou(rec, row):
    sc = body(rec)["scenes"]
    vals = [sc[s]["rows"][row]["all"]["miou"] for s in sorted(sc)]
    return sum(vals) / len(vals)


def select(records):
    """{dataset: record} -> {dataset: dict(level, miou {row: mean})}, plus nor = replica's."""
    out = {}
    for ds, rec in records.items():
        m = {r: mean_miou(rec, r) for r in LANGSPLAT}
        best = max(LANGSPLAT, key=lambda r: (m[r], -LANGSPLAT.index(r)))   # ties: the earlier
        out[ds] = dict(level=best.split("_")[1], row=best, miou=m)
    if "replica" in out:
        out["nor"] = dict(level=out["replica"]["level"], rule="Replica's choice")
    out["rule"] = ("highest mean mIoU ('all' pixels) over the validation scenes; ties -> the "
                   "earlier of s, p, w (configs/sam_baselines_prereg.json selection)")
    return out


def _write(path, obj):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("control")
    c.add_argument("record")
    c.add_argument("reference")
    c.add_argument("--rows", required=True)
    c.add_argument("--tol", type=float, default=0.005)
    c.add_argument("--beta-rel", type=float, default=0.01)
    c.add_argument("--out", default="")
    s = sub.add_parser("select")
    s.add_argument("replica")
    s.add_argument("scannet")
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "control":
        res = control(json.load(open(args.record)), json.load(open(args.reference)),
                      [r for r in args.rows.split(",") if r], args.tol, args.beta_rel)
        res.update(record=os.path.abspath(args.record), reference=os.path.abspath(args.reference))
        _write(args.out, res)
        print(f"control {os.path.basename(args.record)} vs {args.reference}: "
              f"{'ok' if res['ok'] else 'FAILED'}; max metric |diff| {res['max_metric_diff']:.3g} "
              f"({res['max_metric_path']}), max temperature rel diff {res['max_beta_rel']:.3g} "
              f"({res['max_beta_path']}), max count diff {res['max_count_diff']}, "
              f"{res['n_metrics']} metrics")
        for f in res["failures"][:10]:
            print("  " + f)
        return 0 if res["ok"] else 2
    if os.path.exists(args.out):
        print(f"{args.out}: exists, kept (the selection is made once)")
        return 0
    res = select(dict(replica=json.load(open(args.replica)), scannet=json.load(open(args.scannet))))
    res["records"] = dict(replica=os.path.abspath(args.replica), scannet=os.path.abspath(args.scannet))
    _write(args.out, res)
    print(f"selection: Replica {res['replica']['level']}, ScanNet {res['scannet']['level']}, "
          f"NOR {res['nor']['level']} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
