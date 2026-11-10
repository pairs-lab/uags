"""
robust_select.py — attempt 09 (configs/robust_prereg.json): scene lists, the registered selection and
gate, and the test refusal.

    python3 scripts/robust_select.py scenes <replica|scannet> <val|test> <out.json>
    python3 scripts/robust_select.py select <replica_eval.json> <scannet_eval.json> <selection.json>
    python3 scripts/robust_select.py check-test <selection.json>

scenes: the stage's scene list of the Table I protocol (configs/<dataset>_<stage>.json) with each field
  path moved to results/fields_robust (the published fields are never written).
select: S(row) = mean over the two datasets of the mean validation mIoU (points); the best robust row,
  ties (0.01 point) to the smaller c0; accepted for the test iff S(best) >= S(ours_P2) + 1.0 and the
  best row is not lower than ours_P2 on either dataset. Also checks that ours_P2 on the re-mapped
  fields reproduces the published validation mIoU (results/tab1/eval_val.json,
  results/scannet_tab1/eval_val.json) within 1e-3.
check-test: exit 0 only if selection.json says accepted.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = os.path.join(REPO, "configs", "robust_prereg.json")
PUBLISHED = dict(replica=os.path.join(REPO, "results", "tab1", "eval_val.json"),
                 scannet=os.path.join(REPO, "results", "scannet_tab1", "eval_val.json"))


def scenes(dataset, stage, out):
    src = json.load(open(os.path.join(REPO, "configs", f"{dataset}_{stage}.json")))
    sub = "" if dataset == "replica" else "scannet"
    dst = os.path.join(REPO, "results", "fields_robust", sub)
    for s in src:
        s["field"] = os.path.join(dst, os.path.basename(s["field"]))
    json.dump(src, open(out, "w"), indent=1)
    print(" ".join(s["name"] for s in src))


def miou(path, row):
    """Mean over the scenes of the row's mIoU (points) and the per-scene values."""
    m = json.load(open(path))["metrics"]["scenes"]
    per = {s: 100 * v["rows"][row]["all"]["miou"] for s, v in m.items() if row in v["rows"]}
    return sum(per.values()) / len(per), per


def select(rep, sca, out):
    reg = json.load(open(PREREG))
    rows = reg["rows"]
    res = {}
    for row in ["ours_P2"] + rows:
        r, rs = miou(rep, row)
        s_, ss = miou(sca, row)
        res[row] = dict(replica=r, scannet=s_, S=(r + s_) / 2, per_scene={**rs, **ss})
    repro = {}
    for ds, path in (("replica", rep), ("scannet", sca)):
        _, mine = miou(path, "ours_P2")
        _, pub = miou(PUBLISHED[ds], "ours_P2")
        repro[ds] = max(abs(mine[s] - pub[s]) for s in mine)
    order = sorted(rows, key=lambda r: (-round(res[r]["S"], 2), float(r.split("_c")[1])))
    best = order[0]
    ours = res["ours_P2"]
    accepted = (res[best]["S"] >= ours["S"] + 1.0 and res[best]["replica"] >= ours["replica"]
                and res[best]["scannet"] >= ours["scannet"])
    out_d = dict(rule=reg["accept_for_test"], rows=res, best=best, accepted=bool(accepted),
                 reproduction_max_abs_points=repro, reproduction_ok=all(v <= 0.1 for v in repro.values()))
    json.dump(out_d, open(out, "w"), indent=1)
    for r in ["ours_P2"] + rows:
        print(f"{r:13s} Replica {res[r]['replica']:6.2f}  ScanNet {res[r]['scannet']:6.2f}  S {res[r]['S']:6.2f}")
    print(f"best {best}; accepted for the test: {accepted}; reproduction of ours_P2 (max |diff|, points): {repro}")


def check_test(path):
    d = json.load(open(path))
    if not d.get("accepted"):
        raise SystemExit("test refused: validation accepted no variant (configs/robust_prereg.json)")
    print(f"test allowed: {d['best']}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"scenes": lambda: scenes(*sys.argv[2:5]), "select": lambda: select(*sys.argv[2:5]),
     "check-test": lambda: check_test(sys.argv[2])}[cmd]()
