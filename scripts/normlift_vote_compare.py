"""Reproduction check and summary for the NormLift KNN-mode-voting run (configs/normlift_vote_prereg.json).

  python3 scripts/normlift_vote_compare.py tables     # val/test records of Replica and ScanNet
  python3 scripts/normlift_vote_compare.py views      # results/views_normlift_vote vs results/views

tables: for each new record (results/tab1|scannet_tab1/eval_{val,test}_normlift_vote.json), the
averaging and normlift rows must equal the v1 record (every summary metric mean and every
per-scene class IoU; the validation temperatures too); prints the max abs difference, the fitted
temperatures, and the test rows of normlift_vote next to normlift and the v1 ours_P2 (mean over
the test scenes, %: mIoU, Acc, ECE, NLL, Brier, AURC, AUSE of 1-R; plus per-scene counts of
normlift_vote vs normlift).
views: the averaging / normlift AUSE and error counts of the re-mapped k-view fields against
results/views/views_test.json, and the normlift_vote curves (AUSE relative to random needs the
counts; printed as raw AUSE and as relative AUSE where views_test.json has the random level).
"""
import json
import math
import sys

R = "/home/thanhnc/gauss_ws/src/uags/results"
sys.path.insert(0, "/home/thanhnc/gauss_ws/src/uags/scripts")
METRICS = ("miou", "acc", "ece", "nll", "brier", "aurc")


def load(path):
    m = json.load(open(path))
    return m["metrics"]


def mean(x):
    return x["mean"] if isinstance(x, dict) else x


def maxdiff_rows(a, b, rows):
    """max |a - b| over the summary means (metrics and AUSE channels) and the per-scene IoUs."""
    worst, where = 0.0, None
    for r in rows:
        for k in METRICS:
            d = abs(mean(a["summary"][r]["all"][k]) - mean(b["summary"][r]["all"][k]))
            if d > worst:
                worst, where = d, (r, k)
        for ch, v in a["summary"][r]["all"].get("ause", {}).items():
            d = abs(mean(v) - mean(b["summary"][r]["all"]["ause"][ch]))
            if d > worst:
                worst, where = d, (r, "ause/" + ch)
        for s in a["scenes"]:
            ia = a["scenes"][s]["rows"][r]["all"]["iou"]
            ib = b["scenes"][s]["rows"][r]["all"]["iou"]
            if set(ia) != set(ib):
                return float("inf"), (r, s, "classes")
            for c in ia:
                x, y = ia[c], ib[c]
                if not (math.isnan(x) and math.isnan(y)):
                    d = abs(x - y)
                    if d > worst:
                        worst, where = d, (r, s, c)
    return worst, where


def tables():
    for ds, folder in (("Replica", "tab1"), ("ScanNet", "scannet_tab1")):
        for split in ("val", "test"):
            new_p = f"{R}/{folder}/eval_{split}_normlift_vote.json"
            try:
                new = load(new_p)
            except FileNotFoundError:
                print(f"{ds} {split}: {new_p} not found")
                continue
            old = load(f"{R}/{folder}/eval_{split}.json")
            d, w = maxdiff_rows(new, old, ("averaging", "normlift"))
            print(f"{ds} {split}: repro averaging/normlift vs v1 record: max |diff| = {d:.3g} at {w}")
            if split == "val":
                bn, bo = new.get("betas", {}), old.get("betas", {})
                print(f"  temperatures: normlift_vote {bn.get('normlift_vote')}, normlift {bn.get('normlift')} "
                      f"(v1 {bo.get('normlift')}), averaging {bn.get('averaging')} (v1 {bo.get('averaging')})")
            if split == "test":
                s_new, s_old = new["summary"], old["summary"]
                hdr = "  row             " + "  ".join(f"{k:>6s}" for k in METRICS) + "   AUSE(1-R)"
                print(hdr)
                for name, src, r in (("normlift_vote", s_new, "normlift_vote"), ("normlift", s_new, "normlift"),
                                     ("averaging", s_new, "averaging"), ("ours_P2 (v1)", s_old, "ours_P2")):
                    m = src[r]["all"]
                    cells = [100 * mean(m[k]) if k != "nll" else mean(m[k]) for k in METRICS]
                    au = m.get("ause", {})
                    a = au.get("normlift_1-R", au.get("u_ale"))
                    print(f"  {name:16s}" + "  ".join(f"{c:6.2f}" for c in cells) + (f"   {100 * mean(a):6.2f}" if a else ""))
                better = sum(new["scenes"][s]["rows"]["normlift_vote"]["all"]["miou"] >
                             new["scenes"][s]["rows"]["normlift"]["all"]["miou"] for s in new["scenes"])
                print(f"  normlift_vote mIoU > normlift in {better} of {len(new['scenes'])} test scenes")


def views():
    new = json.load(open(f"{R}/views_normlift_vote/views_test.json"))
    old = json.load(open(f"{R}/views/views_test.json"))
    worst, where = 0.0, None
    for s, sc in new["scenes"].items():
        for k, kk in sc["k"].items():
            for r in ("averaging", "normlift"):
                a, b = kk["rows"][r], old["scenes"][s]["k"][k]["rows"][r]
                for key in ("n_e_red", "n_e_irr", "n_pixels"):
                    if a[key] != b[key]:
                        d = abs(a[key] - b[key])
                        if d > worst:
                            worst, where = d, (s, k, r, key)
                for ch, v in a["ause"].items():
                    for t in ("e_red", "e_irr"):
                        d = abs(v[t] - b["ause"][ch][t])
                        if d > worst:
                            worst, where = d, (s, k, r, ch, t)
    print(f"views repro averaging/normlift vs results/views/views_test.json: max |diff| = {worst:.3g} at {where}")
    summ = new["summary"]
    for k in [str(x) for x in new["ks"]]:
        for r in ("normlift", "normlift_vote"):
            row = summ[k][r]
            print(f"  M={k:>3s} {r:14s} " + "  ".join(
                f"{sig}: red {row[sig]['e_red']['mean']:.4f} irr {row[sig]['e_irr']['mean']:.4f}"
                for sig in ("normlift_epi", "normlift_1-R") if sig in row))


if __name__ == "__main__":
    {"tables": tables, "views": views}[sys.argv[1] if len(sys.argv) > 1 else "tables"]()
