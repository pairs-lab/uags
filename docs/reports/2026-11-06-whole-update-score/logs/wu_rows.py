"""Markdown tables of attempt 07 records: per-scene G AP of every row, means, fused, pixel."""
import json
import math
import sys

d, stage = sys.argv[1], sys.argv[2]
scenes = sys.argv[3].split(",")
recs = {s: json.load(open(f"{d}/{stage}/{s}.json")) for s in scenes}
rows = [r for r in ("ours", "dir", "agr", "kish", "normlift", "feat", "ncc", "depth", "dispersion",
                    "feat_mean", "ncc_mean", "depth_mean") if r in recs[scenes[0]]["rows"]]
pct = lambda x: "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * float(x):.1f}"


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    xs = [float(x) for x in xs if x is not None]
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))          # population std (paper)


print(f"Gaussian AP (%), {stage}, per scene; higher is better")
print("| Row | " + " | ".join(scenes) + " | mean | std (pop.) |")
print("|---|" + "---|" * (len(scenes) + 2))
for r in rows:
    v = [recs[s]["rows"][r]["gauss_ap"] for s in scenes]
    print(f"| {r} | " + " | ".join(pct(x) for x in v) + f" | {pct(mean(v))} | {pct(std(v))} |")
print()
print("Means (%): G AP, G AUROC, P AP, P AUROC, fused G AP, fused P AP, NaN share of observed")
print("| Row | G AP | G AUROC | P AP | P AUROC | +depth G AP | +depth P AP | NaN % |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    g = lambda k: mean([recs[s]["rows"][r].get(k) for s in scenes])
    nan = mean([recs[s]["rows"][r]["gauss_n_nan"] / recs[s]["counts"]["n_observed"] for s in scenes])
    print(f"| {r} | {pct(g('gauss_ap'))} | {pct(g('gauss_auroc'))} | {pct(g('pixel_ap'))} | {pct(g('pixel_auroc'))} | "
          f"{pct(g('fused_gauss_ap')) if any('fused_gauss_ap' in recs[s]['rows'][r] for s in scenes) else '-'} | "
          f"{pct(g('fused_pixel_ap')) if any('fused_pixel_ap' in recs[s]['rows'][r] for s in scenes) else '-'} | {pct(nan)} |")
print()
print("Reproduction and checks per scene")
print("| Scene | reproduction ok | max AP/AUROC delta | capture checks | residual identity | wall s | peak MiB |")
print("|---|---|---|---|---|---|---|")
for s in scenes:
    x = recs[s]
    rep = x.get("reproduction") or {}
    print(f"| {s} | {rep.get('ok')} | {rep.get('max_ap_auroc_delta', float('nan')):.1e} | "
          f"{all(x['capture_checks'].values())} | {all(x['residual_identity'].values())} | "
          f"{x['wall_s']:.0f} | {x.get('peak_alloc_mib') or 0:.0f} |")
