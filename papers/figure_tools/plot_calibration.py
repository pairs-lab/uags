"""
plot_calibration.py — the calibration figure of the appendix (A6 of the quantitative review),
from the saved Table I records only (CPU; nothing is re-evaluated):

  (a) per-scene paired dots: ECE and NLL of every test scene for each row, one thin grey line
      per scene joining its values across the rows (the pairing), with the mean over scenes
      (black bar) and the value pooled over pixels (open diamond) beside each other (R2);
  (b) pooled reliability diagrams, one per dataset and row: accuracy against mean confidence
      in the stored 15 equal-width bins summed over the test scenes, the share of pixels per
      bin below each diagram, and both ECEs (mean over scenes / pooled) in each panel.

Identical bins (the 15 bins of usplat.seg_eval.reliability), axes and rows in every panel:
the two datasets show the same rows in the same order, the ECE panels share one y axis and
so do the NLL panels, every diagram spans [0, 1] x [0, 1], and every share strip has the same
y axis. Every scene is shown (R9). Pooled ECE = sum_b C_b |conf_b - acc_b| / sum_b C_b with
the bins summed over scenes; pooled NLL = pixel-weighted (n_pixels). Before plotting, each
scene's ECE recomputed from its bins must equal the stored ECE (scripts/quant_stats.py
ece_check), else nothing is written.

    python3 papers/figure_tools/plot_calibration.py --replica <tab1>/eval_test.json \\
        --scannet <scannet_tab1>/eval_test.json \\
        --rows ours_P2,ours_P0,averaging,f3dgs,@legs,2d_features \\
        --out papers/figs/calibration.pdf --json <logs>/quant/calibration.json

@legs is LEGaussians' row chosen on validation (make_tables.py choose --key legs, read from
the eval_val.json beside each test record; --legs overrides and is checked against it).
Fonts: render_qual.RC (serif, TrueType: pdf.fonttype 42, no Type 3 fonts; the script checks
the PDF). Colours: Okabe-Ito, the order of plot_views.py where rows coincide (ours #0072B2,
averaging #D55E00); every row also has its own marker, and the x tick labels and the panel
titles name it, so no identity is carried by colour alone. Writes <out>.pdf, a 300 dpi
<out>.png beside it, and --json with every plotted number. No paths or names in the figure.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.lines import Line2D                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))              # papers/figure_tools
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))           # plot_views
sys.path.insert(0, HERE)
import quant_stats as QS                                                   # noqa: E402

DATASETS = (("replica", "Replica"), ("scannet", "ScanNet"))
# row: (label, colour, marker); Okabe-Ito
STYLE = {"ours_P2": ("Ours P2", "#0072B2", "o"),
         "ours_P0": ("Ours P0", "#56B4E9", "s"),
         "averaging": ("Averaging", "#D55E00", "D"),
         "f3dgs": ("Feature 3DGS", "#009E73", "^"),
         "@legs": ("LEGaussians", "#CC79A7", "v"),
         "2d_features": ("2D features", "#E69F00", "P")}
INK, INK2, GRID, SCENE = "#0b0b0b", "#52514e", "#e1e0d9", "#b9b8b2"
FIG_W = 7.16                                # in, IEEE double column (figure*)


def rc():
    """render_qual.RC (serif, TrueType) plus matching y ticks and quiet axes."""
    import render_qual
    r = dict(render_qual.RC)
    r.update({"ytick.labelsize": r["xtick.labelsize"], "ytick.major.size": r["xtick.major.size"],
              "ytick.major.width": r["xtick.major.width"], "ytick.major.pad": r["xtick.major.pad"],
              "axes.titlesize": r["font.size"] + 1, "axes.labelsize": r["font.size"],
              "axes.edgecolor": "#898781", "xtick.color": "#898781", "ytick.color": "#898781",
              "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "text.color": INK,
              "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False})
    return r


def resolve_rows(rows, test_path, legs):
    """The record rows of --rows for one dataset (@legs: the validation choice)."""
    out = []
    for r in rows:
        if r == "@legs":
            val = os.path.join(os.path.dirname(test_path), "eval_val.json")
            ch = QS.choose_val(val, "@legs") if os.path.exists(val) else None
            if legs and ch and legs != ch:
                raise SystemExit(f"--legs {legs}: validation chooses {ch}")
            r = legs or ch
            if r is None:
                raise SystemExit("@legs needs --legs or an eval_val.json beside the test record")
        out.append(r)
    return out


def collect(path, rows):
    m = QS.load_record(path)
    check = QS.ece_check(m)
    scenes = list(m["scenes"])
    data = dict(source=QS.source(path), scenes=scenes, ece_bins_check=check, rows={})
    for r in rows:
        alls = [m["scenes"][s]["rows"][r]["all"] for s in scenes]
        n = np.asarray([float(a["n_pixels"]) for a in alls])
        ece = [100 * QS.num(a["ece"]) for a in alls]
        nll = [QS.num(a["nll"]) for a in alls]
        bins = QS.pool_bins([a["reliability"] for a in alls])
        cnt = np.asarray(bins["count"], float)
        data["rows"][r] = dict(
            ece=ece, nll=nll, ece_scene_mean=float(np.mean(ece)),
            ece_pooled=100 * QS.bins_ece(bins), nll_scene_mean=float(np.mean(nll)),
            nll_pooled=float(np.sum(n * np.asarray(nll)) / n.sum()), n_pixels=int(n.sum()),
            bins=dict(bins, share=[float(x) for x in 100 * cnt / cnt.sum()]))
    return data


def no_type3(pdf_path):
    with open(pdf_path, "rb") as f:
        return b"/Type3" not in f.read()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--replica", default=os.path.join(QS.RES, "tab1", "eval_test.json"))
    ap.add_argument("--scannet", default=os.path.join(QS.RES, "scannet_tab1", "eval_test.json"))
    ap.add_argument("--rows", default="ours_P2,ours_P0,averaging,f3dgs,@legs,2d_features")
    ap.add_argument("--legs", default=None, help="LEGaussians rows Replica,ScanNet (checked)")
    ap.add_argument("--out", default=os.path.join(ROOT, "papers", "figs", "calibration.pdf"))
    ap.add_argument("--json", default=os.path.join(QS.OUT, "calibration.json"))
    args = ap.parse_args(argv)
    keys = args.rows.split(",")
    bad = [k for k in keys if k not in STYLE]
    if bad:
        raise SystemExit(f"rows without a style: {bad} (known: {list(STYLE)})")
    legs = dict(zip([d for d, _ in DATASETS], args.legs.split(","))) if args.legs else {}
    paths = {"replica": args.replica, "scannet": args.scannet}
    rows = {d: resolve_rows(keys, paths[d], legs.get(d)) for d, _ in DATASETS}
    data = {d: collect(paths[d], rows[d]) for d, _ in DATASETS}
    nb = len(data["replica"]["rows"][rows["replica"][0]]["bins"]["count"])
    edges = np.linspace(0, 1, nb + 1)

    with matplotlib.rc_context(rc()):
        fig = plt.figure(figsize=(FIG_W, 5.0))
        outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.05], hspace=0.42,
                                 left=0.065, right=0.99, top=0.925, bottom=0.07)
        top = outer[0].subgridspec(1, 4, wspace=0.32)
        bot = outer[1].subgridspec(4, len(keys), height_ratios=[1, 0.32, 1, 0.32],
                                   hspace=0.12, wspace=0.16)
        x = np.arange(len(keys))
        # (a) per-scene paired dots; panels of one metric share the y axis
        lim = {}
        for metric in ("ece", "nll"):
            v = [y for d, _ in DATASETS for r in rows[d] for y in data[d]["rows"][r][metric]]
            pad = 0.05 * (max(v) - min(v))
            lim[metric] = (max(0.0, min(v) - pad), max(v) + pad)
        for j, (d, dl) in enumerate(DATASETS):
            for i, (metric, ml) in enumerate((("ece", "ECE (%)"), ("nll", "NLL (nats)"))):
                ax = fig.add_subplot(top[0, 2 * j + i])
                vals = np.array([data[d]["rows"][r][metric] for r in rows[d]])   # rows x scenes
                for s in range(vals.shape[1]):
                    ax.plot(x - 0.08, vals[:, s], color=SCENE, lw=0.4, zorder=1)
                for k, (key, r) in enumerate(zip(keys, rows[d])):
                    lab, col, mk = STYLE[key]
                    e = data[d]["rows"][r]
                    ax.scatter(np.full(vals.shape[1], x[k] - 0.08), vals[k], s=7, color=col,
                               marker=mk, lw=0, zorder=3)
                    ax.plot([x[k] - 0.2, x[k] + 0.04], [e[f"{metric}_scene_mean"]] * 2,
                            color=INK, lw=0.9, zorder=4, solid_capstyle="butt")
                    ax.scatter([x[k] + 0.2], [e[f"{metric}_pooled"]], s=12, marker="D",
                               facecolor="white", edgecolor=INK, lw=0.6, zorder=4)
                ax.set_xticks(x)
                ax.set_xticklabels([STYLE[k][0] for k in keys], rotation=40, ha="right",
                                   rotation_mode="anchor")
                ax.set_xlim(-0.5, len(keys) - 0.5)
                ax.set_ylim(*lim[metric])
                ax.yaxis.grid(True, color=GRID, lw=0.4)
                ax.set_axisbelow(True)
                ax.set_title(f"{dl} test ($N={vals.shape[1]}$): {ml}", pad=2)
        fig.legend(handles=[Line2D([], [], color=SCENE, lw=0.6, label="one test scene"),
                            Line2D([], [], color=INK, lw=0.9, label="mean of scenes"),
                            Line2D([], [], color=INK, marker="D", mfc="white", lw=0, ms=3.5,
                                   mew=0.6, label="pooled over pixels")],
                   loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 1.0),
                   handlelength=1.6, columnspacing=1.4)
        fig.text(0.005, 0.975, "(a)", fontsize=8, fontweight="bold", va="top")
        fig.text(0.005, 0.60, "(b)", fontsize=8, fontweight="bold", va="top")
        # (b) pooled reliability diagrams + share of pixels per bin
        smax = max(max(data[d]["rows"][r]["bins"]["share"]) for d, _ in DATASETS for r in rows[d])
        for j, (d, dl) in enumerate(DATASETS):
            for k, (key, r) in enumerate(zip(keys, rows[d])):
                lab, col, mk = STYLE[key]
                e = data[d]["rows"][r]
                b = e["bins"]
                ok = np.asarray(b["count"]) > 0
                conf, acc = np.asarray(b["conf"]), np.asarray(b["acc"])
                ax = fig.add_subplot(bot[2 * j, k])
                ax.plot([0, 1], [0, 1], color="#898781", lw=0.5, zorder=1)
                ax.plot(conf[ok], acc[ok], color=col, lw=0.9, marker=mk, ms=2.2, mew=0,
                        zorder=3)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect("auto")
                ax.set_xticks([0, 0.5, 1])
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels(["0", "0.5", "1"])
                ax.tick_params(labelbottom=False)
                if k:
                    ax.tick_params(labelleft=False)
                else:
                    ax.set_ylabel(f"{dl}\naccuracy")
                ax.yaxis.grid(True, color=GRID, lw=0.4)
                ax.xaxis.grid(True, color=GRID, lw=0.4)
                ax.set_axisbelow(True)
                if j == 0:
                    ax.set_title(lab, pad=2)
                ax.text(0.97, 0.03, f"ECE {e['ece_scene_mean']:.1f} (mean)\n"
                                    f"ECE {e['ece_pooled']:.1f} (pooled)",
                        transform=ax.transAxes, va="bottom", ha="right", fontsize=5.5, color=INK,
                        linespacing=1.1, bbox=dict(boxstyle="square,pad=0.15", fc="white",
                                                   ec="none", alpha=0.85), zorder=5)
                hx = fig.add_subplot(bot[2 * j + 1, k])
                hx.bar(edges[:-1], b["share"], width=np.diff(edges) * 0.86, align="edge",
                       color=col, lw=0)
                hx.set_xlim(0, 1)
                hx.set_ylim(0, smax * 1.05)
                hx.set_xticks([0, 0.5, 1])
                hx.set_xticklabels(["0", "0.5", "1"])
                hx.set_yticks([0, round(smax / 10) * 10 if smax >= 10 else round(smax)])
                if k:
                    hx.tick_params(labelleft=False)
                else:
                    hx.set_ylabel("pixels\n(%)")
                if j == 1:
                    hx.set_xlabel("confidence (max. probability)")
                else:
                    hx.tick_params(labelbottom=False)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        base = os.path.splitext(args.out)[0]
        fig.savefig(base + ".pdf", metadata={"CreationDate": None})
        fig.savefig(base + ".png", dpi=300)
        plt.close(fig)
    ok3 = no_type3(base + ".pdf")
    if not ok3:
        raise SystemExit(f"{base}.pdf holds Type 3 fonts")
    out = dict(_command="python3 papers/figure_tools/plot_calibration.py " + " ".join(
                   QS.short(a) if os.path.sep in a and os.path.exists(a) else a
                   for a in (sys.argv[1:] if argv is None else argv)),
               figure=dict(pdf=QS.short(base + ".pdf"), png=QS.short(base + ".png"),
                           no_type3_fonts=ok3, width_in=FIG_W),
               rows={d: dict(zip(keys, rows[d])) for d, _ in DATASETS},
               units="ECE in points (x100), NLL in nats, share in % of the pixels",
               aggregation="mean of scenes and pooled over pixels (bins summed over scenes)",
               bin_edges=[float(e) for e in edges], **data)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {QS.short(base + '.pdf')}, {QS.short(base + '.png')} and {QS.short(args.json)}")


if __name__ == "__main__":
    main()
