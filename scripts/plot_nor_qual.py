"""
plot_nor_qual.py — CPU side of the NeRF-Object-Removal figures and the fused-baseline report,
from the dumps of scripts/dump_nor_qual.py and the published Table III records.

  noise    the run-to-run tolerance: the largest difference of any Table III cell (Gaussian-
           and pixel-level AP and AUROC of the 7 rows, FA and Rec of the 6 tested rows)
           between a record and its repeat; tolerance = max(0.002, 3 x that difference)
  select   the frame rule and the display turn of every scene from the records, the masks and
           the cameras (no method output), checked against the outcomes fixed on 2026-10-21
  change   Fig. qual-change (N1)        query   Fig. qual-query (N2)
  teaser   Fig. 1 (layout A, one validation scene)
  numbers  the N2 numbers (stale-answer rate, delta cos) as JSON and LaTeX rows
  fused    the pre-registered fused baselines (configs/nor_fused_prereg.json): per scene,
           mean +- std, sign tests (family F2, Holm), x/10, median differences, the verdict
  grid     supplementary: the N1 layout for every dumped scene

Every number printed in a figure or caption comes from the published Table III records
(--records), never from a re-run; the dumps give only the images.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

ROWS = ("ncc", "depth", "feat", "normlift", "dispersion", "ours", "fused")
TESTED = ("ncc", "depth", "feat", "normlift", "dispersion", "ours")
NEW_FUSED = ("ncc+depth", "feat+depth", "normlift+depth", "dispersion+depth")
FUSED_ROWS = ("fused",) + NEW_FUSED
FLOOR = 0.002
VAL = ("001", "002", "003", "004")
TEST = ("005", "006", "007", "008", "009", "010", "011", "013", "014", "015")
# fixed 2026-10-21 (qual_review.md section 2), before any new render
EXPECTED_FRAMES = {"005": 201, "006": 178, "007": 270, "008": 424, "009": 354, "010": 519,
                   "011": 496, "013": 556, "014": 748, "015": 675,
                   "001": 450, "002": 373, "003": 307, "004": 266}
LANDSCAPE = ("007", "008", "014", "015")


def body(path):
    d = json.load(open(path))
    return d["metrics"] if isinstance(d.get("metrics"), dict) else d


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def clean(o):
        if isinstance(o, float):
            return o if math.isfinite(o) else str(o)
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return clean(float(o))
        return o
    with open(path, "w") as f:
        json.dump(clean(obj), f, indent=1)


# -------------------------------------------------------------------------------- noise
def table3_cells(m):
    """{cell: value} of one record: G/P AP and AUROC of the 7 rows, FA/Rec of the 6 tested."""
    c = {}
    for lvl in ("gaussian", "pixel"):
        for r in ROWS:
            for k in ("ap", "auroc"):
                c[f"{lvl}/{r}/{k}"] = float(m[lvl]["rows"][r][k])
    for r in TESTED:
        for k in ("fa", "rec"):
            c[f"operating_point/{r}/{k}"] = float(m["operating_point"][r][k])
    return c


def noise(a_path, b_path):
    a, b = body(a_path), body(b_path)
    ca, cb = table3_cells(a), table3_cells(b)
    d = {k: abs(ca[k] - cb[k]) for k in ca}
    worst = max(d, key=d.get)
    fa = [(f["index"], f["n_covered"], f["n_pos"], f["n_mask"]) for f in a["pixel"]["frames"]]
    fb = [(f["index"], f["n_covered"], f["n_pos"], f["n_mask"]) for f in b["pixel"]["frames"]]
    return dict(a=os.path.abspath(a_path), b=os.path.abspath(b_path), scene=a["scene"],
                n_cells=len(d), max_delta=d[worst], max_cell=worst, floor=FLOOR,
                tolerance=max(FLOOR, 3.0 * d[worst]),
                rule="max(0.002, 3 x the largest difference of any Table III cell)",
                views_equal=a["views"] == b["views"], frames_equal=fa == fb,
                n_resets=[a["ours"]["n_resets"], b["ours"]["n_resets"]], deltas=d)


# ------------------------------------------------------------------------------- select
def select(records, root):
    """Frame rule and turn of every scene (records: {scene: record path})."""
    import dump_nor_qual as DQ
    from usplat.datasets import NeRFObjectRemovalDataset
    out, bad = {}, []
    for s, path in sorted(records.items()):
        m = body(path)
        sroot = os.path.join(root, s)

        def mask(i):
            p = os.path.join(sroot, "masks_real", f"{i:03d}.npy")
            return np.load(p) if os.path.exists(p) else None
        n_pos = {f["index"]: f["n_pos"] for f in m["pixel"]["frames"]}
        i, info = DQ.pick_frame(m["views"]["update"], mask, n_pos)
        ds = NeRFObjectRemovalDataset(sroot, part="all")
        k = DQ.quarter_turns(ds.c2w[i])
        exp_k = 0 if s in LANDSCAPE else 1
        ok = i == EXPECTED_FRAMES.get(s) and k == exp_k
        if not ok:
            bad.append(f"{s}: frame {i} (expected {EXPECTED_FRAMES.get(s)}), turns {k} "
                       f"(expected {exp_k})")
        out[s] = dict(frame=i, quarter_turns=k, ok=ok, **info)
    return dict(scenes=out, ok=not bad, failures=bad)


# --------------------------------------------------------------------------- statistics
def sign_test(diffs):
    """Exact two-sided sign test of paired differences (ties dropped): (n_pos, n_neg, p)."""
    from scipy.stats import binomtest
    pos = int(sum(1 for d in diffs if d > 0))
    neg = int(sum(1 for d in diffs if d < 0))
    n = pos + neg
    return pos, neg, (float(binomtest(pos, n, 0.5, alternative="two-sided").pvalue)
                      if n else float("nan"))


def holm(ps):
    """Holm step-down adjusted p-values, in the input order."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def mean_std(v):
    """mean and population std (seg_eval.scene_stats), N."""
    a = np.array([x for x in v if x is not None and math.isfinite(x)], dtype=np.float64)
    return dict(mean=float(a.mean()) if a.size else float("nan"),
                std=float(a.std()) if a.size else float("nan"), n=int(a.size))


FUSED_LABEL = {"fused": "Ours fused with depth residual",
               "ncc+depth": "NCC photometric residual fused with depth residual",
               "feat+depth": "Feature residual, $k=2$, fused with depth residual",
               "normlift+depth": "NormLift reliability drop fused with depth residual",
               "dispersion+depth": "Dispersion $1-\\hat\\rho$ fused with depth residual"}
FUSED_INPUT = {"fused": "F+D", "ncc+depth": "I+D", "feat+depth": "F+D",
               "normlift+depth": "F+D", "dispersion+depth": "F+D"}
METRICS = (("gaussian", "ap"), ("gaussian", "auroc"), ("pixel", "ap"), ("pixel", "auroc"))


def fused_tables(dumps, scenes, records):
    """Per scene the 4 metrics of the 5 fused rows (re-run) and the published fused cells."""
    per = {}
    for s in scenes:
        f = json.load(open(os.path.join(dumps, s, "fused.json")))
        meta = json.load(open(os.path.join(dumps, s, "meta.json")))
        if meta.get("status") != "done" or not (meta.get("reproduction") or {}).get("ok"):
            raise SystemExit(f"{s}: the dump is not done or did not reproduce Table III")
        m = body(records[s])
        per[s] = dict(rows={r: {f"{lvl}_{k}": float(f[lvl]["rows"][r][k]) for lvl, k in METRICS}
                            for r in FUSED_ROWS},
                      common={r: float(f["gaussian"]["rows"][r]["common"]["ap"])
                              for r in FUSED_ROWS},
                      n_common=int(f["gaussian"]["common"]["n"]),
                      published_fused={f"{lvl}_{k}": float(m[lvl]["rows"]["fused"][k])
                                       for lvl, k in METRICS})
    return per


def f2_tests(per, scenes, ref="rerun"):
    """Family F2: ours+depth vs each new fused row at G AP and P AP (8 sign tests, Holm)."""
    tests = []
    for r in NEW_FUSED:
        for key in ("gaussian_ap", "pixel_ap"):
            a = [per[s]["rows"]["fused"][key] if ref == "rerun" else per[s]["published_fused"][key]
                 for s in scenes]
            b = [per[s]["rows"][r][key] for s in scenes]
            d = [x - y for x, y in zip(a, b)]
            pos, neg, p = sign_test(d)
            tests.append(dict(row=r, metric=key, wins=pos, losses=neg, ties=len(d) - pos - neg,
                              n=len(d), x_of_n=f"{pos}/{len(d)}", median_diff=float(np.median(d)),
                              mean_diff=float(np.mean(d)), sign_p=p, diffs=d))
    for t, a in zip(tests, holm([t["sign_p"] for t in tests])):
        t["holm_p"] = a
        t["significant"] = a < 0.05
    return tests


def _pm(ms, scale=100.0):
    return f"{scale * ms['mean']:.1f}{{\\scriptsize$\\,\\pm\\,${scale * ms['std']:.1f}}}"


def fused_report(dumps, val_scenes, test_scenes, records, prereg):
    pre = json.load(open(prereg))
    out = dict(prereg=os.path.abspath(prereg), prereg_registered=pre.get("registered"),
               primary="re-run values of all 5 fused rows from the same dump pass (paired); "
                       "secondary: the published Table III fused cells against the re-run "
                       "new rows (decided before the test dumps)")
    for stage, scenes in (("validation", val_scenes), ("test", test_scenes)):
        if not scenes:
            continue
        per = fused_tables(dumps, scenes, records)
        summ = {r: {key: mean_std([per[s]["rows"][r][key] for s in scenes])
                    for key in (f"{l}_{k}" for l, k in METRICS)} for r in FUSED_ROWS}
        summ["published:fused"] = {key: mean_std([per[s]["published_fused"][key] for s in scenes])
                                   for key in (f"{l}_{k}" for l, k in METRICS)}
        common = {r: mean_std([per[s]["common"][r] for s in scenes]) for r in FUSED_ROWS}
        st = dict(scenes=list(scenes), per_scene=per, summary=summ, common_set_gaussian_ap=common)
        if stage == "test":
            st["f2"] = f2_tests(per, scenes, "rerun")
            st["f2_published_fused"] = f2_tests(per, scenes, "published")
            best = max(FUSED_ROWS, key=lambda r: summ[r]["gaussian_ap"]["mean"])
            ok = best == "fused"
            g = {t["row"]: t["x_of_n"] for t in st["f2"] if t["metric"] == "gaussian_ap"}
            st["verdict"] = dict(
                highest_mean_gaussian_ap=best, claim_may_stay=ok,
                rule=pre.get("claim_rule"),
                text=(("'Ours fused with depth ranks stale Gaussians best' may stay: its test mean "
                       "Gaussian AP is the highest of the 5 fused rows; it carries x/10 against "
                       "each fused row: " + ", ".join(f"{r} {x}" for r, x in g.items()))
                      if ok else
                      (f"The claim may NOT stay: {best} has the highest test mean Gaussian AP "
                       f"({100 * summ[best]['gaussian_ap']['mean']:.1f} vs ours+depth "
                       f"{100 * summ['fused']['gaussian_ap']['mean']:.1f}); the abstract, "
                       f"introduction and results must state the measured comparison")))
        out[stage] = st
    return out


def fused_latex(rep):
    t = rep["test"]
    lines = ["%% generated by scripts/plot_nor_qual.py fused (configs/nor_fused_prereg.json); "
             "test N=%d, mean +- std (population) over scenes, %%" % len(t["scenes"]),
             "%% new rows: dump_nor_qual.py re-run; 'Ours fused' row: the published Table III "
             "record cells (re-run means in the comment)"]
    for r in FUSED_ROWS:
        s = t["summary"]["published:fused"] if r == "fused" else t["summary"][r]
        cells = " & ".join(_pm(s[f"{l}_{k}"]) for l, k in METRICS)
        lines.append(f"{FUSED_LABEL[r]} & {FUSED_INPUT[r]} & {cells} & -- & --\\\\")
        if r == "fused":
            rr = t["summary"]["fused"]
            lines.append("%% re-run: " + ", ".join(f"{l} {k} {100 * rr[f'{l}_{k}']['mean']:.2f}"
                                                   for l, k in METRICS))
    lines.append("%% F2 (ours+depth vs each, exact two-sided sign test, ties dropped, Holm over 8): "
                 "row, metric, x/10, median diff (points), sign p, Holm p")
    stat = ["\\begin{tabular}{@{}llcccc@{}}", "\\toprule",
            "Fused baseline & Level & Ours+D higher & Median diff. & Sign $p$ & Holm $p$\\\\",
            "\\midrule"]
    for x in t["f2"]:
        lvl = "Gaussian AP" if x["metric"] == "gaussian_ap" else "Pixel AP"
        lines.append(f"%% {x['row']} {x['metric']} {x['x_of_n']} {100 * x['median_diff']:+.1f} "
                     f"{x['sign_p']:.3f} {x['holm_p']:.3f}")
        stat.append(f"{FUSED_LABEL[x['row']].replace(' fused with depth residual', '').rstrip(',')} & {lvl} & "
                    f"{x['x_of_n']} & {100 * x['median_diff']:+.1f} & {x['sign_p']:.3f} & "
                    f"{x['holm_p']:.3f}\\\\")
    stat += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n", "\n".join(stat) + "\n"


def fused_scene_md(rep):
    """Per-scene markdown table of the 5 fused rows (G AP / P AP, %)."""
    out = []
    for stage in ("validation", "test"):
        if stage not in rep:
            continue
        st = rep[stage]
        out.append(f"### {stage} (N={len(st['scenes'])}), G AP / P AP in %")
        out.append("| Scene | " + " | ".join(FUSED_ROWS) + " | published fused |")
        out.append("|---" * (len(FUSED_ROWS) + 2) + "|")
        for s in st["scenes"]:
            p = st["per_scene"][s]
            cells = [f"{100 * p['rows'][r]['gaussian_ap']:.1f} / {100 * p['rows'][r]['pixel_ap']:.1f}"
                     for r in FUSED_ROWS]
            pf = p["published_fused"]
            out.append(f"| {s} | " + " | ".join(cells) +
                       f" | {100 * pf['gaussian_ap']:.1f} / {100 * pf['pixel_ap']:.1f} |")
        sm = st["summary"]
        out.append("| mean | " + " | ".join(
            f"{100 * sm[r]['gaussian_ap']['mean']:.1f} / {100 * sm[r]['pixel_ap']['mean']:.1f}"
            for r in FUSED_ROWS) + f" | {100 * sm['published:fused']['gaussian_ap']['mean']:.1f} / "
            f"{100 * sm['published:fused']['pixel_ap']['mean']:.1f} |")
        out.append("")
    return "\n".join(out) + "\n"


N2_SETS = (("before", "Before the update (mapping $\\boldsymbol\\mu^\\star$)"),
           ("nochange", "After, change test off"), ("ours", "After, ours"),
           ("avg_after", "After, averaging"))


def numbers(dumps, scenes, stage):
    per = {}
    for s in scenes:
        c = json.load(open(os.path.join(dumps, s, "contrast.json")))
        per[s] = dict(query=c["query"]["name"],
                      **{k: dict(stale_answer_rate=c["sets"][k]["stale_answer_rate"],
                                 delta_cos=c["sets"][k]["delta_cos"],
                                 n_mask_cov=c["sets"][k]["n_mask_cov"]) for k, _ in N2_SETS})
    summ = {k: {m: mean_std([per[s][k][m] for s in scenes]) for m in ("stale_answer_rate",
                                                                       "delta_cos")}
            for k, _ in N2_SETS}
    below = {f"ours<{ref}": dict(
        stale_answer_rate=sum(per[s]["ours"]["stale_answer_rate"] < per[s][ref]["stale_answer_rate"]
                              for s in scenes),
        delta_cos=sum(per[s]["ours"]["delta_cos"] < per[s][ref]["delta_cos"] for s in scenes),
        n=len(scenes)) for ref in ("nochange", "before", "avg_after")}
    return dict(stage=stage, scenes=list(scenes), per_scene=per, summary=summ, counts=below,
                disclosure="new measurement (not in Table III); defined and debugged on the "
                           "validation scenes 001-004 before the test scenes ran once",
                rule="per scene over the covered pixels of all update keyframes (360x270): "
                     "stale-answer rate = share of covered masks_real pixels whose top class "
                     "(argmax over the 101 classes, P0 render) is the query; delta cos = mean "
                     "query cosine inside masks_real minus outside")


def numbers_latex(rep):
    n = len(rep["scenes"])
    lines = [f"%% generated by scripts/plot_nor_qual.py numbers ({rep['stage']}, N={n}); "
             "stale-answer rate (%) and delta cos (x100), mean +- std (population)"]
    for k, label in N2_SETS:
        s = rep["summary"][k]
        lines.append(f"{label} & {_pm(s['stale_answer_rate'])} & {_pm(s['delta_cos'])}\\\\")
    c = rep["counts"]["ours<nochange"]
    lines.append(f"%% scenes with ours below the no-change-test field: stale-answer rate "
                 f"{c['stale_answer_rate']}/{n}, delta cos {c['delta_cos']}/{n}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- figures
OKABE = dict(blue="#0072B2", vermillion="#D55E00", green="#009E73", orange="#E69F00",
             pink="#CC79A7", sky="#56B4E9", yellow="#F0E442")
RESET = "#E69F00"                  # reset / flagged by our test (as figs/pipeline_v2.pdf)
TINT_ITEMS = (("flagged, stale", "#E69F00"), ("stale, not flagged", "#56B4E9"),
              ("flagged, static", "#D55E00"), ("not flagged, static (map dimmed)", "#7f7f7f"),
              ("no update view", "#404040"))
N1_COLS = (("map", "(a) 3D-GS map\nat the update pose"),
           ("photo", "(b) Update frame\n(object removed)"),
           ("feat", "(c) Feature\nresidual, $k=2$"),
           ("normlift", "(d) NormLift\nreliability drop"),
           ("depth", "(e) Depth\nresidual"),
           ("ours", "(f) Ours: test,\nreset, re-admission"),
           ("fused", "(g) Ours fused with\ndepth residual"),
           ("tint", "(h) Our flags at the\nlabel-free threshold"))
N2_COLS = (("photo", "(a) Update frame\n(object removed)"),
           ("before", "(b) Before the update\n(mapping $\\boldsymbol{\\mu}^{\\star}$)"),
           ("nochange", "(c) After the update,\nchange test off"),
           ("ours", "(d) After the update,\nours"),
           ("avg_after", "(e) After the update,\naveraging"))


def _rq():
    import render_qual as RQ
    return RQ


def load_dump(dump_dir):
    """(meta, rule frame index, frame npz) of one scene's dump."""
    meta = json.load(open(os.path.join(dump_dir, "meta.json")))
    if meta.get("status") != "done":
        raise SystemExit(f"{dump_dir}: dump status {meta.get('status')!r}")
    fi = int(meta["selection"]["index"])
    return meta, fi, np.load(os.path.join(dump_dir, f"frame_{fi}.npz"))


def rot(a, k):
    """k clockwise quarter turns of an (h, w[, c]) array."""
    return np.ascontiguousarray(np.rot90(np.asarray(a), -int(k), axes=(0, 1)))


def nearest(mask, h, w):
    """Nearest-neighbour resize at pixel centres (eval_seg._nearest)."""
    H, W = mask.shape
    r = np.minimum(((np.arange(h) + 0.5) * H / h).astype(int), H - 1)
    c = np.minimum(((np.arange(w) + 0.5) * W / w).astype(int), W - 1)
    return np.asarray(mask)[r][:, c]


def outline(ax, mask, dashed=False, lw=0.55):
    """Contour of a binary mask: a white line over a dark halo (visible on viridis)."""
    m = np.asarray(mask, dtype=float)
    if not (m > 0.5).any():
        return
    ax.contour(m, levels=[0.5], colors="black", linewidths=lw + 0.75)
    ax.contour(m, levels=[0.5], colors="white", linewidths=lw,
               linestyles="dashed" if dashed else "solid")


def _note(ax, text, fs=5.0):
    ax.text(0.03, 0.03, text, transform=ax.transAxes, fontsize=fs, color="white", ha="left",
            va="bottom", bbox=dict(boxstyle="square,pad=0.15", facecolor="black", alpha=0.6,
                                   linewidth=0))


def _frame(ax, colour=None, lw=0.8):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(lw if colour else 0.3)
        s.set_edgecolor(colour or "black")


def scene_maps(z, k, prefix="full"):
    """The rotated maps of one rule frame (full resolution)."""
    RQ = _rq()
    cov = z[f"{prefix}_covered"].astype(bool)
    h, w = cov.shape
    out = dict(photo=z[f"{prefix}_photo"], map=np.clip(z[f"{prefix}_map_rgb"].astype(float), 0, 1),
               tint=np.clip(z[f"{prefix}_tint"].astype(float), 0, 1), covered=cov,
               mask=nearest(z["mask_real"], h, w))
    for r in ROWS:
        out[r] = RQ.u_image(z[f"{prefix}_rank/{r}"], cov, 1.0)
    return {key: rot(v, k) for key, v in out.items()}


def grid_figure(rows, col_titles, fig_w=7.16, left=0.17, gap=0.03, row_gap=0.04,
                title_h=0.24, strip=0.42, right=0.01):
    """A figure of rows of equal-width panels; each row has its own aspect (h / w).
    rows: [dict(label, aspect, panels=[callable(ax) or None])]. Returns (fig, box, layout)."""
    import matplotlib
    from matplotlib.figure import Figure
    RQ = _rq()
    nc = len(col_titles)
    pw = (fig_w - left - right - (nc - 1) * gap) / nc
    hs = [pw * r["aspect"] for r in rows]
    H = title_h + sum(hs) + (len(rows) - 1) * row_gap + strip
    with matplotlib.rc_context(RQ.RC):
        fig = Figure(figsize=(fig_w, H))
        box = lambda x, y, bw, bh: fig.add_axes((x / fig_w, y / H, bw / fig_w, bh / H))
        y = H - title_h
        axes = []
        for ri, (r, ph) in enumerate(zip(rows, hs)):
            y -= ph
            row_axes = []
            for c, draw in enumerate(r["panels"]):
                if draw is None:
                    row_axes.append(None)
                    continue
                ax = box(left + c * (pw + gap), y, pw, ph)
                draw(ax)
                if ri == 0:
                    ax.set_title(col_titles[c], fontsize=6, pad=2.0, linespacing=0.95)
                row_axes.append(ax)
            if r.get("label"):
                fig.text(0.45 * left / fig_w, (y + ph / 2) / H, r["label"], rotation=90,
                         ha="center", va="center", fontsize=7, linespacing=0.95)
            axes.append(row_axes)
            y -= row_gap
    return fig, box, dict(pw=pw, H=H, left=left, gap=gap, strip=strip, fig_w=fig_w, axes=axes)


def _legend_patches(ax, items, ncol, loc="upper left", fs=5.5):
    from matplotlib.patches import Patch
    ax.axis("off")
    ax.legend(handles=[Patch(facecolor=c, edgecolor="black", linewidth=0.3, label=n)
                       for n, c in items], loc=loc, ncol=ncol, frameon=False, fontsize=fs,
              borderaxespad=0.0, borderpad=0.0, handlelength=0.9, handleheight=0.9,
              handletextpad=0.3, columnspacing=0.8, labelspacing=0.25)


def _cbar(fig, box, x, y, w, vmin, vmax, label, ticks=None, fmt="{:g}", cmap="viridis"):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    cax = box(x, y, w, 0.05)
    cb = fig.colorbar(ScalarMappable(Normalize(vmin, vmax), cmap), cax=cax,
                      orientation="horizontal")
    t = ticks if ticks is not None else [vmin, vmax]
    cb.set_ticks(t)
    cb.set_ticklabels([fmt.format(v) for v in t])
    cb.outline.set_linewidth(0.3)
    cax.tick_params(labelsize=5.5, length=1.5, width=0.3, pad=1.0)
    if label:
        cax.set_title(label, fontsize=5.5, pad=1.5)
    return cax


def selection_rule_n1(records):
    """N1 scenes: the test scenes with the highest, the lower-median (5th of 10, ascending)
    and the lowest pixel-level AP of ours (published records)."""
    ap = {s: float(body(records[s])["pixel"]["rows"]["ours"]["ap"]) for s in TEST}
    asc = sorted(TEST, key=lambda s: (ap[s], s))
    return [asc[-1], asc[len(asc) // 2 - 1], asc[0]], ap


def change_figure(dumps, scenes, records, out, stage_label=None, fig_w=7.16):
    """Fig. qual-change (N1). Numbers printed: the published pixel-level AP of (c)-(g) and
    FA / Rec of ours, from the records."""
    RQ = _rq()
    rows, info = [], {}
    for s in scenes:
        meta, fi, z = load_dump(os.path.join(dumps, s))
        k = int(meta["frames"][str(fi)]["quarter_turns"])
        M = scene_maps(z, k)
        rec = body(records[s])
        pap = {r: 100 * float(rec["pixel"]["rows"][r]["ap"]) for r in ROWS}
        op = rec["operating_point"]["ours"]
        info[s] = dict(frame=fi, quarter_turns=k, pixel_ap=pap, fa=100 * op["fa"],
                       rec=100 * op["rec"], selection=meta["selection"])

        def panel(key, M=M, pap=pap, op=op):
            def draw(ax):
                if key == "photo":
                    ax.imshow(M["photo"], interpolation="nearest")
                else:
                    ax.imshow(M[key], interpolation="nearest")
                if key != "map":
                    outline(ax, M["mask"])
                if key in ROWS:
                    _note(ax, f"AP {pap[key]:.1f}")
                if key == "tint":
                    _note(ax, f"FA {100 * op['fa']:.1f}, Rec {100 * op['rec']:.1f}")
                _frame(ax)
            return draw
        h, w = M["covered"].shape
        lab = s if stage_label is None else f"{s} ({stage_label})"
        rows.append(dict(label=lab, aspect=h / w, panels=[panel(c) for c, _ in N1_COLS]))
    fig, box, L = grid_figure(rows, [t for _, t in N1_COLS], strip=0.40, fig_w=fig_w)
    import matplotlib
    with matplotlib.rc_context(RQ.RC):
        x0 = L["left"] + 2 * (L["pw"] + L["gap"])
        _cbar(fig, box, x0, L["strip"] - 0.17, 2 * L["pw"], 0.0, 1.0,
              "(c)-(g): rank among the scene's update pixels", [0, 0.5, 1])
        _legend_patches(box(x0 + 2 * L["pw"] + 0.12, L["strip"] - 0.30, 1.0, 0.25),
                        [("no score", RQ.UNDEF), ("not covered", RQ.BLACK)], ncol=1)
        _legend_patches(box(L["left"] + 5 * (L["pw"] + L["gap"]), 0.0, 3 * L["pw"],
                            L["strip"] - 0.06), [(n, c) for n, c in TINT_ITEMS], ncol=2)
        fig.text(L["left"] / L["fig_w"], (L["strip"] - 0.12) / L["H"],
                 "outline: masks_real\n(dilated silhouette)", fontsize=5.5, va="center")
    paths = RQ.save(fig, out)
    return dict(scenes=list(scenes), info=info, outputs=paths)


def query_figure(dumps, scenes, records, out=None, stage_label=None, fig_w=6.0):
    """Fig. qual-query (N2): the query cosine per pixel before / after the update; one
    viridis scale per row (p5..p99 of panel (b)'s covered pixels), shared by (b)-(e)."""
    RQ = _rq()
    rows, info = [], {}
    for s in scenes:
        meta, fi, z = load_dump(os.path.join(dumps, s))
        k = int(meta["frames"][str(fi)]["quarter_turns"])
        cov = z["full_covered"].astype(bool)
        h, w = cov.shape
        b = z["full_cos/before"].astype(np.float64)
        vals = b[cov & np.isfinite(b)]
        vmin, vmax = (float(np.percentile(vals, 5)), float(np.percentile(vals, 99)))
        imgs = {}
        for key in ("before", "nochange", "ours", "avg_after"):
            c = z[f"full_cos/{key}"].astype(np.float64)
            imgs[key] = rot(RQ.u_image(c - vmin, cov, vmax - vmin), k)
        imgs["photo"] = rot(z["full_photo"], k)
        mask = rot(nearest(z["mask_real"], h, w), k)
        scales = {key: (vmin, vmax) for key in ("before", "nochange", "ours", "avg_after")}
        assert len(set(scales.values())) == 1, "N2: one colour scale per row"
        q = meta["query"]["name"]
        rec = body(records[s])
        info[s] = dict(frame=fi, quarter_turns=k, query=q, vmin=vmin, vmax=vmax,
                       rec=100 * rec["operating_point"]["ours"]["rec"])

        def panel(key, imgs=imgs, mask=mask):
            def draw(ax):
                ax.imshow(imgs[key], interpolation="nearest")
                outline(ax, mask)
                _frame(ax)
            return draw
        lab = f"{s}{'' if stage_label is None else ' (' + stage_label + ')'}\nquery: {q}"
        rows.append(dict(label=lab, aspect=imgs["photo"].shape[0] / imgs["photo"].shape[1], panels=[panel(c) for c, _ in N2_COLS],
                         scale=(vmin, vmax)))
    fig, box, L = grid_figure(rows, [t for _, t in N2_COLS], left=0.30, strip=0.30,
                              right=0.62, fig_w=fig_w)
    import matplotlib
    with matplotlib.rc_context(RQ.RC):
        y = L["H"] - 0.24
        for r, axes in zip(rows, L["axes"]):
            ph = L["pw"] * r["aspect"]
            y -= ph
            vmin, vmax = r["scale"]
            xr = L["left"] + 5 * L["pw"] + 4 * L["gap"] + 0.06
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            cax = box(xr, y + 0.1 * ph, 0.06, 0.8 * ph)
            cb = fig.colorbar(ScalarMappable(Normalize(vmin, vmax), "viridis"), cax=cax)
            cb.set_ticks([vmin, vmax])
            cb.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
            cb.outline.set_linewidth(0.3)
            cax.tick_params(labelsize=5.5, length=1.5, width=0.3, pad=1.0)
            y -= 0.04
        fig.text((L["left"] + 5 * L["pw"] + 4 * L["gap"] + 0.02) / L["fig_w"],
                 (L["H"] - 0.12) / L["H"], "query\ncosine", fontsize=5.5, va="center")
        _legend_patches(box(L["left"], 0.0, 3 * L["pw"], L["strip"] - 0.06),
                        [("not covered", RQ.BLACK), ("no feature", RQ.UNDEF)], ncol=2)
        fig.text((L["left"] + 2 * L["pw"]) / L["fig_w"], (L["strip"] - 0.12) / L["H"],
                 "outline: masks_real (dilated silhouette)\ngeometry fixed: the removed "
                 "object's Gaussians are still rendered", fontsize=5.5, va="center")
    paths = RQ.save(fig, out)
    return dict(scenes=list(scenes), info=info, outputs=paths)


def _grey(img):
    a = np.asarray(img, dtype=float)
    if a.max() > 1.5:
        a = a / 255.0
    g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return np.repeat((0.15 + 0.6 * g)[..., None], 3, axis=2)


def overlay(base_grey, p, colour, covered, strength=0.85):
    """(1 - s p) grey + s p colour where covered (p NaN or not covered: grey)."""
    from matplotlib.colors import to_rgb
    c = np.array(to_rgb(colour))
    a = strength * np.clip(np.nan_to_num(np.asarray(p, dtype=float)), 0, 1)
    a = np.where(covered, a, 0.0)[..., None]
    return (1 - a) * base_grey + a * c


def teaser_figure(dump_dir, record, out):
    """Fig. 1, layout A: one validation scene before / after the update (see the module
    docstring). Printed numbers: the record's Rec / FA of ours and its keyframe count."""
    RQ = _rq()
    import matplotlib
    meta, fi, z = load_dump(dump_dir)
    k = int(meta["frames"][str(fi)]["quarter_turns"])
    rec = body(record)
    cur = json.load(open(os.path.join(dump_dir, "curves.json")))
    cov = z["full_covered"].astype(bool)
    h, w = cov.shape
    R = lambda a: rot(a, k)
    q = meta["query"]["name"]
    photo, maprgb = z["full_photo"], np.clip(z["full_map_rgb"].astype(float), 0, 1)
    g_map, g_photo = _grey(maprgb), _grey(photo)
    stale = z["full_share/stale"].astype(float)
    gt = cov & (np.nan_to_num(stale) >= 0.5)
    ue0, ua0 = z["full_share/u_epi_before"].astype(float), z["full_share/u_ale_before"].astype(float)
    ue1 = z["full_share/u_epi_after"].astype(float)
    ok = lambda a: a[cov & np.isfinite(a)]
    epi_max = RQ.nice_up(float(np.quantile(np.concatenate([ok(ue0), ok(ue1)]), 0.99)))
    ale_max = RQ.nice_up(float(np.quantile(ok(ua0), 0.99))) if ok(ua0).size else 1.0
    pq = {s: z[f"full_pq/{s}"].astype(float) for s in ("before", "ours", "avg_before", "avg_after")}
    flag = z["full_share/flag_ours"].astype(float)
    panels = {
        "a": R(maprgb),
        "b": R(overlay(g_map, pq["avg_before"], OKABE["pink"], cov)),
        "c": R(overlay(g_map, pq["before"], OKABE["pink"], cov)),
        "d": R(RQ.u_image(ua0, cov, ale_max)),
        "e": R(RQ.u_image(ue0, cov, epi_max)),
        "f": R(photo),
        "g": R(overlay(g_photo, pq["avg_after"], OKABE["pink"], cov)),
        "h": R(overlay(g_photo, pq["ours"], OKABE["pink"], cov)),
        "i": R(overlay(g_photo, flag, RESET, cov)),
        "j": R(RQ.u_image(ue1, cov, epi_max))}
    gtR = R(gt.astype(float))
    mean = lambda a: float(np.nanmean(a[gt])) if gt.any() else float("nan")
    measured = dict(n_gt_pixels=int(gt.sum()), mean_p_avg_after=mean(pq["avg_after"]),
                    mean_p_ours_after=mean(pq["ours"]), mean_p_avg_before=mean(pq["avg_before"]),
                    mean_p_ours_before=mean(pq["before"]), mean_u_epi_before=mean(ue0),
                    mean_u_epi_after=mean(ue1), epi_max=epi_max, ale_max=ale_max)
    op = rec["operating_point"]["ours"]
    titles = {"a": "(a) Map render", "b": f"(b) Averaging: p({q})", "c": f"(c) Ours: p({q})",
              "d": "(d) Ours: $u^{\\mathrm{ale}}$", "e": "(e) Ours: $u^{\\mathrm{epi}}$",
              "f": "(f) Camera, removed", "g": f"(g) Averaging: p({q})",
              "h": f"(h) Ours: p({q})", "i": "(i) Ours: reset by test",
              "j": "(j) Ours: $u^{\\mathrm{epi}}$"}
    frame_col = {"b": OKABE["vermillion"], "g": OKABE["vermillion"], "c": OKABE["blue"],
                 "d": OKABE["blue"], "e": OKABE["blue"], "h": OKABE["blue"], "i": OKABE["blue"],
                 "j": OKABE["blue"]}
    W, left, pw, gp = 7.16, 0.17, 0.86, 0.04
    ph = pw * panels["a"].shape[0] / panels["a"].shape[1]
    title_h, strip, rgap = 0.15, 0.30, 0.17
    H = title_h + ph + rgap + ph + strip
    with matplotlib.rc_context(RQ.RC):
        from matplotlib.figure import Figure
        fig = Figure(figsize=(W, H))
        box = lambda x, y, bw, bh: fig.add_axes((x / W, y / H, bw / W, bh / H))
        ys = [H - title_h - ph, H - title_h - ph - rgap - ph]
        for r, keys in enumerate(("abcde", "fghij")):
            for c, key in enumerate(keys):
                ax = box(left + c * (pw + gp), ys[r], pw, ph)
                ax.imshow(panels[key], interpolation="nearest")
                if r == 1:
                    outline(ax, gtR, dashed=True, lw=0.5)
                ax.set_title(titles[key], fontsize=6, pad=1.5)
                _frame(ax, frame_col.get(key))
                if key == "i":
                    _note(ax, f"{100 * op['rec']:.1f}% of stale,\n{100 * op['fa']:.1f}% of static",
                          fs=4.8)
            fig.text(0.45 * left / W, (ys[r] + ph / 2) / H,
                     ("After mapping", "After the update")[r], rotation=90, ha="center",
                     va="center", fontsize=6.5)
        # colour bars under row 2
        yb = 0.12
        x = lambda c: left + c * (pw + gp)
        import matplotlib.colors as mcolors
        cm = mcolors.LinearSegmentedColormap.from_list(
            "pq", [(0.45, 0.45, 0.45), mcolors.to_rgb(OKABE["pink"])])
        _cbar(fig, box, x(1) + 0.05, yb, 2 * pw - 0.1, 0, 1, "p(query), (b, c, g, h)",
              [0, 0.5, 1], cmap=cm)
        cm2 = mcolors.LinearSegmentedColormap.from_list(
            "rs", [(0.45, 0.45, 0.45), mcolors.to_rgb(RESET)])
        _cbar(fig, box, x(3) + 0.05, yb, pw - 0.1, 0, 1, "reset share (i)", [0, 1], cmap=cm2)
        _cbar(fig, box, x(4) + 0.05, yb, pw - 0.1, 0, epi_max, "$u^{\\mathrm{epi}}$ (e, j)")
        _cbar(fig, box, x(0) + 0.05, yb, pw - 0.1, 0, ale_max, "$u^{\\mathrm{ale}}$ (d)")
        # (k) curves
        xk = left + 5 * pw + 4 * gp + 0.42
        wk = W - xk - 0.08
        ax = box(xk, ys[1] + 0.05, wk, ys[0] + ph - ys[1] - 0.05)
        xs = np.arange(len(cur["ours"]["stale_is_query"]))
        for name, col in (("averaging", OKABE["vermillion"]), ("ours", OKABE["blue"])):
            lab = "Averaging" if name == "averaging" else "Ours"
            ax.plot(xs, 100 * np.array(cur[name]["stale_is_query"]), color=col, lw=1.0,
                    label=f"{lab}: removed, top class '{q}'")
            ax.plot(xs, 100 * np.array(cur[name]["static_changed"]), color=col, lw=0.8,
                    ls="--", label=f"{lab}: static, top class changed")
        ax.set_xlim(0, xs[-1])
        ax.set_ylim(0, 100)
        ax.set_xlabel("update keyframes ingested", fontsize=6, labelpad=1)
        ax.set_ylabel("% of Gaussians", fontsize=6, labelpad=1)
        ax.tick_params(labelsize=5.5, length=1.5, width=0.3, pad=1.0)
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        ax.set_title("(k) Top class during the update", fontsize=6, pad=1.5)
        ax.legend(fontsize=4.8, frameon=False, loc="upper right", handlelength=1.6,
                  borderaxespad=0.2, labelspacing=0.25)
    paths = RQ.save(fig, out)
    return dict(scene=meta["scene"], frame=fi, quarter_turns=k, query=q, measured=measured,
                published=dict(rec=op["rec"], fa=op["fa"], n_update=len(rec["views"]["update"])),
                curves=cur, outputs=paths)


# ----------------------------------------------------------------------------- captions
def _f(x):
    return f"{x:.1f}"


def change_caption(info, records):
    s = list(info)
    n_nl = sum(float(body(records[t])["pixel"]["rows"]["normlift"]["ap"])
               > float(body(records[t])["pixel"]["rows"]["ours"]["ap"]) for t in TEST)
    nl_all = all(info[t]["pixel_ap"]["normlift"] > info[t]["pixel_ap"]["ours"] for t in s)
    fr = ", ".join(f"{t} at frame {info[t]['frame']}" for t in s)
    farec = ", ".join(f"{t} {_f(info[t]['fa'])}/{_f(info[t]['rec'])}" for t in s)
    nl_txt = (f"NormLift's drop has a higher pixel-level AP than our test in all three scenes, as "
              f"in Table~\\ref{{tab:change}} ({n_nl} of {len(TEST)} test scenes)"
              if nl_all else
              f"NormLift's drop has a higher pixel-level AP than our test in {n_nl} of "
              f"{len(TEST)} test scenes (Table~\\ref{{tab:change}})")
    return (f"Stale content after a real removal (NeRF-Object-Removal test scenes {fr}). The scenes "
            f"are fixed by rule: the highest, the lower-median and the lowest pixel-level AP of "
            f"ours in Table~\\ref{{tab:change}}; the frame is the update keyframe with the largest "
            f"object region inside the image. The map geometry is fixed: (a) renders the map, "
            f"so the removed object's Gaussians are still there; (b) is the camera image after "
            f"the removal. (c)-(g) render each row's per-Gaussian score, coloured by its rank "
            f"among the covered pixels of all update keyframes of the scene (the population of "
            f"the pixel-level AP; grey: no score, black: not covered); the numbers are the "
            f"rows' pixel-level AP (\\%) in Table~\\ref{{tab:change}}. (h) shows our flags at "
            f"the label-free threshold against the stale Gaussians (orange: flagged and stale, "
            f"light blue: stale and not flagged, vermillion: flagged and static), with FA/Rec.\\ (\\%) of "
            f"ours counted over all update views ({farec}). The outline is masks\\_real, a "
            f"dilated silhouette of the object. {nl_txt}.")


def query_caption(info):
    s = list(info)
    q = ", ".join(f"{t}: '{info[t]['query']}'" for t in s)
    rec = "/".join(_f(info[t]["rec"]) for t in s)
    return (f"Query relevance before and after the update, same scenes, frames and rotations as "
            f"Fig.~\\ref{{fig:qual-change}}. The query of each scene is fixed by rule and never "
            f"re-prompted: the most frequent top class (101 Replica classes, LSeg 5-template "
            f"ensemble) of the mapping-pass $\\boldsymbol\\mu^\\star_n$ of the stale Gaussians "
            f"({q}). Each panel shows the cosine between the rendered feature and the query text, "
            f"one colour scale per row (5th to 99th percentile of (b)). The map geometry is "
            f"fixed: the removed object's Gaussians are still rendered after the update and "
            f"only their semantics change. (c) is the same field after the same update views "
            f"with the change test off; (d) is ours, where only the Gaussians our test flags "
            f"are reset (Rec.\\ {rec}\\% in these scenes), so the missed ones keep the old "
            f"class; (e) is averaging. The outline is masks\\_real, a dilated silhouette.")


def teaser_caption(res, val_scenes_note=True):
    m, p = res["measured"], res["published"]
    q = res["query"]
    c1 = m["mean_p_avg_after"] >= 0.5
    c2 = m["mean_p_ours_after"] < 0.5
    c3 = m["mean_u_epi_after"] > m["mean_u_epi_before"]
    if c1 and c2:
        gh = (f"(g) averaging still answers '{q}' there; (h) ours does not, because (i) our "
              f"test reset {_f(100 * p['rec'])}\\% of the stale Gaussians "
              f"({_f(100 * p['fa'])}\\% of the static ones)")
    else:
        gh = (f"(g, h) p('{q}') of averaging and of ours after the update; (i) the Gaussians "
              f"our test reset ({_f(100 * p['rec'])}\\% of the stale, "
              f"{_f(100 * p['fa'])}\\% of the static ones)")
    jj = ("(j) their $u^{\\mathrm{epi}}$ rises" if c3 else
          "(j) $u^{\\mathrm{epi}}$ after the update")
    head = ("A semantic 3D-GS map that knows how much evidence it has and forgets what changed."
            if (c1 and c2) else
            "A semantic 3D-GS map with two uncertainty channels and a test for changed content.")
    return (f"{head} NeRF-Object-Removal validation scene {res['scene']} (picked by rule: the largest "
            f"removed object of the validation scenes), update keyframe {res['frame']} (the "
            f"largest object region inside the image). Both rows show the same camera pose; the "
            f"map geometry is fixed, so the removed object's Gaussians are still rendered. "
            f"Top, after mapping: (a) map render; (b, c) probability of the text query '{q}' "
            f"(fixed by rule, 101 classes) for averaging and for our field; (d) $u^{{\\mathrm{{ale}}}}$, "
            f"whether the views agree (grey: fewer than 3 views); (e) $u^{{\\mathrm{{epi}}}}$, how "
            f"much view evidence there is. Bottom, after the {p['n_update']} update keyframes: "
            f"(f) camera image (dashed: the removed object's Gaussians, ground truth); {gh}; "
            f"{jj}. (k) Over the update: share of "
            f"the removed object's Gaussians whose top class is '{q}' (solid) and of static "
            f"Gaussians whose top class changed (dashed).")


# --------------------------------------------------------------------------- anonymity
def pdf_check(path):
    """pdffonts (no Type 3) and pdfinfo (no user name or home path)."""
    import getpass
    import subprocess
    fonts = subprocess.run(["pdffonts", path], capture_output=True, text=True).stdout
    info = subprocess.run(["pdfinfo", path], capture_output=True, text=True).stdout
    user = getpass.getuser()
    bad = []
    if "Type 3" in fonts:
        bad.append("Type 3 font")
    for s in (user, "/home/"):
        if s and s in info:
            bad.append(f"pdfinfo holds {s!r}")
    with open(path, "rb") as f:
        raw = f.read()
    for s in (user, "/home/"):
        if s and s.encode() in raw:
            bad.append(f"PDF bytes hold {s!r}")
    return dict(fonts=fonts, info=info, ok=not bad, problems=bad)


# --------------------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("noise")
    p.add_argument("--a", required=True, help="the Table III record")
    p.add_argument("--b", required=True, help="its repeat")
    p.add_argument("--out", required=True)
    p = sub.add_parser("select")
    p.add_argument("--records", required=True, help="the tab2_relmax folder (val/, test/)")
    p.add_argument("--root", required=True, help="the NeRF-Object-Removal scenes folder")
    p.add_argument("--out", required=True)
    p = sub.add_parser("fused")
    p.add_argument("--dumps", required=True, help="folder of the per-scene dumps")
    p.add_argument("--records", required=True, help="the tab2_relmax folder (val/, test/)")
    p.add_argument("--prereg", default=os.path.join(REPO, "configs", "nor_fused_prereg.json"))
    p.add_argument("--val", default=",".join(VAL))
    p.add_argument("--test", default="", help="test scenes (empty: validation only)")
    p.add_argument("--out-dir", required=True)
    for name in ("change", "query", "grid"):
        p = sub.add_parser(name)
        p.add_argument("--dumps", required=True)
        p.add_argument("--records", required=True, help="the tab2_relmax folder (val/, test/)")
        p.add_argument("--scenes", default="rule",
                       help="'rule' (N1: highest, lower-median, lowest test pixel AP of ours) "
                            "or a list")
        p.add_argument("--stage-label", default=None, help="e.g. 'validation' for dev runs")
        p.add_argument("--out", required=True, help="figure path (.pdf; a .png is written too)")
        p.add_argument("--caption", default="", help="write the proposed caption here")
        p.add_argument("--meta", default="", help="write the figure's metadata here (JSON)")
        p.add_argument("--fig-w", type=float, default=7.16 if name != "query" else 6.0,
                       help="figure width [in]")
    p = sub.add_parser("teaser")
    p.add_argument("--dump", required=True, help="the validation scene's dump folder")
    p.add_argument("--record", required=True, help="its Table III record")
    p.add_argument("--out", required=True)
    p.add_argument("--caption", default="")
    p.add_argument("--meta", default="")
    p = sub.add_parser("numbers")
    p.add_argument("--dumps", required=True)
    p.add_argument("--scenes", required=True)
    p.add_argument("--stage", required=True, choices=["val", "test"])
    p.add_argument("--json", required=True)
    p.add_argument("--tex", required=True)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "noise":
        res = noise(args.a, args.b)
        write_json(args.out, res)
        print(f"{res['scene']}: max delta {res['max_delta']:.3g} ({res['max_cell']}); "
              f"tolerance {res['tolerance']:.4g}; views equal {res['views_equal']}, frames "
              f"equal {res['frames_equal']}; wrote {args.out}")
        return res
    if args.cmd == "select":
        recs = {s: os.path.join(args.records, sp, f"{s}.json")
                for sp, ss in (("val", VAL), ("test", TEST)) for s in ss}
        res = select(recs, args.root)
        write_json(args.out, res)
        for s, r in res["scenes"].items():
            print(f"{s}: frame {r['frame']} turns {r['quarter_turns']} ({r['rule']}) "
                  f"{'ok' if r['ok'] else 'MISMATCH'}")
        print(f"select: {'PASS' if res['ok'] else 'FAIL'}; wrote {args.out}")
        if not res["ok"]:
            raise SystemExit(1)
        return res
    if args.cmd == "fused":
        recs = {s: os.path.join(args.records, sp, f"{s}.json")
                for sp, ss in (("val", VAL), ("test", TEST)) for s in ss}
        val = [s for s in args.val.split(",") if s]
        test = [s for s in args.test.split(",") if s]
        rep = fused_report(args.dumps, val, test, recs, args.prereg)
        os.makedirs(args.out_dir, exist_ok=True)
        tag = "test" if test else "val"
        write_json(os.path.join(args.out_dir, f"nor_fused_{tag}.json"), rep)
        with open(os.path.join(args.out_dir, f"nor_fused_{tag}_scenes.md"), "w") as f:
            f.write(fused_scene_md(rep))
        if test:
            rows, stat = fused_latex(rep)
            with open(os.path.join(args.out_dir, "nor_fused_test_rows.tex"), "w") as f:
                f.write(rows)
            with open(os.path.join(args.out_dir, "nor_fused_test_f2.tex"), "w") as f:
                f.write(stat)
            v = rep["test"]["verdict"]
            with open(os.path.join(args.out_dir, "nor_fused_verdict.md"), "w") as f:
                f.write(f"# Fused-baseline claim rule (configs/nor_fused_prereg.json)\n\n"
                        f"Rule: {v['rule']}\n\nVerdict: {v['text']}\n")
            print(v["text"])
        print(fused_scene_md(rep))
        return rep
    if args.cmd in ("change", "query", "grid", "teaser"):
        if args.cmd == "teaser":
            res = teaser_figure(args.dump, args.record, args.out)
            cap = teaser_caption(res)
        else:
            recs = {s: os.path.join(args.records, sp, f"{s}.json")
                    for sp, ss in (("val", VAL), ("test", TEST)) for s in ss}
            if args.scenes == "rule":
                scenes, ap = selection_rule_n1(recs)
            else:
                scenes = [s for s in args.scenes.split(",") if s]
            if args.cmd == "query":
                res = query_figure(args.dumps, scenes, recs, out=args.out,
                                   stage_label=args.stage_label, fig_w=args.fig_w)
                cap = query_caption(res["info"])
            else:
                res = change_figure(args.dumps, scenes, recs, args.out, args.stage_label,
                                    args.fig_w)
                cap = change_caption(res["info"], recs)
            if args.scenes == "rule":
                res["selection_rule"] = dict(rule="highest, lower-median (5th of 10, "
                                                  "ascending), lowest test pixel AP of ours",
                                             pixel_ap_ours=ap, scenes=scenes)
        res["caption"] = cap
        res["pdf_check"] = pdf_check(res["outputs"][0])
        if args.caption:
            with open(args.caption, "w") as f:
                f.write(cap + "\n")
        if args.meta:
            write_json(args.meta, res)
        print(cap)
        print(f"pdf check: {'PASS' if res['pdf_check']['ok'] else 'FAIL'} "
              f"{res['pdf_check']['problems']}; wrote {' '.join(res['outputs'])}")
        return res
    if args.cmd == "numbers":
        rep = numbers(args.dumps, [s for s in args.scenes.split(",") if s], args.stage)
        write_json(args.json, rep)
        with open(args.tex, "w") as f:
            f.write(numbers_latex(rep))
        print(json.dumps(rep["summary"], indent=1))
        print(rep["counts"])
        return rep


if __name__ == "__main__":
    main()
