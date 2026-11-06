"""
plot_views.py — the paper's views figure (fig:views) from a views JSON (scripts/eval_views.py,
merged over scenes by scripts/replica_views.sh) and, for the full-map point M = 180, the
Table I JSON (scripts/eval_seg.py). Does each uncertainty rank the error it is meant for?

    python3 scripts/plot_views.py --stage test            # the paper figure, papers/figs/views.pdf
    python3 scripts/plot_views.py --stage val             # validation, N = 2 (results folder)
    python3 scripts/plot_views.py --stage val --defined   # the secondary analysis

Layout: 2 x 2 panels. Rows: our u_epi (top), our u_ale (bottom). Columns: target e_red (wrong
with M mapping frames, right with the full map) and target e_irr (wrong with both). x-axis: M,
the file's ks (10, 30, 90), log scale. y-axis (--scale relative, the default): each scene's
AUSE divided by the AUSE of a random ranking of the same errors (all uncertainties tied:
usplat.evaluation.sparsification_curve's random_gap, from the scene's pixel and target
counts), so 1 = random and 0 = the oracle; lower is better. --scale raw plots AUSE x100 as in
the summary. Each curve is the mean over the scenes with +- std error bars (population std,
as usplat.seg_eval.scene_stats; --band draws bands). Both rows hold the same references: our total uncertainty, one minus the
maximum probability of our full predictive (P2, the row --total-row, signal "msp"), one minus
the maximum probability of averaging, the scores built from NormLift's quantities
1/(1 + N_eff) and 1 - R, and the view count 1/(1 + n_vis).
READING: raw AUSE grows with the target's share of pixels (and that share varies between
scenes, targets and M), so raw values are compared only WITHIN a column; the relative scale
removes the share, as every curve is divided by the random level of its OWN errors (the
references' predictive is averaging, slightly different from ours). On e_red u_epi should be
lower than u_ale, on e_irr the reverse. The dashed grey line is the random ranking: 1 on the
relative scale (all four panels then share one y axis; the caption names the line), our
errors' random level on the raw scale (labelled "random ranking" in the top panels).
A hollow marker: the point averages fewer scenes than the file has, or one scene has fewer
than MIN_COUNT target pixels (a noisy point).
M = 180, e_irr column only: the field is then the full map, so e_red is empty and e_irr is the
full map's 0/1 error on the pixels it covers. That is Table I's AUSE on covered pixels,
summary[row]["covered"]["ause"][signal] (both scripts score the same frames with the same
ranking rule). Same scenes as the views file: if the Table I file holds other scenes, the
point is recomputed from the per-scene values of the views scenes.
Checks (with the Table I file, primary analysis): the full map of the views file has Table I's
mIoU; the views and Table I score the same covered pixels at every M; the views were scored
with Table I's temperatures (betas of the part files). A failed check stops --stage test
before anything is written (--allow-mismatch overrides); --stage val prints a warning.
--defined plots summary_on_ale_defined instead: the secondary analysis on the pixels where our
u_ale is defined (added after the validation run, labelled post hoc). It has no M = 180 point.
The plotted numbers, the raw AUSE x100 (with --scale relative), the random levels, the
target shares and the smallest target counts are printed as Markdown tables for the report.
Output: a vector PDF, 3.5 x 2.6 in (IEEE column), TrueType fonts (default: papers/figs/views.pdf
for --stage test; <results>/views/plots/views_val.pdf for val; "_defined" added with --defined)
and a 300 dpi PNG preview next to it (for the default test PDF: <results>/views/plots/
views_test.png, so papers/figs holds only the PDF). Colours: Okabe-Ito; they pass the dataviz palette check
on white for all pairs, and every curve also has its own marker.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.lines import Line2D                                        # noqa: E402
from matplotlib.ticker import MaxNLocator                                  # noqa: E402

RESULTS = "/home/thanhnc/gauss_ws/src/uags/results"
FIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers", "figs")
CHANNELS = [("u_epi", r"ours: $u^{\mathrm{epi}}$"), ("u_ale", r"ours: $u^{\mathrm{ale}}$")]
TARGETS = [("e_red", r"reducible error $e_{\mathrm{red}}$"),
           ("e_irr", r"irreducible error $e_{\mathrm{irr}}$")]
# (name in the table, legend label, views row, signal, colour, marker). Ours: the row and the
# signal are set per panel row (--ours-row, CHANNELS). Colours: Okabe-Ito; the dataviz
# validator on white, all pairs: worst CVD dE 7.6 (deutan, #CC79A7 vs #009E73) is in the warn
# band, so each curve also has its own marker; normal-vision floor 15.6 passes.
OURS = ("ours", r"Ours: $u^{\mathrm{epi}}$ (top), $u^{\mathrm{ale}}$ (bottom)", None, None, "#0072B2", "o")
REFS = [
    ("ours, 1 - max p", r"Ours: total, $1-\max\,p$", "ours_P2", "msp", "#0072B2", "o", (0, (3, 1.5))),
    ("averaging, 1 - max p", r"Averaging, $1-\max\,p$", "averaging", "msp", "#D55E00", "s"),
    ("NormLift, 1/(1+N_eff)", r"NormLift, $1/(1+N_{\mathrm{eff}})$", "normlift", "normlift_epi",
     "#009E73", "^"),
    ("NormLift, 1 - R", r"NormLift, $1-\mathcal{R}$", "normlift", "normlift_1-R", "#E69F00", "v"),
    ("view count, 1/(1+n_vis)", "View count", "viewcount", "viewcount", "#56B4E9", "P"),
]
RANDOM = dict(label="Random ranking", color="#898781", ls=(0, (3, 2)), lw=0.7)
SHOW_RANDOM = False             # the random level is 1 on the relative axis (user, 2026-11-06: no line)
MIN_COUNT = 100                   # a scene with fewer target pixels makes the point hollow
DODGE = 0.04                      # error bars: curve c sits at M * exp((c - 3) * DODGE)
INK, INK2, AXIS, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
RC = {"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
      "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 8,
      "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
      "pdf.fonttype": 42, "ps.fonttype": 42,          # TrueType in the PDF, no Type 3 fonts
      "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
      "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2,
      "ytick.labelcolor": INK2, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
      "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.pad": 1.5,
      "ytick.major.pad": 1.5, "axes.spines.top": False, "axes.spines.right": False}


def load(path):
    """A JSON file; a usplat.runlog record (eval_seg.py output) is unwrapped to its metrics."""
    if not os.path.exists(path):
        hint = (" (scripts/replica_views.sh merges views_<stage>.json once every scene is done)"
                if os.path.basename(path).startswith("views_") else "")
        raise SystemExit(f"{path}: no such file{hint}")
    d = json.load(open(path))
    if d.get("status", "ok") != "ok":
        raise SystemExit(f"{path}: status {d.get('status')!r}, not a finished evaluation")
    return d["metrics"] if isinstance(d.get("metrics"), dict) else d


def num(x):
    """A JSON number as a float. None is NaN; so is the string "nan" (usplat.runlog writes
    non-finite floats as strings)."""
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        raise SystemExit(f"not a number: {x!r}") from None


def scene_stats(values):
    """mean, population std and N of the finite values, as usplat.seg_eval.scene_stats."""
    v = [x for x in map(num, values) if math.isfinite(x)]
    if not v:
        return dict(mean=float("nan"), std=float("nan"), n=0, n_nan=len(values))
    return dict(mean=statistics.fmean(v), std=statistics.pstdev(v), n=len(v),
                n_nan=len(values) - len(v))


def random_level(n, n_t, bins=20):
    """AUSE of a random ranking (all uncertainties tied) of a 0/1 error with n_t ones among n
    pixels: usplat.evaluation.sparsification_curve's random_gap, the level a ranking that
    knows nothing reaches in expectation. It grows with the target share n_t / n."""
    if n <= 0:
        return float("nan")
    return statistics.fmean(n_t / n - max(n_t - k, 0) / (n - k)
                            for k in (int(i * n / bins) for i in range(bins)))


def views_stat(summary, k, row, signal, target, path):
    try:
        return summary[str(k)][row][signal][target]
    except KeyError:
        raise SystemExit(f"{path}: no AUSE for M={k}, row {row!r}, signal {signal!r}, "
                         f"target {target!r}") from None


def full_stat(tab1, scenes, row, signal, path):
    """Table I's AUSE of (row, signal) on covered pixels over `scenes`: the file's summary if
    it holds exactly these scenes, else the per-scene values of these scenes."""
    have = list(tab1["scenes"])
    try:
        if sorted(have) == sorted(scenes):
            return tab1["summary"][row]["covered"]["ause"][signal]
        miss = [s for s in scenes if s not in have]
        if miss:
            raise SystemExit(f"{path}: no scenes {miss} (it holds {have})")
        return scene_stats([tab1["scenes"][s]["rows"][row]["covered"]["ranking"][signal]["ause"]
                            for s in scenes])
    except KeyError:
        raise SystemExit(f"{path}: no covered AUSE for row {row!r}, signal {signal!r}") from None


def _ratio(a, n, n_t, bins):
    """AUSE / random-ranking AUSE of the same pixels and target (NaN without errors)."""
    r = random_level(n, n_t, bins)
    return num(a) / r if r > 0 else float("nan")


def views_relative(views, k, row, sig, t, defined, bins, path):
    """Per views scene at M = k: AUSE / random AUSE (the on_ale_defined pixels with
    defined=True; a scene without defined pixels is NaN, as in summary_on_ale_defined)."""
    out = []
    for s, res in views["scenes"].items():
        try:
            r = res["k"][str(k)]["rows"][row]
            if defined:
                r = r["on_ale_defined"]
                a = r["ause"].get(sig, {}).get(t)
            else:
                a = r["ause"][sig][t]
            out.append(_ratio(a, int(r["n_pixels"]), int(r[f"n_{t}"]), bins))
        except KeyError:
            raise SystemExit(f"{path}: scene {s} has no AUSE and counts for M={k}, row {row!r}, "
                             f"signal {sig!r}, target {t!r} (--scale relative needs the "
                             "per-scene values; --scale raw reads the summary)") from None
    return out


def full_relative(tab1, scenes, row, sig, bins, path):
    """Per views scene: Table I's covered AUSE / the random AUSE of the full map's covered
    errors (n_pixels, acc): the M = m_full point on the relative scale."""
    miss = [s for s in scenes if s not in tab1["scenes"]]
    if miss:
        raise SystemExit(f"{path}: no scenes {miss} (it holds {list(tab1['scenes'])})")
    out = []
    for s in scenes:
        try:
            c = tab1["scenes"][s]["rows"][row]["covered"]
            n = int(c["n_pixels"])
            out.append(_ratio(c["ranking"][sig]["ause"], n, round(n * (1.0 - num(c["acc"]))),
                              bins))
        except KeyError:
            raise SystemExit(f"{path}: scene {s} has no covered AUSE, n_pixels and acc for row "
                             f"{row!r}, signal {sig!r}") from None
    return out


def views_counts(views, k, row, target, defined):
    """[(n_pixels, n_target)] of every views scene at M = k (the pixels scored there; the
    on_ale_defined counts with --defined), or None when a scene lacks them."""
    out = []
    for res in views["scenes"].values():
        r = res.get("k", {}).get(str(k), {}).get("rows", {}).get(row) or {}
        if defined:
            r = r.get("on_ale_defined") or {}
        if "n_pixels" not in r or f"n_{target}" not in r:
            return None
        out.append((int(r["n_pixels"]), int(r[f"n_{target}"])))
    return out


def full_counts(tab1, scenes, row):
    """[(n_pixels, n_error)] of the full map on covered pixels (Table I covered n_pixels and
    acc): the e_irr target at M = m_full. None when a scene lacks them."""
    out = []
    for s in scenes:
        c = tab1["scenes"].get(s, {}).get("rows", {}).get(row, {}).get("covered", {})
        if "n_pixels" not in c or "acc" not in c:
            return None
        n = int(c["n_pixels"])
        out.append((n, round(n * (1.0 - num(c["acc"])))))
    return out


def count_stats(counts, bins):
    """Random level and target share (stats over scenes) and the smallest target count."""
    if not counts:
        return None
    return dict(random=scene_stats([random_level(n, t, bins) for n, t in counts]),
                share=scene_stats([t / n if n else float("nan") for n, t in counts]),
                min_count=min(t for _, t in counts))


def curve_keys(ours_row):
    """(row, signal) of every curve: ours u_epi, ours u_ale, then the references."""
    return [(ours_row, c) for c, _ in CHANNELS] + [(r[2], r[3]) for r in REFS]


def ause_bins(views):
    """The AUSE bin count of the views run (provenance of a part file), else 20."""
    for p in (views.get("parts") or {}).values():
        if os.path.exists(p):
            return int(json.load(open(p)).get("provenance", {}).get("config", {})
                       .get("ause_bins", 20))
    return 20


def collect(views, views_path, defined=False, tab1=None, tab1_path="", ours_row="ours_P0",
            m_full=180, bins=20, relative=False):
    """(data, sizes). data = {target: {(row, signal): [(M, stats), ...]}} with stats =
    mean/std/n over scenes of the AUSE (a fraction; relative=True: of AUSE / random AUSE per
    scene); sizes = {target: {row: [(M, count_stats or None)]}} for the rows drawn. M = m_full
    is added to e_irr when the Table I file is given."""
    key = "summary_on_ale_defined" if defined else "summary"
    if key not in views:
        raise SystemExit(f"{views_path}: no {key!r} (an older eval_views.py output?)")
    ks, scenes = [int(k) for k in views["ks"]], list(views["scenes"])
    full = tab1 is not None and not defined
    data, sizes = {}, {}
    for t, _ in TARGETS:
        data[t], sizes[t] = {}, {}
        for row, sig in curve_keys(ours_row):
            if relative:
                pts = [(k, scene_stats(views_relative(views, k, row, sig, t, defined, bins,
                                                      views_path))) for k in ks]
                if t == "e_irr" and full:
                    pts.append((m_full, scene_stats(full_relative(tab1, scenes, row, sig, bins,
                                                                  tab1_path))))
            else:
                pts = [(k, views_stat(views[key], k, row, sig, t, views_path)) for k in ks]
                if t == "e_irr" and full:
                    pts.append((m_full, full_stat(tab1, scenes, row, sig, tab1_path)))
            data[t][(row, sig)] = pts
        for row in dict.fromkeys(r for r, _ in curve_keys(ours_row)):
            pts = [(k, count_stats(views_counts(views, k, row, t, defined), bins)) for k in ks]
            if t == "e_irr" and full:
                pts.append((m_full, count_stats(full_counts(tab1, scenes, row), bins)))
            sizes[t][row] = pts
    return data, sizes


def _ok(st):
    return bool(st) and st.get("n", 0) > 0 and math.isfinite(num(st.get("mean")))


def cell(st, n_all, scale=100.0):
    """'mean +- std' x100 (3 decimals); '(n=..)' if fewer scenes than the file has."""
    if not _ok(st):
        return "--"
    s = f"{scale * num(st['mean']):.3f} +- {scale * num(st['std']):.3f}"
    return s if st["n"] == n_all else f"{s} (n={st['n']})"


def curve_name(row, sig, ours_row):
    if row == ours_row and sig in dict(CHANNELS):
        return f"ours, {sig}"
    return next(r[0] for r in REFS if (r[2], r[3]) == (row, sig))


def _ms(data):
    return sorted({m for t in data for pts in data[t].values() for m, _ in pts})


def table(data, ours_row, n_scenes, scale=100.0, title="curve"):
    """The numbers of one scale as a Markdown table: one line per target and curve (x100 for
    raw AUSE, x1 for AUSE / random AUSE)."""
    ms = _ms(data)
    lines = [f"| target | {title} | JSON row/signal | " + " | ".join(f"M={m}" for m in ms)
             + " |", "|" + "---|" * (3 + len(ms))]
    for t, _ in TARGETS:
        for (row, sig), pts in data[t].items():
            by_m = dict(pts)
            cells = [cell(by_m.get(m), n_scenes, scale) for m in ms]
            lines.append(f"| {t} | {curve_name(row, sig, ours_row)} | {row}/{sig} | "
                         + " | ".join(cells) + " |")
    return "\n".join(lines)


def size_table(sizes, ms, n_scenes):
    """Per target and row (whose errors): the random-ranking AUSE x100, the target share in %
    and the smallest per-scene target count; '--' where the counts are missing."""
    lines = ["| target | errors of row | quantity | " + " | ".join(f"M={m}" for m in ms) + " |",
             "|" + "---|" * (3 + len(ms))]
    for t, _ in TARGETS:
        same = {}                          # rows with the same predictive share their errors
        for row, pts in sizes[t].items():
            same.setdefault(json.dumps(pts, sort_keys=True), []).append(row)
        for rows in same.values():
            row, by_m = " = ".join(rows), dict(sizes[t][rows[0]])
            for q, f in (("random AUSE x100", lambda c: cell(c["random"], n_scenes)),
                         ("target share %", lambda c: cell(c["share"], n_scenes)),
                         ("smallest target count", lambda c: str(c["min_count"]))):
                cells = [f(by_m[m]) if by_m.get(m) else "--" for m in ms]
                lines.append(f"| {t} | {row} | {q} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def full_map_check(views, tab1):
    """Largest |mIoU of the full map in the views file - Table I's mIoU| over the shared
    (scene, row) pairs, and their number: 0 when the M = 180 point and the curves come from
    the same full map scored the same way."""
    diffs = []
    for s, res in views["scenes"].items():
        rows = tab1["scenes"].get(s, {}).get("rows", {})
        for r, m in res.get("full", {}).get("miou", {}).items():
            t = rows.get(r, {}).get("all", {}).get("miou")
            if m is not None and t is not None:
                diffs.append(abs(num(m) - num(t)))
    return (max(diffs) if diffs else None), len(diffs)


def pixel_check(views, tab1):
    """(mismatches, pairs): views n_pixels vs Table I covered n_pixels for every (scene, M,
    row) in both files. Coverage depends only on the geometry, so they must be equal."""
    bad, n = [], 0
    for s, res in views["scenes"].items():
        rows = tab1["scenes"].get(s, {}).get("rows", {})
        for k, rk in res.get("k", {}).items():
            for r, v in rk.get("rows", {}).items():
                c = rows.get(r, {}).get("covered", {})
                if "n_pixels" in v and "n_pixels" in c:
                    n += 1
                    if int(v["n_pixels"]) != int(c["n_pixels"]):
                        bad.append(f"{s} M={k} {r}: {v['n_pixels']} vs {c['n_pixels']}")
    return bad, n


def betas_check(views, tab1):
    """(mismatches, keys, parts) or None: the temperatures of every views part file vs Table
    I's, over the rows both hold. None when the part files or the betas are missing."""
    parts = views.get("parts") or {}
    if not parts or "betas" not in tab1 or not all(os.path.exists(p) for p in parts.values()):
        return None
    bad, n = [], 0
    for s, p in parts.items():
        b = json.load(open(p)).get("betas")
        if b is None:
            return None
        for r in sorted(set(b) & set(tab1["betas"])):
            n += 1
            if json.dumps(b[r], sort_keys=True) != json.dumps(tab1["betas"][r], sort_keys=True):
                bad.append(f"{s} {r}")
    return bad, n, len(parts)


def checks(views, tab1, m_full=180):
    """[(name, PASS | FAIL | NOT RUN, detail)] of the consistency checks."""
    out = []
    d, n = full_map_check(views, tab1)
    if d is None:
        out.append(("full-map mIoU", "NOT RUN", "no full-map mIoU in the views file"))
    else:
        out.append(("full-map mIoU", "PASS" if d <= 1e-4 else "FAIL",
                    f"max |mIoU views full map - Table I| = {d:.2e} over {n} (scene, row) pairs"
                    + ("" if d <= 1e-4 else f" -- the M={m_full} point comes from another "
                                             "evaluation")))
    bad, n = pixel_check(views, tab1)
    out.append(("covered pixels", "NOT RUN" if n == 0 else "FAIL" if bad else "PASS",
                "no pixel counts in both files" if n == 0 else
                f"views n_pixels = Table I covered n_pixels in {n - len(bad)} of {n} "
                f"(scene, M, row) triples" + (f"; first mismatch {bad[0]}" if bad else "")))
    b = betas_check(views, tab1)
    if b is None:
        out.append(("temperatures", "NOT RUN", "no part files or no betas"))
    else:
        bad, n, n_parts = b
        out.append(("temperatures", "FAIL" if bad or n == 0 else "PASS",
                    f"betas of {n_parts} part files = Table I betas in {n - len(bad)} of {n} "
                    f"(part, row) pairs" + (f"; first mismatch {bad[0]}" if bad else "")))
    return out


def header(stage, defined, views, views_path, tab1, tab1_path, m_full=180, results=(),
           relative=False):
    scenes = list(views["scenes"])
    lines = [f"views figure, stage {stage}: "
             + ("SECONDARY analysis (summary_on_ale_defined, post hoc): pixels where our u_ale "
                "is defined" if defined else "primary analysis (summary)"),
             f"views: {views_path}; N = {len(scenes)} scenes: {', '.join(scenes)}"]
    if defined:
        share = [(k, views["summary_on_ale_defined"].get(str(k), {}).get("share"))
                 for k in views["ks"]]
        lines.append("share of covered pixels with u_ale defined: " + ", ".join(
            f"M={k} {100 * num(s['mean']):.1f}%" for k, s in share if _ok(s)))
    elif tab1 is None:
        lines.append(f"M={m_full}: not plotted (no Table I file)")
    else:
        same = sorted(tab1["scenes"]) == sorted(scenes)
        lines.append(f"M={m_full} (e_irr only): Table I AUSE on covered pixels, {tab1_path}, "
                     + ("its summary (same scenes)" if same else
                        f"recomputed over the {len(scenes)} views scenes (the file holds "
                        f"{len(tab1['scenes'])})"))
        for name, status, detail in results:
            lines.append(f"check {name}: {status}: {detail}"
                         + (" -- WARNING" if status == "FAIL" else ""))
    lines.append(("plotted: AUSE / random-ranking AUSE per scene (1 = random, 0 = oracle)"
                  if relative else "plotted: AUSE x100 (raw; compare channels within a target: "
                  "AUSE grows with the target's share)")
                 + ", mean +- population std over scenes; lower is better")
    return "\n".join(lines)


def _xy(pts, scale=100.0):
    x = [m for m, _ in pts]
    y = [scale * num(st["mean"]) if _ok(st) else float("nan") for _, st in pts]
    e = [scale * num(st["std"]) if _ok(st) and math.isfinite(num(st.get("std"))) else 0.0
         for _, st in pts]
    return x, y, e


def hollow(pts, size_pts, n_scenes):
    """Per point: True if it averages fewer scenes than the file has or one scene has fewer
    than MIN_COUNT target pixels."""
    sz = dict(size_pts or [])
    return [_ok(st) and (st["n"] < n_scenes or
                         (sz.get(m) is not None and sz[m]["min_count"] < MIN_COUNT))
            for m, st in pts]


def draw(ax, pts, style, idx, n_curves, band, gid, flags=None, scale=100.0):
    """One curve: mean with +- std error bars (dodged in x) or a +- std band. Points with a
    True flag get a white face (hollow)."""
    colour, marker = style[4], style[5]
    ours = style is OURS
    x, y, e = _xy(pts, scale)
    kw = dict(color=colour, marker=marker, lw=1.3 if ours else 0.8, ms=3.6 if ours else 3.0,
              mec="white", mew=0.4, zorder=3 if ours else 2, clip_on=False,
              ls=style[6] if len(style) > 6 else "-")
    if band:
        lo = [max(v - d, 0.0) for v, d in zip(y, e)]
        ax.fill_between(x, lo, [v + d for v, d in zip(y, e)], color=colour, alpha=0.12,
                        lw=0, zorder=1)
        ax.plot(x, y, gid=gid, **kw)
    else:
        f = math.exp((idx - (n_curves - 1) / 2) * DODGE)
        x = [v * f for v in x]
        ax.errorbar(x, y, yerr=[[min(d, v) if math.isfinite(v) else 0.0
                                 for v, d in zip(y, e)], e],
                    elinewidth=0.6, capsize=1.2, capthick=0.6, gid=gid, **kw)
    if flags and any(flags):
        xs = [v for v, h in zip(x, flags) if h]
        ys = [v for v, h in zip(y, flags) if h]
        ax.plot(xs, ys, ls="none", marker=marker, ms=kw["ms"], mfc="white", mec=colour,
                mew=0.7, zorder=kw["zorder"] + 0.5, clip_on=False, gid=f"{gid}/hollow")


def draw_random(ax, size_pts, gid, label=False, relative=False):
    """The random-ranking level of one row's errors: 1 on the relative scale, else x100 where
    the counts exist; label: write "random ranking" above its first point (no legend entry)."""
    pts = [(m, 1.0 if relative else 100 * num(c["random"]["mean"])) for m, c in size_pts or []
           if c is not None and _ok(c["random"])]
    if pts:
        ax.plot([m for m, _ in pts], [v for _, v in pts], color=RANDOM["color"],
                ls=RANDOM["ls"], lw=RANDOM["lw"], zorder=1.5, clip_on=False, gid=gid)
        if label:
            ax.annotate(RANDOM["label"].lower(), pts[0], xytext=(2, 1.5),
                        textcoords="offset points", ha="left", va="bottom", fontsize=6,
                        color=INK2, fontstyle="italic", gid=f"{gid}/label")


def format_axes(axes, data):
    """Log x with ticks at the M values each column holds (no M = 180 tick under e_red: the
    columns are separate targets, each with its own x range), y from 0."""
    for j, (t, _) in enumerate(TARGETS):
        ticks = sorted({m for pts in data[t].values() for m, _ in pts})
        for ax in axes[:, j]:
            ax.set_xscale("log")
            ax.set_xticks(ticks, [str(m) for m in ticks])
            ax.minorticks_off()
            ax.set_xlim(min(ticks) / 1.35, max(ticks) * 1.3)
    for ax in axes.flat:
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)


def plot(data, out, ours_row="ours_P0", band=False, sizes=None, n_scenes=None, keep=False,
         png=None, relative=False):
    """The 2 x 2 figure (3.5 x 2.6 in, IEEE column width) as a PDF plus a PNG preview (next
    to the PDF unless png is given); keep=True also returns the open figure (for tests)."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.6), sharex="col",       # relative: one
                                 sharey=True if relative else "col",           # y for all
                                 layout="constrained")
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.03, hspace=0.03)
        for i, (ch, ch_label) in enumerate(CHANNELS):
            for j, (t, t_label) in enumerate(TARGETS):
                ax = axes[i, j]
                if sizes and SHOW_RANDOM:
                    draw_random(ax, sizes[t].get(ours_row), f"random/{ours_row}",
                                i == 0 and not relative, relative)
                curves = [(OURS, (ours_row, ch))] + [(r, (r[2], r[3])) for r in REFS]
                for c, (style, key) in enumerate(curves):
                    flags = (hollow(data[t][key], sizes[t].get(key[0]), n_scenes)
                             if sizes and n_scenes else None)
                    draw(ax, data[t][key], style, c, len(curves), band, "/".join(key), flags,
                         1.0 if relative else 100.0)
                if i == 0:
                    ax.set_title(t_label, pad=2)
            axes[i, 1].yaxis.set_label_position("right")
            axes[i, 1].set_ylabel(ch_label, rotation=270, labelpad=8)
        format_axes(axes, data)
        fig.supylabel(r"AUSE / AUSE$_{\mathrm{random}}$ (lower is better)" if relative else
                      r"AUSE $\times 100$ (lower is better)", fontsize=8)
        fig.supxlabel(r"mapping frames $M$", fontsize=8)
        handles = [Line2D([], [], color=s[4], marker=s[5], lw=1.3 if s is OURS else 0.8,
                          ms=3.6 if s is OURS else 3.0, mec="white", mew=0.4,
                          ls=s[6] if len(s) > 6 else "-")
                   for s in [OURS] + REFS]
        fig.legend(handles, [s[1] for s in [OURS] + REFS], loc="outside upper center", ncols=3,
                   frameon=False, handlelength=1.6, handletextpad=0.4, columnspacing=0.9,
                   borderaxespad=0.1)
        png = png or os.path.splitext(out)[0] + ".png"
        for p in (out, png):
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        fig.savefig(out, metadata={"CreationDate": None})          # same input, same bytes
        fig.savefig(png, dpi=300)
        if keep:
            return out, png, fig
        plt.close(fig)
    return out, png


def default_out(stage, defined):
    """Test: the paper's figure in papers/figs; val: next to the data, not in the paper."""
    name = "views" + ("" if stage == "test" else f"_{stage}") + ("_defined" if defined else "")
    return os.path.join(FIGS if stage == "test" else f"{RESULTS}/views/plots", name + ".pdf")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="test", choices=["val", "test"])
    ap.add_argument("--views", default="", help="views JSON (default "
                    f"{RESULTS}/views/views_<stage>.json)")
    ap.add_argument("--tab1", default=None, help="Table I JSON for the M = 180 point (default "
                    f"{RESULTS}/tab1/eval_<stage>.json; '' = no M = 180 point)")
    ap.add_argument("--out", default="", help="output PDF (default papers/figs/views.pdf for "
                    f"test, {RESULTS}/views/plots/views_val.pdf for val, '_defined' added with "
                    "--defined); the PNG preview is written next to it")
    ap.add_argument("--defined", action="store_true", help="plot summary_on_ale_defined (the "
                    "secondary analysis; no M = 180 point)")
    ap.add_argument("--ours-row", default="ours_P0", help="row of our u_epi / u_ale")
    ap.add_argument("--m-full", type=int, default=180, help="M of the full map (nominal; the "
                    "full maps keep 176 to 180 frames)")
    ap.add_argument("--band", action="store_true", help="+- std bands instead of error bars")
    ap.add_argument("--scale", default="relative", choices=["relative", "raw"], help="y: AUSE "
                    "/ random-ranking AUSE per scene (default) or raw AUSE x100")
    ap.add_argument("--allow-mismatch", action="store_true", help="--stage test: write the "
                    "figure even when a consistency check fails")
    a = ap.parse_args(argv)
    views_path = a.views or f"{RESULTS}/views/views_{a.stage}.json"
    tab1_path = f"{RESULTS}/tab1/eval_{a.stage}.json" if a.tab1 is None else a.tab1
    views = load(views_path)
    tab1 = load(tab1_path) if tab1_path and not a.defined else None
    rel, bins = a.scale == "relative", ause_bins(views)
    data, sizes = collect(views, views_path, a.defined, tab1, tab1_path, a.ours_row, a.m_full,
                          bins, relative=rel)
    results = checks(views, tab1, a.m_full) if tab1 is not None else []
    n = len(views["scenes"])
    text = (header(a.stage, a.defined, views, views_path, tab1, tab1_path, a.m_full, results,
                   rel) + "\n\n" + table(data, a.ours_row, n, 1.0 if rel else 100.0,
                                          "curve (AUSE / random)" if rel else "curve"))
    if rel:                                  # the raw AUSE x100 too, for the report
        raw, _ = collect(views, views_path, a.defined, tab1, tab1_path, a.ours_row, a.m_full,
                         bins)
        text += "\n\n" + table(raw, a.ours_row, n, 100.0, "curve (raw AUSE x100)")
    text += "\n\n" + size_table(sizes, _ms(data), n)
    print(text)
    failed = [name for name, status, _ in results if status == "FAIL"]
    if failed and a.stage == "test" and not a.allow_mismatch:
        raise SystemExit(f"failed checks {failed}: no figure written (--allow-mismatch "
                         "overrides)")
    out = a.out or default_out(a.stage, a.defined)
    png = (None if a.out or a.stage != "test" else       # no preview PNG in papers/figs
           f"{RESULTS}/views/plots/{os.path.splitext(os.path.basename(out))[0]}_test.png")
    pdf, png = plot(data, out, a.ours_row, a.band, sizes, n, png=png, relative=rel)
    print(f"\nwrote {pdf}\nwrote {png}")
    return dict(data=data, sizes=sizes, checks=results, text=text, pdf=pdf, png=png)


if __name__ == "__main__":
    main()
