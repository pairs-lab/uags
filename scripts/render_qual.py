"""
render_qual.py — the qualitative figure (fig:qual in papers/main.tex): one evaluation frame of
one Replica scene, rendered from the scene's 3DGS map and saved field, in six panels
    (a) the RGB render of the 3DGS map (its own colours, usplat.gs_io)
    (b) the ground-truth labels      (c) the labels of ours P2 (argmax over the scene's classes)
    (d) u_epi                        (e) u_ale
    (f) the errors: right / wrong; with --kfield the k-view map's errors split into
        e_red (right with the full map) and e_irr (wrong with the full map)
Every value comes from eval_seg.py's code: load_scene (frames, vocabulary, the classes queried
for the scene, text, frozen rows), make_rasterizer, and usplat.seg_eval.evaluate_view (coverage
rule, ties, the row's own temperature from the Tab I validation fit, --betas). So the panels
show the decisions that Table I scores, and e_red / e_irr are those of eval_views.py (labelled
pixels covered in both maps). --frame i is the i-th evaluation frame (Sequence_2, every
--stride-th frame).

Rows of the figure. Without --kfield: one row, the full map. With --kfield: two rows, the full
map and the k-view map (its panels (c)-(f); its (a), (b) slots hold the class legend), so (d),
(e) of the second row are the channels the views figure ranks against its e_red / e_irr.
--one-row keeps only the k-view row.
Colours: black = not covered (alpha < tau_cov) in (c)-(f), unlabelled in (b) and (f). Class
colours: one fixed palette per scene (classes ranked by their GT pixels over all evaluation
frames of the scene), the same in (b) and (c) and in both rows. u in viridis from 0 to the
--u-quantile of the covered values shown (one scale per channel for both rows; --epi-max /
--ale-max fix it); u_ale undefined (support below tau_min views) in grey. (f): white right,
red wrong (with --kfield: red e_irr, blue e_red).
Width 7.16 in (IEEE double column, figure*), 6-7 pt text; writes <out>.pdf and <out>.png.

--list-frames N prints, instead of the figure, the N frames with the most e_red and the most
e_irr pixels (with --kfield; else the most wrong pixels of the full map), to pick a frame, and
writes the counts of every frame to --out (JSON).

The k-view fields of scripts/replica_views.sh are deleted after scoring (KEEP=0): map one again
with its run_mapping.py line (same flags, --max-keyframes k) to draw the e_red / e_irr panels.

    cd uags && Q="--vocab configs/vocab_replica.json --extractor lseg \
        --scenes configs/replica_test.json --scene room_0 --betas <tab1>/eval_val.json \
        --text-cache <fields>/text_cache_lseg.pt --kfield <fields>/room_0_lseg_k30.pt"
    scripts/gpu_gate.sh 4000 python3 scripts/render_qual.py $Q --list-frames 5 \
        --out <results>/qual/frames_room_0.json            # pick a frame
    scripts/gpu_gate.sh 4000 python3 scripts/render_qual.py $Q --frame <f> \
        --out papers/figs/qual.pdf --meta <results>/qual/qual.json
"""
from __future__ import annotations
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import eval_seg as ES                                                      # noqa: E402
from usplat.seg_eval import evaluate_view                                  # noqa: E402

FIG_W = 7.16                       # in, IEEE double-column width (figure*)
GAP, ROW_GAP, TITLE_H = 0.035, 0.03, 0.13    # in
FS_TITLE, FS_SMALL = 7, 6          # pt
BLACK = (0.0, 0.0, 0.0)            # not covered; unlabelled in (b) and (f)
UNDEF = (0.72, 0.72, 0.72)         # u_ale undefined: support below tau_min views
RIGHT = (1.0, 1.0, 1.0)            # (f) right
WRONG = (0.843, 0.098, 0.110)      # (f) wrong; e_irr = wrong at k and with the full map
E_RED = (0.169, 0.514, 0.729)      # (f) e_red = wrong at k, right with the full map
U_CMAP = "viridis"
CHANNELS = ("u_epi", "u_ale")
RC = {"font.family": "serif",
      "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "STIXGeneral",
                     "DejaVu Serif"],
      "mathtext.fontset": "stix", "font.size": FS_SMALL, "legend.fontsize": FS_SMALL,
      "axes.linewidth": 0.4, "xtick.labelsize": FS_SMALL, "xtick.major.size": 1.5,
      "xtick.major.width": 0.3, "xtick.major.pad": 1.0,
      "pdf.fonttype": 42, "ps.fonttype": 42}      # TrueType, no Type 3 fonts (IEEE)


# ------------------------------------------------------------------------------ inputs
def pick_scene(path, name):
    """The entry `name` of a scenes file (eval_seg.py format); the first one if no name."""
    scenes = ES._load_structured(path)
    names = [s["name"] for s in scenes]
    if not name:
        return scenes[0]
    if name not in names:
        raise ValueError(f"scene {name!r} not in {path}; scenes {names}")
    return scenes[names.index(name)]


def figure_row(S, args):
    """The frozen row the figure shows (--row, default ours_P2), with u_epi and u_ale."""
    rows = [r for r in S["rows"] if r.name == args.row]
    if not rows:
        raise ValueError(f"row {args.row!r} not built; rows {[r.name for r in S['rows']]}")
    miss = [c for c in CHANNELS if c not in rows[0].unc]
    if miss:
        raise ValueError(f"row {args.row!r} has no channels {miss} for panels (d), (e)")
    return rows[0]


def load(sc, vocab, betas, args, field=None):
    """eval_seg.load_scene with only the figure's row (no P3, no 2D row); a row without a
    fitted beta is refused here, before any render."""
    args.rows, args.no_p3, args.with_2d = args.row, True, False
    S = ES.load_scene(sc if field is None else dict(sc, field=field), vocab, args, args.device)
    row = figure_row(S, args)
    ES.betas_used([row], betas, ES._raw_probs(args))
    return S, row


def n_mapping_views(path):
    """Keyframes the field was built from (run_mapping.py timing), None if not recorded."""
    timing = torch.load(path, map_location="cpu", weights_only=False, mmap=True).get("timing")
    return (timing or {}).get("keyframes")


def check_frame(S, i):
    n = len(S["frames"])
    if not 0 <= i < n:
        raise ValueError(f"--frame {i}: the scene has {n} evaluation frames (0..{n - 1})")
    return i


# ------------------------------------------------------------------------------ render
@torch.no_grad()
def render_rgb(splats, ras, h, w):
    """(h, w, 3) render of the map's colours composited on black, by the rasterizer of the
    labels (one call with a ones channel, as GSplatFeatureRasterizer.render_image)."""
    c = splats["colors"].float()
    img = ras(torch.cat([c, torch.ones_like(c[:, :1])], 1))
    return img[:, :3].clamp(0, 1).reshape(h, w, 3).float().cpu()


@torch.no_grad()
def frame_maps(S, i, row, betas, args, with_rgb=False):
    """Frame i of scene S through the figure's row, at every pixel: evaluate_view with all
    pixels marked labelled (its decisions do not use the labels), scored against the real GT
    afterwards. (h, w) tensors: gt, pred (-1 = no decision: not covered or tied), covered,
    labelled, correct (labelled pixels only), u {channel: NaN = undefined}; rgb if asked."""
    fr = S["frames"][i]
    h, w = fr["h"], fr["w"]
    ras = ES.make_rasterizer(args.backend, S["splats"], fr, args)
    out = evaluate_view([row], ras, S["text"], torch.zeros(h * w, dtype=torch.long), betas,
                        args.tau_cov, undef_frac=args.undef_frac,
                        raw_probs=ES._raw_probs(args))[row.name]
    gt = fr["gt"].reshape(h, w).cpu()
    pred = out["pred"].reshape(h, w).long()
    lab = gt >= 0
    m = dict(gt=gt, pred=pred, covered=out["covered"].reshape(h, w), labelled=lab,
             correct=lab & (pred == gt), u={k: v.reshape(h, w) for k, v in out["u"].items()})
    if with_rgb:
        m["rgb"] = render_rgb(S["splats"], ras, h, w)
    return m


def frame_counts(full, k=None):
    """Pixel counts of one frame (frame_maps outputs), and the share of labelled covered pixels
    with u_ale defined (of the k-view map if given). With the k-view map's `k`, e_red and
    e_irr on the labelled pixels covered in both maps (as eval_views.py)."""
    lab = full["labelled"]
    shown = full if k is None else k
    cov = lab & shown["covered"]
    c = dict(labelled=int(lab.sum()), right_full=int(full["correct"].sum()),
             wrong_full=int((lab & ~full["correct"]).sum()),
             ale_defined=float((cov & torch.isfinite(shown["u"]["u_ale"])).sum())
             / max(int(cov.sum()), 1))
    if k is not None:
        both = cov & full["covered"]
        c.update(right_k=int(k["correct"].sum()),
                 e_red=int((both & ~k["correct"] & full["correct"]).sum()),
                 e_irr=int((both & ~k["correct"] & ~full["correct"]).sum()))
    return c


# ------------------------------------------------------------------------------ colours
def class_palette(n):
    """n class colours, none black, white or grey: the dark tab20 shades, then the light ones,
    then tab20b and tab20c (greys removed); they repeat after 54 classes."""
    import matplotlib
    t20 = list(matplotlib.colormaps["tab20"].colors)
    base = (t20[0::2] + t20[1::2] + list(matplotlib.colormaps["tab20b"].colors)
            + list(matplotlib.colormaps["tab20c"].colors))
    base = [c for c in base if max(c) - min(c) > 0.05]
    return np.array([base[i % len(base)] for i in range(n)], dtype=float)


def scene_palette(S):
    """(K, 3) colour per queried class of the scene. Classes are ranked by their GT pixels over
    all evaluation frames (ties by query order), so the palette is fixed for the scene (any
    frame, both maps) and the largest classes get the first colours."""
    K = len(S["class_ids"])
    cnt = torch.zeros(K, dtype=torch.long)
    for f in S["frames"]:
        g = f["gt"].reshape(-1)
        cnt += torch.bincount(g[g >= 0], minlength=K)[:K].cpu()
    order = torch.argsort(-cnt, stable=True).numpy()
    pal = np.zeros((K, 3))
    pal[order] = class_palette(K)
    return pal


def label_image(lab, palette):
    """(h, w, 3) class colours; -1 (unlabelled, or no decision) black."""
    lab = np.asarray(lab)
    img = np.zeros(lab.shape + (3,))
    ok = lab >= 0
    img[ok] = palette[lab[ok]]
    return img


def u_image(u, covered, vmax):
    """(h, w, 3) viridis from 0 to vmax; undefined u grey, not covered black."""
    import matplotlib
    u, cov = np.asarray(u, dtype=float), np.asarray(covered, dtype=bool)
    img = matplotlib.colormaps[U_CMAP](np.clip(np.nan_to_num(u) / vmax, 0.0, 1.0))[..., :3]
    img[cov & ~np.isfinite(u)] = UNDEF
    img[~cov] = BLACK
    return img


def error_image(full, k=None):
    """(f): white right, red wrong, black not scored (unlabelled or not covered). With the
    k-view map `k`: its errors on the pixels covered in both maps, blue e_red (right with the
    full map) and red e_irr (wrong with it)."""
    m = full if k is None else k
    ok = (m["labelled"] & m["covered"]).numpy()
    if k is not None:
        ok &= full["covered"].numpy()
    right, cf = m["correct"].numpy(), full["correct"].numpy()
    img = np.zeros(ok.shape + (3,))
    img[ok & right] = RIGHT
    if k is None:
        img[ok & ~right] = WRONG
    else:
        img[ok & ~right & cf] = E_RED
        img[ok & ~right & ~cf] = WRONG
    return img


def nice_up(v):
    """v rounded up to two significant digits (1.0 for v <= 0 or not finite)."""
    if not (v > 0 and math.isfinite(v)):
        return 1.0
    e = 10.0 ** (math.floor(math.log10(v)) - 1)
    return round(math.ceil(v / e - 1e-9) * e, 12)


def u_max(maps, ch, q, fixed=None):
    """Top of the colour bar of channel ch: `fixed`, else the q-quantile of the covered,
    defined values of every map shown, rounded up (nice_up)."""
    if fixed is not None:
        return float(fixed)
    vals = torch.cat([m["u"][ch][m["covered"] & torch.isfinite(m["u"][ch])].double()
                      for m in maps])
    return nice_up(float(torch.quantile(vals, q))) if vals.numel() else 1.0


def legend_classes(names, maps, palette, max_n=15, min_share=0.002):
    """[(name, colour)] of the classes on the frame (GT or a shown prediction), largest first,
    those with at least min_share of the frame's pixels, at most max_n."""
    K = len(names)
    gt = maps[0]["gt"].reshape(-1)
    cnt = torch.bincount(gt[gt >= 0], minlength=K)[:K]
    for m in maps:
        p = m["pred"].reshape(-1)
        cnt = cnt + torch.bincount(p[p >= 0], minlength=K)[:K]
    keep = [int(c) for c in torch.argsort(-cnt, stable=True)
            if cnt[c] > 0 and cnt[c] >= min_share * gt.numel()]
    return [(names[c], tuple(palette[c])) for c in keep[:max_n]]


# ------------------------------------------------------------------------------ figure
def titles(row, kmode):
    who = "Ours" if row == "ours_P2" else row.replace("ours_", "Ours ")       # P2 is our method
    err = "(f) $e_{\\mathrm{red}}$, $e_{\\mathrm{irr}}$" if kmode == "k" else "(f) Errors"
    return ["(a) 3DGS render", "(b) Ground truth", f"(c) {who}", "(d) $u^{\\mathrm{epi}}$",
            "(e) $u^{\\mathrm{ale}}$", err]


def error_items(kmode):
    if kmode is None:
        return [("right", RIGHT), ("wrong", WRONG), ("not scored", BLACK)]
    irr = "$e_{\\mathrm{irr}}$" if kmode == "k" else "wrong, $e_{\\mathrm{irr}}$"
    return [("right", RIGHT), ("$e_{\\mathrm{red}}$", E_RED), (irr, WRONG),
            ("not scored", BLACK)]


def _legend(ax, items, ncol, loc="upper left"):
    from matplotlib.patches import Patch
    ax.axis("off")
    if not items:
        return
    ax.legend(handles=[Patch(facecolor=c, edgecolor="black", linewidth=0.3, label=n)
                       for n, c in items], loc=loc, ncol=ncol, frameon=False,
              fontsize=FS_SMALL, borderaxespad=0.0, borderpad=0.0, handlelength=0.9,
              handleheight=0.9, handletextpad=0.3, columnspacing=0.7, labelspacing=0.25)


def _colorbar(fig, box, x, y, width, vmax, undefined):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    cb_w = 0.62 * width if undefined else 0.8 * width
    cax = box(x + 0.06 * width, y, cb_w, 0.045)
    cb = fig.colorbar(ScalarMappable(Normalize(0.0, vmax), U_CMAP), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([0.0, vmax])
    cb.set_ticklabels(["0", f"{vmax:g}"])
    cb.outline.set_linewidth(0.3)
    if undefined:                        # grey patch: u_ale undefined
        px = x + 0.06 * width + cb_w + 0.05
        pax = box(px, y, 0.045, 0.045)
        pax.add_patch(Rectangle((0, 0), 1, 1, facecolor=UNDEF, edgecolor="black",
                                linewidth=0.3))
        pax.axis("off")
        fig.text((px + 0.065) / fig.get_figwidth(), (y + 0.0225) / fig.get_figheight(),
                 "undef.", fontsize=FS_SMALL, va="center", ha="left")


def compose(rows, col_titles, classes, errors, cbars, notes=None):
    """The figure, FIG_W inches wide. rows: [(row label or None, 6 RGB arrays or None)]; a
    last row starting with two empty slots holds the class legend there, else the legend goes
    in the strip under (a)-(c). classes / errors: [(name, colour)];
    cbars: {column: (vmax, has_undefined)}, drawn under the last row."""
    import matplotlib
    from matplotlib.figure import Figure
    h, w = next(p for _, ps in rows for p in ps[:6] if p is not None).shape[:2]
    R = len(rows)
    left = 0.14 if any(lab for lab, _ in rows) else 0.01
    extra = any(len(ps) > 6 and ps[6] is not None for _, ps in rows)     # the 3D column
    pw = (FIG_W - left - 0.01 - (6 if extra else 5) * GAP) / (6 + (VIEW3D_W if extra else 0))
    ph = pw * h / w
    in_slot = R > 1 and rows[-1][1][0] is None and rows[-1][1][1] is None
    strip = 0.27 if in_slot else 0.36          # colour bars and (f) legend (+ class legend)
    H = TITLE_H + R * ph + (R - 1) * ROW_GAP + strip
    with matplotlib.rc_context(RC):
        fig = Figure(figsize=(FIG_W, H))
        box = lambda x, y, bw, bh: fig.add_axes((x / FIG_W, y / H, bw / FIG_W, bh / H))
        col_x = lambda c: left + c * (pw + GAP)
        row_y = lambda r: H - TITLE_H - (r + 1) * ph - r * ROW_GAP
        for r, (label, panels) in enumerate(rows):
            for c, img in enumerate(panels):
                if img is None:
                    continue
                cw = VIEW3D_W * pw if c == 6 else pw
                ax = box(col_x(c), row_y(r), cw, ph)
                if c == 6 or img.shape[:2] != (h, w):                    # a 3D view: crop to the cell
                    img = crop_to(img, cw / ph)
                ax.imshow(img, interpolation="nearest" if c < 6 and img.shape[:2] == (h, w) else "lanczos")
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0 and c < len(col_titles) and col_titles[c]:
                    ax.set_title(col_titles[c], fontsize=FS_TITLE, pad=2.0)
                note = (notes or {}).get((r, c))
                if note:
                    ax.text(0.03, 0.97, note, transform=ax.transAxes, ha="left", va="top",
                            fontsize=FS_SMALL, color="white",
                            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.0))
            if label:
                fig.text(0.45 * left / FIG_W, (row_y(r) + ph / 2) / H, label, rotation=90,
                         ha="center", va="center", fontsize=FS_TITLE)
        if in_slot:
            _legend(box(col_x(0), row_y(R - 1), 2 * pw + GAP, ph), classes, ncol=3)
        else:
            _legend(box(col_x(0), 0.0, 3 * pw + 2 * GAP, strip - 0.03), classes, ncol=5)
        for c, (vmax, undefined) in cbars.items():
            _colorbar(fig, box, col_x(c), strip - 0.03 - 0.045, pw, vmax, undefined)
        _legend(box(col_x(5), 0.0, pw, strip - 0.03), errors, ncol=2, loc="upper center")
    return fig


VIEW3D_W = 1.5                           # width of the 3D column, in panel widths


def crop_to(img, aspect_wh):
    """Centre crop of an image to the width / height ratio aspect_wh."""
    hh, ww = img.shape[:2]
    if ww / hh > aspect_wh:
        nw = int(round(hh * aspect_wh))
        x0 = (ww - nw) // 2
        return img[:, x0:x0 + nw]
    nh = int(round(ww / aspect_wh))
    y0 = (hh - nh) // 2
    return img[y0:y0 + nh]


def save(fig, out, dpi=600, png_dpi=300):
    """<out>.pdf (images resampled at dpi, nearest) and <out>.png; returns both paths."""
    import matplotlib
    stem = os.path.splitext(out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    with matplotlib.rc_context(RC):
        fig.savefig(stem + ".pdf", dpi=dpi)
        fig.savefig(stem + ".png", dpi=png_dpi)
    return [stem + ".pdf", stem + ".png"]


# ------------------------------------------------------------------------------- runs
def panels(full, m, palette, vmax, first):
    """The six panels of map m (full or k); (a), (b) only in the first row."""
    return [full["rgb"].numpy() if first else None,
            label_image(full["gt"], palette) if first else None,
            label_image(m["pred"], palette),
            u_image(m["u"]["u_epi"], m["covered"], vmax["u_epi"]),
            u_image(m["u"]["u_ale"], m["covered"], vmax["u_ale"]),
            error_image(full, None if m is full else m)]


def make_figure(sc, vocab, betas, args):
    """Render frame --frame of the full map (and of --kfield), draw and save the figure."""
    S, row = load(sc, vocab, betas, args)
    i = check_frame(S, args.frame)
    full = frame_maps(S, i, row, betas, args, with_rgb=True)
    palette, names, idx = scene_palette(S), list(S["class_names"]), int(S["frames"][i]["idx"])
    del S, row                                      # one map on the device at a time
    k, n_k, kmode = None, None, None
    if args.kfield:
        S, row = load(sc, vocab, betas, args, field=args.kfield)
        if S["class_names"] != names:
            raise ValueError("the k-view field queries other classes than the full field")
        k, n_k = frame_maps(S, i, row, betas, args), n_mapping_views(args.kfield)
        kmode = "k" if args.one_row else "both"
        del S, row
    shown = [full] if k is None else ([k] if args.one_row else [full, k])
    fixed = dict(u_epi=args.epi_max, u_ale=args.ale_max)
    vmax = {ch: u_max(shown, ch, args.u_quantile, fixed[ch]) for ch in CHANNELS}
    k_label = f"$M={n_k}$" if n_k else "$k$-view map"
    if k is None:
        grid = [(None, panels(full, full, palette, vmax, True))]
    elif args.one_row:
        grid = [(k_label, panels(full, k, palette, vmax, True))]
    else:
        grid = [("Full map", panels(full, full, palette, vmax, True)),
                (k_label, panels(full, k, palette, vmax, False))]
    classes = legend_classes(names, shown, palette, args.legend_max, args.legend_min_share)
    col_titles, notes = titles(args.row, kmode), {}
    if args.view3d:                     # (g) labels, (h) u_epi in a 3D column; (i) u_ale under (b)
        import matplotlib.pyplot as plt
        v = [plt.imread(p_)[..., :3] for p_ in args.view3d.split(",")]
        grid = [(lab, list(ps) + [None] * (7 - len(ps))) for lab, ps in grid]
        grid[0][1][6] = v[0]
        notes[(0, 6)] = "(g) ours, 3D view"
        if len(grid) > 1:
            grid[1][1][6], grid[1][1][1] = v[1], v[2]
            notes[(1, 6)] = "(h) $u^{\\mathrm{epi}}$, 3D view"
            notes[(1, 1)] = "(i) $u^{\\mathrm{ale}}$, 3D view"
        col_titles = list(col_titles) + ["Full map, 3D view"]
    fig = compose(grid, col_titles, classes, error_items(kmode),
                  {3: (vmax["u_epi"], False), 4: (vmax["u_ale"], True)}, notes)
    paths = save(fig, args.out, args.dpi, args.png_dpi)
    meta = dict(scene=sc["name"], frame=i, frame_index=idx, row=args.row,
                field=sc["field"], kfield=args.kfield or None, k_mapping_views=n_k,
                betas=os.path.abspath(args.betas), vmax=vmax, layout=kmode or "full",
                legend=[n for n, _ in classes], counts=frame_counts(full, k), outputs=paths)
    return dict(meta=meta, maps=dict(full=full, k=k), grid=grid, palette=palette)


def frame_table(sc, vocab, betas, args):
    """frame_counts of every evaluation frame: the full map, and the k-view map if given."""
    S, row = load(sc, vocab, betas, args)
    n, idx = len(S["frames"]), [int(f["idx"]) for f in S["frames"]]
    full = [frame_maps(S, i, row, betas, args) for i in range(n)]
    del S, row
    ks = [None] * n
    if args.kfield:
        S, row = load(sc, vocab, betas, args, field=args.kfield)
        ks = [frame_maps(S, i, row, betas, args) for i in range(n)]
        del S, row
    return [dict(frame=i, index=idx[i], **frame_counts(full[i], ks[i])) for i in range(n)]


def print_frames(table, top, name, kfield):
    """The `top` frames by e_red and by e_irr (by wrong pixels of the full map without k)."""
    keys = ("e_red", "e_irr") if kfield else ("wrong_full",)
    cols = ["frame", "index", "labelled", "acc_full"] + (
        ["acc_k", "e_red", "e_irr"] if kfield else ["wrong_full"]) + ["ale_defined"]
    print(f"{name}: {len(table)} evaluation frames"
          + (f", k-view field {kfield}" if kfield else ", full map only"))
    for key in keys:
        print(f"top {top} frames by {key} (pass the 'frame' column as --frame):")
        print(" ".join(f"{c:>11}" for c in cols))
        for t in sorted(table, key=lambda t: (-t[key], t["frame"]))[:top]:
            n = max(t["labelled"], 1)
            v = dict(t, acc_full=t["right_full"] / n, acc_k=t.get("right_k", 0) / n)
            print(" ".join(f"{v[c]:>11.3f}" if isinstance(v[c], float) else f"{v[c]:>11}"
                           for c in cols))


def run(args):
    if args.device.startswith("cuda"):
        from usplat import gpu_reserve                      # fraction from scripts/gpu_gate.sh
        gpu_reserve.apply(args.device)
    if not args.scenes or not args.betas:
        raise SystemExit("give --scenes and --betas (the eval_seg.py fit, tab1/eval_val.json)")
    vocab = ES.load_vocab(args.vocab)
    ES.check_fit_settings(ES.load_fit_settings(args.betas), args, args.betas)
    betas = ES.load_betas(args.betas)
    sc = pick_scene(args.scenes, args.scene)
    if args.list_frames:
        table = frame_table(sc, vocab, betas, args)
        print_frames(table, args.list_frames, sc["name"], args.kfield)
        out = dict(scene=sc["name"], row=args.row, field=sc["field"], kfield=args.kfield or None,
                   k_mapping_views=n_mapping_views(args.kfield) if args.kfield else None,
                   betas=os.path.abspath(args.betas), frames=table)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(ES._plain_json(out), f, indent=1)
        print(f"wrote {args.out}")
        return out
    return make_figure(sc, vocab, betas, args)


def build_parser():
    ap = ES.build_parser()
    ap.description = __doc__
    ap.add_argument("--scene", default="", help="scene name in --scenes (default: the first)")
    ap.add_argument("--frame", type=int, default=0,
                    help="evaluation frame: position in the scene's frame list (every --stride)")
    ap.add_argument("--row", default="ours_P2", help="the row shown (needs u_epi and u_ale)")
    ap.add_argument("--kfield", default="", help="k-view field: panel (f) as e_red / e_irr")
    ap.add_argument("--view3d", default="", help="three PNGs of papers/figure_tools/views3d.py scene "
                    "(labels, u_epi, u_ale of the full map): (g), (h) in a 3D column, (i) under (b)")
    ap.add_argument("--one-row", action="store_true",
                    help="with --kfield: only the k-view row (default: full map and k-view map)")
    ap.add_argument("--list-frames", type=int, default=0,
                    help="print the N frames with the most e_red / e_irr and write every "
                         "frame's counts to --out (JSON); no figure")
    ap.add_argument("--u-quantile", type=float, default=0.99,
                    help="colour bar top: this quantile of the covered values shown")
    ap.add_argument("--epi-max", type=float, default=None, help="fix the u_epi colour bar top")
    ap.add_argument("--ale-max", type=float, default=None, help="fix the u_ale colour bar top")
    ap.add_argument("--legend-max", type=int, default=15, help="classes in the legend")
    ap.add_argument("--legend-min-share", type=float, default=0.002,
                    help="legend: classes on at least this share of the frame's pixels")
    ap.add_argument("--dpi", type=int, default=600, help="resolution of the PDF's images")
    ap.add_argument("--png-dpi", type=int, default=300)
    ap.add_argument("--meta", default="", help="also write the figure's metadata here (JSON)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    t0 = time.time()
    out = run(args)
    if "meta" in out:
        out["meta"]["provenance"] = dict(
            argv=[sys.argv[0]] + list(argv if argv is not None else sys.argv[1:]),
            start=t0, wall_s=time.time() - t0)
        print(json.dumps(ES._plain_json(out["meta"])))
        if args.meta:
            with open(args.meta, "w") as f:
                json.dump(ES._plain_json(out["meta"]), f, indent=1)
        print(f"wrote {' '.join(out['meta']['outputs'])}")
    return out


if __name__ == "__main__":
    main()
