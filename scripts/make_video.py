"""
make_video.py — the anonymous supplementary video of the paper (RA-L multimedia), CPU only.

    python3 scripts/make_video.py compose --out <results>/video      # clips, master, stills
    python3 scripts/make_video.py deliver --out <results>/video      # two-pass mp4, zip, checks

Inputs (all read-only): papers/main.tex (every number in a caption is parsed from it at build
time: paper_numbers; the build fails on a parse error), the paper figures papers/figs/*.pdf,
the GPU dumps of scripts/video_gpu.sh (<out>/nor/<scene>/timeline.npz, curves.json, meta.json;
<out>/static/room_0_path.npz) and the NOR dumps (qual/nor/<scene>/frame_<i>.npz), Table III
records (nor/tab2_relmax/...) and the ScanNet Table I record (scannet_tab1/eval_test.json),
whose per-scene values appear only as overlays labelled "this scene". A segment whose dump is missing or did not pass its checks stops the build.

Format: 1920x1080, 30 fps, silent, captions burned in, white background, 0.3 s fades from and
to white at every cut. Master: H.264 CRF 12 per segment (imageio-ffmpeg, macro_block_size 8 so
1080 is not padded to 1088), joined with the concat demuxer (-c copy). Delivery: two-pass
H.264 High at the bit rate that fits --max-mb, metadata stripped (-map_metadata -1, bitexact).
Colours (paper figures): ours #0072B2, averaging #D55E00, reset/flagged #E69F00, p(query)
#CC79A7 over the greyscale map (as the teaser).
"""
from __future__ import annotations
import argparse
import functools
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = "/home/thanhnc/gauss_ws/data"
RES = os.path.join(DATA, "uags_results")

W, H, FPS = 1920, 1080, 30
MARGIN = 40                                   # safe area
MIN_FONT = 24
FADE = 9                                      # frames (0.3 s)
INK, INK2, BG = (11, 11, 11), (82, 81, 78), (255, 255, 255)
OURS, AVG, RESET, PQ = (0x00, 0x72, 0xB2), (0xD5, 0x5E, 0x00), (0xE6, 0x9F, 0x00), (0xCC, 0x79, 0xA7)
GREY = (0x89, 0x87, 0x81)
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
U_EPI, U_ALE = "u\u1d49\u1d56\u2071", "u\u1d43\u02e1\u1d49"


# ================================================================ paper numbers (main.tex)
def _clean(cell):
    c = re.sub(r"\{\\scriptsize\$\\,\\pm\\,\$[^}]*\}", "", cell)
    for cmd in ("textbf", "underline", "emph"):
        c = re.sub(r"\\%s\{([^{}]*)\}" % cmd, r"\1", c)
    return c.strip()


def _expand_multicolumn(line):
    return re.sub(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}",
                  lambda m: " & ".join([m.group(2)] * int(m.group(1))), line)


def table_rows(tex, label):
    """[[cells]] of the tabular after \\label{label} (cells cleaned, multicolumns expanded)."""
    i = tex.index("\\label{%s}" % label)
    body = tex[tex.index("\\begin{tabular}", i):tex.index("\\end{tabular}", i)]
    rows = []
    for line in body.split("\\\\"):
        line = "\n".join(s for s in line.split("\n") if not s.strip().startswith("%"))
        line = re.sub(r"\\(toprule|midrule|bottomrule|cmidrule(\([^)]*\))?\{[^}]*\})", "", line)
        if "&" not in line:
            continue
        rows.append([_clean(c) for c in _expand_multicolumn(line).split("&")])
    return rows


def _row(rows, prefix, nth=0, ncells=None):
    hits = [r for r in rows if r[0].startswith(prefix) and (ncells is None or len(r) == ncells)]
    if len(hits) <= nth:
        raise ValueError(f"row {prefix!r} (#{nth}) not found")
    return hits[nth]


def _num(s):
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        raise ValueError(f"not a number: {s!r}")
    return s


def _find(tex, pattern, n=1):
    m = re.search(pattern, tex, flags=re.S)
    if not m:
        raise ValueError(f"pattern not found in main.tex: {pattern}")
    return [m.group(k) for k in range(1, n + 1)]


def paper_numbers(path):
    """{key: number as printed in main.tex}. Raises on any parse failure. RA-L takes no
    supplementary text, so the supplement is not a source (2026-10-27)."""
    tex = open(path).read()
    N = {}
    t1 = table_rows(tex, "tab:replica")      # Method | Replica 7 cols | ScanNet 7 cols
    col = dict(miou=1, ece=3, nll=4, ause=7)
    rows = dict(avg="Averaging", f3dgs="Feature 3DGS", legs="LEGaussians", p0="Ours P0",
                p2="Ours P2", cinter="NormLift\\reimpl, $1-C")
    for rk, prefix in rows.items():
        r = _row(t1, prefix)
        for ds, off in (("rep", 0), ("sn", 7)):
            for cn, c in col.items():
                v = r[c + off]
                if re.fullmatch(r"-?\d+(\.\d+)?", v):
                    N[f"t1.{ds}.{cn}.{rk}"] = v
    N["t1.std.rep"], N["t1.std.sn"] = _find(
        tex, r"the std of mIoU over scenes \(([\d.]+--[\d.]+), ([\d.]+--[\d.]+)\)", 2)
    N["t1.std.rep"], N["t1.std.sn"] = (s.replace("--", "\u2013") for s in
                                       (N["t1.std.rep"], N["t1.std.sn"]))
    N["nll.f3dgs.n"], N["nll.f3dgs.wins"] = _find(
        tex, r"Over the (\d+) test scenes, a sign test finds no difference between ours and "
             r"averaging in any metric, while ours has a lower NLL than Feature 3DGS in (\d+) scenes", 2)
    v = _find(tex, r"\$u\^\{\\mathrm\{epi\}\}\$ ranks \$e_\{\\mathrm\{red\}\}\$ better than "
                   r"\$u\^\{\\mathrm\{ale\}\}\$ \(([\d.]+) vs ([\d.]+) at \$M=10\$", 2)
    N["views.epi_red_10"], N["views.ale_red_10"] = v
    v = _find(tex, r"\$u\^\{\\mathrm\{ale\}\}\$ ranks \$e_\{\\mathrm\{irr\}\}\$ better than "
                   r"\$u\^\{\\mathrm\{epi\}\}\$ \(([\d.]+) vs ([\d.]+) at \$M=90\$", 2)
    N["views.ale_irr_90"], N["views.epi_irr_90"] = v
    t3 = table_rows(tex, "tab:change")       # Method | Input | G AP | G AUROC | P AP | P AUROC | FA | Rec
    for rk, prefix in (("ncc", "NCC"), ("depth", "Depth residual"), ("feat", "Feature residual"),
                       ("nl", "NormLift reliability"), ("disp", "Dispersion"),
                       ("ours", "Ours: test"), ("fused", "Ours fused")):
        r = _row(t3, prefix)
        N[f"t3.{rk}.gap"] = _num(r[2])
        N[f"t3.{rk}.pap"] = _num(r[4])
        if rk != "fused":
            N[f"t3.{rk}.fa"], N[f"t3.{rk}.rec"] = _num(r[6]), _num(r[7])
    for rk, prefix in (("ours", "Ours fused"), ("ncc", "NCC residual +"),       # bottom block
                       ("feat", "Feature residual, $k=2$, +"), ("nl", "NormLift reliability drop +"),
                       ("disp", "Dispersion $1-\\hat\\rho$ +")):
        N[f"fused.{rk}.gap"] = _num(_row(t3, prefix)[2])
    N["stale.before"], N["stale.avg"], N["stale.ours"] = _find(
        tex, r"(\d+)\\% after mapping, (\d+)\\% after the update for averaging and for our "
             r"field with the change test off, and (\d+)\\% for ours", 3)
    a, o = _find(tex, r"still answer its class from (\d+)\\% \(averaging\) to (\d+)\\%", 2)
    if (a, o) != (N["stale.avg"], N["stale.ours"]):
        raise ValueError("abstract and results disagree on the stale-answer rate")
    ab = table_rows(tex, "tab:ablation")
    full = [r for r in ab if r[0] == "Full method" and len(r) == 6][-1]
    nodep = _row(ab, "Re-admission without depth")
    N["kept.full"], N["kept.nodepth"] = _num(full[5]), _num(nodep[5])
    N["lim.rate"] = _find(tex, r"measured median (?:rate )?is \$([\d.]+)\\times10\^\{-3\}\$", 1)[0]
    N["title"] = _find(tex, r"\\title\{([^}]*)\}", 1)[0]
    N["teaser.rec"], N["teaser.fa"] = _find(
        tex, r"\(([\d.]+)\\% of the stale, ([\d.]+)\\% of the static ones\)", 2)
    return N


def fill(template, nums):
    """Replace {key} by nums[key]; an unknown key raises KeyError."""
    return re.sub(r"\{([a-z0-9_.]+)\}", lambda m: nums[m.group(1)], template)


# ===================================================================== drawing helpers
_FONTS = {}


def font(size, bold=False):
    if size < MIN_FONT:
        raise ValueError(f"font {size} px < {MIN_FONT}")
    k = (size, bold)
    if k not in _FONTS:
        _FONTS[k] = ImageFont.truetype(os.path.join(
            FONT_DIR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)
    return _FONTS[k]


_BOTTOM = [0]


def blank():
    _BOTTOM[0] = 0
    return Image.new("RGB", (W, H), BG)


def in_safe(x0, y0, x1, y1):
    if not (MARGIN <= x0 and MARGIN <= y0 and x1 <= W - MARGIN and y1 <= H - MARGIN):
        raise AssertionError(f"box {(x0, y0, x1, y1)} leaves the safe area")


def wrap(s, f, width):
    lines, cur = [], ""
    for word in s.split():
        t = (cur + " " + word).strip()
        if f.getlength(t) <= width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def text(img, xy, s, size=28, bold=False, fill=INK, anchor="la", width=None):
    """Draw (wrapped to width if given) text; returns the bottom y."""
    d, f = ImageDraw.Draw(img), font(size, bold)
    lines = wrap(s, f, width) if width else [s]
    x, y = xy
    lh = int(size * 1.3)
    for ln in lines:
        bb = d.textbbox((x, y), ln, font=f, anchor=anchor)
        in_safe(*bb)
        _BOTTOM[0] = max(_BOTTOM[0], bb[3])
        d.text((x, y), ln, font=f, fill=fill, anchor=anchor)
        y += lh
    return y


CAPTION_LINES = 3


def caption(img, s, size=30):
    """Caption block at the bottom of the safe area (at most 3 lines)."""
    f = font(size)
    lines = wrap(s, f, W - 2 * MARGIN - 20)
    if len(lines) > CAPTION_LINES:
        raise AssertionError(f"caption needs {len(lines)} lines: {s!r}")
    lh = int(size * 1.32)
    y = H - MARGIN - lh * len(lines)
    if _BOTTOM[0] > y - 16:
        raise AssertionError(f"caption at y={y - 14} overlaps content ending at {_BOTTOM[0]}")
    d = ImageDraw.Draw(img)
    d.line([(MARGIN, y - 14), (W - MARGIN, y - 14)], fill=(200, 200, 200), width=2)
    for ln in lines:
        bb = d.textbbox((MARGIN + 10, y), ln, font=f)
        in_safe(*bb)
        d.text((MARGIN + 10, y), ln, font=f, fill=INK)
        y += lh
    return img


def title(img, s, size=40, sub=None):
    d = ImageDraw.Draw(img)
    d.text((MARGIN, MARGIN), s, font=font(size, True), fill=INK)
    if sub:
        d.text((W - MARGIN, MARGIN + 10), sub, font=font(26), fill=INK2, anchor="ra")
    return img


def to_pil(a):
    a = np.asarray(a)
    if a.dtype != np.uint8:
        a = (np.clip(np.nan_to_num(a.astype(np.float32)), 0, 1) * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(a)


def paste(img, panel, x, y, w=None, h=None, border=None, label=None, label_size=26):
    """Paste a panel (PIL or array) resized to (w, h) at (x, y); optional coloured border and
    a label above it."""
    p = panel if isinstance(panel, Image.Image) else to_pil(panel)
    if w is not None:
        p = p.resize((int(w), int(h)), Image.LANCZOS)
    in_safe(x, y, x + p.width, y + p.height)
    img.paste(p, (int(x), int(y)))
    _BOTTOM[0] = max(_BOTTOM[0], int(y) + p.height + (3 if border is not None else 0))
    d = ImageDraw.Draw(img)
    if border is not None:
        d.rectangle([x - 3, y - 3, x + p.width + 2, y + p.height + 2], outline=border, width=4)
    if label:
        bb = d.textbbox((x + p.width / 2, y - 10), label, font=font(label_size, True),
                        anchor="md")
        in_safe(*bb)
        d.text((x + p.width / 2, y - 10), label, font=font(label_size, True), fill=INK,
               anchor="md")
    return p.width, p.height


def grey(rgb):
    rgb = np.asarray(rgb, np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.repeat(g[..., None], 3, -1)


def tint(base_rgb, amount, colour, black=None):
    """base in greyscale (lightened), blended towards colour by amount in [0, 1] (NaN: 0)."""
    g = 0.25 + 0.75 * grey(base_rgb)
    a = np.clip(np.nan_to_num(np.asarray(amount, np.float32)), 0, 1)[..., None]
    c = np.array(colour, np.float32) / 255.0
    out = g * (1 - a) + c * a
    if black is not None:
        out[~black] = 0.0
    return out


def outline(img_arr, mask, colour=(1.0, 1.0, 1.0), width=2):
    """Draw the boundary of a boolean mask (resized to the image) on a float RGB image."""
    h, w = img_arr.shape[:2]
    m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
    mi = Image.fromarray(m.astype(np.uint8) * 255)
    er = np.asarray(mi.filter(ImageFilter.MinFilter(2 * width + 1))) > 0
    edge = m & ~er
    out = np.array(img_arr, np.float32, copy=True)
    if out.max() > 1.5:
        out = out / 255.0
    dark = np.asarray(Image.fromarray(edge.astype(np.uint8) * 255).filter(
        ImageFilter.MaxFilter(3))) > 0
    out[dark & ~edge] = 0.0
    out[edge] = colour
    return out


def rot(a, k):
    """Display turn: k quarter turns clockwise (dump_nor_qual.quarter_turns)."""
    return np.rot90(a, -int(k), axes=(0, 1)).copy() if k else a


def colorbar(img, x, y, w, h, colours, lo, hi, label):
    """A horizontal bar from colours[0] to colours[1] (or a callable cmap) with ends."""
    t = np.linspace(0, 1, w)[None, :, None]
    if callable(colours):
        bar = colours(t[..., 0])[..., :3]
    else:
        c0, c1 = (np.array(c, np.float32) / 255 for c in colours)
        bar = c0 * (1 - t) + c1 * t
    bar = np.repeat(bar, h, 0)
    paste(img, bar, x, y)
    d = ImageDraw.Draw(img)
    d.rectangle([x, y, x + w, y + h], outline=INK, width=1)
    f = font(24)
    d.text((x, y + h + 4), lo, font=f, fill=INK, anchor="la")
    d.text((x + w, y + h + 4), hi, font=f, fill=INK, anchor="ra")
    d.text((x + w / 2, y - 6), label, font=f, fill=INK, anchor="md")


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ========================================================================== inputs
def pdf_png(pdf, out_png, width=None, height=None):
    """Rasterise one PDF page with pdftoppm (fit to width or height)."""
    if not os.path.exists(out_png) or os.path.getmtime(out_png) < os.path.getmtime(pdf):
        stem = out_png[:-4]
        arg = ["-scale-to-x", str(width), "-scale-to-y", "-1"] if width else \
              ["-scale-to-y", str(height), "-scale-to-x", "-1"]
        subprocess.run(["pdftoppm", "-png", "-singlefile", *arg, pdf, stem], check=True)
    return Image.open(out_png).convert("RGB")


def load_nor(out, scene):
    d = os.path.join(out, "nor", scene)
    meta = json.load(open(os.path.join(d, "meta.json")))
    if meta.get("status") != "done" or not meta["reproduction"]["ok"]:
        raise SystemExit(f"NOR timeline {scene}: not done or not reproduced")
    z = np.load(os.path.join(d, "timeline.npz"))
    curves = json.load(open(os.path.join(d, "curves.json")))
    return meta, z, curves


def record(scene):
    st = "val" if scene in ("001", "002", "003", "004") else "test"
    p = os.path.join(RES, "nor", "tab2_relmax", st, f"{scene}.json")
    return json.load(open(p))["metrics"], p


# ======================================================================== segments
class Seg:
    """One shot: n frames from frame(i) -> PIL image; captions and sources for the storyboard."""

    def __init__(self, sid, name, n, frame, captions, sources, still=None, local=()):
        self.sid, self.name, self.n, self.frame = sid, name, int(n), frame
        self.captions, self.sources = captions, sources
        self.local = set(local)          # caption numbers read from data (frame ids, this scene)
        self.still = still if still is not None else self.n // 2


def seconds(s):
    return int(round(s * FPS))


def seg_title(N):
    def frame(i):
        img = blank()
        d = ImageDraw.Draw(img)
        lines = wrap(N["title"], font(56, True), W - 2 * MARGIN - 200)
        y = 330
        for ln in lines:
            d.text((W / 2, y), ln, font=font(56, True), fill=INK, anchor="ma")
            y += 76
        d.text((W / 2, y + 40), "Supplementary video", font=font(36), fill=INK2, anchor="ma")
        d.text((W / 2, y + 100), "Anonymous submission. Silent; captions on screen.",
               font=font(30), fill=INK2, anchor="ma")
        return img
    return Seg("S0", "title", seconds(4), frame, [], ["main.tex \\title"])


def seg_questions(N, static, nor002):
    sp, sm = static
    j = int(np.where(sp["index"] == 290)[0][0])
    vmax = sm["vmax"]
    import render_qual as RQ
    import torch
    epi = RQ.u_image(sp["k_u_epi"][j].astype(np.float32), sp["k_covered"][j], vmax["u_epi"])
    ale = RQ.u_image(sp["full_u_ale"][j].astype(np.float32), sp["full_covered"][j], vmax["u_ale"])
    meta, z, _ = nor002
    kq = meta["quarter_turns"]
    before = rot(outline(z["fixed_map_rgb"].astype(np.float32), z["fixed_mask"]), kq)
    after = rot(z["fixed_photo"], kq)
    cap = ("A robot that reuses a semantic map must know how much evidence supports each part "
           "and must stop trusting content that has changed.")

    def frame(i):
        img = title(blank(), "Three questions about a semantic 3D map")
        y = 190
        paste(img, epi, 60, y, 560, 420, label=f"How much evidence?  ({U_EPI})")
        paste(img, ale, 680, y, 560, 420, label=f"Do the views agree?  ({U_ALE})")
        paste(img, before, 1330, y, 270, 360, label="Is it still there?")
        paste(img, after, 1610, y, 270, 360)
        d = ImageDraw.Draw(img)
        d.text((1465, y + 372), "map", font=font(24), fill=INK2, anchor="ma")
        d.text((1745, y + 372), "camera, later", font=font(24), fill=INK2, anchor="ma")
        text(img, (60, y + 440), "Replica room_0 (test), held-out frame 290: few mapping views "
             "(left, 30 of 180 frames) vs the full map (middle).", size=26, fill=INK2,
             width=1180)
        text(img, (1330, y + 420), "NeRF-Object-Removal 002 (validation): the chair was "
             "removed after mapping.", size=26, fill=INK2, width=540)
        caption(img, cap)
        return img
    return Seg("S1", "questions", seconds(7), frame, [cap],
               ["main.tex abstract, first sentence", "static/room_0_path.npz frame 290",
                "nor/002/timeline.npz (fixed view, frame 373)"])


PIPE_SPOTS = [   # (x0, y0, x1, y1) as fractions of pipeline_v2.pdf; caption (Fig. 1 / Sec. III)
    ((0.000, 0.085, 0.127, 0.885), "(1-2) Each keyframe gives a posed image (and depth, if "
     "available); LSeg turns it into dense features, compressed to 64 dimensions."),
    ((0.129, 0.092, 0.660, 0.482), "(3-5) One render and one backward pass give every Gaussian "
     "its share of the view's features; each keyframe counts as at most one observation per "
     "Gaussian, kept in four running sums."),
    ((0.676, 0.085, 0.853, 0.885), f"(6) The sums give a vMF posterior: the epistemic {U_EPI} "
     f"from the weighted number of views, the aleatoric {U_ALE} from how well the views agree."),
    ((0.129, 0.522, 0.490, 0.940), "(7) A change test compares all earlier evidence with the "
     "current view; its threshold θ(h) is taken without labels from the map's own mapping pass."),
    ((0.500, 0.522, 0.853, 0.885), "(8) A Gaussian that fails the test is reset; with depth, it "
     "rejoins the robust map only after views whose observed depth matches its position."),
    ((0.862, 0.085, 1.000, 0.940), "(9) A predictive render answers text queries; P2 sets each "
     "pixel's temperature from both uncertainties."),
]


def seg_pipeline(figs):
    pdf = os.path.join(REPO, "papers", "figs", "pipeline_v2.pdf")
    fig = pdf_png(pdf, os.path.join(figs, "pipeline_v2_1840.png"), width=W - 2 * MARGIN)
    fx, fy = MARGIN, 150
    cap0 = ("Method: one keyframe update, run over the mapping pass and then over the update "
            "pass (the paper's pipeline figure).")
    per = seconds(3.0)
    n_all = seconds(4)
    arr = np.asarray(fig, np.float32)

    def spot(k):
        (a, b, c, e), _ = PIPE_SPOTS[k]
        x0, y0, x1, y1 = (int(a * fig.width), int(b * fig.height), int(c * fig.width),
                          int(e * fig.height))
        out = arr * 0.3 + 255 * 0.7
        out[y0:y1, x0:x1] = arr[y0:y1, x0:x1]
        im = Image.fromarray(out.astype(np.uint8))
        ImageDraw.Draw(im).rectangle([x0, y0, x1 - 1, y1 - 1], outline=OURS, width=4)
        return im

    spots = [spot(k) for k in range(len(PIPE_SPOTS))]

    def frame(i):
        img = title(blank(), "Method")
        if i < n_all:
            paste(img, fig, fx, fy)
            return caption(img, cap0)
        k = min((i - n_all) // per, len(spots) - 1)
        r = (i - n_all) - k * per
        im = spots[k]
        if r < FADE and k > 0:
            im = Image.blend(spots[k - 1], spots[k], (r + 1) / FADE)
        paste(img, im, fx, fy)
        return caption(img, PIPE_SPOTS[k][1])
    return Seg("S2a", "pipeline", n_all + per * len(PIPE_SPOTS), frame,
               [cap0] + [c for _, c in PIPE_SPOTS],
               ["main.tex fig:pipeline (papers/figs/pipeline_v2.pdf) and its caption; abstract; "
                "Sec. III"], still=n_all + per * 3 + per // 2)


def seg_plane(figs):
    pdf = os.path.join(REPO, "papers", "figs", "uncertainty_plane.pdf")
    fig = pdf_png(pdf, os.path.join(figs, "plane_700.png"), height=700)
    cap = ("Little evidence and conflicting evidence stay separate: our agreement estimate stays "
           "close to the true agreement at few views; a fired change test resets the support to "
           "at most one view.")

    def frame(i):
        img = title(blank(), "Two uncertainties from the same sums")
        paste(img, fig, (W - fig.width) // 2, 130)
        return caption(img, cap)
    return Seg("S2b", "plane", seconds(6), frame, [cap],
               ["papers/figs/uncertainty_plane.pdf (no numbers in the caption)"])


def seg_static_path(static):
    import render_qual as RQ
    import torch
    sp, sm = static
    vmax, names, pal = sm["vmax"], sm["class_names"], sp["palette"]
    idx = sp["index"]
    n = len(idx)
    f_qual = int(sm["frame_index"])                 # fig:qual's frame (its meta)
    j290 = int(np.where(idx == f_qual)[0][0])
    play = [j for j in range(n)]                       # 15 fps: 2 video frames per frame
    hold = seconds(3)
    order = [j for j in play for _ in (0, 1)] + [j290] * hold
    pw, ph, gap, x0 = 282, 212, 8, 140
    xs = [x0 + c * (pw + gap) for c in range(6)]
    y1, y2 = 150, 150 + ph + 48
    cols = ["3DGS render", "Ground truth", "Ours P2 labels", U_EPI, U_ALE, "Errors"]
    t = lambda a: torch.from_numpy(np.asarray(a))

    def maps(tag, j):
        m = dict(labelled=t(sp["full_labelled"][j]), covered=t(sp[f"{tag}_covered"][j]),
                 correct=t(sp[f"{tag}_correct"][j]))
        return m

    # class legend: classes covering >= 0.5% of the pixels over the whole path (GT or labels)
    K = len(names)
    cnt = np.zeros(K)
    for key in ("full_gt", "full_pred", "k_pred"):
        a = sp[key].reshape(-1)
        cnt += np.bincount(a[a >= 0], minlength=K)[:K]
    legend = [c for c in np.argsort(-cnt, kind="stable") if cnt[c] >= 0.005 * sp["full_gt"].size][:14]
    cap_play = (f"Replica room_0 (test), held-out frames {int(idx[0])}\u2013{int(idx[-1])}. Top: full map. Bottom: a "
                "field from M = 30 of the 180 mapping frames on the same geometry. Errors at "
                "M = 30: blue reducible, red irreducible. Scales as in the paper.")
    cap_hold = (f"Frame {f_qual}: with 30 frames, {U_EPI} rises, also "
                f"where the reducible errors are, while {U_ALE} is still undefined on most "
                f"pixels; with the full map, {U_ALE} is highest at the rug, where labels stay wrong.")

    @functools.lru_cache(maxsize=2)
    def rows(j):
        full, k = maps("full", j), maps("k", j)
        row1 = [sp["full_rgb"][j], RQ.label_image(sp["full_gt"][j], pal),
                RQ.label_image(sp["full_pred"][j], pal),
                RQ.u_image(sp["full_u_epi"][j].astype(np.float32), sp["full_covered"][j],
                           vmax["u_epi"]),
                RQ.u_image(sp["full_u_ale"][j].astype(np.float32), sp["full_covered"][j],
                           vmax["u_ale"]),
                RQ.error_image(full)]
        row2 = [None, None, RQ.label_image(sp["k_pred"][j], pal),
                RQ.u_image(sp["k_u_epi"][j].astype(np.float32), sp["k_covered"][j],
                           vmax["u_epi"]),
                RQ.u_image(sp["k_u_ale"][j].astype(np.float32), sp["k_covered"][j],
                           vmax["u_ale"]),
                RQ.error_image(full, k)]
        return row1, row2

    def frame(i):
        j = order[i]
        img = title(blank(), "Static scene: both uncertainties along a held-out path",
                    sub=f"frame {int(idx[j])}")
        d = ImageDraw.Draw(img)
        row1, row2 = rows(j)
        for c in range(6):
            paste(img, row1[c], xs[c], y1, pw, ph, label=cols[c], label_size=24)
            if row2[c] is not None:
                paste(img, row2[c], xs[c], y2, pw, ph)
        for y, lab in ((y1, "Full map"), (y2, "M = 30")):
            tmp = Image.new("RGB", (ph, 40), BG)
            ImageDraw.Draw(tmp).text((ph / 2, 20), lab, font=font(26, True), fill=INK,
                                     anchor="mm")
            img.paste(tmp.rotate(90, expand=True), (x0 - 50, y))
        # legend in the empty slots of row 2
        lx, ly = xs[0], y2
        for q, c in enumerate(legend):
            cx, cy = lx + (q % 2) * (pw + gap), ly + (q // 2) * 30
            d.rectangle([cx, cy + 4, cx + 22, cy + 26], fill=tuple(int(255 * v) for v in pal[c]),
                        outline=INK)
            d.text((cx + 30, cy), names[c], font=font(24), fill=INK)
        yb = y2 + ph + 40
        import matplotlib
        colorbar(img, xs[3] + 20, yb, pw - 40, 18, matplotlib.colormaps["viridis"], "0",
                 f"{vmax['u_epi']:g}", U_EPI)
        colorbar(img, xs[4] + 20, yb, pw - 40, 18, matplotlib.colormaps["viridis"], "0",
                 f"{vmax['u_ale']:g}", f"{U_ALE} (grey: < 3 views)")
        ex = xs[5]
        for q, (lab, col) in enumerate((("right", (255, 255, 255)), ("wrong / irreducible",
                                        (215, 25, 28)), ("reducible (M = 30)", (43, 131, 186)),
                                        ("not scored", (0, 0, 0)))):
            cy = yb - 10 + q * 30
            d.rectangle([ex, cy + 4, ex + 22, cy + 26], fill=col, outline=INK)
            d.text((ex + 30, cy), lab, font=font(24), fill=INK)
        d.text((xs[0], yb + 120 + 0), "Rendered at 640x480; Table I scores 320x240.",
               font=font(24), fill=INK2)
        return caption(img, cap_play if i < 2 * n else cap_hold)
    return Seg("S3a", "static_path", len(order), frame, [cap_play, cap_hold],
               ["static/room_0_path.npz (scripts/video_static_path.py)",
                "main.tex fig:qual caption"], still=2 * n + 5,
               local=[str(int(idx[0])), str(int(idx[-1])), str(f_qual)])


def seg_views(figs, N):
    pdf = os.path.join(REPO, "papers", "figs", "views.pdf")
    fig = pdf_png(pdf, os.path.join(figs, "views_720.png"), height=720)
    cap = fill(f"With few views, {U_EPI} ranks the reducible errors better than {U_ALE} "
               "({views.epi_red_10} vs {views.ale_red_10} at M = 10); with many views, "
               f"{U_ALE} ranks the irreducible errors better than {U_EPI} "
               "({views.ale_irr_90} vs {views.epi_irr_90} at M = 90). Replica test, N = 6.", N)

    def frame(i):
        img = title(blank(), "Does each uncertainty rank its own error?")
        paste(img, fig, (W - fig.width) // 2, 120)
        return caption(img, cap)
    return Seg("S3b", "views", seconds(7), frame, [cap],
               ["main.tex fig:views and Sec. V-C text (0.34 vs 0.90; 0.48 vs 0.83)"])


def seg_compare(figs, N):
    pdf = os.path.join(REPO, "papers", "figs", "qual_compare.pdf")
    fig = pdf_png(pdf, os.path.join(figs, "qual_compare_1830.png"), width=W - 2 * MARGIN - 10)
    a = np.asarray(fig.convert("L")) < 250
    blank_rows = np.where(~a.any(1))[0]
    mid = blank_rows[np.argmin(np.abs(blank_rows - fig.height / 2))]      # gap between blocks
    top = fig.crop((0, 0, fig.width, mid))
    head_end = blank_rows[(blank_rows > 10) & (blank_rows < mid)][0] + 2   # column titles
    low = fig.crop((0, mid, fig.width, fig.height))
    both = Image.new("RGB", (fig.width, head_end + low.height), BG)
    both.paste(fig.crop((0, 0, fig.width, head_end)), (0, 0))
    both.paste(low, (0, head_end))
    halves = [top, both]
    per = seconds(4)
    qc = scene_miou(os.path.join(RES, "scannet_tab1", "eval_test.json"), "scene0140_00",
                    dict(ours="ours_P2", f3dgs="f3dgs", d2="2d_features", avg="averaging"))
    cap = ("Every method on one Replica and one ScanNet test frame, chosen by rules fixed before "
           f"rendering. ScanNet frame, this scene: ours ({qc['ours']} mIoU) is below Feature 3DGS "
           f"({qc['f3dgs']}), the 2D features ({qc['d2']}) and averaging ({qc['avg']}).")

    cap1 = ("Every method on one Replica test frame (scene and frame chosen by rules fixed before "
            "rendering): labels, errors and each method's own uncertainty.")

    def frame(i):
        img = blank()
        h = halves[min(i // per, 1)]
        paste(img, h, (W - h.width) // 2, MARGIN + 10)
        return caption(img, cap1 if i < per else cap)
    return Seg("S3c", "compare", 2 * per, frame, [cap1, cap],
               ["papers/figs/qual_compare.pdf",
                "scannet_tab1/eval_test.json scene0140_00 rows (per-scene mIoU)"],
               local=qc.values())


def scene_miou(path, scene, rows):
    """{key: per-scene mIoU in % (1 decimal) of each eval_seg.py row} of one scene."""
    d = json.load(open(path))
    if d.get("status", "ok") != "ok":
        raise ValueError(f"{path}: status {d.get('status')!r}")
    r = (d["metrics"] if isinstance(d.get("metrics"), dict) else d)["scenes"][scene]["rows"]
    return {k: f"{100 * r[v]['all']['miou']:.1f}" for k, v in rows.items()}


def seg_table1(N):
    rows = [("Averaging", "avg", "avg"), ("Feature 3DGS (trained)", "f3dgs", "f3dgs"),
            ("LEGaussians (trained)", "legs", "legs"), ("Ours (P0 labels, P2 predictive)", "p0", "p2")]
    cap = fill("Ours ties averaging: over the 14 test scenes a sign test finds no difference in "
               "any metric. Lifting without training is better calibrated than the trained maps: "
               "ours has a lower NLL than Feature 3DGS in {nll.f3dgs.wins} of {nll.f3dgs.n} "
               "test scenes.", N)
    note = fill("Mean over test scenes (Replica N = 6, ScanNet N = 8). The std of mIoU over "
                "scenes ({t1.std.rep}, {t1.std.sn}) exceeds most gaps between rows.", N)

    def frame(i):
        img = title(blank(), "Segmentation and calibration (Table I)")
        d = ImageDraw.Draw(img)
        cx = [120, 760, 960, 1160, 1400, 1600, 1800]
        heads = ["", "mIoU \u2191", "ECE \u2193", "NLL \u2193", "mIoU \u2191", "ECE \u2193",
                 "NLL \u2193"]
        y = 200
        d.text(((cx[1] + cx[3]) / 2, y), "Replica", font=font(32, True), fill=INK, anchor="ma")
        d.text(((cx[4] + cx[6]) / 2, y), "ScanNet", font=font(32, True), fill=INK, anchor="ma")
        y += 60
        for c, h in enumerate(heads):
            d.text((cx[c], y), h, font=font(30, True), fill=INK, anchor="la" if c == 0 else "ma")
        d.line([(100, y + 48), (1840, y + 48)], fill=INK, width=2)
        y += 70
        for name, km, kc in rows:
            col = OURS if km == "p0" else (AVG if km == "avg" else INK)
            vals = [N[f"t1.rep.miou.{km}"], N[f"t1.rep.ece.{kc}"], N[f"t1.rep.nll.{kc}"],
                    N[f"t1.sn.miou.{km}"], N[f"t1.sn.ece.{kc}"], N[f"t1.sn.nll.{kc}"]]
            d.text((cx[0], y), name, font=font(32), fill=col)
            for c, v in enumerate(vals):
                d.text((cx[c + 1], y), v, font=font(32), fill=col, anchor="ma")
            y += 64
        d.line([(100, y), (1840, y)], fill=INK, width=2)
        text(img, (120, y + 30), note, size=26, fill=INK2, width=1700)
        return caption(img, cap)
    return Seg("S3d", "table1", seconds(8), frame, [note, cap],
               ["main.tex tab:replica (rows Averaging, Feature 3DGS, LEGaussians, Ours P0/P2; "
                "caption std note)", "main.tex Sec. IV-C (sign test, 13 of 14)"])


def seg_nor_setup(nor002):
    meta, z, _ = nor002
    kq = meta["quarter_turns"]
    m = rot(z["fixed_map_rgb"].astype(np.float32), kq)
    p = rot(z["fixed_photo"], kq)
    cap = ("Real object removal (NeRF-Object-Removal): each scene is recorded twice, before and "
           "after an object is removed. We map the first sequence and update with the second. "
           "Geometry stays fixed; only the semantics are updated.")

    def frame(i):
        img = title(blank(), "Changing scenes: a chair is removed",
                    sub="validation scene 002, frame 373")
        paste(img, m, 420, 150, 480, 640, label="Map render after mapping")
        paste(img, p, 1020, 150, 480, 640, label="Camera during the update")
        return caption(img, cap)
    return Seg("S4a", "nor_setup", seconds(6), frame, [cap],
               ["main.tex Sec. V-A (Object removal)", "qual/nor/002/frame_373.npz"])


def curves_panel(curves, j, query, w, h, landscape):
    """Matplotlib panel of the measured curves up to update keyframe j (0 = after mapping)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    x = np.arange(len(curves["views"]))
    rc = {"font.family": "DejaVu Sans", "font.size": 17, "axes.linewidth": 1.0}
    with matplotlib.rc_context(rc):
        fig = Figure(figsize=(w / 100, h / 100))
        axs = fig.subplots(1, 2) if not landscape else fig.subplots(2, 1)
        a, b = axs
        hx = lambda c: np.array(c) * 100
        for name, col in (("averaging", AVG), ("ours", OURS)):
            c = np.array(col) / 255
            a.plot(x[:j + 1], hx(curves[name]["stale_is_query"])[:j + 1], color=c, lw=3,
                   label=f"{'Averaging' if name == 'averaging' else 'Ours'}: removed, top '{query}'")
            a.plot(x[:j + 1], hx(curves[name]["static_changed"])[:j + 1], color=c, lw=2.5,
                   ls="--", label=f"{'Averaging' if name == 'averaging' else 'Ours'}: static, top changed")
        r = np.array(RESET) / 255
        b.plot(x[:j + 1], hx(curves["reset_stale"])[:j + 1], color=r, lw=3, label="removed object")
        b.plot(x[:j + 1], hx(curves["reset_static"])[:j + 1], color=r, lw=2.5, ls="--",
               label="static")
        b.plot(x[:j + 1], hx(curves["changed_static"])[:j + 1], color=np.array(GREY) / 255,
               lw=2.5, ls=":", label="static, still flagged")
        for ax, keys_ in ((a, (("averaging", "stale_is_query", AVG), ("ours", "stale_is_query", OURS))),
                          (b, ((None, "reset_stale", RESET),))):
            for name, key, col in keys_:
                yv = hx(curves[name][key] if name else curves[key])[j]
                ax.plot([x[j]], [yv], "o", color=np.array(col) / 255, ms=7)
        for ax, t in ((a, "Top class (% of Gaussians)"), (b, "Reset by our test (%)")):
            ax.set_xlim(0, x[-1])
            ax.set_ylim(0, 100)
            ax.set_title(t, fontsize=18)
            ax.axvline(j, color="0.5", lw=1)
            ax.legend(loc="upper left" if ax is b else "upper right", fontsize=12,
                      frameon=True, framealpha=0.55, edgecolor="none")
            ax.set_xlabel("update keyframes ingested", fontsize=16)
        fig.tight_layout(pad=0.4)
        return fig_to_image(fig).resize((w, h), Image.LANCZOS)


def seg_nor_timeline(scene, nor, N, sid, per_kf, label_stage):
    meta, z, curves = nor
    rec, rec_path = record(scene)
    kq, q = meta["quarter_turns"], meta["query"]["name"]
    J = int(meta["n_keyframes"])
    base = z["fixed_map_rgb"].astype(np.float32)
    mask = z["fixed_mask"]
    cov = (z["alpha"].astype(np.float32) >= 0.5)
    fa, rc = rec["operating_point"]["ours"]["fa"] * 100, rec["operating_point"]["ours"]["rec"] * 100
    ch_fa = rec["ours"]["changed_end"]["fa"] * 100
    end_avg = curves["averaging"]["stale_is_query"][-1]
    end_ours = curves["ours"]["stale_is_query"][-1]
    st_avg = curves["averaging"]["static_changed"][-1]
    st_ours = curves["ours"]["static_changed"][-1]
    pa, po, sa, so = 100 * end_avg, 100 * end_ours, 100 * st_avg, 100 * st_ours
    # Neutral end caption (2026-10-27): it reports both curves, since on 002 ours also changes the
    # top class of more static Gaussians (replaces the one-sided caption rule R12).
    cap_end = (f"End of update, this scene: '{q}' is still the top class of {pa:.0f}% of the removed "
               f"object's Gaussians for averaging and {po:.0f}% for ours; the top class of "
               f"{sa:.0f}% (averaging) and {so:.0f}% (ours) of the static Gaussians changed (dashed).")
    cap_play = (f"Each update keyframe, our change test resets Gaussians whose evidence no longer "
                f"agrees with the view (orange). Map shown from one fixed camera (update frame "
                f"{meta['frame']}); white: the removed object's mask.")
    cap_fa = (f"This scene, end of update: our test reset {rc:.1f}% of the removed object's and "
              f"{fa:.1f}% of the static Gaussians; after depth-checked re-admission {ch_fa:.1f}% "
              f"of the static ones stay flagged. Dashed: static Gaussians whose top class changed.")
    hold0, hold1 = seconds(1.5), seconds(7)
    order = [0] * hold0 + [j for j in range(1, J + 1) for _ in range(per_kf)] + [J] * hold1
    portrait = kq % 2 == 1

    def state(j):
        if j == 0:
            g = lambda k: z[f"before_{k}"]
        else:
            g = lambda k: z[k][j - 1]
        return g

    @functools.lru_cache(maxsize=2)
    def panels(j):
        g = state(j)
        avg = tint(base, g("pq_avg").astype(np.float32), PQ)
        ours = tint(base, g("pq_ours").astype(np.float32), PQ)
        rs = tint(base, g("flag").astype(np.float32) * 0.9, RESET)
        for a in (avg, ours, rs):
            a[~cov] = 0.0
        mp = base
        photo = z["photos"][j - 1] if j > 0 else z["fixed_photo"]
        pm = z["masks"][j - 1] if j > 0 else mask
        out = [outline(photo, pm), outline(mp, mask), outline(avg, mask), outline(ours, mask),
               outline(rs, mask)]
        return [rot(a, kq) for a in out]

    @functools.lru_cache(maxsize=2)
    def curve_img(j, w, h):
        return curves_panel(curves, j, q, w, h, False)

    titles = ["Camera now", "Map render (fixed view)", f"Averaging: p({q})", f"Ours: p({q})",
              "Ours: reset by test"]
    borders = [None, None, AVG, OURS, OURS]
    tit = lambda j: (titles if j > 0 else ["Camera after removal"] + titles[1:])

    def keys(img, x, y, w):
        colorbar(img, x, y, w, 18, (tuple(int(255 * (0.25 + 0.75 * 0.5)) for _ in range(3)), PQ),
                 "0", "1", f"p({q})")
        colorbar(img, x, y + 90, w, 18, (tuple(int(255 * (0.25 + 0.75 * 0.5)) for _ in range(3)),
                                         RESET), "0", "1", "reset share")
        text(img, (x, y + 160), "Curves and %: measured on this scene.", size=24, fill=INK2,
             width=w + 40)

    def frame(i):
        j = order[i]
        kf = "after mapping" if j == 0 else f"update keyframe {j}/{J} (frame {int(z['update_index'][j - 1])})"
        img = title(blank(), f"Object removal, {label_stage} scene {scene}: query '{q}'", sub=kf)
        ps = panels(j)
        if portrait:
            pw, ph, gap = 315, 420, 48
            x0 = (W - (5 * pw + 4 * gap)) // 2
            for c in range(5):
                paste(img, ps[c], x0 + c * (pw + gap), 170, pw, ph, border=borders[c],
                      label=tit(j)[c], label_size=24)
            cv = curve_img(j, 1400, 290)
            paste(img, cv, MARGIN + 20, 610)
            keys(img, 1530, 680, 300)
        else:
            pw, ph, gx = 440, 330, 24
            x0 = (W - (4 * pw + 3 * gx)) // 2
            xs = [x0 + c * (pw + gx) for c in range(4)]
            ya, yb = 170, 170 + ph + 50
            for c, (x, y) in enumerate(((xs[0], ya), (xs[1], ya), (xs[0], yb), (xs[1], yb),
                                        (xs[2], yb))):
                paste(img, ps[c], x, y, pw, ph, border=borders[c], label=tit(j)[c],
                      label_size=24)
            cv = curve_img(j, 2 * pw + gx, ph + 30)
            paste(img, cv, xs[2], ya - 30)
            keys(img, xs[3] + 40, yb + 60, pw - 80)
        if i >= len(order) - hold1:
            k = i - (len(order) - hold1)
            return caption(img, cap_fa if k < hold1 // 2 else cap_end, size=28)
        return caption(img, cap_play, size=28)
    srcs = [f"nor/{scene}/timeline.npz, curves.json (scripts/video_nor_timeline.py)",
            f"Table III record {os.path.relpath(rec_path, RES)} (FA, Rec, changed_end)",
            f"end shares (stale_is_query, static_changed): averaging {end_avg:.3f}, {st_avg:.3f}; "
            f"ours {end_ours:.3f}, {st_ours:.3f}"]
    return Seg(sid, f"nor_{scene}", len(order), frame, [cap_play, cap_fa, cap_end], srcs,
               local=[str(meta["frame"]), f"{rc:.1f}", f"{fa:.1f}", f"{ch_fa:.1f}", f"{pa:.0f}",
                      f"{po:.0f}", f"{sa:.0f}", f"{so:.0f}"],
               still=len(order) - 2)


def bar_chart(items, w, h, xlabel, title_s, xmax=100):
    """Horizontal bars: items [(label, value, colour, hatch)]."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 22}):
        fig = Figure(figsize=(w / 100, h / 100))
        ax = fig.subplots()
        ys = np.arange(len(items))[::-1]
        for y, (lab, v, col, hatch) in zip(ys, items):
            ax.barh(y, float(v), color=np.array(col) / 255, hatch=hatch, edgecolor="white")
            ax.text(float(v) + 1, y, v, va="center", fontsize=22)
        ax.set_yticks(ys, [it[0] for it in items])
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel)
        ax.set_title(title_s, fontsize=24)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout(pad=0.5)
        return fig_to_image(fig).resize((w, h), Image.LANCZOS)


def seg_reset_effect(N):
    stale = bar_chart([("after mapping", N["stale.before"], GREY, None),
                       ("averaging", N["stale.avg"], AVG, None),
                       ("ours", N["stale.ours"], OURS, None)], 900, 520,
                      "% of the removed object's covered pixels",
                      "Pixels whose top class is still the query")
    kept = bar_chart([("without depth check", N["kept.nodepth"], GREY, None),
                      ("ours (depth check)", N["kept.full"], OURS, None)], 820, 400,
                     "% of stale Gaussians", "Stale Gaussians left\nin the robust set", xmax=20)
    cap = fill("Resetting the Gaussians our test flags lowers the stale answers from {stale.avg}% "
               "(averaging) to {stale.ours}% (a new measurement on the test scenes); the Gaussians "
               "the test misses keep their old class. The depth check halves what returns to the "
               "robust map.", N)

    def frame(i):
        img = title(blank(), "What the reset changes (10 test scenes)")
        paste(img, stale, 80, 160)
        paste(img, kept, 1040, 240)
        return caption(img, cap)
    return Seg("S4d", "reset_effect", seconds(8), frame, [cap],
               ["main.tex abstract and Sec. V-D (48/34/21 %)", "main.tex tab:ablation Kept "
                "(full method vs re-admission without depth)"])


def seg_ranking(N):
    it = [("NCC photometric", N["t3.ncc.gap"], GREY, None),
          ("depth residual", N["t3.depth.gap"], GREY, None),
          ("feature residual", N["t3.feat.gap"], GREY, None),
          ("NormLift drop", N["t3.nl.gap"], GREY, None),
          ("ours: test", N["t3.ours.gap"], OURS, None),
          ("NCC + depth", N["fused.ncc.gap"], GREY, "//"),
          ("feature + depth", N["fused.feat.gap"], GREY, "//"),
          ("NormLift + depth", N["fused.nl.gap"], GREY, "//"),
          ("ours + depth", N["fused.ours.gap"], OURS, "//")]
    chart = bar_chart(it, 1300, 680, "Gaussian-level AP (%), mean over 10 test scenes",
                      "Ranking stale Gaussians (Table II)")
    cap = fill("Our test alone is not the best ranking (AP {t3.ours.gap} vs {t3.nl.gap} for "
               "NormLift's drop). Fused with depth, the feature residual ({fused.feat.gap}) and "
               "NormLift ({fused.nl.gap}) rank stale Gaussians better than ours ({fused.ours.gap}).",
               N)
    side = fill("At its label-free threshold our test flags {t3.ours.rec}% of the stale "
                "Gaussians at {t3.ours.fa}% false alarms (feature residual: {t3.feat.rec}% at "
                "{t3.feat.fa}%).", N)

    def frame(i):
        img = title(blank(), "Detecting stale content: where ours stands")
        paste(img, chart, 60, 130)
        text(img, (1420, 300), side, size=30, width=440)
        return caption(img, cap)
    return Seg("S4e", "ranking", seconds(9), frame, [side, cap],
               ["main.tex tab:change (G AP, FA, Rec; G AP of the fused rows)"])


def seg_limits(N):
    items = ["Geometry is fixed: moved content is forgotten at its old place, not inserted at "
             "its new one.",
             "Fusion cannot fix a class that LSeg confuses in every view.",
             fill("The change threshold is fit in-sample: on the removal test scenes the measured "
                  "false-alarm rate is {lim.rate}\u00d710\u207b\u00b3 per view, above the nominal "
                  "10\u207b\u00b3.", N),
             "On ScanNet the fitted P2 temperature grows with uncertainty, and our margins over "
             "averaging are smaller than the spread between scenes.",
             "The 5 cm depth check cannot keep out a thin removed object (a book on a table).",
             "All baselines are our re-implementations."]

    def frame(i):
        img = title(blank(), "Limitations")
        y = 170
        for s in items:
            ImageDraw.Draw(img).text((110, y), "\u2022", font=font(32), fill=INK)
            y = text(img, (150, y), s, size=32, width=1620) + 22
        return img
    return Seg("S5", "limitations", seconds(8), frame, items, ["main.tex sec:limitations"])


def seg_end():
    def frame(i):
        img = blank()
        d = ImageDraw.Draw(img)
        d.text((W / 2, 470), "Thank you", font=font(56, True), fill=INK, anchor="ma")
        d.text((W / 2, 560), "Anonymous submission", font=font(32), fill=INK2, anchor="ma")
        return img
    return Seg("S6", "end", seconds(3), frame, [], [])


# ========================================================================== encode
def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def write_clip(path, frames, crf=12, preset="slow"):
    """Frames (PIL, 1920x1080) to H.264 yuv420p, CRF crf, no metadata."""
    import imageio_ffmpeg
    w = imageio_ffmpeg.write_frames(str(path), (W, H), fps=FPS, codec="libx264",
                                    pix_fmt_out="yuv420p", quality=None, macro_block_size=8,
                                    output_params=["-preset", preset, "-crf", str(crf),
                                                   "-map_metadata", "-1", "-fflags", "+bitexact",
                                                   "-flags:v", "+bitexact"])
    w.send(None)
    n = 0
    for f in frames:
        a = np.asarray(f.convert("RGB"), np.uint8)
        assert a.shape == (H, W, 3), a.shape
        w.send(np.ascontiguousarray(a))
        n += 1
    w.close()
    return n


def probe(path):
    """width, height, fps, frame count (decoded), duration, codec/profile line, metadata lines."""
    import imageio_ffmpeg
    r = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    info = r.stderr
    m = re.search(r"Video: (\w+) \(([^)]*)\).*?, (\d+)x(\d+)", info)
    n, secs = imageio_ffmpeg.count_frames_and_secs(str(path))
    meta_lines = [ln.strip() for ln in info.splitlines()
                  if re.match(r"\s+(title|artist|author|comment|encoder|creation_time|"
                              r"handler_name|vendor_id|copyright|album)\s*:", ln)]
    return dict(codec=m.group(1) if m else None, profile=m.group(2) if m else None,
                width=int(m.group(3)) if m else None, height=int(m.group(4)) if m else None,
                frames=n, seconds=secs, fps=float(re.search(r"([\d.]+) fps", info).group(1)),
                metadata=meta_lines, audio="Audio:" in info)


def faded(seg):
    for i in range(seg.n):
        im = seg.frame(i)
        a = min(1.0, (i + 1) / FADE, (seg.n - i) / FADE)
        if a < 1.0:
            im = Image.blend(Image.new("RGB", (W, H), BG), im, a)
        yield im


def _encode_one(args):
    sid, out, figs = args
    segs = {s.sid: s for s in build_segments(out, figs)}
    s = segs[sid]
    path = os.path.join(out, "clips", f"{sid}.mp4")
    t0 = time.time()
    n = write_clip(path, faded(s))
    s.frame(s.still).save(os.path.join(out, "stills", f"{sid}_{s.name}.png"))
    return sid, n, time.time() - t0


_CACHE = {}


def build_segments(out, figs):
    if "segs" in _CACHE:
        return _CACHE["segs"]
    N = paper_numbers(os.path.join(REPO, "papers", "main.tex"))
    sp = np.load(os.path.join(out, "static", "room_0_path.npz"))
    sm = json.load(open(os.path.join(out, "static", "room_0_path.json")))
    if not sm.get("ok"):
        raise SystemExit("static path dump did not pass its checks")
    qm = json.load(open(os.path.join(RES, "qual", "qual_room_0_f29.json")))
    sm["vmax"], sm["frame_index"] = qm["vmax"], qm["frame_index"]
    static = (sp, sm)
    n002, n014 = load_nor(out, "002"), load_nor(out, "014")
    if n002[0]["query"]["name"] != "chair":
        raise SystemExit("002's rule query is not 'chair'")
    segs = [seg_title(N), seg_questions(N, static, n002), seg_pipeline(figs), seg_plane(figs),
            seg_static_path(static), seg_views(figs, N), seg_compare(figs, N), seg_table1(N),
            seg_nor_setup(n002),
            seg_nor_timeline("002", n002, N, "S4b", 6, "validation"),
            seg_nor_timeline("014", n014, N, "S4c", 4, "held-out test"),
            seg_reset_effect(N), seg_ranking(N), seg_limits(N), seg_end()]
    check_caption_numbers(segs, os.path.join(REPO, "papers", "main.tex"))
    _CACHE["segs"], _CACHE["N"] = segs, N
    return segs


def check_caption_numbers(segs, tex_path):
    """Every number in a caption occurs in main.tex, or is a declared data value of its segment
    (frame ids, per-scene values labelled 'this scene')."""
    tex = set(re.findall(r"\d+(?:\.\d+)?", open(tex_path).read()))
    bad = []
    for s in segs:
        for c in s.captions:
            for n in re.findall(r"\d+(?:\.\d+)?", c):
                if n not in tex and n not in s.local:
                    bad.append((s.sid, n, c[:60]))
    if bad:
        raise AssertionError(f"caption numbers not in main.tex: {bad}")


def compose(args):
    out = args.out
    figs = os.path.join(out, "figs")
    for d in ("clips", "stills", "figs"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    t0 = time.time()
    segs = build_segments(out, figs)
    N = _CACHE["N"]
    only = args.only.split(",") if args.only else [s.sid for s in segs]
    if args.dry:                     # one frame per segment, layout asserts only
        for s in segs:
            if s.sid in only:
                s.frame(s.still).save(os.path.join(out, "stills", f"{s.sid}_{s.name}.png"))
        print("dry run ok")
        return
    from multiprocessing import get_context
    jobs = [(s.sid, out, figs) for s in segs if s.sid in only]
    with get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_encode_one, jobs)
    done = {sid: (n, t) for sid, n, t in res}
    for s in segs:
        if s.sid in done and done[s.sid][0] != s.n:
            raise AssertionError(f"{s.sid}: wrote {done[s.sid][0]} of {s.n} frames")
    lst = os.path.join(out, "clips", "list.txt")
    with open(lst, "w") as f:
        for s in segs:
            f.write(f"file '{os.path.join(out, 'clips', s.sid + '.mp4')}'\n")
    master = os.path.join(out, "master.mp4")
    subprocess.run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", lst, "-c", "copy", "-map_metadata", "-1", "-fflags",
                    "+bitexact", master], check=True)
    t, board = 0, []
    for s in segs:
        board.append(dict(id=s.sid, name=s.name, start_s=round(t / FPS, 3),
                          duration_s=round(s.n / FPS, 3), frames=s.n, captions=s.captions,
                          sources=s.sources,
                          encode_s=round(done.get(s.sid, (0, 0))[1], 1)))
        t += s.n
    inputs = {p: sha256(p) for p in [os.path.join(REPO, "papers", "main.tex"),
                                      os.path.join(out, "static", "room_0_path.npz"),
                                      os.path.join(out, "nor", "002", "timeline.npz"),
                                      os.path.join(out, "nor", "014", "timeline.npz"),
                                      os.path.join(REPO, "papers", "figs", "pipeline_v2.pdf")]}
    info = probe(master)
    sb = dict(total_frames=t, total_s=t / FPS, fps=FPS, size=[W, H], segments=board,
              numbers=N, inputs_sha256={os.path.relpath(k, DATA): v for k, v in inputs.items()},
              script_sha256=sha256(os.path.abspath(__file__)), master=info,
              master_mb=os.path.getsize(master) / 1e6, wall_s=time.time() - t0)
    with open(os.path.join(out, "storyboard.json"), "w") as f:
        json.dump(sb, f, indent=1, ensure_ascii=False)
    if info["frames"] != t:
        raise AssertionError(f"master has {info['frames']} frames, expected {t}")
    print(f"master {master}: {t} frames, {t / FPS:.1f} s, {sb['master_mb']:.1f} MB; "
          f"wall {sb['wall_s']:.0f} s")


README = """Supplementary video (anonymous submission)

File: supplementary_video.mp4
Format: MPEG-4 (H.264 High, yuv420p), 1920x1080, 30 fps, {dur:.0f} s, no audio.
Player: any standard player (VLC, QuickTime, a web browser).
Captions are burned into the video; there is no sound.

The video illustrates the method and results of the paper. All numbers shown in captions are
taken from the paper. Values labelled "this scene" are per-scene values of the scene shown;
the paper's tables report means over the test scenes.
"""

SUMMARY = """Summary of the supplementary video ({dur:.0f} s)

{lines}

Data: Replica (as rendered by Semantic-NeRF), ScanNet, NeRF-Object-Removal. The object-removal
timelines replay the paper's update on one validation scene (002) and one test scene (014); the
replay reproduces the paper's per-scene operating point. The test scene is shown for
illustration only; no choice depends on it.
"""


DESCRIBE = dict(
    title="Title",
    questions="The problem: how much evidence, do the views agree, is the content still there",
    pipeline="Method: the paper's pipeline figure, step by step",
    plane="Epistemic vs aleatoric uncertainty: little evidence and conflicting evidence",
    static_path="Replica test scene: labels, both uncertainties and errors along a held-out path",
    views="Does each uncertainty rank its own error (the paper's views figure)",
    compare="Every method on one Replica and one ScanNet test frame",
    table1="Segmentation and calibration numbers (Table I)",
    nor_setup="Real object removal: setup",
    nor_002="Object removal timeline, validation scene 002 (query 'chair')",
    nor_014="Object removal timeline, test scene 014 (query 'bike')",
    reset_effect="What the reset changes: stale answers and the robust set",
    ranking="Ranking stale Gaussians: ours against the baselines (Table II)",
    limitations="Limitations",
    end="End")


def deliver(args):
    out = args.out
    master = os.path.join(out, "master.mp4")
    sb = json.load(open(os.path.join(out, "storyboard.json")))
    dur = sb["total_s"]
    sub = os.path.join(out, "submission")
    tmp = os.path.join(out, "tmp_pass")
    os.makedirs(sub, exist_ok=True)
    os.makedirs(tmp, exist_ok=True)
    kbps = int(math.floor(args.max_mb * 1e6 * 8 / dur / 1000 * 0.97))
    mp4 = os.path.join(sub, "supplementary_video.mp4")
    common = ["-c:v", "libx264", "-preset", "slow", "-profile:v", "high", "-pix_fmt", "yuv420p",
              "-b:v", f"{kbps}k", "-passlogfile", os.path.join(tmp, "x264"), "-an",
              "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact"]
    t0 = time.time()
    subprocess.run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", master, *common,
                    "-pass", "1", "-f", "mp4", os.devnull], check=True, cwd=tmp)
    subprocess.run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", master, *common,
                    "-pass", "2", "-movflags", "+faststart", mp4], check=True, cwd=tmp)
    shutil.rmtree(tmp)
    info = probe(mp4)
    size_mb = os.path.getsize(mp4) / 1e6
    lines = []
    for s in sb["segments"]:
        lines.append(f"{s['start_s']:6.1f} s  {DESCRIBE[s['name']]}")
    readme = README.format(dur=dur)
    summary = SUMMARY.format(dur=dur, lines="\n".join(lines))
    for name, txt in (("ReadMe.txt", readme), ("Summary.txt", summary)):
        txt.encode("ascii")                     # ASCII only
        with open(os.path.join(sub, name), "w") as f:
            f.write(txt)
    zpath = os.path.join(out, "supplementary_video.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for name in ("supplementary_video.mp4", "ReadMe.txt", "Summary.txt"):
            zi = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            zi.external_attr = 0o644 << 16
            with open(os.path.join(sub, name), "rb") as f:
                z.writestr(zi, f.read())
    checks = anonymity_checks([mp4, zpath, master, os.path.join(sub, "ReadMe.txt"),
                               os.path.join(sub, "Summary.txt")])
    master_info = probe(master)
    res = dict(delivery=info, delivery_mb=size_mb, target_kbps=kbps,
               zip_mb=os.path.getsize(zpath) / 1e6, master=master_info,
               frames_equal=info["frames"] == master_info["frames"] == sb["total_frames"],
               anonymity=checks, encode_s=time.time() - t0,
               ok=(info["width"] == W and info["height"] == H and info["codec"] == "h264"
                   and "High" in (info["profile"] or "") and not info["audio"]
                   and 60 <= info["seconds"] <= 180 and size_mb <= args.max_mb
                   and os.path.getsize(zpath) / 1e6 <= 50
                   and info["frames"] == master_info["frames"] and not checks["hits"]
                   and not identifying_metadata(info["metadata"])))
    with open(os.path.join(out, "deliver_checks.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    if not res["ok"]:
        raise SystemExit(2)


GENERIC_TAGS = ("handler_name    : VideoHandler", "vendor_id       : [0][0][0][0]",
                "encoder         : Lavc libx264")


def identifying_metadata(lines):
    """Container/stream tags other than the generic handler and vendor ids written by ffmpeg."""
    return [ln for ln in lines if ln not in GENERIC_TAGS]


BANNED = [b"thanhnc", b"/home/", b"gauss_ws", b"claude", b"anthropic", b"canhthanh",
          b"pairs-lab", b"uags_results"]


def anonymity_checks(paths):
    hits = []
    for p in paths:
        data = open(p, "rb").read().lower()
        for b in BANNED:
            if b in data:
                hits.append(f"{os.path.basename(p)}: {b.decode()}")
    return dict(banned=[b.decode() for b in BANNED], files=[os.path.basename(p) for p in paths],
                hits=hits)


def contact(args):
    """Contact sheets of every --every-th frame of a video (anonymity review)."""
    import imageio_ffmpeg
    path = args.video
    gen = imageio_ffmpeg.read_frames(path)
    meta = gen.__next__()
    w, h = meta["size"]
    tiles, sheets = [], 0
    os.makedirs(args.sheet_dir, exist_ok=True)

    def flush():
        nonlocal tiles, sheets
        if not tiles:
            return
        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 480, rows * 270), BG)
        for k, (fi, t) in enumerate(tiles):
            sheet.paste(t, ((k % cols) * 480, (k // cols) * 270))
        sheet.save(os.path.join(args.sheet_dir, f"sheet_{sheets:02d}.jpg"), quality=85)
        sheets += 1
        tiles = []
    for i, buf in enumerate(gen):
        if i % args.every == 0:
            im = Image.frombuffer("RGB", (w, h), buf, "raw", "RGB", 0, 1).resize((480, 270))
            tiles.append((i, im))
            if len(tiles) == 16:
                flush()
    flush()
    print(f"{sheets} sheets in {args.sheet_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("compose")
    c.add_argument("--out", default=os.path.join(RES, "video"))
    c.add_argument("--only", default="", help="segment ids (debug)")
    c.add_argument("--dry", action="store_true", help="one still per segment, no encode")
    c.add_argument("--workers", type=int, default=8)
    d = sp.add_parser("deliver")
    d.add_argument("--out", default=os.path.join(RES, "video"))
    d.add_argument("--max-mb", type=float, default=44.0)
    s = sp.add_parser("contact")
    s.add_argument("--video", required=True)
    s.add_argument("--every", type=int, default=30)
    s.add_argument("--sheet-dir", required=True)
    s_ = sp.add_parser("numbers")
    s_.add_argument("--tex", default=os.path.join(REPO, "papers", "main.tex"))
    args = ap.parse_args(argv)
    sys.path.insert(0, HERE)
    sys.path.insert(0, REPO)
    if args.cmd == "compose":
        compose(args)
    elif args.cmd == "deliver":
        deliver(args)
    elif args.cmd == "contact":
        contact(args)
    else:
        print(json.dumps(paper_numbers(args.tex), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
