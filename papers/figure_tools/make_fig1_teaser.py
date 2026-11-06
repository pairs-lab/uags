"""
make_fig1_teaser.py — Fig. 1 of papers/main.tex (fig:teaser), layout C (2026-11-06, in the style of
USGLoc's Fig. 1). Left: 2 x 2 panels at the teaser frame's pose (after mapping: the map with the query
and u_ale; after the removal: averaging and ours with the Gaussians our test reset). Right: the same
map from a free camera (teaser_3d.py) coloured by u_epi after mapping, with the reset Gaussians, the
teaser frame's camera and both camera paths. CPU. Every number printed on a panel is measured on the
dump of scripts/dump_nor_qual.py; the helpers and layouts A / B stay in scripts/plot_nor_qual.py.

    scripts/gpu_gate.sh 3000 python3 papers/figure_tools/teaser_3d.py \
        --record results/nor/tab2_relmax/val/002.json --dump results/qual/nor/002 \
        --out results/qual/nor/002/teaser3d.npz
    python3 papers/figure_tools/make_fig1_teaser.py --dump results/qual/nor/002 \
        --record results/nor/tab2_relmax/val/002.json --view3d results/qual/nor/002/teaser3d.npz \
        --out papers/figs/teaser.pdf
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))              # papers/figure_tools
REPO = os.path.dirname(os.path.dirname(HERE))
SCRIPTS = os.path.join(REPO, "scripts")
sys.path.insert(0, REPO)
sys.path.insert(0, SCRIPTS)
import plot_nor_qual as PQ                                                 # noqa: E402


def view3d_image(T, epi_max, reset_strength=0.7):
    """The third-person image of layout C: viridis(u_epi after mapping) shaded by the map's own
    brightness, white where nothing is rendered, and the Gaussians our test reset after the
    removal in the reset colour (rendered share >= 0.5)."""
    RQ = PQ._rq()
    import matplotlib
    import matplotlib.colors as mcolors
    rgb = T["rgb"].astype(float)
    u = T["map/u_epi0"].astype(float)
    shade = rgb.mean(-1, keepdims=True)
    lo, hi = np.quantile(shade[np.isfinite(u)], [0.02, 0.98])
    shade = np.clip((shade - lo) / max(hi - lo, 1e-6), 0, 1)
    heat = matplotlib.colormaps[RQ.U_CMAP](np.clip(np.nan_to_num(u) / epi_max, 0, 1))[..., :3]
    img = np.where(np.isfinite(u)[..., None], heat * (0.35 + 0.65 * shade), rgb)
    fl = np.nan_to_num(T["map/flag"].astype(float)) >= 0.5
    c = np.array(mcolors.to_rgb(PQ.RESET))
    img[fl] = (1 - reset_strength) * img[fl] + reset_strength * c
    return np.clip(img, 0, 1), fl


def teaser_figure_c(dump_dir, record, view3d, out):
    """Fig. 1, layout C (2026-11-06, in the style of USGLoc's Fig. 1). Left, 2 x 2 panels at the
    teaser frame's pose: after mapping, the map with the query and u_ale; after the removal,
    averaging and ours (with the Gaussians our test reset). Right, the same map from a free
    camera (papers/figure_tools/teaser_3d.py, rule in its docstring): u_epi after mapping, the Gaussians our
    test reset after the removal, the teaser frame's camera and both camera paths. The share
    printed on (a), (c), (d) is measured on the dump: the removed object's covered pixels whose
    top class is the query."""
    RQ = PQ._rq()
    import matplotlib
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    meta, fi, z = PQ.load_dump(dump_dir)
    k = int(meta["frames"][str(fi)]["quarter_turns"])
    rec = PQ.body(record)
    T = np.load(view3d)
    info3 = json.loads(str(T["info"]))
    if str(info3["scene"]) != str(meta["scene"]) or int(info3["frame"]) != fi:
        raise SystemExit(f"{view3d}: scene {info3['scene']} frame {info3['frame']}, the dump has "
                         f"scene {meta['scene']} frame {fi}")
    cov = z["full_covered"].astype(bool)
    R = lambda a: PQ.rot(a, k)
    q = meta["query"]["name"]
    photo, maprgb = z["full_photo"], np.clip(z["full_map_rgb"].astype(float), 0, 1)
    g_map, g_photo = PQ._grey(maprgb), PQ._grey(photo)
    gt = cov & (np.nan_to_num(z["full_share/stale"].astype(float)) >= 0.5)
    ue0, ua0 = z["full_share/u_epi_before"].astype(float), z["full_share/u_ale_before"].astype(float)
    ok = lambda a: a[cov & np.isfinite(a)]
    epi_max = RQ.nice_up(float(np.quantile(ok(ue0), 0.99)))
    ale_max = RQ.nice_up(float(np.quantile(ok(ua0), 0.99))) if ok(ua0).size else 1.0
    qi = int(meta["query"]["index"])
    top = {s: (z[f"full_pred/{s}"] == qi).astype(float) for s in ("before", "ours", "avg_after")}
    share = {s: float(top[s][gt].mean()) for s in top}
    flag = z["full_share/flag_ours"].astype(float)
    pink = PQ.OKABE["pink"]
    panels = {"a": R(PQ.overlay(g_map, top["before"], pink, cov)),
              "b": R(RQ.u_image(ua0, cov, ale_max)),
              "c": R(PQ.overlay(g_photo, top["avg_after"], pink, cov)),
              "d": R(PQ.overlay(PQ.overlay(g_photo, flag, PQ.RESET, cov, strength=0.6), top["ours"], pink, cov))}
    obj = R(PQ._object_outline(gt).astype(float))
    titles = {"a": f"(a) Map: top class '{q}'", "b": "(b) $u^{\\mathrm{ale}}$: agreement",
              "c": "(c) Averaging", "d": "(d) Ours"}
    badge = {"a": share["before"], "c": share["avg_after"], "d": share["ours"]}
    frame_col = {"c": PQ.OKABE["vermillion"], "a": PQ.OKABE["blue"], "b": PQ.OKABE["blue"], "d": PQ.OKABE["blue"]}
    img3, fl3 = view3d_image(T, epi_max)
    W, left, gp, top_h, head, strip = 7.16, 0.03, 0.05, 0.13, 0.16, 0.28
    pw = 0.76
    ph = pw * panels["a"].shape[0] / panels["a"].shape[1]
    H = strip + 2 * (ph + top_h + head)
    x3 = left + 2 * pw + gp + 0.12
    w3 = W - x3 - 0.02
    h3 = H - head - top_h - 0.03
    with matplotlib.rc_context(RQ.RC):
        from matplotlib.figure import Figure
        fig = Figure(figsize=(W, H))
        box = lambda x, y, bw, bh: fig.add_axes((x / W, y / H, bw / W, bh / H))
        for r, (keys, hdr) in enumerate(((("a", "b"), "After mapping"),
                                         (("c", "d"), f"After the {q} is removed "
                                                      f"({len(rec['views']['update'])} views)"))):
            y = H - (r + 1) * (head + top_h + ph)
            for c, key in enumerate(keys):
                ax = box(left + c * (pw + gp), y, pw, ph)
                ax.imshow(panels[key], interpolation="nearest")
                if key in "cd":
                    PQ.outline(ax, obj, dashed=False, lw=0.6)
                ax.set_title(titles[key], fontsize=6.0, pad=1.5)
                if key in badge:
                    ax.text(0.04, 0.03, f"'{q}': {100 * badge[key]:.1f}%", transform=ax.transAxes,
                            fontsize=5.5, color="white", ha="left", va="bottom",
                            bbox=dict(boxstyle="square,pad=0.18", facecolor="black", alpha=0.65,
                                      linewidth=0))
                PQ._frame(ax, frame_col.get(key))
            yh = y + ph + top_h + 0.06
            fig.text((left + pw + gp / 2) / W, yh / H, hdr, ha="center", va="center", fontsize=6.5,
                     fontweight="bold")
        PQ._cbar(fig, box, left + pw + gp + 0.08, 0.13, pw - 0.16, 0, ale_max, None)
        PQ._legend_patches(box(left, 0.0, pw + 0.05, strip - 0.03),
                        [("removed object", "white"), (f"top class '{q}'", pink),
                         ("reset", PQ.RESET)], ncol=1, loc="center left", fs=5.0)
        # (e) the third-person view
        from matplotlib.patches import Polygon
        ax = box(x3, H - head - top_h - h3, w3, h3)
        hh, ww = img3.shape[:2]
        ax.imshow(img3, interpolation="none", extent=(0, ww, hh, 0))     # native pixels in the PDF
        # the centred window of the panel's aspect (the overlays keep their pixel coordinates)
        cw, ch = (min(ww, hh * w3 / h3), min(hh, ww * h3 / w3))
        x0, y0 = (ww - cw) / 2, (hh - ch) / 2
        stroke = [pe.withStroke(linewidth=1.6, foreground="black")]
        tm, zm = T["traj_map_uv"], T["traj_map_z"]
        tu, zu = T["traj_update_uv"], T["traj_update_z"]
        inside = lambda uv, zz: ((zz > 0) & (uv[:, 0] >= x0) & (uv[:, 0] < x0 + cw)
                                 & (uv[:, 1] >= y0) & (uv[:, 1] < y0 + ch))
        m_in, u_in = inside(tm, zm), inside(tu, zu)
        ax.scatter(tm[m_in, 0], tm[m_in, 1], s=2.2, c="white", edgecolors="black", linewidths=0.25,
                   zorder=3)
        ax.scatter(tu[u_in, 0], tu[u_in, 1], s=2.2, c=PQ.OKABE["sky"], edgecolors="black",
                   linewidths=0.25, zorder=3)
        fr = T["frustum_uv"]                       # apex (camera centre), then the 4 corners
        ax.add_patch(Polygon(fr[1:], closed=True, facecolor="white", alpha=0.35, edgecolor="none",
                             zorder=4))
        for j in range(1, 5):
            ax.plot([fr[0, 0], fr[j, 0]], [fr[0, 1], fr[j, 1]], color="white", lw=0.6,
                    path_effects=stroke, zorder=4)
        ax.plot(*np.vstack([fr[1:], fr[1:2]]).T, color="white", lw=0.9, path_effects=stroke, zorder=4)
        ax.text(fr[:, 0].min() - 5, fr[:, 1].min() - 5, "keyframe",
                fontsize=5.5, color="white", ha="right", va="bottom", path_effects=stroke, zorder=5)
        ax.set_xlim(x0, x0 + cw)
        ax.set_ylim(y0 + ch, y0)
        ax.set_title("(e) The map from a free camera: $u^{\\mathrm{epi}}$ after mapping; orange: "
                     "reset after the removal", fontsize=6.0, pad=1.5)
        PQ._frame(ax, None)
        ax.legend(handles=[Patch(facecolor=PQ.RESET, edgecolor="black", linewidth=0.3,
                                 label="reset"),
                           Line2D([], [], marker="o", ls="", ms=2.5, mfc="white", mec="black",
                                  mew=0.3, label="mapping camera path"),
                           Line2D([], [], marker="o", ls="", ms=2.5, mfc=PQ.OKABE["sky"], mec="black",
                                  mew=0.3, label="update camera path")],
                  loc="lower left", fontsize=5.0, frameon=True, framealpha=0.8, edgecolor="none",
                  borderpad=0.3, handlelength=1.0, handletextpad=0.4, labelspacing=0.25)
        cb = PQ._cbar(fig, box, x3 + w3 - 1.12, H - head - top_h - h3 + 0.10, 0.95, 0, epi_max,
                   "$u^{\\mathrm{epi}}$ after mapping")
    paths = RQ.save(fig, out)
    op = rec["operating_point"]["ours"]
    return dict(scene=meta["scene"], frame=fi, quarter_turns=k, query=q, layout="C",
                epi_max=epi_max, ale_max=ale_max, shares={k_: round(100 * v, 1) for k_, v in share.items()},
                view3d=dict(path=os.path.abspath(view3d), rule=info3["rule"], eye=info3["eye"],
                            target=info3["target"]),
                published=dict(rec=op["rec"], fa=op["fa"], n_update=len(rec["views"]["update"])),
                outputs=paths)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump", required=True, help="the validation scene's dump folder")
    ap.add_argument("--record", required=True, help="its Table III record")
    ap.add_argument("--view3d", required=True, help="teaser_3d.py output (.npz)")
    ap.add_argument("--out", required=True, help="the PDF")
    ap.add_argument("--png", action="store_true", help="keep the PNG preview next to the PDF")
    ap.add_argument("--meta", default="", help="write the figure record (JSON) here")
    a = ap.parse_args(argv)
    res = teaser_figure_c(a.dump, a.record, a.view3d, a.out)
    res["pdf_check"] = PQ.pdf_check(res["outputs"][0])
    if not a.png:
        os.remove(res["outputs"][1])
        res["outputs"] = res["outputs"][:1]
    if a.meta:
        PQ.write_json(a.meta, res)
    print(json.dumps(dict(shares=res["shares"], published=res["published"], outputs=res["outputs"])))
    print(f"pdf check: {'PASS' if res['pdf_check']['ok'] else 'FAIL'} {res['pdf_check']['problems']}")
    return res


if __name__ == "__main__":
    main()
