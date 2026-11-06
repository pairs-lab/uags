"""
teaser_3d.py — the third-person view of Fig. 1 (layout C): the 3D-GS map of the teaser scene seen
from a free camera, each Gaussian's values rendered as alpha-weighted means (gsplat), with the
teaser frame's camera frustum and the camera paths projected into the view. Every per-Gaussian
value comes from the dump of scripts/dump_nor_qual.py of the same scene (gauss.pt); nothing is
re-estimated here. GPU (run it through scripts/gpu_gate.sh).

Free camera (a geometric rule; it changes no number): it looks at the centroid of the stale
observed Gaussians (the removed object) from the side of the teaser frame's camera, at --dist
times that camera's horizontal distance to the target and --height m above the target (world
up = -y: gravity is +y in the NOR captures), turned by --azimuth degrees about the up axis, with
focal length --focal x the output width. The defaults (2.6, 1.4 m, 35 deg, 0.75) were picked by
eye on 2026-11-06 among five candidate viewpoints of validation scene 002, for a clear view of the
removed chair, the desk and both camera paths.

Rendered channels (each the per-pixel alpha-weighted mean over the Gaussians, NaN where the
alpha is below --alpha-min): u_epi0 / u_ale0 (after mapping), u_epi1 / u_ale1 (after the update,
our field), flag (reset by our test), stale (ground truth), q_before / q_avg / q_ours (the top
class is the scene's query: after mapping, averaging and ours after the update); u_ale uses its
own weight (defined Gaussians only).

    scripts/gpu_gate.sh 3000 python3 papers/figure_tools/teaser_3d.py --record results/nor/tab2_relmax/val/002.json \
        --dump results/qual/nor/002 --out results/qual/nor/002/teaser3d.npz
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))              # papers/figure_tools
REPO = os.path.dirname(os.path.dirname(HERE))
SCRIPTS = os.path.join(REPO, "scripts")                      # shared modules (render_qual, ...)
sys.path.insert(0, REPO)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)
from usplat.paths import moved                                              # noqa: E402
from usplat.gs_io import load_gaussians                                     # noqa: E402
from usplat.datasets import NeRFObjectRemovalDataset                       # noqa: E402

UP = np.array([0.0, -1.0, 0.0])                       # gravity is +y in the NOR captures


def look_at(eye, target, up=UP):
    """c2w (4, 4) of an OpenCV camera (x right, y down, z forward) at eye looking at target."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:
        raise ValueError("the view direction is parallel to the up axis")
    r /= np.linalg.norm(r)
    d = np.cross(f, r)                                # camera +y (image down)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([r, d, f], 1)
    c2w[:3, 3] = eye
    return c2w


def free_camera(target, cam_centre, dist, height, azimuth_deg, up=UP):
    """The rule of the module docstring: eye and c2w."""
    v = np.asarray(cam_centre, float) - np.asarray(target, float)
    vh = v - (v @ up) * up
    a = math.radians(azimuth_deg)
    # rotate the horizontal direction about the up axis (Rodrigues)
    k = up / np.linalg.norm(up)
    vh_rot = vh * math.cos(a) + np.cross(k, vh) * math.sin(a) + k * (k @ vh) * (1 - math.cos(a))
    eye = np.asarray(target, float) + dist * vh_rot + height * up
    return eye, look_at(eye, target, up)


def project(P, c2w, K):
    """(n, 2) pixel coordinates and (n,) camera depths of world points P (n, 3)."""
    w2c = np.linalg.inv(c2w)
    pc = np.asarray(P, float) @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    uv = (pc[:, :2] / np.maximum(z[:, None], 1e-9)) @ np.asarray(K)[:2, :2].T + np.asarray(K)[:2, 2]
    return uv, z


def frustum(c2w, K, hw, depth):
    """World points: the camera centre and the 4 image corners at `depth` m (OpenCV camera)."""
    h, w = hw
    Kinv = np.linalg.inv(np.asarray(K, float))
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float)
    rays = corners @ Kinv.T
    pts_c = rays * depth
    R, t = np.asarray(c2w)[:3, :3], np.asarray(c2w)[:3, 3]
    return np.vstack([t[None], pts_c @ R.T + t])


@torch.no_grad()
def render_channels(splats, c2w, K, w, h, chans, alpha_min, chunk=32, backend="gsplat"):
    """{name: (h, w) float32 alpha-weighted mean} of per-Gaussian (N,) channels, the RGB render
    on white and the alpha. chans: {name: (values (N,), weight (N,) or None)}."""
    from time_field import rasterizer_factory
    dev = splats["means"].device
    viewmat = torch.linalg.inv(torch.as_tensor(c2w, dtype=torch.float64)).float().to(dev)
    ras = rasterizer_factory(backend, chunk)(splats, viewmat, torch.as_tensor(K).float().to(dev),
                                             int(w), int(h))
    cols = [splats["colors"].float(), torch.ones_like(splats["colors"][:, :1]).float()]
    names = []
    for name, (v, wt) in chans.items():
        v = torch.as_tensor(v).float().to(dev)
        wt = torch.ones_like(v) if wt is None else torch.as_tensor(wt).float().to(dev)
        cols += [(torch.nan_to_num(v) * wt)[:, None], wt[:, None]]
        names.append(name)
    out = ras(torch.cat(cols, 1)).reshape(h, w, -1).float().cpu().numpy()
    alpha = out[..., 3]
    rgb = out[..., :3] + (1.0 - alpha[..., None])             # composited on white
    maps = {}
    for j, name in enumerate(names):
        num, den = out[..., 4 + 2 * j], out[..., 5 + 2 * j]
        m = np.full((h, w), np.nan, np.float32)
        ok = (alpha >= alpha_min) & (den > 1e-6)
        m[ok] = num[ok] / den[ok]
        maps[name] = m
    return np.clip(rgb, 0, 1), alpha, maps


def run(a):
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    rec = json.load(open(a.record))
    sys.path.insert(0, SCRIPTS)
    import dump_nor_qual as DQ
    m = DQ.record_body(rec)
    if m["scene"] != meta["scene"]:
        raise SystemExit(f"record scene {m['scene']} != dump scene {meta['scene']}")
    p = m["provenance"]
    G = torch.load(os.path.join(a.dump, "gauss.pt"), map_location="cpu", weights_only=False)
    dev = a.device
    if str(dev).startswith("cuda"):
        from usplat import gpu_reserve
        gpu_reserve.apply(dev)
    splats = load_gaussians(moved(p["ply"]["path"]), device=dev)
    N = int(splats["means"].shape[0])
    if N != int(G["stale"].numel()):
        raise SystemExit(f"map has {N} Gaussians, the dump {G['stale'].numel()}")
    ds = NeRFObjectRemovalDataset(moved(p["data"]), stride=1, part="all")
    fi = int(a.frame if a.frame is not None else next(iter(meta["frames"])))
    if str(fi) not in meta["frames"]:
        raise SystemExit(f"frame {fi} is not a dumped frame of {a.dump}")
    means = splats["means"].float().cpu().numpy()
    sel = (G["stale"] & G["observed"]).numpy()
    target = means[sel].mean(0)
    cam = ds.c2w[fi]
    eye, c2w = free_camera(target, cam[:3, 3], a.dist, a.height, a.azimuth)
    W, H = int(a.width), int(round(a.width / a.aspect))
    K = np.array([[a.focal * W, 0, W / 2], [0, a.focal * W, H / 2], [0, 0, 1]], float)
    qi = int(G["query"]["index"])
    u_ale0, u_ale1 = G["u_ale0"].float(), G["u_ale1"].float()
    chans = dict(u_epi0=(G["u_epi0"], None), u_epi1=(G["u_epi1"], None),
                 u_ale0=(u_ale0, torch.isfinite(u_ale0).float()),
                 u_ale1=(u_ale1, torch.isfinite(u_ale1).float()),
                 flag=(G["flags"]["ours"].float(), None), stale=(G["stale"].float(), None),
                 q_before=((G["top_class"]["before"] == qi).float(), None),
                 q_avg=((G["top_class"]["avg_after"] == qi).float(), None),
                 q_ours=((G["top_class"]["ours"] == qi).float(), None))
    rgb, alpha, maps = render_channels(splats, c2w, K, W, H, chans, a.alpha_min)
    hw0 = ds.image_hw
    fr = frustum(cam, ds.K[fi], hw0, a.frustum_depth)
    fr_uv, fr_z = project(fr, c2w, K)
    upd = [int(v) for v in meta["views"]["update"]]
    mp = [int(v) for v in meta["views"]["mapping"]]
    t_upd, z_upd = project(np.stack([ds.c2w[i][:3, 3] for i in upd]), c2w, K)
    t_map, z_map = project(np.stack([ds.c2w[i][:3, 3] for i in mp]), c2w, K)
    tgt_uv, _ = project(target[None], c2w, K)
    info = dict(scene=meta["scene"], frame=fi, query=G["query"]["name"], query_index=qi,
                rule=dict(dist=a.dist, height=a.height, azimuth=a.azimuth, focal=a.focal,
                          width=W, height_px=H, alpha_min=a.alpha_min,
                          frustum_depth=a.frustum_depth, up=UP.tolist()),
                target=target.tolist(), eye=eye.tolist(), n_stale_observed=int(sel.sum()),
                n_gaussians=N, record=os.path.abspath(a.record), dump=os.path.abspath(a.dump))
    np.savez_compressed(a.out, rgb=rgb.astype(np.float16), alpha=alpha.astype(np.float16),
                        **{f"map/{k}": v.astype(np.float16) for k, v in maps.items()},
                        frustum_uv=fr_uv, frustum_z=fr_z, traj_update_uv=t_upd, traj_update_z=z_upd,
                        traj_map_uv=t_map, traj_map_z=z_map, target_uv=tgt_uv[0], c2w=c2w, K=K,
                        info=json.dumps(info))
    print(json.dumps(info))
    return info


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", required=True, help="the scene's Table III record (val/test)")
    ap.add_argument("--dump", required=True, help="the scene's dump_nor_qual.py directory")
    ap.add_argument("--frame", type=int, default=None, help="the teaser frame (default: dumped)")
    ap.add_argument("--dist", type=float, default=2.6)
    ap.add_argument("--height", type=float, default=1.4)
    ap.add_argument("--azimuth", type=float, default=35.0)
    ap.add_argument("--focal", type=float, default=0.75)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--aspect", type=float, default=1.92)
    ap.add_argument("--alpha-min", type=float, default=0.3)
    ap.add_argument("--frustum-depth", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
