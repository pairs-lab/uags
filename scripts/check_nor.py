"""
check_nor.py — verify the NeRF-Object-Removal loader ("nor", usplat/datasets.py) on every
scene, on the CPU: which reading of the COLMAP poses and which box file match the object
masks, whether the LiDAR points of the object land in that box, the depth scale, and the
depth consistency between frames (also across the removal).

  PYTHONPATH=. python3 scripts/check_nor.py --root <data>/nerf-object-removal \
      --out <dir>/check_nor.json [--scenes 001,002] [--frames 12] [--gap 10] [--revisit 60]

Masks, depth and confidence share one 192x256 grid; K_depth = K scaled to it. "Frames" are
--frames evenly spaced ones of each kind. Per scene:
 counts  frames per part, mask files per part, the spread of the per-frame intrinsics and
         the principal-point step between consecutive frames against random frame pairs
         (much smaller = the per-frame cameras follow the frame order), and the map stride
         whose frame count is nearest 125, with the frames that run_mapping.py's keyframe
         rule (drop a frame < 5 cm AND < 5 deg from the last kept one, its defaults) keeps
 1 pose  for every box file (bbox.txt, bbox.ply, bbox_refined.txt (per-axis min max),
         bbox_refined.ply, bbox_nerf.txt, bbox_nerf_refined.txt) and every reading of the
         images.txt pose (world-to-camera as COLMAP defines it, or camera-to-world; OpenCV
         or OpenGL camera axes) the box silhouette (usplat/box3d.py) is compared with
         masks_real in with-object frames (map part, mask not empty): IoU, cover = share of
         the mask inside the silhouette, inside = share of the silhouette inside the mask.
         The pair with the highest mean IoU wins. bbox_nerf.txt is fitted to bbox.txt by a
         similarity (Umeyama, NeRF -> COLMAP): the same box in a normalised frame gives
         rms ~0 and scale = boxes/meta.json "scale" / 2; that map also takes
         bbox_nerf_refined.txt onto the bbox_refined.txt corners (largest distance reported).
 2 lidar loader geometry, the loader's box (BOX_FILE) inflated by 2 cm, confidence-filtered
         depth (loader default). With-object frames: share of mask pixels whose 3D point is
         in the box (box|mask; below 1 because the masks are dilated), share of in-box points
         inside the mask (mask|box), share of pixels outside the mask landing in the box
         (box|~mask). Update frames: share of mask pixels still landing in the box
         (box|mask after the removal: surfaces left inside the inflated box, e.g. the floor).
         near_after: share of the with-object in-box points (box inflated by 2 cm, and not
         inflated) within 2 cm of an after-removal point: surface that is still there after
         the removal (which a box ground truth labels stale), plus object surface within
         2 cm of it (e.g. its base). The update frames may miss some surface, and a pose
         offset of the update frames (see 3) hides more.
 3 depth pairs (a, b): a's valid depth back-projected, moved into b and compared with b's
         valid depth at the pixel it lands in, dz = z_b - z_pred. map-map: b = a + --gap
         frames; revisit: b the mapping frame with the nearest pose at least --revisit
         frames from a (drift inside the mapping part); map-update: b an update frame, a the
         mapping frame with the nearest pose (pose cost |dC| / 0.1 m + angle / 10 deg),
         masks_real pixels of both frames left out. Median |dz|, share |dz| < 5 cm, median
         z_b / z_pred (= 1 when depth and poses share a unit; ARKit poses are metric),
         overlap = compared / valid pixels of a. For revisit and map-update pairs, the same
         after a rigid ICP fit of b's points onto a's (rigid_*): a residual that drops to the
         map-map level is a pose offset of b (drift), not depth noise. Per ARKit confidence
         level of a's pixels on the map-map pairs (b: high confidence only), and the map-map
         pairs with camera 1's intrinsics for both frames (map_map_camera1: images.txt names
         camera 1 for every frame).
 4 masks update frames: IoU of masks_synthetic with masks_real and with the box silhouette,
         share of masks_synthetic inside masks_real.
Prints a summary per scene and writes every number to --out (JSON). Exit code 1 if a scene
failed to load.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from usplat.box3d import (read_box_corners, box_silhouette, points_in_box,  # noqa: E402
                          box_extents)
from usplat.datasets import (NeRFObjectRemovalDataset, read_colmap_images,  # noqa: E402
                             read_colmap_cameras, qvec_to_rotmat)

BOX_FILES = ("bbox.txt", "bbox.ply", "bbox_refined.txt", "bbox_refined.ply", "bbox_nerf.txt",
             "bbox_nerf_refined.txt")
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])                # OpenGL <-> OpenCV camera axes
INFLATE = 0.02                                        # stale ground truth margin [m]
DZ_OK = 0.05                                          # [m]
KF_TRANS, KF_ROT = 0.05, 5.0                          # run_mapping.py keyframe defaults


def evenly(seq, m):
    seq = list(seq)
    if len(seq) <= m:
        return seq
    return [seq[int(round(x))] for x in np.linspace(0, len(seq) - 1, m)]


def pose_readings(q, t):
    """{reading: c2w (4, 4) OpenCV} for the four readings of one images.txt pose."""
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = qvec_to_rotmat(q), t
    inv = np.linalg.inv(M)
    return {"w2c_opencv": inv, "c2w_opencv": M, "w2c_opengl": inv @ FLIP,
            "c2w_opengl": M @ FLIP}


def overlap(sil, mask):
    inter, union = float((sil & mask).sum()), float((sil | mask).sum())
    nm, ns = float(mask.sum()), float(sil.sum())
    return dict(iou=inter / union if union else 0.0, cover=inter / nm if nm else math.nan,
                inside=inter / ns if ns else math.nan)


def mean_of(rows, key):
    v = [r[key] for r in rows if not math.isnan(r[key])]
    return float(np.mean(v)) if v else math.nan


def similarity_fit(src, dst):
    """(s, R, t, rms) with dst ~ s R src + t (Umeyama 1991); rms in dst units."""
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / (xs ** 2).sum(1).mean())
    t = md - s * R @ ms
    rms = float(np.sqrt((((s * src @ R.T + t) - dst) ** 2).sum(1).mean()))
    return s, R, t, rms


def backproject(depth, K, c2w, keep=None):
    """World points (M, 3) and pixel (v, u) of the depth > 0 pixels (gsplat pixel centres)."""
    ok = depth > 0
    if keep is not None:
        ok &= keep
    v, u = np.nonzero(ok)
    z = depth[v, u].astype(np.float64)
    pc = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
    return pc @ c2w[:3, :3].T + c2w[:3, 3], v, u


def reproject(fa, fb, keep_a=None, keep_b=None):
    """dz = z_b - z_pred (m), z_pred, and a's pixel (v, u), for a's valid depth pixels that
    land on a valid pixel of b. fa, fb: dicts of numpy depth, K_depth, c2w."""
    Pw, va, ua = backproject(fa["depth"], fa["K_depth"], fa["c2w"], keep_a)
    w2c = np.linalg.inv(fb["c2w"])
    P = Pw @ w2c[:3, :3].T + w2c[:3, 3]
    K, (h, w) = fb["K_depth"], fb["depth"].shape
    zp = P[:, 2]
    ok = zp > 0.05
    ub = np.floor(K[0, 0] * P[:, 0] / np.where(ok, zp, 1) + K[0, 2]).astype(np.int64)
    vb = np.floor(K[1, 1] * P[:, 1] / np.where(ok, zp, 1) + K[1, 2]).astype(np.int64)
    ok &= (ub >= 0) & (ub < w) & (vb >= 0) & (vb < h)
    zb = np.zeros_like(zp)
    zb[ok] = fb["depth"][vb[ok], ub[ok]]
    ok &= zb > 0
    if keep_b is not None:
        ok[ok] &= keep_b[vb[ok], ub[ok]]
    return zb[ok] - zp[ok], zp[ok], va[ok], ua[ok], int(len(zp))


def dz_stats(dz, zp, n_src):
    if len(dz) == 0:
        return dict(n=0, overlap=0.0, med_abs_dz=math.nan, ok5cm=math.nan, ratio=math.nan)
    return dict(n=int(len(dz)), overlap=len(dz) / max(n_src, 1),
                med_abs_dz=float(np.median(np.abs(dz))),
                ok5cm=float((np.abs(dz) < DZ_OK).mean()), ratio=float(np.median((dz + zp) / zp)))


def rigid_align(src, dst, iters=30, trim=DZ_OK):
    """4x4 rigid T moving src (M, 3) onto dst: point-to-point ICP from the identity, with
    the nearest-neighbour pairs closer than `trim`."""
    tree, T, cur = cKDTree(dst), np.eye(4), src
    for _ in range(iters):
        d, j = tree.query(cur, distance_upper_bound=trim)
        ok = np.isfinite(d)
        if ok.sum() < 3:
            break
        A, B = cur[ok], dst[j[ok]]
        ma, mb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((B - mb).T @ (A - ma))
        D = np.eye(3)
        D[2, 2] = np.sign(np.linalg.det(U @ Vt))
        step = np.eye(4)
        step[:3, :3] = U @ D @ Vt
        step[:3, 3] = mb - step[:3, :3] @ ma
        cur, T = cur @ step[:3, :3].T + step[:3, 3], step @ T
    return T


def pair_stats(fa, fb, keep_a=None, keep_b=None):
    """dz statistics of the pair (a, b), and the same (rigid_*) after b's pose is corrected
    by a rigid ICP fit of b's points (every 4th) onto a's."""
    dz, zp, _, _, n = reproject(fa, fb, keep_a, keep_b)
    st = dz_stats(dz, zp, n)
    Pa = backproject(fa["depth"], fa["K_depth"], fa["c2w"], keep_a)[0][::4]
    Pb = backproject(fb["depth"], fb["K_depth"], fb["c2w"], keep_b)[0][::4]
    if len(Pa) >= 3 and len(Pb) >= 3:
        fb = dict(fb, c2w=rigid_align(Pb, Pa) @ fb["c2w"])
        dz, zp, _, _, n = reproject(fa, fb, keep_a, keep_b)
    r = dz_stats(dz, zp, n)
    st.update(rigid_med_abs_dz=r["med_abs_dz"], rigid_ok5cm=r["ok5cm"], rigid_ratio=r["ratio"])
    return st


def np_frame(ds, i):
    """The loader's frame i as numpy arrays (depth, K_depth, c2w, masks); no image."""
    f = ds.frame(i)
    out = {k: f[k].numpy() for k in ("depth", "depth_confidence", "K_depth", "c2w", "K")}
    out["K_depth"], out["c2w"] = out["K_depth"].astype(np.float64), out["c2w"].astype(np.float64)
    for k in ("mask_real", "mask_synthetic"):
        if k in f:
            out[k] = f[k].numpy()
    out["index"] = f["index"]
    return out


def suggested_stride(n, target=125, max_stride=8):
    """The stride whose frame count ceil(n / s) is nearest `target` (smaller s on ties)."""
    return min(range(1, max_stride + 1), key=lambda s: (abs(-(-n // s) - target), s))


def keyframes_kept(c2ws):
    """Frames run_mapping.py's select() keeps (default --kf-trans / --kf-rot), no subset."""
    last, kept = None, 0
    for c in c2ws:
        if last is not None:
            dt = float(np.linalg.norm(c[:3, 3] - last[:3, 3]))
            cos = (np.trace(c[:3, :3].T @ last[:3, :3]) - 1.0) / 2.0
            dr = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            if dt < KF_TRANS and dr < KF_ROT:
                continue
        last, kept = c, kept + 1
    return kept


# ----------------------------------------------------------------------------- checks
def check_counts(ds):
    n, (s0, s1) = len(ds.rgb), ds.split
    parts = dict(map=range(0, s0), between=range(s0, s1), update=range(s1, n))
    masks = {}
    for name, rng in parts.items():
        real = [os.path.join(ds.root, "masks_real", f"{i:03d}.npy") for i in rng]
        syn = [os.path.join(ds.root, "masks_synthetic", f"{i:03d}.npy") for i in rng]
        have = [p for p in real if os.path.exists(p)]
        masks[name] = dict(frames=len(rng), real_files=len(have),
                           real_nonempty=int(sum(bool(np.load(p).any()) for p in have)),
                           synthetic_files=int(sum(os.path.exists(p) for p in syn)))
    ims = read_colmap_images(os.path.join(ds.root, "sparse", "images.txt"))
    K = ds.K
    spread = {k: float(np.abs(K[:, a, b] - np.median(K[:, a, b])).max())
              for k, (a, b) in dict(fx=(0, 0), fy=(1, 1), cx=(0, 2), cy=(1, 2)).items()}
    pp = K[:, :2, 2]                                      # principal point per frame
    pp_step = np.linalg.norm(np.diff(pp, axis=0), axis=1)
    pp_rand = np.linalg.norm(pp[np.random.default_rng(0).permutation(n)] - pp, axis=1)
    import cv2
    img_hw = cv2.imread(ds.rgb[0]).shape[:2]
    s = suggested_stride(s0)
    return dict(n=n, split_start=s0, split_end=s1, synthetic_end=ds.meta.get("synthetic_end"),
                parts=masks, image_hw=list(img_hw), camera_hw=list(ds.image_hw),
                images_txt_camera_ids=sorted(set(v[3] for v in ims.values()))[:5],
                K_max_dev_from_median_px=spread,
                pp_step_median_px=float(np.median(pp_step)),
                pp_random_pair_median_px=float(np.median(pp_rand)),
                map_stride=s, map_frames=-(-s0 // s),
                map_keyframes_kept=keyframes_kept(ds.c2w[0:s0:s]),
                update_frames_stride1=n - s1)


def check_pose_and_box(ds, frames, nf):
    """Mean IoU / cover / inside per (box file, pose reading) over the with-object frames."""
    ims = read_colmap_images(os.path.join(ds.root, "sparse", "images.txt"))
    boxes = {b: read_box_corners(os.path.join(ds.root, "boxes", b)) for b in BOX_FILES
             if os.path.exists(os.path.join(ds.root, "boxes", b))}
    table = {}
    for i in frames:
        f = nf[i]
        _, q, t, _ = ims[os.path.basename(ds.rgb[i])]
        for reading, c2w in pose_readings(q, t).items():
            for b, corners in boxes.items():
                sil = box_silhouette(corners, c2w, f["K_depth"], f["depth"].shape)
                table.setdefault((b, reading), []).append(overlap(sil, f["mask_real"]))
    rows = [dict(box=b, reading=r, iou=mean_of(v, "iou"), cover=mean_of(v, "cover"),
                 inside=mean_of(v, "inside"), iou_min=float(min(x["iou"] for x in v)),
                 n=len(v)) for (b, r), v in table.items()]
    rows.sort(key=lambda r: -r["iou"])
    # the loader's c2w is the w2c_opencv reading
    dev = max(float(np.abs(nf[i]["c2w"] - pose_readings(*ims[os.path.basename(ds.rgb[i])][1:3])
                           ["w2c_opencv"]).max()) for i in frames)
    nerf = {}
    if all(b in boxes for b in ("bbox.txt", "bbox_nerf.txt")):
        s, R, t, rms = similarity_fit(boxes["bbox_nerf.txt"], boxes["bbox.txt"])
        meta = json.load(open(os.path.join(ds.root, "boxes", "meta.json")))
        nerf = dict(scale=s, meta_scale_half=float(meta["scale"]) / 2.0, rms_m=rms)
        if all(b in boxes for b in ("bbox_nerf_refined.txt", "bbox_refined.txt")):
            mapped = s * boxes["bbox_nerf_refined.txt"] @ R.T + t
            d = np.linalg.norm(mapped[:, None] - boxes["bbox_refined.txt"][None], axis=-1)
            nerf["refined_max_corner_dist_m"] = float(d.min(1).max())
    sizes = {}
    for b in ("bbox.txt", "bbox_refined.txt"):
        if b in boxes:
            sizes[b] = sorted(round(float(x), 3) for x in box_extents(boxes[b]))
    same_ply = {}
    for txt, ply in (("bbox.txt", "bbox.ply"), ("bbox_refined.txt", "bbox_refined.ply")):
        if txt in boxes and ply in boxes:
            d = np.linalg.norm(boxes[ply][:, None] - boxes[txt][None], axis=-1)
            same_ply[ply] = float(d.min(1).max())
    return dict(ranking=rows, best=rows[0], loader_c2w_max_dev=dev, nerf_frame=nerf,
                box_sizes_m=sizes, ply_vs_txt_max_corner_dist_m=same_ply)


def check_lidar(ds, map_frames, upd_frames, nf):
    corners = read_box_corners(ds.box_path)
    acc = dict(mask_valid=0, mask_in=0, in_box=0, out_valid=0, out_in=0)
    for i in map_frames:
        f = nf[i]
        P, v, u = backproject(f["depth"], f["K_depth"], f["c2w"])
        inb, m = points_in_box(P, corners, INFLATE), f["mask_real"][v, u]
        acc["mask_valid"] += int(m.sum())
        acc["mask_in"] += int((m & inb).sum())
        acc["in_box"] += int(inb.sum())
        acc["out_valid"] += int((~m).sum())
        acc["out_in"] += int((~m & inb).sum())
    up = dict(mask_valid=0, mask_in=0, valid=0, in_box=0)
    for i in upd_frames:
        f = nf[i]
        P, v, u = backproject(f["depth"], f["K_depth"], f["c2w"])
        inb, m = points_in_box(P, corners, INFLATE), f["mask_real"][v, u]
        up["mask_valid"] += int(m.sum())
        up["mask_in"] += int((m & inb).sum())
        up["valid"] += len(inb)
        up["in_box"] += int(inb.sum())
    r = lambda a, b: a / b if b else math.nan
    return dict(box=os.path.basename(ds.box_path), inflate_m=INFLATE,
                box_given_mask=r(acc["mask_in"], acc["mask_valid"]),
                mask_given_box=r(acc["mask_in"], acc["in_box"]),
                box_given_not_mask=r(acc["out_in"], acc["out_valid"]),
                n_mask_valid=acc["mask_valid"], n_in_box=acc["in_box"],
                update_box_given_mask=r(up["mask_in"], up["mask_valid"]),
                update_box_share_of_valid=r(up["in_box"], up["valid"]),
                update_n_mask_valid=up["mask_valid"], update_n_in_box=up["in_box"])


def check_near_after(ds, map_frames, upd_frames, nf, near=0.02):
    """Share of the with-object in-box points that lie within `near` of an after-removal
    point, for the inflated and the tight box."""
    corners = read_box_corners(ds.box_path)
    pts = lambda frames: [backproject(nf[i]["depth"], nf[i]["K_depth"], nf[i]["c2w"])[0]
                          for i in frames]
    after = np.concatenate(pts(upd_frames))
    after = after[points_in_box(after, corners, INFLATE + near)]
    before = np.concatenate(pts(map_frames))
    out = dict(near_m=near, n_after_near_box=int(len(after)))
    for name, infl in (("inflated", INFLATE), ("tight", 0.0)):
        P = before[points_in_box(before, corners, infl)]
        if len(P) == 0 or len(after) == 0:
            out[name] = dict(n=int(len(P)), near_after_share=0.0 if len(P) else math.nan)
            continue
        d, _ = cKDTree(after).query(P, distance_upper_bound=near)
        out[name] = dict(n=int(len(P)), near_after_share=float(np.isfinite(d).mean()))
    return out


def nearest_pose(c2ws, cands, b):
    """The frame of cands whose pose is nearest frame b (|dC| / 0.1 m + angle / 10 deg)."""
    def cost(a):
        dt = np.linalg.norm(c2ws[a][:3, 3] - c2ws[b][:3, 3])
        cos = float(np.clip(c2ws[a][:3, 2] @ c2ws[b][:3, 2], -1, 1))
        return dt / 0.1 + math.degrees(math.acos(cos)) / 10.0
    return min(cands, key=cost)


def check_depth(ds, raw, nf, n_pairs, gap, revisit):
    s0, s1 = ds.split
    n = len(ds.rgb)
    cams = read_colmap_cameras(os.path.join(ds.root, "sparse", "cameras.txt"))
    _, W, H, (fx, fy, cx, cy) = cams[1]
    (h, w) = ds.DEPTH_HW
    K1d = np.array([[fx * w / W, 0, cx * w / W], [0, fy * h / H, cy * h / H], [0, 0, 1.0]])
    mm, mm1, levels = [], [], {L: [] for L in (0, 1, 2)}
    for a in evenly(range(0, max(s0 - gap, 1)), n_pairs):
        b = a + gap
        fa, fb = nf.get(a) or np_frame(ds, a), nf.get(b) or np_frame(ds, b)
        dz, zp, _, _, nsrc = reproject(fa, fb)
        mm.append(dz_stats(dz, zp, nsrc))
        dz, zp, _, _, nsrc = reproject(dict(fa, K_depth=K1d), dict(fb, K_depth=K1d))
        mm1.append(dz_stats(dz, zp, nsrc))
        ra, rb = np_frame(raw, a), np_frame(raw, b)              # unfiltered depth
        rb["depth"] = np.where(rb["depth_confidence"] == 2, rb["depth"], 0.0)
        for L in levels:
            at_level = ra["depth_confidence"] == L
            dz, zp, _, _, _ = reproject(ra, rb, keep_a=at_level)
            levels[L].append((dz, zp, int((at_level & (ra["depth"] > 0)).sum())))
    rv = []
    for a in evenly(range(0, s0), n_pairs):
        cands = [c for c in range(0, s0) if abs(c - a) >= revisit]
        if cands:
            b = nearest_pose(ds.c2w, cands, a)
            st = pair_stats(np_frame(ds, a), np_frame(ds, b))
            st.update(a=int(a), b=int(b))
            rv.append(st)
    mu = []
    for b in evenly(range(s1, n), n_pairs):
        a = nearest_pose(ds.c2w, range(0, s0), b)
        fa, fb = np_frame(ds, a), np_frame(ds, b)
        st = pair_stats(fa, fb, keep_a=~fa["mask_real"], keep_b=~fb["mask_real"])
        st.update(a=int(a), b=int(b))
        mu.append(st)

    def pool(rows):
        keys = [k for k in ("med_abs_dz", "ok5cm", "ratio", "overlap", "rigid_med_abs_dz",
                            "rigid_ok5cm", "rigid_ratio") if rows and k in rows[0]]
        if not rows:
            return dict(n_pairs=0)
        return {**{k: float(np.nanmedian([r[k] for r in rows])) for k in keys},
                "n_pairs": len(rows), "min_ok5cm": float(np.nanmin([r["ok5cm"] for r in rows]))}
    lv = {}
    for L, items in levels.items():
        dz = np.concatenate([x[0] for x in items]) if items else np.zeros(0)
        zp = np.concatenate([x[1] for x in items]) if items else np.zeros(0)
        lv[str(L)] = dz_stats(dz, zp, sum(x[2] for x in items))
        lv[str(L)]["share_of_pixels"] = (sum(x[2] for x in items)
                                         / (len(items) * ds.DEPTH_HW[0] * ds.DEPTH_HW[1]))
    return dict(map_map=pool(mm), map_map_camera1=pool(mm1), revisit=pool(rv),
                map_update=pool(mu), revisit_pairs=rv,
                map_update_pairs=mu, by_confidence=lv, gap=gap, revisit_gap=revisit,
                min_confidence=ds.min_confidence)


def check_masks(ds, upd_frames, nf):
    corners = read_box_corners(ds.box_path)
    rows = []
    for i in upd_frames:
        f = nf[i]
        if "mask_synthetic" not in f or not f["mask_real"].any():
            continue
        sil = box_silhouette(corners, f["c2w"], f["K_depth"], f["depth"].shape)
        syn, real = f["mask_synthetic"], f["mask_real"]
        rows.append(dict(iou_syn_real=overlap(syn, real)["iou"],
                         syn_inside_real=overlap(syn, real)["inside"],
                         iou_sil_real=overlap(sil, real)["iou"],
                         iou_sil_syn=overlap(sil, syn)["iou"]))
    keys = ("iou_syn_real", "syn_inside_real", "iou_sil_real", "iou_sil_syn")
    return {**{k: mean_of(rows, k) for k in keys}, "n": len(rows)}


def check_scene(root, n_frames, gap, revisit):
    ds = NeRFObjectRemovalDataset(root, part="all")
    raw = NeRFObjectRemovalDataset(root, part="all", min_confidence=0)
    s0, s1 = ds.split
    counts = check_counts(ds)
    with_obj = [i for i in range(s0) if np.load(os.path.join(
        root, "masks_real", f"{i:03d}.npy")).any()]
    map_frames = evenly(with_obj, n_frames)
    upd_frames = evenly([i for i in range(s1, len(ds.rgb)) if np.load(os.path.join(
        root, "masks_real", f"{i:03d}.npy")).any()], n_frames)
    nf = {i: np_frame(ds, i) for i in map_frames + upd_frames}
    return dict(counts=counts, pose_box=check_pose_and_box(ds, map_frames, nf),
                lidar=check_lidar(ds, map_frames, upd_frames, nf),
                near_after=check_near_after(ds, map_frames, upd_frames, nf),
                depth=check_depth(ds, raw, nf, n_frames, gap, revisit),
                masks_update=check_masks(ds, upd_frames, nf),
                frames=dict(with_object=map_frames, update=upd_frames))


def _pairs(x):
    if not x.get("n_pairs"):
        return "none"
    s = f"ok5cm {x['ok5cm']:.3f} med {100 * x['med_abs_dz']:.2f}cm ratio {x['ratio']:.4f}"
    if "rigid_med_abs_dz" in x:
        s += f" -> rigid {x['rigid_ok5cm']:.3f} / {100 * x['rigid_med_abs_dz']:.2f}cm"
    return s


def summary_line(name, r):
    c, p, l, d, m = r["counts"], r["pose_box"], r["lidar"], r["depth"], r["masks_update"]
    b, nxt, st = p["best"], p["ranking"][1], r["near_after"]
    lv = d["by_confidence"]
    return (f"{name} N={c['n']} map={c['split_start']} upd={c['n'] - c['split_end']} stride "
            f"{c['map_stride']} -> {c['map_frames']} (kf {c['map_keyframes_kept']}) | pose "
            f"{b['box']}/{b['reading']} IoU {b['iou']:.3f} (min {b['iou_min']:.3f}, cover "
            f"{b['cover']:.3f}, inside {b['inside']:.3f}); 2nd {nxt['box']}/{nxt['reading']} "
            f"{nxt['iou']:.3f} | principal point step {c['pp_step_median_px']:.2f} px "
            f"(random pairs {c['pp_random_pair_median_px']:.2f})"
            f"\n    lidar box|mask {l['box_given_mask']:.3f} mask|box "
            f"{l['mask_given_box']:.3f} box|~mask {l['box_given_not_mask']:.4f}; after removal "
            f"box|mask {l['update_box_given_mask']:.3f}; in-box points near an after-removal "
            f"point {st['inflated']['near_after_share']:.3f} (tight box "
            f"{st['tight']['near_after_share']:.3f})"
            f"\n    depth map-map {_pairs(d['map_map'])} (camera 1: med "
            f"{100 * d['map_map_camera1']['med_abs_dz']:.2f}cm); revisit "
            f"{_pairs(d['revisit'])}; map-upd {_pairs(d['map_update'])}; conf ok5cm 2/1/0 "
            f"{lv['2']['ok5cm']:.3f}/{lv['1']['ok5cm']:.3f}/{lv['0']['ok5cm']:.3f}"
            f"\n    masks after removal IoU(syn,real) {m['iou_syn_real']:.3f} IoU(box,real) "
            f"{m['iou_sil_real']:.3f} IoU(box,syn) {m['iou_sil_syn']:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="folder holding the scene folders 001..016")
    ap.add_argument("--out", required=True, help="output JSON")
    ap.add_argument("--scenes", default="", help="comma-separated scene names (default: all)")
    ap.add_argument("--frames", type=int, default=12, help="frames (and pairs) per check")
    ap.add_argument("--gap", type=int, default=10, help="frame gap of the map-map pairs")
    ap.add_argument("--revisit", type=int, default=60,
                    help="least frame gap of the revisit pairs (inside the mapping part)")
    args = ap.parse_args(argv)
    scenes = ([s.strip() for s in args.scenes.split(",") if s.strip()] if args.scenes else
              sorted(d for d in os.listdir(args.root)
                     if os.path.isfile(os.path.join(args.root, d, "meta.json"))))
    t0 = time.time()
    cmd = ["python3", "scripts/check_nor.py"] + list(sys.argv[1:] if argv is None else argv)
    out = dict(command=" ".join(cmd), start=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
               root=os.path.abspath(args.root), frames=args.frames, gap=args.gap,
               revisit=args.revisit, inflate_m=INFLATE, scenes={})
    failed = []
    for s in scenes:
        try:
            r = check_scene(os.path.join(args.root, s), args.frames, args.gap, args.revisit)
        except Exception as e:                                # keep checking the others
            failed.append(s)
            out["scenes"][s] = dict(error=f"{type(e).__name__}: {e}")
            print(f"{s} FAILED {type(e).__name__}: {e}", flush=True)
            continue
        out["scenes"][s] = r
        print(summary_line(s, r), flush=True)
    ok = [r for r in out["scenes"].values() if "error" not in r]
    winners = sorted({(r["pose_box"]["best"]["box"], r["pose_box"]["best"]["reading"]) for r in ok})
    out["winners"] = [list(w) for w in winners]
    out["wall_s"] = time.time() - t0
    print(f"winners (box, reading) over {len(ok)} scenes: {winners}; failed: {failed}; "
          f"{out['wall_s']:.0f} s")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
