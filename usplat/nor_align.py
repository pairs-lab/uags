"""
nor_align.py — one rigid transform per NeRF-Object-Removal scene that moves the update part
(the frames after the removal) onto the map part (protocol decision D3, 2026-10-17), and the
depth geometry shared with scripts/check_nor.py.

Why: the two parts of a scene come from one ARKit capture, but in scenes 012 and 016 the
update part is offset: its depth lands 3.54 and 2.50 cm (median) from the map part's depth,
and 0.58 / 0.89 cm after a rigid fit per frame pair (scripts/check_nor.py, map-update pairs).

The fit (fit_pairs), one rule for every scene and every method:
  pairs   check_nor.py's pairing: each update frame with the mapping frame of the nearest
          pose. Fit frames: 48 evenly spaced update frames minus the 12 check frames below
          (the check pairs are held out of the fit)
  points  the loader's depth (high confidence only, its default) of the update frame, every
          pixel_step-th pixel, back-projected with its pose; both frames' points OUTSIDE the
          object's box inflated by 2 cm (static content)
  ICP     started at the identity; each iteration projects the moved update points into
          their mapping frame (projective association), takes the mapping frame's point q
          and normal n at that pixel (normals from the depth grid's neighbours; no normal
          across a depth jump > 5 cm) and minimises sum (n . (T p - q))^2 over all pairs
          together (point-to-plane), pairs with |n . (T p - q)| < trim and |T p - q| < 2 trim
  truncation  one Gauss-Newton step per iteration, in coordinates centred on the points'
          mean with rotations scaled by their rms radius L; only the eigen-directions of
          J^T J with at least `tau` (0.1) of the largest eigenvalue move, the others stay at
          the identity start (a minimum-norm step). Without it, directions that the depth
          barely constrains (sliding along the dominant walls) drift by 2-13 cm and wreck the
          photometric check below (measured on 001-004, 012, 016 before any method ran:
          report of 2026-10-17; also for check_nor's point-to-point fit on whole clouds)
  result  T (4x4, float64): x_map = T x_update in world coordinates, so an update camera
          becomes c2w' = T c2w. NeRFObjectRemovalDataset(..., align=<json>) applies T to every
          frame of the update part (frame numbers >= split_end) and to no other frame
Checks on the 12 held-out pairs (check_nor.py's map-update pairs, masks_real pixels left out),
raw pose vs aligned pose:
  depth   median |dz| of the reprojected depth (reproject / dz_stats, check_nor.py's)
  photo   NCC of the update frame's high-passed grey image (minus a Gaussian blur of sigma 8
          px) against the mapping frame's, warped into it with the mapping frame's depth
          (static pixels only; median over pairs). It sees the directions the depth does not.
          Its ceiling without drift: the same NCC on 12 mapping-frame pairs 10 frames apart
          (photo_map_map). It is very sensitive: moving those pairs' second frame by a T that
          displaces the points by 0.3 cm lowers it from 0.82 to 0.55 (scene 015)
  shift   how far T moves the update part's points (median and max |T p - p| over the
          check frames' depth points). T's own translation is not a displacement: T rotates
          about the world origin, which can lie metres from the scene
save_alignment writes the JSON: scene, T, its translation and rotation, the settings, the
pairs, the ICP trace, the shift and the checks. The loader reads only "scene" (it must name
the scene folder) and "T".

Geometry helpers (also used by check_nor.py): backproject, reproject, dz_stats, rigid_align
(point-to-point ICP), nearest_pose, evenly. Frames are dicts of numpy arrays: depth (h, w) [m]
with 0 = invalid, K_depth (3, 3) of that grid, c2w (4, 4) OpenCV camera-to-world.
"""
from __future__ import annotations
import json
import math
import os

import numpy as np
from scipy.spatial import cKDTree

from .box3d import read_box_corners, points_in_box

DZ_OK = 0.05                     # [m] pair residual counted as consistent; ICP pair trim
INFLATE = 0.02                   # [m] the box margin of the stale ground truth (D2)
DEPTH_JUMP = 0.05                # [m] no normal across a larger depth step
PHOTO_SIGMA = 8.0                # [px] blur removed from the grey images of the photo check


# ------------------------------------------------------------------- geometry helpers
def evenly(seq, m):
    """m evenly spaced items of seq (first and last included); all of seq if len <= m."""
    seq = list(seq)
    if len(seq) <= m:
        return seq
    return [seq[int(round(x))] for x in np.linspace(0, len(seq) - 1, m)]


def backproject(depth, K, c2w, keep=None):
    """World points (M, 3) and pixel (v, u) of the depth > 0 pixels (gsplat pixel centres)."""
    ok = depth > 0
    if keep is not None:
        ok &= keep
    v, u = np.nonzero(ok)
    z = depth[v, u].astype(np.float64)
    pc = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
    return pc @ c2w[:3, :3].T + c2w[:3, 3], v, u


def project(P, K, c2w, hw):
    """Pixel (v, u) (floor of the projection, int) and camera depth z of world points P in a
    camera c2w with intrinsics K; ok = in front (z > 5 cm) and inside the (h, w) grid."""
    w2c = np.linalg.inv(c2w)
    Pc = P @ w2c[:3, :3].T + w2c[:3, 3]
    z = Pc[:, 2]
    ok = z > 0.05
    u = np.floor(K[0, 0] * Pc[:, 0] / np.where(ok, z, 1) + K[0, 2]).astype(np.int64)
    v = np.floor(K[1, 1] * Pc[:, 1] / np.where(ok, z, 1) + K[1, 2]).astype(np.int64)
    ok &= (u >= 0) & (u < hw[1]) & (v >= 0) & (v < hw[0])
    return v, u, z, ok


def reproject(fa, fb, keep_a=None, keep_b=None):
    """dz = z_b - z_pred (m), z_pred, and a's pixel (v, u), for a's valid depth pixels that
    land on a valid pixel of b. fa, fb: dicts of numpy depth, K_depth, c2w."""
    Pw, va, ua = backproject(fa["depth"], fa["K_depth"], fa["c2w"], keep_a)
    vb, ub, zp, ok = project(Pw, fb["K_depth"], fb["c2w"], fb["depth"].shape)
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
    the nearest-neighbour pairs closer than `trim` (check_nor.py's per-pair fit)."""
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


def rotation_deg(T):
    """Rotation angle [deg] of a 4x4 (or 3x3) rigid transform."""
    R = np.asarray(T)[:3, :3]
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def rotation_exp(w):
    """Rotation matrix of the rotation vector w (Rodrigues)."""
    th = float(np.linalg.norm(w))
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + math.sin(th) / th * K + (1.0 - math.cos(th)) / th ** 2 * K @ K


def nearest_pose(c2ws, cands, b):
    """The frame of cands whose pose is nearest frame b (|dC| / 0.1 m + angle / 10 deg)."""
    def cost(a):
        dt = np.linalg.norm(c2ws[a][:3, 3] - c2ws[b][:3, 3])
        cos = float(np.clip(c2ws[a][:3, 2] @ c2ws[b][:3, 2], -1, 1))
        return dt / 0.1 + math.degrees(math.acos(cos)) / 10.0
    return min(cands, key=cost)


# ------------------------------------------------------------------- frames
def frame_arrays(ds, i):
    """Frame number i of a NeRFObjectRemovalDataset as numpy: depth (the loader's confidence
    filter), K_depth, c2w (the dataset's pose, float64) and mask_real (None if no file)."""
    depth, _ = ds.read_depth(i)
    p = os.path.join(ds.root, "masks_real", f"{i:03d}.npy")
    mask = np.load(p).astype(bool) if os.path.exists(p) else None
    return dict(depth=depth, K_depth=np.asarray(ds.K_depth[i], dtype=np.float64),
                c2w=np.asarray(ds.c2w[i], dtype=np.float64), mask_real=mask)


def outside_box(f, corners, inflate=INFLATE):
    """(h, w) bool: valid depth pixels whose point is outside the inflated box."""
    P, v, u = backproject(f["depth"], f["K_depth"], f["c2w"])
    keep = np.zeros(f["depth"].shape, bool)
    keep[v, u] = ~points_in_box(P, corners, inflate)
    return keep


def vertex_normal_maps(f, jump=DEPTH_JUMP):
    """World point V (h, w, 3) and unit normal N (h, w, 3) of every pixel, and where the
    normal is defined (the pixel and its 4 neighbours valid, no depth step > jump)."""
    d, K, c2w = f["depth"], f["K_depth"], f["c2w"]
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    pc = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * d, (v + 0.5 - K[1, 2]) / K[1, 1] * d, d], -1)
    V = pc @ c2w[:3, :3].T + c2w[:3, 3]
    ok = d > 0
    n = np.cross(V[1:-1, 2:] - V[1:-1, :-2], V[2:, 1:-1] - V[:-2, 1:-1])
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    good = (ok[1:-1, 1:-1] & ok[1:-1, 2:] & ok[1:-1, :-2] & ok[2:, 1:-1] & ok[:-2, 1:-1]
            & (np.abs(d[1:-1, 2:] - d[1:-1, :-2]) < jump)
            & (np.abs(d[2:, 1:-1] - d[:-2, 1:-1]) < jump) & (nn[..., 0] > 0))
    N, defined = np.zeros_like(V), np.zeros_like(ok)
    N[1:-1, 1:-1] = n / np.maximum(nn, 1e-12)
    defined[1:-1, 1:-1] = good
    return V, N, defined


# ------------------------------------------------------------------- the D3 fit
def _pair_data(ds, pairs, corners, inflate, pixel_step):
    """Per pair: the mapping frame's maps (V, N, usable) and the update frame's points."""
    maps, data = {}, []
    for a, b in pairs:
        if a not in maps:
            fa = frame_arrays(ds, a)
            V, N, defined = vertex_normal_maps(fa)
            maps[a] = (V, N, defined & outside_box(fa, corners, inflate), fa)
        fb = frame_arrays(ds, b)
        sub = np.zeros(fb["depth"].shape, bool)
        sub[::pixel_step, ::pixel_step] = True
        P, _, _ = backproject(fb["depth"], fb["K_depth"], fb["c2w"],
                              outside_box(fb, corners, inflate) & sub)
        data.append((a, P))
    return maps, data


def fit_pairs(ds, pairs, corners, iters=20, trim=DZ_OK, tau=0.1, pixel_step=2,
              inflate=INFLATE, trace=None):
    """T (4x4) of the module docstring's fit; trace (a list) gets one dict per iteration."""
    maps, data = _pair_data(ds, pairs, corners, inflate, pixel_step)
    allP = np.concatenate([P for _, P in data])
    c = allP.mean(0)
    L = float(np.sqrt(((allP - c) ** 2).sum(1).mean()))
    T = np.eye(4)
    for _ in range(iters):
        Js, rs = [], []
        for a, Pb in data:
            V, N, usable, fa = maps[a]
            P = Pb @ T[:3, :3].T + T[:3, 3]
            v, u, _, ok = project(P, fa["K_depth"], fa["c2w"], fa["depth"].shape)
            ok[ok] &= usable[v[ok], u[ok]]
            q, n, p = V[v[ok], u[ok]], N[v[ok], u[ok]], P[ok]
            r = ((p - q) * n).sum(1)
            g = (np.abs(r) < trim) & (np.linalg.norm(p - q, axis=1) < 2 * trim)
            Js.append(np.concatenate([np.cross(p[g] - c, n[g]) / L, n[g]], 1))
            rs.append(r[g])
        J, r = np.concatenate(Js), np.concatenate(rs)
        if len(r) < 6:
            break
        ev, E = np.linalg.eigh(J.T @ J)
        keep = ev >= tau * ev[-1]
        x = E[:, keep] @ (-(E[:, keep].T @ (J.T @ r)) / ev[keep])
        R = rotation_exp(x[:3] / L)
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = R, c - R @ c + x[3:]           # rotation about c, then t
        T = step @ T
        if trace is not None:
            trace.append(dict(pairs=int(len(r)), rms_m=float(np.sqrt((r ** 2).mean())),
                              directions=int(keep.sum()), step_m=float(np.linalg.norm(x[3:])),
                              step_deg=rotation_deg(step), eig=[float(e) for e in ev]))
    return T


# ------------------------------------------------------------------- checks
def pair_errors(ds, pairs, T=None):
    """dz_stats of each (map frame a, update frame b) pair, masks_real pixels of both left
    out; with T, b's pose is moved to T c2w_b first."""
    rows = []
    for a, b in pairs:
        fa, fb = frame_arrays(ds, a), frame_arrays(ds, b)
        if T is not None:
            fb["c2w"] = np.asarray(T, dtype=np.float64) @ fb["c2w"]
        keep = lambda f: None if f["mask_real"] is None else ~f["mask_real"]
        dz, zp, _, _, n = reproject(fa, fb, keep(fa), keep(fb))
        rows.append(dict(a=int(a), b=int(b), **dz_stats(dz, zp, n)))
    return rows


def _grey_highpass(path, sigma=PHOTO_SIGMA):
    import cv2
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), sigma)


def _bilinear(img, x, y):
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    top = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    bot = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bot * fy


def photo_ncc(ds, pairs, corners, T=None, inflate=INFLATE):
    """Per pair (a map, b update): NCC of b's high-passed grey image against a's, warped into
    b with a's depth (pixels outside the inflated box and masks_real only); with T, b's pose
    is T c2w_b. The depth grid is the image grid scaled (ARKit), so a's colour is read at the
    depth pixel's centre."""
    H, W = ds.image_hw
    rows = []
    for a, b in pairs:
        fa, fb = frame_arrays(ds, a), frame_arrays(ds, b)
        keep = outside_box(fa, corners, inflate)
        if fa["mask_real"] is not None:
            keep &= ~fa["mask_real"]
        P, v, u = backproject(fa["depth"], fa["K_depth"], fa["c2w"], keep)
        c2w_b = fb["c2w"] if T is None else np.asarray(T, dtype=np.float64) @ fb["c2w"]
        w2c = np.linalg.inv(c2w_b)
        Pc = P @ w2c[:3, :3].T + w2c[:3, 3]
        Kb, z = ds.K[b], Pc[:, 2]
        ok = z > 0.05
        xb = Kb[0, 0] * Pc[:, 0] / np.where(ok, z, 1) + Kb[0, 2] - 0.5   # pixel-centre coords
        yb = Kb[1, 1] * Pc[:, 1] / np.where(ok, z, 1) + Kb[1, 2] - 0.5
        ha, wa = fa["depth"].shape
        xa, ya = (u + 0.5) * W / wa - 0.5, (v + 0.5) * H / ha - 0.5
        ok &= (xb >= 0) & (xb < W - 1) & (yb >= 0) & (yb < H - 1)
        ok &= (xa >= 0) & (xa < W - 1) & (ya >= 0) & (ya < H - 1)
        ncc = math.nan
        if ok.sum() >= 10:                                       # no overlap: undefined
            ia = _bilinear(_grey_highpass(ds.rgb[a]), xa[ok], ya[ok])
            ib = _bilinear(_grey_highpass(ds.rgb[b]), xb[ok], yb[ok])
            ia, ib = ia - ia.mean(), ib - ib.mean()
            den = math.sqrt(float((ia * ia).sum() * (ib * ib).sum()))
            ncc = float((ia * ib).sum() / den) if den > 0 else math.nan
        rows.append(dict(a=int(a), b=int(b), n=int(ok.sum()), ncc=ncc))
    return rows


def pooled(rows, keys=("med_abs_dz", "ok5cm")):
    """Median over the pairs of each key (check_nor.py's pooling)."""
    out = dict(n_pairs=len(rows))
    for k in keys:
        vals = [r[k] for r in rows if k in r and not math.isnan(r[k])]
        out[k] = float(np.median(vals)) if vals else math.nan
    return out


def displacement(ds, frames, T, every=50):
    """Median and max |T p - p| [m] over every `every`-th depth point of the frames."""
    T = np.asarray(T, dtype=np.float64)
    P = [backproject(f["depth"], f["K_depth"], f["c2w"])[0][::every]
         for f in (frame_arrays(ds, i) for i in frames)]
    P = np.concatenate(P) if P else np.zeros((0, 3))
    if not len(P):
        return dict(median_m=math.nan, max_m=math.nan, n=0)
    d = np.linalg.norm(P @ T[:3, :3].T + T[:3, 3] - P, axis=1)
    return dict(median_m=float(np.median(d)), max_m=float(d.max()), n=int(len(d)))


def estimate_alignment(root, check_pairs=12, fit_frames=48, iters=20, trim=DZ_OK, tau=0.1,
                       pixel_step=2, inflate=INFLATE, map_gap=10):
    """The D3 transform of one scene folder and its record (see the module docstring)."""
    from .datasets import NeRFObjectRemovalDataset
    ds = NeRFObjectRemovalDataset(root, part="all")                    # raw poses
    s0, s1 = ds.split
    n = len(ds.rgb)
    corners = read_box_corners(ds.box_path)
    pair = lambda b: (nearest_pose(ds.c2w, range(0, s0), b), b)
    check = [pair(b) for b in evenly(range(s1, n), check_pairs)]
    held = {b for _, b in check}
    fit = [pair(b) for b in evenly(range(s1, n), fit_frames) if b not in held]
    trace = []
    T = fit_pairs(ds, fit, corners, iters, trim, tau, pixel_step, inflate, trace)
    dz0, dz1 = pair_errors(ds, check), pair_errors(ds, check, T)
    ph0, ph1 = photo_ncc(ds, check, corners, None, inflate), photo_ncc(ds, check, corners, T,
                                                                      inflate)
    mm = [(a, a + map_gap) for a in evenly(range(0, max(s0 - map_gap, 1)), check_pairs)]
    ceiling = pooled(photo_ncc(ds, mm, corners, None, inflate), ("ncc",))["ncc"]
    return dict(scene=os.path.basename(os.path.normpath(root)), root=os.path.abspath(root),
                T=T.tolist(), translation_m=float(np.linalg.norm(T[:3, 3])),
                rotation_deg=rotation_deg(T), shift=displacement(ds, [b for _, b in check], T),
                photo_map_map=dict(ncc=ceiling, gap=map_gap, n_pairs=len(mm)),
                settings=dict(method="pairwise projective point-to-plane ICP, truncated",
                              check_pairs=check_pairs, fit_frames=fit_frames, iters=iters,
                              trim_m=trim, tau=tau, pixel_step=pixel_step, inflate_m=inflate,
                              depth_jump_m=DEPTH_JUMP, photo_sigma_px=PHOTO_SIGMA,
                              map_map_gap=map_gap,
                              min_confidence=ds.min_confidence,
                              box=os.path.basename(ds.box_path), start="identity"),
                fit_pairs=fit, check_pairs=check, icp=trace,
                before=dict(**pooled(dz0), ncc=pooled(ph0, ("ncc",))["ncc"]),
                after=dict(**pooled(dz1), ncc=pooled(ph1, ("ncc",))["ncc"]),
                pairs_before=[dict(**d, ncc=p["ncc"]) for d, p in zip(dz0, ph0)],
                pairs_after=[dict(**d, ncc=p["ncc"]) for d, p in zip(dz1, ph1)])


def save_alignment(path, rec):
    """Write an alignment record as JSON (atomic: never a truncated file)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, path)
    return path
