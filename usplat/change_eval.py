"""
change_eval.py — labelled change evaluation: the stale ground truth per Gaussian (RIO10,
Table II tab:rio10) and the metrics shared with the semantic-swap benchmark (usplat.swap).

(a) STALE GROUND TRUTH, from object annotations that no method reads. The thresholds are
read from configs/rio10_change.json, fixed before any method output exists.
  Object o (an instance id of the mapping scan seqXX_01; usplat.datasets_rio labels.ply):
    removed   >= min_points vertices of o in seqXX_01 and < min_points in seqXX_02
    moved     centroid shift ||c_02 - c_01|| > d_move
    rotated   rotation angle of an ICP fit of the CENTRED point sets > r_deg (point-to-point,
              from the identity: a symmetric object turned onto itself reads as unturned,
              which is also what a camera sees)
    stale = removed | moved | rotated;  static = present in both scans and neither;
    unjudged = an ignored id (0 = unlabelled), fewer than min_points vertices in seqXX_01,
    or nonrigid (3RScan route).
  3RScan route (object_motion_from_transforms): removed / nonrigid lists and the rigid
  instance transform T: shift = ||T c - c|| with c the seqXX_01 centroid, rotation = angle
  of T's rotation; instances in no list are static.
  The 3RScan instance ids must be the ids of the RIO10 labels.ply (datasets_rio V10):
  rscan_id_check refuses lists that name ids absent from it (else every object reads static).
  Gaussian n with centre x_n (gaussian_stale_labels): j = the nearest ANNOTATED seqXX_01
  vertex (ids in ignore_ids, 0 = unlabelled, are not searched); if ||x_n - p_j|| <= d_near
  the Gaussian takes the status of object(j), else it is unjudged ("far"). Gaussians that no
  update view observes, or that the new surface hides in every update view, are unjudged
  too; observed / hidden are INPUT masks, made by visibility_from_depth (centre projection
  into every update view and a depth test against the rescan's depth, with a margin) and
  optionally an update-pass splat mass floor. The update views are the frames the field
  ingests: select_keyframes repeats scripts/run_mapping.py's keyframe rule (finite pose, no
  near-duplicate of the last kept frame), so the judged population is the update frusta.
  Categories are exclusive in the order unobserved > hidden > far > object unjudged, and
  their shares of all Gaussians are reported (the paper's "% unobserved, % hidden").
  Nearest neighbours: scipy cKDTree, exact; a tie at exactly equal distance goes to the
  lowest vertex index among the k_ties nearest (deterministic).
  Segmentation targets (only with a class map RIO10 label -> vocabulary id; rescan_targets):
  a static Gaussian keeps its object's class, a stale one takes the class of the nearest
  seqXX_02 vertex within d_near (what the location shows now), else unknown.

(b) METRICS of a per-Gaussian score s (higher = more likely changed) on the population of
JUDGED Gaussians (label 1 stale / 0 static; -1 unjudged is left out):
  AP     = sum_k (R_k - R_{k-1}) P_k over the DISTINCT score thresholds in descending order
           (usplat.evaluation.average_precision, the scikit-learn definition): a block of tied
           scores enters at once, so AP does not depend on the order of tied elements.
  AUROC  = P(s_pos > s_neg) + 0.5 P(s_pos = s_neg) (mid-ranks; usplat.evaluation.auroc).
  NaN    = no score: ranked as the lowest score (one tie block at the bottom), and the
           coverage n_finite / n_judged is reported with every AP/AUROC.
  object level: object score = mean of the FINITE scores of its judged Gaussians (NaN if
           none, then ranked last); the object label is its status; objects without any
           judged Gaussian are left out and counted.
  common: AP/AUROC on the judged Gaussians that EVERY row of `common_rows` scores (finite),
           so signals are compared on one set (change_baselines.common_defined).
  operating point: flags (N,) bool = the label-free fired flags (ours: reset at least once
           during the update) or s > thr with thr set on VALIDATION (baselines); NaN never
           fires. Recall = mean flag on stale, FA = mean flag on static, Kept = 1 - FA, and
           static reset mass = sum_static m flag / sum_static m with m the update-pass splat
           mass (stale mass removed likewise).
  segmentation pass-through: pred_n = argmax_k f_n . t_k (no decision, i.e. wrong, for
           f_n = 0 or a tie at the top); accuracy on changed and on static Gaussians whose
           target class is known (target >= 0), unweighted and weighted by m.
Every row that a checkpoint cannot provide stays in the record as "missing: <reason>".
Residual rows (ncc, feature, depth) read ChangeBaselines state saved in the field checkpoint,
which must cover the UPDATE pass only (the field's c is likewise cleared at calibration): the
checkpoint holds ckpt["change_scores"] = {"<residual>/<filter>": (N,)} of the update pass
(what scripts/run_mapping.py --change-baselines writes: a second accumulator fed only the update
views, ChangeBaselines.push_view) or ckpt["change_baselines"] = a ChangeBaselines.state_dict()
of the update views only, AND ckpt["change_baselines_scope"] = "update". Without that marker
the rows stay missing: a state that also saw the mapping pass mixes mapping residuals into
kmin / max and is never read. The depth rows are depth_gauss / depth_render (non-oracle depth)
or, from a run with --oracle-depth (rendered mesh depth, e.g. Replica), depth_*_oracle, which
only the row depth_oracle reads. normlift_drop = R before the update - R after it, with R
before from ckpt["normlift_R_before_update"] (N,) (run_mapping.py --baselines) or the NormLift
of ckpt["baselines_before_update"].

A ground-truth file (torch.save of a dict, make_gt) holds: label (N,) int8, object_id (N,)
int64 (-1 none), object_ids / object_labels (O,), mass (N,) or None, target (N,) int64 or
None, classes {ids, names} and text_raw (K, D) for the segmentation block, shares, config.
check_criterion evaluates a pre-registered criterion (configs/*.json "criterion") over the
per-scene records: every condition must hold in a scene; enough scenes must pass.
"""
from __future__ import annotations
import hashlib
import json
import math
from typing import Dict

import numpy as np
import torch

from .evaluation import auroc, average_precision

STALE, STATIC, UNJUDGED = 1, 0, -1
GT_SCHEMA = "uags.change_gt/1"
# Gaussian category codes (exclusive, in this precedence)
CAT_JUDGED, CAT_UNOBSERVED, CAT_HIDDEN, CAT_FAR, CAT_OBJ = 0, 1, 2, 3, 4
CAT_NAMES = {CAT_JUDGED: "judged", CAT_UNOBSERVED: "unobserved", CAT_HIDDEN: "hidden",
             CAT_FAR: "far", CAT_OBJ: "object_unjudged"}


# ------------------------------------------------------------------------------ config
def canonical_sha256(obj) -> str:
    """sha256 of the sorted-key, no-whitespace JSON of obj (equal content, equal hash)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def load_config(path):
    """A frozen JSON config and the hash of its content."""
    with open(path) as f:
        cfg = json.load(f)
    return cfg, canonical_sha256(cfg)


def split_scenes(cfg, split):
    """The scene list of `split` in a frozen config ("scenes" or "splits"); the splits must
    be disjoint (a scene in two splits would let test data in through validation)."""
    sp = cfg.get("scenes") or cfg.get("splits") or {}
    if split not in sp:
        raise ValueError(f"config lists no scenes for split {split!r} (has {sorted(sp)})")
    seen = {}
    for k, lst in sp.items():
        for sc in lst:
            if sc in seen and seen[sc] != k:
                raise ValueError(f"scene {sc!r} is in splits {seen[sc]!r} and {k!r}")
            seen[sc] = k
    return [str(x) for x in sp[split]]


def rio10_thresholds(cfg):
    """Stale-GT thresholds of configs/rio10_change.json in metres / degrees."""
    o, g, v = cfg["object"], cfg["gaussian"], cfg["visibility"]
    out = dict(d_move=float(o["d_move_cm"]) / 100.0, r_deg=float(o["r_deg"]),
               min_points=int(o["min_points"]),
               ignore_ids=tuple(int(i) for i in o.get("ignore_instance_ids", [0])),
               d_near=float(g["d_near_cm"]) / 100.0,
               depth_margin=float(v["depth_margin_cm"]) / 100.0, near=float(v["near_m"]),
               min_mass=float(v.get("observed_min_mass_px", 0.0)))
    for k in ("d_move", "r_deg", "d_near", "depth_margin"):
        if not out[k] > 0:
            raise ValueError(f"rio10 config: {k} must be > 0, got {out[k]}")
    return out


# ------------------------------------------------------------------------------ geometry
def nearest_within(query, ref, radius, k_ties=8):
    """Nearest `ref` point of every `query` point within `radius` (inclusive).
    returns idx (M,) int64 (-1: none within radius), dist (M,) float64 (inf: none).
    Exact (cKDTree); an exact tie goes to the lowest ref index among the k_ties nearest."""
    from scipy.spatial import cKDTree
    q = np.asarray(query, dtype=np.float64).reshape(-1, 3)
    r = np.asarray(ref, dtype=np.float64).reshape(-1, 3)
    M = q.shape[0]
    if r.shape[0] == 0 or M == 0:
        return np.full(M, -1, np.int64), np.full(M, np.inf)
    k = min(int(k_ties), r.shape[0])
    d, i = cKDTree(r).query(q, k=k, distance_upper_bound=radius * (1 + 1e-9) + 1e-12)
    if k == 1:
        d, i = d[:, None], i[:, None]
    d = np.where(d <= radius, d, np.inf)
    dmin = d[:, 0]
    tie = np.isfinite(d) & (d == dmin[:, None])
    idx = np.where(tie, i, np.iinfo(np.int64).max).min(1).astype(np.int64)
    none = ~np.isfinite(dmin)
    idx[none] = -1
    return idx, np.where(none, np.inf, dmin)


def kabsch(A, B):
    """Rotation R minimising sum_i ||R a_i - b_i||^2 for centred rows a_i, b_i."""
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0])
    return Vt.T @ D @ U.T


def rotation_angle_deg(R):
    c = (np.trace(np.asarray(R)[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def icp_rotation(src, dst, iters=50, max_points=4000, seed=0, tol=1e-10):
    """Rotation of a point-to-point ICP between the CENTRED clouds src -> dst, from the
    identity (deterministic subsample of max_points points)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    A = np.asarray(src, np.float64)
    B = np.asarray(dst, np.float64)
    if A.shape[0] > max_points:
        A = A[np.sort(rng.choice(A.shape[0], max_points, replace=False))]
    if B.shape[0] > max_points:
        B = B[np.sort(rng.choice(B.shape[0], max_points, replace=False))]
    A, B = A - A.mean(0), B - B.mean(0)
    tree = cKDTree(B)
    R, prev = np.eye(3), np.inf
    for _ in range(int(iters)):
        d, j = tree.query(A @ R.T)
        R = kabsch(A, B[j])
        err = float(np.mean(d * d))
        if abs(prev - err) <= tol:
            break
        prev = err
    return R


def _by_object(points, ids):
    order = np.argsort(ids, kind="stable")
    u, start, cnt = np.unique(ids[order], return_index=True, return_counts=True)
    return {int(o): points[order[s:s + c]] for o, s, c in zip(u, start, cnt)}


def object_motion_from_points(ref_pts, ref_ids, res_pts, res_ids, min_points=50,
                              ignore_ids=(0,), icp_iters=50, max_points=4000, seed=0):
    """Per seqXX_01 object: {n_ref, n_res, removed, shift_m, rot_deg} from the two labelled
    vertex clouds (same world frame; V7/V8 of usplat.datasets_rio)."""
    A = _by_object(np.asarray(ref_pts, np.float64), np.asarray(ref_ids, np.int64))
    B = _by_object(np.asarray(res_pts, np.float64), np.asarray(res_ids, np.int64))
    out = {}
    for o, P in A.items():
        if o in ignore_ids:
            continue
        Q = B.get(o, np.zeros((0, 3)))
        m = dict(n_ref=int(P.shape[0]), n_res=int(Q.shape[0]), removed=False,
                 shift_m=float("nan"), rot_deg=float("nan"))
        if P.shape[0] >= min_points:
            if Q.shape[0] < min_points:
                m["removed"] = True
            else:
                m["shift_m"] = float(np.linalg.norm(Q.mean(0) - P.mean(0)))
                m["rot_deg"] = rotation_angle_deg(icp_rotation(P, Q, icp_iters, max_points,
                                                               seed))
        out[o] = m
    return out


def object_motion_from_transforms(changes, ref_pts, ref_ids, min_points=50, ignore_ids=(0,)):
    """3RScan route: changes = usplat.datasets_rio.load_3rscan_changes(...)."""
    A = _by_object(np.asarray(ref_pts, np.float64), np.asarray(ref_ids, np.int64))
    out = {}
    for o, P in A.items():
        if o in ignore_ids:
            continue
        m = dict(n_ref=int(P.shape[0]), n_res=None, removed=o in changes["removed"],
                 nonrigid=o in changes["nonrigid"], shift_m=0.0, rot_deg=0.0)
        if o in changes["rigid"]:
            T = np.asarray(changes["rigid"][o][1], np.float64)
            c = P.mean(0)
            m["shift_m"] = float(np.linalg.norm(T[:3, :3] @ c + T[:3, 3] - c))
            m["rot_deg"] = rotation_angle_deg(T)
        out[o] = m
    return out


def rscan_id_check(changes, ref_ids, ignore_ids=(0,), max_absent_share=0.0):
    """V10 of usplat.datasets_rio: the ids of the 3RScan removed / nonrigid / rigid lists
    must exist in the RIO10 labels.ply of the reference scan (same id space). Refuses when
    the share of listed ids absent from it exceeds max_absent_share; returns the counts."""
    present = set(np.unique(np.asarray(ref_ids, np.int64)).tolist()) - set(ignore_ids)
    listed = set(changes["removed"]) | set(changes["nonrigid"]) | set(changes["rigid"])
    absent = sorted(listed - present)
    out = dict(n_listed=len(listed), n_present=len(present), absent=absent,
               absent_share=len(absent) / len(listed) if listed else 0.0)
    if out["absent_share"] > max_absent_share:
        raise ValueError(f"3RScan lists {len(absent)} of {len(listed)} instance ids that "
                         f"labels.ply does not have ({absent[:10]}): the RIO10 and 3RScan ids "
                         f"are not one id space (V10), so objects would read static")
    return out


def select_keyframes(c2ws, kf_trans=0.05, kf_rot=5.0):
    """Indices of the frames scripts/run_mapping.py's select() feeds (update sequence: no
    max_keyframes): finite pose, and not a near-duplicate of the last kept frame (translation
    < kf_trans [m] AND rotation < kf_rot [deg]); _pose_gap's arithmetic on the frames' c2w."""
    last, keep = None, []
    for i, c2w in enumerate(c2ws):
        c2w = torch.as_tensor(c2w)
        if not torch.isfinite(c2w).all():
            continue
        if last is not None:
            dt = (c2w[:3, 3] - last[:3, 3]).norm().item()
            cos = ((c2w[:3, :3].t() @ last[:3, :3]).trace().item() - 1.0) / 2.0
            dr = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            if dt < kf_trans and dr < kf_rot:
                continue
        last = c2w
        keep.append(i)
    return keep


def object_status(motion, d_move, r_deg, min_points=50):
    """{object: (STALE | STATIC | UNJUDGED, reason)} from object_motion_*."""
    out = {}
    for o, m in motion.items():
        if m.get("nonrigid"):
            out[o] = (UNJUDGED, "nonrigid")
        elif m["n_ref"] < min_points:
            out[o] = (UNJUDGED, "few_points")
        elif m["removed"]:
            out[o] = (STALE, "removed")
        elif m["shift_m"] > d_move:
            out[o] = (STALE, "moved")
        elif m["rot_deg"] > r_deg:
            out[o] = (STALE, "rotated")
        else:
            out[o] = (STATIC, "static")
    return out


def gaussian_stale_labels(means, ref_pts, ref_ids, status, d_near, observed=None, hidden=None,
                          ignore_ids=()):
    """Per-Gaussian stale GT (see the module docstring).
    means (N, 3); ref_pts (P, 3), ref_ids (P,) of seqXX_01; status from object_status;
    observed / hidden (N,) bool (None: every Gaussian observed, none hidden); vertices with
    an id in ignore_ids (unlabelled) are not annotated points and are not searched.
    returns dict(label (N,) int8, object_id (N,) int64, category (N,) int8, dist (N,), shares)."""
    x = torch.as_tensor(means).double().cpu().numpy()
    N = x.shape[0]
    ids = np.asarray(ref_ids, np.int64)
    pts = np.asarray(ref_pts, np.float64).reshape(-1, 3)
    if len(ignore_ids):
        ann = ~np.isin(ids, list(ignore_ids))
        pts, ids = pts[ann], ids[ann]
    j, dist = nearest_within(x, pts, d_near)
    obj = np.where(j >= 0, ids[np.maximum(j, 0)], -1)
    code = status_codes(obj, status)
    obs = np.ones(N, bool) if observed is None else torch.as_tensor(observed).cpu().numpy().astype(bool)
    hid = np.zeros(N, bool) if hidden is None else torch.as_tensor(hidden).cpu().numpy().astype(bool)
    if obs.shape != (N,) or hid.shape != (N,):
        raise ValueError(f"observed/hidden must be (N,) = ({N},)")
    cat = np.full(N, CAT_JUDGED, np.int64)
    cat[code == UNJUDGED] = CAT_OBJ
    cat[j < 0] = CAT_FAR
    cat[hid & ~obs] = CAT_HIDDEN              # seen in some view's image, never visible
    cat[~obs & ~hid] = CAT_UNOBSERVED         # an observed Gaussian is never hidden
    label = np.where(cat == CAT_JUDGED, code, UNJUDGED)
    shares = category_shares(cat, label)
    return dict(label=torch.from_numpy(label.astype(np.int8)),
                object_id=torch.from_numpy(np.where(cat == CAT_JUDGED, obj, -1)),
                nearest_object=torch.from_numpy(obj), category=torch.from_numpy(cat.astype(np.int8)),
                dist=torch.from_numpy(dist), shares=shares)


def status_codes(obj, status):
    """STALE / STATIC / UNJUDGED per object id in `obj` (-1 or unknown ids: UNJUDGED)."""
    obj = np.asarray(obj, np.int64)
    if not status:
        return np.full(obj.shape, UNJUDGED, np.int64)
    keys = np.array(sorted(status), np.int64)
    vals = np.array([status[int(k)][0] for k in keys], np.int64)
    pos = np.clip(np.searchsorted(keys, obj), 0, keys.size - 1)
    hit = (keys[pos] == obj) & (obj >= 0)
    return np.where(hit, vals[pos], UNJUDGED)


def rescan_targets(means, label, object_id, ref_class, res_pts, res_ids, res_class, d_near):
    """Target class (index, -1 unknown) per judged Gaussian for the segmentation block: a
    static Gaussian keeps the class of its own object (ref_class[object]); a stale one takes
    the class of the nearest seqXX_02 vertex within d_near, i.e. what its location shows
    now (unknown if none). ref_class / res_class: {instance id: class index} (class map)."""
    lab = torch.as_tensor(label).reshape(-1).cpu().numpy()
    oid = torch.as_tensor(object_id).reshape(-1).cpu().numpy()
    out = np.full(lab.shape, -1, np.int64)
    look = lambda d, ids: np.array([d.get(int(i), -1) for i in ids], np.int64)
    st = lab == STATIC
    out[st] = look(ref_class, oid[st])
    ch = np.nonzero(lab == STALE)[0]
    if ch.size:
        x = torch.as_tensor(means).double().cpu().numpy()[ch]
        j, _ = nearest_within(x, res_pts, d_near)
        rid = np.where(j >= 0, np.asarray(res_ids, np.int64)[np.maximum(j, 0)], -1)
        out[ch] = np.where(j >= 0, look(res_class, rid), -1)
    return torch.from_numpy(out)


def category_shares(cat, label):
    cat = np.asarray(cat)
    label = np.asarray(label)
    N = max(int(cat.size), 1)
    out = {f"n_{CAT_NAMES[c]}": int((cat == c).sum()) for c in CAT_NAMES}
    out.update({f"share_{CAT_NAMES[c]}": float((cat == c).sum()) / N for c in CAT_NAMES})
    out.update(n=int(cat.size), n_stale=int((label == STALE).sum()),
               n_static=int((label == STATIC).sum()))
    return out


@torch.no_grad()
def visibility_from_depth(means, frames, margin=0.05, near=0.05):
    """Observed / hidden masks of the Gaussian centres over the update views.
    frames: iterable of dicts with c2w (4, 4), K (3, 3) of the colour image, depth (Hd, Wd)
    [m] (the rescan's surface; 0 = none) and image (H, W, 3) or hw = (H, W) for K's grid.
    Per view a centre is out (behind the camera or outside the image), visible (z <= d +
    margin), hidden (z > d + margin) or no-depth (in the image, d = 0).
    returns dict(observed: visible in >= 1 view, hidden: in some view's image, never visible,
    behind the surface in >= 1 view, no_depth: in some image, never visible nor hidden,
    in_frustum: in >= 1 view's image, n_views (N,) visible-view counts, n_frames)."""
    X = torch.as_tensor(means, dtype=torch.float64)
    N = X.shape[0]
    vis_n = torch.zeros(N, dtype=torch.long)
    inside_any = torch.zeros(N, dtype=torch.bool)
    hid_any = torch.zeros(N, dtype=torch.bool)
    nf = 0
    for fr in frames:
        c2w = torch.as_tensor(fr["c2w"], dtype=torch.float64)
        if not torch.isfinite(c2w).all():
            continue
        nf += 1
        w2c = torch.linalg.inv(c2w)
        Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
        z = Xc[:, 2]
        K = torch.as_tensor(fr["K"], dtype=torch.float64)
        zc = z.clamp_min(1e-9)
        u = K[0, 0] * Xc[:, 0] / zc + K[0, 2]
        v = K[1, 1] * Xc[:, 1] / zc + K[1, 2]
        D = torch.as_tensor(fr["depth"], dtype=torch.float64)
        Hd, Wd = D.shape
        if fr.get("hw") is not None:
            H, W = (int(s) for s in fr["hw"])
        elif fr.get("image") is not None:
            H, W = int(fr["image"].shape[0]), int(fr["image"].shape[1])
        else:
            H, W = Hd, Wd
        inside = (z > near) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ud = (u * Wd / W).floor().long().clamp(0, Wd - 1)
        vd = (v * Hd / H).floor().long().clamp(0, Hd - 1)
        d = torch.where(inside, D[vd, ud], torch.zeros_like(z))
        has = inside & (d > 0)
        vis_n += (has & (z <= d + margin)).long()
        hid_any |= has & (z > d + margin)
        inside_any |= inside
    observed = vis_n > 0
    hidden = inside_any & ~observed & hid_any
    return dict(observed=observed, hidden=hidden, no_depth=inside_any & ~observed & ~hid_any,
                in_frustum=inside_any, n_views=vis_n, n_frames=nf)


def alignment_check(ref_pts, ref_ids, res_pts, res_ids, static_ids, max_points=200000, seed=0):
    """Surface distance of seqXX_01 static-structure vertices (instance ids `static_ids`,
    e.g. wall/floor/ceiling) to the seqXX_02 vertices of the same ids: median and p95 [m].
    The plan sets d_move above this p95 (configs/rio10_change.json alignment)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    a = np.isin(np.asarray(ref_ids), list(static_ids))
    b = np.isin(np.asarray(res_ids), list(static_ids))
    P, Q = np.asarray(ref_pts)[a], np.asarray(res_pts)[b]
    if P.shape[0] == 0 or Q.shape[0] == 0:
        raise ValueError(f"alignment check: no static-structure vertices (ids {sorted(static_ids)}) "
                         f"in one of the scans ({P.shape[0]}, {Q.shape[0]})")
    if P.shape[0] > max_points:
        P = P[rng.choice(P.shape[0], max_points, replace=False)]
    d, _ = cKDTree(Q).query(P)
    return dict(median_m=float(np.median(d)), p95_m=float(np.percentile(d, 95)),
                n_points=int(P.shape[0]))


def check_alignment(p95_m, d_move):
    if not d_move > p95_m:
        raise ValueError(f"d_move {d_move * 100:.1f} cm is not above the alignment p95 "
                         f"{p95_m * 100:.1f} cm: static objects would read as moved (plan §3)")


# ------------------------------------------------------------------------------ GT file
def make_gt(kind, scene, label, object_id, object_labels, mass=None, target=None,
            classes=None, text_raw=None, shares=None, config=None, extra=None):
    """The ground-truth dict (torch.save it). object_labels: {object id: STALE | STATIC}."""
    label = torch.as_tensor(label).to(torch.int8).cpu()
    N = label.numel()
    if not bool(((label >= -1) & (label <= 1)).all()):
        raise ValueError("label must be in {-1, 0, 1}")
    object_id = torch.as_tensor(object_id).long().cpu()
    if object_id.numel() != N:
        raise ValueError(f"object_id has {object_id.numel()} entries, expected {N}")
    oid = torch.tensor(sorted(int(k) for k in object_labels), dtype=torch.long)
    olab = torch.tensor([int(object_labels[int(k)]) for k in oid.tolist()], dtype=torch.int8)
    for name, x in (("mass", mass), ("target", target)):
        if x is not None and torch.as_tensor(x).numel() != N:
            raise ValueError(f"{name} has {torch.as_tensor(x).numel()} entries, expected {N}")
    if target is not None and classes is None:
        raise ValueError("target needs classes")
    return dict(schema=GT_SCHEMA, kind=kind, scene=scene, N=N, label=label, object_id=object_id,
                object_ids=oid, object_labels=olab,
                mass=None if mass is None else torch.as_tensor(mass).double().cpu(),
                target=None if target is None else torch.as_tensor(target).long().cpu(),
                classes=classes, text_raw=None if text_raw is None else
                torch.as_tensor(text_raw).float().cpu(),
                shares=shares or category_shares(np.zeros(N), label.numpy()),
                config=config, config_sha256=None if config is None else canonical_sha256(config),
                **(extra or {}))


def load_gt(path):
    gt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(gt, dict) or gt.get("schema") != GT_SCHEMA:
        raise ValueError(f"{path}: not a change ground-truth file ({GT_SCHEMA})")
    return gt


# ------------------------------------------------------------------------------ metrics
def _f64(x):
    return torch.as_tensor(x, dtype=torch.float64).reshape(-1).cpu()


def ranking_metrics(score, label):
    """AP / AUROC of `score` against 0/1 `label` (already the population); NaN ranks last."""
    s, y = _f64(score), _f64(label)
    if s.numel() != y.numel():
        raise ValueError(f"{s.numel()} scores for {y.numel()} labels")
    n_nan = int((~torch.isfinite(s)).sum())
    n = int(s.numel())
    return dict(ap=average_precision(s, y) if n else float("nan"),
                auroc=auroc(s, y) if n else float("nan"),
                n=n, n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()), n_nan=n_nan,
                coverage=(n - n_nan) / n if n else float("nan"))


def object_ranking(score, label, object_id, object_ids, object_labels):
    """Object-level AP / AUROC: object score = mean of the finite scores of its judged
    Gaussians (NaN if none); objects with no judged Gaussian are left out (n_dropped)."""
    s, lab = _f64(score), torch.as_tensor(label).reshape(-1).cpu()
    oid = torch.as_tensor(object_id).long().reshape(-1).cpu()
    keep = (lab >= 0) & (oid >= 0)
    known = {int(o): int(y) for o, y in zip(torch.as_tensor(object_ids).tolist(),
                                            torch.as_tensor(object_labels).tolist())}
    objs = sorted(set(oid[keep].tolist()))
    unknown = [o for o in objs if o not in known]
    if unknown:
        raise ValueError(f"objects {unknown[:5]} have judged Gaussians but no object label")
    objs = [o for o in objs if known[o] in (STALE, STATIC)]
    n_lab = sum(1 for y in known.values() if y in (STALE, STATIC))
    if not objs:
        return dict(ranking_metrics([], []), n_objects=0, n_dropped=n_lab)
    ot = torch.tensor(objs, dtype=torch.long)
    pos = torch.searchsorted(ot, oid).clamp(max=len(objs) - 1)
    sel = keep & (ot[pos] == oid)
    pos = pos[sel]
    sv = s[sel]
    fin = torch.isfinite(sv)
    tot = torch.zeros(len(objs), dtype=torch.float64).index_add_(0, pos[fin], sv[fin])
    cnt = torch.zeros(len(objs), dtype=torch.float64).index_add_(0, pos[fin],
                                                                  torch.ones_like(sv[fin]))
    so = torch.where(cnt > 0, tot / cnt.clamp_min(1), torch.full_like(tot, float("nan")))
    yo = torch.tensor([known[o] for o in objs], dtype=torch.float64)
    return dict(ranking_metrics(so, yo), n_objects=len(objs), n_dropped=n_lab - len(objs))


def operating_point(flags, label, mass=None):
    """Recall / FA / Kept of boolean flags on the judged Gaussians (+ reset masses)."""
    f = torch.as_tensor(flags).reshape(-1).cpu().bool()
    lab = torch.as_tensor(label).reshape(-1).cpu()
    st, ch = lab == STATIC, lab == STALE
    mean = lambda m: float(f[m].double().mean()) if bool(m.any()) else float("nan")
    out = dict(recall=mean(ch), fa=mean(st), n_fired=int(f[st | ch].sum()))
    out["kept"] = 1.0 - out["fa"] if math.isfinite(out["fa"]) else float("nan")
    if mass is not None:
        m = _f64(mass)
        wsum = lambda sel: float(m[sel].sum())
        out["static_reset_mass"] = (float((m * f.double())[st].sum()) / wsum(st)
                                    if wsum(st) > 0 else float("nan"))
        out["stale_reset_mass"] = (float((m * f.double())[ch].sum()) / wsum(ch)
                                   if wsum(ch) > 0 else float("nan"))
    return out


def flags_at(score, thr):
    """score > thr; NaN never fires."""
    s = _f64(score)
    return torch.isfinite(s) & (s > float(thr))


def fa_threshold(static_scores, target_fa):
    """Smallest t with share(s > t) <= target_fa over the static scores (NaN never fires):
    t = the (m - a)-th smallest of the m finite scores, a = floor(target_fa n) the number
    allowed to exceed it (-inf when a >= m). Gaussian-level: per-view residuals are not
    stored in the checkpoint."""
    s = _f64(static_scores)
    fin = torch.sort(s[torch.isfinite(s)]).values
    if s.numel() == 0:
        return float("inf")
    allowed = int(math.floor(float(target_fa) * s.numel() + 1e-9))
    if allowed >= fin.numel():
        return float("-inf")
    return float(fin[fin.numel() - 1 - allowed])


@torch.no_grad()
def classify(feature, text):
    """pred (N,) = argmax_k f . t_k; -1 (no decision) where f = 0 / non-finite or tied top."""
    f = torch.as_tensor(feature).double()
    t = torch.as_tensor(text).double().to(f.device)
    s = f @ t.T
    pred = s.argmax(1)
    bad = ~torch.isfinite(f).all(1) | (f.norm(dim=1) == 0)
    if t.shape[0] >= 2:
        top = s.topk(2, dim=1).values
        bad |= top[:, 0] == top[:, 1]
    return torch.where(bad, torch.full_like(pred, -1), pred).cpu()


def region_accuracy(pred, target, label, mass=None):
    """Accuracy on changed (label 1) and static (label 0) Gaussians with target >= 0."""
    p = torch.as_tensor(pred).long().reshape(-1).cpu()
    t = torch.as_tensor(target).long().reshape(-1).cpu()
    lab = torch.as_tensor(label).reshape(-1).cpu()
    out = {}
    for name, code in (("changed", STALE), ("static", STATIC)):
        m = (lab == code) & (t >= 0)
        ok = (p == t) & m
        out[f"n_{name}"] = int(m.sum())
        out[f"acc_{name}"] = float(ok.sum()) / int(m.sum()) if bool(m.any()) else float("nan")
        if mass is not None:
            w = _f64(mass)
            den = float(w[m].sum())
            out[f"acc_{name}_mass"] = float(w[ok].sum()) / den if den > 0 else float("nan")
    return out


# ------------------------------------------------------------------------------ rows
RESIDUAL_ROWS = (                      # (row, ChangeBaselines residual names, input, oracle)
    ("ncc", ("ncc",), "I", False),
    ("feature_residual", ("feat",), "F", False),
    ("depth_residual", ("depth_gauss",), "D", False),
    ("depth_render_residual", ("depth_render",), "D", False),
    ("depth_oracle", ("depth_render_oracle", "depth_gauss_oracle"), "M", True),
)
CB_KEYS = ("change_baselines",)        # ChangeBaselines.state_dict() in the field checkpoint
CB_SCORES_KEY = "change_scores"        # or {"<residual>/<filter>": (N,)} directly
CB_SCOPE_KEY, CB_SCOPE = "change_baselines_scope", "update"   # both: update pass only
NL_BEFORE_KEY = "normlift_R_before_update"   # NormLift R (N,) at the end of the mapping pass


def _row(score=None, fired=None, input="", oracle=False, status="ok", source=""):
    return dict(score=score, fired=fired, input=input, oracle=oracle, status=status,
                source=source)


@torch.no_grad()
def field_change_rows(field):
    """Rows read from a DirectionalSemanticField after the update pass. c is defined during
    the update where c_last is finite (calibrate_change_threshold clears both); c_max of a
    Gaussian never scored is NaN, not its initial 0. fired = reset at least once (the
    mapping pass runs with change_thr = inf, so every reset is an update reset)."""
    c_last = field.c_last.detach().double().cpu()
    defined = torch.isfinite(c_last)
    c_max = torch.where(defined, field.c_max.detach().double().cpu(),
                        torch.full_like(c_last, float("nan")))
    fired = (field.n_resets >= 1).cpu()
    ale, _ = field.aleatoric()
    W1 = field.raw.W1.detach().double().cpu()
    rbar = field.mean_resultant_length("raw").detach().double().cpu()
    one_m = torch.where(W1 > 0, 1.0 - rbar, torch.full_like(rbar, float("nan")))
    src = "field"
    return {"ours_c_max": _row(c_max, fired, "F+D", source=src),
            "ours_c_last": _row(c_last, fired, "F+D", source=src),
            "ours_fired": _row(field.n_resets.double().cpu(), fired, "F+D", source=src),
            "u_ale": _row(ale.detach().double().cpu(), None, "F", source=src),
            "1-rbar": _row(one_m, None, "F", source=src)}


def _cb_scores(ckpt, cb_key=None):
    """({"<residual>/<filter>": (N,)}, None) from the checkpoint, or (None, why not)."""
    have = (CB_SCORES_KEY in ckpt and ckpt[CB_SCORES_KEY] is not None) or \
        any(ckpt.get(k) is not None for k in ([cb_key] if cb_key else CB_KEYS))
    if not have:
        return None, "no ChangeBaselines in the checkpoint"
    if ckpt.get(CB_SCOPE_KEY) != CB_SCOPE:              # mapping residuals would mix in
        return None, (f"ChangeBaselines state not marked update-pass only "
                      f"(ckpt[{CB_SCOPE_KEY!r}] = {ckpt.get(CB_SCOPE_KEY)!r}, need {CB_SCOPE!r})")
    if CB_SCORES_KEY in ckpt and ckpt[CB_SCORES_KEY] is not None:
        return {k: torch.as_tensor(v).double() for k, v in ckpt[CB_SCORES_KEY].items()}, None
    from .change_baselines import ChangeBaselines
    k = next(k for k in ([cb_key] if cb_key else CB_KEYS) if ckpt.get(k) is not None)
    cb = ChangeBaselines.from_state(ckpt[k], device="cpu")
    return {n: v.double() for n, v in cb.scores().items()}, None


def checkpoint_rows(ckpt, field, filt="kmin", cb_key=None):
    """Every Table II row this checkpoint can provide (missing ones say why)."""
    rows = field_change_rows(field)
    cb, why = _cb_scores(ckpt, cb_key)
    for name, residuals, inp, oracle in RESIDUAL_ROWS:
        if cb is None:
            rows[name] = _row(input=inp, oracle=oracle, status=f"missing: {why}")
            continue
        key = next((f"{r}/{filt}" for r in residuals if f"{r}/{filt}" in cb), None)
        rows[name] = (_row(cb[key], None, inp, oracle, source=f"change_baselines:{key}")
                      if key else _row(input=inp, oracle=oracle,
                                       status=f"missing: no residual {residuals} with filter {filt}"))
    css = nl_after = None
    if ckpt.get("baselines"):
        from .baselines import load_baselines, CSS, NormLift
        bs = load_baselines(ckpt["baselines"])
        css = next((b for b in bs if isinstance(b, CSS)), None)
        nl_after = next((b for b in bs if isinstance(b, NormLift)), None)
    if css is not None:
        u = css.uncertainties()
        rows["css_var"] = _row(u["css_var_top"].double(), None, "F+V", source="baselines:CSS")
        rows["yager_conflict"] = (_row(u["yager_conflict"].double(), None, "F+V",
                                       source="baselines:CSS") if "yager_conflict" in u else
                                  _row(input="F+V", status="missing: CSS without Yager"))
    else:
        rows["css_var"] = _row(input="F+V", status="missing: no CSS baseline")
        rows["yager_conflict"] = _row(input="F+V", status="missing: no CSS baseline")
    before = ckpt.get("baselines_before_update")
    r_before = ckpt.get(NL_BEFORE_KEY)
    if nl_after is not None and r_before is not None:
        r_before = torch.as_tensor(r_before).double().reshape(-1)
        rows["normlift_drop"] = (_row(r_before - nl_after.reliability().double(), None, "F",
                                      source=f"baselines:NormLift R drop ({NL_BEFORE_KEY})")
                                 if r_before.numel() == nl_after.N else
                                 _row(input="F", status=f"missing: {NL_BEFORE_KEY} has "
                                                        f"{r_before.numel()} entries, NormLift "
                                                        f"{nl_after.N}"))
    elif nl_after is not None and before:
        from .baselines import load_baselines, NormLift
        nl_before = next((b for b in load_baselines(before) if isinstance(b, NormLift)), None)
        rows["normlift_drop"] = (_row((nl_before.reliability() - nl_after.reliability()).double(),
                                      None, "F", source="baselines:NormLift R drop")
                                 if nl_before is not None else
                                 _row(input="F", status="missing: no NormLift before the update"))
    else:
        rows["normlift_drop"] = _row(input="F", status=f"missing: no NormLift state before the "
                                                        f"update (ckpt[{NL_BEFORE_KEY!r}])")
    return rows


def add_fused_row(rows, gt_label, names=("ours_c_max", "depth_residual"), name="fused"):
    """Pre-registered rank average (change_baselines.fuse_rank) on the judged population."""
    from .change_baselines import fuse_rank
    miss = [n for n in names if n not in rows or rows[n]["status"] != "ok"]
    if miss:
        rows[name] = _row(input="F+D", status=f"missing: inputs {miss}")
        return rows
    mask = torch.as_tensor(gt_label).reshape(-1) >= 0
    rows[name] = _row(fuse_rank([rows[n]["score"] for n in names], mask=mask), None, "F+D",
                      source=f"fuse_rank{tuple(names)}")
    return rows


def add_oracle_gt_row(rows, gt_label):
    """Sanity row: the ground truth itself as the score (AP = AUROC = 1 on any population)."""
    lab = torch.as_tensor(gt_label).double().reshape(-1)
    rows["oracle_gt"] = _row(torch.where(lab >= 0, lab, torch.full_like(lab, float("nan"))),
                             None, "GT", oracle=True, source="ground truth")
    return rows


def evaluate_rows(rows, gt, thresholds=None, common_rows=()):
    """{row: {status, input, oracle, gaussian, object, common, op}} on the judged population."""
    lab = torch.as_tensor(gt["label"]).reshape(-1).long()
    N = lab.numel()
    judged = lab >= 0
    thresholds = thresholds or {}
    common = judged.clone()
    used = [r for r in common_rows if r in rows and rows[r]["status"] == "ok"]
    for r in used:
        common &= torch.isfinite(_f64(rows[r]["score"]))
    out = {}
    for name, r in rows.items():
        e = dict(status=r["status"], input=r["input"], oracle=bool(r["oracle"]), source=r["source"])
        if r["status"] != "ok":
            out[name] = e
            continue
        s = _f64(r["score"])
        if s.numel() != N:
            raise ValueError(f"row {name}: {s.numel()} scores for {N} Gaussians")
        e["gaussian"] = ranking_metrics(s[judged], lab[judged])
        e["object"] = object_ranking(s, lab, gt["object_id"], gt["object_ids"], gt["object_labels"])
        e["common"] = dict(ranking_metrics(s[common], lab[common]), rows=used)
        if r["fired"] is not None:
            e["op"] = dict(operating_point(r["fired"], lab, gt.get("mass")), source="label_free")
        elif name in thresholds:
            e["op"] = dict(operating_point(flags_at(s, thresholds[name]), lab, gt.get("mass")),
                           source="validation_threshold", threshold=float(thresholds[name]))
        out[name] = e
    return out


# ------------------------------------------------------------------------------ criterion
def get_path(d, path):
    """d["a"]["b"]... for "a.b..." (None when a key is missing)."""
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


_OPS = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b}


def check_criterion(records: Dict[str, dict], criterion):
    """records: {scene: metrics of its eval record}; criterion: {scenes: [...] (the split's
    scene list), min_pass, conditions: [{left: path, op, right: path | number, offset}]}.
    A scene passes iff every condition holds (NaN or a missing value fails it). The
    criterion passes iff the records are exactly the listed scenes and >= min_pass pass."""
    want = list(criterion["scenes"])
    got = sorted(records)
    if sorted(want) != got:
        raise ValueError(f"criterion needs records of scenes {sorted(want)}, got {got}")
    per = {}
    for sc in want:
        m = records[sc]
        conds = []
        for c in criterion["conditions"]:
            a = _num(get_path(m, c["left"]))
            b = c["right"]
            b = _num(b) if isinstance(b, (int, float)) else _num(get_path(m, b))
            b = b + float(c.get("offset", 0.0))
            ok = bool(math.isfinite(a) and math.isfinite(b) and _OPS[c["op"]](a, b))
            conds.append(dict(c, left_value=a, right_value=b, ok=ok))
        per[sc] = dict(ok=all(x["ok"] for x in conds), conditions=conds)
    n_pass = sum(v["ok"] for v in per.values())
    return dict(passed=n_pass >= int(criterion["min_pass"]), n_pass=n_pass,
                n_scenes=len(want), min_pass=int(criterion["min_pass"]), scenes=per)
