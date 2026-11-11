"""
sam_lift.py — the SAM-mask baselines LangSplat and OpenGaussian, re-implemented in our protocol,
as rows next to Feature 3DGS and LEGaussians (usplat/distill.py): the scene's fixed 3D-GS map
(geometry frozen), the main field's mapping frames, the LSeg features in the field's PCA basis
(d = 64, unit vectors), the splat-weight renderer and scripts/eval_seg.py's evaluation. What the
two methods add is SAM. Their CLIP feature per mask is replaced by the unit mean of the PCA-64
LSeg features inside the mask, so every row lifts the same 2D features.

SAM masks (LangSplat's preprocessing; scripts/sam_masks.py runs it): a 32 x 32 point grid with
multimask output. The three outputs of a point are three granularities l in LEVELS = s, p, w
(output 0, 1, 2: subpart, part, whole), each filtered on its own as SamAutomaticMaskGenerator
filters (predicted IoU > 0.88, stability >= 0.95, box NMS at IoU 0.7, SAM's clean-up of regions
and holes under 100 px, then masks under 100 px dropped). Each level is painted into one int16
label map at the field's grid (downscaled as the field: area average, kept where >= 1/2),
larger masks first so that smaller masks win overlaps; -1 = no mask (paint_labels).
Mask targets: mask j of a view gets zbar_j = normalize(sum_{p in j} z_p) (z: the view's unit
PCA-64 LSeg features; the painted pixels of j); a pixel in mask j has target zbar_j, a pixel in
no mask has none and is left out of every loss (mask_means, mask_targets).

(h) LangSplat (Qin et al., CVPR 2024, arXiv 2312.16084). A scene autoencoder (its
    autoencoder/model.py, scaled to 64-d): encoder E = Linear 64-32, then [BatchNorm, ReLU,
    Linear] to 16, 8, 3, output L2-normalized; decoder D = Linear 3-8, then [ReLU, Linear] to
    16, 32, 64, output L2-normalized. It is trained on the mask features of all three levels
    of the scene with its loss (autoencoder/train.py)
        L_AE = mean (D(E(x)) - x)^2 + 0.001 (1 - cos(D(E(x)), x)),
    Adam lr 1e-4, 100 epochs of shuffled batches of 64, keeping the epoch of the last 5 with
    the lowest full-data loss mean (D(E(x)) - x)^2 + (1 - cos) (its best_ckpt rule).
    One model per level l: a per-Gaussian latent f_n in R^3, initialized to 0, trained so that
    the rendered latent matches the encoded target,
        L = mean_{p: j(p) >= 0, alpha_p >= alpha_min} |F_p - E(zbar_{j(p)})|_1,   F_p = sum_n w_{p,n} f_n
    (mean over pixels and the 3 channels), Adam lr 0.0025 (its language-feature lr), at most
    30k iterations (its count) and distill.converged's stopping rule. Rows "langsplat_s",
    "langsplat_p", "langsplat_w": the latent rendered, then decoded, D(F_p) (its query rule;
    a render-then-decode row, usplat.rows.Row.decoder). No uncertainty of its own.
(i) OpenGaussian (Wu et al., NeurIPS 2024, arXiv 2406.02058), on the whole-level masks.
    (a) Instance features: theta_n in R^6, rendered as f_n = theta_n / |theta_n| (its
    get_ins_feat), initialized U[0, 1)^6 (its create_from_pcd), F_p = sum_n w_{p,n} f_n, trained
    with its loss L = L_c + 0.1 L_s (its train.py, loss_weight 0.1; its Eq. 1-2 as its code
    computes them), over the masks i of the view with >= 50 covered pixels (its filter_th),
    Mbar_i = mean_{p in i, covered} F_p:
        L_s = mean_i mean_{p in i, covered} |F_p - Mbar_i|_2                  (intra-mask smoothing)
        L_c = 1 / (m (m - 1)) sum_i sum_{j != i} 1 / (|Mbar_i - Mbar_j|^2 + 1)   (inter-mask contrast)
    Adam lr 0.001 (its ins_feat_lr), at most 30k iterations and the stopping rule as above.
    (b) Two-level codebook (its ScanNet setting, paper Appendix A.1: k1 = 64, k2 = 5, position
    weight 1.0): k-means (10 Lloyd rounds, random points as initial centres, its
    kmeans_quantize.py) on [f_n, 1.0 xyz_n] into k1 coarse clusters, then k-means on f_n into
    k2 fine clusters inside each coarse one; Gaussian n belongs to instance c k2 + f (K = 320).
    (c) 2D-3D association (its Sec. 3.3, Eq. 5): per mapping view, instance i's mask I_i holds
    the covered pixels where i's rendered share is >= 1/2 (sum_{n in i} w_{p,n} >= alpha_p / 2,
    the full map's splat weights, so occlusion counts). It is matched to the whole-level mask
    j that maximizes
        S_ij = IoU(I_i, B_j) (1 - d_ij),   d_ij = mean_k |c_ik - p_jk|,
    c_i the instance's fine centre, p_j the mean of F_p / alpha_p over the covered pixels of
    B_j. A match needs S_ij > 0.2 (its b_matched). The instance's feature is
    normalize(sum_v IoU_v zbar_v) over its matched views; an instance never matched keeps the
    zero vector (its pixels get a uniform predictive and a tied top score, as uncovered ones).
    Row "opengs": the per-Gaussian feature is its instance's feature, rendered with the splat
    weights and classified as every feature row. No uncertainty of its own.

Deviations from the papers (shared protocol first):
  * both: geometry is our fixed map (both papers also freeze a pre-trained 3D-GS); the mask
    feature is the unit mean of the PCA-64 LSeg features over the mask's painted pixels, not
    CLIP on the masked, cropped and padded 224 x 224 image; losses only on covered pixels
    (alpha_p >= 0.5); the evaluator's text cosine -> softmax with a validated temperature
    replaces their relevancy scores; the stopping rule can end training before the cap.
  * SAM (both): thresholds 0.88 / 0.95, one crop, no extra mask NMS. LangSplat's preprocess.py
    uses pred_iou 0.7, stability 0.85, crop_n_layers 1 and its masks_update NMS (IoU 0.8,
    score 0.7, inner 0.5).
  * LangSplat: autoencoder widths scaled to 64-d (its 512-256-128-64-32-3 / 3-16-32-64-128-256-
    256-512); the cosine term per feature vector (its code takes it over the batch, dim 0);
    one level per row, chosen on validation by mIoU (it picks, per query, the level with the
    highest relevancy).
  * OpenGaussian: SAM's whole level (its paper: the 'large' level; its ScanNet script reads
    level 0, all masks); L_c without its code's rank-based pair weights (0.1-1, reset after
    35k iterations); stage 1 capped at 30k iterations (its ScanNet setting: 20k); the codebook
    is plain k-means on the trained features (no stage 2 with its straight-through
    pseudo-feature losses), on the unit f_n (its code clusters the raw theta_n); instance
    masks from the full map's splat weights (its code renders each cluster alone and keeps
    alpha > 0.8); the instance feature is the IoU-weighted mean of its matches (its code: the
    plain mean); the query renders the instance features (it ranks instances by cosine).

scripts/sam_masks.py writes the masks, scripts/train_sam_baselines.py trains the files next
to the main field (<main>_langsplat_{s,p,w}.pt, <main>_opengs.pt), row_factory() reads them
for scripts/eval_seg.py (--row-factory usplat.sam_lift:row_factory).
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as Fn

from . import distill as D
from .rows import TensorRow

FAMILY = "sam_lift"                     # the "family" key of every file written here
METHODS = ("langsplat", "opengs")
LEVELS = ("s", "p", "w")                # SAM multimask outputs 0, 1, 2: subpart, part, whole
WHOLE = LEVELS.index("w")

SAM_SETTINGS = dict(points_per_side=32, points_per_batch=64, pred_iou_thresh=0.88,
                    stability_score_thresh=0.95, stability_score_offset=1.0,
                    box_nms_thresh=0.7, min_mask_region_area=100, min_area=100)

AE_DIMS = (64, 32, 16, 8, 3)            # LangSplat's autoencoder, scaled to the 64-d features
AE_EPOCHS, AE_LR, AE_BATCH, AE_KEEP = 100, 1e-4, 64, 5
LANGSPLAT_LR = 0.0025                   # its language_feature_lr

OPENGS_DIM, OPENGS_LR = 6, 1e-3         # its ins_feat_dim, ins_feat_lr
OPENGS_SMOOTH, OPENGS_MIN_PX = 0.1, 50  # its loss_weight (L_s), filter_th
OPENGS_K1, OPENGS_K2, OPENGS_POS_W = 64, 5, 1.0   # ScanNet: root / leaf clusters, xyz weight
KMEANS_ITERS = 10
SHARE_MIN, MIN_SCORE = 0.5, 0.2         # instance mask: share of alpha; match: S_ij > 0.2


def _unit(x, eps=1e-12):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def settings_identity(obj):
    """sha256 of a JSON-plain object (sorted keys)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# --------------------------------------------------------------------------- SAM masks
@torch.no_grad()
def generate_levels(predictor, image, points_per_side=32, points_per_batch=64,
                    pred_iou_thresh=0.88, stability_score_thresh=0.95,
                    stability_score_offset=1.0, box_nms_thresh=0.7, min_mask_region_area=100,
                    min_area=100):
    """SamAutomaticMaskGenerator on one crop (the whole image) with its three multimask outputs
    kept apart: -> [masks_s, masks_p, masks_w], each a list of (H, W) bool arrays. Per level:
    predicted IoU > pred_iou_thresh, stability >= stability_score_thresh (SAM's comparisons),
    box NMS, SAM's postprocess_small_regions, then area >= min_area. predictor: a SamPredictor
    (or anything with its set_image / predict_torch / transform / model.mask_threshold)."""
    from segment_anything import SamAutomaticMaskGenerator
    from segment_anything.utils.amg import (MaskData, batch_iterator, batched_mask_to_box,
                                            build_point_grid, calculate_stability_score,
                                            mask_to_rle_pytorch, rle_to_mask)
    from torchvision.ops.boxes import batched_nms
    H, W = image.shape[:2]
    predictor.set_image(image)
    thr = predictor.model.mask_threshold
    pts = build_point_grid(points_per_side) * np.array([[W, H]])        # (x, y) pixels
    data = [MaskData() for _ in LEVELS]
    for (p,) in batch_iterator(points_per_batch, pts):
        tp = predictor.transform.apply_coords(p, (H, W))
        inp = torch.as_tensor(tp, device=predictor.device)
        lab = torch.ones(inp.shape[0], dtype=torch.int, device=inp.device)
        masks, iou, _ = predictor.predict_torch(inp[:, None, :], lab[:, None],
                                                multimask_output=True, return_logits=True)
        for lv in range(len(LEVELS)):
            d = MaskData(masks=masks[:, lv], iou_preds=iou[:, lv])
            if pred_iou_thresh > 0.0:
                d.filter(d["iou_preds"] > pred_iou_thresh)
            d["stability_score"] = calculate_stability_score(d["masks"], thr,
                                                             stability_score_offset)
            if stability_score_thresh > 0.0:
                d.filter(d["stability_score"] >= stability_score_thresh)
            d["masks"] = d["masks"] > thr
            d["boxes"] = batched_mask_to_box(d["masks"]).cpu()
            d["rles"] = mask_to_rle_pytorch(d["masks"])
            del d["masks"]
            d["iou_preds"] = d["iou_preds"].float().cpu()
            d["stability_score"] = d["stability_score"].float().cpu()
            data[lv].cat(d)
        del masks
    predictor.reset_image()
    out = []
    for d in data:
        if len(d["rles"]):
            keep = batched_nms(d["boxes"].float(), d["iou_preds"],
                               torch.zeros(len(d["rles"])), iou_threshold=box_nms_thresh)
            d.filter(keep)
        if min_mask_region_area > 0 and len(d["rles"]):
            d = SamAutomaticMaskGenerator.postprocess_small_regions(d, min_mask_region_area,
                                                                    box_nms_thresh)
        ms = [rle_to_mask(r) for r in d["rles"]]
        out.append([m.astype(bool) for m in ms if int(m.sum()) >= min_area])
    return out


def downscale_mask(mask, hw):
    """(H, W) bool -> (h, w) bool on the field's grid: the area average of
    usplat.datasets.downscale (cv2 INTER_AREA), kept where at least half the pixel is in."""
    h, w = int(hw[0]), int(hw[1])
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == (h, w):
        return mask.copy()
    import cv2
    return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= 0.5


def paint_labels(masks, hw):
    """masks: list of (H, W) bool -> (h, w) int16 label map on the field's grid. Masks are
    painted by decreasing full-resolution area (ties: list order), so a smaller mask wins
    where it overlaps a larger one; label k = the k-th painted mask; -1 = no mask."""
    if len(masks) > np.iinfo(np.int16).max:
        raise ValueError(f"{len(masks)} masks do not fit an int16 label map")
    area = np.array([int(np.asarray(m).sum()) for m in masks], dtype=np.int64)
    lab = np.full((int(hw[0]), int(hw[1])), -1, dtype=np.int16)
    for k, i in enumerate(np.argsort(-area, kind="stable")):
        lab[downscale_mask(masks[i], hw)] = k
    return lab


def save_masks(path, labels, index, part, meta):
    """Write a masks file: labels (F, L, h, w) int16, the frames' dataset index (F,) and part
    (F,) ('map' | 'update'), meta (JSON: provenance and settings, with meta['identity'])."""
    labels = np.asarray(labels, dtype=np.int16)
    if labels.ndim != 4 or labels.shape[1] != len(LEVELS):
        raise ValueError(f"labels must be (frames, {len(LEVELS)}, h, w), got {labels.shape}")
    index = np.asarray(index, dtype=np.int64)
    part = np.asarray([str(p) for p in part])
    if not (len(index) == len(part) == labels.shape[0]):
        raise ValueError("labels, index and part disagree on the number of frames")
    keys = list(zip(part.tolist(), index.tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError("a frame appears twice")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, labels=labels, index=index, part=part,
                            meta=np.array(json.dumps(meta, sort_keys=True, default=str)))
    os.replace(tmp, path)


def load_masks(paths):
    """One or more masks files (a list or a comma-separated string) -> dict(labels {(part,
    index): (L, h, w) int16}, metas [meta per file], identity: the settings identity they share
    (refused if they differ), files [path])."""
    if isinstance(paths, str):
        paths = [p for p in paths.split(",") if p]
    labels, metas = {}, []
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            for lab, i, pt in zip(z["labels"], z["index"].tolist(), z["part"].tolist()):
                if (pt, int(i)) in labels:
                    raise ValueError(f"{p}: frame {pt}:{i} is in another masks file too")
                labels[(pt, int(i))] = lab
        metas.append(meta)
    ids = {m.get("identity") for m in metas}
    if len(ids) != 1:
        raise ValueError(f"masks files of different SAM settings or code: {sorted(map(str, ids))}")
    return dict(labels=labels, metas=metas, identity=ids.pop(), files=[os.path.abspath(p) for p in paths])


# --------------------------------------------------------------------------- targets
def mask_means(labels, z):
    """labels (P,) mask ids (-1: none), z (P, d) unit features -> (compact (P,) long ids
    0..M-1 (-1: none), means (M, d) unit zbar_j over each mask's pixels)."""
    labels = torch.as_tensor(labels, device=z.device).long().reshape(-1)
    sel = labels >= 0
    ids, inv = torch.unique(labels[sel], return_inverse=True)
    compact = torch.full_like(labels, -1)
    compact[sel] = inv
    s = torch.zeros(len(ids), z.shape[1], dtype=z.dtype, device=z.device).index_add_(0, inv, z[sel])
    return compact, _unit(s)


def mask_targets(labels, z):
    """(P, d) per-pixel targets (zbar of the pixel's mask; 0 outside every mask) and the (P,)
    bool mask of pixels that have one."""
    compact, means = mask_means(labels, z)
    has = compact >= 0
    tgt = torch.zeros_like(z)
    tgt[has] = means[compact[has]]
    return tgt, has


# --------------------------------------------------------------------------- LangSplat
class LangSplatAE(torch.nn.Module):
    """LangSplat's scene autoencoder (autoencoder/model.py) with AE_DIMS widths: encode() and
    decode() L2-normalize their outputs, forward() = decode(encode(x))."""

    def __init__(self, dims=AE_DIMS):
        super().__init__()
        dims = tuple(int(x) for x in dims)
        enc = [torch.nn.Linear(dims[0], dims[1])]
        for a, b in zip(dims[1:-1], dims[2:]):
            enc += [torch.nn.BatchNorm1d(a), torch.nn.ReLU(), torch.nn.Linear(a, b)]
        rev = dims[::-1]
        dec = [torch.nn.Linear(rev[0], rev[1])]
        for a, b in zip(rev[1:-1], rev[2:]):
            dec += [torch.nn.ReLU(), torch.nn.Linear(a, b)]
        self.dims = dims
        self.encoder = torch.nn.Sequential(*enc)
        self.decoder = torch.nn.Sequential(*dec)

    def encode(self, x):
        return _unit(self.encoder(x))

    def decode(self, r):
        return _unit(self.decoder(r))

    def forward(self, x):
        return self.decode(self.encode(x))


class LatentDecoder(torch.nn.Module):
    """The LangSplat row's decoder: rendered latent F_p (P, 3) -> D(F_p) (P, 64), unit."""

    def __init__(self, ae: LangSplatAE):
        super().__init__()
        self.decoder = ae.decoder

    @torch.no_grad()
    def forward(self, r):
        w = self.decoder[0].weight
        return _unit(self.decoder(r.to(w.device, w.dtype)))


def ae_loss(y, x, cos_weight=0.001):
    """Its training loss: mean squared error + cos_weight (1 - cosine), per feature vector."""
    return ((y - x) ** 2).mean() + cos_weight * (1.0 - Fn.cosine_similarity(y, x, dim=-1).mean())


def train_autoencoder(x, dims=AE_DIMS, epochs=AE_EPOCHS, lr=AE_LR, batch=AE_BATCH,
                      keep_last=AE_KEEP, seed=0, device="cpu", log=print):
    """x (M, 64): the scene's unit mask features of every level. Adam(lr), `epochs` passes over
    shuffled batches of `batch` (a batch of one is skipped: BatchNorm needs two); after each
    of the last `keep_last` epochs the full-data loss mse + (1 - cos) is evaluated and the
    lowest is kept (its best_ckpt). Returns (autoencoder (cpu, eval), history [(epoch, loss)])."""
    g = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng(devices=[]):             # the initialization follows `seed`
        torch.manual_seed(seed)
        ae = LangSplatAE(dims)
    ae = ae.to(device)
    x = x.to(device).float()
    M = x.shape[0]
    if M < 2:
        raise ValueError(f"{M} mask features: too few to train the autoencoder")
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    best, best_state, hist = math.inf, None, []
    for ep in range(1, epochs + 1):
        ae.train()
        perm = torch.randperm(M, generator=g).to(device)
        for s in range(0, M, batch):
            b = x[perm[s:s + batch]]
            if b.shape[0] < 2:
                continue
            loss = ae_loss(ae(b), b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep > epochs - keep_last:
            ae.eval()
            with torch.no_grad():
                y = torch.cat([ae(x[s:s + 65536]) for s in range(0, M, 65536)])
                ev = float(((y - x) ** 2).mean() + (1.0 - Fn.cosine_similarity(y, x, dim=-1).mean()))
            hist.append((ep, ev))
            log(f"[langsplat ae] epoch {ep}/{epochs} eval loss {ev:.5f}")
            if ev < best:
                best, best_state = ev, copy.deepcopy(ae.state_dict())
    ae.load_state_dict(best_state)
    return ae.cpu().eval(), hist


@torch.no_grad()
def encode_means(ae, means, device="cpu"):
    """Encoded targets E(zbar) (M, 3) of one view's mask features (eval mode)."""
    ae = ae.to(device).eval()
    if means.shape[0] == 0:
        return torch.zeros(0, ae.dims[-1], device=device)
    return ae.encode(means.to(device).float())


# --------------------------------------------------------------------------- OpenGaussian
def opengs_terms(F, lab, cov, min_px=OPENGS_MIN_PX):
    """Its Eq. 1-2 on one view (as its code computes them). F (P, C) rendered features, lab
    (P,) compact mask ids (-1: none), cov (P,) covered pixels. Masks with fewer than min_px
    covered pixels are left out. -> (L_s, L_c, means (m, C)), or None if no mask is left."""
    lab = lab.long()
    sel = cov & (lab >= 0)
    if not bool(sel.any()):
        return None
    M = int(lab.max()) + 1
    cnt = torch.bincount(lab[sel], minlength=M)
    keep = cnt >= max(int(min_px), 1)
    if not bool(keep.any()):
        return None
    sel = sel & keep[lab.clamp_min(0)]
    ids, inv = torch.unique(lab[sel], return_inverse=True)
    m = ids.numel()
    Fs = F[sel]
    n = torch.bincount(inv, minlength=m).to(F.dtype)
    means = torch.zeros(m, F.shape[1], dtype=F.dtype, device=F.device).index_add(0, inv, Fs) / n[:, None]
    dist = (Fs - means[inv]).norm(dim=-1)                                       # |F_p - Mbar_i|
    ls = (torch.zeros(m, dtype=F.dtype, device=F.device).index_add(0, inv, dist) / n).mean()
    if m < 2:
        lc = F.new_zeros(())
    else:
        d2 = ((means[:, None, :] - means[None, :, :]) ** 2).sum(-1)
        inv_d = (1.0 / (d2 + 1.0)) * (1.0 - torch.eye(m, dtype=F.dtype, device=F.device))
        lc = inv_d.sum() / (m * (m - 1))
    return ls, lc, means


def opengs_loss(F, lab, cov, smooth_weight=OPENGS_SMOOTH, min_px=OPENGS_MIN_PX):
    """Its stage-1 loss L_c + smooth_weight L_s (None: no mask with enough covered pixels)."""
    t = opengs_terms(F, lab, cov, min_px)
    if t is None:
        return None
    ls, lc, _ = t
    return lc + smooth_weight * ls


def _nearest_centre(x, c, chunk=65536):
    return torch.cat([torch.cdist(x[s:s + chunk], c).argmin(1) for s in range(0, x.shape[0], chunk)])


def kmeans(x, k, iters=KMEANS_ITERS, seed=0):
    """Lloyd's k-means with the Euclidean distance (its kmeans_quantize.py): k distinct random
    points as initial centres, `iters` rounds of nearest-centre assignment and mean update
    (a centre that loses every point stays where it is), then a final assignment.
    k is capped at the number of points. -> (centres (k', D), assignment (M,) long)."""
    M = x.shape[0]
    k = min(int(k), M)
    if k <= 0:
        return x.new_zeros(0, x.shape[1]), torch.zeros(0, dtype=torch.long, device=x.device)
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(M, generator=g)[:k].to(x.device)].clone()
    for _ in range(iters):
        a = _nearest_centre(x, c)
        s = torch.zeros_like(c).index_add_(0, a, x)
        n = torch.bincount(a, minlength=k)
        upd = n > 0
        c[upd] = s[upd] / n[upd, None].to(x.dtype)
    return c, _nearest_centre(x, c)


def two_level_codebook(feat, xyz, k1=OPENGS_K1, k2=OPENGS_K2, pos_weight=OPENGS_POS_W,
                       iters=KMEANS_ITERS, seed=0):
    """Its two-level codebook as k-means: coarse on [f, pos_weight xyz] (k1), fine on f inside
    each coarse cluster (k2). -> dict(coarse (N,), instance (N,) = coarse k2 + fine, codes
    (k1 k2, C) the fine centres (0 for unused ids), n_instances k1 k2, coarse_centres)."""
    feat, xyz = feat.float(), xyz.float().to(feat.device)
    N = feat.shape[0]
    cc, ca = kmeans(torch.cat([feat, pos_weight * xyz], 1), k1, iters, seed)
    inst = torch.zeros(N, dtype=torch.long, device=feat.device)
    codes = torch.zeros(k1 * k2, feat.shape[1], dtype=feat.dtype, device=feat.device)
    for c in range(cc.shape[0]):
        sel = torch.nonzero(ca == c).flatten()
        if sel.numel() == 0:
            continue
        fc, fa = kmeans(feat[sel], k2, iters, seed + 1 + c)
        inst[sel] = c * k2 + fa
        codes[c * k2:c * k2 + fc.shape[0]] = fc
    return dict(coarse=ca, instance=inst, codes=codes, n_instances=k1 * k2, coarse_centres=cc)


@torch.no_grad()
def instance_pixels(ras, inst, n_inst, alpha, cov, share_min=SHARE_MIN, chunk=64):
    """(P,) instance id of each pixel (-1: none): the instance whose Gaussians hold at least
    share_min of the pixel's rendered alpha (sum_{n in i} w_{p,n} >= share_min alpha_p), on
    covered pixels. One render of the instances' one-hot indicators per `chunk` instances."""
    P = alpha.shape[0]
    best = torch.full((P,), -1.0, dtype=alpha.dtype, device=alpha.device)
    arg = torch.full((P,), -1, dtype=torch.long, device=alpha.device)
    for c0 in range(0, n_inst, chunk):
        ids = torch.arange(c0, min(n_inst, c0 + chunk), device=inst.device)
        R = ras((inst[:, None] == ids[None, :]).to(alpha.dtype))            # (P, C) shares
        v, a = R.max(1)
        up = v > best
        best = torch.where(up, v, best)
        arg = torch.where(up, a + c0, arg)
    return torch.where(cov & (best >= share_min * alpha), arg, torch.full_like(arg, -1))


@torch.no_grad()
def associate(views, inst, n_inst, codes, feat, alpha_min=0.5, share_min=SHARE_MIN,
              min_score=MIN_SCORE, chunk=64):
    """Its 2D-3D association. views: [(rasterizer, dict(lab (P,) compact whole-level mask ids,
    zbar (M, d) their unit pooled LSeg features))]; inst (N,) instance of each Gaussian; codes
    (n_inst, C) instance codewords; feat (N, C) the per-Gaussian instance features (rendered for
    p_j). -> dict(features (n_inst, d): normalize(sum_v IoU zbar) of the matched masks, 0 if
    never matched; weight (n_inst,) the summed IoU; n_matched (n_inst,) views)."""
    dev = feat.device
    N, C = feat.shape
    d = int(views[0][1]["zbar"].shape[1])
    acc = torch.zeros(n_inst, d, device=dev)
    wsum = torch.zeros(n_inst, device=dev)
    nmatch = torch.zeros(n_inst, dtype=torch.long, device=dev)
    inst = inst.to(dev).long()
    codes = codes.to(dev).float()
    for ras, vd in views:
        base = ras(torch.cat([feat.float(), torch.ones(N, 1, device=dev)], 1))
        alpha = base[:, -1]
        cov = alpha >= alpha_min
        pix = instance_pixels(ras, inst, n_inst, alpha, cov, share_min, chunk)
        lab = vd["lab"].to(dev).long()
        zbar = vd["zbar"].to(dev).float()
        M = zbar.shape[0]
        if M == 0:
            continue
        ok = (pix >= 0) & (lab >= 0)
        inter = torch.bincount(pix[ok] * M + lab[ok], minlength=n_inst * M).reshape(n_inst, M).float()
        a_i = torch.bincount(pix[pix >= 0], minlength=n_inst).float()
        a_j = torch.bincount(lab[lab >= 0], minlength=M).float()
        iou = inter / (a_i[:, None] + a_j[None, :] - inter).clamp_min(1.0)
        mj = cov & (lab >= 0)                       # p_j: mean F / alpha over B_j's covered pixels
        cnt = torch.bincount(lab[mj], minlength=M).float()
        Fp = base[:, :C] / alpha.clamp_min(1e-6)[:, None]
        pj = torch.zeros(M, C, device=dev).index_add_(0, lab[mj], Fp[mj]) / cnt.clamp_min(1.0)[:, None]
        dist = (codes[:, None, :] - pj[None, :, :]).abs().mean(-1)          # d_ij, mean L1
        score = iou * (1.0 - dist)                                          # its Eq. 5
        s, j = score.max(1)
        hit = torch.nonzero((a_i > 0) & (s > min_score)).flatten()
        if hit.numel() == 0:
            continue
        w = iou[hit, j[hit]]
        acc.index_add_(0, hit, w[:, None] * zbar[j[hit]])
        wsum.index_add_(0, hit, w)
        nmatch[hit] += 1
    out = torch.where((wsum > 0)[:, None], _unit(acc), torch.zeros_like(acc))
    return dict(features=out, weight=wsum, n_matched=nmatch)


# --------------------------------------------------------------------------- training
def _optimize(views, param, render, loss_fn, meta, lr, iters, alpha_min, g, device,
              log_every, min_iters, window, tol, ckpt_path, ckpt_every, log):
    """distill.train_features' loop on one parameter: a random view (generator g) per
    iteration, render [render(param) | 1] with its rasterizer, loss_fn(F (P, C), covered (P,),
    view data) (None: nothing to fit in this view), Adam, the stopping rule distill.converged
    and the checkpoint / resume of the full state (param, Adam, g, history)."""
    N = param.shape[0]
    ones = torch.ones(N, 1, device=device)
    opt = torch.optim.Adam([param], lr=lr)
    history, run, start = [], [], 1
    tag = meta.get("name", meta.get("method"))
    if ckpt_path and os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if st.get("meta") != meta:
            raise ValueError(f"{ckpt_path}: checkpoint of another run ({st.get('meta')} != {meta})")
        with torch.no_grad():
            param.copy_(st["param"].to(device))
        opt.load_state_dict(st["opt"])
        g.set_state(st["rng"])
        history, start = st["history"], st["it"] + 1
        log(f"[{tag}] resumed from {ckpt_path} at iter {st['it']}")
    it = start - 1
    for it in range(start, iters + 1):
        ras, vd = views[int(torch.randint(len(views), (1,), generator=g))]
        img = ras(torch.cat([render(param), ones], dim=1))
        loss = loss_fn(img[:, :-1], img[:, -1] >= alpha_min, vd)
        if loss is None:
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run.append(loss.item())
        if it % log_every == 0 or it == iters:
            history.append((it, sum(run) / len(run)))
            log(f"[{tag}] iter {it}/{iters} loss {history[-1][1]:.5f}")
            run = []
            if it >= min_iters and D.converged(history, window, tol):
                log(f"[{tag}] converged at iter {it}")
                break
        if ckpt_path and it % ckpt_every == 0:
            D._save_state(ckpt_path, meta=meta, it=it, param=param.detach().cpu(),
                          opt=opt.state_dict(), rng=g.get_state(), history=history)
    return history, it


def train_langsplat(views, N, iters=30000, lr=LANGSPLAT_LR, alpha_min=0.5, seed=0,
                    device="cpu", log_every=1000, min_iters=5000, window=5, tol=0.01,
                    ckpt_path=None, ckpt_every=2000, name="langsplat", log=print):
    """views: [(rasterizer, dict(lab (P,) compact mask ids of the level (-1: none), tgt (M, 3)
    the encoded mask features E(zbar_j)))]; rasterizer(buffer (N, C)) -> (P, C), linear and
    differentiable in the buffer. Trains the zero-initialized latents (N, 3) with the masked L1
    loss. -> dict(features (N, 3), history, iters_run)."""
    g = torch.Generator().manual_seed(seed)
    C = int(views[0][1]["tgt"].shape[1])
    f = torch.zeros(N, C, device=device, requires_grad=True)

    def loss_fn(F, cov, vd):
        lab = vd["lab"].long()
        sel = cov & (lab >= 0)
        if not bool(sel.any()):
            return None
        return (F[sel] - vd["tgt"][lab[sel]]).abs().mean()
    meta = dict(method="langsplat", name=name, N=int(N), C=C)
    history, it = _optimize(views, f, lambda p: p, loss_fn, meta, lr, iters, alpha_min, g,
                            device, log_every, min_iters, window, tol, ckpt_path, ckpt_every, log)
    return dict(features=f.detach(), history=history, iters_run=it)


def train_opengs(views, N, dim=OPENGS_DIM, iters=30000, lr=OPENGS_LR, smooth_weight=OPENGS_SMOOTH,
                 min_px=OPENGS_MIN_PX, alpha_min=0.5, seed=0, device="cpu", log_every=1000,
                 min_iters=5000, window=5, tol=0.01, ckpt_path=None, ckpt_every=2000,
                 name="opengs", log=print):
    """views: [(rasterizer, dict(lab (P,) compact whole-level mask ids))]. Trains theta (N, dim),
    rendered as theta / |theta|, with L_c + smooth_weight L_s. -> dict(features (N, dim) unit,
    history, iters_run)."""
    g = torch.Generator().manual_seed(seed)
    theta = torch.rand(N, dim, generator=g).to(device).requires_grad_(True)     # its init

    def loss_fn(F, cov, vd):
        return opengs_loss(F, vd["lab"], cov, smooth_weight, min_px)
    meta = dict(method="opengs", name=name, N=int(N), C=int(dim))
    history, it = _optimize(views, theta, lambda p: Fn.normalize(p, dim=-1), loss_fn, meta, lr,
                            iters, alpha_min, g, device, log_every, min_iters, window, tol,
                            ckpt_path, ckpt_every, log)
    return dict(features=Fn.normalize(theta.detach(), dim=-1), history=history, iters_run=it)


# --------------------------------------------------------------------------- files and rows
def variant_name(method, level=None):
    if method == "langsplat":
        if level not in LEVELS:
            raise ValueError(f"LangSplat level {level!r} not in {LEVELS}")
        return f"langsplat_{level}"
    if method == "opengs":
        return "opengs"
    raise ValueError(f"method {method!r} not in {METHODS}")


def row_names():
    return [variant_name("langsplat", lv) for lv in LEVELS] + [variant_name("opengs")]


def is_sam_file(d):
    return isinstance(d, dict) and d.get("family") == FAMILY


def gaussian_features(d, device="cpu"):
    """The per-Gaussian buffer a file renders: LangSplat's latents (N, 3), OpenGaussian's
    instance features expanded to the Gaussians (N, d)."""
    if d["method"] == "langsplat":
        return d["features"].to(device).float()
    if d["method"] == "opengs":
        return d["instance_features"].to(device).float()[d["instance"].to(device).long()]
    raise ValueError(f"unknown method {d.get('method')!r}")


def decoder_of(d, device="cpu"):
    """LangSplat: its LatentDecoder (on device); OpenGaussian: None (no decoder)."""
    if d["method"] != "langsplat":
        return None
    ae = LangSplatAE(tuple(d["ae_dims"]))
    ae.load_state_dict(d["ae"])
    return LatentDecoder(ae.eval()).to(device).eval()


def render_fn(d, device="cpu"):
    """rasterizer -> (P, d) query features of the file (render, then decode for LangSplat)."""
    buf, dec = gaussian_features(d, device), decoder_of(d, device)
    if dec is None:
        return lambda ras: ras(buf)
    return lambda ras: dec(ras(buf))


def rows_from_file(path, device="cpu"):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if not is_sam_file(d):
        raise ValueError(f"{path}: not a {FAMILY} file")
    row = TensorRow(d["name"], "feature", buf=gaussian_features(d, device),
                    decoder=decoder_of(d, device))
    return row, d


def row_factory(ckpt, field=None, device="cpu", field_path=None):
    """scripts/eval_seg.py --row-factory usplat.sam_lift:row_factory: the rows of the files
    next to the main field, <main>_langsplat_{s,p,w}.pt and <main>_opengs.pt (whichever exist).
    Each must be trained in the main field's PCA basis (checked) on as many Gaussians as the
    field has; its identity is appended to ckpt["_aux_identities"] (validation and test must
    use the same code and settings)."""
    main = field_path or (ckpt.get("config") or {}).get("out")
    if not main:
        return []
    from .paths import moved          # fields built before the 2026-10-31 move store old paths
    main = moved(main)
    paths = [D.out_path(main, n) for n in row_names()]
    paths = [p for p in paths if os.path.exists(p)]
    want = D.basis_sha256(ckpt["pca"]) if "pca" in ckpt else None
    rows = []
    for p in paths:
        row, d = rows_from_file(p, device)
        if want is not None and d.get("pca_basis_sha256") != want:
            raise ValueError(f"{p}: trained in another PCA basis than {main}")
        if field is not None and row.buf.shape[0] != field.N:
            raise ValueError(f"{p}: {row.buf.shape[0]} Gaussians, the field has {field.N}")
        ckpt.setdefault("_aux_identities", []).append(d.get("identity", "unknown"))
        rows.append(row)
    return rows
