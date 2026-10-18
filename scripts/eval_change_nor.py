"""
eval_change_nor.py — stale-content evaluation of one NeRF-Object-Removal scene (Table II of
papers/main.tex, label tab:change; protocol D1-D6 of 2026-10-17) on a FIXED 3DGS map.

  scripts/gpu_gate.sh 6000 python3 scripts/eval_change_nor.py \
      --field <fields>/nor_001_lseg.pt --align <results>/nor/align/001.json \
      --out <results>/nor/change/001.json

Inputs: the field run_mapping.py saved after the MAPPING pass only (--dataset nor --part map
--baselines --use-depth, without --update-data; refused otherwise). Its config names the map
(--ply), the scene folder (--data), the stride, the downscale, the extractor and the keyframe
rule; a field that records the sha256 of its map (inputs.ply) is refused on another map file.
--align: the scene's update-pose alignment (scripts/align_nor.py, D3); "none" only for checks.

Views. Mapping keyframes: the field's own (part map, its stride and max_keyframes, the
near-duplicate pose rule kf_trans / kf_rot of run_mapping.py). Update keyframes: part update
with the aligned poses, every --update-stride-th frame (default: the mapping stride), the
same pose rule. Features: the field's extractor at full resolution, bilinear to the
accumulation grid (image / downscale), the field's PCA (run_mapping.py's path). Depth: the
loader's high-confidence LiDAR depth on the same grid (0 = invalid); colour: the image there.

Coverage, ONE rule for every row. A view covers a Gaussian if it gives it at least fast_min
(FieldConfig, 1) of a full view weight in the change-test stream: w_v = min(1, m_{v,n} /
(a0 H W)), m the splat mass of the view (a0 H W = 9.72 px at 360x270), computed as the field
computes it (covered()). It is the gate of the recent bucket of our change score
(fast_memory = 0: the recent bucket is the view itself), so no row can score or flag a
Gaussian in a view that ours cannot test for lack of mass. Every row is tested only in the
views that cover the Gaussian: the residuals see only those views (ChangeBaselines(observed=
the coverage mask)), NormLift and the dispersion are tested only there, and ours needs the
coverage by construction plus its own support and history gates (tau_min, slow_min), as a
residual needs k observations. A Gaussian is OBSERVED if an update keyframe covers it. The
observed Gaussians are the population of every Gaussian-level metric, and every row's score
is NaN outside it. Gaussians that update keyframes see only with less mass (m > 0 but
w_v < fast_min in every view) are counted apart (counts.n_seen_uncovered), as are those no
update keyframe sees. NormLift and the dispersion accumulate every view by their own
evidence rule (pixel mass, view weights), as when they map.

Passes (each update is one render + one backward, the gradient trick):
 1 mapping replay  the mapping keyframes through a fresh field (the saved FieldConfig with
                   change_thr = inf, as mapped), a fresh NormLift on its hooks and
                   ChangeBaselines rendered against the FROZEN saved field and map, all at
                   the coverage above. It gives every row's per-view test statistic on the
                   mapping views (the operating points below) and checks that the replay
                   reproduces the saved field.
 2 update          the update keyframes through
                     ours        a copy of the saved field: calibrate_change_threshold (its
                                 own mapping-pass null, one quantile per history bucket),
                                 then update_view with depth as run_mapping.py does: test,
                                 reset, depth re-admission
                     dispersion  a copy with change_test off (no reset), with the saved NormLift
                                 on its hooks
                     residuals   ChangeBaselines(k) against the frozen saved field and map
Frozen map renders (one call per view): the 3DGS colour (black background), the saved
field's mu* (the feature residual's "map rendered from the mapping field", as CL-Splats and
MIF) and the expected depth sum w z / sum w.

Rows (per-Gaussian score, higher = more likely stale; NaN = never scored, ranked last):
  ncc         I    ncc/kmin            residual rows: the largest value over the update of
  depth       D    depth_render/kmin   the persistent residual, the min of the last k = 2
  feat        F    feat/kmin           covered per-view residuals (NaN before k views)
  normlift    F    R after mapping - R after the update (NormLift reliability)
  dispersion  F    1 - rho_hat after the update (the aleatoric read-out; NaN below tau_min)
  ours        F+D  max over the update views of c_bar (c_max after calibration; NaN if it was
                   never defined in the update)
  fused       F+D  usplat.change_baselines.fuse_rank(ours, depth) on the AP population
Extra scores (JSON only, no table row): depth_gauss/kmin and the "max" filter (k = 1) of
every residual.

Operating points (FA, Rec). Each row is tested in every covered view with a per-view
statistic, and its threshold is the --change-quantile (0.999) empirical quantile of the SAME
statistic on the mapping replay; one quantile per history bucket where the statistic
depends on how much evidence the Gaussian holds, with the bucket edges and the borrowing of
calibrate_change_threshold (DirectionalSemanticField.HIST_EDGES, MIN_NULL: a bucket with
fewer samples takes the quantile of the nearest bucket that has enough). A Gaussian is
flagged if it passes the threshold at least once during the update:
  residual rows  the min of the last k residuals against the frozen map (flagged = score >
                 thr); one pooled quantile: the map does not depend on the Gaussian's history
  normlift       its ranking statistic, the drop of R since the end of mapping, R_map - R_t
                 after view t. Null: the drop R_s - R_t over the last L = min(V, M // 2)
                 mapping keyframes (s = M - L; M mapping, V update keyframes), by NormLift's
                 N_eff at the reference (after keyframe s in the null, at the end of mapping
                 in the update)
  dispersion     1 - rho_hat after the view, by its support after the view
  ours           its own calibration, by its slow-bucket history; flagged = reset at least once
                 during the update (n_resets grew). The end state (field.changed after
                 re-admission) is reported as well
  fused          no threshold: its ranks are relative to the update population (null)
FA = flagged share of the static observed Gaussians, Rec = of the stale observed ones; both
also on the COMMON TESTED set (fa_common, rec_common: the observed Gaussians that every row
with an operating point tested at least once). Realised per-view rates (the paper: "we
measure the actual rate"): fires / tests over the update's Gaussian-views, on the static and
on the stale observed Gaussians (rate_static, rate_stale; the threshold targets 1 - q on
static content).
Kept (ours) = share of the stale observed Gaussians in ours.robust_mask() after the update
(and after the mapping pass, for reference); kept_seen: the same over every stale Gaussian an
update keyframe sees (m > 0), covered or not (untested stale Gaussians keep gaining support).

Ground truth (fixed before any method ran).
  D2 Gaussian: stale iff its centre is in boxes/bbox.txt inflated by --inflate (2 cm) AND
     farther than --near (2 cm) from every high-confidence depth point of ALL update frames
     (stride 1, aligned poses). Box-only counts are stored too (no metric row). The
     population is the observed Gaussians; unobserved ones are counted, not scored.
     Sensitivity (no table row, ground_truth.any_confidence): the same test with the
     after-removal points of ANY ARKit confidence, its stale counts and every row's AP /
     AUROC against those labels on the same population. A static surface that the LiDAR
     measures only at low or medium confidence after the removal is stale under D2 and
     static under this variant.
     Gate (ground_truth.static_shell): the high-confidence depth points of every 4th mapping
     frame (every 2nd pixel) between the box inflated by 5 cm and by 30 cm lie on static
     surfaces; the share of them within --near of a high-confidence after-removal point
     measures how much of the static surface the aligned update part sees in place (a
     residual pose offset after D3 makes D2 call static in-box surfaces stale). Also the
     shares within 5 and 10 cm and the same share for the mapping points inside the box
     inflated by 2 cm. scripts/nor_tab2.py merge applies the split config's gt_gate to it.
  D1 pixel: masks_real of each update keyframe, nearest-resized to the render grid, are the
     positives; covered pixels only (rendered alpha >= --tau-cov), pooled over the update
     keyframes. A row's per-Gaussian score is rendered as sum_n w s_n / sum_n w over its
     defined Gaussians (usplat.seg_eval.render_row: NaN where less than 1 - --undef-frac of
     the alpha is defined).
Metrics: usplat.evaluation.average_precision and auroc with NaN ranked last, and on the
set where every table row is defined (usplat.change_baselines.common_defined), with counts.

Backends: gsplat (GPU) or mock (CPU, eval_seg.py's pinhole mock rasterizer; smoke tests and
tests/test_eval_change.py). The JSON is a usplat.runlog record (metrics under "metrics")
unless --plain-json. Provenance: sha256 of the field, the map and the alignment (with its
fit settings), and of the evaluation code.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import run_mapping as RM                                                    # noqa: E402
from eval_seg import _nearest                                               # noqa: E402
from time_field import (select_keyframes, field_config, rasterizer_factory,  # noqa: E402
                        load_extractor, check_extractor, evenly)
from usplat import baselines as B                                           # noqa: E402
from usplat.box3d import read_box_corners, points_in_box                    # noqa: E402
from usplat.change_baselines import ChangeBaselines, common_defined, fuse_rank  # noqa: E402
from usplat.clip_dense import upsample_tokens                               # noqa: E402
from usplat.datasets import make_dataset, downscale, NeRFObjectRemovalDataset  # noqa: E402
from usplat.evaluation import average_precision, auroc                      # noqa: E402
from usplat.feature_pca import StreamingPCA                                 # noqa: E402
from usplat.gs_io import load_gaussians                                     # noqa: E402
from usplat.nor_align import backproject, INFLATE                           # noqa: E402
from usplat.rows import TensorRow                                           # noqa: E402
from usplat.runlog import to_jsonable                                       # noqa: E402
from usplat.seg_eval import render_row                                      # noqa: E402
from usplat.vmf_field import DirectionalSemanticField                       # noqa: E402

EPS = 1e-12                       # the field's seen rule m > EPS (vmf_field._EPS)
NEAR = 0.02                       # [m] D2: distance to the after-removal depth points
SHELL = (0.05, 0.30)              # [m] D2 gate: the shell between the box inflated by these
SHELL_STEP = (4, 2)               # D2 gate: every 4th mapping frame, every 2nd depth pixel
ALPHA_EPS = 1e-6                  # rendered alpha below this covers nothing (seg_eval)
HIST_EDGES = DirectionalSemanticField.HIST_EDGES    # history buckets [views] of the nulls
MIN_NULL = DirectionalSemanticField.MIN_NULL        # samples a bucket needs for its own thr
# the loader's high-confidence level (its default min_confidence), the D2 truth's points
HIGH_CONFIDENCE = inspect.signature(NeRFObjectRemovalDataset).parameters["min_confidence"].default
RESIDUALS = dict(ncc="ncc", depth="depth_render", feat="feat", depth_gauss="depth_gauss")
ROWS = ("ncc", "depth", "feat", "normlift", "dispersion", "ours", "fused")
TESTED = ("ncc", "depth", "feat", "normlift", "dispersion", "ours")   # rows with a threshold
INPUTS = dict(ncc="I", depth="D", feat="F", normlift="F", dispersion="F", ours="F+D",
              fused="F+D")
EXTRA = (("depth_gauss", "kmin"), ("ncc", "max"), ("depth_render", "max"), ("feat", "max"),
         ("depth_gauss", "max"))
RULES = dict(
    ncc="per-view min of the last k residuals > thr (pooled) at least once (= score > thr)",
    depth="per-view min of the last k residuals > thr (pooled) at least once (= score > thr)",
    feat="per-view min of the last k residuals > thr (pooled) at least once (= score > thr)",
    depth_gauss="per-view min of the last k residuals > thr (pooled) at least once "
                "(= score > thr)",
    normlift="drop of R since the end of mapping > thr after a covered view at least once "
             "(null: the drop over the last mapping keyframes; by N_eff at the reference)",
    dispersion="1 - rho_hat after a covered view > thr at least once (by its support)",
    ours="reset at least once during the update (own calibration, history buckets)",
    fused="none: rank fusion has no mapping-pass score scale")
CODE = ("scripts/eval_change_nor.py", "scripts/time_field.py", "scripts/eval_seg.py",
        "usplat/change_baselines.py", "usplat/evaluation.py", "usplat/seg_eval.py",
        "usplat/rows.py", "usplat/nor_align.py", "usplat/box3d.py")


# ------------------------------------------------------------------------------ inputs
def check_mapping_only(ckpt, path):
    """The protocol's field: nor, part map, --use-depth, --baselines (NormLift), saved after
    the mapping pass (uncalibrated, no reset): the update runs here, once per row."""
    cfg, s = ckpt.get("config") or {}, ckpt["field"]
    bad = []
    if cfg.get("dataset") != "nor":
        bad.append(f"dataset {cfg.get('dataset')!r}, not 'nor'")
    if cfg.get("part") != "map":
        bad.append(f"part {cfg.get('part')!r}, not 'map'")
    if cfg.get("update_data"):
        bad.append("built with --update-data (its update already ran)")
    if not cfg.get("use_depth"):
        bad.append("built without --use-depth")
    if not any(e.get("type") == "NormLift" for e in ckpt.get("baselines") or []):
        bad.append("no NormLift baseline (map with --baselines)")
    if s.get("thr_by_hist") is not None or math.isfinite(float(s["config"]["change_thr"])):
        bad.append("its change threshold is already calibrated")
    if s.get("n_resets") is not None and int(s["n_resets"].sum()) > 0:
        bad.append("it reset Gaussians")
    if bad:
        raise SystemExit(f"{path}: not a mapping-only protocol field: " + "; ".join(bad))


def file_entry(path, hash=True):
    """{path, bytes, sha256} of an input file."""
    h = None
    if hash:
        d = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                d.update(b)
        h = d.hexdigest()
    return dict(path=os.path.abspath(path), bytes=os.path.getsize(path), sha256=h)


def align_entry(path):
    """The alignment's file entry plus the settings of its fit (None for no alignment)."""
    if path is None:
        return None
    with open(path) as f:
        rec = json.load(f)
    return dict(file_entry(path), scene=rec.get("scene"), settings=rec.get("settings"))


def code_state():
    """sha256 of the files that define this evaluation."""
    repo = os.path.dirname(HERE)
    out = {}
    for rel in CODE:
        with open(os.path.join(repo, rel), "rb") as f:
            out[rel] = hashlib.sha256(f.read()).hexdigest()
    return out


def read_views(ds, idx, factor, clip):
    """run_mapping.py's inputs of the kept frames: extractor tokens at full resolution (kept
    on the CPU), pose, intrinsics, colour and depth at the accumulation grid, and masks_real
    (raw 192x256 grid)."""
    views = []
    for i in idx:
        full = ds[i]
        tok = clip.extract_tokens(full["image"]).cpu()
        fr = downscale(full, factor)
        h, w = fr["image"].shape[:2]
        dep = fr.get("depth")
        views.append(dict(index=int(full["index"]), tok=tok, c2w=full["c2w"], K=fr["K"],
                          h=int(h), w=int(w), rgb=fr["image"].reshape(-1, 3).float() / 255.0,
                          depth=None if dep is None else dep.reshape(-1).float(),
                          mask_real=full.get("mask_real")))
    return views


def rasterizer(ctx, v):
    """The view's rasterizer and the camera depth of the Gaussian centres
    (= GSplatFeatureRasterizer.gaussian_depths)."""
    dev = ctx["device"]
    viewmat = torch.linalg.inv(v["c2w"].double()).float().to(dev)
    ras = ctx["rasterizer"](ctx["splats"], viewmat, v["K"].to(dev), v["w"], v["h"])
    return ras, ctx["splats"]["means"] @ viewmat[2, :3] + viewmat[2, 3]


def view_inputs(ctx, v):
    """(z (P, d) features, rasterizer, Gaussian depths, update_view depth keywords)."""
    dev = ctx["device"]
    z = ctx["pca"].project(upsample_tokens(v["tok"].to(dev), (v["h"], v["w"])))
    ras, gz = rasterizer(ctx, v)
    kw = {} if v["depth"] is None else dict(depth_obs=v["depth"].to(dev), gauss_depth=gz)
    return z.reshape(-1, ctx["d"]), ras, gz, kw


@torch.no_grad()
def map_render(ras, gz, frozen):
    """One render of the frozen map: colour (P, 3), the mapping field's mu* (P, d) and the
    expected depth (P,) (0 where nothing is rendered)."""
    one = torch.ones_like(gz)[:, None]
    out = ras(torch.cat([frozen["colors"], frozen["mu"], gz[:, None], one], 1))
    d = frozen["mu"].shape[1]
    alpha = out[:, 4 + d]
    depth = torch.where(alpha > ALPHA_EPS, out[:, 3 + d] / alpha.clamp_min(ALPHA_EPS),
                        torch.zeros_like(alpha))
    return dict(rgb=out[:, :3], feat=out[:, 3:3 + d], depth=depth)


# ---------------------------------------------------------------------------- coverage
@torch.no_grad()
def covered(field, stats):
    """(N,) bool: the Gaussians this view covers (module docstring): seen, with a change-stream
    weight of at least fast_min, as the field's change test computes it (its _change_weights
    of the raw mass, of the centred mass with centre=True). Our score's recent bucket has
    weight nf = w with fast_memory = 0, so its gate nf >= fast_min is exactly this mask."""
    mass = stats.mc if field.cfg.centre else stats.m
    w, seen = field._change_weights(mass, stats)
    return seen & (w >= field.cfg.fast_min)


def dispersion_support(field):
    """(N,) the support that defines the dispersion read-out (aleatoric(): >= tau_min)."""
    return field._support(field._dispersion_channel())


def residuals_view(cb, ras, v, z, ren, gz, kw, observed):
    """One ChangeBaselines update: NCC (colour), feature and depth residuals vs the frozen map,
    with the shared coverage mask. Returns {residual: per-view value (NaN unobserved)}."""
    a = dict(observed_rgb=v["rgb"].to(z.device), rendered_rgb=ren["rgb"], observed_feat=z,
             rendered_feat=ren["feat"], image_hw=(v["h"], v["w"]), observed=observed)
    if kw:
        a.update(depth_obs=kw["depth_obs"], rendered_depth=ren["depth"], gauss_depth=gz)
    return cb.update(ras, v["h"] * v["w"], **a)


def view_kmin(cb, name, per_view):
    """(N,) the persistent residual at this view: the min of the last k observed residuals,
    for the Gaussians observed in it that have k observations; NaN elsewhere. None if the
    residual was never supplied."""
    ts, r = cb.series.get(name), per_view.get(name)
    if ts is None or r is None:
        return None
    full = torch.isfinite(r) & (ts.count >= ts.k)
    return torch.where(full, ts.ring.min(dim=1).values, torch.full_like(r, float("nan")))


def series_score(cb, name, filt, N):
    """The residual's filtered score, NaN everywhere if it was never supplied."""
    if name not in cb.series:
        return torch.full((N,), float("nan"))
    return cb.score(name, filt).detach().cpu()


def _normlift_entry(ckpt):
    return next(e for e in ckpt["baselines"] if e["type"] == "NormLift")


def make_normlift(ckpt, field, saved):
    """The mapping run's NormLift on field's hooks: empty (saved=False) or with its saved
    mapping-pass sums."""
    e = _normlift_entry(ckpt)
    cfg = dict(e["config"])
    dtype = getattr(torch, cfg.pop("dtype", "float32"))
    nl = B.NormLift(**cfg, device=field.device, dtype=dtype)
    if saved:
        nl.load_state_dict(e["state"])
    return nl.attach(field)


def normlift_segment(n_mapping, n_update):
    """NormLift's null segment: the last L = min(V, M // 2) mapping keyframes, after the
    reference = the state after keyframe s = M - L (1-based; L = 0: no null)."""
    L = min(int(n_update), int(n_mapping) // 2)
    return dict(start=int(n_mapping) - L, length=L, n_mapping=int(n_mapping),
                n_update=int(n_update))


# -------------------------------------------------------------------------------- nulls
class Null:
    """The finite per-view statistics of one row on the mapping replay; with edges, one
    sample set per history bucket (torch.bucketize of a per-Gaussian history size, as the
    field's calibrate_change_threshold)."""

    def __init__(self, edges=None):
        self.edges = None if edges is None else [float(e) for e in edges]
        self.parts = [[] for _ in range(1 if edges is None else len(self.edges) + 1)]

    def add(self, x, hist=None):
        ok = torch.isfinite(x)
        if not bool(ok.any()):
            return
        v = x[ok].detach().float().cpu()
        if self.edges is None:
            self.parts[0].append(v)
            return
        b = torch.bucketize(hist[ok].detach().double().cpu(),
                            torch.tensor(self.edges, dtype=torch.float64))
        for i in torch.unique(b).tolist():
            self.parts[i].append(v[b == i])

    def threshold(self, q):
        """{pooled: the empirical q-quantile of all samples, n, and with edges by_history:
        one q-quantile per bucket (a bucket with fewer than MIN_NULL samples takes the
        nearest bucket that has enough; the pooled one if none has), n_by_history, edges};
        NaN thresholds without samples (such a row never flags)."""
        arrs = [torch.cat(p).numpy() if p else np.zeros(0, np.float32) for p in self.parts]
        allv = np.concatenate(arrs)
        out = dict(pooled=float(np.quantile(allv, q)) if allv.size else float("nan"),
                   n=int(allv.size), by_history=None, n_by_history=None, edges=self.edges)
        if self.edges is None:
            return out
        counts = [int(a.size) for a in arrs]
        good = [b for b, c in enumerate(counts) if c >= MIN_NULL]
        per = []
        for b in range(len(arrs)):                          # nearest bucket with data
            src = min(good, key=lambda g: abs(g - b)) if good else None
            per.append(float(np.quantile(arrs[src], q)) if src is not None else out["pooled"])
        out.update(by_history=per, n_by_history=counts)
        return out


def per_gaussian(thr, hist):
    """(N,) the threshold of each Gaussian: its history bucket's (by_history), or pooled."""
    if thr.get("by_history") is None:
        return torch.full_like(hist, thr["pooled"], dtype=torch.float32)
    per = torch.tensor(thr["by_history"], dtype=torch.float32, device=hist.device)
    edges = torch.tensor(thr["edges"], dtype=torch.float64, device=hist.device)
    return per[torch.bucketize(hist.double(), edges)]


# ------------------------------------------------------------------------------ passes
def replay_mapping(ctx, views, k, n_update):
    """Pass 1: the mapping keyframes through a fresh field (as mapped: change_thr = inf), a
    fresh NormLift on its hooks and ChangeBaselines vs the frozen map, every row at the
    shared coverage. Returns (field, {row: Null}, NormLift's null segment)."""
    ckpt, dev, N = ctx["ckpt"], ctx["device"], ctx["N"]
    field = DirectionalSemanticField(N, ctx["d"], field_config(ckpt), device=dev)
    field.set_centre(ctx["zbar"])
    nl = make_normlift(ckpt, field, saved=False)
    cb = ChangeBaselines(N, k=k, view_mass_frac=field.cfg.view_mass_frac, device=dev)
    null = {r: Null() for r in RESIDUALS}
    null.update(normlift=Null(HIST_EDGES), dispersion=Null(HIST_EDGES))
    seg = normlift_segment(len(views), n_update)
    ref = None
    for t, v in enumerate(views, start=1):
        z, ras, gz, kw = view_inputs(ctx, v)
        ren = map_render(ras, gz, ctx["frozen"])
        stats = field.update_view(ras, z, num_pixels=v["h"] * v["w"], **kw)
        cov = covered(field, stats)
        if ref is not None:                         # the drop of R since the reference
            null["normlift"].add((ref[0] - nl.reliability())[cov], ref[1][cov])
        if t == seg["start"] and seg["length"]:
            ref = (nl.reliability().clone(), nl.n_eff().clone())
        ale, ok = field.aleatoric()
        null["dispersion"].add(ale[cov & ok], dispersion_support(field)[cov & ok])
        per = residuals_view(cb, ras, v, z, ren, gz, kw, cov)
        for row, name in RESIDUALS.items():
            cur = view_kmin(cb, name, per)
            if cur is not None:
                null[row].add(cur)
    return field, null, seg


def replay_check(field, saved):
    """How far the replayed field is from the saved one (raw sums and the change-score null
    histogram). GPU sums are not bit-exact; a different keyframe or feature is far off."""
    W1, R = saved["raw_W1"], saved["raw_R"]
    dW1 = float((field.raw.W1.cpu() - W1).abs().max())
    dR = float((field.raw.R.cpu() - R).abs().max())
    h0 = saved.get("c_hist")
    h1 = field.c_hist.cpu()
    h0 = torch.zeros_like(h1) if h0 is None else h0
    tot = float(h0.sum())
    out = dict(max_abs_dW1=dW1, max_rel_dW1=dW1 / max(float(W1.abs().max()), 1e-12),
               max_abs_dR=dR, c_hist_total_saved=tot, c_hist_total_replay=float(h1.sum()),
               c_hist_l1_share=float((h1 - h0).abs().sum()) / max(tot, 1.0))
    out["ok"] = out["max_rel_dW1"] <= 1e-3 and out["c_hist_l1_share"] <= 1e-2
    return out


def run_update(ctx, views, k, thr, q):
    """Pass 2: the update keyframes through ours, the dispersion copy (+ saved NormLift) and
    ChangeBaselines, every row tested at the shared coverage only; scores, flags, per-Gaussian
    test and fire counts, and the coverage."""
    ckpt, dev, N = ctx["ckpt"], ctx["device"], ctx["N"]
    ours = DirectionalSemanticField.from_state(ckpt["field"], device=dev)
    robust_map = ours.robust_mask().cpu()
    thr_ours = ours.calibrate_change_threshold(q)
    resets0 = ours.n_resets.clone()
    disp = DirectionalSemanticField.from_state(ckpt["field"], device=dev)
    disp.cfg.change_test = False                                  # dispersion: no reset
    nl = make_normlift(ckpt, disp, saved=True)
    r_map = nl.reliability().clone()
    thr_nl = per_gaussian(thr["normlift"], nl.n_eff())            # history at the reference
    cb = ChangeBaselines(N, k=k, view_mass_frac=ours.cfg.view_mass_frac, device=dev)
    zeros = lambda dtype: torch.zeros(N, dtype=dtype, device=dev)
    seen, observed, n_views = zeros(torch.bool), zeros(torch.bool), zeros(torch.int32)
    rows = (*RESIDUALS, "normlift", "dispersion", "ours")
    flags = {r: zeros(torch.bool) for r in rows}
    tests = {r: zeros(torch.int32) for r in rows}
    fires = {r: zeros(torch.int32) for r in rows}

    def record(r, tested, fired):
        tests[r] += tested.int()
        fires[r] += fired.int()
        flags[r] |= fired

    for v in views:
        z, ras, gz, kw = view_inputs(ctx, v)
        ren = map_render(ras, gz, ctx["frozen"])
        stats = ours.update_view(ras, z, num_pixels=v["h"] * v["w"], **kw)
        cov = covered(ours, stats)
        if bool((ours.last_tested & ~cov).any()):
            raise RuntimeError("ours tested a Gaussian that the coverage rule does not admit")
        seen |= stats.m > EPS
        observed |= cov
        n_views += cov.int()
        record("ours", ours.last_tested, ours.last_fired)
        disp.update_view(ras, z, num_pixels=v["h"] * v["w"], **kw)
        record("normlift", cov, cov & ((r_map - nl.reliability()) > thr_nl))
        ale, ok = disp.aleatoric()
        t = cov & ok
        record("dispersion", t, t & (ale > per_gaussian(thr["dispersion"],
                                                        dispersion_support(disp))))
        per = residuals_view(cb, ras, v, z, ren, gz, kw, cov)
        for row, name in RESIDUALS.items():
            cur = view_kmin(cb, name, per)
            if cur is not None:
                t = torch.isfinite(cur)
                record(row, t, t & (cur > thr[row]["pooled"]))
    nan = torch.full((N,), float("nan"), device=dev)
    inside = lambda s: torch.where(observed, s, nan).cpu()      # read out on the observed
    scores = dict(ncc=series_score(cb, "ncc", "kmin", N),
                  depth=series_score(cb, "depth_render", "kmin", N),
                  feat=series_score(cb, "feat", "kmin", N),
                  normlift=inside(r_map - nl.reliability()),
                  dispersion=inside(disp.aleatoric()[0]),
                  ours=torch.where(torch.isfinite(ours.c_last), ours.c_max, nan).cpu())
    extra = {f"{n}/{f}": series_score(cb, n, f, N) for n, f in EXTRA}
    flags = {r: f.cpu() for r, f in flags.items()}
    for row, name in RESIDUALS.items():           # the kmin flag is the score above thr
        s = scores.get(row, extra.get(f"{name}/kmin"))
        if not torch.equal(flags[row], torch.isfinite(s) & (s > thr[row]["pooled"])):
            raise RuntimeError(f"{row}: per-view flags differ from score > threshold")
    if not torch.equal(flags["ours"], (ours.n_resets > resets0).cpu()):
        raise RuntimeError("ours: its fires differ from its resets")
    return dict(scores=scores, extra=extra, flags=flags,
                tests={r: t.cpu() for r, t in tests.items()},
                fires={r: f.cpu() for r, f in fires.items()},
                observed=observed.cpu(), seen=seen.cpu(), n_views=n_views.cpu(),
                changed_end=ours.changed.cpu(), robust_end=ours.robust_mask().cpu(),
                robust_map=robust_map, n_resets=int((ours.n_resets - resets0).sum()),
                ours_threshold=dict(pooled=thr_ours, by_history=None if ours.thr_by_hist is None
                                    else ours.thr_by_hist.cpu().tolist(),
                                    history_edges=list(ours.HIST_EDGES),
                                    calib_mode=ours.cfg.calib_mode, quantile=q))


# ------------------------------------------------------------------------ ground truth
def update_depth_points(root, align, corners, margin):
    """Depth points of ALL update frames (stride 1, aligned poses, every finite depth > 0)
    inside the box inflated by `margin` (no other point can be near an in-box Gaussian or
    the shell), their ARKit confidence (0/1/2) and counts (n_points: high confidence)."""
    ds = NeRFObjectRemovalDataset(root, part="update", align=align, min_confidence=0)
    pts, conf, n_all, n_high = [], [], 0, 0
    for i in ds.frame_numbers:
        d, c = ds.read_depth(i)
        P, v, u = backproject(d, ds.K_depth[i], ds.c2w[i])
        cf = c[v, u]
        n_all += len(P)
        n_high += int((cf >= HIGH_CONFIDENCE).sum())
        keep = points_in_box(P, corners, margin)
        pts.append(P[keep])
        conf.append(cf[keep])
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    C = np.concatenate(conf) if conf else np.zeros(0, np.uint8)
    return P, C, dict(n_frames=len(ds.frame_numbers), n_points=n_high,
                      n_points_any_confidence=n_all, min_confidence=HIGH_CONFIDENCE)


def stale_ground_truth(means, corners, after, inflate=INFLATE, near=NEAR):
    """D2: stale iff the centre is in the box inflated by `inflate` AND farther than `near`
    from every after-removal point. Returns bool arrays stale, in_box, near_after."""
    m = np.asarray(means, dtype=np.float64).reshape(-1, 3)
    in_box = points_in_box(m, corners, inflate)
    near_after = np.zeros(len(m), bool)
    if in_box.any() and len(after):
        dist, _ = cKDTree(after).query(m[in_box], distance_upper_bound=near)
        near_after[in_box] = np.isfinite(dist)
    return dict(stale=in_box & ~near_after, in_box=in_box, near_after=near_after)


def static_shell(root, corners, after, near=NEAR, shell=SHELL, steps=SHELL_STEP):
    """The D2 gate (module docstring): share of the mapping depth points in the static shell
    that lie within `near` of an after-removal point (`after`: high confidence, aligned),
    also within 5 and 10 cm, and the same share for the mapping points in the box inflated by
    INFLATE (static and removed surface mixed)."""
    ds = NeRFObjectRemovalDataset(root, part="map")              # high confidence, raw poses
    sub = np.zeros(ds.DEPTH_HW, bool)
    sub[::steps[1], ::steps[1]] = True
    pts = []
    for i in range(ds.start, ds.end, steps[0]):
        d, _ = ds.read_depth(i)
        P, _, _ = backproject(d, ds.K_depth[i], ds.c2w[i], sub)
        pts.append(P[points_in_box(P, corners, shell[1])])
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    dist = np.full(len(P), np.inf)
    if len(P) and len(after):
        dist, _ = cKDTree(after).query(P, distance_upper_bound=0.1)
    ring = ~points_in_box(P, corners, shell[0])
    inner = points_in_box(P, corners, INFLATE)
    frac = lambda m, r: float((dist[m] < r).mean()) if m.any() else float("nan")
    return dict(share_near=frac(ring, near), share_5cm=frac(ring, 0.05),
                share_10cm=frac(ring, 0.1), n_points=int(ring.sum()),
                in_box_share_near=frac(inner, near), n_in_box=int(inner.sum()),
                inner_m=shell[0], outer_m=shell[1], near_m=near, frame_step=steps[0],
                pixel_step=steps[1], min_confidence=ds.min_confidence)


# ----------------------------------------------------------------------------- metrics
def ranking(score, y):
    """AP and AUROC (NaN ranked last) of one score against 0/1 labels, with counts."""
    s, y = score.double().cpu(), y.double().cpu()
    nan = ~torch.isfinite(s)
    return dict(ap=average_precision(s, y), auroc=auroc(s, y), n=int(s.numel()),
                n_pos=int(y.sum()), n_nan=int(nan.sum()), n_nan_pos=int((nan & (y > 0)).sum()))


def ranking_table(scores, y, rows):
    """{row: ranking + its AP/AUROC on the common defined set}, and that set's counts."""
    out = {r: ranking(scores[r], y) for r in rows}
    common = common_defined([scores[r].cpu() for r in rows])
    yc = y.cpu()[common].double()
    for r in rows:
        s = scores[r].cpu()[common].double()
        out[r]["common"] = dict(ap=average_precision(s, yc), auroc=auroc(s, yc))
    return out, dict(n=int(common.sum()), n_pos=int(yc.sum()), rows=list(rows))


def share(mask):
    return float(mask.double().mean()) if mask.numel() else float("nan")


def operating_points(flags, stale, pop, common):
    """FA = flagged share of the static observed Gaussians, Rec = of the stale observed;
    fa_common / rec_common the same on the common tested set."""
    st, sp, cs, cp = pop & stale, pop & ~stale, common & stale, common & ~stale
    return {r: dict(fa=share(f[sp]), rec=share(f[st]), n_flagged_static=int(f[sp].sum()),
                    n_flagged_stale=int(f[st].sum()), fa_common=share(f[cp]),
                    rec_common=share(f[cs]), rule=RULES[r]) for r, f in flags.items()}


def realised_rates(tests, fires, stale, pop):
    """Per row: tests, fires and fires / tests over the update's Gaussian-views, on the
    static and on the stale observed Gaussians (the threshold targets 1 - q on static)."""
    out = {}
    for r in tests:
        row = {}
        for name, m in (("static", pop & ~stale), ("stale", pop & stale)):
            nt, nf = int(tests[r][m].long().sum()), int(fires[r][m].long().sum())
            row.update({f"tests_{name}": nt, f"fires_{name}": nf,
                        f"rate_{name}": nf / nt if nt else float("nan")})
        out[r] = row
    return out


@torch.no_grad()
def pixel_level(ctx, views, scores, rows, tau_cov, undef_frac):
    """D1: every row's scores rendered into the update keyframes, positives = masks_real
    (nearest-resized), covered pixels only, pooled over the keyframes."""
    dev = ctx["device"]
    row = TensorRow("change", "feature", buf=torch.zeros(ctx["N"], 1, device=dev),
                    unc={r: scores[r].to(dev).float() for r in rows})
    gts, cols, frames, no_mask = [], {r: [] for r in rows}, [], 0
    for v in views:
        if v["mask_real"] is None:
            no_mask += 1
            continue
        ras, _ = rasterizer(ctx, v)
        rr = render_row(row, ras, alpha_eps=ALPHA_EPS, undef_frac=undef_frac)
        cov = (rr.alpha >= tau_cov) & (rr.alpha > ALPHA_EPS)
        gt = _nearest(v["mask_real"], v["h"], v["w"]).reshape(-1).to(cov.device)
        gts.append(gt[cov].cpu())
        for r in rows:
            cols[r].append(rr.unc[r][cov].cpu())
        frames.append(dict(index=v["index"], n_covered=int(cov.sum()), n_pos=int(gt[cov].sum()),
                           n_mask=int(gt.sum())))
    if not frames:
        return dict(n_frames=0, n_frames_no_mask=no_mask)
    y = torch.cat(gts)
    table, common = ranking_table({r: torch.cat(cols[r]) for r in rows}, y, rows)
    return dict(rows=table, common=common, n_frames=len(frames), n_frames_no_mask=no_mask,
                n_pix=int(y.numel()), n_pos=int(y.sum()), frames=frames)


def count_stats(n_views, mask):
    """Covering update keyframes per Gaussian over a mask: median, mean, share >= 2."""
    x = n_views[mask].double()
    if not x.numel():
        return dict(n=0)
    return dict(n=int(x.numel()), median=float(x.median()), mean=float(x.mean()),
                share_ge2=float((x >= 2).double().mean()))


# -------------------------------------------------------------------------------- run
def load_context(args):
    """The field, the map, the views and everything both passes share."""
    t0 = time.time()
    ckpt = torch.load(args.field, map_location="cpu", weights_only=False)
    check_mapping_only(ckpt, args.field)
    cfg, dev = ckpt["config"], args.device
    ply, data = args.ply or cfg["ply"], args.data or cfg["data"]
    align = None if args.align == "none" else args.align
    ply_entry = file_entry(ply)
    built_on = ((ckpt.get("inputs") or {}).get("ply") or {}).get("sha256")
    if built_on is not None and built_on != ply_entry["sha256"]:
        raise SystemExit(f"{args.field} was built on a map with sha256 {built_on[:12]}, but "
                         f"{ply} has {ply_entry['sha256'][:12]}: rebuild the field")
    splats = load_gaussians(ply, device=dev)
    N = int(splats["means"].shape[0])
    if N != int(ckpt["field"]["N"]):
        raise SystemExit(f"{ply}: {N} Gaussians, but the field has {ckpt['field']['N']}")
    opt = {} if align is None else dict(align=align)
    map_ds = make_dataset("nor", data, stride=int(cfg["stride"]), part="map", **opt)
    upd_stride = int(args.update_stride or cfg["stride"])
    upd_ds = make_dataset("nor", data, stride=upd_stride, part="update", **opt)
    kf = (float(cfg.get("kf_trans", 0.05)), float(cfg.get("kf_rot", 5.0)))
    map_idx = select_keyframes(map_ds, *kf, int(cfg.get("max_keyframes") or 0))
    saved_kf = (ckpt.get("timing") or {}).get("keyframes")
    if saved_kf is not None and int(saved_kf) != len(map_idx):
        raise SystemExit(f"{len(map_idx)} mapping keyframes here, the field used {saved_kf}")
    upd_idx = select_keyframes(upd_ds, *kf, 0)
    if args.max_update_keyframes:
        upd_idx = evenly(upd_idx, args.max_update_keyframes)
    clip = load_extractor(cfg, dev)
    check_extractor(clip, ckpt)
    factor = int(cfg["downscale"])
    map_views = read_views(map_ds, map_idx, factor, clip)
    upd_views = read_views(upd_ds, upd_idx, factor, clip)
    del clip                                                    # tokens are on the CPU
    f0 = DirectionalSemanticField.from_state(ckpt["field"], device=dev)
    frozen = dict(colors=splats["colors"].float(), mu=f0.mu_post().float())
    del f0
    chunk = int(args.channel_chunk or cfg.get("channel_chunk") or 32)
    cs = ckpt["field"]
    return dict(ckpt=ckpt, cfg=cfg, device=dev, splats=splats, N=N, d=int(cs["d"]),
                pca=StreamingPCA.from_state(ckpt["pca"], device=dev), zbar=cs.get("zbar"),
                frozen=frozen, rasterizer=rasterizer_factory(args.backend, chunk),
                ply=ply, ply_entry=ply_entry, data=data, align=align, map_ds=map_ds,
                map_views=map_views, upd_views=upd_views, upd_stride=upd_stride, kf=kf,
                map_idx=map_idx, upd_idx=upd_idx, t_load=time.time() - t0)


def ground_truth(ctx, args):
    """D2 labels (bool (N,) tensors stale, in_box, near_after), the any-confidence variant,
    the static-shell gate and the after-removal point counts."""
    corners = read_box_corners(ctx["map_ds"].box_path)
    d2 = args.inflate + args.near
    P, C, info = update_depth_points(ctx["data"], ctx["align"], corners,
                                     max(d2, SHELL[1] + args.near))
    near_box, high = points_in_box(P, corners, d2), C >= HIGH_CONFIDENCE
    info.update(n_near_box=int((near_box & high).sum()),
                n_near_box_any_confidence=int(near_box.sum()), margin_m=d2)
    means = ctx["splats"]["means"].cpu().numpy()
    gt = stale_ground_truth(means, corners, P[near_box & high], args.inflate, args.near)
    gt_any = stale_ground_truth(means, corners, P[near_box], args.inflate, args.near)
    shell = static_shell(ctx["data"], corners, P[high], args.near)
    t = lambda a: torch.from_numpy(a)
    return ({k: t(v) for k, v in gt.items()}, {k: t(v) for k, v in gt_any.items()}, shell,
            info)


def evaluate(args):
    """Every Table II score and metric of one scene, as a JSON-ready dict."""
    dev = args.device
    held = 0
    if str(dev).startswith("cuda"):
        from usplat import gpu_reserve                       # fraction from scripts/gpu_gate.sh
        held = gpu_reserve.apply(dev)
    t = {}
    ctx = load_context(args)
    t["load_extract_s"] = ctx["t_load"]
    ckpt, cfg, N, q = ctx["ckpt"], ctx["cfg"], ctx["N"], args.change_quantile
    k = int(args.k or ckpt["field"]["config"]["change_persist"])
    t0 = time.time()
    replay, null, seg = replay_mapping(ctx, ctx["map_views"], k, len(ctx["upd_views"]))
    check = replay_check(replay, ckpt["field"])
    del replay
    if check["max_rel_dW1"] > 1e-2:
        raise SystemExit(f"the mapping replay does not reproduce the saved field: {check}")
    if not check["ok"]:
        warnings.warn(f"the mapping replay differs slightly from the saved field: {check}")
    thr = {r: nl.threshold(q) for r, nl in null.items()}
    del null
    t["replay_s"] = time.time() - t0
    t0 = time.time()
    upd = run_update(ctx, ctx["upd_views"], k, thr, q)
    t["update_s"] = time.time() - t0

    t0 = time.time()
    gt, gt_any, shell, after_info = ground_truth(ctx, args)
    stale, in_box = gt["stale"], gt["in_box"]
    pop, seen = upd["observed"], upd["seen"]
    t["ground_truth_s"] = time.time() - t0

    scores = dict(upd["scores"])
    scores["fused"] = fuse_rank([scores["ours"], scores["depth"]], mask=pop)
    y = stale[pop]
    gauss, gauss_common = ranking_table({r: scores[r][pop] for r in ROWS}, y, ROWS)
    extra_tab = {name: ranking(s[pop], y) for name, s in upd["extra"].items()}
    common = pop.clone()
    for r in TESTED:
        common &= upd["tests"][r] > 0
    ops = operating_points(upd["flags"], stale, pop, common)
    rates = realised_rates(upd["tests"], upd["fires"], stale, pop)
    for r in ops:
        ops[r].update(rates[r])
        if r in thr:
            ops[r].update(threshold=thr[r]["pooled"], threshold_by_history=thr[r]["by_history"],
                          n_null=thr[r]["n"], n_null_by_history=thr[r]["n_by_history"])
    th = upd["ours_threshold"]
    ops["ours"].update(threshold=th["pooled"], threshold_by_history=th["by_history"])
    ops["fused"] = dict(fa=None, rec=None, fa_common=None, rec_common=None,
                        rule=RULES["fused"])
    st, sp, ss = pop & stale, pop & ~stale, seen & stale
    ours = dict(kept=share(upd["robust_end"][st]), kept_after_mapping=share(upd["robust_map"][st]),
                n_kept=int(upd["robust_end"][st].sum()),
                kept_seen=share(upd["robust_end"][ss]),
                kept_seen_after_mapping=share(upd["robust_map"][ss]),
                changed_end=dict(fa=share(upd["changed_end"][sp]), rec=share(upd["changed_end"][st])),
                n_resets=upd["n_resets"], threshold=th)
    y_any = gt_any["stale"][pop]
    any_rows = {r: ranking(scores[r][pop], y_any) for r in ROWS}
    any_conf = dict(n_stale=int(gt_any["stale"].sum()), n_stale_observed=int(y_any.sum()),
                    n_in_box_near_after=int(gt_any["near_after"].sum()),
                    rows={r: dict(ap=v["ap"], auroc=v["auroc"]) for r, v in any_rows.items()})

    t0 = time.time()
    pixel = pixel_level(ctx, ctx["upd_views"], scores, ROWS, args.tau_cov, args.undef_frac)
    t["pixel_s"] = time.time() - t0

    counts = dict(n_gaussians=N, n_observed=int(pop.sum()), n_unobserved=int((~pop).sum()),
                  n_seen=int(seen.sum()), n_seen_uncovered=int((seen & ~pop).sum()),
                  n_in_box=int(in_box.sum()), n_in_box_observed=int((in_box & pop).sum()),
                  n_in_box_near_after=int(gt["near_after"].sum()),
                  n_stale=int(stale.sum()), n_stale_observed=int(st.sum()),
                  n_stale_unobserved=int((stale & ~pop).sum()), n_static_observed=int(sp.sum()),
                  n_stale_seen=int(ss.sum()), n_stale_seen_uncovered=int((ss & ~pop).sum()),
                  n_common_tested=int(common.sum()),
                  n_common_tested_stale=int((common & stale).sum()),
                  update_views_stale=count_stats(upd["n_views"], st),
                  update_views_static=count_stats(upd["n_views"], sp), after_removal=after_info)
    align = None
    if ctx["align"] is not None:
        with open(ctx["align"]) as f:
            rec = json.load(f)
        align = dict(file_entry(ctx["align"]), scene=rec.get("scene"), T=rec.get("T"),
                     translation_m=rec.get("translation_m"), rotation_deg=rec.get("rotation_deg"),
                     before=rec.get("before"), after=rec.get("after"))
    peak = (torch.cuda.max_memory_allocated() / 2 ** 20 if str(dev).startswith("cuda")
            and torch.cuda.is_available() else None)
    fc = ckpt["field"]["config"]
    return dict(
        scene=os.path.basename(os.path.normpath(ctx["data"])), rows=list(ROWS), inputs=INPUTS,
        gaussian=dict(rows=gauss, common=gauss_common, extra=extra_tab),
        pixel=pixel, operating_point=ops, ours=ours, counts=counts,
        ground_truth=dict(static_shell=shell, any_confidence=any_conf),
        thresholds=dict(rows=thr, quantile=q, ours=th, normlift_segment=seg),
        replay_check=check,
        views=dict(mapping=[v["index"] for v in ctx["map_views"]],
                   update=[v["index"] for v in ctx["upd_views"]],
                   n_mapping=len(ctx["map_views"]), n_update=len(ctx["upd_views"]),
                   grid_hw=[ctx["map_views"][0]["h"], ctx["map_views"][0]["w"]]),
        settings=dict(change_quantile=q, k=k, tau_cov=args.tau_cov, undef_frac=args.undef_frac,
                      inflate_m=args.inflate, near_m=args.near, update_stride=ctx["upd_stride"],
                      mapping_stride=int(cfg["stride"]), kf_trans=ctx["kf"][0],
                      kf_rot=ctx["kf"][1], max_keyframes=cfg.get("max_keyframes"),
                      max_update_keyframes=args.max_update_keyframes,
                      downscale=int(cfg["downscale"]), extractor=cfg.get("extractor"),
                      backend=args.backend, device=str(dev),
                      coverage=dict(rule="change-stream view weight >= fast_min",
                                    fast_min=fc.get("fast_min"),
                                    fast_memory=fc.get("fast_memory"),
                                    view_mass_frac=fc.get("view_mass_frac"),
                                    change_weight=fc.get("change_weight", "view")),
                      history_edges=list(HIST_EDGES), min_null=MIN_NULL,
                      shell_m=list(SHELL), shell_steps=list(SHELL_STEP),
                      field_config=fc),
        alignment=align,
        provenance=dict(field=file_entry(args.field), ply=ctx["ply_entry"],
                        align=align_entry(ctx["align"]),
                        data=os.path.abspath(ctx["data"]), eval_code=code_state(),
                        field_code=(ckpt.get("code") or {}).get("mapping_code_sha256"),
                        mapping_code_now=RM.mapping_code_state()["mapping_code_sha256"],
                        gpu_reserved_mib=held, peak_alloc_mib=peak),
        timing=t)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--field", required=True,
                    help="run_mapping.py output after the mapping pass only (--dataset nor "
                         "--part map --baselines --use-depth)")
    ap.add_argument("--align", required=True,
                    help="the scene's alignment JSON (scripts/align_nor.py, D3), or 'none'")
    ap.add_argument("--ply", default="", help="3DGS map (default: the field's --ply)")
    ap.add_argument("--data", default="", help="scene folder (default: the field's --data)")
    ap.add_argument("--update-stride", type=int, default=0,
                    help="stride inside the update part (0: the field's mapping stride)")
    ap.add_argument("--max-update-keyframes", type=int, default=0,
                    help="evenly spaced subset of the update keyframes (smoke runs; 0 = all)")
    ap.add_argument("--change-quantile", type=float, default=0.999,
                    help="q of every row's mapping-pass per-view scores (operating points)")
    ap.add_argument("--k", type=int, default=0,
                    help="persistence of the residual rows (0: the field's change_persist)")
    ap.add_argument("--tau-cov", type=float, default=0.5, help="pixel coverage floor (alpha)")
    ap.add_argument("--undef-frac", type=float, default=0.5,
                    help="a pixel's score is undefined when more than this share of its alpha "
                         "comes from Gaussians without a score")
    ap.add_argument("--inflate", type=float, default=INFLATE, help="[m] box margin (D2)")
    ap.add_argument("--near", type=float, default=NEAR,
                    help="[m] distance to the after-removal depth (D2)")
    ap.add_argument("--backend", choices=["gsplat", "mock"], default="gsplat")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--channel-chunk", type=int, default=0, help="0: the field's")
    ap.add_argument("--plain-json", action="store_true",
                    help="write plain JSON instead of a usplat.runlog record")
    ap.add_argument("--out", required=True)
    return ap


def _fmt(x, digits=4):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.backend == "gsplat" and not str(args.device).startswith("cuda"):
        raise SystemExit("--backend gsplat needs --device cuda (use --backend mock on the CPU)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.plain_json:
        t0 = time.time()
        with torch.no_grad():
            out = evaluate(args)
        out["command"] = dict(argv=sys.argv if argv is None else ["eval_change_nor.py"] + list(argv),
                              wall_s=time.time() - t0, config=vars(args))
        tmp = f"{args.out}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(to_jsonable(out), f, indent=1)
        os.replace(tmp, args.out)
    else:
        from usplat.runlog import RunRecord
        with RunRecord(args.out, config=vars(args), name="eval_change",
                       argv=[sys.argv[0]] + list(argv if argv is not None else sys.argv[1:])) as rec:
            with torch.no_grad():
                out = evaluate(args)
            for key, value in out.items():
                rec.add(key, value)
            for p in (args.field, out["provenance"]["ply"]["path"]):
                rec.add_input(p)
            if out["alignment"]:
                rec.add_input(out["alignment"]["path"])
    g, o = out["gaussian"]["rows"], out["operating_point"]
    for r in ROWS:
        pr = (out["pixel"].get("rows") or {}).get(r, {})
        print(f"{out['scene']} {r:>10} [{INPUTS[r]:>3}]: G AP {g[r]['ap']:.4f} AUROC "
              f"{g[r]['auroc']:.4f} (NaN {g[r]['n_nan']}) | P AP {pr.get('ap', float('nan')):.4f} "
              f"AUROC {pr.get('auroc', float('nan')):.4f} | FA {_fmt(o[r].get('fa'))} Rec "
              f"{_fmt(o[r].get('rec'))} | common FA {_fmt(o[r].get('fa_common'))} Rec "
              f"{_fmt(o[r].get('rec_common'))} | static rate {_fmt(o[r].get('rate_static'), 5)}")
    c, sh = out["counts"], out["ground_truth"]["static_shell"]
    print(f"{out['scene']}: {c['n_observed']}/{c['n_gaussians']} observed ({c['n_seen']} seen), "
          f"stale {c['n_stale_observed']} observed ({c['n_stale']} total, box-only "
          f"{c['n_in_box']}), common tested {c['n_common_tested']}, Kept {out['ours']['kept']:.4f}"
          f" (seen {out['ours']['kept_seen']:.4f}); static shell within {sh['near_m']} m "
          f"{_fmt(sh['share_near'], 3)}; wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
