"""
dump_nor_qual.py — per-scene dump of the Table III (tab:change) update of one
NeRF-Object-Removal scene, for the qualitative figures (fig:qual-change, fig:qual-query, the
teaser) and the pre-registered fused baselines (configs/nor_fused_prereg.json).

    UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5 scripts/gpu_gate.sh 6500 python3 \
        scripts/dump_nor_qual.py --table <results>/nor/tab2_relmax/val/002.json \
        --tolerance <results>/qual/nor/noise/tolerance.json --vocab configs/vocab_replica.json \
        --text-cache <results>/qual/text_cache_lseg_nor.pt \
        --betas <results>/tab1/eval_val.json --curves --out <results>/qual/nor/002

It replays EXACTLY the Table III evaluation of scripts/eval_change_nor.py (imported, never
edited): load_context, replay_mapping + replay_check, run_update, ground_truth, ranking_table,
operating_points and pixel_level, with every setting (field, map, scene folder, alignment,
update stride, tau_cov, undef_frac, k, change quantile, D2 margins) taken from the scene's
Table III record (--table); the field and map must have the record's sha256 and the frozen
evaluation and mapping code must hash as in the record. Inside this process only it
  * records the two fields run_update builds (ours; "disp", the copy with the change test off
    and the saved NormLift on its hooks) with a subclass of DirectionalSemanticField whose
    from_state keeps every instance, and with --curves wraps residuals_view (run_update calls
    it once per update view, after both fields took the view) to read every Gaussian's top
    class after each view. Identity asserts prove that the captured fields are the ones whose
    scores Table III reports (ours = c_max where c_last is defined, robust_end, dispersion,
    the NormLift drop, ours' flags = its resets);
  * captures the pooled covered pixel scores of pixel_level (by wrapping ranking_table during
    that call): the population of Table III's pixel AP, against which the figures rank.
Both patches are undone in finally blocks, which is asserted.

Reproduction (the stop rule; exit 2): mapping and update keyframe lists identical to the
record, per-frame n_covered / n_pos / n_mask identical, every Table III row's Gaussian- and
pixel-level AP and AUROC (NaN last) within the --tolerance file's value
(scripts/plot_nor_qual.py noise), ours' FA / Rec within 0.5 points.

Fused baselines (configs/nor_fused_prereg.json): X+depth = fuse_rank([s_X, s_depth],
mask=observed) for X in ncc, feat, normlift, dispersion, the rule of the paper's fused row
(eval_change_nor.py: fused = fuse_rank([ours, depth], mask=observed)). They are scored by
SEPARATE ranking_table and pixel_level calls on the 5 fused rows, so Table III's own cells and
common sets are untouched.

Query (N2 rule, never re-prompted): the 101 classes of --vocab with the field extractor's
templates (LSeg: the 5-template ensemble), text from --text-cache (a COPY: eval_seg's
text_embeddings writes new prompts into it; --text-only fills it on the CPU, the GPU run
refuses a missing prompt), projected with the scene's PCA without renormalising. The query
is the most frequent argmax class over the mapping-pass mu* of the D2-stale Gaussians that
the mapping pass saw (raw W1 > 0); ties go to the lower class index.

Frame rule (every NOR panel): among the record's update keyframes, the one with the largest
masks_real area whose bounding box stays at least 2 px inside the raw 192x256 grid (ties to
the lower index); if none qualifies, the keyframe with the most covered mask pixels (the
record's pixel.frames[].n_pos). Display turn: the quarter turns (clockwise) that bring
image-down closest to world +y, from the rule frame's camera.

Outputs in --out (a directory; a finished stage-test dump is never overwritten):
  gauss.pt        per Gaussian (CPU): scores of the 7 Table III rows and the 4 fused
                  baselines, flags, tests, fires, observed, seen, stale, in_box, near_after,
                  n_views, changed_end, robust_end, robust_map, u_epi / u_ale before (saved
                  field) and after (ours), the top class and the query cosine of every
                  feature set (before = mapping mu*, nochange = disp mu*, ours = ours mu*,
                  avg_before / avg_after = averaging's feature), the query, and with
                  --save-features the fp16 features
  frame_<i>.npz   the rule frame(s) at full resolution ("full_*") and on the Table III grid
                  ("grid_*"): photo, map render, alpha, every row's rendered score and its
                  mid-rank among the pooled population, the flag tint, the stale and flag
                  shares, u before / after, and per feature set the top class, the query
                  cosine and p(query) (softmax over the classes with --betas' temperature)
  contrast.json   N2 numbers over every update keyframe on the grid: stale-answer rate
                  (share of covered masks_real pixels whose top class is the query) and
                  delta cos (mean query cosine inside masks_real minus outside)
  fused.json      Gaussian- and pixel-level AP / AUROC of the 5 fused rows (+ common sets)
  fa_frames.json  per update keyframe: covered pixels outside masks_real where the rendered
                  share of ours-flagged Gaussians is >= 0.5 (Fig. F's false-alarm rule)
  curves.json     (--curves) per update view: share of the stale observed Gaussians whose top
                  class is the query, and of the static observed ones whose top class changed
                  since the mapping pass, for ours and averaging
  meta.json       argv, wall time, peak memory, sha256 of this script, the code states, the
                  field / map sha256, the reproduction deltas, replay_check, selection
Backends: gsplat (GPU) or mock (CPU, tests/test_dump_nor_qual.py).
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import eval_change_nor as EC                                                # noqa: E402
import eval_seg as ES                                                       # noqa: E402
from usplat import baselines as B                                           # noqa: E402
from usplat.change_baselines import fuse_rank                               # noqa: E402
from usplat.datasets import _imread                                         # noqa: E402
from usplat.rows import TensorRow                                           # noqa: E402
from usplat.seg_eval import render_row, TIE_TOL                             # noqa: E402
from usplat.vmf_field import DirectionalSemanticField                       # noqa: E402

NEW_FUSED = ("ncc+depth", "feat+depth", "normlift+depth", "dispersion+depth")
FUSED_ROWS = ("fused",) + NEW_FUSED
ALL_ROWS = tuple(EC.ROWS) + NEW_FUSED
MARGIN = 2                         # px: the rule frame's mask box stays this far inside
MASK_HW = (192, 256)               # raw masks_real grid (NeRFObjectRemovalDataset.DEPTH_HW)
FA_SHARE = 0.5                     # Fig. F: rendered share of ours-flagged Gaussians
POINTS = 0.005                     # ours' FA / Rec reproduction: 0.5 points
UNSEEN = 255                       # top class of a Gaussian with no feature
# Fig. N1 (h): per-Gaussian categories (Okabe-Ito), composited like the map colours; flagged
# (= reset by our test) in orange as the "reset by test" overlay of figs/pipeline_v2.pdf
TINT = dict(flagged_stale=(0xE6, 0x9F, 0x00), stale_missed=(0x56, 0xB4, 0xE9),
            flagged_static=(0xD5, 0x5E, 0x00), unobserved=(0x40, 0x40, 0x40))
SETS = ("before", "nochange", "ours", "avg_before", "avg_after")
SET_BETA = dict(before="ours_P0", nochange="ours_P0", ours="ours_P0",
                avg_before="averaging", avg_after="averaging")


# ------------------------------------------------------------------------------- utils
def sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def record_body(rec):
    """The metrics of a Table III record (usplat.runlog record or --plain-json)."""
    return rec["metrics"] if isinstance(rec.get("metrics"), dict) else rec


def git_state():
    try:
        head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
        st = subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True,
                            text=True, timeout=10).stdout.splitlines()
        return dict(commit=head or None, tracked_changes=[s for s in st if not s.startswith("??")],
                    untracked=[s[3:] for s in st if s.startswith("??")])
    except Exception as e:                                       # provenance only
        return dict(error=str(e))


def nan_equal(a, b):
    """Equal tensors with NaN == NaN."""
    a, b = a.detach().cpu(), b.detach().cpu()
    if a.shape != b.shape:
        return False
    if a.dtype.is_floating_point or b.dtype.is_floating_point:
        a, b = a.double(), b.double()
        return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())
    return bool(torch.equal(a, b))


# ------------------------------------------------------------------------- selection
def pick_frame(update_idx, mask_of, n_pos=None, margin=MARGIN):
    """The NOR frame rule. update_idx: the record's update keyframes; mask_of(i) -> bool
    (192, 256) masks_real or None; n_pos: {index: covered mask pixels} for the fallback.
    Returns (index, info)."""
    rows = []
    for i in update_idx:
        mk = mask_of(i)
        if mk is None:
            continue
        mk = np.asarray(mk, dtype=bool)
        if not mk.any():
            continue
        ys, xs = np.nonzero(mk)
        H, W = mk.shape
        touch = bool(ys.min() < margin or xs.min() < margin or ys.max() >= H - margin
                     or xs.max() >= W - margin)
        rows.append(dict(index=int(i), area_px=int(mk.sum()), area=float(mk.mean()),
                         touch=touch))
    inside = [r for r in rows if not r["touch"]]
    if inside:
        best = max(inside, key=lambda r: (r["area_px"], -r["index"]))
        return best["index"], dict(rule="largest masks_real inside the frame",
                                   n_keyframes=len(update_idx), n_with_mask=len(rows),
                                   n_inside=len(inside), **best)
    if not n_pos:
        raise ValueError("no keyframe has a mask inside the frame and no n_pos fallback")
    i = max(n_pos, key=lambda j: (n_pos[j], -j))
    return int(i), dict(rule="fallback: most covered mask pixels", n_keyframes=len(update_idx),
                        n_with_mask=len(rows), n_inside=0, index=int(i), n_pos=int(n_pos[i]))


def quarter_turns(c2w, up=(0.0, 1.0, 0.0)):
    """Clockwise quarter turns k (np.rot90(img, -k)) that bring image-down closest to the
    world direction `up` (+y: gravity in the NOR captures). After k turns image-down is the
    camera's +y, +x, -y, -x axis for k = 0, 1, 2, 3."""
    R = np.asarray(c2w, dtype=np.float64)[:3, :3]
    down = [R[:, 1], R[:, 0], -R[:, 1], -R[:, 0]]
    g = np.asarray(up, dtype=np.float64)
    return int(np.argmax([float(d @ g) for d in down]))


def query_class(mu, seen, stale, text, chunk=1 << 18):
    """The N2 query: the most frequent argmax class (over text (K, d)) of mu over the stale
    Gaussians the mapping pass saw. Returns (index, counts (K,))."""
    sel = torch.nonzero(stale.to(mu.device) & seen.to(mu.device)).squeeze(1)
    K = text.shape[0]
    counts = torch.zeros(K, dtype=torch.long)
    t = text.to(mu.device, mu.dtype)
    for s in range(0, sel.numel(), chunk):
        lab = (mu[sel[s:s + chunk]] @ t.T).argmax(1)
        counts += torch.bincount(lab.cpu(), minlength=K)
    if int(counts.sum()) == 0:
        raise ValueError("no stale Gaussian with mapping evidence: no query")
    top = int(torch.argmax(counts))                       # first max = lower class index
    return top, counts


@torch.no_grad()
def top_class(feat, text, chunk=1 << 18):
    """(N,) uint8 argmax class of feat @ text.T (UNSEEN where the feature is 0). For mu* the
    direction of R or of averaging's F gives the same argmax (positive per-row scaling)."""
    N = feat.shape[0]
    out = torch.empty(N, dtype=torch.uint8)
    t = text.to(feat.device, feat.dtype)
    for s in range(0, N, chunk):
        f = feat[s:s + chunk]
        lab = (f @ t.T).argmax(1)
        lab = torch.where(f.abs().sum(1) > 0, lab, torch.full_like(lab, UNSEEN))
        out[s:s + chunk] = lab.to(torch.uint8).cpu()
    return out


def midrank(sorted_pop, x):
    """Mid-rank in [0, 1] of the values x among the sorted population (NaN stays NaN)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    n = len(sorted_pop)
    if n == 0 or not ok.any():
        return out
    lo = np.searchsorted(sorted_pop, x[ok], side="left")
    hi = np.searchsorted(sorted_pop, x[ok], side="right")
    out[ok] = (lo + 0.5 * (hi - lo)) / n
    return out


# --------------------------------------------------------------------------- capture
@contextlib.contextmanager
def capture_fields(per_view=None):
    """Within the block, EC.run_update builds its fields with a subclass of
    DirectionalSemanticField whose from_state appends each instance to the yielded list;
    per_view(made) runs after every EC.residuals_view call (once per update view, after
    ours and disp ingested it). Both patches are undone on exit."""
    made = []
    orig_cls, orig_res = EC.DirectionalSemanticField, EC.residuals_view
    assert orig_cls is DirectionalSemanticField, "EC.DirectionalSemanticField is patched"

    class Captured(orig_cls):
        @classmethod
        def from_state(cls, s, device="cpu"):
            f = super(Captured, cls).from_state(s, device)
            made.append(f)
            return f

    def residuals(*a, **kw):
        out = orig_res(*a, **kw)
        per_view(made)
        return out

    EC.DirectionalSemanticField = Captured
    if per_view is not None:
        EC.residuals_view = residuals
    try:
        yield made
    finally:
        EC.DirectionalSemanticField = orig_cls
        EC.residuals_view = orig_res
    assert EC.DirectionalSemanticField is orig_cls and EC.residuals_view is orig_res


@contextlib.contextmanager
def capture_ranking():
    """Within the block, every EC.ranking_table call's (scores, y) is appended to the list."""
    calls = []
    orig = EC.ranking_table

    def rt(scores, y, rows):
        calls.append(dict(scores={r: s.detach().cpu() for r, s in scores.items()},
                          y=y.detach().cpu(), rows=list(rows)))
        return orig(scores, y, rows)

    EC.ranking_table = rt
    try:
        yield calls
    finally:
        EC.ranking_table = orig
    assert EC.ranking_table is orig


def check_capture(made, upd, nl_saved_r):
    """The identity asserts: the captured fields are the ones whose scores Table III reports."""
    if len(made) != 2:
        raise AssertionError(f"run_update built {len(made)} fields, expected 2 (ours, disp)")
    ours, disp = made
    if not ours.cfg.change_test or disp.cfg.change_test:
        raise AssertionError("captured fields: ours must test, disp must not")
    hooks = [h for h in disp.view_hooks if isinstance(getattr(h, "__self__", None), B.NormLift)]
    if len(hooks) != 1 or len(disp.view_hooks) != 1 or ours.view_hooks:
        raise AssertionError("disp must hold exactly the saved NormLift, ours no hook")
    nl = hooks[0].__self__
    nan = torch.full_like(ours.c_last, float("nan"))
    obs = upd["observed"].to(ours.device)
    checks = dict(
        ours=nan_equal(upd["scores"]["ours"],
                       torch.where(torch.isfinite(ours.c_last), ours.c_max, nan)),
        robust_end=nan_equal(upd["robust_end"], ours.robust_mask()),
        changed_end=nan_equal(upd["changed_end"], ours.changed),
        dispersion=nan_equal(upd["scores"]["dispersion"],
                             torch.where(obs, disp.aleatoric()[0], nan)),
        normlift=nan_equal(upd["scores"]["normlift"],
                           torch.where(obs, nl_saved_r - nl.reliability(), nan)),
        flags_are_resets=nan_equal(upd["flags"]["ours"], ours.n_resets > 0))
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise AssertionError(f"capture identity asserts failed: {bad}")
    return ours, disp, nl, checks


# ---------------------------------------------------------------------- reproduction
def reproduction(m, views, pixel, gauss, ops, tol):
    """Deltas against the Table III record m and the stop decision."""
    fails, d = [], dict(tolerance=tol)
    d["views_equal"] = dict(mapping=views["mapping"] == m["views"]["mapping"],
                            update=views["update"] == m["views"]["update"])
    for k, ok in d["views_equal"].items():
        if not ok:
            fails.append(f"{k} keyframes differ")
    key = lambda f: (f["index"], f["n_covered"], f["n_pos"], f["n_mask"])
    mine, theirs = [key(f) for f in pixel.get("frames", [])], [key(f) for f in m["pixel"]["frames"]]
    d["frames_equal"] = mine == theirs
    if not d["frames_equal"]:
        bad = [(a, b) for a, b in zip(mine, theirs) if a != b][:5]
        fails.append(f"per-frame n_covered / n_pos differ (first {bad}; {len(mine)} vs "
                     f"{len(theirs)} frames)")
    cells, worst = {}, 0.0
    for lvl, mine_rows, rec_rows in (("gaussian", gauss, m["gaussian"]["rows"]),
                                     ("pixel", pixel["rows"], m["pixel"]["rows"])):
        for r in EC.ROWS:
            for met in ("ap", "auroc"):
                a, b = float(mine_rows[r][met]), float(rec_rows[r][met])
                dd = abs(a - b) if math.isfinite(a) and math.isfinite(b) else (
                    0.0 if (math.isnan(a) and math.isnan(b)) else float("inf"))
                cells[f"{lvl}/{r}/{met}"] = dict(rerun=a, record=b, delta=dd)
                worst = max(worst, dd)
                if dd > tol:
                    fails.append(f"{lvl} {r} {met}: |{a:.5f} - {b:.5f}| > {tol}")
            for met in ("common",):
                a, b = mine_rows[r][met]["ap"], rec_rows[r][met]["ap"]
                cells[f"{lvl}/{r}/common_ap"] = dict(rerun=a, record=b, delta=abs(a - b))
    op = {}
    for r in EC.TESTED:
        for met in ("fa", "rec"):
            a, b = float(ops[r][met]), float(m["operating_point"][r][met])
            op[f"{r}/{met}"] = dict(rerun=a, record=b, delta=abs(a - b))
    for met in ("fa", "rec"):
        if op[f"ours/{met}"]["delta"] > POINTS:
            fails.append(f"ours {met}: |{op[f'ours/{met}']['rerun']:.5f} - "
                         f"{op[f'ours/{met}']['record']:.5f}| > {POINTS}")
    d.update(cells=cells, operating_point=op, max_ap_auroc_delta=worst,
             max_operating_point_delta=max(v["delta"] for v in op.values()),
             failures=fails, ok=not fails)
    return d


# --------------------------------------------------------------------------- renders
def _covered(alpha, tau_cov):
    return (alpha >= tau_cov) & (alpha > EC.ALPHA_EPS)


@torch.no_grad()
def class_maps(F, cov, text, qi, beta, chunk=1 << 16):
    """Per pixel of one P0 render F (P, d): top class (-1: not covered or tied, as
    seg_eval.evaluate_view), the query cosine F.t_q / |F| (NaN where F = 0) and p(query) =
    softmax over the classes of beta * F.t (NaN where not covered)."""
    P = F.shape[0]
    t = text.to(F.device).double()
    pred = torch.full((P,), -1, dtype=torch.long)
    pq = torch.full((P,), float("nan"), dtype=torch.float64)
    for s in range(0, P, chunk):
        f = F[s:s + chunk].double()
        sc = f @ t.T
        p = torch.softmax(beta * sc, dim=1)
        conf, top = p.max(1)
        tied = (p >= conf[:, None] - TIE_TOL).sum(1) > 1
        c = cov[s:s + chunk].to(F.device)
        pred[s:s + chunk] = torch.where(c & ~tied, top, torch.full_like(top, -1)).cpu()
        pq[s:s + chunk] = torch.where(c, p[:, qi], torch.full_like(conf, float("nan"))).cpu()
    n = F.double().norm(dim=1)
    cos = torch.where(n > 0, (F.double() @ t[qi]) / n.clamp_min(1e-30),
                      torch.full_like(n, float("nan"))).cpu()
    return pred, cos, pq


def tint_colors(colors, flags, stale, observed):
    """(N, 3) per-Gaussian colours of Fig. N1 (h)."""
    c = colors.float().clone() * 0.5
    tint = {k: torch.tensor(v, dtype=torch.float32) / 255.0 for k, v in TINT.items()}
    dev = c.device
    f, s, o = flags.to(dev), stale.to(dev), observed.to(dev)
    c[o & s & f] = tint["flagged_stale"].to(dev)
    c[o & s & ~f] = tint["stale_missed"].to(dev)
    c[o & ~s & f] = tint["flagged_static"].to(dev)
    c[~o] = tint["unobserved"].to(dev)
    return c


@torch.no_grad()
def render_frame(ctx, c2w, K, w, h, G, scores, pops, text, qi, betas, tau_cov, undef_frac):
    """Every map of one camera (see the module docstring), as numpy arrays (h, w, ...)."""
    dev, N = ctx["device"], ctx["N"]
    viewmat = torch.linalg.inv(c2w.double()).float().to(dev)
    ras = ctx["rasterizer"](ctx["splats"], viewmat, K.float().to(dev), int(w), int(h))
    out = {}
    img = lambda x: x.reshape(h, w, *x.shape[1:]).cpu().numpy()
    col = ctx["splats"]["colors"].float()
    rgb = ras(torch.cat([col, torch.ones_like(col[:, :1])], 1))
    out["map_rgb"] = img(rgb[:, :3].clamp(0, 1))
    row = TensorRow("change", "feature", buf=torch.zeros(N, 1, device=dev),
                    unc={r: scores[r].to(dev).float() for r in ALL_ROWS})
    rr = render_row(row, ras, alpha_eps=EC.ALPHA_EPS, undef_frac=undef_frac)
    cov = _covered(rr.alpha, tau_cov)
    out["alpha"], out["covered"] = img(rr.alpha), img(cov)
    for r in ALL_ROWS:
        v = rr.unc[r].cpu().numpy()
        out[f"score/{r}"] = v.reshape(h, w)
        if r in pops:
            out[f"rank/{r}"] = midrank(pops[r], v).reshape(h, w)
    shares = {"stale": G["stale"].float(), "flag_ours": G["flags"]["ours"].float(),
              "u_epi_before": G["u_epi0"], "u_ale_before": G["u_ale0"],
              "u_epi_after": G["u_epi1"], "u_ale_after": G["u_ale1"]}
    row = TensorRow("shares", "feature", buf=torch.zeros(N, 1, device=dev),
                    unc={k: v.to(dev).float() for k, v in shares.items()})
    rs = render_row(row, ras, alpha_eps=EC.ALPHA_EPS, undef_frac=undef_frac)
    for k in shares:
        out[f"share/{k}"] = img(rs.unc[k])
    tint = tint_colors(col, G["flags"]["ours"], G["stale"], G["observed"])
    t = ras(torch.cat([tint, torch.ones_like(tint[:, :1])], 1))
    out["tint"] = img(t[:, :3].clamp(0, 1))
    for name in SETS:
        F = ras(G["feat"][name].to(dev).float())
        pred, cos, pq = class_maps(F, cov, text, qi, betas[SET_BETA[name]])
        out[f"pred/{name}"], out[f"cos/{name}"] = img(pred), img(cos)
        out[f"pq/{name}"] = img(pq)
    return out


@torch.no_grad()
def contrast_and_fa(ctx, views, G, text, qi, tau_cov, undef_frac):
    """N2 numbers and Fig. F's false-alarm listing over every update keyframe on the grid."""
    dev, N = ctx["device"], ctx["N"]
    acc = {s: dict(n_mask_cov=0, n_query_in_mask=0, sum_in=0.0, n_in=0, sum_out=0.0, n_out=0)
           for s in SETS}
    fa, no_mask = [], 0
    for v in views:
        if v["mask_real"] is None:
            no_mask += 1
            continue
        ras, _ = EC.rasterizer(ctx, v)
        row = TensorRow("flag", "feature", buf=torch.zeros(N, 1, device=dev),
                        unc={"flag": G["flags"]["ours"].to(dev).float()})
        rr = render_row(row, ras, alpha_eps=EC.ALPHA_EPS, undef_frac=undef_frac)
        cov = _covered(rr.alpha, tau_cov).cpu()
        gt = EC._nearest(v["mask_real"], v["h"], v["w"]).reshape(-1).bool().cpu()
        flag = (rr.unc["flag"] >= FA_SHARE).cpu() & cov
        fa.append(dict(index=v["index"], n_covered=int(cov.sum()),
                       n_outside_flagged=int((flag & ~gt).sum()),
                       n_inside_flagged=int((flag & gt).sum()), n_pos=int((gt & cov).sum())))
        for name in SETS:
            F = ras(G["feat"][name].to(dev).float())
            pred, cos, _ = class_maps(F, cov, text, qi, 1.0)
            a = acc[name]
            inm, outm = cov & gt, cov & ~gt
            a["n_mask_cov"] += int(inm.sum())
            a["n_query_in_mask"] += int((inm & (pred == qi)).sum())
            ci, co = cos[inm], cos[outm]
            ci, co = ci[torch.isfinite(ci)], co[torch.isfinite(co)]
            a["sum_in"] += float(ci.sum()); a["n_in"] += int(ci.numel())
            a["sum_out"] += float(co.sum()); a["n_out"] += int(co.numel())
    res = {}
    for name, a in acc.items():
        mi = a["sum_in"] / a["n_in"] if a["n_in"] else float("nan")
        mo = a["sum_out"] / a["n_out"] if a["n_out"] else float("nan")
        res[name] = dict(a, stale_answer_rate=(a["n_query_in_mask"] / a["n_mask_cov"]
                                               if a["n_mask_cov"] else float("nan")),
                         mean_cos_in=mi, mean_cos_out=mo, delta_cos=mi - mo)
    return res, dict(frames=fa, n_frames_no_mask=no_mask, share=FA_SHARE)


# ------------------------------------------------------------------------------ main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", required=True, help="the scene's Table III record (JSON)")
    ap.add_argument("--tolerance", default="",
                    help="tolerance JSON (plot_nor_qual.py noise); required unless --smoke")
    ap.add_argument("--vocab", required=True, help="configs/vocab_replica.json (101 classes)")
    ap.add_argument("--text-cache", required=True, help="a COPY of the LSeg text cache")
    ap.add_argument("--betas", default="", help="eval_seg fit with ours_P0 and averaging betas "
                                                "(tab1/eval_val.json); needed unless --text-only")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--stage", choices=["val", "test", "toy"], required=True)
    ap.add_argument("--frames", default="rule", help="'rule' or update frame numbers i,j")
    ap.add_argument("--curves", action="store_true", help="per-view top-class curves")
    ap.add_argument("--save-features", action="store_true", help="fp16 features in gauss.pt")
    ap.add_argument("--no-full", action="store_true", help="skip the full-resolution renders")
    ap.add_argument("--text-only", action="store_true",
                    help="fill --text-cache with the vocabulary's prompts on the CPU and exit")
    ap.add_argument("--smoke", type=int, default=0,
                    help="evenly spaced subset of N update keyframes; no reproduction check")
    ap.add_argument("--backend", choices=["gsplat", "mock"], default="gsplat")
    ap.add_argument("--device", default="cuda")
    return ap


def ec_namespace(m, args):
    """eval_change_nor's arguments, every value from the record's settings and provenance."""
    s, p = m["settings"], m["provenance"]
    align = "none" if m.get("alignment") is None else p["align"]["path"]
    argv = ["--field", p["field"]["path"], "--align", align,
            "--update-stride", str(s["update_stride"]), "--tau-cov", repr(s["tau_cov"]),
            "--undef-frac", repr(s["undef_frac"]), "--k", str(s["k"]),
            "--change-quantile", repr(s["change_quantile"]), "--inflate", repr(s["inflate_m"]),
            "--near", repr(s["near_m"]), "--backend", args.backend, "--device", args.device,
            "--max-update-keyframes", str(args.smoke or s.get("max_update_keyframes") or 0),
            "--ply", p["ply"]["path"], "--data", p["data"], "--out", os.devnull]
    ns = EC.build_parser().parse_args(argv)
    for key, attr in (("update_stride", "update_stride"), ("tau_cov", "tau_cov"),
                      ("undef_frac", "undef_frac"), ("k", "k"),
                      ("change_quantile", "change_quantile"), ("inflate_m", "inflate"),
                      ("near_m", "near")):
        if getattr(ns, attr) != s[key]:
            raise SystemExit(f"setting {key}: {getattr(ns, attr)!r} != record {s[key]!r}")
    return ns


def text_args(cfg, path):
    return argparse.Namespace(extractor=cfg.get("extractor") or "lseg", clip_model="",
                              clip_pretrained="", text_cache=path, clip_device="cpu")


def missing_prompts(vocab, targs):
    tpl = ES.extractor_templates(vocab, targs.extractor)
    key = lambda q: f"{targs.extractor}|{'|'.join(tpl)}|{q}"
    cache = torch.load(targs.text_cache, map_location="cpu") if os.path.exists(
        targs.text_cache) else {}
    return [q for q in vocab["prompts"] if key(q) not in cache]


def _store(key, a):
    """Display maps in float16; scores and ranks in float32 (key 'grid_rank/...')."""
    if not isinstance(a, np.ndarray) or a.dtype.kind != "f":
        return a
    keep = key.split("_", 1)[-1].startswith(("rank/", "score/"))
    return a.astype(np.float32 if keep else np.float16)


def write_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(ES._plain_json(obj), f, indent=1)
    os.replace(tmp, path)


def run(args, argv):
    t_start = time.time()
    rec = json.load(open(args.table))
    m = record_body(rec)
    vocab = ES.load_vocab(args.vocab)
    if args.text_only:
        targs = text_args(dict(extractor=m["settings"].get("extractor")), args.text_cache)
        miss = missing_prompts(vocab, targs)
        ES.text_embeddings(vocab, list(range(len(vocab["prompts"]))), targs)
        print(f"text cache {args.text_cache}: {len(miss)} prompts added, "
              f"{len(vocab['prompts'])} in the vocabulary")
        return dict(added=miss)
    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, "meta.json")
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        if old.get("status") == "done" and old.get("stage") == "test":
            raise SystemExit(f"{meta_path}: this test scene was dumped already (run once)")
    if not args.smoke and not args.tolerance:
        raise SystemExit("--tolerance is required (plot_nor_qual.py noise)")
    tol = float(json.load(open(args.tolerance))["tolerance"]) if args.tolerance else None
    if not args.betas:
        raise SystemExit("--betas is required")
    ES.check_fit_settings(ES.load_fit_settings(args.betas),
                          argparse.Namespace(undef_frac=m["settings"]["undef_frac"],
                                             tau_cov=m["settings"]["tau_cov"]), args.betas)
    betas_all = ES.load_betas(args.betas)
    betas = {k: float(betas_all[k]) for k in ("ours_P0", "averaging")}
    ns = ec_namespace(m, args)
    meta = dict(stage=args.stage, status="running", scene=m["scene"], argv=argv,
                start=t_start, script_sha256=sha256_file(os.path.abspath(__file__)),
                plot_script_sha256=(sha256_file(os.path.join(HERE, "plot_nor_qual.py"))
                                    if os.path.exists(os.path.join(HERE, "plot_nor_qual.py"))
                                    else None),
                table=os.path.abspath(args.table), table_sha256=sha256_file(args.table),
                tolerance_file=os.path.abspath(args.tolerance) if args.tolerance else None,
                tolerance=tol, git=git_state(), betas=betas, betas_file=os.path.abspath(args.betas))
    write_json(meta_path, meta)
    # frozen code: hashes as in the record
    code_now = EC.code_state()
    rm_now = EC.RM.mapping_code_state()["mapping_code_sha256"]
    rec_code = m["provenance"].get("eval_code")
    meta["code"] = dict(eval_code=code_now, eval_code_equal=code_now == rec_code,
                        mapping_code=rm_now,
                        mapping_code_equal=rm_now == m["provenance"].get("mapping_code_now"))
    if args.stage != "toy" and not (meta["code"]["eval_code_equal"]
                                    and meta["code"]["mapping_code_equal"]):
        raise SystemExit(f"the frozen code differs from the record's: {meta['code']}")
    field_sha = sha256_file(ns.field)
    if field_sha != m["provenance"]["field"]["sha256"]:
        raise SystemExit(f"{ns.field}: sha256 {field_sha[:12]} != record "
                         f"{m['provenance']['field']['sha256'][:12]}")
    dev = args.device
    held = 0
    if str(dev).startswith("cuda"):
        from usplat import gpu_reserve                       # fraction from scripts/gpu_gate.sh
        held = gpu_reserve.apply(dev)
    T = {}
    ctx = EC.load_context(ns)
    T["load_extract_s"] = ctx["t_load"]
    if ctx["ply_entry"]["sha256"] != m["provenance"]["ply"]["sha256"]:
        raise SystemExit("the map's sha256 differs from the record's")
    ckpt, N, q = ctx["ckpt"], ctx["N"], ns.change_quantile
    views = dict(mapping=[v["index"] for v in ctx["map_views"]],
                 update=[v["index"] for v in ctx["upd_views"]])
    targs = text_args(ctx["cfg"], args.text_cache)
    miss = missing_prompts(vocab, targs)
    if miss:
        raise SystemExit(f"{args.text_cache} lacks {len(miss)} prompts: run --text-only first")
    cache_sha = sha256_file(args.text_cache)
    text = ctx["pca"].project_text(
        ES.text_embeddings(vocab, list(range(len(vocab["prompts"]))), targs).to(dev))
    if sha256_file(args.text_cache) != cache_sha:
        raise SystemExit("the text cache changed during the run")
    meta["text_cache_sha256"] = cache_sha

    # ---- before the update: the saved field (CPU) and the saved NormLift (device)
    f0 = DirectionalSemanticField.from_state(ckpt["field"], device="cpu")
    u_epi0, u_ale0 = f0.epistemic().float(), f0.aleatoric()[0].float()
    seen0 = f0.raw.W1 > 0
    del f0
    e = EC._normlift_entry(ckpt)                       # as EC.make_normlift(saved=True)
    cfg_nl = dict(e["config"])
    dtype_nl = getattr(torch, cfg_nl.pop("dtype", "float32"))
    nl0 = B.NormLift(**cfg_nl, device=dev, dtype=dtype_nl)
    nl0.load_state_dict(e["state"])
    r_map0, f_avg0 = nl0.reliability().clone(), nl0.feature()
    del nl0
    mu0 = ctx["frozen"]["mu"]

    # ---- the Table III passes
    k = int(ns.k or ckpt["field"]["config"]["change_persist"])
    t0 = time.time()
    replay, null, seg = EC.replay_mapping(ctx, ctx["map_views"], k, len(ctx["upd_views"]))
    check = EC.replay_check(replay, ckpt["field"])
    del replay
    if check["max_rel_dW1"] > 1e-2:
        raise SystemExit(f"the mapping replay does not reproduce the saved field: {check}")
    thr = {r: nl.threshold(q) for r, nl in null.items()}
    del null
    T["replay_s"] = time.time() - t0

    labels_steps = []

    def per_view(made):
        ours, disp = made[0], made[1]
        nlv = disp.view_hooks[0].__self__
        labels_steps.append((top_class(ours.raw.R, text), top_class(nlv.F, text)))

    t0 = time.time()
    with capture_fields(per_view if args.curves else None) as made:
        upd = EC.run_update(ctx, ctx["upd_views"], k, thr, q)
    ours, disp, nl, cap_checks = check_capture(made, upd, r_map0)
    T["update_s"] = time.time() - t0
    feats = dict(before=mu0, nochange=disp.mu_post(), ours=ours.mu_post(),
                 avg_before=f_avg0, avg_after=nl.feature())
    u_epi1, u_ale1 = ours.epistemic().float().cpu(), ours.aleatoric()[0].float().cpu()
    if args.curves:                        # the R / F shortcut equals mu* / feature at the end
        end_o, end_a = top_class(feats["ours"], text), top_class(feats["avg_after"], text)
        cap_checks["curve_label_agreement"] = dict(
            ours=float((end_o == labels_steps[-1][0]).double().mean()),
            averaging=float((end_a == labels_steps[-1][1]).double().mean()))
    del ours, disp, nl, made

    t0 = time.time()
    gt, gt_any, shell, after_info = EC.ground_truth(ctx, ns)
    stale, in_box = gt["stale"], gt["in_box"]
    pop, seen = upd["observed"], upd["seen"]
    T["ground_truth_s"] = time.time() - t0
    scores = dict(upd["scores"])
    scores["fused"] = fuse_rank([scores["ours"], scores["depth"]], mask=pop)
    for name in NEW_FUSED:
        x = name.split("+")[0]
        scores[name] = fuse_rank([scores[x], scores["depth"]], mask=pop)
    y = stale[pop]
    gauss, gauss_common = EC.ranking_table({r: scores[r][pop] for r in EC.ROWS}, y, EC.ROWS)
    common = pop.clone()
    for r in EC.TESTED:
        common &= upd["tests"][r] > 0
    ops = EC.operating_points(upd["flags"], stale, pop, common)
    f_gauss, f_gauss_common = EC.ranking_table({r: scores[r][pop] for r in FUSED_ROWS}, y,
                                               FUSED_ROWS)
    t0 = time.time()
    with capture_ranking() as calls:
        pixel = EC.pixel_level(ctx, ctx["upd_views"], scores, EC.ROWS, ns.tau_cov, ns.undef_frac)
    pixel_f = EC.pixel_level(ctx, ctx["upd_views"], scores, FUSED_ROWS, ns.tau_cov,
                             ns.undef_frac)
    T["pixel_s"] = time.time() - t0
    if len(calls) != 1:
        raise AssertionError(f"pixel_level called ranking_table {len(calls)} times")
    pops = {r: np.sort(calls[0]["scores"][r].double().numpy()[
        np.isfinite(calls[0]["scores"][r].double().numpy())]) for r in EC.ROWS}
    pop_check = {r: dict(n_defined=int(len(pops[r])),
                         n_pix_minus_nan=int(pixel["n_pix"] - pixel["rows"][r]["n_nan"]),
                         record_n_pix_minus_nan=int(m["pixel"]["n_pix"]
                                                    - m["pixel"]["rows"][r]["n_nan"]))
                 for r in EC.ROWS}
    for r, c in pop_check.items():
        if c["n_defined"] != c["n_pix_minus_nan"]:
            raise AssertionError(f"rank population of {r}: {c}")
    del calls

    smoke = bool(args.smoke)
    rep = None if smoke else reproduction(m, views, pixel, gauss, ops, tol)
    meta.update(views=views, replay_check=check, capture_checks=cap_checks,
                rank_population=pop_check, reproduction=rep,
                n_resets=upd["n_resets"], gpu_reserved_mib=held)
    if rep is not None and not rep["ok"]:
        meta.update(status="FAIL", wall_s=time.time() - t_start)
        write_json(meta_path, meta)
        print(f"{m['scene']}: REPRODUCTION FAILED: {rep['failures']}")
        return dict(meta=meta, exit=2)

    # ---- query (N2 rule)
    qi, counts = query_class(mu0, seen0, stale, text)
    total = int(counts.sum())
    top3 = [dict(index=int(c), name=vocab["names"][int(c)], share=float(counts[c]) / total)
            for c in torch.argsort(-counts, stable=True)[:3]]
    query = dict(index=qi, name=vocab["names"][qi], prompt=vocab["prompts"][qi], top3=top3,
                 n_stale_seen=total, rule="most frequent argmax over the mapping-pass mu* of "
                                          "the D2-stale Gaussians with W1 > 0")
    meta["query"] = query

    # ---- per-Gaussian dump
    G = dict(scores={r: s.cpu() for r, s in scores.items()}, flags=upd["flags"],
             tests=upd["tests"], fires=upd["fires"], observed=pop, seen=seen, stale=stale,
             in_box=in_box, near_after=gt["near_after"], n_views=upd["n_views"],
             changed_end=upd["changed_end"], robust_end=upd["robust_end"],
             robust_map=upd["robust_map"], u_epi0=u_epi0, u_ale0=u_ale0, u_epi1=u_epi1,
             u_ale1=u_ale1, seen_mapping=seen0, query=query, feat=feats)
    tq = text[qi].to(dev)
    G_out = {k: v for k, v in G.items() if k != "feat"}
    G_out["top_class"] = {s: top_class(feats[s], text) for s in SETS}
    G_out["query_cos"] = {s: (feats[s].float() @ tq.float()).cpu() for s in SETS}
    G_out["vocab_names"] = vocab["names"]
    if args.save_features:
        G_out["features"] = {s: feats[s].half().cpu() for s in SETS}
    torch.save(G_out, os.path.join(args.out, "gauss.pt"))
    del G_out

    # ---- curves
    if args.curves:
        st, sp = pop & stale, pop & ~stale
        lab0 = (top_class(mu0, text), top_class(f_avg0, text))
        steps = [lab0] + labels_steps
        curve = dict(views=[0] + views["update"], query=qi,
                     n_stale_observed=int(st.sum()), n_static_observed=int(sp.sum()))
        for j, name in enumerate(("ours", "averaging")):
            curve[name] = dict(
                stale_is_query=[float((s[j][st] == qi).double().mean()) for s in steps],
                static_changed=[float((s[j][sp] != lab0[j][sp]).double().mean()) for s in steps])
        write_json(os.path.join(args.out, "curves.json"), curve)

    # ---- frames
    rule_n_pos = {f["index"]: f["n_pos"] for f in m["pixel"]["frames"]}
    by_index = {v["index"]: v for v in ctx["upd_views"]}
    if args.frames == "rule":
        fi, sel = pick_frame(views["update"], lambda i: (
            None if by_index[i]["mask_real"] is None else by_index[i]["mask_real"].numpy()),
            rule_n_pos)
        frames = [fi]
    else:
        frames, sel = [int(s) for s in args.frames.split(",")], dict(rule="manual")
    meta["selection"] = sel
    ds = ctx["map_ds"]
    G["feat"] = feats
    t0 = time.time()
    meta["frames"] = {}
    for fi in frames:
        if fi not in by_index:
            raise SystemExit(f"frame {fi} is not an update keyframe of the record")
        v = by_index[fi]
        kq = quarter_turns(v["c2w"].numpy())
        dump = dict(index=fi, quarter_turns=kq, query=qi,
                    mask_real=(np.zeros(MASK_HW, bool) if v["mask_real"] is None
                               else v["mask_real"].numpy().astype(bool)),
                    grid_photo=v["rgb"].reshape(v["h"], v["w"], 3).numpy())
        maps = render_frame(ctx, v["c2w"], v["K"], v["w"], v["h"], G, scores, pops, text, qi,
                            betas, ns.tau_cov, ns.undef_frac)
        dump.update({f"grid_{k}": a for k, a in maps.items()})
        fr = next(f for f in m["pixel"]["frames"] if f["index"] == fi)
        gt_mask = EC._nearest(v["mask_real"], v["h"], v["w"]).numpy().astype(bool)
        n_cov = int(maps["covered"].sum())
        n_pos = int((maps["covered"].reshape(-1) & gt_mask.reshape(-1)).sum())
        meta["frames"][str(fi)] = dict(quarter_turns=kq, grid_hw=[v["h"], v["w"]],
                                       n_covered=n_cov, n_pos=n_pos,
                                       record_n_covered=fr["n_covered"], record_n_pos=fr["n_pos"])
        if not smoke and (n_cov, n_pos) != (fr["n_covered"], fr["n_pos"]):
            raise AssertionError(f"frame {fi}: grid render {n_cov}/{n_pos} != record "
                                 f"{fr['n_covered']}/{fr['n_pos']}")
        if not args.no_full:
            Hf, Wf = ds.image_hw
            Kf = torch.from_numpy(ds.K[fi]).float()
            dump["full_photo"] = _imread(ds.rgb[fi])
            maps = render_frame(ctx, v["c2w"], Kf, Wf, Hf, G, scores, pops, text, qi, betas,
                                ns.tau_cov, ns.undef_frac)
            dump.update({f"full_{k}": a for k, a in maps.items()})
            meta["frames"][str(fi)]["full_hw"] = [int(Hf), int(Wf)]
        np.savez_compressed(os.path.join(args.out, f"frame_{fi}.npz"),
                            **{k: _store(k, a) for k, a in dump.items()})
    T["frames_s"] = time.time() - t0

    # ---- N2 numbers and the false-alarm listing
    t0 = time.time()
    con, fa = contrast_and_fa(ctx, ctx["upd_views"], G, text, qi, ns.tau_cov, ns.undef_frac)
    T["contrast_s"] = time.time() - t0
    write_json(os.path.join(args.out, "contrast.json"),
               dict(scene=m["scene"], stage=args.stage, query=query, sets=con,
                    grid_hw=[ctx["upd_views"][0]["h"], ctx["upd_views"][0]["w"]],
                    rule="pooled over the covered pixels of every update keyframe; top class "
                         "= argmax over the classes of the P0 render (ties: no class)"))
    write_json(os.path.join(args.out, "fa_frames.json"), dict(scene=m["scene"], **fa))
    fused = dict(scene=m["scene"], stage=args.stage, rows=list(FUSED_ROWS),
                 gaussian=dict(rows=f_gauss, common=f_gauss_common),
                 pixel={k: v for k, v in pixel_f.items() if k != "frames"},
                 table3=dict(gaussian=gauss, pixel=pixel["rows"]),
                 prereg=os.path.abspath(os.path.join(REPO, "configs", "nor_fused_prereg.json")),
                 prereg_sha256=sha256_file(os.path.join(REPO, "configs", "nor_fused_prereg.json")))
    write_json(os.path.join(args.out, "fused.json"), fused)
    peak = (torch.cuda.max_memory_allocated() / 2 ** 20 if str(dev).startswith("cuda")
            and torch.cuda.is_available() else None)
    meta.update(status="done", timing=T, wall_s=time.time() - t_start, peak_alloc_mib=peak,
                field_sha256=field_sha, ply_sha256=ctx["ply_entry"]["sha256"],
                counts=dict(n_gaussians=N, n_observed=int(pop.sum()),
                            n_stale_observed=int((pop & stale).sum())))
    write_json(meta_path, meta)
    print(f"{m['scene']}: done; frames {frames}, query {query['name']!r} "
          f"({top3[0]['share']:.3f}), reproduction "
          f"{'skipped (smoke)' if rep is None else 'max AP/AUROC delta %.5f' % rep['max_ap_auroc_delta']}"
          f", wall {meta['wall_s']:.1f} s, peak {peak} MiB")
    return dict(meta=meta, exit=0)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.backend == "gsplat" and not str(args.device).startswith("cuda"):
        raise SystemExit("--backend gsplat needs --device cuda")
    full_argv = [sys.argv[0]] + list(argv if argv is not None else sys.argv[1:])
    with torch.no_grad():
        out = run(args, full_argv)
    if out.get("exit"):
        raise SystemExit(out["exit"])
    return out


if __name__ == "__main__":
    main()
