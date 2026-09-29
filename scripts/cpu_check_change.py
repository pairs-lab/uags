"""
cpu_check_change.py — G1: does the change test separate simulated content changes from
static content on REAL dense CLIP features? (internal check, not a paper result)

Protocol (fixed before running):
  * 7-Scenes <scene>/seq-01 frames 0..790 stride 10 (80 frames); CLIP ViT-B/16 laion2b
    MaskCLIP tokens at full resolution (640x480), upsampled to the accumulation resolution
    160x120 (the pipeline default --downscale 4); PCA-64 and the centre zbar are fit on the
    FIRST-HALF frames only.
  * 3D points: 150 pixels with valid depth from each of frames 0,100,...,700; visible in a
    frame if they project inside the image and the frame's depth agrees within 5 cm
    (7-Scenes depth intrinsics f=585 used for colour too, as in the math audit C3b).
    One observation = the 3x3 pixel window of features at the projected pixel.
  * Keep points with >= 8 observations. Half of them (random) become CHANGED: their
    first-half observations are their own, the second half comes from a partner point
    > 0.5 m away (the partner's own second half). This is a PROXY of rearrangement: a
    partner can carry the same class (wall -> wall), which lowers every score's AUROC.
  * Scores (AUROC changed vs static; higher score = more changed):
      legacy 1 - r_bar   (old code path: pixel evidence, raw channel)
      u_ale              (new: Kish estimator on the centred channel)
      c_max              (new: max over time of 1 - cos(slow, fast), centred channel)
      resid_1            (feature residual: max_v 1 - cos(mean of previous obs, obs_v))
      resid_2            (the same, requiring 2 consecutive views: max_v min(res_v, res_v-1))
    plus fired rates at the default threshold, and AUROC on the "strong" subset where the
    two contents differ (raw cosine of their first-half means below the median).
Usage: PYTHONPATH=. python3 scripts/cpu_check_change.py --scene chess --out log.json
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
import numpy as np
import torch
import torch.nn.functional as Fn

from usplat.clip_dense import OpenCLIPDense, upsample_tokens
from usplat.feature_pca import StreamingPCA, RunningMean
from usplat.vmf_field import DirectionalSemanticField, FieldConfig
from usplat.evaluation import auroc

ROOT = "/home/thanhnc/gauss_ws/src/usgloc/data/7scenes_source"
FX = FY = 585.0
CX, CY = 320.0, 240.0
H, W, DS = 120, 160, 4


def load(scene, i):
    import cv2
    base = f"{ROOT}/{scene}/seq-01/frame-{i:06d}"
    img = cv2.cvtColor(cv2.imread(base + ".color.png"), cv2.COLOR_BGR2RGB)
    dep = cv2.imread(base + ".depth.png", cv2.IMREAD_UNCHANGED).astype(np.float64)
    dep[dep == 65535] = 0
    return img, dep / 1000.0, np.loadtxt(base + ".pose.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="chess")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feat-dim", type=int, default=64, help="0 = no PCA (512-d)")
    ap.add_argument("--field-config", default="", help="JSON overrides of FieldConfig")
    ap.add_argument("--variants", default="",
                    help='JSON {name: {"centre": bool, "cfg": {...}}}: evaluate several field '
                         'variants on the SAME observations (overrides --field-config)')
    ap.add_argument("--cache", default="", help="token cache (.pt) to reuse CLIP features")
    ap.add_argument("--change-frac", type=float, default=0.5,
                    help="fraction of a changed point's observations (the LAST ones) taken from "
                         "its partner; 0.5 = the original half/half proxy")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(16)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    frames = list(range(0, 800, 10))
    half = len(frames) // 2

    data, feats, toks = {}, {}, {}
    if a.cache and os.path.exists(a.cache):
        toks = torch.load(a.cache)
    else:
        clip = OpenCLIPDense("ViT-B-16", "laion2b_s34b_b88k", device="cpu")
    for i in frames:
        img, dep, T = load(a.scene, i)
        data[i] = (dep, T)
        if i not in toks:
            toks[i] = clip.extract_tokens(torch.from_numpy(img)).half()
        feats[i] = upsample_tokens(toks[i], (H, W))
    if a.cache and not os.path.exists(a.cache):
        torch.save(toks, a.cache)
    t_ext = time.time() - t0
    D_CLIP = feats[frames[0]].shape[-1]

    d = a.feat_dim or D_CLIP
    if a.feat_dim:
        pca = StreamingPCA(D_CLIP, a.feat_dim)
        for i in frames[:half]:
            pca.accumulate(feats[i])
        pca.fit()
        proj = lambda z: pca.project(z)
    else:
        proj = lambda z: z
    centre = RunningMean(d)
    for i in frames[:half]:
        centre.add(proj(feats[i].reshape(-1, D_CLIP))[::7])

    # ---- 3D points and their observations --------------------------------------
    pts = []
    for i in range(0, 800, 100):
        dep, T = data[i]
        v, u = np.nonzero(dep > 0.3)
        pick = rng.choice(len(u), 150, replace=False)
        u, v = u[pick], v[pick]
        z = dep[v, u]
        Xc = np.stack([(u - CX) * z / FX, (v - CY) * z / FY, z, np.ones_like(z)])
        pts.append((T @ Xc)[:3].T)
    Xw = np.concatenate(pts)
    obs = [[] for _ in range(len(Xw))]
    Xh = np.concatenate([Xw, np.ones((len(Xw), 1))], 1).T
    for i in frames:
        dep, T = data[i]
        Xi = np.linalg.inv(T) @ Xh
        zi = Xi[2]
        ui, vi = FX * Xi[0] / np.maximum(zi, 1e-6) + CX, FY * Xi[1] / np.maximum(zi, 1e-6) + CY
        ok = (zi > 0.1) & (ui >= 8) & (ui < 632) & (vi >= 8) & (vi < 472)
        uu = np.clip(np.round(ui).astype(int), 0, 639)
        vv = np.clip(np.round(vi).astype(int), 0, 479)
        vis = ok & (dep[vv, uu] > 0) & (np.abs(dep[vv, uu] - zi) < 0.05)
        F = proj(feats[i].reshape(-1, D_CLIP)).reshape(H, W, d)
        for j in np.nonzero(vis)[0]:
            y, x = min(max(vv[j] // DS, 1), H - 2), min(max(uu[j] // DS, 1), W - 2)
            obs[j].append(F[y - 1:y + 2, x - 1:x + 2].reshape(9, d))
    keep = [j for j in range(len(obs)) if len(obs[j]) >= 8]
    n = len(keep)
    changed = np.zeros(n, bool)
    changed[rng.choice(n, n // 2, replace=False)] = True
    seqs, strong = [], np.zeros(n, bool)
    cos_pair = np.full(n, np.nan)
    for a_i, j in enumerate(keep):
        s = obs[j]
        if changed[a_i]:
            cand = [k for k in keep if k != j and np.linalg.norm(Xw[k] - Xw[j]) > 0.5]
            k = cand[rng.integers(len(cand))]
            n_new = max(1, int(round(a.change_frac * len(s))))
            hj = len(s) - n_new                                  # own history kept
            mj = torch.stack([o.mean(0) for o in s[:hj]]).mean(0)
            mk = torch.stack([o.mean(0) for o in obs[k][:len(obs[k]) // 2]]).mean(0)
            cos_pair[a_i] = float(Fn.cosine_similarity(mj, mk, dim=0))
            s = s[:hj] + obs[k][-n_new:]
        seqs.append(s)
    med = np.nanmedian(cos_pair)
    strong = changed & (cos_pair < med)

    # ---- feed the field(s) -------------------------------------------------------
    def run(cfg, use_centre=True):
        f = DirectionalSemanticField(n, d, cfg)
        f.set_centre(centre.mean() if use_centre else None)
        L = max(len(s) for s in seqs)
        f.c_steps = torch.full((L, n), float("nan"))          # c after each step
        for v in range(L):
            ids = [i for i in range(n) if v < len(seqs[i])]
            if not ids:
                continue
            ids_rep = torch.tensor(ids).repeat_interleave(9)
            z = torch.cat([seqs[i][v] for i in ids])
            prev = f.c_last.clone()
            f.update_sparse(ids_rep, torch.ones(len(ids_rep)), z, num_pixels=H * W)
            idt = torch.tensor(ids)
            upd = f.c_last[idt]
            new = torch.isfinite(upd) & (upd != prev[idt])   # computed at this step
            f.c_steps[v, idt] = torch.where(new, upd, torch.full_like(upd, float("nan")))
        return f

    def fieldcal(cfg_dict, use_centre, qs=(0.99, 0.999)):
        """Operating point with the FIELD's own calibration (history-conditioned null):
        a null field sees the full sequences of half of the static points (a static
        'mapping' set), is calibrated, and its thresholds are applied to a second field fed
        with the other static half + all changed points (false alarms measured on points
        the calibration never saw)."""
        rng2 = np.random.default_rng(a.seed + 7)
        static_idx = np.nonzero(~changed)[0]
        rng2.shuffle(static_idx)
        cal, held = set(static_idx[: len(static_idx) // 2].tolist()), static_idx[len(static_idx) // 2:]
        out = {}
        for q in qs:
            def feed(ids_keep, f):
                L = max(len(seqs[i]) for i in ids_keep)
                for v in range(L):
                    ids = [i for i in ids_keep if v < len(seqs[i])]
                    if not ids:
                        continue
                    ids_rep = torch.tensor(ids).repeat_interleave(9)
                    z = torch.cat([seqs[i][v] for i in ids])
                    f.update_sparse(ids_rep, torch.ones(len(ids_rep)), z, num_pixels=H * W)
            c0 = dict(cfg_dict); c0["change_thr"] = float("inf")
            fn = DirectionalSemanticField(n, d, FieldConfig(**c0))
            fn.set_centre(centre.mean() if use_centre else None)
            feed(sorted(cal), fn)
            fn.calibrate_change_threshold(q)
            fe = DirectionalSemanticField(n, d, FieldConfig(**c0))
            fe.set_centre(centre.mean() if use_centre else None)
            fe.thr_by_hist = fn.thr_by_hist.clone() if fn.thr_by_hist is not None else None
            fe.cfg.change_thr = fn.cfg.change_thr
            ev = sorted(set(range(n)) - cal)
            feed(ev, fe)
            fired = (fe.n_resets >= 1).numpy()
            out[str(q)] = dict(thr_pooled=fn.cfg.change_thr,
                               thr_by_hist=None if fn.thr_by_hist is None else fn.thr_by_hist.tolist(),
                               fired_changed=float(fired[changed].mean()),
                               fired_static_heldout=float(fired[held].mean()))
        return out

    def selfcal(f, qs=(0.9, 0.95, 0.99)):
        """Threshold from the PRE-change part of every sequence (no labels): each point's
        max c over its first half is a null sample. Evaluate max c over the second half."""
        L = f.c_steps.shape[0]
        lens = torch.tensor([len(s) for s in seqs])
        hist = (lens - (a.change_frac * lens).round().clamp(min=1)).long()
        hmask = torch.arange(L)[:, None] < hist[None]
        c = f.c_steps
        first = torch.where(hmask, c, torch.full_like(c, float("nan")))
        second = torch.where(~hmask, c, torch.full_like(c, float("nan")))
        null = torch.nan_to_num(first, nan=-1.0).max(0).values
        null = null[null > -1]
        alt = torch.nan_to_num(second, nan=-1.0).max(0).values
        out = {}
        for q in qs:
            thr = float(torch.quantile(null, q))
            fired = alt > thr
            out[str(q)] = dict(thr=thr, fired_changed=float(fired[torch.tensor(changed)].float().mean()),
                               fired_static=float(fired[torch.tensor(~changed)].float().mean()))
        return out

    legacy = run(FieldConfig.legacy(), use_centre=False)
    if a.variants:
        variants = json.loads(a.variants)
    else:
        variants = {"ours": {"centre": True,
                             "cfg": json.loads(a.field_config) if a.field_config else {}}}
    fields = {k: run(FieldConfig(**v.get("cfg", {})), use_centre=v.get("centre", True))
              for k, v in variants.items()}

    # feature residual baselines (raw channel, window means)
    r1, r2 = np.zeros(n), np.zeros(n)
    for i, s in enumerate(seqs):
        m = torch.stack([Fn.normalize(o, dim=-1).mean(0) for o in s])
        res = []
        for v in range(3, len(m)):
            prev = m[:v].mean(0)
            res.append(1 - float(Fn.cosine_similarity(prev, m[v], dim=0)))
        res = np.array(res)
        r1[i] = res.max()
        r2[i] = np.minimum(res[1:], res[:-1]).max() if len(res) > 1 else res.max()

    y = torch.tensor(changed, dtype=torch.float64)
    ch_t = torch.tensor(changed)
    st = torch.tensor(~changed | strong)                 # static + strong changes
    scores = {"legacy_1-rbar": 1 - legacy.mean_resultant_length("raw"),
              "resid_1": torch.tensor(r1), "resid_2": torch.tensor(r2)}
    per_variant = {}
    for k, f in fields.items():
        ale, _ = f.aleatoric()
        scores[f"{k}:u_ale"] = ale
        scores[f"{k}:c_max"] = f.c_max
        scores[f"{k}:c_last"] = f.c_last
        per_variant[k] = dict(
            field_config=f.cfg.to_dict(), centred=f.zbar is not None,
            fired_changed=float((f.n_resets[ch_t] >= 1).float().mean()),
            fired_static=float((f.n_resets[~ch_t] >= 1).float().mean()),
            c_max_q_static=np.nanquantile(f.c_max[~ch_t].numpy(), [.5, .9, .95]).tolist(),
            c_max_q_changed=np.nanquantile(f.c_max[ch_t].numpy(), [.1, .5]).tolist(),
            u_ale_defined=float(torch.isfinite(ale).float().mean()),
            selfcal=selfcal(f),
            fieldcal=fieldcal(variants[k].get("cfg", {}), variants[k].get("centre", True)))
    res = dict(scene=a.scene, change_frac=a.change_frac, n_points=n, n_changed=int(changed.sum()),
               n_strong=int(strong.sum()), pair_cos_median=float(med), feat_dim=d, seed=a.seed,
               auroc={k: auroc(v, y) for k, v in scores.items()},
               n_undefined={k: int((~torch.isfinite(torch.as_tensor(v))).sum()) for k, v in scores.items()},
               auroc_strong={k: auroc(v[st], y[st]) for k, v in scores.items()},
               variants=per_variant, seconds_extract=t_ext, seconds_total=time.time() - t0,
               git=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                  text=True).stdout.strip() + "+dirty")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
