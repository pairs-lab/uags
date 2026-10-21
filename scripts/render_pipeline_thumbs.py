"""
render_pipeline_thumbs.py — real thumbnails for the pipeline figure (papers/figs/pipeline_v2.tex,
fig:pipeline). Validation scenes only. Four sub-commands:

  select   (CPU)  apply the selection rules RULES (written before any render) and write
                  <out>/selection.json: the Replica keyframe and the NOR update pose and crop.
  replica  (GPU)  one Replica validation keyframe: the keyframe photo, its LSeg features in the
                  field's PCA basis B, and the saved full field rendered at the keyframe pose
                  (ours P2 labels, u_epi, u_ale) -> <out>/replica.npz
  nor      (GPU)  NeRF-Object-Removal validation scene 002: the Table III update path of ours
                  (eval_change_nor.py's functions, imported, not edited): calibrate q = 0.999 on
                  the field's own mapping pass, update_view with LiDAR depth over the 48 update
                  keyframes, reset and re-admission. Checks the replay against the saved val
                  record, then renders, at the selected update pose, the share of the rendered
                  mass from Gaussians the test reset during the update -> <out>/nor.npz
  compose  (CPU)  NPZ -> small PNG thumbnails, a null-histogram glyph (PDF, no text), a colour
                  bar strip and values.tex (legend colours and names), copied to
                  papers/figs/pipeline_thumbs/; meta.json holds every number used.

Run (from the repo root, environment of the paper):
  O=/home/thanhnc/gauss_ws/data/uags_results/pipeline
  python3 scripts/render_pipeline_thumbs.py select --out $O
  UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5 scripts/gpu_gate.sh 4500 \
      python3 scripts/render_pipeline_thumbs.py replica --out $O
  UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5 scripts/gpu_gate.sh 4500 \
      python3 scripts/render_pipeline_thumbs.py nor --out $O
  python3 scripts/render_pipeline_thumbs.py compose --out $O

Text embeddings: only a COPY of the Replica text cache is passed (eval_seg.text_embeddings
writes to --text-cache): <out>/text_cache_lseg_pipeline.pt.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

DATA = "/home/thanhnc/gauss_ws/data"
OUT = f"{DATA}/uags_results/pipeline"
FIGS = os.path.join(REPO, "papers", "figs", "pipeline_thumbs")
TEXT_CACHE_SRC = f"{DATA}/uags_fields/text_cache_lseg.pt"
TEXT_CACHE = "text_cache_lseg_pipeline.pt"          # our own copy, inside --out
REPLICA_SCENES = os.path.join(REPO, "configs", "replica_val.json")
REPLICA_VOCAB = os.path.join(REPO, "configs", "vocab_replica.json")
REPLICA_BETAS = f"{DATA}/uags_results/tab1/eval_val.json"
NOR_CONFIG = f"{DATA}/uags_results/nor/tab2_relmax/config_val.json"
NOR_RECORD = f"{DATA}/uags_results/nor/tab2_relmax/val/002.json"
NOR_SCENE = "002"

# Selection rules, fixed 2026-10-21 BEFORE any thumbnail was rendered. Validation scenes only.
RULES = {
    "R1_replica_keyframe": (
        "Among the mapping keyframes of the two Replica validation scenes (office_0, room_2: "
        "Sequence_1 every 5th frame, kept by run_mapping.py's pose rule kf_trans 0.05 m / "
        "kf_rot 5 deg, the field's own config), take the keyframe whose ground-truth label "
        "image (configs/vocab_replica.json mapping, id 0 ignored) has the most classes that "
        "each cover at least 0.5% of the pixels. Ties: larger labelled share, then office_0 "
        "before room_2, then the lower frame index. Uses labels only, no method output."),
    "R2_replica_display": (
        "All Replica thumbnails are at that keyframe's pose on the mapping grid (downscale 2, "
        "320x240). (1) the keyframe photo; (2) its LSeg features z_p (the field's extractor, "
        "full-resolution tokens, bilinear to the grid, projected with the field's PCA basis B "
        "and renormalised, as run_mapping.py): the first three coordinates of z_p, each mapped "
        "linearly from its 1st to its 99th percentile over the frame to [0, 1], as RGB; "
        "(6) u_epi and u_ale of the saved full field (after all mapping keyframes), rendered as "
        "in Table I (undef_frac 0.5), viridis from 0 to the 0.99-quantile of the covered "
        "defined pixels of each map, undefined grey, not covered (alpha < 0.5) black; "
        "(9) the labels of ours P2 (argmax over the scene's queried classes, the classes "
        "present in its evaluation frames, as Table I), render_qual.py's scene palette, not "
        "covered black. Legend: the 4 predicted classes with the most pixels in the frame."),
    "R3_nor_scene_pose": (
        "NeRF-Object-Removal validation scene 002 (office chair removed; also the largest "
        "removed-object box of the validation scenes). Pose t*: the update keyframe (of the 48 "
        "of the Table III val record) with the largest masks_real area, ties to the earliest."),
    "R4_nor_state": (
        "State after all 48 update keyframes through the Table III code path of ours "
        "(eval_change_nor.py run_update without the other rows): calibrate_change_threshold "
        "(q = 0.999, history buckets) on the saved mapping-pass field, then update_view with "
        "the high-confidence LiDAR depth: test, reset, depth re-admission. Shown: the share of "
        "the rendered mass at t* from Gaussians reset at least once during the update (Table "
        "III's 'flagged'). Checks before use: thresholds equal to the record; resets within 1%; "
        "flagged Rec and FA within 0.01 of the record."),
    "R5_nor_display": (
        "Full resolution 720x540 with the frame's own K. Rotate 90 deg clockwise (portrait "
        "capture), crop a 4:3 landscape window 540x405 centred vertically on the centroid of "
        "masks_real at t*, clamped to the frame. Base: the after-removal photo in grey "
        "(0.30 + 0.55 luminance); overlay Okabe-Ito orange #E69F00 with opacity 0.85 x share on "
        "covered pixels (alpha >= 0.5). Blur brand text only if it is readable, and say so."),
    "R5b_before_inset (added after the first render, no selection changed)": (
        "The reset overlay alone does not show what was removed, so the figure adds a small "
        "inset: the 3D-GS map rendered at t* (before the update, chair present), same crop."),
    "R6_null_glyph": (
        "Threshold glyph: the mapping-pass null histogram of c-bar of scene 002 (the saved "
        "field's c_hist) in the history bucket with the most samples, re-binned to 60 bins "
        "on [0, 1.6 theta(h)] of that bucket, log counts, with a vertical line at theta(h)."),
}


# ------------------------------------------------------------------------------ helpers
def env_device(args):
    dev = args.device
    if dev.startswith("cuda"):
        from usplat import gpu_reserve                     # fraction from scripts/gpu_gate.sh
        held = gpu_reserve.apply(dev)
        print(f"reserved {held} MiB of GPU memory")
    return dev


def peak_mib(dev):
    if str(dev).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2 ** 20
    return None


def load_json(path):
    with open(path) as f:
        return json.load(f)


def dump_json(obj, path):
    import eval_seg as ES
    with open(path, "w") as f:
        json.dump(ES._plain_json(obj), f, indent=1)
    print(f"wrote {path}")


def nor_entry():
    cfg = load_json(NOR_CONFIG)
    return next(s for s in cfg["scenes"] if s["name"] == NOR_SCENE)


def rot_cw(a):
    """Rotate an (H, W, ...) array 90 deg clockwise (portrait NOR capture -> upright)."""
    return np.rot90(np.asarray(a), k=-1, axes=(0, 1))


# ------------------------------------------------------------------------------ select
def select(args):
    """Rules R1, R3, R5 (crop window) on data only; writes <out>/selection.json."""
    import eval_seg as ES
    from time_field import select_keyframes
    from usplat.datasets import make_dataset
    t0 = time.time()
    vocab = ES.load_vocab(REPLICA_VOCAB)
    table = []
    scenes = ES._load_structured(REPLICA_SCENES)
    for order, sc in enumerate(scenes):
        ck = torch.load(sc["field"], map_location="cpu", weights_only=False, mmap=True)
        cfg = ck["config"]
        ds = make_dataset(cfg["dataset"], cfg["data"], stride=int(cfg["stride"]))
        kf = select_keyframes(ds, float(cfg["kf_trans"]), float(cfg["kf_rot"]),
                              int(cfg.get("max_keyframes") or 0))
        n_saved = (ck.get("timing") or {}).get("keyframes")
        if n_saved is not None and int(n_saved) != len(kf):
            raise SystemExit(f"{sc['name']}: {len(kf)} keyframes here, field used {n_saved}")
        for pos in kf:
            fr = ds[pos]
            lab, _ = ES.map_labels(fr["labels"], vocab)
            lab = lab.reshape(-1)
            cnt = torch.bincount(lab[lab >= 0], minlength=len(vocab["ids"]))
            table.append(dict(scene=sc["name"], scene_order=order, pos=int(pos),
                              index=int(fr["index"]),
                              n_classes=int((cnt >= 0.005 * lab.numel()).sum()),
                              labelled=float((lab >= 0).float().mean())))
    table.sort(key=lambda t: (-t["n_classes"], -t["labelled"], t["scene_order"], t["index"]))
    rep = dict(table[0], top5=table[:5], n_keyframes=len(table))
    print(f"R1 Replica keyframe: {rep['scene']} Sequence_1 frame {rep['index']} "
          f"({rep['n_classes']} classes >= 0.5%, labelled {rep['labelled']:.3f})")

    # R3: NOR 002, the update keyframe with the largest masks_real area (ties: earliest)
    sc = nor_entry()
    rec = load_json(NOR_RECORD)
    upd = [int(i) for i in rec["metrics"]["views"]["update"]]
    root = sc["data"]
    stems = {}
    for i in upd:
        p = os.path.join(root, "masks_real", f"{i:03d}.npy")
        if not os.path.exists(p):                      # file names follow the image names
            import glob
            img = sorted(glob.glob(os.path.join(root, "images", "*.jpg")))[i]
            p = os.path.join(root, "masks_real",
                             os.path.splitext(os.path.basename(img))[0] + ".npy")
        stems[i] = p
    areas = {i: int(np.load(stems[i]).astype(bool).sum()) for i in upd}
    t_star = max(upd, key=lambda i: (areas[i], -i))
    m = rot_cw(np.load(stems[t_star]).astype(bool))       # upright, (256, 192)
    H_up, W_up = 720, 540                                 # upright full resolution
    ys = np.nonzero(m)[0]
    cy = (ys.mean() + 0.5) * H_up / m.shape[0]
    ch = int(round(W_up * 3 / 4))                         # 405
    y0 = int(round(min(max(cy - ch / 2, 0), H_up - ch)))
    nor = dict(scene=NOR_SCENE, t_star=int(t_star), area=areas[t_star],
               area_px_of=int(m.size), position=upd.index(t_star) + 1, n_update=len(upd),
               top3=sorted(upd, key=lambda i: (-areas[i], i))[:3],
               crop=dict(rotate="cw", x0=0, y0=y0, w=W_up, h=ch, centroid_y=float(cy)),
               mask_file=stems[t_star])
    print(f"R3 NOR {NOR_SCENE}: t* = frame {t_star} (area {areas[t_star]} of {m.size}, "
          f"{nor['position']}th of {len(upd)}); crop y0 = {y0}, h = {ch}")
    out = dict(rules=RULES, written=time.strftime("%Y-%m-%d %H:%M:%S"), replica=rep, nor=nor,
               wall_s=time.time() - t0)
    os.makedirs(args.out, exist_ok=True)
    dump_json(out, os.path.join(args.out, "selection.json"))
    return out


# ------------------------------------------------------------------------------ replica
def replica_args(args):
    """render_qual.py's argument set (eval_seg.py parser) for the figure's row ours_P2."""
    import render_qual as RQ
    cache = os.path.join(args.out, TEXT_CACHE)
    if not os.path.exists(cache):
        shutil.copy(TEXT_CACHE_SRC, cache)
    if os.path.samefile(cache, TEXT_CACHE_SRC):
        raise SystemExit("refusing to use the original text cache")
    return RQ.build_parser().parse_args([
        "--vocab", REPLICA_VOCAB, "--extractor", "lseg", "--scenes", REPLICA_SCENES,
        "--betas", REPLICA_BETAS, "--text-cache", cache, "--device", args.device,
        "--backend", "gsplat", "--out", os.path.join(args.out, "unused.pdf")])


@torch.no_grad()
def replica(args):
    import eval_seg as ES
    import render_qual as RQ
    from time_field import load_extractor, check_extractor
    from usplat.clip_dense import upsample_tokens
    from usplat.datasets import make_dataset, downscale
    from usplat.feature_pca import StreamingPCA
    from usplat.seg_eval import evaluate_view
    t0 = time.time()
    dev = env_device(args)
    sel = load_json(os.path.join(args.out, "selection.json"))["replica"]
    sc = RQ.pick_scene(REPLICA_SCENES, sel["scene"])
    ck = torch.load(sc["field"], map_location="cpu", weights_only=False, mmap=True)
    cfg = ck["config"]
    ds = make_dataset(cfg["dataset"], cfg["data"], stride=int(cfg["stride"]))
    full = ds[sel["pos"]]
    if int(full["index"]) != sel["index"]:
        raise SystemExit(f"frame {full['index']} != selected {sel['index']}")
    fr = downscale(full, int(cfg["downscale"]))
    h, w = fr["image"].shape[:2]
    # (2) features as run_mapping.py computes them: full-resolution tokens -> grid -> B
    clip = load_extractor(cfg, dev)
    check_extractor(clip, ck)
    tok = clip.extract_tokens(full["image"])
    del clip
    pca = StreamingPCA.from_state(ck["pca"], device=dev)
    z = pca.project(upsample_tokens(tok.to(dev), (h, w))).reshape(h, w, -1)
    zc = z[..., :3].float().cpu().numpy()
    del tok, z, ck
    t_feat = time.time() - t0
    # (6), (9): the saved full field through eval_seg's loader and evaluate_view (Table I path)
    qa = replica_args(args)
    vocab = ES.load_vocab(qa.vocab)
    ES.check_fit_settings(ES.load_fit_settings(qa.betas), qa, qa.betas)
    betas = ES.load_betas(qa.betas)
    S, row = RQ.load(sc, vocab, betas, qa)
    frd = dict(viewmat=torch.linalg.inv(fr["c2w"].double()).float(), K=fr["K"].float(),
               w=w, h=h)
    ras = ES.make_rasterizer(qa.backend, S["splats"], frd, qa)
    out = evaluate_view([row], ras, S["text"], torch.zeros(h * w, dtype=torch.long), betas,
                        qa.tau_cov, undef_frac=qa.undef_frac,
                        raw_probs=ES._raw_probs(qa))[row.name]
    pred = out["pred"].reshape(h, w).long().numpy()
    cov = out["covered"].reshape(h, w).numpy()
    u = {k: v.reshape(h, w).float().numpy() for k, v in out["u"].items()}
    palette = RQ.scene_palette(S)
    names = list(S["class_names"])
    gt_lab, _ = ES.map_labels(ES._nearest(torch.as_tensor(full["labels"]), h, w), vocab)
    np.savez_compressed(
        os.path.join(args.out, "replica.npz"), rgb=fr["image"].numpy(), zc=zc, pred=pred,
        covered=cov, u_epi=u["u_epi"], u_ale=u["u_ale"], palette=palette,
        names=np.array(names), gt_vocab=gt_lab.numpy(), vocab_names=np.array(vocab["names"]))
    meta = dict(scene=sc["name"], frame_index=sel["index"], grid=[h, w], row=row.name,
                field=sc["field"], ply=sc["ply"], betas=REPLICA_BETAS,
                n_classes_queried=len(names), covered_share=float(cov.mean()),
                u_ale_defined_share_covered=float(np.isfinite(u["u_ale"])[cov].mean()),
                t_features_s=t_feat, wall_s=time.time() - t0, peak_alloc_mib=peak_mib(dev),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    dump_json(meta, os.path.join(args.out, "replica_meta.json"))
    return meta


# ------------------------------------------------------------------------------ nor
@torch.no_grad()
def nor(args):
    import eval_change_nor as EC
    from time_field import select_keyframes, rasterizer_factory, load_extractor, \
        check_extractor
    from usplat.datasets import make_dataset
    from usplat.feature_pca import StreamingPCA
    from usplat.gs_io import load_gaussians
    from usplat.rows import TensorRow
    from usplat.seg_eval import render_row
    from usplat.vmf_field import DirectionalSemanticField
    import render_qual as RQ
    t0 = time.time()
    dev = env_device(args)
    sel = load_json(os.path.join(args.out, "selection.json"))["nor"]
    sc, rec = nor_entry(), load_json(NOR_RECORD)["metrics"]
    ckpt = torch.load(sc["field"], map_location="cpu", weights_only=False)
    EC.check_mapping_only(ckpt, sc["field"])
    cfg = ckpt["config"]
    splats = load_gaussians(sc["ply"], device=dev)
    N = int(splats["means"].shape[0])
    if N != int(ckpt["field"]["N"]):
        raise SystemExit(f"{sc['ply']}: {N} Gaussians, field {ckpt['field']['N']}")
    kf = (float(cfg.get("kf_trans", 0.05)), float(cfg.get("kf_rot", 5.0)))
    upd_stride = int(sc["update_stride"])
    upd_ds = make_dataset("nor", sc["data"], stride=upd_stride, part="update")
    upd_idx = select_keyframes(upd_ds, *kf, 0)
    clip = load_extractor(cfg, dev)
    check_extractor(clip, ckpt)
    views = EC.read_views(upd_ds, upd_idx, int(cfg["downscale"]), clip)
    del clip
    got = [v["index"] for v in views]
    if got != [int(i) for i in rec["views"]["update"]]:
        raise SystemExit(f"update keyframes {got} differ from the val record")
    t_load = time.time() - t0
    chunk = int(cfg.get("channel_chunk") or 32)
    ctx = dict(device=dev, pca=StreamingPCA.from_state(ckpt["pca"], device=dev),
               d=int(ckpt["field"]["d"]), splats=splats,
               rasterizer=rasterizer_factory("gsplat", chunk))
    ours = DirectionalSemanticField.from_state(ckpt["field"], device=dev)
    c_hist = ours.c_hist.double().cpu().numpy()            # the mapping-pass null
    thr = ours.calibrate_change_threshold(0.999)
    by_hist = [float(x) for x in ours.thr_by_hist.cpu()]
    rec_thr = rec["ours"]["threshold"]
    if (abs(thr - rec_thr["pooled"]) > 1e-6
            or max(abs(a - b) for a, b in zip(by_hist, rec_thr["by_history"])) > 1e-6):
        raise SystemExit(f"thresholds {thr}, {by_hist} differ from the record {rec_thr}")
    r0 = ours.n_resets.clone()
    observed = torch.zeros(N, dtype=torch.bool, device=dev)
    t1 = time.time()
    for v in views:
        z, ras, gz, kw = EC.view_inputs(ctx, v)
        stats = ours.update_view(ras, z, num_pixels=v["h"] * v["w"], **kw)
        observed |= EC.covered(ours, stats)
    t_update = time.time() - t1
    flagged = (ours.n_resets > r0)
    n_resets = int((ours.n_resets - r0).sum())
    # Rec / FA against the D2 ground truth, as the record (same functions)
    gctx = dict(map_ds=make_dataset("nor", sc["data"], stride=int(cfg["stride"]), part="map"),
                data=sc["data"], align=None, splats=splats)
    st = rec["settings"]
    gt, _, _, _ = EC.ground_truth(gctx, argparse.Namespace(inflate=st["inflate_m"],
                                                           near=st["near_m"]))
    stale = gt["stale"].to(dev)
    obs_stale, obs_static = observed & stale, observed & ~stale
    rec_ours = float(flagged[obs_stale].float().mean())
    fa_ours = float(flagged[obs_static].float().mean())
    want = rec["operating_point"]["ours"]
    checks = dict(
        thresholds_equal=True,
        resets=n_resets, resets_record=int(rec["ours"]["n_resets"]),
        resets_rel_diff=abs(n_resets - rec["ours"]["n_resets"]) / rec["ours"]["n_resets"],
        rec=rec_ours, rec_record=want["rec"], fa=fa_ours, fa_record=want["fa"],
        n_stale_observed=int(obs_stale.sum()),
        n_stale_observed_record=int(rec["counts"]["n_stale_observed"]))
    checks["pass"] = bool(checks["resets_rel_diff"] <= 0.01
                          and abs(rec_ours - want["rec"]) <= 0.01
                          and abs(fa_ours - want["fa"]) <= 0.01)
    print("checks:", json.dumps(checks))
    if not checks["pass"]:
        raise SystemExit("the replay does not reproduce the val record; no thumbnail")
    # render at t*, full resolution, the frame's own K
    full = upd_ds[(sel["t_star"] - upd_ds.start) // upd_stride]
    if int(full["index"]) != sel["t_star"]:
        raise SystemExit(f"frame {full['index']} != t* {sel['t_star']}")
    H, W = int(full["image"].shape[0]), int(full["image"].shape[1])
    viewmat = torch.linalg.inv(full["c2w"].double()).float().to(dev)
    ras = ctx["rasterizer"](splats, viewmat, full["K"].to(dev), W, H)
    row = TensorRow("reset", "feature", buf=torch.zeros(N, 1, device=dev),
                    unc={"reset": flagged.float(), "stale": stale.float()})
    rr = render_row(row, ras)
    map_rgb = RQ.render_rgb(splats, ras, H, W).numpy()
    np.savez_compressed(
        os.path.join(args.out, "nor.npz"), photo=full["image"].numpy(),
        reset=rr.unc["reset"].reshape(H, W).float().cpu().numpy(),
        stale=rr.unc["stale"].reshape(H, W).float().cpu().numpy(),
        alpha=rr.alpha.reshape(H, W).float().cpu().numpy(), map_rgb=map_rgb,
        mask_real=full["mask_real"].numpy(), c_hist=c_hist,
        thr_by_hist=np.array(by_hist), hist_edges=np.array(ours.HIST_EDGES, dtype=float))
    meta = dict(scene=NOR_SCENE, t_star=sel["t_star"], field=sc["field"], ply=sc["ply"],
                record=NOR_RECORD, n_update_keyframes=len(views), threshold_pooled=thr,
                threshold_by_history=by_hist, checks=checks, hw=[H, W],
                t_load_extract_s=t_load, t_update_s=t_update, wall_s=time.time() - t0,
                peak_alloc_mib=peak_mib(dev),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    dump_json(meta, os.path.join(args.out, "nor_meta.json"))
    return meta


# ------------------------------------------------------------------------------ compose
OK_ORANGE = np.array([0xE6, 0x9F, 0x00]) / 255.0


def save_png(img, path, max_w=None):
    """8-bit PNG; max_w: downsample (Lanczos) to this width, for the small printed insets."""
    from PIL import Image
    a = np.clip(np.asarray(img, dtype=float), 0.0, 1.0)
    im = Image.fromarray((a * 255.0 + 0.5).astype(np.uint8))
    if max_w and im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    im.save(path, optimize=True)
    return path


def pca_image(zc):
    img = np.zeros_like(zc, dtype=float)
    lims = []
    for c in range(3):
        lo, hi = np.percentile(zc[..., c], [1.0, 99.0])
        img[..., c] = np.clip((zc[..., c] - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        lims.append([float(lo), float(hi)])
    return img, lims


def null_glyph(c_hist, thr_by_hist, path):
    """Histogram of the bucket with most samples on [0, 1.6 theta], log counts, theta line;
    no text (the labels are in the TikZ figure)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    b = int(np.argmax(c_hist.sum(1)))
    theta = float(thr_by_hist[b])
    centres = (np.arange(c_hist.shape[1]) + 0.5) / 2000.0     # field bins: width 1/2000
    xmax = 1.6 * theta
    edges = np.linspace(0.0, xmax, 61)
    cnt, _ = np.histogram(centres, bins=edges, weights=c_hist[b])
    fig = Figure(figsize=(0.95, 0.30))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    y = np.log10(1.0 + cnt)
    ax.bar(edges[:-1], y, width=np.diff(edges), align="edge", color="#9a9a9a", linewidth=0)
    ax.axvline(theta, color="#B26E00", linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, y.max() * 1.05)
    ax.axis("off")
    fig.savefig(path, transparent=True)
    tail = float(c_hist[b][centres > theta].sum() / c_hist[b].sum())
    return dict(bucket=b, theta=theta, n=float(c_hist[b].sum()), xmax=xmax,
                share_above_theta=tail, theta_x_frac=theta / xmax)


def colorbar_strip(path, n=256):
    import matplotlib
    cm = matplotlib.colormaps["viridis"](np.linspace(0, 1, n))[:, :3]
    return save_png(np.repeat(cm[None], 8, axis=0), path)


def compose(args):
    import render_qual as RQ
    out, figs = args.out, args.figs
    os.makedirs(figs, exist_ok=True)
    sel = load_json(os.path.join(out, "selection.json"))
    meta = dict(selection=sel, files={})
    R = np.load(os.path.join(out, "replica.npz"))
    cov = R["covered"].astype(bool)
    vmax = {}
    for ch in ("u_epi", "u_ale"):
        vals = R[ch][cov & np.isfinite(R[ch])]
        vmax[ch] = RQ.nice_up(float(np.quantile(vals, 0.99))) if vals.size else 1.0
    pca, lims = pca_image(R["zc"])
    files = {
        "keyframe_rgb.png": R["rgb"].astype(float) / 255.0,
        "features_pca.png": pca,
        "u_epi.png": RQ.u_image(R["u_epi"], cov, vmax["u_epi"]),
        "u_ale.png": RQ.u_image(R["u_ale"], cov, vmax["u_ale"]),
        "labels_p2.png": RQ.label_image(R["pred"], R["palette"]),
    }
    pred = R["pred"]
    cnt = np.bincount(pred[pred >= 0].reshape(-1), minlength=len(R["names"]))
    top = [int(c) for c in np.argsort(-cnt, kind="stable")[:4] if cnt[c] > 0]
    legend = [dict(name=str(R["names"][c]), rgb=[float(x) for x in R["palette"][c]],
                   share=float(cnt[c] / pred.size)) for c in top]
    # NOR change thumbnail
    Nn = np.load(os.path.join(out, "nor.npz"))
    cr = sel["nor"]["crop"]
    photo = rot_cw(Nn["photo"]).astype(float) / 255.0
    reset, alpha = rot_cw(Nn["reset"]), rot_cw(Nn["alpha"])
    sl = (slice(cr["y0"], cr["y0"] + cr["h"]), slice(cr["x0"], cr["x0"] + cr["w"]))
    lum = photo[..., 0] * 0.2126 + photo[..., 1] * 0.7152 + photo[..., 2] * 0.0722
    grey = np.repeat((0.30 + 0.55 * lum)[..., None], 3, axis=2)
    share = np.where((alpha >= 0.5) & np.isfinite(reset), np.nan_to_num(reset), 0.0)
    a = (0.85 * share)[..., None]
    change = (1.0 - a) * grey + a * OK_ORANGE
    files["change_nor002.png"] = change[sl]
    files["change_nor002_photo.png"] = photo[sl]                 # for the brand check only
    files["change_nor002_before.png"] = rot_cw(Nn["map_rgb"])[sl]  # map render, before
    in_crop = (alpha[sl] >= 0.5)
    stale_px = np.nan_to_num(rot_cw(Nn["stale"])[sl]) >= 0.5
    glyph = null_glyph(Nn["c_hist"], Nn["thr_by_hist"], os.path.join(out, "null_hist.pdf"))
    colorbar_strip(os.path.join(out, "viridis_strip.png"))
    width = {"change_nor002.png": 360, "change_nor002_before.png": 240}   # printed 2.0 / 0.9 cm
    for name, img in files.items():
        save_png(img, os.path.join(out, name), width.get(name))
    keep = ["keyframe_rgb.png", "features_pca.png", "u_epi.png", "u_ale.png",
            "labels_p2.png", "change_nor002.png", "change_nor002_before.png", "null_hist.pdf",
            "viridis_strip.png"]
    for name in keep:
        shutil.copy(os.path.join(out, name), os.path.join(figs, name))
    # values.tex: legend colours and names for pipeline_v2.tex (generated, do not edit)
    lines = ["% generated by scripts/render_pipeline_thumbs.py compose; do not edit"]
    for k, e in enumerate(legend):
        r, g, b = e["rgb"]
        lines.append(f"\\definecolor{{lab{'ABCD'[k]}}}{{rgb}}{{{r:.4f},{g:.4f},{b:.4f}}}")
        lines.append(f"\\newcommand{{\\labname{'ABCD'[k]}}}{{{e['name']}}}")
    lines.append(f"\\newcommand{{\\nullthetafrac}}{{{glyph['theta_x_frac']:.4f}}}")
    with open(os.path.join(figs, "values.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    meta.update(vmax=vmax, pca_percentiles=lims, legend=legend, null_glyph=glyph,
                change=dict(covered_share=float(in_crop.mean()),
                            reset_mean_on_stale_px=float(share[sl][stale_px].mean())
                            if stale_px.any() else None,
                            reset_mean_on_other_px=float(share[sl][in_crop & ~stale_px].mean())
                            if (in_crop & ~stale_px).any() else None,
                            stale_px_share=float(stale_px.mean())),
                replica=load_json(os.path.join(out, "replica_meta.json")),
                nor=load_json(os.path.join(out, "nor_meta.json")),
                files={n: os.path.join(figs, n) for n in keep})
    dump_json(meta, os.path.join(out, "meta.json"))
    return meta


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "replica", "nor", "compose"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--figs", default=FIGS)
    ap.add_argument("--device", default="cuda")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    return dict(select=select, replica=replica, nor=nor, compose=compose)[args.cmd](args)


if __name__ == "__main__":
    main()
