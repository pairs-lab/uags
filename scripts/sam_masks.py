"""
sam_masks.py — SAM masks of a scene's frames for the SAM-mask baselines (usplat/sam_lift.py):
LangSplat's preprocessing, its fork's SamAutomaticMaskGenerator (usplat.sam_lift.LevelStreams:
32 x 32 points, one crop layer, four streams default / s / m / l) and its masks_update
(usplat.sam_lift.generate_levels), each level painted in list order into an int16 label map at
the main field's grid (usplat.sam_lift.paint_labels). Levels d, s, p, w = its default, s, m, l.

    scripts/gpu_gate.sh 6000 python3 scripts/sam_masks.py \
        --field /home/thanhnc/gauss_ws/src/uags/results/fields/office_0_lseg.pt

writes <out-dir>/<field stem>_sam.npz (usplat.sam_lift.save_masks): labels (frames, 4, h, w),
the frames' dataset index and part, and meta: the SAM checkpoint's sha256, model and settings,
the segment_anything version, the mask code's hash, the field and its frames, per-frame mask
counts and times. meta["identity"] hashes the settings, the checkpoint and the mask code only,
so files of different scenes made with the same settings share it.
Frames: the main field's mapping frames (distill_features.select_frames). --table <a Table III
record (results/nor/tab2_relmax/<split>/<scene>.json)> writes instead the update keyframes that
scripts/eval_change_nor.py load_context reads for the record, to <field stem>_sam_upd.npz;
scripts/train_sam_baselines.py --table takes both files (--masks a.npz,b.npz).
Resumable per frame: every finished frame is kept in <out>.parts/ until the file is written.
An existing output is kept (--force recomputes). --max-frames N (smoke runs) writes
<field stem>_sam_smoke<N>.npz.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import inspect
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from usplat import sam_lift as SL                                           # noqa: E402
from usplat.paths import DATA, RESULTS, moved                               # noqa: E402

CHECKPOINT = os.path.join(DATA, "weights", "sam", "sam_vit_h_4b8939.pth")
MASK_CODE = (SL.LevelStreams, SL.make_generator, SL._pair_counts, SL.mask_nms, SL.masks_update,
             SL.generate_levels, SL.downscale_mask, SL.paint_labels, SL.save_masks)


def sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def mask_code_sha256():
    """This script and the source of the mask functions of usplat/sam_lift.py (not the rest
    of that module: a change of the training code leaves finished masks valid)."""
    h = hashlib.sha256()
    with open(os.path.abspath(__file__), "rb") as f:
        h.update(hashlib.sha256(f.read()).hexdigest().encode())
    for fn in MASK_CODE:
        h.update(fn.__name__.encode() + b"\0" + inspect.getsource(fn).encode())
    return h.hexdigest()


def mapping_frames(ck):
    """[(part, dataset, position)] of the main field's mapping frames."""
    import distill_features as DF
    from usplat.datasets import make_dataset
    cfg = ck["config"]
    name, root, opts = DF.mapping_dataset(cfg)
    ds = make_dataset(name, root, stride=cfg["stride"], **opts)
    idx = DF.select_frames(ds, cfg)
    saved_kf = (ck.get("timing") or {}).get("keyframes")
    if saved_kf is not None and int(saved_kf) != len(idx):
        raise SystemExit(f"{len(idx)} mapping frames here, the main field used {saved_kf}")
    return [("map", ds, i) for i in idx]


def update_frames(table):
    """[(part, dataset, position)] of the update keyframes eval_change_nor.load_context reads
    for a Table III record (its dataset, update stride, alignment and keyframe rule), and the
    field path of the record."""
    import dump_nor_qual as DQ
    from time_field import select_keyframes, evenly
    from usplat.datasets import make_dataset
    m = DQ.record_body(json.load(open(table)))
    ns = DQ.ec_namespace(m, argparse.Namespace(backend="gsplat", device="cpu", smoke=0))
    ck = torch.load(ns.field, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    data = ns.data or moved(cfg["data"])
    opt = {} if ns.align == "none" else dict(align=ns.align)
    upd_ds = make_dataset("nor", data, stride=int(ns.update_stride or cfg["stride"]),
                          part="update", **opt)
    kf = (float(cfg.get("kf_trans", 0.05)), float(cfg.get("kf_rot", 5.0)))
    idx = select_keyframes(upd_ds, *kf, 0)
    if ns.max_update_keyframes:
        idx = evenly(idx, ns.max_update_keyframes)
    return [("update", upd_ds, i) for i in idx], ns.field, ck


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--field", help="the scene's main field (run_mapping --out): mapping frames")
    src.add_argument("--table", help="a NeRF-Object-Removal Table III record: its update keyframes")
    ap.add_argument("--out-dir", default=os.path.join(RESULTS, "sam_masks"))
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--model", default="vit_h")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=0, help="first N frames only (smoke runs)")
    ap.add_argument("--force", action="store_true", help="recompute an existing output")
    for k, v in SL.SAM_SETTINGS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    t_start = time.time()
    if args.table:
        frames, field, ck = update_frames(args.table)
        tag = "_sam_upd"
    else:
        field = args.field
        ck = torch.load(field, map_location="cpu", weights_only=False)
        frames = mapping_frames(ck)
        tag = "_sam"
    if args.max_frames:
        frames = frames[:args.max_frames]
        tag += f"_smoke{args.max_frames}"
    stem = os.path.splitext(os.path.basename(field))[0]
    out = os.path.join(os.path.abspath(args.out_dir), stem + tag + ".npz")
    if os.path.exists(out) and not args.force:
        print(f"{out}: exists, kept (--force recomputes)")
        return out
    settings = {k: getattr(args, k) for k in SL.SAM_SETTINGS}
    ck_sha = sha256_file(args.checkpoint)
    import importlib.metadata as md
    sam_meta = dict(model=args.model, checkpoint=os.path.abspath(args.checkpoint),
                    checkpoint_sha256=ck_sha, segment_anything=md.version("segment-anything"))
    code = mask_code_sha256()
    ident = SL.settings_identity(dict(settings=settings, model=args.model,
                                      checkpoint_sha256=ck_sha, code=code))
    factor = int(ck["config"]["downscale"])
    parts = out + ".parts"
    os.makedirs(parts, exist_ok=True)

    held = 0
    if str(args.device).startswith("cuda"):
        from usplat import gpu_reserve                       # fraction from scripts/gpu_gate.sh
        held = gpu_reserve.apply(args.device)
    from segment_anything import SamPredictor, sam_model_registry
    from usplat.datasets import downscale
    sam = sam_model_registry[args.model](checkpoint=args.checkpoint).to(args.device).eval()
    generator = SL.make_generator(SamPredictor(sam), **settings)
    upd = dict(update_iou_thr=settings["update_iou_thr"], update_score_thr=settings["update_score_thr"],
               update_inner_thr=settings["update_inner_thr"], device=args.device)
    t_load = time.time() - t_start
    print(f"{stem}: {len(frames)} frames ({tag[1:]}), SAM {args.model} on {args.device}, "
          f"downscale {factor} -> {out}")
    labels, index, part, per = [], [], [], []
    for k, (pt, ds, i) in enumerate(frames):
        full = ds[i]
        idx = int(full.get("index", getattr(ds, "offset", 0) + i * ds.stride))
        pfile = os.path.join(parts, f"{pt}_{idx}.npz")
        hw = tuple(downscale(full, factor)["image"].shape[:2])
        done = None
        if os.path.exists(pfile):
            with np.load(pfile, allow_pickle=False) as z:
                info = json.loads(str(z["info"]))
                if info.get("identity") == ident and tuple(z["labels"].shape[1:]) == hw:
                    done = (z["labels"].copy(), info)
        if done is None:
            t0 = time.time()
            img = np.ascontiguousarray(full["image"].numpy().astype(np.uint8))
            levels, before = SL.generate_levels(generator, img, **upd)
            lab = np.stack([SL.paint_labels(ms, hw) for ms in levels])
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            info = dict(identity=ident, part=pt, index=idx, n_masks=[len(ms) for ms in levels],
                        n_before_update=before,
                        n_painted=[int(len(np.unique(lv[lv >= 0]))) for lv in lab],
                        covered=[float((lv >= 0).mean()) for lv in lab],
                        image_hw=list(img.shape[:2]), seconds=time.time() - t0)
            tmp = f"{pfile}.tmp{os.getpid()}"
            with open(tmp, "wb") as f:
                np.savez_compressed(f, labels=lab, info=np.array(json.dumps(info)))
            os.replace(tmp, pfile)
            done = (lab, info)
            print(f"  [{k + 1}/{len(frames)}] {pt}:{idx} masks d/s/p/w {info['n_masks']} "
                  f"({info['seconds']:.1f} s)", flush=True)
        labels.append(done[0])
        index.append(idx)
        part.append(pt)
        per.append(done[1])
    meta = dict(identity=ident, settings=settings, sam=sam_meta, mask_code_sha256=code,
                field=os.path.abspath(field), table=os.path.abspath(args.table) if args.table else None,
                table_sha256=sha256_file(args.table) if args.table else None,
                field_config={k: ck["config"].get(k) for k in
                              ("dataset", "data", "part", "stride", "downscale", "max_keyframes",
                               "kf_trans", "kf_rot")},
                levels=list(SL.LEVELS), grid_hw=list(labels[0].shape[1:]) if labels else None,
                frames=[f"{p}:{i}" for p, i in zip(part, index)], per_frame=per,
                max_frames=args.max_frames or None, device=args.device, gpu_reserved_mib=held,
                peak_alloc_mib=(torch.cuda.max_memory_allocated() / 2 ** 20
                                if str(args.device).startswith("cuda") and torch.cuda.is_available()
                                else None),
                load_s=t_load, total_s=time.time() - t_start,
                created=datetime.datetime.now().isoformat(timespec="seconds"))
    SL.save_masks(out, np.stack(labels), index, part, meta)
    shutil.rmtree(parts)
    n = np.array([p["n_masks"] for p in per])
    print(f"wrote {out}: {len(per)} frames, masks per frame d/s/p/w {n.mean(0).round(1).tolist()}, "
          f"{sum(p['seconds'] for p in per):.0f} s")
    return out


if __name__ == "__main__":
    main()
