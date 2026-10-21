"""
video_static_path.py — the animated qualitative figure of the supplementary video: the panels of
fig:qual (scripts/render_qual.py) along consecutive frames of the held-out Replica sequence,
for the full map and the M=30 field, with the classes, temperatures, coverage and tie rules of
Table I.

    UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5 scripts/gpu_gate.sh 4500 python3 \
        scripts/video_static_path.py --vocab configs/vocab_replica.json --extractor lseg \
        --scenes configs/replica_test.json --scene room_0 --betas <results>/tab1/eval_val.json \
        --text-cache <results>/video/text_cache_lseg_replica.pt \
        --kfield <results>/qual/kfields/room_0_lseg_k30.pt --path 230:351 --path-downscale 1 \
        --check-frame 29 --check-meta <results>/qual/qual_room_0_f29.json \
        --out <results>/video/static/room_0_path.npz

Rule (fixed before rendering, design R1/R2/R4): room_0 and evaluation frame 29 (Sequence_2 frame
290) are the scene and frame of fig:qual; the path is Sequence_2 frames 230-350 (290 +- 60); the
M=30 field is the one fig:qual uses. No new choice.

Code paths (render_qual.py and eval_seg.py, imported, never edited):
  * the scene, its query classes (those of the stride-10 evaluation frames), text and frozen
    row come from RQ.load (eval_seg.load_scene), so the class set is the paper's;
  * the path frames come from eval_seg.load_frames on the same scene entry with stride 1 (its
    poses, scaled intrinsics and vocabulary labels), mapped to the scene's query index with
    load_scene's `local` table;
  * every map is RQ.frame_maps (evaluate_view: coverage, ties, the row's own P2 temperature).
Check (exit 2): at --check-frame, through RQ.frame_maps on the scene's own evaluation frame,
RQ.frame_counts of the full and the M=30 maps equal the counts of fig:qual's meta, and the
path frame built here for the same image at the evaluation resolution has the same gt, pose
and intrinsics as the evaluation frame.
"""
from __future__ import annotations
import copy
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import eval_seg as ES                                                      # noqa: E402
import render_qual as RQ                                                   # noqa: E402


def build_parser():
    ap = RQ.build_parser()
    ap.description = __doc__
    ap.add_argument("--path", default="230:351", help="Sequence_2 frames a:b (stride 1)")
    ap.add_argument("--path-downscale", type=int, default=1)
    ap.add_argument("--check-frame", type=int, default=29)
    ap.add_argument("--check-meta", default="", help="fig:qual's meta JSON (counts)")
    return ap


def path_frames(S, sc, vocab, args, a, b, downscale):
    """Frames a..b-1 (stride 1) of the scene's sequence, as eval_seg.load_frames builds them,
    with gt in the scene's query index (load_scene's `local` table)."""
    pa = copy.copy(args)
    pa.max_views, pa.downscale, pa.with_2d = b, downscale, False
    frames, _, _ = ES.load_frames(dict(sc, stride=1), vocab, pa)
    q = [vocab["ids"].index(c) for c in S["class_ids"]]
    local = torch.full((len(vocab["ids"]) + 1,), ES.IGNORE, dtype=torch.long)
    local[torch.tensor(q, dtype=torch.long)] = torch.arange(len(q))
    out = []
    for f in frames:
        if a <= f["idx"] < b:
            lab = f["labels"]
            f["gt"] = torch.where(lab >= 0, local[lab.clamp_min(0)], lab)
            out.append(f)
    return out


def pack(m, rgb=False):
    d = dict(pred=m["pred"].numpy().astype(np.int16), covered=m["covered"].numpy(),
             labelled=m["labelled"].numpy(), correct=m["correct"].numpy(),
             u_epi=m["u"]["u_epi"].numpy().astype(np.float16),
             u_ale=m["u"]["u_ale"].numpy().astype(np.float16))
    if rgb:
        d["rgb"] = (m["rgb"].numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        d["gt"] = m["gt"].numpy().astype(np.int16)
    return d


def run(args):
    t0 = time.time()
    if args.device.startswith("cuda"):
        from usplat import gpu_reserve
        gpu_reserve.apply(args.device)
    vocab = ES.load_vocab(args.vocab)
    ES.check_fit_settings(ES.load_fit_settings(args.betas), args, args.betas)
    betas = ES.load_betas(args.betas)
    sc = RQ.pick_scene(args.scenes, args.scene)
    a, b = (int(x) for x in args.path.split(":"))
    cache_sha = ES_sha(args.text_cache)
    out, check = {}, {}
    names, palette = None, None
    for tag, field in (("full", None), ("k", args.kfield or None)):
        if tag == "k" and field is None:
            continue
        S, row = RQ.load(sc, vocab, betas, args, field=field)
        if names is None:
            names, palette = list(S["class_names"]), RQ.scene_palette(S)
        elif list(S["class_names"]) != names:
            raise SystemExit("the k field queries other classes")
        i = RQ.check_frame(S, args.check_frame)
        check[tag] = RQ.frame_maps(S, i, row, betas, args)
        ev = S["frames"][i]
        mine = path_frames(S, sc, vocab, args, ev["idx"], ev["idx"] + 1, args.downscale)
        same = (len(mine) == 1 and torch.equal(mine[0]["gt"], ev["gt"])
                and torch.equal(mine[0]["viewmat"], ev["viewmat"])
                and torch.equal(mine[0]["K"], ev["K"]))
        check[f"{tag}_frame_built_equal"] = bool(same)
        vf = path_frames(S, sc, vocab, args, a, b, args.path_downscale)
        Sv = dict(S, frames=vf)
        maps = [pack(RQ.frame_maps(Sv, j, row, betas, args, with_rgb=(tag == "full")),
                     rgb=(tag == "full")) for j in range(len(vf))]
        for key in maps[0]:
            out[f"{tag}_{key}"] = np.stack([mm[key] for mm in maps])
        out["index"] = np.array([f["idx"] for f in vf])
        del S, row, Sv, vf
    counts = RQ.frame_counts(check["full"], check.get("k"))
    meta = dict(scene=sc["name"], path=[a, b], path_downscale=args.path_downscale,
                check_frame=args.check_frame, counts=counts,
                frame_built_equal={k: v for k, v in check.items() if k.endswith("_equal")},
                class_names=names, script_sha256=ES_sha(os.path.abspath(__file__)))
    fails = [k for k, v in meta["frame_built_equal"].items() if not v]
    if args.check_meta:
        ref = json.load(open(args.check_meta))["counts"]
        meta["counts_ref"] = ref
        fails += [f"count {k}: {counts.get(k)} != {v}" for k, v in ref.items()
                  if (abs(counts.get(k, -1) - v) > 1e-12)]
    if ES_sha(args.text_cache) != cache_sha:
        fails.append("text cache changed")
    meta.update(ok=not fails, failures=fails, wall_s=time.time() - t0,
                peak_alloc_mib=(torch.cuda.max_memory_allocated() / 2 ** 20
                                if args.device.startswith("cuda") else None))
    stem = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    with open(stem + ".json", "w") as f:
        json.dump(ES._plain_json(meta), f, indent=1)
    if fails:
        print(f"CHECK FAILED: {fails}")
        raise SystemExit(2)
    np.savez_compressed(stem + ".npz", palette=palette, **out)
    print(f"done: {len(out['index'])} path frames, counts {counts}, wall {meta['wall_s']:.0f} s")
    return meta


def ES_sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main(argv=None):
    args = build_parser().parse_args(argv)
    with torch.no_grad():
        return run(args)


if __name__ == "__main__":
    main()
