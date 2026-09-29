"""
extract_clip.py — cache the dense CLIP token grid of every frame (plan F8 / K6).

The cache holds the projected ViT patch tokens (gh, gw, 512) in fp16, BEFORE upsampling and
normalisation (usplat.clip_dense.OpenCLIPDense.extract_tokens); load them with
usplat.clip_dense.upsample_tokens(tokens, (h, w)) at the accumulation resolution. A 640x480
frame is 30x40x512 fp16 = 1.2 MB, against 39 MB for a per-pixel PCA-64 cache.

One file per frame: <out>/<scene>/<sequence>/tok_<index>.pt ; frames already cached are skipped,
so the script can be stopped and resumed.
Usage:
  PYTHONPATH=. python3 scripts/extract_clip.py --dataset replica_semnerf \
      --root /home/thanhnc/gauss_ws/data/replica_semnerf/Replica_Dataset \
      --scenes office_0,room_2 --sequences Sequence_1:5,Sequence_2:10 \
      --out /home/thanhnc/gauss_ws/data/uags_cache/clip_tokens
"""
from __future__ import annotations
import argparse
import os
import time
import torch

from usplat.clip_dense import OpenCLIPDense
from usplat.datasets import make_dataset


def cache_path(out, scene, seq, index):
    return os.path.join(out, scene, seq, f"tok_{index:06d}.pt")


def load_tokens(out, scene, seq, index):
    return torch.load(cache_path(out, scene, seq, index))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="replica_semnerf")
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenes", required=True, help="comma-separated scene names")
    ap.add_argument("--sequences", default="Sequence_1:5,Sequence_2:10",
                    help="comma-separated <sequence folder>:<stride>")
    ap.add_argument("--clip-model", default="ViT-B-16")
    ap.add_argument("--clip-pretrained", default="laion2b_s34b_b88k")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    clip = OpenCLIPDense(a.clip_model, a.clip_pretrained, device=a.device)
    t0, n_new, n_skip = time.time(), 0, 0
    for scene in a.scenes.split(","):
        for spec in a.sequences.split(","):
            seq, stride = spec.split(":")
            ds = make_dataset(a.dataset, os.path.join(a.root, scene, seq), stride=int(stride))
            os.makedirs(os.path.join(a.out, scene, seq), exist_ok=True)
            for i in range(len(ds)):
                index = i * int(stride)
                p = cache_path(a.out, scene, seq, index)
                if os.path.exists(p):
                    n_skip += 1
                    continue
                tok = clip.extract_tokens(ds[i]["image"]).half().cpu()
                torch.save(tok.clone(), p + ".tmp")
                os.replace(p + ".tmp", p)                 # atomic: no half-written files
                n_new += 1
            print(f"{scene}/{seq} stride {stride}: {len(ds)} frames "
                  f"({time.time() - t0:.0f} s elapsed)", flush=True)
    print(f"done: {n_new} extracted, {n_skip} already cached, {time.time() - t0:.0f} s, "
          f"device {a.device}")


if __name__ == "__main__":
    main()
