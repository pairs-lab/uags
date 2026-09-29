"""
check_clip2d.py — plan K8 capacity gate: are 2D dense CLIP features good enough on Replica?

Protocol (fixed before running; VALIDATION scenes only):
  * scenes office_0 and room_2 (Semantic-NeRF Replica), Sequence_1, 20 frames each
    (indices 0, 45, ..., 855), full resolution 640x480;
  * vocabulary = the classes with GT pixels in these 20 frames of the scene (id 0 ignored),
    prompts from configs/vocab_replica.json, one of two templates;
  * every labelled pixel is classified by argmax cosine to the class text embeddings;
    metrics: mIoU over the present classes and pixel accuracy;
  * variants:
      pixel      per-pixel dense CLIP (usplat.clip_dense.OpenCLIPDense, ViT-B/16 laion2b,
                 ClearCLIP last block; run 1 of this script used a broken extractor, see report)
      mask       FastSAM masks + CLIP crop embeddings (usgloc MaskClipEncoder, read-only
                 import); a pixel takes its smallest covering mask, else the pixel feature
      pixel_pca  per-pixel features through PCA-64 (fit on these frames' features, no text)
                 scored with the decode-first text projection (F7), as the field does
  * pass: best variant >= 15 mIoU on both scenes (plan threshold, an assumption).
Usage: PYTHONPATH=. python3 scripts/check_clip2d.py --out log.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as Fn

from usplat.datasets import ReplicaSemNeRFDataset
from usplat.clip_dense import OpenCLIPDense
from usplat.feature_pca import StreamingPCA

ROOT = "/home/thanhnc/gauss_ws/data/replica_semnerf/Replica_Dataset"
USGLOC = "/home/thanhnc/gauss_ws/src/usgloc"
TEMPLATES = {"photo": ("a photo of a {}.",), "room": ("a photo of a {} in a room.",),
             "ensemble": ("a photo of a {}.", "a photo of the {}.", "there is a {} in the scene.",
                          "a photo of a {} in a room.", "a cropped photo of a {}.")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="office_0,room_2")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--no-mask", action="store_true", help="skip the FastSAM variant")
    ap.add_argument("--pretrained", default="laion2b_s34b_b88k", help="open_clip weights tag")
    ap.add_argument("--extractor", default="clearclip",
                    help="usplat.extractors name (clearclip = OpenCLIPDense default)")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = json.load(open(os.path.join(os.path.dirname(__file__), "..", "configs",
                                        "vocab_replica.json")))
    from usplat.extractors import get_extractor
    kw = {"pretrained": a.pretrained} if a.extractor == "clearclip" else {}
    clip = get_extractor(a.extractor, device=dev, **kw)
    D = clip.d
    masker = None
    if not a.no_mask:
        sys.path.insert(0, USGLOC)
        from usgloc.mapping import MaskClipEncoder
        masker = MaskClipEncoder(fastsam_weights=os.path.join(USGLOC, "FastSAM-s.pt"), device=dev)
    out = dict(protocol=__doc__.split("Protocol")[1].split("Usage")[0].strip(), results={})
    t0 = time.time()
    for scene in a.scenes.split(","):
        ds = ReplicaSemNeRFDataset(os.path.join(ROOT, scene, "Sequence_1"))
        idx = [int(i) for i in np.linspace(0, 855, a.frames)]
        present = sorted(set().union(*[set(torch.unique(ds[i]["labels"]).tolist()) for i in idx])
                         - set(vocab["ignore_ids"]))
        cls = torch.tensor(present)
        texts = {tn: clip.text_embedding([vocab["classes"][str(c)]["prompt"] for c in present],
                                         templates=tpl).cpu() for tn, tpl in TEMPLATES.items()}
        # pass 1: PCA basis from the frames' features (no text, no labels)
        pca = StreamingPCA(D, a.pca_dim)
        for i in idx:
            pca.accumulate(clip.extract(ds[i]["image"]).cpu().reshape(-1, D), max_samples=20000)
        pca.fit()
        # pass 2: score every frame, accumulate intersections / unions per class
        stats = {}
        for i in idx:
            fr = ds[i]
            gt = fr["labels"]
            feat = clip.extract(fr["image"]).cpu()                      # (H,W,512)
            variants = {"pixel": feat}
            if masker is not None:
                mf = feat.clone()
                pairs = masker.encode_frame(fr["image"].numpy()[..., ::-1].copy())
                pairs.sort(key=lambda p: -int(p[0].sum()))              # small masks last
                for m, e in pairs:
                    mf[m] = Fn.normalize(e.float(), dim=0)
                variants["mask"] = mf
            fp = pca.project(feat.reshape(-1, D)).reshape(*feat.shape[:2], a.pca_dim)
            for tn, text in texts.items():
                scores = {vn: f @ text.t() for vn, f in variants.items()}
                scores["pixel_pca"] = fp @ pca.project_text(text).t()
                for vn, sc in scores.items():
                    pred = cls[sc.argmax(-1)]
                    st = stats.setdefault(f"{vn}/{tn}", dict(inter=np.zeros(len(present)),
                                                             union=np.zeros(len(present)),
                                                             gtn=np.zeros(len(present)),
                                                             correct=0, total=0))
                    valid = gt > 0
                    st["correct"] += int((pred[valid] == gt[valid]).sum())
                    st["total"] += int(valid.sum())
                    for j, c in enumerate(present):
                        p_, g_ = pred == c, gt == c
                        st["inter"][j] += int((p_ & g_).sum())
                        st["union"][j] += int((p_ | g_).sum())
                        st["gtn"][j] += int(g_.sum())
        res = {}
        for key, st in stats.items():
            has = st["gtn"] > 0
            m = float(np.mean(st["inter"][has] / np.maximum(st["union"][has], 1)))
            res[key] = dict(miou=m, acc=st["correct"] / max(st["total"], 1))
            print(f"{scene:9s} {key:18s} mIoU {m * 100:5.1f}  acc {res[key]['acc'] * 100:5.1f}  "
                  f"({len(present)} classes)", flush=True)
        out["results"][scene] = dict(n_classes=len(present), classes=present, frames=idx, metrics=res)
    best = {}
    for key in next(iter(out["results"].values()))["metrics"]:
        best[key] = min(r["metrics"][key]["miou"] for r in out["results"].values())
    out["worst_scene_miou"] = best
    out["pass_15"] = max(best.values()) >= 0.15
    out["seconds"] = time.time() - t0
    out["device"] = dev
    out["pretrained"] = a.pretrained
    out["extractor"] = a.extractor
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(dict(worst_scene_miou=best, pass_15=out["pass_15"]), indent=1))


if __name__ == "__main__":
    main()
