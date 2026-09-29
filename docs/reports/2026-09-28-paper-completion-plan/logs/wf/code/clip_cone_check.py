"""CPU check of the real dense-CLIP features the uags pipeline would feed the vMF field.

Uses usplat.clip_dense.OpenCLIPDense (MaskCLIP trick) with the cached ViT-B-16
laion2b weights on a few 7-Scenes frames (read-only from the usgloc data dir).
Measures:
  1. cone effect: cosine between random pixel features (same frame / other scene)
  2. r_bar a Gaussian would get from a 50/50 mix of two different contents
  3. PCA 512->64 (usplat.feature_pca.StreamingPCA, as run_mapping.py): text energy kept
     and per-pixel zero-shot argmax agreement before/after PCA
  4. extraction time at 640x480 and at the run_mapping default downscale 4 (160x120)
"""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import cv2
import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, "/home/thanhnc/gauss_ws/src/uags")
from usplat.clip_dense import OpenCLIPDense
from usplat.feature_pca import StreamingPCA

torch.manual_seed(0)
torch.set_num_threads(8)
ROOT = "/home/thanhnc/gauss_ws/src/usgloc/data/7scenes_source"
FRAMES = [("chess", "seq-01", 0), ("chess", "seq-01", 500),
          ("office", "seq-01", 0), ("office", "seq-01", 500),
          ("redkitchen", "seq-01", 0), ("redkitchen", "seq-01", 500)]
VOCAB = ("wall", "floor", "ceiling", "door", "window", "chair", "sofa", "table",
         "desk", "shelf", "cabinet", "monitor", "keyboard", "mouse", "book", "cup",
         "bottle", "plant", "lamp", "stairs", "curtain", "box", "bag",
         "kitchen appliance", "sink")   # same 25 classes as usgloc INDOOR_VOCABULARY

def load(scene, seq, i):
    p = f"{ROOT}/{scene}/{seq}/frame-{i:06d}.color.png"
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)

clip = OpenCLIPDense("ViT-B-16", pretrained="laion2b_s34b_b88k", device="cpu")
print(f"model ViT-B-16 laion2b_s34b_b88k, feat dim {clip.d}")
text = clip.text_embedding(list(VOCAB))            # (25, 512)

res = {}
for tag, factor in (("full640x480", 1), ("down4_160x120", 4)):
    feats, times = [], []
    for (sc, sq, i) in FRAMES:
        img = load(sc, sq, i)
        if factor > 1:
            h, w = img.shape[0] // factor, img.shape[1] // factor
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)  # = usplat.datasets.downscale
        t0 = time.time()
        f = clip.extract(torch.from_numpy(img))
        times.append(time.time() - t0)
        feats.append(f.reshape(-1, clip.d))
    res[tag] = feats
    print(f"\n[{tag}] input {img.shape[1]}x{img.shape[0]}, patch grid "
          f"{-(-img.shape[1] // 16)}x{-(-img.shape[0] // 16)}, "
          f"extract time mean {np.mean(times):.2f}s (CPU, 8 threads)")

    # 1. cone effect
    g = torch.Generator().manual_seed(1)
    same, cross = [], []
    for k, f in enumerate(feats):
        a = f[torch.randint(0, f.shape[0], (4000,), generator=g)]
        b = f[torch.randint(0, f.shape[0], (4000,), generator=g)]
        same.append((a * b).sum(-1))
        o = feats[(k + 2) % len(feats)]                 # a frame of another scene
        c = o[torch.randint(0, o.shape[0], (4000,), generator=g)]
        cross.append((a * c).sum(-1))
    same, cross = torch.cat(same), torch.cat(cross)
    q = lambda x: [round(float(v), 3) for v in torch.quantile(x, torch.tensor([0.05, 0.5, 0.95]))]
    print(f"cos(random pixel pair, same frame)   p5/p50/p95 = {q(same)}")
    print(f"cos(random pixel pair, other scene)  p5/p50/p95 = {q(cross)}")
    # 2. r_bar of an equal-weight mix of two contents: ||a+b||/2 = sqrt((1+cos)/2)
    rbar_mix = ((1 + cross.clamp(-1, 1)) / 2).sqrt()
    print(f"r_bar of 50/50 mix (other-scene pixel pairs) p5/p50/p95 = {q(rbar_mix)}")
    print(f"  fraction of mixes with r_bar <= 0.55 (r_unstable): {(rbar_mix <= 0.55).float().mean():.3f}; "
          f">= 0.85 (r_stable): {(rbar_mix >= 0.85).float().mean():.3f}")
    # whole-frame resultant length (all pixels of one frame into one Gaussian)
    rf = [float(f.mean(0).norm()) for f in feats]
    print(f"r_bar of ALL pixels of one frame pooled: {[round(x, 3) for x in rf]}")

    # 3. PCA 512 -> 64 as in run_mapping.py (fit on these frames, uncentered)
    pca = StreamingPCA(clip.d, 64)
    for f in feats:
        pca.accumulate(f)
    expl = pca.fit()
    B = pca.basis                                     # (512, 64)
    txt_keep = (text @ B).norm(dim=-1) ** 2           # energy of unit text vec in subspace
    img_all = torch.cat(feats)
    img_keep = (img_all[:20000] @ B).norm(dim=-1) ** 2
    print(f"PCA 512->64 explained second moment (image feats): {expl:.3f}")
    print(f"energy kept: image pixels mean {img_keep.mean():.3f}; text embeds mean "
          f"{txt_keep.mean():.3f} (min {txt_keep.min():.3f}, max {txt_keep.max():.3f})")
    lab512 = (img_all @ text.t()).argmax(-1)
    lab64 = (pca.project(img_all) @ pca.project(text).t()).argmax(-1)
    print(f"per-pixel zero-shot argmax agreement 512-d vs PCA-64: "
          f"{(lab512 == lab64).float().mean():.3f} over {img_all.shape[0]} pixels")
    cos_img_txt = img_all[:20000] @ text.t()
    print(f"cos(pixel, text) 512-d p5/p50/p95 = {q(cos_img_txt.flatten())}")

# agreement of per-pixel labels between full-res and downscale-4 extraction
agree = []
for f_full, f_down, (sc, sq, i) in zip(res["full640x480"], res["down4_160x120"], FRAMES):
    lf = (f_full @ text.t()).argmax(-1).reshape(480, 640)
    ld = (f_down @ text.t()).argmax(-1).reshape(120, 160)
    lf_ds = lf[2::4, 2::4]
    agree.append(float((lf_ds == ld).float().mean()))
print(f"\nzero-shot label agreement, full-res vs downscale-4 extraction, per frame: "
      f"{[round(a, 3) for a in agree]}")
