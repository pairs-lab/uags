"""Static re-observation vs changed content: r_bar separation with real MaskCLIP features.

Static proxy: the same pixel in frame i and frame i+3 of a 30 Hz 7-Scenes sequence
(camera moves a few mm; content at that pixel is nearly the same surface).
Changed-content proxy: the same pixel location in a frame of ANOTHER scene.
For each pair (a, b) the r_bar a Gaussian would get from a 50/50 mix is ||a+b||/2.
Reported both in raw 512-d and after the run_mapping PCA-64 projection (renormalized).
Input resolution: full 640x480 (best case for the extractor).
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import cv2, torch
sys.path.insert(0, "/home/thanhnc/gauss_ws/src/uags")
from usplat.clip_dense import OpenCLIPDense
from usplat.feature_pca import StreamingPCA

torch.set_num_threads(8)
ROOT = "/home/thanhnc/gauss_ws/src/usgloc/data/7scenes_source"
def load(sc, i):
    return torch.from_numpy(cv2.cvtColor(cv2.imread(f"{ROOT}/{sc}/seq-01/frame-{i:06d}.color.png"), cv2.COLOR_BGR2RGB))

clip = OpenCLIPDense("ViT-B-16", pretrained="laion2b_s34b_b88k", device="cpu")
scenes = ["chess", "office", "redkitchen"]
A = {sc: clip.extract(load(sc, 300)).reshape(-1, 512) for sc in scenes}
B = {sc: clip.extract(load(sc, 303)).reshape(-1, 512) for sc in scenes}
pca = StreamingPCA(512, 64)
for sc in scenes:
    pca.accumulate(A[sc]); pca.accumulate(B[sc])
print(f"PCA explained: {pca.fit():.3f}")

g = torch.Generator().manual_seed(0)
idx = torch.randint(0, 480 * 640, (20000,), generator=g)
q = lambda x: [round(float(v), 3) for v in torch.quantile(x, torch.tensor([0.05, 0.5, 0.95]))]
for space in ("raw512", "pca64"):
    proj = (lambda z: z) if space == "raw512" else (lambda z: pca.project(z))
    st, mv = [], []
    for k, sc in enumerate(scenes):
        a, b = proj(A[sc][idx]), proj(B[sc][idx])
        o = proj(A[scenes[(k + 1) % 3]][idx])
        st.append(((a + b) / 2).norm(dim=-1))
        mv.append(((a + o) / 2).norm(dim=-1))
    st, mv = torch.cat(st), torch.cat(mv)
    print(f"[{space}] r_bar static re-observation (pixel, frame i vs i+3) p5/p50/p95 = {q(st)}")
    print(f"[{space}] r_bar changed content (pixel, other scene)          p5/p50/p95 = {q(mv)}")
    for thr in (0.55, 0.85):
        print(f"   r_bar<={thr}: static {(st <= thr).float().mean():.3f}  changed {(mv <= thr).float().mean():.3f}")
    # AUROC of 1-r_bar for separating changed from static
    s = torch.cat([1 - st, 1 - mv]); y = torch.cat([torch.zeros_like(st), torch.ones_like(mv)])
    order = torch.argsort(s); ranks = torch.empty_like(s); ranks[order] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    npos, nneg = y.sum(), (1 - y).sum()
    auroc = (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    print(f"   AUROC(1-r_bar: changed vs static) = {float(auroc):.3f}")
