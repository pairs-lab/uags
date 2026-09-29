"""Check plan F3: centre at read-out, R_c = R - n*zbar, and use R_c for dispersion (u_ale).

Question 1: for a Gaussian whose observations are all IDENTICAL (zero aleatoric spread),
what does the centred dispersion give?  With n identical unit obs z0, R_c/n = z0 - zbar,
so rbar_c = ||z0 - zbar|| (not 1).  If this varies a lot across pixels/classes, u_ale
(= 1 - rbar_c * n/(n+c0)) depends on the class, not on the spread.

Question 2: static-vs-changed AUROC for three dispersion read-outs on a 50/50 mix:
  raw:        ||(a+b)/2||
  F3 (plan):  ||(a+b)/2 - zbar||                (centre at read-out, no renormalisation)
  renorm:     ||(a_hat+b_hat)/2||, a_hat = (a-zbar)/||a-zbar||  (per-pixel centre + renormalise;
              needs its own accumulation channel)
Static = same pixel, frame i vs i+3. Changed = pixel p of frame i vs random pixel of a frame
>=60 frames away in the SAME scene (same proxy style as math C3b; not a real rearrangement).
CPU only, MaskCLIP ViT-B-16 laion2b via the uags OpenCLIPDense (read-only import).
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import cv2, torch
sys.path.insert(0, "/home/thanhnc/gauss_ws/src/uags")
from usplat.clip_dense import OpenCLIPDense

torch.set_num_threads(8)
ROOT = "/home/thanhnc/gauss_ws/src/usgloc/data/7scenes_source"
def load(sc, i):
    return torch.from_numpy(cv2.cvtColor(cv2.imread(f"{ROOT}/{sc}/seq-01/frame-{i:06d}.color.png"), cv2.COLOR_BGR2RGB))

clip = OpenCLIPDense("ViT-B-16", pretrained="laion2b_s34b_b88k", device="cpu")
scenes = ["chess", "office", "redkitchen"]
frames = [0, 3, 100, 103, 200, 203, 300, 303]
g = torch.Generator().manual_seed(0)
q = lambda x: [round(float(v), 3) for v in torch.quantile(x, torch.tensor([0.05, 0.5, 0.95]))]

def auroc(s_static, s_changed):  # score = 1 - rbar ; higher = more "changed"
    s = torch.cat([s_static, s_changed]); y = torch.cat([torch.zeros_like(s_static), torch.ones_like(s_changed)])
    order = torch.argsort(s); ranks = torch.empty_like(s); ranks[order] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    npos, nneg = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))

allc = {"raw": ([], []), "F3": ([], []), "renorm": ([], [])}
cons_all = []
for sc in scenes:
    F = {i: clip.extract(load(sc, i)).reshape(-1, 512) for i in frames}
    zbar = torch.cat(list(F.values())).mean(0)                 # scene mean feature (not unit)
    print(f"[{sc}] ||zbar|| = {zbar.norm():.3f}")
    # Q1: consistent Gaussian -> rbar_c = ||z0 - zbar||
    cons = (F[0] - zbar).norm(dim=-1)
    cons_all.append(cons)
    print(f"[{sc}] consistent content: rbar_c = ||z0 - zbar||  p5/p50/p95 = {q(cons)}  (raw rbar would be 1.0)")
    idx = torch.randint(0, F[0].shape[0], (20000,), generator=g)
    idx2 = torch.randint(0, F[0].shape[0], (20000,), generator=g)
    for (i, j, far) in [(0, 3, 200), (100, 103, 300), (200, 203, 0), (300, 303, 100)]:
        a, b, o = F[i][idx], F[j][idx], F[far][idx2]
        allc["raw"][0].append(((a + b) / 2).norm(dim=-1)); allc["raw"][1].append(((a + o) / 2).norm(dim=-1))
        allc["F3"][0].append(((a + b) / 2 - zbar).norm(dim=-1)); allc["F3"][1].append(((a + o) / 2 - zbar).norm(dim=-1))
        nz = lambda x: (x - zbar) / (x - zbar).norm(dim=-1, keepdim=True)
        allc["renorm"][0].append(((nz(a) + nz(b)) / 2).norm(dim=-1)); allc["renorm"][1].append(((nz(a) + nz(o)) / 2).norm(dim=-1))

cons_all = torch.cat(cons_all)
print(f"[all] consistent content rbar_c p5/p50/p95 = {q(cons_all)}; so u_ale(n=10,c0=1) for ZERO spread = "
      f"{q(1 - cons_all * 10 / 11)}")
for k, (st, ch) in allc.items():
    st, ch = torch.cat(st), torch.cat(ch)
    print(f"[{k:6s}] static rbar p5/p50/p95 = {q(st)} | changed = {q(ch)} | AUROC(1-rbar) = {auroc(1 - st, 1 - ch):.3f}")
# how much of the F3 static score is explained by ||z - zbar|| of the content (class effect)?
st = torch.cat(allc["F3"][0]); ch = torch.cat(allc["F3"][1])
print(f"[F3] fraction of STATIC pairs with lower rbar_c than the median CHANGED pair: {(st < ch.median()).float().mean():.3f}")
