"""C3: real dense CLIP features (the project's own MaskCLIP extractor, CPU) on 7-Scenes chess.
Measures: spatial correlation of pixel features (does n_eff = sum over pixels overcount?),
one-view r_bar of a Gaussian-sized footprint, cross-view r_bar of STATIC 3D points,
and what the PCA-64 compression does to text queries and to r_bar.
Project files are only imported (PYTHONDONTWRITEBYTECODE=1), never written."""
import sys, os, time
import numpy as np, torch, torch.nn.functional as Fn
from PIL import Image
sys.path.insert(0, "/home/thanhnc/gauss_ws/src/uags")
from usplat.clip_dense import OpenCLIPDense  # noqa: E402

torch.set_num_threads(16)
rng = np.random.default_rng(0)
ROOT = "/home/thanhnc/gauss_ws/src/usgloc/data/7scenes_source/chess/seq-01"
FX = FY = 585.0; CX, CY = 320.0, 240.0          # 7-Scenes depth intrinsics (approx. for colour too)

t0 = time.time()
clip = OpenCLIPDense("ViT-B-16", "laion2b_s34b_b88k", device="cpu")
print(f"model loaded in {time.time()-t0:.1f}s, d={clip.d}, patch={clip.patch}")


def load(i, ds):
    im = Image.open(f"{ROOT}/frame-{i:06d}.color.png").convert("RGB")
    if ds > 1:
        im = im.resize((640 // ds, 480 // ds), Image.BILINEAR)
    dep = np.array(Image.open(f"{ROOT}/frame-{i:06d}.depth.png")).astype(np.float64)
    dep[dep == 65535] = 0; dep /= 1000.0
    T = np.loadtxt(f"{ROOT}/frame-{i:06d}.pose.txt")
    return np.array(im), dep, T


def feats(img):
    with torch.no_grad():
        return clip.extract(torch.from_numpy(img)).numpy().astype(np.float64)   # (H,W,512) unit


def rbar(Z, w=None):
    w = np.ones(len(Z)) if w is None else w
    return np.linalg.norm((w[:, None] * Z).sum(0)) / w.sum()


# ---------------- (i) spatial correlation at the two resolutions ----------------------------
for ds in (4, 1):
    img, _, _ = load(0, ds)
    t1 = time.time(); F = feats(img); te = time.time() - t1
    H, W, _ = F.shape
    row = []
    for dlt in (1, 2, 4, 8, 16, 32, 64):
        if dlt >= W: break
        c = (F[:, :-dlt] * F[:, dlt:]).sum(-1).mean()
        row.append(f"{dlt}px:{c:.3f}")
    # global mean cosine of random pixel pairs (the "unrelated" floor)
    a = F.reshape(-1, 512)[rng.integers(0, H * W, 4000)]; b = F.reshape(-1, 512)[rng.integers(0, H * W, 4000)]
    print(f"[i] ds={ds} image {W}x{H} (extract {te:.1f}s, {int(np.ceil(H/16))}x{int(np.ceil(W/16))} tokens): "
          f"mean cos at horizontal offset {' '.join(row)} | random pairs {np.mean((a*b).sum(-1)):.3f}")

# ---------------- (ii) one-view footprint r_bar ------------------------------------------------
img, _, _ = load(0, 4); F4 = feats(img); H, W, _ = F4.shape
yy, xx = np.mgrid[0:H, 0:W]
print("[ii] ds=4, one view, disc footprints (equal weights): n_pixels ~ n_eff if alpha=1")
for rad in (2, 4, 8, 16):
    rbs, nps, ntok = [], [], []
    for _ in range(300):
        cy, cx = rng.integers(rad, H - rad), rng.integers(rad, W - rad)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2
        Z = F4[m]; rbs.append(rbar(Z)); nps.append(m.sum())
        ntok.append(len(set(zip((yy[m] // 16).tolist(), (xx[m] // 16).tolist()))))
    rbs = np.array(rbs)
    print(f"     radius {rad:>2}px: n_pix={np.mean(nps):6.1f}  distinct 16px tokens={np.mean(ntok):4.1f}  "
          f"r_bar median={np.median(rbs):.3f} min={rbs.min():.3f}  frac r_bar>=0.85 (r_stable): {np.mean(rbs>=0.85):.2f}")

# ---------------- (iii) cross-view r_bar of static 3D points -------------------------------------
frames = list(range(0, 200, 10))
data = {i: load(i, 4) for i in frames}
Fs = {i: feats(data[i][0]) for i in frames}
_, dep0, T0 = data[0]
v, u = np.nonzero(dep0 > 0.3)
pick = rng.choice(len(u), 600, replace=False); u, v = u[pick], v[pick]
z = dep0[v, u]
Xc = np.stack([(u - CX) * z / FX, (v - CY) * z / FY, z, np.ones_like(z)], 0)
Xw = T0 @ Xc
obs = [[] for _ in range(len(u))]
for i in frames:
    _, dep, T = data[i]
    Xi = np.linalg.inv(T) @ Xw
    zi = Xi[2]; ui = FX * Xi[0] / zi + CX; vi = FY * Xi[1] / zi + CY
    ok = (zi > 0.1) & (ui >= 0) & (ui < 639.5) & (vi >= 0) & (vi < 479.5)
    uu = np.clip(np.round(ui).astype(int), 0, 639); vv = np.clip(np.round(vi).astype(int), 0, 479)
    dz = np.abs(dep[vv, uu] - zi)
    vis = ok & (dep[vv, uu] > 0) & (dz < 0.05)
    for j in np.nonzero(vis)[0]:
        obs[j].append(Fs[i][min(vv[j] // 4, 119), min(uu[j] // 4, 159)])
nv = np.array([len(o) for o in obs]); keep = nv >= 5
rb_x = np.array([rbar(np.array(o)) for o, k in zip(obs, keep) if k])
nvk = nv[keep]
rho = np.array([np.sqrt(max((n * r * r - 1) / (n - 1), 0)) for n, r in zip(nvk, rb_x)])
q = lambda a: " ".join(f"{p}%:{np.percentile(a, p):.3f}" for p in (10, 50, 90))
print(f"[iii] static chess points seen in >=5 of {len(frames)} frames (stride 10): {keep.sum()} pts, "
      f"median views {np.median(nvk):.0f}")
print(f"      cross-view r_bar (1 obs per view): {q(rb_x)} | frac <=0.55 (r_unstable): {np.mean(rb_x<=0.55):.2f} "
      f"| frac >=0.85 (r_stable): {np.mean(rb_x>=0.85):.2f}")
print(f"      bias-corrected rho_hat: {q(rho)}")
# same points, pixel-weighted within ONE view (5x5 window, 25 px) -> what a single keyframe gives
rb_1 = []
for j in np.nonzero(keep)[0]:
    y0, x0 = min(v[j] // 4, 119), min(u[j] // 4, 159)
    Z = Fs[0][max(y0-2, 0):y0+3, max(x0-2, 0):x0+3].reshape(-1, 512); rb_1.append(rbar(Z))
print(f"      same points, one view, 5x5 px window: r_bar {q(np.array(rb_1))}")

# ---------------- (iv) PCA-64 (uncentred second moment, as usplat/feature_pca.py) ---------------
Zall = np.concatenate([Fs[i].reshape(-1, 512) for i in frames[:10]])
Zs = Zall[rng.choice(len(Zall), min(len(Zall), 20000), replace=False)]
M = Zs.T @ Zs / len(Zs)
ev, V = np.linalg.eigh(M); o = np.argsort(ev)[::-1][:64]; B = V[:, o]
print(f"[iv] PCA-64 explained second moment on image feats: {ev[o].sum()/ev.sum():.3f}")
prompts = ["chess board", "chess piece", "table", "chair", "wall", "floor",
           "computer monitor", "keyboard", "cabinet", "window", "door", "carpet"]
T_ = clip.text_embedding(prompts).numpy().astype(np.float64)
ret_t = (np.linalg.norm(T_ @ B, axis=1) ** 2)
ret_i = (np.linalg.norm(Zs @ B, axis=1) ** 2)
print(f"     energy kept by PCA-64: image pixels mean {ret_i.mean():.3f} | text prompts mean {ret_t.mean():.3f} "
      f"(min {ret_t.min():.3f}, max {ret_t.max():.3f})")
full_lab = (Zs @ T_.T).argmax(1)
Pz = Zs @ B; Pz /= np.linalg.norm(Pz, axis=1, keepdims=True)
Pt = T_ @ B; Pt /= np.linalg.norm(Pt, axis=1, keepdims=True)
pca_lab = (Pz @ Pt.T).argmax(1)
print(f"     argmax-label agreement over {len(prompts)} prompts, full-512 vs PCA-64 space: {np.mean(full_lab==pca_lab):.3f}")
print(f"     mean image-text cos full: {(Zs @ T_.T).mean():.3f} | after PCA+renorm: {(Pz @ Pt.T).mean():.3f}")
rb_p = []
for o_, k in zip(obs, keep):
    if k:
        Zp = np.array(o_) @ B; Zp /= np.linalg.norm(Zp, axis=1, keepdims=True); rb_p.append(rbar(Zp))
print(f"     cross-view r_bar of the static points in PCA-64 space: {q(np.array(rb_p))} (vs 512-D above)")
print(f"total time {time.time()-t0:.1f}s")
