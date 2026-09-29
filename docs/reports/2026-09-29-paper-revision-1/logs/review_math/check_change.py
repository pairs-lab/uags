# Change family in the code (update_sparse) vs the paper's formulas.
import sys; sys.path.insert(0, "/home/thanhnc/gauss_ws/src/uags")
import torch, math
import torch.nn.functional as Fn
from usplat.vmf_field import DirectionalSemanticField, FieldConfig
torch.manual_seed(0)
N, d, V, P = 50, 8, 25, 1000   # P = num_pixels -> ref = 1e-4*1000 = 0.1
ref = 1e-4 * P
f = DirectionalSemanticField(N, d, FieldConfig(), dtype=torch.float64)
cfg = f.cfg
print("defaults: fast_memory", cfg.fast_memory, "persist", cfg.change_persist, "tau_min", cfg.tau_min,
      "q", cfg.change_quantile, "calib_mode", cfg.calib_mode, "a0", cfg.view_mass_frac)
base = Fn.normalize(torch.randn(N, d, dtype=torch.float64), dim=-1)
hist_u, hist_w = [], []
c_paper = [[] for _ in range(N)]   # defined per-view c by the paper's lambda=0 formula
maxdiff = 0.0; maxdiff_R = 0.0
for v in range(V):
    # pixel mass per Gaussian: some full views (m >= ref), some partial, some unseen
    m = torch.rand(N, dtype=torch.float64) * 0.2
    m[torch.rand(N) < 0.2] = 0.0
    u = Fn.normalize(base + 0.6*torch.randn(N, d, dtype=torch.float64), dim=-1) * (0.7 + 0.3*torch.rand(N, 1, dtype=torch.float64))
    gid = torch.arange(N)[m > 0]
    # update_sparse normalises z per triple; pass one triple per Gaussian with R_v = m u
    # -> give z = u/||u|| and weight m*||u||? No: R_v/m_v must equal u, so use two triples.
    # Simplest: set normalize_z=False, z = u (non-unit), weight = m: R_v = m u, m_v = m.
    f.update_sparse(gid, m[gid], u[gid], num_pixels=P, normalize_z=False)
    w = torch.where(m > 0, (m/ref).clamp(max=1.0), torch.zeros_like(m))
    hist_u.append(u); hist_w.append(w)
    # paper, lambda = 0: c_t = 1 - cos(sum_{v<t} w u, u_t), defined if support>=3, w_t>=1, older mass>=1
    Wu = torch.stack(hist_w); U = torch.stack(hist_u)
    old = (Wu[:-1, :, None]*U[:-1]).sum(0); nold = Wu[:-1].sum(0)
    tot1 = Wu.sum(0); tot2 = (Wu**2).sum(0)
    supp = torch.minimum(tot1, tot1**2/tot2.clamp_min(1e-12))
    defined = (w > 0) & (supp >= 3) & (w >= 1) & (nold >= 1)
    c = 1 - Fn.cosine_similarity(old, u, dim=-1, eps=1e-8)
    for n in range(N):
        if defined[n]:
            c_paper[n].append(float(c[n]))
    cbar = torch.tensor([min(x[-2:]) if len(x) >= 2 else float("nan") for x in c_paper], dtype=torch.float64)
    ok = torch.isfinite(cbar) & defined
    if ok.any():
        maxdiff = max(maxdiff, float((cbar[ok] - f.c_ring.min(1).values[ok]).abs().max()))
    # buckets split R without overlap
    maxdiff_R = max(maxdiff_R, float((f.Rs + f.Rf - f.raw.R).abs().max()))
print(f"max |c_bar(paper formula) - c_bar(code)| over defined views: {maxdiff:.2e}")
print(f"max |R_old + R_new - R| : {maxdiff_R:.2e}")
print("null histogram counts per history bucket:", f.c_hist.sum(1).tolist())
