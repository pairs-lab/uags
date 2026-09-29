# Is s = min(n_K, W1) ever different from W1 when view weights are in (0, 1]?
import sys; sys.path.insert(0, "/home/thanhnc/gauss_ws/src/uags")
import torch
import torch.nn.functional as Fn
from usplat.vmf_field import DirectionalSemanticField, FieldConfig
torch.manual_seed(0)
N, d, P = 20000, 8, 307200
ref = 1e-4 * P
f = DirectionalSemanticField(N, d, FieldConfig(change_test=False), dtype=torch.float64)
g = DirectionalSemanticField(N, d, FieldConfig(change_test=False, epistemic="views"), dtype=torch.float64)
scale = torch.exp(torch.randn(N, dtype=torch.float64) * 2.0) * ref       # footprints from tiny to huge
for v in range(30):
    m = scale * torch.rand(N, dtype=torch.float64) * (torch.rand(N) < 0.6)
    gid = torch.arange(N)[m > 0]
    u = Fn.normalize(torch.randn(len(gid), d, dtype=torch.float64), dim=-1)
    f.update_sparse(gid, m[gid], u, num_pixels=P, normalize_z=False)
    g.update_sparse(gid, m[gid], u, num_pixels=P, normalize_z=False)
W1, nK, s = f.raw.W1, f.raw.kish(), f.raw.support()
seen = W1 > 0
print("Gaussians seen:", int(seen.sum()))
print("share with n_K < W1 (Kish binds):", float((nK[seen] < W1[seen] - 1e-12).float().mean()))
print("max |s - W1|:", float((s - W1).abs().max()))
print("max |u_epi(support) - u_epi('views' = 1/(1+W1))|:", float((f.epistemic() - g.epistemic()).abs().max()))
print("min over seen of n_K - W1:", float((nK[seen] - W1[seen]).min()))
