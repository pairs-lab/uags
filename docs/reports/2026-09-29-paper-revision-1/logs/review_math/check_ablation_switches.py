# What do the ablation switches change? evidence="pixels" and estimator="legacy".
import sys; sys.path.insert(0, "/home/thanhnc/gauss_ws/src/uags")
import torch
import torch.nn.functional as Fn
from usplat.vmf_field import DirectionalSemanticField, FieldConfig
from scipy.stats import spearmanr
torch.manual_seed(0)
N, d, P = 2000, 16, 307200            # 640x480 -> ref = a0*H*W = 30.72 px of mass
ref = 1e-4 * P
cfgs = dict(views=FieldConfig(change_test=False),
            pixels=FieldConfig(change_test=False, evidence="pixels"),
            pixels_epi=FieldConfig(change_test=False, evidence="pixels", epistemic="pixels"),
            joint=FieldConfig(change_test=False, estimator="joint"),
            legacy_est=FieldConfig(change_test=False, estimator="legacy"))
fs = {k: DirectionalSemanticField(N, d, c, dtype=torch.float64) for k, c in cfgs.items()}
base = Fn.normalize(torch.randn(N, d, dtype=torch.float64), dim=-1)
nviews = torch.randint(1, 12, (N,))
scale = torch.exp(torch.randn(N, dtype=torch.float64) * 1.5) * 40   # pixel footprints vary a lot
for v in range(12):
    seen = nviews > v
    m = scale * (0.5 + torch.rand(N, dtype=torch.float64))
    m[~seen] = 0
    u = Fn.normalize(base + 0.8*torch.randn(N, d, dtype=torch.float64), dim=-1) * 0.9
    gid = torch.arange(N)[m > 0]
    for f in fs.values():
        f.update_sparse(gid, m[gid], u[gid], num_pixels=P, normalize_z=False)
e = {k: f.epistemic() for k, f in fs.items()}
print("Spearman(u_epi views, u_epi pixels[evidence only])      =", round(spearmanr(e['views'], e['pixels']).correlation, 4))
print("max |u_epi views - u_epi pixels[evidence only]|          =", float((e['views'] - e['pixels']).abs().max()))
print("Spearman(u_epi views, u_epi pixels[+epistemic=pixels])  =", round(spearmanr(e['views'], e['pixels_epi']).correlation, 4))
fv, fp = fs['views'], fs['pixels']
print("Kish n_K (views weights) vs Kish n_K (pixel weights), first 5:",
      fv.raw.kish()[:5].tolist(), fp.raw.kish()[:5].tolist())
# aleatoric: defined sets and values
av, okv = fv.aleatoric(); ap, okp = fp.aleatoric()
print("u_ale defined: views", int(okv.sum()), " pixels", int(okp.sum()),
      " Gaussians with 1 view defined (pixels):", int((okp & (nviews == 1)).sum()))
both = okv & okp
print("u_ale on commonly defined: max |diff| =", float((av[both] - ap[both]).abs().max()))
# estimator="legacy" also swaps the support gate
fl, fj = fs['legacy_est'], fs['joint']
_, okl = fl.aleatoric(); _, okj = fj.aleatoric()
print("gate: support>=3 (views/joint) defined:", int(okj.sum()), "; estimator=legacy defined:", int(okl.sum()),
      "; of those with support<3:", int((okl & (fv.raw.support() < 3)).sum()))
