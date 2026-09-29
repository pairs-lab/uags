"""Support / aleatoric-coverage statistics of saved Replica fields (CPU, read-only)."""
import sys, torch
from usplat.vmf_field import DirectionalSemanticField
from usplat.gs_io import load_gaussians
for s in sys.argv[1:]:
    ck = torch.load(f"/home/thanhnc/gauss_ws/data/uags_fields/{s}_lseg.pt", map_location="cpu", weights_only=False)
    f = DirectionalSemanticField.from_state(ck["field"])
    g = load_gaussians(f"/home/thanhnc/gauss_ws/data/uags_maps/{s}_seq1.pt", device="cpu")
    W1, pm = f.raw.W1, f.pixel_mass
    op = g["opacities"].reshape(-1)
    seen = pm > 0
    # rendering importance proxy: total pixel mass over the mapping views
    imp = pm / pm.sum()
    print(f"== {s}: N {f.N}, keyframes {ck['timing']['keyframes']}, a0 {f.cfg.view_mass_frac}, tau_min {f.cfg.tau_min}")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        print(f"  support W1 q{q:.2f} = {torch.quantile(W1[seen][::7].float(), q):.3f}   pixel mass q{q:.2f} = {torch.quantile(pm[seen][::7].float(), q):.2f}")
    for tau in (1.0, 2.0, 3.0):
        d = W1 >= tau
        print(f"  support >= {tau}: {d.float().mean()*100:.1f}% of Gaussians, {imp[d].sum()*100:.1f}% of pixel mass")
    ale, valid = f.aleatoric()
    print(f"  u_ale defined {valid.float().mean()*100:.1f}% (mass {imp[valid].sum()*100:.1f}%); u_ale mean {ale[valid].mean():.4f}, q90 {torch.quantile(ale[valid][::3].float(), 0.9):.4f}, q99 {torch.quantile(ale[valid][::3].float(), 0.99):.4f}")
    # what if a0 were smaller: a view counts fully at pixel mass a0*H*W; W1 scales roughly as min(1, m/ref)
    print(f"  mean pixel mass per keyframe (seen) {(pm[seen] / ck['timing']['keyframes']).mean():.3f} px; ref a0*H*W = {1e-4*240*320:.2f} px")
