"""Numerical checks of uags formula behaviour (CPU, uses the real usplat classes)."""
import sys, torch
import torch.nn.functional as Fn
sys.path.insert(0, "/home/thanhnc/gauss_ws/src/uags")
from usplat.vmf_field import DirectionalSemanticField
from usplat.consolidation import Consolidator
from usplat.efe_planner import EFEPlanner

d = 64
z = Fn.normalize(torch.randn(d), dim=0)
f = DirectionalSemanticField(6, d, kappa_lik=10.0)
# G0: one observation, weight 1   G1: 0.3 obs  G2: 50 identical obs
# G3: 50 obs, half z half -z     G4: 5 obs, cos 0.6 mix   G5: never observed
f.R[0] = z;            f.n_eff[0] = 1.0
f.R[1] = 0.3 * z;      f.n_eff[1] = 0.3
f.R[2] = 50 * z;       f.n_eff[2] = 50.0
f.R[3] = 0 * z;        f.n_eff[3] = 50.0
y = Fn.normalize(torch.randn(d), dim=0); y = Fn.normalize(y - (y @ z) * z, dim=0)
w = 0.6 * z + 0.8 * y
f.R[4] = 2.5 * z + 2.5 * w; f.n_eff[4] = 5.0
rbar = f.mean_resultant_length(); rt = f.r_bar_corrected()
ale, valid = f.aleatoric(min_obs=5.0)
print("n_eff      ", [round(x, 2) for x in f.n_eff.tolist()])
print("r_bar      ", [round(x, 3) for x in rbar.tolist()])
print("r_tilde    ", [round(x, 3) for x in rt.tolist()], "(paper eq. l.332)")
print("u_epi      ", [round(x, 3) for x in f.epistemic().tolist()])
print("u_ale(>=5) ", [round(x, 3) if x == x else 'nan' for x in ale.tolist()])
print("kappa_post ", [round(x, 1) for x in f.kappa_post().tolist()])
# kappa_lik has no effect on mu_post, r_bar, u_epi, u_ale when prior_kappa=0
f2 = DirectionalSemanticField(6, d, kappa_lik=1000.0); f2.R.copy_(f.R); f2.n_eff.copy_(f.n_eff)
print("kappa_lik 10 vs 1000: max |mu diff| =",
      float((f.mu_post() - f2.mu_post()).abs().max()),
      "| r_bar equal:", bool(torch.equal(f.mean_resultant_length(), f2.mean_resultant_length())))
# consolidation decay leaves r_bar and mu unchanged, only shrinks n_eff
f.R[3] = 10 * z; f.n_eff[3] = 50.0          # r_bar 0.2 -> unstable
c = Consolidator(f, min_obs=5.0, r_unstable=0.5, r_stable=0.85, forget_lambda=0.9)
r_before, mu_before = f.mean_resultant_length()[3].item(), f.mu_post()[3].clone()
steps = 0
while c.step()["unstable"][3]:
    steps += 1
print(f"G3 (r_bar {r_before:.3f}, n_eff 50): decayed for {steps} steps until n_eff "
      f"{f.n_eff[3]:.2f} < min_obs; r_bar after {f.mean_resultant_length()[3]:.3f}; "
      f"mu change {float((f.mu_post()[3] - mu_before).abs().max()):.1e}; "
      f"in robust set: {bool(c.robust_mask()[3])}; u_epi now {f.epistemic()[3]:.3f}")
# EFE: unobserved/under-min_obs Gaussians get discount 1 (not in paper eq.)
p = EFEPlanner(f, lam=0.0)
vis = torch.eye(6)
print("EFE epistemic value per Gaussian (visibility = one-hot):",
      [round(x, 3) for x in p.epistemic_value(vis).tolist()])
print("paper formula nu*u_epi*(1-u_ale), u_ale undefined -> nan:",
      [round(x, 3) if x == x else 'nan' for x in (f.epistemic() * (1 - ale)).tolist()])
