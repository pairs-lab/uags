# A_d(kappa) = I_{d/2}(kappa)/I_{d/2-1}(kappa): shrink factor of E[mu] under vMF(mu*, kappa), d=64.
# kappa_hat from the joint prior with rbar=0.85 (real static MaskCLIP rho_hat ~0.85, math E5/C3-iii);
# posterior concentration of mu = kappa_hat * n * rbar for n views.
from scipy.special import ive
from scipy.optimize import brentq
d = 64
A = lambda k: ive(d/2, k) / ive(d/2 - 1, k)
kh = brentq(lambda k: A(k) - 0.85, 1e-3, 1e6)
print(f"d={d}: kappa_hat for A_d=0.85 -> {kh:.1f}")
for n in (1, 2, 5, 10, 50):
    kp = kh * n * 0.85
    print(f"n={n:3d} views: kappa_post={kp:8.1f}  A_d(kappa_post)={A(kp):.4f}  (effective softmax temperature factor)")
