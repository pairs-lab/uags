"""C1: vMF tempered-likelihood conjugacy, kappa* formula, angular-variance approximation,
and what a fixed kappa means in high dimension. Read-only w.r.t. the project."""
import numpy as np
import mpmath as mp
from scipy.stats import vonmises_fisher

rng = np.random.default_rng(0)


def A(d, k):
    """Mean resultant length of vMF_d(kappa): I_{d/2}(k)/I_{d/2-1}(k) (exact, mpmath)."""
    return float(mp.besseli(d / 2, k) / mp.besseli(d / 2 - 1, k))


def logC(d, k):
    """log normaliser of vMF_d(kappa) w.r.t. surface measure."""
    k = mp.mpf(k)
    return float((d / 2 - 1) * mp.log(k) - (d / 2) * mp.log(2 * mp.pi) - mp.log(mp.besseli(d / 2 - 1, k)))


# ---- 1. conjugacy of the tempered product on S^2 (d=3) ---------------------------------
d, kappa, kappa0 = 3, 7.0, 2.0
mu0 = np.array([0, 0, 1.0])
Z = rng.normal(size=(40, 3)); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
k = rng.uniform(0, 1, size=40)
M = 400_000                                  # Fibonacci sphere quadrature
i = np.arange(M) + 0.5
phi = np.arccos(1 - 2 * i / M); th = np.pi * (1 + 5 ** 0.5) * i
G = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
# tempered product INCLUDING the C_d(kappa)^k factors, then normalised numerically
logpost = kappa0 * G @ mu0 + (k[:, None] * (logC(3, kappa) + kappa * (Z @ G.T))).sum(0)
post = np.exp(logpost - logpost.max()); post /= post.sum() * (4 * np.pi / M)
eta = kappa0 * mu0 + kappa * (k[:, None] * Z).sum(0)
kst = np.linalg.norm(eta)
vmf = np.exp(logC(3, kst) + G @ eta)
print(f"[1] conjugacy d=3: kappa*={kst:.3f}  max|post-vMF|/max(vMF) = "
      f"{np.abs(post - vmf).max() / vmf.max():.2e}")

# ---- 2. kappa* vs kappa * n_eff * r_bar ------------------------------------------------
R = (k[:, None] * Z).sum(0); n = k.sum(); rbar = np.linalg.norm(R) / n
print(f"[2] kappa0=0: ||kappa R|| = {kappa*np.linalg.norm(R):.6f}  kappa*n*rbar = {kappa*n*rbar:.6f} (identical)")
print(f"    kappa0={kappa0}: ||eta|| = {kst:.4f}  kappa*n*rbar = {kappa*n*rbar:.4f}  "
      f"bound [{kappa*n*rbar-kappa0:.4f}, {kappa*n*rbar+kappa0:.4f}]")

# ---- 3. posterior angular variance (d-1)/kappa* ------------------------------------------
print("[3] E[theta^2] of vMF_d(kappa*) (MC, 20000 draws) vs (d-1)/kappa*;  exact E[cos]=A_d")
print(f"    {'d':>4} {'kappa*':>8} {'MC E[th^2]':>11} {'(d-1)/k*':>10} {'ratio':>7} {'A_d(k*)':>8}")
for d in (3, 64, 512):
    for ks in (10, 50, 200, 1000, 10000):
        mu = np.zeros(d); mu[0] = 1
        x = vonmises_fisher(mu, ks).rvs(20000, random_state=rng)
        th2 = np.arccos(np.clip(x[:, 0], -1, 1)) ** 2
        approx = (d - 1) / ks
        print(f"    {d:>4} {ks:>8} {th2.mean():>11.4f} {approx:>10.4f} {approx/th2.mean():>7.2f} {A(d, ks):>8.4f}")

# ---- 4. what a FIXED kappa=10 means for the observation model ----------------------------
print("[4] fixed likelihood kappa=10: expected cos(z, mu) = A_d(10); kappa implied by E[cos]=0.9")
for d in (16, 64, 512):
    kk = mp.findroot(lambda x: mp.besseli(d / 2, x) / mp.besseli(d / 2 - 1, x) - 0.9, d * 5)
    print(f"    d={d:>3}: A_d(10) = {A(d, 10):.4f}   kappa with A_d=0.9: {float(kk):9.1f}"
          f"   Banerjee approx r(d-r^2)/(1-r^2) at r=0.9: {0.9*(d-0.81)/0.19:9.1f}")
