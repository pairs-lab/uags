# Monte Carlo: E[rho_hat], E[bracket], E[rbar], E[||R||/(W1+1)] for vMF view means in d=64.
import sys; sys.path.insert(0, "/home/thanhnc/gauss_ws/src/uags")
import numpy as np
from usplat.vmf_math import a_d_scalar
from scipy.optimize import brentq
rng = np.random.default_rng(0)
d = 64

def sample_vmf(mu, kappa, n):
    # Wood (1994)
    dim = mu.shape[0]
    b = (-2*kappa + np.sqrt(4*kappa**2 + (dim-1)**2))/(dim-1)
    x0 = (1-b)/(1+b); c = kappa*x0 + (dim-1)*np.log(1-x0**2)
    w = np.empty(n); i = 0
    while i < n:
        m = n - i
        z = rng.beta((dim-1)/2, (dim-1)/2, size=m)
        ww = (1-(1+b)*z)/(1-(1-b)*z)
        u = rng.uniform(size=m)
        ok = kappa*ww + (dim-1)*np.log(1-x0*ww) - c >= np.log(u)
        k = ok.sum(); w[i:i+k] = ww[ok]; i += k
    v = rng.standard_normal((n, dim)); v -= (v@mu)[:, None]*mu; v /= np.linalg.norm(v, axis=1, keepdims=True)
    return w[:, None]*mu + np.sqrt(1-w**2)[:, None]*v

mu = np.zeros(d); mu[0] = 1
T = 20000
print("rho  n  | E[bracket unclamped]  E[rho_hat]  E[rbar]  E[||R||/(W1+1)] | rms-formula rbar, shrunk")
for rho in (0.0, 0.1, 0.35, 0.6, 0.9):
    kappa = 0.0 if rho == 0 else brentq(lambda k: a_d_scalar(d, k) - rho, 1e-6, 1e5)
    for n in (2, 3, 5, 12):
        if kappa == 0:
            X = rng.standard_normal((T*n, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            X = sample_vmf(mu, kappa, T*n)
        X = X.reshape(T, n, d)
        R = X.sum(1); W1 = n; W2 = n; Q = n
        Rn2 = (R**2).sum(1)
        br = (Rn2 - Q)/(W1**2 - W2)
        rh = np.sqrt(np.clip(br, 0, 1))
        rbar = np.sqrt(Rn2)/W1
        sh = np.sqrt(Rn2)/(W1+1)
        f_r = np.sqrt(rho**2 + (1-rho**2)/n)
        print(f"{rho:.2f} {n:2d} | {br.mean():.4f}  {rh.mean():.4f}  {rbar.mean():.4f}  {sh.mean():.4f} | {f_r:.4f} {f_r*n/(n+1):.4f}")
