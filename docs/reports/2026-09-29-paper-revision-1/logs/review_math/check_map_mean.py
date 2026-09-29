import sys; sys.path.insert(0, "/home/thanhnc/gauss_ws/src/uags")
import numpy as np, torch
from scipy.special import ive
from scipy.optimize import minimize_scalar
from usplat.vmf_math import a_d, a_d_inv, a_d_scalar
from vmf_sampler import sample_vmf
rng = np.random.default_rng(1)
d = 64; nu = d/2 - 1
def logC(k):  # log C_d(kappa) = nu log k - (d/2) log 2pi - log I_nu(k)
    return nu*np.log(k) - (d/2)*np.log(2*np.pi) - (np.log(ive(nu, k)) + k)
print("(1) MAP kappa of the joint prior (m0 = 0, c0 = 1)")
for W1, Rn in ((1, 0.9), (3, 2.5), (10, 3.0), (100, 90.0), (5, 0.2)):
    c0 = 1.0
    obj = lambda lk: -((W1+c0)*logC(np.exp(lk)) + np.exp(lk)*Rn)
    res = minimize_scalar(obj, bounds=(-10, 15), method="bounded", options=dict(xatol=1e-10))
    k_num = np.exp(res.x)
    k_code = float(a_d_inv(d, torch.tensor([Rn/(W1+c0)], dtype=torch.float64))[0])
    print(f"  W1={W1:5.1f} ||R||={Rn:5.2f}: numeric argmax {k_num:10.4f}  code a_d_inv {k_code:10.4f}  rel.diff {abs(k_num-k_code)/k_num:.2e}")
print("(2) E[mu] = A_d(kappa) mu* (vMF sample mean)")
mu = np.zeros(d); mu[0] = 1
for k in (5.0, 42.0, 300.0):
    X = sample_vmf(mu, k, 200000, rng)
    print(f"  kappa={k:6.1f}: sample ||mean||={np.linalg.norm(X.mean(0)):.4f}, mean.mu={X.mean(0)@mu:.4f}, A_d={a_d_scalar(d,k):.4f}, code a_d={float(a_d(d, torch.tensor([k],dtype=torch.float64))[0]):.4f}")
