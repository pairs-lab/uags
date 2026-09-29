# Is the joint-prior MAP (plans A,B F2) a new estimator, or NormLift's reliability?
# joint prior (m0=0):  A_d(kappa_hat) = ||R|| / (n + c0) = rbar * n/(n+c0)
# NormLift:            Rel = ||f|| * N/(N+beta)        = rbar * N/(N+beta)
# Kish (plan C F2):    rho^2 = (n rbar^2 - 1)/(n - 1)
import numpy as np
from scipy.stats import spearmanr
rng = np.random.default_rng(0)
n = rng.integers(1, 21, 20000).astype(float) + rng.random(20000)*0.0   # views 1..20
rbar = rng.uniform(0.3, 1.0, 20000)
jp = rbar*n/(n+1.0)
nl = rbar*n/(n+1.0)
m = n > 1
kish = np.sqrt(np.clip((n*rbar**2-1)/np.maximum(n-1,1e-9), 0, None))
print("max|jointprior(c0=1) - NormLift(beta=1)| =", np.abs(jp-nl).max())
print("Spearman(jointprior, NormLift) =", spearmanr(jp, nl).correlation)
print("Spearman(Kish rho_hat, NormLift) on n>1 =", round(spearmanr(kish[m], nl[m]).correlation,4))
