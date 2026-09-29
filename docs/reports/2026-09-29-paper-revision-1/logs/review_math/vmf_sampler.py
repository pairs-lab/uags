import numpy as np
def sample_vmf(mu, kappa, n, rng):
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
