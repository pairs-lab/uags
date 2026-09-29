"""C2: the paper's small-sample shrinkage  r~ = r - (1-r)/(n+1)  vs the standard bias
correction of the mean resultant length, incl. a weighted (Kish) version."""
import numpy as np
from scipy.stats import vonmises_fisher
import mpmath as mp

rng = np.random.default_rng(1)


def paper(r, n):             # main.tex eq. for r~ ; usplat/vmf_field.py r_bar_corrected
    return np.clip(r - (1 - r) / (n + 1), 0, 1)


def unbiased_rho(R, w):
    """rho^2 unbiased for iid unit vectors with weights w:
       E||sum w z||^2 = sum w^2 + ((sum w)^2 - sum w^2) rho^2.
       With w=1: rho^2 = (n r^2 - 1)/(n - 1); in Kish form (nK r^2 - 1)/(nK - 1)."""
    s1, s2 = w.sum(), (w ** 2).sum()
    rho2 = (np.dot(R, R) - s2) / (s1 ** 2 - s2)
    return np.sqrt(max(rho2, 0.0))


# ---- (a) degenerate cases -----------------------------------------------------------------
print("[a] one observation (n=1, r=1): paper r~ =", paper(1.0, 1.0),
      "| n identical obs (r=1), n=2,5,50:", [paper(1.0, n) for n in (2, 5, 50)])
print("    => 1 - r~ = (1-r)(n+2)/(n+1): multiplicative in (1-r), so r=1 stays r~=1 for every n")
print("    standard unbiased rho^2 at n=1: (1*1-1)/(1-1) = 0/0 -> undefined (correctly 'no information')")

# ---- (b) iid vMF samples, true population rho = A_d(kappa) -----------------------------------
d = 64
print(f"[b] d={d}: mean over 4000 trials of r_bar, paper r~, unbiased rho_hat (true rho in header)")
for kappa in (0.0, 30.0, 300.0):
    rho = 0.0 if kappa == 0 else float(mp.besseli(d / 2, kappa) / mp.besseli(d / 2 - 1, kappa))
    mu = np.zeros(d); mu[0] = 1
    print(f"    kappa={kappa:>5} true rho={rho:.3f}")
    for n in (2, 3, 5, 10, 50):
        rb, rt, ru = [], [], []
        for _ in range(4000):
            if kappa == 0:
                z = rng.normal(size=(n, d)); z /= np.linalg.norm(z, axis=1, keepdims=True)
            else:
                z = vonmises_fisher(mu, kappa).rvs(n, random_state=rng).reshape(n, d)
            R = z.sum(0); r = np.linalg.norm(R) / n
            rb.append(r); rt.append(paper(r, n)); ru.append(unbiased_rho(R, np.ones(n)))
        print(f"      n={n:>3}: r_bar={np.mean(rb):.3f}  paper r~={np.mean(rt):.3f}  "
              f"unbiased rho_hat={np.mean(ru):.3f}")

# ---- (c) weighted observations: what n to plug in -------------------------------------------
print("[c] weighted obs (w ~ splat weights, many tiny), d=64, true rho=0 (pure conflict)")
for P in (20, 200):
    rt_sum, ru_sum, nk_sum, rb_sum, ns = 0, 0, 0, 0, 0
    for _ in range(3000):
        w = rng.beta(0.3, 2.0, size=P)            # skewed weights like T*alpha
        z = rng.normal(size=(P, d)); z /= np.linalg.norm(z, axis=1, keepdims=True)
        R = (w[:, None] * z).sum(0); n = w.sum(); r = np.linalg.norm(R) / n
        nk = n ** 2 / (w ** 2).sum()
        rb_sum += r; rt_sum += paper(r, n); ru_sum += unbiased_rho(R, w); nk_sum += nk; ns += n
    T = 3000
    print(f"    P={P}: mean n_eff=sum w={ns/T:.1f}  Kish n={nk_sum/T:.1f}  r_bar={rb_sum/T:.3f}  "
          f"paper r~={rt_sum/T:.3f}  weighted-unbiased rho_hat={ru_sum/T:.3f}  (truth 0)")
