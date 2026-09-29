"""C5: (a) is the vMF update a 'continuous generalization' of the Dirichlet-categorical BKI
update (does the Dirichlet come out as a special case for one-hot z)?
(b) EFE: what does the paper's own vMF model say the information gain of a view is?"""
import numpy as np
import mpmath as mp

# ---- (a) one-hot observations, K=3 labels -----------------------------------------------------
K, alpha0, kappa = 3, 1.0, 10.0
for counts in ([3, 1, 0], [5, 5, 0], [50, 50, 0]):
    y = np.array(counts, float)                       # sum_p k_{p,n} y_p
    alpha = alpha0 + y
    p_dir = alpha / alpha.sum()                       # Dirichlet posterior predictive
    R = y                                             # vMF resultant: SAME statistic
    mu = R / np.linalg.norm(R); kst = kappa * np.linalg.norm(R)
    print(f"[a] counts={counts}: Dirichlet predictive p={np.round(p_dir,3)}  | vMF mu*={np.round(mu,3)} "
          f"kappa*={kst:.1f} r_bar={np.linalg.norm(R)/y.sum():.3f}")
print("    -> same sufficient statistic and same argmax, but the vMF posterior over a mean direction")
print("       does not return class probabilities: a 50/50 split gives mu* between two labels")
print("       with kappa* growing like n (certain about a direction that is neither label).")


# ---- (b) information gain under the paper's fixed-kappa vMF posterior -------------------------
def H(d, k):   # differential entropy of vMF_d(k) on S^{d-1}
    k = mp.mpf(k)
    logC = (d/2 - 1)*mp.log(k) - (d/2)*mp.log(2*mp.pi) - mp.log(mp.besseli(d/2 - 1, k))
    A = mp.besseli(d/2, k) / mp.besseli(d/2 - 1, k)
    return float(-logC - k*A)


d, kap, m = 64, 10.0, 5.0
print(f"[b] d={d}, kappa={kap}, new view adds mass m={m} with the same agreement r_bar (aligned)")
print("     n_eff  r_bar  paper u_epi*(1-u_ale)   model IG=H_old-H_new [nats]")
for n in (5, 10, 50):
    for r in (0.95, 0.6):
        ko, kn = kap*n*r, kap*(n*r + m*r)
        rt = r - (1 - r)/(n + 1)
        paper = (1/(1 + n)) * rt
        print(f"     {n:>5} {r:>6} {paper:>20.4f} {H(d, ko) - H(d, kn):>22.3f}")
print("    -> under the paper's own model the gain falls with n; its r_bar effect shrinks as n grows")
print("       (kappa* < d at small n). The paper gate keeps a fixed r~ factor at every n.")
