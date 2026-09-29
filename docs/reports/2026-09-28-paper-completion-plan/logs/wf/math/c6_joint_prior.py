"""C6: joint conjugate prior on (mu, kappa): p ∝ C_d(kappa)^c0 exp(kappa m0^T mu).
Posterior after (R, n): C_d(kappa)^(c0+n) exp(kappa (m0+R)^T mu).  Joint MAP: A_d(kappa_hat) = ||m0+R||/(c0+n).
Check d/dk log C_d(k) = -A_d(k) numerically, and the MAP by brute-force 1-D search."""
import mpmath as mp
d = 64
logC = lambda k: (d/2-1)*mp.log(k) - (d/2)*mp.log(2*mp.pi) - mp.log(mp.besseli(d/2-1, k))
A = lambda k: mp.besseli(d/2, k)/mp.besseli(d/2-1, k)
for k in (5, 50, 500):
    print(f"k={k}: dlogC/dk={float(mp.diff(logC, k)):.6f}  -A_d={-float(A(k)):.6f}")
c0 = 2.0
for n, Rn in ((1.0, 1.0), (10.0, 9.0), (100.0, 90.0)):          # one obs r=1; r=0.9 cases
    obj = lambda k: (c0+n)*logC(k) + k*Rn                          # mu at its optimum
    ks = [mp.mpf(x)/10 for x in range(1, 40000)]
    kbest = max(ks, key=obj)
    print(f"n={n} ||R||={Rn} (r_bar={Rn/n:.2f}): brute-force kappa_MAP={float(kbest):.1f}  "
          f"A_d(kappa_MAP)={float(A(kbest)):.4f}  formula ||R||/(n+c0)={Rn/(n+c0):.4f}")
