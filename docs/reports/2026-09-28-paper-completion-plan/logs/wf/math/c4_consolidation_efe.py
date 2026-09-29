"""C4: consolidation decay semantics and its interplay with the EFE gate, using the
project's own classes (imported read-only; PYTHONDONTWRITEBYTECODE=1)."""
import sys
import torch, torch.nn.functional as Fn
sys.path.insert(0, "/home/thanhnc/gauss_ws/src/uags")
from usplat.vmf_field import DirectionalSemanticField   # noqa: E402
from usplat.consolidation import Consolidator          # noqa: E402
from usplat.efe_planner import EFEPlanner              # noqa: E402

torch.manual_seed(0)
d = 64
A = Fn.normalize(torch.randn(d), dim=0)
B = Fn.normalize(torch.randn(d), dim=0)
noise = lambda v, s: Fn.normalize(v + s * torch.randn(d) / d ** 0.5, dim=0)

# ---- (a) decay leaves r_bar unchanged ------------------------------------------------
f = DirectionalSemanticField(1, d)
for t in range(10):
    f.update_sparse(torch.tensor([0]), torch.tensor([3.0]), (A if t % 2 else B)[None])
r0 = f.mean_resultant_length().item()
c = Consolidator(f, min_obs=5.0, r_unstable=0.9, r_stable=0.95, forget_lambda=0.9)
m = c.step()
print(f"[a] before decay r_bar={r0:.4f} n={30.0:.1f} | after decay r_bar={f.mean_resultant_length().item():.4f} "
      f"n={f.n_eff.item():.1f} unstable={bool(m['unstable'][0])}  (lambda scales R and n alike)")

# ---- (b) moved object: A for 50 frames, then B (e.g. wall behind) for 50 frames ------
# default run_mapping.py settings: consolidate every 10 frames, lambda=0.9,
# r_unstable=0.55, r_stable=0.85, min_obs=5; per-frame mass m=20 pixel-equivalents.
f = DirectionalSemanticField(1, d)
c = Consolidator(f, min_obs=5.0, r_unstable=0.55, r_stable=0.85, forget_lambda=0.9)
log = []
for t in range(200):
    z = noise(A if t < 50 else B, 0.3)
    f.update_sparse(torch.tensor([0]), torch.tensor([20.0]), z[None])
    if (t + 1) % 10 == 0:
        ms = c.step()
        mu = f.mu_post()[0]
        log.append((t + 1, f.mean_resultant_length().item(), f.n_eff.item(),
                    bool(ms["unstable"][0]), bool(c.robust_mask()[0]), (mu @ A).item(), (mu @ B).item()))
print("[b] moved object (A until t=50, then B); every 10th frame:")
print("     t   r_bar   n_eff  unstable robust cos(mu,A) cos(mu,B)")
for row in log[:12] + log[-2:]:
    print("    {:>3} {:6.3f} {:8.1f} {!s:>8} {!s:>6} {:9.3f} {:9.3f}".format(*row))

# ---- (c) forgetting with NO new observations: EFE gate re-opens -----------------------
f = DirectionalSemanticField(2, d)
for t in range(20):                  # Gaussian 0: dynamic (alternating), Gaussian 1: static
    f.update_sparse(torch.tensor([0, 1]), torch.tensor([2.0, 2.0]),
                    torch.stack([noise(A if t % 2 else B, 0.2), noise(A, 0.2)]))
c = Consolidator(f, min_obs=5.0, r_unstable=0.8, r_stable=0.85, forget_lambda=0.9)
pl = EFEPlanner(f, lam=0.0, aleatoric_min_obs=5.0)
vis = torch.eye(2)                   # view 0 sees only the dynamic G, view 1 only the static G
print("[c] dynamic G0 vs static G1, repeated consolidation steps without new data:")
for s in range(0, 40):
    if s in (0, 1, 30, 31, 32, 39):
        efe, best = pl.score(vis)
        print(f"    step {s:>2}: n_eff={[round(x,2) for x in f.n_eff.tolist()]} r_bar={[round(x,3) for x in f.mean_resultant_length().tolist()]} "
              f"epistemic value (view0, view1)={[round(-x,3) for x in efe.tolist()]} best view={best}")
    c.step()

# ---- (d) steady-state mass of an unstable Gaussian under continued observation -------
lam, m_int = 0.9, 200.0              # mass gained per consolidation interval
n = 0.0
for _ in range(200):
    n = lam * (n + m_int)
print(f"[d] steady state n* = lambda*m/(1-lambda) = {lam*m_int/(1-lam):.1f}; iterated: {n:.1f}")
