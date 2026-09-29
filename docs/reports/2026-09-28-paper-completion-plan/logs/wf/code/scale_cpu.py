"""CPU proxy for the memory/time of the gradient-trick accumulation (not gsplat).

The rasterizer is replaced by a sparse (P, N) blending-weight matrix with K
contributors per pixel (torch.sparse.mm is linear + differentiable in the
buffer, like alpha compositing). Everything else is the real uags code path:
DirectionalSemanticField.update_from_rasterizer -> bki_grad_accumulate.
Usage: python scale_cpu.py N d H W K
"""
import sys, time, resource, gc
import torch
sys.path.insert(0, "/home/thanhnc/gauss_ws/src/uags")
from usplat.vmf_field import DirectionalSemanticField

torch.set_num_threads(8)
N, d, H, W, K = map(int, sys.argv[1:6])
P = H * W

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2   # KB -> GB

g = torch.Generator().manual_seed(0)
rows = torch.arange(P).repeat_interleave(K)
cols = torch.randint(0, N, (P * K,), generator=g)
vals = torch.rand(P * K, generator=g) / K
Wm = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (P, N)).coalesce()
z = torch.nn.functional.normalize(torch.randn(P, d, generator=g), dim=-1)
field = DirectionalSemanticField(N, d)
gc.collect()
base = rss_gb()
ras = lambda buf: torch.sparse.mm(Wm, buf)
ts = []
for _ in range(3):
    t0 = time.time()
    field.update_from_rasterizer(ras, z)
    ts.append(time.time() - t0)
peak = rss_gb()
state = (field.R.numel() + field.n_eff.numel()) * 4 / 1e9
print(f"N={N} d={d} P={H}x{W} K={K}: accumulate time/frame (CPU) "
      f"{min(ts):.2f}s | peak RSS {peak:.2f} GB (before first update {base:.2f} GB, "
      f"delta {peak - base:.2f} GB) | field state R+n_eff {state:.2f} GB | "
      f"N*(d+1)*4 = {N * (d + 1) * 4 / 1e9:.2f} GB")
