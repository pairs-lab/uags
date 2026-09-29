# When is a reset Gaussian re-admitted? (views counted from the reset view on?)
import sys; sys.path.insert(0, "/home/thanhnc/gauss_ws/src/uags")
import torch
import torch.nn.functional as Fn
from usplat.vmf_field import DirectionalSemanticField, FieldConfig, ViewStats
torch.manual_seed(0)
N, d = 400, 8
f = DirectionalSemanticField(N, d, FieldConfig(), dtype=torch.float64)
a = Fn.normalize(torch.randn(N, d, dtype=torch.float64), dim=-1)
b = Fn.normalize(torch.randn(N, d, dtype=torch.float64), dim=-1)
ref = 1.0
def view(dirs, noise=0.3):
    u = Fn.normalize(dirs + noise*torch.randn(N, d, dtype=torch.float64), dim=-1)
    m = torch.full((N,), 2.0, dtype=torch.float64)          # full views (w = 1)
    gd = torch.full((N,), 2.0, dtype=torch.float64)        # Gaussian centre depth 2 m
    s = ViewStats(u*m[:, None], m, None, None, gd*m, m, ref)  # observed depth = centre depth
    f._ingest(s, gd)
for _ in range(40):
    view(a)
f.calibrate_change_threshold()
log = []
for t in range(8):
    view(b)                                               # content changed, depth unchanged
    log.append((t, int(f.changed.sum()), int(f.n_resets.sum()), float(f.views_since_reset.mean())))
for r in log: print("update view %d: flagged=%d resets=%d mean views_since_reset=%.2f" % r)
