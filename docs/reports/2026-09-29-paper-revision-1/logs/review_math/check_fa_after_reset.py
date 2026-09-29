# Static content after a reset: is the per-view false-alarm rate near 1 - q = 1e-3?
# Map 40 static views, calibrate, then keep showing the SAME static content (no change),
# but force a reset of every Gaussian first, so all histories restart (as after a real change).
import sys; sys.path.insert(0, "/home/thanhnc/gauss_ws/src/uags")
import torch
import torch.nn.functional as Fn
from usplat.vmf_field import DirectionalSemanticField, FieldConfig, ViewStats
torch.manual_seed(0)
N, d = 20000, 8
f = DirectionalSemanticField(N, d, FieldConfig(), dtype=torch.float64)
a = Fn.normalize(torch.randn(N, d, dtype=torch.float64), dim=-1)
def view(noise=0.3):
    u = Fn.normalize(a + noise*torch.randn(N, d, dtype=torch.float64), dim=-1)
    m = torch.full((N,), 2.0, dtype=torch.float64)
    f._ingest(ViewStats(u*m[:, None], m, None, None, None, None, 1.0), None)
for _ in range(40):
    view()
counts = f.c_hist.sum(1).tolist()
f.calibrate_change_threshold()
print("null samples per history bucket:", counts)
print("thresholds per bucket:", [round(x, 4) for x in f.thr_by_hist.tolist()])
# phase A: continue static (long histories)
tests = fires = 0
for _ in range(20):
    before = f.n_resets.sum().item(); ring_ok = torch.isfinite(f.c_ring.min(1).values)
    view()
    fires += f.n_resets.sum().item() - before
    tests += int(torch.isfinite(f.c_last).sum())  # approx: defined tests
print(f"static, long histories: fires/defined-tests over 20 views = {fires}/{20*N} = {fires/(20*N):.2e}")
# phase B: reset everything, continue static: histories restart as after a detected change
f.raw.reset(torch.ones(N, dtype=torch.bool)); f.Rs.zero_(); f.ns.zero_(); f.Rf.zero_(); f.nf.zero_()
f.T2.zero_(); f.TQ.zero_(); f.c_ring.fill_(float('nan'))
per_view = []
for t in range(12):
    before = f.n_resets.sum().item()
    hb = torch.bucketize(f.ns, f._hist_edges)
    view()
    per_view.append(f.n_resets.sum().item() - before)
print("static after a reset: fires per view (N=%d):" % N, per_view)
