# calibrate_change_threshold: per-history-bucket q-quantile, nearest bucket if < 200 samples.
import sys; sys.path.insert(0, "/home/thanhnc/gauss_ws/src/uags")
import torch, numpy as np
from usplat.vmf_field import DirectionalSemanticField, FieldConfig
f = DirectionalSemanticField(4, 4, FieldConfig(), dtype=torch.float64)
rng = np.random.default_rng(0)
samples = {0: rng.beta(2, 30, 5000), 1: rng.beta(2, 40, 150), 2: rng.beta(2, 60, 8000),
           3: rng.beta(2, 80, 8000), 4: rng.beta(2, 90, 50), 5: rng.beta(2, 100, 3000)}
for b, v in samples.items():
    bins = np.minimum((v * 2000).astype(int), 3999)
    np.add.at(f.c_hist[b].numpy(), bins, 1.0)
thr = f.calibrate_change_threshold()
q = 0.999
exp = {b: np.quantile(v, q) for b, v in samples.items()}
for b in range(6):
    print(b, len(samples[b]), "code thr", round(float(f.thr_by_hist[b]), 5), " own-bucket np.quantile", round(exp[b], 5))
print("pooled thr returned:", round(thr, 5), " np pooled:", round(np.quantile(np.concatenate(list(samples.values())), q), 5))
print("edges", f.HIST_EDGES, "bucketize(ns) of 4.0, 4.5, 40.0, 41:",
      torch.bucketize(torch.tensor([4.0, 4.5, 40.0, 41.0], dtype=torch.float64), f._hist_edges).tolist())
