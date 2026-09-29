"""Semantic-NeRF Replica loader: layout, label ids, and geometry (intrinsics + OpenCV c2w)
checked by depth reprojection between two nearby frames. Skipped when the data is absent."""
import os
import pytest
import torch
from usplat.datasets import ReplicaSemNeRFDataset

ROOT = "/home/thanhnc/gauss_ws/data/replica_semnerf/Replica_Dataset"
SEQ = os.path.join(ROOT, "office_0", "Sequence_1")


@pytest.mark.skipif(not os.path.isdir(SEQ), reason="Replica (Semantic-NeRF) not downloaded")
def test_replica_semnerf_geometry_and_labels():
    ds = ReplicaSemNeRFDataset(SEQ)
    assert len(ds) == 900
    a, b = ds[0], ds[3]
    assert a["image"].shape == (480, 640, 3) and a["labels"].shape == (480, 640)
    names = ReplicaSemNeRFDataset.classes(ROOT, "office_0")
    ids = set(torch.unique(a["labels"]).tolist()) - {0}
    assert ids <= set(names), ids - set(names)
    # back-project frame a, project into frame b, compare with b's depth
    K = a["K"]
    H, W = a["depth"].shape
    v, u = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    z = a["depth"]
    ok = z > 0.1
    x = torch.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, torch.ones_like(z)], -1)[ok]
    Xw = (a["c2w"] @ x.T)
    Xb = torch.linalg.inv(b["c2w"]) @ Xw
    zb = Xb[2]
    ub = K[0, 0] * Xb[0] / zb + K[0, 2]
    vb = K[1, 1] * Xb[1] / zb + K[1, 2]
    inb = (zb > 0.1) & (ub >= 0) & (ub <= W - 1) & (vb >= 0) & (vb <= H - 1)
    db = b["depth"][vb[inb].round().long(), ub[inb].round().long()]
    rel = ((db - zb[inb]).abs() / zb[inb])
    frac = (rel < 0.02).float().mean().item()
    assert inb.float().mean() > 0.5 and frac > 0.9, (inb.float().mean(), frac)
    print(f"PASS Replica loader: 900 frames, {len(ids)} classes in frame 0, depth reprojection "
          f"agrees (<2%) on {frac * 100:.1f}% of co-visible pixels")


if __name__ == "__main__":
    test_replica_semnerf_geometry_and_labels()
