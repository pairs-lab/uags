"""gs_io: write a synthetic INRIA-style ply / gsplat .pt, load, check activations."""
import os
import tempfile
import numpy as np
import torch

from usplat.gs_io import load_gaussians, SH_C0


def _write_inria_ply(path, N=50):
    from plyfile import PlyData, PlyElement
    rng = np.random.default_rng(0)
    names = (["x", "y", "z", "nx", "ny", "nz"]
             + [f"f_dc_{i}" for i in range(3)]
             + ["opacity"]
             + [f"scale_{i}" for i in range(3)]
             + [f"rot_{i}" for i in range(4)])
    data = np.zeros(N, dtype=[(n, "f4") for n in names])
    for n in names:
        data[n] = rng.normal(size=N).astype(np.float32)
    el = PlyElement.describe(data, "vertex")
    PlyData([el]).write(path)
    return data


def test_ply_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "point_cloud.ply")
        raw = _write_inria_ply(p)
        g = load_gaussians(p)
        N = raw.shape[0]
        assert g["means"].shape == (N, 3)
        assert g["quats"].shape == (N, 4)
        assert torch.allclose(g["quats"].norm(dim=-1),
                              torch.ones(N), atol=1e-5)
        assert (g["scales"] > 0).all()                      # exp applied
        assert ((g["opacities"] > 0) & (g["opacities"] < 1)).all()  # sigmoid
        # colors = SH dc -> rgb
        expect = np.clip(SH_C0 * raw["f_dc_0"] + 0.5, 0, 1)
        assert torch.allclose(g["colors"][:, 0],
                              torch.from_numpy(expect), atol=1e-5)
    print("PASS: ply load + activations")


def test_pt_roundtrip():
    N = 30
    torch.manual_seed(0)
    ckpt = dict(means=torch.randn(N, 3), quats=torch.randn(N, 4),
                scales=torch.randn(N, 3),               # log-scale (negative vals)
                opacities=torch.randn(N),               # logits
                sh0=torch.randn(N, 1, 3))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ckpt.pt")
        torch.save(dict(splats=ckpt), p)
        g = load_gaussians(p)
        assert (g["scales"] > 0).all()
        assert ((g["opacities"] > 0) & (g["opacities"] < 1)).all()
        assert g["colors"].shape == (N, 3)
    print("PASS: pt load + auto-activation")


if __name__ == "__main__":
    test_ply_roundtrip()
    test_pt_roundtrip()
