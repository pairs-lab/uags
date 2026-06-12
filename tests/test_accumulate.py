"""Verify: gradient-trick accumulation == explicit weighted-sum accumulation."""
import torch
from usplat.mock_rasterizer import MockFeatureRasterizer
from usplat.bki_accumulate import bki_grad_accumulate, bki_bruteforce_accumulate


def test_grad_equals_bruteforce():
    torch.manual_seed(0)
    N, d, P = 12, 16, 200
    means2d = torch.rand(N, 2) * 32
    sigmas = 1.0 + torch.rand(N) * 3.0
    opacities = 0.2 + 0.7 * torch.rand(N)
    depths = torch.rand(N)
    pixels = torch.rand(P, 2) * 32
    z_obs = torch.nn.functional.normalize(torch.randn(P, d), dim=-1)

    ras = MockFeatureRasterizer(means2d, sigmas, opacities, depths, pixels)

    R_g, n_g = bki_grad_accumulate(ras, z_obs, num_gaussians=N)
    R_b, n_b = bki_bruteforce_accumulate(ras.weights(), z_obs)

    assert torch.allclose(R_g, R_b, atol=1e-5), (R_g - R_b).abs().max()
    assert torch.allclose(n_g, n_b, atol=1e-5), (n_g - n_b).abs().max()
    print(f"R   max abs diff: {(R_g - R_b).abs().max().item():.2e}")
    print(f"n_eff max abs diff: {(n_g - n_b).abs().max().item():.2e}")
    print("PASS: gradient-trick accumulation is exact.")


if __name__ == "__main__":
    test_grad_equals_bruteforce()
