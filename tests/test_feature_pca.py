"""feature_pca: subspace recovery + cosine preservation on the retained subspace."""
import torch
import torch.nn.functional as Fn

from usplat.feature_pca import StreamingPCA


def test_recovers_subspace_and_preserves_cosine():
    torch.manual_seed(0)
    D, k, M = 128, 16, 4000
    # data concentrated on a k-dim subspace + small noise
    basis = torch.linalg.qr(torch.randn(D, k)).Q
    z = Fn.normalize(torch.randn(M, k) @ basis.t() + 0.01 * torch.randn(M, D), dim=-1)

    pca = StreamingPCA(D, k)
    for chunk in z.split(500):
        pca.accumulate(chunk)
    explained = pca.fit()
    assert explained > 0.95, explained

    p = pca.project(z)
    assert p.shape == (M, k)
    assert torch.allclose(p.norm(dim=-1), torch.ones(M), atol=1e-4)

    # cosine similarities preserved through projection
    idx = torch.randperm(M)[:200]
    cos_orig = z[idx[:100]] @ z[idx[100:]].t()
    cos_proj = p[idx[:100]] @ p[idx[100:]].t()
    err = (cos_orig - cos_proj).abs().max()
    assert err < 0.05, err
    print(f"PASS: PCA explained={explained:.3f}, max cosine distortion={err:.4f}")


if __name__ == "__main__":
    test_recovers_subspace_and_preserves_cosine()
