"""state_dict / from_state round trip (text prior, dtype, change flags, centre) and the
inference_mode guard of the gradient-trick update."""
import pytest
import torch
import torch.nn.functional as Fn
from usplat.vmf_field import DirectionalSemanticField, FieldConfig
from usplat.mock_rasterizer import MockFeatureRasterizer


def test_roundtrip_with_prior_and_flags():
    torch.manual_seed(0)
    N, d = 40, 8
    prior = Fn.normalize(torch.randn(N, d), dim=-1)
    f = DirectionalSemanticField(N, d, FieldConfig(prior_kappa=5.0, fast_memory=0.5,
                                                   change_thr=0.5, readmit_depth=None,
                                                   view_mass_frac=0.01),
                                 prior_mu=prior, dtype=torch.float64)
    f.set_centre(torch.zeros(d, dtype=torch.float64))
    A = Fn.normalize(torch.randn(N, d), dim=-1)
    B = Fn.normalize(torch.randn(N, d), dim=-1)
    for v in range(10):
        f.update_sparse(torch.arange(N), torch.ones(N), A if v < 6 else B, num_pixels=100)
    assert f.n_resets.sum() > 0
    g = DirectionalSemanticField.from_state(f.state_dict())
    assert g.dtype == torch.float64 and g.prior_mu is not None
    for name in ("mu_post", "kappa_post", "epistemic", "agreement", "robust_mask"):
        a, b = getattr(f, name)(), getattr(g, name)()
        assert torch.allclose(a.double(), b.double(), equal_nan=True), name
    assert torch.equal(f.changed, g.changed) and torch.equal(f.n_resets, g.n_resets)
    print("PASS round trip: prior, dtype, posterior, uncertainty, flags preserved")


def test_inference_mode_is_rejected():
    N, d = 5, 4
    ras = MockFeatureRasterizer(torch.rand(N, 2) * 8, torch.ones(N), 0.5 * torch.ones(N),
                                torch.rand(N), torch.rand(20, 2) * 8)
    f = DirectionalSemanticField(N, d, FieldConfig(centre=False))
    z = Fn.normalize(torch.randn(20, d), dim=-1)
    with torch.no_grad():                                     # fine
        f.update_view(ras, z)
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="inference_mode"):
            f.update_view(ras, z)
    print("PASS update_view works under no_grad and gives a clear error under inference_mode")


if __name__ == "__main__":
    test_roundtrip_with_prior_and_flags()
    test_inference_mode_is_rejected()
