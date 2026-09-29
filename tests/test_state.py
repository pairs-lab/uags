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


def test_old_pooled_name_loads_as_history():
    """Field states saved before the calib_mode rename called the history-conditioned rule
    "pooled"; from_state maps them to "history". New states keep "pooled" as the flat mode."""
    from usplat.vmf_field import DirectionalSemanticField, FieldConfig
    f = DirectionalSemanticField(10, 4, FieldConfig(calib_mode="pooled"))
    s = f.state_dict()
    assert DirectionalSemanticField.from_state(s).cfg.calib_mode == "pooled"
    old = dict(s)
    old.pop("calib_names")
    assert DirectionalSemanticField.from_state(old).cfg.calib_mode == "history"
    print("PASS calib_mode rename: old 'pooled' -> 'history', new 'pooled' kept")


def test_zero_tensors_saved_as_none_roundtrip():
    """All-zero statistics (centred channel with centre=False, buckets with change_test=False)
    are saved as None and come back as zeros; every read-out is unchanged."""
    import torch.nn.functional as Fn
    from usplat.vmf_field import DirectionalSemanticField, FieldConfig
    torch.manual_seed(0)
    N, d = 50, 8
    f = DirectionalSemanticField(N, d, FieldConfig(change_test=False))
    for _ in range(4):
        f.update_sparse(torch.arange(N), torch.ones(N), Fn.normalize(torch.randn(N, d), dim=-1),
                        num_pixels=100)
    s = f.state_dict()
    assert s["cen_R"] is None and s["Rf"] is None and s["raw_R"] is not None
    g = DirectionalSemanticField.from_state(s)
    for a, b in ((f.raw.R, g.raw.R), (f.cen.R, g.cen.R), (f.Rf, g.Rf), (f.c_ring, g.c_ring)):
        assert torch.equal(a.nan_to_num(-7), b.nan_to_num(-7))
    assert torch.allclose(f.aleatoric()[0], g.aleatoric()[0], equal_nan=True)
    assert torch.equal(f.epistemic(), g.epistemic())
    print("PASS zero tensors saved as None, read-outs unchanged after the round trip")
