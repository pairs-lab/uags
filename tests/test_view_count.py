"""F1: evidence is counted in views, not pixels.

 * The same view seen k times gives k (Kish) views at ANY resolution; the old pixel
   count grows with the square of the resolution.
 * After ONE view no Gaussian has a defined aleatoric value or belongs to the robust set
   (the old pixel count made every well-covered Gaussian "stable" after one frame).
 * The multi-group accumulation equals the brute-force weighted sums, masks included.
"""
import torch
from usplat.vmf_field import DirectionalSemanticField, FieldConfig
from usplat.mock_rasterizer import MockFeatureRasterizer
from usplat.bki_accumulate import bki_grad_accumulate_groups


def _scene(scale, N=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    means2d = torch.rand(N, 2, generator=g) * 32 * scale
    sigmas = (1.5 + 2.0 * torch.rand(N, generator=g)) * scale
    opac = 0.5 + 0.4 * torch.rand(N, generator=g)
    depths = torch.rand(N, generator=g)
    H = W = int(32 * scale)
    grid = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy"), -1)
    pixels = grid.reshape(-1, 2).float() + 0.5
    return MockFeatureRasterizer(means2d, sigmas, opac, depths, pixels), H * W


def test_views_are_resolution_invariant():
    d, k = 8, 5
    z0 = torch.nn.functional.normalize(torch.randn(d), dim=0)
    out = {}
    for scale in (1, 3):
        ras, P = _scene(scale)
        for mode in ("views", "pixels"):
            f = DirectionalSemanticField(ras.N, d, FieldConfig(view_mass_frac=1e-3, evidence=mode))
            for _ in range(k):
                f.update_view(ras, z0.expand(P, d).contiguous(), num_pixels=P)
            out[scale, mode] = (f.kish_views(), f.n_eff.clone(), f.pixel_mass.clone())
    full = ((out[1, "views"][2] >= k * 1e-3 * 1024) &
            (out[3, "views"][2] >= k * 1e-3 * 9216))          # a full view every time
    assert full.sum() >= 10, int(full.sum())
    for scale in (1, 3):                                      # F1: W1 counts views
        w1 = out[scale, "views"][1][full]
        assert torch.allclose(w1, torch.full_like(w1, float(k)), atol=1e-4), w1[:3]
    ratio_px = (out[3, "pixels"][1][full] / out[1, "pixels"][1][full]).median().item()
    assert 7.0 < ratio_px < 11.0, ratio_px                    # legacy: grows ~ scale^2
    assert torch.allclose(out[3, "views"][0][full], out[1, "views"][0][full], atol=1e-4)
    print(f"PASS views: W1 = {k} views at both resolutions on {int(full.sum())} Gaussians; "
          f"pixel evidence grew {ratio_px:.1f}x")


def test_partial_view_weight():
    """w = min(1, m/ref) exactly; pixel mode keeps the raw mass."""
    d = 4
    for mode, expect in (("views", [0.5, 1.0]), ("pixels", [5.0, 20.0])):
        f = DirectionalSemanticField(2, d, FieldConfig(view_mass_frac=0.1, evidence=mode,
                                                       centre=False))
        f.update_sparse(torch.tensor([0, 1]), torch.tensor([5.0, 20.0]),
                        torch.randn(2, d), num_pixels=100)      # ref = 10
        assert torch.allclose(f.raw.W1, torch.tensor(expect)), (mode, f.raw.W1)
    print("PASS partial view: w = m/ref (0.5) and clamps at 1; pixel mode keeps the mass")


def test_one_view_is_not_stable():
    d = 8
    ras, P = _scene(2)
    z0 = torch.nn.functional.normalize(torch.randn(d), dim=0)

    def one_view(cfg):
        f = DirectionalSemanticField(ras.N, d, cfg)
        f.update_view(ras, z0.expand(P, d).contiguous(), num_pixels=P)
        return f
    new = one_view(FieldConfig())
    ale, valid = new.aleatoric()
    assert not valid.any() and not new.robust_mask().any()
    old = one_view(FieldConfig.legacy(r_stable=0.85))
    frac_old = old.robust_mask().float().mean().item()
    assert frac_old > 0.5, frac_old
    # F1 alone (views, old estimator): the view count gates it
    f1 = one_view(FieldConfig.legacy(evidence="views", epistemic="views"))
    assert not f1.aleatoric()[1].any()
    # F2 alone (pixel evidence, Kish estimator): one view has no dispersion estimate
    f2 = one_view(FieldConfig.legacy(estimator="kish"))
    assert not f2.aleatoric()[1].any()
    print(f"PASS one view: 0% defined with the fixes (and with F1 or F2 alone); legacy "
          f"calls {frac_old * 100:.0f}% stable")


def test_groups_equal_bruteforce():
    torch.manual_seed(0)
    ras, P = _scene(1)
    d = 6
    z = torch.randn(P, d)
    a1 = (torch.rand(P) > 0.3).float()
    y2 = torch.randn(P, 1)
    out = bki_grad_accumulate_groups(ras, [(z, None), (z * 2, a1), (y2, None), (None, a1)], ras.N)
    W = ras.weights()
    ref = [(W.t() @ z, W.sum(0)), (W.t() @ (2 * z * a1[:, None]), W.t() @ a1),
           (W.t() @ y2, W.sum(0)), (None, W.t() @ a1)]
    for (S, m), (Sr, mr) in zip(out, ref):
        if Sr is not None:
            assert torch.allclose(S, Sr, atol=1e-4), (S - Sr).abs().max()
        assert torch.allclose(m, mr, atol=1e-4), (m - mr).abs().max()
    print("PASS groups: multi-target accumulation equals brute force (masks included)")


if __name__ == "__main__":
    test_views_are_resolution_invariant()
    test_partial_view_weight()
    test_one_view_is_not_stable()
    test_groups_equal_bruteforce()
