"""F2/F3: agreement estimators, the centred channel, and the vMF special function.

 * identical observations: Kish rho_hat = 1; joint prior = n/(n+c0) (1/(1+c0) at one view);
   the old r_tilde stays 1 at one view (the flaw it was meant to fix);
 * uniformly random directions: Kish rho_hat ~ 0 (r_bar is biased up by ~1/sqrt(n));
 * one view: Kish is undefined (NaN) -> "no information";
 * joint-prior MAP: A_d(kappa_hat) = ||R||/(n+c0) matches a brute-force maximisation of
   the joint posterior over kappa (math audit C6: 0.3330 vs 0.3333);
 * centred channel: constant content has agreement ~1 on the centred channel even when
   it is close to the scene mean direction in the raw space.
"""
import math
import torch
import torch.nn.functional as Fn
from usplat.vmf_field import DirectionalSemanticField, FieldConfig
from usplat.vmf_math import a_d, a_d_inv, a_d_scalar


def _feed(field, obs, P=100):
    """Feed one view per row of obs (V, N, d) via update_sparse (one pixel each)."""
    N = obs.shape[1]
    for v in range(obs.shape[0]):
        field.update_sparse(torch.arange(N), torch.full((N,), 1.0), obs[v], num_pixels=P,
                            normalize_z=True)


def test_identical_and_uniform():
    torch.manual_seed(0)
    N, d, V = 2000, 16, 8
    z0 = Fn.normalize(torch.randn(N, d), dim=-1)
    for est, expect in (("kish", 1.0), ("joint", V / (V + 1.0))):
        f = DirectionalSemanticField(N, d, FieldConfig(estimator=est, centre=False,
                                                       change_test=False, view_mass_frac=0.01))
        _feed(f, z0.expand(V, N, d))
        a = f.agreement("raw")
        assert torch.allclose(a, torch.full_like(a, expect), atol=1e-5), (est, a[:3])
    # uniform directions
    f = DirectionalSemanticField(N, d, FieldConfig(estimator="kish", centre=False,
                                                   change_test=False, view_mass_frac=0.01))
    _feed(f, Fn.normalize(torch.randn(V, N, d), dim=-1))
    rho = f.agreement("raw")
    rbar = f.mean_resultant_length("raw")
    assert rho.mean() < 0.15, rho.mean()
    assert rbar.mean() > 0.3, rbar.mean()
    print(f"PASS identical: kish 1, joint {V}/{V + 1}; uniform: kish mean {rho.mean():.3f} "
          f"vs raw r_bar {rbar.mean():.3f}")


def test_one_view():
    N, d = 50, 8
    z0 = Fn.normalize(torch.randn(N, d), dim=-1)
    out = {}
    for est in ("kish", "joint", "legacy"):
        f = DirectionalSemanticField(N, d, FieldConfig(estimator=est, centre=False,
                                                       change_test=False, view_mass_frac=0.01))
        _feed(f, z0[None])
        out[est] = f.agreement("raw")
    assert torch.isnan(out["kish"]).all()
    assert torch.allclose(out["joint"], torch.full((N,), 0.5), atol=1e-6)
    assert torch.allclose(out["legacy"], torch.ones(N), atol=1e-6)   # the old flaw
    print("PASS one view: kish undefined, joint 1/(1+c0)=0.5, legacy r_tilde stays 1.0")


def test_joint_prior_map_bruteforce():
    """Maximise (n+c0) log C_d(k) + k ||R|| over k on a grid; compare with a_d_inv."""
    from scipy.special import ive
    d, c0 = 8, 1.0
    for n, Rn in ((2.0, 1.0), (4.0, 3.75), (10.0, 9.7)):
        nu = d / 2 - 1
        ks = torch.logspace(-3, 4, 200001, dtype=torch.float64)
        # log C_d(k) = nu log k - log I_nu(k) + const ; log I_nu = log ive + k
        logI = torch.log(torch.as_tensor(ive(nu, ks.numpy()))) + ks
        obj = (n + c0) * (nu * torch.log(ks) - logI) + ks * Rn
        k_bf = ks[obj.argmax()].item()
        k_map = a_d_inv(d, torch.tensor([Rn / (n + c0)], dtype=torch.float64)).item()
        assert abs(k_bf - k_map) / k_map < 1e-3, (n, Rn, k_bf, k_map)
        assert abs(a_d_scalar(d, k_map) - Rn / (n + c0)) < 1e-6
    print("PASS joint prior: A_d(kappa_hat) = ||R||/(n+c0) matches the brute-force MAP")


def test_special_function():
    for d in (3, 64, 512):
        k = torch.logspace(-3, 5, 60, dtype=torch.float64)
        r = a_d(d, k)
        assert (r[1:] >= r[:-1]).all()
        m = (r > 1e-6) & (r < 0.9999)
        assert ((a_d_inv(d, r[m]) - k[m]).abs() / k[m]).max() < 1e-6
    for k in (0.1, 1.0, 10.0, 100.0):
        assert abs(a_d_scalar(3, k) - (1 / math.tanh(k) - 1 / k)) < 1e-9
    assert abs(a_d_inv(64, torch.tensor([0.85], dtype=torch.float64)).item() - 193.47) < 0.05
    print("PASS A_d: monotone, invertible, d=3 closed form, d=64 A=0.85 -> kappa 193.5")


def test_centred_channel_restores_consistency():
    """Raw CLIP-like features: a shared component + small class-specific parts. Constant
    content must look consistent on the centred channel; the centre is the scene mean."""
    torch.manual_seed(0)
    N, d, V = 300, 32, 6
    common = Fn.normalize(torch.randn(d), dim=0)
    cls = Fn.normalize(torch.randn(N, d), dim=-1)
    obs = Fn.normalize(3.0 * common + cls, dim=-1)            # cos(random pair) ~ 0.9
    f = DirectionalSemanticField(N, d, FieldConfig(estimator="kish", change_test=False,
                                                   view_mass_frac=0.01, centre=True))
    f.set_centre(obs.mean(0))
    _feed(f, obs.expand(V, N, d))
    a = f.agreement("centred")
    assert torch.allclose(a, torch.ones_like(a), atol=1e-4), a.min()
    # two different contents mixed half/half: raw agreement stays high, centred drops
    mix = torch.cat([obs.expand(V // 2, N, d), obs[torch.randperm(N)].expand(V // 2, N, d)])
    g = DirectionalSemanticField(N, d, FieldConfig(estimator="kish", change_test=False,
                                                   view_mass_frac=0.01, centre=True))
    g.set_centre(obs.mean(0))
    _feed(g, mix)
    raw, cen = g.agreement("raw").mean().item(), g.agreement("centred").mean().item()
    assert raw > 0.9 and cen < 0.8, (raw, cen)
    print(f"PASS centred: constant content 1.0; mixed content raw {raw:.3f} vs centred {cen:.3f}")


def test_kish_multi_pixel_fractional():
    """Views that average K noisy pixels (||u|| < 1) with fractional view weights: the
    Kish estimate of rho^2 = ||E u||^2 is unbiased (Monte-Carlo reference)."""
    torch.manual_seed(0)
    N, d, V, K, P = 3000, 8, 6, 4, 100
    mu = Fn.normalize(torch.randn(N, d), dim=-1)
    f = DirectionalSemanticField(N, d, FieldConfig(estimator="kish", centre=False,
                                                   change_test=False, view_mass_frac=0.1))
    ref = 0.1 * P
    draw = lambda n: Fn.normalize(mu[:, None] + 0.8 * torch.randn(N, n, d), dim=-1)
    for _ in range(V):
        w_v = 0.2 + 0.8 * torch.rand(N)                        # fractional view weight
        z = draw(K).reshape(N * K, d)
        ids = torch.arange(N).repeat_interleave(K)
        f.update_sparse(ids, (w_v * ref / K).repeat_interleave(K), z, num_pixels=P)
    assert (f.raw.within() < 0.99).all()                       # ||u|| < 1 really occurs
    Rn2 = f.raw.R.pow(2).sum(-1)
    rho2_hat = ((Rn2 - f.raw.Q) / (f.raw.W1 ** 2 - f.raw.W2))  # unclamped estimator
    u_mc = draw(K * 4000 // K).reshape(N, 4000 // K, K, d).mean(2)  # many view means
    rho2_mc = u_mc.mean(1).pow(2).sum(-1)
    rel = abs(rho2_hat.mean().item() - rho2_mc.mean().item()) / rho2_mc.mean().item()
    naive = (f.raw.rbar() ** 2).mean().item()
    assert rel < 0.03, (rho2_hat.mean().item(), rho2_mc.mean().item())
    assert naive > 1.05 * rho2_mc.mean().item()                # r_bar^2 is biased up
    print(f"PASS Kish (K={K} px/view, fractional w): mean rho2_hat {rho2_hat.mean():.4f} vs "
          f"Monte-Carlo {rho2_mc.mean():.4f} (r_bar^2 {naive:.4f})")


def test_a_d_against_mpmath():
    import mpmath as mp
    mp.mp.dps = 50
    for d in (64, 512):
        nu = d / 2 - 1
        for k in (0.5, 5.0, 20.0, 0.9 * nu, 1.1 * nu, 3.0 * d):   # both branches
            ref = float(mp.besseli(nu + 1, k) / mp.besseli(nu, k))
            got = a_d_scalar(d, k)
            got_t = a_d(d, torch.tensor([k], dtype=torch.float64)).item()
            assert abs(got - ref) < 1e-10 * max(ref, 1e-3), (d, k, got, ref)
            assert abs(got_t - ref) < 2e-4 * max(ref, 1e-2), (d, k, got_t, ref)
    print("PASS A_d vs mpmath (d=64, 512; continued fraction and scipy branches; table interp)")


def test_centred_update_view_path():
    """F3 through update_view (not update_sparse): features with a large shared component,
    per-pixel content by band; half the views swap every band's content."""
    from usplat.mock_rasterizer import MockFeatureRasterizer
    torch.manual_seed(1)
    N, d, H, W, B = 60, 24, 32, 32, 4
    common = Fn.normalize(torch.randn(d), dim=0)
    protos = Fn.normalize(torch.randn(2 * B, d), dim=-1)
    means2d = torch.rand(N, 2) * 32
    ras = MockFeatureRasterizer(means2d, 1.5 + torch.rand(N), 0.6 + 0.3 * torch.rand(N),
                                torch.rand(N), torch.stack(torch.meshgrid(
                                    torch.arange(W), torch.arange(H), indexing="xy"),
                                    -1).reshape(-1, 2).float())
    band = (torch.arange(W).repeat(H) * B // W)                 # (P,) band by x
    feat = lambda k: Fn.normalize(3.0 * common + protos[k], dim=-1)
    f = DirectionalSemanticField(N, d, FieldConfig(estimator="kish", change_test=False,
                                                   centre=True))
    f.set_centre(feat(band).mean(0))                            # mapping-frame mean
    for v in range(8):
        f.update_view(ras, feat(band + (B if v >= 4 else 0)), num_pixels=H * W)
    ok = f.support("centred") >= 3
    raw = f.agreement("raw")[ok].mean().item()
    cen = f.agreement("centred")[ok].mean().item()
    assert raw > 0.85 and cen < raw - 0.2, (raw, cen)
    print(f"PASS centred update_view: swapped content agreement raw {raw:.3f} vs centred {cen:.3f}")


def test_query_decode_first():
    """F7: query(project_text(t)) is the FULL-space cosine between the decoded direction
    B mu and t; renormalising the projected text changes the ranking."""
    from usplat.feature_pca import StreamingPCA
    torch.manual_seed(2)
    D, k, N, K = 64, 8, 200, 6
    pca = StreamingPCA(D, k)
    img = Fn.normalize(torch.randn(5000, k) @ torch.randn(k, D), dim=-1)
    pca.accumulate(img)
    pca.fit()
    text = Fn.normalize(torch.randn(K, D) + 0.5 * img[:K], dim=-1)
    f = DirectionalSemanticField(N, k, FieldConfig(centre=False, change_test=False,
                                                   view_mass_frac=0.01))
    f.update_sparse(torch.arange(N), torch.ones(N), pca.project(img[:N]), num_pixels=100)
    s = f.query(pca.project_text(text))
    full = (f.mu_post() @ pca.basis.t()) @ text.t()            # (B mu) . t
    assert torch.allclose(s, full, atol=1e-5)
    renorm = f.query(pca.project(text))
    assert (renorm.argmax(1) != s.argmax(1)).float().mean() > 0.05
    print(f"PASS F7: query == full-space cosine; renormalised text flips "
          f"{(renorm.argmax(1) != s.argmax(1)).float().mean() * 100:.0f}% of labels")


if __name__ == "__main__":
    test_identical_and_uniform()
    test_one_view()
    test_joint_prior_map_bruteforce()
    test_special_function()
    test_centred_channel_restores_consistency()
    test_kish_multi_pixel_fractional()
    test_a_d_against_mpmath()
    test_centred_update_view_path()
    test_query_decode_first()


def test_rbar_estimator_is_uncorrected_mean_length():
    """estimator="rbar" (Tab III ablation): agreement = ||R|| / W1, biased up at few views."""
    import torch.nn.functional as Fn
    from usplat.vmf_field import DirectionalSemanticField, FieldConfig
    torch.manual_seed(0)
    N, d = 200, 16
    f = DirectionalSemanticField(N, d, FieldConfig(estimator="rbar", change_test=False))
    for _ in range(4):
        z = Fn.normalize(torch.randn(N, d), dim=-1)             # rho = 0: no agreement
        f.update_sparse(torch.arange(N), torch.ones(N), z, num_pixels=100)
    r = f.agreement()
    assert torch.allclose(r, f.raw.R.norm(dim=-1) / f.raw.W1)
    kish = f.raw.agreement("kish", 1.0)
    assert r.mean() > 0.4 and kish.nan_to_num(0).mean() < r.mean() - 0.2   # ~1/sqrt(4) bias
    print(f"PASS rbar: mean {r.mean():.3f} at rho = 0 (kish {kish.nan_to_num(0).mean():.3f})")
