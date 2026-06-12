"""GPU validation of the gradient-trick accumulation on the REAL gsplat rasterizer.

Three checks that together certify the method end-to-end on real CUDA splatting:
 1. LINEARITY: render(a*b1 + b2) == a*render(b1) + render(b2). The gradient
    identity R_n = d/df_n sum_p z_p.F_p is exactly the linearity of compositing.
 2. MASS: with a constant ones-channel, sum_n n_eff_n == sum over pixels of the
    rendered alpha (the total blending mass) — an independent cross-check that
    the recovered n_eff are the true splat weights.
 3. vMF INVARIANTS: feeding a constant direction z to all pixels must give
    mu_post == z and r_bar == 1 for every observed Gaussian; feeding opposite
    directions on alternating frames must drive r_bar down (aleatoric up).

Skipped (exit 0 with a message) when CUDA or the gsplat backend is unavailable.
"""
import sys
import torch
import torch.nn.functional as Fn


def _make_scene(N=300, device="cuda", seed=0):
    torch.manual_seed(seed)
    means = torch.randn(N, 3, device=device) * 0.6
    quats = Fn.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales = 0.02 + 0.05 * torch.rand(N, 3, device=device)
    opac = 0.4 + 0.55 * torch.rand(N, device=device)
    viewmat = torch.eye(4, device=device)
    viewmat[2, 3] = 2.5                                  # camera back from origin
    K = torch.tensor([[120.0, 0, 80], [0, 120.0, 60], [0, 0, 1]], device=device)
    return means, quats, scales, opac, viewmat, K


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return
    try:
        from gsplat.cuda._backend import _C
        assert _C is not None
    except Exception:
        print("SKIP: gsplat CUDA backend not built")
        return

    from usplat.gsplat_adapter import GSplatFeatureRasterizer
    from usplat.bki_accumulate import bki_grad_accumulate
    from usplat.vmf_field import DirectionalSemanticField

    device = "cuda"
    N, d, W, H = 300, 24, 160, 120
    means, quats, scales, opac, viewmat, K = _make_scene(N, device)
    ras = GSplatFeatureRasterizer(means, quats, scales, opac, viewmat, K, W, H,
                                  channel_chunk=16)

    # --- 1. linearity --------------------------------------------------------
    b1 = torch.randn(N, d + 1, device=device)
    b2 = torch.randn(N, d + 1, device=device)
    a = 2.7
    with torch.no_grad():
        lhs = ras(a * b1 + b2)
        rhs = a * ras(b1) + ras(b2)
    lin_err = (lhs - rhs).abs().max().item()
    assert lin_err < 1e-3, f"compositing not linear?! max err {lin_err}"
    print(f"PASS linearity: max |render(a*b1+b2) - a*render(b1)-render(b2)| "
          f"= {lin_err:.2e}")

    # --- 2. n_eff mass == rendered alpha mass --------------------------------
    z = Fn.normalize(torch.randn(H * W, d, device=device), dim=-1)
    R, n_eff = bki_grad_accumulate(ras, z, N, device=device)
    with torch.no_grad():
        _, alpha = ras.render_image(torch.ones(N, 1, device=device))
    mass_gap = abs(n_eff.sum().item() - alpha.sum().item())
    rel = mass_gap / max(alpha.sum().item(), 1e-9)
    assert rel < 1e-3, f"n_eff mass mismatch: {rel:.2e}"
    assert (n_eff >= -1e-5).all()
    print(f"PASS mass: sum n_eff = {n_eff.sum():.1f} vs alpha mass "
          f"{alpha.sum():.1f} (rel gap {rel:.2e})")

    # --- 3. vMF invariants on the real rasterizer ----------------------------
    field = DirectionalSemanticField(N, d, kappa_lik=10.0, device=device)
    z0 = Fn.normalize(torch.randn(d, device=device), dim=0)
    field.update_from_rasterizer(ras, z0.expand(H * W, d).contiguous())
    seen = field.n_eff > 1.0
    cos = (field.mu_post()[seen] * z0).sum(-1)
    rbar = field.mean_resultant_length()[seen]
    assert (cos > 0.999).all(), cos.min()
    assert (rbar > 0.999).all(), rbar.min()
    print(f"PASS constant-direction: min cos(mu_post, z) = {cos.min():.5f}, "
          f"min r_bar = {rbar.min():.5f} on {int(seen.sum())} observed Gaussians")

    # conflicting evidence: opposite directions on alternating updates
    field.reset()
    for t in range(6):
        zt = z0 if t % 2 == 0 else -z0
        field.update_from_rasterizer(ras, zt.expand(H * W, d).contiguous())
    ale, valid = field.aleatoric(min_obs=3.0)
    assert valid.any()
    assert ale[valid].mean() > 0.5, ale[valid].mean()
    print(f"PASS conflict: mean aleatoric under flipping semantics = "
          f"{ale[valid].mean():.3f} (> 0.5)")
    print("ALL GPU CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
