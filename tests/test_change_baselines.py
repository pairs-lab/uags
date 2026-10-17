"""K13 change baselines: per-pixel residuals, lifting, temporal filter, NaN, rank fusion.

CPU only (mock rasterizer with an explicit weight matrix W). Checks:
  NCC      identical images give 1 - NCC ~ 0; a gain/offset change still gives ~0;
           flat windows in EITHER image are masked; one pixel of a non-square image
           matches a direct window computation; min_valid_frac at a padding edge.
  lifting  every residual equals the brute-force W-weighted mean sum_p w a r / sum_p w a
           (non-square images, so a transposed H/W cannot pass).
  coverage the default rule is the field's m > EPS; a gate w_v >= g binds with w_v from
           a0 * num_pixels; a shared `observed` mask replaces the gate.
  filter   the k-consecutive min ignores a single-view spike; unobserved views keep state;
           the EWMA uses the view weight; a restored state continues like an unbroken run.
  NaN      never-observed scores are NaN; kmin is NaN until k observations.
  fusion   fuse_rank on a toy case, ties, all-NaN rows, monotone invariance, float64 out,
           and invariance to Gaussians outside the evaluation population (mask).
"""
import math
import pytest
import torch
import torch.nn.functional as Fn

from usplat.mock_rasterizer import MockFeatureRasterizer
from usplat.change_baselines import (ChangeBaselines, TemporalScore, ncc_residual,
                                     feature_residual, to_grey, normalized_rank, fuse_rank,
                                     common_defined)
from usplat.evaluation import auroc, average_precision


def _grid_scene(H=24, W=24, step=3, sigma=1.5, off=(), seed=0):
    """Gaussians on a regular grid over an H x W image; ids in `off` are moved off-screen."""
    g = torch.Generator().manual_seed(seed)
    ys, xs = torch.meshgrid(torch.arange(1, H, step).float(), torch.arange(1, W, step).float(),
                            indexing="ij")
    means = torch.stack([xs.flatten(), ys.flatten()], -1)             # (N, 2) as (x, y)
    N = means.shape[0]
    for i in off:
        means[i] = torch.tensor([500.0, 500.0])
    py, px = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    pixels = torch.stack([px.flatten(), py.flatten()], -1)            # row-major P = H*W
    ras = MockFeatureRasterizer(means, torch.full((N,), sigma), torch.full((N,), 0.8),
                                torch.rand(N, generator=g), pixels)
    return ras, means, N


def _mixed_scene(H=20, W=28, N=80, seed=0):
    """Random Gaussians, half small (sigma 0.5: little pixel mass), half large (sigma 2);
    the last one is off-screen."""
    g = torch.Generator().manual_seed(seed)
    means = torch.stack([torch.rand(N, generator=g) * W, torch.rand(N, generator=g) * H], -1)
    sig = torch.where(torch.arange(N) % 2 == 0, torch.full((N,), 0.5), torch.full((N,), 2.0))
    means[-1] = torch.tensor([500.0, 500.0])                         # one off-screen
    py, px = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    ras = MockFeatureRasterizer(means, sig, torch.full((N,), 0.7), torch.rand(N, generator=g),
                                torch.stack([px.flatten(), py.flatten()], -1))
    return ras, N


def _same(x, y):
    """Exact equality that treats NaN == NaN."""
    return torch.equal(torch.isnan(x), torch.isnan(y)) and torch.equal(x.nan_to_num(-1),
                                                                         y.nan_to_num(-1))


# ------------------------------------------------------------------ NCC
def _texture(H, W, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(H * W, 3, generator=g)


def _direct_ncc(obs, ren, H, W, y, x, half=3):
    """1 - NCC of the full (2 half + 1)^2 window at interior pixel (y, x), by hand."""
    go, gr = to_grey(obs).reshape(H, W), to_grey(ren).reshape(H, W)
    wo = go[y - half:y + half + 1, x - half:x + half + 1].double().flatten()
    wr = gr[y - half:y + half + 1, x - half:x + half + 1].double().flatten()
    wo = (wo - wo.mean()) / wo.std(unbiased=False)
    wr = (wr - wr.mean()) / wr.std(unbiased=False)
    return 1.0 - (wo * wr).mean().item()


def test_ncc_identical_and_exposure():
    H, W = 20, 24
    ren = _texture(H, W, 1)
    res, ok = ncc_residual(ren.clone(), ren, (H, W))
    assert ok.float().mean() > 0.99
    assert res[ok].abs().max() < 1e-6, res[ok].abs().max()
    # gain/offset (exposure) change of all channels: grey' = 0.6 grey + 0.15
    obs = 0.6 * ren + 0.15
    res, ok = ncc_residual(obs, ren, (H, W))
    assert ok.float().mean() > 0.99
    assert res[ok].abs().max() < 1e-5, res[ok].abs().max()
    # unrelated content: 1 - NCC ~ 1 on average; contrast inversion gives ~2
    res, ok = ncc_residual(_texture(H, W, 2), ren, (H, W))
    assert 0.8 < res[ok].mean() < 1.2, res[ok].mean()
    res, ok = ncc_residual(1.0 - ren, ren, (H, W))
    assert (res[ok] > 2 - 1e-5).all()


def test_ncc_flat_windows_masked_and_direct_value():
    H, W = 14, 22                                          # non-square: H/W swaps show
    ren = _texture(H, W, 3).reshape(H, W, 3)
    obs = _texture(H, W, 4).reshape(H, W, 3)
    ren[1:13, 1:9] = 0.5                                   # flat block in the RENDER
    obs[1:13, 14:21] = 0.3                                 # flat block in the OBSERVED image
    ren, obs = ren.reshape(-1, 3), obs.reshape(-1, 3)
    res, ok = ncc_residual(obs, ren, (H, W), window=7)
    okm = ok.reshape(H, W)
    assert not okm[4:10, 4:6].any()                         # windows inside the render block
    assert not okm[4:10, 17:18].any()                       # windows inside the observed block
    assert okm[:, 11].all()                                 # textured in both: defined
    assert (res[~ok] == 0).all()
    # direct window computation at interior pixels (window touches neither block)
    for y, x in ((3, 11), (10, 11), (7, 11)):
        ref = _direct_ncc(obs, ren, H, W, y, x)
        assert abs(res.reshape(H, W)[y, x].item() - ref) < 1e-5, (y, x)
    # the same check on unmasked textures at off-centre pixels of a wide image
    H, W = 11, 25
    obs, ren = _texture(H, W, 6), _texture(H, W, 7)
    res, ok = ncc_residual(obs, ren, (H, W))
    for y, x in ((3, 3), (7, 20), (5, 12), (3, 21)):
        assert abs(res.reshape(H, W)[y, x].item() - _direct_ncc(obs, ren, H, W, y, x)) < 1e-5


def test_ncc_float64_statistics():
    """Window variance is E[x^2] - E[x]^2: at a large offset it needs float64 inside."""
    H, W = 16, 20
    t = _texture(H, W, 8)
    ren = 100.0 + 0.2 * t                                  # large mean, small spread
    obs = 100.0 + 0.4 * t                                  # affine in grey: 1 - NCC = 0
    res, ok = ncc_residual(obs, ren, (H, W))
    assert ok.all()
    assert res.abs().max() < 1e-4, res.abs().max()


def test_ncc_border_and_pixel_valid():
    H, W = 16, 16
    ren = _texture(H, W, 5)
    valid = torch.ones(H, W, dtype=torch.bool)
    valid[:, :4] = False                                    # padding columns
    obs = ren.clone().reshape(H, W, 3)
    obs[:, :4] = 0.0                                       # garbage in the padding
    res, ok = ncc_residual(obs.reshape(-1, 3), ren, (H, W), valid=valid.flatten())
    okm = ok.reshape(H, W)
    assert not okm[:, :4].any()
    assert okm[:, 4:].all()                                 # incl. border pixels (0, W-1)
    assert res[ok].abs().max() < 1e-6                       # padding never enters a window
    # min_valid_frac at the padding edge: column 4 sees 4/7 valid columns, column 5 5/7
    _, ok = ncc_residual(obs.reshape(-1, 3), ren, (H, W), valid=valid.flatten(),
                         min_valid_frac=0.6)
    okm = ok.reshape(H, W)
    assert not okm[:, 4].any() and okm[:, 5:].all()
    # an isolated 2x2 valid island: 4/49 valid in its window -> masked at the default 0.5
    island = torch.zeros(H, W, dtype=torch.bool)
    island[7:9, 7:9] = True
    other = _texture(H, W, 9)
    _, ok = ncc_residual(other, ren, (H, W), valid=island.flatten())
    assert not ok.any()
    _, ok = ncc_residual(other, ren, (H, W), valid=island.flatten(), min_valid_frac=0.05)
    assert torch.equal(ok, island.flatten())


def test_feature_residual():
    g = torch.Generator().manual_seed(0)
    z = Fn.normalize(torch.randn(50, 8, generator=g), dim=-1)
    res, ok = feature_residual(z, 0.3 * z)                  # un-normalised render
    assert ok.all() and res.abs().max() < 1e-6
    res, ok = feature_residual(z, -z)
    assert (res - 2).abs().max() < 1e-6
    ren = z.clone(); ren[:5] = 0
    res, ok = feature_residual(z, ren)
    assert not ok[:5].any() and ok[5:].all() and (res[:5] == 0).all()
    obs = z.clone(); obs[10:13] = 0                         # empty observation (padding)
    res, ok = feature_residual(obs, z)
    assert not ok[10:13].any() and ok[13:].all() and (res[10:13] == 0).all()


# ------------------------------------------------------------------ lifting
def _lift(Wm, r, a):
    m = Wm.t() @ a.double()
    return (Wm.t() @ (a.double() * r.double())) / m.clamp_min(1e-300), m


def test_lifted_residual_equals_bruteforce():
    H, W = 16, 22
    ras, means, N = _grid_scene(H, W, step=3, off=(0, 7))
    P = H * W
    g = torch.Generator().manual_seed(1)
    obs_rgb, ren_rgb = torch.rand(P, 3, generator=g), torch.rand(P, 3, generator=g)
    ren_rgb[:60] = 0.4                                     # a flat strip: NCC-masked pixels
    d = 6
    obs_f, ren_f = torch.randn(P, d, generator=g), torch.randn(P, d, generator=g)
    ren_f[100:130] = 0                                     # empty render: feature-masked
    dep = 1.0 + torch.rand(P, generator=g)
    dep[::7] = 0.0                                         # invalid depth
    dep[3] = float("nan")
    rdep = 1.0 + torch.rand(P, generator=g)
    rdep[200:220] = 0.0                                    # no rendered depth there
    gd = 1.0 + torch.rand(N, generator=g)
    valid = torch.ones(P); valid[-20:] = 0                 # padding

    cb = ChangeBaselines(N, k=2, view_mass_frac=1e-4)
    out = cb.update(ras, P, observed_rgb=obs_rgb, rendered_rgb=ren_rgb, observed_feat=obs_f,
                    rendered_feat=ren_f, depth_obs=dep, rendered_depth=rdep, gauss_depth=gd,
                    image_hw=(H, W), pixel_valid=valid)
    assert set(out) == {"ncc", "feat", "depth_gauss", "depth_render"}

    Wm = ras.weights().double()                            # (P, N)
    r_ncc, ok_ncc = ncc_residual(obs_rgb, ren_rgb, (H, W), valid=valid > 0)
    r_f, ok_f = feature_residual(obs_f, ren_f)
    good = torch.isfinite(dep) & (dep > 0)
    dep0 = torch.where(good, dep, torch.zeros_like(dep))
    g2 = good & (rdep > 0)
    ref = {
        "ncc": _lift(Wm, r_ncc, valid * ok_ncc),
        "feat": _lift(Wm, r_f, valid * ok_f),
        "depth_render": _lift(Wm, torch.where(g2, (dep0 - rdep).abs(), torch.zeros_like(dep0)),
                              valid * g2),
    }
    mean_d, m_d = _lift(Wm, dep0, valid * good)
    ref["depth_gauss"] = ((mean_d - gd.double()).abs(), m_d)
    for name, (rb, m) in ref.items():
        seen = m > 1e-6
        assert seen.sum() > N // 2, name
        err = (out[name].double()[seen] - rb[seen]).abs().max().item()
        assert err < 1e-4, (name, err)
        assert torch.isnan(out[name][m == 0]).all(), name  # off-screen Gaussians
    assert torch.isnan(out["feat"][[0, 7]]).all()


def test_update_uses_rasterizer_size_and_errors():
    H, W = 12, 20                                          # non-square
    ras, _, N = _grid_scene(H, W, step=4)
    P = H * W
    ras.height, ras.width = H, W                           # like GSplatFeatureRasterizer
    obs, ren = _texture(H, W, 0), _texture(H, W, 1)        # different content: NCC != 0
    cb = ChangeBaselines(N)
    out = cb.update(ras, P, observed_rgb=obs, rendered_rgb=ren)
    r, ok = ncc_residual(obs, ren, (H, W))
    rb, m = _lift(ras.weights().double(), r, ok.double())
    seen = m > 1e-6
    assert seen.sum() == N
    assert (out["ncc"].double()[seen] - rb[seen]).abs().max() < 1e-5
    assert out["ncc"][seen].min() > 0.1                    # real residual, not a zero test
    with pytest.raises(ValueError):
        cb.update(ras, P, depth_obs=torch.ones(P))          # no gauss/rendered depth
    with pytest.raises(ValueError):
        cb.update(ras, P)
    with pytest.raises(ValueError):                        # a half pair is never dropped silently
        cb.update(ras, P, observed_rgb=obs, observed_feat=torch.ones(P, 4),
                  rendered_feat=torch.ones(P, 4))
    with torch.inference_mode(), pytest.raises(RuntimeError):
        cb.update(ras, P, observed_rgb=obs, rendered_rgb=ren)
    # shape checks: every per-pixel input has P rows, every per-Gaussian input N entries
    dep = torch.full((P,), 2.0)
    bad = [dict(depth_obs=dep, gauss_depth=torch.ones(1)),          # used to broadcast
           dict(depth_obs=dep, gauss_depth=torch.ones(N - 1)),
           dict(depth_obs=dep, rendered_depth=torch.ones(P - 1)),
           dict(depth_obs=dep, gauss_depth=torch.ones(N), pixel_valid=torch.ones(P + 1)),
           dict(observed_feat=torch.ones(P, 4), rendered_feat=torch.ones(P - 1, 4)),
           dict(observed_feat=torch.ones(P, 4), rendered_feat=torch.ones(P, 4),
                observed=torch.ones(N - 1, dtype=torch.bool)),
           dict(observed_feat=torch.ones(P, 4), rendered_feat=torch.ones(P, 4),
                observed=torch.ones(N))]                            # not a bool mask
    for kw in bad:
        with pytest.raises(ValueError):
            cb.update(ras, P, **kw)
    assert cb.num_views == 1                                # rejected views change nothing


def test_inputs_are_detached():
    """Per-pixel inputs that carry a caller's graph (even a freed one) do not enter the
    lifting backward pass, and no gradient leaks into the caller's leaves."""
    H, W = 10, 14
    ras, _, N = _grid_scene(H, W, step=3)
    P = H * W
    dep = 1.5 + 0.1 * torch.rand(P, generator=torch.Generator().manual_seed(0))
    leaf = torch.full((N, 1), 1.5, requires_grad=True)
    rd = ras(leaf)[:, 0]
    rd.sum().backward()                                    # the caller already used the graph
    leaf.grad = None
    cb = ChangeBaselines(N)
    out = cb.update(ras, P, depth_obs=dep, rendered_depth=rd, gauss_depth=torch.ones(N))
    assert torch.isfinite(out["depth_render"]).any()
    leaf2 = torch.full((N, 1), 1.5, requires_grad=True)
    rd2 = ras(leaf2)[:, 0]                                 # live graph
    feat = Fn.normalize(torch.randn(P, 4), dim=-1).requires_grad_(True)
    cb.update(ras, P, depth_obs=dep * 1.0, rendered_depth=rd2, observed_feat=feat,
              rendered_feat=feat.detach() + 0.1, pixel_valid=torch.ones(P, requires_grad=True))
    assert leaf2.grad is None and feat.grad is None


# ------------------------------------------------------------------ coverage rule
def test_coverage_rule_default_gate_and_shared_mask():
    """Default: observed iff m > EPS, the field's own seen rule. A gate binds on
    w_v = min(1, m / (a0 * num_pixels)); a shared `observed` mask replaces the gate."""
    from usplat.vmf_field import DirectionalSemanticField, FieldConfig
    H, W = 20, 28
    ras, N = _mixed_scene(H, W)
    P = H * W
    a0 = 1e-2
    g = torch.Generator().manual_seed(3)
    ren = Fn.normalize(torch.randn(P, 8, generator=g), dim=-1)
    views = [Fn.normalize(ren + 0.8 * torch.randn(P, 8, generator=g), dim=-1)
             for _ in range(3)]
    m = ras.weights().double().sum(0)                      # all pixels valid for feat
    seen = m > 1e-12
    wv = (m / (a0 * P)).clamp(max=1.0)
    small = seen & (wv < 0.5)
    assert small.sum() >= 5 and (seen & (wv >= 0.5)).sum() >= 5 and (~seen).sum() >= 1

    # default rule == the field's seen rule on the same view
    field = DirectionalSemanticField(N, 8, FieldConfig(centre=False, change_test=False,
                                                       view_mass_frac=a0))
    got = []
    field.view_hooks.append(lambda s: got.append(s.m > 1e-12))
    field.update_view(ras, views[0])
    cb0 = ChangeBaselines(N, view_mass_frac=a0)
    r0 = cb0.update(ras, P, observed_feat=views[0], rendered_feat=ren)["feat"]
    assert torch.equal(torch.isfinite(r0), got[0]) and torch.equal(torch.isfinite(r0), seen)

    # gate 0.5: small-mass Gaussians stay NaN and their ring stays empty
    cb5 = ChangeBaselines(N, view_mass_frac=a0, min_view_weight=0.5)
    for z in views:
        r5 = cb5.update(ras, P, observed_feat=z, rendered_feat=ren)["feat"]
        assert torch.equal(torch.isfinite(r5), seen & (wv >= 0.5))
    ts = cb5.series["feat"]
    assert torch.isnan(ts.ring[small]).all() and (ts.count[small] == 0).all()
    assert torch.isnan(cb5.score("feat", "kmin")[small]).all()
    # the reference mass is a0 * num_pixels: 4x the pixel count divides every w_v by 4
    cb5b = ChangeBaselines(N, view_mass_frac=a0, min_view_weight=0.5)
    r5b = cb5b.update(ras, 4 * P, observed_feat=views[0], rendered_feat=ren)["feat"]
    assert torch.equal(torch.isfinite(r5b), seen & (m / (a0 * 4 * P) >= 0.5))

    # EWMA through update uses the view weight: e = r1 + min(1, rate * w_v) (r2 - r1)
    cbe = ChangeBaselines(N, view_mass_frac=a0, ewma_rate=0.8)
    r1 = cbe.update(ras, P, observed_feat=views[0], rendered_feat=ren)["feat"].double()
    r2 = cbe.update(ras, P, observed_feat=views[1], rendered_feat=ren)["feat"].double()
    e_ref = r1 + (0.8 * wv).clamp(max=1.0) * (r2 - r1)
    e = cbe.score("feat", "ewma").double()
    assert (e[seen] - e_ref[seen]).abs().max() < 1e-5
    assert (small & ((r2 - r1).abs() > 0.05)).any()         # the weight really matters here

    # a shared mask replaces the gate; unobserved Gaussians keep their state
    shared = seen.clone(); shared[torch.nonzero(seen)[:4, 0]] = False
    cbs = ChangeBaselines(N, view_mass_frac=a0, min_view_weight=0.5)
    rs = cbs.update(ras, P, observed_feat=views[0], rendered_feat=ren, observed=seen)["feat"]
    assert torch.equal(torch.isfinite(rs), seen)             # small ones included
    before = {k: v.clone() for k, v in cbs.series["feat"].state_dict().items()}
    rs = cbs.update(ras, P, observed_feat=views[1], rendered_feat=ren, observed=shared)["feat"]
    assert torch.equal(torch.isfinite(rs), shared)
    after = cbs.series["feat"].state_dict()
    held = seen & ~shared
    for k in before:
        assert _same(before[k][held], after[k][held]), k
    assert torch.equal(after["count"][held], torch.ones(int(held.sum()), dtype=torch.long))


def test_common_defined():
    nan = float("nan")
    a = torch.tensor([0.1, nan, 0.3, 0.4, nan])
    b = torch.tensor([1.0, 2.0, nan, 4.0, nan])
    assert common_defined([a, b]).tolist() == [True, False, False, True, False]
    pop = torch.tensor([False, True, True, True, True])
    assert common_defined([a, b], pop).tolist() == [False, False, False, True, False]
    with pytest.raises(ValueError):
        common_defined([a, b[:3]])


# ------------------------------------------------------------------ temporal filter
def test_k_consecutive_filter_ignores_single_spike():
    nan = float("nan")
    # rows: Gaussians; columns: views; NaN = not observed in that view
    seq = torch.tensor([
        [0.1, 0.9, 0.1, 0.1, 0.1],     # 0 single-view spike           -> kmin 0.1
        [0.1, 0.9, 0.9, 0.9, 0.9],     # 1 persistent change            -> kmin 0.9
        [0.1, 0.9, nan, 0.9, 0.1],     # 2 unobserved view does not break the run
        [nan, nan, nan, nan, nan],     # 3 never observed               -> all NaN
        [nan, nan, 0.7, nan, nan],     # 4 observed once (k=2)          -> kmin NaN
        [0.9, 0.1, 0.9, 0.1, 0.9],     # 5 flicker                      -> kmin 0.1
    ])
    ts = TemporalScore(seq.shape[0], k=2, ewma_rate=0.5)
    for t in range(seq.shape[1]):
        r = seq[:, t]
        ts.push(r, torch.isfinite(r), None)
    assert torch.allclose(ts.kmin[[0, 1, 2, 5]], torch.tensor([0.1, 0.9, 0.9, 0.1]))
    assert torch.allclose(ts.max[[0, 1, 2, 4, 5]], torch.tensor([0.9, 0.9, 0.9, 0.7, 0.9]))
    assert torch.allclose(ts.last[[0, 1, 2, 4]], torch.tensor([0.1, 0.9, 0.1, 0.7]))
    for f in ("kmin", "max", "ewma", "last"):
        assert torch.isnan(ts.get(f)[3]), f
    assert torch.isnan(ts.kmin[4]) and torch.isfinite(ts.ewma[4])
    # EWMA, rate 0.5, unit weights: Gaussian 0: 0.1 -> 0.5 -> 0.3 -> 0.2 -> 0.15
    assert abs(ts.ewma[0].item() - 0.15) < 1e-6
    # Gaussian 2 skips view 3: 0.1 -> 0.5 -> (kept) -> 0.7 -> 0.4
    assert abs(ts.ewma[2].item() - 0.4) < 1e-6
    # k = 3 needs three consecutive high views: Gaussian 1 yes, Gaussian 2 no
    ts3 = TemporalScore(seq.shape[0], k=3)
    for t in range(seq.shape[1]):
        ts3.push(seq[:, t], torch.isfinite(seq[:, t]))
    assert abs(ts3.kmin[1].item() - 0.9) < 1e-6 and abs(ts3.kmin[2].item() - 0.1) < 1e-6
    # k = 1 is the plain max
    ts1 = TemporalScore(seq.shape[0], k=1)
    for t in range(seq.shape[1]):
        ts1.push(seq[:, t], torch.isfinite(seq[:, t]))
    assert torch.equal(torch.isnan(ts1.kmin), torch.isnan(ts1.max))
    ok = torch.isfinite(ts1.max)
    assert torch.allclose(ts1.kmin[ok], ts1.max[ok])


def test_ewma_view_weight_and_clamp():
    obs = torch.ones(3, dtype=torch.bool)
    ts = TemporalScore(3, k=2, ewma_rate=0.5)
    ts.push(torch.full((3,), 0.1), obs, torch.tensor([1.0, 1.0, 1.0]))
    ts.push(torch.full((3,), 0.9), obs, torch.tensor([0.5, 1.0, 0.0]))
    # rate * w = 0.25, 0.5, 0: 0.1 + 0.25 * 0.8 = 0.3; 0.1 + 0.5 * 0.8 = 0.5; 0.1
    assert torch.allclose(ts.ewma, torch.tensor([0.3, 0.5, 0.1]), atol=1e-6)
    # rate * w is clamped to 1: the EWMA never overshoots the new value
    ts = TemporalScore(2, k=1, ewma_rate=2.0)
    ts.push(torch.full((2,), 0.1), torch.ones(2, dtype=torch.bool))
    ts.push(torch.full((2,), 0.9), torch.ones(2, dtype=torch.bool), torch.tensor([1.0, 0.25]))
    assert torch.allclose(ts.ewma, torch.tensor([0.9, 0.5]), atol=1e-6)


def test_scene_spike_vs_persistent_change_and_nan():
    """Mock scene, 3 views. Region A changes in view 0 only (a spike); region B changes in
    every view; one Gaussian is off-screen. feat/kmin keeps B and drops A; feat/max keeps both."""
    H, W = 24, 24
    ras, means, N = _grid_scene(H, W, step=3, off=(63,))
    P = H * W
    g = torch.Generator().manual_seed(2)
    d = 16
    f_map = Fn.normalize(torch.randn(N, d, generator=g), dim=-1)
    ren = ras(f_map)                                       # rendered features (P, d)
    py, px = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    regA = ((px < 12) & (py < 12)).flatten()
    regB = ((px >= 12) & (py >= 12)).flatten()
    inside = lambda x0, x1, y0, y1: ((means[:, 0] > x0) & (means[:, 0] < x1)
                                     & (means[:, 1] > y0) & (means[:, 1] < y1))
    gA, gB = inside(-1, 9, -1, 9), inside(15, 24, 15, 24)
    gS = inside(15, 24, -1, 9) | inside(-1, 9, 15, 24)   # static, away from both regions
    gA[63] = gB[63] = gS[63] = False
    cb = ChangeBaselines(N, k=2, view_mass_frac=1e-3, min_view_weight=0.5)
    for v in range(3):
        obs = Fn.normalize(ren + 0.05 * torch.randn(P, d, generator=g), dim=-1)
        new = Fn.normalize(torch.randn(P, d, generator=g), dim=-1)
        chg = regB | (regA if v == 0 else torch.zeros_like(regA))
        obs = torch.where(chg[:, None], new, obs)
        cb.update(ras, P, observed_feat=obs, rendered_feat=ren)
    s = cb.scores()
    kmin, mx = s["feat/kmin"], s["feat/max"]
    assert kmin[gB].min() > 0.5 and mx[gA].min() > 0.5
    assert kmin[gA].max() < 0.2 and kmin[gS].max() < 0.2, (kmin[gA].max(), kmin[gS].max())
    lab = gB.float()
    keep = gA | gB | gS
    assert auroc(kmin[keep], lab[keep]) == 1.0
    # at a 0.5 threshold the one-shot max flags every spike Gaussian, the k-min none
    neg = keep & ~gB
    assert (mx[neg] > 0.5).sum() == gA.sum() and (kmin[neg] > 0.5).sum() == 0
    for key, val in s.items():
        assert torch.isnan(val[63]), key                   # never observed
    assert cb.num_views == 3
    # scores() hands out copies: editing them does not touch the accumulator
    kept = kmin.clone()
    s["feat/kmin"].fill_(-1.0)
    assert _same(cb.scores()["feat/kmin"], kept) and _same(cb.score("feat", "kmin"), kept)


def test_oracle_naming_and_state_roundtrip():
    H, W = 12, 12
    ras, _, N = _grid_scene(H, W, step=4)
    P = H * W
    dep = torch.full((P,), 2.0)
    cb = ChangeBaselines(N, oracle_depth=True)
    for _ in range(2):
        cb.update(ras, P, depth_obs=dep, gauss_depth=torch.full((N,), 2.5),
                  rendered_depth=torch.full((P,), 2.0))
    assert set(cb.names) == {"depth_gauss_oracle", "depth_render_oracle"}
    s = cb.scores()
    assert "depth_gauss_oracle/kmin" in s and "depth_gauss/kmin" not in s
    assert torch.allclose(s["depth_gauss_oracle/kmin"], torch.full((N,), 0.5), atol=1e-5)
    assert torch.allclose(s["depth_render_oracle/max"], torch.zeros(N), atol=1e-6)
    cb2 = ChangeBaselines.from_state(cb.state_dict())
    s2 = cb2.scores()
    assert s2.keys() == s.keys() and all(torch.equal(s[k], s2[k]) for k in s)
    assert ChangeBaselines(N).depth_gauss_name == "depth_gauss"


def test_state_roundtrip_continues_like_unbroken_run():
    """Save after 2 views, restore, feed 2 more: equal to 4 views in one go (ring, count and
    dtype survive; float64 stays float64)."""
    H, W = 10, 16
    ras64, _, N = _grid_scene(H, W, step=3)
    ras64 = MockFeatureRasterizer(ras64.means2d.double(), ras64.sigmas.double(),
                                  ras64.opacities.double(), ras64.depths.double(),
                                  ras64.pixels.double(), dtype=torch.float64)
    P = H * W
    g = torch.Generator().manual_seed(5)
    ren = torch.randn(P, 6, generator=g, dtype=torch.float64)
    views = [ren + s * torch.randn(P, 6, generator=g, dtype=torch.float64)
             for s in (0.1, 2.0, 0.1, 2.0)]
    full = ChangeBaselines(N, k=3, dtype=torch.float64)
    part = ChangeBaselines(N, k=3, dtype=torch.float64)
    for z in views:
        full.update(ras64, P, observed_feat=z, rendered_feat=ren)
    for z in views[:2]:
        part.update(ras64, P, observed_feat=z, rendered_feat=ren)
    restored = ChangeBaselines.from_state(part.state_dict())
    assert restored.dtype == torch.float64 and restored.config() == part.config()
    for z in views[2:]:
        restored.update(ras64, P, observed_feat=z, rendered_feat=ren)
    a, b = full.scores(), restored.scores()
    assert a.keys() == b.keys() and restored.num_views == 4
    for k in a:
        assert b[k].dtype == torch.float64 and _same(a[k], b[k]), k
    assert torch.isfinite(a["feat/kmin"]).all()             # k = 3 reached across the cut


def test_push_view_gives_the_update_only_state():
    """run_mapping.py's update-pass-only state: an accumulator fed (push_view) the per-view
    residuals and view weights of the update views that the whole-run accumulator returned
    equals, bit for bit, one that ran update() on the update views alone (every residual,
    every filter, the ewma's view weights, partial coverage); the whole-run accumulator,
    whose ring the residual null reads, is not touched."""
    H, W = 20, 28
    ras, N = _mixed_scene(H, W)
    P = H * W
    g = torch.Generator().manual_seed(3)
    ren_rgb = torch.rand(P, 3, generator=g)
    ren_f = torch.randn(P, 5, generator=g)
    rd = 1.0 + torch.rand(P, generator=g)
    gd = 1.0 + torch.rand(N, generator=g)

    def view(t):
        valid = torch.rand(P, generator=g) > 0.2 * (t % 2)          # partial coverage
        return dict(observed_rgb=(ren_rgb + 0.1 * t * torch.rand(P, 3, generator=g)).clamp(0, 1),
                    rendered_rgb=ren_rgb, observed_feat=ren_f + 0.5 * torch.randn(P, 5, generator=g),
                    rendered_feat=ren_f, depth_obs=rd + 0.1 * torch.rand(P, generator=g),
                    rendered_depth=rd, gauss_depth=gd, image_hw=(H, W), pixel_valid=valid)
    views = [view(t) for t in range(7)]
    whole = ChangeBaselines(N, k=2, view_mass_frac=1e-2)
    for v in views[:3]:                                              # mapping pass
        whole.update(ras, P, **v)
    upd = whole.empty_like()
    assert upd.config() == whole.config() and upd.names == [] and upd.num_views == 0
    alone = ChangeBaselines(N, k=2, view_mass_frac=1e-2)
    for v in views[3:]:                                              # update pass
        pv = whole.update(ras, P, **v)
        ring = {n: s.ring.clone() for n, s in whole.series.items()}
        upd.push_view(pv, whole.last_view_weights)
        assert all(_same(ring[n], s.ring) for n, s in whole.series.items())
        alone.update(ras, P, **v)
    a, b = alone.scores(), upd.scores()
    assert a.keys() == b.keys() and len(a) == 4 * 4 and upd.num_views == alone.num_views == 4
    for k in a:
        assert _same(a[k], b[k]), k
    assert not _same(whole.scores()["feat/max"], a["feat/max"])      # the whole run differs
    w = ChangeBaselines(N, k=2, view_mass_frac=1e-2)
    w.push_view(pv)                                                  # no weights: ewma rate 1
    assert torch.isfinite(w.scores()["feat/ewma"]).any()


# ------------------------------------------------------------------ fusion
def test_fuse_rank_toy():
    nan = float("nan")
    f64 = lambda *v: torch.tensor(v, dtype=torch.float64)
    s1 = torch.tensor([0.1, 0.5, 0.9, nan])
    s2 = torch.tensor([nan, 3.0, 2.0, 1.0])
    assert torch.allclose(normalized_rank(s1), f64(1 / 3, 2 / 3, 1.0, 0.0))
    f = fuse_rank([s1, s2])
    assert f.dtype == torch.float64
    assert torch.allclose(f, f64(1 / 6, 5 / 6, 5 / 6, 1 / 6), atol=1e-12)
    # ties share their mean rank; an all-NaN row stays NaN
    t = torch.tensor([2.0, 2.0, 1.0, nan, 5.0])
    assert torch.allclose(normalized_rank(t), f64(0.625, 0.625, 0.25, 0.0, 1.0))
    f = fuse_rank([torch.tensor([nan, 1.0]), torch.tensor([nan, nan])])
    assert math.isnan(f[0].item()) and f[1].item() == pytest.approx(0.5)
    # rank fusion ignores monotone rescaling (no fitting, no units)
    g = torch.Generator().manual_seed(0)
    a, b = torch.rand(100, generator=g), torch.rand(100, generator=g)
    assert torch.allclose(fuse_rank([a, b]), fuse_rank([a.exp() * 7, 1e3 * b - 5]))
    # constant score adds nothing to the ordering
    fa = fuse_rank([a, torch.ones(100)])
    assert torch.equal(torch.argsort(fa), torch.argsort(a))
    with pytest.raises(ValueError):
        fuse_rank([a, b[:5]])
    # float16 inputs: the fused output is float64 and keeps its resolution
    n = 20000
    h1, h2 = torch.rand(n, generator=g).half(), torch.rand(n, generator=g).half()
    fh = fuse_rank([h1, h2])
    assert fh.dtype == torch.float64
    assert torch.unique(fh).numel() == torch.unique(fuse_rank([h1.double(), h2.double()])).numel()


def test_fuse_rank_population_mask():
    """Ranks are taken inside the evaluation population only: Gaussians outside the mask
    (hidden, unobserved) that are finite in one score cannot move the fused order."""
    g = torch.Generator().manual_seed(7)
    n_in, n_out = 300, 900
    lab = (torch.rand(n_in, generator=g) < 0.2).float()
    c_in = lab + 0.8 * torch.randn(n_in, generator=g)
    d_in = lab + 0.8 * torch.randn(n_in, generator=g)
    c_in[::9] = float("nan")                                # some c undefined inside
    c_out = torch.full((n_out,), float("nan"))              # hidden: no c ...
    d_out = 0.5 + 0.01 * torch.randn(n_out, generator=g)    # ... but a depth residual
    c, d = torch.cat([c_in, c_out]), torch.cat([d_in, d_out])
    pop = torch.cat([torch.ones(n_in, dtype=torch.bool), torch.zeros(n_out, dtype=torch.bool)])
    f_sub = fuse_rank([c_in, d_in])
    f_mask = fuse_rank([c, d], mask=pop)
    assert torch.isnan(f_mask[~pop]).all()
    assert torch.allclose(f_mask[pop], f_sub, atol=1e-12)
    assert torch.allclose(normalized_rank(d, pop)[pop], normalized_rank(d_in), atol=1e-12)
    # without the mask the out-of-population depth values shift the order on the eval set
    f_all = fuse_rank([c, d])[pop]
    assert not torch.equal(torch.argsort(f_all, stable=True), torch.argsort(f_sub, stable=True))
    assert average_precision(f_mask[pop], lab) == pytest.approx(average_precision(f_sub, lab))
    with pytest.raises(ValueError):
        fuse_rank([c, d], mask=pop.float())


def test_depth_gauss_matches_field_depth_residual():
    """depth_gauss after one view == DirectionalSemanticField.depth_residual (same definition),
    with the default coverage rule."""
    from usplat.vmf_field import DirectionalSemanticField, FieldConfig
    H, W = 16, 16
    ras, _, N = _grid_scene(H, W, step=3, off=(2,))
    P = H * W
    g = torch.Generator().manual_seed(4)
    dep = 1.0 + torch.rand(P, generator=g)
    dep[::5] = 0.0
    gd = 1.0 + torch.rand(N, generator=g)
    field = DirectionalSemanticField(N, 8, FieldConfig(centre=False, change_test=False))
    field.update_view(ras, Fn.normalize(torch.randn(P, 8, generator=g), dim=-1),
                      depth_obs=dep, gauss_depth=gd)
    cb = ChangeBaselines(N)
    out = cb.update(ras, P, depth_obs=dep, gauss_depth=gd)["depth_gauss"]
    ref = field.depth_residual()
    assert torch.equal(torch.isnan(out), torch.isnan(ref))
    ok = torch.isfinite(ref)
    assert ok.sum() == N - 1
    assert torch.allclose(out[ok], ref[ok], atol=1e-6)
