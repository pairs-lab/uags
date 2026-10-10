"""F4/F5: temporal change test, reset, and re-admission.

Views are fed with update_sparse (one pixel per Gaussian, full view weight) so the test
controls exactly what each Gaussian sees. Groups of Gaussians:
  static   : content A in every view                   -> never flagged
  swapped  : A for 8 views, then B (same surface)       -> flagged, then re-admitted once
                                                           3 views agree AND depth fits
  removed  : A for 8 views, then B seen BEHIND it       -> flagged, never re-admitted
  unseen   : A for 8 views, then not observed           -> keeps its state, no decay
Known limit (recorded in the report): the test fires in a short window after the change;
with only 4 history views ~8% of orthogonal changes peak just below c = 0.5 and are missed.
"""
import torch
import torch.nn.functional as Fn
from usplat.vmf_field import DirectionalSemanticField, FieldConfig


def _view(field, ids, feats, depth_obs=None, gdepth=None, P=100):
    """One view: Gaussian ids[i] sees feats[i] with full weight (and optional depth)."""
    N, d = field.N, field.d
    R_v = torch.zeros(N, d); m_v = torch.zeros(N)
    R_v[ids] = feats; m_v[ids] = 1.0
    Rc_v = mc_v = None
    if field.cfg.centre:
        zc = feats - field.zbar if field.zbar is not None else feats
        Rc_v = torch.zeros(N, d); mc_v = torch.zeros(N)
        Rc_v[ids] = Fn.normalize(zc, dim=-1); mc_v[ids] = 1.0
    Ds = Dm = None
    if depth_obs is not None:
        Ds = torch.zeros(N); Dm = torch.zeros(N)
        Ds[ids] = depth_obs; Dm[ids] = 1.0
    from usplat.vmf_field import ViewStats
    field._ingest(ViewStats(R_v, m_v, Rc_v, mc_v, Ds, Dm, field.cfg.view_mass_frac * P), gdepth)


def _setup(n_per=50, d=16, seed=0, **cfg):
    g = torch.Generator().manual_seed(seed)
    A = Fn.normalize(torch.randn(4 * n_per, d, generator=g), dim=-1)
    B = torch.randn(4 * n_per, d, generator=g)
    B = Fn.normalize(B - (B * A).sum(-1, keepdim=True) * A, dim=-1)   # new content _|_ old
    groups = {k: torch.arange(i * n_per, (i + 1) * n_per) for i, k in
              enumerate(("static", "swapped", "removed", "unseen"))}
    cfg.setdefault("centre", True)                         # these tests cover the centred path
    cfg.setdefault("fast_memory", 0.5)                     # mechanics test: short memory
    cfg.setdefault("change_thr", 0.5)                      # and a fixed threshold
    cfg.setdefault("change_persist", 1)                    # no persistence filter
    field = DirectionalSemanticField(4 * n_per, d, FieldConfig(view_mass_frac=0.01, **cfg))
    field.set_centre(torch.zeros(d))
    gdepth = torch.full((4 * n_per,), 2.0)                # Gaussian centres at 2 m
    return field, A, B, groups, gdepth


def test_change_reset_readmit():
    field, A, B, G, gdepth = _setup()
    noise = lambda x: Fn.normalize(x + 0.05 * torch.randn_like(x), dim=-1)
    allid = torch.arange(field.N)
    for t in range(8):                                     # before the change (history)
        field_depth = torch.full((field.N,), 2.0)
        _view(field, allid, noise(A), field_depth, gdepth)
    assert not field.changed.any()
    state_unseen = field.raw.R[G["unseen"]].clone()
    seen = torch.cat([G["static"], G["swapped"], G["removed"]])
    fired_at = None
    for t in range(6):                                      # after the change
        feats = noise(A[seen].clone())
        n = len(G["static"])
        feats[n:] = noise(B[seen[n:]])                     # swapped + removed see B
        dep = torch.full((len(seen),), 2.0)
        dep[2 * n:] = 3.0                                   # removed: surface 1 m behind
        _view(field, seen, feats, dep, gdepth)
        if fired_at is None and (field.n_resets[G["swapped"]] >= 1).float().mean() > 0.9:
            fired_at = t
    assert fired_at is not None and fired_at <= 1, fired_at
    assert field.changed[G["static"]].float().mean() == 0
    assert field.n_resets[G["swapped"]].min() >= 1
    # swapped: re-admitted (views since reset >= 3 and depth fits); removed: stays flagged
    assert (~field.changed[G["swapped"]]).float().mean() > 0.9
    fired_rm = field.n_resets[G["removed"]] >= 1
    assert fired_rm.float().mean() > 0.9, fired_rm.float().mean()
    assert field.changed[G["removed"]][fired_rm].all()     # depth blocks re-admission
    # after the reset the posterior follows B
    cos = (field.mu_post()[G["swapped"]] * B[G["swapped"]]).sum(-1)
    assert cos.min() > 0.95, cos.min()
    # unseen Gaussians are untouched
    assert torch.equal(field.raw.R[G["unseen"]], state_unseen)
    assert not field.changed[G["unseen"]].any()
    rob = field.robust_mask()
    assert rob[G["static"]].float().mean() > 0.9 and not rob[G["removed"]].any()
    print(f"PASS change: fired within {fired_at + 1} view(s); static 0 flagged; swapped "
          f"re-admitted; removed stays flagged; unseen untouched")


def test_readmit_without_depth_uses_views():
    field, A, B, G, gdepth = _setup(readmit_depth=None)
    allid = torch.arange(field.N)
    for t in range(4):
        _view(field, allid, A)
    for t in range(6):
        _view(field, G["swapped"], B[G["swapped"]])
    assert (field.n_resets[G["swapped"]] >= 1).all()          # it did fire
    assert (~field.changed[G["swapped"]]).all()
    print("PASS re-admission on views alone when readmit_depth=None")


def test_buckets_partition_and_decay():
    """fast + slow = all centred evidence (lambda = 1); decay acts only on the slow
    bucket and only when the Gaussian is observed."""
    f, A, B, G, _ = _setup(change_test=True)
    ids = G["static"]
    for _ in range(3):
        _view(f, ids, A[ids])
    assert torch.allclose(f.Rs[ids] + f.Rf[ids], f.cen.R[ids], atol=1e-5)
    assert torch.allclose(f.ns[ids] + f.nf[ids], f.cen.W1[ids], atol=1e-5)
    g, *_ = _setup(slow_decay=0.5)
    _view(g, ids, A[ids]); _view(g, ids, A[ids])           # ns = 0.5 (leak of view 1)
    ns = g.ns[ids].clone()
    _view(g, G["unseen"], A[G["unseen"]])                  # ids not observed: unchanged
    assert torch.equal(g.ns[ids], ns)
    nf = g.nf[ids].clone()
    _view(g, ids, A[ids])                                  # ns <- 0.5 ns + 0.5 nf
    assert torch.allclose(g.ns[ids], 0.5 * ns + 0.5 * nf)
    print("PASS buckets: slow + fast = all evidence; decay per observed view only")


def test_normalized_score_is_scale_free():
    """Under NO change the normalised score has mean ~1 at any noise level, fast memory and
    history length (the no-change expectation includes the shared-view term), while the
    raw c grows with the noise; the bucket weight recursions equal explicit sums."""
    from usplat.vmf_field import ViewStats
    torch.manual_seed(0)
    N, d = 2000, 16
    mu = Fn.normalize(torch.randn(N, d), dim=-1)
    raw_by_noise = {}
    for beta in (0.5, 0.8, 0.9):
        for V in (10, 30):
            for noise in (0.3, 0.9):
                f = DirectionalSemanticField(N, d, FieldConfig(
                    view_mass_frac=0.01, fast_memory=beta, change_score="normalized",
                    tau_min=1.0, slow_min=0.5, fast_min=0.5, change_persist=1, centre=True))
                g = DirectionalSemanticField(N, d, FieldConfig(
                    view_mass_frac=0.01, fast_memory=beta, tau_min=1.0, slow_min=0.5,
                    fast_min=0.5, change_persist=1, centre=True))
                for fld in (f, g):
                    fld.set_centre(torch.zeros(d))
                ws = []
                for v in range(V):
                    w = 0.3 + 0.7 * torch.rand(N)              # partial views
                    ws.append(w)
                    x = Fn.normalize(mu + noise * torch.randn(N, d), dim=-1)
                    m = 100.0 * w                              # ref = 1 -> view weight w
                    st = ViewStats(x * m[:, None], m, x * m[:, None], m, None, None, 100.0)
                    for fld in (f, g):
                        fld._ingest(st, None)
                mean_n = f.c_last.mean().item()
                assert 0.85 < mean_n < 1.15, (beta, V, noise, mean_n)
                raw_by_noise.setdefault((beta, V), []).append(g.c_last.mean().item())
                Wt = torch.stack(ws)                           # explicit fast weights
                keep = beta ** Wt
                bw = torch.stack([Wt[i] * torch.prod(keep[i + 1:], 0) for i in range(V)])
                assert torch.allclose(f.fB2, (bw * bw).sum(0), rtol=1e-4)
                assert torch.allclose(f.fC, (Wt * bw).sum(0), rtol=1e-4)
                assert torch.allclose(f.ns, (Wt - bw).sum(0), rtol=1e-4, atol=1e-4)
    for key, (lo, hi) in raw_by_noise.items():
        assert hi > 2.0 * lo, (key, lo, hi)
    print("PASS normalized: no-change mean in [0.85, 1.15] for beta 0.5/0.8/0.9, 10/30 views, "
          "2 noise levels; raw c grows > 2x with noise; recursions exact")


def test_calibrated_threshold():
    """Map a static sequence (threshold inf: scores recorded), calibrate, then change half of
    the Gaussians. Pooled mode: thr is the q-quantile of ALL per-view mapping scores (a
    per-view false-alarm rate); max mode: the q-quantile of per-Gaussian maxima, which the
    short early histories inflate. Defaults (beta 0, k 2)."""
    torch.manual_seed(0)
    N, d = 2000, 16
    mu = Fn.normalize(torch.randn(N, d), dim=-1)
    new = torch.randn(N, d)
    new = Fn.normalize(new - (new * mu).sum(-1, keepdim=True) * mu, dim=-1)
    out = {}
    for mode in ("history", "pooled", "max"):
        torch.manual_seed(1)
        f = DirectionalSemanticField(N, d, FieldConfig(view_mass_frac=0.01, readmit_depth=None,
                                                       calib_mode=mode))
        noisy = lambda m: Fn.normalize(m + 0.3 * torch.randn(N, d), dim=-1)
        ids = torch.arange(N)
        scores = []
        for _ in range(20):
            _view(f, ids, noisy(mu))
            ok = torch.isfinite(f.c_ring).all(1)
            scores.append(f.c_ring.min(1).values[ok])
        thr = f.calibrate_change_threshold(0.99)
        if mode in ("history", "pooled"):
            ref = torch.quantile(torch.cat(scores).double(), 0.99).item()
            assert abs(thr - ref) < 2e-3, (thr, ref)            # histogram bin = 5e-4
            assert (f.thr_by_hist is None) == (mode == "pooled")
        changed = torch.arange(N) < N // 2
        for _ in range(20):
            _view(f, ids, noisy(torch.where(changed[:, None], new, mu)))
        fired = f.n_resets >= 1
        out[mode] = (thr, fired[changed].float().mean().item(), fired[~changed].float().mean().item())
    thr, tp, fp = out["history"]
    assert tp > 0.9 and fp < 0.3, out
    assert out["pooled"][1] > 0.9 and out["pooled"][2] < 0.3, out
    assert out["max"][0] > thr                                   # max-based null is higher
    print(f"PASS calibrated (pooled): thr {thr:.3f} = q99 of per-view mapping scores; flagged "
          f"{tp:.2f} changed / {fp:.2f} static; max-mode thr {out['max'][0]:.3f} "
          f"(flags {out['max'][1]:.2f} / {out['max'][2]:.2f})")


def test_default_config_detects_after_calibration():
    """FieldConfig() unmodified: map (threshold inf, nothing fires), calibrate, update with
    a clean orthogonal switch of half the Gaussians. No depth stream: re-admission falls
    back to views (with a warning)."""
    import warnings
    torch.manual_seed(0)
    N, d = 1000, 16
    mu = Fn.normalize(torch.randn(N, d), dim=-1)
    new = torch.randn(N, d)
    new = Fn.normalize(new - (new * mu).sum(-1, keepdim=True) * mu, dim=-1)
    f = DirectionalSemanticField(N, d)                          # all defaults
    f.set_centre(torch.zeros(d))
    noisy = lambda m: Fn.normalize(m + 0.3 * torch.randn(N, d), dim=-1)
    ids = torch.arange(N)
    for _ in range(15):
        _view(f, ids, noisy(mu))
    assert f.n_resets.sum() == 0                               # uncalibrated: record only
    f.calibrate_change_threshold()
    changed = torch.arange(N) < N // 2
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        for _ in range(15):
            _view(f, ids, noisy(torch.where(changed[:, None], new, mu)))
    fired = f.n_resets >= 1
    tp, fp = fired[changed].float().mean().item(), fired[~changed].float().mean().item()
    assert tp > 0.9 and fp < 0.05, (tp, fp)                    # q = 0.999 per view
    assert any("no depth" in str(w.message) for w in wl)
    readmitted = (~f.changed[changed]).float().mean().item()  # re-admitted on views; a few
    assert readmitted > 0.9, readmitted                        # may re-fire (1%/view null)
    print(f"PASS defaults + calibration: flagged {tp:.2f} changed / {fp:.2f} static; "
          f"{readmitted:.2f} re-admitted on views without a depth stream")


def test_readmit_timing_and_partition():
    field, A, B, G, gdepth = _setup(readmit_depth=None)
    allid = torch.arange(field.N)
    for _ in range(8):
        _view(field, allid, A)
    sw = G["swapped"]
    first = None
    for t in range(8):
        _view(field, sw, B[sw])
        fired = field.n_resets[sw] >= 1
        if first is None and fired.all():
            first = t
            assert field.changed[sw].all()                     # flagged right after firing
            assert torch.allclose(field.Rs[sw] + field.Rf[sw], field.cen.R[sw], atol=1e-5)
        if first is not None and t - first < 2:                # < readmit_views new views
            assert field.changed[sw].all(), t
    assert first is not None and not field.changed[sw].any()
    print(f"PASS re-admission timing: flagged at view {first + 1}, held for < 3 views, then cleared")


def test_depth_stream_fails_closed_and_nan_safe():
    from usplat.mock_rasterizer import MockFeatureRasterizer
    torch.manual_seed(0)
    # (a) depth stream exists but a changed Gaussian got no depth since its reset -> stays
    field, A, B, G, gdepth = _setup()
    allid = torch.arange(field.N)
    for _ in range(8):
        _view(field, allid, A, torch.full((field.N,), 2.0), gdepth)
    sw = G["swapped"]
    for _ in range(6):
        _view(field, sw, B[sw])                                # no depth for these views
    assert (field.n_resets[sw] >= 1).all() and field.changed[sw].all()
    # (b) NaN / inf depth pixels never reach the statistics
    N, d, H, W = 30, 8, 16, 16
    ras = MockFeatureRasterizer(torch.rand(N, 2) * 16, 1.5 + torch.rand(N), 0.8 * torch.ones(N),
                                torch.rand(N), torch.stack(torch.meshgrid(
                                    torch.arange(W), torch.arange(H), indexing="xy"),
                                    -1).reshape(-1, 2).float())
    f = DirectionalSemanticField(N, d)
    f.set_centre(torch.zeros(d))
    dep = torch.full((H * W,), 2.0)
    dep[::7] = float("nan"); dep[::11] = float("inf")
    f.update_view(ras, Fn.normalize(torch.randn(H * W, d), dim=-1), depth_obs=dep,
                  gauss_depth=torch.full((N,), 2.0))
    assert torch.isfinite(f.depth_abs).all() and torch.isfinite(f.depth_w).all()
    print("PASS depth: fails closed without depth evidence after a reset; NaN/inf pixels ignored")


def test_change_test_on_raw_channel():
    field, A, B, G, gdepth = _setup(centre=False, readmit_depth=None)
    allid = torch.arange(field.N)
    for _ in range(8):
        _view(field, allid, A)
    sw = G["swapped"]
    for _ in range(4):
        _view(field, sw, B[sw])
    assert (field.n_resets[sw] >= 1).float().mean() > 0.9
    print("PASS centre=False: the change test runs on the raw channel")


def test_persistence_and_residual_special_case():
    """k = 2 ignores a one-view spike and fires on a persistent change; with fast_memory = 0
    the score is exactly the classic feature residual min_{last 2} (1 - cos(mean of the
    previous views, current view))."""
    torch.manual_seed(0)
    N, d = 3, 16
    A = Fn.normalize(torch.randn(N, d), dim=-1)
    B = torch.randn(N, d); B = Fn.normalize(B - (B * A).sum(-1, keepdim=True) * A, dim=-1)
    seq = [A] * 6 + [B] + [A] * 3 + [B] * 3                # spike at view 6, change at 10
    f = DirectionalSemanticField(N, d, FieldConfig(view_mass_frac=0.01, fast_memory=0.0,
                                                   change_persist=2, change_thr=0.5,
                                                   tau_min=2.0, readmit_depth=None))
    fired_at, scores = None, []
    for v, x in enumerate(seq):
        _view(f, torch.arange(N), x)
        scores.append(f.c_last.clone())
        if fired_at is None and bool((f.n_resets >= 1).all()):
            fired_at = v
    assert fired_at == 11, fired_at                        # not at the spike (6), 2nd B view
    # residual reference on the same sequence (before any reset)
    res = []
    for v in range(1, 11):
        prev = torch.stack(seq[:v]).mean(0)
        res.append(1 - Fn.cosine_similarity(prev, seq[v], dim=-1))
    ref = torch.minimum(res[-1], res[-2])                  # views 9 and 10
    assert torch.allclose(scores[10], ref, atol=1e-5), (scores[10], ref)
    print(f"PASS persistence: k=2 skips the spike, fires at view {fired_at}; beta=0 score = "
          f"feature residual ({scores[10][0]:.4f})")


def _stats(U, m, centre):
    """ViewStats of one view: every Gaussian n sees the unit feature U[n] with pixel mass m[n]
    (ref = 100: view weight min(1, m/100))."""
    from usplat.vmf_field import ViewStats
    R = U * m[:, None]
    return ViewStats(R, m, R if centre else None, m if centre else None, None, None, 100.0)


def test_slow_decay_uses_stream_view_weight():
    """slow_decay < 1 decays the older bucket by lambda^w per view, w = the view weight of the
    change stream. Regression: the decay read `wc`, the centred channel's EVIDENCE weight, so
    centre=False raised NameError and evidence="pixels" decayed by lambda^(pixel mass).
    Brute force with fast_memory = 0 (view i leaks fully into S at view i + 1, then decays):
        S_t = sum_{i<t} w_i u_i prod_{j=i+2..t} lambda^{w_j},  F_t = w_t u_t."""
    torch.manual_seed(0)
    N, d, V, lam = 6, 8, 7, 0.6
    Wt = 0.2 + 0.8 * torch.rand(V, N)                      # partial view weights
    U = Fn.normalize(torch.randn(V, N, d), dim=-1)
    for kw in (dict(centre=False), dict(centre=True, evidence="pixels")):
        f = DirectionalSemanticField(N, d, FieldConfig(slow_decay=lam, fast_memory=0.0,
                                                       readmit_depth=None, **kw))
        if f.cfg.centre:
            f.set_centre(torch.zeros(d))
        for t in range(V):
            f._ingest(_stats(U[t], 100.0 * Wt[t], f.cfg.centre), None)
            Rs, ns = torch.zeros(N, d), torch.zeros(N)
            for i in range(t):
                c = torch.prod(lam ** Wt[i + 2:t + 1], dim=0)  # empty product = 1
                Rs += (c * Wt[i])[:, None] * U[i]
                ns += c * Wt[i]
            assert torch.allclose(f.ns, ns, atol=1e-5), (kw, t)
            assert torch.allclose(f.Rs, Rs, atol=1e-5), (kw, t)
            assert torch.allclose(f.nf, Wt[t], atol=1e-6), (kw, t)
        if kw.get("evidence") == "pixels":                 # the posterior still counts pixels
            assert torch.allclose(f.raw.W1, 100.0 * Wt.sum(0), rtol=1e-5)
    print("PASS slow_decay: S decays by lambda^w (stream view weight), centre on or off")


def _brute_tests(views, thr, k=2, tau=3, edges=(4.0, 6.0, 10.0, 20.0, 40.0)):
    """Brute-force change test for full-weight views and fast_memory = 0. views: list of
    (ids, U (n, d)); thr(v, bucket) -> threshold at view v. Per Gaussian: the views since its
    last reset; c = 1 - cos(sum of the earlier ones, current) is defined from the tau-th view
    on; the test uses the min of the last k defined c, in the history bucket of ns = views - 1;
    a fire restarts the history at the current view and clears the ring.
    returns per view (tested ids, fired ids, tested buckets, fired buckets)."""
    hist, ring, out = {}, {}, []
    for v, (ids, U) in enumerate(views):
        tested, fired, tb, fb = [], [], [], []
        for n, u in zip(ids.tolist(), U.double()):
            h = hist.setdefault(n, [])
            h.append(u)
            if len(h) < tau:
                continue
            c = 1 - Fn.cosine_similarity(torch.stack(h[:-1]).sum(0), u, dim=0, eps=1e-8)
            r = ring.setdefault(n, [])
            r.append(float(c))
            del r[:-k]
            if len(r) < k:
                continue
            b = int(torch.bucketize(torch.tensor(float(len(h) - 1)), torch.tensor(edges)))
            tested.append(n); tb.append(b)
            if min(r) > thr(v, b):
                fired.append(n); fb.append(b)
                hist[n], ring[n] = [u], []
        out.append((tested, fired, tb, fb))
    return out


def test_test_counters_and_state():
    """last_tested / last_fired per view and n_tests / n_fires per history bucket equal a
    brute-force change test (mapping pass with threshold inf, then a calibrated per-history
    threshold loose enough to fire); the mapping-pass test count equals the null histogram;
    the counters survive state_dict / from_state, and a state saved before them loads with
    zeros."""
    torch.manual_seed(0)
    N, d, V, half = 300, 8, 30, 15
    mu = Fn.normalize(torch.randn(N, d), dim=-1)
    views = []
    for v in range(V):
        ids = torch.nonzero(torch.rand(N) < 0.7).squeeze(1)       # partial coverage per view
        views.append((ids, Fn.normalize(mu[ids] + 0.4 * torch.randn(len(ids), d), dim=-1)))
    f = DirectionalSemanticField(N, d, FieldConfig(view_mass_frac=0.01, readmit_depth=None))
    got = []
    for v, (ids, U) in enumerate(views):
        if v == half:
            assert torch.equal(f.n_tests.double(), f.c_hist.sum(1))   # mapping: tests = null
            assert int(f.n_fires.sum()) == 0
            f.calibrate_change_threshold(0.9)              # a loose threshold: fires happen
        _view(f, ids, U)
        got.append((f.last_tested.clone(), f.last_fired.clone()))
    thr = f.thr_by_hist.tolist()
    ref = _brute_tests(views, lambda v, b: float("inf") if v < half else thr[b])
    nt, nf = torch.zeros(6, dtype=torch.long), torch.zeros(6, dtype=torch.long)
    for v, ((tested, fired), (t_ids, f_ids, tb, fb)) in enumerate(zip(got, ref)):
        assert torch.nonzero(tested).squeeze(1).tolist() == sorted(t_ids), v
        assert torch.nonzero(fired).squeeze(1).tolist() == sorted(f_ids), v
        nt += torch.bincount(torch.tensor(tb, dtype=torch.long), minlength=6)
        nf += torch.bincount(torch.tensor(fb, dtype=torch.long), minlength=6)
    assert torch.equal(f.n_tests, nt) and torch.equal(f.n_fires, nf)
    assert int(f.n_fires.sum()) == int(f.n_resets.sum()) > 0
    s = f.state_dict()
    g = DirectionalSemanticField.from_state(s)
    for name in ("last_tested", "last_fired", "n_tests", "n_fires"):
        assert torch.equal(getattr(f, name), getattr(g, name)), name
    old = {k: v for k, v in s.items() if k not in ("last_tested", "last_fired", "n_tests",
                                                    "n_fires", "m_max")}
    h = DirectionalSemanticField.from_state(old)
    assert int(h.n_tests.sum()) == 0 and not h.last_tested.any()
    assert torch.equal(h.raw.R, f.raw.R)
    print(f"PASS counters = brute force: {int(f.n_tests.sum())} tests, "
          f"{int(f.n_fires.sum())} fires; round trip and old states")

def test_defaults_match_base_commit(tmp_path):
    """FieldConfig() is unchanged by the test record and the opt-in change weight: the field at
    git 3eb0e86 (Table I v1) and the current one, fed the same mock sequence with a content
    change, observed depth and a calibrated threshold (resets and re-admissions happen), agree
    bit for bit in every state tensor they share."""
    import importlib
    import os
    import subprocess
    import sys
    import pytest
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg = tmp_path / "usplat_v1"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for name in ("vmf_field.py", "bki_accumulate.py", "vmf_math.py"):
        r = subprocess.run(["git", "-C", repo, "show", f"3eb0e86:usplat/{name}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip("git history with 3eb0e86 not available")
        (pkg / name).write_text(r.stdout)
    sys.path.insert(0, str(tmp_path))
    try:
        old_mod = importlib.import_module("usplat_v1.vmf_field")
    finally:
        sys.path.remove(str(tmp_path))
    from usplat.mock_rasterizer import MockFeatureRasterizer
    from usplat.clip_dense import MockDenseCLIP
    torch.manual_seed(5)
    N, d, H = 120, 16, 32
    grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(H), indexing="xy"),
                       -1).reshape(-1, 2).float()
    means2d = torch.rand(N, 2) * H
    ras = MockFeatureRasterizer(means2d, 0.5 + 2.5 * torch.rand(N), 0.3 + 0.6 * torch.rand(N),
                                torch.rand(N), grid)
    clip = MockDenseCLIP(feat_dim=d, num_protos=5, noise=0.2, seed=5)
    proto = (means2d[:, 0] / H * 5).long().clamp(0, 4)
    moved = torch.where(torch.arange(N) % 3 == 0, (proto + 1) % 5, proto)
    dom = ras.weights().argmax(1)
    W = ras.weights()
    gdepth = 1.0 + torch.rand(N)
    ren = W @ gdepth / W.sum(1).clamp_min(1e-12)
    views = [(clip.extract_from_labels((proto if v < 10 else moved)[dom].reshape(H, H))
              .reshape(-1, d), ren + 0.01 * torch.randn(H * H)) for v in range(20)]
    old = old_mod.DirectionalSemanticField(N, d, old_mod.FieldConfig(view_mass_frac=1e-3))
    new = DirectionalSemanticField(N, d, FieldConfig(view_mass_frac=1e-3))
    for f in (old, new):
        for v, (z, dep) in enumerate(views):
            if v == 10:
                f.calibrate_change_threshold(0.99)
            f.update_view(ras, z, depth_obs=dep, gauss_depth=gdepth)
    assert int(new.n_resets.sum()) > 0 and int(new.n_fires.sum()) == int(new.n_resets.sum())
    so, sn = old.state_dict(), new.state_dict()
    shared = [k for k in so if isinstance(so[k], torch.Tensor)]
    assert len(shared) > 20
    for k in shared:
        assert sn[k] is not None and torch.equal(so[k].nan_to_num(-7), sn[k].nan_to_num(-7)), k
    for k in so:
        if so[k] is None:
            assert sn[k] is None, k
    assert so["config"] == {k: v for k, v in sn["config"].items() if k != "change_weight"}
    print(f"PASS defaults = git 3eb0e86 bit for bit ({len(shared)} tensors, "
          f"{int(new.n_resets.sum())} resets)")


if __name__ == "__main__":
    test_slow_decay_uses_stream_view_weight()
    test_test_counters_and_state()
    test_change_reset_readmit()
    test_readmit_without_depth_uses_views()
    test_buckets_partition_and_decay()
    test_normalized_score_is_scale_free()
    test_calibrated_threshold()
    test_default_config_detects_after_calibration()
    test_readmit_timing_and_partition()
    test_depth_stream_fails_closed_and_nan_safe()
    test_change_test_on_raw_channel()
    test_persistence_and_residual_special_case()
