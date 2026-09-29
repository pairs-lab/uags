"""usplat.ablation_rows: read-out variants share the field's statistics without changing the
field; re-mapped variants are read from <main>_<variant>.pt and skipped when missing."""
import warnings
import torch
import torch.nn.functional as Fn
from usplat.vmf_field import DirectionalSemanticField, FieldConfig
from usplat import ablation_rows as AR


def _field(cfg=None, seed=0, N=40, d=8):
    g = torch.Generator().manual_seed(seed)
    f = DirectionalSemanticField(N, d, cfg or FieldConfig(change_test=False))
    base = Fn.normalize(torch.randn(N, d, generator=g), dim=-1)
    for v in range(6):
        noise = 0.3 * (1 + torch.arange(N) % 3)[:, None] * torch.randn(N, d, generator=g)
        z = Fn.normalize(base + noise, dim=-1)
        n = N if v < 2 else N // 2                      # half the Gaussians get more views
        f.update_sparse(torch.arange(n), torch.full((n,), 5.0), z[:n], num_pixels=100)
    return f


def test_readout_rows_share_statistics(tmp_path):
    f = _field()
    before = (f.cfg.estimator, f.cfg.epistemic, f.cfg.kappa_mode)
    main = str(tmp_path / "scene_lseg.pt")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rows = AR.row_factory(dict(config=dict(out=main)), f, "cpu")
    assert sum("no field at" in str(x.message) for x in w) == 3     # pixels, centre, legacy
    names = [r.name for r in rows]
    for v in AR.READOUT:
        assert f"abl_{v}_P0" in names and f"abl_{v}_P2" in names
    assert (f.cfg.estimator, f.cfg.epistemic, f.cfg.kappa_mode) == before   # main untouched
    by = {r.name: r for r in rows}
    ours_ale, ours_epi = f.aleatoric()[0], f.epistemic()
    joint = by["abl_est_joint_P0"].uncertainties()["u_ale"]
    ok = ~torch.isnan(joint)                           # undefined below tau_min views
    assert torch.equal(ok, ~torch.isnan(ours_ale)) and 0 < int(ok.sum()) < f.N
    assert torch.allclose(joint[ok], 1 - f.raw.agreement("joint", f.cfg.c0)[ok])
    assert not torch.allclose(joint, ours_ale, equal_nan=True)
    rbar = by["abl_est_rbar_P0"].uncertainties()["u_ale"]
    assert torch.allclose(rbar[ok], 1 - f.raw.rbar()[ok])
    assert torch.allclose(by["abl_est_joint_P0"].uncertainties()["u_epi"], ours_epi)
    # the copies share the statistics: an update of the field is seen by every read-out
    assert by["abl_est_joint_P0"].field.raw is f.raw
    assert torch.equal(by["abl_est_joint_P0"].buffer(), by["abl_est_rbar_P0"].buffer())
    assert len(rows) == 6
    print(f"PASS read-out ablations: {len(rows)} rows, main config unchanged")


def test_remapped_variant_is_loaded(tmp_path):
    f = _field()
    main = str(tmp_path / "scene_lseg.pt")
    pix = _field(FieldConfig(change_test=False, evidence="pixels", epistemic="pixels"))
    torch.save(dict(field=pix.state_dict()), AR.variant_path(main, "pixels"))
    rows = AR.row_factory(dict(config=dict(out=main)), f, "cpu", variants=["pixels"])
    assert [r.name for r in rows] == ["abl_pixels_P0", "abl_pixels_P2"]
    u = rows[0].uncertainties()["u_epi"]
    assert torch.allclose(u, pix.epistemic()) and not torch.allclose(u, f.epistemic())
    print("PASS re-mapped ablation variant loaded from", AR.variant_path(main, "pixels"))
