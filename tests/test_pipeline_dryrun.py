"""End-to-end smoke test on the CPU mock backend.

A scene where 25% of the Gaussians change semantic identity half-way through. Each pixel
shows the prototype of its DOMINANT Gaussian, so a "dynamic" Gaussian that is never the
visible surface produces no observable change. Ground truth is therefore per footprint:
  observably changed = more than half of the Gaussian's splat weight lies on pixels whose
  label changed; unchanged = less than 10% does. Only well-observed Gaussians count (a full
  view every frame); the mock scene is small and heavily overlapped.
(b) runs the pipeline's own flow with FieldConfig() defaults: the first half of the frames
is the mapping pass (scores recorded, nothing reset), then calibrate_change_threshold(),
then the changed second half. Checked: ranking (AUROC of c_max over the update), recall
and false alarms at the calibrated threshold, and the posterior after the reset.
 (a) dispersion only (change test off): the aleatoric channel is higher on observably
     changed Gaussians than on unchanged ones;
 (b) with the temporal change test: most observably changed Gaussians are flagged, few
     unchanged ones are, and the flagged Gaussians' posterior points at the NEW content.
"""
import torch
from usplat.vmf_field import DirectionalSemanticField, FieldConfig
from usplat.mock_rasterizer import MockFeatureRasterizer
from usplat.clip_dense import MockDenseCLIP


def _run(cfg, seed=1, frames=12, calibrate=False):
    torch.manual_seed(seed)
    N, d, protos = 120, 32, 6
    clip = MockDenseCLIP(feat_dim=d, num_protos=protos, noise=0.1, seed=seed)
    field = DirectionalSemanticField(N, d, cfg)

    means2d = torch.rand(N, 2) * 64
    sigmas = 1.0 + torch.rand(N) * 2.0
    opac = 0.3 + 0.6 * torch.rand(N)
    depths = torch.rand(N)
    true_proto = (means2d[:, 0] / 64.0 * protos).long().clamp(0, protos - 1)
    dynamic = torch.rand(N) < 0.25

    H = Wd = 40
    grid = torch.stack(torch.meshgrid(torch.arange(Wd), torch.arange(H), indexing="xy"), -1)
    pixels = grid.reshape(-1, 2).float()
    new_proto = true_proto.clone()
    new_proto[dynamic] = (new_proto[dynamic] + 1) % protos

    ras = MockFeatureRasterizer(means2d, sigmas, opac, depths, pixels)
    W = ras.weights()
    dom = W.argmax(1)
    if cfg.centre:
        field.set_centre(clip.extract_from_labels(true_proto[dom].reshape(H, Wd))
                         .reshape(-1, d).mean(0))           # mapping-frame mean
    for t in range(frames):
        if calibrate and t == frames // 2:
            field.calibrate_change_threshold()
        p = new_proto if t >= frames // 2 else true_proto
        z = clip.extract_from_labels(p[dom].reshape(H, Wd)).reshape(-1, d)
        field.update_view(ras, z)
    pix_changed = (true_proto[dom] != new_proto[dom]).float()
    frac = (W.t() @ pix_changed) / W.sum(0).clamp_min(1e-12)
    well = field.pixel_mass / frames >= field.cfg.view_mass_frac * H * Wd
    return field, frac, clip, new_proto, well


def test_dispersion_separates_dynamic():
    field, frac, _, _, well = _run(FieldConfig(change_test=False))
    ale, valid = field.aleatoric()
    dyn, sta = (frac > 0.5) & valid & well, (frac < 0.1) & valid & well
    assert field.n_eff.sum() > 0
    assert dyn.any() and sta.any()
    assert ale[dyn].mean() > ale[sta].mean(), (ale[dyn].mean(), ale[sta].mean())
    print(f"PASS dispersion: aleatoric dynamic={ale[dyn].mean():.3f} > static={ale[sta].mean():.3f}")


def test_change_test_flags_dynamic():
    from usplat.evaluation import auroc
    field, frac, clip, new_proto, well = _run(FieldConfig(), calibrate=True)
    dyn, sta = (frac > 0.5) & well, (frac < 0.1) & well
    assert dyn.sum() >= 5 and sta.sum() >= 5, (int(dyn.sum()), int(sta.sum()))
    fired = field.n_resets >= 1
    sel = dyn | sta
    auc = auroc(field.c_max[sel], dyn[sel].float())
    false_alarm = fired[sta].float().mean().item()
    recall = fired[dyn].float().mean().item()
    assert auc > 0.8, auc
    assert recall > 0.8, recall
    assert false_alarm < 0.2, false_alarm
    # after the reset the posterior follows the new content
    flagged = (field.n_resets >= 1) & dyn
    cos_new = (field.mu_post()[flagged] * clip.protos[new_proto[flagged]]).sum(-1)
    assert flagged.any() and cos_new.mean() > 0.8, cos_new.mean()
    print(f"PASS change test: AUROC(c_max) {auc:.3f}; at the calibrated threshold recall "
          f"{recall:.2f}, false alarms {false_alarm:.2f}; cos(mu*, new content) {cos_new.mean():.3f}")


if __name__ == "__main__":
    test_dispersion_separates_dynamic()
    test_change_test_flags_dynamic()
