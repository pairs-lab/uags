"""End-to-end smoke test on the CPU mock backend."""
import torch
from usplat.vmf_field import DirectionalSemanticField
from usplat.mock_rasterizer import MockFeatureRasterizer
from usplat.clip_dense import MockDenseCLIP


def test_pipeline_runs_and_separates_dynamic():
    torch.manual_seed(1)
    N, d, protos = 120, 32, 6
    clip = MockDenseCLIP(feat_dim=d, num_protos=protos, noise=0.1, seed=1)
    field = DirectionalSemanticField(N, d, kappa_lik=10.0)

    means2d = torch.rand(N, 2) * 64
    sigmas = 1.0 + torch.rand(N) * 2.0
    opac = 0.3 + 0.6 * torch.rand(N)
    depths = torch.rand(N)
    true_proto = (means2d[:, 0] / 64.0 * protos).long().clamp(0, protos - 1)
    dynamic = torch.rand(N) < 0.25

    H = Wd = 40
    grid = torch.stack(torch.meshgrid(torch.arange(Wd), torch.arange(H), indexing="xy"), -1)
    pixels = grid.reshape(-1, 2).float()

    for t in range(12):
        ras = MockFeatureRasterizer(means2d, sigmas, opac, depths, pixels)
        dom = ras.weights().argmax(1)
        p = true_proto.clone()
        if t > 6:
            p[dynamic] = (p[dynamic] + 1) % protos
        z = clip.extract_from_labels(p[dom].reshape(H, Wd)).reshape(-1, d)
        field.update_from_rasterizer(ras, z)

    ale, valid = field.aleatoric(min_obs=5.0)
    dyn = dynamic & valid
    sta = (~dynamic) & valid
    assert field.n_eff.sum() > 0
    assert dyn.any() and sta.any()
    # dynamic regions must, on average, carry higher aleatoric uncertainty
    assert ale[dyn].mean() > ale[sta].mean(), (ale[dyn].mean(), ale[sta].mean())
    print(f"PASS: aleatoric dynamic={ale[dyn].mean():.3f} > static={ale[sta].mean():.3f}")


if __name__ == "__main__":
    test_pipeline_runs_and_separates_dynamic()
