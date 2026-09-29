"""FieldConfig.legacy() reproduces the field at git ab28fb4 (the ablation baseline).

The original vmf_field.py / bki_accumulate.py / consolidation.py are read from git
(`git show ab28fb4:<path>`) into a temporary package and fed the same mock sequence
(a quarter of the Gaussians change content half-way). n_eff, R, mu*, kappa*, epistemic,
aleatoric (+ validity), r_bar and one legacy forgetting step must agree.
"""
import importlib
import os
import subprocess
import sys
import tempfile
import torch
from usplat.vmf_field import DirectionalSemanticField, FieldConfig
from usplat.consolidation import Consolidator
from usplat.mock_rasterizer import MockFeatureRasterizer
from usplat.clip_dense import MockDenseCLIP

BASE = "ab28fb4"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _head_package():
    tmp = tempfile.mkdtemp()
    pkg = os.path.join(tmp, "usplat_head")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    for name in ("vmf_field.py", "bki_accumulate.py", "consolidation.py"):
        src = subprocess.run(["git", "-C", REPO, "show", f"{BASE}:usplat/{name}"],
                             capture_output=True, text=True, check=True).stdout
        open(os.path.join(pkg, name), "w").write(src)
    sys.path.insert(0, tmp)
    return (importlib.import_module("usplat_head.vmf_field"),
            importlib.import_module("usplat_head.consolidation"))


def test_legacy_matches_head():
    old_field_mod, old_cons_mod = _head_package()
    torch.manual_seed(3)
    N, d, protos = 80, 16, 5
    clip = MockDenseCLIP(feat_dim=d, num_protos=protos, noise=0.2, seed=3)
    means2d = torch.rand(N, 2) * 32
    ras = MockFeatureRasterizer(means2d, 1.0 + 2 * torch.rand(N), 0.3 + 0.6 * torch.rand(N),
                                torch.rand(N), torch.stack(torch.meshgrid(
                                    torch.arange(32), torch.arange(32), indexing="xy"),
                                    -1).reshape(-1, 2).float())
    dom = ras.weights().argmax(1)
    proto = (means2d[:, 0] / 32 * protos).long().clamp(0, protos - 1)
    moved = torch.rand(N) < 0.25
    old = old_field_mod.DirectionalSemanticField(N, d, kappa_lik=10.0)
    new = DirectionalSemanticField(N, d, FieldConfig.legacy())
    for t in range(10):
        p = torch.where(moved & (t >= 5), (proto + 1) % protos, proto)
        z = clip.extract_from_labels(p[dom].reshape(32, 32)).reshape(-1, d)
        old.update_from_rasterizer(ras, z)
        new.update_view(ras, z)
    close = lambda a, b: torch.allclose(a, b, atol=1e-5, rtol=1e-5, equal_nan=True)
    assert close(new.n_eff, old.n_eff)
    assert close(new.raw.R, old.R)
    assert close(new.mu_post(), old.mu_post())
    assert close(new.kappa_post(), old.kappa_post())
    assert close(new.epistemic(), old.epistemic())
    assert close(new.mean_resultant_length(), old.mean_resultant_length())
    a_new, v_new = new.aleatoric()
    a_old, v_old = old.aleatoric(min_obs=5.0)
    assert torch.equal(v_new, v_old) and close(a_new, a_old)
    # one legacy forgetting step, with the old default thresholds
    m_old = old_cons_mod.Consolidator(old).step()
    m_new = Consolidator(new, legacy=True).step()
    assert torch.equal(m_old["unstable"], m_new["unstable"])
    assert torch.equal(m_old["stable"], m_new["stable"])
    assert close(new.raw.R, old.R) and close(new.n_eff, old.n_eff)
    print(f"PASS legacy preset == git {BASE}: n_eff, R, mu*, kappa*, epistemic, aleatoric "
          f"({int(v_old.sum())} valid), r_bar, consolidation step")


if __name__ == "__main__":
    test_legacy_matches_head()
