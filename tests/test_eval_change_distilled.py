"""scripts/eval_change_nor_distilled.py on the CPU toy scene of tests/test_eval_change.py:
* a "distilled" Feature 3DGS file whose features are the saved field's mu* reproduces the 'feat' row
  of scripts/eval_change_nor.py: the Gaussian level and FA / Rec exactly, the threshold to 1e-7 (it
  renders mu* alone, not with the colour and depth channels), the pixel
  level to 1e-6 (the pixel render composites another number of score channels, so its sums round
  differently); so does the control row, and its check against eval_change_nor.py's record;
* a LEGaussians file (codes + decoder) runs the render-then-decode path, flags = score > threshold;
* a file in another PCA basis or with another Gaussian count is refused."""
import importlib.util
import json
import pathlib

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TE = _load("test_eval_change_toy", REPO / "tests" / "test_eval_change.py")


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    from usplat import distill as DI
    from usplat.vmf_field import DirectionalSemanticField
    tmp = tmp_path_factory.mktemp("eval_change_distilled")
    ec, _, root, align, sp, *_ = TE.build(tmp)
    ec.load_extractor = lambda cfg, device: TE.Fake()
    base = ["--field", str(tmp / "field.pt"), "--backend", "mock", "--device", "cpu", "--plain-json",
            "--align", "none"]
    ec.main(base + ["--out", str(tmp / "ref.json")])
    ck = torch.load(tmp / "field.pt", map_location="cpu", weights_only=False)
    mu = DirectionalSemanticField.from_state(ck["field"]).mu_post().float()
    sha = DI.basis_sha256(ck["pca"])
    torch.save(dict(name="f3dgs", features=mu.clone(), u=None, decoder=None, pca_basis_sha256=sha,
                    identity="toy", config={}), tmp / "field_f3dgs.pt")
    g = torch.Generator().manual_seed(0)
    codebook = torch.nn.functional.normalize(torch.randn(16, mu.shape[1], generator=g), dim=1)
    dec = DI.LEGDecoder(codebook, dim=DI.LEGS_DIM, hidden=8)
    torch.save(dict(name="legs_l1", features=torch.randn(mu.shape[0], DI.LEGS_DIM, generator=g),
                    u=torch.zeros(mu.shape[0]), decoder=dec.state_dict(), decoder_hidden=8,
                    pca_basis_sha256=sha, identity="toy", config={}), tmp / "field_legs_l1.pt")
    ed = _load("eval_change_nor_distilled", REPO / "scripts" / "eval_change_nor_distilled.py")
    ed.ECN.load_extractor = lambda cfg, device: TE.Fake()
    files = f"{tmp / 'field_f3dgs.pt'},{tmp / 'field_legs_l1.pt'}"
    out = ed.main(base + ["--distilled", files, "--reference", str(tmp / "ref.json"),
                          "--out", str(tmp / "out.json")])
    return dict(ref=json.loads((tmp / "ref.json").read_text()), out=out, ed=ed, tmp=tmp, base=base,
                sha=sha, mu=mu)


def test_mu_star_as_distilled_features_is_the_feat_row(toy):
    ref, out = toy["ref"], toy["out"]
    s = out["summary"]
    for r in ("f3dgs_feat", "feat_control"):
        assert s[r]["g_ap"] == ref["gaussian"]["rows"]["feat"]["ap"]
        assert s[r]["g_auroc"] == ref["gaussian"]["rows"]["feat"]["auroc"]
        assert abs(s[r]["p_ap"] - ref["pixel"]["rows"]["feat"]["ap"]) <= 1e-6
        assert abs(s[r]["p_auroc"] - ref["pixel"]["rows"]["feat"]["auroc"]) <= 1e-6
        assert s[r]["fa"] == ref["operating_point"]["feat"]["fa"]
        assert s[r]["rec"] == ref["operating_point"]["feat"]["rec"]
        tol = 0.0 if r == "feat_control" else 1e-7   # f3dgs renders mu* alone (64, not 69 channels)
        assert abs(s[r]["threshold"] - ref["thresholds"]["rows"]["feat"]["pooled"]) <= tol
    cc = out["control_check"]
    assert cc["max_abs_diff"] <= 1e-6, cc
    assert all(cc[k]["exact"] for k in ("g_ap", "g_auroc", "fa", "rec", "threshold", "n_observed")), cc
    assert s["f3dgs_feat"]["n_observed"] == ref["counts"]["n_observed"]


def test_legs_runs_the_decoder_path(toy):
    s = toy["out"]["summary"]["legs_l1_feat"]
    assert 0.0 <= s["g_ap"] <= 1.0 and 0.0 <= s["fa"] <= 1.0 and s["threshold"] == s["threshold"]
    assert toy["out"]["inputs"]["legs_l1_feat"] == "F"


def test_refusals(toy):
    tmp, ed, base = toy["tmp"], toy["ed"], toy["base"]
    torch.save(dict(name="f3dgs", features=toy["mu"], decoder=None, pca_basis_sha256="other"),
               tmp / "bad_basis.pt")
    torch.save(dict(name="f3dgs", features=toy["mu"][:-1], decoder=None, pca_basis_sha256=toy["sha"]),
               tmp / "bad_n.pt")
    for bad in ("bad_basis.pt", "bad_n.pt"):
        with pytest.raises(SystemExit):
            ed.main(base + ["--distilled", str(tmp / bad), "--out", str(tmp / "x.json")])
