"""Fusion with the current view (usplat/view_fusion.py, scripts/eval_fusion.py,
scripts/fusion_select.py, scripts/fusion_tab1.sh, configs/fusion_prereg.json). CPU only.

 * the code's forms are the prereg's (names, bases, features, order) and its choices resolve;
 * endpoints are exact: w = 0 gives the map row's per-pixel outputs (seg_eval.evaluate_view)
   bit for bit on covered pixels and the 2D row's on uncovered pixels (P2's NaN beta there is
   never read); w = 1 gives the 2D row's everywhere; for the bases P0, P2 and averaging;
 * fit_weights recovers known coefficients (UG form and a constant) on a toy drawn from the
   model, and its NLL is not above either endpoint's;
 * eval_fusion.py end to end on the mock backend (toy scene with a hole, the NormLift and
   view-count sums, a fake 2D extractor): the single-source betas EQUAL eval_seg.py's own fit
   (same order and seed), the reference rows' metrics equal eval_seg.py's, the map rows'
   map_covered variant equals eval_seg's covered variant, a --betas run reproduces the
   validation evaluation, theta = -1e300 / +1e300 make the fused rows equal the map rows (on
   map-covered pixels) and the 2D row (all pixels) in every metric, and the refusals hold;
 * fusion_select's validation gate and claim family on synthetic records;
 * the driver end to end in a scratch git repository (mock backend): its gates, val ->
   selection (with the Table I reproduction check) -> test once -> table + claim.
"""
import copy
import importlib.util
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from usplat import view_fusion as vf
from usplat.rows import TensorRow
from usplat.seg_eval import evaluate_view, render_row
from usplat.vmf_field import DirectionalSemanticField, FieldConfig
from usplat.baselines import default_baselines, baselines_state

REPO = pathlib.Path(__file__).resolve().parents[1]
PREREG = REPO / "configs" / "fusion_prereg.json"
TOY_EXTRACTOR = '''"""Deterministic fake 2D extractor (tests/test_view_fusion.py only): chair (e0) tokens
on the left half, table (e1) on the right, the bottom-left quarter wrong (table), fixed noise."""
import torch


class ToyTokens:
    d = 16

    def __init__(self, device="cpu"):
        self.device = device

    def extract_tokens(self, image):
        H, W = int(image.shape[0]), int(image.shape[1])
        gh, gw = H // 16, W // 16
        tok = torch.zeros(gh, gw, 16)
        tok[:, : gw // 2, 0] = 1.0
        tok[:, gw // 2:, 1] = 1.0
        tok[gh // 2:, : gw // 4, 0] = 0.0
        tok[gh // 2:, : gw // 4, 1] = 1.0
        g = torch.Generator().manual_seed(0)
        return tok + 0.6 * torch.randn(gh, gw, 16, generator=g)

    def text_embedding(self, prompts, templates=("a photo of a {}.",)):
        raise RuntimeError("toy extractor: the texts come from the text cache")


def build(device="cpu", **kw):
    return ToyTokens(device)
'''


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy_extractor():
    ns = {}
    exec(TOY_EXTRACTOR, ns)
    return ns["ToyTokens"]()


# ------------------------------------------------------------------------------ prereg
def test_forms_match_prereg():
    p = json.loads(PREREG.read_text())
    assert p["forms"] == [f.spec() for f in vf.FORMS]
    names = [f.name for f in vf.FORMS]
    assert p["choices"]["@ug"]["rows"] == ["fuse_p0_ug", "fuse_p2_ug"]
    for ug, c in p["paired_const"].items():
        assert vf.FORMS_BY_NAME[ug].base == vf.FORMS_BY_NAME[c].base and not vf.FORMS_BY_NAME[c].features
    for b in p["claim"]["baselines"]:
        assert b == "@paired_const" or b in names
    assert set(p["rows"]["labels"]) == set(names)
    assert p["rows"]["reference"] == ["ours_P0", "ours_P2", "averaging", "2d_features"]
    assert p["pipeline"].startswith("scripts/fusion_tab1.sh")
    assert vf.FORMS_BY_NAME["fuse_p0_ug"].features == vf.FORMS_BY_NAME["fuse_p2_ug"].features
    print(f"PASS forms: {len(names)} fused rows = prereg; @ug {p['choices']['@ug']['rows']}")


# --------------------------------------------------------------------------- endpoints
class _Src:
    """2D source like eval_seg.PixelSource: z (P, d) and alpha = 1."""

    def __init__(self, z):
        self.z = z

    def __call__(self, buf):
        return torch.cat([self.z, torch.ones(self.z.shape[0], 1)], 1)


def _unit_view(seed=0):
    t = _load("test_seg_eval_vf", REPO / "tests" / "test_seg_eval.py")
    ras, lab, gt, text = t._scene(hole=(12, 12), seed=seed)
    g = torch.Generator().manual_seed(seed)
    N, K, d = lab.numel(), text.shape[0], text.shape[1]
    feat = text[lab] + 0.8 * torch.randn(N, d, generator=g)
    feat = feat / feat.norm(dim=1, keepdim=True)
    u_ale = torch.rand(N, generator=g)
    u_ale[torch.rand(N, generator=g) < 0.3] = float("nan")
    ours = TensorRow("ours_P0", "feature", buf=feat,
                     unc={"u_epi": torch.rand(N, generator=g), "u_ale": u_ale})
    ours2 = TensorRow("ours_P2", "feature", adaptive=True, buf=feat, unc=dict(ours.unc))
    afeat = text[lab] + 1.0 * torch.randn(N, d, generator=g)
    afeat = afeat / afeat.norm(dim=1, keepdim=True)
    avg = TensorRow("averaging", "feature", buf=afeat)
    avg_ch = TensorRow("averaging_channels", "feature", buf=afeat,
                       unc={"viewcount": torch.rand(N, generator=g),
                            "normlift_epi": torch.rand(N, generator=g),
                            "normlift_1-Cinter": torch.rand(N, generator=g)})
    gtf = gt.reshape(-1)
    z = text[gtf] + 1.2 * torch.randn(gtf.numel(), d, generator=g)
    z = z / z.norm(dim=1, keepdim=True)
    row2d = TensorRow("2d_features", "feature", buf=torch.zeros(1, d))
    betas = {"ours_P0": 9.0, "averaging": 7.0, "2d_features": 5.0,
             "ours_P2": dict(beta0=12.0, a=0.7, b=1.3, g=0.4)}
    return ras, gt, text, ours, ours2, avg, avg_ch, row2d, _Src(z), betas


def _same(a, b, keys=("pred", "conf", "entropy", "nll", "brier", "correct", "tied"), mask=None):
    for k in keys:
        x, y = a[k], b[k]
        if mask is not None:
            x, y = x[mask], y[mask]
        assert torch.equal(x, y), k


def test_endpoints_reproduce_rows_exactly():
    ras, gt, text, ours, ours2, avg, avg_ch, row2d, src, betas = _unit_view()
    ref = evaluate_view([ours, ours2, avg], ras, text, gt, betas, 0.5)
    ref.update(evaluate_view([row2d], src, text, gt, betas, 0.5))
    b = vf.pixel_bundle(render_row(ours, ras), render_row(avg_ch, ras), render_row(row2d, src),
                        text, gt, 0.5)
    cov = ref["ours_P0"]["covered"]
    assert torch.equal(cov, b.cov) and int((~cov).sum()) > 0 and int(cov.sum()) > 0
    hole = ~cov
    assert torch.isnan(b.chan["u_epi"][hole]).any()      # empty pixels: undefined channels
    base_row = {"ours_P0": "ours_P0", "ours_P2": "ours_P2", "averaging": "averaging"}
    for f in vf.FORMS:
        n = len(f.params())
        lo = [-1e300] + [0.0] * (n - 1)
        hi = [1e300] + [0.0] * (n - 1)
        w0 = vf.fused_outputs(f, lo, b, betas)
        w1 = vf.fused_outputs(f, hi, b, betas)
        _same(w0, ref[base_row[f.base]], mask=cov)             # the map row on covered pixels
        _same(w0, ref["2d_features"], mask=hole)               # the 2D row on uncovered pixels
        _same(w1, ref["2d_features"])                          # the 2D row everywhere
        assert torch.equal(w0["u"]["w2d"][cov], torch.zeros(int(cov.sum())))
        assert torch.equal(w0["u"]["w2d"][hole], torch.ones(int(hole.sum())))
        mid = vf.fused_outputs(f, [0.0] * n, b, betas)       # w = 1/2: differs from both
        assert not torch.equal(mid["nll"][cov], w0["nll"][cov])
        assert torch.equal(mid["nll"][hole], w1["nll"][hole])   # holes: always the 2D row
    # a non-finite weight feature on a covered pixel is refused
    bad = copy.copy(b)
    bad.chan = dict(b.chan, viewcount=b.chan["viewcount"].clone())
    bad.chan["viewcount"][torch.nonzero(cov)[0, 0]] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        vf.fused_outputs(vf.FORMS_BY_NAME["fuse_avg_vc"], [0.0, 0.0, 0.0], bad, betas)
    with pytest.raises(ValueError, match="theta has"):
        vf.fused_outputs(vf.FORMS_BY_NAME["fuse_p0_ug"], [0.0], b, betas)
    print(f"PASS endpoints: w=0 = map row on {int(cov.sum())} covered px and the 2D row on "
          f"{int(hole.sum())} uncovered px; w=1 = 2D row; 7 forms, bases P0/P2/averaging")


# --------------------------------------------------------------------------------- fit
def _synthetic_chunks(theta, n=60000, K=5, F=4, seed=0, parts=3):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(parts):
        m = n // parts
        Lm = 3.0 * torch.randn(m, K, generator=g, dtype=torch.float64)
        L2 = 3.0 * torch.randn(m, K, generator=g, dtype=torch.float64)
        X = torch.rand(m, F, generator=g, dtype=torch.float64)
        th = torch.tensor(theta, dtype=torch.float64)
        w = torch.sigmoid(th[0] + X @ th[1:])
        p = torch.softmax(vf.fuse(Lm, L2, w), 1)
        y = torch.multinomial(p, 1, generator=g).squeeze(1)
        out.append(vf.Chunk(Lm, L2, X, y))
    return out


def test_fit_recovers_known_coefficients():
    true = [0.5, -2.0, 1.5, 1.0, -1.0]
    chunks = _synthetic_chunks(true)
    th, info = vf.fit_weights(chunks, 4)
    err = max(abs(a - b) for a, b in zip(th, true))
    assert err < 0.2, (th, true)
    assert info["converged"] and info["nll"] <= min(info["nll_map"], info["nll_2d"])
    assert abs(info["nll"] - vf.chunk_nll(true, chunks)) < 2e-3
    # a constant weight: w* = 0.3
    c = _synthetic_chunks([math.log(0.3 / 0.7)], F=0, seed=1)
    thc, ic = vf.fit_weights(c, 0)
    wc = 1.0 / (1.0 + math.exp(-thc[0]))
    assert abs(wc - 0.3) < 0.02, wc
    # the endpoints: data drawn from the map alone push w toward 0
    d = _synthetic_chunks([-30.0], F=0, seed=2)
    thd, idd = vf.fit_weights(d, 0)
    assert 1.0 / (1.0 + math.exp(-thd[0])) < 0.02 and idd["nll"] <= idd["nll_map"] + 1e-6
    print(f"PASS fit: UG theta {[round(x, 3) for x in th]} vs {true} (max err {err:.3f}); "
          f"constant w {wc:.3f} vs 0.3")


# ------------------------------------------------------------------------- CLI (mock)
def _toy(root, extractor="clearclip"):
    """A toy 7-Scenes scene (2 frames; tests/test_seg_eval.py layout) with a field built from
    noisy partial views, the NormLift + view-count sums, the map path, and a hole in the map
    (no Gaussians at x > 0.55, y > 0.35: uncovered pixels)."""
    t = _load("test_seg_eval_vf2", REPO / "tests" / "test_seg_eval.py")
    from usplat.feature_pca import StreamingPCA
    root.mkdir(parents=True, exist_ok=True)

    def labels():
        lab = np.zeros((480, 640), np.int64)
        lab[40:, :320] = 3
        lab[40:, 320:] = 7
        return lab
    t._fake_7scenes(root / "data", labels)
    D, d = 16, 8
    pca = StreamingPCA(D, d)
    pca.basis = torch.eye(D)[:, :d]
    tx = {"chair": torch.eye(D)[0], "table": torch.eye(D)[1], "sofa": torch.eye(D)[2]}
    xs = torch.arange(-1.15, 1.2, 0.04)
    ys = torch.arange(-0.9, 0.95, 0.04)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    means = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.full((gx.numel(),), 2.0)], -1)
    means = means[~((means[:, 0] > 0.55) & (means[:, 1] > 0.35))]
    N = means.shape[0]
    splats = dict(means=means, quats=torch.tensor([1.0, 0, 0, 0]).expand(N, 4).clone(),
                  scales=torch.full((N, 3), 0.03), opacities=torch.full((N,), 0.9))
    torch.save(splats, root / "gs.pt")
    right = means[:, 0] > 0
    feat = torch.where(right[:, None], pca.project(tx["table"][None]),
                       pca.project(tx["chair"][None]))
    f = DirectionalSemanticField(N, d, FieldConfig(change_test=False))
    f.set_centre(torch.zeros(d))
    bs = default_baselines(f)
    g = torch.Generator().manual_seed(0)
    for _ in range(4):
        sub = torch.rand(N, generator=g) > 0.4
        ids = torch.nonzero(sub).squeeze(1)
        z = feat[ids] + 0.9 * torch.randn(len(ids), d, generator=g)
        f.update_sparse(ids, torch.rand(len(ids), generator=g) * 0.05, z, num_pixels=100)
    torch.save(dict(field=f.state_dict(), pca=pca.state_dict(), baselines=baselines_state(bs),
                    config=dict(ply=str(root / "gs.pt"))), root / "field.pt")
    vocab = dict(templates=["a photo of a {}."], ignore=[0],
                 classes={3: {"name": "chair"}, 7: {"name": "table"}, 9: "sofa"})
    (root / "vocab.json").write_text(json.dumps(vocab))
    tag = "m|p" if extractor == "clearclip" else extractor
    torch.save({f"{tag}|a photo of a {{}}.|{k}": v for k, v in tx.items()}, root / "text.pt")
    scenes = [dict(name="toy", field=str(root / "field.pt"), ply=str(root / "gs.pt"),
                   data=str(root / "data"), dataset="7scenes", stride=1,
                   label_pattern="{root}/frame-{idx:06d}.label.png")]
    (root / "scenes.json").write_text(json.dumps(scenes))
    base = ["--vocab", str(root / "vocab.json"), "--backend", "mock", "--device", "cpu",
            "--downscale", "16", "--tau-cov", "0.5", "--extractor", extractor,
            "--text-cache", str(root / "text.pt")]
    if extractor == "clearclip":
        base += ["--clip-model", "m", "--clip-pretrained", "p"]
    return base


ES_ROWS = "ours_P0,ours_P2,averaging,viewcount,normlift,2d_features"
KEYS = ("miou", "acc", "ece", "nll", "brier", "aurc", "n_pixels", "coverage")


def test_eval_fusion_cli_mock(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    ef = _load("eval_fusion_t", REPO / "scripts" / "eval_fusion.py")
    toy = _toy_extractor()
    monkeypatch.setattr(ef.es, "_extractor", lambda args: toy)
    base = _toy(tmp_path)
    S = str(tmp_path / "scenes.json")
    es_out = ef.es.main(base + ["--val-scenes", S, "--scenes", S, "--with-2d", "--rows", ES_ROWS,
                                "--row-factory", "usplat.baselines:row_factory", "--no-p3",
                                "--plain-json", "--out", str(tmp_path / "es.json")])
    val = tmp_path / "val.json"
    out = ef.main(base + ["--val-scenes", S, "--scenes", S, "--plain-json", "--out", str(val)])
    # stage 1 = eval_seg's own fit, value for value
    assert out["betas"] == es_out["betas"], (out["betas"], es_out["betas"])
    sc, esc = out["scenes"]["toy"], es_out["scenes"]["toy"]
    for r in ("ours_P0", "ours_P2", "averaging", "2d_features"):
        for k in KEYS:
            assert sc["rows"][r]["all"][k] == esc["rows"][r]["all"][k], (r, k)
    for r in ("ours_P0", "ours_P2", "averaging"):          # map_covered = eval_seg's covered
        for k in KEYS:
            assert sc["rows"][r]["map_covered"][k] == esc["rows"][r]["covered"][k], (r, k)
    n_unc = sc["rows"]["ours_P0"]["n_map_uncovered"]
    assert n_unc > 0 and sc["rows"]["ours_P0"]["all"]["coverage"] < 1.0
    # stage 2: every form fitted, both variants, the fused rows decide every pixel
    assert sorted(out["fusion"]) == sorted(f.name for f in vf.FORMS)
    for f in vf.FORMS:
        e = out["fusion"][f.name]
        assert e["spec"] == f.spec() and len(e["theta"]) == len(f.params())
        assert all(math.isfinite(x) for x in e["theta"]) and e["fit"]["n_pixels"] > 0
        assert e["fit"]["nll"] <= min(e["fit"]["nll_map"], e["fit"]["nll_2d"]) + 1e-9
        r = sc["rows"][f.name]
        assert r["all"]["coverage"] == 1.0 and 0.0 <= r["w2d_mean_map_covered"] <= 1.0
        assert r["map_covered"]["n_pixels"] == sc["rows"]["ours_P0"]["map_covered"]["n_pixels"]
    # --betas: the validation evaluation again, from the record
    again = ef.main(base + ["--betas", str(val), "--scenes", S, "--plain-json",
                            "--out", str(tmp_path / "again.json")])
    for r, m in sc["rows"].items():
        for var in ("all", "map_covered"):
            for k in KEYS:
                assert again["scenes"]["toy"]["rows"][r][var][k] == m[var][k], (r, var, k)
    # endpoints at metric level: w = 0 -> the map row on map-covered pixels, w = 1 -> 2D row
    rec = json.loads(val.read_text())
    for name in ("fuse_p0_const", "fuse_p2_ug", "fuse_avg_vc"):
        rec["fusion"][name]["theta"] = [-1e300] + [0.0] * (len(rec["fusion"][name]["theta"]) - 1)
    rec["fusion"]["fuse_avg_const"]["theta"] = [1e300]
    (tmp_path / "ends.json").write_text(json.dumps(rec))
    ends = ef.main(base + ["--betas", str(tmp_path / "ends.json"), "--scenes", S, "--plain-json",
                           "--out", str(tmp_path / "ends_out.json")])["scenes"]["toy"]["rows"]
    for fused, mrow in (("fuse_p0_const", "ours_P0"), ("fuse_p2_ug", "ours_P2"),
                        ("fuse_avg_vc", "averaging")):
        for k in ("miou", "acc", "ece", "nll", "brier", "aurc"):
            assert ends[fused]["map_covered"][k] == ends[mrow]["map_covered"][k], (fused, k)
    for k in ("miou", "acc", "ece", "nll", "brier", "aurc"):
        for var in ("all", "map_covered"):
            assert ends["fuse_avg_const"][var][k] == ends["2d_features"][var][k], (var, k)
    # refusals
    with pytest.raises(ValueError, match="fixes its rows"):
        ef.main(base + ["--val-scenes", S, "--rows", "ours_P0", "--plain-json",
                        "--out", str(tmp_path / "x.json")])
    bad = json.loads(val.read_text())
    bad["fusion"]["fuse_p0_ug"]["spec"]["features"] = ["u_epi"]
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="differs from the code"):
        ef.main(base + ["--betas", str(tmp_path / "bad.json"), "--scenes", S, "--plain-json",
                        "--out", str(tmp_path / "y.json")])
    del bad["fusion"]
    (tmp_path / "bad2.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="no fitted fusion"):
        ef.main(base + ["--betas", str(tmp_path / "bad2.json"), "--scenes", S, "--plain-json",
                        "--out", str(tmp_path / "z.json")])
    with pytest.raises(ValueError, match="not both"):
        ef.main(base + ["--betas", str(val), "--val-scenes", S, "--scenes", S, "--plain-json",
                        "--out", str(tmp_path / "w.json")])
    print(f"PASS eval_fusion (mock): betas = eval_seg's; reference rows = eval_seg's; "
          f"{n_unc} uncovered px; endpoints exact at metric level; refusals")


# ---------------------------------------------------------------------- select rules
def _body(nll):
    """A validation body with one scene: {row: pooled NLL} at 100 pixels each."""
    return dict(scenes={"s": dict(rows={r: {"all": dict(nll=v, n_pixels=100)}
                                        for r, v in nll.items()})})


def test_select_gate_and_claim():
    fs = _load("fusion_select_t", REPO / "scripts" / "fusion_select.py")
    p = json.loads(PREREG.read_text())
    rows = {f.name: 0.50 for f in vf.FORMS}
    rows["fuse_p0_ug"] = 0.40
    chosen = {"@ug": "fuse_p0_ug"}
    assert fs.binding(p, chosen) == ["fuse_avg_const", "fuse_p0_const", "fuse_avg_vc",
                                     "fuse_avg_nl"]
    g = fs.validation_gate(p, _body(rows), chosen)
    assert g["passed"] and not g["enabled"]
    rows["fuse_avg_nl"] = 0.39995                          # within tie tol: not beaten
    assert not fs.validation_gate(p, _body(rows), chosen)["passed"]
    # @ug tie rule: P2 must beat P0 by more than tol
    rows = {"fuse_p0_ug": 0.40, "fuse_p2_ug": 0.40 - 5e-5}
    assert fs.mt.choose_v2(_body(rows), "@ug", p)[0] == "fuse_p0_ug"
    rows["fuse_p2_ug"] = 0.40 - 2e-4
    assert fs.mt.choose_v2(_body(rows), "@ug", p)[0] == "fuse_p2_ug"
    # claim: 20 tests over the pooled scenes, '@paired_const' resolved per dataset
    def table(dataset, ug, n, win):
        b = fs.binding(p, {"@ug": ug})
        comps = {m: {x: dict(diffs={f"{dataset}{i}": (1.0 if i < win else -1.0) for i in range(n)})
                     for x in b} for m in fs.CLAIM_METRICS}
        return dict(dataset=dataset, C1=dict(baselines=b, comparisons=comps,
                                             verdict={m: True for m in fs.CLAIM_METRICS},
                                             writable={m: True for m in fs.CLAIM_METRICS}))
    c = fs.claim(p, [table("replica", "fuse_p0_ug", 6, 6), table("scannet", "fuse_p2_ug", 8, 8)])
    assert c["n_tests"] == 20 and all(t["n"] == 14 for t in c["tests"])
    assert all(t["significant"] for t in c["tests"])        # 14/14 wins: p 1.2e-4, Holm 0.0024
    c = fs.claim(p, [table("replica", "fuse_p0_ug", 6, 4), table("scannet", "fuse_p2_ug", 8, 5)])
    assert not any(t["significant"] for t in c["tests"])
    with pytest.raises(SystemExit, match="claim needs"):
        fs.claim(p, [table("replica", "fuse_p0_ug", 6, 6)])
    print("PASS select: gate, tie rule (P0 first), F3f 20 tests with Holm")


# ---------------------------------------------------------------------------- driver
def _git(root, *a):
    subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True,
                   env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                            GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"))


def test_fusion_driver(tmp_path, monkeypatch):
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("needs git and bash")
    pytest.importorskip("cv2")
    toy = tmp_path / "toy"
    base = _toy(toy, extractor="fusiontoy")
    # the "Table I" validation record of the toy: eval_seg's rows on the same scene
    ef = _load("eval_fusion_t2", REPO / "scripts" / "eval_fusion.py")
    monkeypatch.setattr(ef.es, "_extractor", lambda args: _toy_extractor())
    t1 = tmp_path / "table1_val.json"
    ef.es.main(base + ["--val-scenes", str(toy / "scenes.json"), "--scenes", str(toy / "scenes.json"),
                       "--with-2d", "--rows", ES_ROWS, "--row-factory", "usplat.baselines:row_factory",
                       "--no-p3", "--plain-json", "--out", str(t1)])
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "configs").mkdir()
    for f in ("fusion_tab1.sh", "fusion_select.py", "eval_fusion.py", "eval_seg.py",
              "make_tables.py", "quant_stats.py"):
        shutil.copy2(REPO / "scripts" / f, repo / "scripts" / f)
    (repo / "scripts" / "gpu_gate.sh").write_text("#!/usr/bin/env bash\nshift\nexec \"$@\"\n")
    (repo / "scripts" / "gpu_gate.sh").chmod(0o755)
    shutil.copytree(REPO / "usplat", repo / "usplat", ignore=shutil.ignore_patterns("__pycache__"))
    (repo / "usplat" / "extractors" / "fusiontoy.py").write_text(TOY_EXTRACTOR)
    sc = json.loads((toy / "scenes.json").read_text())
    (repo / "configs" / "toy_val.json").write_text(json.dumps(sc))
    (repo / "configs" / "toy_test.json").write_text(json.dumps([dict(sc[0], name="toy_test")]))
    out = tmp_path / "out"
    p = json.loads(PREREG.read_text())
    p["datasets"] = {"toy": dict(scenes=dict(validation="configs/toy_val.json",
                                             test="configs/toy_test.json"),
                                 eval=dict(vocab=str(toy / "vocab.json"), dataset="7scenes",
                                           stride=1, downscale=16, tau_cov=0.5,
                                           extractor="fusiontoy"),
                                 text_cache_source=str(toy / "text.pt"), out=str(out),
                                 table1=dict(val=str(t1)), test_look="toy test look")}
    p["claim"]["min_wins"] = {"toy": 1}
    pr = "configs/fusion_prereg_toy.json"
    (repo / pr).write_text(json.dumps(p, indent=1))
    git = lambda *a: _git(repo, *a)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "init")
    env = dict(os.environ, PREREG=pr, DATASET="toy", BACKEND="mock", DEVICE="cpu")
    for k in ("OUT", "EVAL_EXTRA", "SCENES"):
        env.pop(k, None)
    env = {k: v for k, v in env.items() if not k.startswith("UAGS_LSEG_")}

    def run(stage, **kw):
        return subprocess.run(["bash", str(repo / "scripts" / "fusion_tab1.sh")],
                              env=dict(env, STAGE=stage, **kw), capture_output=True, text=True,
                              timeout=1200)

    def refused(r, msg):
        return r.returncode == 1 and msg in r.stdout + r.stderr

    r = run("test")
    assert refused(r, "is not a finished val record"), r.stdout + r.stderr
    r = run("val", SCENES="configs/toy_test.json")
    assert refused(r, "validation scenes are ['toy']"), r.stdout + r.stderr
    r = run("val", OUT="/home/thanhnc/gauss_ws/data/uags_results/tab1")
    assert refused(r, "never touched"), r.stdout + r.stderr
    r = run("val", OUT="/home/thanhnc/gauss_ws/data/uags_results/smooth/scannet")
    assert refused(r, "never touched"), r.stdout + r.stderr
    r = run("val", EVAL_EXTRA="--clip-model m --rows ours_P2")
    assert refused(r, "EVAL_EXTRA option '--rows'"), r.stdout + r.stderr
    r = run("val", UAGS_LSEG_FLIP="1")
    assert refused(r, "UAGS_LSEG_FLIP"), r.stdout + r.stderr
    (repo / pr).write_text((repo / pr).read_text().replace('"tol": 0.0001', '"tol": 0.001'))
    r = run("val")
    assert refused(r, "is not committed"), r.stdout + r.stderr
    git("checkout", "--", pr)
    vp = repo / "usplat" / "view_fusion.py"
    vp.write_text(vp.read_text() + "\n# local edit\n")
    r = run("val")
    assert refused(r, "uncommitted changes to tracked files"), r.stdout + r.stderr
    git("checkout", "--", "usplat/view_fusion.py")
    r = run("val", DRY_RUN="1")
    assert r.returncode == 0 and r.stdout.count("[dry-run]") == 1, r.stdout + r.stderr
    r = run("val")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    sel = json.loads((out / "selection_val.json").read_text())
    assert sel["status"] == "ok"
    s = sel["metrics"]["selection"]
    assert s["chosen"]["@ug"] in ("fuse_p0_ug", "fuse_p2_ug")
    assert s["reproduction"]["status"] == "pass", s["reproduction"]
    assert s["paired_const"] == p["paired_const"][s["chosen"]["@ug"]]
    v = json.loads((out / "eval_val.json").read_text())
    assert v["status"] == "ok" and v["git"]["dirty_tracked"] is False
    assert (out / "text_cache_fusiontoy.pt").exists()
    # test: refuses code committed after validation, then a Table I record that does not match
    git("commit", "--allow-empty", "-qm", "empty")              # an empty commit is fine
    vp.write_text(vp.read_text() + "\n# a later commit\n")
    git("commit", "-qam", "later")
    r = run("test")
    assert refused(r, "differ from the validation commit"), r.stdout + r.stderr
    git("revert", "--no-edit", "HEAD")
    t1_ok = t1.read_text()
    t1d = json.loads(t1_ok)
    t1d["scenes"]["toy"]["rows"]["2d_features"]["all"]["nll"] += 0.05
    t1.write_text(json.dumps(t1d))
    r = run("test")
    assert refused(r, "do not reproduce Table I"), r.stdout + r.stderr
    t1.write_text(t1_ok)
    r = run("test")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    look = json.loads((out / "look_test.json").read_text())
    assert len(look["attempts"]) == 1 and look["chosen"] == s["chosen"]
    tab = json.loads((out / "table_fusion.json").read_text())
    assert tab["scenes"] == ["toy_test"] and len(tab["rows"]) == 4 + len(vf.FORMS)
    assert set(tab["C1"]["verdict"]) == {"miou", "ece", "nll", "brier", "aurc"}
    for m in ("ece", "nll", "brier"):                      # R2: writable needs both aggregations
        pooled_ok = all(c["pooled"]["better"] for c in tab["C1"]["comparisons"][m].values())
        assert tab["C1"]["writable"][m] == (tab["C1"]["verdict"][m] and pooled_ok)
    assert set(tab["rows"]["fuse_p0_ug"]) == {"all", "map_covered", "pooled"}
    claim = json.loads((out / "claim_fusion.json").read_text())
    assert claim["n_tests"] == 20 and (out / "table_fusion.tex").exists()
    r = run("test")
    assert refused(r, "finished test record"), r.stdout + r.stderr
    r = run("val")
    assert refused(r, "the test stage has started"), r.stdout + r.stderr
    r = run("table")
    assert r.returncode == 0, r.stdout + r.stderr
    print(f"PASS driver: val -> selection {s['chosen']} (reproduction pass) -> test once -> "
          f"table + claim; gates refuse")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
