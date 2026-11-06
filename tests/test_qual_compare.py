"""papers/figure_tools/qual_compare.py (fig:qual-compare) on a CPU toy (mock backend): the 7-Scenes-style scene
of tests/test_seg_eval.py with the saved baselines (averaging, NormLift, view count, CSS), a
Feature 3DGS and a LEGaussians file next to the field, a fake 2D extractor, a hole in the map
(not covered), a block seen in two views only (u_ale undefined) and noisy views:
 * tokens: the saved tokens are the extractor's tokens of the frame image;
 * render: every column's labels, coverage and correctness equal eval_seg's own evaluation
   of the frame (evaluate_view with the real labels; the 2D row through frame_features_2d and
   PixelSource); the rank panels equal a brute-force mid-rank among the covered labelled
   defined pixels of all evaluation frames; the re-scored mIoU / AUSE equal the "published"
   eval_seg JSON exactly (repro PASS), and a wrong table fails it (exit 2);
 * figure: 7.16 in wide PDF with TrueType fonts (no Type 3), numbers read from the table,
   caption generated from them; a changed table is refused;
 * mid-ranks and the confident-error rule on small cases.
"""
import importlib.util
import json
import pathlib
import re

import numpy as np
import pytest
import torch
import torch.nn.functional as Fn

from usplat.baselines import default_baselines, baselines_state
from usplat.feature_pca import StreamingPCA
from usplat.seg_eval import evaluate_view
from usplat.vmf_field import DirectionalSemanticField, FieldConfig
from usplat import distill as DI

REPO = pathlib.Path(__file__).resolve().parents[1]
RF = "usplat.baselines:row_factory,usplat.distill:row_factory"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TS = _load("test_seg_eval_for_compare", REPO / "tests" / "test_seg_eval.py")   # _fake_7scenes
QC = _load("qual_compare", REPO / "papers" / "figure_tools" / "qual_compare.py")
ES, RQ = QC.ES, QC.RQ


class Fake:
    """2D extractor: chair tokens (e0) on the left half, table tokens (e1) on the right."""
    d = 16

    def extract_tokens(self, image):
        H, W = image.shape[:2]
        tok = torch.zeros(H // 16, W // 16, 16)
        tok[:, : W // 32, 0] = 1.0
        tok[:, W // 32:, 1] = 1.0
        tok[:, W // 32 - 2: W // 32 + 2, 2] = 0.7       # a sofa-ish strip in the middle
        return tok


def _labels():
    lab = np.zeros((480, 640), np.int64)
    lab[40:, :320] = 3                                 # chair
    lab[40:, 320:] = 7                                 # table
    lab[40:80, 320:] = 8                               # merged into table
    return lab


def _build(tmp, plain=False):
    """plain: no hole, every Gaussian in every view, no noise (the betas are fit on it, as in
    tests/test_render_qual.py: fit on the changed toy, P2 runs away)."""
    TS._fake_7scenes(tmp / "data", _labels, n=3)
    D, d = 16, 8
    pca = StreamingPCA(D, d)
    pca.basis = torch.eye(D)[:, :d]
    t = {"chair": torch.eye(D)[0], "table": torch.eye(D)[1], "sofa": torch.eye(D)[2]}
    xs = torch.arange(-1.15, 1.2, 0.04)
    ys = torch.arange(-0.9, 0.95, 0.04)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    means = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.full((gx.numel(),), 2.0)], -1)
    N = means.shape[0]
    hole = (means[:, 0] > 0.7) & (means[:, 1] < -0.5) & (not plain)   # image top right
    op = torch.where(hole, torch.zeros(N), torch.full((N,), 0.9))
    splats = dict(means=means, quats=torch.tensor([1.0, 0, 0, 0]).expand(N, 4).clone(),
                  scales=torch.full((N, 3), 0.03), opacities=op,
                  colors=torch.rand(N, 3, generator=torch.Generator().manual_seed(1)))
    torch.save(splats, tmp / "gs.pt")
    feat = torch.where((means[:, 0] > 0)[:, None], pca.project(t["table"][None]),
                       pca.project(t["chair"][None]))
    few = (means[:, 0] < -0.5) & (means[:, 1] > 0.4) & (not plain)   # bottom left: 2 views
    f = DirectionalSemanticField(N, d, FieldConfig(change_test=False))
    f.set_centre(torch.zeros(d))
    text = pca.project_text(torch.stack([t["chair"], t["table"], t["sofa"]]))
    bs = default_baselines(f, text, class_ids=[3, 7, 9])
    g = torch.Generator().manual_seed(0)
    for v in range(4):
        ids = torch.arange(N) if v < 2 else torch.nonzero(~few).squeeze(1)
        noise = (0.0 if plain else 0.4) * torch.rand(len(ids), 1, generator=g) \
            * torch.randn(len(ids), d, generator=g)
        f.update_sparse(ids, torch.ones(len(ids)), feat[ids] + noise, num_pixels=100)
        bs[2].update_sparse(ids, torch.ones(len(ids)), feat[ids])
    main = str(tmp / "field.pt")
    torch.save(dict(field=f.state_dict(), pca=pca.state_dict(), baselines=baselines_state(bs),
                    config=dict(out=main)), main)
    sha = DI.basis_sha256(pca.state_dict())
    f3 = feat + 0.3 * torch.randn(N, d, generator=g)
    dec = DI.LEGDecoder(Fn.normalize(torch.randn(4, d, generator=g), dim=-1))
    torch.save(dict(name="f3dgs", features=f3, u=None, decoder=None, identity="id-f3dgs",
                    pca_basis_sha256=sha), DI.out_path(main, "f3dgs"))
    torch.save(dict(name="legs_l4", features=torch.randn(N, DI.LEGS_DIM, generator=g),
                    u=torch.rand(N, generator=g), decoder=dec.state_dict(),
                    identity="id-legs_l4", pca_basis_sha256=sha), DI.out_path(main, "legs_l4"))
    vocab = dict(templates=["a photo of a {}."], ignore=[0], merges={8: 7},
                 classes={3: {"name": "chair"}, 7: {"name": "table"}, 9: "sofa"})
    (tmp / "vocab.json").write_text(json.dumps(vocab))
    key = lambda p: f"m|p|a photo of a {{}}.|{p}"
    torch.save({key(k): v for k, v in t.items()}, tmp / "text.pt")
    scenes = [dict(name="toy", field=main, ply=str(tmp / "gs.pt"), data=str(tmp / "data"),
                   dataset="7scenes", stride=1, label_pattern="{root}/frame-{idx:06d}.label.png")]
    (tmp / "scenes.json").write_text(json.dumps(scenes))
    return ["--vocab", str(tmp / "vocab.json"), "--backend", "mock", "--device", "cpu",
            "--downscale", "16", "--tau-cov", "0.5", "--extractor", "clearclip",
            "--clip-model", "m", "--clip-pretrained", "p", "--text-cache", str(tmp / "text.pt")]


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("compare")
    fake = Fake()
    orig = ES._extractor
    ES._extractor = lambda args: fake
    try:
        scenes = str(tmp / "scenes.json")
        rows = ["--rows", "ours_P2,averaging,normlift,f3dgs,legs_l4,2d_features", "--no-p3"]
        base = _build(tmp, plain=True)
        ES.main(base + ["--val-scenes", scenes, "--with-2d", "--row-factory", RF, "--plain-json",
                        "--out", str(tmp / "fit.json")] + rows)
        base = _build(tmp)
        ES.main(base + ["--scenes", scenes, "--betas", str(tmp / "fit.json"), "--with-2d",
                        "--row-factory", RF, "--plain-json", "--out", str(tmp / "table.json")]
                + rows)
        cmp_dir = tmp / "cmp"
        tok = QC.main(base + ["--stage", "tokens", "--scenes", scenes, "--target", "toy:1",
                              "--out-dir", str(cmp_dir)])
        render = ["--stage", "render", "--scenes", scenes, "--scene", "toy", "--frame", "1",
                  "--legs", "legs_l4", "--betas", str(tmp / "fit.json"),
                  "--row-factory", RF, "--out-dir", str(cmp_dir), "--dataset-label", "Toy"]
        rend = QC.main(base + render + ["--table", str(tmp / "table.json")])
    finally:
        ES._extractor = orig
        ES._EXTRACTOR.clear()
    return dict(tmp=tmp, base=base, scenes=scenes, cmp=cmp_dir, tok=tok, rend=rend,
                render=render, fake=fake)


def _reference(toy):
    """eval_seg's own path (evaluate_scene): every evaluation frame with the real labels, the
    shown rows and the 2D row (tokens extracted by load_scene)."""
    args = ES.build_parser().parse_args(toy["base"] + [
        "--scenes", toy["scenes"], "--row-factory", RF, "--with-2d", "--no-p3",
        "--rows", "ours_P2,normlift,f3dgs,legs_l4,2d_features", "--out", "unused"])
    orig = ES._extractor
    ES._extractor = lambda a: toy["fake"]
    try:
        S = ES.load_scene(ES._load_structured(toy["scenes"])[0], ES.load_vocab(args.vocab),
                          args, "cpu")
    finally:
        ES._extractor = orig
        ES._EXTRACTOR.clear()
    betas = ES.load_betas(str(toy["tmp"] / "fit.json"))
    outs = []
    for f in S["frames"]:
        ras = ES.make_rasterizer("mock", S["splats"], f, args)
        o = evaluate_view(S["rows"], ras, S["text"], f["gt"], betas, args.tau_cov,
                          undef_frac=args.undef_frac)
        o.update(evaluate_view(S["rows2d"], ES.PixelSource(ES.frame_features_2d(S, f, args)),
                               S["text"], f["gt"], betas, args.tau_cov,
                               undef_frac=args.undef_frac))
        outs.append(o)
    return S, outs


def test_tokens_are_the_extractor_tokens(toy):
    (p,) = toy["tok"]["outputs"]
    d = torch.load(p, weights_only=False)
    assert d["scene"] == "toy" and d["frame"] == 1 and d["index"] == 1
    assert torch.equal(d["tok"], toy["fake"].extract_tokens(np.zeros((480, 640, 3))))
    assert (d["h"], d["w"]) == (30, 40) and d["image_shape"] == [480, 640, 3]
    print(f"PASS tokens: {tuple(d['tok'].shape)} for toy frame 1")


def test_render_equals_eval_seg_and_ranks(toy):
    S, outs = _reference(toy)
    rend = toy["rend"]
    maps, ranks, meta = rend["maps"], rend["ranks"], rend["meta"]
    ref = outs[1]
    for r in ("ours_P2", "normlift", "f3dgs", "legs_l4", "2d_features"):
        pix = ref[r]["pix"]
        flat = lambda t: t.reshape(-1)[pix]
        assert torch.equal(flat(maps[r]["pred"]), ref[r]["pred"].long()), r
        assert torch.equal(flat(maps[r]["covered"]), ref[r]["covered"]), r
        assert torch.equal(flat(maps[r]["correct"]), ref[r]["correct"]), r
    cov = maps["ours_P2"]["covered"]
    assert (~cov).any() and cov.any(), "the hole must leave pixels not covered"
    assert maps["2d_features"]["covered"].all()                 # the 2D row covers every pixel
    und = cov & ~torch.isfinite(maps["ours_P2"]["u"]["u_ale"])
    assert und.any(), "the two-view block must leave u_ale undefined"
    # brute-force mid-ranks among the covered labelled defined pixels of all frames
    for r, ch in meta["own_channels"].items():
        pool = torch.cat([o[r]["u"][ch][o[r]["covered"] & torch.isfinite(o[r]["u"][ch])]
                          for o in outs]).double().numpy()
        assert meta["pool_sizes"][r] == pool.size > 0
        u = maps[r]["u"][ch].double().numpy()
        c = maps[r]["covered"].numpy()
        rk = ranks[r].numpy()
        ok = c & np.isfinite(u)
        want = ((pool[None, :] < u[ok][:, None]).sum(1) + (pool[None, :] <= u[ok][:, None]).sum(1)) \
            / (2.0 * pool.size)
        assert np.allclose(rk[ok], want, atol=1e-6), r
        assert np.isnan(rk[~ok]).all() and rk[ok].min() >= 0 and rk[ok].max() <= 1
    assert meta["repro"]["status"] == "PASS" and meta["repro"]["max_abs_delta"] <= 1e-12
    assert meta["own_channels"] == {"ours_P2": "u_ale", "normlift": "normlift_1-Cinter",
                                    "legs_l4": "legs_u"}
    assert meta["columns"]["normlift"] == "normlift" and meta["table_rows"]["normlift"] == "averaging"
    print(f"PASS render = eval_seg on {ref['ours_P2']['pix'].numel()} labelled px; ranks = "
          f"brute force (pools {meta['pool_sizes']}); repro max |delta| "
          f"{meta['repro']['max_abs_delta']:.1e}")


def test_repro_check_fails_on_another_table(toy, tmp_path):
    t = json.loads((toy["tmp"] / "table.json").read_text())
    t["scenes"]["toy"]["rows"]["ours_P2"]["all"]["miou"] -= 0.01
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(t))
    orig = ES._extractor
    try:
        with pytest.raises(SystemExit) as e:
            QC.main(toy["base"] + [a if a != str(toy["cmp"]) else str(tmp_path)
                                   for a in toy["render"]]
                    + ["--table", str(bad), "--tokens",
                       str(QC.tok_path(str(toy["cmp"]), "toy", 1))])
        assert e.value.code == 2
    finally:
        ES._extractor = orig
    meta = json.loads((tmp_path / "toy_f1.json").read_text())
    assert meta["repro"]["status"] == "FAIL"
    assert abs(meta["repro"]["rows"]["ours_P2"]["miou_delta"] - 0.01) < 1e-9
    with pytest.raises(SystemExit, match="repro"):
        QC.main(["--vocab", toy["base"][1], "--stage", "figure", "--blocks",
                 str(tmp_path / "toy_f1.pt"), "--out", str(tmp_path / "x.pdf")])
    print("PASS repro: a table 1 point off fails (exit 2) and the figure refuses the block")


def _ihdr_width(path):
    data = pathlib.Path(path).read_bytes()
    return int.from_bytes(data[16:20], "big")


def test_figure_numbers_from_table_and_files(toy, tmp_path):
    block = str(QC.block_path(str(toy["cmp"]), "toy", 1))
    out = QC.main(["--vocab", toy["base"][1], "--stage", "figure", "--blocks", block,
                   "--out", str(tmp_path / "figs" / "cmp.pdf"), "--png-dpi", "100",
                   "--meta", str(tmp_path / "cmp.json")])
    pdf, png, txt = out["outputs"]
    data = pathlib.Path(pdf).read_bytes()
    assert data[:5] == b"%PDF-" and b"/Type3" not in data and b"/FontFile2" in data
    box = re.search(rb"/MediaBox \[ *0 0 ([\d.]+) ([\d.]+) *\]", data)
    assert box and abs(float(box.group(1)) / 72 - RQ.FIG_W) < 0.01, box
    assert _ihdr_width(png) == round(RQ.FIG_W * 100)
    t = json.loads((toy["tmp"] / "table.json").read_text())["scenes"]["toy"]["rows"]
    (info,) = out["meta"]["info"]
    assert info["miou"]["ours_P2"] == t["ours_P2"]["all"]["miou"]
    assert info["miou"]["normlift"] == t["averaging"]["all"]["miou"]
    assert info["miou"]["2d_features"] == t["2d_features"]["all"]["miou"]
    assert info["ause"]["ours_P2"] == t["ours_P2"]["all"]["ranking"]["u_ale"]["ause"]
    assert info["ause"]["legs"] == t["legs_l4"]["all"]["ranking"]["legs_u"]["ause"]
    assert set(info["ause"]) == {"normlift", "legs", "ours_P2"}       # no 2D / F3DGS own u
    cap = pathlib.Path(txt).read_text()
    assert f"{100 * t['ours_P2']['all']['miou']:.1f}" in cap and "\n" == cap[-1]
    assert cap.count("\n") == 1 and "/home" not in cap                  # one line, no paths
    # a table changed after the render is refused
    tab = toy["tmp"] / "table.json"
    keep = tab.read_text()
    try:
        tab.write_text(keep + " ")
        with pytest.raises(SystemExit, match="changed"):
            QC.main(["--vocab", toy["base"][1], "--stage", "figure", "--blocks", block,
                     "--out", str(tmp_path / "y.pdf")])
    finally:
        tab.write_text(keep)
    print(f"PASS figure: {float(box.group(1)) / 72:.2f} in wide, TrueType; numbers = table; "
          f"caption: {cap[:80]}...")


def test_rescore_equals_summarize(toy):
    """The cheap re-score (mIoU and own-channel AUSE only) equals SegAccumulator.metrics."""
    S, outs = _reference(toy)
    acc = QC.SegAccumulator(class_names=S["class_names"])
    for o in outs:
        acc.add({k: v for k, v in o.items() if k != "2d_features"})
    rows = ["ours_P2", "normlift", "f3dgs", "legs_l4"]
    own = {r: QC.own_channel(r) for r in rows if QC.own_channel(r)}
    fast, full = QC.rescore(acc, rows, own, 20), acc.metrics(15, 20)
    for r in rows:
        assert fast[r]["all"]["miou"] == full[r]["all"]["miou"], r
        for ch, v in fast[r]["all"]["ranking"].items():
            assert v["ause"] == full[r]["all"]["ranking"][ch]["ause"], (r, ch)
    print("PASS rescore = SegAccumulator.metrics (mIoU, own-channel AUSE)")


def test_prefix_frames_equal_full_list(toy, monkeypatch):
    """prefix_frames gives the same frame at position i as the full load_frames list, also
    when frames with a non-finite pose are skipped; a position past the end is refused."""
    args = ES.build_parser().parse_args(toy["base"] + ["--scenes", toy["scenes"], "--out", "x"])
    args.with_2d = False
    sc = ES._load_structured(toy["scenes"])[0]
    vocab = ES.load_vocab(args.vocab)
    full, _, _ = ES.load_frames(sc, vocab, args)
    for i in range(len(full)):
        assert QC.prefix_frames(sc, vocab, args, i)[i]["idx"] == full[i]["idx"]
    assert args.max_views is None
    from usplat import datasets as DS
    orig = DS.make_dataset

    def bad_first(*a, **k):                      # frame 0 gets a non-finite pose
        ds = orig(*a, **k)
        get = ds.__class__.__getitem__

        class Wrap(ds.__class__):
            def __getitem__(self, j):
                fr = get(self, j)
                if j == 0:
                    fr = dict(fr, c2w=torch.full((4, 4), float("nan")))
                return fr
        ds.__class__ = Wrap
        return ds
    monkeypatch.setattr(DS, "make_dataset", bad_first)
    full, _, skipped = ES.load_frames(sc, vocab, args)
    assert skipped == 1 and len(full) == 2
    assert QC.prefix_frames(sc, vocab, args, 1)[1]["idx"] == full[1]["idx"] == 2
    with pytest.raises(ValueError, match="not in"):
        QC.prefix_frames(sc, vocab, args, 2)
    print("PASS prefix_frames = load_frames (with a skipped frame)")


def test_mid_ranks():
    pool = np.array([0.1, 0.2, 0.2, 0.4])
    r = QC.mid_ranks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, np.nan], pool)
    assert np.allclose(r[:6], [0.0, 0.125, 0.5, 0.75, 0.875, 1.0]) and np.isnan(r[6])
    assert np.isnan(QC.mid_ranks([0.3], np.array([]))).all()
    print("PASS mid-ranks: ties get the mean rank, outside values 0 / 1, NaN stays NaN")


def test_confident_rule(tmp_path):
    t = lambda *v: torch.tensor(v)
    nan = float("nan")
    per = [dict(covered=t(True, True, True, False), correct=t(False, False, True, False),
                u_ale=t(0.1, nan, 0.1, 0.0), u_epi=t(0.1, 0.1, 0.1, 0.0)),
           dict(covered=t(True, True, True, True), correct=t(False, False, True, True),
                u_ale=t(0.1, 0.9, 0.5, 0.5), u_epi=t(0.1, 0.1, 0.9, 0.9))]
    c = QC.confident_counts(per, [0, 10])
    # covered defined u_ale: 0.1, 0.1, 0.1, 0.9, 0.5, 0.5 -> median 0.3; u_epi median 0.1
    assert c["median_u_ale"] == pytest.approx(0.3) and c["median_u_epi"] == pytest.approx(0.1)
    assert [f["confident_errors"] for f in c["frames"]] == [1, 1]     # NaN u_ale not confident
    metas = []
    for k, name in enumerate(("a", "b")):
        m = dict(scene=name, repro=dict(status="PASS"), confident=c)
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(m))
        metas.append(str(p))
    out = QC.main(["--vocab", "x", "--stage", "confident", "--metas", ",".join(metas),
                   "--out", str(tmp_path / "conf.json")])
    assert out["pick"]["scene"] == "a" and out["pick"]["frame"] == 0   # ties: order, frame
    print("PASS confident rule: scene medians, undefined u_ale not confident, tie order")
