"""scripts/render_qual.py (fig:qual) on the toy scene of tests/test_seg_eval.py (_cli_toy: mock
backend, CPU), with a hole in the map (transparent Gaussians: pixels not covered), a block seen
in two views only (u_ale undefined there) and noisy views (u_ale varies):
 * the per-pixel maps equal eval_seg's evaluation of the frame (evaluate_view with the real
   labels): same decisions, coverage, correctness and rendered u on every labelled pixel;
 * the panels: one palette for GT and prediction (no black or grey class colour), no decision
   and not covered black, u_ale undefined grey, (f) white right / red wrong; with a k-view
   field e_red blue and e_irr red;
 * per-frame e_red / e_irr summed over the frames equal scripts/eval_views.py's counts;
 * --list-frames ranks the frames; the files are a 7.16 in wide PDF (fonts embedded, no Type 3)
   and PNG; a frame outside the scene is refused.
"""
import importlib.util
import json
import pathlib
import re

import numpy as np
import pytest
import torch

from usplat.feature_pca import StreamingPCA
from usplat.seg_eval import evaluate_view
from usplat.vmf_field import DirectionalSemanticField, FieldConfig

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TS = _load("test_seg_eval_for_qual", REPO / "tests" / "test_seg_eval.py")     # _cli_toy
RQ = _load("render_qual", REPO / "scripts" / "render_qual.py")
ES = RQ.ES


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """The toy scene with betas fit on the plain toy (as on validation), then the hole, the
    two-view block and the noisy views added; and a 1-view k-field ('table' everywhere).
    (Fit after the changes, P2 runs away to a, g ~ -1e244: every covered pixel with u_ale
    undefined, the two-view block, is right, so the NLL keeps falling as their beta grows.)"""
    tmp = tmp_path_factory.mktemp("qual")
    base = TS._cli_toy(tmp)
    fit = tmp / "fit.json"
    ES.main(base + ["--val-scenes", str(tmp / "scenes.json"), "--plain-json", "--out", str(fit)])
    gs = torch.load(tmp / "gs.pt")
    m = gs["means"]
    hole = (m[:, 0] > 0.7) & (m[:, 1] < -0.5)              # image top right (y points down)
    gs["opacities"] = torch.where(hole, torch.zeros_like(gs["opacities"]), gs["opacities"])
    torch.save(gs, tmp / "gs.pt")
    ck = torch.load(tmp / "field.pt", weights_only=False)
    pca = StreamingPCA.from_state(ck["pca"])
    N, d = m.shape[0], pca.basis.shape[1]
    chair, table = pca.project(torch.eye(16)[0][None]), pca.project(torch.eye(16)[1][None])
    feat = torch.where((m[:, 0] > 0)[:, None], table, chair)
    few = (m[:, 0] < -0.5) & (m[:, 1] > 0.4)               # bottom left: two views only
    g = torch.Generator().manual_seed(0)
    f = DirectionalSemanticField(N, d, FieldConfig(change_test=False))
    f.set_centre(torch.zeros(d))
    for v in range(4):
        ids = torch.arange(N) if v < 2 else torch.nonzero(~few).squeeze(1)
        noise = 0.3 * torch.rand(len(ids), 1, generator=g) * torch.randn(len(ids), d, generator=g)
        f.update_sparse(ids, torch.ones(len(ids)), feat[ids] + noise, num_pixels=100)
    torch.save(dict(field=f.state_dict(), pca=ck["pca"], timing=dict(keyframes=4)),
               tmp / "field.pt")
    fk = DirectionalSemanticField(N, d, FieldConfig(change_test=False))
    fk.set_centre(torch.zeros(d))
    fk.update_sparse(torch.arange(N), torch.ones(N), table.expand(N, d), num_pixels=100)
    torch.save(dict(field=fk.state_dict(), pca=ck["pca"], timing=dict(keyframes=1)),
               tmp / "field_k1.pt")
    argv = base + ["--scenes", str(tmp / "scenes.json"), "--betas", str(fit)]
    return dict(tmp=tmp, argv=argv, fit=fit, kfield=str(tmp / "field_k1.pt"))


def _reference(toy, frame):
    """eval_seg's own evaluation of one frame (real labels: labelled pixels only)."""
    args = RQ.build_parser().parse_args(toy["argv"] + ["--out", str(toy["tmp"] / "unused.pdf")])
    betas = ES.load_betas(str(toy["fit"]))
    S, row = RQ.load(RQ.pick_scene(args.scenes, ""), ES.load_vocab(args.vocab), betas, args)
    fr = S["frames"][frame]
    ras = ES.make_rasterizer("mock", S["splats"], fr, args)
    return evaluate_view([row], ras, S["text"], fr["gt"], betas, args.tau_cov,
                         undef_frac=args.undef_frac)[row.name]


def test_maps_equal_eval_seg_and_panels(toy, tmp_path):
    out = RQ.main(toy["argv"] + ["--frame", "1", "--out", str(tmp_path / "qual.pdf"),
                                 "--meta", str(tmp_path / "qual.json")])
    full = out["maps"]["full"]
    ref = _reference(toy, 1)
    pix = ref["pix"]
    flat = lambda t: t.reshape(-1)[pix]
    assert torch.equal(flat(full["pred"]), ref["pred"].long())
    assert torch.equal(flat(full["covered"]), ref["covered"])
    assert torch.equal(flat(full["correct"]), ref["correct"])
    for ch in RQ.CHANNELS:
        assert torch.allclose(flat(full["u"][ch]), ref["u"][ch], atol=1e-7, equal_nan=True), ch
    cov, lab = full["covered"].numpy(), full["labelled"].numpy()
    ua = full["u"]["u_ale"].numpy()
    und = cov & np.isnan(ua)
    assert (~cov & lab).any(), "the hole should cover labelled pixels"
    assert und.any() and (cov & np.isfinite(ua)).any() and np.nanmax(ua[cov]) > 0
    # panels
    (label, panels), = out["grid"]
    rgb, gt_img, pred_img, ue_img, ua_img, err_img = panels
    gt, pred, ok = full["gt"].numpy(), full["pred"].numpy(), full["correct"].numpy()
    same = (gt >= 0) & (pred == gt)
    assert label is None and same.any() and np.array_equal(gt_img[same], pred_img[same])
    assert np.all(gt_img[gt < 0] == 0) and np.all(pred_img[pred < 0] == 0)       # black
    assert (pred[~cov] == -1).all()                        # not covered: no decision
    pal = out["palette"]
    assert all(np.ptp(c) > 0.05 for c in pal)              # no black, white or grey class
    assert np.all(ue_img[~cov] == 0) and np.all(ua_img[~cov] == 0)
    assert np.allclose(ua_img[und], RQ.UNDEF) and not np.allclose(ue_img[und], RQ.UNDEF)
    assert np.allclose(err_img[lab & cov & ok], RQ.RIGHT)
    assert (lab & cov & ~ok).any() and np.allclose(err_img[lab & cov & ~ok], RQ.WRONG)
    assert np.all(err_img[~(lab & cov)] == 0)
    assert rgb.shape == gt_img.shape and np.all(rgb[~cov].max(-1) < 0.5)          # dark hole
    meta = json.loads((tmp_path / "qual.json").read_text())
    assert meta["frame"] == 1 and meta["counts"]["labelled"] == int(lab.sum())
    px = [int((gt == c).sum() + (pred == c).sum()) for c in range(2)]     # GT + prediction
    want = ["chair", "table"] if px[0] >= px[1] else ["table", "chair"]   # largest first
    assert meta["layout"] == "full" and meta["legend"] == want, (meta["legend"], px)
    print(f"PASS maps = eval_seg on {pix.numel()} labelled px; {int((~cov).sum())} not covered, "
          f"{int(und.sum())} u_ale undefined; vmax {meta['vmax']}")


def _ihdr_width(path):
    data = pathlib.Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return int.from_bytes(data[16:20], "big")


def test_files_ieee_width_and_fonts(toy, tmp_path):
    out = RQ.main(toy["argv"] + ["--frame", "0", "--out", str(tmp_path / "figs" / "qual.pdf"),
                                 "--png-dpi", "200"])
    pdf, png = out["meta"]["outputs"]
    data = pathlib.Path(pdf).read_bytes()
    assert data[:5] == b"%PDF-" and b"/Type3" not in data and b"/FontFile2" in data
    box = re.search(rb"/MediaBox \[ *0 0 ([\d.]+) ([\d.]+) *\]", data)
    assert box and abs(float(box.group(1)) / 72 - RQ.FIG_W) < 0.01, box
    assert _ihdr_width(png) == round(RQ.FIG_W * 200)
    print(f"PASS files: PDF {float(box.group(1)) / 72:.2f} x {float(box.group(2)) / 72:.2f} in, "
          f"TrueType fonts; PNG {_ihdr_width(png)} px wide")


def test_kfield_e_red_e_irr_equal_eval_views(toy, tmp_path):
    out = RQ.main(toy["argv"] + ["--frame", "0", "--kfield", toy["kfield"],
                                 "--out", str(tmp_path / "qual_k.pdf")])
    full, k = out["maps"]["full"], out["maps"]["k"]
    assert [lab for lab, _ in out["grid"]] == ["Full map", "$M=1$"]
    top, bottom = out["grid"][0][1], out["grid"][1][1]
    assert bottom[0] is None and bottom[1] is None and all(p is not None for p in top)
    err = bottom[5]
    both = (full["labelled"] & full["covered"] & k["covered"]).numpy()
    ck, cf = k["correct"].numpy(), full["correct"].numpy()
    e_red, e_irr = both & ~ck & cf, both & ~ck & ~cf
    assert e_red.sum() > 10 and np.allclose(err[e_red], RQ.E_RED)     # the chair half
    assert np.allclose(err[e_irr], RQ.WRONG) and np.allclose(err[both & ck], RQ.RIGHT)
    assert np.isnan(k["u"]["u_ale"].numpy()[k["covered"].numpy()]).all()   # 1 view: undefined
    assert out["meta"]["counts"]["e_red"] == int(e_red.sum())
    # --one-row: the k-view map alone
    one = RQ.main(toy["argv"] + ["--frame", "0", "--kfield", toy["kfield"], "--one-row",
                                 "--out", str(tmp_path / "qual_k1.pdf")])
    assert len(one["grid"]) == 1 and np.array_equal(one["grid"][0][1][5], err)
    # per-frame counts over all frames = scripts/eval_views.py on the same row
    table = RQ.main(toy["argv"] + ["--kfield", toy["kfield"], "--list-frames", "2",
                                   "--out", str(tmp_path / "frames.json")])["frames"]
    saved = json.loads((tmp_path / "frames.json").read_text())
    assert saved["frames"] == table and saved["k_mapping_views"] == 1
    sc = json.loads((toy["tmp"] / "scenes.json").read_text())
    sc[0]["fields"] = {"1": toy["kfield"]}
    (tmp_path / "views.json").write_text(json.dumps(sc))
    EV = _load("eval_views_for_qual", REPO / "scripts" / "eval_views.py")
    base = [a for a in toy["argv"]]
    i = base.index("--scenes")
    base[i + 1] = str(tmp_path / "views.json")
    ev = EV.main(base + ["--rows", "ours_P2", "--out", str(tmp_path / "views_out.json")])
    r = ev["scenes"]["toy"]["k"]["1"]["rows"]["ours_P2"]
    assert sum(t["e_red"] for t in table) == r["n_e_red"] > 0
    assert sum(t["e_irr"] for t in table) == r["n_e_irr"]
    assert [t["frame"] for t in table] == [0, 1]
    assert table[0]["e_red"] == out["meta"]["counts"]["e_red"]          # same frame, same count
    print(f"PASS k-view: e_red {r['n_e_red']} / e_irr {r['n_e_irr']} px = eval_views.py")


def test_list_frames_ranking(capsys):
    table = [dict(frame=0, index=0, labelled=100, right_full=90, wrong_full=10, ale_defined=0.5,
                  right_k=60, e_red=30, e_irr=10),
             dict(frame=1, index=10, labelled=100, right_full=70, wrong_full=30, ale_defined=0.4,
                  right_k=50, e_red=20, e_irr=30)]
    RQ.print_frames(table, 1, "toy", "k.pt")
    lines = capsys.readouterr().out.splitlines()
    red = lines[lines.index(next(l for l in lines if "by e_red" in l)) + 2].split()
    irr = lines[lines.index(next(l for l in lines if "by e_irr" in l)) + 2].split()
    assert red[0] == "0" and irr[0] == "1"                 # most e_red: frame 0; e_irr: frame 1
    RQ.print_frames(table, 2, "toy", "")
    out = capsys.readouterr().out
    assert "by wrong_full" in out and "e_red" not in out
    print("PASS --list-frames: ranked by e_red, e_irr (wrong pixels without k)")


def test_palette_scale_and_frame_check(toy, tmp_path):
    S = dict(class_ids=[1, 2, 3], frames=[dict(gt=torch.tensor([[0, 1, 1, 1, 1, 2, 2, -1]]))])
    pal = RQ.scene_palette(S)
    base = RQ.class_palette(3)
    assert np.allclose(pal[[1, 2, 0]], base)              # class 1 has the most GT pixels
    assert len({tuple(c) for c in RQ.class_palette(54)}) == 54
    assert RQ.nice_up(0.0347) == 0.035 and RQ.nice_up(0.98) == 0.98 and RQ.nice_up(1.0) == 1.0
    assert RQ.nice_up(0.0) == 1.0 and RQ.nice_up(float("nan")) == 1.0 and RQ.nice_up(123) == 130
    m = dict(covered=torch.tensor([True, True, False]),
             u=dict(u_ale=torch.tensor([0.01, float("nan"), 5.0])))
    assert RQ.u_max([m], "u_ale", 0.99) == 0.01            # not covered and undefined left out
    assert RQ.u_max([m], "u_ale", 0.99, fixed=0.2) == 0.2
    with pytest.raises(ValueError, match="evaluation frames"):
        RQ.main(toy["argv"] + ["--frame", "5", "--out", str(tmp_path / "x.pdf")])
    with pytest.raises(ValueError, match="not in"):
        RQ.main(toy["argv"] + ["--scene", "nope", "--out", str(tmp_path / "x.pdf")])
    print("PASS palette by scene GT pixels, nice colour-bar tops, frame and scene checks")
