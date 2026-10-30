"""T6: ScanNet as the held-out set of Table I v2 (usplat.datasets.ScanNetDataset,
prepare_scannet, eval_seg.py plumbing, configs/*scannet*.json, scripts/scannet_tab1.sh).

Synthetic ScanNet scans in the SensReader layout (color/ depth/ pose/ intrinsic/ label-filt/,
scannetv2-labels.combined.tsv two levels up, as in the download):
 * labels: raw ScanNet ids -> NYU40 through the tsv (unlisted ids -> 0), resized to the
   colour grid by centre-aligned nearest neighbour; index = frame number; K_depth;
 * stride / offset: mapping frames 0, 10, ... and evaluation frames 5, 25, ... are disjoint;
   make_dataset passes the offset and refuses it for loaders without one;
 * the tsv is searched in the scan and its two parents; label-filt without it is an error;
   without label-filt the frames are those of the old loader (labels None);
 * prepare_scannet: colour / 2 as usplat.datasets.downscale (image and K), the rest
   symlinked, idempotent, other settings and incomplete exports refused;
 * scannet_frame_view: the evaluation frames with a finite pose, so run_mapping.py's CSS
   vocabulary is exactly the classes eval_seg.py queries (not those of mapping frames);
 * eval_seg.py end to end on CPU (mock backend) with the real configs/vocab_scannet.json:
   offset frames only, the lost-tracking frame skipped, NYU40 ids outside the 19 classes
   ignored, no unknown ids;
 * the frozen vocabulary and scene lists (OpenGaussian's 19 classes and 10 scenes, 2 val /
   8 test, disjoint; paths those of the script);
 * scripts/scannet_tab1.sh on a throw-away git copy of the repo: dry runs print the protocol
   (mapping stride 10, CSS on the evaluation-frame view, eval stride 20 offset 5, --v2-rows
   iff eval_seg has it); the selection takes the settings of the validation run (another
   environment is refused) and records the make_tables choices and the sha256 of
   eval_val.json; the test stage refuses a dirty tree, a missing / stale selection, another
   protocol, code changed since the validation commit, a finished test record and another
   selection than an earlier attempt; val / select are refused once the test started;
 * the v2 selection (make_tables.selection under configs/scannet_v2_prereg.json, a stand-in
   with the same rule when T4a's make_tables.py is not in the tree): made by default with
   --v2-rows; the test stage refuses without it, with another choice or an edited prereg;
   the prereg's rules equal configs/tab1_v2_prereg.json's when that file exists.
"""
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest
import torch

from usplat.datasets import (ScanNetDataset, make_dataset, prepare_scannet, downscale,
                             scannet_label_lut)
from usplat.vmf_field import DirectionalSemanticField, FieldConfig

REPO = pathlib.Path(__file__).resolve().parents[1]
TSV_HEADER = "id\traw_category\tcategory\tcount\tnyu40id\teigen13id\tnyuClass\tnyu40class"
# raw ScanNet id -> NYU40 id (the real tsv's first rows: 1 wall, 2 chair, 3 floor ...)
RAW2NYU = {1: 1, 2: 5, 3: 2, 7: 39, 40: 40}
OPENGAUSSIAN_19 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]
OPENGAUSSIAN_SCENES = ["scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00",
                       "scene0140_00", "scene0200_00", "scene0347_00", "scene0400_00",
                       "scene0590_00", "scene0645_00"]


def _write_tsv(path):
    rows = [TSV_HEADER] + [f"{r}\tname{r}\tcat{r}\t10\t{n}\t1\tx\ty" for r, n in RAW2NYU.items()]
    rows.append("55\tblank\tcat\t1\t\t\t\t")                     # malformed: no nyu40id
    path.write_text("\n".join(rows) + "\n")


def _raw_labels(h, w, i=0):
    """Raw ids: left half 2 (chair), right half 1 (wall), top rows 7 (otherfurniture),
    a 999 (not in the tsv) block."""
    lab = np.ones((h, w), np.uint16)
    lab[:, : w // 2] = 2
    lab[: h // 6] = 7
    lab[h - 2:, :2] = 999
    lab[h // 2, (i % 3)] = 3                                    # a floor pixel (frame-dependent)
    return lab


def _fake_scan(root, n=6, hw=(48, 64), label_hw=(24, 32), lost=(), labels=True, tsv=True,
               fx=None, color_fn=None, quality=95):
    """A SensReader-layout scan <root> (tsv two levels up, as data/scannetv2-labels... beside
    scans/<scene>)."""
    import cv2
    H, W = hw
    for sub in ("color", "depth", "pose", "intrinsic") + (("label-filt",) if labels else ()):
        (root / sub).mkdir(parents=True, exist_ok=True)
    fx = fx or 0.9 * W
    K = np.eye(4)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fx, (W - 1) / 2, (H - 1) / 2
    np.savetxt(root / "intrinsic" / "intrinsic_color.txt", K)
    Kd = np.eye(4)
    Kd[0, 0], Kd[1, 1], Kd[0, 2], Kd[1, 2] = 0.89 * 80, 0.89 * 80, 39.5, 29.5  # depth: 80x60
    np.savetxt(root / "intrinsic" / "intrinsic_depth.txt", Kd)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = color_fn(i) if color_fn else rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        cv2.imwrite(str(root / "color" / f"{i}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        cv2.imwrite(str(root / "depth" / f"{i}.png"), np.full((480 // 8, 640 // 8), 2000 + i,
                                                              np.uint16))
        c2w = np.eye(4)
        c2w[0, 3] = 0.001 * i
        if i in lost:
            c2w[:] = -np.inf                                     # ScanNet: lost tracking
        np.savetxt(root / "pose" / f"{i}.txt", c2w)
        if labels:
            cv2.imwrite(str(root / "label-filt" / f"{i}.png"), _raw_labels(*label_hw, i))
    if tsv:
        _write_tsv(root.parent.parent / "scannetv2-labels.combined.tsv")
    return root


def _expected_labels(raw, h, w):
    lut = np.zeros(1000, np.int64)
    for r, c in RAW2NYU.items():
        lut[r] = c
    H, W = raw.shape
    r = np.minimum(((np.arange(h) + 0.5) * H / h).astype(int), H - 1)
    c = np.minimum(((np.arange(w) + 0.5) * W / w).astype(int), W - 1)
    return lut[raw.astype(np.int64)][r][:, c]


# ------------------------------------------------------------------------------ loader
def test_labels_nyu40_resized_index_and_kdepth(tmp_path):
    scan = _fake_scan(tmp_path / "data" / "scans" / "scene0001_00", n=3)
    ds = ScanNetDataset(str(scan))
    assert len(ds) == 3 and ds.label_map.endswith("scannetv2-labels.combined.tsv")
    fr = ds[2]
    assert fr["index"] == 2 and fr["image"].shape == (48, 64, 3)
    lab = fr["labels"]
    assert lab.dtype == torch.int64 and lab.shape == (48, 64)    # on the colour grid
    want = _expected_labels(_raw_labels(24, 32, 2), 48, 64)
    assert np.array_equal(lab.numpy(), want)
    assert set(np.unique(want)) == {0, 1, 2, 5, 39}                # 999 -> 0 (not in the tsv)
    assert fr["depth"].shape == (60, 80) and torch.allclose(fr["depth"], torch.tensor(2.002))
    assert fr["K_depth"][0, 0] == pytest.approx(0.89 * 80) and fr["K"][0, 0] == pytest.approx(0.9 * 64)
    assert torch.isfinite(fr["c2w"]).all() and fr["c2w"][0, 3] == pytest.approx(0.002)
    lut = scannet_label_lut(ds.label_map)
    assert lut[2] == 5 and lut[7] == 39 and len(lut) == 41       # malformed row 55 skipped
    print("PASS ScanNet labels: raw -> NYU40, centre nearest-neighbour to the colour grid")


def test_stride_offset_disjoint_and_lost_pose(tmp_path):
    scan = _fake_scan(tmp_path / "d" / "scans" / "s", n=50, lost=(25,))
    mapping = make_dataset("scannet", str(scan), stride=10)
    evals = make_dataset("scannet", str(scan), stride=20, offset=5)
    m_idx = [mapping[i]["index"] for i in range(len(mapping))]
    e_idx = [evals[i]["index"] for i in range(len(evals))]
    assert m_idx == [0, 10, 20, 30, 40] and e_idx == [5, 25, 45]
    assert not set(m_idx) & set(e_idx)
    assert not torch.isfinite(evals[1]["c2w"]).all()                # returned as is
    assert len(make_dataset("scannet", str(scan), stride=20, offset=49)) == 1
    assert len(make_dataset("scannet", str(scan), stride=20, offset=50)) == 0
    with pytest.raises(ValueError, match="no frame offset"):
        make_dataset("replica_semnerf", str(scan), stride=1, offset=5)
    with pytest.raises(ValueError, match="negative"):
        ScanNetDataset(str(scan), offset=-1)
    print("PASS ScanNet split: mapping 0,10,..,40 and evaluation 5,25,45 are disjoint")


def test_label_map_search_errors_and_unlabelled(tmp_path):
    scan = _fake_scan(tmp_path / "a" / "scans" / "s", n=2, tsv=False)
    with pytest.raises(FileNotFoundError, match="scannetv2-labels"):
        ScanNetDataset(str(scan))
    other = tmp_path / "elsewhere.tsv"
    _write_tsv(other)
    ds = ScanNetDataset(str(scan), label_map=str(other))
    assert ds[0]["labels"] is not None
    (scan / "scannetv2-labels.combined.tsv").write_text("id\tcategory\n1\twall\n")
    with pytest.raises(ValueError, match="nyu40id"):              # found in <root>, no column
        ScanNetDataset(str(scan))
    # no label-filt: the frames of the loader before labels (plus labels None, index)
    plain = _fake_scan(tmp_path / "b" / "scans" / "s", n=2, labels=False, tsv=False)
    fr = ScanNetDataset(str(plain))[1]
    assert fr["labels"] is None and fr["index"] == 1
    import cv2
    img = cv2.cvtColor(cv2.imread(str(plain / "color" / "1.jpg")), cv2.COLOR_BGR2RGB)
    assert np.array_equal(fr["image"].numpy(), img)
    K = np.loadtxt(plain / "intrinsic" / "intrinsic_color.txt")[:3, :3]
    assert np.allclose(fr["K"].numpy(), K)
    print("PASS ScanNet label map: found two levels up, explicit path, errors, unlabelled scan")


def test_prepare_scannet_matches_downscale(tmp_path):
    def smooth(i):                                   # JPEG-friendly content (noise is not)
        y, x = np.mgrid[0:96, 0:128]
        return np.stack([x * 2, y * 2, (x + y + 20 * i) % 256], -1).astype(np.uint8)
    src = _fake_scan(tmp_path / "data" / "scans" / "scene0002_00", n=4, hw=(96, 128),
                     label_hw=(48, 64), quality=100, color_fn=smooth)
    dst = tmp_path / "data" / "uags_f2" / "scene0002_00"
    rec = prepare_scannet(str(src), str(dst), factor=2, quality=100)
    assert rec["status"] == "ok" and rec["hw"] == [48, 64] and rec["native_hw"] == [96, 128]
    assert json.loads((dst / "prepare.json").read_text())["status"] == "ok"
    for name in ("depth", "pose", "label-filt", "scannetv2-labels.combined.tsv"):
        assert (dst / name).is_symlink(), name
    native, prep = ScanNetDataset(str(src)), ScanNetDataset(str(dst))
    assert len(prep) == 4 and prep.label_map == str(dst / "scannetv2-labels.combined.tsv")
    for i in range(4):
        a, b = downscale(native[i], 2), prep[i]
        assert b["image"].shape == a["image"].shape == (48, 64, 3)
        diff = (b["image"].float() - a["image"].float()).abs().mean().item()
        assert diff < 1.5, diff                                  # JPEG re-encode only
        assert torch.allclose(b["K"], a["K"], atol=1e-5)          # downscale's K rule
        assert np.array_equal(b["labels"].numpy(), _expected_labels(_raw_labels(48, 64, i), 48, 64))
        assert torch.equal(b["depth"], native[i]["depth"]) and torch.equal(b["c2w"], native[i]["c2w"])
        assert torch.equal(b["K_depth"], native[i]["K_depth"]) and b["index"] == i
    # the 3DGS fit's frames (scripts/fit_gs.py --max-res 0): prepared resolution, the depth
    # on the colour grid with K_depth rescaled to it, depth points back-projected
    from usplat import gs_fit
    frames, fac = gs_fit.load_frames(prep, max_res=0)
    Kd = native[0]["K_depth"].clone()
    Kd[0] *= 64 / 80
    Kd[1] *= 48 / 60
    assert fac == 1 and frames[0]["depth"].shape == (48, 64)
    assert torch.allclose(frames[0]["K_depth"], Kd)
    pts, _ = gs_fit.backproject_depth(frames[0], pixel_step=4)
    assert len(pts) > 100 and torch.allclose(pts[:, 2], torch.tensor(2.0), atol=1e-4)
    # idempotent rerun (fills in a missing frame only); other settings refused
    os.remove(dst / "color" / "3.jpg")
    mtime = os.path.getmtime(dst / "color" / "0.jpg")
    prepare_scannet(str(src), str(dst), factor=2, quality=100)
    assert (dst / "color" / "3.jpg").exists() and os.path.getmtime(dst / "color" / "0.jpg") == mtime
    with pytest.raises(ValueError, match="other settings"):
        prepare_scannet(str(src), str(dst), factor=4, quality=100)
    os.remove(src / "pose" / "2.txt")
    with pytest.raises(ValueError, match="no pose file"):
        prepare_scannet(str(src), str(tmp_path / "x"))
    os.remove(src / "color" / "1.jpg")
    with pytest.raises(ValueError, match="frame numbers"):
        prepare_scannet(str(src), str(tmp_path / "y"))
    print(f"PASS prepare_scannet: colour / 2 equals downscale(native, 2) up to JPEG "
          f"(mean abs diff {diff:.2f}); K, labels, depth, poses identical")


def test_frame_view_gives_css_the_evaluation_classes(tmp_path):
    """The CSS vocabulary (run_mapping.py css_vocabulary on the view, stride 1) equals the
    classes of the frames eval_seg.py evaluates: floor only in frames 10 and 15 (mapping /
    unused) and cabinet only in the lost evaluation frame 25 stay out; every 5th frame of the
    whole scan (the old rule) would add both."""
    import argparse
    import cv2
    import sys
    scan = _fake_scan(tmp_path / "d" / "scans" / "scene0004_00", n=50, hw=(24, 32),
                      label_hw=(24, 32), lost=(25,))
    with open(tmp_path / "d" / "scannetv2-labels.combined.tsv", "a") as f:
        f.write("5\tcabinet\tcabinet\t1\t3\t1\tx\ty\n")             # raw 5 -> NYU40 3 (cabinet)
    for i in range(50):
        lab = _raw_labels(24, 32, i)
        if i not in (10, 15):
            lab[lab == 3] = 1
        if i == 25:
            lab[10:14, 10:14] = 5
        cv2.imwrite(str(scan / "label-filt" / f"{i}.png"), lab)
    from usplat.datasets import scannet_frame_view
    view = tmp_path / "view"
    rec = scannet_frame_view(str(scan), str(view), stride=20, offset=5)
    assert rec["frames"] == [5, 45] and rec["lost"] == [25]
    assert sorted(os.listdir(view / "color")) == ["45.jpg", "5.jpg"]
    for name in ("depth", "pose", "label-filt", "intrinsic", "scannetv2-labels.combined.tsv"):
        assert (view / name).is_symlink(), name
    ds = ScanNetDataset(str(view))
    assert [ds[i]["index"] for i in range(len(ds))] == [5, 45]
    assert ds.label_map == str(view / "scannetv2-labels.combined.tsv")
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import run_mapping
    finally:
        sys.path.remove(str(REPO / "scripts"))
    css = lambda data, stride: run_mapping.css_vocabulary(argparse.Namespace(
        vocab=str(REPO / "configs" / "vocab_scannet.json"), dataset="scannet",
        css_eval_data=str(data), css_eval_stride=stride, extractor="lseg"))[0]
    cli = _load_cli()
    vocab = cli.load_vocab(str(REPO / "configs" / "vocab_scannet.json"))
    args = cli.build_parser().parse_args(["--vocab", "v", "--out", "o", "--downscale", "1"])
    frames, _, skipped = cli.load_frames(dict(name="s", data=str(scan), dataset="scannet",
                                              stride=20, offset=5), vocab, args)
    present = torch.unique(torch.cat([f["labels"].reshape(-1) for f in frames]))
    queried = [vocab["ids"][i] for i in present.tolist() if i >= 0]
    assert [f["idx"] for f in frames] == [5, 45] and skipped == 1
    assert css(view, 1) == queried == [1, 5]                       # wall, chair
    assert css(scan, 5) == [1, 2, 3, 5]                            # + floor, cabinet
    # idempotent; other settings and stray frames refused
    assert scannet_frame_view(str(scan), str(view), stride=20, offset=5) == rec
    with pytest.raises(ValueError, match="another"):
        scannet_frame_view(str(scan), str(view), stride=20, offset=0)
    os.symlink(scan / "color" / "10.jpg", view / "color" / "10.jpg")
    os.remove(view / "view.json")
    with pytest.raises(ValueError, match="outside the view"):
        scannet_frame_view(str(scan), str(view), stride=20, offset=5)
    print(f"PASS scannet_frame_view: CSS vocabulary {css(view, 1)} = the evaluation frames' "
          f"classes {queried}; every 5th frame would give {css(scan, 5)}")


# ---------------------------------------------------------------------------- eval_seg
def _load_cli():
    spec = importlib.util.spec_from_file_location("eval_seg", REPO / "scripts" / "eval_seg.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy_scannet_field(tmp_path):
    """A 640x480 synthetic scan (labels at 320x240) and a field on the plane z = 2 m whose
    left half is 'chair', right half 'wall' (texts e0, e1 of a 16-d space, identity PCA)."""
    from usplat.feature_pca import StreamingPCA
    scan = _fake_scan(tmp_path / "data" / "scans" / "scene0003_00", n=50, hw=(480, 640),
                      label_hw=(240, 320), lost=(25,), fx=585.0,
                      color_fn=lambda i: np.zeros((480, 640, 3), np.uint8))
    D, d = 16, 8
    pca = StreamingPCA(D, d)
    pca.basis = torch.eye(D)[:, :d]
    t_wall, t_chair, t_floor = torch.eye(D)[0], torch.eye(D)[1], torch.eye(D)[2]
    xs = torch.arange(-1.15, 1.2, 0.04)
    ys = torch.arange(-0.9, 0.95, 0.04)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    means = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.full((gx.numel(),), 2.0)], -1)
    N = means.shape[0]
    torch.save(dict(means=means, quats=torch.tensor([1.0, 0, 0, 0]).expand(N, 4).clone(),
                    scales=torch.full((N, 3), 0.03), opacities=torch.full((N,), 0.9)),
               tmp_path / "gs.pt")
    right = means[:, 0] > 0
    feat = torch.where(right[:, None], pca.project(t_wall[None]), pca.project(t_chair[None]))
    f = DirectionalSemanticField(N, d, FieldConfig(change_test=False))
    f.set_centre(torch.zeros(d))
    for _ in range(4):
        f.update_sparse(torch.arange(N), torch.ones(N), feat, num_pixels=100)
    torch.save(dict(field=f.state_dict(), pca=pca.state_dict()), tmp_path / "field.pt")
    vocab = json.loads((REPO / "configs" / "vocab_scannet.json").read_text())
    tpl = vocab["templates_by_extractor"]["lseg"]
    key = lambda p: f"lseg|{'|'.join(tpl)}|{p}"
    torch.save({key("wall"): t_wall, key("chair"): t_chair, key("floor"): t_floor},
               tmp_path / "text.pt")                          # no extractor is loaded
    scenes = [dict(name="scene0003_00", field=str(tmp_path / "field.pt"),
                   ply=str(tmp_path / "gs.pt"), data=str(scan), dataset="scannet",
                   stride=20, offset=5)]
    (tmp_path / "scenes.json").write_text(json.dumps(scenes))
    return scenes[0]


def test_eval_seg_scannet_end_to_end(tmp_path):
    cli = _load_cli()
    sc = _toy_scannet_field(tmp_path)
    vocab = cli.load_vocab(str(REPO / "configs" / "vocab_scannet.json"))
    args = cli.build_parser().parse_args(["--vocab", "v", "--out", "o", "--downscale", "16"])
    frames, n_unknown, skipped = cli.load_frames(sc, vocab, args)
    assert [f["idx"] for f in frames] == [5, 45] and skipped == 1 and n_unknown == 0
    lab = frames[0]["labels"]
    assert lab.shape == (30, 40)
    names = {vocab["names"][i] for i in torch.unique(lab).tolist() if i >= 0}
    assert names == {"wall", "chair", "floor"} or names == {"wall", "chair"}
    assert (lab[:5] == -1).all()                                  # otherfurniture: ignored
    out = tmp_path / "out" / "eval.json"
    cli.main(["--vocab", str(REPO / "configs" / "vocab_scannet.json"), "--backend", "mock",
              "--device", "cpu", "--downscale", "16", "--tau-cov", "0.5", "--extractor", "lseg",
              "--text-cache", str(tmp_path / "text.pt"), "--queries", "present",
              "--rows", "ours_P0,ours_P1", "--scenes", str(tmp_path / "scenes.json"),
              "--val-scenes", str(tmp_path / "scenes.json"), "--plain-json", "--out", str(out)])
    res = json.loads(out.read_text())["scenes"]["scene0003_00"]
    assert res["n_views"] == 2 and res["n_skipped_frames"] == 1
    assert res["n_unknown_label_pixels"] == 0
    assert set(res["classes"]) <= {"wall", "chair", "floor"} and {"wall", "chair"} <= set(res["classes"])
    assert res["rows"]["ours_P0"]["all"]["acc"] > 0.9, res["rows"]["ours_P0"]["all"]
    print(f"PASS eval_seg on ScanNet: views 5 and 45 (25 lost), classes {res['classes']}, "
          f"acc {res['rows']['ours_P0']['all']['acc']:.3f}")


# ------------------------------------------------------------------- frozen configs
def test_vocab_and_scene_configs():
    cli = _load_cli()
    v = cli.load_vocab(str(REPO / "configs" / "vocab_scannet.json"))
    assert v["ids"] == OPENGAUSSIAN_19
    assert v["names"][:3] == ["wall", "floor", "cabinet"] and v["prompts"][15] == "shower curtain"
    assert v["ignore"] == set(range(41)) - set(OPENGAUSSIAN_19) and 39 in v["ignore"]
    rep = json.loads((REPO / "configs" / "vocab_replica.json").read_text())
    assert v["templates_by_extractor"] == rep["templates_by_extractor"]  # nothing re-tuned
    lab, unk = cli.map_labels(torch.arange(41)[None], v)
    assert unk == 0 and (lab[0, OPENGAUSSIAN_19] >= 0).all() and int((lab >= 0).sum()) == 19
    val = json.loads((REPO / "configs" / "scannet_val.json").read_text())
    test = json.loads((REPO / "configs" / "scannet_test.json").read_text())
    vn, tn = [s["name"] for s in val], [s["name"] for s in test]
    assert vn == OPENGAUSSIAN_SCENES[:2] and len(tn) == 8 and not set(vn) & set(tn)
    assert sorted(vn + tn) == OPENGAUSSIAN_SCENES
    script = (REPO / "scripts" / "scannet_tab1.sh").read_text()
    var = lambda k: next(ln.split("=", 1)[1] for ln in script.splitlines() if ln.startswith(k + "="))
    data = var("DATA")
    roots = dict(prep=var("PREP").replace("$DATA", data), maps=var("MAPS"), fields=var("FIELDS"))
    for s in val + test:
        assert (s["dataset"], s["stride"], s["offset"]) == ("scannet", 20, 5)
        assert s["data"] == f"{roots['prep']}/{s['name']}"
        assert s["ply"] == f"{roots['maps']}/{s['name']}.pt"
        assert s["field"] == f"{roots['fields']}/{s['name']}_lseg.pt"
    assert data == "/home/thanhnc/gauss_ws/src/uags/data/scannet"
    assert "/results/scannet_tab1}" in var("OUT")
    print(f"PASS ScanNet configs: 19 classes, {len(v['ignore'])} ignored ids, val {vn}, "
          f"test {len(tn)} scenes")


# ------------------------------------------------------------------------- the script
def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
                           *a], capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope="module")
def script_repo(tmp_path_factory):
    """A committed copy of scripts/ usplat/ configs/ in a throw-away git repo."""
    repo = tmp_path_factory.mktemp("repo")
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in ("scripts", "usplat", "configs"):
        shutil.copytree(REPO / d, repo / d, ignore=ign)
    # empty map and field folders, in the script and in its scene configs: the dry run then
    # plans every fit and mapping, whatever the real results/ holds (its ScanNet maps exist)
    sh = repo / "scripts" / "scannet_tab1.sh"
    cfgs = [repo / "configs" / f"scannet_{s}.json" for s in ("val", "test")]
    t, ct = sh.read_text(), [c.read_text() for c in cfgs]
    for k in ("MAPS", "FIELDS"):
        line = next(ln for ln in t.splitlines() if ln.startswith(f"{k}="))
        (repo / k.lower()).mkdir()
        t = t.replace(line, f"{k}={repo / k.lower()}")
        ct = [c.replace(line.split("=", 1)[1] + "/", f"{repo / k.lower()}/") for c in ct]
    sh.write_text(t)
    for c, x in zip(cfgs, ct):
        c.write_text(x)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "copy")
    return repo


def _run(repo, out, **env):
    e = dict(os.environ, OUT=str(out), DRY_RUN="1", PYTHONDONTWRITEBYTECODE="1")
    e.pop("PYTHONPATH", None)
    e.update({k: str(v) for k, v in env.items()})
    return subprocess.run(["bash", str(repo / "scripts" / "scannet_tab1.sh")], env=e,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          timeout=600)                        # stdout: both streams


# the Table I v2 selection API of scripts/make_tables.py (wip/t4a-predict): appended to the
# throw-away copy only when its make_tables.py has none (the T6 branch alone); with T4a
# merged the script runs against the real one
MAKE_TABLES_V2_STANDIN = '''

if "selection" not in globals():         # test stand-in (tests/test_scannet_loader.py)
    import hashlib as _hashlib
    import os as _os
    V2_KEYS = ("@p1", "@p4", "@ours_v2")

    def load_prereg(path):
        with open(path) as f:
            return json.load(f)

    def choose_v2(b, key, p):
        tol, vals = float(p["tie"]["tol"]), {}
        for c in p["choices"][key]["rows"]:
            r = choose_v2(b, c, p)[0] if c.startswith("@") else c
            ms = [sc["rows"][r]["all"] for sc in b["scenes"].values()]
            vals[c] = [r, sum(m["nll"] * m["n_pixels"] for m in ms)
                       / sum(m["n_pixels"] for m in ms)]
        best = min(v for _, v in vals.values())
        first = next(c for c, (_, v) in vals.items() if v <= best + tol)
        return vals[first][0], vals

    def selection(val_path, prereg_path=None):
        p, b = load_prereg(prereg_path), body(val_path)
        sha = lambda q: _hashlib.sha256(open(q, "rb").read()).hexdigest()
        return dict(prereg=dict(path=_os.path.abspath(prereg_path), sha256=sha(prereg_path)),
                    validation=dict(path=_os.path.abspath(val_path), sha256=sha(val_path)),
                    chosen={k: choose_v2(b, k, p)[0] for k in V2_KEYS})
'''
PREREG = REPO / "configs" / "scannet_v2_prereg.json"


def _v2_rows():
    p = json.loads(PREREG.read_text())
    rows = {r for c in p["choices"].values() for r in c["rows"] if not r.startswith("@")}
    return sorted(rows | set(p["ablations"]))


# validation NLL of the v2 candidates (pooled over the two scenes): @p1 = P1 c0 0.1,
# @p4 = P4 alpha0 0.1 lambda 1, @ours_v2 = @p1 (P2 0.80, P2p1 0.79); the rest 0.9 and above
V2_NLL = {"ours_P2": 0.80, "ours_P2p1": 0.79, "ours_P1_c0.1": 0.70, "ours_P1_c0.03": 0.72,
          "ours_P4_a0.1_s1": 0.76, "ours_P4_inf": 1.5}
V2_CHOSEN = {"@p1": "ours_P1_c0.1", "@p4": "ours_P4_a0.1_s1", "@ours_v2": "ours_P1_c0.1"}


def _fake_eval_val(path, repo, v2=False, tau_cov=0.5, drop_beta=None, git=None):
    """A finished eval_seg record of STAGE=val: its config, git state (a commit of `repo`),
    betas, summary (every summary metric of eval_seg.py) for the make_tables choices and,
    per validation scene, the v2 candidates' NLL and labelled pixels."""
    keys = ("miou", "acc", "ece", "brier", "aurc", "coverage", "n_tied", "n_uncovered")

    def row(nll, ause=None):
        m = {k: {"mean": 0.5, "std": 0.0, "n": 2} for k in keys}
        m["nll"] = {"mean": nll, "std": 0.0, "n": 2}
        m["ause"] = {} if ause is None else {"legs_u": {"mean": ause, "std": 0.0, "n": 2}}
        return {"all": m, "covered": m}
    summ = {"css": row(0.9), "css_a0.5": row(0.7), "legs_l1": row(1.0, 0.30),
            "legs_l2": row(1.0, 0.20), "legs_l4": row(1.0, 0.25), "ours_P0": row(0.8),
            "ours_P2": row(0.8)}
    betas = {"ours_P0": 10.0, "ours_P2": 10.0, "css_a0.5": 1.0}
    v2rows = _v2_rows() if v2 else []
    nll = {r: V2_NLL.get(r, 0.9 + 0.01 * i) for i, r in enumerate(v2rows)}
    nll["ours_P2"] = V2_NLL["ours_P2"]
    scenes = {s: {"rows": {r: {"all": {"nll": v + d, "n_pixels": n}} for r, v in nll.items()}}
              for s, d, n in (("scene0000_00", 0.0, 1000), ("scene0062_00", 0.02, 3000))}
    betas.update({r: 5.0 for r in v2rows if r != drop_beta})
    cfg = dict(dataset="scannet", vocab="configs/vocab_scannet.json", stride=20, downscale=2,
               tau_cov=tau_cov, val_scenes="configs/scannet_val.json",
               scenes="configs/scannet_val.json")
    head = _git(repo, "rev-parse", "HEAD").strip()
    rec = dict(status="ok", argv=["scripts/eval_seg.py"] + (["--v2-rows"] if v2 else []),
               config=cfg, git=git or dict(commit=head, dirty=False, dirty_tracked=False),
               metrics=dict(betas=betas, scenes=scenes, summary=summ,
                            beta_fit_settings={"tau_cov": tau_cov, "field_identities": ["abc"]}))
    path.write_text(json.dumps(rec))


def _settings(repo, out, **env):
    """The protocol JSON the script prints (a dry STAGE=prep: no other check runs)."""
    r = _run(repo, out, STAGE="prep", **env)
    assert r.returncode == 0, r.stdout
    return json.loads(next(ln for ln in r.stdout.splitlines()
                           if ln.startswith("protocol: "))[len("protocol: "):])


def _val_run(repo, out, **kw):
    """What a finished STAGE=val leaves before its selection: eval_val.json and the settings
    record protocol_val.json (the script's record_protocol)."""
    import hashlib
    _fake_eval_val(out / "eval_val.json", repo, **kw)
    sha = hashlib.sha256((out / "eval_val.json").read_bytes()).hexdigest()
    (out / "protocol_val.json").write_text(json.dumps(dict(
        status="ok", eval_val_sha256=sha, protocol=_settings(repo, out))))


def _eval_seg_has_v2():
    return "--v2-rows" in _load_cli().build_parser().format_help()


def _has_v2_selection(repo):
    src = (repo / "scripts" / "make_tables.py").read_text()
    return "def choose_v2" in src and "def selection" in src


def test_script_dry_run_val(script_repo, tmp_path):
    assert subprocess.run(["bash", "-n", str(REPO / "scripts" / "scannet_tab1.sh")]).returncode == 0
    r = _run(script_repo, tmp_path / "o", STAGE="bogus")
    assert r.returncode == 2, r.stdout
    r = _run(script_repo, tmp_path / "o", STAGE="val")
    assert r.returncode == 0, r.stdout
    o = r.stdout
    assert o.count("prepare_scannet") == 2 and "scenes: scene0000_00 scene0062_00" in o
    assert o.count("scannet_frame_view") == 2 and "eval_s20o5 (stride 20, offset 5)" in o
    fit = next(ln for ln in o.splitlines() if "fit_gs.py" in ln)
    assert "--dataset scannet" in fit and "--stride 10" in fit and "--max-res 0" in fit
    mp = next(ln for ln in o.splitlines() if "run_mapping.py" in ln)
    assert "--stride 10" in mp and "vocab_scannet.json" in mp
    assert "--css-eval-data /home/thanhnc/gauss_ws/src/uags/data/scannet/uags_f2/scene0000_00/eval_s20o5 " \
           "--css-eval-stride 1 " in mp
    ev = next(ln for ln in o.splitlines() if "eval_seg.py" in ln)
    assert "--val-scenes configs/scannet_val.json" in ev and "--with-2d" in ev
    v2 = _eval_seg_has_v2()
    assert ("--v2-rows" in ev) == v2 and f"gpu_gate.sh {10000 if v2 else 7000} " in ev
    assert o.count("distill_features.py") == 2 * 4                 # f3dgs + 3 lambda_u per scene
    r0 = _run(script_repo, tmp_path / "o", STAGE="val", V2_ROWS=0)
    assert r0.returncode == 0 and "--v2-rows" not in r0.stdout and "gpu_gate.sh 7000 " in r0.stdout
    r1 = _run(script_repo, tmp_path / "o", STAGE="val", V2_ROWS=1)
    if v2:
        assert r1.returncode == 0 and "--v2-rows" in r1.stdout
    else:
        assert r1.returncode == 1 and "refused" in r1.stdout
    assert _run(script_repo, tmp_path / "o", STAGE="val", V2_ROWS="x").returncode == 2
    assert not any((tmp_path / "o").iterdir())                    # a dry run writes nothing
    # the validation evaluation must name a commit (the test stage compares the code with it)
    with open(script_repo / "usplat" / "datasets.py", "a") as f:
        f.write("\n")
    try:
        r = _run(script_repo, tmp_path / "o", STAGE="val")
        assert r.returncode == 1 and "uncommitted changes" in r.stdout, r.stdout
        assert _run(script_repo, tmp_path / "o", STAGE="val", EVAL=0).returncode == 0
    finally:
        _git(script_repo, "checkout", "--", "usplat/datasets.py")
    print("PASS scannet_tab1.sh STAGE=val dry run: protocol strides, CSS = evaluation frames, "
          "V2_ROWS, clean tree")


def test_script_selection_and_test_refusals(script_repo, tmp_path):
    import hashlib
    out = tmp_path / "o"
    out.mkdir()
    head = _git(script_repo, "rev-parse", "HEAD").strip()
    r = _run(script_repo, out, STAGE="test")
    assert r.returncode == 1 and "refused: no finished validation selection" in r.stdout, r.stdout
    # select needs the settings record of the validation run, and those settings now
    _fake_eval_val(out / "eval_val.json", script_repo)
    r = _run(script_repo, out, STAGE="select", DRY_RUN=0)
    assert r.returncode == 1 and "no settings record" in r.stdout, r.stdout
    _val_run(script_repo, out)
    for env, what in ((dict(TAU_COV=0.3), "tau_cov"), (dict(MAP_EXTRA="--some-flag"), "map_extra")):
        r = _run(script_repo, out, STAGE="select", DRY_RUN=0, **env)
        assert r.returncode == 1 and "differ from the validation run" in r.stdout and what in r.stdout
    assert not (out / "selection_val.json").exists()
    r = _run(script_repo, out, STAGE="select")                     # dry: printed, not written
    assert r.returncode == 0 and "not written" in r.stdout and not (out / "selection_val.json").exists()
    r = _run(script_repo, out, STAGE="select", DRY_RUN=0)
    assert r.returncode == 0, r.stdout
    sel = json.loads((out / "selection_val.json").read_text())
    assert sel["status"] == "ok" and sel["v2_rows"] is False and "v2" not in sel
    assert sel["eval_val_sha256"] == hashlib.sha256((out / "eval_val.json").read_bytes()).hexdigest()
    assert sel["choices"]["legs"] == "legs_l2" and sel["choices"]["css"] == "css_a0.5"
    assert sel["choices"]["p3"] == "ours_P3_b20"                   # default: no P3 rows
    p = sel["protocol"]
    assert (p["map_stride"], p["eval_stride"], p["eval_offset"], p["css_frames"],
            p["downscale"], p["prep_factor"], p["tau_cov"]) == (10, 20, 5, "eval_s20o5", 2, 2, 0.5)
    assert set(p["configs"]) == {"configs/vocab_scannet.json", "configs/scannet_val.json",
                                 "configs/scannet_test.json", "configs/scannet_v2_prereg.json"}
    assert sel["eval_val_git"]["commit"] == head
    # the test stage passes every check on the clean copy and runs once
    r = _run(script_repo, out, STAGE="test")
    assert r.returncode == 0, r.stdout
    ev = next(ln for ln in r.stdout.splitlines() if "eval_seg.py" in ln)
    assert "--betas" in ev and "configs/scannet_test.json" in ev and "--v2-rows" not in ev
    assert "--method legs --lam-u 2" in r.stdout and "--lam-u 1" not in r.stdout.replace(
        "--method f3dgs --lam-u 1", "")
    assert r.stdout.count("prepare_scannet") == 8 and "look record" in r.stdout
    assert not (out / "look_test.json").exists()                   # dry: nothing written
    # another protocol setting than the selection's
    r = _run(script_repo, out, STAGE="test", DOWNSCALE=4)
    assert r.returncode == 1 and "protocol differs" in r.stdout
    r = _run(script_repo, out, STAGE="test", MAP_EXTRA="--css-vocab full")
    assert r.returncode == 1 and "map_extra" in r.stdout
    for flag in ("--css-full-vocab", "--css-full", "--seed 0 --css-f"):   # no css_full look
        for stage in ("val", "test", "prep"):
            r = _run(script_repo, out, STAGE=stage, MAP_EXTRA=flag)
            assert r.returncode == 1 and "validation-only control" in r.stdout, r.stdout
    # eval_val.json changed after the selection; one not made with the recorded settings
    orig = (out / "eval_val.json").read_text()
    _fake_eval_val(out / "eval_val.json", script_repo, tau_cov=0.3)
    r = _run(script_repo, out, STAGE="test")
    assert r.returncode == 1 and "changed after the selection" in r.stdout
    proto_orig = (out / "protocol_val.json").read_text()
    proto = json.loads(proto_orig)
    proto["eval_val_sha256"] = hashlib.sha256((out / "eval_val.json").read_bytes()).hexdigest()
    (out / "protocol_val.json").write_text(json.dumps(proto))
    r = _run(script_repo, out, STAGE="select")
    assert r.returncode == 1 and "was not made with these settings" in r.stdout and "tau_cov" in r.stdout
    (out / "eval_val.json").write_text(orig)
    (out / "protocol_val.json").write_text(proto_orig)
    # the code: the validation record named a clean commit, scripts/ usplat/ configs/ unchanged
    with open(script_repo / "usplat" / "rows.py", "a") as f:
        f.write("\n# changed after the validation run\n")
    _git(script_repo, "commit", "-qam", "change the evaluation code")
    try:
        r = _run(script_repo, out, STAGE="test")
        assert r.returncode == 1 and "differ from the commit" in r.stdout, r.stdout
    finally:
        _git(script_repo, "reset", "-q", "--hard", head)
    _fake_eval_val(out / "eval_val.json", script_repo,
                   git=dict(commit=head, dirty=True, dirty_tracked=True))
    sel_orig = (out / "selection_val.json").read_text()
    s = json.loads(sel_orig)
    s["eval_val_sha256"] = hashlib.sha256((out / "eval_val.json").read_bytes()).hexdigest()
    (out / "selection_val.json").write_text(json.dumps(s))
    r = _run(script_repo, out, STAGE="test")
    assert r.returncode == 1 and "was made with uncommitted changes" in r.stdout, r.stdout
    (out / "eval_val.json").write_text(orig)
    # the rules must still give the recorded choices
    s = json.loads(sel_orig)
    s["choices"]["legs"] = "legs_l4"
    (out / "selection_val.json").write_text(json.dumps(s))
    r = _run(script_repo, out, STAGE="test")
    assert r.returncode == 1 and "the rules give" in r.stdout
    (out / "selection_val.json").write_text(sel_orig)
    assert _run(script_repo, out, STAGE="test").returncode == 0
    # once the test evaluation started (look record, a record in any status, partial
    # results), val and select are frozen; a crashed test may be rerun with the same selection
    for name, content in (("eval_test.json.partial.json", {"scenes": {"scene0070_00": {}}}),
                          ("eval_test.json", {"status": "failed"}),
                          ("look_test.json", {"selection_sha256": hashlib.sha256(
                              sel_orig.encode()).hexdigest(), "attempts": [{}]})):
        (out / name).write_text(json.dumps(content))
        for st in ("val", "select"):
            r = _run(script_repo, out, STAGE=st)
            assert r.returncode == 1 and "frozen" in r.stdout, (name, st, r.stdout)
        r = _run(script_repo, out, STAGE="test")
        assert r.returncode == 0, (name, r.stdout)
        os.remove(out / name)
    (out / "look_test.json").write_text(json.dumps({"selection_sha256": "0" * 64, "attempts": [{}]}))
    r = _run(script_repo, out, STAGE="test")
    assert r.returncode == 1 and "started before with another selection" in r.stdout
    os.remove(out / "look_test.json")
    (out / "eval_test.json").write_text(json.dumps({"status": "ok"}))
    r = _run(script_repo, out, STAGE="test")
    assert r.returncode == 1 and "finished test run" in r.stdout
    os.remove(out / "eval_test.json")
    # a dirty tree is refused (tracked file under scripts/ usplat/ configs/ modified)
    with open(script_repo / "configs" / "scannet_val.json", "a") as f:
        f.write("\n")
    try:
        r = _run(script_repo, out, STAGE="test")
        assert r.returncode == 1 and "uncommitted changes" in r.stdout
    finally:
        _git(script_repo, "checkout", "--", "configs/scannet_val.json")
    # Table I from the finished test record of these betas
    r = _run(script_repo, out, STAGE="table")
    assert r.returncode == 1 and "no finished test run" in r.stdout
    test_rec = dict(status="ok", metrics=dict(betas_path=str(out / "elsewhere.json"),
                                              summary=json.loads(orig)["metrics"]["summary"]))
    (out / "eval_test.json").write_text(json.dumps(test_rec))
    r = _run(script_repo, out, STAGE="table")
    assert r.returncode == 1 and "evaluated with the betas of" in r.stdout
    test_rec["metrics"]["betas_path"] = str(out / "eval_val.json")
    (out / "eval_test.json").write_text(json.dumps(test_rec))
    r = _run(script_repo, out, STAGE="table", DRY_RUN=0)
    assert r.returncode == 0, r.stdout
    tex = (out / "table1_scannet.tex").read_text()
    assert tex.startswith("% generated by scripts/make_tables.py tab1") and "--v2" not in tex
    print("PASS scannet_tab1.sh: selection from the validation run's settings (forged settings "
          "and dry runs refused / not written); test refuses no/stale selection, other "
          "protocol, changed or dirty code, other choices, another selection than an earlier "
          "attempt, a finished record, a dirty tree; val/select frozen once the test started")


@pytest.fixture(scope="module")
def v2_repo(tmp_path_factory):
    """script_repo with the v2 selection API in make_tables.py (the real one when T4a is
    merged, else MAKE_TABLES_V2_STANDIN)."""
    repo = tmp_path_factory.mktemp("repo_v2")
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in ("scripts", "usplat", "configs"):
        shutil.copytree(REPO / d, repo / d, ignore=ign)
    if not _has_v2_selection(repo):
        with open(repo / "scripts" / "make_tables.py", "a") as f:
            f.write(MAKE_TABLES_V2_STANDIN)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "copy")
    return repo


def test_script_v2_selection(v2_repo, script_repo, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    # a v2 validation needs a beta for every v2 row of the prereg
    _val_run(v2_repo, out, v2=True, drop_beta="ours_P4_inf")
    r = _run(v2_repo, out, STAGE="select", DRY_RUN=0)
    assert r.returncode == 1 and "no fitted beta for the v2 rows ['ours_P4_inf']" in r.stdout, r.stdout
    _val_run(v2_repo, out, v2=True)
    r = _run(v2_repo, out, STAGE="select", DRY_RUN=0)
    assert r.returncode == 0, r.stdout
    sel = json.loads((out / "selection_val.json").read_text())
    import hashlib
    assert sel["v2_rows"] is True and sel["v2"]["chosen"] == V2_CHOSEN
    assert sel["v2"]["prereg"]["sha256"] == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert sel["v2"]["validation"]["sha256"] == sel["eval_val_sha256"]
    sel_orig = (out / "selection_val.json").read_text()
    # the test stage: --v2-rows with the GPU cap of the v2 rows (when eval_seg has the flag)
    r = _run(v2_repo, out, STAGE="test")
    if _eval_seg_has_v2():
        assert r.returncode == 0, r.stdout
        ev = next(ln for ln in r.stdout.splitlines() if "eval_seg.py" in ln)
        assert "--v2-rows" in ev and "gpu_gate.sh 10000 " in ev
    else:
        assert r.returncode == 1 and "does not have" in r.stdout, r.stdout
    # refused: no v2 selection, another v2 choice, the prereg edited (and committed) after it
    s = json.loads(sel_orig)
    del s["v2"]
    (out / "selection_val.json").write_text(json.dumps(s))
    r = _run(v2_repo, out, STAGE="test")
    assert r.returncode == 1 and "holds no v2 selection" in r.stdout, r.stdout
    s = json.loads(sel_orig)
    s["v2"]["chosen"]["@ours_v2"] = "ours_P2"
    (out / "selection_val.json").write_text(json.dumps(s))
    r = _run(v2_repo, out, STAGE="test")
    assert r.returncode == 1 and "the v2 rules give" in r.stdout, r.stdout
    (out / "selection_val.json").write_text(sel_orig)
    head = _git(v2_repo, "rev-parse", "HEAD").strip()
    pre = v2_repo / "configs" / "scannet_v2_prereg.json"
    p = json.loads(pre.read_text())
    p["tie"]["tol"] = 0.5
    pre.write_text(json.dumps(p))
    _git(v2_repo, "commit", "-qam", "edit the prereg after the selection")
    summ = json.loads((out / "eval_val.json").read_text())["metrics"]["summary"]
    (out / "eval_test.json").write_text(json.dumps(dict(status="ok", metrics=dict(
        betas_path=str(out / "eval_val.json"), summary=summ))))
    try:
        r = _run(v2_repo, out, STAGE="test")                       # (finished record aside)
        assert r.returncode == 1 and "finished test run" in r.stdout, r.stdout
        os.rename(out / "eval_test.json", out / "eval_test.done")
        r = _run(v2_repo, out, STAGE="test")
        assert r.returncode == 1 and "protocol differs" in r.stdout, r.stdout
        assert "scannet_v2_prereg.json" in r.stdout
        os.rename(out / "eval_test.done", out / "eval_test.json")
        r = _run(v2_repo, out, STAGE="table")                      # no protocol check there
        assert r.returncode == 1 and "changed after the selection" in r.stdout, r.stdout
    finally:
        _git(v2_repo, "reset", "-q", "--hard", head)
    if _has_v2_selection(script_repo):    # the real make_tables.py (T4a merged): v2 rows
        r = _run(v2_repo, out, STAGE="table")
        assert r.returncode == 0 and "--v2" in r.stdout, r.stdout
    os.remove(out / "eval_test.json")
    # make_tables.py without the v2 selection: a v2 validation cannot be selected
    if not _has_v2_selection(script_repo):
        out2 = tmp_path / "o2"
        out2.mkdir()
        _val_run(script_repo, out2, v2=True)
        r = _run(script_repo, out2, STAGE="select", DRY_RUN=0)
        assert r.returncode == 1 and "no Table I v2 selection" in r.stdout, r.stdout
    # with T4a merged: STAGE=val refuses an untracked prereg before any GPU step (a modified
    # one is refused as a dirty tree)
    if _eval_seg_has_v2():
        _git(v2_repo, "rm", "-q", "--cached", "configs/scannet_v2_prereg.json")
        _git(v2_repo, "commit", "-qm", "untrack the prereg")
        try:
            r = _run(v2_repo, tmp_path / "o3", STAGE="val")
            assert r.returncode == 1 and "is not committed" in r.stdout, r.stdout
        finally:
            _git(v2_repo, "reset", "-q", "--hard", head)
    print(f"PASS scannet_tab1.sh v2: selection {sel['v2']['chosen']} under the ScanNet prereg; "
          f"test refuses a missing / other v2 choice and an edited prereg")


def test_scannet_prereg_copies_the_v2_rules():
    """configs/scannet_v2_prereg.json: the ScanNet scene files, and the rules of
    configs/tab1_v2_prereg.json (when that file is in the tree, i.e. T4a merged)."""
    p = json.loads(PREREG.read_text())
    assert p["scenes"] == {"validation": "configs/scannet_val.json",
                           "test": "configs/scannet_test.json"}
    assert set(p["choices"]) == {"@p1", "@p4", "@ours_v2"} and p["metric"]["pooling"] == "pixels"
    assert p["choices"]["@ours_v2"]["rows"][0] == "ours_P2" and "v1" not in p  # first look
    rep = REPO / "configs" / "tab1_v2_prereg.json"
    if rep.exists():
        r = json.loads(rep.read_text())
        for k in ("grids", "metric", "tie", "choices", "ablations"):
            assert p[k] == r[k], k
    print(f"PASS ScanNet v2 prereg: {len(_v2_rows())} rows; rules "
          f"{'equal to configs/tab1_v2_prereg.json' if rep.exists() else '(Replica prereg not in this tree)'}")


def test_script_refuses_config_paths_not_its_own(script_repo, tmp_path):
    cfg = script_repo / "configs" / "scannet_test.json"
    orig = cfg.read_text()
    for key, bad in (("data", "/somewhere/else"), ("offset", 0)):  # offset 0: mapping frames
        sc = json.loads(orig)
        sc[0][key] = bad
        cfg.write_text(json.dumps(sc))
        try:
            r = _run(script_repo, tmp_path / "o", STAGE="prep")
            assert r.returncode == 1 and "does not match" in r.stdout and key in r.stdout, r.stdout
        finally:
            cfg.write_text(orig)
    r = _run(script_repo, tmp_path / "o", STAGE="prep")
    assert r.returncode == 0 and r.stdout.count("prepare_scannet") == 10, r.stdout
    print("PASS scannet_tab1.sh: scene configs must name the script's paths and evaluation "
          "frames; prep = 10 scans")
