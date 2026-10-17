"""RIO10 loader (usplat.datasets_rio) on synthetic layouts of the documented format.

No RIO10 data here: every test writes a tiny scene in the readme's layout (seqXX.zip,
semanticsXX.zip, modelsXX.zip unzipped side by side, or merged) and checks
  frames     frame dicts like usplat.datasets (image, depth, c2w, K, index, name), the
             rendered depth only on request (it is the rescan's ORACLE), instances, parts
             (first / second half), stride, the "::part" root suffix
  intrinsics camera.yaml, intrinsics.txt fallback, principal point inside the image
  checks     every format assumption fails loudly with the offending path (non-rigid pose,
             missing pose, wrong folder name, RGB instance image, 8-bit depth, missing
             semantics frame, unresolvable layout lists the tried paths)
  models     labels.ply instance property (objectId; "label" only when named: it may hold
             class ids; error), instances.txt
  3RScan     removed / nonrigid / rigid lists, transform storage order detection
"""
import json
import os

import numpy as np
import pytest
import torch

from usplat import datasets_rio as R

H, W = 48, 27                                    # portrait, like RIO10's 960x540
FX, FY, CX, CY = 40.0, 41.0, 13.2, 24.7


def _pose(i):
    a = 0.1 * i
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4)
    m[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    m[:3, 3] = [0.1 * i, -0.05 * i, 0.02]
    return m


def write_ply(path, pts, ids, prop="objectId"):
    from plyfile import PlyData, PlyElement
    v = np.zeros(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), (prop, "u2")])
    v["x"], v["y"], v["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    v[prop] = ids
    os.makedirs(os.path.dirname(path), exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")]).write(str(path))


def write_rio(root, scene="01", seqs=(1, 2), n=4, layout="split", camera_yaml=True,
              poses=None, depth_fn=None, inst_fn=None, ply=None, names=None):
    """A RIO10 scene in the documented layout; returns {seq: frames dir}."""
    import cv2
    out = {}
    for k in seqs:
        q = f"seq{scene}_{k:02d}"
        if layout == "split":
            fd = os.path.join(root, f"seq{scene}", q)
            sd = os.path.join(root, f"semantics{scene}", q)
            md = os.path.join(root, f"models{scene}", q)
        elif layout == "scene":
            fd = os.path.join(root, f"scene{scene}", f"seq{scene}", q)
            sd = os.path.join(root, f"scene{scene}", f"semantics{scene}", q)
            md = os.path.join(root, f"scene{scene}", f"models{scene}", q)
        else:                                            # merged: one folder per sequence
            fd = sd = md = os.path.join(root, f"seq{scene}", q)
        for d in (fd, sd, md):
            os.makedirs(d, exist_ok=True)
        for i in range(n):
            img = np.full((H, W, 3), 10 * i, np.uint8)
            img[..., 2] = 200                            # BGR on disk: red channel 200
            cv2.imwrite(os.path.join(fd, f"frame-{i:06d}.color.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
            P = (poses or {}).get((k, i), _pose(i))
            np.savetxt(os.path.join(fd, f"frame-{i:06d}.pose.txt"), P)
            dep = depth_fn(k, i) if depth_fn else np.full((H, W), 1500 + i, np.uint16)
            cv2.imwrite(os.path.join(fd, f"frame-{i:06d}.rendered.depth.png"), dep)
            inst = inst_fn(k, i) if inst_fn else np.full((H, W), 3 + i, np.uint16)
            cv2.imwrite(os.path.join(sd, f"frame-{i:06d}.instances.png"), inst)
        if camera_yaml:
            with open(os.path.join(fd, "camera.yaml"), "w") as f:
                f.write(f"camera_intrinsics:\n  model: [{FX}, {FY}, {CX}, {CY}]\n")
        with open(os.path.join(sd, "instances.txt"), "w") as f:
            for iid, nm in (names or {1: "wall", 2: "floor", 3: "chair", 4: "table"}).items():
                f.write(f"{iid}\t{nm}\n")
        pts, ids = ply[k] if ply else (np.random.default_rng(k).random((30, 3)),
                                       np.repeat([1, 2, 3], 10))
        write_ply(os.path.join(md, "labels.ply"), pts, ids)
        out[k] = fd
    return out


def test_frames_match_the_loader_contract(tmp_path):
    write_rio(str(tmp_path))
    sc = R.RIO10Scene(str(tmp_path), 1)
    ds = sc.sequence(1)
    assert len(ds) == 4 and ds.H == H and ds.W == W
    fr = ds[2]
    assert fr["image"].shape == (H, W, 3) and fr["image"].dtype == torch.uint8
    assert int(fr["image"][..., 0].float().mean()) in range(195, 206)        # RGB order
    assert fr["depth"] is None and fr["depth_source"] == "none"              # oracle off
    assert torch.allclose(fr["c2w"], torch.from_numpy(_pose(2)).float(), atol=1e-6)
    assert torch.allclose(fr["K"], torch.tensor([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]]))
    assert fr["index"] == 2 and fr["name"] == "seq01_01/frame-000002"
    assert "instances" not in fr
    dr = sc.sequence(2, depth="rendered", instances=True)
    f2 = dr[1]
    assert f2["depth_source"] == "rendered_mesh" and f2["depth"].dtype == torch.float32
    assert torch.allclose(f2["depth"], torch.full((H, W), 1.501))
    assert f2["instances"].dtype == torch.int64 and int(f2["instances"][0, 0]) == 4
    assert sc.instance_names(1) == {1: "wall", 2: "floor", 3: "chair", 4: "table"}
    lab = sc.labels(2)
    assert lab["points"].shape == (30, 3) and lab["property"] == "objectId"
    assert lab["ids"].dtype == np.int64 and sorted(set(lab["ids"].tolist())) == [1, 2, 3]
    print("PASS RIO10 frames: image/depth/c2w/K/index/name/instances, oracle depth opt-in")


def test_parts_stride_and_root_suffix(tmp_path):
    fd = write_rio(str(tmp_path), seqs=(2,), n=10)[2]
    assert R.RIO10Dataset(fd).numbers == list(range(10))
    assert R.RIO10Dataset(fd, part="first_half").numbers == [0, 1, 2, 3, 4]
    assert R.RIO10Dataset(fd, part="second_half", stride=2).numbers == [5, 7, 9]
    assert R.RIO10Dataset(fd + "::second_half").numbers == [5, 6, 7, 8, 9]
    assert R.RIO10Dataset(fd, stride=3)[1]["index"] == 3
    with pytest.raises(ValueError, match="part"):
        R.RIO10Dataset(fd, part="middle")


def test_intrinsics_txt_and_principal_point(tmp_path):
    fd = write_rio(str(tmp_path), seqs=(1,), camera_yaml=False)[1]
    with pytest.raises(R.RIO10FormatError, match="no camera.yaml"):
        R.RIO10Dataset(fd)
    (tmp_path / "intrinsics.txt").write_text(f"seq01_01 {FX} {FY} {CX} {CY}\nseq01_02 1 1 1 1\n")
    ds = R.RIO10Dataset(fd)
    assert float(ds.K[0, 2]) == pytest.approx(CX) and float(ds.K[1, 1]) == pytest.approx(FY)
    # landscape intrinsics (cx ~ 490) on a portrait image: refused
    (tmp_path / "intrinsics.txt").write_text("seq01_01 756.0 756.8 492.9 270.4\n")
    with pytest.raises(R.RIO10FormatError, match="principal point"):
        R.RIO10Dataset(fd)
    ds = R.RIO10Dataset(fd, intrinsics=(FX, FY, CX, CY))
    assert float(ds.K[0, 0]) == pytest.approx(FX)


def test_format_checks_fail_loudly(tmp_path):
    import cv2
    bad = _pose(1)
    bad[:3, :3] *= 1.1                                   # not a rotation
    fd = write_rio(str(tmp_path / "a"), seqs=(1,), poses={(1, 1): bad})[1]
    ds = R.RIO10Dataset(fd)
    with pytest.raises(R.RIO10FormatError, match="frame-000001.pose.txt.*orthonormal"):
        ds[1]
    fd = write_rio(str(tmp_path / "b"), seqs=(1,))[1]
    os.remove(os.path.join(fd, "frame-000002.pose.txt"))
    with pytest.raises(R.RIO10FormatError, match="no pose file"):
        R.RIO10Dataset(fd)
    assert len(R.RIO10Dataset(fd, require_poses=False)) == 4
    renamed = str(tmp_path / "b" / "seq01" / "sequence1")
    fd2 = write_rio(str(tmp_path / "c"), seqs=(1,))[1]
    os.rename(fd2, renamed)
    with pytest.raises(R.RIO10FormatError, match="seqXX_YY"):
        R.RIO10Dataset(renamed)
    # an RGB-coded instance image and an 8-bit depth are not the documented format
    fd = write_rio(str(tmp_path / "d"), seqs=(1,),
                   inst_fn=lambda k, i: np.zeros((H, W, 3), np.uint8),
                   depth_fn=lambda k, i: np.zeros((H, W), np.uint8))[1]
    sem = str(tmp_path / "d" / "semantics01" / "seq01_01")
    with pytest.raises(R.RIO10FormatError, match="single-channel uint8/uint16 instance"):
        R.RIO10Dataset(fd, semantics_dir=sem)[0]
    with pytest.raises(R.RIO10FormatError, match="uint16 depth"):
        R.RIO10Dataset(fd, depth="rendered")[0]
    os.remove(os.path.join(sem, "frame-000003.instances.png"))
    with pytest.raises(R.RIO10FormatError, match="no instance image"):
        R.RIO10Dataset(fd, semantics_dir=sem)
    with pytest.raises(R.RIO10FormatError, match="tried"):
        R.RIO10Scene(str(tmp_path / "d"), 7).frames_dir(1)
    # a frame of another size than the first
    fd = write_rio(str(tmp_path / "e"), seqs=(1,))[1]
    cv2.imwrite(os.path.join(fd, "frame-000003.color.jpg"), np.zeros((H, W + 2, 3), np.uint8))
    with pytest.raises(R.RIO10FormatError, match="of the first frame"):
        R.RIO10Dataset(fd)[3]


def test_non_finite_pose_and_external_depth(tmp_path):
    inf = np.full((4, 4), np.inf)
    fd = write_rio(str(tmp_path), seqs=(2,), poses={(2, 0): inf})[2]
    ds = R.RIO10Dataset(fd)
    fr = ds[0]
    assert not torch.isfinite(fr["c2w"]).all()           # passed through, skipped downstream
    assert not torch.isfinite(ds.pose(0)).all()          # pose only: the frame's own c2w
    assert torch.equal(ds.pose(2), ds[2]["c2w"])
    ext = tmp_path / "mast3r"
    ext.mkdir()
    for i in range(4):
        d = np.full((H // 2, W // 2), 2.0 + i, np.float32)
        d[0, 0] = np.nan
        np.save(ext / f"frame-{i:06d}.depth.npy", d)
    fr = R.RIO10Dataset(fd, depth=str(ext))[1]
    assert fr["depth_source"] == "external" and fr["depth"].shape == (H // 2, W // 2)
    assert float(fr["depth"][0, 0]) == 0.0 and float(fr["depth"][1, 1]) == 3.0
    with pytest.raises(ValueError, match="directory"):
        R.RIO10Dataset(fd, depth=str(tmp_path / "missing"))


def test_layouts_resolve(tmp_path):
    for layout in ("scene", "merged"):
        root = str(tmp_path / layout)
        write_rio(root, scene="04", layout=layout)
        sc = R.RIO10Scene(root, "04")
        assert len(sc.sequence(2, instances=True)) == 4
        assert sc.labels(1)["points"].shape == (30, 3)
        assert sc.instance_names(2)[3] == "chair"


def test_labels_ply_and_instances_txt(tmp_path):
    pts = np.arange(12, dtype=np.float64).reshape(4, 3)
    write_ply(str(tmp_path / "a.ply"), pts, np.array([0, 5, 5, 7]), prop="label")
    # "label" may hold semantic class ids (ScanNet-style): never taken silently (V7)
    with pytest.raises(R.RIO10FormatError, match="semantic class ids.*id_property='label'"):
        R.read_labels_ply(str(tmp_path / "a.ply"))
    d = R.read_labels_ply(str(tmp_path / "a.ply"), id_property="label")
    assert d["property"] == "label" and d["ids"].tolist() == [0, 5, 5, 7]
    assert np.allclose(d["points"], pts)
    write_ply(str(tmp_path / "c.ply"), pts, np.array([0, 5, 5, 7]), prop="instance")
    assert R.read_labels_ply(str(tmp_path / "c.ply"))["property"] == "instance"
    from plyfile import PlyData, PlyElement
    v = np.zeros(2, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1")])
    PlyData([PlyElement.describe(v, "vertex")]).write(str(tmp_path / "b.ply"))
    with pytest.raises(R.RIO10FormatError, match="no instance-id vertex property.*red"):
        R.read_labels_ply(str(tmp_path / "b.ply"))
    (tmp_path / "i.txt").write_text("# id label\n1\twall\n2, floor\n7 office chair\n")
    assert R.read_instances_txt(str(tmp_path / "i.txt")) == {1: "wall", 2: "floor",
                                                            7: "office chair"}
    (tmp_path / "j.txt").write_text("1 wall\n1 floor\n")
    with pytest.raises(R.RIO10FormatError, match="duplicate"):
        R.read_instances_txt(str(tmp_path / "j.txt"))
    (tmp_path / "k.txt").write_text("wall 1\n")
    with pytest.raises(R.RIO10FormatError, match="<id> <label>"):
        R.read_instances_txt(str(tmp_path / "k.txt"))


def test_3rscan_changes_and_transform_order(tmp_path):
    T = np.eye(4)
    T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    T[:3, 3] = [0.3, -0.2, 0.1]
    assert np.allclose(R.transform_from_list(T.reshape(-1).tolist()), T)           # row-major
    assert np.allclose(R.transform_from_list(T.reshape(-1, order="F").tolist()), T)  # column
    Rz = np.eye(4)
    Rz[:3, :3] = T[:3, :3]                               # zero translation: `order` decides
    assert np.allclose(R.transform_from_list(Rz.reshape(-1, order="F").tolist()), Rz)
    assert np.allclose(R.transform_from_list(Rz.reshape(-1).tolist(), order="C"), Rz)
    with pytest.raises(R.RIO10FormatError, match="homogeneous"):
        R.transform_from_list([1.0] * 16)
    meta = [{"reference": "ref0", "type": "train", "scans": [
        {"reference": "res1", "transform": np.eye(4).reshape(-1).tolist(),
         "removed": [9, {"instance_reference": 11}], "nonrigid": [4],
         "rigid": [{"instance_reference": 5, "instance_rescan": 5, "symmetry": 0,
                    "transform": T.reshape(-1, order="F").tolist()}]}]}]
    p = tmp_path / "3RScan.json"
    p.write_text(json.dumps(meta))
    ch = R.load_3rscan_changes(str(p), "ref0", "res1")
    assert ch["removed"] == {9, 11} and ch["nonrigid"] == {4}
    assert set(ch["rigid"]) == {5} and np.allclose(ch["rigid"][5][1], T)
    with pytest.raises(R.RIO10FormatError, match="no rescan"):
        R.load_3rscan_changes(meta, "ref0", "res2")
    with pytest.raises(R.RIO10FormatError, match="no reference"):
        R.load_3rscan_changes(meta, "refX", "res1")
