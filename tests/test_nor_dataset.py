"""NeRF-Object-Removal loader ("nor", usplat/datasets.py), the K_depth handling of downscale,
the run_mapping / fit_gs part flags and the check_nor.py machinery (CPU). Checks:
  synthetic scene  parts map / update / all with a stride, part required, per-image cameras
                   (images.txt names camera 1 for every frame), c2w = inverse of the COLMAP
                   world-to-camera pose, K_depth = K scaled to 192x256, depth invalid below
                   min_confidence / non-finite / <= 0 (set to 0), masks only where files exist
  COLMAP readers   images.txt with and without 2D points, quaternion order (w, x, y, z)
  downscale        a depth on its own grid moves onto the image grid and K_depth follows it
                   (registered and unregistered K_depth); no K_depth key is invented
  scripts          fit_gs --part and run_mapping --part / --update-part reach make_dataset;
                   check_nor's reprojection gives dz = 0 / ratio 1 on an exact plane and
                   ratio 1.1 when the depth is scaled by 1.1
  scene 001        (skipped when the data is absent) part sizes, K_depth, the box silhouette
                   against masks_real, LiDAR mask pixels inside the box, depth consistency
                   between frames 100 and 110, gs_fit.load_frames on nor frames
"""
import importlib.util
import json
import os
import pathlib

import numpy as np
import pytest
import torch

from usplat.datasets import (NeRFObjectRemovalDataset, make_dataset, downscale,
                             read_colmap_images, read_colmap_cameras, qvec_to_rotmat)

REPO = pathlib.Path(__file__).resolve().parents[1]
SCENE = "/home/thanhnc/gauss_ws/src/uags/data/nerf_object_removal/nerf-object-removal/001"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _quat(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.radians(deg) / 2
    return np.r_[np.cos(h), np.sin(h) * a]


def _write_scene(root, n=7, split=(3, 5), hw=(54, 72)):
    """A tiny scene in the dataset layout; frame i has depth 1 + 0.1 i and camera fx 50 + i."""
    import cv2
    H, W = hw
    for d in ("images", "depth", "confidence", "masks_real", "masks_synthetic", "sparse"):
        os.makedirs(root / d)
    json.dump({"split_start": split[0], "split_end": split[1], "synthetic_end": -1},
              open(root / "meta.json", "w"))
    cams, ims, poses = [], ["# Image list with two lines of data per image:"], {}
    for i in range(n):
        cv2.imwrite(str(root / "images" / f"{i:03d}.jpg"), np.full((H, W, 3), 20 * i, np.uint8))
        d = np.full((192, 256), 1.0 + 0.1 * i, np.float32)
        d[0, 0], d[0, 1] = np.nan, -1.0
        d.tofile(root / "depth" / f"depth_{i}.bin")
        c = np.full((192, 256), 2, np.uint8)
        c[:, 10:20], c[:, 20:30] = 1, 0
        c.tofile(root / "confidence" / f"depthConfidence_{i}.bin")
        m = np.zeros((192, 256), bool)
        m[50:60, 70:90] = True
        np.save(root / "masks_real" / f"{i:03d}.npy", m)
        if i >= split[1]:
            np.save(root / "masks_synthetic" / f"{i:03d}.npy", ~m)
        cams.append(f"{i + 1} PINHOLE {W} {H} {50.0 + i} {50.0 + i} {W / 2} {H / 2}")
        q, t = _quat([1, 2, 3], 10 * i), np.array([0.1 * i, -0.2, 1.0])
        poses[i] = (q, t)
        ims.append(f"{i + 1} {' '.join(map(str, q))} {' '.join(map(str, t))} 1 {i:03d}.jpg")
        ims.append("" if i % 2 else "1.0 2.0 -1 3.0 4.0 7")         # POINTS2D line
    (root / "sparse" / "cameras.txt").write_text("\n".join(cams) + "\n")
    (root / "sparse" / "images.txt").write_text("\n".join(ims) + "\n")
    return poses


# ------------------------------------------------------------------- synthetic scene
def test_parts_strides_and_part_required(tmp_path):
    _write_scene(tmp_path)
    for part, stride, want in (("map", 1, [0, 1, 2]), ("map", 2, [0, 2]),
                               ("update", 1, [5, 6]), ("all", 3, [0, 3, 6])):
        ds = make_dataset("nor", str(tmp_path), stride=stride, part=part)
        assert ds.frame_numbers == want and len(ds) == len(want)
        assert [ds[k]["index"] for k in range(len(ds))] == want
    with pytest.raises(ValueError):
        make_dataset("nor", str(tmp_path))
    with pytest.raises(ValueError):
        NeRFObjectRemovalDataset(str(tmp_path), part="before")
    ds = NeRFObjectRemovalDataset(str(tmp_path), part="all")
    assert [ds[i]["part"] for i in range(7)] == ["map"] * 3 + ["between"] * 2 + ["update"] * 2
    with pytest.raises(TypeError):                         # other datasets take no part
        make_dataset("replica", str(tmp_path), part="map")


def test_geometry_depth_and_masks(tmp_path):
    poses = _write_scene(tmp_path)
    ds = NeRFObjectRemovalDataset(str(tmp_path), part="all")
    for i in (0, 4, 6):
        f = ds[i]
        q, t = poses[i]
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = qvec_to_rotmat(q), t
        assert np.allclose(f["c2w"].double().numpy() @ w2c, np.eye(4), atol=1e-6)
        assert f["K"][0, 0] == 50 + i                       # the frame's own camera
        Kd = f["K"].clone()
        Kd[0] *= 256 / 72
        Kd[1] *= 192 / 54
        assert torch.allclose(f["K_depth"], Kd)
        assert f["image"].shape == (54, 72, 3) and f["image"].dtype == torch.uint8
        d, c = f["depth"], f["depth_confidence"]
        assert d.shape == (192, 256) and d.dtype == torch.float32 and c.dtype == torch.uint8
        assert d[0, 0] == 0 and d[0, 1] == 0                # NaN and negative depth
        assert (d[:, 10:30] == 0).all()                     # confidence 1 and 0
        assert torch.allclose(d[1:, 30:], torch.tensor(1.0 + 0.1 * i))
        assert f["mask_real"].dtype == torch.bool and f["mask_real"].sum() == 200
        assert ("mask_synthetic" in f) == (i >= 5)
    d1 = NeRFObjectRemovalDataset(str(tmp_path), part="all", min_confidence=1)[0]["depth"]
    assert (d1[1:, 10:20] > 0).all() and (d1[:, 20:30] == 0).all()


def test_colmap_readers(tmp_path):
    poses = _write_scene(tmp_path, n=4)
    ims = read_colmap_images(tmp_path / "sparse" / "images.txt")
    assert sorted(ims) == ["000.jpg", "001.jpg", "002.jpg", "003.jpg"]
    iid, q, t, cid = ims["002.jpg"]
    assert iid == 3 and cid == 1 and np.allclose(q, poses[2][0]) and np.allclose(t, poses[2][1])
    cams = read_colmap_cameras(tmp_path / "sparse" / "cameras.txt")
    assert cams[2] == ("PINHOLE", 72, 54, [51.0, 51.0, 36.0, 27.0])
    assert np.allclose(qvec_to_rotmat([1, 0, 0, 0]), np.eye(3))
    assert np.allclose(qvec_to_rotmat(_quat([0, 0, 1], 90)), [[0, -1, 0], [1, 0, 0], [0, 0, 1]])


# ------------------------------------------------------------------- downscale
def _frame(Kd_scale=0.5, depth_hw=(6, 8)):
    K = torch.tensor([[20.0, 0, 8.0], [0, 20.0, 6.0], [0, 0, 1]])
    Kd = K.clone()
    Kd[:2] *= Kd_scale
    return dict(image=torch.zeros(12, 16, 3, dtype=torch.uint8), K=K, K_depth=Kd,
                depth=torch.arange(np.prod(depth_hw), dtype=torch.float32).reshape(depth_hw) + 1,
                c2w=torch.eye(4))


def test_downscale_keeps_K_depth_with_the_depth():
    f = _frame()                                            # registered, half resolution
    g = downscale(f, 1)
    assert g["depth"].shape == (12, 16) and torch.allclose(g["K_depth"], f["K"])
    g = downscale(f, 2)
    assert g["depth"].shape == (6, 8) and torch.allclose(g["K_depth"], g["K"])
    u = _frame(Kd_scale=1.2, depth_hw=(12, 16))             # unregistered, same grid size
    g = downscale(u, 2)
    want = u["K_depth"].clone()
    want[:2] *= 0.5
    assert torch.allclose(g["K_depth"], want) and not torch.allclose(g["K_depth"], g["K"])
    assert torch.equal(f["K_depth"], _frame()["K_depth"])  # the input is not modified
    plain = {k: v for k, v in f.items() if k != "K_depth"}
    assert "K_depth" not in downscale(plain, 1) and "K_depth" not in downscale(plain, 2)


# ------------------------------------------------------------------- scripts
def test_fit_gs_and_run_mapping_pass_the_part(tmp_path):
    fit_gs = _script("fit_gs")
    a = fit_gs.build_parser().parse_args(["--dataset", "nor", "--data", "x", "--part", "map",
                                          "--out", "y.pt"])
    assert a.part == "map" and a.dataset == "nor"
    rm = _script("run_mapping")
    assert rm._part("") == {} and rm._part("update") == {"part": "update"}
    _write_scene(tmp_path)
    ds = make_dataset("nor", str(tmp_path), stride=1, **rm._part("update"))
    assert ds.frame_numbers == [5, 6]


def test_check_nor_reprojection_on_a_plane():
    cn = _script("check_nor")
    K = np.array([[200.0, 0, 128], [0, 200.0, 96], [0, 0, 1]])

    def view(cx, scale=1.0):                                # the plane z = 2 m, seen along +z
        c2w = np.eye(4)
        c2w[:3, 3] = [cx, 0.0, 0.0]
        return dict(depth=np.full((192, 256), 2.0 * scale, np.float32), K_depth=K, c2w=c2w)
    dz, zp, _, _, n = cn.reproject(view(0.0), view(0.1))
    st = cn.dz_stats(dz, zp, n)
    assert st["med_abs_dz"] < 1e-6 and abs(st["ratio"] - 1) < 1e-6 and st["overlap"] > 0.85
    dz, zp, _, _, n = cn.reproject(view(0.0), view(0.1, scale=1.1))
    assert abs(cn.dz_stats(dz, zp, n)["ratio"] - 1.1) < 1e-6
    assert cn.suggested_stride(257) == 2 and cn.suggested_stride(100) == 1
    assert cn.suggested_stride(396) == 3


# ------------------------------------------------------------------- real scene 001
@pytest.mark.skipif(not os.path.isdir(SCENE), reason="NeRF-Object-Removal scene 001 absent")
def test_scene_001_geometry():
    from usplat.box3d import read_box_corners, box_silhouette, points_in_box
    cn = _script("check_nor")
    assert len(make_dataset("nor", SCENE, part="map")) == 257
    assert len(make_dataset("nor", SCENE, part="update")) == 188
    assert len(make_dataset("nor", SCENE, stride=2, part="map")) == 129
    ds = NeRFObjectRemovalDataset(SCENE, part="all")
    corners = read_box_corners(ds.box_path)
    f = ds[100]
    assert f["image"].shape == (540, 720, 3) and f["depth"].shape == (192, 256)
    assert torch.allclose(f["K_depth"][:2], f["K"][:2] * 256 / 720)
    R = f["c2w"][:3, :3].double()
    assert torch.allclose(R.T @ R, torch.eye(3, dtype=torch.float64), atol=1e-5)
    for i in (0, 100, 200):
        g = cn.np_frame(ds, i)
        sil = box_silhouette(corners, g["c2w"], g["K_depth"], (192, 256))
        assert cn.overlap(sil, g["mask_real"])["iou"] > 0.6, i
        P, v, u = cn.backproject(g["depth"], g["K_depth"], g["c2w"])
        inb, m = points_in_box(P, corners, 0.02), g["mask_real"][v, u]
        assert (inb & m).sum() / m.sum() > 0.7 and (inb & m).sum() / inb.sum() > 0.95, i
    dz, zp, _, _, n = cn.reproject(cn.np_frame(ds, 100), cn.np_frame(ds, 110))
    st = cn.dz_stats(dz, zp, n)
    assert st["ok5cm"] > 0.95 and abs(st["ratio"] - 1) < 0.01, st


@pytest.mark.skipif(not os.path.isdir(SCENE), reason="NeRF-Object-Removal scene 001 absent")
def test_scene_001_through_gs_fit_load_frames():
    from usplat import gs_fit
    ds = make_dataset("nor", SCENE, stride=100, part="map")
    frames, f = gs_fit.load_frames(ds, max_res=640)
    assert f == 2 and len(frames) == 3
    for fr in frames:
        assert fr["image"].shape == (270, 360, 3) and fr["depth"].shape == (270, 360)
        assert torch.allclose(fr["K_depth"], fr["K"], atol=1e-4)    # registered depth
    P, _ = gs_fit.backproject_depth(frames[1], pixel_step=4)
    assert len(P) > 1000 and torch.isfinite(P).all()
