"""Update-pose alignment of NeRF-Object-Removal scenes (protocol D3): usplat/nor_align.py, the
loader's align option and scripts/align_nor.py, on the CPU. Checks:
  loader    align=<json> moves exactly the update part (frames >= split_end) by T and no other
            frame; a record of another scene or a non-rigid T is refused; run_mapping and
            fit_gs pass --align on
  fit       a synthetic textured room (6 walls, each part sweeps 360 deg of yaw) seen by a
            map part and an update part whose images.txt poses carry a known rigid offset:
            with every direction free (tau 0.01) estimate_alignment recovers the inverse
            offset (points moved back within 1 mm); with the protocol's tau 0.1 the two
            weakest directions (8-9% of the strongest here) stay at the identity, and the
            held-out depth error still drops from 2.8 cm to < 5 mm and the photometric NCC
            rises; the JSON round-trips through align_nor.py into the loader, and
            --summary writes the run summary elsewhere (summary.json is kept)
  helpers   project / reproject agree with a hand computation; rotation_exp is a rotation
"""
import importlib.util
import json
import math
import os
import pathlib

import numpy as np
import pytest
import torch

from usplat import nor_align as A
from usplat.datasets import NeRFObjectRemovalDataset, make_dataset

REPO = pathlib.Path(__file__).resolve().parents[1]
H, W = 192, 256                        # image grid = depth grid in these scenes
K = np.array([[180.0, 0, 128.0], [0, 180.0, 96.0], [0, 0, 1.0]])
ROOM = (np.array([-1.5, -0.9, -1.5]), np.array([1.5, 0.9, 2.5]))   # all walls seen


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rigid(axis, deg, t):
    T = np.eye(4)
    T[:3, :3] = A.rotation_exp(np.radians(deg) * np.asarray(axis, float) / np.linalg.norm(axis))
    T[:3, 3] = t
    return T


def quat(R):
    """(w, x, y, z) of a rotation matrix (COLMAP order)."""
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def room_view(c2w):
    """Depth (H, W) [m] and a textured grey image (H, W, 3) uint8 of the room interior."""
    v, u = np.mgrid[0:H, 0:W]
    d = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1],
                  np.ones((H, W))], -1) @ c2w[:3, :3].T                # camera z = 1
    C = c2w[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = np.stack([np.where(d[..., i] > 0, (ROOM[1][i] - C[i]) / d[..., i],
                                (ROOM[0][i] - C[i]) / d[..., i]) for i in range(3)], -1)
    t = np.nanmin(np.where(ts > 0, ts, np.inf), -1)                       # exit through a wall
    P = C + t[..., None] * d
    tex = (np.sin(2 * np.pi * P[..., 0] / 0.37) * np.sin(2 * np.pi * P[..., 1] / 0.29)
           + np.sin(2 * np.pi * P[..., 2] / 0.31))
    img = np.clip(128 + 50 * tex, 0, 255).astype(np.uint8)
    return t.astype(np.float32), np.repeat(img[..., None], 3, -1)


def write_room(root, offset, n_map=16, n_between=2, n_upd=16):
    """A NOR-layout scene of the room; images.txt holds the true poses of the map part and
    offset @ true for the update part (depth and images come from the true poses)."""
    import cv2
    for d in ("images", "depth", "confidence", "masks_real", "sparse", "boxes"):
        os.makedirs(root / d)
    n = n_map + n_between + n_upd
    json.dump({"split_start": n_map, "split_end": n_map + n_between, "synthetic_end": -1},
              open(root / "meta.json", "w"))
    box = np.array([[x, y, z] for x in (0.8, 1.2) for y in (-0.9, -0.5) for z in (1.6, 2.0)])
    np.savetxt(root / "boxes" / "bbox.txt", box)
    cams, ims, true = [], [], {}
    for i in range(n):                    # each part sweeps 360 deg of yaw: every wall head-on
        j = i if i < n_map else i - n_map - n_between + 0.5
        a = 2 * np.pi * j / (n_map if i < n_map else n_upd)
        c2w = rigid([0, 1, 0], math.degrees(a), [0.2 * math.cos(a), 0.1 * math.sin(2 * a),
                                                   0.5 + 0.2 * math.sin(a)])
        true[i] = c2w
        depth, img = room_view(c2w)
        cv2.imwrite(str(root / "images" / f"{i:03d}.jpg"), img)
        depth.tofile(root / "depth" / f"depth_{i}.bin")
        np.full((H, W), 2, np.uint8).tofile(root / "confidence" / f"depthConfidence_{i}.bin")
        np.save(root / "masks_real" / f"{i:03d}.npy", np.zeros((H, W), bool))
        pose = offset @ c2w if i >= n_map + n_between else c2w
        w2c = np.linalg.inv(pose)
        cams.append(f"{i + 1} PINHOLE {W} {H} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}")
        ims.append(f"{i + 1} {' '.join(map(str, quat(w2c[:3, :3])))} "
                   f"{' '.join(map(str, w2c[:3, 3]))} 1 {i:03d}.jpg")
        ims.append("")
    (root / "sparse" / "cameras.txt").write_text("\n".join(cams) + "\n")
    (root / "sparse" / "images.txt").write_text("\n".join(ims) + "\n")
    return true


# ------------------------------------------------------------------- loader option
def test_loader_aligns_the_update_part_only(tmp_path):
    root = tmp_path / "007"
    off = rigid([1, 2, 3], 2.0, [0.03, -0.02, 0.05])
    write_room(root, off, n_map=4, n_between=2, n_upd=3)
    raw = NeRFObjectRemovalDataset(str(root), part="all")
    T = np.linalg.inv(off)
    rec = root.parent / "007.json"
    rec.write_text(json.dumps(dict(scene="007", T=T.tolist())))
    for align in (str(rec), dict(scene="007", T=T.tolist())):
        ds = NeRFObjectRemovalDataset(str(root), part="all", align=align)
        for i in range(9):
            want = T @ raw.c2w[i] if i >= 6 else raw.c2w[i]
            assert np.allclose(ds.c2w[i], want, atol=1e-9), i
            assert torch.allclose(ds[i]["c2w"].double(), torch.from_numpy(want), atol=1e-5)
    upd = make_dataset("nor", str(root), part="update", align=str(rec))
    assert np.allclose(upd.c2w[6], T @ raw.c2w[6])
    with pytest.raises(ValueError, match="scene"):
        NeRFObjectRemovalDataset(str(root), part="all", align=dict(scene="012", T=T.tolist()))
    bad = T.copy()
    bad[0, 0] *= 1.1                                       # not a rotation
    with pytest.raises(ValueError, match="rigid"):
        NeRFObjectRemovalDataset(str(root), part="all", align=dict(scene="007", T=bad.tolist()))
    rm = _script("run_mapping")
    assert rm._part("update", str(rec)) == {"part": "update", "align": str(rec)}
    assert rm._part("map") == {"part": "map"} and rm._part("") == {}
    a = _script("fit_gs").build_parser().parse_args(
        ["--dataset", "nor", "--data", "x", "--part", "map", "--align", "a.json", "--out", "y"])
    assert a.align == "a.json"


# ------------------------------------------------------------------- the fit
def test_alignment_recovers_a_known_offset(tmp_path):
    root = tmp_path / "001"
    off = rigid([0.3, 1.0, 0.2], 1.0, [0.03, 0.015, -0.025])        # update = off @ true
    true = write_room(root, off)
    pts = np.array([[x, y, z] for x in (-1.5, 1.5) for y in (-0.9, 0.9) for z in (-1.5, 2.5)])
    rec = A.estimate_alignment(str(root), check_pairs=6, fit_frames=16, iters=15, tau=0.01)
    T = np.array(rec["T"])
    moved = pts @ (T @ off)[:3, :3].T + (T @ off)[:3, 3]
    assert np.abs(moved - pts).max() < 1e-3, T @ off               # the offset is undone
    assert rec["icp"][-1]["directions"] == 6
    b, a = rec["before"], rec["after"]
    assert b["med_abs_dz"] > 0.02 and a["med_abs_dz"] < 1e-3, (b, a)
    assert a["ncc"] > b["ncc"] + 0.2 and a["ncc"] > 0.9, (b, a)
    rec = A.estimate_alignment(str(root), check_pairs=6, fit_frames=16, iters=15)  # tau 0.1
    b, a = rec["before"], rec["after"]
    assert rec["icp"][-1]["directions"] < 6 and a["med_abs_dz"] < 0.005, (rec["icp"][-1], a)
    assert a["ncc"] > b["ncc"] and rec["shift"]["median_m"] > 0.02
    held = {p[1] for p in rec["check_pairs"]}
    assert held.isdisjoint({p[1] for p in rec["fit_pairs"]})        # checks are held out
    out = tmp_path / "align"
    assert _script("align_nor").main(["--root", str(tmp_path), "--out-dir", str(out),
                                      "--scenes", "001", "--check-pairs", "6",
                                      "--fit-frames", "16", "--tau", "0.01"]) == 0
    ds = NeRFObjectRemovalDataset(str(root), part="update", align=str(out / "001.json"))
    for i in ds.frame_numbers:                                      # aligned = true poses
        assert np.abs(ds.c2w[i] - true[i]).max() < 1e-3, i
    summary = json.loads((out / "summary.json").read_text())
    assert summary["failed"] == [] and "001" in summary["scenes"]
    other = tmp_path / "summary_001.json"                         # --summary: keep summary.json
    assert _script("align_nor").main(["--root", str(tmp_path), "--out-dir", str(out),
                                      "--scenes", "001", "--check-pairs", "6", "--fit-frames",
                                      "16", "--iters", "2", "--summary", str(other)]) == 0
    assert json.loads(other.read_text())["scenes"]["001"]["icp_iterations"] == 2
    assert json.loads((out / "summary.json").read_text()) == summary


def test_project_and_reproject_by_hand():
    c2w = rigid([0, 1, 0], 10, [0.1, 0.0, -0.2])
    P = np.array([[0.3, -0.2, 2.0], [0.0, 0.0, -1.0]])
    v, u, z, ok = A.project(P, K, c2w, (H, W))
    pc = (P - c2w[:3, 3]) @ c2w[:3, :3]
    assert np.allclose(z, pc[:, 2]) and list(ok) == [True, False]
    assert u[0] == math.floor(K[0, 0] * pc[0, 0] / pc[0, 2] + K[0, 2])
    assert v[0] == math.floor(K[1, 1] * pc[0, 1] / pc[0, 2] + K[1, 2])
    depth, _ = room_view(c2w)
    f = dict(depth=depth, K_depth=K, c2w=c2w)
    dz, zp, _, _, n = A.reproject(f, dict(f, c2w=c2w))              # a frame onto itself
    assert n == H * W and np.abs(dz).max() < 1e-4
    R = A.rotation_exp(np.array([0.2, -0.1, 0.4]))
    assert np.allclose(R.T @ R, np.eye(3)) and math.isclose(np.linalg.det(R), 1.0)
    assert math.isclose(A.rotation_deg(rigid([1, 0, 0], 3.0, [0, 0, 0])), 3.0, rel_tol=1e-9)
