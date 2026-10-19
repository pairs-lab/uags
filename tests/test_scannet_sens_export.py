"""scripts/scannet_sens_export.py on a synthetic two-frame .sens file (CPU):
 * header fields, intrinsics and poses (incl. a lost-tracking -inf pose) round-trip;
 * colour = the stored JPEG bytes, depth = the zlib uint16 image as a 16-bit PNG;
 * a truncated file and an unsupported compression type stop with a message.
"""
import importlib.util
import pathlib
import struct
import zlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sens", REPO / "scripts" / "scannet_sens_export.py")
sx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sx)


def _sens(path, color_type=2, truncate=False):
    import cv2
    K = np.diag([500.0, 500.0, 1.0, 1.0]).astype(np.float32)
    K[0, 2], K[1, 2] = 320.0, 240.0
    E = np.eye(4, dtype=np.float32)
    H, W = 6, 8
    b = struct.pack("<I", 4) + struct.pack("<Q", 3) + b"abc"
    for m in (K, E, K, E):
        b += struct.pack("<16f", *m.flatten())
    b += struct.pack("<ii", color_type, 1) + struct.pack("<IIII", W, H, W, H)
    b += struct.pack("<f", 1000.0) + struct.pack("<Q", 2)
    frames = []
    for i in range(2):
        c2w = np.eye(4, dtype=np.float32)
        c2w[0, 3] = i
        if i == 1:
            c2w[:] = -np.inf                          # lost tracking
        img = np.full((H, W, 3), 40 * (i + 1), np.uint8)
        jpg = cv2.imencode(".jpg", img)[1].tobytes()
        depth = (np.arange(H * W, dtype=np.uint16) + 1000 * i).reshape(H, W)
        dz = zlib.compress(depth.tobytes())
        b += struct.pack("<16f", *c2w.flatten()) + struct.pack("<QQ", i, i)
        b += struct.pack("<QQ", len(jpg), len(dz)) + jpg + dz
        frames.append((c2w, jpg, depth))
    if truncate:
        b = b[:-10]
    path.write_bytes(b)
    return frames, K


def test_export_round_trip(tmp_path):
    import cv2
    frames, K = _sens(tmp_path / "s.sens")
    out = tmp_path / "scene"
    h = sx.export(str(tmp_path / "s.sens"), str(out), log=lambda *_: None)
    assert h["num_frames"] == 2 and h["sensor_name"] == "abc" and h["depth_shift"] == 1000.0
    assert np.allclose(np.loadtxt(out / "intrinsic" / "intrinsic_color.txt"), K)
    for i, (c2w, jpg, depth) in enumerate(frames):
        assert (out / "color" / f"{i}.jpg").read_bytes() == jpg
        d = cv2.imread(str(out / "depth" / f"{i}.png"), cv2.IMREAD_UNCHANGED)
        assert d.dtype == np.uint16 and np.array_equal(d, depth)
        p = np.loadtxt(out / "pose" / f"{i}.txt")
        assert (np.isinf(p).all() and i == 1) or np.allclose(p, c2w)
    print("PASS .sens export: header, intrinsics, colour bytes, 16-bit depth, poses (-inf kept)")


def test_refusals(tmp_path):
    _sens(tmp_path / "t.sens", truncate=True)
    with pytest.raises(EOFError):
        sx.export(str(tmp_path / "t.sens"), str(tmp_path / "o1"), log=lambda *_: None)
    _sens(tmp_path / "p.sens", color_type=1)
    with pytest.raises(ValueError, match="only jpeg"):
        sx.export(str(tmp_path / "p.sens"), str(tmp_path / "o2"), log=lambda *_: None)
    print("PASS refusals: truncated file, unsupported colour type")
