"""
scannet_sens_export.py — export a ScanNet .sens file into the layout scripts/scannet_tab1.sh reads,
in Python 3 (ScanNet's own SensReader/python is Python 2 only):

    scans/<scene>/color/<i>.jpg      the frame's JPEG bytes as stored in the .sens (no re-encoding;
                                     the official reader decodes and re-saves them with imageio)
    scans/<scene>/depth/<i>.png      uint16 depth in millimetres (depth_shift 1000), 16-bit PNG
    scans/<scene>/pose/<i>.txt       camera-to-world 4x4, one row per line (np.savetxt '%f', as the
                                     official export_poses; lost tracking stays -inf)
    scans/<scene>/intrinsic/{intrinsic,extrinsic}_{color,depth}.txt

The .sens format (ScanNet SensReader/python/SensorData.py, version 4): header (version, sensor name,
intrinsic/extrinsic of colour and depth as 4x4 float32, compression types, sizes, depth shift,
frame count), then per frame: camera_to_world (16 float32), two uint64 timestamps, the byte sizes
of the colour and depth data, and the two blobs. Colour must be JPEG (type 2) and depth
zlib-compressed uint16 (type 1), the types of every ScanNet v2 scan. Frames are streamed one at a
time (the official reader holds every frame in memory).

    python3 scripts/scannet_sens_export.py <data root>/scans/scene0000_00/scene0000_00.sens \
        --out <data root>/scans/scene0000_00 [--skip-existing]
"""
from __future__ import annotations
import argparse
import os
import struct
import sys
import zlib

import numpy as np

COLOR_TYPES = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
DEPTH_TYPES = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


def _read(f, fmt):
    n = struct.calcsize(fmt)
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"truncated .sens: wanted {n} bytes, got {len(b)}")
    return struct.unpack(fmt, b)


def _mat(f):
    return np.asarray(_read(f, "<16f"), dtype=np.float32).reshape(4, 4)


def read_header(f):
    """The .sens header as a dict (file positioned at the first frame afterwards)."""
    (version,) = _read(f, "<I")
    if version != 4:
        raise ValueError(f"unsupported .sens version {version} (expected 4)")
    (strlen,) = _read(f, "<Q")
    name = f.read(strlen).decode("ascii", "replace")
    h = dict(version=version, sensor_name=name, intrinsic_color=_mat(f), extrinsic_color=_mat(f),
             intrinsic_depth=_mat(f), extrinsic_depth=_mat(f))
    ct, dt = _read(f, "<ii")
    h["color_compression"], h["depth_compression"] = COLOR_TYPES.get(ct), DEPTH_TYPES.get(dt)
    h["color_width"], h["color_height"], h["depth_width"], h["depth_height"] = _read(f, "<IIII")
    (h["depth_shift"],) = _read(f, "<f")
    (h["num_frames"],) = _read(f, "<Q")
    return h


def frames(f, n):
    """Yield (index, camera_to_world, colour bytes, depth bytes) for n frames."""
    for i in range(n):
        c2w = _mat(f)
        _read(f, "<QQ")                                   # colour and depth timestamps
        cs, ds = _read(f, "<QQ")
        color, depth = f.read(cs), f.read(ds)
        if len(color) != cs or len(depth) != ds:
            raise EOFError(f"truncated .sens at frame {i}")
        yield i, c2w, color, depth


def save_mat(m, path):
    with open(path, "w") as fh:
        for row in np.asarray(m):
            np.savetxt(fh, row[np.newaxis], fmt="%f")


def export(sens, out, skip_existing=False, log=print):
    import cv2
    for d in ("color", "depth", "pose", "intrinsic"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    with open(sens, "rb") as f:
        h = read_header(f)
        if h["color_compression"] != "jpeg" or h["depth_compression"] != "zlib_ushort":
            raise ValueError(f"{sens}: colour {h['color_compression']}, depth "
                             f"{h['depth_compression']}: only jpeg / zlib_ushort are supported")
        for name in ("intrinsic_color", "extrinsic_color", "intrinsic_depth", "extrinsic_depth"):
            save_mat(h[name], os.path.join(out, "intrinsic", name + ".txt"))
        n, H, W = h["num_frames"], h["depth_height"], h["depth_width"]
        written = 0
        for i, c2w, color, depth in frames(f, n):
            cp = os.path.join(out, "color", f"{i}.jpg")
            dp = os.path.join(out, "depth", f"{i}.png")
            pp = os.path.join(out, "pose", f"{i}.txt")
            if skip_existing and os.path.exists(cp) and os.path.exists(dp) and os.path.exists(pp):
                continue
            with open(cp, "wb") as fh:
                fh.write(color)
            d = np.frombuffer(zlib.decompress(depth), dtype=np.uint16).reshape(H, W)
            if not cv2.imwrite(dp, d):
                raise IOError(f"could not write {dp}")
            save_mat(c2w, pp)
            written += 1
            if written % 1000 == 0:
                log(f"  {os.path.basename(out)}: {i + 1}/{n} frames")
    log(f"{sens}: {n} frames ({written} written), colour {h['color_width']}x{h['color_height']}, "
        f"depth {W}x{H}, depth shift {h['depth_shift']:g}")
    return h


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sens")
    ap.add_argument("--out", required=True, help="the scene folder (scans/<scene>)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="keep frames whose colour, depth and pose files exist (resume)")
    a = ap.parse_args(argv)
    export(a.sens, a.out, a.skip_existing)


if __name__ == "__main__":
    sys.exit(main())
