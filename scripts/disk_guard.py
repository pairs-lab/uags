"""
disk_guard.py — refuse a mapping step that the disk cannot hold (the workstation disk is
nearly full, and a field written to a full disk fails only at its final save, after the GPU
time is spent).

    python3 scripts/disk_guard.py DIR MAP BYTES_PER_GAUSSIAN [--factor F]

N = the number of Gaussians of the 3DGS map MAP (.pt: its "means", read memory-mapped; .ply:
the vertex count of its header). The step needs F x N x BYTES_PER_GAUSSIAN bytes free on the
file system of DIR (F = 1.5 by default: the atomic save holds the temp file next to the old
file, and the estimate is rough). Exits 1 with "refused: ..." when less is free, 0 otherwise;
F = 0 turns the check off. Per-Gaussian sizes (cross-track review, 2026-10-10, d = 64): a
field after an update pass ~0.86 KB, + NormLift / averaging / view count ~0.28 KB, + the
residual change scores ~0.06 KB (the swap reset field, ~1.2 KB); a ScanNet field with CSS at
most ~1.55 KB.
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys


def num_gaussians(path):
    """Number of Gaussians of a 3DGS map (.ply header, or the means of a .pt checkpoint)."""
    p = str(path)
    if p.endswith(".ply"):
        with open(p, "rb") as f:
            for line in f:
                if line.startswith(b"element vertex"):
                    return int(line.split()[2])
                if line.strip() == b"end_header":
                    break
        raise ValueError(f"{p}: no 'element vertex' in the PLY header")
    import torch
    try:
        ck = torch.load(p, map_location="cpu", mmap=True, weights_only=True)
    except RuntimeError:                          # not the zip format: no memory map
        ck = torch.load(p, map_location="cpu", weights_only=True)
    if "splats" in ck:
        ck = ck["splats"]
    if "means" not in ck:
        raise KeyError(f"{p}: no 'means' (found {sorted(ck)})")
    return int(ck["means"].shape[0])


def check(directory, map_path, bytes_per_gaussian, factor=1.5):
    """(ok, message) for a step that writes factor x N x bytes_per_gaussian into directory."""
    n = num_gaussians(map_path)
    need = float(factor) * n * float(bytes_per_gaussian)
    d = directory
    while not os.path.exists(d):                  # the directory may not exist yet
        d = os.path.dirname(os.path.abspath(d))
    free = shutil.disk_usage(d).free
    msg = (f"disk: {free / 2 ** 30:.1f} GiB free in {directory}, the step needs "
           f"{need / 2 ** 30:.1f} GiB ({factor:g} x {n} Gaussians x {bytes_per_gaussian:g} B)")
    return free >= need, msg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir", help="directory the step writes into")
    ap.add_argument("map", help="the 3DGS map of the step (.pt or .ply)")
    ap.add_argument("bytes_per_gaussian", type=float)
    ap.add_argument("--factor", type=float, default=1.5)
    a = ap.parse_args(argv)
    if a.factor <= 0:
        return 0
    ok, msg = check(a.dir, a.map, a.bytes_per_gaussian, a.factor)
    if not ok:
        print(f"refused: {msg}: free disk space first (DISK_FACTOR=0 skips this check)")
        return 1
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
