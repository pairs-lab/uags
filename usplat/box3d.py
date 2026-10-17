"""
box3d.py — oriented 3D boxes given by their 8 corners (in any order): reading box files,
point membership and the image silhouette (the NeRF-Object-Removal ground truth; checked by
scripts/check_nor.py).

A box is the convex hull of its corners, kept as half-spaces A x + b <= 0 (unit outward
normals: the Qhull facets of scipy.spatial.ConvexHull; a box face appears there as two
triangles on the same plane). Inflating by d moves every face plane out by d, so each extent
grows by 2 d (the 2 cm margin of the stale ground truth); edges and corners stay sharp (no
rounded Minkowski sum).

Files (read_box_corners):
  .txt, 8 rows "x y z"        the corners (bbox.txt, bbox_nerf.txt, bbox_nerf_refined.txt)
  .txt, 3 rows "min max"      an axis-aligned box, one row per axis (bbox_refined.txt)
  .ply                        the vertices (binary little-endian or ASCII; the vertex element
                              must come first, as in the files written by Open3D)

Silhouette: pixel (i, j) of an (h, w) grid is in the silhouette iff the ray from the camera
centre through the pixel centre (j + 0.5, i + 0.5) (the gsplat convention, as in
usplat.gs_fit.backproject_depth) meets the box at a distance >= 0. For a box in front of the
camera this is the convex hull of the projected corners; unlike that hull it stays right when
some corners are behind the camera.
"""
from __future__ import annotations
import numpy as np

_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2",
              "int16": "i2", "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4",
              "uint": "u4", "uint32": "u4", "float": "f4", "float32": "f4",
              "double": "f8", "float64": "f8"}


def _ply_header(f, path):
    """(format, vertex count, [(name, numpy type)]) of a PLY header; f is left after it."""
    fmt, n, props, element, first = None, None, [], None, None
    while True:
        line = f.readline()
        if not line:
            raise ValueError(f"{path}: PLY header has no end_header")
        p = line.decode("ascii", "replace").split()
        if not p:
            continue
        if p[0] == "format":
            fmt = p[1]
        elif p[0] == "element":
            element = p[1]
            first = first or element
            if element == "vertex":
                n = int(p[2])
        elif p[0] == "property" and element == "vertex":
            if p[1] == "list":
                raise ValueError(f"{path}: list property in the vertex element")
            props.append((p[2], _PLY_TYPES[p[1]]))
        elif p[0] == "end_header":
            break
    if first != "vertex" or n is None:
        raise ValueError(f"{path}: the vertex element must be the first element")
    return fmt, n, props


def read_ply_vertices(path) -> np.ndarray:
    """(n, 3) float64 x, y, z of the vertices of a PLY file."""
    with open(path, "rb") as f:
        fmt, n, props = _ply_header(f, path)
        names = [name for name, _ in props]
        if fmt == "ascii":
            rows = [f.readline().split()[:len(props)] for _ in range(n)]
            v = {k: np.array([float(r[i]) for r in rows]) for i, k in enumerate(names)}
        elif fmt == "binary_little_endian":
            dt = np.dtype([(k, "<" + t) for k, t in props])
            v = np.frombuffer(f.read(dt.itemsize * n), dtype=dt, count=n)
        else:
            raise ValueError(f"{path}: unsupported PLY format {fmt}")
    return np.stack([np.asarray(v[k], dtype=np.float64) for k in ("x", "y", "z")], 1)


def read_box_corners(path) -> np.ndarray:
    """(8, 3) float64 box corners from a .txt (8 corners, or per-axis min max) or .ply file."""
    if str(path).endswith(".ply"):
        c = read_ply_vertices(path)
    else:
        a = np.loadtxt(path, dtype=np.float64)
        if a.shape == (3, 2):                                   # rows x, y, z: min max
            c = np.array([[x, y, z] for x in a[0] for y in a[1] for z in a[2]])
        else:
            c = a
    if c.shape != (8, 3):
        raise ValueError(f"{path}: expected 8 box corners, got an array of shape {c.shape}")
    return c


def box_halfspaces(corners):
    """(A (F, 3), b (F,)) with unit outward normals: x is inside iff A x + b <= 0."""
    from scipy.spatial import ConvexHull
    eq = ConvexHull(np.asarray(corners, dtype=np.float64)).equations
    norm = np.linalg.norm(eq[:, :3], axis=1, keepdims=True)
    eq = eq / norm
    return eq[:, :3], eq[:, 3]


def points_in_box(points, corners, inflate=0.0) -> np.ndarray:
    """(M,) bool: points (M, 3) inside the box whose faces are moved out by `inflate`."""
    A, b = box_halfspaces(corners)
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return ((P @ A.T + b) <= inflate).all(axis=1)


def box_extents(corners) -> np.ndarray:
    """(3,) side lengths of a cuboid along its own axes (ValueError if not a cuboid)."""
    A, _ = box_halfspaces(corners)
    axes = []
    for n in A:                                     # one normal per pair of opposite faces
        if all(abs(float(n @ a)) < 0.999 for a in axes):
            axes.append(n)
    if len(axes) != 3:
        raise ValueError(f"not a cuboid: {len(axes)} face directions")
    c = np.asarray(corners, dtype=np.float64)
    return np.array([np.ptp(c @ a) for a in axes])


def box_silhouette(corners, c2w, K, hw, inflate=0.0) -> np.ndarray:
    """(h, w) bool: pixels whose ray meets the (inflated) box; c2w (4, 4) OpenCV
    camera-to-world, K (3, 3) intrinsics of the (h, w) grid."""
    A, b = box_halfspaces(corners)
    b = b - inflate
    c2w, K = np.asarray(c2w, dtype=np.float64), np.asarray(K, dtype=np.float64)
    h, w = hw
    v, u = np.mgrid[0:h, 0:w]
    d = np.stack([(u.ravel() + 0.5 - K[0, 2]) / K[0, 0],
                  (v.ravel() + 0.5 - K[1, 2]) / K[1, 1], np.ones(h * w)], 1) @ c2w[:3, :3].T
    s0 = A @ c2w[:3, 3] + b                         # (F,) facet values at the camera centre
    sd = d @ A.T                                    # (P, F) their rate along each ray
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -s0 / sd                                # where the ray crosses each facet plane
    lo = np.where(sd < 0, t, -np.inf).max(axis=1)   # inside needs s0 + t sd <= 0 for all F
    hi = np.where(sd > 0, t, np.inf).min(axis=1)
    parallel_ok = np.where(sd == 0, s0 <= 0, True).all(axis=1)
    return ((np.maximum(lo, 0.0) <= hi) & parallel_ok).reshape(h, w)
