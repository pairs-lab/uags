"""usplat/box3d.py on synthetic boxes (CPU, no data). Checks:
  files       8-corner .txt, per-axis min/max .txt, binary and ASCII .ply give the corners;
              a wrong shape is an error
  membership  a rotated cuboid: points just inside / outside each face; the inflation moves
              every face out by d (in at 1.5 cm with d = 2 cm, out with d = 1 cm); the corner
              order does not matter; box_extents gives the side lengths
  silhouette  box in front of the camera = the convex hull of the projected corners at the
              pixel centres (exact test); camera inside the box = every pixel; box behind the
              camera = no pixel; corners behind the camera agree with dense ray sampling;
              the inflated silhouette contains the plain one
"""
import numpy as np
import pytest
from scipy.spatial import Delaunay

from usplat.box3d import (read_box_corners, points_in_box, box_extents, box_silhouette,
                          read_ply_vertices)


def _rot(seed):
    q = np.random.default_rng(seed).normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _box(centre=(0.2, -0.1, 3.0), ext=(0.4, 0.25, 0.6), seed=0):
    """Corners (8, 3) of a rotated cuboid, its rotation R (columns = axes) and half extents."""
    R, h = _rot(seed), np.asarray(ext) / 2
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    return np.asarray(centre) + (signs * h) @ R.T, R, h


def _cam(h=48, w=64, f=60.0):
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return K, np.eye(4), (h, w)


def test_read_box_files(tmp_path):
    C, _, _ = _box()
    np.savetxt(tmp_path / "c.txt", C)
    assert np.allclose(read_box_corners(tmp_path / "c.txt"), C)
    np.savetxt(tmp_path / "aabb.txt", np.array([[0, 1], [2, 4], [-1, 0.5]]))
    A = read_box_corners(tmp_path / "aabb.txt")
    assert A.shape == (8, 3)
    assert np.allclose(A.min(0), [0, 2, -1]) and np.allclose(A.max(0), [1, 4, 0.5])
    head = ("ply\nformat {}\ncomment test\nelement vertex 8\nproperty double x\n"
            "property double y\nproperty double z\nelement face 0\n"
            "property list uchar uint vertex_indices\nend_header\n")
    with open(tmp_path / "b.ply", "wb") as f:
        f.write(head.format("binary_little_endian 1.0").encode())
        f.write(C.astype("<f8").tobytes())
    assert np.allclose(read_box_corners(tmp_path / "b.ply"), C)
    with open(tmp_path / "a.ply", "w") as f:
        f.write(head.format("ascii 1.0") + "\n".join(" ".join(f"{x:.17g}" for x in r) for r in C))
    assert np.allclose(read_ply_vertices(tmp_path / "a.ply"), C)
    np.savetxt(tmp_path / "bad.txt", np.zeros((4, 3)))
    with pytest.raises(ValueError):
        read_box_corners(tmp_path / "bad.txt")


def test_points_in_box_faces_inflation_and_order():
    C, R, h = _box(seed=3)
    c = C.mean(0)
    for axis in range(3):
        for sign in (-1, 1):
            n = sign * R[:, axis]
            inside = c + n * (h[axis] - 0.005)
            near = c + n * (h[axis] + 0.015)
            assert points_in_box(inside[None], C)[0]
            assert not points_in_box(near[None], C)[0]
            assert points_in_box(near[None], C, inflate=0.02)[0]
            assert not points_in_box(near[None], C, inflate=0.01)[0]
    P = c + np.random.default_rng(1).normal(scale=0.3, size=(500, 3))
    perm = np.random.default_rng(2).permutation(8)
    assert np.array_equal(points_in_box(P, C), points_in_box(P, C[perm]))
    local = np.abs((P - c) @ R)                       # coordinates along the box axes
    assert np.array_equal(points_in_box(P, C), (local <= h).all(1))
    assert np.allclose(sorted(box_extents(C)), sorted(2 * h))


def test_silhouette_matches_projected_hull_in_front():
    C, _, _ = _box(centre=(0.1, 0.05, 2.0), seed=5)
    K, c2w, (h, w) = _cam()
    sil = box_silhouette(C, c2w, K, (h, w))
    uv = (C @ K.T)[:, :2] / C[:, 2:]
    v, u = np.mgrid[0:h, 0:w]
    centres = np.stack([u.ravel() + 0.5, v.ravel() + 0.5], 1)
    hull = (Delaunay(uv).find_simplex(centres) >= 0).reshape(h, w)
    assert sil.sum() > 100 and np.array_equal(sil, hull)
    assert (box_silhouette(C, c2w, K, (h, w), inflate=0.05) >= sil).all()


def test_silhouette_inside_behind_and_crossing_the_camera():
    K, c2w, hw = _cam()
    C, _, _ = _box(centre=(0, 0, 0.1), ext=(1, 1, 1))         # camera inside the box
    assert box_silhouette(C, c2w, K, hw).all()
    C, _, _ = _box(centre=(0, 0, -2.0), ext=(0.5, 0.5, 0.5))  # behind the camera
    assert not box_silhouette(C, c2w, K, hw).any()
    # x in [0.2, 1], y in [-0.3, 0.3], z in [-0.5, 1]: four corners behind, camera outside
    C = np.array([[x, y, z] for x in (0.2, 1.0) for y in (-0.3, 0.3) for z in (-0.5, 1.0)])
    sil = box_silhouette(C, c2w, K, hw)
    v, u = np.mgrid[0:hw[0], 0:hw[1]]
    d = np.stack([(u.ravel() + 0.5 - K[0, 2]) / K[0, 0], (v.ravel() + 0.5 - K[1, 2]) / K[1, 1],
                  np.ones(u.size)], 1)
    t = np.linspace(0, 1.2, 1201)                              # 1 mm steps along each ray
    hit = np.zeros(u.size, bool)
    for k in range(0, len(t), 500):
        P = (d[:, None, :] * t[None, k:k + 500, None]).reshape(-1, 3)
        hit |= points_in_box(P, C).reshape(u.size, -1).any(1)
    assert 0 < sil.sum() < sil.size
    assert (sil.ravel() != hit).mean() < 0.01                  # grazing rays only
