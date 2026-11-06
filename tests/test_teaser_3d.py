"""papers/figure_tools/teaser_3d.py geometry on the CPU (the render itself needs gsplat):
 * look_at gives a right-handed OpenCV camera (x right, y down, z forward) that sees the target
   at the image centre and a point above it (world up = -y) above the centre;
 * free_camera follows its rule: horizontal distance dist x that of the reference camera, turned
   by the azimuth about the up axis, height m above the target;
 * the frustum corners lie at the given depth along the camera's z axis and project to the
   image corners of their own camera.
"""
import importlib.util
import math
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("teaser_3d", REPO / "papers" / "figure_tools" / "teaser_3d.py")
T3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T3)
K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


def test_look_at_camera():
    eye, target = np.array([2.0, -1.0, -3.0]), np.array([0.5, 0.3, -1.0])
    c2w = T3.look_at(eye, target)
    R = c2w[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and math.isclose(np.linalg.det(R), 1.0)
    uv, z = T3.project(np.stack([target, target + 0.1 * T3.UP]), c2w, K)
    assert np.allclose(uv[0], [320, 240]) and z[0] > 0
    assert uv[1, 1] < uv[0, 1]                         # 'up' in the world is up in the image


def test_free_camera_rule():
    target, cam = np.array([0.0, 0.0, 0.0]), np.array([1.0, -0.5, 2.0])
    up = T3.UP
    horiz = lambda v: v - (v @ up) * up
    for az in (0.0, 35.0, -90.0):
        eye, c2w = T3.free_camera(target, cam, 2.6, 1.4, az)
        d = eye - target
        assert math.isclose(d @ up, 1.4, abs_tol=1e-12)
        assert math.isclose(np.linalg.norm(horiz(d)), 2.6 * np.linalg.norm(horiz(cam - target)))
        a = math.degrees(math.atan2(np.cross(horiz(cam - target), horiz(d)) @ up,
                                    horiz(cam - target) @ horiz(d)))
        assert math.isclose(a, az, abs_tol=1e-9)
        uv, _ = T3.project(target[None], c2w, K)
        assert np.allclose(uv[0], [320, 240])


def test_frustum_corners():
    c2w = T3.look_at(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    P = T3.frustum(c2w, K, (480, 640), 0.25)
    assert np.allclose(P[0], 0) and np.allclose(P[1:, 2], 0.25)
    uv, z = T3.project(P[1:], c2w, K)
    assert np.allclose(uv, [[0, 0], [640, 0], [640, 480], [0, 480]]) and np.allclose(z, 0.25)
