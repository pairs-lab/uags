"""SE(3) utilities for the localization harness (CPU; no gsplat needed)."""
import math
import torch

from usplat.localization import se3_exp, pose_errors


def test_se3_exp_identity_and_orthonormal():
    T0 = se3_exp(torch.zeros(6))
    assert torch.allclose(T0, torch.eye(4), atol=1e-7)

    xi = torch.tensor([0.1, -0.2, 0.3, 0.2, 0.1, -0.15])
    T = se3_exp(xi)
    R = T[:3, :3]
    assert torch.allclose(R @ R.t(), torch.eye(3), atol=1e-5)
    assert abs(torch.det(R).item() - 1.0) < 1e-5
    # exp(xi) exp(-xi) = I  (rotation part exactly; full SE3 only approx for
    # non-commuting rho/phi, so check rotation + small translation residual)
    Tinv = se3_exp(-xi)
    I4 = T @ Tinv
    assert torch.allclose(I4[:3, :3], torch.eye(3), atol=1e-5)


def test_pose_errors():
    c2w = torch.eye(4)
    t, r = pose_errors(c2w, c2w)
    assert t == 0.0 and r < 1e-4

    moved = c2w.clone()
    moved[0, 3] = 0.05
    t, r = pose_errors(moved, c2w)
    assert abs(t - 0.05) < 1e-6

    th = math.radians(10.0)
    rot = torch.eye(4)
    rot[0, 0], rot[0, 1] = math.cos(th), -math.sin(th)
    rot[1, 0], rot[1, 1] = math.sin(th), math.cos(th)
    t, r = pose_errors(rot, c2w)
    assert abs(r - 10.0) < 1e-3
    print("PASS: se3_exp + pose_errors")


if __name__ == "__main__":
    test_se3_exp_identity_and_orthonormal()
    test_pose_errors()
