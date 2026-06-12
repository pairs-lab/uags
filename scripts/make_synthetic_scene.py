"""
make_synthetic_scene.py — build a tiny synthetic 3DGS scene + Replica-format
posed RGB dataset by rendering the scene itself with gsplat. Lets the full
`run_mapping.py --backend gsplat` CLI (and the localization harness) be
exercised end-to-end with no dataset download. Requires GPU + gsplat.

  python scripts/make_synthetic_scene.py --out /tmp/synth_scene
  python scripts/run_mapping.py --backend gsplat \
      --ply /tmp/synth_scene/scene.pt --data /tmp/synth_scene \
      --dataset replica --clip-pretrained none --feat-dim 16 \
      --stride 1 --downscale 2 --out /tmp/synth_scene/field.pt
"""
from __future__ import annotations
import argparse
import math
import os
import numpy as np
import torch
import torch.nn.functional as Fn


def make_splats(num=2000, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # three colored blobs + a floor slab
    centers = torch.tensor([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, -0.5, 0.4]])
    cols = torch.tensor([[0.9, 0.2, 0.2], [0.2, 0.4, 0.9], [0.2, 0.8, 0.3]])
    per = num // 4
    means, colors = [], []
    for c, col in zip(centers, cols):
        means.append(c + 0.15 * torch.randn(per, 3, generator=g))
        colors.append(col.expand(per, 3) + 0.05 * torch.randn(per, 3, generator=g))
    nfloor = num - 3 * per
    floor = torch.stack([2.0 * (torch.rand(nfloor, generator=g) - 0.5),
                         torch.full((nfloor,), 0.45),
                         2.0 * (torch.rand(nfloor, generator=g) - 0.5)], -1)
    means.append(floor)
    colors.append(torch.tensor([0.6, 0.6, 0.55]).expand(nfloor, 3).clone())
    means = torch.cat(means).to(device)
    colors = torch.cat(colors).clamp(0, 1).to(device)
    quats = Fn.normalize(torch.randn(means.shape[0], 4, generator=g), dim=-1).to(device)
    scales = (0.01 + 0.03 * torch.rand(means.shape[0], 3, generator=g)).to(device)
    opac = (0.6 + 0.39 * torch.rand(means.shape[0], generator=g)).to(device)
    return dict(means=means, quats=quats, scales=scales,
                opacities=opac, colors=colors)


def orbit_poses(n, radius=2.2, height=-0.3):
    """c2w poses on a circle looking at the origin (OpenCV convention)."""
    poses = []
    for i in range(n):
        a = 2 * math.pi * i / n * 0.5 - math.pi * 0.25   # 90-degree arc
        eye = np.array([radius * math.sin(a), height, -radius * math.cos(a)])
        fwd = -eye / np.linalg.norm(eye)                  # z: look at origin
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, fwd); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        c2w = np.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, eye
        poses.append(c2w)
    return np.stack(poses)


def main():
    import cv2
    from gsplat import rasterization
    from usplat.datasets import ReplicaDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--num", type=int, default=2000)
    args = ap.parse_args()

    device = "cuda"
    W, H = ReplicaDataset.W, ReplicaDataset.H            # match replica intrinsics
    K = torch.tensor([[ReplicaDataset.FX, 0, ReplicaDataset.CX],
                      [0, ReplicaDataset.FY, ReplicaDataset.CY],
                      [0, 0, 1.0]], device=device)

    splats = make_splats(args.num, device=device)
    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)
    torch.save(dict(splats={k: v.cpu() for k, v in splats.items()}),
               os.path.join(args.out, "scene.pt"))

    poses = orbit_poses(args.frames)
    np.savetxt(os.path.join(args.out, "traj.txt"), poses.reshape(args.frames, 16))
    for i in range(args.frames):
        viewmat = torch.from_numpy(
            np.linalg.inv(poses[i])).float().to(device)
        img, _, _ = rasterization(
            means=splats["means"], quats=splats["quats"], scales=splats["scales"],
            opacities=splats["opacities"], colors=splats["colors"],
            viewmats=viewmat[None], Ks=K[None], width=W, height=H,
            render_mode="RGB")
        rgb = (img[0].clamp(0, 1) * 255).byte().cpu().numpy()
        cv2.imwrite(os.path.join(args.out, "results", f"frame{i:06d}.jpg"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"wrote {args.frames} frames + scene.pt + traj.txt -> {args.out}")


if __name__ == "__main__":
    main()
