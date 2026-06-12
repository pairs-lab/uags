"""
gs_io.py — load pre-trained 3D Gaussian Splatting checkpoints.

Supports:
  * INRIA / graphdeco ``point_cloud.ply`` (the de-facto interchange format
    written by the original 3DGS, Splatfacto, OpenSplat, ...). Stored values
    are pre-activation: scales are log, opacities are logits, rotations are
    unnormalized wxyz quaternions, color is SH (f_dc + f_rest).
  * gsplat-style ``.pt`` checkpoints holding a dict with keys
    means / quats / scales / opacities (+ sh0/shN or colors). Pre- vs.
    post-activation is auto-detected from value ranges.

Returns a plain dict of post-activation tensors ready for gsplat:
    means (N,3), quats (N,4) unit wxyz, scales (N,3) positive,
    opacities (N,) in (0,1), colors (N,3) in [0,1] (SH degree 0 only).
"""
from __future__ import annotations
import torch

SH_C0 = 0.28209479177387814


def load_gaussians_ply(path, device="cpu", dtype=torch.float32):
    from plyfile import PlyData
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    names = {p.name for p in v.properties}

    def col(*keys):
        return torch.stack(
            [torch.from_numpy(v[k].copy()) for k in keys], dim=-1).to(device, dtype)

    means = col("x", "y", "z")
    scales = torch.exp(col("scale_0", "scale_1", "scale_2"))
    quats = col("rot_0", "rot_1", "rot_2", "rot_3")          # wxyz
    quats = torch.nn.functional.normalize(quats, dim=-1)
    opacities = torch.sigmoid(
        torch.from_numpy(v["opacity"].copy()).to(device, dtype))
    if {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
        colors = (SH_C0 * col("f_dc_0", "f_dc_1", "f_dc_2") + 0.5).clamp(0, 1)
    else:
        colors = torch.full_like(means, 0.5)
    return dict(means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors)


def load_gaussians_pt(path, device="cpu", dtype=torch.float32):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    # common containers: {"splats": {...}} (gsplat examples) or flat dict
    if "splats" in ckpt:
        ckpt = ckpt["splats"]
    g = {k: torch.as_tensor(ckpt[k]).to(device, dtype)
         for k in ("means", "quats", "scales", "opacities") if k in ckpt}
    if len(g) < 4:
        raise KeyError(f"checkpoint {path} missing keys; found {sorted(ckpt)}")
    # auto-detect pre-activation storage
    if g["scales"].min() < 0:
        g["scales"] = torch.exp(g["scales"])
    if g["opacities"].min() < 0 or g["opacities"].max() > 1:
        g["opacities"] = torch.sigmoid(g["opacities"])
    g["opacities"] = g["opacities"].reshape(-1)
    g["quats"] = torch.nn.functional.normalize(g["quats"], dim=-1)
    if "sh0" in ckpt:
        sh0 = torch.as_tensor(ckpt["sh0"]).to(device, dtype).reshape(-1, 3)
        g["colors"] = (SH_C0 * sh0 + 0.5).clamp(0, 1)
    elif "colors" in ckpt:
        g["colors"] = torch.as_tensor(ckpt["colors"]).to(device, dtype)
    else:
        g["colors"] = torch.full_like(g["means"], 0.5)
    return g


def load_gaussians(path, device="cpu", dtype=torch.float32):
    p = str(path)
    if p.endswith(".ply"):
        return load_gaussians_ply(p, device, dtype)
    return load_gaussians_pt(p, device, dtype)
