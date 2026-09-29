"""
datasets.py — minimal posed-RGB(-D) loaders for the gsplat mapping backend.

Each dataset yields dicts:
    image : (H, W, 3) uint8 tensor
    depth : (H, W) float32 tensor in meters, or None
    c2w   : (4, 4) float32 camera-to-world (OpenCV convention: x right, y down,
            z forward — what 3DGS / gsplat expect)
    K     : (3, 3) float32 intrinsics

Supported layouts:
  * Replica (the iMAP/NICE-SLAM rendering):
      <root>/results/frame%06d.jpg, <root>/results/depth%06d.png, <root>/traj.txt
  * ScanNet (exported):
      <root>/color/%d.jpg, <root>/depth/%d.png, <root>/pose/%d.txt,
      <root>/intrinsic/intrinsic_color.txt
  * 7-Scenes:
      <root>/frame-%06d.color.png, .depth.png, .pose.txt
"""
from __future__ import annotations
import os
import glob
import numpy as np
import torch


def _imread(path):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class _PosedRGBD:
    def __init__(self, stride=1):
        self.stride = max(1, int(stride))

    def __len__(self):
        return (self._n + self.stride - 1) // self.stride

    def __getitem__(self, i):
        return self.frame(i * self.stride)

    def frame(self, idx):
        raise NotImplementedError


class ReplicaDataset(_PosedRGBD):
    W, H = 1200, 680
    FX = FY = 600.0
    CX, CY = 599.5, 339.5
    DEPTH_SCALE = 6553.5

    def __init__(self, root, stride=1):
        super().__init__(stride)
        self.root = root
        self.rgb = sorted(glob.glob(os.path.join(root, "results", "frame*.jpg")))
        self.dep = sorted(glob.glob(os.path.join(root, "results", "depth*.png")))
        traj = np.loadtxt(os.path.join(root, "traj.txt")).reshape(-1, 4, 4)
        self._n = len(self.rgb)
        self.poses = traj[:self._n]
        self.K = torch.tensor([[self.FX, 0, self.CX],
                               [0, self.FY, self.CY],
                               [0, 0, 1]], dtype=torch.float32)

    def frame(self, idx):
        img = torch.from_numpy(_imread(self.rgb[idx]))
        depth = None
        if idx < len(self.dep):
            depth = torch.from_numpy(
                _imread(self.dep[idx]).astype(np.float32)) / self.DEPTH_SCALE
        c2w = torch.from_numpy(self.poses[idx].astype(np.float32))
        return dict(image=img, depth=depth, c2w=c2w, K=self.K.clone())


class ScanNetDataset(_PosedRGBD):
    DEPTH_SCALE = 1000.0

    def __init__(self, root, stride=1):
        super().__init__(stride)
        self.root = root
        self.rgb = sorted(glob.glob(os.path.join(root, "color", "*.jpg")),
                          key=lambda p: int(os.path.basename(p).split(".")[0]))
        self._n = len(self.rgb)
        K = np.loadtxt(os.path.join(root, "intrinsic", "intrinsic_color.txt"))
        self.K = torch.from_numpy(K[:3, :3].astype(np.float32))

    def frame(self, idx):
        stem = os.path.basename(self.rgb[idx]).split(".")[0]
        img = torch.from_numpy(_imread(self.rgb[idx]))
        dp = os.path.join(self.root, "depth", stem + ".png")
        depth = None
        if os.path.exists(dp):
            d = _imread(dp).astype(np.float32) / self.DEPTH_SCALE
            depth = torch.from_numpy(d)
        pose = np.loadtxt(os.path.join(self.root, "pose", stem + ".txt"))
        c2w = torch.from_numpy(pose.astype(np.float32))
        return dict(image=img, depth=depth, c2w=c2w, K=self.K.clone())


class SevenScenesDataset(_PosedRGBD):
    FX = FY = 585.0
    CX, CY = 320.0, 240.0
    DEPTH_SCALE = 1000.0
    DEPTH_INVALID = 65535

    def __init__(self, root, stride=1):
        super().__init__(stride)
        self.root = root
        self.rgb = sorted(glob.glob(os.path.join(root, "frame-*.color.png")))
        self._n = len(self.rgb)
        self.K = torch.tensor([[self.FX, 0, self.CX],
                               [0, self.FY, self.CY],
                               [0, 0, 1]], dtype=torch.float32)

    def frame(self, idx):
        base = self.rgb[idx][:-len(".color.png")]
        img = torch.from_numpy(_imread(self.rgb[idx]))
        d = _imread(base + ".depth.png").astype(np.float32)
        d[d == self.DEPTH_INVALID] = 0.0
        depth = torch.from_numpy(d / self.DEPTH_SCALE)
        pose = np.loadtxt(base + ".pose.txt")
        c2w = torch.from_numpy(pose.astype(np.float32))
        return dict(image=img, depth=depth, c2w=c2w, K=self.K.clone())


class ReplicaSemNeRFDataset(_PosedRGBD):
    """Replica as rendered by Semantic-NeRF (Zhi et al., ICCV 2021):
        <root>/<scene>/Sequence_<k>/{rgb/rgb_<i>.png, depth/depth_<i>.png,
                                     semantic_class/semantic_class_<i>.png, traj_w_c.txt}
        <root>/semantic_info/<scene>/info_semantic.json
    640x480, 90 deg horizontal FoV (fx = fy = 320, c = ((W-1)/2, (H-1)/2)), depth in mm,
    traj_w_c = camera-to-world in the OpenCV convention (the dataset readme). Labels are the
    ids of info_semantic.json "classes" (1..101); 0 = unlabelled.
    `root` may be the scene's Sequence_<k> folder or (root, scene, sequence) via make()."""
    W, H = 640, 480
    HFOV_DEG = 90.0
    DEPTH_SCALE = 1000.0

    def __init__(self, root, stride=1):
        super().__init__(stride)
        import math
        import re
        self.root = root
        num = lambda p: int(re.findall(r"(\d+)\.png$", p)[0])
        self.rgb = sorted(glob.glob(os.path.join(root, "rgb", "rgb_*.png")), key=num)
        self.dep = sorted(glob.glob(os.path.join(root, "depth", "depth_*.png")), key=num)
        self.sem = sorted(glob.glob(os.path.join(root, "semantic_class", "semantic_class_*.png")),
                          key=num)
        self._n = len(self.rgb)
        self.poses = np.loadtxt(os.path.join(root, "traj_w_c.txt")).reshape(-1, 4, 4)
        assert len(self.poses) == self._n, (len(self.poses), self._n)
        f = self.W / 2.0 / math.tan(math.radians(self.HFOV_DEG / 2.0))
        self.K = torch.tensor([[f, 0, (self.W - 1) / 2.0],
                               [0, f, (self.H - 1) / 2.0],
                               [0, 0, 1]], dtype=torch.float32)

    def frame(self, idx):
        img = torch.from_numpy(_imread(self.rgb[idx]))
        depth = torch.from_numpy(_imread(self.dep[idx]).astype(np.float32) / self.DEPTH_SCALE)
        labels = torch.from_numpy(_imread(self.sem[idx]).astype(np.int64)) if self.sem else None
        c2w = torch.from_numpy(self.poses[idx].astype(np.float32))
        return dict(image=img, depth=depth, c2w=c2w, K=self.K.clone(), labels=labels,
                    index=idx)

    @staticmethod
    def classes(dataset_root, scene):
        """{class id: name} from semantic_info/<scene>/info_semantic.json (0 = unlabelled)."""
        import json
        info = json.load(open(os.path.join(dataset_root, "semantic_info", scene,
                                           "info_semantic.json")))
        return {int(c["id"]): c["name"] for c in info["classes"]}


DATASETS = {
    "replica_semnerf": ReplicaSemNeRFDataset,
    "replica": ReplicaDataset,
    "scannet": ScanNetDataset,
    "7scenes": SevenScenesDataset,
}


def make_dataset(name, root, stride=1):
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'; options: {sorted(DATASETS)}")
    return DATASETS[name](root, stride=stride)


def downscale(frame, factor):
    """Downscale image/depth/K by an integer factor (for feature-resolution work)."""
    import cv2
    img = frame["image"].numpy()
    H, W = img.shape[:2]
    if factor <= 1:
        dep = frame.get("depth")
        if dep is not None and tuple(dep.shape) != (H, W):
            # e.g. ScanNet: 480x640 depth vs 968x1296 colour -> depth on the colour grid
            out = dict(frame)
            out["depth"] = torch.from_numpy(cv2.resize(dep.numpy(), (W, H),
                                                       interpolation=cv2.INTER_NEAREST))
            return out
        return frame
    h, w = H // factor, W // factor
    out = dict(frame)
    out["image"] = torch.from_numpy(
        cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
    if frame.get("depth") is not None:
        out["depth"] = torch.from_numpy(cv2.resize(
            frame["depth"].numpy(), (w, h), interpolation=cv2.INTER_NEAREST))
    K = frame["K"].clone()
    K[0] *= w / W
    K[1] *= h / H
    out["K"] = K
    return out
