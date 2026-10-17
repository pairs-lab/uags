"""
datasets.py — minimal posed-RGB(-D) loaders for the gsplat mapping backend.

Each dataset yields dicts:
    image : (H, W, 3) uint8 tensor
    depth : (H, W) float32 tensor in meters, or None
    c2w   : (4, 4) float32 camera-to-world (OpenCV convention: x right, y down,
            z forward — what 3DGS / gsplat expect)
    K     : (3, 3) float32 intrinsics
    K_depth : (3, 3) float32 intrinsics of the depth grid, only when the depth has its own
            grid (nor: 192x256 ARKit depth next to a 540x720 image)

Supported layouts:
  * Replica (the iMAP/NICE-SLAM rendering):
      <root>/results/frame%06d.jpg, <root>/results/depth%06d.png, <root>/traj.txt
  * ScanNet (exported):
      <root>/color/%d.jpg, <root>/depth/%d.png, <root>/pose/%d.txt,
      <root>/intrinsic/intrinsic_color.txt
  * 7-Scenes:
      <root>/frame-%06d.color.png, .depth.png, .pose.txt
  * NeRF-Object-Removal ("nor", NeRFObjectRemovalDataset): one ARKit capture per scene,
      before and after an object is removed; depth on its own grid with frame["K_depth"]
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


def read_colmap_cameras(path):
    """{camera id: (model, width, height, params)} from a COLMAP cameras.txt."""
    cams = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if p and not p[0].startswith("#"):
                cams[int(p[0])] = (p[1], int(p[2]), int(p[3]), [float(x) for x in p[4:]])
    return cams


def read_colmap_images(path):
    """{name: (image id, qvec (w, x, y, z), tvec, camera id)} from a COLMAP images.txt: the
    world-to-camera pose x_cam = R(qvec) x_world + tvec. Each image takes two lines, the pose
    and its 2D points (possibly empty), as COLMAP's own reader assumes."""
    out = {}
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            p = line.split()
            if not p or p[0].startswith("#"):
                continue
            out[p[9]] = (int(p[0]), np.array(p[1:5], dtype=np.float64),
                         np.array(p[5:8], dtype=np.float64), int(p[8]))
            f.readline()                                      # the POINTS2D line
    return out


def qvec_to_rotmat(q):
    """Rotation matrix of the quaternion (w, x, y, z) (COLMAP order; normalised first)."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class NeRFObjectRemovalDataset(_PosedRGBD):
    """NeRF-Object-Removal (Weder et al., "Removing Objects From Neural Radiance Fields",
    CVPR 2023, arXiv 2212.11966): per scene one continuous ARKit capture (iPhone LiDAR);
    frames [0, split_start) show the object (part "map"), frames [split_end, N) are taken
    after it was removed (part "update"), the frames in between show the removal ("all" only).
        <root>/images/NNN.jpg                      720x540 colour
        <root>/depth/depth_<i>.bin                 float32 192x256 LiDAR depth [m], row-major
        <root>/confidence/depthConfidence_<i>.bin  uint8 192x256 ARKit confidence 0/1/2
                                                   (low / medium / high)
        <root>/masks_real/NNN.npy                  bool 192x256, the region of the real object
                                                   (a dilated silhouette), in frames before
                                                   AND after the removal (where it was); empty
                                                   when it is out of view and in most removal
                                                   frames
        <root>/masks_synthetic/NNN.npy             bool 192x256, frames >= split_end only: the
                                                   silhouette of a virtual object placed near
                                                   the removed one (not the real object)
        <root>/sparse/{cameras,images}.txt         COLMAP text: one PINHOLE camera per frame,
                                                   world-to-camera poses
        <root>/boxes/bbox.txt                      the object's box, 8 corners, COLMAP frame
        <root>/meta.json                           {split_start, split_end, synthetic_end}
    Geometry, measured by scripts/check_nor.py on the 16 scenes: images.txt holds
    world-to-camera poses with OpenCV camera axes (c2w = its inverse); bbox.txt is in that
    frame (bbox_nerf*.txt are in a normalised NeRF frame, bbox_refined.* is a much larger
    region). The depth, confidence and mask grids are the colour image scaled by 256/720
    (ARKit registers its depth to the colour camera), so K_depth = K with rows 0 and 1
    scaled by 256/W and 192/H.
    Intrinsics: cameras.txt has one camera per frame (ids = image ids) while images.txt names
    camera 1 for every frame; each frame takes its own camera. The principal point moves by
    up to ~12 px within a capture, smoothly from frame to frame (optical stabilisation; fx
    changes by < 0.5 px), so camera 1 would be wrong for most frames.
    Depth: pixels with confidence < min_confidence, non-finite or <= 0 are set to 0, the
    invalid value of this repo (the default keeps high-confidence pixels only; see
    check_nor.py for the measured error per level). The raw confidence stays available.
    `part` (map | update | all) has no default, so a caller cannot mix the two states by
    accident; `stride` applies inside the part (frame numbers start, start + stride, ...).
    Frame dict: image, depth (on its own grid), K_depth, depth_confidence, c2w, K, index
    (the frame number of the file names), part ("map" | "between" | "update"), and
    mask_real / mask_synthetic (bool, depth grid) when the file exists. downscale() moves the
    depth (and K_depth) onto the image grid but leaves the masks on the raw 192x256 grid,
    whose intrinsics are the raw frame's K_depth."""
    PARTS = ("map", "update", "all")
    DEPTH_HW = (192, 256)                       # ARKit sceneDepth grid (rows, columns)
    BOX_FILE = "bbox.txt"                       # the object's box in the COLMAP frame

    def __init__(self, root, stride=1, part=None, min_confidence=2):
        super().__init__(stride)
        import json
        if part not in self.PARTS:
            raise ValueError(f"nor: part must be one of {self.PARTS}, got {part!r}")
        self.root, self.part, self.min_confidence = root, part, int(min_confidence)
        with open(os.path.join(root, "meta.json")) as f:
            self.meta = json.load(f)
        num = lambda p: int(os.path.splitext(os.path.basename(p))[0])
        self.rgb = sorted(glob.glob(os.path.join(root, "images", "*.jpg")), key=num)
        n = len(self.rgb)
        if [num(p) for p in self.rgb] != list(range(n)):
            raise ValueError(f"{root}/images: frames are not numbered 0..{n - 1}")
        self.split = (int(self.meta["split_start"]), int(self.meta["split_end"]))
        self.start, self.end = dict(map=(0, self.split[0]), update=(self.split[1], n),
                                    all=(0, n))[part]
        self._n = self.end - self.start
        self.c2w, self.K, self.image_hw = self._read_cameras()
        self.K_depth = self.K.copy()
        self.K_depth[:, 0] *= self.DEPTH_HW[1] / self.image_hw[1]
        self.K_depth[:, 1] *= self.DEPTH_HW[0] / self.image_hw[0]
        self.box_path = os.path.join(root, "boxes", self.BOX_FILE)

    def _read_cameras(self):
        """c2w (N, 4, 4) and K (N, 3, 3) float64 per frame number, and the image (H, W)."""
        cams = read_colmap_cameras(os.path.join(self.root, "sparse", "cameras.txt"))
        ims = read_colmap_images(os.path.join(self.root, "sparse", "images.txt"))
        per_image = sorted(cams) == sorted(v[0] for v in ims.values())
        n = len(self.rgb)
        c2w, K, hw = np.zeros((n, 4, 4)), np.zeros((n, 3, 3)), set()
        for i, path in enumerate(self.rgb):
            iid, q, t, cid = ims[os.path.basename(path)]
            model, width, height, (fx, fy, cx, cy) = cams[iid if per_image else cid]
            if model != "PINHOLE":
                raise ValueError(f"{self.root}: camera model {model}, expected PINHOLE")
            K[i] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            R = qvec_to_rotmat(q)
            c2w[i, :3, :3], c2w[i, :3, 3], c2w[i, 3, 3] = R.T, -R.T @ t, 1.0
            hw.add((height, width))
        if len(hw) != 1:
            raise ValueError(f"{self.root}: several image sizes {sorted(hw)}")
        return c2w, K, hw.pop()

    @property
    def frame_numbers(self):
        """The frame numbers this dataset yields, in order."""
        return list(range(self.start, self.end, self.stride))

    def part_of(self, i):
        return "map" if i < self.split[0] else ("update" if i >= self.split[1] else "between")

    def read_depth(self, i):
        """(depth float32 m, confidence uint8), both (192, 256), of frame number i; depth is
        0 where confidence < min_confidence or the value is non-finite or <= 0."""
        h, w = self.DEPTH_HW
        d = np.fromfile(os.path.join(self.root, "depth", f"depth_{i}.bin"), dtype="<f4")
        c = np.fromfile(os.path.join(self.root, "confidence", f"depthConfidence_{i}.bin"),
                        dtype=np.uint8)
        if d.size != h * w or c.size != h * w:
            raise ValueError(f"{self.root} frame {i}: depth / confidence sizes {d.size} / "
                             f"{c.size}, expected {h} x {w}")
        d, c = d.reshape(h, w).astype(np.float32), c.reshape(h, w)
        d[~np.isfinite(d) | (d <= 0) | (c < self.min_confidence)] = 0.0
        return d, c

    def frame(self, idx):
        i = self.start + idx
        stem = os.path.splitext(os.path.basename(self.rgb[i]))[0]
        depth, conf = self.read_depth(i)
        out = dict(image=torch.from_numpy(_imread(self.rgb[i])), depth=torch.from_numpy(depth),
                   K_depth=torch.from_numpy(self.K_depth[i]).float(),
                   depth_confidence=torch.from_numpy(conf),
                   c2w=torch.from_numpy(self.c2w[i]).float(),
                   K=torch.from_numpy(self.K[i]).float(), index=i, part=self.part_of(i))
        for key, sub in (("mask_real", "masks_real"), ("mask_synthetic", "masks_synthetic")):
            p = os.path.join(self.root, sub, stem + ".npy")
            if os.path.exists(p):
                out[key] = torch.from_numpy(np.load(p).astype(bool))
        return out


DATASETS = {
    "replica_semnerf": ReplicaSemNeRFDataset,
    "replica": ReplicaDataset,
    "scannet": ScanNetDataset,
    "7scenes": SevenScenesDataset,
    "nor": NeRFObjectRemovalDataset,
}


def make_dataset(name, root, stride=1, **options):
    """Dataset `name` at `root`; `options` go to its constructor (e.g. part="map" for nor)."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'; options: {sorted(DATASETS)}")
    return DATASETS[name](root, stride=stride, **options)


def _rescale_K_depth(out, frame, hw):
    """out["K_depth"] = frame["K_depth"] for frame["depth"] resized to hw (if it has one)."""
    if frame.get("K_depth") is None:
        return
    Kd = frame["K_depth"].clone()
    Hd, Wd = frame["depth"].shape[:2]
    Kd[0] *= hw[1] / Wd
    Kd[1] *= hw[0] / Hd
    out["K_depth"] = Kd


def downscale(frame, factor):
    """Downscale image/depth/K by an integer factor (for feature-resolution work).

    Depth on its own grid is resized onto the output image grid (nearest neighbour), which
    assumes it covers the colour image's field of view (ARKit depth does); a K_depth is
    rescaled with it, so it keeps describing the returned depth."""
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
            _rescale_K_depth(out, frame, (H, W))
            return out
        return frame
    h, w = H // factor, W // factor
    out = dict(frame)
    out["image"] = torch.from_numpy(
        cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
    if frame.get("depth") is not None:
        out["depth"] = torch.from_numpy(cv2.resize(
            frame["depth"].numpy(), (w, h), interpolation=cv2.INTER_NEAREST))
        _rescale_K_depth(out, frame, (h, w))
    K = frame["K"].clone()
    K[0] *= w / W
    K[1] *= h / H
    out["K"] = K
    return out
