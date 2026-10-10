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
      <root>/intrinsic/intrinsic_color.txt, [<root>/label-filt/%d.png] (ScanNetDataset)
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
    def __init__(self, stride=1, offset=0):
        self.stride = max(1, int(stride))
        self.offset = int(offset)            # frames offset, offset + stride, ... (default 0)
        if self.offset < 0:
            raise ValueError(f"negative frame offset {offset}")

    def __len__(self):
        return (max(0, self._n - self.offset) + self.stride - 1) // self.stride

    def __getitem__(self, i):
        return self.frame(self.offset + i * self.stride)

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


SCANNET_LABEL_MAP = "scannetv2-labels.combined.tsv"


def scannet_label_lut(path, to="nyu40id"):
    """(max id + 1,) int64 lookup table: raw ScanNet label id -> column `to` of
    scannetv2-labels.combined.tsv (id -> nyu40id by default); ids the file does not list -> 0."""
    import csv
    with open(path, newline="") as f:
        rd = csv.DictReader(f, delimiter="\t")
        if not rd.fieldnames or "id" not in rd.fieldnames or to not in rd.fieldnames:
            raise ValueError(f"{path}: needs columns 'id' and '{to}' (got {rd.fieldnames})")
        pairs = []
        for row in rd:
            try:
                pairs.append((int(row["id"]), int(row[to])))
            except (TypeError, ValueError):
                continue                                    # blank or malformed row
    if not pairs:
        raise ValueError(f"{path}: no 'id' -> '{to}' rows")
    lut = np.zeros(max(i for i, _ in pairs) + 1, np.int64)
    for i, c in pairs:
        lut[i] = c
    return lut


def _nearest_labels(lab, h, w):
    """Label map (H, W) -> (h, w) by centre-aligned nearest neighbour (no blending; the rule of
    scripts/eval_seg.py _nearest)."""
    H, W = lab.shape
    if (H, W) == (h, w):
        return lab
    r = np.minimum(((np.arange(h) + 0.5) * H / h).astype(np.int64), H - 1)
    c = np.minimum(((np.arange(w) + 0.5) * W / w).astype(np.int64), W - 1)
    return lab[r][:, c]


class ScanNetDataset(_PosedRGBD):
    """ScanNet v2 (Dai et al., CVPR 2017) as exported by the official SensReader, with
    scene*_2d-label-filt.zip unzipped into the scan folder:
        <root>/color/<i>.jpg, depth/<i>.png (mm, 0 = none), pose/<i>.txt (camera-to-world,
        OpenCV), intrinsic/intrinsic_color.txt [, intrinsic_depth.txt], [label-filt/<i>.png]
    <i> = frame number in the .sens stream. The dataset holds the sorted frames offset,
    offset + stride, ... (frame numbers for a complete export, which prepare_scannet checks);
    each frame carries index = its frame number.

    Labels (label-filt present): the 16-bit raw ScanNet ids are mapped through
    scannetv2-labels.combined.tsv (id -> nyu40id; ids it does not list -> 0, unannotated) and
    resized to the colour resolution by centre-aligned nearest neighbour. NYU40 because the
    ScanNet class sets used for evaluation are lists of NYU40 ids: the 20-class ScanNet
    benchmark and its 19-class subset without "other furniture", on which open-vocabulary
    3DGS work reports ScanNet (OpenGaussian, NeurIPS 2024, scripts/eval_scannet.py target_id;
    NormLift follows that protocol). Which subset is evaluated is the vocabulary's choice
    (configs/vocab_scannet.json: those 19 classes, every other NYU40 id ignored), not the
    loader's. The tsv is `label_map`, else the first of <root>, <root>/.., <root>/../.. that
    holds it (the ScanNet download keeps it beside scans/); label-filt without it is an error
    (raw ids are not class ids). Without label-filt, labels is None.

    Depth (640x480) is not registered to the colour image (1296x968; the normalised
    intrinsics differ by about 1%): intrinsic_depth.txt is returned as K_depth, which
    usplat.gs_fit's depth init uses (usplat.datasets.downscale only resamples the depth onto
    the colour grid). A lost-tracking pose (-inf entries) is returned as is; the consumers
    skip non-finite poses, as for every loader here (eval_seg.py load_frames, run_mapping.py
    select, gs_fit.fit_dataset)."""
    DEPTH_SCALE = 1000.0

    def __init__(self, root, stride=1, offset=0, label_map=None):
        super().__init__(stride, offset)
        self.root = root
        self.rgb = sorted(glob.glob(os.path.join(root, "color", "*.jpg")),
                          key=lambda p: int(os.path.basename(p).split(".")[0]))
        self._n = len(self.rgb)
        K = np.loadtxt(os.path.join(root, "intrinsic", "intrinsic_color.txt"))
        self.K = torch.from_numpy(K[:3, :3].astype(np.float32))
        kd = os.path.join(root, "intrinsic", "intrinsic_depth.txt")
        self.K_depth = (torch.from_numpy(np.loadtxt(kd)[:3, :3].astype(np.float32))
                        if os.path.exists(kd) else None)
        self.label_dir = os.path.join(root, "label-filt")
        self.label_map, self.lut = None, None
        if os.path.isdir(self.label_dir):
            self.label_map = label_map or self.find_label_map(root)
            if self.label_map is None:
                raise FileNotFoundError(
                    f"{self.label_dir} holds raw ScanNet ids but no {SCANNET_LABEL_MAP} was "
                    f"found in {root} or its two parents (pass label_map=)")
            self.lut = scannet_label_lut(self.label_map)

    @staticmethod
    def find_label_map(root):
        """The first of <root>, <root>/.., <root>/../.. holding scannetv2-labels.combined.tsv."""
        for d in (root, os.path.join(root, os.pardir), os.path.join(root, os.pardir, os.pardir)):
            p = os.path.join(d, SCANNET_LABEL_MAP)
            if os.path.isfile(p):
                return os.path.normpath(p)
        return None

    def labels(self, stem, hw):
        """NYU40 ids (h, w) int64 of frame `stem` on the colour grid hw; None without a label."""
        p = os.path.join(self.label_dir, stem + ".png")
        if self.lut is None or not os.path.exists(p):
            return None
        raw = _imread(p)
        if raw.ndim != 2:
            raise ValueError(f"{p}: expected a single-channel label map")
        raw = raw.astype(np.int64)
        lab = np.where(raw < len(self.lut), self.lut[np.minimum(raw, len(self.lut) - 1)], 0)
        return torch.from_numpy(np.ascontiguousarray(_nearest_labels(lab, *hw)))

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
        out = dict(image=img, depth=depth, c2w=c2w, K=self.K.clone(),
                   labels=self.labels(stem, tuple(img.shape[:2])), index=int(stem))
        if self.K_depth is not None:
            out["K_depth"] = self.K_depth.clone()
        return out


def prepare_scannet(src, dst, factor=2, quality=95, label_map=None):
    """Write the ScanNet scan `src` (SensReader export) with its colour downscaled by `factor`
    into `dst`, a layout ScanNetDataset reads: each colour frame resized as
    usplat.datasets.downscale does (cv2 INTER_AREA to (W // f, H // f)) and stored as JPEG
    (quality `quality`), intrinsic_color.txt scaled by w / W and h / H (downscale's rule), the
    other intrinsic files copied, depth/, pose/, label-filt/ and the label map symlinked.
    dst/prepare.json records the settings (status "running" until every frame is written);
    a rerun with the same settings only fills in missing frames, other settings are refused.

    Why: at the native 1296x968 LSeg would run at 4x the pixels of Replica's 640x480 frames
    (its settings were frozen on Replica validation) and run_mapping.py's CPU token store
    (fp32, token stride 8) would hold ~40 MB per mapping frame; at f = 2 (648x484) the
    frames match Replica and the depth's 640x480, as ScanNet work runs at reduced resolution
    too (640x480: NICE-SLAM, SplaTAM; 320x240: OpenScene's 2D export). The export must be
    complete (frame numbers 0..n-1, a pose file per frame) so that stride / offset select
    frame numbers; ValueError otherwise."""
    import cv2
    import json
    import shutil
    f = int(factor)
    if f < 1:
        raise ValueError(f"factor must be >= 1 (got {factor})")
    num = lambda p: int(os.path.basename(p).split(".")[0])
    ids = sorted(num(p) for p in glob.glob(os.path.join(src, "color", "*.jpg")))
    if not ids:
        raise FileNotFoundError(f"{src}/color: no frames")
    if ids != list(range(len(ids))):
        miss = sorted(set(range(ids[-1] + 1)) - set(ids))
        raise ValueError(f"{src}: frame numbers are not 0..{len(ids) - 1} (missing {miss[:10]}"
                         f"{' ...' if len(miss) > 10 else ''}); export every frame (frame_skip 1)")
    poses = {num(p) for p in glob.glob(os.path.join(src, "pose", "*.txt"))}
    if not set(ids) <= poses:
        raise ValueError(f"{src}/pose: {len(set(ids) - poses)} frames have no pose file")
    lm = label_map or ScanNetDataset.find_label_map(src)
    if os.path.isdir(os.path.join(src, "label-filt")) and lm is None:
        raise FileNotFoundError(f"{src}/label-filt needs {SCANNET_LABEL_MAP} (pass label_map=)")
    H, W = _imread(os.path.join(src, "color", "0.jpg")).shape[:2]
    h, w = H // f, W // f
    K = np.loadtxt(os.path.join(src, "intrinsic", "intrinsic_color.txt"))
    Ks = K.copy()
    Ks[0] *= w / W
    Ks[1] *= h / H
    rec = dict(src=os.path.abspath(src), factor=f, quality=int(quality), n_frames=len(ids),
               native_hw=[H, W], hw=[h, w], K_native=K[:3, :3].tolist(), K=Ks[:3, :3].tolist(),
               label_map=None if lm is None else os.path.abspath(lm), status="running")
    rp = os.path.join(dst, "prepare.json")
    if os.path.exists(rp):
        old = json.load(open(rp))
        diff = {k: (old.get(k), rec[k]) for k in ("src", "factor", "quality", "n_frames", "hw")
                if old.get(k) != rec[k]}
        if diff:
            raise ValueError(f"{dst} was prepared with other settings {diff}; remove it first")
    os.makedirs(os.path.join(dst, "color"), exist_ok=True)
    os.makedirs(os.path.join(dst, "intrinsic"), exist_ok=True)
    with open(rp, "w") as fh:
        json.dump(rec, fh, indent=1)
    for i in ids:
        out = os.path.join(dst, "color", f"{i}.jpg")
        if os.path.exists(out):
            continue
        img = cv2.imread(os.path.join(src, "color", f"{i}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(os.path.join(src, "color", f"{i}.jpg"))
        if img.shape[:2] != (H, W):
            raise ValueError(f"{src}/color/{i}.jpg: {img.shape[:2]} != {(H, W)} of frame 0")
        ok, buf = cv2.imencode(".jpg", cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA),
                               [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise OSError(f"JPEG encoding failed for frame {i}")
        with open(out + ".part", "wb") as fh:           # atomic: never a truncated frame
            fh.write(buf.tobytes())
        os.replace(out + ".part", out)
    np.savetxt(os.path.join(dst, "intrinsic", "intrinsic_color.txt"), Ks, fmt="%.6f")
    for p in glob.glob(os.path.join(src, "intrinsic", "*.txt")):
        if os.path.basename(p) != "intrinsic_color.txt":
            shutil.copy2(p, os.path.join(dst, "intrinsic", os.path.basename(p)))
    for name, target in (("depth", os.path.join(src, "depth")),
                         ("pose", os.path.join(src, "pose")),
                         ("label-filt", os.path.join(src, "label-filt")),
                         (SCANNET_LABEL_MAP, lm)):
        if target and os.path.exists(target) and not os.path.lexists(os.path.join(dst, name)):
            os.symlink(os.path.abspath(target), os.path.join(dst, name))
    rec["status"] = "ok"
    with open(rp, "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


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


def make_dataset(name, root, stride=1, offset=0):
    """offset: index of the first frame (then every stride-th), for the loaders that take one
    (ScanNet: held-out evaluation frames between the mapping frames of the same scan)."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'; options: {sorted(DATASETS)}")
    if offset:
        import inspect
        if "offset" not in inspect.signature(DATASETS[name]).parameters:
            raise ValueError(f"dataset '{name}' takes no frame offset")
        return DATASETS[name](root, stride=stride, offset=offset)
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
