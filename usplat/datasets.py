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
      <root>/intrinsic/intrinsic_color.txt, [<root>/label-filt/%d.png] (ScanNetDataset)
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
    def __init__(self, stride=1, offset=0):
        self.stride = max(1, int(stride))
        self.offset = int(offset)            # frames offset, offset + stride, ... (default 0)
        if self.offset < 0:
            raise ValueError(f"negative frame offset {offset}")

    def __len__(self):
        return (max(0, self._n - self.offset) + self.stride - 1) // self.stride

    def __getitem__(self, i):
        # no negative or out-of-range index: a dataset part (nor: map | update) must never
        # walk into the neighbouring frames of the capture (also ends a for-loop over ds)
        if not 0 <= i < len(self):
            raise IndexError(f"frame index {i} outside 0..{len(self) - 1}")
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
    loader's. (OpenGaussian scores 3D points, the *_vh_clean_2.labels.ply vertices; these are
    2D per-pixel labels, so its numbers are not comparable with ours.) The tsv is
    `label_map`, else the first of <root>, <root>/.., <root>/../.. that holds it (the ScanNet
    download keeps it beside scans/); label-filt without it is an error (raw ids are not
    class ids). Without label-filt, labels is None. eval_seg.py resamples the labels once
    more onto its render grid, again by nearest neighbour: on a prepared scan (labels
    968x1296 -> colour 484x648 -> render 242x324) the two steps pick a label pixel about one
    native pixel (1/4 render pixel) off the direct nearest one, a small boundary shift shared
    by every row.

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


def scannet_frame_view(src, dst, stride, offset=0):
    """Write `dst`, a ScanNet layout holding only the frames offset, offset + stride, ... of
    the scan `src` (positions in its sorted colour frames, as ScanNetDataset selects them)
    whose pose is finite: color/<i>.jpg symlinked per frame; depth/, pose/, label-filt/,
    intrinsic/ and the label map symlinked whole. ScanNetDataset(dst) (stride 1) then holds
    exactly the frames scripts/eval_seg.py evaluates with that stride and offset (it skips
    lost-tracking poses). dst/view.json records the settings and frame numbers; a rerun with
    the same settings only fills in missing links, other settings are refused.

    Why: run_mapping.py's css_vocabulary gives the closed-set CSS baseline the classes of
    --css-eval-data read with --css-eval-stride (no offset). On Replica those are the
    evaluation frames; over a whole ScanNet scan (e.g. every 5th frame) they would add the
    classes of mapping frames, columns whose votes CSS's rows(class_ids) then discards."""
    import json
    stride, offset = int(stride), int(offset)
    if stride < 1 or offset < 0:
        raise ValueError(f"stride must be >= 1 and offset >= 0 (got {stride}, {offset})")
    rgb = sorted(glob.glob(os.path.join(src, "color", "*.jpg")),
                 key=lambda p: int(os.path.basename(p).split(".")[0]))
    if not rgb:
        raise FileNotFoundError(f"{src}/color: no frames")
    frames, lost = [], []
    for p in rgb[offset::stride]:
        stem = os.path.basename(p).split(".")[0]
        pose = np.loadtxt(os.path.join(src, "pose", stem + ".txt"))
        (frames if np.isfinite(pose).all() else lost).append(int(stem))
    rec = dict(src=os.path.abspath(src), stride=stride, offset=offset, frames=frames, lost=lost)
    rp = os.path.join(dst, "view.json")
    if os.path.exists(rp):
        old = json.load(open(rp))
        diff = sorted(k for k, v in rec.items() if old.get(k) != v)
        if diff:
            raise ValueError(f"{dst} is a view with another {', '.join(diff)}; remove it first")
    os.makedirs(os.path.join(dst, "color"), exist_ok=True)
    extra = sorted(set(os.listdir(os.path.join(dst, "color"))) - {f"{i}.jpg" for i in frames})
    if extra:
        raise ValueError(f"{dst}/color holds frames outside the view: {extra[:10]}")
    for i in frames:
        link = os.path.join(dst, "color", f"{i}.jpg")
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(os.path.join(src, "color", f"{i}.jpg")), link)
    lm = ScanNetDataset.find_label_map(src)
    for name, target in (("depth", os.path.join(src, "depth")),
                         ("pose", os.path.join(src, "pose")),
                         ("label-filt", os.path.join(src, "label-filt")),
                         ("intrinsic", os.path.join(src, "intrinsic")),
                         (SCANNET_LABEL_MAP, lm)):
        if target and os.path.exists(target) and not os.path.lexists(os.path.join(dst, name)):
            os.symlink(os.path.abspath(target), os.path.join(dst, name))
    with open(rp + ".tmp", "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(rp + ".tmp", rp)
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
    Update-pose alignment (protocol D3): align = the JSON of usplat/nor_align.py (a path, or
    its dict) gives one rigid T (4x4, x_map = T x_update) per scene; every frame of the update
    part (frame numbers >= split_end) then gets c2w = T c2w, every other frame keeps its pose.
    The JSON must name this scene folder ("scene"). align=None: the poses of images.txt.
    Frame dict: image, depth (on its own grid), K_depth, depth_confidence, c2w, K, index
    (the frame number of the file names), part ("map" | "between" | "update"), and
    mask_real / mask_synthetic (bool, depth grid) when the file exists. downscale() moves the
    depth (and K_depth) onto the image grid but leaves the masks on the raw 192x256 grid,
    whose intrinsics are the raw frame's K_depth."""
    PARTS = ("map", "update", "all")
    DEPTH_HW = (192, 256)                       # ARKit sceneDepth grid (rows, columns)
    BOX_FILE = "bbox.txt"                       # the object's box in the COLMAP frame

    def __init__(self, root, stride=1, part=None, min_confidence=2, align=None):
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
        self.align_T = self._read_align(align)
        if self.align_T is not None:                           # D3: the update part only
            self.c2w[self.split[1]:] = self.align_T @ self.c2w[self.split[1]:]
        self.K_depth = self.K.copy()
        self.K_depth[:, 0] *= self.DEPTH_HW[1] / self.image_hw[1]
        self.K_depth[:, 1] *= self.DEPTH_HW[0] / self.image_hw[0]
        self.box_path = os.path.join(root, "boxes", self.BOX_FILE)

    def _read_align(self, align):
        """T (4, 4) float64 of an alignment record (JSON path or dict), None for None. The
        record must name this scene folder and T must be rigid."""
        import json
        if align is None:
            return None
        if isinstance(align, (str, os.PathLike)):
            with open(align) as f:
                align = json.load(f)
        here = os.path.basename(os.path.normpath(self.root))
        if align.get("scene") != here:
            raise ValueError(f"nor: the alignment is for scene {align.get('scene')!r}, "
                             f"not {here!r}")
        T = np.asarray(align["T"], dtype=np.float64)
        R = T[:3, :3] if T.shape == (4, 4) else None
        if (R is None or not np.allclose(T[3], [0, 0, 0, 1]) or not np.allclose(R.T @ R, np.eye(3),
                                                                             atol=1e-6)
                or np.linalg.det(R) < 0):
            raise ValueError(f"nor: alignment T of {here} is not a rigid 4x4 transform")
        return T

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


def make_dataset(name, root, stride=1, offset=0, **options):
    """Dataset `name` at `root`; `options` go to its constructor (e.g. part="map" for nor).

    offset selects the first frame, then every stride-th, for loaders that support it
    (ScanNet: held-out evaluation frames between the mapping frames of the same scan).
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'; options: {sorted(DATASETS)}")
    if offset:
        import inspect
        if "offset" not in inspect.signature(DATASETS[name]).parameters:
            raise ValueError(f"dataset '{name}' takes no frame offset")
        options["offset"] = offset
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
