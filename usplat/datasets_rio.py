"""
datasets_rio.py — RIO10 loader (Wald, Sattler, Golodetz, Cavallari, Tombari, "Beyond
Controlled Environments: 3D Camera Re-Localization in Changing Indoor Scenes", ECCV 2020)
and the object annotations the stale ground truth (usplat.change_eval) is built from.

Frame dicts match usplat.datasets:
    image : (H, W, 3) uint8          c2w : (4, 4) float32 camera-to-world (OpenCV axes)
    depth : (H', W') float32 [m] or None (see "Depth")
    K     : (3, 3) float32           index : file frame number     name : "seqXX_YY/frame-%06d"
    instances : (H'', W'') int64 instance ids (when the semantics are present), else absent
    depth_source : "none" | "rendered_mesh" | "external"
so RIO10Dataset(root, stride) can be registered in usplat.datasets.DATASETS as "rio10" (root =
the sequence folder, as for the other loaders).

Layout (docs/reports/2026-09-28-paper-completion-plan/logs/wf/data/rio10_readme.md, l.13-63):
three zips per scene, each holding one folder per sequence seqXX_YY:
    seqXX.zip        seqXX_YY/frame-%06d.color.jpg, .pose.txt, .rendered.depth.png (+ camera.yaml)
    semanticsXX.zip  seqXX_YY/instances.txt, frame-%06d.instances.png
    modelsXX.zip     seqXX_YY/labels.ply, mesh.obj, mesh.refined_0.png, mesh.refined.mtl
seqXX_01 is the reference (mapping) scan, seqXX_02 the validation rescan (update); only these
two carry poses and semantics. Rescan poses are aligned with the reference scan (readme).
RIO10Scene resolves where the three folders were unzipped (_FRAME_DIRS, _SEM_DIRS,
_MODEL_DIRS: several plausible layouts, the first that holds the marker file wins); explicit
paths override the search. Every miss names all the paths tried.

Depth. RIO10 has NO sensor depth: *.rendered.depth.png is rendered from the sequence's own
mesh, i.e. for seqXX_02 it shows the new state of the scene and is an ORACLE for the update.
depth="none" (default) therefore yields depth None, so nothing reads it by accident (e.g.
run_mapping.py --use-depth as the re-admission depth); depth="rendered" loads it (stale GT
visibility and the oracle row only) and depth=<dir> loads a non-oracle depth (Tango via
3RScan, or MASt3R) from <dir>/frame-%06d.depth.png (uint16, mm) or .npy (float, m).

Parts. The plan updates with the FIRST half of seqXX_02 and evaluates semantics on the
SECOND half: part="first_half" | "second_half" | "all" (or the root suffix "::first_half",
so make_dataset(name, root, stride) can carry it); halves split the sorted frame list at
n // 2, then the stride applies within the part.

3RScan fallback (plan §3, "if the instance ids do not match"): load_3rscan_changes reads the
per-rescan "removed", "nonrigid" and "rigid" (instance transform) lists of 3RScan.json for a
(reference, rescan) scan-id pair; the RIO10 -> 3RScan scan-id map is not in RIO10's metadata
and must be supplied (configs/rio10_change.json "rio_to_3rscan").

VERIFY ON REAL DATA (each assumption below is checked at load time where it can be; the rest
must be checked on val scene 01 the day the data arrives):
  V1  sequence folders are named seqXX_YY and the zips unzip into one of the layouts in
      _FRAME_DIRS / _SEM_DIRS / _MODEL_DIRS (checked: marker files, clear error otherwise)
  V2  frame-%06d.pose.txt holds a 4x4 camera-to-world matrix (OpenCV axes: x right, y down,
      z forward), aligned across rescans (checked: shape, last row, orthonormal R, det > 0;
      non-finite poses are passed through and skipped by run_mapping's keyframe selection).
      The OpenCV-axes claim is NOT checkable here: back-project the rendered depth of two
      frames as tests/test_replica_loader.py does
  V3  intrinsics fx fy cx cy from <seq>/camera.yaml (camera_intrinsics.model, as the ACE-G
      setup script reads them) or a line "seqXX_YY fx fy cx cy" of intrinsics.txt, in pixels
      of the colour image (checked: 0 < cx < W and 0 < cy < H; RIO10 is 540x960 portrait,
      cx ~ 270, cy ~ 490, so a landscape/portrait mix-up fails here)
  V4  *.rendered.depth.png is uint16 in millimetres (DEPTH_SCALE = 1000, 0 = no surface)
      (checked: single channel, uint16; the SCALE is not checkable: compare with the mesh)
  V5  frame-%06d.instances.png is a single-channel uint8/uint16 instance-id image whose ids
      are those of instances.txt and of the labels.ply vertex property (checked: channels,
      dtype, one instance image per colour frame)
  V6  instances.txt lines are "<id> <label...>" (whitespace, tab or comma separated);
      duplicate ids are an error
  V7  labels.ply is a vertex cloud/mesh whose vertices carry an integer instance property
      (first of ID_PROPERTIES; 3RScan names it "objectId") in the SAME world frame as the
      poses of its sequence (checked: property present; frame: alignment_check in
      usplat.change_eval on static structure). "label" / "id" are NOT searched: in
      ScanNet-style plys "label" holds semantic class ids, which would merge every chair
      into one object; name it explicitly (id_property, configs/rio10_change.json
      labels_id_property) once checked. The property used is recorded in the ground truth
      and must be the same for seqXX_01 and seqXX_02 (checked by scripts/eval_change.py)
  V8  the instance ids of seqXX_01 and seqXX_02 name the same physical object (plan §3: NOT
      checked; compare label names and the centroids of static ids on val 01)
  V9  3RScan.json transforms are 16 numbers of a 4x4 homogeneous matrix; the storage order is
      detected from where the (0, 0, 0, 1) row sits, else `order` decides (column-major "F"
      by default, as Eigen reads them)
  V10 3RScan route: the instance ids of the RIO10 labels.ply of seqXX_01 ARE the 3RScan
      objectIds of the mapped reference scan, so the removed / nonrigid / rigid lists index
      them (checked in part: usplat.change_eval.rscan_id_check refuses lists naming ids that
      labels.ply lacks; a full check compares with 3RScan's own
      labels.instances.annotated.v2.ply of the reference scan on val 01)
"""
from __future__ import annotations
import glob
import json
import os
import re

import numpy as np
import torch

DEPTH_SCALE = 1000.0                        # V4: rendered depth in mm
ID_PROPERTIES = ("objectId", "objectid", "object_id", "instance", "instance_id",
                 "instanceId")              # V7: never "label" / "id" (class ids) by default
SEQ_RE = re.compile(r"^seq(\d{2})_(\d{2})$")
PARTS = ("all", "first_half", "second_half")
# where seqXX.zip / semanticsXX.zip / modelsXX.zip may have been unzipped, relative to the
# RIO10 root; {s} = scene id "02", {q} = sequence "seq02_01"
_FRAME_DIRS = ("seq{s}/{q}", "scene{s}/seq{s}/{q}", "{q}", "seq{s}/seq{s}/{q}")
_SEM_DIRS = ("semantics{s}/{q}", "scene{s}/semantics{s}/{q}", "semantics/{q}",
             "semantics{s}/semantics{s}/{q}", "seq{s}/{q}", "{q}")
_MODEL_DIRS = ("models{s}/{q}", "scene{s}/models{s}/{q}", "models/{q}",
               "models{s}/models{s}/{q}", "seq{s}/{q}", "{q}")


class RIO10FormatError(ValueError):
    """A RIO10 file or layout does not match the documented format (see VERIFY ON REAL DATA)."""


# ------------------------------------------------------------------------------ readers
def _imread(path):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img


def read_color(path):
    import cv2
    img = _imread(path)
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise RIO10FormatError(f"{path}: expected a colour image, got shape {img.shape}")
    return cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)


def read_pose(path, atol_rot=1e-3, atol_row=1e-4):
    """4x4 camera-to-world (V2). Non-finite matrices are returned as they are (lost tracking);
    finite ones must be rigid."""
    try:
        m = np.loadtxt(path, dtype=np.float64)
    except ValueError as e:
        raise RIO10FormatError(f"{path}: not a numeric pose file ({e})") from None
    if m.size != 16:
        raise RIO10FormatError(f"{path}: expected 16 numbers (4x4 pose), got {m.size}")
    m = m.reshape(4, 4)
    if not np.isfinite(m).all():
        return m
    if not np.allclose(m[3], [0.0, 0.0, 0.0, 1.0], atol=atol_row):
        raise RIO10FormatError(f"{path}: last row {m[3].tolist()} != [0, 0, 0, 1]")
    R = m[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=atol_rot) or np.linalg.det(R) <= 0:
        raise RIO10FormatError(f"{path}: rotation is not orthonormal with det +1 "
                               f"(|R^T R - I| = {np.abs(R.T @ R - np.eye(3)).max():.2e}, "
                               f"det {np.linalg.det(R):.4f})")
    return m


def read_depth_png(path, scale=DEPTH_SCALE):
    """uint16 depth image -> float32 metres, 0 = invalid (V4)."""
    d = _imread(path)
    if d.ndim != 2 or d.dtype != np.uint16:
        raise RIO10FormatError(f"{path}: expected a single-channel uint16 depth image, got "
                               f"shape {d.shape} dtype {d.dtype}")
    return d.astype(np.float32) / float(scale)


def read_instance_png(path):
    """Single-channel uint8/uint16 instance-id image -> int64 (V5)."""
    m = _imread(path)
    if m.ndim != 2 or m.dtype not in (np.uint8, np.uint16):
        raise RIO10FormatError(f"{path}: expected a single-channel uint8/uint16 instance "
                               f"image, got shape {m.shape} dtype {m.dtype}")
    return m.astype(np.int64)


def read_instances_txt(path):
    """{instance id: label} from instances.txt (V6)."""
    out = {}
    with open(path) as f:
        for n, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = [t for t in re.split(r"[\s,]+", s) if t]
            try:
                iid = int(tok[0])
            except ValueError:
                raise RIO10FormatError(f"{path}:{n}: expected '<id> <label>', got {s!r}") from None
            if iid in out:
                raise RIO10FormatError(f"{path}:{n}: duplicate instance id {iid}")
            out[iid] = " ".join(tok[1:]).strip("'\"")
    if not out:
        raise RIO10FormatError(f"{path}: no instances")
    return out


def read_labels_ply(path, id_property=None):
    """labels.ply vertices -> dict(points (P, 3) float64, ids (P,) int64, property) (V7)."""
    from plyfile import PlyData
    ply = PlyData.read(str(path))
    if "vertex" not in [e.name for e in ply.elements]:
        raise RIO10FormatError(f"{path}: no vertex element")
    v = ply["vertex"]
    names = [p.name for p in v.properties]
    for k in ("x", "y", "z"):
        if k not in names:
            raise RIO10FormatError(f"{path}: vertex property {k!r} missing (has {names})")
    prop = id_property or next((p for p in ID_PROPERTIES if p in names), None)
    if prop is None or prop not in names:
        hint = [p for p in ("label", "id") if p in names and not id_property]
        raise RIO10FormatError(f"{path}: no instance-id vertex property; tried "
                               f"{[id_property] if id_property else list(ID_PROPERTIES)}, "
                               f"the file has {names}"
                               + (f" ({hint} may hold semantic class ids: pass "
                                  f"id_property={hint[0]!r} only after checking V7)" if hint
                                  else ""))
    pts = np.stack([np.asarray(v[k], dtype=np.float64) for k in ("x", "y", "z")], -1)
    raw = np.asarray(v[prop])
    if not np.issubdtype(raw.dtype, np.integer):
        if not np.all(np.isfinite(raw)) or np.any(raw != np.round(raw)):
            raise RIO10FormatError(f"{path}: property {prop!r} is not integer-valued")
    if not np.isfinite(pts).all():
        raise RIO10FormatError(f"{path}: non-finite vertex coordinates")
    return dict(points=pts, ids=raw.astype(np.int64), property=prop, path=str(path))


def read_camera_yaml(path):
    """(fx, fy, cx, cy) from camera.yaml: camera_intrinsics.model (V3)."""
    import yaml
    with open(path) as f:
        y = yaml.safe_load(f)
    try:
        m = y["camera_intrinsics"]["model"]
        fx, fy, cx, cy = (float(x) for x in m[:4])
    except (KeyError, TypeError, ValueError):
        raise RIO10FormatError(f"{path}: expected camera_intrinsics.model = [fx, fy, cx, cy]") \
            from None
    return fx, fy, cx, cy


def read_intrinsics_txt(path, sequence):
    """(fx, fy, cx, cy) of `sequence` from RIO10's intrinsics.txt (V3)."""
    with open(path) as f:
        for line in f:
            tok = line.split()
            if tok and tok[0] == sequence:
                if len(tok) < 5:
                    raise RIO10FormatError(f"{path}: line for {sequence} has {len(tok)} fields")
                return tuple(float(x) for x in tok[1:5])
    raise RIO10FormatError(f"{path}: no line for {sequence}")


def _frame_number(path):
    m = re.search(r"frame-(\d+)\.", os.path.basename(path))
    if m is None:
        raise RIO10FormatError(f"{path}: not a frame-%06d.* file name")
    return int(m.group(1))


def _split_part(root):
    if "::" in str(root):
        root, part = str(root).rsplit("::", 1)
        return root, part
    return root, None


# ------------------------------------------------------------------------------ dataset
class RIO10Dataset:
    """One RIO10 sequence (root = its frame folder seqXX_YY). See the module docstring."""
    W, H = 540, 960                          # nominal (portrait); the files decide

    def __init__(self, root, stride=1, depth="none", part=None, semantics_dir=None,
                 intrinsics=None, sequence=None, depth_scale=DEPTH_SCALE,
                 require_poses=True):
        root, sfx = _split_part(root)
        part = part or sfx or "all"
        if part not in PARTS:
            raise ValueError(f"part {part!r} not in {PARTS}")
        self.root, self.part = str(root), part
        self.stride = max(1, int(stride))
        self.sequence = sequence or os.path.basename(os.path.normpath(self.root))
        if not SEQ_RE.match(self.sequence):
            raise RIO10FormatError(f"{self.root}: folder name {self.sequence!r} is not "
                                   f"seqXX_YY (pass sequence=...)")
        files = glob.glob(os.path.join(self.root, "frame-*.color.jpg"))
        if not files:
            raise RIO10FormatError(f"{self.root}: no frame-*.color.jpg files")
        files = sorted(files, key=_frame_number)
        nums = [_frame_number(p) for p in files]
        if len(set(nums)) != len(nums):
            raise RIO10FormatError(f"{self.root}: duplicate frame numbers")
        half = len(files) // 2
        sel = {"all": slice(None), "first_half": slice(0, half),
               "second_half": slice(half, None)}[part]
        self.all_numbers = nums
        self.numbers = nums[sel][:: self.stride]
        if not self.numbers:
            raise RIO10FormatError(f"{self.root}: part {part!r} with stride {self.stride} "
                                   f"selects no frame")
        self.require_poses = bool(require_poses)
        if self.require_poses:
            miss = [n for n in self.numbers if not os.path.exists(self._path(n, "pose.txt"))]
            if miss:
                raise RIO10FormatError(f"{self.root}: {len(miss)} frames have no pose file, "
                                       f"first frame-{miss[0]:06d}.pose.txt (only seqXX_01 and "
                                       f"seqXX_02 carry poses)")
        self.depth_mode, self.depth_scale = depth, float(depth_scale)
        if depth not in ("none", "rendered") and not os.path.isdir(str(depth)):
            raise ValueError(f"depth must be 'none', 'rendered' or a directory, got {depth!r}")
        self.semantics_dir = semantics_dir
        if semantics_dir is not None:
            miss = [n for n in self.numbers if not os.path.exists(
                os.path.join(semantics_dir, f"frame-{n:06d}.instances.png"))]
            if miss:
                raise RIO10FormatError(f"{semantics_dir}: {len(miss)} of {len(self.numbers)} "
                                       f"frames have no instance image, first "
                                       f"frame-{miss[0]:06d}.instances.png")
        fx, fy, cx, cy = self._intrinsics(intrinsics)
        self.K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
        h, w = read_color(self._path(self.numbers[0], "color.jpg")).shape[:2]
        self.H, self.W = int(h), int(w)
        if not (0 < cx < w and 0 < cy < h):
            raise RIO10FormatError(f"{self.root}: principal point ({cx}, {cy}) outside the "
                                   f"{w}x{h} colour image (portrait/landscape or intrinsics "
                                   f"of another resolution?)")

    def _path(self, n, suffix):
        return os.path.join(self.root, f"frame-{n:06d}.{suffix}")

    def _intrinsics(self, intrinsics):
        if intrinsics is not None and not isinstance(intrinsics, (str, os.PathLike)):
            return tuple(float(x) for x in intrinsics)
        if intrinsics is not None:
            return read_intrinsics_txt(str(intrinsics), self.sequence)
        cam = os.path.join(self.root, "camera.yaml")
        if os.path.exists(cam):
            return read_camera_yaml(cam)
        d, tried = os.path.abspath(self.root), []
        for _ in range(4):                               # intrinsics.txt up to 4 levels up
            d = os.path.dirname(d)
            for p in (os.path.join(d, "intrinsics.txt"), os.path.join(d, "data", "intrinsics.txt")):
                tried.append(p)
                if os.path.exists(p):
                    return read_intrinsics_txt(p, self.sequence)
        raise RIO10FormatError(f"{self.root}: no camera.yaml and no intrinsics.txt (tried "
                               f"{tried}); pass intrinsics=(fx, fy, cx, cy) or a path")

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, i):
        return self.frame(self.numbers[i])

    def pose(self, i):
        """c2w of item i without reading its images (NaN when absent, require_poses off)."""
        p = self._path(self.numbers[i], "pose.txt")
        if os.path.exists(p):
            return torch.from_numpy(read_pose(p).astype(np.float32))
        if self.require_poses:
            raise FileNotFoundError(p)
        return torch.full((4, 4), float("nan"))

    def frame(self, n):
        img = read_color(self._path(n, "color.jpg"))
        if img.shape[:2] != (self.H, self.W):
            raise RIO10FormatError(f"{self._path(n, 'color.jpg')}: {img.shape[1]}x"
                                   f"{img.shape[0]} != {self.W}x{self.H} of the first frame")
        pose_p = self._path(n, "pose.txt")
        if os.path.exists(pose_p):
            c2w = torch.from_numpy(read_pose(pose_p).astype(np.float32))
        elif self.require_poses:
            raise FileNotFoundError(pose_p)
        else:
            c2w = torch.full((4, 4), float("nan"))
        depth, src = None, "none"
        if self.depth_mode == "rendered":
            depth = torch.from_numpy(read_depth_png(self._path(n, "rendered.depth.png"),
                                                    self.depth_scale))
            src = "rendered_mesh"
        elif self.depth_mode != "none":
            depth, src = torch.from_numpy(self._external_depth(n)), "external"
        out = dict(image=torch.from_numpy(img), depth=depth, c2w=c2w, K=self.K.clone(),
                   index=n, name=f"{self.sequence}/frame-{n:06d}", depth_source=src)
        if self.semantics_dir is not None:
            out["instances"] = torch.from_numpy(read_instance_png(
                os.path.join(self.semantics_dir, f"frame-{n:06d}.instances.png")))
        return out

    def _external_depth(self, n):
        base = os.path.join(str(self.depth_mode), f"frame-{n:06d}.depth")
        if os.path.exists(base + ".npy"):
            d = np.load(base + ".npy").astype(np.float32)
            if d.ndim != 2:
                raise RIO10FormatError(f"{base}.npy: expected (H, W) depth, got {d.shape}")
            return np.where(np.isfinite(d) & (d > 0), d, 0.0).astype(np.float32)
        if os.path.exists(base + ".png"):
            return read_depth_png(base + ".png", self.depth_scale)
        raise FileNotFoundError(f"{base}.npy / .png (external depth)")


# ------------------------------------------------------------------------------ scene
class RIO10Scene:
    """The folders of one scene XX (frames, semantics, models per sequence) under a RIO10
    root, resolved from the layouts in _FRAME_DIRS / _SEM_DIRS / _MODEL_DIRS (V1)."""

    def __init__(self, root, scene, frames_dir=None, semantics_dir=None, models_dir=None):
        self.root = str(root)
        self.scene = f"{int(scene):02d}"
        self._over = dict(frames=frames_dir, semantics=semantics_dir, models=models_dir)

    def seq(self, k):
        return f"seq{self.scene}_{int(k):02d}"

    def _resolve(self, kind, k, marker, patterns, required=True):
        q = self.seq(k)
        over = self._over[kind]
        cands = [over.format(s=self.scene, q=q)] if over else \
            [os.path.join(self.root, p.format(s=self.scene, q=q)) for p in patterns]
        for c in cands:
            if glob.glob(os.path.join(c, marker)):
                return c
        if required:
            raise RIO10FormatError(f"scene {self.scene} {q}: no {kind} folder holding {marker}; "
                                   f"tried {cands}")
        return None

    def frames_dir(self, k):
        return self._resolve("frames", k, "frame-*.color.jpg", _FRAME_DIRS)

    def semantics_dir(self, k, required=True):
        return self._resolve("semantics", k, "instances.txt", _SEM_DIRS, required)

    def models_dir(self, k, required=True):
        return self._resolve("models", k, "labels.ply", _MODEL_DIRS, required)

    def sequence(self, k, stride=1, depth="none", part=None, instances=False, **kw):
        sem = self.semantics_dir(k) if instances else None
        return RIO10Dataset(self.frames_dir(k), stride=stride, depth=depth, part=part,
                            semantics_dir=sem, **kw)

    def instance_names(self, k):
        return read_instances_txt(os.path.join(self.semantics_dir(k), "instances.txt"))

    def labels(self, k, id_property=None):
        return read_labels_ply(os.path.join(self.models_dir(k), "labels.ply"), id_property)


# ------------------------------------------------------------------------------ 3RScan
def transform_from_list(vals, order="F", atol=1e-6):
    """4x4 matrix from 16 numbers (V9). The storage order is read from where the
    homogeneous row (0, 0, 0, 1) sits; when both orders fit (zero translation), `order`
    decides ("F": column-major, "C": row-major)."""
    a = np.asarray(vals, dtype=np.float64).reshape(-1)
    if a.size != 16:
        raise RIO10FormatError(f"transform needs 16 numbers, got {a.size}")
    row_major = np.allclose(a[12:16], [0, 0, 0, 1], atol=atol)       # C: last row at 12..15
    col_major = np.allclose(a[[3, 7, 11, 15]], [0, 0, 0, 1], atol=atol)
    if row_major and not col_major:
        return a.reshape(4, 4)
    if col_major and not row_major:
        return a.reshape(4, 4, order="F")
    if row_major and col_major:
        return a.reshape(4, 4, order=order)
    raise RIO10FormatError(f"transform {a.tolist()} has no (0, 0, 0, 1) homogeneous row")


def load_3rscan_changes(meta, reference, rescan, order="F"):
    """Changes of `rescan` against `reference` from 3RScan.json (path or parsed list):
    dict(removed: set, nonrigid: set, rigid: {ref instance: (rescan instance, 4x4)},
    scan_transform: 4x4 or None)."""
    if isinstance(meta, (str, os.PathLike)):
        with open(meta) as f:
            meta = json.load(f)
    ref = next((m for m in meta if m.get("reference") == reference), None)
    if ref is None:
        raise RIO10FormatError(f"3RScan.json: no reference scan {reference!r}")
    scan = next((s for s in ref.get("scans", []) if s.get("reference") == rescan), None)
    if scan is None:
        raise RIO10FormatError(f"3RScan.json: reference {reference!r} has no rescan {rescan!r} "
                               f"(has {[s.get('reference') for s in ref.get('scans', [])]})")
    rigid = {}
    for r in scan.get("rigid", []):
        iref = int(r["instance_reference"])
        if iref in rigid:
            raise RIO10FormatError(f"3RScan.json: instance {iref} listed twice as rigid")
        rigid[iref] = (int(r.get("instance_rescan", iref)),
                       transform_from_list(r["transform"], order))
    st = scan.get("transform")
    return dict(removed=_instance_set(scan.get("removed", []), "removed"),
                nonrigid=_instance_set(scan.get("nonrigid", []), "nonrigid"), rigid=rigid,
                scan_transform=None if st is None else transform_from_list(st, order))


def _instance_set(items, what):
    """Instance ids of a 3RScan list: plain ids or {"instance_reference"/"instance": id}."""
    out = set()
    for it in items:
        if isinstance(it, dict):
            key = next((k for k in ("instance_reference", "instance", "id") if k in it), None)
            if key is None:
                raise RIO10FormatError(f"3RScan.json: {what} entry {it} has no instance id")
            it = it[key]
        out.add(int(it))
    return out
