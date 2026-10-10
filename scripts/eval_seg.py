"""
eval_seg.py — segmentation + calibration evaluation of saved fields (plan K10b, Tab I).

For every scene: load the saved field (run_mapping.py --out: {"field", "pca", ...}), the 3DGS
checkpoint it was built on, the evaluation frames (poses, intrinsics) and their label maps;
map raw label ids through the frozen vocabulary (merges, ignore list); query the classes
present in the scene (plan §3) with texts projected into the field space WITHOUT
renormalising (StreamingPCA.project_text); render every row once per frame
(usplat.seg_eval.evaluate_view) and write all metrics per row as JSON.

Temperatures are fit on the VAL scenes only (--val-scenes: one beta per row, P2 parameters
for the adaptive row) or read from an earlier fit (--betas; not both). The test scenes
(--scenes) are then evaluated with them; the summary gives mean +- std over scenes with N.
Every evaluated row needs a fitted beta (a probs row may opt out with --raw-probs, which
evaluates its raw probabilities); the beta each row used is written per scene (betas_used).
The fit records its settings (beta_fit_settings: tau_cov, undef_frac, ...); a --betas file
fit with another --undef-frac is refused, another --tau-cov gives a warning.

The vocabulary of Tab I is configs/vocab_replica.json (K8); Replica frames come from
usplat.datasets "replica_semnerf" (Sequence_<k> folder as data), which carries the labels.
The ScanNet held-out set (scripts/scannet_tab1.sh) uses configs/vocab_scannet.json and the
"scannet" loader on the prepared scans (usplat.datasets.prepare_scannet), which carries them too.
Scenes file (YAML or JSON): a list of
    {name, field, ply, data, dataset (default --dataset), stride (default --stride),
     offset (default 0), label_pattern}
                          offset: first dataset frame (then every stride-th), for loaders that
                          take one: ScanNet evaluation frames 5, 25, 45, ... between the mapping
                          frames 0, 10, 20, ... of the same scan (configs/scannet_*.json)
                          label_pattern: "{root}/semantic_class/semantic_class_{idx}.png",
                          formatted with root = data and idx = dataset frame index
(frames that already carry labels under "semantic"/"labels"/"semantic_class" use those:
Replica Semantic-NeRF class ids, ScanNet NYU40 ids from label-filt via
scannetv2-labels.combined.tsv, see usplat.datasets.ScanNetDataset and configs/vocab_scannet.json).
Vocabulary file (YAML or JSON):
    templates: ["a photo of a {}."]           (or template: "...")
    ignore: [0]                               raw ids never evaluated (or ignore_ids)
    merges: {93: 12}                          raw id -> raw id, applied first
    classes: {12: {name: wall, prompt: wall}, 13: floor, ...}   prompt defaults to name
Raw ids that are neither classes nor ignored are ignored too and counted per scene.

Rows: ours_P0/P1/P2 (usplat.rows.field_rows), ours_scalar (the 1 - A_d spread only),
ours_P3_b<beta_g> (usplat.rows.FieldProbRow: the per-Gaussian predictive rendered as a
mixture, one row per beta_g of usplat.rows.P3_BETAS; --no-p3 leaves them out), and
the rows of --row-factory "module:function", called as fn(ckpt, field, device) -> [Row]
(e.g. baselines stored in the field checkpoint). --rows selects by name.
--with-2d adds the row "2d_features": the extractor run on the EVALUATION frame itself, the
same tokens -> upsample -> PCA path as run_mapping.py (the map's PCA), no 3D map. It covers
every pixel (alpha = 1), has no own uncertainty (MSP/entropy only) and gets its own beta.
It loads the extractor on --device, so gate the run for it (LSeg fp32 needs > 3.3 GB).

Backends: gsplat (GPU, the real renderer) or mock (CPU: pinhole projection of the Gaussian
centres + usplat.mock_rasterizer, isotropic footprint fx * mean scale / z; smoke tests only).

    cd uags && scripts/gpu_gate.sh 3000 python3 scripts/eval_seg.py \
        --vocab configs/vocab_replica.json --val-scenes val.yaml --scenes test.yaml \
        --tau-cov 0.5 --downscale 2 --out outputs/tab1/eval_seg.json
"""
from __future__ import annotations
import argparse
import importlib
import json
import math
import os
import sys
import time
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from usplat.vmf_field import DirectionalSemanticField                      # noqa: E402
from usplat.feature_pca import StreamingPCA                                # noqa: E402
from usplat.rows import field_rows, FieldRow, TensorRow, p3_rows           # noqa: E402
from usplat.seg_eval import (View, BetaFitter, SegAccumulator, evaluate_view,  # noqa: E402
                             freeze_rows, scene_stats, row_beta, IGNORE)

LABEL_KEYS = ("semantic", "labels", "semantic_class")


# --------------------------------------------------------------------------- inputs
def _load_structured(path):
    import yaml                                    # YAML is a superset of JSON
    with open(path) as f:
        return yaml.safe_load(f)


def load_vocab(path):
    """-> dict(ids, names, prompts, templates, ignore, merges) in file order."""
    v = _load_structured(path)
    templates = v.get("templates") or [v.get("template", "a photo of a {}.")]
    if isinstance(templates, str):
        templates = [templates]
    ids, names, prompts = [], [], []
    classes = v["classes"]
    items = classes.items() if isinstance(classes, dict) else \
        [(c["id"], c) for c in classes]
    for cid, c in items:
        name = c if isinstance(c, str) else c["name"]
        prompt = name if isinstance(c, str) else c.get("prompt", name)
        ids.append(int(cid)); names.append(str(name)); prompts.append(str(prompt))
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate class ids")
    ignore = {int(i) for i in (v.get("ignore") or v.get("ignore_ids") or [])}
    merges = {int(a): int(b) for a, b in (v.get("merges") or {}).items()}
    bad = ignore & set(ids)
    if bad:
        raise ValueError(f"{path}: ids {sorted(bad)} are both classes and ignored")
    return dict(ids=ids, names=names, prompts=prompts, templates=list(templates),
                ignore=ignore, merges=merges,
                templates_by_extractor=dict(v.get("templates_by_extractor") or {}))


def extractor_templates(vocab, extractor):
    """The prompt templates frozen for this extractor (vocab templates_by_extractor), else
    the vocabulary's default templates."""
    t = vocab.get("templates_by_extractor", {}).get(extractor)
    return list(t) if t else list(vocab["templates"])


def map_labels(raw, vocab):
    """raw ids (H, W) -> vocabulary index (ignore -> -1). Returns (labels, n_unknown)."""
    raw = torch.as_tensor(raw).long()
    if vocab["merges"]:
        src = torch.tensor(list(vocab["merges"].keys()))
        dst = torch.tensor(list(vocab["merges"].values()))
        hit = raw[..., None] == src
        raw = torch.where(hit.any(-1), dst[hit.float().argmax(-1)], raw)
    top = max(int(raw.max()), max(vocab["ids"]), max(vocab["ignore"], default=0)) + 1
    lut = torch.full((top,), -2, dtype=torch.long)          # -2 = unknown
    for i in vocab["ignore"]:
        lut[i] = IGNORE
    for k, cid in enumerate(vocab["ids"]):
        lut[cid] = k
    if int(raw.min()) < 0:
        raise ValueError("negative raw label id")
    out = lut[raw]
    unknown = out == -2
    return torch.where(unknown, torch.full_like(out, IGNORE), out), int(unknown.sum())


def text_embeddings(vocab, idx, args):
    """(len(idx), D) unit text embeddings of the chosen vocabulary entries (cached), from the
    SAME extractor that produced the map's image features (args.extractor)."""
    tpl = extractor_templates(vocab, args.extractor)
    tag = (f"{args.clip_model}|{args.clip_pretrained}" if args.extractor == "clearclip"
           else args.extractor)
    key = lambda p: f"{tag}|{'|'.join(tpl)}|{p}"
    cache = {}
    if args.text_cache and os.path.exists(args.text_cache):
        cache = torch.load(args.text_cache, map_location="cpu")
    prompts = [vocab["prompts"][i] for i in idx]
    missing = sorted({p for p in prompts if key(p) not in cache})
    if missing:
        from usplat.extractors import get_extractor
        kw = (dict(model_name=args.clip_model, pretrained=args.clip_pretrained)
              if args.extractor == "clearclip" else {})
        clip = get_extractor(args.extractor, device=args.clip_device, **kw)
        emb = clip.text_embedding(missing, templates=tuple(tpl)).cpu()
        for p, e in zip(missing, emb):
            cache[key(p)] = e
        if args.text_cache:
            os.makedirs(os.path.dirname(os.path.abspath(args.text_cache)), exist_ok=True)
            torch.save(cache, args.text_cache)
    return torch.stack([cache[key(p)].float() for p in prompts])


def _nearest(lab, h, w):
    H, W = lab.shape
    if (H, W) == (h, w):
        return lab
    r = ((torch.arange(h) + 0.5) * H / h).long().clamp(max=H - 1)
    c = ((torch.arange(w) + 0.5) * W / w).long().clamp(max=W - 1)
    return lab[r][:, c]


def _read_label(path):
    import cv2
    lab = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise FileNotFoundError(path)
    if lab.ndim == 3:
        raise ValueError(f"{path}: expected a single-channel label map")
    return torch.from_numpy(lab.astype("int64"))


def load_frames(sc, vocab, args):
    """Poses, intrinsics (scaled by --downscale) and vocabulary-index labels per frame."""
    from usplat.datasets import make_dataset
    ds = make_dataset(sc.get("dataset", args.dataset), sc["data"],
                      stride=int(sc.get("stride", args.stride)), offset=int(sc.get("offset", 0)))
    frames, n_unknown, skipped = [], 0, 0
    n = len(ds) if args.max_views is None else min(len(ds), args.max_views)
    for i in range(n):
        fr = ds[i]
        idx = int(fr.get("index", getattr(ds, "offset", 0) + i * ds.stride))
        c2w = fr["c2w"]
        if not torch.isfinite(c2w).all():
            skipped += 1
            continue
        raw = next((fr[k] for k in LABEL_KEYS if fr.get(k) is not None), None)
        if raw is None:
            if not sc.get("label_pattern"):
                raise ValueError(f"scene {sc['name']}: frame {idx} carries no labels; set "
                                 f"label_pattern")
            raw = _read_label(sc["label_pattern"].format(root=sc["data"], idx=idx))
        H, W = int(fr["image"].shape[0]), int(fr["image"].shape[1])
        f = max(1, int(args.downscale))
        h, w = H // f, W // f
        K = fr["K"].clone().float()
        K[0] *= w / W
        K[1] *= h / H
        lab, unk = map_labels(_nearest(torch.as_tensor(raw), h, w), vocab)
        n_unknown += unk
        frames.append(dict(idx=idx, viewmat=torch.linalg.inv(c2w.double()).float(), K=K,
                           w=w, h=h, labels=lab,
                           image=fr["image"] if getattr(args, "with_2d", False) else None))
    return frames, n_unknown, skipped


# ------------------------------------------------------------------------ rasterizers
class PinholeMockRasterizer:
    """CPU stand-in for GSplatFeatureRasterizer: Gaussian centres projected with K, an
    isotropic screen footprint fx * mean(scale) / z, usplat.mock_rasterizer compositing."""

    def __init__(self, splats, viewmat, K, width, height, near=0.01):
        from usplat.mock_rasterizer import MockFeatureRasterizer
        R, t = viewmat[:3, :3], viewmat[:3, 3]
        cam = splats["means"].float() @ R.t() + t
        z = cam[:, 2]
        front = z > near
        zc = z.clamp_min(near)
        uv = cam[:, :2] / zc[:, None] * torch.stack([K[0, 0], K[1, 1]]) + K[:2, 2]
        sig = K[0, 0] * splats["scales"].float().mean(-1) / zc
        op = torch.where(front, splats["opacities"].float().reshape(-1),
                         torch.zeros_like(zc))
        ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        pix = torch.stack([xs, ys], -1).reshape(-1, 2).float() + 0.5   # pixel centres
        self._r = MockFeatureRasterizer(uv, sig.clamp_min(1e-3), op, z, pix)
        self.num_gaussians = self._r.N

    def __call__(self, buf):
        return self._r(buf)


def make_rasterizer(backend, splats, fr, args):
    if backend == "mock":
        return PinholeMockRasterizer(splats, fr["viewmat"], fr["K"], fr["w"], fr["h"])
    from usplat.gsplat_adapter import GSplatFeatureRasterizer
    dev = splats["means"].device
    return GSplatFeatureRasterizer.from_splats(splats, fr["viewmat"].to(dev), fr["K"].to(dev),
                                               fr["w"], fr["h"],
                                               channel_chunk=args.channel_chunk)


# ------------------------------------------------------------------------------ scenes
def load_scene(sc, vocab, args, device):
    from usplat.gs_io import load_gaussians
    ckpt = torch.load(sc["field"], map_location="cpu", weights_only=False)
    ext = ckpt.get("extractor") or (ckpt.get("config") or {}).get("extractor")
    if ext is not None and ext != args.extractor:
        raise ValueError(f"scene {sc['name']}: map built with extractor {ext!r} but "
                         f"--extractor {args.extractor!r}")
    # The field and the baselines stay on the CPU: the rows are frozen there and only the
    # frozen buffers go to the GPU (a 1M-Gaussian field with its change buckets and every
    # row's buffer would not fit beside the extractor of the 2D row).
    field = DirectionalSemanticField.from_state(ckpt["field"], device="cpu")
    pca = StreamingPCA.from_state(ckpt["pca"], device=device)
    frames, n_unknown, skipped = load_frames(sc, vocab, args)
    if not frames:
        raise ValueError(f"scene {sc['name']}: no frames")
    want2d = getattr(args, "with_2d", False) and (
        not args.rows or ROW_2D in [s.strip() for s in args.rows.split(",")])
    if want2d:                           # tokens first, then the extractor leaves the GPU
        for f in frames:
            f["tok2d"] = _extractor(args).extract_tokens(f["image"]).cpu()
            f["image"] = None
        _EXTRACTOR.clear()                   # its blocks stay in this process's cache
    splats = load_gaussians(sc["ply"], device=device)
    if splats["means"].shape[0] != field.N:
        raise ValueError(f"scene {sc['name']}: {splats['means'].shape[0]} Gaussians in "
                         f"{sc['ply']} but the field has {field.N}")
    if args.queries == "present":
        present = torch.unique(torch.cat([f["labels"].reshape(-1) for f in frames]))
        q = [int(i) for i in present.tolist() if i >= 0]
    else:
        q = list(range(len(vocab["ids"])))
    local = torch.full((len(vocab["ids"]) + 1,), IGNORE, dtype=torch.long)
    local[torch.tensor(q, dtype=torch.long)] = torch.arange(len(q))
    for f in frames:                     # vocabulary index -> scene query index
        lab = f["labels"]
        f["gt"] = torch.where(lab >= 0, local[lab.clamp_min(0)], lab)
    text = pca.project_text(text_embeddings(vocab, q, args).to(device))    # (K, d), F7
    rows = candidate_rows(field, ckpt, "cpu", args, class_ids=[vocab["ids"][i] for i in q],
                          text=text)
    rows = frozen_on(rows, device)
    rows2d = []
    if want2d:
        rows2d = [TensorRow(ROW_2D, "feature", buf=torch.zeros(1, field.d, device=device))]
    code = (ckpt.get("code") or {}).get("mapping_code_sha256")
    return dict(name=sc["name"], field=field, splats=splats, frames=frames, text=text, pca=pca,
                field_identity=field_identity(ckpt), field_code=code,
                aux_identities=sorted(set(ckpt.get("_aux_identities") or [])),
                rows2d=rows2d,
                class_ids=[vocab["ids"][i] for i in q], class_names=[vocab["names"][i] for i in q],
                rows=rows, n_unknown=n_unknown, n_skipped=skipped)


IDENTITY_ARGS = ("extractor", "feat_dim", "pca_frames", "downscale", "stride", "kf_trans",
                 "kf_rot", "max_keyframes", "seed", "baselines", "channel_chunk")


def field_identity(ckpt):
    """sha256 of what decides a field besides its scene: mapping code hash, extractor options,
    FieldConfig and the protocol arguments of run_mapping.py. The validation fields (which
    set the temperatures) and the test fields must share it."""
    import hashlib
    cfg = ckpt.get("config") or {}
    key = dict(code=(ckpt.get("code") or {}).get("mapping_code_sha256"),
               extractor_options=ckpt.get("extractor_options"),
               field_config=ckpt.get("field_config"),
               args={k: cfg.get(k) for k in IDENTITY_ARGS})
    return hashlib.sha256(json.dumps(_plain_json(key), sort_keys=True).encode()).hexdigest()


def check_field_identity(ident, aux, settings, scene, args):
    """In --betas mode, refuse a scene built differently from the validation scenes the betas
    were fit on: its field identity must be one of theirs, and every auxiliary file a row
    factory read (e.g. a distilled baseline) must match one read on validation (validation may
    hold more, e.g. a hyperparameter grid). A field without a recorded code version counts as
    different. Called right after the scene is loaded, before any evaluation."""
    if not args.betas or getattr(args, "allow_field_mismatch", False):
        return
    val = (settings or {}).get("field_identities")
    if not val:
        raise ValueError(f"{args.betas}: records no field identities; refit the betas with this "
                         f"eval_seg.py (or pass --allow-field-mismatch)")
    if ident not in val:
        raise ValueError(f"scene {scene}: field identity {ident[:12]} is not one of the "
                         f"validation fields' {[v[:12] for v in val]} (different mapping code, "
                         f"extractor options, FieldConfig or protocol); rebuild the fields")
    extra = sorted(set(aux) - set((settings or {}).get("aux_identities") or []))
    if extra:
        raise ValueError(f"scene {scene}: auxiliary files {[e[:12] for e in extra]} were not "
                         f"built like any read on validation (another code or config)")


@torch.no_grad()
def frozen_on(rows, device):
    """Freeze the rows (on their own device), move the snapshots to `device`, and share
    identical tensors (e.g. mu* of ours P0/P2/scalar, the averaging feature of three
    baselines, u_epi / u_ale of every ours row) so each is stored once."""
    kept = []

    def same(a, b):                          # NaN-aware: u_ale is NaN where undefined
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.is_floating_point():
            na, nb = torch.isnan(a), torch.isnan(b)
            return torch.equal(na, nb) and torch.equal(a[~na], b[~nb])
        return torch.equal(a, b)

    def share(x):
        for y_cpu, y_dev in kept:
            if same(y_cpu, x):
                return y_dev
        y = x.to(device)
        kept.append((x, y))
        return y
    out = []
    for r in freeze_rows(rows):
        dec = r.decoder.to(device) if hasattr(r.decoder, "to") else r.decoder
        out.append(TensorRow(r.name, r.kind, adaptive=r.adaptive,
                             p2_channels=tuple(r.p2_channels), decoder=dec, buf=share(r.buf),
                             unc={k: share(v) for k, v in r.unc.items()}))
    return out


def candidate_rows(field, ckpt, device, args, class_ids=None, text=None):
    """The rows of one scene. --v2-rows adds the Table I v2 candidates (usplat.rows.v2_rows:
    ours_P2p1, ours_P1_c<c0>, and with the scene's text ours_P4_a<alpha0>_s<lam>,
    ours_P4_inf, ours_P4_rbar), read-outs of the same saved field; default off: the v1 rows only."""
    rows = field_rows(field, "ours")
    rows.append(FieldRow("ours_scalar", "feature", field=field, mode="P0", channels=("spread",)))
    if text is not None and not getattr(args, "no_p3", False):
        rows += p3_rows(field, text.to(field.device))  # P3: mixture of per-Gaussian predictives
    if getattr(args, "v2_rows", False):
        from usplat.rows import v2_rows
        rows += v2_rows(field, None if text is None else text.to(field.device))
    import inspect
    for spec in [s.strip() for s in (args.row_factory or "").split(",") if s.strip()]:
        mod, fn = spec.split(":")
        factory = getattr(importlib.import_module(mod), fn)
        if "class_ids" in inspect.signature(factory).parameters:   # CSS columns = queries
            rows += list(factory(ckpt, field, device, class_ids=class_ids))
        else:
            rows += list(factory(ckpt, field, device))
    names = [r.name for r in rows]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate row names {names}")
    if args.rows:
        want = [s.strip() for s in args.rows.split(",") if s.strip()]
        if getattr(args, "with_2d", False):
            want = [w for w in want if w != ROW_2D]          # the 2D row is built apart
        miss = [w for w in want if w not in names]
        if miss:
            raise ValueError(f"unknown rows {miss}; available {names}")
        rows = [r for r in rows if r.name in want]
    return rows


def scene_views(S, args):
    for f in S["frames"]:
        yield View(make_rasterizer(args.backend, S["splats"], f, args), f["gt"], S["text"],
                   S["class_ids"], name=str(f["idx"]))


ROW_2D = "2d_features"
_EXTRACTOR = {}


def _extractor(args):
    if args.extractor not in _EXTRACTOR:
        from usplat.extractors import get_extractor
        kw = (dict(model_name=args.clip_model, pretrained=args.clip_pretrained)
              if args.extractor == "clearclip" else {})
        _EXTRACTOR[args.extractor] = get_extractor(args.extractor, device=args.device, **kw)
    return _EXTRACTOR[args.extractor]


class PixelSource:
    """'Rasterizer' of the 2D row: returns the frame's own features z (P, d) and alpha = 1,
    whatever the (1, d + 1) dummy buffer holds (render_row appends the ones channel)."""

    def __init__(self, z):
        self.z = z

    def __call__(self, buf):
        if buf.shape[1] != self.z.shape[1] + 1:
            raise ValueError(f"2D row: buffer has {buf.shape[1]} channels, expected "
                             f"{self.z.shape[1] + 1} (features + ones, no uncertainties)")
        ones = torch.ones(self.z.shape[0], 1, device=self.z.device, dtype=self.z.dtype)
        return torch.cat([self.z, ones], 1)


@torch.no_grad()
def frame_features_2d(S, f, args):
    """(h*w, d) features of the evaluation frame: the run_mapping.py path (full-resolution
    tokens -> bilinear upsample to the label grid -> unit -> the map's PCA -> unit)."""
    from usplat.clip_dense import upsample_tokens
    tok = f["tok2d"] if f.get("tok2d") is not None else _extractor(args).extract_tokens(f["image"])
    z = S["pca"].project(upsample_tokens(tok.to(S["pca"].basis.device), (f["h"], f["w"])))
    return z.reshape(-1, z.shape[-1]).float()


def scene_views_2d(S, args):
    for f in S["frames"]:
        yield View(PixelSource(frame_features_2d(S, f, args)), f["gt"], S["text"],
                   S["class_ids"], name=str(f["idx"]))


def fit_on(scenes, vocab, args, device):
    fitter = BetaFitter(args.tau_cov, max_pixels_per_view=args.fit_pixels, seed=args.seed,
                        undef_frac=args.undef_frac)
    ids, aux = set(), set()
    for sc in scenes:
        S = load_scene(sc, vocab, args, device)
        ids.add(S["field_identity"])
        aux |= set(S["aux_identities"])
        fitter.add(S["rows"], scene_views(S, args))
        if S["rows2d"]:
            fitter.add(S["rows2d"], scene_views_2d(S, args))
        del S
    if len(ids) > 1:
        warnings.warn(f"validation fields were built differently ({len(ids)} identities): "
                      f"rebuild them with one code version")
    betas = fitter.fit()
    settings = dict(fitter.settings, field_identities=sorted(ids), aux_identities=sorted(aux))
    return betas, fitter.info, settings


def _raw_probs(args):
    return {s.strip() for s in (args.raw_probs or "").split(",") if s.strip()}


def betas_used(rows, betas, raw_probs):
    """{row: the beta it is evaluated with}; raises before any render if one is missing."""
    used, bad = {}, {}
    for r in rows:
        try:
            used[r.name] = row_beta(r, betas, raw_probs)
        except ValueError as e:
            bad[r.name] = str(e)
    if bad:
        raise ValueError(f"rows {sorted(bad)} have no usable fitted beta (fit on --val-scenes "
                         f"with these rows, or --raw-probs for probs rows); betas has "
                         f"{sorted(betas)}; reasons: {bad}")
    return used


def evaluate_scene(sc, vocab, betas, args, device, settings=None):
    S = load_scene(sc, vocab, args, device)
    check_field_identity(S["field_identity"], S["aux_identities"], settings, sc["name"], args)
    raw = _raw_probs(args)
    used = betas_used(S["rows"] + S["rows2d"], betas, raw)
    rows = freeze_rows(S["rows"])
    rows2d = freeze_rows(S["rows2d"])
    acc = SegAccumulator(class_names=S["class_names"])
    t0 = time.perf_counter()
    views2d = scene_views_2d(S, args) if rows2d else None
    for v in scene_views(S, args):
        out = evaluate_view(rows, v.rasterizer, v.text, v.gt, betas, args.tau_cov,
                            undef_frac=args.undef_frac, raw_probs=raw) if rows else {}
        if views2d is not None:
            v2 = next(views2d)
            if v2.name != v.name:
                raise RuntimeError(f"2D view {v2.name} != 3D view {v.name}")
            out.update(evaluate_view(rows2d, v2.rasterizer, v2.text, v2.gt, betas,
                                     args.tau_cov, undef_frac=args.undef_frac, raw_probs=raw))
        acc.add(out)
    return dict(n_views=acc.n_views, classes=S["class_names"], class_ids=S["class_ids"],
                field_identity=S["field_identity"], field_code=S["field_code"],
                aux_identities=S["aux_identities"],
                betas_used=used, raw_probs=sorted(raw & set(used)),
                n_unknown_label_pixels=S["n_unknown"], n_skipped_frames=S["n_skipped"],
                eval_s=time.perf_counter() - t0, rows=acc.metrics(args.ece_bins, args.ause_bins))


SUMMARY_KEYS = ("miou", "acc", "ece", "nll", "brier", "aurc", "coverage", "n_tied",
                "n_uncovered")


def summarize_scenes(results):
    """{row: {variant: {metric: mean/std/n over scenes}}} (ranking: AUSE per signal)."""
    out = {}
    rows = sorted({r for res in results.values() for r in res["rows"]})
    for r in rows:
        out[r] = {}
        for var in ("all", "covered"):
            per = [res["rows"][r][var] for res in results.values() if r in res["rows"]]
            m = {k: scene_stats([p.get(k, float("nan")) for p in per]) for k in SUMMARY_KEYS}
            sigs = sorted({s for p in per for s in p.get("ranking", {})})
            m["ause"] = {s: scene_stats([p["ranking"][s]["ause"] if s in p.get("ranking", {})
                                         else float("nan") for p in per]) for s in sigs}
            out[r][var] = m
    return out


def _record_body(path):
    d = json.load(open(path))
    if isinstance(d, dict) and isinstance(d.get("metrics"), dict):
        d = d["metrics"]                            # usplat.runlog record
    return d


def load_betas(path):
    d = _record_body(path)
    if isinstance(d, dict) and isinstance(d.get("betas"), dict):
        d = d["betas"]
    return {k: (v if isinstance(v, dict) else float(v)) for k, v in d.items()}


def load_fit_settings(path):
    """The BetaFitter settings stored with the betas (None for a bare betas file)."""
    d = _record_body(path)
    s = d.get("beta_fit_settings") if isinstance(d, dict) else None
    return s if isinstance(s, dict) else None


def check_fit_settings(settings, args, path=""):
    """Refuse betas fit with another undef_frac (it changes u_ale, hence P2); warn on
    another tau_cov (it changes only the fit pixel set)."""
    if settings is None:
        warnings.warn(f"{path}: no beta_fit_settings recorded; cannot check undef_frac/tau_cov")
        return
    uf = settings.get("undef_frac")
    if uf is None or not math.isclose(float(uf), args.undef_frac, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{path}: betas were fit with undef_frac {uf}, evaluation uses "
                         f"--undef-frac {args.undef_frac}")
    tc = settings.get("tau_cov")
    if tc is not None and not math.isclose(float(tc), args.tau_cov, rel_tol=0, abs_tol=1e-12):
        warnings.warn(f"{path}: betas were fit with tau_cov {tc}, evaluation uses "
                      f"--tau-cov {args.tau_cov}")


def _plain_json(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, dict):
        return {str(k): _plain_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain_json(v) for v in obj]
    if torch.is_tensor(obj):
        return _plain_json(obj.tolist())
    return obj


# -------------------------------------------------------------------------------- main
def run(args, rec=None):
    device = args.device
    if device.startswith("cuda"):
        from usplat import gpu_reserve                      # fraction from scripts/gpu_gate.sh
        held = gpu_reserve.apply(device)
        if held:
            print(f"reserved {held} MiB of GPU memory")
    vocab = load_vocab(args.vocab)
    out = dict(vocab=dict(path=os.path.abspath(args.vocab), n_classes=len(vocab["ids"]),
                          extractor=args.extractor,
                          templates=extractor_templates(vocab, args.extractor),
                          ignore=sorted(vocab["ignore"])))
    if args.val_scenes and args.betas:
        raise ValueError("give --val-scenes (fit) or --betas (read a fit), not both")
    if args.val_scenes:
        betas, info, settings = fit_on(_load_structured(args.val_scenes), vocab, args, device)
        out.update(betas=betas, beta_fit=info, beta_fit_settings=settings)
        print("fitted betas on val:", json.dumps(betas))
    elif args.betas:
        settings = load_fit_settings(args.betas)
        check_fit_settings(settings, args, args.betas)
        out.update(betas=load_betas(args.betas), beta_fit_settings=settings,
                   betas_path=os.path.abspath(args.betas))
    else:
        out["betas"] = {}
    if args.scenes:
        res = {}
        partial = f"{args.out}.partial.json"
        for sc in _load_structured(args.scenes):
            res[sc["name"]] = evaluate_scene(sc, vocab, out["betas"], args, device,
                                             settings=out.get("beta_fit_settings"))
            with open(partial, "w") as f:           # a crash later keeps the finished scenes
                json.dump(_plain_json(dict(out, scenes=res)), f, indent=1)
            for r, m in res[sc["name"]]["rows"].items():
                a = m["all"]
                print(f"{sc['name']:>10} {r:>14}: mIoU {a.get('miou', float('nan')):.4f} "
                      f"acc {a.get('acc', float('nan')):.4f} ECE {a.get('ece', float('nan')):.4f} "
                      f"NLL {a.get('nll', float('nan')):.4f} coverage {a.get('coverage', 0):.3f}")
        out["scenes"] = res
        out["summary"] = summarize_scenes(res)
        out["field_identities"] = sorted({r["field_identity"] for r in res.values()})
    if rec is not None:
        for k, v in out.items():
            rec.add(k, v)
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vocab", required=True, help="frozen vocabulary file (YAML/JSON)")
    ap.add_argument("--scenes", default="", help="scenes to evaluate (YAML/JSON list)")
    ap.add_argument("--val-scenes", default="", help="VAL scenes: fit one beta per row here")
    ap.add_argument("--betas", default="", help="betas from an earlier fit (JSON)")
    ap.add_argument("--tau-cov", type=float, default=0.5,
                    help="coverage floor on the rendered alpha [A: set on Replica val]")
    ap.add_argument("--undef-frac", type=float, default=0.5,
                    help="a pixel's rendered u is undefined when more than this share of "
                         "its alpha comes from Gaussians with undefined u")
    ap.add_argument("--dataset", default="replica_semnerf")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--downscale", type=int, default=2, help="render/label resolution divisor")
    ap.add_argument("--max-views", type=int, default=None, help="first n frames (smoke runs)")
    ap.add_argument("--queries", choices=["present", "all"], default="present")
    ap.add_argument("--rows", default="", help="comma-separated row names (default: all)")
    ap.add_argument("--row-factory", default="",
                    help="module:function[,module:function...] -> extra rows")
    ap.add_argument("--allow-field-mismatch", action="store_true",
                    help="with --betas: evaluate fields built differently from the val fields")
    ap.add_argument("--no-p3", action="store_true", help="leave out the P3 mixture rows")
    ap.add_argument("--v2-rows", action="store_true",
                    help="add the Table I v2 candidates (usplat.rows.v2_rows: ours_P2p1, the P1 "
                         "c0 grid, the P4 vote grid; configs/tab1_v2_prereg.json)")
    ap.add_argument("--with-2d", action="store_true",
                    help="add the 2D row: the extractor on each evaluation frame (no map)")
    ap.add_argument("--raw-probs", default="",
                    help="comma-separated probs rows evaluated with their raw probabilities "
                         "(beta 1) when they have no fitted beta")
    ap.add_argument("--backend", choices=["gsplat", "mock"], default="gsplat")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--channel-chunk", type=int, default=32)
    ap.add_argument("--extractor", default="lseg",
                    help="usplat.extractors name; must match the map's extractor")
    ap.add_argument("--clip-model", default="ViT-B-16")
    ap.add_argument("--clip-pretrained", default="laion2b_s34b_b88k")
    ap.add_argument("--clip-device", default="cpu", help="device for the text encoder")
    ap.add_argument("--text-cache", default="", help=".pt cache of text embeddings")
    ap.add_argument("--fit-pixels", type=int, default=2000,
                    help="max covered labelled pixels per val view for the beta fit")
    ap.add_argument("--ece-bins", type=int, default=15)
    ap.add_argument("--ause-bins", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plain-json", action="store_true",
                    help="write plain JSON instead of a usplat.runlog record")
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not (args.scenes or args.val_scenes):
        raise SystemExit("nothing to do: give --scenes and/or --val-scenes")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    try:
        from usplat.runlog import RunRecord
    except ImportError:
        RunRecord = None
    if RunRecord is not None and not args.plain_json:
        with RunRecord(args.out, config=vars(args), seed=args.seed, name="eval_seg",
                       argv=[sys.argv[0]] + list(argv if argv is not None else sys.argv[1:])) as rec:
            out = run(args, rec)
    else:
        if RunRecord is None and not args.plain_json:
            warnings.warn("usplat.runlog not found: writing plain JSON")
        torch.manual_seed(args.seed)
        t0 = time.time()
        out = run(args)
        out["provenance"] = dict(argv=sys.argv if argv is None else argv,
                                 start=t0, wall_s=time.time() - t0, config=vars(args))
        with open(args.out, "w") as f:
            json.dump(_plain_json(out), f, indent=1)
    print(f"wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
