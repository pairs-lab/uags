"""
time_field.py — run time and memory of the mapping pass, for the paragraph "Run time and
memory" of papers/main.tex: one validation scene, numbers per keyframe.

The main field (--field, a scripts/run_mapping.py output) fixes the scene and the settings: its
config gives data, dataset, stride, downscale, extractor and keyframe rule, and its PCA basis and
centre are reused (no PCA fit, as run_mapping.py --pca-from). The pass repeats run_mapping.py:
the same keyframes, all extractor tokens first (then the extractor leaves the GPU), then one
update per keyframe of a fresh field (change test recording only, as in the mapping run).
Parts of a keyframe, each one closed by torch.cuda.synchronize at both ends:
    extract          extractor tokens of the full-resolution frame (LSeg)
    project          tokens upsampled to the accumulation grid and PCA-projected
    render_backward  the one render + one backward of the gradient trick
                     (usplat.bki_accumulate.bki_grad_accumulate_groups, called by update_view)
    stats            statistics, change test and re-admission (DirectionalSemanticField._ingest)
    other            the rest of the update (pose, rasterizer setup, view sums)
    total            extract + project + render_backward + stats + other
A second pass attaches the mapping run's baselines and times what they add: "hooks"
(averaging/NormLift and the view count read our sums: no render) and "css" (CSS.update_view: its
own render + backward for the class votes). The first --warmup keyframes of each pass are not
counted; mean and std (population) are over the rest. Reading frames from disk and moving the
tokens between CPU and GPU are not in the total ("to_cpu" is reported apart). Peak GPU memory
is the torch allocator peak of each phase (extraction; field update).
Also recorded: the 3D-GS fit time (init + optimisation) from the fit meta scripts/fit_gs.py saves
in the map, else from the JSON line of its fit log; the field state in bytes per Gaussian (the
saved state_dict tensors / N, with and without the change-test buckets, and every tensor the
field holds in memory, also the all-zero ones the saved state leaves out); the GPU and CPU
(nvidia-smi, lscpu) and the GPU's other compute processes; every per-keyframe time. A LaTeX
sentence with the numbers is printed and saved in the JSON.

    scripts/gpu_gate.sh 9000 python3 scripts/time_field.py \
        --field /home/thanhnc/gauss_ws/data/uags_fields/office_0_lseg.pt \
        --fit-maps "/home/thanhnc/gauss_ws/data/uags_maps/*_seq1.pt" \
        --out /home/thanhnc/gauss_ws/data/uags_results/timing/time_office_0.json
    python3 scripts/time_field.py --latex-from <replica.json> [<rio10.json>]

On the CPU (GPU busy): --device cpu --backend mock renders with eval_seg.py's pinhole mock
rasterizer (smoke tests only; tests/test_time_field.py runs it on a tiny synthetic scene).
"""
from __future__ import annotations
import argparse
import contextlib
import functools
import gc
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time
from importlib import metadata
from unittest import mock

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import run_mapping as RM                                                   # noqa: E402
from usplat import vmf_field                                               # noqa: E402
from usplat import baselines as B                                          # noqa: E402
from usplat.clip_dense import upsample_tokens                              # noqa: E402
from usplat.datasets import make_dataset, downscale                        # noqa: E402
from usplat.feature_pca import StreamingPCA                                # noqa: E402
from usplat.gs_io import load_gaussians                                    # noqa: E402
from usplat.vmf_field import DirectionalSemanticField, FieldConfig        # noqa: E402

# state_dict tensors only the change test needs; the buckets are its recent and older sums
CHANGE_BUCKETS = ("Rf", "nf", "Rs", "ns")
CHANGE_TEST = CHANGE_BUCKETS + ("fB2", "fC", "fCq", "fBq2", "T2", "TQ", "c_ring", "c_last",
                                "c_max", "c_hist", "thr_by_hist")
POSTERIOR = ("raw_R", "raw_W1", "raw_W2", "raw_Q", "raw_Wr")
EXTRACTOR_OPTIONS = ("token_stride", "flip", "scales", "half", "base_size", "crop_size")
DATASET_NAMES = {"replica_semnerf": "Replica", "replica": "Replica", "scannet": "ScanNet",
                 "7scenes": "7-Scenes"}
EXTRACTOR_NAMES = {"lseg": "LSeg", "clearclip": "ClearCLIP"}
MISSING = "\\ph{}"                       # the placeholder of papers/main.tex (red XX.X)


# ------------------------------------------------------------------------------- timing
class Clock:
    """Named parts of each keyframe, in seconds. On a CUDA device every part is closed by
    torch.cuda.synchronize at both ends, so the GPU work of a part is counted in that part."""

    def __init__(self, device):
        self.device = str(device)
        self.cuda = self.device.startswith("cuda")
        self.frames, self.calls = [], {}

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize(torch.device(self.device))

    def frame(self):
        self.frames.append({})

    @contextlib.contextmanager
    def part(self, name):
        self.sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            f = self.frames[-1]
            f[name] = f.get(name, 0.0) + time.perf_counter() - t0
            self.calls[name] = self.calls.get(name, 0) + 1

    def wrap(self, fn, name):
        """fn, with every call timed as the part `name` of the current keyframe."""
        @functools.wraps(fn)
        def timed(*a, **kw):
            with self.part(name):
                return fn(*a, **kw)
        return timed


def mean_std(values, scale=1.0):
    """{mean, std (population), n} of the values times scale (None when there are none)."""
    v = [scale * float(x) for x in values]
    if not v:
        return dict(mean=None, std=None, n=0)
    return dict(mean=statistics.fmean(v), std=statistics.pstdev(v), n=len(v))


def summarize(frames, warmup):
    """{part: mean_std in ms} over the keyframes after the warm-up."""
    kept = frames[warmup:]
    parts = sorted({k for f in kept for k in f})
    return {k: mean_std([f[k] for f in kept if k in f], 1000.0) for k in parts}


# ---------------------------------------------------------------------------- keyframes
def select_keyframes(ds, kf_trans, kf_rot, max_keyframes=0):
    """run_mapping.py's mapping frames (`select` in run_gsplat): an evenly spaced subset with
    --max-keyframes, finite poses only, and no near-duplicate of the last kept frame (pose gap
    below kf_trans AND kf_rot). Copied, not imported: it is a closure there."""
    last, keep = None, []
    order = range(len(ds))
    if max_keyframes and max_keyframes < len(ds):
        order = sorted(set(torch.linspace(0, len(ds) - 1, max_keyframes).round().long().tolist()))
    for i in order:
        c2w = ds[i]["c2w"]
        if not torch.isfinite(c2w).all():
            continue
        if last is not None:
            dt, dr = RM._pose_gap(c2w, last)
            if dt < kf_trans and dr < kf_rot:
                continue
        last = c2w
        keep.append(i)
    return keep


def evenly(idx, n):
    """n evenly spaced entries of idx (all of them for n <= 0 or n >= len(idx))."""
    if n <= 0 or n >= len(idx):
        return list(idx)
    return [idx[p] for p in torch.linspace(0, len(idx) - 1, n).round().long().tolist()]


def scene_name(cfg):
    """office_0 for .../office_0/Sequence_1, else the name of the data folder."""
    p = os.path.normpath(cfg["data"])
    base = os.path.basename(p)
    return os.path.basename(os.path.dirname(p)) if base.lower().startswith("sequence") else base


# ------------------------------------------------------------------------------- set-up
def load_field_file(path):
    """The main field checkpoint, memory-mapped (only the small parts are read)."""
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def field_config(ckpt):
    """The mapping run's FieldConfig with change_thr = inf: the mapping pass records the change
    scores and resets nothing (run_mapping.py)."""
    s = ckpt["field"]
    cfg = dict(s["config"])
    if s.get("calib_names", 1) < 2 and cfg.get("calib_mode") == "pooled":
        cfg["calib_mode"] = "history"         # the old name, as DirectionalSemanticField.from_state
    cfg = FieldConfig(**cfg)
    cfg.change_thr = float("inf")
    return cfg


def rasterizer_factory(backend, channel_chunk=32):
    """(splats, viewmat, K, w, h) -> rasterizer: run_mapping.py's GSplatFeatureRasterizer, or
    eval_seg.py's CPU pinhole mock (smoke tests only)."""
    if backend == "mock":
        import eval_seg
        return lambda splats, viewmat, K, w, h: eval_seg.PinholeMockRasterizer(
            splats, viewmat, K, w, h)
    from usplat.gsplat_adapter import GSplatFeatureRasterizer
    return lambda splats, viewmat, K, w, h: GSplatFeatureRasterizer.from_splats(
        splats, viewmat, K, w, h, channel_chunk=channel_chunk)


def context(ckpt, args):
    """What both passes share: the map, the dataset, the keyframes, the PCA and the centre."""
    cfg = ckpt["config"]
    dev = args.device
    splats = load_gaussians(cfg["ply"], device=dev)
    N = int(splats["means"].shape[0])
    if N != int(ckpt["field"]["N"]):
        raise SystemExit(f"{cfg['ply']}: {N} Gaussians, but the field has {ckpt['field']['N']}")
    ds = make_dataset(cfg["dataset"], cfg["data"], stride=cfg["stride"])
    keep = select_keyframes(ds, cfg.get("kf_trans", 0.05), cfg.get("kf_rot", 5.0),
                            cfg.get("max_keyframes") or 0)
    idx = evenly(keep, args.frames)
    if len(idx) <= args.warmup:
        raise SystemExit(f"{len(idx)} keyframes to time with --warmup {args.warmup}: "
                         "nothing left to count")
    return dict(ckpt=ckpt, cfg=cfg, device=dev, scene=scene_name(cfg), splats=splats, N=N,
                d=int(ckpt["field"]["d"]), ds=ds, keyframes=keep, idx=idx,
                saved_keyframes=(ckpt.get("timing") or {}).get("keyframes"),
                pca=StreamingPCA.from_state(ckpt["pca"], device=dev),
                zbar=ckpt["field"].get("zbar"), use_depth=bool(cfg.get("use_depth")),
                rasterizer=rasterizer_factory(args.backend, int(cfg.get("channel_chunk") or 32)))


def load_extractor(cfg, device):
    """The mapping run's extractor, built as run_mapping.py builds it."""
    from usplat.extractors import get_extractor
    kw = (dict(model_name=cfg["clip_model"], pretrained=cfg["clip_pretrained"])
          if cfg["extractor"] == "clearclip" else {})
    return get_extractor(cfg["extractor"], device=device, **kw)


def _plain(x):
    """JSON-plain copy (tuples -> lists), for comparisons and the output file."""
    return json.loads(json.dumps(x, default=str))


def check_extractor(clip, ckpt):
    """The extractor must be the mapping run's: the same options (run_mapping.py saves them as
    extractor_options) and the PCA's input size; else the time is of another extractor."""
    opts = _plain({k: getattr(clip, k) for k in EXTRACTOR_OPTIONS if hasattr(clip, k)})
    saved = _plain(ckpt.get("extractor_options") or {})
    if opts != saved:
        raise SystemExit(f"extractor options {opts} != the mapping run's {saved}")
    if int(clip.d) != int(ckpt["pca"]["in_dim"]):
        raise SystemExit(f"the extractor gives {clip.d}-d tokens, the field's PCA takes "
                         f"{ckpt['pca']['in_dim']}")
    return opts


def make_baselines(field, ckpt, device):
    """The mapping run's baselines, empty: run_mapping.py --baselines builds
    usplat.baselines.default_baselines (NormLift with averaging, the view count, CSS); the CSS
    texts and class ids, its prior and NormLift's beta come from the saved configs."""
    cfgs = {e["type"]: e["config"] for e in ckpt["baselines"]}
    unknown = set(cfgs) - {"NormLift", "ViewCount", "CSS"}
    if unknown:
        raise SystemExit(f"saved baselines {sorted(unknown)} are not the default set")
    css = cfgs.get("CSS") or {}
    return B.default_baselines(field, text=css["text"].to(device) if css else None,
                               class_ids=css.get("class_ids"), alpha0=css.get("alpha0", 1.0),
                               beta=(cfgs.get("NormLift") or {}).get("beta", 1.0),
                               yager=css.get("yager", True))


# ------------------------------------------------------------------------------- passes
def extract_views(clip, ds, idx, factor, clock):
    """Phase 1, as run_mapping.py: the extractor tokens of every timed keyframe at full
    resolution ("extract"), moved to the CPU so the extractor can leave the GPU ("to_cpu", not
    in the total), with the pose, intrinsics and depth at the accumulation resolution."""
    views = []
    for i in idx:
        full = ds[i]
        clock.frame()
        with clock.part("extract"):
            tok = clip.extract_tokens(full["image"])
        with clock.part("to_cpu"):
            tok = tok.cpu()
        fr = downscale(full, factor)
        h, w = fr["image"].shape[:2]
        views.append(dict(index=i, tok=tok, c2w=full["c2w"], K=fr["K"], h=int(h), w=int(w),
                          depth=fr.get("depth"), image_hw=tuple(full["image"].shape[:2])))
    return views


def check_calls(clock, n, n_hooks):
    """Each keyframe ran exactly one render + backward and one _ingest (and every hook once);
    else update_view's code path changed and the parts would be wrong."""
    want = dict(render_backward=n, stats=n, hooks=n * n_hooks)
    got = {k: clock.calls.get(k, 0) for k in want}
    if got != want:
        raise RuntimeError(f"timed calls {got} != expected {want}: usplat/vmf_field.py "
                           "update_view no longer runs the parts this script times")


def map_pass(ctx, views, baselines=False):
    """Phase 2: a fresh field (with baselines=True also the mapping run's baselines) fed the
    timed keyframes in order, as run_mapping.py run_sequence. Returns (field, baselines, clock).
    The accumulation and _ingest are timed through wrappers, removed again at the end."""
    dev, d, clock = ctx["device"], ctx["d"], Clock(ctx["device"])
    field = DirectionalSemanticField(ctx["N"], d, field_config(ctx["ckpt"]), device=dev)
    field.set_centre(ctx["zbar"])
    bs = make_baselines(field, ctx["ckpt"], dev) if baselines else []
    css = [b for b in bs if isinstance(b, B.CSS)]
    field.view_hooks[:] = [clock.wrap(h, "hooks") for h in field.view_hooks]
    field._ingest = clock.wrap(field._ingest, "stats")
    accumulate = clock.wrap(vmf_field.bki_grad_accumulate_groups, "render_backward")
    means = ctx["splats"]["means"]
    try:
        with mock.patch.object(vmf_field, "bki_grad_accumulate_groups", accumulate):
            for v in views:
                tok = v["tok"].to(dev)                      # inputs on the device: not timed
                depth = (v["depth"].reshape(-1).to(dev)
                         if ctx["use_depth"] and v["depth"] is not None else None)
                clock.frame()
                with clock.part("project"):
                    z = ctx["pca"].project(upsample_tokens(tok, (v["h"], v["w"]))).reshape(-1, d)
                with clock.part("update"):
                    viewmat = torch.linalg.inv(v["c2w"].double()).float().to(dev)
                    ras = ctx["rasterizer"](ctx["splats"], viewmat, v["K"].to(dev), v["w"], v["h"])
                    kw = {}
                    if depth is not None:          # = GSplatFeatureRasterizer.gaussian_depths
                        kw = dict(depth_obs=depth,
                                  gauss_depth=means @ viewmat[2, :3] + viewmat[2, 3])
                    field.update_view(ras, z, num_pixels=v["h"] * v["w"], **kw)
                for c in css:                               # CSS: its own one-hot pass
                    with clock.part("css"):
                        c.update_view(ras, z)
    finally:
        del field._ingest                                   # back to the class method
    check_calls(clock, len(views), len(field.view_hooks))
    return field, bs, clock


def ours_frames(ext, ours):
    """Per-keyframe parts of our method (seconds): phase 1 joined with the pass without the
    baselines; other = update - render_backward - stats; total = the sum of the parts."""
    out = []
    for e, u in zip(ext.frames, ours.frames):
        f = dict(extract=e["extract"], to_cpu=e["to_cpu"], project=u["project"],
                 render_backward=u["render_backward"], stats=u["stats"])
        f["other"] = u["update"] - f["render_backward"] - f["stats"] - u.get("hooks", 0.0)
        f["total"] = f["extract"] + f["project"] + f["render_backward"] + f["stats"] + f["other"]
        out.append(f)
    return out


def change_record(field):
    """What the change test did in the timed pass. A static mapping pass resets nothing, so the
    re-admission step only tests field.changed.any()."""
    return dict(n_scored=int(torch.isfinite(field.c_last).sum()),
                n_resets=int(field.n_resets.sum()), n_changed=int(field.changed.sum()))


def reset_peak(device):
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def peak_mib(device):
    """Torch allocator peaks since the last reset, in MiB (None on the CPU)."""
    if not str(device).startswith("cuda"):
        return None
    dev = torch.device(device)
    return dict(allocated=torch.cuda.max_memory_allocated(dev) / 2 ** 20,
                reserved=torch.cuda.max_memory_reserved(dev) / 2 ** 20)


def saved_timing(ckpt, ext, with_b, mem):
    """The mapping run's own timing dict (run_mapping.py: no inner synchronize, every keyframe
    averaged) next to the same sums measured here over every keyframe: extraction + copy to the
    CPU, field update with the hooks, that plus CSS, and the peaks. Same GPU only."""
    t = ckpt.get("timing") or {}
    ms = lambda v: 1000.0 * statistics.fmean(v) if v else None
    meas = dict(extract=ms([f["extract"] + f["to_cpu"] for f in ext.frames]),
                peak_alloc_extract_mib=(mem.get("extract") or {}).get("allocated"),
                peak_alloc_mib=(mem.get("map_with_baselines") or {}).get("allocated"))
    if with_b is not None:
        meas["field"] = ms([f["update"] for f in with_b.frames])
        meas["update"] = ms([f["update"] + f.get("css", 0.0) for f in with_b.frames])
    saved = dict(gpu=t.get("gpu"), keyframes=t.get("keyframes"),
                 extract=t.get("ms_per_keyframe_extract"), field=t.get("ms_per_keyframe_field"),
                 update=t.get("ms_per_keyframe_update"),
                 peak_alloc_extract_mib=t.get("peak_alloc_extract_mib"),
                 peak_alloc_mib=t.get("peak_alloc_mib"))
    return dict(saved=saved, measured=meas)


# ------------------------------------------------------------------- state, fit, machine
def state_bytes(state):
    """Bytes per Gaussian of a saved field state (DirectionalSemanticField.state_dict): the sum
    of its tensors / N. All-zero tensors are saved as None and count 0 (e.g. the centred channel
    with centre=False). Without the change-test buckets: minus the recent and older sums (Rf, nf,
    Rs, ns); without the change test: minus every tensor only the change test uses (also its
    weight sums, scores and null histogram, a fixed 192 kB per scene)."""
    N = int(state["N"])
    per = {k: v.numel() * v.element_size() / N for k, v in state.items() if torch.is_tensor(v)}
    total = sum(per.values())
    part = lambda keys: sum(per.get(k, 0.0) for k in keys)
    return dict(n_gaussians=N, feat_dim=int(state["d"]), dtype=state.get("dtype"), total=total,
                without_change_buckets=total - part(CHANGE_BUCKETS),
                without_change_test=total - part(CHANGE_TEST), posterior=part(POSTERIOR),
                by_tensor=per, saved_as_none=sorted(k for k, v in state.items() if v is None))


def field_tensor_bytes(field):
    """Bytes per Gaussian the field object holds in memory: every tensor, also the ones the
    saved state leaves out because they are all zero (e.g. the unused centred channel)."""
    ts = [t for obj in (field, field.raw, field.cen) for t in vars(obj).values()
          if torch.is_tensor(t)]
    return sum(t.numel() * t.element_size() for t in ts) / field.N


def fit_meta(ply):
    """The fit meta scripts/fit_gs.py saves in the map (ckpt["meta"]), or None."""
    if str(ply).endswith(".ply"):
        return None
    try:
        ckpt = torch.load(ply, map_location="cpu", weights_only=True, mmap=True)
    except Exception as e:                                  # a record, not a reason to stop
        print(f"WARNING: cannot read the fit meta of {ply}: {e}")
        return None
    return ckpt.get("meta") if isinstance(ckpt, dict) else None


def fit_from_log(ply, logs):
    """The JSON line scripts/fit_gs.py prints last, from the first log whose "out" is this map."""
    want = os.path.abspath(ply)
    for path in sorted(glob.glob(logs)) if logs else []:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
        for line in reversed(lines):
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and os.path.abspath(str(rec.get("out", ""))) == want:
                return rec, path
    return None, None


def fit_time(ply, logs=""):
    """Fit time of one map in seconds: init + optimisation (time_init_s + time_train_s) from the
    saved fit meta, else the optimisation time in its fit log (the log line has no init time).
    Frame loading and the held-out evaluation are not counted."""
    meta, src = fit_meta(ply), ply
    if not meta or meta.get("time_train_s") is None:
        meta, src = fit_from_log(ply, logs)
    if not meta or meta.get("time_train_s") is None:
        return dict(map=os.path.abspath(ply), fit_s=None, source=None)
    keys = [k for k in ("time_init_s", "time_train_s") if meta.get(k) is not None]
    return dict(map=os.path.abspath(ply), fit_s=sum(float(meta[k]) for k in keys), parts=keys,
                wall_s=meta.get("wall_time_s"), gpu=meta.get("gpu"),
                n_gaussians=meta.get("n_gaussians"), iters=meta.get("iters"),
                source=os.path.abspath(src))


def fit_times(ply, maps="", logs=""):
    """The scene's fit time and, over the maps of --fit-maps (default: the scene's map only),
    mean and std in minutes, with the GPUs the fits ran on."""
    paths = sorted(glob.glob(maps)) if maps else []
    if maps and not paths:
        print(f"WARNING: --fit-maps {maps!r} matches no file: the scene's map only")
    per = [fit_time(p, logs) for p in (paths or [ply])]
    done = [r for r in per if r["fit_s"] is not None]
    return dict(scene=fit_time(ply, logs), maps=per,
                minutes=mean_std([r["fit_s"] for r in done], 1 / 60.0),
                gpus=sorted({r.get("gpu") or "unknown" for r in done}))


def _run(cmd):
    """stdout of a command, or 'error: ...' (the machine record never stops the timing)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                           env=dict(os.environ, LC_ALL="C"))
    except (OSError, subprocess.SubprocessError) as e:
        return f"error: {e}"
    return r.stdout.strip() if r.returncode == 0 else f"error: {r.stderr.strip() or r.returncode}"


def cpu_model(lscpu_text):
    """'Intel Core i7-13700KF' from lscpu's 'Model name:' line ((R), (TM), '13th Gen' dropped)."""
    for line in lscpu_text.splitlines():
        if line.strip().startswith("Model name:"):
            name = " ".join(re.sub(r"\((R|TM)\)", " ", line.split(":", 1)[1]).split())
            return re.sub(r"^\d+(st|nd|rd|th) Gen ", "", name)
    return None


def gpu_processes():
    """Compute processes on the GPUs now: [{pid, name, memory}] from nvidia-smi."""
    out = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader"])
    if out.startswith("error"):
        return out
    keys = ("pid", "name", "memory")
    return [dict(zip(keys, [s.strip() for s in ln.split(",")])) for ln in out.splitlines()
            if ln.strip()]


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def hardware(device):
    """The machine: the CPU (lscpu) and, on a CUDA device, the GPU (torch, nvidia-smi) and its
    compute processes when the timing starts (the paper says no other job was running)."""
    hw = dict(device=str(device), cpu=cpu_model(_run(["lscpu"])), torch=torch.__version__,
              cuda=torch.version.cuda, gsplat=_version("gsplat"), pid=os.getpid())
    if str(device).startswith("cuda"):
        hw["gpu"] = torch.cuda.get_device_name(torch.device(device))
        hw["nvidia_smi"] = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,"
                                 "power.limit", "--format=csv,noheader"])
        hw["gpu_processes_start"] = gpu_processes()
    return hw


def other_gpu_jobs(hw):
    """GPU compute processes other than this one, at the start or the end of the timing."""
    me, seen = str(hw.get("pid")), []
    for key in ("gpu_processes_start", "gpu_processes_end"):
        procs = hw.get(key)
        if isinstance(procs, list):
            seen += [p for p in procs if p.get("pid") != me and p not in seen]
    return seen


# -------------------------------------------------------------------------------- LaTeX
def _pm(st, digits=1):
    """'$m\\pm s$' of a mean_std dict, or the placeholder."""
    if not st or not st.get("n"):
        return MISSING
    return f"${st['mean']:.{digits}f}\\pm{st['std']:.{digits}f}$"


def _both(results, f):
    """'a' for one result, 'a (b)' for two: the paper writes 'Replica (RIO10)'."""
    vals = [f(r) for r in results]
    return vals[0] + "".join(f" ({v})" for v in vals[1:])


def _gb(r):
    m = r["memory_mib"].get("peak_allocated")
    return MISSING if m is None else f"{m / 1024:.1f}"


def _fit_min(r):
    st = r["fit"]["minutes"]
    if not st.get("n"):
        return MISSING
    return f"{st['mean']:.1f}" if st["n"] < 2 else _pm(st)


def _mib(m):
    return "n/a" if not m else f"{m['allocated']:.0f}"


def _article(word):
    return "an" if word and word[0].lower() in "aeiou" else "a"


def latex(results):
    """The sentences of "Run time and memory" (papers/main.tex) with the measured numbers:
    mean +- std over the counted keyframes; two results read 'first (second)', like the paper's
    'Replica (RIO10)'. Comment lines (%%) give the context and every warning."""
    r0, hw = results[0], results[0]["hardware"]
    both = lambda f: _both(results, f)
    ms = lambda part: (lambda r: _pm(r["ms"].get(part)))
    others = [p for r in results for p in other_gpu_jobs(r["hardware"])]
    # claim "no other job" only when nvidia-smi listed the GPU's processes for every result
    checked = all(isinstance(r["hardware"].get("gpu_processes_start"), list) for r in results)
    cpu = hw.get("cpu")
    where = ((f"one {hw['gpu']}" + (f" with {_article(cpu)} {cpu}" if cpu else ""))
             if hw.get("gpu") else f"the CPU ({cpu or 'model unknown'})")
    text = [f"We time one validation scene per dataset on {where}"
            + (" and no other job running." if checked and not others else "."),
            f"On {both(lambda r: DATASET_NAMES.get(r['dataset'], r['dataset']))}, a keyframe "
            f"takes {both(ms('total'))}\\,ms in total: {both(ms('extract'))}\\,ms for "
            f"{EXTRACTOR_NAMES.get(r0['extractor'], r0['extractor'])} feature extraction, "
            f"{both(ms('render_backward'))}\\,ms for the render and backward pass, and "
            f"{both(ms('stats'))}\\,ms for the statistics, change test and re-admission.",
            f"Peak GPU memory is {both(_gb)}\\,GB.",
            "Averaging and NormLift reuse the sums of our backward pass and add no render or "
            "backward pass.",
            "CSS needs its own backward pass for the class votes, which adds "
            f"{both(lambda r: _pm(r['ms_added'].get('css')))}\\,ms.",
            f"Fitting the 3D-GS map takes {both(_fit_min)}\\,min per scene, and the field state "
            f"takes {both(lambda r: format(r['state_bytes_per_gaussian']['total'], '.0f'))} "
            "bytes per Gaussian."]
    notes = []
    for r in results:
        k, m, s = r["keyframes"], r["memory_mib"], r["state_bytes_per_gaussian"]
        notes += [f"%% scripts/time_field.py: {r['scene']} ({r['dataset']}), {k['counted']} "
                  f"keyframes counted after {k['warmup']} warm-up, {r['device']}/{r['backend']}, "
                  f"{r['field']}",
                  f"%%   ms not in the text: project {_pm(r['ms'].get('project'))}, other "
                  f"{_pm(r['ms'].get('other'))}, tokens to the CPU {_pm(r['ms'].get('to_cpu'))}, "
                  f"hooks (averaging/NormLift, view count) {_pm(r['ms_added'].get('hooks'))}",
                  f"%%   peak GPU MiB (torch allocator): extraction {_mib(m.get('extract'))}, "
                  f"update {_mib(m.get('map'))}, update with the baselines "
                  f"{_mib(m.get('map_with_baselines'))}",
                  f"%%   field state per Gaussian (saved): {s['total']:.0f} B; without the "
                  f"change-test buckets {s['without_change_buckets']:.0f} B; without the change "
                  f"test {s['without_change_test']:.0f} B; posterior sums {s['posterior']:.0f} B; "
                  f"in memory (every tensor) {s.get('in_memory', float('nan')):.0f} B"]
        notes += [f"%%   WARNING: {w}" for w in r.get("warnings", [])]
    if others:
        notes.append(f"%% WARNING: other GPU compute processes ran: {others}")
    elif not checked:
        notes.append("%% the GPU's processes were not listed (CPU run or no nvidia-smi): "
                     "'no other job running' is left out")
    return "\n".join(notes + [" ".join(text)])


# ---------------------------------------------------------------------------------- run
def check_args(args):
    dev = args.device
    if args.backend == "gsplat" and not dev.startswith("cuda"):
        raise SystemExit("--backend gsplat needs a CUDA device; on the CPU use --device cpu "
                         "--backend mock")
    if args.backend == "mock" and dev != "cpu":
        raise SystemExit("--backend mock runs on the CPU only (--device cpu)")
    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if args.warmup < 0 or (args.frames > 0 and args.frames <= args.warmup):
        raise SystemExit(f"--frames {args.frames} with --warmup {args.warmup}: nothing left "
                         "to count")
    return dev


def warnings_for(ctx, args, hw, fit, code_same):
    """Everything that makes a number not the paper's number, in plain words."""
    out = []
    if args.backend == "mock":
        out.append("mock backend: a CPU smoke test, not a paper number")
    if ctx["saved_keyframes"] not in (None, len(ctx["keyframes"])):
        out.append(f"the mapping run used {ctx['saved_keyframes']} keyframes, the copied rule "
                   f"gives {len(ctx['keyframes'])}")
    if not code_same:
        out.append("the mapping code changed since the field was built")
    if ctx["cfg"].get("update_data"):
        out.append(f"the update sequence {ctx['cfg']['update_data']} is not timed (mapping "
                   "pass only)")
    if not ctx["ckpt"].get("baselines") or args.no_baselines:
        out.append("no pass with the baselines: the CSS time is missing")
    gpus = set(fit["gpus"])
    if fit["minutes"]["n"] and gpus != {hw.get("gpu")}:
        out.append(f"the fit times come from {sorted(gpus)}, the timing from "
                   f"{hw.get('gpu') or 'the CPU'}")
    return out


def report(out):
    """A few lines on the console (the JSON holds everything)."""
    line = lambda d, keys: " | ".join(f"{k} {d[k]['mean']:.2f} +- {d[k]['std']:.2f}"
                                      for k in keys if d.get(k) and d[k]["n"])
    print("ms per keyframe, ours: " + line(out["ms"], ("extract", "project", "render_backward",
                                                       "stats", "other", "total")))
    if out["ms_added"]:
        print("ms per keyframe added by the baselines: " + line(out["ms_added"], ("hooks", "css")))
    print(f"peak GPU MiB: {_plain(out['memory_mib'])}")
    s, m = out["state_bytes_per_gaussian"], out["fit"]["minutes"]
    print(f"field state: {s['total']:.1f} B per Gaussian ({s['without_change_buckets']:.1f} "
          f"without the change-test buckets, {s['without_change_test']:.1f} without the change "
          f"test)")
    if m["n"]:
        print(f"3D-GS fit: {m['mean']:.2f} +- {m['std']:.2f} min per scene (N = {m['n']}, "
              f"GPUs {out['fit']['gpus']})")
    for w in out["warnings"]:
        print(f"WARNING: {w}")


def run(args):
    """Phase 1 (tokens), the pass without and the pass with the baselines; returns the record."""
    t0 = time.time()
    dev = check_args(args)
    held = 0
    if dev.startswith("cuda"):
        from usplat import gpu_reserve                     # fraction from scripts/gpu_gate.sh
        held = gpu_reserve.apply(dev)
    hw = hardware(dev)
    ckpt = load_field_file(args.field)
    ctx = context(ckpt, args)
    cfg = ctx["cfg"]
    print(f"{ctx['scene']}: {ctx['N']} Gaussians, {len(ctx['keyframes'])} mapping keyframes "
          f"(saved run: {ctx['saved_keyframes']}); timing {len(ctx['idx'])}, the first "
          f"{args.warmup} not counted, on {dev}/{args.backend}", flush=True)

    reset_peak(dev)                                        # phase 1: extractor tokens
    clip = load_extractor(cfg, dev)
    ext_opts = check_extractor(clip, ckpt)
    ext = Clock(dev)
    views = extract_views(clip, ctx["ds"], ctx["idx"], cfg["downscale"], ext)
    del clip                                               # leaves the GPU, as in run_mapping.py
    mem = dict(extract=peak_mib(dev))

    reset_peak(dev)                                        # phase 2: our update alone
    field, _, ours = map_pass(ctx, views)
    mem["map"] = peak_mib(dev)
    change = change_record(field)
    in_memory = field_tensor_bytes(field)
    del field
    with_b = None
    if ckpt.get("baselines") and not args.no_baselines:   # + the mapping run's baselines
        reset_peak(dev)
        field, bs, with_b = map_pass(ctx, views, baselines=True)
        mem["map_with_baselines"] = peak_mib(dev)
        del field, bs
    if dev.startswith("cuda"):
        hw["gpu_processes_end"] = gpu_processes()
    peaks = [m["allocated"] for m in (mem["extract"], mem["map"]) if m]
    mem["peak_allocated"] = max(peaks) if peaks else None   # ours: max over the two phases

    frames = ours_frames(ext, ours)
    added = summarize(with_b.frames, args.warmup) if with_b else {}
    fit = fit_times(cfg["ply"], args.fit_maps, args.fit_logs)
    code_now = RM.mapping_code_state()
    code_field = (ckpt.get("code") or {}).get("mapping_code_sha256")
    v0 = views[0]
    out = dict(
        field=os.path.abspath(args.field), scene=ctx["scene"], dataset=cfg["dataset"],
        extractor=cfg["extractor"], extractor_options=ext_opts, device=dev,
        backend=args.backend, n_gaussians=ctx["N"], feat_dim=ctx["d"],
        image_hw=list(v0["image_hw"]), grid_hw=[v0["h"], v0["w"]],
        keyframes=dict(mapping=len(ctx["keyframes"]), saved_run=ctx["saved_keyframes"],
                       timed=len(ctx["idx"]), warmup=args.warmup,
                       counted=len(ctx["idx"]) - args.warmup, indices=ctx["idx"]),
        ms=summarize(frames, args.warmup),
        ms_added={k: v for k, v in added.items() if k in ("hooks", "css")},
        ms_with_baselines=added, memory_mib=mem, gpu_reserved_mib=held, change_test=change,
        state_bytes_per_gaussian=dict(state_bytes(ckpt["field"]), in_memory=in_memory), fit=fit,
        saved_timing=saved_timing(ckpt, ext, with_b, mem), hardware=hw,
        code=dict(field=code_field, now=code_now["mapping_code_sha256"],
                  same=code_field == code_now["mapping_code_sha256"], git=code_now.get("git")),
        per_keyframe_s=dict(ours=frames, with_baselines=with_b.frames if with_b else None))
    out["warnings"] = warnings_for(ctx, args, hw, fit, out["code"]["same"])
    out["latex"] = latex([out])
    out["provenance"] = dict(start=t0, wall_s=time.time() - t0)
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--field", help="main field .pt (scripts/run_mapping.py --out)")
    ap.add_argument("--frames", type=int, default=0,
                    help="keyframes to time, warm-up included, evenly spaced over the mapping "
                         "keyframes (0 = all: the mapping pass itself)")
    ap.add_argument("--warmup", type=int, default=3,
                    help="first keyframes of every pass that are not counted")
    ap.add_argument("--out", help="output JSON")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", choices=["gsplat", "mock"], default="gsplat",
                    help="mock: eval_seg.py's CPU pinhole rasterizer (smoke tests only)")
    ap.add_argument("--no-baselines", action="store_true",
                    help="skip the pass with the baselines (no hooks or CSS time)")
    ap.add_argument("--fit-maps", default="",
                    help="glob of 3D-GS maps whose fit times are averaged (default: the scene's)")
    ap.add_argument("--fit-logs", default="",
                    help="glob of fit_gs.py logs, read for a map without fit meta")
    ap.add_argument("--latex-from", nargs="+", default=None,
                    help="print the LaTeX of saved result JSON(s) (two: 'first (second)')")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.latex_from:
        text = latex([json.load(open(p)) for p in args.latex_from])
        print(text)
        return text
    if not (args.field and args.out):
        raise SystemExit("give --field and --out (or --latex-from)")
    out = run(args)
    out["provenance"]["argv"] = [sys.argv[0]] + list(argv if argv is not None else sys.argv[1:])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    report(out)
    print(out["latex"])
    print(f"wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
