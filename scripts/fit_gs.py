"""
fit_gs.py — fit a 3DGS map on a posed RGB(-D) sequence with gsplat (plan K9).

  scripts/gpu_gate.sh 3000 python3 scripts/fit_gs.py --dataset 7scenes \
      --data <7scenes>/chess/seq-01 --stride 10 --iters 3000 --max-res 640 --out chess.pt

Every FitConfig field of usplat/gs_fit.py is a flag (--max-gaussians, --init-voxel, ...;
booleans take 1/0, true/false, e.g. --opacity-reset 0 for the gsplat 1.5.3 behaviour).
Every --holdout-every-th frame (after the stride) is held out for PSNR. Progress goes to
stderr; the last stdout line is one JSON object with the metrics. The output .pt is loadable
by usplat.gs_io.load_gaussians_pt and carries the same metrics in ckpt["meta"], plus the
code identity: git '<hash>[+dirty]' (untracked files count as dirty), the sha256 of the
files that define the fit (CODE_FILES) and the config sha256.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from usplat.gs_fit import FitConfig, fit_dataset, save_gaussians_pt  # noqa: E402
from usplat.datasets import make_dataset, DATASETS  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the files whose content defines a fit (loader, fitter, CLI)
CODE_FILES = ("usplat/gs_fit.py", "usplat/datasets.py", "scripts/fit_gs.py")


def _git_rev(root):
    """'<hash>' or '<hash>+dirty' of the repo holding `root`; None without git. Untracked
    (not ignored) files count as dirty: new code need not be in any commit."""
    try:
        h = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        if not h:
            return None
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except Exception:
        return None


def code_sha256(root=ROOT, files=CODE_FILES):
    """{relative path: sha256 of the file} (None for a missing file)."""
    out = {}
    for rel in files:
        try:
            with open(os.path.join(root, rel), "rb") as f:
                out[rel] = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            out[rel] = None
    return out


def _bool(v):
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean (1/0, true/false), got '{v}'")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--data", required=True, help="dataset root")
    ap.add_argument("--stride", type=int, default=1, help="take every stride-th frame")
    ap.add_argument("--out", required=True, help="output .pt")
    ap.add_argument("--device", default="cuda")
    choices = dict(init=["auto", "depth", "random"])
    for f in fields(FitConfig):                                # --iters, --max-gaussians, ...
        typ = _bool if isinstance(f.default, bool) else type(f.default)
        ap.add_argument("--" + f.name.replace("_", "-"), type=typ,
                        default=f.default, choices=choices.get(f.name))
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = FitConfig(**{f.name: getattr(args, f.name) for f in fields(FitConfig)})
    if not torch.cuda.is_available():
        sys.exit("fit_gs.py needs CUDA (gsplat)")
    frac = os.environ.get("UAGS_GPU_MEM_FRACTION")          # set by scripts/gpu_gate.sh
    if frac:
        torch.cuda.set_per_process_memory_fraction(float(frac))

    # code identity before the fit: files edited during a long run do not change it
    code = dict(git=_git_rev(ROOT), code_sha256=code_sha256())
    ds = make_dataset(args.dataset, args.data, stride=args.stride)
    print(f"[fit] dataset {args.dataset}: {len(ds)} frames (stride {args.stride})",
          file=sys.stderr, flush=True)
    splats, meta = fit_dataset(ds, cfg, device=args.device,
                               log=lambda s: print(s, file=sys.stderr, flush=True))
    meta.update(args=vars(args), **code,
                gsplat=str(__import__("gsplat").__version__), torch=str(torch.__version__),
                gpu=torch.cuda.get_device_name(), mem_fraction=frac)
    save_gaussians_pt(args.out, splats, meta)
    keys = ("psnr", "ssim", "psnr_init", "psnr_train", "n_gaussians", "n_init", "init_method",
            "cap_hit_step", "n_opacity_resets", "reset_every", "peak_mem_mib",
            "peak_reserved_mib", "proc_mem_mib", "proc_peak_mib", "wall_time_s",
            "time_train_s", "iters", "n_train", "n_test", "height", "width", "downscale",
            "scene_scale", "n_dropped", "dropped_frames", "git", "config_sha256")
    print(json.dumps(dict({k: meta[k] for k in keys}, out=os.path.abspath(args.out))),
          flush=True)


if __name__ == "__main__":
    main()
