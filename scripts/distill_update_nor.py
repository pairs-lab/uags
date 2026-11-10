"""
distill_update_nor.py — Feature 3DGS / LEGaussians re-trained with the update frames of one
NeRF-Object-Removal scene (configs/stale_baselines_prereg.json): the optimization-based baselines
of Table I (usplat/distill.py, scripts/distill_features.py: imported, never edited) trained on
the scene's mapping AND update keyframes, so that they have seen the frames our field has seen
after its update.

    scripts/gpu_gate.sh 5000 python3 scripts/distill_update_nor.py \
        --table results/nor/tab2_relmax/test/008.json --method f3dgs

Frames: the mapping and the update keyframes that scripts/eval_change_nor.py load_context reads
for the scene's Table III record (its field, update stride, alignment and keyframe rule; the
arguments come from scripts/dump_nor_qual.py ec_namespace). Targets, optimiser, iterations, stop
rule and seed are distill_features.py's defaults (Table I, Table III).
Output: <field>_<name>_upd.pt (name: f3dgs or legs_l<lambda>), in distill_features.py's format
plus n_map_frames / n_update_frames; --out overrides the path.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import distill_features as DF                                               # noqa: E402
import dump_nor_qual as DQ                                                  # noqa: E402
from usplat import distill as D                                             # noqa: E402

EC = DQ.EC
SUFFIX = "_upd"
CODE = ("scripts/distill_update_nor.py",) + tuple(DF.CODE)


def code_sha256():
    """distill_features.code_sha256() extended by this file."""
    with open(os.path.abspath(__file__), "rb") as f:
        own = hashlib.sha256(f.read()).hexdigest()
    return hashlib.sha256((DF.code_sha256() + "\n" + own).encode()).hexdigest()


def training_views(ctx):
    """[(rasterizer, target_fn)] of the mapping keyframes, then the update keyframes: the
    rasterizer and the PCA-64 feature map of eval_change_nor.view_inputs."""
    from usplat.clip_dense import upsample_tokens
    dev, d = ctx["device"], ctx["d"]
    views = []
    for v in list(ctx["map_views"]) + list(ctx["upd_views"]):
        ras, _ = EC.rasterizer(ctx, v)

        def target(v=v):
            return ctx["pca"].project(upsample_tokens(v["tok"].to(dev), (v["h"], v["w"]))).reshape(-1, d)
        views.append((ras, target))
    return views


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", required=True, help="the scene's Table III record (JSON)")
    ap.add_argument("--method", choices=D.METHODS, required=True)
    ap.add_argument("--lam-u", type=float, default=1.0, help="LEGaussians lambda_u")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--min-iters", type=int, default=5000)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha-min", type=float, default=0.5)
    ap.add_argument("--codes", type=int, default=D.LEGS_CODES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["gsplat", "mock"], default="gsplat")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="", help="default: <field>_<name>_upd.pt")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.smoke = 0
    m = DQ.record_body(json.load(open(args.table)))
    ns = DQ.ec_namespace(m, args)
    dev = args.device
    torch.manual_seed(args.seed)
    held = 0
    if str(dev).startswith("cuda"):
        from usplat import gpu_reserve                       # fraction from scripts/gpu_gate.sh
        held = gpu_reserve.apply(dev)
    t0 = time.time()
    ctx = EC.load_context(ns)
    N, d = ctx["N"], ctx["d"]
    name = D.variant_name(args.method, args.lam_u) + SUFFIX
    out = args.out or D.out_path(ns.field, name)
    views = training_views(ctx)
    n_map, n_upd = len(ctx["map_views"]), len(ctx["upd_views"])
    print(f"{name}: {N} Gaussians, {n_map} mapping + {n_upd} update keyframes (field {ns.field})")
    t_ext = time.time() - t0
    if torch.cuda.is_available() and str(dev).startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.time()
    res = D.train_features(views, N, d, method=args.method, iters=args.iters, lr=args.lr,
                           lam_u=args.lam_u, alpha_min=args.alpha_min, seed=args.seed,
                           device=dev, n_codes=args.codes, min_iters=args.min_iters,
                           tol=args.tol, ckpt_path=f"{out}.ckpt")
    if torch.cuda.is_available() and str(dev).startswith("cuda"):
        torch.cuda.synchronize()
    t_train = time.time() - t1
    conf = dict(method=args.method, lam_u=args.lam_u if args.method == "legs" else None,
                codes=args.codes if args.method == "legs" else None,
                iters=args.iters, min_iters=args.min_iters, tol=args.tol, lr=args.lr,
                alpha_min=args.alpha_min, seed=args.seed, frames="mapping+update")
    code = code_sha256()
    ident = hashlib.sha256(json.dumps(dict(code=code, **conf), sort_keys=True).encode()).hexdigest()
    rec = dict(name=name, features=res["features"].cpu(),
               u=None if res["u"] is None else res["u"].cpu(), history=res["history"],
               decoder=None if res["decoder"] is None else res["decoder"].state_dict(),
               decoder_hidden=D.LEGS_HIDDEN, config=conf, identity=ident, code_sha256=code,
               main_field=os.path.abspath(ns.field), table=os.path.abspath(args.table),
               table_sha256=DQ.sha256_file(args.table),
               pca_basis_sha256=D.basis_sha256(ctx["ckpt"]["pca"]),
               n_frames=n_map + n_upd, n_map_frames=n_map, n_update_frames=n_upd,
               update_index=[int(v["index"]) for v in ctx["upd_views"]],
               iters_run=res["iters_run"],
               timing=dict(extract_s=t_ext, train_s=t_train, iters=res["iters_run"],
                           ms_per_iter=1000 * t_train / max(res["iters_run"], 1),
                           peak_alloc_mib=(torch.cuda.max_memory_allocated() / 2 ** 20
                                           if torch.cuda.is_available() and str(dev).startswith("cuda")
                                           else None),
                           gpu_reserved_mib=held))
    tmp = f"{out}.tmp{os.getpid()}"
    torch.save(rec, tmp)
    os.replace(tmp, out)
    if os.path.exists(f"{out}.ckpt"):
        os.remove(f"{out}.ckpt")                   # the training state is no longer needed
    print(f"saved {name} -> {out} ({res['iters_run']} iterations, train {t_train:.0f} s, "
          f"{rec['timing']['ms_per_iter']:.1f} ms/iter, final loss {res['history'][-1][1]:.4f})")
    return rec


if __name__ == "__main__":
    main()
