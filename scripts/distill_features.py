"""
distill_features.py — train the optimization-based baselines of Table I (usplat/distill.py:
Feature 3DGS and LEGaussians, re-implemented) on a scene's 3D-GS map with the SAME mapping
frames, LSeg tokens and PCA basis as the scene's main field (run_mapping.py --out), and save
them next to it as <main>_f3dgs.pt or <main>_legs.pt.

    scripts/gpu_gate.sh 7000 python3 scripts/distill_features.py \
        --field /home/thanhnc/gauss_ws/data/uags_fields/office_0_lseg.pt --method legs

The mapping frames are those of the main field: its dataset, --stride, --downscale, the
evenly spaced --max-keyframes subset and the near-duplicate pose filter (kf_trans, kf_rot).
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
from usplat import distill as D                                             # noqa: E402

CODE = ("scripts/distill_features.py", "usplat/distill.py")


def code_sha256():
    repo = os.path.dirname(HERE)
    import run_mapping
    h = hashlib.sha256()
    for rel in CODE + tuple(run_mapping.MAPPING_CODE):
        with open(os.path.join(repo, rel), "rb") as f:
            h.update(rel.encode() + b"\0" + hashlib.sha256(f.read()).hexdigest().encode() + b"\n")
    return h.hexdigest()


def select_frames(ds, cfg):
    """run_mapping.py's frame selection: evenly spaced subset, finite poses, near-duplicate
    pose filter against the last kept frame."""
    import run_mapping
    last, keep = None, []
    order = range(len(ds))
    mk = cfg.get("max_keyframes") or 0
    if mk and mk < len(ds):
        order = sorted(set(torch.linspace(0, len(ds) - 1, mk).round().long().tolist()))
    for i in order:
        c2w = ds[i]["c2w"]
        if not torch.isfinite(c2w).all():
            continue
        if last is not None:
            dt, dr = run_mapping._pose_gap(c2w, last)
            if dt < cfg["kf_trans"] and dr < cfg["kf_rot"]:
                continue
        last = c2w
        keep.append(i)
    return keep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--field", required=True, help="the scene's main field (run_mapping --out)")
    ap.add_argument("--method", choices=D.METHODS, required=True)
    ap.add_argument("--lam-u", type=float, default=1.0, help="LEGaussians lambda_u (its value: 1)")
    ap.add_argument("--iters", type=int, default=30000, help="at most (the papers' count)")
    ap.add_argument("--min-iters", type=int, default=5000)
    ap.add_argument("--tol", type=float, default=0.01,
                    help="stop when the loss improves less than this over 5k iterations")
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam lr (both papers: 1e-3)")
    ap.add_argument("--alpha-min", type=float, default=0.5)
    ap.add_argument("--codes", type=int, default=D.LEGS_CODES, help="LEGaussians codebook size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channel-chunk", type=int, default=32)
    ap.add_argument("--out", default="", help="default: next to --field (usplat.distill.out_path)")
    args = ap.parse_args(argv)

    from usplat import gpu_reserve
    from usplat.gs_io import load_gaussians
    from usplat.datasets import make_dataset, downscale
    from usplat.clip_dense import upsample_tokens
    from usplat.feature_pca import StreamingPCA
    from usplat.gsplat_adapter import GSplatFeatureRasterizer
    from usplat.extractors import get_extractor

    device = "cuda"
    torch.manual_seed(args.seed)
    held = gpu_reserve.apply(device)
    t0 = time.time()
    ck = torch.load(args.field, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    name = D.variant_name(args.method, args.lam_u)
    out = args.out or D.out_path(args.field, name)
    pca = StreamingPCA.from_state(ck["pca"], device=device)
    d = int(cfg["feat_dim"])
    splats = load_gaussians(cfg["ply"], device=device)
    N = splats["means"].shape[0]
    ds = make_dataset(cfg["dataset"], cfg["data"], stride=cfg["stride"])
    idx = select_frames(ds, cfg)
    print(f"{name}: {N} Gaussians, {len(idx)} mapping frames of {len(ds)} (main field {args.field})")

    ext = get_extractor(cfg["extractor"], device=device)
    toks = [ext.extract_tokens(ds[i]["image"]).cpu() for i in idx]           # extractor phase
    del ext
    t_ext = time.time() - t0

    views = []
    for i, tok in zip(idx, toks):
        fr = downscale(ds[i], cfg["downscale"])
        h, w = fr["image"].shape[:2]
        viewmat = torch.linalg.inv(fr["c2w"].double()).float().to(device)
        ras = GSplatFeatureRasterizer.from_splats(splats, viewmat, fr["K"].to(device), w, h,
                                                  channel_chunk=args.channel_chunk)

        def target(tok=tok, h=h, w=w):
            return pca.project(upsample_tokens(tok.to(device), (h, w))).reshape(-1, d)
        views.append((ras, target))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    res = D.train_features(views, N, d, method=args.method, iters=args.iters, lr=args.lr,
                           lam_u=args.lam_u, alpha_min=args.alpha_min, seed=args.seed,
                           device=device, n_codes=args.codes, min_iters=args.min_iters,
                           tol=args.tol, ckpt_path=f"{out}.ckpt")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_train = time.time() - t1
    conf = dict(method=args.method, lam_u=args.lam_u if args.method == "legs" else None,
                codes=args.codes if args.method == "legs" else None,
                iters=args.iters, min_iters=args.min_iters, tol=args.tol, lr=args.lr,
                alpha_min=args.alpha_min, seed=args.seed)
    code = code_sha256()
    ident = hashlib.sha256(json.dumps(dict(code=code, **conf), sort_keys=True).encode()).hexdigest()
    rec = dict(name=name, features=res["features"].cpu(),
               u=None if res["u"] is None else res["u"].cpu(), history=res["history"],
               decoder=None if res["decoder"] is None else res["decoder"].state_dict(),
               decoder_hidden=D.LEGS_HIDDEN,
               config=conf, identity=ident, code_sha256=code, main_field=os.path.abspath(args.field),
               pca_basis_sha256=D.basis_sha256(ck["pca"]), n_frames=len(idx),
               iters_run=res["iters_run"],
               timing=dict(extract_s=t_ext, train_s=t_train, iters=res["iters_run"],
                           ms_per_iter=1000 * t_train / max(res["iters_run"], 1),
                           peak_alloc_mib=(torch.cuda.max_memory_allocated() / 2 ** 20
                                           if torch.cuda.is_available() else None),
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
