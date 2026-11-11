"""
distill_update_swap.py — Feature 3DGS / LEGaussians re-trained with the update frames of one
Replica semantic-swap scene (configs/swap_trained_prereg.json): the optimization-based baselines
of Table I (usplat/distill.py train_features at scripts/distill_features.py's defaults; both
imported, never edited) trained on the swap field's mapping keyframes AND its update keyframes,
whose features went through the field's own swap transform (usplat.swap: the same bank, spec,
seed and frames as the field's frames log), in the field's PCA basis.

    scripts/gpu_gate.sh 5000 python3 scripts/distill_update_swap.py \
        --field results/swap_relmax/fields/office_0_swap.pt --method legs --lam-u 4 \
        --frames-ref results/swap_relmax/fields/office_0_swap.frames.json

Frames: run_mapping.py's rule (distill_features.select_frames: the field's stride, evenly spaced
max_keyframes subset, finite poses, near-duplicate filter kf_trans / kf_rot) on the field's
mapping sequence (Sequence_1) and on its update sequence (Sequence_2, no subset). The update
frames must be the frames the field's transform logged, mapping + update the field's keyframe
count. Features as run_mapping.py makes them: the extractor's tokens (same options as the field)
-> bilinear upsample to the accumulation grid -> the field's saved PCA; update frames then pass
through the transform built from the field's --feature-transform-arg (bank, spec, scene; never
its log), which must log exactly the field's frames (i, dataset index, swapped pixels per
source). Only the swapped rows are kept per update frame and put back into each target.
Refused: a field without the swap transform; a bank, spec, scene or frames log that disagree; a
usplat/swap.py other than the one that wrote the log; a map file other than the field's; a
--frames-ref other than the field's log; an output or a resume checkpoint of another PCA basis.
Output: <field>_<name>_upd.pt (name: f3dgs or legs_l<lambda_u>), distill_features.py's format
plus provenance: frames, n_map_frames / n_update_frames, bank, spec, frames-log and code sha256.
Backends: gsplat (GPU) or mock (CPU, tests/test_swap_trained.py).
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
from usplat import distill as D                                             # noqa: E402
from usplat import swap as SW                                               # noqa: E402
from usplat.paths import moved                                              # noqa: E402

SUFFIX = "_upd"
FACTORY = "usplat.swap:swap_factory"
FRAMES = "mapping+update (swap transform)"
CODE = ("scripts/distill_update_swap.py", "usplat/swap.py") + tuple(DF.CODE)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_sha256():
    """distill_features.code_sha256() extended by this file and usplat/swap.py."""
    with open(os.path.abspath(__file__), "rb") as f:
        own = hashlib.sha256(f.read()).hexdigest()
    return hashlib.sha256((DF.code_sha256() + "\n" + own + "\n" + SW.code_sha256())
                          .encode()).hexdigest()


def log_frames(log):
    """(i, dataset index, sorted swapped pixels per source) of every frame of a frames log."""
    return [[int(e["i"]), int(e["index"]), sorted([str(k), int(v)] for k, v in e["n_px"].items())]
            for e in log["frames"]]


def _pairs(x):
    return [(int(p["source_id"]), int(p["target_id"])) for p in x["pairs"]]


def same_log(a, b):
    """Two frames logs of the same swap: scene, spec, pairs, transform code and frames."""
    return (a.get("scene") == b.get("scene") and a.get("spec_sha256") == b.get("spec_sha256")
            and a.get("code_sha256") == b.get("code_sha256") and _pairs(a) == _pairs(b)
            and log_frames(a) == log_frames(b))


def swap_inputs(ckpt):
    """The swap field's transform inputs, checked against each other: dict(arg, spec, spec_sha,
    bank, bank_sha, log, log_path)."""
    cfg = ckpt.get("config") or {}
    if cfg.get("feature_transform") != FACTORY:
        raise SystemExit(f"not a swap field: feature transform {cfg.get('feature_transform')!r} "
                         f"!= {FACTORY!r}")
    a = SW.parse_arg(cfg.get("feature_transform_arg"))
    miss = [k for k in ("spec", "scene", "log") if k not in a]
    if miss:
        raise SystemExit(f"the field's transform argument names no {miss}: cannot check its inputs")
    for k in ("bank", "spec", "log"):
        a[k] = moved(a[k])
        if not os.path.isfile(a[k]):
            raise SystemExit(f"the field's {k} {a[k]} is missing (rebuild it: scripts/swap_trained.sh)")
    spec, spec_sha = SW.load_spec(a["spec"])
    bank = SW.load_bank(a["bank"])
    if bank["spec_sha256"] != spec_sha:
        raise SystemExit(f"{a['bank']} was built from another spec than {a['spec']}")
    if bank["scene"] != a["scene"]:
        raise SystemExit(f"bank scene {bank['scene']!r} != the field's {a['scene']!r}")
    log = json.load(open(a["log"]))
    if log.get("schema") != "uags.swaplog/1" or log.get("scene") != a["scene"] or \
            log.get("spec_sha256") != spec_sha or _pairs(log) != _pairs(bank):
        raise SystemExit(f"{a['log']}: not the frames log of this bank, spec and scene")
    if log.get("code_sha256") != SW.code_sha256():
        raise SystemExit(f"usplat/swap.py changed since {a['log']} was written (code "
                         f"{str(log.get('code_sha256'))[:12]} != {SW.code_sha256()[:12]}): the "
                         f"transform cannot be replayed")
    if not log["frames"]:
        raise SystemExit(f"{a['log']}: no frames")
    return dict(arg=a, spec=spec, spec_sha=spec_sha, bank=bank, bank_sha=sha256_file(a["bank"]),
                log=log, log_path=a["log"])


def datasets(cfg):
    """The field's mapping and update datasets, as run_mapping.py opened them."""
    import run_mapping as RM
    from usplat.datasets import make_dataset
    align = cfg.get("align") or ""
    m = make_dataset(cfg["dataset"], moved(cfg["data"]), stride=cfg["stride"],
                     **RM._part(cfg.get("part"), align))
    u = make_dataset(cfg["dataset"], moved(cfg["update_data"]), stride=cfg["update_stride"],
                     **RM._part(cfg.get("update_part"), align))
    return m, u


def select_frames(ckpt, map_ds, upd_ds, log):
    """(mapping, update) frame indices of the field (run_mapping.py's selection), checked
    against its frames log and keyframe count."""
    cfg = ckpt["config"]
    map_idx = DF.select_frames(map_ds, cfg)
    upd_idx = DF.select_frames(upd_ds, dict(cfg, max_keyframes=0))     # run_mapping: select(upd)
    logged = [int(e["i"]) for e in log["frames"]]
    if upd_idx != logged:
        raise SystemExit(f"{len(upd_idx)} update frames by run_mapping.py's rule, the field's "
                         f"transform logged {len(logged)} other frames")
    kf = (ckpt.get("timing") or {}).get("keyframes")
    if kf is not None and int(kf) != len(map_idx) + len(upd_idx):
        raise SystemExit(f"{len(map_idx)} mapping + {len(upd_idx)} update frames here, the field "
                         f"ingested {kf}")
    return map_idx, upd_idx


def check_map(ckpt, ply):
    ent = (ckpt.get("inputs") or {}).get("ply")
    if ent and ent.get("sha256") and sha256_file(ply) != ent["sha256"]:
        raise SystemExit(f"{ply} is not the map the field was built on (sha256 differs)")


def check_basis(out, basis):
    """Refuse an existing output, and a resume checkpoint of another PCA basis (train_features
    resumes from <out>.ckpt, checking only the method and N)."""
    if os.path.exists(out):
        d = torch.load(out, map_location="cpu", weights_only=False)
        if d.get("pca_basis_sha256") != basis:
            raise SystemExit(f"{out} exists, trained in another PCA basis than the field: delete it")
        raise SystemExit(f"{out} exists (same basis): delete it to train again")
    ck, side = f"{out}.ckpt", f"{out}.ckpt.json"
    if os.path.exists(ck):
        prev = json.load(open(side)) if os.path.exists(side) else {}
        if prev.get("pca_basis_sha256") != basis:
            raise SystemExit(f"{ck}: a resume checkpoint of another PCA basis (or of unknown "
                             f"basis): delete it")


def build_views(ckpt, sw, map_ds, upd_ds, map_idx, upd_idx, ext, pca, splats, backend, device,
                channel_chunk=32):
    """[(rasterizer, target_fn)] of the mapping frames, then of the update frames (their
    targets carry the swapped rows of the replayed transform), and per update frame the
    number of swapped rows."""
    from argparse import Namespace
    from eval_seg import make_rasterizer
    from usplat.clip_dense import upsample_tokens
    from usplat.datasets import downscale
    cfg = ckpt["config"]
    d, f = int(cfg["feat_dim"]), int(cfg["downscale"])
    a = sw["arg"]
    xform = SW.swap_factory(f"bank={a['bank']},spec={a['spec']},scene={a['scene']}", pca, device)
    rargs = Namespace(channel_chunk=channel_chunk)

    def project(tok, h, w):                       # run_mapping.run_sequence's z
        return pca.project(upsample_tokens(tok.to(device), (h, w))).reshape(-1, d)

    def view(ds, i, rep=None):
        fr_full = ds[i]
        fr = downscale(fr_full, f)
        h, w = fr["image"].shape[:2]
        tok = ext.extract_tokens(fr_full["image"]).cpu()
        vm = torch.linalg.inv(fr_full["c2w"].double()).float().to(device)
        ras = make_rasterizer(backend, splats, dict(viewmat=vm, K=fr["K"].to(device), w=w, h=h),
                              rargs)
        return fr, h, w, tok, ras

    views, n_rows = [], []
    for i in map_idx:
        _, h, w, tok, ras = view(map_ds, i)
        views.append((ras, lambda tok=tok, h=h, w=w: project(tok, h, w)))
    for i in upd_idx:
        fr, h, w, tok, ras = view(upd_ds, i)
        z = project(tok, h, w)
        out = xform(z, fr, i, (h, w))
        rows = torch.nonzero((out != z).any(1)).flatten()
        rep = (rows.cpu(), out[rows].cpu())

        def target(tok=tok, h=h, w=w, rep=rep):
            z = project(tok, h, w)
            z[rep[0].to(z.device)] = rep[1].to(z.device)
            return z
        if not torch.equal(target(), out):
            raise SystemExit(f"update frame {i}: the stored swap does not reproduce the transform")
        views.append((ras, target))
        n_rows.append(int(rows.numel()))
    if log_frames(dict(frames=xform.log)) != log_frames(sw["log"]):
        raise SystemExit("the replayed transform swapped other pixels than the field's frames log")
    return views, n_rows


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--field", required=True, help="the scene's swap field (run_mapping --out)")
    ap.add_argument("--method", choices=D.METHODS, required=True)
    ap.add_argument("--lam-u", type=float, default=4.0,
                    help="LEGaussians lambda_u (4: Table I's Replica choice, row legs_l4)")
    ap.add_argument("--frames-ref", default="",
                    help="a kept frames log the field's own log must equal (the paper run's)")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--min-iters", type=int, default=5000)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha-min", type=float, default=0.5)
    ap.add_argument("--codes", type=int, default=D.LEGS_CODES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channel-chunk", type=int, default=32)
    ap.add_argument("--backend", choices=["gsplat", "mock"], default="gsplat")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="", help="default: <field>_<name>_upd.pt")
    return ap


def main(argv=None, extractor=None):
    """extractor: tests pass a stand-in for the field's extractor."""
    args = build_parser().parse_args(argv)
    from usplat.feature_pca import StreamingPCA
    from usplat.gs_io import load_gaussians
    dev = args.device
    torch.manual_seed(args.seed)
    held = 0
    if str(dev).startswith("cuda"):
        from usplat import gpu_reserve                       # fraction from scripts/gpu_gate.sh
        held = gpu_reserve.apply(dev)
    t0 = time.time()
    ckpt = torch.load(args.field, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    sw = swap_inputs(ckpt)
    if args.frames_ref and not same_log(sw["log"], json.load(open(args.frames_ref))):
        raise SystemExit(f"the field's frames log {sw['log_path']} is not {args.frames_ref} "
                         f"(scene, spec, pairs, transform code or frames differ)")
    basis = D.basis_sha256(ckpt["pca"])
    name = D.variant_name(args.method, args.lam_u) + SUFFIX
    out = args.out or D.out_path(args.field, name)
    check_basis(out, basis)
    ply = moved(cfg["ply"])
    check_map(ckpt, ply)
    map_ds, upd_ds = datasets(cfg)
    map_idx, upd_idx = select_frames(ckpt, map_ds, upd_ds, sw["log"])
    pca = StreamingPCA.from_state(ckpt["pca"], device=dev)
    d = int(cfg["feat_dim"])
    if tuple(pca.basis.shape)[1] != d:
        raise SystemExit(f"PCA basis {tuple(pca.basis.shape)} != feat_dim {d}")
    splats = load_gaussians(ply, device=dev)
    N = splats["means"].shape[0]
    if int(ckpt["field"]["N"]) != N:
        raise SystemExit(f"the map has {N} Gaussians, the field {ckpt['field']['N']}")
    ext = extractor
    if ext is None:
        from usplat.extractors import get_extractor
        ext = get_extractor(cfg["extractor"], device=dev)
    opts = {k: getattr(ext, k) for k in ("token_stride", "flip", "scales", "half", "base_size",
                                         "crop_size") if hasattr(ext, k)}
    if ckpt.get("extractor_options") is not None and opts != ckpt["extractor_options"]:
        raise SystemExit(f"extractor options {opts} != the field's {ckpt['extractor_options']}")
    views, n_rows = build_views(ckpt, sw, map_ds, upd_ds, map_idx, upd_idx, ext, pca, splats,
                                args.backend, dev, args.channel_chunk)
    del ext
    n_map, n_upd = len(map_idx), len(upd_idx)
    print(f"{name}: {N} Gaussians, {n_map} mapping + {n_upd} update keyframes, "
          f"{sum(n_rows)} swapped rows (field {args.field})")
    conf = dict(method=args.method, lam_u=args.lam_u if args.method == "legs" else None,
                codes=args.codes if args.method == "legs" else None,
                iters=args.iters, min_iters=args.min_iters, tol=args.tol, lr=args.lr,
                alpha_min=args.alpha_min, seed=args.seed, channel_chunk=args.channel_chunk,
                frames=FRAMES)
    code = code_sha256()
    ident = hashlib.sha256(json.dumps(dict(code=code, **conf), sort_keys=True).encode()).hexdigest()
    with open(f"{out}.ckpt.json", "w") as fh:                # the basis of a resume checkpoint
        json.dump(dict(pca_basis_sha256=basis, identity=ident), fh)
    t_ext = time.time() - t0
    cuda = torch.cuda.is_available() and str(dev).startswith("cuda")
    if cuda:
        torch.cuda.synchronize()
    t1 = time.time()
    res = D.train_features(views, N, d, method=args.method, iters=args.iters, lr=args.lr,
                           lam_u=args.lam_u, alpha_min=args.alpha_min, seed=args.seed,
                           device=dev, n_codes=args.codes, min_iters=args.min_iters,
                           tol=args.tol, ckpt_path=f"{out}.ckpt")
    if cuda:
        torch.cuda.synchronize()
    t_train = time.time() - t1
    a = sw["arg"]
    rec = dict(name=name, features=res["features"].cpu(),
               u=None if res["u"] is None else res["u"].cpu(), history=res["history"],
               decoder=None if res["decoder"] is None else res["decoder"].state_dict(),
               decoder_hidden=D.LEGS_HIDDEN, config=conf, identity=ident, code_sha256=code,
               main_field=os.path.abspath(args.field), scene=a["scene"],
               pca_basis_sha256=basis,
               n_frames=n_map + n_upd, n_map_frames=n_map, n_update_frames=n_upd,
               frames=dict(mapping=[int(i) for i in map_idx], update=log_frames(sw["log"]),
                           swapped_rows=n_rows),
               bank=dict(path=os.path.abspath(a["bank"]), sha256=sw["bank_sha"]),
               spec=dict(path=os.path.abspath(a["spec"]), sha256=sha256_file(a["spec"]),
                         spec_sha256=sw["spec_sha"]),
               frames_log=dict(path=os.path.abspath(sw["log_path"]),
                               sha256=sha256_file(sw["log_path"])),
               frames_ref=(dict(path=os.path.abspath(args.frames_ref),
                                sha256=sha256_file(args.frames_ref)) if args.frames_ref else None),
               transform_code_sha256=SW.code_sha256(),
               iters_run=res["iters_run"],
               timing=dict(extract_s=t_ext, train_s=t_train, iters=res["iters_run"],
                           ms_per_iter=1000 * t_train / max(res["iters_run"], 1),
                           peak_alloc_mib=(torch.cuda.max_memory_allocated() / 2 ** 20
                                           if cuda else None),
                           gpu_reserved_mib=held))
    tmp = f"{out}.tmp{os.getpid()}"
    torch.save(rec, tmp)
    os.replace(tmp, out)
    for p in (f"{out}.ckpt", f"{out}.ckpt.json"):             # the training state is done
        if os.path.exists(p):
            os.remove(p)
    print(f"saved {name} -> {out} ({res['iters_run']} iterations, train {t_train:.0f} s, "
          f"{rec['timing']['ms_per_iter']:.1f} ms/iter, final loss {res['history'][-1][1]:.4f})")
    return rec


if __name__ == "__main__":
    main()
