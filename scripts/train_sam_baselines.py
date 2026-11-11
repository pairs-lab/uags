"""
train_sam_baselines.py — train the SAM-mask baselines (usplat/sam_lift.py: LangSplat and
OpenGaussian, re-implemented) on a scene's 3D-GS map with the SAME mapping frames, LSeg tokens
and PCA basis as the scene's main field (run_mapping.py --out), from the scene's SAM masks
(scripts/sam_masks.py), and save them next to it as <main>_langsplat_<level>.pt (level s, p, w)
or <main>_opengs.pt, in the format of scripts/distill_features.py (name, identity, code and PCA
basis hashes, provenance) plus the masks' identity.

    scripts/gpu_gate.sh 5000 python3 scripts/train_sam_baselines.py \
        --field /home/thanhnc/gauss_ws/src/uags/results/fields/office_0_lseg.pt \
        --masks /home/thanhnc/gauss_ws/src/uags/results/sam_masks/office_0_lseg_sam.npz \
        --method langsplat --level all

The mapping frames are those of the main field (distill_features.select_frames). LangSplat's
autoencoder is trained once per scene and frame set on every level's mask features and kept at
<main>_langsplat_ae.pt, so all three levels decode with the same one; each level's file holds a
copy. --table <a Table III record (results/nor/tab2_relmax/<split>/<scene>.json)>: train on the
mapping AND update keyframes that scripts/eval_change_nor.py reads for the record (as
scripts/distill_update_nor.py does, for Table IV); outputs <field>_<name>_upd.pt and
<field>_langsplat_ae_upd.pt, and --masks must hold the frames of both parts. An existing
output is kept (use --force to retrain); an interrupted training resumes from <out>.ckpt.
--smoke N trains on the first N frames only (GPU smoke tests); it needs --out-dir.
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
from usplat import sam_lift as SL                                           # noqa: E402

CODE = ("scripts/train_sam_baselines.py", "usplat/sam_lift.py", "usplat/distill.py",
        "scripts/distill_features.py")
CODE_TABLE = ("scripts/distill_update_nor.py", "scripts/dump_nor_qual.py",
              "scripts/eval_change_nor.py", "scripts/time_field.py")
SUFFIX = "_upd"


def code_sha256(table=False):
    repo = os.path.dirname(HERE)
    import run_mapping
    h = hashlib.sha256()
    for rel in CODE + (CODE_TABLE if table else ()) + tuple(run_mapping.MAPPING_CODE):
        with open(os.path.join(repo, rel), "rb") as f:
            h.update(rel.encode() + b"\0" + hashlib.sha256(f.read()).hexdigest().encode() + b"\n")
    return h.hexdigest()


def sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def load_extractor(cfg, device):
    """The mapping run's extractor (tests replace this function)."""
    import time_field
    return time_field.load_extractor(cfg, device)


def frame_index(full, ds, i):
    """The dataset frame index of position i (the key of the masks file)."""
    return int(full.get("index", getattr(ds, "offset", 0) + i * ds.stride))


def record_field(table):
    """The main field of a Table III record (dump_nor_qual.ec_namespace's --field)."""
    import dump_nor_qual as DQ
    from usplat.paths import moved
    return moved(DQ.record_body(json.load(open(table)))["provenance"]["field"]["path"])


def field_frames(args, device):
    """The main field's mapping frames as distill_features.main reads them: [dict(part, index,
    ras, z_fn, h, w)] with z_fn() -> (P, d) the frame's PCA-64 features, and the scene context."""
    import distill_features as DF
    import time_field as TF
    from usplat.clip_dense import upsample_tokens
    from usplat.datasets import make_dataset, downscale
    from usplat.feature_pca import StreamingPCA
    from usplat.gs_io import load_gaussians
    from usplat.paths import moved
    ck = torch.load(args.field, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    pca = StreamingPCA.from_state(ck["pca"], device=device)
    d = int(cfg["feat_dim"])
    splats = load_gaussians(moved(cfg["ply"]), device=device)
    name_, root, opts = DF.mapping_dataset(cfg)
    ds = make_dataset(name_, root, stride=cfg["stride"], **opts)
    idx = DF.select_frames(ds, cfg)
    saved_kf = (ck.get("timing") or {}).get("keyframes")
    if saved_kf is not None and int(saved_kf) != len(idx):
        raise SystemExit(f"{len(idx)} mapping frames here, the main field used {saved_kf}")
    if args.smoke:
        idx = idx[:args.smoke]
    ext = load_extractor(cfg, args.extractor_device or device)
    rasf = TF.rasterizer_factory(args.backend, args.channel_chunk)
    frames = []
    for i in idx:
        full = ds[i]
        tok = ext.extract_tokens(full["image"]).cpu()
        fr = downscale(full, cfg["downscale"])
        h, w = fr["image"].shape[:2]
        viewmat = torch.linalg.inv(fr["c2w"].double()).float().to(device)
        ras = rasf(splats, viewmat, fr["K"].to(device), w, h)

        def z_fn(tok=tok, h=h, w=w):
            return pca.project(upsample_tokens(tok.to(device), (h, w))).reshape(-1, d)
        frames.append(dict(part="map", index=frame_index(full, ds, i), ras=ras, z_fn=z_fn,
                           h=int(h), w=int(w)))
    del ext
    return dict(ckpt=ck, splats=splats, N=int(splats["means"].shape[0]), d=d, frames=frames,
                field=os.path.abspath(args.field), table=None)


def table_frames(args, device):
    """--table: the mapping and the update keyframes of eval_change_nor.load_context for the
    scene's Table III record (as distill_update_nor.training_views)."""
    import dump_nor_qual as DQ
    from usplat.clip_dense import upsample_tokens
    EC = DQ.EC
    args_ns = argparse.Namespace(backend=args.backend, device=device, smoke=0)
    m = DQ.record_body(json.load(open(args.table)))
    ns = DQ.ec_namespace(m, args_ns)
    ctx = EC.load_context(ns)
    d = ctx["d"]
    frames = []
    for part, vs in (("map", ctx["map_views"]), ("update", ctx["upd_views"])):
        vs = vs[:args.smoke] if args.smoke else vs
        for v in vs:
            ras, _ = EC.rasterizer(ctx, v)

            def z_fn(v=v):
                return ctx["pca"].project(upsample_tokens(v["tok"].to(device), (v["h"], v["w"]))).reshape(-1, d)
            frames.append(dict(part=part, index=int(v["index"]), ras=ras, z_fn=z_fn,
                               h=int(v["h"]), w=int(v["w"])))
    return dict(ckpt=ctx["ckpt"], splats=ctx["splats"], N=ctx["N"], d=d, frames=frames,
                field=os.path.abspath(ns.field), table=os.path.abspath(args.table))


def mask_inputs(scene, masks, device):
    """Per frame and level: the compact mask ids (L, P) int16 (cpu) and the unit mask features
    zbar (M, d); the frames' features are dropped after this."""
    out = []
    for f in scene["frames"]:
        key = (f["part"], f["index"])
        if key not in masks["labels"]:
            raise SystemExit(f"the masks lack frame {f['part']}:{f['index']} ({masks['files']})")
        lab = masks["labels"][key]
        if tuple(lab.shape[1:]) != (f["h"], f["w"]):
            raise SystemExit(f"frame {key}: masks of {tuple(lab.shape[1:])}, the field's grid is "
                             f"{(f['h'], f['w'])}")
        z = f["z_fn"]()
        comp, means = [], []
        for lv in range(len(SL.LEVELS)):
            c, mu = SL.mask_means(torch.from_numpy(lab[lv].reshape(-1)).to(device), z)
            comp.append(c.to(torch.int16).cpu())
            means.append(mu)
        out.append(dict(ras=f["ras"], lab=torch.stack(comp), means=means))
        f["z_fn"] = None
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--field", help="the scene's main field (run_mapping --out)")
    src.add_argument("--table", help="a NeRF-Object-Removal Table III record: mapping + update")
    ap.add_argument("--masks", required=True, help="scripts/sam_masks.py file(s), comma-separated")
    ap.add_argument("--method", choices=SL.METHODS, required=True)
    ap.add_argument("--level", choices=SL.LANGSPLAT_LEVELS + ("all",), default="all",
                    help="LangSplat's SAM level (s, p, w) or all three; OpenGaussian: all (it "
                         "uses the default level)")
    ap.add_argument("--iters", type=int, default=30000, help="at most (the papers' count)")
    ap.add_argument("--min-iters", type=int, default=5000)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--log-every", type=int, default=1000, help="loss log points (stopping rule)")
    ap.add_argument("--lr", type=float, default=None,
                    help=f"Adam lr (default: LangSplat {SL.LANGSPLAT_LR}, OpenGaussian {SL.OPENGS_LR})")
    ap.add_argument("--alpha-min", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ae-epochs", type=int, default=SL.AE_EPOCHS)
    ap.add_argument("--ae-lr", type=float, default=SL.AE_LR)
    ap.add_argument("--ae-batch", type=int, default=SL.AE_BATCH)
    ap.add_argument("--smooth-weight", type=float, default=SL.OPENGS_SMOOTH)
    ap.add_argument("--min-mask-px", type=int, default=SL.OPENGS_MIN_PX)
    ap.add_argument("--k1", type=int, default=SL.OPENGS_K1)
    ap.add_argument("--k2", type=int, default=SL.OPENGS_K2)
    ap.add_argument("--pos-weight", type=float, default=SL.OPENGS_POS_W)
    ap.add_argument("--kmeans-iters", type=int, default=SL.KMEANS_ITERS)
    ap.add_argument("--share-min", type=float, default=SL.SHARE_MIN)
    ap.add_argument("--min-score", type=float, default=SL.MIN_SCORE)
    ap.add_argument("--min-views", type=int, default=SL.MIN_VIEWS,
                    help="OpenGaussian: an instance matched in fewer views gets no feature")
    ap.add_argument("--switch", type=int, default=SL.OPENGS_SWITCH,
                    help="OpenGaussian: iteration of its separation-weight switch")
    ap.add_argument("--inst-chunk", type=int, default=64, help="instances per association render")
    ap.add_argument("--channel-chunk", type=int, default=32)
    ap.add_argument("--backend", choices=["gsplat", "mock"], default="gsplat")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--extractor-device", default="",
                    help="device of the extractor (default: --device; cpu for small smoke gates)")
    ap.add_argument("--out-dir", default="", help="default: next to the field")
    ap.add_argument("--smoke", type=int, default=0, help="first N frames only (needs --out-dir)")
    ap.add_argument("--force", action="store_true", help="retrain an existing output")
    return ap


def _conf(args, name, level, frames):
    c = dict(method=args.method, iters=args.iters, min_iters=args.min_iters, tol=args.tol,
             log_every=args.log_every,
             lr=args.lr, alpha_min=args.alpha_min, seed=args.seed, frames=frames,
             smoke=args.smoke or None)
    if args.extractor_device:                     # other numerics: another identity
        c["extractor_device"] = args.extractor_device
    if args.method == "langsplat":
        c.update(level=level, ae_dims=list(args.ae_dims), ae_epochs=args.ae_epochs,
                 ae_lr=args.ae_lr, ae_batch=args.ae_batch, ae_keep=SL.AE_KEEP)
    else:
        c.update(level=SL.OPENGS_LEVEL, dim=SL.OPENGS_DIM, smooth_weight=args.smooth_weight,
                 min_views=args.min_views, switch=args.switch,
                 min_mask_px=args.min_mask_px, k1=args.k1, k2=args.k2, pos_weight=args.pos_weight,
                 kmeans_iters=args.kmeans_iters, share_min=args.share_min,
                 min_score=args.min_score)
    return c


def _save(path, rec):
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(rec, tmp)
    os.replace(tmp, path)


def autoencoder(args, inputs, stem, base, device, log=print):
    """The scene's LangSplat autoencoder: read from <stem>_langsplat_ae<suffix>.pt if it was
    trained on the same masks, frames, settings and code, else trained and written there."""
    path = D.out_path(stem, "langsplat_ae" + base["suffix"])
    ident = SL.settings_identity(dict(code=base["code_sha256"], masks=base["masks"]["sha256"],
                                      frames=base["frames"], basis=base["pca_basis_sha256"],
                                      dims=list(args.ae_dims), epochs=args.ae_epochs,
                                      lr=args.ae_lr, batch=args.ae_batch, keep=SL.AE_KEEP,
                                      seed=args.seed))
    if os.path.exists(path):
        st = torch.load(path, map_location="cpu", weights_only=False)
        if st.get("identity") == ident:
            ae = SL.LangSplatAE(tuple(st["ae_dims"]))
            ae.load_state_dict(st["ae"])
            log(f"langsplat: autoencoder read from {path}")
            return ae.eval(), st, path, 0.0
        log(f"langsplat: {path} was trained on other inputs; retraining it")
    x = torch.cat([mu for v in inputs for mu in v["means"]])
    t0 = time.time()
    ae, hist = SL.train_autoencoder(x, dims=args.ae_dims, epochs=args.ae_epochs, lr=args.ae_lr,
                                    batch=args.ae_batch, seed=args.seed, device=device, log=log)
    t_ae = time.time() - t0
    st = dict(family=SL.FAMILY + "_ae", ae=ae.state_dict(), ae_dims=list(args.ae_dims),
              history=hist, identity=ident, n_features=int(x.shape[0]),
              per_level=[int(sum(v["means"][lv].shape[0] for v in inputs))
                         for lv in range(len(SL.LEVELS))],
              masks=base["masks"], pca_basis_sha256=base["pca_basis_sha256"],
              main_field=base["main_field"], train_s=t_ae)
    _save(path, st)
    log(f"langsplat: autoencoder on {x.shape[0]} mask features ({st['per_level']} per level), "
        f"eval loss {hist[-1][1]:.5f}, {t_ae:.0f} s -> {path}")
    return ae, st, path, t_ae


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.smoke and not args.out_dir:
        raise SystemExit("--smoke needs --out-dir (a smoke run never writes next to the field)")
    if args.backend == "gsplat" and not str(args.device).startswith("cuda"):
        raise SystemExit("--backend gsplat needs --device cuda (use --backend mock on the CPU)")
    if args.lr is None:
        args.lr = SL.LANGSPLAT_LR if args.method == "langsplat" else SL.OPENGS_LR
    field = os.path.abspath(record_field(args.table) if args.table else args.field)
    suffix = SUFFIX if args.table else ""
    stem = field
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        stem = os.path.join(os.path.abspath(args.out_dir), os.path.basename(field))
    if args.method == "langsplat":
        levels = list(SL.LANGSPLAT_LEVELS) if args.level == "all" else [args.level]
    elif args.level != "all":
        raise SystemExit("OpenGaussian trains on the default level only (--level all)")
    else:
        levels = [SL.OPENGS_LEVEL]
    name_of = lambda lv: SL.variant_name(args.method, lv if args.method == "langsplat" else None)
    todo = [lv for lv in levels if args.force or not os.path.exists(D.out_path(stem, name_of(lv) + suffix))]
    for lv in levels:
        if lv not in todo:
            print(f"{name_of(lv)}{suffix}: exists, kept (--force retrains)")
    if not todo:
        return []
    dev = args.device
    torch.manual_seed(args.seed)
    held = 0
    if str(dev).startswith("cuda"):
        from usplat import gpu_reserve                       # fraction from scripts/gpu_gate.sh
        held = gpu_reserve.apply(dev)
    cuda = str(dev).startswith("cuda") and torch.cuda.is_available()
    sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)
    t0 = time.time()
    masks = SL.load_masks(args.masks)
    scene = table_frames(args, dev) if args.table else field_frames(args, dev)
    table = scene["table"] is not None
    if scene["field"] != field:
        raise SystemExit(f"the record's field {scene['field']} != {field}")
    args.ae_dims = (scene["d"],) + tuple(SL.AE_DIMS[1:])        # LangSplat's AE on d-dim features
    inputs = mask_inputs(scene, masks, dev)
    sync()
    t_ext = time.time() - t0
    N, ck = scene["N"], scene["ckpt"]
    code = code_sha256(table)
    frames = [(f["part"], f["index"]) for f in scene["frames"]]
    base = dict(family=SL.FAMILY, method=args.method, code_sha256=code, main_field=scene["field"],
                pca_basis_sha256=D.basis_sha256(ck["pca"]), n_frames=len(frames),
                frames=[f"{p}:{i}" for p, i in frames],
                n_map_frames=sum(p == "map" for p, _ in frames),
                n_update_frames=sum(p == "update" for p, _ in frames), suffix=suffix,
                masks=dict(files=masks["files"], identity=masks["identity"],
                           sha256=[sha256_file(p) for p in masks["files"]],
                           settings=masks["metas"][0].get("settings")),
                table=scene["table"],
                table_sha256=sha256_file(scene["table"]) if table else None)
    frames_kind = "mapping+update" if table else "mapping"
    print(f"{args.method}: {N} Gaussians, {len(frames)} frames ({base['n_map_frames']} mapping, "
          f"{base['n_update_frames']} update), field {scene['field']}")
    saved = []

    def finish(name, level, res, extra, t_train, t_post):
        conf = _conf(args, name, level, frames_kind)
        ident = hashlib.sha256(json.dumps(dict(code=code, masks=masks["identity"], **conf),
                                          sort_keys=True).encode()).hexdigest()
        rec = dict({k: v for k, v in base.items() if k != "suffix"}, name=name + suffix,
                   level=level, history=res["history"], iters_run=res["iters_run"], config=conf,
                   identity=ident,
                   timing=dict(extract_s=t_ext, train_s=t_train, post_s=t_post,
                               iters=res["iters_run"],
                               ms_per_iter=1000 * t_train / max(res["iters_run"], 1),
                               peak_alloc_mib=(torch.cuda.max_memory_allocated() / 2 ** 20
                                               if cuda else None),
                               gpu_reserved_mib=held), **extra)
        out = D.out_path(stem, name + suffix)
        _save(out, rec)
        if os.path.exists(f"{out}.ckpt"):
            os.remove(f"{out}.ckpt")               # the training state is no longer needed
        print(f"saved {rec['name']} -> {out} ({res['iters_run']} iterations, train {t_train:.0f} s, "
              f"{rec['timing']['ms_per_iter']:.1f} ms/iter, final loss {res['history'][-1][1]:.5f})")
        saved.append(rec)

    if args.method == "langsplat":
        ae, ae_st, ae_path, _ = autoencoder(args, inputs, stem, base, dev)
        for lv in todo:
            li = SL.LEVELS.index(lv)
            name = SL.variant_name("langsplat", lv)
            out = D.out_path(stem, name + suffix)
            views = [(v["ras"], dict(lab=v["lab"][li].to(dev),
                                     tgt=SL.encode_means(ae, v["means"][li], dev)))
                     for v in inputs]
            sync()
            t1 = time.time()
            res = SL.train_langsplat(views, N, iters=args.iters, lr=args.lr,
                                     alpha_min=args.alpha_min, seed=args.seed, device=dev,
                                     min_iters=args.min_iters, tol=args.tol,
                                     log_every=args.log_every, ckpt_path=f"{out}.ckpt",
                                     name=name + suffix)
            sync()
            finish(name, lv, res, dict(
                features=res["features"].cpu(),
                ae={k: x.cpu() for k, x in ae.state_dict().items()}, ae_dims=list(args.ae_dims),
                ae_identity=ae_st["identity"], ae_history=ae_st["history"],
                ae_path=os.path.abspath(ae_path),
                n_masks=int(sum(v["means"][li].shape[0] for v in inputs))),
                time.time() - t1, 0.0)
    else:
        name = SL.variant_name("opengs")
        out = D.out_path(stem, name + suffix)
        views = [(v["ras"], dict(lab=v["lab"][SL.DEFAULT].to(dev))) for v in inputs]
        sync()
        t1 = time.time()
        res = SL.train_opengs(views, N, iters=args.iters, lr=args.lr,
                              smooth_weight=args.smooth_weight, min_px=args.min_mask_px,
                              switch=args.switch,
                              alpha_min=args.alpha_min, seed=args.seed, device=dev,
                              min_iters=args.min_iters, tol=args.tol, log_every=args.log_every,
                              ckpt_path=f"{out}.ckpt", name=name + suffix)
        sync()
        t_train = time.time() - t1
        t1 = time.time()
        feat = res["features"]
        cb = SL.two_level_codebook(feat, scene["splats"]["means"].detach(), args.k1, args.k2,
                                   args.pos_weight, args.kmeans_iters, args.seed)
        assoc = SL.associate([(v["ras"], dict(lab=v["lab"][SL.DEFAULT].to(dev),
                                              zbar=v["means"][SL.DEFAULT])) for v in inputs],
                             cb["instance"], cb["n_instances"], cb["codes"], feat,
                             alpha_min=args.alpha_min, share_min=args.share_min,
                             min_score=args.min_score, min_views=args.min_views,
                             chunk=args.inst_chunk)
        sync()
        t_post = time.time() - t1
        used = torch.bincount(cb["instance"], minlength=cb["n_instances"]) > 0
        matched = assoc["n_matched"] > 0
        featured = assoc["features"].norm(dim=-1) > 0          # matched in >= min_views views
        n_g = int(featured[cb["instance"]].sum())
        print(f"{name}: {int(used.sum())} of {cb['n_instances']} instances hold Gaussians, "
              f"{int((used & matched).sum())} matched, {int((used & featured).sum())} in >= "
              f"{args.min_views} views; {n_g} of {N} Gaussians get a feature")
        finish(name, SL.OPENGS_LEVEL, res, dict(
            ins_feat=feat.cpu(), instance=cb["instance"].to(torch.int32).cpu(),
            coarse=cb["coarse"].to(torch.int32).cpu(), codes=cb["codes"].cpu(),
            instance_features=assoc["features"].cpu(), match_score=assoc["score"].cpu(),
            n_matched=assoc["n_matched"].cpu(), n_instances=cb["n_instances"],
            association=dict(n_used=int(used.sum()), n_matched=int((used & matched).sum()),
                             n_with_feature=int((used & featured).sum()),
                             n_gaussians_with_feature=n_g)), t_train, t_post)
    return saved


if __name__ == "__main__":
    main()
