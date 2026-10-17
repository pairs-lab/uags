"""
run_mapping.py — orchestration of the uncertainty-aware semantic mapping loop.

Pipeline per keyframe:
    1. dense CLIP features (token grid at full resolution, upsampled to the accumulation
       resolution), projected to the PCA subspace       z_obs : (P, d)
    2. one render + one backward through the feature rasterizer gives the per-view sums;
       the field turns them into ONE view observation per Gaussian (F1), runs the change
       test on the centred channel (F3, F4) and updates the posterior
    3. read posterior, uncertainty, change flags, robust set

Backends:
    --backend mock   : pure-CPU, MockFeatureRasterizer + MockDenseCLIP, with a scene change
    --backend gsplat : real (GPU + a 3DGS checkpoint + a posed RGB(-D) sequence)

Change handling needs two sequences: the MAPPING sequence (--data) is run with the change
test recording scores only (change_thr = inf); the field then calibrates its own threshold
(the q-quantile of its mapping-pass scores, --change-quantile), and the UPDATE sequence
(--update-data, same scene after a change) runs with that threshold. A dataset with parts
(nor) gives both from one scene folder: --part map --update-part update.

The PCA basis and the centre zbar are fit on the first --pca-frames MAPPING frames only
(no text, no labels). Text queries are projected without renormalisation (F7).
"""
from __future__ import annotations
import argparse
import json
import os
import math
import torch

from usplat.vmf_field import DirectionalSemanticField, FieldConfig
from usplat.efe_planner import EFEPlanner
from usplat.evaluation import sparsification_curve, semantic_error_vs_text, auroc


def run_mock(num_gaussians=200, feat_dim=32, num_frames=16, num_protos=6, seed=0,
             field_config="", change_quantile=0.999):
    torch.manual_seed(seed)
    from usplat.mock_rasterizer import MockFeatureRasterizer
    from usplat.clip_dense import MockDenseCLIP

    clip = MockDenseCLIP(feat_dim=feat_dim, num_protos=num_protos, noise=0.12, seed=seed)
    cfg = FieldConfig(**json.loads(field_config)) if field_config else FieldConfig()
    cfg.change_thr = float("inf")                       # mapping pass: record only
    field = DirectionalSemanticField(num_gaussians, feat_dim, cfg)

    means2d = torch.rand(num_gaussians, 2) * 64
    sigmas = 1.0 + torch.rand(num_gaussians) * 2.0
    opacities = 0.3 + 0.6 * torch.rand(num_gaussians)
    depths = torch.rand(num_gaussians)
    # hidden "true" prototype, spatially coherent; 20% change identity half-way through
    true_proto = (means2d[:, 0] / 64.0 * num_protos).long().clamp(0, num_protos - 1)
    dynamic = torch.rand(num_gaussians) < 0.2
    new_proto = true_proto.clone()
    new_proto[dynamic] = (new_proto[dynamic] + 1) % num_protos

    H = W = 48
    pix = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy"), -1)
    ras = MockFeatureRasterizer(means2d, sigmas, opacities, depths, pix.reshape(-1, 2).float())
    Wm = ras.weights()
    dom = Wm.argmax(dim=1)                              # dominant Gaussian per pixel
    half = num_frames // 2
    # centre = mean feature of the mapping frames (here: the pre-change content)
    field.set_centre(clip.extract_from_labels(true_proto[dom].reshape(H, W)).reshape(-1, feat_dim).mean(0))
    for t in range(num_frames):
        if t == half:                                   # end of the mapping sequence
            thr = field.calibrate_change_threshold(change_quantile)
            print(f"calibrated change threshold (q={change_quantile}): {thr:.3f}")
        proto_t = new_proto if t >= half else true_proto
        z_obs = clip.extract_from_labels(proto_t[dom].reshape(H, W)).reshape(-1, feat_dim)
        field.update_view(ras, z_obs)

    # observable change: share of each Gaussian's splat weight on pixels that changed
    pix_changed = (true_proto[dom] != new_proto[dom]).float()
    frac = (Wm.t() @ pix_changed) / Wm.sum(0).clamp_min(1e-12)
    well = field.pixel_mass / num_frames >= field.cfg.view_mass_frac * H * W
    changed_gt, static_gt = (frac > 0.5) & well, (frac < 0.1) & well
    epi = field.epistemic()
    ale, valid = field.aleatoric()
    print(f"gaussians: {num_gaussians} | views: {num_frames} | mean support "
          f"{field.support('raw').mean():.1f} | well-observed {int(well.sum())}")
    print(f"mean epistemic {epi.mean():.3f} | mean aleatoric (valid) {ale[valid].mean():.3f}")
    fired = field.n_resets >= 1
    sel = changed_gt | static_gt
    print(f"change test: AUROC(c_max) {auroc(field.c_max[sel], changed_gt[sel].float()):.3f} | "
          f"flagged {fired[changed_gt].float().mean() * 100:.0f}% of changed, "
          f"{fired[static_gt].float().mean() * 100:.0f}% of unchanged "
          f"(n={int(changed_gt.sum())}/{int(static_gt.sum())})")
    print(f"robust set: {int(field.robust_mask().sum())} Gaussians")

    V = 5
    visibility = torch.rand(V, num_gaussians)          # mock expected blending mass
    efe, best = EFEPlanner(field, lam=1.0).score(visibility, query_embed=clip.text_embedding(0))
    print(f"EFE per candidate view: {[round(x, 3) for x in efe.tolist()]} | best=v{best}")

    gt_dirs = clip.protos[new_proto]                    # current content
    err = semantic_error_vs_text(field.mu_post(), gt_dirs)
    ok = valid & well
    out = sparsification_curve(err[ok], ale[ok], num_bins=10)
    print(f"AUSE(aleatoric -> semantic error): {out['ause']:.4f} "
          f"(random gap {out['random_gap']:.4f}; lower = better)")
    print("done.")


def css_vocabulary(args):
    """CSS is closed-set: it receives, at mapping time, the classes present in the scene's
    evaluation frames (the same rule as the queries of scripts/eval_seg.py), in vocabulary
    order, with the extractor's frozen templates."""
    from usplat.datasets import make_dataset
    v = json.load(open(args.vocab))
    ignore = set(int(i) for i in (v.get("ignore") or v.get("ignore_ids") or []))
    order = [int(k) for k in v["classes"]]
    ds = make_dataset(args.dataset, args.css_eval_data, stride=args.css_eval_stride)
    present = set()
    for i in range(len(ds)):
        present |= set(torch.unique(ds[i]["labels"]).tolist())
    ids = [c for c in order if c in present and c not in ignore]
    prompts = [v["classes"][str(c)].get("prompt", v["classes"][str(c)]["name"]) for c in ids]
    tpl = (v.get("templates_by_extractor") or {}).get(args.extractor) \
        or v.get("templates") or [v.get("template", "a photo of a {}.")]
    return ids, prompts, tpl


# files whose content decides a field (mapping + extractor + field code): their hash must be
# equal for the validation fields (temperatures) and the test fields (checked by eval_seg)
MAPPING_CODE = ("scripts/run_mapping.py", "usplat/vmf_field.py", "usplat/vmf_math.py",
                "usplat/bki_accumulate.py", "usplat/feature_pca.py", "usplat/gsplat_adapter.py",
                "usplat/baselines.py", "usplat/datasets.py", "usplat/clip_dense.py",
                "usplat/gs_io.py", "usplat/extractors/__init__.py", "usplat/extractors/lseg.py")


def mapping_code_state():
    """{mapping_code_sha256, files: {path: sha256}, git: commit/dirty (usplat.runlog)}."""
    import hashlib
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = {}
    for rel in MAPPING_CODE:
        with open(os.path.join(repo, rel), "rb") as f:
            files[rel] = hashlib.sha256(f.read()).hexdigest()
    total = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in sorted(files.items())).encode())
    try:
        from usplat.runlog import git_state
        g = git_state(repo)
        git = {k: g.get(k) for k in ("commit", "dirty", "code_sha256")}
    except Exception as e:                                  # provenance only
        git = {"error": str(e)}
    return dict(mapping_code_sha256=total.hexdigest(), files=files, git=git)


def _part(part):
    """make_dataset options for a sequence part ({} when none is given)."""
    return {"part": part} if part else {}


def _pose_gap(c2w_a, c2w_b):
    dt = (c2w_a[:3, 3] - c2w_b[:3, 3]).norm().item()
    cos = ((c2w_a[:3, :3].t() @ c2w_b[:3, :3]).trace().item() - 1.0) / 2.0
    return dt, math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def run_gsplat(args):
    """Real pipeline: pre-trained 3DGS checkpoint + posed RGB(-D) stream + dense CLIP."""
    import os
    from tqdm import tqdm
    from usplat.gs_io import load_gaussians
    from usplat.datasets import make_dataset, downscale
    from usplat.clip_dense import upsample_tokens
    from usplat.feature_pca import StreamingPCA, RunningMean
    from usplat.gsplat_adapter import GSplatFeatureRasterizer
    from usplat.extractors import get_extractor
    from usplat import baselines as B

    device = "cuda"
    torch.manual_seed(args.seed)                            # PCA pixel subsample (randperm)
    from usplat import gpu_reserve                          # fraction from scripts/gpu_gate.sh
    held = gpu_reserve.apply(device)
    if held:
        print(f"reserved {held} MiB of GPU memory")
    splats = load_gaussians(args.ply, device=device)
    N = splats["means"].shape[0]
    print(f"loaded {N} Gaussians from {args.ply}")
    ds = make_dataset(args.dataset, args.data, stride=args.stride, **_part(args.part))
    print(f"dataset {args.dataset}: {len(ds)} frames (stride {args.stride})")
    kw = (dict(model_name=args.clip_model, pretrained=args.clip_pretrained)
          if args.extractor == "clearclip" else {})
    clip = get_extractor(args.extractor, device=device, **kw)
    ext_opts = {k: getattr(clip, k) for k in ("token_stride", "flip", "scales", "half",
                                                "base_size", "crop_size") if hasattr(clip, k)}
    print(f"extractor {args.extractor}: {clip.d}-d features {ext_opts}")

    def feats(frame_full, hw):
        tok = clip.extract_tokens(frame_full["image"])          # full-resolution tokens
        return upsample_tokens(tok, hw)

    # ---- PCA basis + centre from the first mapping frames (no text, no labels) ----
    # --pca-from <field.pt>: reuse another field's basis and centre (the k-view and ablation
    # fields of a scene must share the main field's basis: its text projection scores them)
    if args.pca_from:
        src = torch.load(args.pca_from, map_location="cpu", weights_only=False)
        pca = StreamingPCA.from_state(src["pca"], device=device)
        if pca.basis.shape != (clip.d, args.feat_dim):
            raise ValueError(f"--pca-from basis {tuple(pca.basis.shape)} != ({clip.d}, {args.feat_dim})")
        zbar = src["field"]["zbar"].to(device)
        explained = src.get("pca_explained")
        del src
        print(f"PCA {clip.d} -> {args.feat_dim} dims from {args.pca_from}; ||zbar|| = {zbar.norm():.3f}")
    else:
        pca = StreamingPCA(clip.d, args.feat_dim, device=device)
        raw = []
        for i in range(min(args.pca_frames, len(ds))):
            fr = ds[i]
            h, w = downscale(fr, args.downscale)["image"].shape[:2]
            z = feats(fr, (h, w))
            pca.accumulate(z)
            # a strided view would keep the whole (h*w, D) frame alive: copy the sample
            raw.append(z.reshape(-1, clip.d)[:: max(1, (h * w) // 5000)].clone())
            del z
        explained = pca.fit()
        centre = RunningMean(args.feat_dim, device=device)
        for zs in raw:
            centre.add(pca.project(zs))
        del raw
        zbar = centre.mean()
        print(f"PCA {clip.d} -> {args.feat_dim} dims (explained second moment {explained:.3f}); "
              f"||zbar|| = {zbar.norm():.3f}")

    import time
    times, field_times, ext_times = [], [], []

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def select(seq, max_keyframes=None):
        """Frame indices the sequence feeds: evenly spaced subset, finite poses, and no
        near-duplicate of the last kept frame (pose gap below kf_trans AND kf_rot)."""
        last, keep = None, []
        order = range(len(seq))
        if max_keyframes and max_keyframes < len(seq):             # evenly spaced subset
            order = sorted(set(torch.linspace(0, len(seq) - 1, max_keyframes).round().long().tolist()))
        for i in order:
            c2w = seq[i]["c2w"]
            if not torch.isfinite(c2w).all():
                continue
            if last is not None:                                # drop near-duplicates
                dt, dr = _pose_gap(c2w, last)
                if dt < args.kf_trans and dr < args.kf_rot:
                    continue
            last = c2w
            keep.append(i)
        return keep

    def extract(seq, idx, desc):
        """Full-resolution extractor tokens of the kept frames, stored on the CPU (fp32, the
        values the field would get), so the extractor can leave the GPU before mapping."""
        out = []
        for i in tqdm(idx, desc=f"extract {desc}"):
            fr_full = seq[i]
            sync()
            t_e = time.perf_counter()
            out.append(clip.extract_tokens(fr_full["image"]).cpu())
            sync()
            ext_times.append(time.perf_counter() - t_e)             # extractor only
        return out

    # Extract every token grid first, then free the extractor: its activations (LSeg fp32:
    # > 3.3 GB at 640x480) and the field + baselines + renders no longer peak together.
    map_idx = select(ds, args.max_keyframes)
    map_tok = extract(ds, map_idx, "mapping")
    upd, upd_idx, upd_tok = None, [], []
    if args.update_data:
        upd = make_dataset(args.dataset, args.update_data, stride=args.update_stride,
                           **_part(args.update_part))
        upd_idx = select(upd)
        upd_tok = extract(upd, upd_idx, "update")
    query_t = None
    if args.query:
        query_t = clip.text_embedding([q.strip() for q in args.query.split(",")])
    css_ids, css_text = None, None
    if args.baselines and args.vocab and args.css_eval_data:
        css_ids, prompts, tpl = css_vocabulary(args)
        css_text = pca.project_text(clip.text_embedding(prompts, templates=tuple(tpl)))
    del clip                                                # its blocks stay in our cache
    peak_extract = (torch.cuda.max_memory_allocated() / 2 ** 20
                    if torch.cuda.is_available() else None)

    cfg = FieldConfig(**json.loads(args.field_config)) \
        if args.field_config else FieldConfig()
    if not args.use_depth and cfg.readmit_depth is not None:
        print("no --use-depth: re-admission after a change uses views only")
        cfg.readmit_depth = None
    cfg.change_thr = float("inf")                               # mapping pass: record only
    field = DirectionalSemanticField(N, args.feat_dim, cfg, device=device)
    field.set_centre(zbar)

    # ---- baselines fed by the same per-view sums (plan K11) --------------------------
    bs = []
    if args.baselines:
        text = None
        if css_ids is not None:
            text = css_text
            print(f"CSS vocabulary: {len(css_ids)} classes present in {args.css_eval_data}")
        bs = B.default_baselines(field, text=text, class_ids=css_ids)
    css = [b for b in bs if isinstance(b, B.CSS)]

    def run_sequence(seq, idx, toks, desc):
        used = 0
        for i, tok in tqdm(list(zip(idx, toks)), desc=desc):
            fr_full = seq[i]
            c2w = fr_full["c2w"]
            fr = downscale(fr_full, args.downscale)
            h, w = fr["image"].shape[:2]
            z = pca.project(upsample_tokens(tok.to(device), (h, w))).reshape(-1, args.feat_dim)
            viewmat = torch.linalg.inv(c2w.double()).float().to(device)
            ras = GSplatFeatureRasterizer.from_splats(splats, viewmat, fr["K"].to(device),
                                                      w, h, channel_chunk=args.channel_chunk)
            kw = {}
            if args.use_depth and fr.get("depth") is not None:
                kw = dict(depth_obs=fr["depth"].reshape(-1).to(device),
                          gauss_depth=ras.gaussian_depths())
            sync()
            t0 = time.perf_counter()
            field.update_view(ras, z, num_pixels=h * w, **kw)   # + hooked baselines
            sync()
            field_times.append(time.perf_counter() - t0)
            for c in css:                                       # CSS: its own one-hot pass
                c.update_view(ras, z)
            sync()
            times.append(time.perf_counter() - t0)              # field + baselines update
            used += 1
        return used

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    used = run_sequence(ds, map_idx, map_tok, "mapping")
    del map_tok
    if args.update_data:
        thr = field.calibrate_change_threshold(args.change_quantile)
        print(f"calibrated change threshold (q={args.change_quantile}): {thr:.4f}")
        used += run_sequence(upd, upd_idx, upd_tok, "update")

    # ---- readout ------------------------------------------------------------------
    epi = field.epistemic()
    ale, valid = field.aleatoric()
    print(f"keyframes used {used} | observed Gaussians {(field.pixel_mass > 0).float().mean() * 100:.1f}% "
          f"| mean support {field.support('raw')[field.pixel_mass > 0].mean():.2f} views")
    print(f"epistemic mean {epi.mean():.3f} | aleatoric mean (valid, {valid.float().mean() * 100:.1f}%) "
          f"{ale[valid].mean():.3f} | changed {int(field.changed.sum())} | robust "
          f"{int(field.robust_mask().sum())}/{N}")

    if args.query:
        prompts = [q.strip() for q in args.query.split(",")]
        t = pca.project_text(query_t)                           # no renormalisation (F7)
        sims = field.query(t)
        for k, p in enumerate(prompts):
            s = sims[valid, k]
            print(f"query '{p}': max {s.max():.3f} | mean {s.mean():.3f}")

    ms = lambda v: 1000 * sum(v) / max(len(v), 1)
    timing = dict(keyframes=used, ms_per_keyframe_update=ms(times),
                  ms_per_keyframe_field=ms(field_times),      # field + view-hook baselines
                  ms_per_keyframe_extract=ms(ext_times),      # extractor (tokens) only
                  n_css=len(css), n_baselines=len(bs),
                  peak_alloc_extract_mib=peak_extract,        # extractor phase (PCA + tokens)
                  gpu=(torch.cuda.get_device_name() if torch.cuda.is_available() else None),
                  peak_alloc_mib=(torch.cuda.max_memory_allocated() / 2 ** 20
                                  if torch.cuda.is_available() else None),
                  peak_reserved_mib=(torch.cuda.max_memory_reserved() / 2 ** 20
                                     if torch.cuda.is_available() else None))
    print(f"timing: {timing}")
    out = dict(field=field.state_dict(), pca=pca.state_dict(), config=vars(args), timing=timing,
               field_config=field.cfg.to_dict(), extractor=args.extractor,
               extractor_options=ext_opts, pca_explained=explained,
               code=mapping_code_state(), gpu_reserved_mib=held,
               baselines=B.baselines_state(bs) if bs else [], css_class_ids=css_ids)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = f"{args.out}.tmp{os.getpid()}"                      # atomic: never a truncated field
    torch.save(out, tmp)
    os.replace(tmp, args.out)
    print(f"saved field -> {args.out} (mapping code {out['code']['mapping_code_sha256'][:12]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "gsplat"], default="mock")
    ap.add_argument("--ply", help="pre-trained 3DGS checkpoint (.ply or .pt)")
    ap.add_argument("--data", help="dataset root")
    ap.add_argument("--dataset", default="replica",
                    choices=["replica", "replica_semnerf", "scannet", "7scenes", "nor"])
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--part", default="",
                    help="part of --data, for datasets with parts (nor: map | update | all)")
    ap.add_argument("--downscale", type=int, default=4,
                    help="accumulation resolution = image / downscale (CLIP runs at full res)")
    ap.add_argument("--extractor", default="lseg", help="usplat.extractors name")
    ap.add_argument("--baselines", action="store_true",
                    help="also accumulate the K11 baselines (NormLift/averaging, view count, CSS)")
    ap.add_argument("--vocab", default="", help="frozen vocabulary (CSS classes + templates)")
    ap.add_argument("--css-eval-data", default="",
                    help="evaluation sequence whose present classes form the CSS vocabulary")
    ap.add_argument("--css-eval-stride", type=int, default=10)
    ap.add_argument("--clip-model", default="ViT-B-16")
    ap.add_argument("--clip-pretrained", default="laion2b_s34b_b88k")
    ap.add_argument("--feat-dim", type=int, default=64,
                    help="PCA-compressed semantic dimension on the splats")
    ap.add_argument("--pca-frames", type=int, default=10)
    ap.add_argument("--pca-from", default="",
                    help="field .pt whose PCA basis and centre to reuse (no PCA fit)")
    ap.add_argument("--seed", type=int, default=0, help="torch seed (PCA pixel subsample)")
    ap.add_argument("--field-config", default="",
                    help='JSON overrides of FieldConfig, e.g. \'{"change_thr": 0.4}\'')
    ap.add_argument("--kf-trans", type=float, default=0.05, help="keyframe gap [m]")
    ap.add_argument("--kf-rot", type=float, default=5.0, help="keyframe gap [deg]")
    ap.add_argument("--max-keyframes", type=int, default=0,
                    help="use this many evenly spaced mapping frames (0 = all)")
    ap.add_argument("--update-data", default="",
                    help="root of the UPDATE sequence (same scene after a change)")
    ap.add_argument("--update-stride", type=int, default=5)
    ap.add_argument("--update-part", default="",
                    help="part of --update-data (nor: --data and --update-data are the same "
                         "scene, --part map --update-part update)")
    ap.add_argument("--change-quantile", type=float, default=0.999,
                    help="q of the mapping-pass null used to set the change threshold")
    ap.add_argument("--use-depth", action="store_true",
                    help="feed the dataset depth as the observed depth (re-admission check)")
    ap.add_argument("--channel-chunk", type=int, default=32)
    ap.add_argument("--query", default="", help="comma-separated open-vocab text queries")
    ap.add_argument("--out", default="outputs/field.pt")
    args = ap.parse_args()
    if args.backend == "mock":
        run_mock(field_config=args.field_config, change_quantile=args.change_quantile)
    else:
        if not (args.ply and args.data):
            raise SystemExit("--backend gsplat requires --ply and --data")
        run_gsplat(args)


if __name__ == "__main__":
    main()
