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
(nor) gives both from one scene folder: --part map --update-part update; --align <json> moves
the update part's poses onto the map part (usplat/nor_align.py, protocol D3).

The PCA basis and the centre zbar are fit on the first --pca-frames MAPPING frames only
(no text, no labels). Text queries are projected without renormalisation (F7).

Static false-alarm run (--summary-json PATH, a usplat.runlog record): on a static scene every
fire of the update pass is a false alarm. The record holds, for the UPDATE pass, the defined
tests and fires of every view and in total, the realised per-view false-alarm rate
fires / tests overall and per history bucket (the threshold targets 1 - q), the share of the
observed pixel mass that was reset, the calibrated thresholds and the config (ChangeTally).
--change-baselines runs the residual baselines of usplat.change_baselines (NCC, feature
residual, depth residuals where depth exists) on the same views, with the same label-free
rule (ResidualNull: the q-quantile of their own mapping-pass scores, pooled), counted the same
way. --fixed-change-thr replaces the calibration by one fixed threshold (ablation).
--no-save-field writes no field file, so an existing field at --out is never overwritten.

Labelled change evaluation (scripts/eval_change.py) reads, from a field saved after an update
pass: change_scores = {"<residual>/<filter>": (N,)} of the residual baselines over the UPDATE
views only (a second ChangeBaselines fed the update views' residuals, ChangeBaselines.push_view;
the accumulator of the whole run keeps the mapping-pass null) with change_baselines_scope =
"update", and normlift_R_before_update = NormLift's reliability R at the end of the mapping pass
(--baselines; the row normlift_drop). --oracle-depth marks the --use-depth input as rendered mesh
depth (Replica): the depth residuals are then the oracle rows depth_*_oracle.

Feature transform (--feature-transform MODULE:FACTORY, --feature-transform-arg STRING): at
startup xform = FACTORY(arg, pca, device); on the UPDATE sequence only, right after z (h*w, d)
is in field space and before the field and CSS see it, z = xform(z, fr, i, (h, w)), with fr
the downscaled frame dict (its 'labels', if any, keep the dataset resolution: downscale()
resizes image, depth and K only) and i the frame index in the update dataset. Default: no
transform. The saved field and the run record store both strings (config).
"""
from __future__ import annotations
import argparse
import json
import os
import math
import sys
import torch

from usplat.vmf_field import DirectionalSemanticField, FieldConfig
from usplat.efe_planner import EFEPlanner
from usplat.evaluation import sparsification_curve, semantic_error_vs_text, auroc


def _bucket_labels(edges):
    """Names of the history buckets of torch.bucketize(ns, edges) (right-closed bins)."""
    e = [float(x) for x in edges]
    return ([f"h<={e[0]:g}"] + [f"{a:g}<h<={b:g}" for a, b in zip(e[:-1], e[1:])]
            + [f"h>{e[-1]:g}"])


def load_feature_transform(spec, arg, pca, device):
    """--feature-transform MODULE:FACTORY -> FACTORY(arg, pca, device), a callable
    xform(z, fr, i, (h, w)) -> z for the update sequence (see the module docstring); None
    without a spec."""
    if not spec:
        return None
    mod, sep, fac = spec.partition(":")
    if not (sep and mod and fac):
        raise SystemExit(f"--feature-transform must be MODULE:FACTORY, got {spec!r}")
    from importlib import import_module
    return getattr(import_module(mod), fac)(arg, pca, device)


def calibrate_field(field, q, fixed=None):
    """End of the mapping pass: the threshold from the field's own null (calib_mode), or one
    fixed threshold (--fixed-change-thr); returns its description for the summary.
    borrows_from (rule "history" only): the bucket whose null sets a bucket's threshold when it
    has fewer than MIN_NULL samples. null_exceedance: per history bucket, the share of its
    mapping-pass scores above the threshold applied to it (thr_by_hist, else change_thr), on
    the field's 4000-bin histogram (a bin counts when its lower edge >= the threshold: exact for
    the bin-edge thresholds of "history" and "pooled", a lower bound for "max" and "fixed").
    Nominal 1 - q; a bucket with fewer than 1/(1 - q) samples gets its largest score as
    threshold, i.e. exceedance 0 in sample, so the update-pass rate can deviate from 1 - q
    through small buckets as well as through a mapping / update mismatch."""
    counts = field.c_hist.sum(1)
    labels = _bucket_labels(field.HIST_EDGES)
    good = [b for b in range(len(counts)) if counts[b] >= field.MIN_NULL]
    if fixed is None:
        thr = field.calibrate_change_threshold(q)
    else:
        thr = field.set_change_threshold(fixed)
    rule = "fixed" if fixed is not None else field.cfg.calib_mode
    by_hist = None if field.thr_by_hist is None else [float(x) for x in field.thr_by_hist]
    applied = by_hist if by_hist is not None else [float(thr)] * len(counts)
    lo = field._c_edges[:-1]
    above = [float(field.c_hist[b][lo >= applied[b] - 1e-6].sum())   # float32 thr_by_hist
             for b in range(len(counts))]
    n = float(counts.sum())
    return dict(rule=rule, quantile=q, change_thr=thr, history_buckets=labels,
                thr_by_hist=by_hist, null_samples=[int(c) for c in counts],
                borrows_from=[None if rule != "history" or b in good else
                              (labels[min(good, key=lambda g: abs(g - b))] if good else "pooled")
                              for b in range(len(counts))],
                null_exceedance=[above[b] / float(counts[b]) if counts[b] > 0 else None
                                 for b in range(len(counts))],
                null_exceedance_pooled=sum(above) / n if n > 0 else None)


class ChangeTally:
    """Realised per-view false-alarm counts of one sequence (--summary-json: the UPDATE pass).

    On a static sequence every fire is a false alarm, so
        rate = sum_v fires_v / sum_v tests_v
    is the realised per-view false-alarm rate that the threshold targets (1 - q), overall and
    per history bucket. Field: tests_v = #last_tested, fires_v = #last_fired, per bucket the
    increments of n_tests / n_fires. Residual baselines: ResidualNull.test() counts, pooled.
    A fired Gaussian is reset and needs k new defined views before its next test, so tests
    count the views at which a Gaussian was actually tested. The per-view scores of the tested
    Gaussians are kept in a histogram on the field's bins (4000 on [0, 2]): the validation
    null of the fixed-threshold ablation. reset_pixel_mass_share = pixel mass of the Gaussians
    reset during the sequence / pixel mass of the observed Gaussians (all views so far);
    gaussian_fa = Gaussians reset / Gaussians observed in the sequence (the paper's FA: the
    share of static Gaussians flagged at least once during the update, on the Gaussians the
    update observed, as usplat.change_eval judges them; it grows with the number of views that
    test a Gaussian); reset_share_of_tested = Gaussians reset / Gaussians tested at least once
    (a diagnostic: its denominator leaves out the never-tested Gaussians, so it is larger than
    gaussian_fa by the inverse coverage)."""

    def __init__(self, field):
        self.field = field
        self._nt, self._nf = field.n_tests.clone(), field.n_fires.clone()
        self.resets0 = field.n_resets.clone()
        self.mass0 = field.pixel_mass.clone()
        self.ever_tested = torch.zeros_like(field.last_tested)
        self.score_hist = torch.zeros(4000, dtype=torch.float64)
        self.views = []

    @torch.no_grad()
    def step(self, index, baselines=None):
        """After the field ingested view `index`; baselines: {name: (tests, fires)}."""
        f = self.field
        dt, df = f.n_tests - self._nt, f.n_fires - self._nf
        self._nt, self._nf = f.n_tests.clone(), f.n_fires.clone()
        self.ever_tested |= f.last_tested
        v = f.c_last[f.last_tested].double().clamp(0.0, 2.0 - 1e-9)
        self.score_hist += torch.bincount((v * 2000).long().clamp(max=3999),
                                          minlength=4000).double().cpu()
        row = dict(index=int(index), tests=int(dt.sum()), fires=int(df.sum()),
                   tests_by_hist=dt.tolist(), fires_by_hist=df.tolist())
        if baselines:
            row["baselines"] = {n: dict(tests=t, fires=k) for n, (t, k) in baselines.items()}
        self.views.append(row)
        return row

    @torch.no_grad()
    def summary(self):
        f = self.field
        rate = lambda k, t: k / t if t else None
        T = sum(r["tests"] for r in self.views)
        F = sum(r["fires"] for r in self.views)
        tb = [sum(r["tests_by_hist"][b] for r in self.views) for b in range(len(f.n_tests))]
        fb = [sum(r["fires_by_hist"][b] for r in self.views) for b in range(len(f.n_tests))]
        labels = _bucket_labels(f.HIST_EDGES)
        reset = (f.n_resets - self.resets0) > 0
        pm = f.pixel_mass.double()
        obs = pm > 0
        upd = (f.pixel_mass - self.mass0).double()
        out = dict(views=len(self.views), tests=T, fires=F, rate=rate(F, T),
                   by_hist=[dict(bucket=labels[b], tests=tb[b], fires=fb[b],
                                 rate=rate(fb[b], tb[b])) for b in range(len(tb))],
                   gaussians_observed=int(obs.sum()), gaussians_reset=int(reset.sum()),
                   gaussians_observed_update=int((upd > 0).sum()),
                   gaussians_tested=int(self.ever_tested.sum()),
                   gaussian_fa=rate(int(reset.sum()), int((upd > 0).sum())),
                   reset_share_of_tested=rate(int(reset.sum()), int(self.ever_tested.sum())),
                   reset_pixel_mass_share=(float(pm[reset].sum() / pm[obs].sum())
                                           if bool(obs.any()) else None),
                   reset_update_mass_share=(float(upd[reset].sum() / upd.sum())
                                            if float(upd.sum()) > 0 else None),
                   score_hist=[int(x) for x in self.score_hist.tolist()],
                   per_view=self.views)
        tot = {}
        for r in self.views:
            for n, c in r.get("baselines", {}).items():
                t = tot.setdefault(n, [0, 0])
                t[0] += c["tests"]; t[1] += c["fires"]
        if tot:
            out["baselines"] = {n: dict(tests=t, fires=k, rate=rate(k, t))
                                for n, (t, k) in sorted(tot.items())}
        return out


def change_summary(field, tally, calib, nulls=None, args=None):
    """--summary-json metrics: the update-pass tally, the calibration and the config."""
    out = dict(update=tally.summary(), calibration=calib,
               field_config=field.cfg.to_dict(),
               config=dict(change_quantile=calib.get("quantile"),
                           calib_mode=field.cfg.calib_mode, k=field.cfg.change_persist,
                           fixed_change_thr=calib["change_thr"] if calib["rule"] == "fixed"
                           else None,
                           readmit_depth=field.cfg.readmit_depth,
                           change_weight=getattr(field.cfg, "change_weight", "view")))
    if nulls is not None:
        out["baselines_null"] = nulls.summary()
    if args is not None:
        out["args"] = dict(args)
    return out


def _record(path, args, name, argv=None):
    """usplat.runlog record of a --summary-json run (status, git state, config)."""
    from usplat.runlog import RunRecord
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return RunRecord(path, config=vars(args), seed=args.seed, name=name, argv=argv)


def run_mock(num_gaussians=200, feat_dim=32, num_frames=16, num_protos=6, seed=0,
             field_config="", change_quantile=0.999, static=False, feature_transform="",
             feature_transform_arg="", change_baselines=False, fixed_change_thr=None,
             summary=False):
    """CPU pipeline: the first half of the frames maps, the second half updates (20% of the
    Gaussians change content unless static). Returns the --summary-json metrics when summary
    (else None). The feature-transform factory gets pca=None and device "cpu", and the mock
    frame dict is {labels: (H, W) prototype ids, index}; the change baselines see palette
    colours and the rendered depth plus noise (both from their own generator: the default
    run draws the same random numbers)."""
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
    if static:                                          # static scene: nothing changes
        dynamic[:] = False
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
    xform = load_feature_transform(feature_transform, feature_transform_arg, None, "cpu")
    cb = nulls = None
    if change_baselines:
        from usplat.change_baselines import ChangeBaselines, ResidualNull
        cb = ChangeBaselines(num_gaussians, k=max(1, int(cfg.change_persist)),
                             view_mass_frac=cfg.view_mass_frac)
        nulls = ResidualNull()
        gen = torch.Generator().manual_seed(seed + 1)
        palette = torch.rand(num_protos, 3, generator=gen)
        colors = palette[true_proto]                    # the map's colours: mapping content
        gdepth = 1.0 + depths                           # Gaussian centre depths [m]
        alpha = Wm.sum(1)
        ren_depth = torch.where(alpha > 1e-6, Wm @ gdepth / alpha.clamp_min(1e-12),
                                torch.zeros_like(alpha))
    tally = calib = None
    for t in range(num_frames):
        if t == half:                                   # end of the mapping sequence
            calib = calibrate_field(field, change_quantile, fixed_change_thr)
            thr = calib["change_thr"]
            if fixed_change_thr is None:
                print(f"calibrated change threshold (q={change_quantile}): {thr:.3f}")
            else:
                print(f"fixed change threshold {thr:.3f} (no calibration)")
            if nulls is not None:
                print(f"residual baselines (q={change_quantile}): "
                      f"{nulls.calibrate(change_quantile)}")
            if summary:
                tally = ChangeTally(field)
        proto_t = new_proto if t >= half else true_proto
        lab = proto_t[dom].reshape(H, W)
        z_obs = clip.extract_from_labels(lab).reshape(-1, feat_dim)
        if xform is not None and t >= half:             # update sequence only
            z_obs = xform(z_obs, dict(labels=lab, index=t - half), t - half, (H, W))
        pv = None
        if cb is not None:                              # before the field ingests the view
            obs_rgb = palette[lab.reshape(-1)] + 0.02 * torch.randn(H * W, 3, generator=gen)
            obs_dep = ren_depth + 0.002 * torch.randn(H * W, generator=gen)
            pv = cb.update(ras, H * W, observed_rgb=obs_rgb.clamp(0, 1),
                           rendered_rgb=Wm @ colors,
                           observed_feat=z_obs, rendered_feat=Wm @ field.mu_post(),
                           depth_obs=obs_dep, rendered_depth=ren_depth, gauss_depth=gdepth,
                           image_hw=(H, W))
        field.update_view(ras, z_obs)
        if pv is not None and t < half:
            nulls.record(cb, pv)
        if tally is not None:
            tally.step(t - half, nulls.test(cb, pv) if pv is not None else None)

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
    if summary:
        s = change_summary(field, tally, calib, nulls)
        u = s["update"]
        print(f"update: {u['views']} views, {u['tests']} tests, {u['fires']} fires, "
              f"per-view rate {u['rate']}")
        return s


def css_vocabulary(args, full=False):
    """CSS is closed-set: it receives, at mapping time, the classes present in the scene's
    evaluation frames (the same rule as the queries of scripts/eval_seg.py), in vocabulary
    order, with the extractor's frozen templates. full=True (--css-full-vocab, the control
    css_full): every class of the vocabulary; no evaluation frame is read."""
    from usplat.datasets import make_dataset
    v = json.load(open(args.vocab))
    ignore = set(int(i) for i in (v.get("ignore") or v.get("ignore_ids") or []))
    order = [int(k) for k in v["classes"]]
    if full:
        present = set(order)
    else:
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


def input_entry(path):
    """{path, bytes, sha256} of an input file (the 3DGS map), saved with the field so a later
    step can refuse a field built on an earlier map at the same path; None without a path."""
    import hashlib
    if not path or not os.path.exists(path):
        return None
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return dict(path=os.path.abspath(path), bytes=os.path.getsize(path), sha256=d.hexdigest())


def _part(part, align=""):
    """make_dataset options for a sequence part and an update-pose alignment ({} when none
    is given)."""
    return {**({"part": part} if part else {}), **({"align": align} if align else {})}


def _pose_gap(c2w_a, c2w_b):
    dt = (c2w_a[:3, 3] - c2w_b[:3, 3]).norm().item()
    cos = ((c2w_a[:3, :3].t() @ c2w_b[:3, :3]).trace().item() - 1.0) / 2.0
    return dt, math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def run_gsplat(args, rec=None):
    """Real pipeline: pre-trained 3DGS checkpoint + posed RGB(-D) stream + dense CLIP.
    rec: the usplat.runlog record of --summary-json (gets the change summary)."""
    import os
    from tqdm import tqdm
    from usplat.gs_io import load_gaussians
    from usplat.datasets import make_dataset, downscale
    from usplat.clip_dense import upsample_tokens
    from usplat.feature_pca import StreamingPCA, RunningMean
    from usplat.gsplat_adapter import GSplatFeatureRasterizer
    from usplat.extractors import get_extractor
    from usplat import baselines as B

    device = args.device
    torch.manual_seed(args.seed)                            # PCA pixel subsample (randperm)
    from usplat import gpu_reserve                          # fraction from scripts/gpu_gate.sh
    held = gpu_reserve.apply(device)
    if held:
        print(f"reserved {held} MiB of GPU memory")
    splats = load_gaussians(args.ply, device=device)
    N = splats["means"].shape[0]
    print(f"loaded {N} Gaussians from {args.ply}")
    ds = make_dataset(args.dataset, args.data, stride=args.stride,
                      **_part(args.part, args.align))
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
    xform = load_feature_transform(args.feature_transform, args.feature_transform_arg, pca,
                                   device)
    if xform is not None:
        print(f"feature transform on the update sequence: {args.feature_transform} "
              f"({args.feature_transform_arg!r})")

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
                           **_part(args.update_part, args.align))
        upd_idx = select(upd)
        upd_tok = extract(upd, upd_idx, "update")
    query_t = None
    if args.query:
        query_t = clip.text_embedding([q.strip() for q in args.query.split(",")])
    css_ids, css_text = None, None
    if args.baselines and args.vocab and args.css_eval_data:
        css_ids, prompts, tpl = css_vocabulary(args)
        css_text = pca.project_text(clip.text_embedding(prompts, templates=tuple(tpl)))
    full_ids, full_text = None, None
    if args.css_full_vocab:                                 # control: CSS over every class
        full_ids, prompts, tpl = css_vocabulary(args, full=True)
        full_text = pca.project_text(clip.text_embedding(prompts, templates=tuple(tpl)))
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
    if args.robust_lift and not args.baselines:
        raise SystemExit("--robust-lift needs --baselines")
    if args.baselines:
        text = None
        if css_ids is not None:
            text = css_text
            print(f"CSS vocabulary: {len(css_ids)} classes present in {args.css_eval_data}")
        bs = B.default_baselines(field, text=text, class_ids=css_ids)
        for c0 in [float(x) for x in args.robust_lift.split(",") if x.strip()]:
            bs.append(B.RobustLift.for_field(field, c0=c0, tau=cfg.tau_min))
            print(f"RobustLift: c0 {c0:g}, tau {cfg.tau_min:g} (row {bs[-1].name})")
        if full_ids is not None:
            bs.append(B.full_vocab_css(field, full_text, full_ids))
            print(f"CSS full vocabulary: {len(full_ids)} classes of {args.vocab} (rows "
                  f"{B.CSS_FULL}*), counts {N} x {len(full_ids)} fp32 = "
                  f"{N * len(full_ids) * 4 / 2 ** 20:.0f} MiB in memory, saved row-sparse "
                  f"(dense if not smaller)")
    css = [b for b in bs if isinstance(b, B.CSS)]

    # ---- residual change baselines on the same views (--change-baselines) --------------
    cb = nulls = tally = cb_upd = nl_before = None
    cb_times = []
    if args.change_baselines:
        from usplat.change_baselines import ChangeBaselines, ResidualNull
        cb = ChangeBaselines(N, k=max(1, int(cfg.change_persist)),
                             view_mass_frac=cfg.view_mass_frac, oracle_depth=args.oracle_depth,
                             device=device)
        nulls = ResidualNull()

    def baseline_view(ras, fr, z, kw):
        """Residuals of one view against the map rendered at its pose, BEFORE the field
        ingests the view (the feature render is the field's state from the earlier views):
        NCC on the 3DGS colours, feature residual on mu*, depth residuals with --use-depth."""
        h, w = fr["image"].shape[:2]
        dkw = {}
        if "depth_obs" in kw:
            dkw = dict(depth_obs=kw["depth_obs"], gauss_depth=kw["gauss_depth"],
                       rendered_depth=ras.render_depth()[0].reshape(-1))
        ren_feat = ras.render_image(field.mu_post())[0].reshape(-1, args.feat_dim)
        return cb.update(ras, h * w,
                         observed_rgb=fr["image"].reshape(-1, 3).to(device).float() / 255.0,
                         rendered_rgb=ras.render_image(splats["colors"])[0].reshape(-1, 3),
                         observed_feat=z, rendered_feat=ren_feat, image_hw=(h, w), **dkw)

    def run_sequence(seq, idx, toks, desc):
        used = 0
        for i, tok in tqdm(list(zip(idx, toks)), desc=desc):
            fr_full = seq[i]
            c2w = fr_full["c2w"]
            fr = downscale(fr_full, args.downscale)
            h, w = fr["image"].shape[:2]
            z = pca.project(upsample_tokens(tok.to(device), (h, w))).reshape(-1, args.feat_dim)
            if xform is not None and desc == "update":         # --feature-transform
                z = xform(z, fr, i, (h, w))
            viewmat = torch.linalg.inv(c2w.double()).float().to(device)
            ras = GSplatFeatureRasterizer.from_splats(splats, viewmat, fr["K"].to(device),
                                                      w, h, channel_chunk=args.channel_chunk)
            kw = {}
            if args.use_depth and fr.get("depth") is not None:
                kw = dict(depth_obs=fr["depth"].reshape(-1).to(device),
                          gauss_depth=ras.gaussian_depths())
            pv = None
            if cb is not None:
                sync()
                t_b = time.perf_counter()
                pv = baseline_view(ras, fr, z, kw)
                sync()
                cb_times.append(time.perf_counter() - t_b)
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
            if pv is not None and desc == "mapping":
                nulls.record(cb, pv)                            # static by assumption: null
            if pv is not None and cb_upd is not None and desc == "update":
                cb_upd.push_view(pv, cb.last_view_weights)      # update-pass-only state
            if tally is not None and desc == "update":
                tally.step(i, nulls.test(cb, pv) if pv is not None else None)
        return used

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    used = run_sequence(ds, map_idx, map_tok, "mapping")
    del map_tok
    calib = None
    if args.update_data:
        calib = calibrate_field(field, args.change_quantile, args.fixed_change_thr)
        thr = calib["change_thr"]
        if args.fixed_change_thr is None:
            print(f"calibrated change threshold (q={args.change_quantile}): {thr:.4f}")
        else:
            print(f"fixed change threshold {thr:.4f} (--fixed-change-thr, no calibration)")
        if nulls is not None:
            print(f"residual baselines (q={args.change_quantile}): "
                  f"{nulls.calibrate(args.change_quantile)}")
        if cb is not None:          # cb keeps the null's state (ResidualNull reads its ring)
            cb_upd = cb.empty_like()
        nl = next((b for b in bs if isinstance(b, B.NormLift)), None)
        if nl is not None:
            nl_before = nl.reliability().float().cpu()
        if rec is not None:
            tally = ChangeTally(field)
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
    if cb is not None:
        timing["ms_per_keyframe_change_baselines"] = ms(cb_times)  # renders + residual pass
    print(f"timing: {timing}")
    if rec is not None:
        s = change_summary(field, tally, calib, nulls, args=vars(args))
        s.update(timing=timing, code=mapping_code_state(), num_gaussians=N,
                 keyframes_mapping=len(map_idx), keyframes_update=len(upd_idx))
        rec.update(s)
        rec.add_input(args.ply)
        u = s["update"]
        print(f"update pass: {u['views']} views, {u['tests']} tests, {u['fires']} fires, "
              f"per-view false-alarm rate {u['rate']} (nominal {1 - args.change_quantile:.3g})")
        for n, b in u.get("baselines", {}).items():
            print(f"  {n}: {b['tests']} tests, {b['fires']} fires, rate {b['rate']}")
    save_field(args.out, lambda: dict(
        field=field.state_dict(), pca=pca.state_dict(), config=vars(args), timing=timing,
        field_config=field.cfg.to_dict(), extractor=args.extractor,
        extractor_options=ext_opts, pca_explained=explained,
        code=mapping_code_state(), gpu_reserved_mib=held, inputs=dict(ply=input_entry(args.ply)),
        baselines=B.baselines_state(bs) if bs else [], css_class_ids=css_ids,
        normlift_R_before_update=nl_before, **change_state(cb_upd)),
        no_save=args.no_save_field)


def change_state(cb_upd):
    """Checkpoint keys of the residual baselines' update-pass-only state (usplat.change_eval):
    every filter's score in float32, its configuration, the number of update views and the
    scope marker; {} without --change-baselines or an update pass."""
    if cb_upd is None:
        return {}
    return dict(change_scores={k: v.float().cpu() for k, v in cb_upd.scores().items()},
                change_baselines_config=cb_upd.config(),
                change_baselines_views=cb_upd.num_views, change_baselines_scope="update")


def save_field(path, build, no_save=False):
    """torch.save(build(), path) atomically (never a truncated field). no_save
    (--no-save-field): nothing is built or written, so an existing file at path is never
    touched. Returns whether a field was written."""
    if no_save:
        print(f"--no-save-field: no field written ({path} is not touched)")
        return False
    out = build()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"                          # atomic: never a truncated field
    try:
        torch.save(out, tmp)
        os.replace(tmp, path)
    except BaseException:                    # a full disk or a failed pickle: no stray temp file
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print(f"saved field -> {path} (mapping code {out['code']['mapping_code_sha256'][:12]})")
    return True


def build_parser():
    """The command line (scripts/nor_tab2.py reuse parses it too: a saved field's config must
    equal the options a new run would get)."""
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
    ap.add_argument("--robust-lift", default="",
                    help="attempt 09: comma-separated c0 values; one RobustLift accumulator (rows "
                         "robust_c<c0>, trimmed view mean, tau = the field's tau_min) per value; needs --baselines")
    ap.add_argument("--vocab", default="", help="frozen vocabulary (CSS classes + templates)")
    ap.add_argument("--css-eval-data", default="",
                    help="evaluation sequence whose present classes form the CSS vocabulary")
    ap.add_argument("--css-eval-stride", type=int, default=10)
    ap.add_argument("--css-full-vocab", action="store_true",
                    help="with --baselines: also accumulate CSS over every class of --vocab "
                         "(rows css_full*: the control for CSS being told the scene's classes)")
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
    ap.add_argument("--align", default="",
                    help="nor: update-pose alignment JSON (scripts/align_nor.py); applied to "
                         "the update part of --data and --update-data")
    ap.add_argument("--change-quantile", type=float, default=0.999,
                    help="q of the mapping-pass null used to set the change threshold")
    ap.add_argument("--use-depth", action="store_true",
                    help="feed the dataset depth as the observed depth (re-admission check)")
    ap.add_argument("--summary-json", default="",
                    help="change-test record of the UPDATE pass (tests, fires, realised "
                         "per-view false-alarm rate, thresholds; a usplat.runlog record)")
    ap.add_argument("--change-baselines", action="store_true",
                    help="also run the residual change baselines (NCC, feature, depth) with "
                         "label-free thresholds from their own mapping-pass scores")
    ap.add_argument("--oracle-depth", action="store_true",
                    help="with --change-baselines --use-depth: the dataset depth is rendered "
                         "mesh depth (Replica), so the depth residuals are the oracle rows "
                         "depth_*_oracle (upper bound, never G3)")
    ap.add_argument("--fixed-change-thr", type=float, default=None,
                    help="one fixed change threshold instead of the calibration (ablation)")
    ap.add_argument("--no-save-field", action="store_true",
                    help="write no field file (an existing --out is never touched)")
    ap.add_argument("--feature-transform", default="",
                    help="MODULE:FACTORY; FACTORY(arg, pca, device) -> xform(z, fr, i, (h, w)) "
                         "applied to the update sequence's field-space features")
    ap.add_argument("--feature-transform-arg", default="",
                    help="string passed to the feature-transform factory")
    ap.add_argument("--mock-frames", type=int, default=16,
                    help="mock backend: frames (first half maps, second half updates)")
    ap.add_argument("--mock-static", action="store_true",
                    help="mock backend: no content change in the update half")
    ap.add_argument("--channel-chunk", type=int, default=32)
    ap.add_argument("--device", default="cuda",
                    help="gsplat backend device (cpu only with a CPU rasterizer, in tests)")
    ap.add_argument("--query", default="", help="comma-separated open-vocab text queries")
    ap.add_argument("--out", default="outputs/field.pt")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    cli = [sys.argv[0]] + list(argv if argv is not None else sys.argv[1:])
    if args.backend == "mock":
        kw = dict(field_config=args.field_config, change_quantile=args.change_quantile)
        if (args.summary_json or args.change_baselines or args.feature_transform
                or args.mock_static or args.fixed_change_thr is not None
                or args.mock_frames != 16):
            kw.update(num_frames=args.mock_frames, seed=args.seed, static=args.mock_static,
                      feature_transform=args.feature_transform,
                      feature_transform_arg=args.feature_transform_arg,
                      change_baselines=args.change_baselines,
                      fixed_change_thr=args.fixed_change_thr)
        if args.summary_json:
            with _record(args.summary_json, args, "run_mapping_mock", cli) as rec:
                rec.update(run_mock(summary=True, **kw))
        else:
            run_mock(**kw)
    else:
        if args.css_full_vocab and not (args.baselines and args.vocab):   # before extraction
            raise SystemExit("--css-full-vocab needs --baselines and --vocab")
        if args.oracle_depth and not (args.change_baselines and args.use_depth):
            raise SystemExit("--oracle-depth names the depth rows of --change-baselines "
                             "--use-depth")
        if not (args.ply and args.data):
            raise SystemExit("--backend gsplat requires --ply and --data")
        if args.summary_json and not args.update_data:
            raise SystemExit("--summary-json counts the UPDATE pass: give --update-data")
        if args.summary_json:
            with _record(args.summary_json, args, "run_mapping_static_fa", cli) as rec:
                run_gsplat(args, rec)
        else:
            run_gsplat(args)


if __name__ == "__main__":
    main()
