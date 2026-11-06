"""
eval_change_nor_distilled.py — the feature residual of the optimization-based baselines of Table I
(Feature 3DGS, LEGaussians; usplat/distill.py, scripts/distill_features.py) as stale-content rows of
Table III (tab:change) on one NeRF-Object-Removal scene (registered: configs/nor_distilled_prereg.json).

  scripts/gpu_gate.sh 6000 python3 scripts/eval_change_nor_distilled.py \
      --field results/fields/nor_001_lseg_relmax.pt --align none --update-stride 1 --tau-cov 0.5 \
      --distilled results/fields/nor_001_lseg_relmax_f3dgs.pt,results/fields/nor_001_lseg_relmax_legs_l1.pt \
      --reference results/nor/tab2_relmax/val/001.json --out results/nor/distilled/val/001.json

Each distilled file D gives the row "<D.name>_feat" (f3dgs_feat, legs_l<lambda>_feat): the 'feat' row
of scripts/eval_change_nor.py with the rendered features of D in place of the saved field's mu*.
Feature 3DGS renders its per-Gaussian features; LEGaussians renders its 8-d codes and decodes the
composite with its MLP (its query rule, usplat.distill.LEGDecoder). D is trained on the mapping
frames only and stays frozen during the update (these methods cannot ingest a view without
retraining). Everything else is eval_change_nor.py's, reused from it: the views and features
(load_context), the coverage rule (covered() of the field replayed over the mapping keyframes, and
of the saved field updated with depth over the update keyframes, as eval_change_nor.py does),
ChangeBaselines with the field's k (the min of the last k covered per-view residuals, kmin), the
pooled 0.999-quantile threshold of the row's own mapping-view statistic, the D2 Gaussian and D1
pixel ground truth, the observed population, AP / AUROC with NaN ranked last, FA / Rec.
A control row "feat_control" recomputes eval_change_nor.py's 'feat' row in the same passes;
--reference compares it, and the observed population, with the scene's Table III record.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import eval_change_nor as ECN                                               # noqa: E402
from usplat import distill as DI                                            # noqa: E402
from usplat.change_baselines import ChangeBaselines                         # noqa: E402
from usplat.runlog import to_jsonable                                       # noqa: E402
from usplat.vmf_field import DirectionalSemanticField                       # noqa: E402

CONTROL = "feat_control"
CODE = ("scripts/eval_change_nor_distilled.py", "usplat/distill.py") + tuple(ECN.CODE)


def code_state():
    repo = os.path.dirname(HERE)
    out = {}
    for rel in CODE:
        with open(os.path.join(repo, rel), "rb") as f:
            out[rel] = hashlib.sha256(f.read()).hexdigest()
    return out


def load_sources(ctx, paths, control=True):
    """{row: render(ras, ren) -> (P, d) features (ren: the frozen map render of the view)} and the
    distilled files' provenance."""
    dev, N, ckpt = ctx["device"], ctx["N"], ctx["ckpt"]
    want = DI.basis_sha256(ckpt["pca"])
    src, prov = {}, {}
    if control:
        src[CONTROL] = lambda ras, ren: ren["feat"]              # eval_change_nor.py's own
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        if d.get("pca_basis_sha256") != want:
            raise SystemExit(f"{p}: distilled in another PCA basis than {ctx['cfg'].get('out')}")
        f = d["features"].to(dev).float()
        if f.shape[0] != N:
            raise SystemExit(f"{p}: {f.shape[0]} Gaussians, the map has {N}")
        row = f"{d['name']}_feat"
        if d.get("decoder") is not None:                     # LEGaussians: render, then decode
            dec = DI.LEGDecoder(d["decoder"]["codebook"], dim=f.shape[1],
                                hidden=d.get("decoder_hidden", DI.LEGS_HIDDEN))
            dec.load_state_dict(d["decoder"])
            dec = dec.to(dev).eval()
            src[row] = lambda ras, ren, f=f, dec=dec: dec(ras(f))
        else:                                                # Feature 3DGS: its features
            src[row] = lambda ras, ren, f=f: ras(f)
        prov[row] = dict(ECN.file_entry(p), name=d["name"], identity=d.get("identity"),
                         config=d.get("config"), main_field=d.get("main_field"),
                         n_frames=d.get("n_frames"), iters_run=d.get("iters_run"),
                         train_s=(d.get("timing") or {}).get("train_s"))
    return src, prov


def replay(ctx, views, k, src):
    """Pass 1 (eval_change_nor.replay_mapping, the residual part): the mapping keyframes through a
    fresh field with a fresh NormLift on its hooks (the coverage), one ChangeBaselines per row."""
    ckpt, dev, N = ctx["ckpt"], ctx["device"], ctx["N"]
    field = DirectionalSemanticField(N, ctx["d"], ECN.field_config(ckpt), device=dev)
    field.set_centre(ctx["zbar"])
    ECN.make_normlift(ckpt, field, saved=False)
    cbs = {r: ChangeBaselines(N, k=k, view_mass_frac=field.cfg.view_mass_frac, device=dev) for r in src}
    null = {r: ECN.Null() for r in src}
    for v in views:
        z, ras, gz, kw = ECN.view_inputs(ctx, v)
        ren = ECN.map_render(ras, gz, ctx["frozen"])
        feats = {r: f(ras, ren) for r, f in src.items()}
        stats = field.update_view(ras, z, num_pixels=v["h"] * v["w"], **kw)
        cov = ECN.covered(field, stats)
        for r in src:              # eval_change_nor.py's call; only the rendered features differ
            per = ECN.residuals_view(cbs[r], ras, v, z, dict(ren, feat=feats[r]), gz, kw, cov)
            cur = ECN.view_kmin(cbs[r], "feat", per)
            if cur is not None:
                null[r].add(cur)
    return field, null


def update(ctx, views, k, src, thr, q):
    """Pass 2 (eval_change_nor.run_update, the residual part): the update keyframes through a copy
    of the saved field (its own calibration, update_view with depth: the coverage), one
    ChangeBaselines per row; scores, flags, tests, fires, the observed and seen Gaussians."""
    ckpt, dev, N = ctx["ckpt"], ctx["device"], ctx["N"]
    ours = DirectionalSemanticField.from_state(ckpt["field"], device=dev)
    ours.calibrate_change_threshold(q)
    cbs = {r: ChangeBaselines(N, k=k, view_mass_frac=ours.cfg.view_mass_frac, device=dev) for r in src}
    zeros = lambda dtype: torch.zeros(N, dtype=dtype, device=dev)
    seen, observed, n_views = zeros(torch.bool), zeros(torch.bool), zeros(torch.int32)
    flags = {r: zeros(torch.bool) for r in src}
    tests = {r: zeros(torch.int32) for r in src}
    fires = {r: zeros(torch.int32) for r in src}
    for v in views:
        z, ras, gz, kw = ECN.view_inputs(ctx, v)
        ren = ECN.map_render(ras, gz, ctx["frozen"])
        feats = {r: f(ras, ren) for r, f in src.items()}
        stats = ours.update_view(ras, z, num_pixels=v["h"] * v["w"], **kw)
        cov = ECN.covered(ours, stats)
        seen |= stats.m > ECN.EPS
        observed |= cov
        n_views += cov.int()
        for r in src:
            per = ECN.residuals_view(cbs[r], ras, v, z, dict(ren, feat=feats[r]), gz, kw, cov)
            cur = ECN.view_kmin(cbs[r], "feat", per)
            if cur is not None:
                t = torch.isfinite(cur)
                fired = t & (cur > thr[r]["pooled"])
                tests[r] += t.int()
                fires[r] += fired.int()
                flags[r] |= fired
    scores = {r: ECN.series_score(cbs[r], "feat", "kmin", N) for r in src}
    flags = {r: f.cpu() for r, f in flags.items()}
    for r in src:                                    # the kmin flag is the score above thr
        s = scores[r]
        if not torch.equal(flags[r], torch.isfinite(s) & (s > thr[r]["pooled"])):
            raise RuntimeError(f"{r}: per-view flags differ from score > threshold")
    return dict(scores=scores, flags=flags, tests={r: t.cpu() for r, t in tests.items()},
                fires={r: f.cpu() for r, f in fires.items()}, observed=observed.cpu(),
                seen=seen.cpu(), n_views=n_views.cpu())


def compare(control, ref):
    """The control row and the observed population against the scene's Table III record."""
    m = ref.get("metrics", ref)
    pairs = dict(g_ap=(control["g_ap"], m["gaussian"]["rows"]["feat"]["ap"]),
                 g_auroc=(control["g_auroc"], m["gaussian"]["rows"]["feat"]["auroc"]),
                 p_ap=(control["p_ap"], m["pixel"]["rows"]["feat"]["ap"]),
                 p_auroc=(control["p_auroc"], m["pixel"]["rows"]["feat"]["auroc"]),
                 fa=(control["fa"], m["operating_point"]["feat"]["fa"]),
                 rec=(control["rec"], m["operating_point"]["feat"]["rec"]),
                 threshold=(control["threshold"], m["thresholds"]["rows"]["feat"]["pooled"]),
                 n_observed=(control["n_observed"], m["counts"]["n_observed"]),
                 n_stale_observed=(control["n_stale_observed"], m["counts"]["n_stale_observed"]))
    out = {k: dict(here=a, record=b, abs_diff=abs(float(a) - float(b)), exact=float(a) == float(b))
           for k, (a, b) in pairs.items()}
    out["all_exact"] = all(v["exact"] for v in out.values() if isinstance(v, dict))
    out["max_abs_diff"] = max(v["abs_diff"] for v in out.values() if isinstance(v, dict))
    return out


def evaluate(args):
    dev = args.device
    held = 0
    if str(dev).startswith("cuda"):
        from usplat import gpu_reserve                       # fraction from scripts/gpu_gate.sh
        held = gpu_reserve.apply(dev)
    t = {}
    ctx = ECN.load_context(args)
    t["load_extract_s"] = ctx["t_load"]
    ckpt, q = ctx["ckpt"], args.change_quantile
    k = int(args.k or ckpt["field"]["config"]["change_persist"])
    src, prov = load_sources(ctx, [p for p in args.distilled.split(",") if p], not args.no_control)
    rows = list(src)
    t0 = time.time()
    field, null = replay(ctx, ctx["map_views"], k, src)
    check = ECN.replay_check(field, ckpt["field"])
    del field
    if check["max_rel_dW1"] > 1e-2:
        raise SystemExit(f"the mapping replay does not reproduce the saved field: {check}")
    thr = {r: nl.threshold(q) for r, nl in null.items()}
    t["replay_s"] = time.time() - t0
    t0 = time.time()
    upd = update(ctx, ctx["upd_views"], k, src, thr, q)
    t["update_s"] = time.time() - t0
    t0 = time.time()
    gt, _, _, _ = ECN.ground_truth(ctx, args)
    stale, pop = gt["stale"], upd["observed"]
    t["ground_truth_s"] = time.time() - t0
    y = stale[pop]
    gauss = {r: ECN.ranking(upd["scores"][r][pop], y) for r in rows}
    t0 = time.time()
    pixel = ECN.pixel_level(ctx, ctx["upd_views"], upd["scores"], rows, args.tau_cov, args.undef_frac)
    t["pixel_s"] = time.time() - t0
    st, sp = pop & stale, pop & ~stale
    rates = ECN.realised_rates(upd["tests"], upd["fires"], stale, pop)
    ops = {r: dict(fa=ECN.share(upd["flags"][r][sp]), rec=ECN.share(upd["flags"][r][st]),
                   n_flagged_static=int(upd["flags"][r][sp].sum()),
                   n_flagged_stale=int(upd["flags"][r][st].sum()), threshold=thr[r]["pooled"],
                   n_null=thr[r]["n"], rule=ECN.RULES["feat"], **rates[r]) for r in rows}
    summary = {r: dict(g_ap=gauss[r]["ap"], g_auroc=gauss[r]["auroc"],
                       p_ap=(pixel.get("rows") or {}).get(r, {}).get("ap", float("nan")),
                       p_auroc=(pixel.get("rows") or {}).get(r, {}).get("auroc", float("nan")),
                       fa=ops[r]["fa"], rec=ops[r]["rec"], threshold=thr[r]["pooled"],
                       g_unscored=gauss[r]["n_nan"] / max(gauss[r]["n"], 1),
                       n_observed=int(pop.sum()), n_stale_observed=int(st.sum())) for r in rows}
    control_check = None
    if args.reference and CONTROL in summary:
        with open(args.reference) as f:
            control_check = compare(summary[CONTROL], json.load(f))
        control_check["reference"] = ECN.file_entry(args.reference)
    counts = dict(n_gaussians=ctx["N"], n_observed=int(pop.sum()), n_seen=int(upd["seen"].sum()),
                  n_stale=int(stale.sum()), n_stale_observed=int(st.sum()),
                  n_static_observed=int(sp.sum()))
    peak = (torch.cuda.max_memory_allocated() / 2 ** 20 if str(dev).startswith("cuda")
            and torch.cuda.is_available() else None)
    return dict(scene=os.path.basename(os.path.normpath(ctx["data"])), rows=rows,
                inputs={r: "F" for r in rows}, summary=summary, gaussian=dict(rows=gauss),
                pixel=pixel, operating_point=ops, thresholds=dict(rows=thr, quantile=q),
                counts=counts, control_check=control_check, replay_check=check,
                views=dict(n_mapping=len(ctx["map_views"]), n_update=len(ctx["upd_views"])),
                settings=dict(change_quantile=q, k=k, tau_cov=args.tau_cov,
                              undef_frac=args.undef_frac, inflate_m=args.inflate, near_m=args.near,
                              update_stride=ctx["upd_stride"], backend=args.backend),
                provenance=dict(field=ECN.file_entry(args.field), ply=ctx["ply_entry"],
                                distilled=prov, eval_code=code_state(), gpu_reserved_mib=held,
                                peak_alloc_mib=peak),
                timing=t)


def build_parser():
    ap = ECN.build_parser()
    ap.description = __doc__.split("\n\n")[0]
    ap.add_argument("--distilled", required=True,
                    help="distilled files (scripts/distill_features.py), comma-separated")
    ap.add_argument("--reference", default="",
                    help="the scene's Table III record (eval_change_nor.py): checks the control row")
    ap.add_argument("--no-control", action="store_true", help="skip the control row")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.backend == "gsplat" and not str(args.device).startswith("cuda"):
        raise SystemExit("--backend gsplat needs --device cuda (use --backend mock on the CPU)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.plain_json:
        t0 = time.time()
        with torch.no_grad():
            out = evaluate(args)
        out["command"] = dict(argv=sys.argv if argv is None else ["eval_change_nor_distilled.py"] + list(argv),
                              wall_s=time.time() - t0, config=vars(args))
        tmp = f"{args.out}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(to_jsonable(out), f, indent=1)
        os.replace(tmp, args.out)
    else:
        from usplat.runlog import RunRecord
        with RunRecord(args.out, config=vars(args), name="eval_change_distilled",
                       argv=[sys.argv[0]] + list(argv if argv is not None else sys.argv[1:])) as rec:
            with torch.no_grad():
                out = evaluate(args)
            for key, value in out.items():
                rec.add(key, value)
            for p in [args.field, out["provenance"]["ply"]["path"]] + \
                     [e["path"] for e in out["provenance"]["distilled"].values()]:
                rec.add_input(p)
    fmt = lambda x: "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"
    for r in out["rows"]:
        s = out["summary"][r]
        print(f"{out['scene']} {r:>16}: G AP {fmt(s['g_ap'])} AUROC {fmt(s['g_auroc'])} | P AP "
              f"{fmt(s['p_ap'])} AUROC {fmt(s['p_auroc'])} | FA {fmt(s['fa'])} Rec {fmt(s['rec'])}")
    cc = out["control_check"]
    if cc is not None:
        print(f"{out['scene']} control vs Table III record: all exact {cc['all_exact']}, "
              f"max |diff| {cc['max_abs_diff']:.3g}")
    print(f"wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
