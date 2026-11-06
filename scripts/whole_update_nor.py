"""
whole_update_nor.py — one NeRF-Object-Removal scene of attempt 07 (configs/whole_update_prereg.json;
usplat/whole_update.py): change scores that compare each Gaussian's evidence over the whole
update with its evidence after mapping, from our field's own sums, rank the stale Gaussians
(Table III's Gaussian- and pixel-level AP). The reset, its test and every operating point are
unchanged.

    UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5 scripts/gpu_gate.sh 6500 python3 \
        scripts/whole_update_nor.py --table <results>/nor/tab2_relmax/val/002.json \
        --tolerance <results>/qual/nor/noise/tolerance.json \
        --prereg configs/whole_update_prereg.json --stage val --out <out>/val/002.json

The scene run is Table III's (scripts/eval_change_nor.py, imported, never edited): the mapping
replay and the update, with the update's two fields captured (dump_nor_qual.capture_fields and
its identity asserts) and every update view's per-view residuals read by a wrapper of
eval_change_nor.residuals_view that returns them unchanged; then the ground truth and the Table
III reproduction (stop rule, exit 2, as attempts 03 and 06). The variants come from the saved
field (M) and the captured copy with the change test off (E); the baseline mean rows from the
read residuals. Each variant and the NCC and feature mean rows are fused with Table III's depth
residual (fuse_rank on the observed Gaussians), and so are Table III's other rows (the fused
baselines of Table IV, for reference). Backends: gsplat (GPU) or mock (CPU,
tests/test_whole_update.py).
"""
from __future__ import annotations
import argparse
import contextlib
import gc
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import dump_nor_qual as DQ                                                  # noqa: E402
EC = DQ.EC
from usplat import baselines as B                                           # noqa: E402
from usplat.change_baselines import fuse_rank                              # noqa: E402
from usplat.vmf_field import DirectionalSemanticField                       # noqa: E402
from usplat.whole_update import (VARIANTS, RESIDUALS, channel_sums,        # noqa: E402
                                 variant_scores, ResidualMeans)

CODE = ("scripts/whole_update_nor.py", "usplat/whole_update.py", "scripts/dump_nor_qual.py")
TABLE3 = ("ncc", "depth", "feat", "normlift", "dispersion", "ours")    # EC.ROWS without 'fused'
FUSED3 = dict(ours="fused", ncc="ncc+depth", feat="feat+depth", normlift="normlift+depth",
              dispersion="dispersion+depth")                            # Table III's fused rows


def code_state():
    return {rel: DQ.sha256_file(os.path.join(REPO, rel)) for rel in CODE}


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    DQ.write_json(path, obj)


def free_gpu():
    """Collect reference cycles (the captured fields). Never torch.cuda.empty_cache(): it would
    hand the memory reserved by usplat.gpu_reserve back to other users of the shared card."""
    gc.collect()


@contextlib.contextmanager
def read_residuals(acc):
    """Within the block, every EC.residuals_view call's per-view residuals are added to acc and
    returned unchanged; yields the list of the ChangeBaselines objects the calls used."""
    orig = EC.residuals_view
    cbs = []

    def residuals(cb, *a, **kw):
        per = orig(cb, *a, **kw)
        if not cbs or cbs[-1] is not cb:
            cbs.append(cb)
        acc.add(per)
        return per
    EC.residuals_view = residuals
    try:
        yield cbs
    finally:
        EC.residuals_view = orig
    assert EC.residuals_view is orig


def saved_reliability(ckpt, dev):
    """The saved NormLift's reliability after mapping (dump_nor_qual.check_capture's reference)."""
    e = EC._normlift_entry(ckpt)
    cfg = dict(e["config"])
    dtype = getattr(torch, cfg.pop("dtype", "float32"))
    nl = B.NormLift(**cfg, device=dev, dtype=dtype)
    nl.load_state_dict(e["state"])
    return nl.reliability().clone()


def row_entry(g, p, gf=None, pf=None):
    """One row's Gaussian- and pixel-level metrics (and its fused row's)."""
    x = dict(gauss_ap=g["ap"], gauss_auroc=g["auroc"], gauss_n_nan=g["n_nan"],
             gauss_n_nan_pos=g["n_nan_pos"], gauss_common_ap=g["common"]["ap"],
             pixel_ap=p["ap"], pixel_auroc=p["auroc"])
    if gf is not None:
        x.update(fused_gauss_ap=gf["ap"], fused_gauss_auroc=gf["auroc"], fused_pixel_ap=pf["ap"],
                 fused_pixel_auroc=pf["auroc"])
    return x


def run(args, argv):
    t_start = time.time()
    if os.path.exists(args.out):
        old = json.load(open(args.out))
        if old.get("status") == "done" and old.get("stage") == "test":
            raise SystemExit(f"{args.out}: this test scene was run already (run once)")
    prereg = json.load(open(args.prereg))
    want = list(VARIANTS) if args.variants == "all" else [v for v in args.variants.split(",") if v]
    if not want or any(v not in VARIANTS for v in want) or len(set(want)) != len(want):
        raise SystemExit(f"--variants {args.variants!r}: not a subset of {list(VARIANTS)}")
    if args.stage == "test" and len(want) != 1:
        raise SystemExit("the test stage runs the accepted variant only (prereg test_once)")
    if tuple(prereg["variants"]) != VARIANTS:
        raise SystemExit("the prereg's variants differ from usplat/whole_update.py VARIANTS")
    if tuple(prereg["baselines"]["rows"]) != tuple(RESIDUALS):
        raise SystemExit("the prereg's baseline rows differ from usplat/whole_update.py RESIDUALS")
    run_v = [v for v in VARIANTS if v in want]                      # registered order
    mean_rows = tuple(RESIDUALS)
    fuse_new = tuple(run_v) + ("ncc_mean", "feat_mean")
    m = DQ.record_body(json.load(open(args.table)))
    if not args.smoke and not args.tolerance:
        raise SystemExit("--tolerance is required (plot_nor_qual.py noise)")
    tol = float(json.load(open(args.tolerance))["tolerance"]) if args.tolerance else None
    ns = DQ.ec_namespace(m, args)
    q = ns.change_quantile
    out = dict(stage=args.stage, status="running", scene=m["scene"], argv=argv, start=t_start,
               code=code_state(), prereg=os.path.abspath(args.prereg),
               prereg_sha256=DQ.sha256_file(args.prereg), table=os.path.abspath(args.table),
               table_sha256=DQ.sha256_file(args.table), tolerance=tol, git=DQ.git_state(),
               smoke=args.smoke, variants=run_v)
    write_json(args.out, out)
    code_now, rm_now = EC.code_state(), EC.RM.mapping_code_state()["mapping_code_sha256"]
    out["frozen_code"] = dict(eval_code_equal=code_now == m["provenance"].get("eval_code"),
                              mapping_code_equal=rm_now == m["provenance"].get("mapping_code_now"),
                              note="recorded, not required (prereg 'code.frozen')")
    if DQ.sha256_file(ns.field) != m["provenance"]["field"]["sha256"]:
        raise SystemExit(f"{ns.field}: sha256 differs from the record's")
    dev = args.device
    held = 0
    if str(dev).startswith("cuda"):
        from usplat import gpu_reserve
        held = gpu_reserve.apply(dev)
    T = {}
    ctx = EC.load_context(ns)
    T["load_extract_s"] = ctx["t_load"]
    if ctx["ply_entry"]["sha256"] != m["provenance"]["ply"]["sha256"]:
        raise SystemExit("the map's sha256 differs from the record's")
    ckpt, N = ctx["ckpt"], ctx["N"]
    views = dict(mapping=[v["index"] for v in ctx["map_views"]],
                 update=[v["index"] for v in ctx["upd_views"]])

    # ---- M: the saved field's raw channel (after mapping)
    f0 = DirectionalSemanticField.from_state(ckpt["field"], device="cpu")
    M = channel_sums(f0.raw)
    del f0
    r_map0 = saved_reliability(ckpt, dev)

    # ---- Table III's mapping replay (the rows' operating-point thresholds)
    k = int(ns.k or ckpt["field"]["config"]["change_persist"])
    t0 = time.time()
    replay, null, seg = EC.replay_mapping(ctx, ctx["map_views"], k, len(ctx["upd_views"]))
    check = EC.replay_check(replay, ckpt["field"])
    del replay
    if check["max_rel_dW1"] > 1e-2:
        raise SystemExit(f"the mapping replay does not reproduce the saved field: {check}")
    thr = {r: nl.threshold(q) for r, nl in null.items()}
    del null
    T["replay_s"] = time.time() - t0

    # ---- Table III's update, its fields captured and its per-view residuals read
    t0 = time.time()
    acc = ResidualMeans(N)
    with DQ.capture_fields(None) as made, read_residuals(acc) as cbs:
        upd = EC.run_update(ctx, ctx["upd_views"], k, thr, q)
    ours, disp, nl, cap_checks = DQ.check_capture(made, upd, r_map0)
    E = channel_sums(disp.raw)
    if len(cbs) != 1 or acc.n_views != len(ctx["upd_views"]):
        raise AssertionError(f"read {acc.n_views} views from {len(cbs)} accumulators, expected "
                             f"{len(ctx['upd_views'])} from 1")
    res_ident = acc.identity(cbs[0])
    if not all(res_ident.values()):
        raise AssertionError(f"read residuals differ from the residual series: {res_ident}")
    del ours, disp, nl, made, cbs
    free_gpu()
    T["update_s"] = time.time() - t0

    # ---- Table III reproduction (stop rule)
    t0 = time.time()
    gt, _, _, _ = EC.ground_truth(ctx, ns)
    stale, pop = gt["stale"], upd["observed"]
    scores3 = dict(upd["scores"])
    scores3["fused"] = fuse_rank([scores3["ours"], scores3["depth"]], mask=pop)
    y = stale[pop]
    gauss3, _ = EC.ranking_table({r: scores3[r][pop] for r in EC.ROWS}, y, EC.ROWS)
    common = pop.clone()
    for r in EC.TESTED:
        common &= upd["tests"][r] > 0
    ops3 = EC.operating_points(upd["flags"], stale, pop, common)
    pixel3 = EC.pixel_level(ctx, ctx["upd_views"], scores3, EC.ROWS, ns.tau_cov, ns.undef_frac)
    T["table3_s"] = time.time() - t0
    rep = None if args.smoke else DQ.reproduction(m, views, pixel3, gauss3, ops3, tol)
    out.update(views=views, replay_check=check, capture_checks=cap_checks,
               residual_identity=res_ident, reproduction=rep, gpu_reserved_mib=held)
    if rep is not None and not rep["ok"]:
        out.update(status="FAIL", wall_s=time.time() - t_start)
        write_json(args.out, out)
        print(f"{m['scene']}: REPRODUCTION FAILED: {rep['failures']}")
        return dict(out=out, exit=2)

    # ---- the new scores (NaN outside the observed Gaussians, as every Table III row)
    t0 = time.time()
    nan = torch.full((N,), float("nan"), dtype=torch.float64)
    inside = lambda s: torch.where(pop, s.double().cpu(), nan)
    new = {v: inside(s) for v, s in variant_scores(M, E, run_v).items()}
    new.update({r: inside(s) for r, s in acc.scores().items()})
    for r in fuse_new:
        new[f"{r}+depth"] = fuse_rank([new[r], scores3["depth"]], mask=pop)
    for r in ("ncc", "feat", "normlift", "dispersion"):          # Table IV's fused baselines
        new[FUSED3[r]] = fuse_rank([scores3[r], scores3["depth"]], mask=pop)
    rows_new = (tuple(run_v) + mean_rows + tuple(f"{r}+depth" for r in fuse_new)
                + tuple(FUSED3[r] for r in ("ncc", "feat", "normlift", "dispersion")))
    gauss, gauss_common = EC.ranking_table({r: new[r][pop] for r in rows_new}, y, rows_new)
    pixel = EC.pixel_level(ctx, ctx["upd_views"], new, rows_new, ns.tau_cov, ns.undef_frac)
    T["metrics_s"] = time.time() - t0

    G, P = dict(gauss3, **gauss), dict(pixel3["rows"], **pixel["rows"])
    rows = {}
    for r in TABLE3:
        f = FUSED3.get(r)
        rows[r] = row_entry(G[r], P[r], *((G[f], P[f]) if f else ()))
        if r in EC.TESTED:
            rows[r].update(fa=ops3[r]["fa"], rec=ops3[r]["rec"])
    for r in tuple(run_v) + mean_rows:
        f = f"{r}+depth"
        rows[r] = row_entry(G[r], P[r], *((G[f], P[f]) if f in G else ()))
    st, sp = pop & stale, pop & ~stale
    peak = (torch.cuda.max_memory_allocated() / 2 ** 20 if str(dev).startswith("cuda")
            and torch.cuda.is_available() else None)
    out.update(status="done", rows=rows, gauss_common=gauss_common, pixel_common=pixel["common"],
               counts=dict(n_gaussians=N, n_observed=int(pop.sum()), n_stale_observed=int(st.sum()),
                           n_static_observed=int(sp.sum()),
                           n_mapping_unseen_observed=int((pop & (M["W1"] <= 0)).sum())),
               timing=T, wall_s=time.time() - t_start, peak_alloc_mib=peak)
    write_json(args.out, out)
    for r in ("ours", "normlift") + tuple(run_v) + mean_rows:
        x = rows[r]
        fz = f" (+depth {x['fused_gauss_ap']:.4f})" if "fused_gauss_ap" in x else ""
        print(f"{m['scene']} {r:>10}: G AP {x['gauss_ap']:.4f}{fz} AUROC {x['gauss_auroc']:.4f} "
              f"NaN {x['gauss_n_nan']} | P AP {x['pixel_ap']:.4f}")
    print(f"{m['scene']}: done, wall {out['wall_s']:.1f} s, peak {peak} MiB; wrote {args.out}")
    return dict(out=out, exit=0, scores=new, observed=pop, stale=stale, M=M, E=E)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", required=True, help="the scene's Table III record (JSON)")
    ap.add_argument("--tolerance", default="", help="plot_nor_qual.py noise tolerance JSON")
    ap.add_argument("--prereg", required=True, help="configs/whole_update_prereg.json")
    ap.add_argument("--stage", choices=["val", "test", "toy"], required=True)
    ap.add_argument("--variants", default="all", help="'all' (validation) or the accepted variant (test)")
    ap.add_argument("--smoke", type=int, default=0,
                    help="evenly spaced subset of N update keyframes; no reproduction check")
    ap.add_argument("--backend", choices=["gsplat", "mock"], default="gsplat")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="output JSON")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.backend == "gsplat" and not str(args.device).startswith("cuda"):
        raise SystemExit("--backend gsplat needs --device cuda")
    full = [sys.argv[0]] + list(argv if argv is not None else sys.argv[1:])
    with torch.no_grad():
        res = run(args, full)
    if res.get("exit"):
        raise SystemExit(res["exit"])
    return res


if __name__ == "__main__":
    main()
