"""
cusum_reset_nor.py — one NeRF-Object-Removal scene of attempt 06 (configs/cusum_reset_prereg.json;
usplat/cusum_reset.py): a CUSUM over our own per-view change score fires our field's reset
instead of the per-view test; the reset, depth re-admission and robust set are ours.

    UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5 scripts/gpu_gate.sh 6500 python3 \
        scripts/cusum_reset_nor.py --table <results>/nor/tab2_relmax/val/002.json \
        --tolerance <results>/qual/nor/noise/tolerance.json --vocab configs/vocab_replica.json \
        --text-cache <out>/text_cache_lseg_nor.pt --dump <results>/qual/nor/002 \
        --prereg configs/cusum_reset_prereg.json --stage val --out <out>/val/002.json

The scene run follows scripts/reset_fused_nor.py (attempt 03; imported, never edited): the
mapping replay with the component capture, ours' Table III update, the Table III and N2
reproduction (stop rule, exit 2), one variant pass per CUSUM variant through
reset_fused_nor.run_variant, and the same metrics. Differences: the capture also keeps the
covered Gaussians of every mapping view (the CUSUM null needs per-Gaussian sequences); each
variant is also fused with the depth residual (fuse_rank, as Table III's fused rows); the hashes
of the frozen code are recorded, not required (they differ from the records by the path lines
of the 2026-10-31 move only; the reproduction is the equivalence check, see the prereg).
Backends: gsplat (GPU) or mock (CPU, tests/test_cusum_reset.py).
"""
from __future__ import annotations
import argparse
import contextlib
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import reset_fused_nor as RF                                                # noqa: E402
DQ, EC, ES = RF.DQ, RF.EC, RF.ES
from usplat.change_baselines import fuse_rank                              # noqa: E402
from usplat.cusum_reset import (VARIANTS, mapping_delta, cusum_null,       # noqa: E402
                                cusum_threshold, cusum_decider)
from usplat.reset_fused import COMPONENTS, field_score                    # noqa: E402
from usplat.vmf_field import DirectionalSemanticField                       # noqa: E402

CODE = ("scripts/cusum_reset_nor.py", "usplat/cusum_reset.py", "scripts/reset_fused_nor.py",
        "usplat/reset_fused.py", "scripts/dump_nor_qual.py")


def code_state():
    return {rel: DQ.sha256_file(os.path.join(REPO, rel)) for rel in CODE}


@contextlib.contextmanager
def capture_replay_idx():
    """reset_fused_nor.capture_replay with the covered Gaussians of every view ('idx')."""
    with RF.capture_replay(0) as (rec, state):
        orig = EC.residuals_view

        def residuals(cb, ras, v, z, ren, gz, kw, observed):
            per = orig(cb, ras, v, z, ren, gz, kw, observed)
            rec["views"][-1]["idx"] = torch.nonzero(observed).squeeze(1).int().cpu()   # int32: RAM
            return per
        EC.residuals_view = residuals
        try:
            yield rec, state
        finally:
            EC.residuals_view = orig
        assert EC.residuals_view is orig


def run(args, argv):
    t_start = time.time()
    if os.path.exists(args.out):
        old = json.load(open(args.out))
        if old.get("status") == "done" and old.get("stage") == "test":
            raise SystemExit(f"{args.out}: this test scene was run already (run once)")
    prereg = json.load(open(args.prereg))
    q = float(prereg["threshold"]["quantile"])
    run_v = list(VARIANTS) if args.variants == "all" else [v for v in args.variants.split(",") if v]
    if not run_v or any(v not in VARIANTS for v in run_v):
        raise SystemExit(f"--variants {args.variants!r}: not a subset of {list(VARIANTS)}")
    if args.stage == "test" and len(run_v) != 1:
        raise SystemExit("the test stage runs the accepted variant only (prereg test_once)")
    ROWS = ("ours",) + tuple(run_v)
    FUSED = tuple(f"{r}+depth" for r in ROWS)
    N2_SETS = ROWS + RF.CONTEXT_SETS
    if {k: float(v) for k, v in prereg["variants"].items()} != VARIANTS:
        raise SystemExit("the prereg's variants differ from usplat/cusum_reset.py VARIANTS")
    m = DQ.record_body(json.load(open(args.table)))
    if not args.smoke and not args.tolerance:
        raise SystemExit("--tolerance is required (plot_nor_qual.py noise)")
    tol = float(json.load(open(args.tolerance))["tolerance"]) if args.tolerance else None
    vocab = ES.load_vocab(args.vocab)
    ns = DQ.ec_namespace(m, args)
    if abs(ns.change_quantile - q) > 0:
        raise SystemExit(f"prereg quantile {q} != the record's {ns.change_quantile}")
    out = dict(stage=args.stage, status="running", scene=m["scene"], argv=argv, start=t_start,
               code=code_state(), prereg=os.path.abspath(args.prereg),
               prereg_sha256=DQ.sha256_file(args.prereg), table=os.path.abspath(args.table),
               table_sha256=DQ.sha256_file(args.table), tolerance=tol, git=DQ.git_state(),
               smoke=args.smoke, variants=run_v)
    RF.write_json(args.out, out)
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
    targs = DQ.text_args(ctx["cfg"], args.text_cache)
    if DQ.missing_prompts(vocab, targs):
        raise SystemExit(f"{args.text_cache} lacks prompts (fill a copy with dump_nor_qual.py --text-only)")
    cache_sha = DQ.sha256_file(args.text_cache)
    text = ctx["pca"].project_text(
        ES.text_embeddings(vocab, list(range(len(vocab["prompts"]))), targs).to(dev))
    if DQ.sha256_file(args.text_cache) != cache_sha:
        raise SystemExit("the text cache changed during the run")
    out["text_cache_sha256"] = cache_sha

    # ---- before the update
    f0 = DirectionalSemanticField.from_state(ckpt["field"], device="cpu")
    seen0 = f0.raw.W1 > 0
    del f0
    mu0 = ctx["frozen"]["mu"]
    e = EC._normlift_entry(ckpt)
    cfg_nl = dict(e["config"])
    dtype_nl = getattr(torch, cfg_nl.pop("dtype", "float32"))
    from usplat import baselines as B
    nl0 = B.NormLift(**cfg_nl, device=dev, dtype=dtype_nl)
    nl0.load_state_dict(e["state"])
    r_map0 = nl0.reliability().clone()
    del nl0

    # ---- mapping replay with the component capture (and the covered Gaussians per view)
    k = int(ns.k or ckpt["field"]["config"]["change_persist"])
    t0 = time.time()
    seg = EC.normlift_segment(len(ctx["map_views"]), len(ctx["upd_views"]))
    with capture_replay_idx() as (rec, state):
        state["seg"] = seg
        replay, null, seg_ec = EC.replay_mapping(ctx, ctx["map_views"], k, len(ctx["upd_views"]))
    if seg_ec != seg:
        raise AssertionError(f"segment {seg} != replay's {seg_ec}")
    check = EC.replay_check(replay, ckpt["field"])
    if check["max_rel_dW1"] > 1e-2:
        raise SystemExit(f"the mapping replay does not reproduce the saved field: {check}")
    ident = RF.null_identity(rec, null)
    del replay
    rec["field"].clear(); rec["nl"].clear()
    if not all(ident[c] for c in COMPONENTS):
        raise AssertionError(f"captured mapping samples differ from the replay nulls: {ident}")
    if any("idx" not in v or v["idx"].numel() != v["ours"].numel() for v in rec["views"]):
        raise AssertionError("a captured mapping view lacks its covered Gaussians")
    thr = {r: nl.threshold(q) for r, nl in null.items()}
    del null
    cus = {}
    for name in run_v:
        qd = VARIANTS[name]
        delta = mapping_delta(rec["views"], qd)
        nul = cusum_null(rec["views"], N, delta)
        cus[name] = dict(q_delta=qd, delta=delta, thr=cusum_threshold(nul, q), n_null=int(nul.size),
                         share_zero=float((nul == 0).mean()) if nul.size else float("nan"))
    del rec
    T["replay_s"] = time.time() - t0

    # ---- ours: Table III's run_update, captured
    t0 = time.time()
    with DQ.capture_fields(None) as made:
        upd = EC.run_update(ctx, ctx["upd_views"], k, thr, q)
    ours, disp, nl, cap_checks = DQ.check_capture(made, upd, r_map0)
    feats = dict(before=mu0.float().cpu(), nochange=disp.mu_post().float().cpu(),
                 ours=ours.mu_post().float().cpu(), avg_after=nl.feature().float().cpu())
    thr_ours_hist = ours.thr_by_hist.clone()
    del ours, disp, nl, made
    RF.free_gpu()
    T["ours_s"] = time.time() - t0

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
    out.update(views=views, replay_check=check, capture_checks=cap_checks, null_identity=ident,
               reproduction=rep, gpu_reserved_mib=held, cusum=cus)
    if rep is not None and not rep["ok"]:
        out.update(status="FAIL", wall_s=time.time() - t_start)
        RF.write_json(args.out, out)
        print(f"{m['scene']}: REPRODUCTION FAILED: {rep['failures']}")
        return dict(out=out, exit=2)

    # ---- the CUSUM variants
    res, cov_check = {}, {}
    for v in run_v:
        t0 = time.time()
        RF.free_gpu()
        r = RF.run_variant(ctx, ctx["upd_views"], k, q,
                           cusum_decider(cus[v]["delta"], cus[v]["thr"], N, dev), need_normlift=False)
        n_diff = int((r["observed"] != pop.cpu()).sum())
        cov_check[v] = dict(n_differ=n_diff, share=n_diff / max(int(pop.sum()), 1))
        if cov_check[v]["share"] > RF.COV_TOL:      # GPU sums are not bit-exact (atomics)
            raise AssertionError(f"{v}: coverage differs from ours': {cov_check[v]}")
        feats[v] = r.pop("mu")
        r.pop("covs")
        res[v] = r
        T[f"{v}_s"] = time.time() - t0

    # ---- metrics
    t0 = time.time()
    st, sp = pop & stale, pop & ~stale
    scores = dict(ours=scores3["ours"], **{v: res[v]["score"].float() for v in run_v})
    for r in ROWS:                                   # each fused with the depth residual
        scores[f"{r}+depth"] = fuse_rank([scores[r], scores3["depth"]], mask=pop)
    flags = dict(ours=upd["flags"]["ours"], **{v: res[v]["flags"] for v in run_v})
    tests = dict(ours=upd["tests"]["ours"], **{v: res[v]["tests"] for v in run_v})
    fires = dict(ours=upd["fires"]["ours"], **{v: res[v]["fires"] for v in run_v})
    robust = dict(ours=upd["robust_end"], **{v: res[v]["robust_end"] for v in run_v})
    all_rows = ROWS + FUSED
    gauss, gauss_common = EC.ranking_table({r: scores[r][pop] for r in all_rows}, y, all_rows)
    pixel = EC.pixel_level(ctx, ctx["upd_views"], scores, all_rows, ns.tau_cov, ns.undef_frac)
    rules = dict(EC.RULES)
    for v in run_v:
        rules[v] = (f"CUSUM S of c_p with delta = the {VARIANTS[v]:g} mapping quantile > the {q:g} "
                    f"mapping quantile of S, not changed; reset")
    saved_rules = EC.RULES
    EC.RULES = rules
    try:
        ops = EC.operating_points(flags, stale, pop, common)
    finally:
        EC.RULES = saved_rules
    assert EC.RULES is saved_rules
    rates = EC.realised_rates(tests, fires, stale, pop)
    lab0 = DQ.top_class(feats["before"], text)
    T["metrics_s"] = time.time() - t0

    # ---- N2 stale-answer rates
    t0 = time.time()
    qi, counts = DQ.query_class(mu0, seen0, stale, text)
    total = int(counts.sum())
    query = dict(index=qi, name=vocab["names"][qi], share=float(counts[qi]) / total, n_stale_seen=total)
    G = dict(flags=dict(ours=flags["ours"]), feat=feats)
    saved_sets = DQ.SETS
    DQ.SETS = N2_SETS
    try:
        con, _ = DQ.contrast_and_fa(ctx, ctx["upd_views"], G, text, qi, ns.tau_cov, ns.undef_frac)
    finally:
        DQ.SETS = saved_sets
    assert DQ.SETS is saved_sets
    T["n2_s"] = time.time() - t0
    dump_check = None
    if args.dump:
        dc = json.load(open(os.path.join(args.dump, "contrast.json")))["sets"]
        dump_check = {s: dict(rerun=con[s]["stale_answer_rate"], dump=float(dc[s]["stale_answer_rate"]))
                      for s in ("ours", "nochange", "avg_after", "before")}
        for d in dump_check.values():
            d["delta"] = abs(d["rerun"] - d["dump"])
        dump_check["ok"] = max(d["delta"] for d in dump_check.values()) <= RF.STALE_TOL
        if not dump_check["ok"] and not args.smoke:
            out.update(status="FAIL", dump_check=dump_check, wall_s=time.time() - t_start)
            RF.write_json(args.out, out)
            print(f"{m['scene']}: N2 does not reproduce the dump: {dump_check}")
            return dict(out=out, exit=2)

    rows = {}
    for r in ROWS:
        rows[r] = dict(gauss_ap=gauss[r]["ap"], gauss_auroc=gauss[r]["auroc"],
                       gauss_n_nan=gauss[r]["n_nan"], gauss_common_ap=gauss[r]["common"]["ap"],
                       pixel_ap=pixel["rows"][r]["ap"], pixel_auroc=pixel["rows"][r]["auroc"],
                       fused_gauss_ap=gauss[f"{r}+depth"]["ap"], fused_gauss_auroc=gauss[f"{r}+depth"]["auroc"],
                       fused_pixel_ap=pixel["rows"][f"{r}+depth"]["ap"],
                       fused_pixel_auroc=pixel["rows"][f"{r}+depth"]["auroc"],
                       fa=ops[r]["fa"], rec=ops[r]["rec"], fa_common=ops[r]["fa_common"],
                       rec_common=ops[r]["rec_common"], rule=ops[r]["rule"], **rates[r],
                       kept=RF.share(robust[r][st]),
                       n_resets=(upd["n_resets"] if r == "ours" else res[r]["n_resets"]),
                       stale_answer=con[r]["stale_answer_rate"], delta_cos=con[r]["delta_cos"],
                       n_mask_cov=con[r]["n_mask_cov"],
                       static_relabel=RF.static_relabel(feats[r], lab0, sp, text))
        if r in run_v:
            rows[r].update(**cus[r])
    context = {s: dict(stale_answer=con[s]["stale_answer_rate"], delta_cos=con[s]["delta_cos"],
                       static_relabel=RF.static_relabel(feats[s], lab0, sp, text))
               for s in RF.CONTEXT_SETS}
    peak = (torch.cuda.max_memory_allocated() / 2 ** 20 if str(dev).startswith("cuda")
            and torch.cuda.is_available() else None)
    out.update(status="done", rows=rows, context=context, query=query, dump_check=dump_check,
               gauss_common=gauss_common, pixel_common=pixel["common"], coverage_check=cov_check,
               ours_threshold_by_history=thr_ours_hist.cpu().tolist(),
               counts=dict(n_gaussians=N, n_observed=int(pop.sum()), n_stale_observed=int(st.sum()),
                           n_static_observed=int(sp.sum())),
               timing=T, wall_s=time.time() - t_start, peak_alloc_mib=peak)
    RF.write_json(args.out, out)
    for r in ROWS:
        x = rows[r]
        print(f"{m['scene']} {r:>4}: G AP {x['gauss_ap']:.4f} (+depth {x['fused_gauss_ap']:.4f}) P AP "
              f"{x['pixel_ap']:.4f} | FA {x['fa']:.4f} Rec {x['rec']:.4f} Kept {x['kept']:.4f} | "
              f"stale-answer {x['stale_answer']:.4f} static relabel {x['static_relabel']:.4f}")
    print(f"{m['scene']}: done, wall {out['wall_s']:.1f} s, peak {peak} MiB; wrote {args.out}")
    return dict(out=out, exit=0)


def build_parser():
    ap = RF.build_parser()
    ap.description = __doc__.split("\n\n")[0]
    for a in ap._actions:
        if a.dest == "prereg":
            a.help = "configs/cusum_reset_prereg.json"
    ap.add_argument("--variants", default="all", help="'all' (validation) or the accepted variant (test)")
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
