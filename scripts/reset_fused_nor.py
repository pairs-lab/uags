"""
reset_fused_nor.py — one NeRF-Object-Removal scene of the fused-reset attempt
(configs/reset_fused_prereg.json; usplat/reset_fused.py): our field's reset is fired by a
fused change detector (V1 feat+depth, V2 ours+feat+depth, V3 NormLift+depth) instead of our
own test; the reset, depth re-admission and robust set are ours.

    UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5 scripts/gpu_gate.sh 6500 python3 \
        scripts/reset_fused_nor.py --table <results>/nor/tab2_relmax/val/002.json \
        --tolerance <results>/qual/nor/noise/tolerance.json --vocab configs/vocab_replica.json \
        --text-cache <out>/text_cache_lseg_nor.pt --dump <results>/qual/nor/002 \
        --prereg configs/reset_fused_prereg.json --stage val --out <out>/val/002.json

It replays the Table III evaluation of scripts/eval_change_nor.py (imported, never edited)
with every setting from the scene's Table III record, as scripts/dump_nor_qual.py does (its
functions are reused: ec_namespace, capture_fields, check_capture, reproduction, query_class,
top_class, contrast_and_fa). Inside this process only it
  * wraps EC.residuals_view, EC.make_normlift and EC.DirectionalSemanticField during
    EC.replay_mapping to read every component's per-view value on the mapping views; the
    samples must equal the replay's own nulls (feat, depth, NormLift: sorted values
    identical; ours: as many samples as the replay field's null histogram);
  * captures run_update's fields (dump_nor_qual.capture_fields) for ours' end state;
  * sets dump_nor_qual.SETS to this script's feature sets during contrast_and_fa.
Every patch is undone in a finally block, which is asserted.
Stop rule (exit 2): the Table III reproduction of dump_nor_qual.reproduction (keyframes,
per-frame counts, every AP/AUROC within --tolerance, ours' FA / Rec within 0.5 points).

Variant passes (one per variant, after ours): a copy of the saved field (ExternalResetField),
calibrated like ours and with its own test off; every update keyframe goes through it with
depth exactly as run_update feeds ours; the coverage (EC.covered) must equal ours' (CPU: exactly; GPU: up to 1e-4 of the observed Gaussians, the backward's atomic sums
are not bit-exact); ChangeBaselines (k) against the frozen map at that coverage; for V3 a second copy with
the change test off and the saved NormLift on its hooks (run_update's "disp"). Per view:
s_V on the covered Gaussians, fire = s_V > thr_V and not changed, external_reset.

Outputs (--out, one JSON; a finished test scene is never redone):
  rows ours, V1, V2, V3:
    firing score  ours: Table III's (c_max where c_last is defined); V: max over the update
                  views of s_V (NaN if never defined); Gaussian AP / AUROC on the observed
                  Gaussians (EC.ranking_table, NaN last) and pixel AP / AUROC (EC.pixel_level)
    fa, rec       flagged = reset at least once in the update, on the static / stale observed
                  Gaussians (EC.operating_points; + the common-tested set of Table III)
    rate          fires / tests over the update's Gaussian-views (static, stale)
    kept          share of the stale observed Gaussians in the robust set at the end
    stale_answer  N2 (dump_nor_qual.contrast_and_fa): share of the covered masks_real pixels
                  of every update keyframe whose top class (101 classes) is the scene's query
    static_relabel share of the static observed Gaussians whose top class at the end differs
                  from the mapping-pass mu*'s
  also: the N2 numbers of before / change test off / averaging, the thresholds and null sizes,
  the Table III reproduction, ours' stale-answer rate against the scene's dump (--dump), the
  code / prereg / record hashes and the timing.
Backends: gsplat (GPU) or mock (CPU, tests/test_reset_fused.py).
"""
from __future__ import annotations
import argparse
import contextlib
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import dump_nor_qual as DQ                                                  # noqa: E402
EC = DQ.EC
ES = DQ.ES
from usplat.change_baselines import ChangeBaselines, fuse_rank             # noqa: E402
from usplat.reset_fused import (VARIANTS, COMPONENTS, MappingRanks,         # noqa: E402
                                ExternalResetField, fused_stat, field_score)
from usplat.vmf_field import DirectionalSemanticField                       # noqa: E402

ROWS = ("ours",) + tuple(VARIANTS)
CONTEXT_SETS = ("before", "nochange", "avg_after")       # N2 context rows (no reset decision)
N2_SETS = ROWS + CONTEXT_SETS
STALE_TOL = 0.005                  # ours' stale-answer rate vs the dump: 0.5 points
COV_TOL = 1e-4                     # observed set of a variant pass vs ours' (share of observed)
CODE = ("scripts/reset_fused_nor.py", "usplat/reset_fused.py", "scripts/dump_nor_qual.py")


def free_gpu():
    """Collect reference cycles (the captured fields), so that their GPU tensors go back to
    this process's caching allocator. Never torch.cuda.empty_cache(): it would hand the memory
    reserved by usplat.gpu_reserve back to other users of the shared card."""
    gc.collect()


def code_state():
    return {rel: DQ.sha256_file(os.path.join(REPO, rel)) for rel in CODE}


# --------------------------------------------------------------------- mapping capture
@contextlib.contextmanager
def capture_replay(n_update):
    """During the block, EC.replay_mapping's field, NormLift and every view's components are
    recorded: yields a dict with 'views' (list of {cov, ours, feat, depth, normlift} CPU
    tensors restricted to the covered Gaussians) and 'field', 'nl'."""
    rec = dict(views=[], field=[], nl=[], t=0)
    orig_cls, orig_res, orig_mk = EC.DirectionalSemanticField, EC.residuals_view, EC.make_normlift
    assert orig_cls is DirectionalSemanticField, "EC.DirectionalSemanticField is patched"
    state = dict(ref=None, seg=None)

    class Recorded(orig_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            rec["field"].append(self)

    def make_normlift(ckpt, field, saved):
        nl = orig_mk(ckpt, field, saved)
        rec["nl"].append(nl)
        return nl

    def residuals(cb, ras, v, z, ren, gz, kw, observed):
        per = orig_res(cb, ras, v, z, ren, gz, kw, observed)
        rec["t"] += 1
        t = rec["t"]
        field, nl = rec["field"][-1], rec["nl"][-1]
        seg = state["seg"]
        if seg is None:
            raise RuntimeError("the segment was not set")
        cov = observed
        nan = torch.full((int(cov.numel()),), float("nan"), device=cov.device)
        if t == seg["start"] and seg["length"]:            # EC set its reference just now
            state["ref"] = nl.reliability().clone()
        drop = (torch.where(cov, state["ref"] - nl.reliability(), nan)
                if state["ref"] is not None and t > seg["start"] else nan)
        comp = dict(ours=field_score(field), feat=EC.view_kmin(cb, "feat", per),
                    depth=EC.view_kmin(cb, "depth_render", per), normlift=drop)
        idx = torch.nonzero(cov).squeeze(1)
        out = dict(t=t, in_segment=bool(t > seg["start"] and seg["length"]))
        for name, x in comp.items():
            x = nan if x is None else x
            out[name] = x[idx].float().cpu()
        rec["views"].append(out)
        return per

    EC.DirectionalSemanticField = Recorded
    EC.residuals_view = residuals
    EC.make_normlift = make_normlift
    try:
        yield rec, state
    finally:
        EC.DirectionalSemanticField = orig_cls
        EC.residuals_view = orig_res
        EC.make_normlift = orig_mk
    assert (EC.DirectionalSemanticField is orig_cls and EC.residuals_view is orig_res
            and EC.make_normlift is orig_mk)


def null_identity(rec, null):
    """The captured component samples equal the replay's own nulls."""
    out = {}
    for name, row in (("feat", "feat"), ("depth", "depth"), ("normlift", "normlift")):
        mine = torch.cat([v[name] for v in rec["views"]])
        mine = torch.sort(mine[torch.isfinite(mine)]).values
        parts = [p for part in null[row].parts for p in part]
        theirs = torch.sort(torch.cat(parts)).values if parts else torch.zeros(0)
        out[name] = bool(mine.numel() == theirs.numel() and torch.equal(mine, theirs.float()))
    field = rec["field"][-1]
    n_ours = sum(int(torch.isfinite(v["ours"]).sum()) for v in rec["views"])
    out["ours"] = n_ours == int(round(float(field.c_hist.sum())))
    out["n"] = dict(ours=n_ours, **{k: int(sum(int(torch.isfinite(v[k]).sum())
                                                for v in rec["views"]))
                                     for k in ("feat", "depth", "normlift")})
    return out


def variant_thresholds(rec, ranks, q):
    """{V: dict(thr, n)}: the q-quantile of s_V over the mapping replay's covered
    Gaussian-views (V3: the NormLift segment views only), float64."""
    out = {}
    for v, names in VARIANTS.items():
        vals = []
        for view in rec["views"]:
            if "normlift" in names and not view["in_segment"]:
                continue
            s = fused_stat(ranks, {n: view[n] for n in names}, names)
            vals.append(s[torch.isfinite(s)].cpu().numpy())
        a = np.concatenate(vals) if vals else np.zeros(0)
        # eval_change_nor.Null.threshold's rule (numpy linear quantile), kept in float64
        out[v] = dict(thr=float(np.quantile(a, q)) if a.size else float("nan"), n=int(a.size),
                      views="NormLift segment" if "normlift" in names else "all mapping")
    return out


# ------------------------------------------------------------------------ variant pass
def run_variant(ctx, views, k, q, decide, need_normlift):
    """One update pass of our field with an external firing rule.
    decide(field, comps, cov) -> (fire bool (N,), stat float (N,) NaN = not tested)
    comps: per-view components ours / feat / depth (/ normlift). Returns the per-Gaussian
    record (CPU)."""
    ckpt, dev, N = ctx["ckpt"], ctx["device"], ctx["N"]
    f = ExternalResetField.from_state(ckpt["field"], device=dev)
    robust_map = f.robust_mask().cpu()
    f.calibrate_change_threshold(q)
    f.disable_own_test()
    resets0 = f.n_resets.clone()
    disp = nl = r_map = None
    if need_normlift:
        disp = DirectionalSemanticField.from_state(ckpt["field"], device=dev)
        disp.cfg.change_test = False
        nl = EC.make_normlift(ckpt, disp, saved=True)
        r_map = nl.reliability().clone()
    cb = ChangeBaselines(N, k=k, view_mass_frac=f.cfg.view_mass_frac, device=dev)
    zeros = lambda dtype: torch.zeros(N, dtype=dtype, device=dev)
    observed, flags = zeros(torch.bool), zeros(torch.bool)
    tests, fires = zeros(torch.int32), zeros(torch.int32)
    smax = torch.full((N,), float("nan"), dtype=torch.float64, device=dev)
    covs = []
    for v in views:
        z, ras, gz, kw = EC.view_inputs(ctx, v)
        ren = EC.map_render(ras, gz, ctx["frozen"])
        stats = f.update_view(ras, z, num_pixels=v["h"] * v["w"], **kw)
        cov = EC.covered(f, stats)
        observed |= cov
        covs.append(cov.cpu())
        if disp is not None:
            disp.update_view(ras, z, num_pixels=v["h"] * v["w"], **kw)
        per = EC.residuals_view(cb, ras, v, z, ren, gz, kw, cov)
        nan = torch.full((N,), float("nan"), device=dev)
        comps = dict(ours=field_score(f), feat=EC.view_kmin(cb, "feat", per),
                     depth=EC.view_kmin(cb, "depth_render", per))
        comps = {c: (nan if x is None else x) for c, x in comps.items()}
        if nl is not None:
            comps["normlift"] = torch.where(cov, r_map - nl.reliability(), nan)
        fire, stat = decide(f, comps, cov)
        fire = fire & cov
        tested = torch.isfinite(stat) & cov
        tests += tested.int()
        fires += fire.int()
        flags |= fire
        smax = torch.where(tested, torch.fmax(smax, stat.double()), smax)
        f.external_reset(stats, fire, gauss_depth=kw.get("gauss_depth"))
    if not torch.equal(flags, f.n_resets > resets0):
        raise RuntimeError("flags differ from resets")
    return dict(score=torch.where(observed, smax, torch.full_like(smax, float("nan"))).cpu(),
                flags=flags.cpu(), tests=tests.cpu(), fires=fires.cpu(),
                observed=observed.cpu(), covs=covs, robust_end=f.robust_mask().cpu(),
                robust_map=robust_map, changed_end=f.changed.cpu(),
                n_resets=int((f.n_resets - resets0).sum()), mu=f.mu_post().float().cpu())


def fused_decider(ranks, names, thr):
    """V's rule: s_V on the covered Gaussians; fire where s_V > thr and not changed."""
    def decide(f, comps, cov):
        s = fused_stat(ranks, comps, names, mask=cov)
        fire = torch.isfinite(s) & (s > thr) & ~f.changed
        return fire, s
    return decide


def native_decider(thr_by_hist):
    """Our own rule through the external path (identity checks): c_p > the calibrated
    threshold of the Gaussian's history bucket, where the test was defined."""
    def decide(f, comps, cov):
        hb = torch.bucketize(f.ns, f._hist_edges)
        c = comps["ours"]
        fire = torch.isfinite(c) & (c > thr_by_hist.to(c.device)[hb])
        return fire, c
    return decide


# ------------------------------------------------------------------------------- metrics
def share(mask):
    return float(mask.double().mean()) if mask.numel() else float("nan")


def static_relabel(feat, lab0, sp, text):
    lab = DQ.top_class(feat, text)
    return share(lab[sp] != lab0[sp])


# ---------------------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", required=True, help="the scene's Table III record (JSON)")
    ap.add_argument("--tolerance", default="", help="plot_nor_qual.py noise tolerance JSON")
    ap.add_argument("--vocab", required=True, help="configs/vocab_replica.json")
    ap.add_argument("--text-cache", required=True, help="a COPY of the LSeg text cache (read only)")
    ap.add_argument("--dump", default="", help="the scene's dump_nor_qual.py output directory "
                                               "(ours' stale-answer rate cross-check)")
    ap.add_argument("--prereg", required=True, help="configs/reset_fused_prereg.json")
    ap.add_argument("--stage", choices=["val", "test", "toy"], required=True)
    ap.add_argument("--smoke", type=int, default=0,
                    help="evenly spaced subset of N update keyframes; no reproduction check")
    ap.add_argument("--backend", choices=["gsplat", "mock"], default="gsplat")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="output JSON")
    return ap


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    DQ.write_json(path, obj)


def run(args, argv):
    t_start = time.time()
    if os.path.exists(args.out):
        old = json.load(open(args.out))
        if old.get("status") == "done" and old.get("stage") == "test":
            raise SystemExit(f"{args.out}: this test scene was run already (run once)")
    prereg = json.load(open(args.prereg))
    q = float(prereg["threshold"]["quantile"])
    if prereg["variants"] != {v: list(c) for v, c in VARIANTS.items()}:
        raise SystemExit("the prereg's variants differ from usplat/reset_fused.py VARIANTS")
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
               smoke=args.smoke)
    write_json(args.out, out)
    code_now, rm_now = EC.code_state(), EC.RM.mapping_code_state()["mapping_code_sha256"]
    out["frozen_code"] = dict(eval_code_equal=code_now == m["provenance"].get("eval_code"),
                              mapping_code_equal=rm_now == m["provenance"].get("mapping_code_now"))
    if args.stage != "toy" and not all(out["frozen_code"].values()):
        raise SystemExit(f"the frozen code differs from the record's: {out['frozen_code']}")
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
    miss = DQ.missing_prompts(vocab, targs)
    if miss:
        raise SystemExit(f"{args.text_cache} lacks {len(miss)} prompts (fill a copy with "
                         "dump_nor_qual.py --text-only)")
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
    e = EC._normlift_entry(ckpt)
    cfg_nl = dict(e["config"])
    dtype_nl = getattr(torch, cfg_nl.pop("dtype", "float32"))
    from usplat import baselines as B
    nl0 = B.NormLift(**cfg_nl, device=dev, dtype=dtype_nl)
    nl0.load_state_dict(e["state"])
    r_map0 = nl0.reliability().clone()
    del nl0
    mu0 = ctx["frozen"]["mu"]

    # ---- mapping replay with the component capture
    k = int(ns.k or ckpt["field"]["config"]["change_persist"])
    t0 = time.time()
    seg = EC.normlift_segment(len(ctx["map_views"]), len(ctx["upd_views"]))
    with capture_replay(len(ctx["upd_views"])) as (rec, state):
        state["seg"] = seg
        replay, null, seg_ec = EC.replay_mapping(ctx, ctx["map_views"], k, len(ctx["upd_views"]))
    if seg_ec != seg:
        raise AssertionError(f"segment {seg} != replay's {seg_ec}")
    check = EC.replay_check(replay, ckpt["field"])
    if check["max_rel_dW1"] > 1e-2:
        raise SystemExit(f"the mapping replay does not reproduce the saved field: {check}")
    ident = null_identity(rec, null)
    del replay
    rec["field"].clear(); rec["nl"].clear()
    if not all(ident[c] for c in COMPONENTS):
        raise AssertionError(f"captured mapping samples differ from the replay nulls: {ident}")
    thr = {r: nl.threshold(q) for r, nl in null.items()}
    del null
    samples = {c: torch.cat([v[c] for v in rec["views"]]) for c in COMPONENTS}
    seg_views = [v for v in rec["views"] if v["in_segment"]]
    samples["normlift"] = (torch.cat([v["normlift"] for v in seg_views]) if seg_views
                           else torch.zeros(0))
    ranks = MappingRanks(samples)
    vthr = variant_thresholds(rec, ranks, q)
    del rec, samples
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
    free_gpu()            # capture_fields' subclass closes over its list: a reference cycle
    T["ours_s"] = time.time() - t0

    # ---- Table III reproduction (stop rule)
    t0 = time.time()
    gt, _, _, _ = EC.ground_truth(ctx, ns)
    stale, pop, seen = gt["stale"], upd["observed"], upd["seen"]
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
               null_identity=ident, reproduction=rep, gpu_reserved_mib=held)
    if rep is not None and not rep["ok"]:
        out.update(status="FAIL", wall_s=time.time() - t_start)
        write_json(args.out, out)
        print(f"{m['scene']}: REPRODUCTION FAILED: {rep['failures']}")
        return dict(out=out, exit=2)

    # ---- the variants
    res, cov_check = {}, {}
    for v, names in VARIANTS.items():
        t0 = time.time()
        free_gpu()
        r = run_variant(ctx, ctx["upd_views"], k, q, fused_decider(ranks, names, vthr[v]["thr"]),
                        need_normlift="normlift" in names)
        n_diff = int((r["observed"] != pop.cpu()).sum())
        cov_check[v] = dict(n_differ=n_diff, share=n_diff / max(int(pop.sum()), 1))
        if cov_check[v]["share"] > COV_TOL:        # GPU sums are not bit-exact (atomics)
            raise AssertionError(f"{v}: coverage differs from ours': {cov_check[v]}")
        feats[v] = r.pop("mu")
        r.pop("covs")
        res[v] = r
        T[f"{v}_s"] = time.time() - t0

    # ---- metrics of ours and the variants
    t0 = time.time()
    st, sp = pop & stale, pop & ~stale
    scores = dict(ours=scores3["ours"], **{v: res[v]["score"].float() for v in VARIANTS})
    flags = dict(ours=upd["flags"]["ours"], **{v: res[v]["flags"] for v in VARIANTS})
    tests = dict(ours=upd["tests"]["ours"], **{v: res[v]["tests"] for v in VARIANTS})
    fires = dict(ours=upd["fires"]["ours"], **{v: res[v]["fires"] for v in VARIANTS})
    robust = dict(ours=upd["robust_end"], **{v: res[v]["robust_end"] for v in VARIANTS})
    gauss, gauss_common = EC.ranking_table({r: scores[r][pop] for r in ROWS}, y, ROWS)
    pixel = EC.pixel_level(ctx, ctx["upd_views"], scores, ROWS, ns.tau_cov, ns.undef_frac)
    rules = dict(EC.RULES)
    for v in VARIANTS:
        rules[v] = (f"s_{v} (mapping-ranked fusion of {'+'.join(VARIANTS[v])}) > the 0.999 "
                    f"mapping quantile, not changed; reset")
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
    query = dict(index=qi, name=vocab["names"][qi], share=float(counts[qi]) / total,
                 n_stale_seen=total)
    G = dict(flags=dict(ours=flags["ours"]), feat=feats)
    saved_sets = DQ.SETS
    DQ.SETS = N2_SETS
    try:
        con, _ = DQ.contrast_and_fa(ctx, ctx["upd_views"], G, text, qi, ns.tau_cov,
                                    ns.undef_frac)
    finally:
        DQ.SETS = saved_sets
    assert DQ.SETS is saved_sets
    T["n2_s"] = time.time() - t0
    dump_check = None
    if args.dump:
        dc = json.load(open(os.path.join(args.dump, "contrast.json")))["sets"]
        dump_check = {s: dict(rerun=con[s]["stale_answer_rate"],
                              dump=float(dc[s]["stale_answer_rate"]))
                      for s in ("ours", "nochange", "avg_after", "before")}
        for s, d in dump_check.items():
            d["delta"] = abs(d["rerun"] - d["dump"])
        worst = max(d["delta"] for d in dump_check.values())
        dump_check["ok"] = worst <= STALE_TOL
        if not dump_check["ok"] and not args.smoke:
            out.update(status="FAIL", dump_check=dump_check, wall_s=time.time() - t_start)
            write_json(args.out, out)
            print(f"{m['scene']}: N2 does not reproduce the dump: {dump_check}")
            return dict(out=out, exit=2)

    rows = {}
    for r in ROWS:
        rows[r] = dict(gauss_ap=gauss[r]["ap"], gauss_auroc=gauss[r]["auroc"],
                       gauss_n_nan=gauss[r]["n_nan"], gauss_common_ap=gauss[r]["common"]["ap"],
                       pixel_ap=pixel["rows"][r]["ap"], pixel_auroc=pixel["rows"][r]["auroc"],
                       fa=ops[r]["fa"], rec=ops[r]["rec"], fa_common=ops[r]["fa_common"],
                       rec_common=ops[r]["rec_common"], rule=ops[r]["rule"], **rates[r],
                       kept=share(robust[r][st]), n_resets=(upd["n_resets"] if r == "ours"
                                                           else res[r]["n_resets"]),
                       stale_answer=con[r]["stale_answer_rate"], delta_cos=con[r]["delta_cos"],
                       n_mask_cov=con[r]["n_mask_cov"],
                       static_relabel=static_relabel(feats[r], lab0, sp, text))
        if r in VARIANTS:
            rows[r].update(threshold=vthr[r]["thr"], n_null=vthr[r]["n"],
                           null_views=vthr[r]["views"], components=list(VARIANTS[r]))
    context = {s: dict(stale_answer=con[s]["stale_answer_rate"], delta_cos=con[s]["delta_cos"],
                       static_relabel=static_relabel(feats[s], lab0, sp, text))
               for s in CONTEXT_SETS}
    peak = (torch.cuda.max_memory_allocated() / 2 ** 20 if str(dev).startswith("cuda")
            and torch.cuda.is_available() else None)
    out.update(status="done", rows=rows, context=context, query=query, dump_check=dump_check,
               gauss_common=gauss_common, pixel_common=pixel["common"],
               component_null_n={c: ranks.n(c) for c in COMPONENTS}, coverage_check=cov_check,
               ours_threshold_by_history=thr_ours_hist.cpu().tolist(),
               counts=dict(n_gaussians=N, n_observed=int(pop.sum()),
                           n_stale_observed=int(st.sum()), n_static_observed=int(sp.sum())),
               timing=T, wall_s=time.time() - t_start, peak_alloc_mib=peak)
    write_json(args.out, out)
    for r in ROWS:
        x = rows[r]
        print(f"{m['scene']} {r:>4}: G AP {x['gauss_ap']:.4f} P AP {x['pixel_ap']:.4f} | FA "
              f"{x['fa']:.4f} Rec {x['rec']:.4f} Kept {x['kept']:.4f} | stale-answer "
              f"{x['stale_answer']:.4f} static relabel {x['static_relabel']:.4f}")
    print(f"{m['scene']}: done, wall {out['wall_s']:.1f} s, peak {peak} MiB; wrote {args.out}")
    return dict(out=out, exit=0)


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
