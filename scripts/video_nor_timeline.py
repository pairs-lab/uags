"""
video_nor_timeline.py — the object-removal timeline of the supplementary video: the Table III
(tab:change) update of one NeRF-Object-Removal scene, replayed keyframe by keyframe, with the
map read out after every update keyframe from ONE fixed camera (the scene's rule frame of the
NOR figures, e.g. 002 -> update frame 373).

    UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5 scripts/gpu_gate.sh 6500 python3 \
        scripts/video_nor_timeline.py --table <results>/nor/tab2_relmax/val/002.json \
        --dump <results>/qual/nor/002 --tolerance <results>/qual/nor/noise/tolerance.json \
        --vocab configs/vocab_replica.json --text-cache <results>/video/text_cache_lseg_nor.pt \
        --betas <results>/tab1/eval_val.json --out <results>/video/nor/002

No new replay code: the passes are those of scripts/dump_nor_qual.py (imported, never edited),
which runs eval_change_nor.py's own load_context / replay_mapping / replay_check / run_update /
ground_truth / ranking_table / operating_points with every setting from the scene's Table III
record. The only addition is dump_nor_qual.capture_fields(per_view=...): after every update
keyframe (run_update calls residuals_view once per view, after ours and the change-test-off
copy with the saved NormLift took it), the hook READS the two fields (no state is written)
and renders, at the fixed camera, at the full 720x540 resolution:
  * p(query) of ours (P0 render of mu*) and of averaging (NormLift.feature(), the same
    feature averaging uses), softmax over the 101 classes with the Replica validation
    temperatures (as the NOR figures: dump_nor_qual.class_maps);
  * whether the top class is the query, per pixel;
  * the rendered share of the Gaussians our test has reset so far (n_resets above the saved
    field's count; at the end this is Table III's 'flagged' set) and of those changed now.
It also keeps per update keyframe, per Gaussian, the top class of ours and of averaging and
the reset bits, from which the curves are computed after the ground truth (D2) is known.

Query and frame are those of the existing NOR dump (--dump, computed there by rule) and are
re-derived here: the frame by dump_nor_qual.pick_frame on the record's keyframes.

Checks (exit 2 on failure, nothing shown from a failed run):
  * keyframes equal the record's; replay_check ok (as dump_nor_qual)
  * Gaussian-level AP/AUROC of every Table III row within the tolerance file's value; ours'
    FA, Rec, Kept and changed_end (FA, Rec) within 0.5 points of the record
  * the end state equals the NOR dump bit for bit (flags, changed_end, robust_end, observed,
    stale) and the end/before renders equal its frame npz (p(query), max abs diff <= 2e-3,
    float16 storage)
  * --curves-ref (the dump's curves.json, 002 only): the per-keyframe stale-is-query curves
    are equal
Outputs in --out: timeline.npz (per keyframe maps, photos, masks), curves.json, meta.json.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import dump_nor_qual as DNQ                                                 # noqa: E402
from usplat.change_baselines import fuse_rank                               # noqa: E402
from usplat.datasets import _imread                                         # noqa: E402
from usplat.vmf_field import DirectionalSemanticField                       # noqa: E402

EC, ES = DNQ.EC, DNQ.ES
POINTS = DNQ.POINTS            # 0.5 points: ours' operating point, Kept, changed_end
PQ_TOL = 2e-3                  # renders vs the dump's float16 frame maps


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="the scene's Table III record (JSON)")
    ap.add_argument("--dump", required=True, help="the scene's NOR dump (dump_nor_qual.py)")
    ap.add_argument("--tolerance", required=True, help="plot_nor_qual.py noise output")
    ap.add_argument("--vocab", required=True, help="configs/vocab_replica.json")
    ap.add_argument("--text-cache", required=True, help="a COPY of the text cache")
    ap.add_argument("--betas", required=True, help="tab1/eval_val.json (as the NOR dump)")
    ap.add_argument("--curves-ref", default="", help="the dump's curves.json, if it has one")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0, help="(tests) first n update keyframes")
    ap.add_argument("--backend", choices=["gsplat", "mock"], default="gsplat")
    ap.add_argument("--device", default="cuda")
    return ap


def share_render(ras, x, ones):
    """Rendered alpha-share of a per-Gaussian value x (N,) and the alpha (P,)."""
    out = ras(torch.stack([x.float(), ones], 1))
    a = out[:, 1]
    return torch.where(a > EC.ALPHA_EPS, out[:, 0] / a.clamp_min(EC.ALPHA_EPS),
                       torch.full_like(a, float("nan"))), a


@torch.no_grad()
def render_state(ras, h, w, Fo, Fa, flag, changed, text, qi, betas, tau_cov):
    """The fixed camera's maps of one state, numpy (h, w)."""
    N = Fo.shape[0]
    ones = torch.ones(N, device=Fo.device)
    fl, alpha = share_render(ras, flag, ones)
    ch, _ = share_render(ras, changed, ones)
    cov = DNQ._covered(alpha, tau_cov)
    out = dict(alpha=alpha, covered=cov, flag=fl, changed=ch)
    for name, F, b in (("ours", Fo, "ours_P0"), ("avg", Fa, "averaging")):
        pred, _, pq = DNQ.class_maps(ras(F.float()), cov, text, qi, betas[b])
        out[f"pq_{name}"], out[f"isq_{name}"] = pq, (pred == qi)
    return {k: v.reshape(h, w).cpu().numpy() for k, v in out.items()}


def share(mask):
    n = int(mask.numel())
    return float(mask.double().mean()) if n else float("nan")


def run(args, argv):
    t_start = time.time()
    m = DNQ.record_body(json.load(open(args.table)))
    dmeta = json.load(open(os.path.join(args.dump, "meta.json")))
    if dmeta.get("status") != "done" or dmeta.get("scene") != m["scene"]:
        raise SystemExit(f"{args.dump}: not a finished dump of scene {m['scene']}")
    fi, qi = int(dmeta["selection"]["index"]), int(dmeta["query"]["index"])
    vocab = ES.load_vocab(args.vocab)
    if vocab["names"][qi] != dmeta["query"]["name"]:
        raise SystemExit("the vocabulary differs from the dump's")
    tol = float(json.load(open(args.tolerance))["tolerance"])
    ES.check_fit_settings(ES.load_fit_settings(args.betas),
                          argparse.Namespace(undef_frac=m["settings"]["undef_frac"],
                                             tau_cov=m["settings"]["tau_cov"]), args.betas)
    betas_all = ES.load_betas(args.betas)
    betas = {k: float(betas_all[k]) for k in ("ours_P0", "averaging")}
    if args.backend != "mock" and betas != dmeta["betas"]:
        raise SystemExit(f"betas {betas} differ from the dump's {dmeta['betas']}")
    ns = DNQ.ec_namespace(m, args)
    meta = dict(scene=m["scene"], status="running", argv=argv, start=t_start,
                script_sha256=DNQ.sha256_file(os.path.abspath(__file__)),
                dump_script_sha256=DNQ.sha256_file(os.path.join(HERE, "dump_nor_qual.py")),
                table_sha256=DNQ.sha256_file(args.table), tolerance=tol, betas=betas,
                frame=fi, query=dmeta["query"])
    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, "meta.json")
    DNQ.write_json(meta_path, meta)
    code_now = EC.code_state()
    rm_now = EC.RM.mapping_code_state()["mapping_code_sha256"]
    meta["code_equal"] = dict(eval=code_now == m["provenance"].get("eval_code"),
                              mapping=rm_now == m["provenance"].get("mapping_code_now"))
    if args.backend != "mock" and not all(meta["code_equal"].values()):
        raise SystemExit(f"the frozen code differs from the record's: {meta['code_equal']}")
    if DNQ.sha256_file(ns.field) != m["provenance"]["field"]["sha256"]:
        raise SystemExit("the field's sha256 differs from the record's")
    dev = args.device
    if str(dev).startswith("cuda"):
        from usplat import gpu_reserve
        meta["gpu_reserved_mib"] = gpu_reserve.apply(dev)

    ctx = EC.load_context(ns)
    ckpt, N, q = ctx["ckpt"], ctx["N"], ns.change_quantile
    views = dict(mapping=[v["index"] for v in ctx["map_views"]],
                 update=[v["index"] for v in ctx["upd_views"]])
    if not args.smoke and views != {k: m["views"][k] for k in ("mapping", "update")}:
        raise SystemExit("keyframes differ from the record's")
    targs = DNQ.text_args(ctx["cfg"], args.text_cache)
    if DNQ.missing_prompts(vocab, targs):
        raise SystemExit(f"{args.text_cache} lacks prompts (dump_nor_qual.py --text-only)")
    cache_sha = DNQ.sha256_file(args.text_cache)
    text = ctx["pca"].project_text(
        ES.text_embeddings(vocab, list(range(len(vocab["prompts"]))), targs).to(dev))
    if DNQ.sha256_file(args.text_cache) != cache_sha:
        raise SystemExit("the text cache changed during the run")

    # ---- the fixed camera: the rule frame, re-derived
    by_index = {v["index"]: v for v in ctx["upd_views"]}
    rule_n_pos = {f["index"]: f["n_pos"] for f in m["pixel"]["frames"]}
    if not args.smoke:
        fr, _ = DNQ.pick_frame(views["update"], lambda i: (
            None if by_index[i]["mask_real"] is None else by_index[i]["mask_real"].numpy()),
            rule_n_pos)
        if fr != fi:
            raise SystemExit(f"rule frame {fr} != the dump's {fi}")
    ds = ctx["map_ds"]
    H, W = ds.image_hw
    v_fix = by_index[fi]
    viewmat = torch.linalg.inv(v_fix["c2w"].double()).float().to(dev)
    Kf = torch.from_numpy(np.asarray(ds.K[fi])).float().to(dev)
    ras = ctx["rasterizer"](ctx["splats"], viewmat, Kf, int(W), int(H))

    # ---- before the update (saved field, saved NormLift), as dump_nor_qual
    f0 = DirectionalSemanticField.from_state(ckpt["field"], device=dev)
    r0 = f0.n_resets.clone()
    zero = torch.zeros(N, device=dev)
    e = EC._normlift_entry(ckpt)
    cfg_nl = dict(e["config"])
    dtype_nl = getattr(torch, cfg_nl.pop("dtype", "float32"))
    nl0 = DNQ.B.NormLift(**cfg_nl, device=dev, dtype=dtype_nl)
    nl0.load_state_dict(e["state"])
    r_map0 = nl0.reliability().clone()
    before = render_state(ras, H, W, f0.mu_post(), nl0.feature(), zero, zero, text, qi, betas,
                          ns.tau_cov)
    tc0 = (DNQ.top_class(f0.mu_post(), text), DNQ.top_class(nl0.feature(), text))
    del f0, nl0

    k = int(ns.k or ckpt["field"]["config"]["change_persist"])
    t0 = time.time()
    replay, null, _ = EC.replay_mapping(ctx, ctx["map_views"], k, len(ctx["upd_views"]))
    check = EC.replay_check(replay, ckpt["field"])
    del replay
    if check["max_rel_dW1"] > 1e-2:
        raise SystemExit(f"the mapping replay does not reproduce the saved field: {check}")
    thr = {r: nl.threshold(q) for r, nl in null.items()}
    del null
    meta["replay_check"] = check
    T = dict(replay_s=time.time() - t0)

    steps = []

    def per_view(made):
        ours, disp = made[0], made[1]
        nlv = disp.view_hooks[0].__self__
        flag = ours.n_resets > r0
        maps = render_state(ras, H, W, ours.mu_post(), nlv.feature(), flag, ours.changed,
                            text, qi, betas, ns.tau_cov)
        steps.append(dict(maps=maps, flag=flag.cpu(), changed=ours.changed.cpu(),
                          tc=(DNQ.top_class(ours.raw.R, text), DNQ.top_class(nlv.F, text))))

    t0 = time.time()
    with DNQ.capture_fields(per_view) as made:
        upd = EC.run_update(ctx, ctx["upd_views"], k, thr, q)
    ours, disp, nl, cap = DNQ.check_capture(made, upd, r_map0)
    end_feats = (ours.mu_post(), nl.feature())
    end_tc = (DNQ.top_class(end_feats[0], text), DNQ.top_class(end_feats[1], text))
    del ours, disp, nl, made
    T["update_and_renders_s"] = time.time() - t0
    if len(steps) != len(ctx["upd_views"]):
        raise AssertionError(f"{len(steps)} hook calls for {len(ctx['upd_views'])} views")

    # ---- reproduction of the Table III record
    t0 = time.time()
    gt, _, _, _ = EC.ground_truth(ctx, ns)
    stale, pop = gt["stale"], upd["observed"]
    scores = dict(upd["scores"])
    scores["fused"] = fuse_rank([scores["ours"], scores["depth"]], mask=pop)
    gauss, _ = EC.ranking_table({r: scores[r][pop] for r in EC.ROWS}, stale[pop], EC.ROWS)
    common = pop.clone()
    for r in EC.TESTED:
        common &= upd["tests"][r] > 0
    ops = EC.operating_points(upd["flags"], stale, pop, common)
    st, sp = pop & stale, pop & ~stale
    mine = dict(fa=ops["ours"]["fa"], rec=ops["ours"]["rec"], kept=share(upd["robust_end"][st]),
                changed_fa=share(upd["changed_end"][sp]), changed_rec=share(upd["changed_end"][st]))
    rec = dict(fa=m["operating_point"]["ours"]["fa"], rec=m["operating_point"]["ours"]["rec"],
               kept=m["ours"]["kept"], changed_fa=m["ours"]["changed_end"]["fa"],
               changed_rec=m["ours"]["changed_end"]["rec"])
    fails = []
    op = {key: dict(rerun=mine[key], record=rec[key], delta=abs(mine[key] - rec[key]))
          for key in mine}
    if not args.smoke:
        fails += [f"{key}: {d}" for key, d in op.items() if d["delta"] > POINTS]
        cells = {}
        for r in EC.ROWS:
            for met in ("ap", "auroc"):
                a, b = float(gauss[r][met]), float(m["gaussian"]["rows"][r][met])
                cells[f"{r}/{met}"] = abs(a - b)
                if not abs(a - b) <= tol:
                    fails.append(f"gaussian {r} {met}: {a} vs {b}")
        meta["gaussian_max_delta"] = max(cells.values())
    # ---- equality with the NOR dump's end state and renders
    G = torch.load(os.path.join(args.dump, "gauss.pt"), map_location="cpu",
                   weights_only=False, mmap=True)
    eq = dict(flags=torch.equal(G["flags"]["ours"], upd["flags"]["ours"]),
              changed_end=torch.equal(G["changed_end"], upd["changed_end"]),
              robust_end=torch.equal(G["robust_end"], upd["robust_end"]),
              observed=torch.equal(G["observed"], pop), stale=torch.equal(G["stale"], stale),
              last_hook_flags=torch.equal(steps[-1]["flag"], upd["flags"]["ours"]),
              last_hook_changed=torch.equal(steps[-1]["changed"], upd["changed_end"]),
              top_class_ours=torch.equal(G["top_class"]["ours"], end_tc[0]),
              top_class_avg=torch.equal(G["top_class"]["avg_after"], end_tc[1]),
              top_class_before=torch.equal(G["top_class"]["before"], tc0[0]),
              top_class_avg_before=torch.equal(G["top_class"]["avg_before"], tc0[1]))
    del G
    fz = np.load(os.path.join(args.dump, f"frame_{fi}.npz"))
    diff = lambda a, b: float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    nan_same = lambda a, b: bool(np.array_equal(np.isnan(a.astype(np.float64)),
                                                np.isnan(b.astype(np.float64))))
    last = steps[-1]["maps"]
    pqd = dict(ours_end=diff(last["pq_ours"], fz["full_pq/ours"]),
               avg_end=diff(last["pq_avg"], fz["full_pq/avg_after"]),
               ours_before=diff(before["pq_ours"], fz["full_pq/before"]),
               avg_before=diff(before["pq_avg"], fz["full_pq/avg_before"]),
               flag_end=diff(last["flag"], fz["full_share/flag_ours"]))
    nan_eq = dict(ours_end=nan_same(last["pq_ours"], fz["full_pq/ours"]),
                  avg_end=nan_same(last["pq_avg"], fz["full_pq/avg_after"]))
    if not args.smoke:
        fails += [f"dump equality: {k_}" for k_, ok in eq.items() if not ok]
        fails += [f"render vs dump {k_}: {d}" for k_, d in pqd.items() if not d <= PQ_TOL]
        fails += [f"render NaN pattern {k_}" for k_, ok in nan_eq.items() if not ok]
    T["checks_s"] = time.time() - t0

    # ---- curves (Gaussian level, over the observed Gaussians at the end, as curves.json)
    stq = lambda tc: float((tc[st] == qi).double().mean())
    chg = lambda tc, t0_: float((tc[sp] != t0_[sp]).double().mean())
    curves = dict(views=[0] + views["update"], query=qi, n_stale_observed=int(st.sum()),
                  n_static_observed=int(sp.sum()),
                  ours=dict(stale_is_query=[stq(tc0[0])] + [stq(s["tc"][0]) for s in steps],
                            static_changed=[0.0] + [chg(s["tc"][0], tc0[0]) for s in steps]),
                  averaging=dict(stale_is_query=[stq(tc0[1])] + [stq(s["tc"][1]) for s in steps],
                                 static_changed=[0.0] + [chg(s["tc"][1], tc0[1])
                                                         for s in steps]),
                  reset_stale=[0.0] + [share(s["flag"][st]) for s in steps],
                  reset_static=[0.0] + [share(s["flag"][sp]) for s in steps],
                  changed_stale=[0.0] + [share(s["changed"][st]) for s in steps],
                  changed_static=[0.0] + [share(s["changed"][sp]) for s in steps])
    if args.curves_ref:
        ref = json.load(open(args.curves_ref))
        same = (ref["views"] == curves["views"] and all(
            np.allclose(ref[n]["stale_is_query"], curves[n]["stale_is_query"], atol=1e-12, rtol=0)
            and np.allclose(ref[n]["static_changed"], curves[n]["static_changed"], atol=1e-12,
                            rtol=0) for n in ("ours", "averaging")))
        meta["curves_equal_ref"] = bool(same)
        if not same and not args.smoke:
            fails.append("curves differ from the dump's curves.json")
    meta.update(reproduction=dict(operating_point=op, failures=fails, ok=not fails),
                dump_equal=eq, render_vs_dump=pqd, nan_pattern_equal=nan_eq, capture=cap,
                views=views, n_keyframes=len(steps), image_hw=[int(H), int(W)],
                quarter_turns=int(dmeta["frames"][str(fi)]["quarter_turns"]))
    if fails:
        meta.update(status="FAIL", wall_s=time.time() - t_start)
        DNQ.write_json(meta_path, meta)
        print(f"{m['scene']}: REPRODUCTION FAILED: {fails}")
        return dict(meta=meta, exit=2)

    # ---- photos and masks of every update keyframe (display only)
    photos = np.stack([_imread(ds.rgb[i]) for i in views["update"]])
    masks = np.stack([np.zeros(DNQ.MASK_HW, bool) if by_index[i]["mask_real"] is None
                      else by_index[i]["mask_real"].numpy().astype(bool)
                      for i in views["update"]])
    keys = ("pq_ours", "pq_avg", "isq_ours", "isq_avg", "flag", "changed")
    out = {f"before_{k_}": before[k_] for k_ in keys}
    for k_ in keys:
        a = np.stack([s["maps"][k_] for s in steps])
        out[k_] = a.astype(np.float16) if a.dtype.kind == "f" else a
    out.update(alpha=steps[0]["maps"]["alpha"].astype(np.float16), photos=photos, masks=masks,
               fixed_photo=fz["full_photo"], fixed_map_rgb=fz["full_map_rgb"],
               fixed_mask=fz["mask_real"], update_index=np.array(views["update"]),
               fixed_index=np.array(fi))
    np.savez_compressed(os.path.join(args.out, "timeline.npz"), **out)
    DNQ.write_json(os.path.join(args.out, "curves.json"), curves)
    peak = (torch.cuda.max_memory_allocated() / 2 ** 20 if str(dev).startswith("cuda")
            and torch.cuda.is_available() else None)
    meta.update(status="done", timing=T, wall_s=time.time() - t_start, peak_alloc_mib=peak)
    DNQ.write_json(meta_path, meta)
    print(f"{m['scene']}: done; {len(steps)} keyframes at frame {fi}, query "
          f"{dmeta['query']['name']!r}; FA/Rec/Kept {mine['fa']:.4f}/{mine['rec']:.4f}/"
          f"{mine['kept']:.4f} (record {rec['fa']:.4f}/{rec['rec']:.4f}/{rec['kept']:.4f}); "
          f"render diffs {pqd}; wall {meta['wall_s']:.0f} s, peak {peak} MiB")
    return dict(meta=meta, exit=0)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.backend == "gsplat" and not str(args.device).startswith("cuda"):
        raise SystemExit("--backend gsplat needs --device cuda")
    full = [sys.argv[0]] + list(argv if argv is not None else sys.argv[1:])
    with torch.no_grad():
        out = run(args, full)
    if out.get("exit"):
        raise SystemExit(out["exit"])
    return out


if __name__ == "__main__":
    main()
