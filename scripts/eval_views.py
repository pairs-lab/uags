"""
eval_views.py — the views figure (plan §3, Fig views): how the k-view map's uncertainty
channels rank the errors that more views would fix and the errors they would not.

For every scene and every k, the SAME evaluation pixels are scored twice: with the field built
from k mapping views (run_mapping.py --max-keyframes k, on the same 3DGS map) and with the full
field (all mapping views, the Tab I field). Per row (usplat.seg_eval.views_cross_ause):
    e_red = wrong at k AND right at full     (reducible by more views)
    e_irr = wrong at k AND wrong at full     (irreducible)
and the AUSE of every ranking signal of the k-view row (own channels such as u_epi / u_ale,
plus msp and entropy) against each target, on pixels covered in both maps (holes of the
shared geometry are counted, not scored). Lower AUSE = better ranking.
Temperatures come from an eval_seg.py fit on the full validation maps (--betas) and are the
same for every k. Everything else (vocabulary, text cache, frames, coverage rule) is
eval_seg.py's, which this script imports.

Added analyses (fixed 2026-10-10 before any run with them). Additive JSON keys only:
every key above keeps its meaning and value.
 (a) "reducibility" per row: among the pixels covered in both maps and WRONG at k, how well a
     score separates e_red (right at full, the positive class) from e_irr (wrong at full):
     the AUROC (tie-aware Mann-Whitney, usplat.evaluation.auroc) of every ranking signal of
     the k-view row (own channels, msp, entropy) on the SAME key as the AUSE
     (seg_eval.ranking_signals: an undefined u sits above every finite u, the uncovered tier
     above that), and of "-u_ale", the negated key (a high aleatoric u claims the error is NOT
     reducible; the undefined tier, most uncertain for the AUSE, is then least reducible).
     AUROC(-s) = 1 - AUROC(s); 0.5 = no separation; n_undefined counts the wrong pixels with
     undefined u. The key is turned into dense ranks first (ties kept), so the +inf tier
     stays on top (auroc itself would rank a non-finite score last). With few views the
     undefined tier holds most pixels and decides "-u_ale"; "auroc_defined" {"-u_ale"} is
     the same AUROC on the wrong pixels where u_ale is defined (NaN if none), which reads
     u_ale alone.
 (b) "random_gap" {e_red, e_irr} and "ause_norm" {signal: {target: AUSE / random_gap}} per
     row, also inside on_ale_defined: 0 = the oracle order, 1 = a random order (NaN when the
     gap is 0: no error or only errors). The random gap depends on the errors only, so the
     ratio compares AUSE across k, scenes and error rates.
 (c) "ale_split" per row with a u_ale channel: u_ale's AUSE on the covered-in-both pixels where
     that row's u_ale is defined, and on its undefined tier (one key there, so the AUSE equals
     the tier's random gap), each with n_pixels, share, n_e_red, n_e_irr and the error rates;
     the rates say whether ranking the undefined tier first helps the primary AUSE.
Summaries over scenes: summary_ause_norm, summary_reducibility, summary_ale_split.

PRE-REGISTERED REDUCIBILITY CRITERION (REDUCIBILITY_CRITERION, fixed 2026-10-10 before any
run of (a)): the epistemic-separation claim ("u_epi ranks the errors that more views fix") is
kept only if, at M = 10 mapping views, the reducibility AUROC of u_epi of row ours_P0 is
> 0.60 on BOTH validation scenes (office_0, room_2) AND above the AUROC of the same row's
MSP on both. A NaN AUROC fails. Otherwise the claim is dropped (no other M, row, threshold or
scene set is tried). M is the k-view field label; a field whose recorded mapping keyframes
(n_mapping_views) differ from M leaves its scene missing (undecided). Validation only:
scripts/replica_views.sh scores one scene per run, and
    python3 scripts/eval_views.py verdict <views>/views_val_office_0.json
        <views>/views_val_room_2.json --out <views>/reducibility_val.json      (one line)
merges them and writes the verdict once (an existing verdict file is never overwritten).
Every run also stores "reducibility_criterion" ("undecided" unless both scenes are in it).

Scenes file (JSON/YAML list), as for eval_seg.py plus the k-view fields:
    {name, field: <full field>, fields: {"10": <path>, "30": <path>, "90": <path>}, ply, data,
     dataset, stride}

    cd uags && scripts/gpu_gate.sh 4000 python3 scripts/eval_views.py \
        --vocab configs/vocab_replica.json --extractor lseg --betas <tab1>/eval_val.json \
        --scenes configs/replica_views_val.json --out <out>/views_val.json
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
import eval_seg as ES                                                      # noqa: E402
from usplat.seg_eval import (SegAccumulator, evaluate_view, freeze_rows,   # noqa: E402
                             views_cross_ause, scene_stats, cross_ause, error_targets,
                             ranking_signals, _aligned_keep)
from usplat.evaluation import auroc, sparsification_curve                 # noqa: E402


def accumulate(S, betas, args, rows_wanted):
    raw = ES._raw_probs(args)
    rows = [r for r in S["rows"] if r.name in rows_wanted]
    ES.betas_used(rows, betas, raw)
    fr = freeze_rows(rows)
    acc = SegAccumulator(class_names=S["class_names"])
    for v in ES.scene_views(S, args):
        acc.add(evaluate_view(fr, v.rasterizer, v.text, v.gt, betas, args.tau_cov,
                              undef_frac=args.undef_frac, raw_probs=raw))
    return acc


def row_names(S, args):
    names = [r.name for r in S["rows"]]
    if not args.rows:
        return names
    want = [s.strip() for s in args.rows.split(",") if s.strip()]
    miss = [w for w in want if w not in names]
    if miss:
        raise ValueError(f"unknown rows {miss}; available {names}")
    return want


REF_ROW, REF_CHANNEL = "ours_P0", "u_ale"


def on_defined(res, acc_k, acc_full, rows, args):
    """Secondary analysis (added after the validation run, labelled as such in the paper):
    every row's signals scored on the SAME pixels, those covered in both maps where our
    u_ale is defined (support >= tau_min views). The primary AUSE ranks an undefined u_ale as
    the most uncertain (the rule fixed before any run), which mixes low evidence into the
    aleatoric ranking; this asks how the channels rank the errors once u_ale exists."""
    if REF_ROW not in rows:
        return
    ref, ref_full = acc_k.arrays(REF_ROW), acc_full.arrays(REF_ROW)
    if REF_CHANNEL not in ref["u"]:
        return
    keep, _ = _aligned_keep(ref, ref_full, True)
    sub = keep & torch.isfinite(ref["u"][REF_CHANNEL])
    if not bool(sub.any()):                   # u_ale undefined everywhere (e.g. 1 view)
        for r in rows:
            res[r]["on_ale_defined"] = dict(ause={}, n_pixels=0, share=0.0, n_e_red=0, n_e_irr=0,
                                            random_gap={}, ause_norm={})
        return
    for r in rows:
        ak, af = acc_k.arrays(r), acc_full.arrays(r)
        t = error_targets(ak["correct"][sub], af["correct"][sub])
        sig = ranking_signals(ak)
        gaps = random_gaps(t, args.ause_bins)
        res[r]["on_ale_defined"] = dict(
            ause=cross_ause({s: v[sub] for s, v in sig.items()}, t, args.ause_bins),
            n_pixels=int(sub.sum()), share=float(sub.sum()) / max(int(keep.sum()), 1),
            n_e_red=int(t["e_red"].sum()), n_e_irr=int(t["e_irr"].sum()), random_gap=gaps)
        res[r]["on_ale_defined"]["ause_norm"] = normalised(res[r]["on_ale_defined"]["ause"], gaps)


# ------------------------------------------------------- added analyses (a), (b), (c)
TARGETS = ("e_red", "e_irr")
NEGATED = (REF_CHANNEL,)             # high aleatoric u = the error is NOT reducible: -u
# Pre-registered (2026-10-10, before any run of the reducibility AUROC); see the header.
REDUCIBILITY_CRITERION = dict(row=REF_ROW, score="u_epi", against="msp", k=10, auroc_min=0.60,
                              scenes=("office_0", "room_2"), stage="val")
_NAN = float("nan")


def _rank_key(key):
    """Dense ranks of a ranking key (ties kept, the +inf tier on top), finite so it can be
    negated and given to usplat.evaluation.auroc (which ranks non-finite scores last)."""
    if bool(torch.isnan(key).any()):
        raise ValueError("ranking keys have no NaN (seg_eval.ranking_signals tiers them)")
    return torch.unique(key, sorted=True, return_inverse=True)[1].double()


def reducibility_auroc(out_k, out_full, covered_only=True):
    """(a) Among the kept pixels (covered in both maps by default) WRONG at k: the AUROC of
    every ranking signal of the k-view row for e_red (right at full, positive) against e_irr
    (wrong at full), on the AUSE key, plus the negated key of each NEGATED channel."""
    keep, _ = _aligned_keep(out_k, out_full, covered_only)
    wrong = keep & ~out_k["correct"].bool()
    red = out_full["correct"].bool()[wrong].double()
    cov = out_k["covered"].bool()[wrong]
    au, au_def = {}, {}
    for s, v in ranking_signals(out_k).items():
        r = _rank_key(v[wrong])
        au[s] = auroc(r, red)
        if s in NEGATED:
            au[f"-{s}"] = auroc(-r, red)
            d = cov & torch.isfinite(out_k["u"][s][wrong])   # the finite-u tier alone
            au_def[f"-{s}"] = auroc(-r[d], red[d]) if bool(d.any()) else _NAN
    return dict(auroc=au, auroc_defined=au_def, n_pixels=int(wrong.sum()),
                n_e_red=int(red.sum()), n_e_irr=int((1.0 - red).sum()),
                n_undefined={k: int((cov & ~torch.isfinite(u[wrong])).sum())
                             for k, u in out_k["u"].items()})


def random_gaps(targets, ause_bins=20):
    """(b) {target: AUSE of a random order} (sparsification_curve random_gap: it depends on
    the errors only); NaN for an empty target."""
    return {t: (sparsification_curve(e, torch.zeros_like(e), num_bins=ause_bins)["random_gap"]
                if e.numel() else _NAN) for t, e in targets.items()}


def normalised(ause, gaps):
    """(b) {signal: {target: AUSE / random gap}}; NaN where the gap is 0 (no or only errors)."""
    return {s: {t: (a / gaps[t] if gaps.get(t, 0.0) > 1e-12 else _NAN) for t, a in d.items()}
            for s, d in ause.items()}


def ale_split(out_k, out_full, channel=REF_CHANNEL, covered_only=True, ause_bins=20):
    """(c) The AUSE of `channel` on the kept pixels (covered at k) where it is defined and on
    its undefined tier (one key: AUSE = the tier's random gap). None without the channel."""
    if channel not in out_k["u"]:
        return None
    keep, _ = _aligned_keep(out_k, out_full, covered_only)
    keep = keep & out_k["covered"].bool()
    key = ranking_signals(out_k)[channel]
    fin = torch.isfinite(out_k["u"][channel])
    n_keep, out = int(keep.sum()), {}
    for tier, m in (("defined", keep & fin), ("undefined", keep & ~fin)):
        t = error_targets(out_k["correct"][m], out_full["correct"][m])
        n = int(m.sum())
        d = dict(n_pixels=n, share=n / max(n_keep, 1), n_e_red=int(t["e_red"].sum()),
                 n_e_irr=int(t["e_irr"].sum()))
        if n:
            gaps = random_gaps(t, ause_bins)
            ause = {tn: sparsification_curve(e, key[m], num_bins=ause_bins,
                                             undefined="last")["ause"] for tn, e in t.items()}
            d.update(rate={tn: float(e.mean()) for tn, e in t.items()}, ause=ause,
                     random_gap=gaps, ause_norm=normalised({channel: ause}, gaps)[channel])
        out[tier] = d
    return out


def extra_analyses(out_k, out_full, ause, ause_bins=20):
    """(a), (b), (c) for one row: the additive keys of its per-k result. `ause` is the row's
    views_cross_ause()["ause"] (covered-in-both pixels)."""
    keep, _ = _aligned_keep(out_k, out_full, True)
    gaps = random_gaps(error_targets(out_k["correct"][keep], out_full["correct"][keep]),
                       ause_bins)
    out = dict(random_gap=gaps, ause_norm=normalised(ause, gaps),
               reducibility=reducibility_auroc(out_k, out_full))
    split = ale_split(out_k, out_full, ause_bins=ause_bins)
    if split is not None:
        out["ale_split"] = split
    return out


def evaluate_scene(sc, vocab, betas, args, device, ks):
    t0 = time.perf_counter()
    S_full = ES.load_scene(sc, vocab, args, device)
    rows = row_names(S_full, args)
    acc_full = accumulate(S_full, betas, args, rows)
    full_metrics = acc_full.metrics(args.ece_bins, args.ause_bins)
    out = dict(full=dict(field=sc["field"], n_views=acc_full.n_views,
                         miou={r: full_metrics[r]["all"].get("miou") for r in rows}),
               k={})
    del S_full
    for k in ks:
        path = sc["fields"][str(k)]
        S_k = ES.load_scene(dict(sc, field=path), vocab, args, device)
        timing = torch.load(path, map_location="cpu", weights_only=False, mmap=True).get("timing")
        n_map = (timing or {}).get("keyframes")
        acc_k = accumulate(S_k, betas, args, rows)
        mk = acc_k.metrics(args.ece_bins, args.ause_bins)
        res = {}
        for r in rows:
            ak, af = acc_k.arrays(r), acc_full.arrays(r)
            res[r] = views_cross_ause(ak, af, ause_bins=args.ause_bins)
            res[r]["miou_k"] = mk[r]["all"].get("miou")
            res[r].update(extra_analyses(ak, af, res[r]["ause"], args.ause_bins))
            del ak, af
        on_defined(res, acc_k, acc_full, rows, args)
        out["k"][str(k)] = dict(field=path, n_mapping_views=n_map, rows=res)
        for r in rows:
            a = res[r]["ause"]
            line = " ".join(f"{s}:{a[s]['e_red']:.3f}/{a[s]['e_irr']:.3f}" for s in a)
            print(f"{sc['name']:>9} k={k:>4} {r:>14} mIoU {res[r]['miou_k']:.3f} "
                  f"n_red {res[r]['n_e_red']} n_irr {res[r]['n_e_irr']} "
                  f"AUSE red/irr {line}", flush=True)
            au = res[r]["reducibility"]["auroc"]
            print(f"{sc['name']:>9} k={k:>4} {r:>14} AUROC red-vs-irr (wrong at k) "
                  + " ".join(f"{s}:{v:.3f}" for s, v in au.items()), flush=True)
        del S_k, acc_k
    out["eval_s"] = time.perf_counter() - t0
    return out


def summarize(results, ks):
    """{k: {row: {signal: {target: mean/std/n over scenes}}}}."""
    summ = {}
    for k in ks:
        per = [res["k"][str(k)]["rows"] for res in results.values()]
        rows = sorted(set.intersection(*[set(p) for p in per])) if per else []
        summ[str(k)] = {}
        for r in rows:
            sig = per[0][r]["ause"]
            summ[str(k)][r] = {s: {t: scene_stats([p[r]["ause"][s][t] for p in per])
                                   for t in sig[s]} for s in sig}
    return summ


def summarize_on_defined(results, ks):
    """The secondary analysis (on_defined): {k: {row: {signal: {target: stats}}, share}}."""
    summ = {}
    for k in ks:
        per = [res["k"][str(k)]["rows"] for res in results.values()]
        rows = sorted(set.intersection(*[set(p) for p in per])) if per else []
        rows = [r for r in rows if all("on_ale_defined" in p[r] for p in per)]
        summ[str(k)] = {}
        for r in rows:                    # a scene with no defined pixel counts as NaN
            sigs = sorted({s for p in per for s in p[r]["on_ale_defined"]["ause"]})
            get = lambda p, s, t: p[r]["on_ale_defined"]["ause"].get(s, {}).get(t, float("nan"))
            summ[str(k)][r] = {s: {t: scene_stats([get(p, s, t) for p in per])
                                   for t in ("e_red", "e_irr")} for s in sigs}
        if rows:
            summ[str(k)]["share"] = scene_stats([p[rows[0]]["on_ale_defined"]["share"]
                                                 for p in per])
    return summ


def _get(d, *keys):
    """d[k0][k1]...; NaN if missing. A saved JSON holds non-finite floats as strings
    (eval_seg._plain_json): they come back as floats."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return _NAN
        d = d[k]
    return float(d) if d in ("nan", "inf", "-inf") else d


def summarize_extra(results, ks):
    """The added analyses over scenes (a scene without a value counts as NaN):
    {"ause_norm": {k: {row: {signal: {target: stats}}}}, "reducibility": {k: {row: {signal:
    stats}}} ("<signal>@defined": auroc_defined), "ale_split": {k: {row: {tier: {share, rate,
    ause, ause_norm}}}}}."""
    out = {"ause_norm": {}, "reducibility": {}, "ale_split": {}}
    for k in ks:
        per = [res["k"][str(k)]["rows"] for res in results.values()]
        rows = sorted(set.intersection(*[set(p) for p in per])) if per else []
        an, ra, sp = {}, {}, {}
        for r in rows:
            g = [p[r] for p in per]
            sigs = sorted({s for x in g for s in x.get("ause_norm", {})})
            an[r] = {s: {t: scene_stats([_get(x, "ause_norm", s, t) for x in g]) for t in TARGETS}
                     for s in sigs}
            sigs = sorted({s for x in g for s in x.get("reducibility", {}).get("auroc", {})})
            ra[r] = {s: scene_stats([_get(x, "reducibility", "auroc", s) for x in g]) for s in sigs}
            sigs = sorted({s for x in g
                           for s in x.get("reducibility", {}).get("auroc_defined", {})})
            ra[r].update({f"{s}@defined": scene_stats([_get(x, "reducibility", "auroc_defined", s)
                                                       for x in g]) for s in sigs})
            if all("ale_split" in x for x in g):
                sp[r] = {tier: dict(share=scene_stats([_get(x, "ale_split", tier, "share") for x in g]),
                                    **{key: {t: scene_stats([_get(x, "ale_split", tier, key, t)
                                                             for x in g]) for t in TARGETS}
                                       for key in ("rate", "ause", "ause_norm")})
                         for tier in ("defined", "undefined")}
        out["ause_norm"][str(k)], out["reducibility"][str(k)], out["ale_split"][str(k)] = an, ra, sp
    return out


def reducibility_verdict(scenes, crit=REDUCIBILITY_CRITERION):
    """The pre-registered criterion on {scene: evaluate_scene output}: "kept" if on every
    criterion scene the AUROC of crit["score"] (row crit["row"], M = crit["k"]) is above
    crit["auroc_min"] and above that of crit["against"]; "dropped" otherwise (a NaN fails);
    "undecided" while a criterion scene, its k or its row is missing, or its k-view field
    records other than k mapping keyframes. Other scenes are listed and ignored."""
    per, missing = {}, []
    for sc in crit["scenes"]:
        a = _get(scenes, sc, "k", str(crit["k"]), "rows", crit["row"], "reducibility", "auroc")
        n_map = _get(scenes, sc, "k", str(crit["k"]), "n_mapping_views")
        if isinstance(n_map, (int, float)) and n_map == n_map and int(n_map) != int(crit["k"]):
            a = None                                    # the field label is not M views
        if not isinstance(a, dict):
            missing.append(sc)
            continue
        su, sm = float(a.get(crit["score"], _NAN)), float(a.get(crit["against"], _NAN))
        per[sc] = {crit["score"]: su, crit["against"]: sm,
                   "passed": bool(su > crit["auroc_min"] and su > sm)}
    kept = None if missing else all(p["passed"] for p in per.values())
    return dict(criterion=dict(crit, scenes=list(crit["scenes"])),
                status="undecided" if missing else ("kept" if kept else "dropped"),
                kept=kept, per_scene=per, missing=missing,
                ignored_scenes=sorted(set(scenes) - set(crit["scenes"])))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verdict_main(argv):
    """`eval_views.py verdict <json>... [--out path]`: the pre-registered criterion from
    eval_views.py outputs (one per scene, as scripts/replica_views.sh writes them, or merged),
    with the added analyses summarised over the merged scenes (the k common to all files)."""
    ap = argparse.ArgumentParser(prog="eval_views.py verdict",
                                 description="pre-registered reducibility criterion "
                                             "(REDUCIBILITY_CRITERION) from eval_views.py outputs")
    ap.add_argument("inputs", nargs="+", help="eval_views.py output JSON files")
    ap.add_argument("--out", default="", help="write the verdict here (never overwritten; "
                                              "an undecided verdict is not written)")
    a = ap.parse_args(argv)
    if a.out and os.path.exists(a.out):
        raise SystemExit(f"refused: {a.out} exists (the verdict is written once)")
    scenes, src, ks = {}, {}, None
    for path in a.inputs:
        with open(path) as f:
            d = json.load(f)
        for sc, v in d["scenes"].items():
            if sc in scenes:
                raise SystemExit(f"scene {sc} is in both {src[sc]} and {path}")
            scenes[sc], src[sc] = v, path
        ks = set(d["ks"]) if ks is None else ks & set(d["ks"])
    out = reducibility_verdict(scenes)
    ex = summarize_extra(scenes, sorted(ks))                # over the merged scenes
    out.update(summary_ause_norm=ex["ause_norm"], summary_reducibility=ex["reducibility"],
               summary_ale_split=ex["ale_split"])
    out["inputs"] = {os.path.abspath(p): _sha256(p) for p in a.inputs}
    out["provenance"] = dict(argv=[sys.argv[0], "verdict"] + list(argv), time=time.time())
    c = out["criterion"]
    print(f"reducibility criterion ({c['row']} {c['score']} AUROC > {c['auroc_min']} and > "
          f"{c['against']} at M = {c['k']} on {c['scenes']}): {out['status'].upper()}")
    for sc, p in out["per_scene"].items():
        print(f"  {sc}: {c['score']} {p[c['score']]:.4f}  {c['against']} {p[c['against']]:.4f}  "
              f"{'pass' if p['passed'] else 'FAIL'}")
    if out["missing"]:
        print(f"  missing: {out['missing']}")
    if a.out and out["status"] == "undecided":   # the file is final: only a decided verdict
        raise SystemExit(f"refused: the verdict is undecided (missing {out['missing']}); "
                         f"nothing written to {a.out}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(ES._plain_json(out), f, indent=1)
        print(f"wrote {a.out}")
    return out


def run(args):
    device = args.device
    if device.startswith("cuda"):
        from usplat import gpu_reserve                      # fraction from scripts/gpu_gate.sh
        gpu_reserve.apply(device)
    if not args.betas:
        raise SystemExit("--betas is required (an eval_seg.py fit on the full val maps)")
    vocab = ES.load_vocab(args.vocab)
    settings = ES.load_fit_settings(args.betas)
    ES.check_fit_settings(settings, args, args.betas)
    betas = ES.load_betas(args.betas)
    scenes = ES._load_structured(args.scenes)
    ks = sorted({int(k) for sc in scenes for k in sc["fields"]})
    for sc in scenes:
        miss = [k for k in ks if str(k) not in sc["fields"]]
        if miss:
            raise ValueError(f"scene {sc['name']}: no field for k = {miss}")
    res = {sc["name"]: evaluate_scene(sc, vocab, betas, args, device, ks) for sc in scenes}
    ex = summarize_extra(res, ks)
    return dict(betas_path=os.path.abspath(args.betas), betas=betas, ks=ks, scenes=res,
                summary=summarize(res, ks), summary_on_ale_defined=summarize_on_defined(res, ks),
                summary_ause_norm=ex["ause_norm"], summary_reducibility=ex["reducibility"],
                summary_ale_split=ex["ale_split"], reducibility_criterion=reducibility_verdict(res))


def build_parser():
    ap = ES.build_parser()
    ap.description = __doc__
    return ap


def main(argv=None):
    rest = list(sys.argv[1:] if argv is None else argv)
    if rest[:1] == ["verdict"]:                    # the pre-registered criterion, no GPU
        return verdict_main(rest[1:])
    args = build_parser().parse_args(argv)
    if not args.scenes:
        raise SystemExit("give --scenes (with per-k 'fields')")
    if args.val_scenes:
        raise SystemExit("eval_views.py reads a fit (--betas); it does not fit")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.manual_seed(args.seed)
    t0 = time.time()
    out = run(args)
    out["provenance"] = dict(argv=[sys.argv[0]] + list(argv if argv is not None else sys.argv[1:]),
                             start=t0, wall_s=time.time() - t0, config=vars(args))
    with open(args.out, "w") as f:
        json.dump(ES._plain_json(out), f, indent=1)
    print(f"wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
