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

Scenes file (JSON/YAML list), as for eval_seg.py plus the k-view fields:
    {name, field: <full field>, fields: {"10": <path>, "30": <path>, "90": <path>}, ply, data,
     dataset, stride}

    cd uags && scripts/gpu_gate.sh 4000 python3 scripts/eval_views.py \
        --vocab configs/vocab_replica.json --extractor lseg --betas <tab1>/eval_val.json \
        --scenes configs/replica_views_val.json --out <out>/views_val.json
"""
from __future__ import annotations
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
                             views_cross_ause, scene_stats)


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
            res[r] = views_cross_ause(acc_k.arrays(r), acc_full.arrays(r),
                                      ause_bins=args.ause_bins)
            res[r]["miou_k"] = mk[r]["all"].get("miou")
        out["k"][str(k)] = dict(field=path, n_mapping_views=n_map, rows=res)
        for r in rows:
            a = res[r]["ause"]
            line = " ".join(f"{s}:{a[s]['e_red']:.3f}/{a[s]['e_irr']:.3f}" for s in a)
            print(f"{sc['name']:>9} k={k:>4} {r:>14} mIoU {res[r]['miou_k']:.3f} "
                  f"n_red {res[r]['n_e_red']} n_irr {res[r]['n_e_irr']} "
                  f"AUSE red/irr {line}", flush=True)
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
    return dict(betas_path=os.path.abspath(args.betas), betas=betas, ks=ks, scenes=res,
                summary=summarize(res, ks))


def build_parser():
    ap = ES.build_parser()
    ap.description = __doc__
    return ap


def main(argv=None):
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
