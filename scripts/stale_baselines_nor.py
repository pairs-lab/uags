"""
stale_baselines_nor.py — the stale-answer rate of Table IV (tab:stale) for more baselines on one
NeRF-Object-Removal scene (configs/stale_baselines_prereg.json). The measure is
scripts/dump_nor_qual.py's N2 number (contrast_and_fa), with that script's coverage, class maps,
text and query; dump_nor_qual.py, eval_change_nor.py and eval_change_nor_distilled.py are
imported and never edited.

    scripts/gpu_gate.sh 6000 python3 scripts/stale_baselines_nor.py \
        --table results/nor/tab2_relmax/test/008.json --dump results/qual/nor/008 \
        --vocab configs/vocab_replica.json --text-cache <out>/text_cache_lseg_nor.pt \
        --distilled results/fields/nor_008_lseg_relmax_f3dgs_upd.pt,... --stage test \
        --out <out>/test/008_trained.json

Rows (one count over the covered masks_real pixels of every update keyframe with a mask):
  before    control: the mapping field's mu*, rendered. It must reproduce the dump's 'before'
            set (rate within 0.5 points, pixel count within 0.5%), else exit 2
  2d        the update keyframe's own features in the field's PCA basis (no map); skipped with
            --no-2d
  <name>    one row per --distilled file (its 'name': f3dgs, legs_l1, f3dgs_upd, legs_l1_upd):
            Feature 3DGS renders its features, LEGaussians renders its codes and decodes them
            (eval_change_nor_distilled.load_sources)
The query is the dump's (contrast.json), never chosen again. A finished test record is never
overwritten. Backends: gsplat (GPU) or mock (CPU, tests/test_stale_baselines.py).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import dump_nor_qual as DQ                                                  # noqa: E402
import eval_change_nor_distilled as ECD                                     # noqa: E402
from usplat.rows import TensorRow                                           # noqa: E402
from usplat.seg_eval import render_row                                      # noqa: E402

EC, ES = DQ.EC, DQ.ES
CONTROL = "before"
POINTS, COUNT_REL = 0.005, 0.005        # control: 0.5 points on the rate, 0.5% on the pixel count
CODE = ("scripts/stale_baselines_nor.py", "scripts/dump_nor_qual.py",
        "scripts/eval_change_nor_distilled.py", "usplat/distill.py") + tuple(EC.CODE)


def code_state():
    repo = os.path.dirname(HERE)
    return {rel: DQ.sha256_file(os.path.join(repo, rel)) for rel in dict.fromkeys(CODE)}


@torch.no_grad()
def count_rows(ctx, views, sources, text, qi, tau_cov, undef_frac):
    """{row: dict(n_mask_cov, n_query_in_mask, stale_answer_rate)} and the number of views
    without a mask. sources: {row: fn(view, rasterizer) -> (P, d) feature map}. Coverage, mask
    and top class as dump_nor_qual.contrast_and_fa (beta 1; ties count as no class)."""
    dev, N = ctx["device"], ctx["N"]
    acc = {r: dict(n_mask_cov=0, n_query_in_mask=0) for r in sources}
    no_mask = 0
    zero = torch.zeros(N, device=dev)
    for v in views:
        if v["mask_real"] is None:
            no_mask += 1
            continue
        ras, _ = EC.rasterizer(ctx, v)
        row = TensorRow("flag", "feature", buf=torch.zeros(N, 1, device=dev), unc={"flag": zero})
        rr = render_row(row, ras, alpha_eps=EC.ALPHA_EPS, undef_frac=undef_frac)
        cov = DQ._covered(rr.alpha, tau_cov).cpu()
        gt = EC._nearest(v["mask_real"], v["h"], v["w"]).reshape(-1).bool().cpu()
        inm = cov & gt
        for r, fn in sources.items():
            pred, _, _ = DQ.class_maps(fn(v, ras), cov, text, qi, 1.0)
            acc[r]["n_mask_cov"] += int(inm.sum())
            acc[r]["n_query_in_mask"] += int((inm & (pred == qi)).sum())
    for a in acc.values():
        a["stale_answer_rate"] = (a["n_query_in_mask"] / a["n_mask_cov"] if a["n_mask_cov"]
                                  else float("nan"))
    return acc, no_mask


def control_check(row, dump_set):
    """The recomputed 'before' row against the dump's set."""
    d_rate = abs(row["stale_answer_rate"] - dump_set["stale_answer_rate"])
    d_cnt = abs(row["n_mask_cov"] - dump_set["n_mask_cov"]) / max(dump_set["n_mask_cov"], 1)
    return dict(rerun=row["stale_answer_rate"], dump=dump_set["stale_answer_rate"],
                abs_diff=d_rate, n_mask_cov=row["n_mask_cov"], n_mask_cov_dump=dump_set["n_mask_cov"],
                rel_count_diff=d_cnt, ok=bool(d_rate <= POINTS and d_cnt <= COUNT_REL))


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", required=True, help="the scene's Table III record (JSON)")
    ap.add_argument("--dump", required=True, help="the scene's dump directory (contrast.json)")
    ap.add_argument("--vocab", required=True, help="configs/vocab_replica.json (101 classes)")
    ap.add_argument("--text-cache", required=True, help="a COPY of the LSeg text cache")
    ap.add_argument("--distilled", default="", help="comma-separated distilled files")
    ap.add_argument("--no-2d", action="store_true", help="skip the per-frame 2D row")
    ap.add_argument("--stage", choices=["val", "test", "toy"], required=True)
    ap.add_argument("--out", required=True, help="output record (JSON)")
    ap.add_argument("--backend", choices=["gsplat", "mock"], default="gsplat")
    ap.add_argument("--device", default="cuda")
    return ap


def run(args, argv):
    t0 = time.time()
    if os.path.exists(args.out):
        old = json.load(open(args.out))
        if old.get("status") == "ok" and old.get("stage") == "test":
            raise SystemExit(f"{args.out}: this test scene was scored already (run once)")
    args.smoke = 0
    m = DQ.record_body(json.load(open(args.table)))
    con = json.load(open(os.path.join(args.dump, "contrast.json")))
    if str(con["scene"]) != str(m["scene"]):
        raise SystemExit(f"the dump is of scene {con['scene']}, the record of {m['scene']}")
    if args.stage != "toy" and con["stage"] != args.stage:
        raise SystemExit(f"the dump is a {con['stage']} dump, --stage is {args.stage}")
    vocab = ES.load_vocab(args.vocab)
    ns = DQ.ec_namespace(m, args)
    dev = args.device
    held = 0
    if str(dev).startswith("cuda"):
        from usplat import gpu_reserve                       # fraction from scripts/gpu_gate.sh
        held = gpu_reserve.apply(dev)
    ctx = EC.load_context(ns)
    targs = DQ.text_args(ctx["cfg"], args.text_cache)
    miss = DQ.missing_prompts(vocab, targs)
    if miss:
        raise SystemExit(f"{args.text_cache} lacks {len(miss)} prompts")
    text = ctx["pca"].project_text(
        ES.text_embeddings(vocab, list(range(len(vocab["prompts"]))), targs).to(dev))
    qi = int(con["query"]["index"])
    if vocab["prompts"][qi] != con["query"]["prompt"]:
        raise SystemExit(f"query {qi}: prompt {vocab['prompts'][qi]!r} != the dump's "
                         f"{con['query']['prompt']!r}")
    mu0 = ctx["frozen"]["mu"]
    sources = {CONTROL: lambda v, ras: ras(mu0)}
    if not args.no_2d:
        sources["2d"] = lambda v, ras: EC.view_inputs(ctx, v)[0]
    paths = [p for p in args.distilled.split(",") if p]
    src, prov = ECD.load_sources(ctx, paths, control=False)
    names = {}
    for row, fn in src.items():
        name = row[:-len("_feat")] if row.endswith("_feat") else row
        if name in sources:
            raise SystemExit(f"two sources are named {name!r}")
        sources[name] = lambda v, ras, fn=fn: fn(ras, None)
        names[name] = row
    rows, no_mask = count_rows(ctx, ctx["upd_views"], sources, text, qi, ns.tau_cov, ns.undef_frac)
    control = control_check(rows[CONTROL], con["sets"]["before"])
    out = dict(status="ok" if control["ok"] else "control failed", stage=args.stage,
               scene=m["scene"], query=con["query"], rows=rows, control=control,
               n_update_keyframes=len(ctx["upd_views"]), n_frames_no_mask=no_mask,
               distilled={n: prov[r] for n, r in names.items()},
               settings=dict(tau_cov=ns.tau_cov, undef_frac=ns.undef_frac,
                             update_stride=ns.update_stride, backend=args.backend),
               provenance=dict(argv=argv, git=DQ.git_state(), code=code_state(),
                               table=os.path.abspath(args.table),
                               table_sha256=DQ.sha256_file(args.table),
                               contrast_sha256=DQ.sha256_file(os.path.join(args.dump, "contrast.json")),
                               text_cache_sha256=DQ.sha256_file(args.text_cache)),
               timing=dict(wall_s=time.time() - t0, gpu_reserved_mib=held))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    DQ.write_json(args.out, out)
    for r, a in rows.items():
        print(f"{m['scene']} {r:14s} stale answers {100 * a['stale_answer_rate']:6.2f}%  "
              f"({a['n_query_in_mask']} of {a['n_mask_cov']} px)")
    print(f"{m['scene']} control: rerun {100 * control['rerun']:.2f}% vs dump {100 * control['dump']:.2f}% "
          f"-> {'ok' if control['ok'] else 'FAILED'}")
    out["exit"] = 0 if control["ok"] else 2
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    return run(build_parser().parse_args(argv), argv)


if __name__ == "__main__":
    sys.exit(main()["exit"])
