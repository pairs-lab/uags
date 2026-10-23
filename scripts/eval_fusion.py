"""
eval_fusion.py — Table I block "with the current view": the map predictive fused per pixel
with the predictive of the evaluation frame's own 2D features (usplat/view_fusion.py; rules in
configs/fusion_prereg.json; driver scripts/fusion_tab1.sh). It imports scripts/eval_seg.py and
usplat.seg_eval and edits neither: scenes, frames, labels, texts, renders, the 2D row and the
metrics are theirs.

Rows (fixed, not selectable):
    reference  ours_P0, ours_P2, averaging (eval_seg's rows, evaluate_view) and 2d_features
               (eval_seg --with-2d: PixelSource, frame_features_2d)
    fused      every form of usplat.view_fusion.FORMS (fuse_*), fused_outputs
The channel rows viewcount and normlift (usplat.baselines:row_factory; their buffer is the
averaging feature, checked) are rendered only for their channels. Rows are frozen by
eval_seg.load_scene (--rows is set here; giving --rows, --row-factory, --v2-rows or
--raw-probs is refused).

Fit (--val-scenes; the same file as --scenes in the validation stage):
  1. eval_seg's BetaFitter in Table I's order (per scene: every view of the map rows, then
     every view of the 2D row, same seed and --fit-pixels), so the single-source temperatures
     reproduce Table I's validation betas (independent check in scripts/fusion_select.py);
  2. a third pass over the views collects the fusion fit pixels (covered labelled pixels, at
     most --fit-pixels per view, generator seed + 1: view_fusion.FusionFitData), then every
     form's theta is fit by NLL with the single-source logits fixed (view_fusion.fit_weights).
  --betas reads both (betas, fusion) from an earlier record of this script; the fit settings
  and field identities are checked as eval_seg does, the forms against the code.
Variants per row and scene: "all" (every labelled pixel; a map row counts an uncovered pixel
as wrong with the uniform predictive, the 2D and fused rows decide there) and "map_covered"
(the labelled pixels with map alpha >= tau_cov, the same set for every row): the fused rows
fill the holes with the 2D prediction, the map_covered variant shows the gain without them.
Fused rows also record w2d_mean_map_covered (the mean 2D weight on map-covered pixels).

    cd uags && scripts/gpu_gate.sh 7000 python3 scripts/eval_fusion.py \
        --vocab configs/vocab_replica.json --val-scenes configs/replica_val.json \
        --scenes configs/replica_val.json --text-cache OUT/text_cache_lseg.pt --out OUT/eval_val.json
"""
from __future__ import annotations
import json
import os
import sys
import time
import warnings

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import eval_seg as es                                                      # noqa: E402
from usplat import view_fusion as vf                                       # noqa: E402
from usplat.rows import TensorRow                                          # noqa: E402
from usplat.seg_eval import (BetaFitter, SegAccumulator, evaluate_view, render_row,  # noqa: E402
                             summarize, scene_stats)

REFERENCE = ("ours_P0", "ours_P2", "averaging")
CHANNEL_ROWS = ("viewcount", "normlift")
ROW_2D = es.ROW_2D
VARIANTS = ("all", "map_covered")
FACTORY = "usplat.baselines:row_factory"
REFUSED_OPTS = ("rows", "row_factory", "v2_rows", "raw_probs")


# ------------------------------------------------------------------------------ setup
def configure(args):
    """Fix the rows of this script on an eval_seg argument namespace (refuses row options)."""
    given = [o for o in REFUSED_OPTS if getattr(args, o, None)]
    if given:
        raise ValueError(f"eval_fusion.py fixes its rows; do not pass "
                         f"{['--' + o.replace('_', '-') for o in given]}")
    args.rows = ",".join(REFERENCE + CHANNEL_ROWS + (ROW_2D,))
    args.row_factory = FACTORY
    args.with_2d = True
    args.no_p3 = True
    args.v2_rows = False
    args.raw_probs = ""
    names = [s.strip() for s in (args.forms or "").split(",") if s.strip()]
    args.form_list = vf.forms_by_names(names) if names else list(vf.FORMS)
    return args


def scene_rows(S):
    """{name: frozen row} and the averaging-channel row: the averaging buffer with the
    channels of viewcount and normlift (one render). Refuses channel rows whose buffer is not
    the averaging feature."""
    rows = {r.name: r for r in S["rows"]}
    miss = [r for r in REFERENCE + CHANNEL_ROWS if r not in rows]
    if miss or len(S["rows2d"]) != 1:
        raise ValueError(f"scene {S['name']}: rows {miss} missing (the field checkpoint needs the "
                         f"NormLift and view-count baselines) or no 2D row")
    avg = rows["averaging"]
    for c in CHANNEL_ROWS:
        if not torch.equal(rows[c].buf, avg.buf):
            raise ValueError(f"scene {S['name']}: row {c!r} does not render the averaging feature")
    unc = dict(rows["viewcount"].unc)
    unc.update(rows["normlift"].unc)
    avg_ch = TensorRow("averaging_channels", "feature", buf=avg.buf, unc=unc)
    return rows, avg_ch, S["rows2d"][0]


def view_renders(S, f, args, rows, avg_ch, row2d):
    """(rasterizer, 2D source, bundle) of one evaluation frame."""
    ras = es.make_rasterizer(args.backend, S["splats"], f, args)
    src = es.PixelSource(es.frame_features_2d(S, f, args))
    r_o = render_row(rows["ours_P0"], ras, undef_frac=args.undef_frac)
    r_a = render_row(avg_ch, ras, undef_frac=args.undef_frac)
    r_2 = render_row(row2d, src, undef_frac=args.undef_frac)
    return ras, src, vf.pixel_bundle(r_o, r_a, r_2, S["text"], f["gt"], args.tau_cov)


# -------------------------------------------------------------------------------- fit
def fit_on(scenes, vocab, args, device):
    """Stage 1 (single-source temperatures, eval_seg's fit_on order) and stage 2 (fusion)."""
    fitter = BetaFitter(args.tau_cov, max_pixels_per_view=args.fit_pixels, seed=args.seed,
                        undef_frac=args.undef_frac)
    data = vf.FusionFitData(cap=args.fit_pixels, seed=args.seed + vf.FIT_SEED_OFFSET)
    ids, aux = set(), set()
    for sc in scenes:
        S = es.load_scene(sc, vocab, args, device)
        ids.add(S["field_identity"])
        aux |= set(S["aux_identities"])
        fitter.add(S["rows"], es.scene_views(S, args))          # as eval_seg.fit_on
        fitter.add(S["rows2d"], es.scene_views_2d(S, args))
        rows, avg_ch, row2d = scene_rows(S)
        for f in S["frames"]:
            data.add(view_renders(S, f, args, rows, avg_ch, row2d)[2])
        del S
    if len(ids) > 1:
        warnings.warn(f"validation fields were built differently ({len(ids)} identities)")
    betas = fitter.fit()
    settings = dict(fitter.settings, field_identities=sorted(ids), aux_identities=sorted(aux))
    fusion = data.fit(args.form_list, betas)
    fsettings = dict(max_pixels_per_view=args.fit_pixels, seed=data.seed, iters=vf.FIT_ITERS,
                     n_views=len(data.bundles), forms=[f.spec() for f in args.form_list])
    return betas, fitter.info, settings, fusion, fsettings


# ----------------------------------------------------------------------------- metrics
class FusionAccumulator(SegAccumulator):
    """SegAccumulator that keeps the map coverage of every labelled pixel ('map_covered').
    The fusion weight of a fused row (u['w2d']) is taken out of its channels (it is not a
    ranking signal of Table I) and kept as its mean on map-covered pixels."""
    KEYS = SegAccumulator.KEYS + ("map_covered",)

    def __init__(self, class_names=None):
        super().__init__(class_names)
        self.wsum = {}

    def add(self, view_out, map_covered):
        for name, d in view_out.items():
            if d["gt"].numel() != map_covered.numel():
                raise ValueError("map coverage and row outputs differ in pixels")
            d["map_covered"] = map_covered
            if "w2d" in d["u"]:
                d["u"] = dict(d["u"])
                w = d["u"].pop("w2d")[map_covered].double()
                s, n = self.wsum.get(name, (0.0, 0))
                self.wsum[name] = (s + float(w.sum()), n + int(w.numel()))
        super().add(view_out)

    @staticmethod
    def subset(d, mask):
        return dict({k: d[k][mask] for k in FusionAccumulator.KEYS},
                    u={k: v[mask] for k, v in d["u"].items()})

    def metrics(self, num_bins=15, ause_bins=20):
        out = {}
        for name in self.chunks:
            d = self.arrays(name)
            m = d["map_covered"]
            s = summarize(d, self.class_names, num_bins, ause_bins)
            if torch.equal(d["covered"], m):           # a map row: its covered variant
                r = {"all": s["all"], "map_covered": s["covered"]}
            else:                                      # 2D / fused rows decide every pixel
                r = {"all": s["all"], "map_covered": summarize(self.subset(d, m),
                                                               self.class_names, num_bins,
                                                               ause_bins)["all"]}
            if name in self.wsum:
                ws, wn = self.wsum[name]
                r["w2d_mean_map_covered"] = ws / wn if wn else float("nan")
            r["n_map_uncovered"] = int((~m).sum())
            out[name] = r
        return out


def evaluate_scene(sc, vocab, betas, fusion, args, device, settings=None):
    S = es.load_scene(sc, vocab, args, device)
    es.check_field_identity(S["field_identity"], S["aux_identities"], settings, sc["name"], args)
    used = es.betas_used(S["rows"] + S["rows2d"], betas, set())
    rows, avg_ch, row2d = scene_rows(S)
    refs = [rows[n] for n in REFERENCE]
    acc = FusionAccumulator(class_names=S["class_names"])
    t0 = time.perf_counter()
    for f in S["frames"]:
        ras, src, b = view_renders(S, f, args, rows, avg_ch, row2d)
        out = evaluate_view(refs, ras, S["text"], f["gt"], betas, args.tau_cov,
                            undef_frac=args.undef_frac)
        out.update(evaluate_view([row2d], src, S["text"], f["gt"], betas, args.tau_cov,
                                 undef_frac=args.undef_frac))
        for form in args.form_list:
            out[form.name] = vf.fused_outputs(form, fusion[form.name]["theta"], b, betas)
        mc = out["ours_P0"]["covered"]
        if not torch.equal(mc, b.cov.cpu()):
            raise RuntimeError("the bundle's map coverage differs from evaluate_view's")
        acc.add(out, mc)
    return dict(n_views=acc.n_views, classes=S["class_names"], class_ids=S["class_ids"],
                field_identity=S["field_identity"], field_code=S["field_code"],
                aux_identities=S["aux_identities"], betas_used=used,
                fusion_used={f.name: fusion[f.name]["theta"] for f in args.form_list},
                n_unknown_label_pixels=S["n_unknown"], n_skipped_frames=S["n_skipped"],
                eval_s=time.perf_counter() - t0, rows=acc.metrics(args.ece_bins, args.ause_bins))


def summarize_scenes(results):
    out = {}
    rows = sorted({r for res in results.values() for r in res["rows"]})
    for r in rows:
        out[r] = {}
        for var in VARIANTS:
            per = [res["rows"][r][var] for res in results.values() if r in res["rows"]]
            out[r][var] = {k: scene_stats([p.get(k, float("nan")) for p in per])
                           for k in es.SUMMARY_KEYS}
    return out


def load_fusion(path):
    d = es._record_body(path)
    return d.get("fusion") if isinstance(d, dict) else None


# -------------------------------------------------------------------------------- main
def run(args, rec=None):
    device = args.device
    if device.startswith("cuda"):
        from usplat import gpu_reserve
        held = gpu_reserve.apply(device)
        if held:
            print(f"reserved {held} MiB of GPU memory")
    vocab = es.load_vocab(args.vocab)
    out = dict(vocab=dict(path=os.path.abspath(args.vocab), n_classes=len(vocab["ids"]),
                          extractor=args.extractor,
                          templates=es.extractor_templates(vocab, args.extractor),
                          ignore=sorted(vocab["ignore"])),
               rows=dict(reference=list(REFERENCE) + [ROW_2D],
                         fused=[f.spec() for f in args.form_list]))
    if args.val_scenes and args.betas:
        raise ValueError("give --val-scenes (fit) or --betas (read a fit), not both")
    if args.val_scenes:
        betas, info, settings, fusion, fset = fit_on(es._load_structured(args.val_scenes),
                                                     vocab, args, device)
        out.update(betas=betas, beta_fit=info, beta_fit_settings=settings, fusion=fusion,
                   fusion_fit_settings=fset)
        print("fitted betas on val:", json.dumps(betas))
        for n, e in fusion.items():
            print(f"fusion {n}: theta {dict(zip(e['params'], [round(t, 4) for t in e['theta']]))} "
                  f"fit NLL {e['fit']['nll']:.4f} (map {e['fit']['nll_map']:.4f}, 2D "
                  f"{e['fit']['nll_2d']:.4f}), converged {e['fit']['converged']}")
    elif args.betas:
        settings = es.load_fit_settings(args.betas)
        es.check_fit_settings(settings, args, args.betas)
        fusion = load_fusion(args.betas)
        vf.check_fusion_record(fusion, args.form_list)
        out.update(betas=es.load_betas(args.betas), beta_fit_settings=settings,
                   betas_path=os.path.abspath(args.betas),
                   fusion={f.name: fusion[f.name] for f in args.form_list})
    else:
        raise ValueError("give --val-scenes or --betas: every row needs fitted parameters")
    if args.scenes:
        res = {}
        partial = f"{args.out}.partial.json"
        for sc in es._load_structured(args.scenes):
            res[sc["name"]] = evaluate_scene(sc, vocab, out["betas"], out["fusion"], args, device,
                                             settings=out.get("beta_fit_settings"))
            with open(partial, "w") as f:
                json.dump(es._plain_json(dict(out, scenes=res)), f, indent=1)
            for r, m in res[sc["name"]]["rows"].items():
                a, c = m["all"], m["map_covered"]
                print(f"{sc['name']:>12} {r:>14}: mIoU {a.get('miou', float('nan')):.4f} "
                      f"acc {a.get('acc', float('nan')):.4f} ECE {a.get('ece', float('nan')):.4f} "
                      f"NLL {a.get('nll', float('nan')):.4f} | map-covered mIoU "
                      f"{c.get('miou', float('nan')):.4f} NLL {c.get('nll', float('nan')):.4f}")
        out["scenes"] = res
        out["summary"] = summarize_scenes(res)
        out["field_identities"] = sorted({r["field_identity"] for r in res.values()})
    if rec is not None:
        for k, v in out.items():
            rec.add(k, v)
    return out


def build_parser():
    ap = es.build_parser()
    ap.description = __doc__.split("\n\n")[0].strip()
    ap.add_argument("--forms", default="",
                    help="comma-separated fused rows (default: every form of "
                         "usplat.view_fusion.FORMS)")
    return ap


def main(argv=None):
    args = configure(build_parser().parse_args(argv))
    if not (args.scenes or args.val_scenes):
        raise SystemExit("nothing to do: give --scenes and/or --val-scenes")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    from usplat.runlog import RunRecord
    if not args.plain_json:
        cfg = {k: v for k, v in vars(args).items() if k != "form_list"}
        with RunRecord(args.out, config=cfg, seed=args.seed, name="eval_fusion",
                       argv=[sys.argv[0]] + list(argv if argv is not None else sys.argv[1:])) as rec:
            out = run(args, rec)
    else:
        torch.manual_seed(args.seed)
        t0 = time.time()
        out = run(args)
        out["provenance"] = dict(argv=sys.argv if argv is None else argv, start=t0,
                                 wall_s=time.time() - t0,
                                 config={k: v for k, v in vars(args).items() if k != "form_list"})
        with open(args.out, "w") as f:
            json.dump(es._plain_json(out), f, indent=1)
    print(f"wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
