"""
extractor_2d.py — stage A of configs/extractor_prereg.json: the per-frame 2D segmentation of
one extractor on the VALIDATION evaluation frames of one dataset, through the path of
Table I's row "2D features, per frame" (scripts/eval_seg.py --with-2d, frame_features_2d),
without a map:

  PCA     per scene, StreamingPCA(d, 64) fit on the first --pca-frames (20) frames of the
          scene's MAPPING sequence at --downscale (2), as scripts/run_mapping.py does it
          (torch.manual_seed(seed) right before, so the CUDA randperm subsample is the one of a
          fresh run_mapping process: for LSeg the basis is Table I's);
  frames  the scene's evaluation frames and labels from eval_seg.load_frames (the scene file of
          the prereg's validation split; --downscale 2 label grid), queries = the vocabulary
          classes present in those frames (eval_seg's rule), text = the vocabulary's templates
          for this extractor (templates_by_extractor), cached in eval_seg's key format;
  2D path tokens -> bilinear upsample to the label grid -> unit -> PCA project (unit) -> logits
          with the PCA-projected text (no renormalisation, F7) -> argmax; every labelled pixel
          counts (the 2D row covers every pixel);
  metrics per scene: mIoU over the present classes (usplat.seg_eval.confusion_iou, the 'all'
          variant of Table I) and pixel accuracy, for the PCA-64 path (primary: the path every
          map row and the 2D row use) and the full-dimensional path (secondary, reported only).

Refuses any scene that is not in the prereg's validation split of --dataset, writes a
usplat.runlog record (git state, memory, timing) and the per-frame extraction time.
--max-views / --max-pca-frames (smoke runs) mark the record smoke = true; scripts/extractor_select.py
refuses smoke records.

    scripts/gpu_gate.sh 6000 python3 scripts/extractor_2d.py --prereg configs/extractor_prereg.json \
        --dataset replica --extractor radseg --text-cache OUT/text_cache_radseg.pt --out OUT/a_replica_radseg.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)


def _load_prereg(path):
    with open(path) as f:
        return json.load(f)


def dataset_spec(prereg, dataset):
    d = (prereg.get("datasets") or {}).get(dataset)
    if d is None:
        raise SystemExit(f"refused: dataset {dataset!r} is not in the prereg")
    return d


def validation_scenes(prereg, dataset, root=ROOT):
    """The prereg's validation scene list of `dataset` (eval_seg scene dicts), checked to be
    disjoint from its test list by name."""
    import eval_seg
    d = dataset_spec(prereg, dataset)
    sc = {}
    for split in ("validation", "test"):
        p = d["scenes"][split]
        sc[split] = eval_seg._load_structured(p if os.path.isabs(p) else os.path.join(root, p))
    both = sorted({s["name"] for s in sc["validation"]} & {s["name"] for s in sc["test"]})
    if both:
        raise SystemExit(f"refused: validation and test scenes overlap: {both}")
    return sc["validation"]


def mapping_data(spec, scene):
    """(dataset loader, mapping sequence root, stride) of a scene, from the prereg."""
    m = spec["mapping"]
    return m["dataset"], m["data"].format(name=scene), int(m["stride"])


def text_for(ext, extractor, vocab, idx, cache_path):
    """(len(idx), D) unit text embeddings in eval_seg.text_embeddings' cache format (key
    '<extractor>|<templates joined by |>|<prompt>'), computed with the extractor already loaded
    (no second copy on the GPU) for the prompts missing from the cache."""
    import eval_seg
    tpl = eval_seg.extractor_templates(vocab, extractor)
    key = lambda p: f"{extractor}|{'|'.join(tpl)}|{p}"
    cache = torch.load(cache_path, map_location="cpu") if cache_path and os.path.exists(cache_path) else {}
    prompts = [vocab["prompts"][i] for i in idx]
    missing = sorted({p for p in prompts if key(p) not in cache})
    if missing:
        emb = ext.text_embedding(missing, templates=tuple(tpl)).cpu()
        for p, e in zip(missing, emb):
            cache[key(p)] = e
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            torch.save(cache, cache_path)
    return torch.stack([cache[key(p)].float() for p in prompts]), tpl, len(missing)


def sync():
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


@torch.no_grad()
def fit_pca(ext, spec, scene, args, device):
    """run_mapping.py's PCA: the first pca_frames frames of the mapping sequence."""
    from usplat.datasets import make_dataset, downscale
    from usplat.clip_dense import upsample_tokens
    from usplat.feature_pca import StreamingPCA
    loader, data, stride = mapping_data(spec, scene)
    ds = make_dataset(loader, data, stride=stride)
    torch.manual_seed(args.seed)                     # as a fresh run_mapping process
    pca = StreamingPCA(ext.d, args.feat_dim, device=device)
    n = min(args.pca_frames, len(ds))
    if args.max_pca_frames is not None:
        n = min(n, args.max_pca_frames)
    for i in range(n):
        fr = ds[i]
        h, w = downscale(fr, args.downscale)["image"].shape[:2]
        z = upsample_tokens(ext.extract_tokens(fr["image"]), (h, w))
        pca.accumulate(z)
        del z
    explained = pca.fit()
    return pca, dict(loader=loader, data=data, stride=stride, frames=n,
                     explained=float(explained))


@torch.no_grad()
def score_scene(ext, sc, vocab, spec, args, device):
    import eval_seg
    from usplat.clip_dense import upsample_tokens
    from usplat.seg_eval import confusion_iou
    pca, pinfo = fit_pca(ext, spec, sc["name"], args, device)
    ns = argparse.Namespace(downscale=args.downscale, max_views=args.max_views, with_2d=True,
                            dataset=spec["eval"]["dataset"], stride=spec["eval"]["stride"])
    frames, n_unknown, skipped = eval_seg.load_frames(sc, vocab, ns)
    present = torch.unique(torch.cat([f["labels"].reshape(-1) for f in frames]))
    q = [int(i) for i in present.tolist() if i >= 0]
    local = torch.full((len(vocab["ids"]) + 1,), -1, dtype=torch.long)
    local[torch.tensor(q, dtype=torch.long)] = torch.arange(len(q))
    text, tpl, n_new = text_for(ext, args.extractor, vocab, q, args.text_cache)
    text = text.to(device)
    t_pca = pca.project_text(text)                                       # (K, 64), F7
    gts, preds, preds_full, ext_s = [], [], [], []
    for f in frames:
        sync()
        t0 = time.perf_counter()
        tok = ext.extract_tokens(f["image"])
        sync()
        ext_s.append(time.perf_counter() - t0)
        u = upsample_tokens(tok, (f["h"], f["w"]))                       # (h, w, D) unit
        z = pca.project(u).reshape(-1, t_pca.shape[1])
        lab = f["labels"].reshape(-1)
        gt = torch.where(lab >= 0, local[lab.clamp_min(0)], lab)
        keep = gt >= 0
        preds.append((z @ t_pca.t()).argmax(-1).cpu()[keep])
        preds_full.append((u.reshape(-1, u.shape[-1]) @ text.t()).argmax(-1).cpu()[keep])
        gts.append(gt[keep])
        f["image"] = None
        del tok, u, z
    gt = torch.cat(gts)
    out = dict(n_frames=len(frames), n_skipped=skipped, n_unknown=n_unknown,
               n_pixels=int(gt.numel()), n_classes=len(q),
               classes=[vocab["names"][i] for i in q], templates=len(tpl), text_computed=n_new,
               pca_fit=pinfo, ms_per_frame_extract=1000.0 * sum(ext_s) / max(len(ext_s), 1))
    for name, pr in (("pca", preds), ("full", preds_full)):
        p = torch.cat(pr)
        iou, gc = confusion_iou(gt, p, num_classes=len(q))
        has = gc > 0
        out[name] = dict(miou=float(iou[has].mean()), acc=float((p == gt).double().mean()),
                         iou={vocab["names"][q[i]]: float(iou[i])
                              for i in torch.nonzero(has).squeeze(1).tolist()})
    return out


def run(args, rec=None):
    import eval_seg
    from usplat.extractors import get_extractor
    prereg = _load_prereg(args.prereg)
    spec = dataset_spec(prereg, args.dataset)
    if args.extractor not in [prereg["incumbent"]] + list(prereg["candidates"]):
        raise SystemExit(f"refused: extractor {args.extractor!r} is not in the prereg")
    scenes = validation_scenes(prereg, args.dataset)
    if args.scene:
        scenes = [s for s in scenes if s["name"] in args.scene.split(",")]
        if not scenes or len(scenes) != len(args.scene.split(",")):
            raise SystemExit(f"refused: --scene {args.scene} is not a validation scene of "
                             f"{args.dataset}")
    device = args.device
    held = 0
    if device.startswith("cuda"):
        from usplat import gpu_reserve                  # fraction from scripts/gpu_gate.sh
        held = gpu_reserve.apply(device)
    vocab = eval_seg.load_vocab(spec["eval"]["vocab"])
    t0 = time.time()
    ext = get_extractor(args.extractor, device=device)
    t_build = time.time() - t0
    ext_opts = {k: getattr(ext, k) for k in ("token_stride", "flip", "scales", "half",
                                              "base_size", "crop_size") if hasattr(ext, k)}
    res = {}
    for sc in scenes:
        res[sc["name"]] = score_scene(ext, sc, vocab, spec, args, device)
        r = res[sc["name"]]
        print(f"{args.dataset} {sc['name']:>13} {args.extractor:>12}: 2D mIoU (PCA-64) "
              f"{r['pca']['miou'] * 100:.2f} acc {r['pca']['acc'] * 100:.2f} | full-dim mIoU "
              f"{r['full']['miou'] * 100:.2f} | {r['n_frames']} frames, {r['n_classes']} classes, "
              f"{r['ms_per_frame_extract']:.0f} ms/frame", flush=True)
        if rec is not None:
            rec.add("scenes", res)
            rec.flush()
    mean = lambda k, m: sum(r[k][m] for r in res.values()) / len(res)
    summary = {k: dict(miou=mean(k, "miou"), acc=mean(k, "acc")) for k in ("pca", "full")}
    out = dict(dataset=args.dataset, extractor=args.extractor, extractor_options=ext_opts,
               d=int(ext.d), scenes=res, summary=summary, n_scenes=len(res),
               smoke=bool(args.max_views is not None or args.max_pca_frames is not None
                          or args.scene),
               build_s=t_build, gpu_reserved_mib=held,
               peak_alloc_mib=(torch.cuda.max_memory_allocated() / 2 ** 20
                               if device.startswith("cuda") else None),
               prereg=dict(path=os.path.abspath(args.prereg),
                           sha256=__import__("hashlib").sha256(open(args.prereg, "rb").read()).hexdigest()),
               vocab=os.path.abspath(spec["eval"]["vocab"]))
    print(f"{args.dataset} {args.extractor}: mean 2D mIoU (PCA-64) {summary['pca']['miou'] * 100:.2f} "
          f"over {len(res)} validation scenes")
    if rec is not None:
        for k, v in out.items():
            rec.add(k, v)
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--extractor", required=True)
    ap.add_argument("--text-cache", default="", help=".pt cache (eval_seg format); a COPY")
    ap.add_argument("--device", default="cuda")
    # the protocol (Table I's: 2, 64, 20, 0) comes from the prereg's stage_a.protocol; a flag
    # may only repeat it
    ap.add_argument("--downscale", type=int, default=None)
    ap.add_argument("--feat-dim", type=int, default=None)
    ap.add_argument("--pca-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scene", default="", help="smoke: a subset of the validation scenes")
    ap.add_argument("--max-views", type=int, default=None, help="smoke: first n eval frames")
    ap.add_argument("--max-pca-frames", type=int, default=None, help="smoke: fewer PCA frames")
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    prereg = _load_prereg(args.prereg)
    fixed = (prereg.get("stage_a") or {}).get("protocol") or {}
    if sorted(fixed) != ["downscale", "feat_dim", "pca_frames", "seed"]:
        raise SystemExit("refused: the prereg's stage_a.protocol must give downscale, feat_dim, "
                         "pca_frames and seed")
    bad = {k: (getattr(args, k), v) for k, v in fixed.items()
           if getattr(args, k) is not None and getattr(args, k) != v}
    if bad:
        raise SystemExit(f"refused: settings differ from the prereg's stage A protocol: {bad}")
    for k, v in fixed.items():
        setattr(args, k, int(v))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    from usplat.runlog import RunRecord
    with RunRecord(args.out, config=vars(args), name="extractor_2d",
                   argv=[sys.argv[0]] + list(argv if argv is not None else sys.argv[1:])) as rec:
        out = run(args, rec)
    print(f"wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
