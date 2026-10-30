"""
make_swap_bank.py — the feature bank and the resolved (A -> B) pairs of one Replica scene for
the semantic-swap benchmark (usplat.swap, configs/swap_replica.json).

  1. label statistics of the mapping frames (Sequence_1) and the update frames (Sequence_2)
     at the accumulation grid (spec downscale): labels only. Every frame of the strided
     sequences counts (run_mapping.py's near-duplicate pose filter drops at most a few);
  2. text embeddings of the present classes with the extractor's frozen templates;
  3. the pair rule of the spec -> the pairs (labels + text only, no method output);
  4. the target segments from every frame_step-th mapping frame: extractor tokens ->
     upsample to the accumulation grid -> unit D-dim features (before any PCA) -> crop, fill,
     resize, fp16 (usplat.swap.build_bank).
Writes the bank (--out, .pt) and a run record (<out>.json) holding the pairs and statistics.
The extractor runs on --device: gate it like every GPU step. CPU tests pass a fake extractor
to run().

    scripts/gpu_gate.sh 4000 python3 scripts/make_swap_bank.py --spec configs/swap_replica.json \
        --scene office_0 --root /home/thanhnc/gauss_ws/src/uags/data/replica_semnerf/Replica_Dataset \
        --out /home/thanhnc/gauss_ws/src/uags/results/swap/banks/office_0.pt
"""
from __future__ import annotations
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from usplat import swap as SW                                              # noqa: E402

ROOT = "/home/thanhnc/gauss_ws/src/uags/data/replica_semnerf/Replica_Dataset"


def run(args, rec=None, extractor=None):
    from usplat.datasets import make_dataset
    from eval_seg import load_vocab, extractor_templates
    spec, sha = SW.load_spec(args.spec)
    if args.scene not in sum(spec["scenes"].values(), []):
        raise ValueError(f"scene {args.scene!r} is not in the spec's splits {spec['scenes']}")
    vocab = load_vocab(args.vocab or spec["vocab"])
    ext_name = args.extractor or spec["extractor"]
    tpl = extractor_templates(vocab, ext_name)
    root = os.path.join(args.root, args.scene)
    map_ds = make_dataset(spec["dataset"], os.path.join(root, spec["mapping"]["sequence"]),
                          stride=int(spec["mapping"]["stride"]))
    upd_ds = make_dataset(spec["dataset"], os.path.join(root, spec["update"]["sequence"]),
                          stride=int(spec["update"]["stride"]))
    if extractor is None:
        from usplat.extractors import get_extractor
        device = args.device
        if device.startswith("cuda"):
            from usplat import gpu_reserve
            gpu_reserve.apply(device)
        extractor = get_extractor(ext_name, device=device)
    bank = SW.build_bank(spec, sha, args.scene, map_ds, range(len(map_ds)), upd_ds,
                         range(len(upd_ds)), extractor, vocab, tpl, spec["downscale"],
                         extractor_name=ext_name)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = f"{args.out}.tmp{os.getpid()}"
    try:
        torch.save(bank, tmp)
        os.replace(tmp, args.out)
    except BaseException:                    # a full disk: no stray temp file
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    n_seg = {int(b): len(v) for b, v in bank["segments"].items()}
    print(f"{args.scene}: bank of {sum(n_seg.values())} segments {n_seg} -> {args.out}")
    if rec is not None:
        rec.add_input(args.spec)
        rec.add_input(args.vocab or spec["vocab"])
        rec.add_output(args.out)
        for k in ("scene", "spec_sha256", "pairs", "classes", "stats", "extractor", "templates",
                  "downscale"):
            rec.add(k, bank[k])
        rec.add("segments", {str(b): [{k: s[k] for k in ("frame", "component", "area", "bbox")}
                                      for s in v] for b, v in bank["segments"].items()})
    return bank


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", default="configs/swap_replica.json")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--root", default=ROOT, help="Replica_Dataset root (Semantic-NeRF renders)")
    ap.add_argument("--vocab", default="", help="default: the spec's vocabulary")
    ap.add_argument("--extractor", default="", help="default: the spec's extractor")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--record", default="", help="run record (default: <out>.json)")
    ap.add_argument("--plain-json", action="store_true", help="no run record")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="bank .pt")
    return ap


def main(argv=None, extractor=None):
    args = build_parser().parse_args(argv)
    if args.plain_json:
        return run(args, extractor=extractor)
    from usplat.runlog import RunRecord
    path = args.record or f"{args.out}.json"
    with RunRecord(path, config=vars(args), seed=args.seed, name="make_swap_bank",
                   argv=[sys.argv[0]] + list(argv if argv is not None else sys.argv[1:])) as rec:
        bank = run(args, rec, extractor=extractor)
    print(f"wrote {path}")
    return bank


if __name__ == "__main__":
    main()
