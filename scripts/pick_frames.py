"""
pick_frames.py — the selection rules of the qualitative figures, applied to ground-truth labels
and to published result JSONs only (no method output is rendered or read per pixel).

Subcommand `static` (fig:qual-compare and the supplementary grid supp_compare_all; rules fixed
on 2026-10-21 before any new render, docs/reports/2026-10-20-figures-video/logs/design/
qual_review.md section 2):
  scene rule  per dataset, the test scene with the lower-median per-scene mIoU of ours_P2 in
              the published Table I JSON (rows.ours_P2.all.miou): position (n - 1) // 2 of the
              scenes sorted by (mIoU, name), so the lower of the two middle scenes when n is
              even. Needs the table (third item of --set).
  frame rule  per scene, the evaluation frame with the most ground-truth classes that each
              cover at least --min-share (0.5%) of the frame's labelled pixels, on the
              evaluation grid (eval_seg.load_frames: every stride-th frame from offset, frames
              with a non-finite pose skipped, labels resized by nearest neighbour to
              1/--downscale and mapped through the vocabulary). Ties go to the earliest frame.
              "frame" is the position in that list, the --frame of render_qual.py and
              qual_compare.py; "index" is the dataset frame number.
--check compares the outcomes with EXPECTED (the review's main and supplementary frames) and
exits 2 if any differ, so a figure is never drawn from a changed rule outcome.

    cd uags && python3 scripts/pick_frames.py static --check \
        --set replica=configs/replica_test.json,configs/vocab_replica.json,<results>/tab1/eval_test.json \
        --set scannet=configs/scannet_test.json,configs/vocab_scannet.json,<results>/scannet_tab1/eval_test.json \
        --out <results>/qual/select/static.json

CPU only. Other figures' rules (NOR) are not in this file.
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

MIN_SHARE = 0.005
SCENE_ROW = "ours_P2"
# Outcomes fixed in the review (qual_review.md section 2 and "Supplementary"): set -> scene,
# scene -> evaluation frame (position in the frame list).
EXPECTED = dict(
    scenes=dict(replica="room_1", scannet="scene0140_00"),
    frames=dict(room_0=18, room_1=66, office_1=25, office_2=75, office_3=15, office_4=29,
                scene0070_00=46, scene0097_00=2, scene0140_00=57, scene0200_00=36,
                scene0347_00=0, scene0400_00=0, scene0590_00=104, scene0645_00=156))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def table_body(path):
    d = json.load(open(path))
    return d["metrics"] if isinstance(d.get("metrics"), dict) else d


# ------------------------------------------------------------------------------- rules
def lower_median_scene(table, row=SCENE_ROW):
    """(pick, ranking): the scene at position (n - 1) // 2 of the scenes sorted by the row's
    per-scene mIoU (variant "all"), ties by scene name."""
    rank = sorted((float(v["rows"][row]["all"]["miou"]), s) for s, v in table["scenes"].items())
    if not rank:
        raise ValueError("the table has no scenes")
    return rank[(len(rank) - 1) // 2][1], [[s, m] for m, s in rank]


def frame_classes(labels, min_share=MIN_SHARE):
    """(classes, n_labelled): vocabulary indices covering >= min_share of the labelled pixels
    (label >= 0) of one frame, sorted by index."""
    lab = torch.as_tensor(labels).reshape(-1)
    lab = lab[lab >= 0]
    n = int(lab.numel())
    if n == 0:
        return [], 0
    cnt = torch.bincount(lab)
    return [int(c) for c in torch.nonzero(cnt >= min_share * n).squeeze(1)], n


def pick_frame(table):
    """The row of `table` (dicts with frame, n_classes) with the most classes, then the
    earliest frame."""
    return min(table, key=lambda t: (-t["n_classes"], t["frame"]))


def frame_table(sc, vocab, args):
    """Per evaluation frame of scene sc: frame, index, labelled, n_classes, classes."""
    frames, _, skipped = ES.load_frames(sc, vocab, args)
    rows = []
    for i, f in enumerate(frames):
        cls, n = frame_classes(f["labels"], args.min_share)
        rows.append(dict(frame=i, index=int(f["idx"]), labelled=n, n_classes=len(cls),
                         classes=[vocab["names"][c] for c in cls]))
    return rows, skipped


def static(args):
    out = dict(rule=dict(
        scene=f"per dataset, the test scene with the lower-median per-scene mIoU of {SCENE_ROW} "
              f"in the published Table I JSON: position (n-1)//2 of the scenes sorted by "
              f"(mIoU, name)",
        frame=f"per scene, the evaluation frame with the most GT classes each covering >= "
              f"{args.min_share:g} of the frame's labelled pixels (downscale {args.downscale}); "
              f"ties -> earliest frame"),
        fixed="2026-10-21, before any new render (qual_review.md section 2)", sets={})
    for spec in args.set:
        name, files = spec.split("=", 1)
        files = files.split(",")
        scenes_path, vocab_path = files[0], files[1]
        table_path = files[2] if len(files) > 2 else ""
        vocab = ES.load_vocab(vocab_path)
        entry = dict(scenes=scenes_path, vocab=vocab_path, scenes_sha256=sha256(scenes_path),
                     vocab_sha256=sha256(vocab_path), frames={})
        if table_path:
            pick, rank = lower_median_scene(table_body(table_path))
            entry.update(table=table_path, table_sha256=sha256(table_path),
                         scene_rule=dict(row=SCENE_ROW, ranking=rank, pick=pick))
            print(f"{name}: scene rule -> {pick}  (ranking by {SCENE_ROW} mIoU: "
                  + ", ".join(f"{s} {100 * m:.2f}" for s, m in rank) + ")")
        for sc in ES._load_structured(scenes_path):
            t0 = time.time()
            table, skipped = frame_table(sc, vocab, args)
            best = pick_frame(table)
            entry["frames"][sc["name"]] = dict(
                n_frames=len(table), n_skipped=skipped, pick=best,
                top5=sorted(table, key=lambda t: (-t["n_classes"], t["frame"]))[:5],
                table=[{k: t[k] for k in ("frame", "index", "labelled", "n_classes")}
                       for t in table])
            print(f"{name}: {sc['name']} -> frame {best['frame']} (index {best['index']}, "
                  f"{best['n_classes']} classes); {len(table)} frames, {skipped} skipped, "
                  f"{time.time() - t0:.1f} s", flush=True)
        out["sets"][name] = entry
    if args.check:
        out["check"] = check(out)
    return out


def check(out):
    """Compare the outcomes with EXPECTED (only the scenes and sets present)."""
    bad, ok = [], []
    for name, e in out["sets"].items():
        want = EXPECTED["scenes"].get(name)
        if want is not None and "scene_rule" in e:
            (ok if e["scene_rule"]["pick"] == want else bad).append(
                f"{name}: scene {e['scene_rule']['pick']} (expected {want})")
        for s, f in e["frames"].items():
            want = EXPECTED["frames"].get(s)
            if want is not None:
                (ok if f["pick"]["frame"] == want else bad).append(
                    f"{s}: frame {f['pick']['frame']} (expected {want})")
    return dict(status="PASS" if not bad and ok else "FAIL", n_checked=len(ok) + len(bad),
                mismatches=bad)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("static", help="scene and frame rules of the static comparison figure")
    st.add_argument("--set", action="append", required=True,
                    help="name=scenes.json,vocab.json[,table.json] (repeatable); the table "
                         "(published eval_seg JSON) enables the scene rule")
    st.add_argument("--min-share", type=float, default=MIN_SHARE)
    st.add_argument("--downscale", type=int, default=2, help="evaluation grid divisor")
    st.add_argument("--dataset", default="replica_semnerf", help="default dataset of a scene")
    st.add_argument("--stride", type=int, default=10, help="default stride of a scene")
    st.add_argument("--check", action="store_true",
                    help="compare with the review's outcomes; exit 2 if any differ")
    st.add_argument("--out", required=True, help="JSON with the rule outcomes and frame tables")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.max_views, args.with_2d = None, False             # every evaluation frame, no images
    t0 = time.time()
    out = static(args)
    out["provenance"] = dict(argv=[os.path.basename(sys.argv[0])] + list(
        argv if argv is not None else sys.argv[1:]), wall_s=time.time() - t0,
        script_sha256=sha256(os.path.abspath(__file__)))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")
    if args.check:
        c = out["check"]
        print(f"check {c['status']}: {c['n_checked']} outcomes compared"
              + ("" if not c["mismatches"] else "; MISMATCH " + "; ".join(c["mismatches"])))
        if c["status"] != "PASS":
            raise SystemExit(2)
    return out


if __name__ == "__main__":
    main()
