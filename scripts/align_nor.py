"""
align_nor.py — fit the update-pose alignment (protocol D3) of NeRF-Object-Removal scenes and
save one JSON per scene, on the CPU (usplat/nor_align.py says how).

  PYTHONPATH=. python3 scripts/align_nor.py --root <data>/nerf-object-removal \
      --out-dir <results>/nor/align [--scenes 012,016]

Writes <out-dir>/<scene>.json (the record the loader reads with align=<json>, see
usplat/datasets.py NeRFObjectRemovalDataset) and <out-dir>/summary.json (--summary: another
path, e.g. for a run on some scenes that must not replace the summary of all), and prints per
scene the transform (translation, rotation), the ICP pairs and directions, and on the
held-out check pairs the median depth error and the photometric NCC before and after the
alignment. Exit code 1 if a scene failed.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from usplat.nor_align import DZ_OK, INFLATE, estimate_alignment, save_alignment  # noqa: E402


def summary_line(rec):
    b, a, ic = rec["before"], rec["after"], rec["icp"]
    nan = float("nan")
    first = ic[0] if ic else dict(pairs=0, rms_m=nan, directions=0)
    last = ic[-1] if ic else dict(pairs=0, rms_m=nan, step_m=nan, directions=0)
    sh = rec["shift"]
    return (f"{rec['scene']} shift median {100 * sh['median_m']:.2f} cm (max "
            f"{100 * sh['max_m']:.2f}) rot {rec['rotation_deg']:.3f} deg | ICP {len(ic)} its on "
            f"{len(rec['fit_pairs'])} "
            f"pairs: {last['pairs']} point pairs, rms {100 * first['rms_m']:.2f} -> "
            f"{100 * last['rms_m']:.2f} cm, {last['directions']}/6 directions, last step "
            f"{1000 * last['step_m']:.3f} mm | check ({b['n_pairs']} pairs): med |dz| "
            f"{100 * b['med_abs_dz']:.2f} -> {100 * a['med_abs_dz']:.2f} cm, ok5cm "
            f"{b['ok5cm']:.3f} -> {a['ok5cm']:.3f}, photo NCC {b['ncc']:.3f} -> {a['ncc']:.3f} "
            f"(map-map {rec['photo_map_map']['ncc']:.3f})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="folder holding the scene folders 001..016")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--summary", default="", help="summary JSON (default <out-dir>/summary.json)")
    ap.add_argument("--scenes", default="", help="comma-separated scene names (default: all)")
    ap.add_argument("--check-pairs", type=int, default=12,
                    help="held-out map-update pairs of the checks (check_nor.py's pairs)")
    ap.add_argument("--fit-frames", type=int, default=48,
                    help="evenly spaced update frames for the fit (minus the check frames)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--trim", type=float, default=DZ_OK, help="[m] ICP pair distance limit")
    ap.add_argument("--tau", type=float, default=0.1,
                    help="eigen-directions below tau * the largest stay at the identity")
    ap.add_argument("--pixel-step", type=int, default=2)
    ap.add_argument("--inflate", type=float, default=INFLATE, help="[m] box margin left out")
    args = ap.parse_args(argv)
    scenes = ([s.strip() for s in args.scenes.split(",") if s.strip()] if args.scenes else
              sorted(d for d in os.listdir(args.root)
                     if os.path.isfile(os.path.join(args.root, d, "meta.json"))))
    t0 = time.time()
    cmd = ["python3", "scripts/align_nor.py"] + list(sys.argv[1:] if argv is None else argv)
    summary, failed = dict(command=" ".join(cmd), start=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                           scenes={}), []
    for s in scenes:
        t1 = time.time()
        try:
            rec = estimate_alignment(os.path.join(args.root, s), check_pairs=args.check_pairs,
                                     fit_frames=args.fit_frames, iters=args.iters,
                                     trim=args.trim, tau=args.tau, pixel_step=args.pixel_step,
                                     inflate=args.inflate)
        except Exception as e:                                # keep fitting the others
            failed.append(s)
            summary["scenes"][s] = dict(error=f"{type(e).__name__}: {e}")
            print(f"{s} FAILED {type(e).__name__}: {e}", flush=True)
            continue
        rec.update(command=" ".join(cmd), wall_s=time.time() - t1)
        path = save_alignment(os.path.join(args.out_dir, f"{s}.json"), rec)
        summary["scenes"][s] = dict(path=os.path.abspath(path), T=rec["T"],
                                    shift=rec["shift"], rotation_deg=rec["rotation_deg"],
                                    before=rec["before"], after=rec["after"],
                                    photo_map_map=rec["photo_map_map"],
                                    icp_iterations=len(rec["icp"]))
        print(summary_line(rec), flush=True)
    summary.update(failed=failed, wall_s=time.time() - t0)
    save_alignment(args.summary or os.path.join(args.out_dir, "summary.json"), summary)
    print(f"{len(scenes) - len(failed)} scenes aligned, failed: {failed}; "
          f"{summary['wall_s']:.0f} s; wrote {args.out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
