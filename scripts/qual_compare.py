"""
qual_compare.py — the static comparison figure (fig:qual-compare) and the supplementary grid
(supp_compare_all): one evaluation frame of a test scene, every Table I method side by side.

Block of one scene (6 columns x 3 rows, at the evaluation grid: Replica 320x240, ScanNet
324x242):
    columns  Reference | 2D features, per frame | Feature 3DGS | Averaging / NormLift |
             LEGaussians (lambda_u of Table I: --legs legs_l4 on Replica, legs_l1 on ScanNet) |
             Ours P2
    row 1    the 3D-GS render | the label maps (render_qual.scene_palette; black = no decision:
             not covered or tied). Averaging and NormLift share one predictive (same feature,
             same beta), so one map serves both.
    row 2    the ground truth | the errors (white right, vermillion wrong, black not scored:
             unlabelled or not covered)
    row 3    a key | each method's OWN uncertainty only: none for the 2D features and Feature
             3DGS ("no own uncertainty", the '--' of Table I), NormLift 1 - C_inter, the learned
             u of LEGaussians, our u_ale. Each pixel shows its mid-rank among the pooled
             population of that channel: the covered labelled pixels with a defined value of
             ALL evaluation frames of the scene (the finite part of the AUSE population of
             Table I). Grey = undefined, black = not covered. No P2 confidence panel.
    numbers  per-scene mIoU (row 1) and the AUSE of the own uncertainty (row 3), in %, read
             from the published Table I JSON (--table) at figure time; never recomputed.
Every per-pixel value comes from eval_seg.py's code (load_scene, make_rasterizer,
usplat.seg_eval.evaluate_view, frame_features_2d, PixelSource) and render_qual.frame_maps, so
the panels show the decisions that Table I scores.

Stages (the two GPU stages are separate processes, each behind scripts/gpu_gate.sh):
  tokens     (GPU, LSeg) the 2D row's tokens of the chosen frames: eval_seg.load_frames
             (images kept) -> eval_seg._extractor(args).extract_tokens, saved on the CPU,
             one file <out-dir>/tok_<scene>_f<frame>.pt per --target scene:frame.
  render     (GPU, gsplat) one scene: load_scene with the shown rows (no P3, no 2D row); one
             pass over every evaluation frame with the real labels (evaluate_view) that
             (a) pools each own channel for the ranks, (b) re-scores the shown rows with
             SegAccumulator and compares mIoU and the own-channel AUSE with the published
             table (a check only, |delta| <= --repro-tol, else exit 2; the figure never prints
             these), (c) counts, per frame, the confident errors of ours P2 for figure F; then
             the chosen frame at every pixel (frame_maps; the 2D row from the saved tokens via
             frame_features_2d + PixelSource). Writes <out-dir>/<scene>_f<frame>.pt + .json.
  figure     (CPU) composes saved blocks into one figure (two blocks for the paper; --pages k
             for a multi-page grid), reads the numbers from each block's table (sha256
             checked), writes <out>.pdf/.png, a caption (<out>.txt) generated from the same
             numbers, and <out>.json.
  confident  (CPU) the static confident-error listing of figure F from the render metas of
             the 14 test scenes: per scene, medians of u_ale and u_epi over the covered
             labelled pixels with a defined value; per frame, the covered labelled pixels that
             ours P2 gets wrong with u_ale defined and <= its median and u_epi <= its median
             (undefined u_ale is not confident); the frame with the most such pixels, ties to
             config order (the order of --metas), then the earliest frame.

    cd uags && E="--vocab configs/vocab_replica.json --scenes configs/replica_test.json"
    UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5 scripts/gpu_gate.sh 4500 python3 \
        scripts/qual_compare.py --stage tokens $E --target room_1:66 --out-dir <cmp>
    UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5 scripts/gpu_gate.sh 4500 python3 \
        scripts/qual_compare.py --stage render $E --scene room_1 --frame 66 --legs legs_l4 \
        --betas <results>/tab1/eval_val.json --table <results>/tab1/eval_test.json \
        --text-cache <results>/qual/text_cache_lseg_replica.pt \
        --row-factory usplat.baselines:row_factory,usplat.distill:row_factory --out-dir <cmp>
    python3 scripts/qual_compare.py --stage figure --vocab configs/vocab_replica.json \
        --blocks <cmp>/room_1_f66.pt,<cmp>/scene0140_00_f57.pt --out papers/figs/qual_compare.pdf
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import eval_seg as ES                                                      # noqa: E402
import render_qual as RQ                                                   # noqa: E402
from usplat.rows import TensorRow                                          # noqa: E402
from usplat.seg_eval import (evaluate_view, SegAccumulator, confusion_iou,  # noqa: E402
                             ranking_signals)
from usplat.evaluation import sparsification_curve                        # noqa: E402
from usplat.paths import moved                                              # noqa: E402

# colours: Okabe-Ito vermillion for wrong; class colours from render_qual.scene_palette;
# viridis ranks (RQ.U_CMAP), RQ.UNDEF grey undefined, black not covered / not scored
RIGHT = (1.0, 1.0, 1.0)
WRONG = (0.835, 0.369, 0.0)              # Okabe-Ito vermillion #D55E00
BLACK = RQ.BLACK
NONE_BG = (0.93, 0.93, 0.93)             # "no own uncertainty" slot
LEFT = 0.32                              # in: block label + row labels
BLOCK_GAP = 0.10                         # in, between blocks
FS_NOTE = 5.5                            # pt, numbers inside panels
LEGEND_NCOL = 8
COLS = ("ref", "2d_features", "f3dgs", "normlift", "legs", "ours_P2")
TITLES = ("Reference", "2D features, per frame", "Feature 3DGS", "Averaging / NormLift",
          "LEGaussians", "Ours P2")
ROW_LABELS = ("labels", "errors", "own uncertainty")
OWN_NOTE = {"normlift": "$1-C_{\\mathrm{inter}}$:", "legs": "$u$:", "ours_P2": "$u^{\\mathrm{ale}}$:"}
REPRO_TOL = 0.002                        # |published - re-scored| for mIoU and AUSE (fractions)
LSEG_ENV = ("UAGS_LSEG_STRIDE", "UAGS_LSEG_FLIP", "UAGS_LSEG_SCALES", "UAGS_LSEG_HALF")


# ------------------------------------------------------------------------------ helpers
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_head():
    import subprocess
    try:
        r = subprocess.run(["git", "-C", os.path.dirname(HERE), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:                                    # noqa: BLE001
        return None


def table_body(path):
    d = json.load(open(path))
    return d["metrics"] if isinstance(d.get("metrics"), dict) else d


def own_channel(row):
    """The row's own uncertainty channel (Table I AUSE_u column), None if it has none."""
    if row == "normlift":
        return "normlift_1-Cinter"
    if row.startswith("legs_l"):
        return "legs_u"
    if row == "ours_P2":
        return "u_ale"
    return None


def col_rows(legs):
    """{column: the row whose decisions it shows}, and {column: the table row of its mIoU}."""
    shown = {"2d_features": ES.ROW_2D, "f3dgs": "f3dgs", "normlift": "normlift", "legs": legs,
             "ours_P2": "ours_P2"}
    table = dict(shown, normlift="averaging")
    return shown, table


def mid_ranks(values, pool):
    """Mid-rank of each value in the sorted pool, in [0, 1]: (#{pool < v} + #{pool <= v}) /
    (2 n). NaN values stay NaN; an empty pool gives NaN."""
    v = np.asarray(values, dtype=np.float64)
    pool = np.asarray(pool, dtype=np.float64)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if pool.size == 0 or not ok.any():
        return out
    lo = np.searchsorted(pool, v[ok], side="left")
    hi = np.searchsorted(pool, v[ok], side="right")
    out[ok] = (lo + hi) / (2.0 * pool.size)
    return out


def maps_from_out(o, gt, h, w):
    """render_qual.frame_maps' dict from one evaluate_view output made with gt = zeros (every
    pixel), scored against the real gt (h*w,) afterwards (used for the 2D row)."""
    gt = gt.reshape(h, w).cpu()
    pred = o["pred"].reshape(h, w).long()
    lab = gt >= 0
    return dict(gt=gt, pred=pred, covered=o["covered"].reshape(h, w), labelled=lab,
                correct=lab & (pred == gt), u={k: v.reshape(h, w) for k, v in o["u"].items()})


def error_rgb(m):
    """(h, w, 3): white right, vermillion wrong, black not scored (unlabelled or not covered)."""
    ok = (m["labelled"] & m["covered"]).numpy()
    right = m["correct"].numpy()
    img = np.zeros(ok.shape + (3,))
    img[ok & right] = RIGHT
    img[ok & ~right] = WRONG
    return img


def gpu_setup(args):
    if str(args.device).startswith("cuda"):
        from usplat import gpu_reserve                      # fraction from scripts/gpu_gate.sh
        held = gpu_reserve.apply(args.device)
        print(f"reserved {held} MiB of GPU memory", flush=True)
        return held
    return 0


def gpu_peak(args):
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        return dict(peak_alloc_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                    reserved_mib=torch.cuda.memory_reserved() / 2 ** 20)
    return {}


def parse_targets(spec):
    out = []
    for t in [s.strip() for s in spec.split(",") if s.strip()]:
        name, f = t.rsplit(":", 1)
        out.append((name, int(f)))
    return out


def tok_path(out_dir, scene, frame):
    return os.path.join(out_dir, f"tok_{scene}_f{frame}.pt")


def block_path(out_dir, scene, frame):
    return os.path.join(out_dir, f"{scene}_f{frame}.pt")


def rule_frame(select, scene):
    """The frame the static rule picked for scene (scripts/pick_frames.py static output)."""
    d = json.load(open(select))
    for e in d["sets"].values():
        if scene in e["frames"]:
            return int(e["frames"][scene]["pick"]["frame"])
    raise ValueError(f"{select}: no rule outcome for scene {scene!r}")


# ------------------------------------------------------------------------------- tokens
def prefix_frames(sc, vocab, args, i):
    """The evaluation frames of eval_seg.load_frames up to position i at least: a prefix of
    the full list (--max-views cuts the dataset frames; frames with a non-finite pose are
    skipped, so the cut grows until position i exists)."""
    keep, n = args.max_views, i + 1
    try:
        while True:
            args.max_views = n
            frames, _, skipped = ES.load_frames(sc, vocab, args)
            if len(frames) > i:
                return frames
            if len(frames) + skipped < n:          # the dataset ended before position i
                raise ValueError(f"{sc['name']}: frame {i} not in 0..{len(frames) - 1}")
            n += i + 1 - len(frames)
    finally:
        args.max_views = keep


def stage_tokens(args):
    """LSeg tokens of each --target frame (the 2D row), saved on the CPU."""
    held = gpu_setup(args)
    env = {k: os.environ[k] for k in LSEG_ENV if k in os.environ}
    if env:
        raise SystemExit(f"LSeg options {env} are set; Table I ran without them")
    vocab = ES.load_vocab(args.vocab)
    os.makedirs(args.out_dir, exist_ok=True)
    args.with_2d = True                                   # load_frames keeps the images
    written = []
    for name, i in parse_targets(args.target):
        t0 = time.time()
        sc = RQ.pick_scene(args.scenes, name)
        frames = prefix_frames(sc, vocab, args, i)
        fr = frames[i]
        img = fr["image"]
        tok = ES._extractor(args).extract_tokens(img).cpu()
        p = tok_path(args.out_dir, name, i)
        torch.save(dict(tok=tok, scene=name, frame=i, index=int(fr["idx"]), h=fr["h"],
                        w=fr["w"], image_shape=list(img.shape),
                        image_sha256=hashlib.sha256(np.ascontiguousarray(
                            np.asarray(img)).tobytes()).hexdigest(),
                        extractor=args.extractor, lseg_env=env), p)
        written.append(p)
        print(f"{name} frame {i} (index {int(fr['idx'])}): tokens {tuple(tok.shape)} -> {p} "
              f"({time.time() - t0:.1f} s)", flush=True)
        del frames
    return dict(stage="tokens", outputs=written, gpu_reserved_mib=held, **gpu_peak(args))


# ------------------------------------------------------------------------------- render
def scene_pass(S, rows, betas, args, own):
    """Every evaluation frame with its real labels: the SegAccumulator of the shown rows, the
    pooled own-channel values (covered, labelled, defined), and the per-frame arrays of ours
    P2 for the confident-error listing."""
    raw = ES._raw_probs(args)
    acc = SegAccumulator(class_names=S["class_names"])
    pools = {r: [] for r in own}
    per_frame = []
    for f in S["frames"]:
        ras = ES.make_rasterizer(args.backend, S["splats"], f, args)
        out = evaluate_view(rows, ras, S["text"], f["gt"], betas, args.tau_cov,
                            undef_frac=args.undef_frac, raw_probs=raw)
        acc.add(out)
        for r, ch in own.items():
            u, cov = out[r]["u"][ch], out[r]["covered"]
            pools[r].append(u[cov & torch.isfinite(u)])
        if "ours_P2" in out:
            o = out["ours_P2"]
            per_frame.append(dict(covered=o["covered"], correct=o["correct"],
                                  u_ale=o["u"]["u_ale"], u_epi=o["u"]["u_epi"]))
    pooled = {r: np.sort(torch.cat(v).double().numpy()) for r, v in pools.items()}
    return acc, pooled, per_frame


def confident_counts(per_frame, idx):
    """Scene medians of u_ale, u_epi over covered labelled pixels with a defined value, then per
    frame the covered labelled pixels ours P2 gets wrong with u_ale defined and <= its median
    and u_epi <= its median (undefined u_ale is not confident)."""
    if not per_frame:
        return None
    cov = torch.cat([p["covered"] for p in per_frame])
    ua = torch.cat([p["u_ale"] for p in per_frame])
    ue = torch.cat([p["u_epi"] for p in per_frame])
    med_a = float(np.median(ua[cov & torch.isfinite(ua)].double().numpy()))
    med_e = float(np.median(ue[cov & torch.isfinite(ue)].double().numpy()))
    frames = []
    for i, p in enumerate(per_frame):
        a, e = p["u_ale"], p["u_epi"]
        conf = torch.isfinite(a) & (a <= med_a) & torch.isfinite(e) & (e <= med_e)
        wrong = p["covered"] & ~p["correct"]
        frames.append(dict(frame=i, index=idx[i], confident_errors=int((wrong & conf).sum()),
                           covered_wrong=int(wrong.sum()), covered=int(p["covered"].sum())))
    return dict(median_u_ale=med_a, median_u_epi=med_e, frames=frames)


def rescore(acc, rows, own, ause_bins):
    """{row: {"all": {miou, ranking: {own channel: {ause}}}}}: the two numbers the figure
    prints, re-scored exactly as usplat.seg_eval.summarize does for the "all" variant
    (confusion_iou over the classes present; sparsification_curve of the three-tier
    ranking_signals key against the 0/1 error), without its other metrics."""
    out = {}
    for r in rows:
        d = acc.arrays(r)
        iou, gc = confusion_iou(d["gt"], d["pred"])
        e = dict(miou=float(iou[gc > 0].mean()), ranking={})
        ch = own.get(r)
        if ch:
            key = ranking_signals(d)[ch]
            err = 1.0 - d["correct"].double()
            e["ranking"][ch] = dict(ause=sparsification_curve(
                err, key, num_bins=ause_bins, undefined="last")["ause"])
        out[r] = {"all": e}
    return out


def repro_check(metrics, table_scene, shown_rows, own, tol, n_views, n_frames):
    """Re-scored mIoU / own-channel AUSE of the shown rows vs the published scene record."""
    rows, worst = {}, 0.0
    for r in shown_rows:
        pub = table_scene["rows"][r]["all"]
        new = metrics[r]["all"]
        e = dict(miou_published=pub["miou"], miou_rescored=new["miou"],
                 miou_delta=new["miou"] - pub["miou"])
        worst = max(worst, abs(e["miou_delta"]))
        ch = own.get(r)
        if ch:
            e.update(channel=ch, ause_published=pub["ranking"][ch]["ause"],
                     ause_rescored=new["ranking"][ch]["ause"],
                     ause_delta=new["ranking"][ch]["ause"] - pub["ranking"][ch]["ause"])
            worst = max(worst, abs(e["ause_delta"]))
        rows[r] = e
    same = table_scene["rows"]["averaging"]["all"]["miou"] == table_scene["rows"]["normlift"][
        "all"]["miou"]
    ok = worst <= tol and same and n_views == n_frames
    return dict(status="PASS" if ok else "FAIL", tol=tol, max_abs_delta=worst, rows=rows,
                averaging_equals_normlift=same, n_views_published=n_views, n_frames=n_frames)


def stage_render(args):
    t_start = time.time()
    held = gpu_setup(args)
    if not (args.scenes and args.betas and args.table and args.legs):
        raise SystemExit("render needs --scenes, --betas, --table and --legs")
    vocab = ES.load_vocab(args.vocab)
    settings = ES.load_fit_settings(args.betas)
    ES.check_fit_settings(settings, args, args.betas)
    betas = ES.load_betas(args.betas)
    sc = RQ.pick_scene(args.scenes, args.scene)
    name = sc["name"]
    if args.select:
        want = rule_frame(args.select, name)
        if want != args.frame:
            raise SystemExit(f"{name}: --frame {args.frame} but the rule picked {want}")
    table = table_body(args.table)
    if name not in table["scenes"]:
        raise SystemExit(f"{args.table} has no scene {name}")
    tsc = table["scenes"][name]
    shown, tab_rows = col_rows(args.legs)
    map_rows = ["ours_P2", "normlift", "f3dgs", args.legs]
    miss = [r for r in map_rows + [ES.ROW_2D, "averaging"] if r not in tsc["rows"]]
    if miss:
        raise SystemExit(f"{args.table} scene {name} has no rows {miss}")
    tok_file = args.tokens or tok_path(args.out_dir, name, args.frame)
    tok = torch.load(tok_file, map_location="cpu", weights_only=False)
    if (tok["scene"], tok["frame"]) != (name, args.frame):
        raise SystemExit(f"{tok_file}: tokens of {tok['scene']} frame {tok['frame']}")
    # the shown rows only, frozen on the device (eval_seg.load_scene); no P3, no 2D row
    args.rows, args.no_p3, args.with_2d = ",".join(map_rows), True, False
    t0 = time.time()
    S = ES.load_scene(sc, vocab, args, args.device)
    ES.check_field_identity(S["field_identity"], S["aux_identities"], settings, name, args)
    t_load = time.time() - t0
    rows = {r.name: r for r in S["rows"]}
    dev = S["text"].device
    row2d = TensorRow(ES.ROW_2D, "feature", buf=torch.zeros(1, S["field"].d, device=dev))
    used = ES.betas_used([rows[r] for r in map_rows] + [row2d], betas, ES._raw_probs(args))
    i = RQ.check_frame(S, args.frame)
    fr = S["frames"][i]
    if int(fr["idx"]) != tok["index"] or (fr["h"], fr["w"]) != (tok["h"], tok["w"]):
        raise SystemExit(f"{tok_file}: frame index/grid differ from the scene's frame {i}")
    own = {r: own_channel(r) for r in map_rows if own_channel(r)}
    # (a)-(c): every evaluation frame with the real labels
    t0 = time.time()
    acc, pooled, per_frame = scene_pass(S, [rows[r] for r in map_rows], betas, args, own)
    t_pass = time.time() - t0
    idx = [int(f["idx"]) for f in S["frames"]]
    t0 = time.time()
    metrics = rescore(acc, map_rows, own, args.ause_bins)
    del acc
    t_metrics = time.time() - t0
    full_run = args.max_views is None
    rep = repro_check(metrics, tsc, map_rows, own, args.repro_tol, int(tsc["n_views"]),
                      len(S["frames"])) if full_run else dict(status="SKIPPED (--max-views)")
    conf = confident_counts(per_frame, idx)
    del per_frame
    # the chosen frame at every pixel
    h, w = fr["h"], fr["w"]
    maps = {}
    for k, r in enumerate(map_rows):
        maps[r] = RQ.frame_maps(S, i, rows[r], betas, args, with_rgb=(k == 0))
    rgb = maps["ours_P2"].pop("rgb")
    z = ES.frame_features_2d(S, dict(tok2d=tok["tok"], h=h, w=w), args)
    o2d = evaluate_view([row2d], ES.PixelSource(z), S["text"], torch.zeros(h * w, dtype=torch.long),
                        betas, args.tau_cov, undef_frac=args.undef_frac,
                        raw_probs=ES._raw_probs(args))[ES.ROW_2D]
    maps[ES.ROW_2D] = maps_from_out(o2d, fr["gt"], h, w)
    ranks = {r: torch.from_numpy(mid_ranks(np.where(maps[r]["covered"].numpy(),
                                                    maps[r]["u"][ch].double().numpy(), np.nan),
                                           pooled[r]).astype(np.float32))
             for r, ch in own.items()}
    palette = RQ.scene_palette(S)
    names = list(S["class_names"])
    gt = maps["ours_P2"]["gt"]
    slim = {r: {k: m[k] for k in ("pred", "covered", "labelled", "correct")}
            for r, m in maps.items()}
    os.makedirs(args.out_dir, exist_ok=True)
    out_pt = block_path(args.out_dir, name, i)
    meta = dict(
        scene=name, dataset_label=args.dataset_label, frame=i, index=int(fr["idx"]), h=h, w=w,
        n_frames=len(S["frames"]), rows=map_rows, legs=args.legs, own_channels=own,
        columns=dict(zip(COLS[1:], [shown[c] for c in COLS[1:]])),
        table_rows=dict(zip(COLS[1:], [tab_rows[c] for c in COLS[1:]])),
        pool_sizes={r: int(p.size) for r, p in pooled.items()},
        betas_used={k: v for k, v in used.items()}, betas=args.betas,
        betas_sha256=sha256(args.betas), table=args.table, table_sha256=sha256(args.table),
        tokens=tok_file, tokens_image_sha256=tok.get("image_sha256"),
        field_identity=S["field_identity"], class_names=names,
        rows_queried=[r.name for r in S["rows"]], repro=rep, confident=conf,
        text_cache=args.text_cache,
        text_cache_sha256=sha256(args.text_cache) if args.text_cache and os.path.exists(
            args.text_cache) else None,
        git_head=git_head(), script_sha256=sha256(os.path.abspath(__file__)),
        timing=dict(load_s=t_load, pass_s=t_pass, metrics_s=t_metrics,
                    wall_s=time.time() - t_start),
        gpu_reserved_mib=held, **gpu_peak(args))
    torch.save(dict(meta=meta, rgb=rgb, gt=gt, palette=palette, maps=slim, ranks=ranks), out_pt)
    with open(os.path.splitext(out_pt)[0] + ".json", "w") as f:
        json.dump(ES._plain_json(meta), f, indent=1)
    print(f"{name} frame {i}: repro {rep['status']}"
          + (f" (max |delta| {rep['max_abs_delta']:.5f})" if "max_abs_delta" in rep else "")
          + f"; pools {meta['pool_sizes']}; wrote {out_pt}", flush=True)
    if full_run and rep["status"] != "PASS":
        print(json.dumps(ES._plain_json(rep), indent=1))
        raise SystemExit(2)
    return dict(stage="render", meta=meta, maps=maps, ranks=ranks, pooled=pooled, rgb=rgb,
                palette=palette, outputs=[out_pt])


# ------------------------------------------------------------------------------- figure
def published(block):
    """{column: mIoU}, {column: own-channel AUSE} of the block's scene, read from its table
    (whose sha256 must equal the one recorded at render time)."""
    m = block["meta"]
    table = moved(m["table"])                    # blocks rendered before the 2026-10-31 move
    if sha256(table) != m["table_sha256"]:
        raise SystemExit(f"{m['table']} changed since the block was rendered")
    sc = table_body(table)["scenes"][m["scene"]]["rows"]
    miou = {c: sc[r]["all"]["miou"] for c, r in m["table_rows"].items()}
    ause = {}
    for c, r in m["columns"].items():
        ch = m["own_channels"].get(r)
        if ch:
            ause[c] = sc[r]["all"]["ranking"][ch]["ause"]
    if sc["averaging"]["all"]["miou"] != sc["normlift"]["all"]["miou"]:
        raise SystemExit(f"{m['scene']}: averaging and normlift mIoU differ in the table")
    return miou, ause


def block_panels(block, legend_max, legend_min_share):
    """The 3 x 6 panel specs of one block and its class legend."""
    m = block["meta"]
    pal = block["palette"]
    maps = block["maps"]
    miou, ause = published(block)
    gt = block["gt"]
    lab_img = lambda r: RQ.label_image(maps[r]["pred"], pal)
    r1 = [dict(kind="image", img=block["rgb"].numpy(), note="3DGS render")]
    r2 = [dict(kind="image", img=RQ.label_image(gt, pal), note="ground truth")]
    r3 = [dict(kind="key")]
    for c in COLS[1:]:
        r = m["columns"][c]
        mm = dict(maps[r], gt=gt)
        r1.append(dict(kind="image", img=lab_img(r), note=f"mIoU {100 * miou[c]:.1f}"))
        r2.append(dict(kind="image", img=error_rgb(mm)))
        if c in ause:
            rk = block["ranks"][r].numpy()
            r3.append(dict(kind="image", img=RQ.u_image(rk, maps[r]["covered"].numpy(), 1.0),
                           note=f"{OWN_NOTE[c]} AUSE {100 * ause[c]:.1f}"))
        else:
            r3.append(dict(kind="none"))
    shown = [dict(gt=gt, pred=maps[m["columns"][c]]["pred"]) for c in COLS[1:]]
    legend = RQ.legend_classes(m["class_names"], shown, pal, legend_max, legend_min_share)
    # classes on the frame (GT or a prediction) left out of the legend, for the record
    K = len(m["class_names"])
    cnt = torch.zeros(K, dtype=torch.long)
    for t in [gt] + [s["pred"] for s in shown]:              # as legend_classes counts
        t = t.reshape(-1)
        cnt += torch.bincount(t[t >= 0], minlength=K)[:K]
    names = {n for n, _ in legend}
    left_out = {m["class_names"][k]: int(cnt[k]) for k in range(K)
                if cnt[k] > 0 and m["class_names"][k] not in names}
    return [r1, r2, r3], legend, dict(miou=miou, ause=ause, legend_left_out=left_out)


def block_label(m):
    return f"{m['dataset_label']} {m['scene']}, frame {m['frame']}".strip()


def _note(ax, text):
    ax.text(0.025, 0.035, text, transform=ax.transAxes, fontsize=FS_NOTE, color="white",
            ha="left", va="bottom",
            bbox=dict(boxstyle="square,pad=0.12", facecolor="black", edgecolor="none",
                      alpha=0.62))


def _rank_key(fig, box, x, y, pw, ph):
    """Row 3, slot 0: what the rank panels show."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    ax = box(x, y, pw, ph)
    ax.axis("off")
    ax.text(0.5, 0.97, "own uncertainty:\nrank among the\nscene's pixels", transform=ax.transAxes,
            ha="center", va="top", fontsize=RQ.FS_SMALL, linespacing=1.05)
    cax = box(x + 0.10 * pw, y + 0.40 * ph, 0.80 * pw, 0.045)
    cb = fig.colorbar(ScalarMappable(Normalize(0.0, 1.0), RQ.U_CMAP), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([0.0, 1.0])
    cb.set_ticklabels(["0 low", "1 high"])
    cb.outline.set_linewidth(0.3)
    RQ._legend(box(x + 0.10 * pw, y, 0.90 * pw, 0.26 * ph),
               [("undefined", RQ.UNDEF), ("not covered", BLACK)], ncol=1)


def _none_slot(ax):
    from matplotlib.patches import Rectangle
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=NONE_BG, edgecolor="none"))
    ax.text(0.5, 0.5, "no own\nuncertainty", ha="center", va="center", fontsize=RQ.FS_SMALL,
            color=(0.35, 0.35, 0.35))
    for s in ax.spines.values():
        s.set_linewidth(0.3)
        s.set_color((0.6, 0.6, 0.6))


def compose_blocks(blocks, legend_max=24, legend_min_share=0.002):
    """One figure, RQ.FIG_W wide: column titles, then per block 3 rows of 6 panels (block label
    and row labels on the left) and a strip with the class and error legends. Uses render_qual's
    fonts (RC), sizes and legend helper. Returns (fig, per-block info)."""
    import matplotlib
    from matplotlib.figure import Figure
    specs = [block_panels(b, legend_max, legend_min_share) for b in blocks]
    pw = (RQ.FIG_W - LEFT - 0.01 - 5 * RQ.GAP) / 6
    geo = []
    for b, (_, legend, _) in zip(blocks, specs):
        ph = pw * b["meta"]["h"] / b["meta"]["w"]
        n_leg = max(1, math.ceil(len(legend) / LEGEND_NCOL))
        strip = 0.06 + max(n_leg, 2) * 0.1           # the error legend has 2 rows
        geo.append((ph, strip))
    H = RQ.TITLE_H + sum(3 * ph + 2 * RQ.ROW_GAP + st for ph, st in geo) \
        + (len(blocks) - 1) * BLOCK_GAP
    info = []
    with matplotlib.rc_context(RQ.RC):
        fig = Figure(figsize=(RQ.FIG_W, H))
        box = lambda x, y, bw, bh: fig.add_axes((x / RQ.FIG_W, y / H, bw / RQ.FIG_W, bh / H))
        col_x = lambda c: LEFT + c * (pw + RQ.GAP)
        top = H - RQ.TITLE_H
        for bi, (b, (panels, legend, extra), (ph, strip)) in enumerate(zip(blocks, specs, geo)):
            for r in range(3):
                y = top - (r + 1) * ph - r * RQ.ROW_GAP
                for c in range(6):
                    p = panels[r][c]
                    if p["kind"] == "key":
                        _rank_key(fig, box, col_x(c), y, pw, ph)
                        continue
                    ax = box(col_x(c), y, pw, ph)
                    if p["kind"] == "none":
                        _none_slot(ax)
                    else:
                        ax.imshow(p["img"], interpolation="nearest")
                        ax.set_xticks([])
                        ax.set_yticks([])
                        for s in ax.spines.values():
                            s.set_linewidth(0.3)
                        if p.get("note"):
                            _note(ax, p["note"])
                    if r == 0 and bi == 0:
                        ax.set_title(TITLES[c], fontsize=RQ.FS_TITLE, pad=2.0)
                fig.text((LEFT - 0.075) / RQ.FIG_W, (y + ph / 2) / H, ROW_LABELS[r],
                         rotation=90, ha="center", va="center", fontsize=RQ.FS_SMALL)
            y_bot = top - 3 * ph - 2 * RQ.ROW_GAP
            fig.text(0.075 / RQ.FIG_W, (y_bot + (top - y_bot) / 2) / H, block_label(b["meta"]),
                     rotation=90, ha="center", va="center", fontsize=RQ.FS_TITLE)
            RQ._legend(box(col_x(0), y_bot - strip, 5 * pw + 4 * RQ.GAP, strip - 0.04),
                       legend, ncol=LEGEND_NCOL)
            RQ._legend(box(col_x(5), y_bot - strip, pw, strip - 0.04),
                       [("right", RIGHT), ("wrong", WRONG), ("not scored", BLACK)], ncol=2)
            info.append(dict(scene=b["meta"]["scene"], frame=b["meta"]["frame"],
                             legend=[n for n, _ in legend], **extra))
            top = y_bot - strip - BLOCK_GAP
    return fig, info


def _f(v):
    return f"{100 * v:.1f}"


def _tex(s):
    return s.replace("_", "\\_")


COL_TEX = {"2d_features": "the 2D features", "f3dgs": "Feature 3DGS", "normlift": "averaging",
           "legs": "LEGaussians", "ours_P2": "ours"}
OWN_TEX = {"normlift": "NormLift's $1-C_{\\mathrm{inter}}$", "legs": "the $u$ of LEGaussians",
           "ours_P2": "$u^{\\mathrm{ale}}$"}


def _r(v):
    """The value as printed (%, one decimal): comparisons in the caption use it."""
    return round(100 * v, 1)


def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def disclosures(blocks, info):
    """Sentences generated from the published numbers, compared as printed (one decimal): per
    block, the columns whose mIoU is above or equal to ours, and the own uncertainties whose
    AUSE is below or equal to that of u_ale. "slightly" marks a gap below 0.5 points."""
    out = []
    ale = "$u^{\\mathrm{ale}}$"
    for b, inf in zip(blocks, info):
        s = _tex(b["meta"]["scene"])
        miou, ause = inf["miou"], inf["ause"]
        o = _r(miou["ours_P2"])
        above = sorted([c for c in COLS[1:-1] if _r(miou[c]) > o], key=lambda c: -miou[c])
        tie = [c for c in COLS[1:-1] if _r(miou[c]) == o]
        if above:
            txt = (f"On {s}, ours ({o:.1f} mIoU) is below "
                   + _join([f"{COL_TEX[c]} ({_r(miou[c]):.1f})" for c in above]))
            if tie:
                txt += " and ties with " + _join([COL_TEX[c] for c in tie])
            out.append(txt + ".")
        elif tie:
            out.append(f"On {s}, ours ties with {_join([COL_TEX[c] for c in tie])} "
                       f"({o:.1f} mIoU) and is above the others.")
        else:
            out.append(f"On {s}, ours has the highest mIoU ({o:.1f}).")
        a = _r(ause["ours_P2"])
        others = [c for c in COLS[1:-1] if c in ause]
        better = sorted([c for c in others if _r(ause[c]) < a], key=lambda c: ause[c])
        tie = [c for c in others if _r(ause[c]) == a]
        if better:
            slight = all(abs(ause[c] - ause["ours_P2"]) < 0.005 for c in better)
            verb = "ranks" if len(better) == 1 else "rank"
            out.append(f"On {s}, " + _join([OWN_TEX[c] for c in better])
                       + f" {verb} the errors {'slightly ' if slight else ''}better than {ale} "
                       f"(AUSE " + ", ".join(f"{_r(ause[c]):.1f}" for c in better)
                       + f" vs {a:.1f}).")
        elif tie:
            out.append(f"On {s}, {ale} and " + _join([OWN_TEX[c] for c in tie])
                       + f" rank the errors on par (AUSE {a:.1f}).")
        else:
            out.append(f"On {s}, {ale} has the lowest AUSE of the own uncertainties ({a:.1f} "
                       f"vs " + ", ".join(f"{_r(ause[c]):.1f}" for c in others) + ").")
    return out


def caption(blocks, info, supp=False):
    parts = []
    for b in blocks:
        m = b["meta"]
        parts.append(f"{m['dataset_label']} {_tex(m['scene'])} (frame {m['frame']})")
    where = ", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]
    lam = sorted({(b["meta"]["dataset_label"], b["meta"]["legs"]) for b in blocks})
    lam_txt = ", ".join(f"{l[len('legs_l'):]} ({d})" for d, l in lam)
    # main figure (user, 2026-11-02: "choose our best case"): until then the lower-median scene
    sel = ("Each scene is our best test scene of its dataset, a favorable case: the largest "
           "per-scene mIoU margin of ours P2 over the best other row of "
           "Table~\\ref{tab:replica}. Each frame is the evaluation frame with the most "
           "ground-truth classes that cover at least 0.5\\% of its labeled pixels, a rule on "
           "the labels fixed before rendering.")
    if supp:
        sel = ("Every test scene, at the evaluation frame with the most ground-truth classes "
               "that cover at least 0.5\\% of its labeled pixels (a rule on the labels only, "
               "fixed before rendering).")
    body = (f"Segmentation and own uncertainty of every method on {where}. {sel} "
            "Row 1: the 3DGS render and the labels (black: no decision); row 2: the ground "
            "truth and the errors (vermillion: wrong; black: not scored); row 3: each method's "
            "own uncertainty, NormLift's $1-C_{\\mathrm{inter}}$, the learned $u$ of "
            "LEGaussians and our $u^{\\mathrm{ale}}$, as its rank among the covered labeled "
            "pixels with a defined value in all evaluation frames of the scene (grey: "
            "undefined; black: not covered). The 2D features and Feature 3DGS have no own "
            "uncertainty (-- in Table~\\ref{tab:replica}); we show no P2 confidence. Numbers "
            "are per-scene values over all evaluation frames, read from "
            "Table~\\ref{tab:replica}: mIoU (row 1) and the AUSE of the own uncertainty "
            f"(row 3), in \\%. LEGaussians $\\lambda_u$ = {lam_txt}, chosen on validation.")
    if not supp:
        body += " " + " ".join(disclosures(blocks, info))
    return body


def save_pages(figs, out, dpi):
    import matplotlib
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with matplotlib.rc_context(RQ.RC):
        with PdfPages(out, metadata={"Creator": None, "Producer": None}) as pdf:
            for fig in figs:
                fig.savefig(pdf, format="pdf", dpi=dpi)
    return out


def stage_figure(args):
    paths = [p.strip() for p in args.blocks.split(",") if p.strip()]
    blocks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    for p, b in zip(paths, blocks):
        rep = b["meta"]["repro"]
        if rep.get("status") != "PASS" and not args.allow_repro_fail:
            raise SystemExit(f"{p}: repro check {rep.get('status')}; refusing to draw it")
    per_page = args.per_page or len(blocks)
    figs, info = [], []
    for k in range(0, len(blocks), per_page):
        fig, inf = compose_blocks(blocks[k:k + per_page], args.legend_max, args.legend_min_share)
        figs.append(fig)
        info += inf
    stem = os.path.splitext(args.out)[0]
    if len(figs) == 1:
        outputs = RQ.save(figs[0], args.out, args.dpi, args.png_dpi)
    else:
        outputs = [save_pages(figs, stem + ".pdf", args.dpi)]
        import matplotlib
        with matplotlib.rc_context(RQ.RC):
            for j, fig in enumerate(figs):
                png = os.path.join(args.png_dir or os.path.dirname(os.path.abspath(stem)),
                                   f"{os.path.basename(stem)}_p{j + 1}.png")
                os.makedirs(os.path.dirname(png), exist_ok=True)
                fig.savefig(png, dpi=args.png_dpi)
                outputs.append(png)
    cap = caption(blocks, info, supp=len(figs) > 1)
    with open(stem + ".txt", "w") as f:
        f.write(cap + "\n")
    outputs.append(stem + ".txt")
    meta = dict(blocks=[os.path.basename(p) for p in paths],
                block_sha256=[sha256(p) for p in paths], info=info, caption=cap,
                outputs=[os.path.basename(o) for o in outputs],
                script_sha256=sha256(os.path.abspath(__file__)), git_head=git_head())
    meta_path = args.meta or os.path.join(os.path.dirname(os.path.abspath(paths[0])),
                                          os.path.basename(stem) + "_figure.json")
    with open(meta_path, "w") as f:
        json.dump(ES._plain_json(meta), f, indent=1)
    print(f"wrote {' '.join(outputs)} {meta_path}")
    return dict(stage="figure", meta=meta, outputs=outputs, figs=figs)


# ---------------------------------------------------------------------------- confident
def stage_confident(args):
    """Figure F's static confident-error frame from the render metas (config order)."""
    paths = [p.strip() for p in args.metas.split(",") if p.strip()]
    best, scenes = None, []
    for order, p in enumerate(paths):
        m = json.load(open(p))
        c = m.get("confident")
        if not c or m.get("repro", {}).get("status") != "PASS":
            raise SystemExit(f"{p}: no confident-error counts or repro not PASS")
        top = min(c["frames"], key=lambda t: (-t["confident_errors"], t["frame"]))
        scenes.append(dict(scene=m["scene"], median_u_ale=c["median_u_ale"],
                           median_u_epi=c["median_u_epi"], best=top,
                           top5=sorted(c["frames"], key=lambda t: (-t["confident_errors"],
                                                                   t["frame"]))[:5]))
        key = (-top["confident_errors"], order, top["frame"])
        if best is None or key < best[0]:
            best = (key, dict(scene=m["scene"], **top))
    out = dict(rule="over the test scenes, the evaluation frame with the most covered labelled "
                    "pixels that ours P2 gets wrong while u_ale is defined and <= the scene's "
                    "median u_ale and u_epi <= the scene's median u_epi (medians over covered "
                    "labelled pixels with a defined value); ties: config order, then the "
                    "earliest frame", pick=best[1], scenes=scenes)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"confident-error frame: {best[1]['scene']} frame {best[1]['frame']} "
          f"(index {best[1]['index']}, {best[1]['confident_errors']} px); wrote {args.out}")
    return out


# --------------------------------------------------------------------------------- main
def build_parser():
    ap = ES.build_parser()
    ap.description = __doc__.split("\n\n")[0]
    for a in ap._actions:                     # --out is not used by every stage
        if a.dest == "out":
            a.required = False
            a.default = ""
    ap.add_argument("--stage", choices=["tokens", "render", "figure", "confident"],
                    required=True)
    ap.add_argument("--target", default="", help="tokens: scene:frame[,scene:frame...]")
    ap.add_argument("--scene", default="", help="render: scene name in --scenes")
    ap.add_argument("--frame", type=int, default=0, help="render: evaluation frame (position)")
    ap.add_argument("--select", default="",
                    help="render: pick_frames.py static output; --frame must equal its pick")
    ap.add_argument("--legs", default="", help="render: the LEGaussians row of Table I")
    ap.add_argument("--table", default="", help="render: the published eval_seg JSON")
    ap.add_argument("--tokens", default="", help="render: token file (default in --out-dir)")
    ap.add_argument("--dataset-label", default="", help="block label prefix, e.g. Replica")
    ap.add_argument("--repro-tol", type=float, default=REPRO_TOL)
    ap.add_argument("--out-dir", default="", help="tokens / render: output directory")
    ap.add_argument("--blocks", default="", help="figure: block files, comma-separated")
    ap.add_argument("--per-page", type=int, default=0,
                    help="figure: blocks per page (multi-page PDF); 0 = all on one page")
    ap.add_argument("--png-dir", default="", help="figure: where the per-page PNGs go")
    ap.add_argument("--allow-repro-fail", action="store_true")
    ap.add_argument("--legend-max", type=int, default=24)
    ap.add_argument("--legend-min-share", type=float, default=0.002)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--png-dpi", type=int, default=300)
    ap.add_argument("--meta", default="", help="figure: metadata JSON path")
    ap.add_argument("--metas", default="", help="confident: render metas in config order")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    t0 = time.time()
    fn = dict(tokens=stage_tokens, render=stage_render, figure=stage_figure,
              confident=stage_confident)[args.stage]
    if args.stage in ("tokens", "render") and not args.out_dir:
        raise SystemExit(f"--stage {args.stage} needs --out-dir")
    if args.stage in ("figure", "confident") and not args.out:
        raise SystemExit(f"--stage {args.stage} needs --out")
    out = fn(args)
    if isinstance(out, dict):
        out["wall_s"] = time.time() - t0
        if args.stage == "tokens":
            print(json.dumps(ES._plain_json({k: v for k, v in out.items()})))
    return out


if __name__ == "__main__":
    main()
