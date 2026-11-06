"""
decode_tab1.py — attempt 08 (configs/decoding_prereg.json): every Table I row of one dataset and
stage with the paper's decision rule ('base', which must reproduce the published record) and the
registered label-free rules of usplat/decoding.py. The saved fields, maps and fitted temperatures
are read; scripts/eval_seg.py's loaders, views and evaluate_view are imported, never edited.
GPU, through scripts/gpu_gate.sh:

    scripts/gpu_gate.sh 7000 python3 scripts/decode_tab1.py --dataset replica --stage val \
        --prereg configs/decoding_prereg.json --out results/decoding/replica_val.json

Per scene. Pass 1 renders every row on every evaluation frame and adds each row's covered pixels to
its label-free statistics (decoding.SceneStats: class-score means, predicted prior, a fixed pixel
subsample); the rules' parameters follow. Pass 2 runs eval_seg.evaluate_view (the paper's rule;
every published metric through a SegAccumulator) and each rule's decisions on the labelled pixels.
The base decisions must equal evaluate_view's labels, and the base metrics must equal the
published record cell by cell (prereg tolerance), or the run stops with exit 2.

Output: one record per rule (<out stem>_<rule>.json; eval_seg's layout, so scripts/make_tables.py
tab1pair / perclasspair read it; NLL and Brier are the base predictive's) and the run record
<out> (rule parameters per scene and row, reproduction, timing).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import eval_seg as ES                                                      # noqa: E402
from usplat import decoding as D                                           # noqa: E402
from usplat.paths import moved                                             # noqa: E402
from usplat.seg_eval import (SegAccumulator, evaluate_view, freeze_rows, render_row,  # noqa: E402
                             class_scores, row_beta, pixel_beta, _coverage)

CODE = ("scripts/decode_tab1.py", "scripts/decode_select.py", "scripts/decode_tab1.sh",
        "usplat/decoding.py")
REPRO_KEYS = ("miou", "acc", "macc", "ece", "nll", "brier", "aurc")
DROP = ("--val-scenes", "--betas", "--scenes", "--out", "--rows", "--device", "--backend")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def git_state():
    run = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()
    return dict(commit=run("rev-parse", "HEAD"),
                dirty=[l for l in run("status", "--porcelain", "--untracked-files=no").splitlines()])


def rule_keys(prereg):
    """[(key, rule, value)] of the registered grid, 'base' first."""
    out = [("base", "base", None)]
    for c in prereg["rules"]:
        for v in c.get("values") or [None]:
            key = c["rule"] if v is None else f"{c['rule']}_{v:g}"
            out.append((key, c["rule"], v))
    return out


def patch_distill():
    """usplat.distill.row_factory finds the distilled files next to the field's own recorded path,
    which names the folders before the 2026-10-31 move; give it the moved path (no file edit)."""
    import usplat.distill as DS
    if getattr(DS.row_factory, "_moved", False):
        return
    orig = DS.row_factory

    def row_factory(ckpt, field=None, device="cpu", class_ids=None, field_path=None):
        main = field_path or moved((ckpt.get("config") or {}).get("out"))
        return orig(ckpt, field, device, class_ids=class_ids, field_path=main)
    row_factory._moved = True
    DS.row_factory = row_factory


def eval_args(rec, betas_path, scenes_path, rows, device, backend):
    argv = list(rec["argv"][1:])
    keep, i = [], 0
    while i < len(argv):
        if argv[i] in DROP:
            i += 2
            continue
        keep.append(argv[i])
        i += 1
    keep += ["--betas", betas_path, "--scenes", scenes_path, "--rows", ",".join(rows),
             "--out", os.devnull, "--device", device, "--backend", backend]
    a = ES.build_parser().parse_args(keep)
    if getattr(a, "text_cache", None):
        a.text_cache = moved(a.text_cache)
    return a


@torch.no_grad()
def row_scores(row, ras, text, betas, raw, tau_cov, undef_frac, idx=None):
    """(s, p, alpha, covered) of one row on one frame, as evaluate_view computes them (on the pixels
    idx, or all)."""
    b0 = row_beta(row, betas, raw)
    rr = render_row(row, ras, 1e-6, undef_frac)
    P = rr.alpha.shape[0]
    sel = torch.arange(P, device=rr.feat.device) if idx is None else idx.to(rr.feat.device)
    s = class_scores(rr, text, 1e-6)[sel].double()
    K = s.shape[1]
    b = pixel_beta(row, b0, {k: v[sel] for k, v in rr.unc.items()})
    cov = _coverage(rr.alpha[sel], tau_cov, 1e-6)
    if torch.is_tensor(b):
        b = torch.where(cov, b, torch.ones_like(b))
    logits = (b[:, None].double() * s) if torch.is_tensor(b) else b * s
    p = torch.softmax(logits, dim=1)
    p = torch.where(cov[:, None], p, torch.full_like(p, 1.0 / K))
    a = rr.alpha[sel].double() if rr.kind == "feature" else torch.ones_like(rr.alpha[sel]).double()
    return s, p, a, cov


def views_of(S, args, rows2d):
    v2 = ES.scene_views_2d(S, args) if rows2d else None
    for v in ES.scene_views(S, args):
        w = next(v2) if v2 is not None else None
        if w is not None and w.name != v.name:
            raise RuntimeError(f"2D view {w.name} != 3D view {v.name}")
        yield v, w


def scene_run(sc, vocab, betas, settings, args, keys, published, tol):
    S = ES.load_scene(sc, vocab, args, args.device)
    ES.check_field_identity(S["field_identity"], S["aux_identities"], settings, sc["name"], args)
    raw = ES._raw_probs(args)
    used = ES.betas_used(S["rows"] + S["rows2d"], betas, raw)
    rows, rows2d = freeze_rows(S["rows"]), freeze_rows(S["rows2d"])
    K = len(S["class_names"])
    t0 = time.perf_counter()
    stats = {r.name: D.SceneStats(K, len(S["frames"]), seed=i) for i, r in enumerate(rows + rows2d)}
    for v, w in views_of(S, args, rows2d):                                  # pass 1
        for group, view in ((rows, v), (rows2d, w)):
            for r in group:
                s, p, a, cov = row_scores(r, view.rasterizer, view.text, betas, raw, args.tau_cov,
                                          args.undef_frac)
                stats[r.name].add(s[cov], p[cov], a[cov])
    fin = {n: st.finalize() for n, st in stats.items()}
    params = {n: {k: D.scene_parameters(fin[n], rule, val) for k, rule, val in keys} for n in fin}
    t1 = time.perf_counter()
    acc = SegAccumulator(class_names=S["class_names"])
    dec = {(n, k): dict(lab=[], conf=[]) for n in fin for k, _, _ in keys if k != "base"}
    n_base_mismatch = 0
    for v, w in views_of(S, args, rows2d):                                  # pass 2
        out = evaluate_view(rows, v.rasterizer, v.text, v.gt, betas, args.tau_cov,
                            undef_frac=args.undef_frac, raw_probs=raw) if rows else {}
        if rows2d:
            out.update(evaluate_view(rows2d, w.rasterizer, w.text, w.gt, betas, args.tau_cov,
                                     undef_frac=args.undef_frac, raw_probs=raw))
        acc.add(out)
        for group, view in ((rows, v), (rows2d, w)):
            for r in group:
                o = out[r.name]
                s, p, a, cov = row_scores(r, view.rasterizer, view.text, betas, raw, args.tau_cov,
                                          args.undef_frac, idx=o["pix"])
                lab0, _, _ = D.decide("base", s, p, cov)
                n_base_mismatch += int((lab0.cpu() != o["pred"].long()).sum())
                for k, rule, _ in keys:
                    if k == "base":
                        continue
                    lab, conf, _ = D.decide(rule, s, p, cov, a, params[r.name][k])
                    dec[(r.name, k)]["lab"].append(lab.cpu().short())
                    dec[(r.name, k)]["conf"].append(conf.cpu())
    base = acc.metrics(args.ece_bins, args.ause_bins)
    t2 = time.perf_counter()
    by_rule = {"base": base}
    for k, _, _ in keys:
        if k == "base":
            continue
        by_rule[k] = {}
        for n in fin:
            arr = acc.arrays(n)
            m = D.decision_metrics(arr["gt"].long(), torch.cat(dec[(n, k)]["lab"]).long(),
                                   torch.cat(dec[(n, k)]["conf"]), arr["covered"].bool(), u=arr["u"],
                                   num_bins=args.ece_bins, ause_bins=args.ause_bins,
                                   class_names=S["class_names"])
            for var in m:                                   # the predictive's NLL and Brier
                if m[var].get("n_pixels"):
                    m[var]["nll"], m[var]["brier"] = base[n][var]["nll"], base[n][var]["brier"]
            by_rule[k][n] = m
    # the base rule against the published record, cell by cell
    pub = published["scenes"][sc["name"]]["rows"]
    deltas = {}
    for n, m in base.items():
        for var in ("all", "covered"):
            for key in REPRO_KEYS:
                deltas[f"{n}/{var}/{key}"] = abs(float(m[var][key]) - float(pub[n][var][key]))
            for c, val in m[var]["iou"].items():
                deltas[f"{n}/{var}/iou/{c}"] = abs(float(val) - float(pub[n][var]["iou"][c]))
    worst = max(deltas.items(), key=lambda kv: kv[1])
    repro = dict(ok=worst[1] <= tol and n_base_mismatch == 0, tolerance=tol, max_delta=worst[1],
                 worst=worst[0], n_cells=len(deltas), base_label_mismatches=n_base_mismatch)
    meta = dict(n_views=acc.n_views, classes=S["class_names"], class_ids=S["class_ids"],
                field_identity=S["field_identity"], field_code=S["field_code"],
                aux_identities=S["aux_identities"], betas_used=used,
                raw_probs=sorted(raw & set(used)), n_unknown_label_pixels=S["n_unknown"],
                n_skipped_frames=S["n_skipped"])
    stat_rec = {n: dict(n_pixels=fin[n]["n_pixels"], n_subsample=fin[n]["n_subsample"],
                        m=fin[n]["m"].tolist(), pi=fin[n]["pi"].tolist(),
                        params={k: (None if params[n][k] is None else params[n][k].tolist()) for k in params[n]})
                for n in fin}
    timing = dict(pass1_s=t1 - t0, pass2_s=t2 - t1, metrics_s=time.perf_counter() - t2)
    return by_rule, meta, stat_rec, repro, timing


def run(a):
    prereg = json.load(open(a.prereg))
    ds = prereg["datasets"][a.dataset]
    stage = "val" if a.stage == "toy" else a.stage
    if a.stage == "test":
        out_test = a.out
        if os.path.exists(out_test) and json.load(open(out_test)).get("status") == "done":
            raise SystemExit(f"{out_test}: this test stage already ran (run once)")
    rec_path = moved(a.record or ds[f"record_{stage}"])
    betas_path = moved(a.betas or ds["record_val"])
    published = ES._record_body(rec_path) if hasattr(ES, "_record_body") else json.load(open(rec_path))["metrics"]
    rec = json.load(open(rec_path))
    scenes_path = a.scenes or ds[f"scenes_{stage}"]
    rows = a.rows.split(",") if a.rows else ds["rows"]
    keys = rule_keys(prereg)
    patch_distill()
    if str(a.device).startswith("cuda"):
        from usplat import gpu_reserve
        gpu_reserve.apply(a.device)
    args = eval_args(rec, betas_path, scenes_path, rows, a.device, a.backend)
    vocab = ES.load_vocab(args.vocab)
    settings = ES.load_fit_settings(args.betas)
    ES.check_fit_settings(settings, args, args.betas)
    betas = ES.load_betas(args.betas)
    head = dict(stage=a.stage, dataset=a.dataset, status="running", start=time.time(),
                argv=sys.argv, prereg=os.path.abspath(a.prereg), prereg_sha256=sha256_file(a.prereg),
                record=rec_path, record_sha256=sha256_file(rec_path), betas=betas_path,
                betas_sha256=sha256_file(betas_path), scenes=scenes_path, rows=rows,
                rules=[k for k, _, _ in keys], git=git_state(),
                code={c: sha256_file(os.path.join(REPO, c)) for c in CODE if os.path.exists(os.path.join(REPO, c))})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(head, open(a.out, "w"), indent=1)
    res = {k: {} for k, _, _ in keys}
    scene_meta, scene_stats, scene_repro, scene_time = {}, {}, {}, {}
    for sc in ES._load_structured(scenes_path):
        by_rule, meta, st, repro, timing = scene_run(sc, vocab, betas, settings, args, keys,
                                                     published, float(prereg["reproduction"]["tolerance"]))
        print(f"{sc['name']}: reproduction {'ok' if repro['ok'] else 'FAILED'} (max {repro['max_delta']:.2e} "
              f"at {repro['worst']}, base label mismatches {repro['base_label_mismatches']}); "
              f"{timing}")
        for k in by_rule:
            res[k][sc["name"]] = dict(meta, rows=by_rule[k])
        scene_meta[sc["name"]], scene_stats[sc["name"]] = meta, st
        scene_repro[sc["name"]], scene_time[sc["name"]] = repro, timing
        for k in res:
            r0 = res[k][sc["name"]]["rows"]
            line = " ".join(f"{n}={100 * r0[n]['all']['miou']:.1f}" for n in rows if n in r0)
            print(f"  {k:>10}: mIoU {line}")
        if not repro["ok"] and a.stage != "toy":
            head.update(status="failed", reproduction=scene_repro, scenes=list(scene_repro))
            json.dump(head, open(a.out, "w"), indent=1)
            print("the base rule did not reproduce the published record: stop")
            return 2
    stem = os.path.splitext(a.out)[0]
    outs = {}
    for k in res:
        body = dict(vocab=published.get("vocab"), betas=betas, betas_path=betas_path,
                    beta_fit_settings=settings, scenes=res[k], summary=ES.summarize_scenes(res[k]),
                    field_identities=sorted({r["field_identity"] for r in res[k].values()}),
                    decoding=dict(rule=k, prereg=os.path.abspath(a.prereg)))
        path = f"{stem}_{k}.json"
        json.dump(dict(status="ok", stage=a.stage, metrics=ES._plain_json(body)), open(path, "w"), indent=1)
        outs[k] = path
    head.update(status="done", end=time.time(), outputs=outs, reproduction=scene_repro,
                stats=scene_stats, timing=scene_time)
    json.dump(ES._plain_json(head), open(a.out, "w"), indent=1)
    print(f"wrote {a.out} and {len(outs)} rule records")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, choices=["replica", "scannet", "toy"])
    ap.add_argument("--stage", required=True, choices=["val", "test", "toy"])
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", default="", help="the published eval_seg record (default: prereg)")
    ap.add_argument("--betas", default="", help="the fitted temperatures (default: the val record)")
    ap.add_argument("--scenes", default="", help="the scenes file (default: prereg)")
    ap.add_argument("--rows", default="", help="comma list (default: prereg)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="gsplat")
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
