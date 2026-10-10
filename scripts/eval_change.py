"""
eval_change.py — labelled change evaluation of fields saved after an update pass
(scripts/run_mapping.py --update-data): Table II (tab:rio10) and the Replica semantic swap.

Subcommands; each writes a usplat.runlog run record (as scripts/eval_seg.py; --plain-json for
plain JSON) whose metrics hold the results:
  eval        score every row of ONE field checkpoint against ONE ground-truth file (CPU).
              Rows (usplat.change_eval.checkpoint_rows): ours_c_max (the primary score: max
              over the update of the k-persistent c), ours_c_last, ours_fired (reset count;
              the label-free operating point of every ours row), u_ale (dispersion 1 - rho),
              1-rbar, the ChangeBaselines residuals saved in the checkpoint (ncc,
              feature_residual, depth_residual, depth_render_residual, depth_oracle), css_var,
              yager_conflict, normlift_drop, fused (rank average of the config's "fused"
              rows) and oracle_gt (the GT as a score: a sanity row). A row the checkpoint
              cannot provide is kept as "missing: <reason>". With targets in the GT (swap),
              the segmentation block scores ours (mu*), averaging and every --compare-field
              (e.g. ours_noreset) on changed and static Gaussians. --dump-scores saves the
              per-Gaussian scores for `thresholds`.
  gt-swap     render the swap ground truth (usplat.swap.SwapGT) over the frames the transform
              logged, with the run's 3DGS map and downscale (GPU with --backend gsplat).
  gt-rio10    build the RIO10 stale ground truth (usplat.change_eval) from labels.ply of
              seqXX_01 / seqXX_02 (or the 3RScan lists), visibility from the rescan's rendered
              depth, optionally an update-pass mass floor (render; GPU with gsplat).
  thresholds  one threshold per baseline row at a Gaussian-level false-alarm rate on the
              static VALIDATION Gaussians (from eval --dump-scores files).
  criterion   the pre-registered criterion of a config over the per-scene eval records of one
              split (the records must use the same frozen config).

    python3 scripts/eval_change.py eval --field F.pt --gt gt.pt --config configs/swap_replica.json \
        --scene room_0 --split test --compare-field ours_noreset=F_noreset.pt --out eval.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from usplat import change_eval as CE                                       # noqa: E402
from usplat.vmf_field import DirectionalSemanticField                      # noqa: E402
from usplat.feature_pca import StreamingPCA                                # noqa: E402


# --------------------------------------------------------------------------- helpers
def _load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _calibrated(field):
    return field.thr_by_hist is not None or field.cfg.change_thr != float("inf")


def _read_thresholds(path):
    if not path:
        return {}
    d = json.load(open(path))
    if isinstance(d.get("metrics"), dict):
        if d.get("status", "ok") != "ok":
            raise ValueError(f"{path}: status {d.get('status')!r}, not a finished thresholds run")
        d = d["metrics"]
    d = d.get("thresholds", d)
    return {k: float(v) for k, v in d.items()}


def _averaging_feature(ckpt):
    if not ckpt.get("baselines"):
        return None
    from usplat.baselines import load_baselines, Averaging
    for b in load_baselines(ckpt["baselines"]):
        if isinstance(b, Averaging):               # NormLift is an Averaging subclass
            return b.feature()
    return None


def _seg_block(gt, ckpt, field, compare):
    """Gaussian-level accuracy on changed / static Gaussians (change_eval.region_accuracy)."""
    if gt.get("target") is None or gt.get("text_raw") is None:
        return dict(status="no targets in the ground truth")
    out = dict(status="ok", classes=gt["classes"], rows={})

    def text_for(c):
        return StreamingPCA.from_state(c["pca"]).project_text(gt["text_raw"])
    feats = {"ours": (field.mu_post(), text_for(ckpt))}
    avg = _averaging_feature(ckpt)
    if avg is not None:
        feats["averaging"] = (avg, text_for(ckpt))
    for name, (c, f) in compare.items():
        feats[name] = (f.mu_post(), text_for(c))
    for name, (f, t) in feats.items():
        pred = CE.classify(f, t)
        out["rows"][name] = CE.region_accuracy(pred, gt["target"], gt["label"], gt.get("mass"))
    if avg is None:
        out["rows"]["averaging"] = dict(status="missing: no averaging baseline in the checkpoint")
    return out


def _row_settings(cfg):
    r = cfg.get("rows") or {}
    e = cfg.get("eval") or {}
    return (r.get("residual_filter", "kmin"), tuple(r.get("fused", ("ours_c_max", "depth_residual"))),
            tuple(r.get("common_rows", e.get("common_rows", ()))))


# --------------------------------------------------------------------------- eval
def run_eval(args, rec=None):
    cfg, cfg_sha = CE.load_config(args.config)
    gt = CE.load_gt(args.gt)
    ckpt = _load_ckpt(args.field)
    field = DirectionalSemanticField.from_state(ckpt["field"], device="cpu")
    if field.N != gt["N"]:
        raise ValueError(f"field has {field.N} Gaussians, the ground truth {gt['N']}")
    if not _calibrated(field) and not args.allow_uncalibrated:
        raise ValueError(f"{args.field}: the change threshold was never calibrated, i.e. the "
                         f"field saw no update pass (--allow-uncalibrated to score it anyway)")
    if gt.get("scene") is not None and args.scene and gt["scene"] != args.scene:
        raise ValueError(f"ground truth of scene {gt['scene']!r}, --scene {args.scene!r}")
    compare = {}
    for spec in args.compare_field or []:
        name, path = spec.split("=", 1)
        c = _load_ckpt(path)
        f = DirectionalSemanticField.from_state(c["field"], device="cpu")
        if f.N != field.N:
            raise ValueError(f"--compare-field {name}: {f.N} Gaussians, the field {field.N}")
        compare[name] = (c, f)
        if rec is not None:
            rec.add_input(path)
    filt, fused, common = _row_settings(cfg)
    if args.filter:
        filt = args.filter
    rows = CE.checkpoint_rows(ckpt, field, filt=filt, cb_key=args.cb_key)
    CE.add_fused_row(rows, gt["label"], fused)
    CE.add_oracle_gt_row(rows, gt["label"])
    thr = _read_thresholds(args.thresholds)
    res = CE.evaluate_rows(rows, gt, thresholds=thr, common_rows=common)
    lab = gt["label"].long()
    judged = lab >= 0
    reset = CE.operating_point(field.n_resets >= 1, lab, gt.get("mass"))
    c_def = torch.isfinite(field.c_last)
    reset.update(n_resets_total=int(field.n_resets.sum()),
                 c_coverage_judged=float(c_def[judged].double().mean()) if bool(judged.any())
                 else float("nan"),
                 c_coverage_all=float(c_def.double().mean()),
                 threshold=field.cfg.change_thr,
                 thr_by_hist=None if field.thr_by_hist is None else field.thr_by_hist.tolist())
    seg = _seg_block(gt, ckpt, field, compare)
    cfg_rec = (ckpt.get("config") or {})
    out = dict(scene=args.scene or gt.get("scene"), split=args.split, kind=gt.get("kind"),
               config_path=os.path.abspath(args.config), config_sha256=cfg_sha,
               gt=dict(path=os.path.abspath(args.gt), kind=gt.get("kind"), shares=gt["shares"],
                       config_sha256=gt.get("config_sha256"), n=gt["N"],
                       n_judged=int(judged.sum()), n_stale=int((lab == 1).sum()),
                       n_static=int((lab == 0).sum())),
               field=dict(path=os.path.abspath(args.field),
                          mapping_code=(ckpt.get("code") or {}).get("mapping_code_sha256"),
                          field_config=ckpt.get("field_config"),
                          feature_transform=cfg_rec.get("feature_transform"),
                          feature_transform_arg=cfg_rec.get("feature_transform_arg")),
               compare_fields={k: os.path.abspath(p.split("=", 1)[1])
                               for k, p in zip(compare, args.compare_field or [])},
               residual_filter=filt, thresholds=thr, rows=res, reset=reset, seg=seg)
    if args.dump_scores:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_scores)), exist_ok=True)
        torch.save(dict(scene=out["scene"], split=args.split, label=gt["label"],
                        scores={k: r["score"].float() for k, r in rows.items()
                                if r["status"] == "ok"},
                        label_free=[k for k, r in rows.items() if r["fired"] is not None],
                        oracle=[k for k, r in rows.items() if r["oracle"]]), args.dump_scores)
    for name, r in res.items():
        g = r.get("gaussian")
        msg = (f"AP {g['ap']:.4f} AUROC {g['auroc']:.4f} coverage {g['coverage']:.3f}"
               if g else r["status"])
        print(f"{out['scene']:>10} {name:>22}: {msg}")
    if rec is not None:
        rec.add_input(args.field)
        rec.add_input(args.gt)
        rec.add_input(args.config)
        for k, v in out.items():
            rec.add(k, v)
    return out


# --------------------------------------------------------------------------- gt-swap
def _rasterizer(backend, splats, c2w, K, w, h, channel_chunk=32):
    from eval_seg import make_rasterizer
    fr = dict(viewmat=torch.linalg.inv(c2w.double()).float().to(splats["means"].device),
              K=K.to(splats["means"].device), w=w, h=h)
    return make_rasterizer(backend, splats, fr, argparse.Namespace(channel_chunk=channel_chunk))


def run_gt_swap(args, rec=None):
    from usplat import swap as SW
    from usplat.datasets import make_dataset, downscale
    from usplat.gs_io import load_gaussians
    spec, spec_sha = SW.load_spec(args.spec)
    bank = SW.load_bank(args.bank)
    if bank["spec_sha256"] != spec_sha:
        raise ValueError(f"{args.bank} was built from another spec than {args.spec}")
    if bank["scene"] != args.scene:
        raise ValueError(f"bank scene {bank['scene']!r} != --scene {args.scene!r}")
    log = json.load(open(args.frames_log))
    if log.get("spec_sha256") != spec_sha or log.get("scene") != args.scene:
        raise ValueError(f"{args.frames_log}: logged by another spec/scene "
                         f"({log.get('scene')!r}, {str(log.get('spec_sha256'))[:12]})")
    if [(p["source_id"], p["target_id"]) for p in log["pairs"]] != \
            [(p["source_id"], p["target_id"]) for p in bank["pairs"]]:
        raise ValueError(f"{args.frames_log}: pairs differ from the bank's")
    if int(args.downscale) != int(spec["downscale"]):
        raise ValueError(f"--downscale {args.downscale} != the spec's {spec['downscale']}")
    device = args.device
    if device.startswith("cuda"):
        from usplat import gpu_reserve
        gpu_reserve.apply(device)
    splats = load_gaussians(args.ply, device=device)
    N = splats["means"].shape[0]
    ds = make_dataset(args.dataset, args.data, stride=args.update_stride)
    acc = SW.SwapGT(N, bank["classes"]["ids"], [(p["source_id"], p["target_id"])
                                               for p in bank["pairs"]], device=device)
    frames = log["frames"]
    if not frames:
        raise ValueError(f"{args.frames_log}: no frames")
    for e in frames:
        fr = ds[e["i"]]
        if int(fr.get("index", e["i"])) != int(e["index"]):
            raise ValueError(f"update frame {e['i']}: dataset index {fr.get('index')} != logged "
                             f"{e['index']} (another --update-stride or data?)")
        frd = downscale(fr, args.downscale)
        h, w = frd["image"].shape[:2]
        lab = SW.nearest_labels(fr["labels"], h, w)
        for a, n in e["n_px"].items():
            if int((lab == int(a)).sum()) != int(n):
                raise ValueError(f"update frame {e['i']}: {int((lab == int(a)).sum())} source "
                                 f"pixels of class {a}, the transform logged {n}")
        ras = _rasterizer(args.backend, splats, fr["c2w"], frd["K"], w, h, args.channel_chunk)
        acc.add_view(ras, lab)
    gt = SW.swap_gt(acc, spec, args.scene, bank,
                    extra=dict(frames=[e["i"] for e in frames], ply=os.path.abspath(args.ply),
                               data=os.path.abspath(args.data), downscale=int(args.downscale)))
    _save(gt, args.out)
    print(f"{args.scene}: {acc.n_views} update views | swapped {gt['shares']['n_swapped']} | "
          f"static {gt['shares']['n_static']} | unjudged {gt['N'] - gt['shares']['n_swapped'] - gt['shares']['n_static']}")
    if rec is not None:
        for p in (args.ply, args.bank, args.frames_log, args.spec):
            rec.add_input(p)
        rec.add_output(args.out)
        rec.add("shares", gt["shares"])
        rec.add("pairs", bank["pairs"])
        rec.add("spec_sha256", spec_sha)
    return gt


def _save(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- gt-rio10
def run_gt_rio10(args, rec=None):
    from usplat.datasets_rio import RIO10Scene, load_3rscan_changes
    from usplat.gs_io import load_gaussians
    cfg, cfg_sha = CE.load_config(args.config)
    th = CE.rio10_thresholds(cfg)
    sc = RIO10Scene(args.root or cfg["data_root"], args.scene)
    ref, res = sc.labels(1), sc.labels(2)
    names = {}
    try:
        names = sc.instance_names(1)
    except Exception as e:                              # alignment needs names; reported
        warnings.warn(f"instances.txt of {sc.seq(1)}: {e}")
    static_names = {s.lower() for s in cfg["alignment"]["static_labels"]}
    static_ids = {i for i, n in names.items() if n.lower() in static_names}
    align = None
    if static_ids:
        align = CE.alignment_check(ref["points"], ref["ids"], res["points"], res["ids"], static_ids)
        if cfg["alignment"].get("require_d_move_above_p95", True):
            CE.check_alignment(align["p95_m"], th["d_move"])
    elif cfg["alignment"].get("require_d_move_above_p95", True) and not args.skip_alignment:
        raise ValueError(f"no static-structure instance ({sorted(static_names)}) in "
                         f"{sc.seq(1)}/instances.txt: the alignment check cannot run "
                         f"(--skip-alignment to override)")
    o = cfg["object"]
    if cfg.get("gt_source", "instances") == "3rscan":
        m = cfg.get("rio_to_3rscan") or {}
        if sc.seq(1) not in m or sc.seq(2) not in m:
            raise ValueError(f"gt_source 3rscan needs rio_to_3rscan entries for {sc.seq(1)} "
                             f"and {sc.seq(2)}")
        ch = load_3rscan_changes(args.rscan_meta, m[sc.seq(1)], m[sc.seq(2)])
        motion = CE.object_motion_from_transforms(ch, ref["points"], ref["ids"], th["min_points"],
                                                  th["ignore_ids"])
    else:
        motion = CE.object_motion_from_points(ref["points"], ref["ids"], res["points"], res["ids"],
                                              th["min_points"], th["ignore_ids"],
                                              int(o.get("icp_iters", 50)),
                                              int(o.get("icp_max_points", 4000)),
                                              int(o.get("seed", 0)))
    status = CE.object_status(motion, th["d_move"], th["r_deg"], th["min_points"])
    device = args.device
    splats = load_gaussians(args.ply, device=device)
    means = splats["means"].detach().cpu()
    u = cfg["update"]
    upd = sc.sequence(u["sequence"], stride=u["stride"], depth=cfg["visibility"]["depth"],
                      part=u.get("part", "all"))
    vis = CE.visibility_from_depth(means, (upd[i] for i in range(len(upd))),
                                   th["depth_margin"], th["near"])
    observed, mass = vis["observed"], None
    if args.mass_backend != "none":
        from usplat.datasets import downscale
        from usplat.bki_accumulate import bki_grad_accumulate_groups
        mass = torch.zeros(means.shape[0], dtype=torch.float64)
        f = int(cfg["visibility"].get("downscale", 4))
        for i in range(len(upd)):
            fr = upd[i]
            if not torch.isfinite(fr["c2w"]).all():
                continue
            frd = downscale(fr, f)
            h, w = frd["image"].shape[:2]
            ras = _rasterizer(args.mass_backend, splats, fr["c2w"], frd["K"], w, h,
                              args.channel_chunk)
            with torch.enable_grad():
                (_, m), = bki_grad_accumulate_groups(ras, [(None, torch.ones(h * w, device=device))],
                                                     means.shape[0], device=device)
            mass += m.double().cpu()
        observed = observed & (mass >= th["min_mass"])
    lab = CE.gaussian_stale_labels(means, ref["points"], ref["ids"], status, th["d_near"],
                                   observed=observed, hidden=vis["hidden"])
    olab = {k: v[0] for k, v in status.items() if v[0] in (CE.STALE, CE.STATIC)}
    shares = dict(lab["shares"], n_no_depth=int(vis["no_depth"].sum()), n_frames=vis["n_frames"])
    cmap = json.load(open(args.class_map)) if args.class_map else (cfg.get("class_map") or {})
    target = classes = text_raw = None
    if cmap:
        target, classes, text_raw = _rio10_targets(args, sc, cmap, means, lab, res, th["d_near"])
    gt = CE.make_gt("rio10", sc.scene, lab["label"], lab["object_id"], olab, mass=mass,
                    target=target, classes=classes, text_raw=text_raw, shares=shares, config=cfg,
                    extra=dict(object_status={int(k): list(v) for k, v in status.items()},
                               object_motion={int(k): v for k, v in motion.items()},
                               alignment=align, category=lab["category"],
                               instance_names=names, ply=os.path.abspath(args.ply)))
    _save(gt, args.out)
    n_st = sum(1 for v in status.values() if v[0] == CE.STALE)
    print(f"scene {sc.scene}: {len(status)} objects ({n_st} stale) | Gaussians judged "
          f"{shares['n_judged']} (stale {shares['n_stale']}, static {shares['n_static']}) | "
          f"unobserved {shares['share_unobserved']:.3f} hidden {shares['share_hidden']:.3f}")
    if rec is not None:
        rec.add_input(args.ply)
        rec.add_input(args.config)
        rec.add_output(args.out)
        rec.add("shares", shares)
        rec.add("alignment", align)
        rec.add("config_sha256", cfg_sha)
        rec.add("objects", {str(k): dict(status=v[1], **{kk: vv for kk, vv in motion[k].items()})
                            for k, v in status.items()})
    return gt


def _rio10_targets(args, sc, cmap, means, lab, res, d_near):
    """Segmentation targets through a class map {RIO10 label name: vocabulary id}; the text
    embeddings come from scripts/eval_seg.py's cache / extractor (same keys and templates)."""
    from eval_seg import load_vocab, text_embeddings
    if not args.vocab:
        raise ValueError("a class map needs --vocab (and --text-cache / --extractor)")
    vocab = load_vocab(args.vocab)
    norm = lambda s: str(s).strip().lower()
    cm = {norm(k): int(v) for k, v in cmap.items()}
    bad = sorted(set(cm.values()) - set(vocab["ids"]))
    if bad:
        raise ValueError(f"class map ids {bad} are not in {args.vocab}")
    ids = [c for c in vocab["ids"] if c in set(cm.values())]
    col = {c: k for k, c in enumerate(ids)}
    to_cls = lambda names: {i: col[cm[norm(n)]] for i, n in names.items() if norm(n) in cm}
    target = CE.rescan_targets(means, lab["label"], lab["object_id"], to_cls(sc.instance_names(1)),
                               res["points"], res["ids"], to_cls(sc.instance_names(2)), d_near)
    idx = [vocab["ids"].index(c) for c in ids]
    targs = argparse.Namespace(extractor=args.extractor, clip_model=args.clip_model,
                               clip_pretrained=args.clip_pretrained, text_cache=args.text_cache,
                               clip_device=args.clip_device)
    text_raw = text_embeddings(vocab, idx, targs).float().cpu()
    return target, dict(ids=ids, names=[vocab["names"][i] for i in idx]), text_raw


# --------------------------------------------------------------------------- thresholds
def run_thresholds(args, rec=None):
    cfg, cfg_sha = CE.load_config(args.config)
    target = args.target_fa if args.target_fa is not None else \
        float((cfg.get("operating_point") or {}).get("baseline_target_fa", 1e-3))
    pooled, label_free, oracle, scenes = {}, set(), set(), []
    for p in args.dumps:
        d = torch.load(p, map_location="cpu", weights_only=False)
        if d.get("split") != "val":
            raise ValueError(f"{p}: split {d.get('split')!r}; thresholds come from validation only")
        scenes.append(d.get("scene"))
        st = torch.as_tensor(d["label"]).long() == CE.STATIC
        label_free |= set(d["label_free"])
        oracle |= set(d.get("oracle", []))
        for k, s in d["scores"].items():
            pooled.setdefault(k, []).append(torch.as_tensor(s).double()[st])
    if len(set(scenes)) != len(scenes):
        raise ValueError(f"duplicate scenes in the dumps: {scenes}")
    want = args.rows.split(",") if args.rows else sorted(pooled)
    thr = {k: CE.fa_threshold(torch.cat(pooled[k]), target) for k in want
           if k in pooled and k not in label_free and k not in oracle}
    out = dict(thresholds=thr, target_fa=target, scenes=scenes, config_sha256=cfg_sha,
               level="gaussian")
    print(json.dumps(out, indent=1))
    if rec is not None:
        for p in args.dumps:
            rec.add_input(p)
        for k, v in out.items():
            rec.add(k, v)
    return out


# --------------------------------------------------------------------------- criterion
def run_criterion(args, rec=None):
    cfg, cfg_sha = CE.load_config(args.config)
    crits = cfg.get("criteria") or {cfg["criterion"].get("id", "criterion"): cfg["criterion"]}
    scenes = (cfg.get("scenes") or cfg.get("splits"))[args.split]
    records = {}
    for p in args.records:
        d = json.load(open(p))
        if d.get("status", "ok") != "ok":
            raise ValueError(f"{p}: status {d.get('status')!r}, not a finished evaluation")
        m = d["metrics"] if isinstance(d.get("metrics"), dict) else d
        if m.get("split") != args.split:
            raise ValueError(f"{p}: split {m.get('split')!r} != {args.split!r}")
        if m.get("config_sha256") != cfg_sha:
            raise ValueError(f"{p}: evaluated with another config (sha {str(m.get('config_sha256'))[:12]} "
                             f"!= {cfg_sha[:12]})")
        if m["scene"] in records:
            raise ValueError(f"two records of scene {m['scene']!r}")
        records[m["scene"]] = m
    codes = sorted({str((m.get("field") or {}).get("mapping_code")) for m in records.values()})
    if len(codes) > 1:
        raise ValueError(f"the {args.split} fields were built with {len(codes)} mapping code "
                         f"versions {[c[:12] for c in codes]}: rebuild them with one")
    if args.reference:
        d = json.load(open(args.reference))
        ref = d["metrics"] if isinstance(d.get("metrics"), dict) else d
        if d.get("status", "ok") != "ok" or ref.get("config_sha256") != cfg_sha:
            raise ValueError(f"{args.reference}: not a finished criterion record of this config")
        if ref.get("mapping_codes") != codes:
            raise ValueError(f"{args.split} fields use mapping code {[c[:12] for c in codes]}, "
                             f"the reference ({ref.get('split')}) "
                             f"{[str(c)[:12] for c in ref.get('mapping_codes') or []]}")
    res = {}
    for name, c in crits.items():
        r = CE.check_criterion(records, dict(c, scenes=scenes))
        r["binding"] = c.get("split") == args.split       # a val run of a test criterion reports
        r["statement"] = c.get("statement")
        res[name] = r
        print(f"{name} ({args.split}{'' if r['binding'] else ', not binding'}): "
              f"{r['n_pass']}/{r['n_scenes']} scenes pass, need {r['min_pass']} -> "
              f"{'PASS' if r['passed'] else 'FAIL'}")
    rule = cfg.get("criteria_rule", "all_of")
    vals = [r["passed"] for r in res.values()]
    out = dict(split=args.split, config_sha256=cfg_sha, scenes=scenes, criteria=res, rule=rule,
               passed=any(vals) if rule == "any_of" else all(vals), mapping_codes=codes,
               reference=os.path.abspath(args.reference) if args.reference else None)
    if rec is not None:
        for p in args.records:
            rec.add_input(p)
        for k, v in out.items():
            rec.add(k, v)
    return out


# --------------------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--plain-json", action="store_true",
                       help="write plain JSON instead of a usplat.runlog record")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", required=True)
    e = sub.add_parser("eval", help="score one field checkpoint against one GT file")
    e.add_argument("--field", required=True, help="checkpoint saved after the update pass")
    e.add_argument("--gt", required=True, help="ground truth (gt-swap / gt-rio10)")
    e.add_argument("--config", required=True, help="the frozen config (swap spec or rio10)")
    e.add_argument("--scene", default="")
    e.add_argument("--split", required=True, choices=["val", "test"])
    e.add_argument("--compare-field", action="append", default=[],
                   help="NAME=PATH: another field for the segmentation block (repeatable)")
    e.add_argument("--thresholds", default="", help="validation thresholds (thresholds run)")
    e.add_argument("--filter", default="", help="residual filter (default: the config's)")
    e.add_argument("--cb-key", default="", help="checkpoint key of the ChangeBaselines state")
    e.add_argument("--dump-scores", default="", help=".pt with the per-Gaussian scores")
    e.add_argument("--allow-uncalibrated", action="store_true")
    common(e)
    g = sub.add_parser("gt-swap", help="render the swap ground truth")
    g.add_argument("--ply", required=True)
    g.add_argument("--data", required=True, help="update sequence root (Sequence_2)")
    g.add_argument("--dataset", default="replica_semnerf")
    g.add_argument("--update-stride", type=int, default=5)
    g.add_argument("--frames-log", required=True, help="the transform's log (log=...)")
    g.add_argument("--bank", required=True)
    g.add_argument("--spec", required=True)
    g.add_argument("--scene", required=True)
    g.add_argument("--downscale", type=int, required=True)
    g.add_argument("--backend", choices=["gsplat", "mock"], default="gsplat")
    g.add_argument("--device", default="cuda")
    g.add_argument("--channel-chunk", type=int, default=32)
    g.add_argument("--record", default="", help="run record (default: <out>.json)")
    g.add_argument("--plain-json", action="store_true")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", required=True, help="ground truth .pt")
    r = sub.add_parser("gt-rio10", help="build the RIO10 stale ground truth")
    r.add_argument("--root", default="", help="RIO10 root (default: the config's data_root)")
    r.add_argument("--scene", required=True, help="scene id, e.g. 01")
    r.add_argument("--config", default="configs/rio10_change.json")
    r.add_argument("--ply", required=True, help="3DGS map fit on seqXX_01")
    r.add_argument("--rscan-meta", default="", help="3RScan.json (gt_source 3rscan)")
    r.add_argument("--mass-backend", choices=["none", "gsplat", "mock"], default="none",
                   help="render the update-pass mass for the observed floor")
    r.add_argument("--device", default="cpu")
    r.add_argument("--channel-chunk", type=int, default=32)
    r.add_argument("--skip-alignment", action="store_true")
    r.add_argument("--class-map", default="",
                   help="JSON {RIO10 label: vocabulary id} for the segmentation targets "
                        "(default: the config's class_map; empty = no targets)")
    r.add_argument("--vocab", default="", help="frozen vocabulary (with a class map)")
    r.add_argument("--text-cache", default="", help="scripts/eval_seg.py text cache")
    r.add_argument("--extractor", default="lseg")
    r.add_argument("--clip-model", default="ViT-B-16")
    r.add_argument("--clip-pretrained", default="laion2b_s34b_b88k")
    r.add_argument("--clip-device", default="cpu")
    r.add_argument("--record", default="")
    r.add_argument("--plain-json", action="store_true")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)
    t = sub.add_parser("thresholds", help="baseline thresholds from validation dumps")
    t.add_argument("--config", required=True)
    t.add_argument("--dumps", nargs="+", required=True)
    t.add_argument("--rows", default="")
    t.add_argument("--target-fa", type=float, default=None)
    common(t)
    c = sub.add_parser("criterion", help="pre-registered criterion over per-scene records")
    c.add_argument("--config", required=True)
    c.add_argument("--split", required=True, choices=["val", "test"])
    c.add_argument("--records", nargs="+", required=True)
    c.add_argument("--reference", default="",
                   help="criterion record of another split whose mapping code must match "
                        "(test: the validation record)")
    common(c)
    return ap


RUNNERS = {"eval": run_eval, "gt-swap": run_gt_swap, "gt-rio10": run_gt_rio10,
           "thresholds": run_thresholds, "criterion": run_criterion}


def main(argv=None):
    args = build_parser().parse_args(argv)
    fn = RUNNERS[args.cmd]
    path = getattr(args, "record", "") or (args.out if args.cmd in ("eval", "thresholds",
                                                                    "criterion")
                                           else f"{args.out}.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not args.plain_json:
        from usplat.runlog import RunRecord
        with RunRecord(path, config=vars(args), seed=args.seed, name=f"eval_change_{args.cmd}",
                       argv=[sys.argv[0]] + list(argv if argv is not None else sys.argv[1:])) as rec:
            out = fn(args, rec)
    else:
        torch.manual_seed(args.seed)
        out = fn(args)
        if args.cmd in ("eval", "thresholds", "criterion"):
            from usplat.runlog import to_jsonable
            with open(path, "w") as f:
                json.dump(to_jsonable(dict(out, config=vars(args))), f, indent=1)
    print(f"wrote {path}")
    return out


if __name__ == "__main__":
    main()
