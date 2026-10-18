"""
nor_tab2.py — the CPU steps of scripts/nor_tab2.sh: Table II of papers/main.tex (label
tab:change) on NeRF-Object-Removal, protocol D1-D6 of 2026-10-17 (configs/nor_val.json,
configs/nor_test.json).

  python3 scripts/nor_tab2.py scenes configs/nor_val.json
  python3 scripts/nor_tab2.py reuse map|field|eval <file> -- <the arguments that build it>
  python3 scripts/nor_tab2.py frozen <out>/change_val.json --pipeline "<settings>"
  python3 scripts/nor_tab2.py merge --config configs/nor_test.json --dir <out>/test \
      --out <out>/change_test.json --pipeline "<settings>" [--frozen <out>/change_val.json]

scenes  one line per scene of a split config, "name data stride update_stride align ply
        field" (align "none": no alignment), for the shell loop.
reuse   exit 1 when an existing output was built with other options or other code than a
        new run would use: the given arguments are parsed with the producing script's own
        parser (fit_gs.py for a map, run_mapping.py for a field, eval_change_nor.py for an
        evaluation), defaults included, and every option but "out" and "device" must equal
        the saved one (a map's meta["args"], a field's config, a record's config; a record
        must also be finished, and its field and map unchanged since it ran); the code
        identity it records (fit_gs.py CODE_FILES, run_mapping.py MAPPING_CODE,
        eval_change_nor.py CODE) must be the code of now. So a smoke
        map, field or evaluation left at a protocol path, or one from older code, is
        refused, never reused silently (the merge would refuse the mix later anyway).
frozen  exit 1 when the code that builds and scores maps and fields now (fit_gs.py
        CODE_FILES, run_mapping.py MAPPING_CODE, eval_change_nor.py CODE) or the pipeline
        settings differ from those of a finished validation merge: the test split runs
        with the method frozen on validation.
merge   the eval_change_nor.py records <dir>/<scene>.json of every scene of the config ->
        mean and population std over the scenes (usplat.seg_eval.scene_stats: NaN values
        are counted apart, never averaged). Nothing is merged (status "failed", exit 1)
        unless
          every scene  has a finished record of that scene, made from the config's field,
                       map, data and alignment, with the config's strides and keyframe
                       counts (D4: n_map_keyframes, n_update_keyframes), no keyframe subset;
                       its field and map still have the sha256 the evaluation read, and the
                       map was fit on the config's mapping frames
          all scenes   share the rows, the settings (SAME), the eval code, the field mapping
                       code, the map fit code and the FitConfig
          --frozen V   the settings, code and --pipeline settings equal those of the
                       finished validation merge V (the test merge)
        The output is a usplat.runlog record; under "metrics":
          summary[row][metric]  {mean, std, n, n_nan} over the scenes, metric in
              g_ap g_auroc p_ap p_auroc  Gaussian / pixel level, NaN ranked last (Table II)
              *_common                   the same on the set where every row is defined
              fa rec                     the row's operating point (fused: none, n = 0)
              g_unscored p_unscored      share of the population the row never scores
              threshold                  its operating-point threshold
          extra[score][g_ap|g_auroc]  the JSON-only scores (depth_gauss/kmin, */max)
          ours[...]      kept, kept_after_mapping, changed_end_fa / _rec, n_resets
          counts[...]    per-scene counts: Gaussians, unobserved share, stale (D2) and
                         box-only counts, views, pixel population (pooled_unobserved_share:
                         all unobserved / all Gaussians)
          maps[...]      n_gaussians, psnr, ssim of the 3DGS maps
          scenes[name]   every per-scene value above, the record path and sha256
          split, config, rows, inputs, settings, code, pipeline, replay
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

# eval_change_nor.py settings every scene (and the test against validation) must share
SAME = ("change_quantile", "k", "tau_cov", "undef_frac", "inflate_m", "near_m", "kf_trans",
        "kf_rot", "max_keyframes", "max_update_keyframes", "downscale", "extractor",
        "backend", "field_config")
NOT_COMPARED = ("out", "device")          # reuse: the path itself and the device
ROW_METRICS = ("g_ap", "g_auroc", "p_ap", "p_auroc", "fa", "rec", "g_ap_common",
               "g_auroc_common", "p_ap_common", "p_auroc_common", "g_unscored", "p_unscored",
               "threshold")


# ------------------------------------------------------------------------------ helpers
def num(x):
    """float of a record value ("nan" / "inf" strings of usplat.runlog too); None if none."""
    if x is None or isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def share(k, n):
    k, n = num(k), num(n)
    return None if k is None or not n else k / n


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_config(path):
    """A split config {split, rule, root, scenes: [entry]} (configs/nor_val.json)."""
    with open(path) as f:
        cfg = json.load(f)
    names = [e["name"] for e in cfg["scenes"]]
    if not names or len(set(names)) != len(names):
        raise SystemExit(f"{path}: no scenes or repeated names {names}")
    return cfg


def _abs(p):
    return os.path.abspath(p) if p else p


def _key(v):
    return json.dumps(v, sort_keys=True)


def _groups(values):
    """'[001, 002] vs [003]': the scenes grouped by equal value."""
    by = {}
    for s, v in values.items():
        by.setdefault(_key(v), []).append(s)
    return " vs ".join("[" + ", ".join(g) + "]" for g in by.values())


def shared(values, what):
    """(the value every scene shares, problems)."""
    if len({_key(v) for v in values.values()}) == 1:
        return next(iter(values.values())), []
    return None, [f"{what} differs between scenes: {_groups(values)}"]


def _diff(a, b):
    """The keys whose values differ between two dicts (or [''] for two unequal non-dicts)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return sorted(k for k in set(a) | set(b) if _key(a.get(k)) != _key(b.get(k)))
    return [] if _key(a) == _key(b) else [""]


# ------------------------------------------------------------------------------- scenes
def cmd_scenes(a):
    for e in read_config(a.config)["scenes"]:
        vals = [e["name"], e["data"], e["stride"], e["update_stride"], e.get("align") or "none",
                e["ply"], e["field"]]
        if any(not str(v) or " " in str(v) for v in vals):
            raise SystemExit(f"{a.config}: scene {e['name']}: empty value or a space in {vals}")
        print(" ".join(str(v) for v in vals))
    return 0


# -------------------------------------------------------------------------------- reuse
def _parser(kind):
    if kind == "map":
        import fit_gs
        return fit_gs.build_parser()
    if kind == "field":
        import run_mapping
        return run_mapping.build_parser()
    import eval_change_nor as eval_change
    return eval_change.build_parser()


def saved_options(kind, path):
    """(the saved options of an output, the code identity it records, why it cannot be
    reused or None)."""
    if kind == "eval":
        with open(path) as f:
            d = json.load(f)
        st = d.get("status")
        code = ((d.get("metrics") or {}).get("provenance") or {}).get("eval_code")
        return (d.get("config") or {}, dict(eval_code=code),
                None if st == "ok" else f"status {st!r}, not finished")
    import torch
    try:                                            # memory-mapped: the tensors stay on disk
        ck = torch.load(path, map_location="cpu", weights_only=kind == "map", mmap=True)
    except RuntimeError:
        ck = torch.load(path, map_location="cpu", weights_only=kind == "map")
    if kind == "map":
        meta = ck.get("meta") or {}
        return (meta.get("args") or {}, dict(fit_code=meta.get("code_sha256")),
                None if "means" in ck else "not a map")
    return (ck.get("config") or {}, dict(field_code=(ck.get("code") or {}).get("mapping_code_sha256")),
            None if "field" in ck else "not a field")


def kind_code_now(kind):
    """The code identity an output of this kind built now would record."""
    if kind == "map":
        import fit_gs
        return dict(fit_code=fit_gs.code_sha256())
    if kind == "field":
        import run_mapping
        return dict(field_code=run_mapping.mapping_code_state()["mapping_code_sha256"])
    import eval_change_nor as eval_change
    return dict(eval_code=eval_change.code_state())


def same_value(saved, given):
    """A saved option equals the parsed one (numbers as numbers, paths as normalised paths)."""
    if isinstance(saved, bool) or isinstance(given, bool) or saved is None or given is None:
        return saved == given
    if isinstance(given, (int, float)):
        return num(saved) is not None and num(saved) == float(given)
    s, g = str(saved), str(given)
    if s and g and (os.path.isabs(s) or os.path.isabs(g)):
        return os.path.normpath(s) == os.path.normpath(g)
    return s == g


def changed_inputs(rec, keys=("field", "ply")):
    """The input files of an eval_change_nor.py record (its field, its map) that are missing or
    no longer have the sha256 the evaluation read."""
    pv = (rec.get("metrics") or {}).get("provenance") or {}
    out = []
    for key in keys:
        e = pv.get(key) or {}
        p = e.get("path")
        if not p or not os.path.isfile(p) or sha256(p) != e.get("sha256"):
            out.append(f"its {key} {p} changed after the evaluation (sha256)")
    return out


def reuse_problems(kind, path, args):
    """Why the output at `path` was not built by `args` with the code of now (empty: it was)."""
    saved, code, problem = saved_options(kind, path)
    given = vars(_parser(kind).parse_args(args))
    out = [] if problem is None else [problem]
    if kind == "eval" and problem is None:
        with open(path) as f:
            out += changed_inputs(json.load(f))
    for k, v in kind_code_now(kind).items():
        d = _diff(code.get(k), v)
        if d:
            out.append(f"built by other code ({k}" + (f": {', '.join(d)}" if d != [""] else "")
                       + ")")
    for k, v in given.items():
        if k in NOT_COMPARED:
            continue
        if k not in saved:
            out.append(f"{k}: not saved (a run now: {v!r})")
        elif not same_value(saved[k], v):
            out.append(f"{k}: {saved[k]!r} saved, {v!r} now")
    return out


def cmd_reuse(a):
    args = a.args[1:] if a.args[:1] == ["--"] else a.args
    bad = reuse_problems(a.kind, a.file, args)
    if bad:
        print(f"refused: {a.file} ({a.kind}) was not built by this run's options and code "
              f"({'; '.join(bad)}); delete it to rebuild")
        return 1
    print(f"reusing {a.file} ({a.kind}: the same options)")
    return 0


# ------------------------------------------------------------------------------- frozen
def finished_merge(path):
    """The metrics of a finished merge record (exit if missing or not finished)."""
    if not os.path.isfile(path):
        raise SystemExit(f"refused: no validation merge {path} (run STAGE=val first)")
    with open(path) as f:
        d = json.load(f)
    if d.get("status") != "ok" or not isinstance(d.get("metrics"), dict) \
            or "summary" not in d["metrics"]:
        raise SystemExit(f"refused: {path} is not a finished merge (status {d.get('status')!r})")
    return d["metrics"]


def code_now():
    """The code identity a map, field and evaluation built now would record."""
    import eval_change_nor as eval_change
    import fit_gs
    import run_mapping
    return dict(eval_code=eval_change.code_state(), fit_code=fit_gs.code_sha256(),
                field_code=run_mapping.mapping_code_state()["mapping_code_sha256"])


def frozen_problems(val, code=None, settings=None, pipeline=None):
    """How the given code / settings / pipeline settings differ from the validation merge."""
    out = []
    if val.get("split") != "val":
        out.append(f"the frozen merge is of split {val.get('split')!r}, not 'val'")
    for k, v in (code or {}).items():
        d = _diff(v, (val.get("code") or {}).get(k))
        if d:
            out.append(f"{k} differs from validation" + (f" ({', '.join(d)})" if d != [""] else ""))
    if settings is not None:
        d = _diff(settings, val.get("settings"))
        if d:
            out.append(f"settings differ from validation ({', '.join(d)})")
    if pipeline is not None and pipeline != val.get("pipeline"):
        out.append(f"pipeline settings {pipeline!r} differ from validation's "
                   f"{val.get('pipeline')!r}")
    return out


def cmd_frozen(a):
    bad = frozen_problems(finished_merge(a.val), code=code_now(), pipeline=a.pipeline)
    if bad:
        print("refused: the test run must use the method frozen on validation: " + "; ".join(bad))
        return 1
    print(f"frozen: the same code and pipeline settings as the validation merge {a.val}")
    return 0


# -------------------------------------------------------------------------------- merge
def check_scene(entry, rec):
    """Problems of one finished eval_change_nor.py record against its config entry."""
    m, name = rec["metrics"], entry["name"]
    s, v, pv = m["settings"], m["views"], m["provenance"]
    align = entry.get("align") or "none"
    want = dict(scene=(m.get("scene"), name),
                field=(pv["field"]["path"], _abs(entry["field"])),
                ply=(pv["ply"]["path"], _abs(entry["ply"])),
                data=(pv["data"], _abs(entry["data"])),
                align=((m.get("alignment") or {}).get("path"),
                       None if align == "none" else _abs(align)),
                stride=(s["mapping_stride"], entry["stride"]),
                update_stride=(s["update_stride"], entry["update_stride"]),
                n_map_keyframes=(v["n_mapping"], entry["n_map_keyframes"]),
                n_update_keyframes=(v["n_update"], entry["n_update_keyframes"]),
                max_keyframes=(s.get("max_keyframes") or 0, 0),
                max_update_keyframes=(s.get("max_update_keyframes") or 0, 0))
    out = [f"{name}: {k} {got!r} in the record, {exp!r} in the config"
           for k, (got, exp) in want.items() if got != exp]
    a = m.get("alignment")
    if a and a.get("scene") != name:
        out.append(f"{name}: its alignment is for scene {a.get('scene')!r}")
    return out


def map_meta(entry, rec):
    """(the 3DGS map's identity and quality, problems): the file the evaluation read, fit on
    the config's mapping frames."""
    import torch
    path, name = entry["ply"], entry["name"]
    if not os.path.isfile(path):
        return {}, [f"{name}: map {path} is missing"]
    out = []
    sha = sha256(path)
    if sha != rec["metrics"]["provenance"]["ply"].get("sha256"):
        out.append(f"{name}: map {path} changed after the evaluation (sha256)")
    meta = torch.load(path, map_location="cpu", weights_only=True).get("meta") or {}
    args = meta.get("args") or {}
    for k, exp in (("dataset", "nor"), ("part", "map"), ("stride", entry["stride"]),
                   ("data", entry["data"])):
        if not same_value(args.get(k), exp):
            out.append(f"{name}: map {k} {args.get(k)!r}, {exp!r} in the config")
    return dict(sha256=sha, fit_code=meta.get("code_sha256"),
                fit_config_sha256=meta.get("config_sha256"), fit_config=meta.get("config"),
                n_gaussians=meta.get("n_gaussians"), psnr=meta.get("psnr"),
                ssim=meta.get("ssim"), git=meta.get("git")), out


def row_values(m, r):
    """The per-scene values of one row (fractions; None where the record has none)."""
    g = m["gaussian"]["rows"].get(r) or {}
    p = (m.get("pixel") or {}).get("rows", {}).get(r) or {}
    o = m["operating_point"].get(r) or {}
    gc, pc = g.get("common") or {}, p.get("common") or {}
    return dict(g_ap=num(g.get("ap")), g_auroc=num(g.get("auroc")),
                p_ap=num(p.get("ap")), p_auroc=num(p.get("auroc")),
                fa=num(o.get("fa")), rec=num(o.get("rec")),
                g_ap_common=num(gc.get("ap")), g_auroc_common=num(gc.get("auroc")),
                p_ap_common=num(pc.get("ap")), p_auroc_common=num(pc.get("auroc")),
                g_unscored=share(g.get("n_nan"), g.get("n")),
                p_unscored=share(p.get("n_nan"), p.get("n")),
                threshold=num(o.get("threshold")))


def scene_counts(m):
    """The per-scene populations: Gaussians (D2 stale, box-only), views and pixels (D1)."""
    c, v = m["counts"], m["views"]
    p = m.get("pixel") or {}
    gc, pc = m["gaussian"].get("common") or {}, p.get("common") or {}
    return dict(
        n_gaussians=c["n_gaussians"], n_observed=c["n_observed"],
        n_unobserved=c["n_unobserved"], unobserved_share=share(c["n_unobserved"], c["n_gaussians"]),
        n_stale=c["n_stale"], n_stale_observed=c["n_stale_observed"],
        n_stale_unobserved=c["n_stale_unobserved"],
        stale_share=share(c["n_stale_observed"], c["n_observed"]),
        n_in_box=c["n_in_box"], n_in_box_observed=c["n_in_box_observed"],
        n_in_box_near_after=c["n_in_box_near_after"],
        box_static_share=share(c["n_in_box_observed"] - c["n_stale_observed"],
                               c["n_in_box_observed"]),
        n_mapping_views=v["n_mapping"], n_update_views=v["n_update"],
        g_common_n=gc.get("n"), g_common_share=share(gc.get("n"), c["n_observed"]),
        p_frames=p.get("n_frames"), p_frames_no_mask=p.get("n_frames_no_mask"),
        p_n=p.get("n_pix"), p_pos_share=share(p.get("n_pos"), p.get("n_pix")),
        p_common_share=share(pc.get("n"), p.get("n_pix")))


def ours_values(m):
    o = m["ours"]
    return dict(kept=num(o.get("kept")), kept_after_mapping=num(o.get("kept_after_mapping")),
                changed_end_fa=num((o.get("changed_end") or {}).get("fa")),
                changed_end_rec=num((o.get("changed_end") or {}).get("rec")),
                n_resets=num(o.get("n_resets")))


def load_scenes(cfg, folder):
    """({name: record}, {name: record path}, problems) of every scene of the config."""
    recs, paths, out = {}, {}, []
    for e in cfg["scenes"]:
        p = os.path.join(folder, f"{e['name']}.json")
        if not os.path.isfile(p):
            out.append(f"{e['name']}: no record {p}")
            continue
        with open(p) as f:
            rec = json.load(f)
        if rec.get("status") != "ok":
            out.append(f"{e['name']}: {p} is not finished (status {rec.get('status')!r})")
            continue
        recs[e["name"]], paths[e["name"]] = rec, p
        out += check_scene(e, rec)
    return recs, paths, out


def merge(config, folder, frozen=None, pipeline=None):
    """Every Table II number of one split as a JSON-ready dict (SystemExit on a problem)."""
    from usplat.seg_eval import scene_stats
    cfg = read_config(config)
    recs, paths, bad = load_scenes(cfg, folder)
    names = [e["name"] for e in cfg["scenes"] if e["name"] in recs]
    entries = {e["name"]: e for e in cfg["scenes"]}
    maps = {}
    for s in names:
        maps[s], probs = map_meta(entries[s], recs[s])
        bad += probs + [f"{s}: {p}" for p in changed_inputs(recs[s], ("field",))]
    ms = {s: recs[s]["metrics"] for s in names}
    if names:
        rows, p1 = shared({s: m["rows"] for s, m in ms.items()}, "rows")
        settings, p2 = shared({s: {k: m["settings"].get(k) for k in SAME} for s, m in ms.items()},
                              "settings")
        code, p3 = {}, []
        for k, get in (("eval_code", lambda s: ms[s]["provenance"]["eval_code"]),
                       ("field_code", lambda s: ms[s]["provenance"]["field_code"]),
                       ("fit_code", lambda s: maps[s].get("fit_code")),
                       ("fit_config_sha256", lambda s: maps[s].get("fit_config_sha256"))):
            code[k], p = shared({s: get(s) for s in names}, k)
            p3 += p
        bad += p1 + p2 + p3
        inputs, _ = shared({s: m.get("inputs") for s, m in ms.items()}, "inputs")
    if frozen and not bad:
        bad += frozen_problems(finished_merge(frozen), code=code, settings=settings,
                               pipeline=pipeline)
    if bad:
        raise SystemExit(f"{config}: not merged ({len(bad)} problems):\n  " + "\n  ".join(bad))

    per = {s: dict(rows={r: row_values(m, r) for r in rows}, counts=scene_counts(m),
                   ours=ours_values(m),
                   extra={k: dict(g_ap=num(e.get("ap")), g_auroc=num(e.get("auroc")))
                          for k, e in m["gaussian"]["extra"].items()},
                   maps={k: maps[s][k] for k in ("n_gaussians", "psnr", "ssim")},
                   replay_ok=bool(m["replay_check"].get("ok")),
                   replay_max_rel_dW1=num(m["replay_check"].get("max_rel_dW1")),
                   alignment=m.get("alignment") and {k: m["alignment"].get(k) for k in
                                                     ("path", "sha256", "translation_m",
                                                      "rotation_deg")},
                   record=dict(path=os.path.abspath(paths[s]), sha256=sha256(paths[s])),
                   map=dict(path=os.path.abspath(entries[s]["ply"]), sha256=maps[s]["sha256"],
                            git=maps[s]["git"]))
           for s, m in ms.items()}

    def stats(get):
        return scene_stats([get(per[s]) for s in names])

    summary = {r: {k: stats(lambda v: v["rows"][r][k]) for k in ROW_METRICS} for r in rows}
    extra_keys = sorted({k for s in names for k in per[s]["extra"]})
    extra = {k: {mk: stats(lambda v: (v["extra"].get(k) or {}).get(mk)) for mk in
                 ("g_ap", "g_auroc")} for k in extra_keys}
    counts = {k: stats(lambda v: num(v["counts"][k])) for k in per[names[0]]["counts"]}
    counts["pooled_unobserved_share"] = share(
        sum(per[s]["counts"]["n_unobserved"] for s in names),
        sum(per[s]["counts"]["n_gaussians"] for s in names))
    replay_bad = [s for s in names if not per[s]["replay_ok"]]
    return dict(
        split=cfg.get("split"), config=dict(path=os.path.abspath(config), sha256=sha256(config)),
        scenes_order=names, rows=rows, inputs=inputs, summary=summary, extra=extra,
        ours={k: stats(lambda v: v["ours"][k]) for k in per[names[0]]["ours"]},
        counts=counts, maps={k: stats(lambda v: num(v["maps"][k])) for k in
                             ("n_gaussians", "psnr", "ssim")},
        scenes=per, settings=settings, code=code, pipeline=pipeline,
        fit_config=maps[names[0]].get("fit_config"),
        replay=dict(all_ok=not replay_bad, not_ok=replay_bad,
                    max_rel_dW1=max(per[s]["replay_max_rel_dW1"] or 0.0 for s in names)),
        frozen=None if not frozen else dict(path=os.path.abspath(frozen), sha256=sha256(frozen)))


def summary_lines(out):
    """One printed line per row (means in %) and the counts, for the logs."""
    lines, sm = [], out["summary"]
    for r in out["rows"]:
        v = {k: sm[r][k]["mean"] for k in ("g_ap", "g_auroc", "p_ap", "p_auroc", "fa", "rec")}
        lines.append(f"{r:>10} [{(out['inputs'] or {}).get(r, '?'):>3}]: " + " ".join(
            f"{k} {100 * x:.1f}" for k, x in v.items()) + f" (N = {sm[r]['g_ap']['n']})")
    c = out["counts"]
    lines.append(f"scenes {out['scenes_order']}: unobserved {100 * c['unobserved_share']['mean']:.1f}% "
                 f"(pooled {100 * c['pooled_unobserved_share']:.1f}%), stale observed "
                 f"{c['n_stale_observed']['mean']:.0f} per scene, Kept "
                 f"{100 * out['ours']['kept']['mean']:.1f}%; replay ok {out['replay']['all_ok']}")
    return lines


def cmd_merge(a):
    from usplat.runlog import RunRecord
    argv = [sys.argv[0]] + list(a.argv)
    with RunRecord(a.out, config=dict(config=a.config, dir=a.dir, frozen=a.frozen,
                                      pipeline=a.pipeline, out=a.out),
                   name="nor_tab2_merge", argv=argv) as rec:
        out = merge(a.config, a.dir, a.frozen or None, a.pipeline)
        for k, v in out.items():
            rec.add(k, v)
        rec.add_input(a.config)
        for s in out["scenes_order"]:
            rec.add_input(out["scenes"][s]["record"]["path"])
        if a.frozen:
            rec.add_input(a.frozen)
    print("\n".join(summary_lines(out)))
    print(f"merged {len(out['scenes_order'])} scenes of {a.config} -> {a.out}")
    return 0


# --------------------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scenes", help="one line per scene of a split config")
    p.add_argument("config")
    p = sub.add_parser("reuse", help="exit 1 if an output was built with other options")
    p.add_argument("kind", choices=["map", "field", "eval"])
    p.add_argument("file")
    p.add_argument("args", nargs=argparse.REMAINDER,
                   help="-- and the arguments of fit_gs.py / run_mapping.py / eval_change_nor.py")
    p = sub.add_parser("frozen", help="exit 1 if code or settings differ from validation")
    p.add_argument("val", help="the finished validation merge (change_val.json)")
    p.add_argument("--pipeline", default=None, help="the pipeline settings of scripts/nor_tab2.sh")
    p = sub.add_parser("merge", help="mean and population std over the scenes")
    p.add_argument("--config", required=True)
    p.add_argument("--dir", required=True, help="folder of the <scene>.json records")
    p.add_argument("--out", required=True)
    p.add_argument("--frozen", default="", help="the validation merge the test must match")
    p.add_argument("--pipeline", default=None, help="the pipeline settings of scripts/nor_tab2.sh")
    return ap


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    a = build_parser().parse_args(argv)
    a.argv = argv
    return dict(scenes=cmd_scenes, reuse=cmd_reuse, frozen=cmd_frozen, merge=cmd_merge)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
