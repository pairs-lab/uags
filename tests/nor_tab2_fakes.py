"""Fake outputs of the NeRF-Object-Removal pipeline (scripts/nor_tab2.sh) for CPU tests, and a
mock of scripts/gpu_gate.sh that writes them (GPU_GATE=<a script that runs this file>):

  python3 tests/nor_tab2_fakes.py <need_mib> python3 scripts/fit_gs.py|run_mapping.py|eval_change_nor.py <args>

Run from the repo root (the pipeline's cwd). Each step's arguments are parsed with the real
script's parser and saved as that script saves them (a map's meta["args"], a field's
config, an eval_change_nor.py record's config), with the code identity of the repo in the cwd,
so scripts/nor_tab2.py reuse / frozen / merge check them as real outputs.
  fit_gs.py       a tiny map .pt (usplat.gs_fit.save_gaussians_pt) with meta: args, FitConfig,
                  config_sha256, code_sha256, n_gaussians, psnr, ssim, git
  run_mapping.py  a field .pt: {field, config, code: run_mapping.mapping_code_state()}
  eval_change_nor.py  a usplat.runlog record whose metrics follow eval_change_nor.py's layout, with
                  the numbers of fake_values(scene index) and the views / strides / paths of
                  the scene's entry in configs/nor_val.json or configs/nor_test.json
Every call appends "<script> <out>" to $FAKE_CALLS (if set). FAKE_CRASH_ONCE=<scene>: the
first evaluation of that scene writes a "failed" record and exits 1 (a CUDA time-out).
"""
from __future__ import annotations
import hashlib
import json
import os
import sys

ROWS = ("ncc", "depth", "feat", "normlift", "dispersion", "ours", "fused")
INPUTS = dict(ncc="I", depth="D", feat="F", normlift="F", dispersion="F", ours="F+D",
              fused="F+D")
EXTRA = ("depth_gauss/kmin", "ncc/max", "depth_render/max", "feat/max", "depth_gauss/max")


def fake_values(i):
    """The per-row values of scene number i (fractions; fused has no operating point). Each
    row j and scene i has its own value, so a merged mean tells where it came from."""
    out = {}
    for j, r in enumerate(ROWS):
        out[r] = dict(g_ap=0.05 + 0.08 * j + 0.01 * i, g_auroc=0.60 + 0.04 * j - 0.005 * i,
                      p_ap=0.10 + 0.07 * j + 0.004 * i, p_auroc=0.55 + 0.05 * j + 0.003 * i,
                      g_ap_common=0.06 + 0.08 * j + 0.01 * i, g_auroc_common=0.61 + 0.04 * j,
                      p_ap_common=0.11 + 0.07 * j, p_auroc_common=0.56 + 0.05 * j,
                      fa=None if r == "fused" else 0.10 - 0.012 * j + 0.001 * i,
                      rec=None if r == "fused" else 0.30 + 0.08 * j - 0.002 * i,
                      threshold=None if r == "fused" else 0.5 + 0.1 * j + 0.01 * i,
                      n_nan=10 * j + i, p_n_nan=20 * j + i)
    return out


def fake_counts(i):
    return dict(n_gaussians=10000 + 100 * i, n_observed=9000 + 50 * i, n_unobserved=1000 + 50 * i,
                n_in_box=600 + i, n_in_box_observed=580 + i, n_in_box_near_after=80 + i,
                n_stale=520, n_stale_observed=500, n_stale_unobserved=20,
                n_static_observed=8500 + 50 * i)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def file_entry(path):
    return dict(path=os.path.abspath(path), bytes=os.path.getsize(path), sha256=sha256(path))


def scene_entry(field):
    """(index in its split, entry) of the config scene whose field is `field`."""
    for split in ("val", "test"):
        p = os.path.join("configs", f"nor_{split}.json")
        if not os.path.isfile(p):
            continue
        for i, e in enumerate(json.load(open(p))["scenes"]):
            if os.path.abspath(e["field"]) == os.path.abspath(field):
                return i, e
    raise SystemExit(f"fake eval: no config scene with field {field}")


def eval_metrics(args, i, entry, field_ck, eval_code, mapping_now):
    """eval_change_nor.py's metrics dict of one scene, with fake numbers."""
    v, c = fake_values(i), fake_counts(i)
    n_obs, n_pix = c["n_observed"], 200000
    cfg = field_ck["config"]
    g = {r: dict(ap=x["g_ap"], auroc=x["g_auroc"], n=n_obs, n_pos=c["n_stale_observed"],
                 n_nan=x["n_nan"], n_nan_pos=0,
                 common=dict(ap=x["g_ap_common"], auroc=x["g_auroc_common"])) for r, x in v.items()}
    p = {r: dict(ap=x["p_ap"], auroc=x["p_auroc"], n=n_pix, n_pos=20000, n_nan=x["p_n_nan"],
                 n_nan_pos=0, common=dict(ap=x["p_ap_common"], auroc=x["p_auroc_common"]))
         for r, x in v.items()}
    op = {r: dict(fa=x["fa"], rec=x["rec"], threshold=x["threshold"], rule="fake")
          for r, x in v.items()}
    align = None
    if args.align != "none":
        rec = json.load(open(args.align))
        align = dict(file_entry(args.align), scene=rec.get("scene"), T=rec.get("T"),
                     translation_m=0.01, rotation_deg=0.1, before=None, after=None)
    return dict(
        scene=os.path.basename(os.path.normpath(cfg["data"])), rows=list(ROWS), inputs=INPUTS,
        gaussian=dict(rows=g, common=dict(n=n_obs - 500, n_pos=450, rows=list(ROWS)),
                      extra={k: dict(ap=0.2 + 0.01 * i, auroc=0.8, n=n_obs, n_pos=500, n_nan=0,
                                     n_nan_pos=0) for k in EXTRA}),
        pixel=dict(rows=p, common=dict(n=n_pix - 1000, n_pos=19000, rows=list(ROWS)),
                   n_frames=entry["n_update_keyframes"], n_frames_no_mask=0, n_pix=n_pix,
                   n_pos=20000, frames=[]),
        operating_point=op,
        ours=dict(kept=0.05 + 0.01 * i, kept_after_mapping=0.2, n_kept=25,
                  changed_end=dict(fa=0.002, rec=0.4 + 0.01 * i), n_resets=300 + i,
                  threshold=dict(pooled=0.06)),
        counts=dict(c, update_views_stale=dict(n=0), update_views_static=dict(n=0),
                    after_removal=dict(n_frames=1)),
        thresholds=dict(rows={r: x["threshold"] for r, x in v.items()}, quantile=0.999),
        replay_check=dict(ok=True, max_rel_dW1=0.0, c_hist_l1_share=0.0),
        views=dict(mapping=list(range(entry["n_map_keyframes"])),
                   update=list(range(entry["n_update_keyframes"])),
                   n_mapping=entry["n_map_keyframes"], n_update=entry["n_update_keyframes"],
                   grid_hw=[270, 360]),
        settings=dict(change_quantile=args.change_quantile, k=args.k or 2, tau_cov=args.tau_cov,
                      undef_frac=args.undef_frac, inflate_m=args.inflate, near_m=args.near,
                      update_stride=int(args.update_stride or cfg["stride"]),
                      mapping_stride=int(cfg["stride"]), kf_trans=cfg["kf_trans"],
                      kf_rot=cfg["kf_rot"], max_keyframes=cfg["max_keyframes"],
                      max_update_keyframes=args.max_update_keyframes,
                      downscale=cfg["downscale"], extractor=cfg["extractor"],
                      backend=args.backend, device=args.device,
                      field_config=dict(change_persist=2, calib_mode="history")),
        alignment=align,
        provenance=dict(field=file_entry(args.field), ply=file_entry(cfg["ply"]),
                        data=os.path.abspath(cfg["data"]), eval_code=eval_code,
                        field_code=field_ck["code"]["mapping_code_sha256"],
                        mapping_code_now=mapping_now, gpu_reserved_mib=0, peak_alloc_mib=1.0),
        timing=dict(update_s=1.0))


def fake_fit(argv):
    import torch
    import fit_gs
    from dataclasses import asdict, fields
    from usplat.gs_fit import FitConfig, config_sha256, save_gaussians_pt
    a = fit_gs.build_parser().parse_args(argv)
    cfg = FitConfig(**{f.name: getattr(a, f.name) for f in fields(FitConfig)})
    n = 64
    splats = dict(means=torch.zeros(n, 3), quats=torch.tensor([1.0, 0, 0, 0]).repeat(n, 1),
                  scales=torch.zeros(n, 3), opacities=torch.zeros(n), sh0=torch.zeros(n, 1, 3))
    meta = dict(args=vars(a), config=asdict(cfg), config_sha256=config_sha256(cfg),
                code_sha256=fit_gs.code_sha256(), git="fake", n_gaussians=n,
                psnr=24.0 + a.stride, ssim=0.8, iters=a.iters)
    save_gaussians_pt(a.out, splats, meta)
    print(json.dumps(dict(psnr=meta["psnr"], n_gaussians=n, out=os.path.abspath(a.out))))
    return a.out


def fake_map(argv):
    import torch
    import run_mapping
    a = run_mapping.build_parser().parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(dict(field=dict(N=64), config=vars(a), code=run_mapping.mapping_code_state(),
                    timing=dict(keyframes=0)), a.out)
    print(f"keyframes used 0 | saved field -> {a.out}")
    return a.out


def fake_eval(argv):
    import torch
    import eval_change_nor as eval_change
    import run_mapping
    from usplat.runlog import RunRecord
    a = eval_change.build_parser().parse_args(argv)
    i, entry = scene_entry(a.field)
    crash = os.environ.get("FAKE_CRASH_ONCE")
    marker = f"{a.out}.crashed"
    ck = torch.load(a.field, map_location="cpu", weights_only=False)
    with RunRecord(a.out, config=vars(a), name="eval_change",
                   argv=["scripts/eval_change_nor.py"] + list(argv)) as rec:
        if crash == entry["name"] and not os.path.exists(marker):
            open(marker, "w").close()
            raise RuntimeError("CUDA error: the launch timed out and was terminated (fake)")
        m = eval_metrics(a, i, entry, ck, eval_change.code_state(),
                         run_mapping.mapping_code_state()["mapping_code_sha256"])
        for k, v in m.items():
            rec.add(k, v)
    for r in ROWS:
        print(f"{m['scene']} {r:>10} [{INPUTS[r]:>3}]: fake")
    return a.out


def main(argv):
    need, py, script, *rest = argv
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
    name = os.path.basename(script)
    print(f"[gpu_gate] fake: need {need} MiB; running: {py} {script} ...", flush=True)
    out = dict(**{"fit_gs.py": fake_fit, "run_mapping.py": fake_map,
                  "eval_change_nor.py": fake_eval})[name](rest)
    calls = os.environ.get("FAKE_CALLS")
    if calls:
        with open(calls, "a") as f:
            f.write(f"{name} {need} {os.path.abspath(out)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
