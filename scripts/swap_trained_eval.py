"""
swap_trained_eval.py — accuracy on swapped and static Gaussians of the re-trained Feature 3DGS /
LEGaussians rows (scripts/distill_update_swap.py) on one Replica semantic-swap scene
(configs/swap_trained_prereg.json). The measure is scripts/eval_change.py's segmentation block,
imported and never edited: one _seg_block call scores the field's own rows (ours, averaging) and
every trained row as a compare field: change_eval.classify (argmax over the field's PCA
projection of the ground truth's class text, ties = no class) -> change_eval.region_accuracy
(acc_changed on the swapped Gaussians, acc_static on the static ones, target >= 0, same GT).

    python3 scripts/swap_trained_eval.py --field results/swap_relmax/fields/office_0_swap.pt \
        --gt results/swap_relmax/gt/office_0.pt --config configs/swap_replica.json \
        --scene office_0 --split val --record results/swap_relmax/eval_val_office_0.json \
        --trained <field>_f3dgs_upd.pt,<field>_legs_l4_upd.pt \
        --prereg configs/swap_trained_prereg.json --out results/swap_relmax/trained_val_office_0.json

Per-Gaussian features: Feature 3DGS its trained features; LEGaussians each Gaussian's own 8-d
code decoded by its MLP, softmax(MLP(s_n)) S (its query rule, at one Gaussian).
Control: the same call recomputes the row 'averaging' from the field; every value of it must
equal the scene's eval record (scripts/eval_change.py eval, --record) within 1e-9, else the
status is "control failed" and the exit code 2. --check-only runs the control alone (no
--trained, no record written).
Refused: a trained file of another PCA basis, scene, frames, bank, spec or transform code than
the field and ground truth; with --prereg, rows or settings other than the registered ones.
--split test: refused with uncommitted changes in scripts/ usplat/ configs/, over a finished
record, without --prereg, and without --reference (the finished validation records of every
validation scene, control passed, same code).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import distill_update_swap as DUS                                           # noqa: E402
import eval_change as EC                                                    # noqa: E402
from usplat import change_eval as CE                                        # noqa: E402
from usplat import distill as D                                             # noqa: E402

TOL = 1e-9
CONTROL = "averaging"
FIELD_ROWS = ("ours", "averaging")              # the field's own rows in _seg_block
CODE = ("scripts/swap_trained_eval.py", "scripts/distill_update_swap.py", "scripts/eval_change.py",
        "usplat/change_eval.py", "usplat/distill.py", "usplat/swap.py")
SETTINGS = ("iters", "min_iters", "tol", "lr", "alpha_min", "seed", "channel_chunk")
LEGS_SETTINGS = ("codes", "lam_u")


def code_state():
    return {rel: DUS.sha256_file(os.path.join(REPO, rel)) for rel in CODE}


class Features:
    """A trained row as a compare field of eval_change._seg_block (it reads f.mu_post())."""

    def __init__(self, x):
        self.x = x

    def mu_post(self):
        return self.x


@torch.no_grad()
def gaussian_features(d, chunk=1 << 17):
    """(N, d) per-Gaussian features of a trained file: Feature 3DGS its features, LEGaussians
    its codes decoded by its MLP (softmax(MLP(s_n)) S)."""
    f = d["features"].float()
    if d.get("decoder") is None:
        return f
    dec = D.LEGDecoder(d["decoder"]["codebook"], dim=f.shape[1],
                       hidden=d.get("decoder_hidden", D.LEGS_HIDDEN))
    dec.load_state_dict(d["decoder"])
    dec.eval()
    return torch.cat([dec(f[k:k + chunk]) for k in range(0, f.shape[0], chunk)])


def check_settings(d, prereg, path):
    """A trained file's row name and settings against the registration."""
    if d["name"] not in prereg["rows"]:
        raise SystemExit(f"{path}: row {d['name']!r} is not registered ({sorted(prereg['rows'])})")
    want = prereg["training"]["settings"]
    keys = SETTINGS + (LEGS_SETTINGS if d["config"]["method"] == "legs" else ())
    bad = [f"{k} {d['config'].get(k)!r} != {want[k]!r}" for k in keys
           if float(d["config"].get(k)) != float(want[k])]
    if bad:
        raise SystemExit(f"{path}: not the registered settings: " + "; ".join(bad))


def load_trained(path, ckpt, gt, scene, cfg_sha, N, prereg=None):
    """(name, (N, d) features, provenance) of one trained file, checked against the field, its
    frames log and bank, the spec and the ground truth."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    bad = []
    if d.get("pca_basis_sha256") != D.basis_sha256(ckpt["pca"]):
        raise SystemExit(f"{path}: distilled in another PCA basis than the field")
    if d["features"].shape[0] != N:
        bad.append(f"{d['features'].shape[0]} Gaussians, the field has {N}")
    sw = DUS.swap_inputs(ckpt)
    if d.get("scene") != scene:
        bad.append(f"scene {d.get('scene')!r} != {scene!r}")
    if (d.get("frames") or {}).get("update") != DUS.log_frames(sw["log"]):
        bad.append("its update frames are not the field's frames log")
    if d.get("n_update_frames") != len(sw["log"]["frames"]):
        bad.append(f"{d.get('n_update_frames')} update frames, the log has {len(sw['log']['frames'])}")
    kf = (ckpt.get("timing") or {}).get("keyframes")
    if kf is not None and d.get("n_frames") != int(kf):
        bad.append(f"{d.get('n_frames')} frames, the field ingested {kf}")
    if (d.get("bank") or {}).get("sha256") != sw["bank_sha"]:
        bad.append("another bank than the field's")
    if (d.get("spec") or {}).get("spec_sha256") != cfg_sha:
        bad.append("another spec")
    if d.get("transform_code_sha256") != gt.get("transform_code_sha256"):
        bad.append("another transform code than the ground truth's")
    if (d.get("config") or {}).get("frames") != DUS.FRAMES:
        bad.append(f"frames {(d.get('config') or {}).get('frames')!r} != {DUS.FRAMES!r}")
    if bad:
        raise SystemExit(f"{path}: not trained on this field's swap: " + "; ".join(bad))
    if prereg is not None:
        check_settings(d, prereg, path)
    prov = dict(path=os.path.abspath(path), sha256=DUS.sha256_file(path), name=d["name"],
                identity=d.get("identity"), config=d.get("config"), code_sha256=d.get("code_sha256"),
                n_frames=d.get("n_frames"), n_map_frames=d.get("n_map_frames"),
                n_update_frames=d.get("n_update_frames"), iters_run=d.get("iters_run"),
                bank_sha256=d["bank"]["sha256"], frames_log=d.get("frames_log"),
                frames_ref=d.get("frames_ref"), timing=d.get("timing"))
    return d["name"], gaussian_features(d), prov


def _diff(a, b):
    a, b = CE._num(a), CE._num(b)
    if math.isnan(a) and math.isnan(b):
        return 0.0
    return abs(a - b) if math.isfinite(a) and math.isfinite(b) else float("inf")


def control_check(row, ref, tol=TOL):
    """The recomputed row against the record's: every key, within tol (NaN equals NaN)."""
    keys = sorted(set(row) | set(ref))
    diffs = {k: (_diff(row[k], ref[k]) if k in row and k in ref else float("inf")) for k in keys}
    worst = max(diffs.values()) if diffs else float("inf")
    return dict(ok=bool(keys) and worst <= tol, tol=tol, max_abs_diff=worst, diffs=diffs,
                rerun=row, record=ref)


def read_record(path, scene, split, cfg_sha, gt):
    """The scene's finished eval record (scripts/eval_change.py eval) of this spec and GT."""
    d = json.load(open(path))
    m = EC._metrics(d)
    bad = []
    if d.get("status", "ok") != "ok":
        bad.append(f"status {d.get('status')!r}")
    for k, want in (("scene", scene), ("split", split), ("config_sha256", cfg_sha)):
        if m.get(k) != want:
            bad.append(f"{k} {m.get(k)!r} != {want!r}")
    g = m.get("gt") or {}
    if g.get("n") != gt["N"] or g.get("transform_code_sha256") != gt.get("transform_code_sha256"):
        bad.append("another ground truth (size or transform code)")
    row = ((m.get("seg") or {}).get("rows") or {}).get(CONTROL) or {}
    if not row or "status" in row:
        bad.append(f"no {CONTROL} row ({row.get('status', 'missing')})")
    if bad:
        raise SystemExit(f"{path}: not the eval record of this scene: " + "; ".join(bad))
    return m


def check_reference(paths, cfg, cfg_sha):
    """The test split's gate: a finished validation record per validation scene, control
    passed, made with this code."""
    if not paths:
        raise SystemExit("refused: the test split is scored after the validation records: give "
                         "--reference <trained_val_<scene>.json of every validation scene>")
    code, scenes = code_state(), []
    for p in paths:
        d = json.load(open(p))
        if d.get("status") != "ok" or d.get("split") != "val" or \
                d.get("config_sha256") != cfg_sha or not (d.get("control") or {}).get("ok"):
            raise SystemExit(f"refused: {p} is not a finished validation record of this spec "
                             f"with its control passed")
        if (d.get("provenance") or {}).get("code") != code:
            raise SystemExit(f"refused: {p} was made with other code than this run")
        scenes.append(d["scene"])
    if sorted(scenes) != sorted(CE.split_scenes(cfg, "val")):
        raise SystemExit(f"refused: validation records of {scenes}, the validation scenes are "
                         f"{CE.split_scenes(cfg, 'val')}")


def test_gate(args, cfg, cfg_sha):
    if os.path.exists(args.out):
        try:
            done = json.load(open(args.out)).get("status") == "ok"
        except (OSError, ValueError):
            done = False
        if done:
            raise SystemExit(f"refused: {args.out} holds a finished test record (the test split runs once)")
    dirty = EC._dirty_tree()
    if dirty:
        raise SystemExit(f"refused: uncommitted changes in scripts/ usplat/ configs/ ({len(dirty)} "
                         f"files): the test run must name a commit")
    if not args.prereg:
        raise SystemExit("refused: the test split needs --prereg (the registered rows and settings)")
    check_reference(args.reference, cfg, cfg_sha)


def write_json(path, obj):
    from usplat.runlog import to_jsonable
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(to_jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--field", required=True, help="the scene's swap field (with the averaging baseline)")
    ap.add_argument("--gt", required=True, help="its ground truth (eval_change.py gt-swap)")
    ap.add_argument("--config", required=True, help="the frozen swap spec")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--record", required=True, help="the scene's eval record (the control)")
    ap.add_argument("--trained", default="", help="comma-separated distill_update_swap.py files")
    ap.add_argument("--prereg", default="", help="configs/swap_trained_prereg.json")
    ap.add_argument("--reference", nargs="*", default=[],
                    help="--split test: the finished validation records")
    ap.add_argument("--check-only", action="store_true", help="the control alone, no record")
    ap.add_argument("--out", default="", help="output record (JSON)")
    return ap


def run(args, argv):
    t0 = time.time()
    from usplat.vmf_field import DirectionalSemanticField
    cfg, cfg_sha = CE.load_config(args.config)
    if args.scene not in CE.split_scenes(cfg, args.split):
        raise SystemExit(f"scene {args.scene!r} is not a {args.split} scene of {args.config}")
    paths = [p for p in args.trained.split(",") if p]
    if args.check_only and paths:
        raise SystemExit("--check-only scores no trained file")
    if not args.check_only and not (paths and args.out):
        raise SystemExit("give --trained and --out (or --check-only)")
    if args.split == "test" and not args.check_only:
        test_gate(args, cfg, cfg_sha)
    prereg = json.load(open(args.prereg)) if args.prereg else None
    gt = CE.load_gt(args.gt)
    if str(gt.get("scene")) != args.scene:
        raise SystemExit(f"{args.gt}: ground truth of scene {gt.get('scene')!r}")
    m = read_record(args.record, args.scene, args.split, cfg_sha, gt)
    ckpt = EC._load_ckpt(args.field)
    field = DirectionalSemanticField.from_state(ckpt["field"], device="cpu")
    if field.N != gt["N"]:
        raise SystemExit(f"the field has {field.N} Gaussians, the ground truth {gt['N']}")
    EC._check_swap_frames(gt, ckpt)            # the field saw the frames the GT was rendered over
    compare, prov = {}, {}
    for p in paths:
        name, feat, pv = load_trained(p, ckpt, gt, args.scene, cfg_sha, field.N, prereg)
        if name in compare or name in FIELD_ROWS:
            raise SystemExit(f"two rows are named {name!r}")
        compare[name] = (ckpt, Features(feat), p)         # the field's PCA projects the text
        prov[name] = pv
    if prereg is not None and paths and sorted(compare) != sorted(prereg["rows"]):
        raise SystemExit(f"rows {sorted(compare)}, registered {sorted(prereg['rows'])}")
    seg = EC._seg_block(gt, ckpt, field, compare)
    if seg.get("status") != "ok":
        raise SystemExit(f"{args.gt}: {seg.get('status')}")
    control = control_check(seg["rows"][CONTROL], m["seg"]["rows"][CONTROL])
    control["ours_info"] = control_check(seg["rows"]["ours"], m["seg"]["rows"].get("ours") or {})
    control["field_config_equal"] = ckpt.get("field_config") == (m.get("field") or {}).get("field_config")
    for name in compare:
        r = seg["rows"][name]
        print(f"{args.scene} {name:12s} acc changed {r['acc_changed']:.4f} ({r['n_changed']}) "
              f"static {r['acc_static']:.4f} ({r['n_static']})")
    print(f"{args.scene} control {CONTROL}: max |rerun - record| {control['max_abs_diff']:.3g} "
          f"(tol {TOL:g}) -> {'ok' if control['ok'] else 'FAILED'}; ours {control['ours_info']['max_abs_diff']:.3g}")
    out = dict(status="ok" if control["ok"] else "control failed", split=args.split,
               scene=args.scene, config_sha256=cfg_sha, control=control, exit=0 if control["ok"] else 2)
    if args.check_only:
        return out
    labels = {k: (prereg["rows"][k]["label"] if prereg else k) for k in compare}
    out.update(rows={k: seg["rows"][k] for k in compare}, labels=labels, classes=seg["classes"],
               recomputed={k: seg["rows"][k] for k in FIELD_ROWS},
               record_rows=m["seg"]["rows"],
               record=dict(path=os.path.abspath(args.record), sha256=DUS.sha256_file(args.record)),
               gt=dict(path=os.path.abspath(args.gt), n=gt["N"], shares=gt.get("shares"),
                       transform_code_sha256=gt.get("transform_code_sha256")),
               field=dict(path=os.path.abspath(args.field),
                          pca_basis_sha256=D.basis_sha256(ckpt["pca"]),
                          mapping_code=(ckpt.get("code") or {}).get("mapping_code_sha256"),
                          feature_transform_arg=(ckpt.get("config") or {}).get("feature_transform_arg")),
               trained=prov,
               prereg=(dict(path=os.path.abspath(args.prereg), sha256=DUS.sha256_file(args.prereg))
                       if args.prereg else None),
               reference=[os.path.abspath(p) for p in args.reference],
               provenance=dict(argv=argv, git=_git(), code=code_state()),
               timing=dict(wall_s=time.time() - t0))
    write_json(args.out, {k: v for k, v in out.items() if k != "exit"})
    print(f"wrote {args.out}")
    return out


def _git():
    from usplat.runlog import git_state
    g = git_state(REPO)
    return {k: g.get(k) for k in ("commit", "branch", "dirty", "dirty_tracked", "code_sha256", "error")}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    return run(build_parser().parse_args(argv), argv)


if __name__ == "__main__":
    sys.exit(main()["exit"])
