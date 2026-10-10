"""
static_fa_report.py — the static false-alarm protocol of scripts/replica_static_fa.sh: the
run_mapping.py arguments of a frozen config, the validation selection and the test criterion.

Replica Sequence_2 shows the scene of Sequence_1 from another trajectory: nothing changes, so
every fire of the update pass is a false alarm. Each run (run_mapping.py --summary-json) counts
the defined change tests and the fires of every update view; the realised per-view
false-alarm rate is fires / tests, and the calibrated threshold targets 1 - q.

  args CONFIG [--selection SEL]   run_mapping.py arguments of a configs/static_fa*.json, one
                                  per line (the fixed-threshold config reads its threshold
                                  from the validation selection)
  select --results OUT            after STAGE=val: the fixed threshold carried over from
                                  validation (rule below) and the validation rates of every
                                  config -> OUT/val_selection.json
  criterion --results OUT         after STAGE=test: the PRE-REGISTERED criterion of
                                  configs/static_fa.json on the test runs, and the rates of
                                  every config -> OUT/test_criterion.json

Runs live at OUT/runs/<config name>/<scene>.json (usplat.runlog records; only status "ok"
counts). select and criterion refuse missing or unfinished runs of the frozen config, and any
run whose recorded argv does not contain the config's run_mapping arguments (`args`, in order).

Fixed-threshold rule (written 2026-10-10, before any run; configs/static_fa_fixed.json): the
q-quantile (q = 0.999, upper edge of the q bin of the field's 4000-bin histogram on [0, 2]) of
the per-view scores of the tested Gaussians in the UPDATE passes of the frozen config
(static_fa) on the validation scenes, pooled: the threshold whose per-view false-alarm rate on
the static validation scenes is 1 - q (the rule the paper uses for the baselines' thresholds).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

CONFIG_KEYS = ("field_config", "change_quantile", "change_baselines", "fixed_change_thr",
               "use_depth", "update_stride")


def load_config(path):
    c = json.load(open(path))
    missing = [k for k in CONFIG_KEYS if k not in c]
    if missing:
        raise SystemExit(f"{path}: missing keys {missing}")
    fc = c["field_config"]
    if "change_quantile" in fc and fc["change_quantile"] != c["change_quantile"]:
        raise SystemExit(f"{path}: field_config.change_quantile != change_quantile")
    return c


def hist_quantile(counts, q, lo=0.0, hi=2.0):
    """Upper edge of the q bin of a histogram on [lo, hi] (the estimator of
    DirectionalSemanticField.calibrate_change_threshold)."""
    n = float(sum(counts))
    if n <= 0:
        raise ValueError("empty histogram")
    acc = 0.0
    for i, c in enumerate(counts):
        acc += c
        if acc / n >= q:
            return lo + (i + 1) * (hi - lo) / len(counts)
    return hi


def run_args(cfg, selection=None):
    """run_mapping.py arguments of a config (paths and data flags are the script's)."""
    a = ["--field-config", json.dumps(cfg["field_config"], sort_keys=True),
         "--change-quantile", repr(float(cfg["change_quantile"])),
         "--update-stride", str(int(cfg["update_stride"]))]
    if cfg["use_depth"]:
        a.append("--use-depth")
    if cfg["change_baselines"]:
        a.append("--change-baselines")
    fixed = cfg["fixed_change_thr"]
    if fixed == "from_validation":
        if selection is None:
            raise SystemExit(f"{cfg['name']}: the fixed threshold comes from the validation "
                             "selection (--selection)")
        sel = _load_finished(selection, "validation selection")
        fixed = sel["metrics"]["fixed_change_thr"]
    if fixed is not None:
        a += ["--fixed-change-thr", repr(float(fixed))]
    return a


def _load_finished(path, what):
    if not os.path.exists(path):
        raise SystemExit(f"refused: no {what} at {path}")
    d = json.load(open(path))
    if d.get("status") != "ok":
        raise SystemExit(f"refused: {path} is not a finished {what} (status {d.get('status')})")
    return d


def _run_matches(rec, cfg, selection=None):
    """The run used the config's run_mapping arguments (from its record's argv)."""
    argv = rec.get("argv") or []
    want = run_args(cfg, selection)
    i = 0
    for a in want:                                   # want is a subsequence of argv
        try:
            i = argv.index(a, i) + 1
        except ValueError:
            return False
    return True


def load_runs(results, cfg, scenes, selection=None, allow_missing=False):
    """{scene: record metrics} of OUT/runs/<cfg name>/<scene>.json, finished runs only."""
    out = {}
    for s in scenes:
        p = os.path.join(results, "runs", cfg["name"], f"{s}.json")
        if allow_missing and not os.path.exists(p):
            continue
        rec = _load_finished(p, f"{cfg['name']} run of {s}")
        if not _run_matches(rec, cfg, selection):
            raise SystemExit(f"refused: {p} was not run with the arguments of {cfg['name']} "
                             f"({' '.join(run_args(cfg, selection))})")
        out[s] = rec["metrics"]
    return out


def _rates(runs):
    rows = {}
    for s, m in runs.items():
        u = m["update"]
        rows[s] = dict(rate=u["rate"], tests=u["tests"], fires=u["fires"],
                       reset_pixel_mass_share=u["reset_pixel_mass_share"],
                       threshold=m["calibration"]["change_thr"],
                       by_hist=[b["rate"] for b in u["by_hist"]],
                       baselines={n: b["rate"] for n, b in u.get("baselines", {}).items()})
    return rows


def _configs(config_dir, names):
    return {n: load_config(os.path.join(config_dir, f"{n}.json")) for n in names}


def select(args, rec=None):
    """Validation: the fixed threshold of the fixed-threshold ablation and the val rates."""
    frozen = load_config(args.config)
    scenes = frozen["data"]["val_scenes"]
    runs = load_runs(args.results, frozen, scenes)
    q = float(frozen["change_quantile"])
    pooled = [0] * 4000
    for m in runs.values():
        pooled = [a + b for a, b in zip(pooled, m["update"]["score_hist"])]
    thr = hist_quantile(pooled, q)
    others = {}
    for name, c in _configs(os.path.dirname(os.path.abspath(args.config)),
                            frozen["ablations"]).items():
        if c["fixed_change_thr"] == "from_validation":
            continue                                 # defined by this selection
        others[name] = _rates(load_runs(args.results, c, scenes, allow_missing=True))
    out = dict(fixed_change_thr=thr, quantile=q, val_scenes=scenes,
               rule=frozen["fixed_rule"], null_samples=int(sum(pooled)),
               rates={frozen["name"]: _rates(runs), **others})
    print(f"fixed threshold from validation ({', '.join(scenes)}; q={q:g}, "
          f"{int(sum(pooled))} update-pass scores): {thr:.4f}")
    _print_rates(out["rates"], q)
    if rec is not None:
        rec.update(out)
    return out


def criterion(args, rec=None):
    """Test: the pre-registered criterion of the frozen config and every config's rates."""
    frozen = load_config(args.config)
    crit = frozen["criterion"]
    scenes = frozen["data"]["test_scenes"]
    runs = load_runs(args.results, frozen, scenes)
    lo, hi = crit["low"] * crit["nominal"], crit["high"] * crit["nominal"]
    per = {s: _rates({s: m})[s]["rate"] for s, m in runs.items()}
    inside = {s: (r is not None and lo <= r <= hi) for s, r in per.items()}
    n_in = sum(inside.values())
    passed = n_in >= crit["min_scenes"]
    sel = os.path.join(args.results, "val_selection.json")
    others = {}
    for name, c in _configs(os.path.dirname(os.path.abspath(args.config)),
                            frozen["ablations"]).items():
        others[name] = _rates(load_runs(args.results, c, scenes, selection=sel,
                                        allow_missing=True))
    out = dict(criterion=crit, test_scenes=scenes, rate_per_scene=per, inside=inside,
               scenes_inside=n_in, passed=passed,
               rates={frozen["name"]: _rates(runs), **others})
    _print_rates(out["rates"], float(frozen["change_quantile"]))
    print(f"CRITERION (pre-registered): {crit['text']}")
    print(f"  {n_in}/{len(scenes)} test scenes inside [{lo:.2g}, {hi:.2g}] -> "
          f"{'PASS' if passed else 'FAIL'}")
    if rec is not None:
        rec.update(out)
    return out


def _print_rates(rates, q):
    fmt = lambda r: "   n/a" if r is None or (isinstance(r, float) and math.isnan(r)) \
        else f"{r:.2e}"
    for name, rows in rates.items():
        for s, r in rows.items():
            bl = " ".join(f"{n} {fmt(v)}" for n, v in sorted(r["baselines"].items()))
            print(f"{name:>16} {s:>9}: rate {fmt(r['rate'])} ({r['fires']}/{r['tests']}; "
                  f"nominal {1 - q:.0e}) thr {r['threshold']:.4f} reset mass "
                  f"{100 * (r['reset_pixel_mass_share'] or 0):.2f}% {bl}")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("args", help="run_mapping.py arguments of a config, one per line")
    a.add_argument("config")
    a.add_argument("--selection", default=None, help="OUT/val_selection.json")
    for name in ("select", "criterion"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/static_fa.json", help="frozen config")
        p.add_argument("--results", required=True, help="OUT of replica_static_fa.sh")
        p.add_argument("--out", default="", help="record (default OUT/val_selection.json or "
                                                 "OUT/test_criterion.json)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "args":
        cfg = load_config(args.config)
        print("\n".join(run_args(cfg, args.selection)))
        return None
    out = args.out or os.path.join(args.results, "val_selection.json" if args.cmd == "select"
                                   else "test_criterion.json")
    from usplat.runlog import RunRecord
    cli = list(argv if argv is not None else sys.argv[1:])
    with RunRecord(out, config=vars(args), name=f"static_fa_{args.cmd}",
                   argv=[sys.argv[0]] + cli) as rec:
        rec.add_input(args.config)
        res = (select if args.cmd == "select" else criterion)(args, rec)
    print(f"wrote {out}")
    return res


if __name__ == "__main__":
    main()
