"""
audit_change_coverage.py — how much of a mapped field can the change test score? (CPU only)

The change score c of a Gaussian is defined once its support reaches tau_min views and each
bucket holds one view, counted in VIEW units: a view of splat mass m counts min(1, m/(a0 H W)),
and a0 H W = 7.68 px at the Replica accumulation resolution (320x240). The median Gaussian of
the Table I fields collects 1.6-4.2 px over ALL its mapping keyframes, so most Gaussians may
never be tested at all. This audit measures that on a saved field (the dict that
scripts/run_mapping.py saves; its "field" entry is DirectionalSemanticField.state_dict()):

  observed     Gaussians with pixel mass > 0 (pixel_mass = sum over the keyframes of m_v)
  defined      isfinite(c_last): a persistent score was defined at least once. c_last keeps
               the last one; calibrate_change_threshold clears it, so on a field saved after
               an update pass it covers the update pass only (reported as calibrated_at_save)
  shares       defined share of the observed Gaussians and of their pixel mass
  deciles      the observed Gaussians sorted by pixel mass, 10 equal-count groups (decile 0 =
               smallest): size, pixel-mass range and defined share of each
  view mass    W1 (view units) of the observed Gaussians and the share with W1 >= tau_min,
               a necessary condition for a defined score
  null         c_hist samples per history bucket (h = slow-bucket mass; edges 4, 6, 10, 20, 40
               views); a bucket with fewer than MIN_NULL samples borrows the nearest bucket
               that has enough (the rule of calibrate_change_threshold)
  thresholds   calibrate_change_threshold(q) on a deep copy (the audited field is not
               changed), q = 0.99 and 0.999, in the field's own calib_mode

PRE-REGISTERED decision (written 2026-10-10, before any audit run; constants below):
  PASS iff >= 50% of the observed pixel mass AND >= 50% of the observed Gaussians in the
  lower 5 pixel-mass deciles have a defined score. FAIL means the change test, as configured,
  never scores most of the small Gaussians: coverage, not the statistic, limits it.

    PYTHONPATH=. python3 scripts/audit_change_coverage.py \\
        --field /home/thanhnc/gauss_ws/data/uags_fields/office_0_lseg.pt \\
        --out /home/thanhnc/gauss_ws/data/uags_results/static_fa/audit_office_0.json
Several --field paths give one report per field. The JSON is a usplat.runlog record
(--plain-json: plain JSON) with metrics.fields[<path>] = the report and metrics.passed.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from usplat.vmf_field import DirectionalSemanticField                      # noqa: E402

# pre-registered decision (2026-10-10): see the module docstring
PASS_MASS_SHARE = 0.5          # defined share of the observed pixel mass
PASS_LOWER_SHARE = 0.5         # defined share of the observed Gaussians in the lower deciles
LOWER_DECILES = 5              # deciles 0..4 of pixel mass (the smaller half)
QUANTILES = (0.99, 0.999)
RULE = (f"PASS iff >= {PASS_MASS_SHARE:.0%} of the observed pixel mass AND >= "
        f"{PASS_LOWER_SHARE:.0%} of the observed Gaussians in the lower {LOWER_DECILES} "
        f"pixel-mass deciles have a defined change score (isfinite(c_last))")


def load_field(path):
    """DirectionalSemanticField on the CPU from a run_mapping.py checkpoint (or a bare
    state_dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["field"] if isinstance(ckpt, dict) and "field" in ckpt else ckpt
    return DirectionalSemanticField.from_state(state, device="cpu")


def bucket_labels(edges):
    """History-bucket names of torch.bucketize(ns, edges) (right-closed bins)."""
    e = [float(x) for x in edges]
    return ([f"h<={e[0]:g}"] + [f"{a:g}<h<={b:g}" for a, b in zip(e[:-1], e[1:])]
            + [f"h>{e[-1]:g}"])


def _q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    if x.numel() == 0:
        return {}
    v = torch.quantile(x.double(), torch.tensor(qs, dtype=torch.float64))
    return {f"q{int(round(q * 100)):02d}": float(t) for q, t in zip(qs, v)}


@torch.no_grad()
def audit(field, quantiles=QUANTILES):
    """The coverage report of one field (a dict; see the module docstring)."""
    cfg = field.cfg
    pm = field.pixel_mass.double()
    obs = pm > 0
    defined = torch.isfinite(field.c_last) & obs
    n_obs = int(obs.sum())
    mass_obs = float(pm[obs].sum())
    out = dict(num_gaussians=field.N, observed=n_obs,
               observed_share=n_obs / max(field.N, 1), observed_pixel_mass=mass_obs,
               defined=int(defined.sum()),
               calibrated_at_save=bool(field.thr_by_hist is not None
                                       or math.isfinite(cfg.change_thr)),
               config={k: getattr(cfg, k) for k in (
                   "change_test", "tau_min", "fast_min", "slow_min", "change_persist",
                   "fast_memory", "view_mass_frac", "evidence", "centre", "calib_mode",
                   "change_quantile")})
    out["config"]["change_weight"] = getattr(cfg, "change_weight", "view")
    if n_obs == 0:
        out.update(defined_share_gaussians=None, defined_share_pixel_mass=None, deciles=[],
                   lower_defined_share=None)
    else:
        out["defined_share_gaussians"] = float(defined[obs].double().mean())
        out["defined_share_pixel_mass"] = float(pm[defined].sum()) / mass_obs
        po, do = pm[obs], defined[obs]
        order = torch.argsort(po, stable=True)
        dec = torch.empty(n_obs, dtype=torch.long)
        dec[order] = torch.arange(n_obs) * 10 // n_obs     # equal-count groups by rank
        rows = []
        for k in range(10):
            sel = dec == k
            if not bool(sel.any()):
                continue
            rows.append(dict(decile=k, n=int(sel.sum()),
                             pixel_mass_min=float(po[sel].min()),
                             pixel_mass_max=float(po[sel].max()),
                             pixel_mass_median=float(po[sel].median()),
                             pixel_mass_share=float(po[sel].sum()) / mass_obs,
                             defined_share=float(do[sel].double().mean())))
        out["deciles"] = rows
        lower = dec < LOWER_DECILES
        out["lower_defined_share"] = float(do[lower].double().mean())
        out["pixel_mass_quantiles"] = _q(po)
        W1 = field.raw.W1[obs]
        out["view_mass_quantiles"] = _q(W1)
        out["view_mass_ge_tau_share"] = float((W1 >= cfg.tau_min).double().mean())
    # the mapping-pass null per history bucket, and which buckets borrow
    counts = field.c_hist.sum(1)
    labels = bucket_labels(field.HIST_EDGES)
    good = [b for b in range(len(counts)) if counts[b] >= field.MIN_NULL]
    null = []
    for b in range(len(counts)):
        own = b in good
        src = None if own else (labels[min(good, key=lambda g: abs(g - b))] if good
                                else "pooled")
        null.append(dict(bucket=labels[b], samples=int(counts[b]), own=own, borrows_from=src))
    out["null"] = null
    out["null_samples"] = int(counts.sum())
    out["min_null"] = field.MIN_NULL
    if bool(field.n_tests.any()):                         # fields mapped with the test record
        out["tests_by_hist"] = field.n_tests.tolist()
        out["fires_by_hist"] = field.n_fires.tolist()
    thr = {}
    for q in quantiles:
        g = copy.deepcopy(field)                          # calibration clears c_last / c_max
        try:
            t = g.calibrate_change_threshold(q)
            thr[f"{q:g}"] = dict(change_thr=t, calib_mode=g.cfg.calib_mode,
                                 by_hist=(None if g.thr_by_hist is None
                                          else [float(x) for x in g.thr_by_hist]))
        except RuntimeError as e:                         # no defined score was recorded
            thr[f"{q:g}"] = dict(error=str(e))
        del g
    out["thresholds"] = thr
    mass, low = out["defined_share_pixel_mass"], out["lower_defined_share"]
    out["decision"] = dict(
        rule=RULE, defined_share_pixel_mass=mass, lower_defined_share=low,
        passed=bool(mass is not None and low is not None
                    and mass >= PASS_MASS_SHARE and low >= PASS_LOWER_SHARE))
    return out


def format_report(path, r):
    pct = lambda x: "  n/a" if x is None else f"{100 * x:5.1f}%"
    lines = [f"== {path}",
             f"observed {r['observed']}/{r['num_gaussians']} Gaussians "
             f"({pct(r['observed_share'])}); defined score: {pct(r['defined_share_gaussians'])} "
             f"of observed Gaussians, {pct(r['defined_share_pixel_mass'])} of their pixel mass"
             + ("  [field calibrated at save: c_last covers the update pass only]"
                if r["calibrated_at_save"] else "")]
    if r.get("view_mass_quantiles"):
        qs = lambda d: " ".join(f"{k} {v:.3g}" for k, v in d.items())
        lines.append(f"pixel mass {qs(r['pixel_mass_quantiles'])}; view mass W1 "
                     f"{qs(r['view_mass_quantiles'])}; W1 >= tau_min "
                     f"({r['config']['tau_min']:g}): {pct(r['view_mass_ge_tau_share'])}")
    for d in r.get("deciles", []):
        lines.append(f"  decile {d['decile']}: n {d['n']:>8}  pixel mass "
                     f"[{d['pixel_mass_min']:.3g}, {d['pixel_mass_max']:.3g}] "
                     f"(share {pct(d['pixel_mass_share'])})  defined {pct(d['defined_share'])}")
    lines.append("null (c_hist) per history bucket: " + ", ".join(
        f"{n['bucket']} {n['samples']}" + ("" if n["own"] else f" -> {n['borrows_from']}")
        for n in r["null"]))
    for q, t in r["thresholds"].items():
        if "error" in t:
            lines.append(f"threshold q={q}: {t['error']}")
        else:
            bh = "" if t["by_hist"] is None else " by history " + ", ".join(
                f"{x:.4f}" for x in t["by_hist"])
            lines.append(f"threshold q={q} ({t['calib_mode']}): pooled {t['change_thr']:.4f}{bh}")
    dcs = r["decision"]
    lines.append(f"DECISION (pre-registered): {'PASS' if dcs['passed'] else 'FAIL'}  "
                 f"(defined pixel mass {pct(dcs['defined_share_pixel_mass'])}, lower "
                 f"{LOWER_DECILES} deciles {pct(dcs['lower_defined_share'])}; {RULE})")
    return "\n".join(lines)


def run(args, rec=None):
    reports = {}
    for path in args.field:
        if rec is not None:
            rec.add_input(path, hash=not args.no_hash)
        field = load_field(path)
        reports[path] = audit(field, tuple(float(q) for q in args.quantiles.split(",")))
        del field
        print(format_report(path, reports[path]), flush=True)
    out = dict(fields=reports, rule=RULE,
               passed={p: r["decision"]["passed"] for p, r in reports.items()})
    if rec is not None:
        rec.update(out)
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--field", nargs="+", required=True,
                    help="field checkpoint(s) saved by run_mapping.py")
    ap.add_argument("--quantiles", default=",".join(f"{q:g}" for q in QUANTILES),
                    help="q of calibrate_change_threshold (comma-separated)")
    ap.add_argument("--no-hash", action="store_true",
                    help="list the field files in the record without their sha256")
    ap.add_argument("--plain-json", action="store_true",
                    help="write plain JSON instead of a usplat.runlog record")
    ap.add_argument("--out", default="", help="report JSON (default: print only)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.out:
        return run(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.plain_json:
        out = run(args)
        with open(args.out, "w") as f:
            json.dump(dict(out, config=vars(args)), f, indent=1)
    else:
        from usplat.runlog import RunRecord
        cli = list(argv if argv is not None else sys.argv[1:])
        with RunRecord(args.out, config=vars(args), name="audit_change_coverage",
                       argv=[sys.argv[0]] + cli) as rec:
            out = run(args, rec)
    print(f"wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
