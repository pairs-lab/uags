#!/usr/bin/env bash
# Table I v2 on ScanNet v2: a fresh held-out set (real data; no choice of the method, its
# templates or its settings was made on it). Fixed 2026-10-10, before any ScanNet run:
#   scenes   the 10 ScanNet scenes of OpenGaussian (Wu et al., NeurIPS 2024,
#            scripts/eval_scannet.py scene_list, read 2026-10-10; the concurrent NormLift
#            reports ScanNet with that protocol, 10 scenes and 19 classes, per
#            docs/reports/2026-09-28-paper-completion-plan - its scene ids were not re-checked);
#            validation = the two lowest scene ids
#            (configs/scannet_val.json: scene0000_00, scene0062_00), test = the other eight
#            (configs/scannet_test.json); the configs are the only scene lists used here
#   classes  OpenGaussian's 19 (ScanNet-20 without "other furniture"), as NYU40 ids from
#            label-filt through scannetv2-labels.combined.tsv (configs/vocab_scannet.json);
#            queries = the classes present in each scene's evaluation frames (as on Replica)
#   scoring  2D, per pixel, on held-out frames. Same scenes and classes as OpenGaussian, but
#            OpenGaussian and NormLift score the 3D points of *_vh_clean_2.labels.ply with all
#            19 classes queried: their published numbers cannot be put in the same table
#   method   LSeg, the Replica templates, PCA-64 from the first 20 mapping frames, mapping /
#            evaluation downscale 2, tau_cov 0.5, 3DGS fit settings: those of Replica Table I
#   v2 rules configs/scannet_v2_prereg.json: the Table I v2 rules of configs/tab1_v2_prereg.json
#            (candidates, lowest pooled validation NLL, tie rule) naming the ScanNet scene files,
#            applied by scripts/make_tables.py (selection, choose_v2)
# Protocol within one scan (frame numbers of the complete SensReader export):
#   mapping frames     0, 10, 20, ... (every 10th): the 3DGS fit (scripts/fit_gs.py, which
#                      holds out every 8th of them for its PSNR) and the semantic field
#                      (run_mapping.py, with its near-duplicate pose filter)
#   evaluation frames  5, 25, 45, ... (every 20th, offset 5: configs/scannet_*.json) - never
#                      fitted, never mapped; each is 5 frames (1/6 s) from a mapping frame
#   lost tracking      frames with a non-finite (-inf) pose are skipped by every step
#   CSS vocabulary     the classes present in the evaluation frames, as on Replica:
#                      run_mapping.py reads them from $PREP/<scene>/eval_s20o5 (symlinks to the
#                      evaluation frames with a finite pose, usplat.datasets.scannet_frame_view)
#                      with --css-eval-stride 1
# Data root /home/thanhnc/gauss_ws/data/scannet (ScanNet's download-scannet.py -o <root>
# --id <scene> with --type .sens and --type _2d-label-filt.zip, and --label_map), layout:
#   scannetv2-labels.combined.tsv                 (top level of the ScanNet download)
#   scans/<scene>/color/<i>.jpg depth/<i>.png pose/<i>.txt intrinsic/intrinsic_color.txt
#                 intrinsic/intrinsic_depth.txt   (ScanNet SensReader/python, Python 2.7 only,
#                                                  every frame:
#       python2 reader.py --filename scans/<scene>/<scene>.sens --output_path scans/<scene> \
#           --export_color_images --export_depth_images --export_poses --export_intrinsics)
#   scans/<scene>/label-filt/<i>.png              (cd scans/<scene> && unzip <scene>_2d-label-filt.zip)
#   uags_f2/<scene>/                              written by this script (CPU):
#       usplat.datasets.prepare_scannet - colour / 2 (648x484, as usplat.datasets.downscale),
#       intrinsics scaled, depth/pose/label-filt/tsv symlinked to scans/<scene>;
#       eval_s20o5/ - the evaluation frames only (usplat.datasets.scannet_frame_view)
# Memory: run_mapping.py keeps the fp32 LSeg tokens of every mapping frame on the CPU
# (~10 MB per frame at 648x484): scene0000_00 (5578 frames, 558 mapping frames) needs ~6 GB.
# Stages:
#   STAGE=prep   : prepare the 10 scans only (CPU; val and test prepare their own scans)
#   STAGE=val    : prepare + fit + map + evaluate the validation scans; fit one temperature
#                  per row on them -> $OUT/eval_val.json, its settings -> $OUT/protocol_val.json;
#                  then the validation selection -> $OUT/selection_val.json: the row choices of
#                  scripts/make_tables.py (css, p3, legs), with --v2-rows also the Table I v2
#                  choices (make_tables.selection under configs/scannet_v2_prereg.json: the
#                  sha256 of the prereg and of eval_val.json, every candidate's value), the
#                  protocol settings and the sha256 of eval_val.json. The evaluation is refused
#                  with uncommitted changes to tracked files (the test stage compares the code
#                  with the commit its record names).
#   STAGE=select : redo only the selection from $OUT/eval_val.json (e.g. after a refusal), with
#                  the settings of $OUT/protocol_val.json: refused when the current settings
#                  differ or eval_val.json was not made with them. DRY_RUN=1 prints it only.
#   STAGE=test   : prepare + fit + map the test scans and evaluate them ONCE with the frozen
#                  betas. Refused on a dirty tree; when $OUT/eval_test.json holds a finished
#                  run (no override; a crashed run, status != ok, may be rerun with the same
#                  selection); without a finished selection of eval_val.json (sha256) whose
#                  choices the rules still give; when the protocol settings (downscale,
#                  tau_cov, iters, strides, MAP_EXTRA, the configs' sha256) differ from the
#                  selection's; when scripts/ usplat/ configs/ differ from the commit of the
#                  validation evaluation (which must have had no uncommitted changes); with
#                  --v2-rows, without the v2 selection or when configs/scannet_v2_prereg.json
#                  changed after it. Each attempt is appended to $OUT/look_test.json before the
#                  evaluation (every attempt: the same selection). val and select are refused
#                  once the test evaluation has started ($OUT/look_test.json, $OUT/eval_test.json
#                  in any status, or its .partial.json exists): the selection is frozen.
#   STAGE=table  : Table I from the finished test record -> $OUT/table1_scannet.tex
#                  (make_tables.py tab1 --val eval_val.json, plus --v2 --prereg
#                  configs/scannet_v2_prereg.json when the selection has the v2 rows); refused
#                  unless the selection still matches eval_val.json, the prereg and the rules
#   EVAL=0       : prepare + maps and fields only
# Options:
#   V2_ROWS=auto|1|0  (val) pass --v2-rows to eval_seg.py when it has the flag (auto), always
#                     (1: refused when it has not), never (0); with it, refused before any GPU
#                     step unless make_tables.py has the v2 selection and the prereg is
#                     committed. The test stage passes it iff the validation evaluation did
#                     (its recorded argv)
#   MAP_EXTRA=<args>  extra run_mapping.py arguments; part of the protocol recorded by val, so
#                     select and test refuse another value. An existing field is reused only
#                     when it was mapped with the same MAP_EXTRA ($FIELDS/<scene>_lseg.pt.map_extra.json,
#                     written after its mapping). --css-full-vocab is refused: css_full is a
#                     validation-only control (usplat/baselines.py CSS_FULL_RULE) and this
#                     prereg has no css_full choice, so a test number would have no recorded
#                     validation choice
#   DRY_RUN=1         print the GPU and data steps instead of running them; the checks run
#   DISK_FACTOR=<f>   each field mapping is refused unless f (1.5) x its size (N of the map x
#                     1.6 KB, scripts/disk_guard.py) is free in $FIELDS; 0: no check. Budget:
#                     up to ~2.6 GB per validation scan and ~2.1 GB per test scan (field, CSS,
#                     distilled baselines at the 1M-Gaussian cap), plus the prepared frames
#   EVAL_NEED=<MiB>   evaluation GPU cap: 7000, or 10000 with --v2-rows (its frozen P1 and P4
#                     buffers, as scripts/replica_tab1_v2.sh)
# Every GPU step goes through scripts/gpu_gate.sh (never beside a USGLoc job, poll every 5 s).
# Existing prepared scans, maps and fields are reused; eval_seg.py --betas refuses test fields
# built with other mapping code, options or protocol than the validation fields.
# Logs: $OUT/<step>_<scene>.<timestamp>.log (never overwritten). Results: $OUT/eval_<stage>.json
# (+ .partial.json after every scene), $OUT/protocol_val.json, $OUT/selection_val.json,
# $OUT/look_test.json, $OUT/table1_scannet.tex.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_USGLOC_EXTRA_MIB=${UAGS_GPU_USGLOC_EXTRA_MIB:-99999}   # never beside USGLoc
export UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-5}
DATA=/home/thanhnc/gauss_ws/data/scannet
PREP=$DATA/uags_f2
MAPS=/home/thanhnc/gauss_ws/data/uags_maps/scannet
FIELDS=/home/thanhnc/gauss_ws/data/uags_fields/scannet
OUT=${OUT:-/home/thanhnc/gauss_ws/data/uags_results/scannet_tab1}
VOCAB=configs/vocab_scannet.json
PREREG=configs/scannet_v2_prereg.json   # Table I v2 rules on ScanNet
STAGE=${STAGE:-val}
EVAL=${EVAL:-1}
DRY_RUN=${DRY_RUN:-0}
ITERS=${ITERS:-30000}
DOWNSCALE=${DOWNSCALE:-2}
TAU_COV=${TAU_COV:-0.5}          # coverage rule of Replica Table I
V2_ROWS=${V2_ROWS:-auto}
MAP_EXTRA=${MAP_EXTRA:-}
DISK_FACTOR=${DISK_FACTOR:-1.5}
case " $MAP_EXTRA " in         # argparse also takes a unique prefix (--css-full, --css-f)
  *" --css-f"*) echo "refused: MAP_EXTRA $MAP_EXTRA: css_full is a validation-only control (usplat/baselines.py CSS_FULL_RULE); configs/scannet_v2_prereg.json registers no css_full choice"; exit 1 ;;
esac
FACTOR=2                         # prepared colour = native / FACTOR
MAP_STRIDE=10                    # mapping frames 0, 10, 20, ...
EVAL_STRIDE=20 EVAL_OFFSET=5     # evaluation frames 5, 25, 45, ... (configs/scannet_*.json)
VIEW=eval_s${EVAL_STRIDE}o${EVAL_OFFSET}   # $PREP/<scene>/$VIEW: evaluation frames (CSS)
# GPU memory caps (MiB), as replica_tab1.sh; EVAL_NEED once --v2-rows is known
MAP_NEED=${MAP_NEED:-9000}
DISTILL_NEED=${DISTILL_NEED:-7000}
VAL=$OUT/eval_val.json PROTO=$OUT/protocol_val.json SEL=$OUT/selection_val.json
TEST=$OUT/eval_test.json LOOK=$OUT/look_test.json
TS=$(date +%Y%m%d-%H%M%S)
case "$STAGE" in
  prep)       SPLITS="val test" ;;
  val|select) SPLITS=val ;;
  test)       SPLITS=test ;;
  table)      SPLITS="" ;;
  *)          echo "STAGE must be prep, val, select, test or table (got '$STAGE')"; exit 2 ;;
esac
mkdir -p "$OUT"
[ "$DRY_RUN" = 1 ] || [ -z "$SPLITS" ] || [ "$STAGE" = select ] || mkdir -p "$PREP" "$MAPS" "$FIELDS"

scenes() {    # names in configs/scannet_<split>.json; refused unless its paths and evaluation
              # frames are this script's
  python3 - "configs/scannet_$1.json" "$PREP" "$MAPS" "$FIELDS" "$EVAL_STRIDE" "$EVAL_OFFSET" <<'PY'
import json, sys
path, prep, maps, fields, stride, offset = sys.argv[1:]
sc = json.load(open(path))
bad = [f"{s['name']}.{k}: {s.get(k)!r} != {v!r}" for s in sc
       for k, v in dict(data=f"{prep}/{s['name']}", ply=f"{maps}/{s['name']}.pt",
                        field=f"{fields}/{s['name']}_lseg.pt", dataset="scannet",
                        stride=int(stride), offset=int(offset)).items()
       if s.get(k) != v]
if bad:
    sys.exit(f"refused: {path} does not match scripts/scannet_tab1.sh: {bad}")
print(" ".join(s["name"] for s in sc))
PY
}
protocol() {  # the settings the test stage must share with the validation selection (JSON)
  python3 - "$DOWNSCALE" "$TAU_COV" "$ITERS" "$FACTOR" "$MAP_STRIDE" "$EVAL_STRIDE" \
    "$EVAL_OFFSET" "$VIEW" "$MAP_EXTRA" "$VOCAB" "$PREREG" <<'PY'
import hashlib, json, sys
ds, tau, iters, factor, ms, es, eo, view, extra, vocab, prereg = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
print(json.dumps(dict(downscale=int(ds), tau_cov=float(tau), iters=int(iters),
                      prep_factor=int(factor), map_stride=int(ms), eval_stride=int(es),
                      eval_offset=int(eo), css_frames=view, map_extra=extra,
                      configs={p: sha(p) for p in (vocab, "configs/scannet_val.json",
                                                   "configs/scannet_test.json", prereg)}),
                 sort_keys=True))
PY
}
has_v2() {    # eval_seg.py has --v2-rows (captured first: grep -q under pipefail can SIGPIPE)
  local help
  help=$(python3 scripts/eval_seg.py --help 2>/dev/null || true)
  [[ $help == *--v2-rows* ]]
}
v2_ready() {  # the v2 rules can be applied: $PREREG committed, make_tables.py's v2 selection
  if ! git ls-files --error-unmatch "$PREREG" > /dev/null 2>&1 || \
     [ -n "$(git status --porcelain -- "$PREREG")" ]; then
    echo "refused: $PREREG is not committed (the v2 rules must be fixed before any run)"; return 1
  fi
  python3 -c "import sys; sys.path.insert(0, 'scripts'); import make_tables as m; sys.exit(0 if all(hasattr(m, k) for k in ('selection', 'choose_v2', 'load_prereg', 'V2_KEYS')) else 1)" || {
    echo "refused: scripts/make_tables.py has no Table I v2 selection (selection, choose_v2): the v2 rows could not be selected"; return 1; }
}
test_done() { # $TEST is a finished run record
  [ -f "$TEST" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status', 'ok') == 'ok' else 1)" "$TEST"
}
test_started() {  # the test evaluation has started (even if it crashed: it may have scored scenes)
  [ -e "$LOOK" ] || [ -e "$TEST" ] || [ -e "$TEST.partial.json" ]
}
step() {      # step <log> <command...>: run it, output to <log>; on failure show the tail, stop
  local L=$1; shift
  if [ "$DRY_RUN" = 1 ]; then echo "[dry-run] $*"; return 0; fi
  "$@" > "$L" 2>&1 || { tail -20 "$L"; exit 1; }
}
record_protocol() {   # STAGE=val, after its evaluation: the settings $VAL was made with
  python3 - "$VAL" "$PROTO" "$PROTOCOL" <<'PY'
import hashlib, json, os, sys
val, out, proto = sys.argv[1:]
rec = dict(status="ok", eval_val=os.path.abspath(val),
           eval_val_sha256=hashlib.sha256(open(val, "rb").read()).hexdigest(),
           protocol=json.loads(proto))
with open(out + ".tmp", "w") as f:
    json.dump(rec, f, indent=1)
os.replace(out + ".tmp", out)
PY
}
select_val() {   # $VAL (made with the settings of $PROTO) -> $SEL
  python3 - "$VAL" "$SEL" "$PROTO" "$PROTOCOL" "$PREREG" "$VOCAB" "$DRY_RUN" <<'PY'
import hashlib, json, os, subprocess, sys
sys.path.insert(0, "scripts")
import make_tables as mt
val, out, proto_path, proto, prereg, vocab, dry = sys.argv[1:]
proto = json.loads(proto)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
body = mt.body(val)                         # refuses an unfinished evaluation
d = json.load(open(val))
if not body.get("betas"):
    sys.exit(f"refused: {val} holds no fitted betas")
# the settings come from the validation run (recorded after its evaluation), never from a
# later environment: the test stage holds the selection's settings to its own
try:
    pv = json.load(open(proto_path))
except (OSError, ValueError) as e:
    sys.exit(f"refused: no settings record of the validation run ({e}); rerun STAGE=val")
if pv.get("eval_val_sha256") != sha(val):
    sys.exit(f"refused: {proto_path} was recorded for another {val}; rerun STAGE=val")
rec_p = pv.get("protocol") or {}
diff = {k: (rec_p.get(k), proto.get(k)) for k in sorted(set(rec_p) | set(proto))
        if rec_p.get(k) != proto.get(k)}
if diff:
    sys.exit(f"refused: settings differ from the validation run's (recorded, now): {diff}")
cfg, fit = d.get("config") or {}, body.get("beta_fit_settings") or {}
want = dict(dataset="scannet", vocab=vocab, stride=proto["eval_stride"],
            downscale=proto["downscale"], tau_cov=proto["tau_cov"],
            val_scenes="configs/scannet_val.json", scenes="configs/scannet_val.json")
bad = {k: (cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v}
if fit.get("tau_cov") != proto["tau_cov"]:
    bad["beta_fit_settings.tau_cov"] = (fit.get("tau_cov"), proto["tau_cov"])
if bad:
    sys.exit(f"refused: {val} was not made with these settings (recorded, expected): {bad}")
argv = d.get("argv") or []
rec = dict(status="ok", eval_val=os.path.abspath(val), eval_val_sha256=sha(val),
           git=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip(),
           eval_val_git=d.get("git"), protocol=proto, v2_rows="--v2-rows" in argv,
           eval_argv=argv, betas_rows=sorted(body["betas"]),
           field_identities=fit.get("field_identities"),
           choices={k.lstrip("@"): mt.choose(body["summary"], k) for k in mt.CHOICES})
if rec["v2_rows"]:                          # the v2 choices, by the prereg's rules
    if not all(hasattr(mt, k) for k in ("selection", "choose_v2", "load_prereg", "V2_KEYS")):
        sys.exit(f"refused: {val} has the v2 rows but scripts/make_tables.py has no Table I "
                 f"v2 selection (selection, choose_v2)")
    p = mt.load_prereg(prereg)
    rows = {r for c in p["choices"].values() for r in c["rows"] if not r.startswith("@")}
    miss = sorted((rows | set(p.get("ablations", []))) - set(body["betas"]))
    if miss:
        sys.exit(f"refused: {val} has no fitted beta for the v2 rows {miss}")
    rec["v2"] = mt.selection(val, prereg)   # prereg / validation sha256, every value, chosen
msg = (f"{rec['choices']}, v2 rows {rec['v2_rows']}"
       + (f", v2 {rec['v2']['chosen']}" if "v2" in rec else ""))
if dry == "1":
    print(f"[dry-run] validation selection (not written to {out}): {msg}")
    sys.exit(0)
with open(out + ".tmp", "w") as f:
    json.dump(rec, f, indent=1)
os.replace(out + ".tmp", out)
print(f"validation selection -> {out}: {msg}")
PY
}
check_selection() {   # check_selection test|table: refuses unless a finished selection matches
                      # $VAL and the rules (test: also the protocol and the code)
  python3 - "$1" "$SEL" "$VAL" "$PROTOCOL" "$PREREG" "$TEST" <<'PY'
import hashlib, json, os, subprocess, sys
mode, sel_path, val, proto, prereg, test = sys.argv[1:]
proto = json.loads(proto)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
try:
    sel = json.load(open(sel_path))
except (OSError, ValueError) as e:
    sys.exit(f"refused: no finished validation selection ({e}); run STAGE=val first")
if sel.get("status") != "ok":
    sys.exit(f"refused: {sel_path} is not finished (status {sel.get('status')!r})")
try:
    vsha = sha(val)
except OSError as e:
    sys.exit(f"refused: {e}")
if vsha != sel.get("eval_val_sha256"):
    sys.exit(f"refused: {val} changed after the selection (sha256 {vsha[:12]} != "
             f"{str(sel.get('eval_val_sha256'))[:12]}); rerun STAGE=select")
sys.path.insert(0, "scripts")
import make_tables as mt
b = mt.body(val)
if mode == "test":
    sp = sel.get("protocol") or {}
    diff = {k: (sp.get(k), proto.get(k)) for k in sorted(set(sp) | set(proto))
            if sp.get(k) != proto.get(k)}
    if diff:
        sys.exit(f"refused: protocol differs from the validation selection (selection, now): {diff}")
    # the evaluation code: the test numbers come from the code that fit the betas
    g = json.load(open(val)).get("git") or {}
    if g.get("error") or not g.get("commit"):
        sys.exit(f"refused: {val} records no git commit ({g.get('error', 'no git state')})")
    if g.get("dirty_tracked", g.get("dirty", True)):
        sys.exit(f"refused: {val} was made with uncommitted changes; rerun STAGE=val on a "
                 f"committed tree")
    r = subprocess.run(["git", "diff", "--quiet", g["commit"], "--", "scripts", "usplat",
                        "configs", ":(exclude,glob)**/*.pyc"], capture_output=True, text=True)
    if r.returncode == 1:
        sys.exit(f"refused: scripts/ usplat/ configs/ differ from the commit "
                 f"{g['commit'][:12]} of the validation evaluation; rerun STAGE=val")
    if r.returncode:
        sys.exit(f"refused: cannot compare the code with {g['commit'][:12]}: {r.stderr.strip()}")
else:                                       # table: the test record of these betas
    t = json.load(open(test))
    tb = t["metrics"] if isinstance(t.get("metrics"), dict) else t
    if tb.get("betas_path") != os.path.abspath(val):
        sys.exit(f"refused: {test} was evaluated with the betas of {tb.get('betas_path')}, "
                 f"not {os.path.abspath(val)}")
again = {k.lstrip("@"): mt.choose(b["summary"], k) for k in mt.CHOICES}
if again != sel.get("choices"):
    sys.exit(f"refused: the rules give {again}, the selection holds {sel.get('choices')}")
if sel.get("v2_rows"):
    v2 = sel.get("v2")
    if not v2:
        sys.exit(f"refused: the validation evaluation has the v2 rows but {sel_path} holds no "
                 f"v2 selection; rerun STAGE=select")
    if not all(hasattr(mt, k) for k in ("choose_v2", "load_prereg", "V2_KEYS")):
        sys.exit("refused: scripts/make_tables.py has no Table I v2 selection (choose_v2)")
    if v2["prereg"]["sha256"] != sha(prereg):
        sys.exit(f"refused: {prereg} changed after the selection (sha256 {sha(prereg)[:12]} != "
                 f"{v2['prereg']['sha256'][:12]})")
    if v2["validation"]["sha256"] != vsha:
        sys.exit(f"refused: the v2 selection was made on another {val}")
    p = mt.load_prereg(prereg)
    again = {k: mt.choose_v2(b, k, p)[0] for k in mt.V2_KEYS}
    if again != v2["chosen"]:
        sys.exit(f"refused: the v2 rules give {again}, the selection holds {v2['chosen']}")
print(f"validation selection {sel_path}: git {sel.get('git', '')[:12]}, {len(sel['betas_rows'])} "
      f"rows with betas, choices {sel['choices']}, v2 rows {sel['v2_rows']}"
      + (f", v2 {sel['v2']['chosen']}" if sel.get("v2") else ""))
PY
}
look_test() {   # look_test check|add: every test attempt evaluates with one selection
  python3 - "$1" "$SEL" "$LOOK" "$TS" "${L:-}" <<'PY'
import hashlib, json, os, subprocess, sys
mode, sel_path, out, ts, log = sys.argv[1:]
sha = hashlib.sha256(open(sel_path, "rb").read()).hexdigest()
rec = json.load(open(out)) if os.path.exists(out) else None
if rec is not None and rec.get("selection_sha256") != sha:
    sys.exit(f"refused: the test stage started before with another selection ({out}: "
             f"{len(rec.get('attempts') or [])} attempt(s), selection sha256 "
             f"{str(rec.get('selection_sha256'))[:12]} != {sha[:12]}); the selection is final")
if mode == "add":
    rec = rec or dict(selection=os.path.abspath(sel_path), selection_sha256=sha, attempts=[])
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    rec["attempts"].append(dict(started=ts, log=log, git=git.stdout.strip()))
    with open(out + ".tmp", "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(out + ".tmp", out)
    print(f"look record {out}: attempt {len(rec['attempts'])}")
PY
}
sel_get() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$SEL" "$1"; }

SCENES=""
for sp in $SPLITS; do SCENES="$SCENES $(scenes "$sp")"; done
PROTOCOL=$(protocol)
DIRTY=$(git status --porcelain --untracked-files=no -- scripts usplat configs | wc -l)
echo "[$STAGE] $TS git $(git rev-parse --short HEAD) dirty: $DIRTY files; scenes:$SCENES"
if [[ $STAGE == val || $STAGE == select ]] && test_started; then
  echo "refused: the test evaluation has started ($LOOK, $TEST or $TEST.partial.json exists); the validation selection is frozen"; exit 1
fi
if [ "$STAGE" = select ]; then select_val; exit 0; fi
if [ "$STAGE" = table ]; then
  test_done || { echo "refused: $TEST holds no finished test run"; exit 1; }
  check_selection table
  TAB=()
  [ "$(sel_get v2_rows)" = True ] && TAB=(--v2 --prereg "$PREREG")
  if [ "$DRY_RUN" = 1 ]; then
    python3 scripts/make_tables.py tab1 "$TEST" --val "$VAL" ${TAB[@]+"${TAB[@]}"}
    echo "done (dry run): table (not written)"; exit 0
  fi
  python3 scripts/make_tables.py tab1 "$TEST" --val "$VAL" ${TAB[@]+"${TAB[@]}"} \
    > "$OUT/table1_scannet.tex.tmp"
  mv "$OUT/table1_scannet.tex.tmp" "$OUT/table1_scannet.tex"
  cat "$OUT/table1_scannet.tex"
  echo "done: table -> $OUT/table1_scannet.tex"; exit 0
fi
V2=0
if [ "$STAGE" = val ] && [ "$EVAL" = 1 ]; then
  # the record must name a commit: the test stage compares scripts/ usplat/ configs/ with it
  if ! python3 -c "import sys; from usplat.runlog import git_state; g = git_state('.'); sys.exit(1 if g.get('error') or g['dirty_tracked'] else 0)"; then
    git status --short --untracked-files=no
    echo "refused: uncommitted changes to tracked files outside docs/ papers/ outputs/ (the validation evaluation must name a commit)"; exit 1
  fi
  case "$V2_ROWS" in
    0)    V2=0 ;;
    1)    has_v2 || { echo "refused: V2_ROWS=1 but scripts/eval_seg.py has no --v2-rows"; exit 1; }; V2=1 ;;
    auto) if has_v2; then V2=1; fi ;;
    *)    echo "V2_ROWS must be auto, 1 or 0 (got '$V2_ROWS')"; exit 2 ;;
  esac
  if [ "$V2" = 1 ]; then v2_ready || exit 1; fi
fi
if [ "$STAGE" = test ]; then
  if [ "$DIRTY" != 0 ]; then
    echo "refused: uncommitted changes in scripts/ usplat/ configs/ (the test run must name a commit)"; exit 1
  fi
  if [ "$EVAL" = 1 ]; then
    if test_done; then
      echo "refused: $TEST holds a finished test run (the test split runs once; no override)"; exit 1
    fi
    check_selection test
    look_test check
    [ "$(sel_get v2_rows)" = True ] && V2=1
    if [ "$V2" = 1 ] && ! has_v2; then
      echo "refused: the validation evaluation used --v2-rows, which scripts/eval_seg.py does not have"; exit 1
    fi
  fi
fi
if [ "$V2" = 1 ]; then EVAL_NEED=${EVAL_NEED:-10000}; else EVAL_NEED=${EVAL_NEED:-7000}; fi
echo "protocol: $PROTOCOL"
# trained baselines: validation trains LEGaussians for each lambda_u of usplat.distill
# LEGS_LAMBDAS; the test stage trains only the lambda_u of the validation selection
DISTILL_JOBS=""
if [ "$STAGE" = val ]; then
  DISTILL_JOBS="f3dgs $(python3 -c "from usplat.distill import LEGS_LAMBDAS as L; print(' '.join(f'legs:{l:g}' for l in L))")"
elif [ "$STAGE" = test ] && [ "$EVAL" = 1 ]; then
  chosen=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['choices']['legs'])" "$SEL")
  DISTILL_JOBS="f3dgs legs:${chosen#legs_l}"
  echo "LEGaussians lambda_u chosen on validation: ${chosen#legs_l} ($chosen)"
fi

for s in $SCENES; do
  if [ "$DRY_RUN" = 1 ]; then
    echo "[dry-run] prepare_scannet $DATA/scans/$s -> $PREP/$s (colour / $FACTOR); scannet_frame_view -> $PREP/$s/$VIEW (stride $EVAL_STRIDE, offset $EVAL_OFFSET)"
  else
    [ -d "$DATA/scans/$s/label-filt" ] || { echo "$DATA/scans/$s/label-filt missing (unzip ${s}_2d-label-filt.zip there)"; exit 1; }
    python3 -c 'import sys; from usplat.datasets import prepare_scannet as p; r = p(sys.argv[1], sys.argv[2], factor=int(sys.argv[3])); print(sys.argv[2], r["n_frames"], "frames", r["native_hw"], "->", r["hw"])' \
      "$DATA/scans/$s" "$PREP/$s" "$FACTOR"
    python3 -c 'import sys; from usplat.datasets import scannet_frame_view as v; r = v(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])); print(sys.argv[2], len(r["frames"]), "evaluation frames,", len(r["lost"]), "lost")' \
      "$PREP/$s" "$PREP/$s/$VIEW" "$EVAL_STRIDE" "$EVAL_OFFSET"
  fi
  [ "$STAGE" = prep ] && continue
  if [ ! -f "$MAPS/$s.pt" ]; then         # --max-res 0: fit at the prepared resolution
    L="$OUT/fit_${s}.$TS.log"
    step "$L" scripts/gpu_gate.sh 3000 python3 scripts/fit_gs.py --dataset scannet \
      --data "$PREP/$s" --stride "$MAP_STRIDE" --iters "$ITERS" --max-gaussians 1000000 \
      --max-res 0 --out "$MAPS/$s.pt"
    [ "$DRY_RUN" = 1 ] || { grep -E "gpu_gate|PSNR|psnr|Gaussians|dropped|saved" "$L" | tail -5 || true; }
  fi
  if [ -f "$FIELDS/${s}_lseg.pt" ]; then   # reused only if mapped with this MAP_EXTRA
    python3 - "$FIELDS/${s}_lseg.pt.map_extra.json" "$MAP_EXTRA" <<'PY' || exit 1
import json, os, sys
p, want = sys.argv[1], sys.argv[2]
if not os.path.exists(p):
    sys.exit(f"refused: no {p}: the MAP_EXTRA of the existing field is unknown (move the field "
             f"aside to map it again)")
got = json.load(open(p)).get("map_extra")
if got != want:
    sys.exit(f"refused: the existing field was mapped with MAP_EXTRA={got!r}, now {want!r} "
             f"(move it aside to map it again, or use its MAP_EXTRA)")
PY
  else
    [ "$DRY_RUN" = 1 ] || python3 scripts/disk_guard.py "$FIELDS" "$MAPS/$s.pt" 1600 \
      --factor "$DISK_FACTOR" || exit 1
    L="$OUT/map_${s}.$TS.log"
    # shellcheck disable=SC2086          # MAP_EXTRA: word-split on purpose
    step "$L" scripts/gpu_gate.sh "$MAP_NEED" python3 scripts/run_mapping.py --backend gsplat \
      --dataset scannet --data "$PREP/$s" --stride "$MAP_STRIDE" --downscale "$DOWNSCALE" \
      --ply "$MAPS/$s.pt" --extractor lseg --baselines --vocab "$VOCAB" \
      --css-eval-data "$PREP/$s/$VIEW" --css-eval-stride 1 --feat-dim 64 --pca-frames 20 \
      --seed 0 --out "$FIELDS/${s}_lseg.pt" $MAP_EXTRA
    [ "$DRY_RUN" = 1 ] || { grep -E "gpu_gate|PCA|CSS|keyframes used|timing|saved" "$L" || true; }
    [ "$DRY_RUN" = 1 ] || python3 -c 'import json, sys; json.dump(dict(map_extra=sys.argv[2]), open(sys.argv[1], "w"))' \
      "$FIELDS/${s}_lseg.pt.map_extra.json" "$MAP_EXTRA"
  fi
  for job in $DISTILL_JOBS; do            # trained baselines: f3dgs, legs:<lambda_u>
    m=${job%%:*}; lam=${job#*:}; [ "$m" = "$job" ] && lam=1
    name=$m; [ "$m" = legs ] && name="legs_l$(python3 -c "print(f'{float($lam):g}')")"
    [ -f "$FIELDS/${s}_lseg_${name}.pt" ] && continue
    L="$OUT/distill_${s}_${name}.$TS.log"
    if [ "$DRY_RUN" = 1 ]; then
      echo "[dry-run] scripts/gpu_gate.sh $DISTILL_NEED python3 scripts/distill_features.py --field $FIELDS/${s}_lseg.pt --method $m --lam-u $lam"
      continue
    fi
    ok=0
    for try in 1 2 3; do                 # a CUDA launch time-out kills the process: resume
      scripts/gpu_gate.sh "$DISTILL_NEED" python3 scripts/distill_features.py \
        --field "$FIELDS/${s}_lseg.pt" --method "$m" --lam-u "$lam" >> "$L" 2>&1 && { ok=1; break; }
      echo "distill $s $name: attempt $try failed ($(grep -m1 -oE 'CUDA error: [^.]*' "$L" | tail -1)); resuming"
    done
    [ "$ok" = 1 ] || { tail -20 "$L"; exit 1; }
    grep -E "gpu_gate|mapping frames|resumed|converged|saved" "$L" || true
  done
done

[ "$STAGE" = prep ] && { echo "done: prepared$SCENES"; exit 0; }
[ "$EVAL" = 1 ] || { echo "done: $STAGE maps and fields only (EVAL=0)"; exit 0; }
COMMON=(--vocab "$VOCAB" --row-factory usplat.baselines:row_factory,usplat.distill:row_factory
        --extractor lseg --dataset scannet --stride "$EVAL_STRIDE" --downscale "$DOWNSCALE"
        --tau-cov "$TAU_COV" --text-cache "$FIELDS/text_cache_lseg.pt" --with-2d)
[ "$V2" = 1 ] && COMMON+=(--v2-rows)
L="$OUT/eval_${STAGE}.$TS.log"
if [ "$STAGE" = val ]; then
  ARGS=(--val-scenes configs/scannet_val.json --scenes configs/scannet_val.json --out "$VAL")
else
  ARGS=(--betas "$VAL" --scenes configs/scannet_test.json --out "$TEST")
fi
if [ "$DRY_RUN" = 1 ]; then
  echo "[dry-run] scripts/gpu_gate.sh $EVAL_NEED python3 scripts/eval_seg.py ${COMMON[*]} ${ARGS[*]}"
  [ "$STAGE" = val ] && echo "[dry-run] settings -> $PROTO; validation selection -> $SEL"
  [ "$STAGE" = test ] && echo "[dry-run] look record $LOOK: attempt (selection $SEL)"
  echo "done (dry run): $STAGE"; exit 0
fi
[ "$STAGE" = test ] && look_test add      # from here on the selection is final
ok=0
for try in 1 2 3; do                     # retried only after a crash (no result file is written)
  scripts/gpu_gate.sh "$EVAL_NEED" python3 scripts/eval_seg.py "${COMMON[@]}" "${ARGS[@]}" \
    >> "$L" 2>&1 && { ok=1; break; }
  echo "eval $STAGE: attempt $try failed ($(grep -oE 'CUDA error: [^.]*|Error: [^.]*' "$L" | tail -1)); retrying"
done
[ "$ok" = 1 ] || { tail -30 "$L"; exit 1; }
grep -vE "it/s|s/it|waiting" "$L" | tail -90
if [ "$STAGE" = val ]; then record_protocol; select_val; fi
echo "done: $STAGE -> $OUT (log $L)"
