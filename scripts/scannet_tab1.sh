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
#   method   LSeg, the Replica templates, PCA-64 from the first 20 mapping frames, mapping /
#            evaluation downscale 2, tau_cov 0.5, 3DGS fit settings: those of Replica Table I
# Protocol within one scan (frame numbers of the complete SensReader export):
#   mapping frames     0, 10, 20, ... (every 10th): the 3DGS fit (scripts/fit_gs.py, which
#                      holds out every 8th of them for its PSNR) and the semantic field
#                      (run_mapping.py, with its near-duplicate pose filter)
#   evaluation frames  5, 25, 45, ... (every 20th, offset 5: configs/scannet_*.json) - never
#                      fitted, never mapped; each is 5 frames (1/6 s) from a mapping frame
#   lost tracking      frames with a non-finite (-inf) pose are skipped by every step
#   CSS vocabulary     the classes present in every 5th frame: run_mapping.py's css_vocabulary
#                      takes no offset, and frames 0, 5, 10, ... contain every evaluation
#                      frame, so each queried class has a CSS column (on Replica the CSS
#                      vocabulary is exactly the evaluation frames' classes)
# Data root /home/thanhnc/gauss_ws/data/scannet (ScanNet's download-scannet.py -o <root>
# --id <scene> with --type .sens and --type _2d-label-filt.zip, and --label_map), layout:
#   scannetv2-labels.combined.tsv                 (top level of the ScanNet download)
#   scans/<scene>/color/<i>.jpg depth/<i>.png pose/<i>.txt intrinsic/intrinsic_color.txt
#                 intrinsic/intrinsic_depth.txt   (ScanNet SensReader/python, every frame:
#       python reader.py --filename scans/<scene>/<scene>.sens --output_path scans/<scene> \
#           --export_color_images --export_depth_images --export_poses --export_intrinsics)
#   scans/<scene>/label-filt/<i>.png              (cd scans/<scene> && unzip <scene>_2d-label-filt.zip)
#   uags_f2/<scene>/                              written by this script (CPU):
#       usplat.datasets.prepare_scannet - colour / 2 (648x484, as usplat.datasets.downscale),
#       intrinsics scaled, depth/pose/label-filt/tsv symlinked to scans/<scene>
# Memory: run_mapping.py keeps the fp32 LSeg tokens of every mapping frame on the CPU
# (~10 MB per frame at 648x484): scene0000_00 (5578 frames, 558 mapping frames) needs ~6 GB.
# Stages:
#   STAGE=prep   : prepare the 10 scans only (CPU; val and test prepare their own scans)
#   STAGE=val    : prepare + fit + map + evaluate the validation scans; fit one temperature
#                  per row on them -> $OUT/eval_val.json; then the validation selection
#                  -> $OUT/selection_val.json (the row choices of scripts/make_tables.py, the
#                  protocol settings and the sha256 of eval_val.json)
#   STAGE=select : redo only the selection from $OUT/eval_val.json (e.g. with V2_SELECT)
#   STAGE=test   : prepare + fit + map the test scans and evaluate them ONCE with the frozen
#                  betas. Refused on a dirty tree, when $OUT/eval_test.json holds a finished
#                  run (no override; a crashed run, status != ok, may be rerun), without a
#                  finished selection whose sha256 matches eval_val.json, or when the
#                  protocol settings (downscale, tau_cov, iters, MAP_EXTRA, the configs'
#                  sha256) differ from the selection's. val and select are refused once the
#                  test split ran (the selection is frozen).
#   EVAL=0       : prepare + maps and fields only
# Options:
#   V2_ROWS=auto|1|0  (val) pass --v2-rows to eval_seg.py when it has the flag (auto), always
#                     (1: refused when it has not), never (0); the test stage passes it iff
#                     the validation evaluation did (its recorded argv)
#   V2_SELECT=<cmd>   (val, select) run as `<cmd> $OUT/eval_val.json`; its stdout, one JSON
#                     object, is stored in the selection as "v2" (the v2 selection rules)
#   MAP_EXTRA=<args>  extra run_mapping.py arguments (e.g. a CSS vocabulary option); part of
#                     the protocol, so test refuses another value than validation's
#   DRY_RUN=1         print the GPU and data steps instead of running them; the checks run
# Every GPU step goes through scripts/gpu_gate.sh (never beside a USGLoc job, poll every 5 s).
# Existing prepared scans, maps and fields are reused; eval_seg.py --betas refuses test fields
# built with other mapping code, options or protocol than the validation fields.
# Logs: $OUT/<step>_<scene>.<timestamp>.log (never overwritten). Results: $OUT/eval_<stage>.json
# (+ .partial.json after every scene), $OUT/selection_val.json.
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
STAGE=${STAGE:-val}
EVAL=${EVAL:-1}
DRY_RUN=${DRY_RUN:-0}
ITERS=${ITERS:-30000}
DOWNSCALE=${DOWNSCALE:-2}
TAU_COV=${TAU_COV:-0.5}          # coverage rule of Replica Table I
V2_ROWS=${V2_ROWS:-auto}
V2_SELECT=${V2_SELECT:-}
MAP_EXTRA=${MAP_EXTRA:-}
FACTOR=2                         # prepared colour = native / FACTOR
MAP_STRIDE=10                    # mapping frames 0, 10, 20, ...
EVAL_STRIDE=20 EVAL_OFFSET=5     # evaluation frames 5, 25, 45, ... (configs/scannet_*.json)
CSS_STRIDE=5                     # CSS vocabulary frames, a superset of the evaluation frames
# GPU memory caps (MiB), as replica_tab1.sh
MAP_NEED=${MAP_NEED:-9000}
EVAL_NEED=${EVAL_NEED:-7000}
DISTILL_NEED=${DISTILL_NEED:-7000}
TS=$(date +%Y%m%d-%H%M%S)
case "$STAGE" in
  prep)       SPLITS="val test" ;;
  val|select) SPLITS=val ;;
  test)       SPLITS=test ;;
  *)          echo "STAGE must be prep, val, select or test (got '$STAGE')"; exit 2 ;;
esac
mkdir -p "$OUT"
[ "$DRY_RUN" = 1 ] || mkdir -p "$PREP" "$MAPS" "$FIELDS"

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
    "$EVAL_OFFSET" "$CSS_STRIDE" "$MAP_EXTRA" <<'PY'
import hashlib, json, sys
ds, tau, iters, factor, ms, es, eo, cs, extra = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
print(json.dumps(dict(downscale=int(ds), tau_cov=float(tau), iters=int(iters),
                      prep_factor=int(factor), map_stride=int(ms), eval_stride=int(es),
                      eval_offset=int(eo), css_eval_stride=int(cs), map_extra=extra,
                      configs={p: sha(p) for p in ("configs/vocab_scannet.json",
                                                   "configs/scannet_val.json",
                                                   "configs/scannet_test.json")}),
                 sort_keys=True))
PY
}
has_v2() {    # eval_seg.py has --v2-rows (captured first: grep -q under pipefail can SIGPIPE)
  local help
  help=$(python3 scripts/eval_seg.py --help 2>/dev/null || true)
  [[ $help == *--v2-rows* ]]
}
test_done() { # $OUT/eval_test.json is a finished run record
  [ -f "$OUT/eval_test.json" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status', 'ok') == 'ok' else 1)" "$OUT/eval_test.json"
}
step() {      # step <log> <command...>: run it, output to <log>; on failure show the tail, stop
  local L=$1; shift
  if [ "$DRY_RUN" = 1 ]; then echo "[dry-run] $*"; return 0; fi
  "$@" > "$L" 2>&1 || { tail -20 "$L"; exit 1; }
}
select_val() {   # $OUT/eval_val.json -> $OUT/selection_val.json
  python3 - "$OUT/eval_val.json" "$OUT/selection_val.json" "$PROTOCOL" "$V2_SELECT" <<'PY'
import hashlib, json, os, shlex, subprocess, sys
sys.path.insert(0, "scripts")
import make_tables as mt
val, out, proto, v2_select = sys.argv[1:]
body = mt.body(val)                         # refuses an unfinished evaluation
d = json.load(open(val))
if not body.get("betas"):
    sys.exit(f"refused: {val} holds no fitted betas")
argv = d.get("argv") or []
rec = dict(status="ok", eval_val=os.path.abspath(val),
           eval_val_sha256=hashlib.sha256(open(val, "rb").read()).hexdigest(),
           git=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip(),
           protocol=json.loads(proto), v2_rows="--v2-rows" in argv, eval_argv=argv,
           betas_rows=sorted(body["betas"]),
           field_identities=(body.get("beta_fit_settings") or {}).get("field_identities"),
           choices={k.lstrip("@"): mt.choose(body["summary"], k) for k in mt.CHOICES})
pre = "configs/tab1_v2_prereg.json"
if os.path.exists(pre):
    rec["prereg"] = dict(path=pre, sha256=hashlib.sha256(open(pre, "rb").read()).hexdigest())
    if not v2_select:
        print(f"note: {pre} exists but V2_SELECT is empty: the selection holds only the "
              f"scripts/make_tables.py choices", file=sys.stderr)
if v2_select:
    r = subprocess.run(shlex.split(v2_select) + [val], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"refused: V2_SELECT failed (exit {r.returncode}): {r.stderr[-800:]}")
    rec.update(v2=json.loads(r.stdout), v2_select=v2_select)
with open(out + ".tmp", "w") as f:
    json.dump(rec, f, indent=1)
os.replace(out + ".tmp", out)
print(f"validation selection -> {out}: {rec['choices']}, v2 rows {rec['v2_rows']}"
      + (f", v2 {rec['v2']}" if "v2" in rec else ""))
PY
}
check_selection() {   # refuses unless a finished selection matches eval_val.json and the protocol
  python3 - "$OUT/selection_val.json" "$OUT/eval_val.json" "$PROTOCOL" <<'PY'
import hashlib, json, sys
sel_path, val, proto = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
try:
    sel = json.load(open(sel_path))
except (OSError, ValueError) as e:
    sys.exit(f"refused: no finished validation selection ({e}); run STAGE=val first")
if sel.get("status") != "ok":
    sys.exit(f"refused: {sel_path} is not finished (status {sel.get('status')!r})")
try:
    sha = hashlib.sha256(open(val, "rb").read()).hexdigest()
except OSError as e:
    sys.exit(f"refused: {e}")
if sha != sel.get("eval_val_sha256"):
    sys.exit(f"refused: {val} changed after the selection (sha256 {sha[:12]} != "
             f"{str(sel.get('eval_val_sha256'))[:12]}); rerun STAGE=select")
diff = {k: (sel.get("protocol", {}).get(k), v) for k, v in proto.items()
        if sel.get("protocol", {}).get(k) != v}
if diff:
    sys.exit(f"refused: protocol differs from the validation selection (selection, now): {diff}")
print(f"validation selection {sel_path}: git {sel.get('git', '')[:12]}, {len(sel['betas_rows'])} "
      f"rows with betas, choices {sel['choices']}, v2 rows {sel['v2_rows']}")
PY
}
sel_get() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$OUT/selection_val.json" "$1"; }

SCENES=""
for sp in $SPLITS; do SCENES="$SCENES $(scenes "$sp")"; done
PROTOCOL=$(protocol)
DIRTY=$(git status --porcelain --untracked-files=no -- scripts usplat configs | wc -l)
echo "[$STAGE] $TS git $(git rev-parse --short HEAD) dirty: $DIRTY files; scenes:$SCENES"
if [[ $STAGE == val || $STAGE == select ]] && test_done; then
  echo "refused: $OUT/eval_test.json holds a finished test run; the validation selection is frozen"; exit 1
fi
if [ "$STAGE" = select ]; then select_val; exit 0; fi
V2=0
if [ "$STAGE" = val ] && [ "$EVAL" = 1 ]; then
  case "$V2_ROWS" in
    0)    V2=0 ;;
    1)    has_v2 || { echo "refused: V2_ROWS=1 but scripts/eval_seg.py has no --v2-rows"; exit 1; }; V2=1 ;;
    auto) if has_v2; then V2=1; fi ;;
    *)    echo "V2_ROWS must be auto, 1 or 0 (got '$V2_ROWS')"; exit 2 ;;
  esac
fi
if [ "$STAGE" = test ]; then
  if [ "$DIRTY" != 0 ]; then
    echo "refused: uncommitted changes in scripts/ usplat/ configs/ (the test run must name a commit)"; exit 1
  fi
  if [ "$EVAL" = 1 ]; then
    if test_done; then
      echo "refused: $OUT/eval_test.json holds a finished test run (the test split runs once; no override)"; exit 1
    fi
    check_selection
    [ "$(sel_get v2_rows)" = True ] && V2=1
    if [ "$V2" = 1 ] && ! has_v2; then
      echo "refused: the validation evaluation used --v2-rows, which scripts/eval_seg.py does not have"; exit 1
    fi
  fi
fi
echo "protocol: $PROTOCOL"
# trained baselines: validation trains LEGaussians for each lambda_u of usplat.distill
# LEGS_LAMBDAS; the test stage trains only the lambda_u of the validation selection
DISTILL_JOBS=""
if [ "$STAGE" = val ]; then
  DISTILL_JOBS="f3dgs $(python3 -c "from usplat.distill import LEGS_LAMBDAS as L; print(' '.join(f'legs:{l:g}' for l in L))")"
elif [ "$STAGE" = test ] && [ "$EVAL" = 1 ]; then
  chosen=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['choices']['legs'])" "$OUT/selection_val.json")
  DISTILL_JOBS="f3dgs legs:${chosen#legs_l}"
  echo "LEGaussians lambda_u chosen on validation: ${chosen#legs_l} ($chosen)"
fi

for s in $SCENES; do
  if [ "$DRY_RUN" = 1 ]; then
    echo "[dry-run] prepare_scannet $DATA/scans/$s -> $PREP/$s (colour / $FACTOR)"
  else
    [ -d "$DATA/scans/$s/label-filt" ] || { echo "$DATA/scans/$s/label-filt missing (unzip ${s}_2d-label-filt.zip there)"; exit 1; }
    python3 -c 'import sys; from usplat.datasets import prepare_scannet as p; r = p(sys.argv[1], sys.argv[2], factor=int(sys.argv[3])); print(sys.argv[2], r["n_frames"], "frames", r["native_hw"], "->", r["hw"])' \
      "$DATA/scans/$s" "$PREP/$s" "$FACTOR"
  fi
  [ "$STAGE" = prep ] && continue
  if [ ! -f "$MAPS/$s.pt" ]; then         # --max-res 0: fit at the prepared resolution
    L="$OUT/fit_${s}.$TS.log"
    step "$L" scripts/gpu_gate.sh 3000 python3 scripts/fit_gs.py --dataset scannet \
      --data "$PREP/$s" --stride "$MAP_STRIDE" --iters "$ITERS" --max-gaussians 1000000 \
      --max-res 0 --out "$MAPS/$s.pt"
    [ "$DRY_RUN" = 1 ] || { grep -E "gpu_gate|PSNR|psnr|Gaussians|dropped|saved" "$L" | tail -5 || true; }
  fi
  if [ ! -f "$FIELDS/${s}_lseg.pt" ]; then
    L="$OUT/map_${s}.$TS.log"
    # shellcheck disable=SC2086          # MAP_EXTRA: word-split on purpose
    step "$L" scripts/gpu_gate.sh "$MAP_NEED" python3 scripts/run_mapping.py --backend gsplat \
      --dataset scannet --data "$PREP/$s" --stride "$MAP_STRIDE" --downscale "$DOWNSCALE" \
      --ply "$MAPS/$s.pt" --extractor lseg --baselines --vocab "$VOCAB" \
      --css-eval-data "$PREP/$s" --css-eval-stride "$CSS_STRIDE" --feat-dim 64 --pca-frames 20 \
      --seed 0 --out "$FIELDS/${s}_lseg.pt" $MAP_EXTRA
    [ "$DRY_RUN" = 1 ] || { grep -E "gpu_gate|PCA|CSS|keyframes used|timing|saved" "$L" || true; }
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
  ARGS=(--val-scenes configs/scannet_val.json --scenes configs/scannet_val.json --out "$OUT/eval_val.json")
else
  ARGS=(--betas "$OUT/eval_val.json" --scenes configs/scannet_test.json --out "$OUT/eval_test.json")
fi
if [ "$DRY_RUN" = 1 ]; then
  echo "[dry-run] scripts/gpu_gate.sh $EVAL_NEED python3 scripts/eval_seg.py ${COMMON[*]} ${ARGS[*]}"
  [ "$STAGE" = val ] && echo "[dry-run] validation selection -> $OUT/selection_val.json"
  echo "done (dry run): $STAGE"; exit 0
fi
ok=0
for try in 1 2 3; do                     # retried only after a crash (no result file is written)
  scripts/gpu_gate.sh "$EVAL_NEED" python3 scripts/eval_seg.py "${COMMON[@]}" "${ARGS[@]}" \
    >> "$L" 2>&1 && { ok=1; break; }
  echo "eval $STAGE: attempt $try failed ($(grep -oE 'CUDA error: [^.]*|Error: [^.]*' "$L" | tail -1)); retrying"
done
[ "$ok" = 1 ] || { tail -30 "$L"; exit 1; }
grep -vE "it/s|s/it|waiting" "$L" | tail -90
[ "$STAGE" = val ] && select_val
echo "done: $STAGE -> $OUT (log $L)"
