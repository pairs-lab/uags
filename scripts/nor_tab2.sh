#!/usr/bin/env bash
# Table II on NeRF-Object-Removal (papers/main.tex, label tab:change; protocol D1-D6 of
# 2026-10-17; scenes, strides and keyframe counts in configs/nor_<stage>.json). Per scene:
#   align  the D3 update-pose alignment (scripts/align_nor.py, CPU), kept if it exists
#   fit    a 3DGS map on the mapping frames (scripts/fit_gs.py --part map: 30k iterations,
#          <= 1M Gaussians, --max-res 720 so the 720x540 images keep their size)
#   map    the mapping-only field (scripts/run_mapping.py --part map --baselines
#          --use-depth): LSeg, PCA-64 from the first 20 mapping frames, downscale 2, seed 0,
#          the Replica settings of replica_tab1.sh except the data; no CSS (D5)
#   eval   the update passes with every Table II row (scripts/eval_change_nor.py), the only step
#          that reads the update part: the D3 transform moves no mapping frame, so the maps
#          and fields do not depend on it (a change of the D3 rule reruns only the evaluations)
# then the merge over the scenes (mean +- population std; scripts/nor_tab2.py merge) and the
# LaTeX rows (scripts/make_tables.py tab2).
#   STAGE=val  : validation scenes 001-004 -> $OUT/change_val.json
#   STAGE=test : test scenes 005-016, run ONCE. Refused when scripts/ usplat/ configs/ hold
#                uncommitted changes or untracked source (the run must name a commit), when
#                $OUT/change_test.json holds a finished merge (FORCE_TEST=1 overrides), and
#                without a finished $OUT/change_val.json made by the same code and settings
#                (nor_tab2.py frozen; the test merge checks it again with --frozen)
#   EVAL=0     : alignments (if any), maps and fields only, no evaluation
#   DRY_RUN=1  : print the steps instead of running them (the guards still run)
#   ABLATION=1 : with VARIANT and STAGE=test, a Table III ablation of the final method: scored on
#                the test scenes only (no choice depends on it), so it has no validation run and
#                skips the frozen checks; the stage log says so. Still runs once per variant.
#   VARIANT=<name> FIELD_CONFIG=<json> : a field variant chosen on validation (e.g. the change
#                weight): run_mapping.py --field-config <json>, fields <field>_<name>.pt, outputs
#                under $OUT (default .../nor/tab2_<name>); maps are shared with the default
# An existing alignment is kept; an existing map, field or evaluation is reused only if it
# was built with the same options and code (nor_tab2.py reuse), else the run stops: a smoke
# output at a protocol path is never reused silently. A finished evaluation is never rerun
# (each test scene is scored once); a crashed one (status "running" / "failed") is rerun, up
# to 3 attempts (a CUDA launch time-out kills the process: Xorg shares this GPU).
# Every GPU step goes through scripts/gpu_gate.sh (GPU_GATE overrides it, for tests): never
# beside a USGLoc job, 1 GiB headroom on this 12 GB card, poll every 5 s.
# Logs: $OUT/<step>_<scene>.<timestamp>.log (never overwritten); the stage log
# $OUT/nor_tab2_<stage>.<timestamp>.log records the git state and every reuse decision.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -x "$HOME/venvs/uags/bin/python3" ]; then export PATH="$HOME/venvs/uags/bin:$PATH"; fi
export CUDA_HOME=${CUDA_HOME:-$HOME/.local/cuda-12.8}
export PATH="$CUDA_HOME/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_USGLOC_EXTRA_MIB=${UAGS_GPU_USGLOC_EXTRA_MIB:-99999}   # never beside USGLoc
export UAGS_GPU_HEADROOM_MIB=${UAGS_GPU_HEADROOM_MIB:-1024}            # 12 GB, Xorg holds 2 GB
export UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-5}
VARIANT=${VARIANT:-}
FIELD_CONFIG=${FIELD_CONFIG:-}
ABLATION=${ABLATION:-0}
[ "$ABLATION" = 0 ] || [ -n "$VARIANT" ] || { echo "ABLATION needs VARIANT"; exit 2; }
[ -z "$VARIANT" ] || [ -n "$FIELD_CONFIG" ] || { echo "VARIANT needs FIELD_CONFIG"; exit 2; }
OUT=${OUT:-/home/thanhnc/gauss_ws/src/uags/results/nor/tab2${VARIANT:+_$VARIANT}}
STAGE=${STAGE:-val}
EVAL=${EVAL:-1}
DRY_RUN=${DRY_RUN:-0}
GATE=${GPU_GATE:-scripts/gpu_gate.sh}
ITERS=${ITERS:-30000}
MAX_RES=${MAX_RES:-720}          # fit_gs.py downscales to a long side <= max_res: 720 keeps 720x540
DOWNSCALE=${DOWNSCALE:-2}        # accumulation grid 360x270 (Replica: 2 as well)
TAU_COV=${TAU_COV:-0.5}          # pixel coverage floor, fixed before any run
# GPU memory caps (MiB): the fit peaked at < 0.3 GB on Replica (0.46M Gaussians); mapping
# holds the field, NormLift and the renders after LSeg has left the GPU; the evaluation holds
# two field copies, NormLift and the residual filters
FIT_NEED=${FIT_NEED:-3000}
MAP_NEED=${MAP_NEED:-7000}
EVAL_NEED=${EVAL_NEED:-6500}
# the options every scene shares (the per-scene ones come from the config); the frozen check
# compares this text between validation and test, besides the code
FIT_FIXED=(--iters "$ITERS" --max-gaussians 1000000 --max-res "$MAX_RES")
MAP_FIXED=(--downscale "$DOWNSCALE" --extractor lseg --baselines --use-depth --feat-dim 64
           --pca-frames 20 --seed 0)
[ -z "$FIELD_CONFIG" ] || MAP_FIXED+=(--field-config "$FIELD_CONFIG")
EVAL_FIXED=(--tau-cov "$TAU_COV")
PIPELINE="fit ${FIT_FIXED[*]} | map ${MAP_FIXED[*]} | eval ${EVAL_FIXED[*]}"
TS=$(date +%Y%m%d-%H%M%S)
case "$STAGE" in
  val|test) CFG=configs/nor_$STAGE.json ;;
  *)        echo "STAGE must be val or test (got '$STAGE')"; exit 2 ;;
esac
mkdir -p "$OUT/$STAGE"
if [ -n "$VARIANT" ]; then    # the split config with the variant's field paths, for the loop and the merge
  VCFG="$OUT/config_$STAGE.json"
  python3 -c "import json, sys
c = json.load(open(sys.argv[1]))
for e in c['scenes']:
    e['field'] = e['field'][:-3] + '_' + sys.argv[3] + '.pt' if e['field'].endswith('.pt') else e['field'] + '_' + sys.argv[3]
c['variant'] = dict(name=sys.argv[3], field_config=sys.argv[4], base_config=sys.argv[1])
json.dump(c, open(sys.argv[2], 'w'), indent=1)" "$CFG" "$VCFG" "$VARIANT" "$FIELD_CONFIG"
  CFG=$VCFG
fi
SLOG="$OUT/nor_tab2_${STAGE}.$TS.log"
say() { echo "$*" | tee -a "$SLOG"; }
run() {                       # run a step with its output appended to the log $1 (or print it)
  local log=$1; shift
  if [ "$DRY_RUN" = 1 ]; then say "[dry run] $* >> $log"; return 0; fi
  "$@" >> "$log" 2>&1
}
dirty() {                     # changed tracked files and untracked source (no bytecode)
  git status --porcelain --untracked-files=all -- scripts usplat configs \
    | grep -vE '(__pycache__/|\.pyc$)' || true
}
finished() {                  # a JSON run record with status "ok"
  python3 -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status') == 'ok' else 1)" "$1"
}
say "[$STAGE] $TS git $(git rev-parse --short HEAD) dirty: $(dirty | wc -l) files; $CFG sha256 $(sha256sum "$CFG" | cut -c1-12); $PIPELINE"
if [ "$STAGE" = test ]; then
  if [ -n "$(dirty)" ]; then
    say "refused: uncommitted changes or untracked source in scripts/ usplat/ configs/ (the test run must name a commit):"
    dirty | tee -a "$SLOG"; exit 1
  fi
  if [ "$EVAL" = 1 ]; then
    if [ -f "$OUT/change_test.json" ] && [ "${FORCE_TEST:-0}" != 1 ] && finished "$OUT/change_test.json"; then
      say "refused: $OUT/change_test.json holds a finished test run (the test split runs once; FORCE_TEST=1 to override)"
      exit 1
    fi
    if [ "$ABLATION" != 0 ]; then
      say "ablation $VARIANT: test only, no validation run, frozen checks skipped (Table III)"
    else
      CUDA_VISIBLE_DEVICES= python3 scripts/nor_tab2.py frozen "$OUT/change_val.json" \
        --pipeline "$PIPELINE" 2>&1 | tee -a "$SLOG"
    fi
  fi
fi

mapfile -t SCENES < <(python3 scripts/nor_tab2.py scenes "$CFG")
[ "${#SCENES[@]}" -gt 0 ] || { say "no scenes in $CFG"; exit 1; }
for line in "${SCENES[@]}"; do
  read -r s data stride ustride align ply field <<< "$line"
  if [ "$align" != none ] && [ ! -f "$align" ]; then          # D3 (reads the GT box to mask the object), CPU
    [ "$(basename "$align")" = "$s.json" ] || { say "$CFG: $s: align must be <dir>/$s.json"; exit 1; }
    L="$OUT/align_${s}.$TS.log"
    run "$L" env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
      python3 scripts/align_nor.py --root "$(dirname "$data")" --out-dir "$(dirname "$align")" \
      --scenes "$s" --summary "$OUT/align_${s}.$TS.json" || { tail -20 "$L"; exit 1; }
    [ "$DRY_RUN" = 1 ] || grep -E "^$s " "$L" | tee -a "$SLOG" || true
  fi
  FIT=(--dataset nor --data "$data" --part map --stride "$stride" "${FIT_FIXED[@]}" --out "$ply")
  if [ -f "$ply" ]; then
    CUDA_VISIBLE_DEVICES= python3 scripts/nor_tab2.py reuse map "$ply" -- "${FIT[@]}" | tee -a "$SLOG"
  else
    L="$OUT/fit_${s}.$TS.log"
    mkdir -p "$(dirname "$ply")"
    run "$L" "$GATE" "$FIT_NEED" python3 scripts/fit_gs.py "${FIT[@]}" || { tail -20 "$L"; exit 1; }
    [ "$DRY_RUN" = 1 ] || { grep -E "gpu_gate|frames" "$L"; tail -1 "$L" | cut -c1-400; } | tee -a "$SLOG" || true
  fi
  MAP=(--backend gsplat --dataset nor --data "$data" --part map --stride "$stride" --ply "$ply"
       "${MAP_FIXED[@]}" --out "$field")
  if [ -f "$field" ]; then
    CUDA_VISIBLE_DEVICES= python3 scripts/nor_tab2.py reuse field "$field" -- "${MAP[@]}" | tee -a "$SLOG"
  else
    L="$OUT/map_${s}.$TS.log"
    mkdir -p "$(dirname "$field")"
    run "$L" "$GATE" "$MAP_NEED" python3 scripts/run_mapping.py "${MAP[@]}" || { tail -20 "$L"; exit 1; }
    [ "$DRY_RUN" = 1 ] || grep -E "gpu_gate|PCA|keyframes used|timing|saved" "$L" | tee -a "$SLOG" || true
  fi
  [ "$EVAL" = 1 ] || continue
  E="$OUT/$STAGE/$s.json"
  EV=(--field "$field" --align "$align" --update-stride "$ustride" "${EVAL_FIXED[@]}" --out "$E")
  if [ -f "$E" ] && finished "$E"; then                       # scored once: never rerun
    CUDA_VISIBLE_DEVICES= python3 scripts/nor_tab2.py reuse eval "$E" -- "${EV[@]}" | tee -a "$SLOG"
    continue
  fi
  L="$OUT/eval_${s}.$TS.log"
  ok=0
  for try in 1 2 3; do                    # retried only after a crash (no finished record)
    run "$L" "$GATE" "$EVAL_NEED" python3 scripts/eval_change_nor.py "${EV[@]}" && { ok=1; break; }
    say "eval $s: attempt $try failed ($(grep -oE 'CUDA error: [^.]*|Error: [^.]*' "$L" | tail -1)); retrying"
  done
  [ "$ok" = 1 ] || { tail -30 "$L"; exit 1; }
  [ "$DRY_RUN" = 1 ] || grep -E "gpu_gate|^$s" "$L" | tee -a "$SLOG" || true
done

[ "$EVAL" = 1 ] || { say "done: $STAGE alignments, maps and fields only (EVAL=0)"; exit 0; }
L="$OUT/merge_${STAGE}.$TS.log"
M=(merge --config "$CFG" --dir "$OUT/$STAGE" --out "$OUT/change_$STAGE.json" --pipeline "$PIPELINE")
[ "$STAGE" = test ] && [ "$ABLATION" = 0 ] && M+=(--frozen "$OUT/change_val.json")
run "$L" env CUDA_VISIBLE_DEVICES= python3 scripts/nor_tab2.py "${M[@]}" || { tail -30 "$L"; exit 1; }
T="$OUT/tab2_${STAGE}.$TS.tex"
if [ "$DRY_RUN" = 1 ]; then
  say "[dry run] python3 scripts/make_tables.py tab2 $OUT/change_$STAGE.json > $T"
else
  python3 scripts/make_tables.py tab2 "$OUT/change_$STAGE.json" > "$T"
  { cat "$L"; cat "$T"; } | tee -a "$SLOG"
fi
say "done: $STAGE -> $OUT/change_$STAGE.json, rows $T (stage log $SLOG)"
