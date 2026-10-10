#!/usr/bin/env bash
# Static false-alarm rate of the label-free change threshold on Replica (paper, change test:
# "we measure the actual rate"). Replica Sequence_2 shows the scene of Sequence_1 from another
# trajectory: nothing changes, so every fire of the update pass is a false alarm.
#   map    : Sequence_1 every 5th frame, as scripts/replica_tab1.sh builds the Table I fields
#            (downscale 2, LSeg, PCA-64 from the first 20 frames, seed 0, the Table I 3DGS maps)
#   update : Sequence_2 every 5th frame (--update-data) with the dataset depth (--use-depth),
#            under the threshold the field calibrated on its own mapping pass
#   record : run_mapping.py --summary-json: defined tests and fires of every update view, the
#            realised per-view rate fires/tests overall and per history bucket, the reset share
#            of the observed pixel mass, the thresholds, the config; the residual baselines (NCC,
#            feature, depth) with the same label-free rule. --no-save-field: no field file is
#            written, so the Table I fields are never touched.
# Configs, frozen 2026-10-10 before any run: configs/static_fa.json (q = 0.999, history null,
# k = 2, 5 cm re-admission depth tolerance) and the ablations configs/static_fa_pooled.json,
# _max (calib_mode), _fixed (one threshold carried over from validation), _k1 (k = 1).
#   STAGE=val   : office_0, room_2, every config but _fixed; then scripts/static_fa_report.py
#                 select -> $OUT/val_selection.json (the fixed threshold for the test stage).
#                 Refused once any test run record exists (validation is closed).
#   STAGE=test  : the 6 test scenes, every config, run ONCE. Refused with uncommitted changes in
#                 scripts/ usplat/ configs/, without a finished $OUT/val_selection.json, or when
#                 $OUT/test_criterion.json holds a finished test run (FORCE_TEST=1 overrides that
#                 check only). A finished per-scene record is never overwritten (a crashed run,
#                 status "running"/"failed", is rerun). Then scripts/static_fa_report.py
#                 criterion -> $OUT/test_criterion.json.
#   STAGE=audit : CPU only: scripts/audit_change_coverage.py on the 2 VALIDATION Table I fields
#                 -> $OUT/audit_val_fields.json (coverage of the change test, its pre-registered
#                 decision). Any configuration choice it informs (e.g. the opt-in change weight)
#                 is taken on these fields only; the 6 test fields may be audited by hand, as a
#                 descriptive number, only after the change-evaluation configs are frozen.
# PRE-REGISTERED criterion (configs/static_fa.json "criterion"): the realised per-view false-alarm
# rate of static_fa at q = 0.999 lies within [0.5x, 2x] of the nominal 1e-3 on >= 5 of the 6 test
# scenes. Every GPU step goes through scripts/gpu_gate.sh. Logs: $OUT/logs/<config>_<scene>.<ts>.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_USGLOC_EXTRA_MIB=${UAGS_GPU_USGLOC_EXTRA_MIB:-99999}   # never beside USGLoc
export UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-5}
ROOT=/home/thanhnc/gauss_ws/data/replica_semnerf/Replica_Dataset
MAPS=/home/thanhnc/gauss_ws/data/uags_maps
FIELDS=/home/thanhnc/gauss_ws/data/uags_fields
OUT=${OUT:-/home/thanhnc/gauss_ws/data/uags_results/static_fa}
STAGE=${STAGE:-val}
# mapping peaks at ~4.8-5.7 GB (Table I); the residual baselines add three renders per view
MAP_NEED=${MAP_NEED:-10000}
FROZEN=configs/static_fa.json
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT/runs" "$OUT/logs"
cfgval() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); [v := v[k] for k in sys.argv[2:]]; print(' '.join(v) if isinstance(v, list) else v)" "$@"; }
finished() { [ -f "$1" ] && python3 -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status') == 'ok' else 1)" "$1"; }
VAL_SCENES=$(cfgval "$FROZEN" data val_scenes)
TEST_SCENES=$(cfgval "$FROZEN" data test_scenes)
CONFIGS="static_fa $(cfgval "$FROZEN" ablations)"
case "$STAGE" in
  val)   SCENES=$VAL_SCENES ;;
  test)  SCENES=$TEST_SCENES ;;
  audit) SCENES=$VAL_SCENES ;;                  # decisions on validation fields only
  *)     echo "STAGE must be val, test or audit (got '$STAGE')"; exit 2 ;;
esac
echo "[$STAGE] $TS git $(git rev-parse --short HEAD) dirty: $(git status --porcelain --untracked-files=no -- scripts usplat configs | wc -l) files"

if [ "$STAGE" = audit ]; then
  FIELD_LIST=()
  for s in $SCENES; do FIELD_LIST+=("$FIELDS/${s}_lseg.pt"); done
  L="$OUT/logs/audit.$TS.log"
  python3 scripts/audit_change_coverage.py --field "${FIELD_LIST[@]}" \
    --out "$OUT/audit_val_fields.json" 2>&1 | tee "$L"
  exit 0
fi

if [ "$STAGE" = test ]; then
  if [ -n "$(git status --porcelain --untracked-files=no -- scripts usplat configs)" ]; then
    echo "refused: uncommitted changes in scripts/ usplat/ configs/ (the test run must name a commit)"; exit 1
  fi
  finished "$OUT/val_selection.json" || {
    echo "refused: $OUT/val_selection.json is not a finished validation selection (run STAGE=val first)"; exit 1; }
  if finished "$OUT/test_criterion.json" && [ "${FORCE_TEST:-0}" != 1 ]; then
    echo "refused: $OUT/test_criterion.json holds a finished test run (the test split runs once; FORCE_TEST=1 to override)"; exit 1
  fi
else
  for s in $TEST_SCENES; do
    for c in $CONFIGS; do
      if [ -f "$OUT/runs/$c/$s.json" ]; then
        echo "refused: test record $OUT/runs/$c/$s.json exists: validation is closed"; exit 1
      fi
    done
  done
fi

for s in $SCENES; do
  [ -f "$MAPS/${s}_seq1.pt" ] || {
    echo "missing map $MAPS/${s}_seq1.pt (fit it with STAGE=$STAGE EVAL=0 scripts/replica_tab1.sh)"; exit 1; }
  for c in $CONFIGS; do
    cfg=configs/$c.json
    if [ "$STAGE" = val ] && [ "$(cfgval "$cfg" fixed_change_thr)" = from_validation ]; then
      continue                                  # defined by the validation selection
    fi
    rec="$OUT/runs/$c/$s.json"
    if finished "$rec"; then
      echo "$c $s: finished, kept ($rec)"; continue
    fi
    mkdir -p "$OUT/runs/$c"
    SEL=()
    [ "$STAGE" = test ] && SEL=(--selection "$OUT/val_selection.json")
    mapfile -t CFG_ARGS < <(python3 scripts/static_fa_report.py args "$cfg" ${SEL[@]+"${SEL[@]}"})
    [ "${#CFG_ARGS[@]}" -gt 0 ] || { echo "no run_mapping arguments for $cfg"; exit 1; }
    L="$OUT/logs/${c}_${s}.$TS.log"
    scripts/gpu_gate.sh "$MAP_NEED" python3 scripts/run_mapping.py --backend gsplat \
      --dataset replica_semnerf --data "$ROOT/$s/Sequence_1" --stride 5 --downscale 2 \
      --ply "$MAPS/${s}_seq1.pt" --extractor lseg --feat-dim 64 --pca-frames 20 --seed 0 \
      --update-data "$ROOT/$s/Sequence_2" "${CFG_ARGS[@]}" \
      --summary-json "$rec" --no-save-field > "$L" 2>&1 || { tail -20 "$L"; exit 1; }
    grep -E "gpu_gate|PCA|calibrated|fixed change|residual baselines|update pass|^  [a-z_]+: [0-9]+ tests|timing" "$L"
  done
done

if [ "$STAGE" = val ]; then
  python3 scripts/static_fa_report.py select --config "$FROZEN" --results "$OUT" \
    2>&1 | tee "$OUT/logs/select.$TS.log"
else
  python3 scripts/static_fa_report.py criterion --config "$FROZEN" --results "$OUT" \
    2>&1 | tee "$OUT/logs/criterion.$TS.log"
fi
echo "done: $STAGE -> $OUT"
