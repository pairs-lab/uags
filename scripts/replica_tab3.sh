#!/usr/bin/env bash
# Tab III ablations on Replica (plan section 3), rows from usplat/ablation_rows.py:
#   read-out variants (estimator, epistemic, kappa) reuse the Tab I field's sums;
#   re-mapped variants (pixels, centre, legacy) get their own mapping pass on the same 3DGS
#   map, saved as <field>_<variant>.pt next to the Tab I field.
#   map, with the main field's PCA basis (--pca-from), saved as <field>_<variant>.pt next to
#   the Tab I field. The AUSE of Tab III is each row's own u_epi / u_ale against the 0/1
#   error on the full map (eval_seg "ranking"); the e_red / e_irr split is the views figure.
#   STAGE=val  : map the variants of office_0, room_2; fit one beta per row -> $OUT/eval_val.json
#   STAGE=test : map the test-scene variants, evaluate with the frozen betas (run ONCE)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_USGLOC_EXTRA_MIB=${UAGS_GPU_USGLOC_EXTRA_MIB:-99999}   # never beside USGLoc
export UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-5}
TS=$(date +%Y%m%d-%H%M%S)
ROOT=/home/thanhnc/gauss_ws/data/replica_semnerf/Replica_Dataset
MAPS=/home/thanhnc/gauss_ws/data/uags_maps
FIELDS=/home/thanhnc/gauss_ws/data/uags_fields
OUT=${OUT:-/home/thanhnc/gauss_ws/data/uags_results/tab3}
STAGE=${STAGE:-val}
DOWNSCALE=${DOWNSCALE:-2}
TAU_COV=${TAU_COV:-0.5}
# GPU memory caps (MiB): mapping peaks at ~4.8 GB for 620k Gaussians and grows with N
# (room_2, 954k, went past 5.7 GB); the evaluation holds LSeg and the field together
MAP_NEED=${MAP_NEED:-9000}
EVAL_NEED=${EVAL_NEED:-7000}
VARIANTS=${VARIANTS:-"pixels centre legacy"}
mkdir -p "$OUT"
case "$STAGE" in
  val)  SCENES="office_0 room_2"; CFG=configs/replica_val.json ;;
  test) SCENES="room_0 room_1 office_1 office_2 office_3 office_4"; CFG=configs/replica_test.json ;;
  *)    echo "STAGE must be val or test (got '$STAGE')"; exit 2 ;;
esac
echo "[tab3 $STAGE] $TS git $(git rev-parse --short HEAD)"
if [ "$STAGE" = test ] && [ -f "$OUT/eval_test.json" ] && [ "${FORCE_TEST:-0}" != 1 ]; then
  echo "refused: $OUT/eval_test.json exists (the test split runs once; FORCE_TEST=1 to override)"; exit 1
fi

for s in $SCENES; do
  [ -f "$FIELDS/${s}_lseg.pt" ] || { echo "no Tab I field for $s: run replica_tab1.sh first"; exit 1; }
  for v in $VARIANTS; do
    f="$FIELDS/${s}_lseg_${v}.pt"
    [ -f "$f" ] && continue
    # segmentation-only fields: no change test (its buckets would double the file size)
    fc=$(python3 -c "import json; from usplat.ablation_rows import field_config_json as f; c = json.loads(f('$v')); c['change_test'] = False; print(json.dumps(c))")
    L="$OUT/map_${s}_${v}.$TS.log"
    scripts/gpu_gate.sh "$MAP_NEED" python3 scripts/run_mapping.py --backend gsplat \
      --dataset replica_semnerf --data "$ROOT/$s/Sequence_1" --stride 5 --downscale "$DOWNSCALE" \
      --ply "$MAPS/${s}_seq1.pt" --extractor lseg --feat-dim 64 --pca-frames 20 --seed 0 \
      --pca-from "$FIELDS/${s}_lseg.pt" --field-config "$fc" --out "$f" > "$L" 2>&1 \
      || { tail -20 "$L"; exit 1; }
    grep -E "keyframes used|timing|saved" "$L"
  done
done

COMMON=(--vocab configs/vocab_replica.json --row-factory usplat.ablation_rows:row_factory
        --extractor lseg --dataset replica_semnerf --stride 10 --downscale "$DOWNSCALE"
        --tau-cov "$TAU_COV" --text-cache "$FIELDS/text_cache_lseg.pt")
L="$OUT/eval_${STAGE}.$TS.log"
if [ "$STAGE" = val ]; then
  scripts/gpu_gate.sh "$EVAL_NEED" python3 scripts/eval_seg.py "${COMMON[@]}" \
    --val-scenes "$CFG" --scenes "$CFG" --out "$OUT/eval_val.json" > "$L" 2>&1 || { tail -30 "$L"; exit 1; }
else
  [ -f "$OUT/eval_val.json" ] || { echo "run STAGE=val first"; exit 1; }
  scripts/gpu_gate.sh "$EVAL_NEED" python3 scripts/eval_seg.py "${COMMON[@]}" \
    --betas "$OUT/eval_val.json" --scenes "$CFG" --out "$OUT/eval_test.json" > "$L" 2>&1 || { tail -30 "$L"; exit 1; }
fi
grep -vE "it/s|s/it|waiting" "$L" | tail -60
echo "done: tab3 $STAGE -> $OUT"
