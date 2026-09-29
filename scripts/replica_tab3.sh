#!/usr/bin/env bash
# Tab III ablations on Replica (plan section 3), rows from usplat/ablation_rows.py:
#   read-out variants (estimator, epistemic, kappa) reuse the Tab I field's sums;
#   re-mapped variants (pixels, centre, legacy) get their own mapping pass on the same 3DGS
#   map, saved as <field>_<variant>.pt next to the Tab I field.
#   STAGE=val  : map the variants of office_0, room_2; fit one beta per row -> $OUT/eval_val.json
#   STAGE=test : map the test-scene variants, evaluate with the frozen betas (run once)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
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
EVAL_NEED=${EVAL_NEED:-9000}
VARIANTS=${VARIANTS:-"pixels centre legacy"}
mkdir -p "$OUT"
if [ "$STAGE" = val ]; then SCENES="office_0 room_2"; CFG=configs/replica_val.json
else SCENES="room_0 room_1 office_1 office_2 office_3 office_4"; CFG=configs/replica_test.json; fi

for s in $SCENES; do
  [ -f "$FIELDS/${s}_lseg.pt" ] || { echo "no Tab I field for $s: run replica_tab1.sh first"; exit 1; }
  for v in $VARIANTS; do
    f="$FIELDS/${s}_lseg_${v}.pt"
    [ -f "$f" ] && continue
    # segmentation-only fields: no change test (its buckets would double the file size)
    fc=$(python3 -c "import json; from usplat.ablation_rows import field_config_json as f; c = json.loads(f('$v')); c['change_test'] = False; print(json.dumps(c))")
    scripts/gpu_gate.sh "$MAP_NEED" python3 scripts/run_mapping.py --backend gsplat \
      --dataset replica_semnerf --data "$ROOT/$s/Sequence_1" --stride 5 --downscale "$DOWNSCALE" \
      --ply "$MAPS/${s}_seq1.pt" --extractor lseg --feat-dim 64 --pca-frames 20 \
      --field-config "$fc" --out "$f" > "$OUT/map_${s}_${v}.log" 2>&1
    grep -E "keyframes used|timing|saved" "$OUT/map_${s}_${v}.log" || { tail -20 "$OUT/map_${s}_${v}.log"; exit 1; }
  done
done

COMMON=(--vocab configs/vocab_replica.json --row-factory usplat.ablation_rows:row_factory
        --extractor lseg --dataset replica_semnerf --stride 10 --downscale "$DOWNSCALE"
        --tau-cov "$TAU_COV" --text-cache "$FIELDS/text_cache_lseg.pt")
if [ "$STAGE" = val ]; then
  scripts/gpu_gate.sh "$EVAL_NEED" python3 scripts/eval_seg.py "${COMMON[@]}" \
    --val-scenes "$CFG" --scenes "$CFG" --out "$OUT/eval_val.json"
else
  [ -f "$OUT/eval_val.json" ] || { echo "run STAGE=val first"; exit 1; }
  scripts/gpu_gate.sh "$EVAL_NEED" python3 scripts/eval_seg.py "${COMMON[@]}" \
    --betas "$OUT/eval_val.json" --scenes "$CFG" --out "$OUT/eval_test.json"
fi
echo "done: tab3 $STAGE -> $OUT"
