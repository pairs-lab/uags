#!/usr/bin/env bash
# Table I on Replica (Semantic-NeRF renders). Plan section 3:
#   map on Sequence_1 every 5th frame (180), evaluate on Sequence_2 every 10th frame (90);
#   validation scenes office_0, room_2 fit the temperatures; test scenes run ONCE with them.
#   STAGE=val  : fit + map + evaluate the validation scenes, fit betas -> $OUT/betas_val.json
#   STAGE=test : fit + map the test scenes, evaluate them with the frozen betas (run once)
# Every GPU step goes through scripts/gpu_gate.sh. Existing maps/fields are reused.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
ROOT=/home/thanhnc/gauss_ws/data/replica_semnerf/Replica_Dataset
MAPS=/home/thanhnc/gauss_ws/data/uags_maps
FIELDS=/home/thanhnc/gauss_ws/data/uags_fields
OUT=${OUT:-/home/thanhnc/gauss_ws/data/uags_results/tab1}
STAGE=${STAGE:-val}
ITERS=${ITERS:-30000}
DOWNSCALE=${DOWNSCALE:-2}
TAU_COV=${TAU_COV:-0.5}
mkdir -p "$MAPS" "$FIELDS" "$OUT"
if [ "$STAGE" = val ]; then SCENES="office_0 room_2"; else SCENES="room_0 room_1 office_1 office_2 office_3 office_4"; fi

for s in $SCENES; do
  if [ ! -f "$MAPS/${s}_seq1.pt" ]; then
    scripts/gpu_gate.sh 6000 python3 scripts/fit_gs.py --dataset replica_semnerf \
      --data "$ROOT/$s/Sequence_1" --stride 5 --iters "$ITERS" --max-gaussians 1000000 \
      --out "$MAPS/${s}_seq1.pt" 2>&1 | tail -3
  fi
  if [ ! -f "$FIELDS/${s}_lseg.pt" ]; then
    scripts/gpu_gate.sh 6000 python3 scripts/run_mapping.py --backend gsplat \
      --dataset replica_semnerf --data "$ROOT/$s/Sequence_1" --stride 5 --downscale "$DOWNSCALE" \
      --ply "$MAPS/${s}_seq1.pt" --extractor lseg --baselines --vocab configs/vocab_replica.json \
      --css-eval-data "$ROOT/$s/Sequence_2" --css-eval-stride 10 --feat-dim 64 --pca-frames 20 \
      --out "$FIELDS/${s}_lseg.pt" 2>&1 | grep -E "loaded|PCA|CSS|keyframes|timing|saved" || true
  fi
done

COMMON=(--vocab configs/vocab_replica.json --row-factory usplat.baselines:row_factory
        --extractor lseg --dataset replica_semnerf --stride 10 --downscale "$DOWNSCALE"
        --tau-cov "$TAU_COV" --text-cache "$FIELDS/text_cache_lseg.pt")
if [ "$STAGE" = val ]; then
  scripts/gpu_gate.sh 4000 python3 scripts/eval_seg.py "${COMMON[@]}" \
    --val-scenes configs/replica_val.json --scenes configs/replica_val.json --out "$OUT/eval_val.json"
else
  [ -f "$OUT/eval_val.json" ] || { echo "run STAGE=val first"; exit 1; }
  scripts/gpu_gate.sh 4000 python3 scripts/eval_seg.py "${COMMON[@]}" \
    --betas "$OUT/eval_val.json" --scenes configs/replica_test.json --out "$OUT/eval_test.json"
fi
echo "done: $STAGE -> $OUT"
