#!/usr/bin/env bash
# Attempt 09 (configs/robust_prereg.json): the trimmed view mean (usplat.baselines.RobustLift) on the
# Table I protocol. Same maps, frames, LSeg, PCA, coverage and evaluation code as replica_tab1.sh and
# scannet_tab1.sh; the fields are re-mapped with --robust-lift into results/fields_robust (the published
# fields are never written).
#   STAGE=val  : map + evaluate the 4 validation scenes, then the registered selection
#                (results/robust/val/selection.json)
#   STAGE=test : refused unless the validation accepted a variant; then the 14 test scenes, once
# Every GPU step goes through scripts/gpu_gate.sh. Logs: results/robust/<stage>/*.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_USGLOC_EXTRA_MIB=${UAGS_GPU_USGLOC_EXTRA_MIB:-99999}
export UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-5}
STAGE=${STAGE:-val}
R=/home/thanhnc/gauss_ws/src/uags/results
FR=$R/fields_robust
OUT=$R/robust/$STAGE
ROOT=/home/thanhnc/gauss_ws/src/uags/data/replica_semnerf/Replica_Dataset
PREP=/home/thanhnc/gauss_ws/src/uags/data/scannet/uags_f2
MAP_NEED=${MAP_NEED:-9000}
EVAL_NEED=${EVAL_NEED:-7000}
mkdir -p "$FR/scannet" "$OUT"
[ "$STAGE" = test ] && python3 scripts/robust_select.py check-test "$R/robust/val/selection.json"
C0=$(python3 -c "import json; print(','.join(f'{c:g}' for c in json.load(open('configs/robust_prereg.json'))['variants']['c0']))")
TS=$(date +%Y%m%d-%H%M%S)
echo "# $(date '+%F %T') git $(git rev-parse --short HEAD) stage $STAGE c0 $C0" | tee "$OUT/driver.$TS.log"

REP=$(python3 scripts/robust_select.py scenes replica "$STAGE" "$OUT/replica_scenes.json")
SCA=$(python3 scripts/robust_select.py scenes scannet "$STAGE" "$OUT/scannet_scenes.json")
for s in $REP; do
  [ -f "$FR/${s}_lseg.pt" ] && continue
  scripts/gpu_gate.sh "$MAP_NEED" python3 scripts/run_mapping.py --backend gsplat \
    --dataset replica_semnerf --data "$ROOT/$s/Sequence_1" --stride 5 --downscale 2 \
    --ply "$R/maps/${s}_seq1.pt" --extractor lseg --baselines --vocab configs/vocab_replica.json \
    --css-eval-data "$ROOT/$s/Sequence_2" --css-eval-stride 10 --feat-dim 64 --pca-frames 20 \
    --seed 0 --robust-lift "$C0" --out "$FR/${s}_lseg.pt" > "$OUT/map_${s}.$TS.log" 2>&1 \
    || { tail -20 "$OUT/map_${s}.$TS.log"; exit 1; }
  grep -E "RobustLift|keyframes used|saved" "$OUT/map_${s}.$TS.log" | tee -a "$OUT/driver.$TS.log"
done
for s in $SCA; do
  [ -f "$FR/scannet/${s}_lseg.pt" ] && continue
  scripts/gpu_gate.sh "$MAP_NEED" python3 scripts/run_mapping.py --backend gsplat \
    --dataset scannet --data "$PREP/$s" --stride 10 --downscale 2 \
    --ply "$R/maps/scannet/$s.pt" --extractor lseg --baselines --vocab configs/vocab_scannet.json \
    --css-eval-data "$PREP/$s/eval_s20o5" --css-eval-stride 1 --feat-dim 64 --pca-frames 20 \
    --seed 0 --robust-lift "$C0" --out "$FR/scannet/${s}_lseg.pt" > "$OUT/map_${s}.$TS.log" 2>&1 \
    || { tail -20 "$OUT/map_${s}.$TS.log"; exit 1; }
  grep -E "RobustLift|keyframes used|saved" "$OUT/map_${s}.$TS.log" | tee -a "$OUT/driver.$TS.log"
done

# evaluation: Table I's evaluator on the robust fields (text caches: copies)
cp -n "$R/fields/text_cache_lseg.pt" "$OUT/text_cache_replica.pt"
cp -n "$R/fields/scannet/text_cache_lseg.pt" "$OUT/text_cache_scannet.pt"
if [ "$STAGE" = val ]; then
  RA=(--val-scenes "$OUT/replica_scenes.json" --scenes "$OUT/replica_scenes.json")
  SA=(--val-scenes "$OUT/scannet_scenes.json" --scenes "$OUT/scannet_scenes.json")
else
  RA=(--betas "$R/robust/val/replica_eval.json" --scenes "$OUT/replica_scenes.json")
  SA=(--betas "$R/robust/val/scannet_eval.json" --scenes "$OUT/scannet_scenes.json")
fi
scripts/gpu_gate.sh "$EVAL_NEED" python3 scripts/eval_seg.py --vocab configs/vocab_replica.json \
  --row-factory usplat.baselines:row_factory --extractor lseg --dataset replica_semnerf --stride 10 \
  --downscale 2 --tau-cov 0.5 --text-cache "$OUT/text_cache_replica.pt" "${RA[@]}" \
  --out "$OUT/replica_eval.json" > "$OUT/eval_replica.$TS.log" 2>&1 || { tail -30 "$OUT/eval_replica.$TS.log"; exit 1; }
scripts/gpu_gate.sh "$EVAL_NEED" python3 scripts/eval_seg.py --vocab configs/vocab_scannet.json \
  --row-factory usplat.baselines:row_factory --extractor lseg --dataset scannet --stride 20 \
  --downscale 2 --tau-cov 0.5 --text-cache "$OUT/text_cache_scannet.pt" "${SA[@]}" \
  --out "$OUT/scannet_eval.json" > "$OUT/eval_scannet.$TS.log" 2>&1 || { tail -30 "$OUT/eval_scannet.$TS.log"; exit 1; }
[ "$STAGE" = val ] && python3 scripts/robust_select.py select "$OUT/replica_eval.json" "$OUT/scannet_eval.json" \
  "$OUT/selection.json" | tee -a "$OUT/driver.$TS.log"
echo "done: $STAGE -> $OUT"
