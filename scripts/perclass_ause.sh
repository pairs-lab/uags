#!/usr/bin/env bash
# Per-class error ranking (configs/perclass_ause_prereg.json): the Table I test evaluations of
# Replica and ScanNet again, with eval_seg.py --per-class-ranking, for the rows that output an
# uncertainty. The temperatures are read from the Table I validation records (no fit, no
# selection). New record files only (results/perclass_ause/); Table I's records are never written.
# Control: every recomputed per-class IoU, accuracy and AUSE must equal the Table I record
# (papers/table_tools/perclass_ause_table.py refuses otherwise).
#   scripts/perclass_ause.sh replica | scannet | all
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-10}
R=/home/thanhnc/gauss_ws/src/uags/results
OUT=$R/perclass_ause
mkdir -p "$OUT/logs"
[ -f "$OUT/text_cache_replica.pt" ] || cp "$R/fields/text_cache_lseg.pt" "$OUT/text_cache_replica.pt"
[ -f "$OUT/text_cache_scannet.pt" ] || cp "$R/fields/scannet/text_cache_lseg.pt" "$OUT/text_cache_scannet.pt"
FACT=usplat.baselines:row_factory,usplat.distill:row_factory
NEED=${NEED:-7000}               # --with-2d: LSeg fp32 on the evaluation frames too
if [ -n "$(git status --porcelain --untracked-files=no -- scripts usplat configs)" ]; then
  echo "refused: uncommitted changes in scripts/ usplat/ configs/ (the run must name a commit)"; exit 2
fi
echo "[$(date '+%F %T')] per-class ranking, git $(git rev-parse --short HEAD)"

run() {       # run <name> <vocab> <dataset> <stride> <text cache> <betas> <scenes> <rows>
  local name=$1
  [ -f "$OUT/${name}_test.json" ] && { echo "exists: $name"; return; }
  echo "[$(date '+%F %T')] $name test"
  scripts/gpu_gate.sh "$NEED" python3 scripts/eval_seg.py --vocab "$2" --row-factory "$FACT" \
    --extractor lseg --dataset "$3" --stride "$4" --downscale 2 --tau-cov 0.5 --text-cache "$5" \
    --betas "$6" --scenes "$7" --rows "$8" --with-2d --per-class-ranking \
    --out "$OUT/${name}_test.json" > "$OUT/logs/${name}_test.log" 2>&1 \
    || { tail -30 "$OUT/logs/${name}_test.log"; exit 1; }
}

replica() { run replica configs/vocab_replica.json replica_semnerf 10 "$OUT/text_cache_replica.pt" \
  "$R/tab1/eval_val.json" configs/replica_test.json ours_P0,ours_P2,normlift,viewcount,f3dgs,legs_l4,css_a0.01,2d_features; }
scannet() { run scannet configs/vocab_scannet.json scannet 20 "$OUT/text_cache_scannet.pt" \
  "$R/scannet_tab1/eval_val.json" configs/scannet_test.json ours_P0,ours_P2,normlift,viewcount,f3dgs,legs_l1,css_a0.1,2d_features; }

case "${1:-all}" in
  replica) replica ;;
  scannet) scannet ;;
  all) replica; scannet ;;
  *) echo "usage: $0 replica|scannet|all"; exit 2 ;;
esac
echo "[$(date '+%F %T')] done: ${1:-all}"
