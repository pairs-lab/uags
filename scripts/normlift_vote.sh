#!/usr/bin/env bash
# NormLift with its Sec. 3.4 KNN mode voting (row normlift_vote; configs/normlift_vote_prereg.json):
# the Table I / II evaluations on Replica and ScanNet (validation first: the temperature of every
# row is fitted there by NLL; then the test once with those temperatures) and the views
# experiment (Fig. views, Replica test). New record files only; the v1 records are never written.
# Each run also recomputes averaging and normlift: they must equal their v1 records (repro check).
#   scripts/normlift_vote.sh val | test | views | all
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-10}
R=/home/thanhnc/gauss_ws/src/uags/results
WORK=$R/normlift_vote                        # text-cache copies and logs (eval_seg adds prompts)
mkdir -p "$WORK/logs"
[ -f "$WORK/text_cache_replica.pt" ] || cp "$R/fields/text_cache_lseg.pt" "$WORK/text_cache_replica.pt"
[ -f "$WORK/text_cache_scannet.pt" ] || cp "$R/fields/scannet/text_cache_lseg.pt" "$WORK/text_cache_scannet.pt"
ROWS=averaging,normlift,normlift_vote
FACT=usplat.baselines:row_factory,usplat.baselines:normlift_vote_factory
NEED=${NEED:-7000}

replica() {   # replica <val|test>
  local split=$1 extra
  if [ "$split" = val ]; then extra="--val-scenes configs/replica_val.json --scenes configs/replica_val.json"
  else extra="--betas $R/tab1/eval_val_normlift_vote.json --scenes configs/replica_test.json"; fi
  [ -f "$R/tab1/eval_${split}_normlift_vote.json" ] && { echo "exists: tab1 $split"; return; }
  echo "[$(date '+%F %T')] replica $split"
  scripts/gpu_gate.sh "$NEED" python3 scripts/eval_seg.py --vocab configs/vocab_replica.json \
    --row-factory "$FACT" --extractor lseg --dataset replica_semnerf --stride 10 --downscale 2 \
    --tau-cov 0.5 --text-cache "$WORK/text_cache_replica.pt" --rows "$ROWS" $extra \
    --out "$R/tab1/eval_${split}_normlift_vote.json" > "$WORK/logs/replica_${split}.log" 2>&1 \
    || { tail -30 "$WORK/logs/replica_${split}.log"; exit 1; }
  grep -E "normlift_vote: replaced" "$WORK/logs/replica_${split}.log" || true
}

scannet() {   # scannet <val|test>
  local split=$1 extra
  if [ "$split" = val ]; then extra="--val-scenes configs/scannet_val.json --scenes configs/scannet_val.json"
  else extra="--betas $R/scannet_tab1/eval_val_normlift_vote.json --scenes configs/scannet_test.json"; fi
  [ -f "$R/scannet_tab1/eval_${split}_normlift_vote.json" ] && { echo "exists: scannet $split"; return; }
  echo "[$(date '+%F %T')] scannet $split"
  scripts/gpu_gate.sh "$NEED" python3 scripts/eval_seg.py --vocab configs/vocab_scannet.json \
    --row-factory "$FACT" --extractor lseg --dataset scannet --stride 20 --downscale 2 \
    --tau-cov 0.5 --text-cache "$WORK/text_cache_scannet.pt" --rows "$ROWS" $extra \
    --out "$R/scannet_tab1/eval_${split}_normlift_vote.json" > "$WORK/logs/scannet_${split}.log" 2>&1 \
    || { tail -30 "$WORK/logs/scannet_${split}.log"; exit 1; }
  grep -E "normlift_vote: replaced" "$WORK/logs/scannet_${split}.log" || true
}

views() {     # the views experiment on the Replica test scenes, k-view fields re-mapped
  echo "[$(date '+%F %T')] views test"
  OUT=$R/views_normlift_vote STAGE=test ROWS=$ROWS ROW_FACTORY=$FACT \
    BETAS=$R/tab1/eval_val_normlift_vote.json scripts/replica_views.sh
}

case "${1:-all}" in
  val)   replica val; scannet val ;;
  test)  replica test; scannet test ;;
  views) views ;;
  all)   replica val; scannet val; replica test; scannet test; views ;;
  *) echo "usage: $0 val|test|views|all"; exit 2 ;;
esac
echo "[$(date '+%F %T')] done: ${1:-all}"
