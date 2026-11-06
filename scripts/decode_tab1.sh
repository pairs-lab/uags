#!/usr/bin/env bash
# decode_tab1.sh — attempt 08 (configs/decoding_prereg.json): the label-free decision rules of
# usplat/decoding.py for every Table I row, on the validation scenes and then, once, on the test
# scenes, through scripts/gpu_gate.sh (scripts/decode_tab1.py, scripts/decode_select.py).
#   STAGE=val   : both datasets' validation scenes, then the selection -> $OUT/selection.json
#   STAGE=test  : check-test (registration and code committed, selection accepted, no test yet),
#                 then both datasets' test scenes, once
#   STAGE=table : the test table and the adoption verdict -> $OUT/table
# Every stage is refused while the prereg or this attempt's code is untracked or has uncommitted
# changes. A run whose base rule does not reproduce the published record stops (exit 2).
# DRY_RUN=1 prints the commands and runs nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_HEADROOM_MIB=${UAGS_GPU_HEADROOM_MIB:-1024} UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-5}
P=configs/decoding_prereg.json
OUT=${OUT:-results/decoding}
STAGE=${STAGE:-val}
NEED=${NEED:-7000}
TS=$(date +%Y%m%d-%H%M%S)
run() { echo "+ $*"; if [ "${DRY_RUN:-0}" != 1 ]; then "$@"; fi; }
gate() {
  for f in "$P" scripts/decode_tab1.py scripts/decode_select.py scripts/decode_tab1.sh usplat/decoding.py; do
    git ls-files --error-unmatch "$f" > /dev/null 2>&1 || { echo "refused: $f is not tracked"; exit 1; }
    git diff --quiet HEAD -- "$f" || { echo "refused: $f has uncommitted changes"; exit 1; }
  done
}
stage_run() {   # $1 = val | test
  for ds in replica scannet; do
    log="$OUT/logs/$1_$ds.$TS.log"
    echo "[decoding] $(date '+%F %T') $1 $ds git $(git rev-parse --short HEAD) -> $log"
    if [ "${DRY_RUN:-0}" = 1 ]; then
      echo "+ scripts/gpu_gate.sh $NEED python3 scripts/decode_tab1.py --dataset $ds --stage $1 --prereg $P --out $OUT/${ds}_$1.json"
    else
      scripts/gpu_gate.sh "$NEED" python3 scripts/decode_tab1.py --dataset "$ds" --stage "$1" \
        --prereg "$P" --out "$OUT/${ds}_$1.json" 2>&1 | tee "$log"
    fi
  done
}
mkdir -p "$OUT/logs"
case "$STAGE" in
  val)
    gate
    stage_run val
    run python3 scripts/decode_select.py select --prereg "$P" --dir "$OUT" --out "$OUT/selection.json" ;;
  test)
    gate
    run python3 scripts/decode_select.py check-test --prereg "$P" --selection "$OUT/selection.json" --dir "$OUT"
    stage_run test ;;
  table)
    run python3 scripts/decode_select.py table --prereg "$P" --selection "$OUT/selection.json" \
      --dir "$OUT" --out-dir "$OUT/table" ;;
  *) echo "STAGE must be val, test or table (got '$STAGE')"; exit 2 ;;
esac
echo "[decoding] $(date '+%F %T') $STAGE done"
