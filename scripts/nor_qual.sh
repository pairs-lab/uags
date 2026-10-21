#!/usr/bin/env bash
# nor_qual.sh — the NeRF-Object-Removal figure dumps (scripts/dump_nor_qual.py), one scene at a
# time through scripts/gpu_gate.sh (6500 MiB, 1 GiB headroom, poll 5 s), then the CPU steps.
#
#   STAGE=val  scripts/nor_qual.sh          # validation 001-004 (repeatable)
#   STAGE=test scripts/nor_qual.sh          # test scenes, once (dump_nor_qual.py refuses a
#                                           # finished test dump)
# The first failing dump (exit 2: Table III not reproduced) stops the loop.
# Every setting of a dump comes from the scene's Table III record; the tolerance file comes from
# the noise repeat (plot_nor_qual.py noise). Logs: $Q/logs/dump_<scene>.<timestamp>.log.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/venvs/uags/bin:$HOME/.local/cuda-12.8/bin:$PATH
export CUDA_HOME=$HOME/.local/cuda-12.8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5
D=${D:-/home/thanhnc/gauss_ws/data}
Q=${Q:-$D/uags_results/qual/nor}
T=${T:-$D/uags_results/nor/tab2_relmax}
STAGE=${STAGE:?STAGE=val or test}
SCENES=${SCENES:-}
FEATURES=" 002 006 013 015 "          # fp16 features for the figure scenes (qual_review step h)
case "$STAGE" in
  val)  SCENES=${SCENES:-001 002 003 004}; CURVES=--curves ;;
  test) SCENES=${SCENES:-005 006 007 008 009 010 011 013 014 015}; CURVES= ;;
  *) echo "STAGE must be val or test"; exit 2 ;;
esac
mkdir -p "$Q/logs"
TS=$(date +%Y%m%d-%H%M%S)
for s in $SCENES; do
  extra=()
  [[ "$FEATURES" == *" $s "* ]] && extra+=(--save-features)
  log=$Q/logs/dump_$s.$TS.log
  echo "[nor_qual] $(date '+%F %T') $STAGE $s git $(git rev-parse --short HEAD) -> $log"
  scripts/gpu_gate.sh 6500 python3 scripts/dump_nor_qual.py --table "$T/$STAGE/$s.json" \
    --tolerance "$Q/noise/tolerance.json" --vocab configs/vocab_replica.json \
    --text-cache "$D/uags_results/qual/text_cache_lseg_nor.pt" \
    --betas "$D/uags_results/tab1/eval_val.json" --stage "$STAGE" $CURVES "${extra[@]}" \
    --out "$Q/$s" > "$log" 2>&1
  rc=$?
  tail -1 "$log"
  if [ $rc -ne 0 ]; then
    echo "[nor_qual] $s failed (exit $rc): stopping"; exit $rc
  fi
done
echo "[nor_qual] $(date '+%F %T') $STAGE done"
