#!/usr/bin/env bash
# whole_update.sh — attempt 07, whole-update change scores on NeRF-Object-Removal (configs/whole_update_prereg.json).
#
#   STAGE=val   scripts/whole_update.sh     # validation 001-004 (every variant), then the selection
#   STAGE=test  scripts/whole_update.sh     # test scenes, once (only after an accepted selection)
#   STAGE=table scripts/whole_update.sh     # CPU: per-scene table and the adoption rule
#   DRY_RUN=1   prints the commands and runs only the gates
#
# Gates (every stage): the prereg and this attempt's code are tracked and committed unchanged;
# OUT is never a Table III directory. STAGE=test needs scripts/whole_update_select.py check-test
# (finished, reproducible and ACCEPTED validation selection). GPU jobs: scripts/gpu_gate.sh 6500
# with 1 GiB headroom, poll 5 s. The first failing scene (exit 2: Table III not reproduced)
# stops the loop.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/venvs/uags/bin:$HOME/.local/cuda-12.8/bin:$PATH
export CUDA_HOME=$HOME/.local/cuda-12.8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5
D=${D:-/home/thanhnc/gauss_ws/src/uags}
T=${T:-$D/results/nor/tab2_relmax}
Q=${Q:-$D/results/qual/nor}
OUT=${OUT:-$D/results/whole_update}
PREREG=configs/whole_update_prereg.json
STAGE=${STAGE:?STAGE=val, test or table}
DRY=${DRY_RUN:-0}
CODE="scripts/whole_update_nor.py scripts/whole_update_select.py scripts/whole_update.sh usplat/whole_update.py scripts/dump_nor_qual.py"

fail() { echo "[whole_update] REFUSED: $*"; exit 3; }
for f in $PREREG $CODE; do
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 || fail "$f is not tracked (commit it first)"
  git diff --quiet HEAD -- "$f" || fail "$f has uncommitted changes"
done
case "$(readlink -f "$OUT")" in
  "$(readlink -f "$T")"*|"$(readlink -f "$Q")"*) fail "OUT=$OUT is a Table III or dump directory" ;;
esac
mkdir -p "$OUT/logs"
SEL=$OUT/selection.json
TS=$(date +%Y%m%d-%H%M%S)
run_scenes() {        # $1 = val|test, $2 = variants, $3.. = scenes
  local st=$1 vs=$2; shift 2
  for s in "$@"; do
    local log=$OUT/logs/${st}_$s.$TS.log
    local cmd=(scripts/gpu_gate.sh 6500 python3 scripts/whole_update_nor.py
      --table "$T/$st/$s.json" --tolerance "$Q/noise/tolerance.json"
      --prereg $PREREG --stage "$st" --variants "$vs" --out "$OUT/$st/$s.json")
    echo "[whole_update] $(date '+%F %T') $st $s git $(git rev-parse --short HEAD) -> $log"
    if [ "$DRY" = 1 ]; then echo "  ${cmd[*]}"; continue; fi
    mkdir -p "$OUT/$st"
    "${cmd[@]}" > "$log" 2>&1
    local rc=$?
    tail -9 "$log"
    [ $rc -eq 0 ] || { echo "[whole_update] $s failed (exit $rc): stopping"; exit $rc; }
  done
}
case "$STAGE" in
  val)
    run_scenes val all 001 002 003 004
    [ "$DRY" = 1 ] && exit 0
    python3 scripts/whole_update_select.py select --prereg $PREREG --dir "$OUT/val" --out "$SEL" \
      | tee "$OUT/logs/select.$TS.log" ;;
  test)
    python3 scripts/whole_update_select.py check-test --prereg $PREREG --selection "$SEL" \
      --dir "$OUT/val" || exit 3
    CHOSEN=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['chosen'])" "$SEL")
    run_scenes test "$CHOSEN" 005 006 007 008 009 010 011 013 014 015 ;;
  table)
    python3 scripts/whole_update_select.py table --prereg $PREREG --selection "$SEL" \
      --test-dir "$OUT/test" --out-dir "$OUT/table" | tee "$OUT/logs/table.$TS.log" ;;
  *) fail "STAGE must be val, test or table" ;;
esac
echo "[whole_update] $(date '+%F %T') $STAGE done"
