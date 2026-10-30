#!/usr/bin/env bash
# reset_fused.sh — the fused-reset attempt on NeRF-Object-Removal (configs/reset_fused_prereg.json).
#
#   STAGE=val   scripts/reset_fused.sh     # validation 001-004, then the selection
#   STAGE=test  scripts/reset_fused.sh     # test scenes, once (after check-test passes)
#   STAGE=table scripts/reset_fused.sh     # CPU: per-scene table, claim rule, verdict
#   DRY_RUN=1   prints the commands and runs only the gates
#
# Gates (every stage): the prereg and this attempt's code are tracked and committed unchanged;
# OUT is never a Table III or dump directory; the text cache is a COPY in OUT (made once from
# the dumps' copy, never written by the runs). STAGE=test additionally needs
# scripts/reset_fused_select.py check-test (finished, reproducible validation selection made
# with this prereg). GPU jobs: scripts/gpu_gate.sh 6500 with 1 GiB headroom, poll 5 s. The first
# failing scene (exit 2: Table III or the dump's N2 not reproduced) stops the loop.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/venvs/uags/bin:$HOME/.local/cuda-12.8/bin:$PATH
export CUDA_HOME=$HOME/.local/cuda-12.8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5
D=${D:-/home/thanhnc/gauss_ws/src/uags}
T=${T:-$D/results/nor/tab2_relmax}
Q=${Q:-$D/results/qual/nor}
OUT=${OUT:-$D/results/reset_fused}
PREREG=configs/reset_fused_prereg.json
CACHE_SRC=${CACHE_SRC:-$D/results/qual/text_cache_lseg_nor.pt}
STAGE=${STAGE:?STAGE=val, test or table}
DRY=${DRY_RUN:-0}
CODE="scripts/reset_fused_nor.py scripts/reset_fused_select.py scripts/reset_fused.sh usplat/reset_fused.py scripts/dump_nor_qual.py"

fail() { echo "[reset_fused] REFUSED: $*"; exit 3; }
for f in $PREREG $CODE; do
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 || fail "$f is not tracked (commit it first)"
  git diff --quiet HEAD -- "$f" || fail "$f has uncommitted changes"
done
case "$(readlink -f "$OUT")" in
  "$(readlink -f "$T")"*|"$(readlink -f "$Q")"*) fail "OUT=$OUT is a Table III or dump directory" ;;
esac
mkdir -p "$OUT/logs"
CACHE=$OUT/text_cache_lseg_nor.pt
if [ ! -f "$CACHE" ]; then
  [ -f "$CACHE_SRC" ] || fail "no text cache at $CACHE_SRC"
  cp "$CACHE_SRC" "$CACHE"
fi
SEL=$OUT/selection.json
TS=$(date +%Y%m%d-%H%M%S)

run_scenes() {        # $1 = val|test, $2.. = scenes
  local st=$1; shift
  for s in "$@"; do
    local log=$OUT/logs/${st}_$s.$TS.log
    local cmd=(scripts/gpu_gate.sh 6500 python3 scripts/reset_fused_nor.py
      --table "$T/$st/$s.json" --tolerance "$Q/noise/tolerance.json"
      --vocab configs/vocab_replica.json --text-cache "$CACHE" --dump "$Q/$s"
      --prereg $PREREG --stage "$st" --out "$OUT/$st/$s.json")
    echo "[reset_fused] $(date '+%F %T') $st $s git $(git rev-parse --short HEAD) -> $log"
    if [ "$DRY" = 1 ]; then echo "  ${cmd[*]}"; continue; fi
    "${cmd[@]}" > "$log" 2>&1
    local rc=$?
    tail -2 "$log"
    [ $rc -eq 0 ] || { echo "[reset_fused] $s failed (exit $rc): stopping"; exit $rc; }
  done
}

case "$STAGE" in
  val)
    run_scenes val 001 002 003 004
    [ "$DRY" = 1 ] && exit 0
    python3 scripts/reset_fused_select.py select --prereg $PREREG --dir "$OUT/val" --out "$SEL" \
      | tee "$OUT/logs/select.$TS.log" ;;
  test)
    python3 scripts/reset_fused_select.py check-test --prereg $PREREG --selection "$SEL" \
      --dir "$OUT/val" || exit 3
    run_scenes test 005 006 007 008 009 010 011 013 014 015 ;;
  table)
    python3 scripts/reset_fused_select.py table --prereg $PREREG --selection "$SEL" \
      --test-dir "$OUT/test" --out-dir "$OUT/table" | tee "$OUT/logs/table.$TS.log" ;;
  *) fail "STAGE must be val, test or table" ;;
esac
echo "[reset_fused] $(date '+%F %T') $STAGE done"
