#!/usr/bin/env bash
# cusum_reset.sh — attempt 06, the CUSUM change test on NeRF-Object-Removal (configs/cusum_reset_prereg.json).
#
#   STAGE=val   scripts/cusum_reset.sh     # validation 001-004, then the selection
#   STAGE=test  scripts/cusum_reset.sh     # test scenes, once (only after an accepted selection)
#   STAGE=table scripts/cusum_reset.sh     # CPU: per-scene table, object-query split, adoption rule
#   DRY_RUN=1   prints the commands and runs only the gates
#
# Gates (every stage): the prereg and this attempt's code are tracked and committed unchanged;
# OUT is never a Table III or dump directory; the text cache is a COPY in OUT. STAGE=test needs
# scripts/cusum_reset_select.py check-test (finished, reproducible and ACCEPTED validation
# selection). GPU jobs: scripts/gpu_gate.sh 6500 with 1 GiB headroom, poll 5 s. The first failing
# scene (exit 2: Table III or the dump's N2 not reproduced) stops the loop.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/venvs/uags/bin:$HOME/.local/cuda-12.8/bin:$PATH
export CUDA_HOME=$HOME/.local/cuda-12.8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5
D=${D:-/home/thanhnc/gauss_ws/src/uags}
T=${T:-$D/results/nor/tab2_relmax}
Q=${Q:-$D/results/qual/nor}
OUT=${OUT:-$D/results/cusum_reset}
PREREG=configs/cusum_reset_prereg.json
CACHE_SRC=${CACHE_SRC:-$D/results/qual/text_cache_lseg_nor.pt}
STAGE=${STAGE:?STAGE=val, test or table}
DRY=${DRY_RUN:-0}
CODE="scripts/cusum_reset_nor.py scripts/cusum_reset_select.py scripts/cusum_reset.sh usplat/cusum_reset.py scripts/reset_fused_nor.py usplat/reset_fused.py scripts/dump_nor_qual.py"

fail() { echo "[cusum_reset] REFUSED: $*"; exit 3; }
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
run_scenes() {        # $1 = val|test, $2 = variants, $3.. = scenes
  local st=$1 vs=$2; shift 2
  for s in "$@"; do
    local log=$OUT/logs/${st}_$s.$TS.log
    local cmd=(scripts/gpu_gate.sh 6500 python3 scripts/cusum_reset_nor.py
      --table "$T/$st/$s.json" --tolerance "$Q/noise/tolerance.json"
      --vocab configs/vocab_replica.json --text-cache "$CACHE" --dump "$Q/$s"
      --prereg $PREREG --stage "$st" --variants "$vs" --out "$OUT/$st/$s.json")
    echo "[cusum_reset] $(date '+%F %T') $st $s git $(git rev-parse --short HEAD) -> $log"
    if [ "$DRY" = 1 ]; then echo "  ${cmd[*]}"; continue; fi
    mkdir -p "$OUT/$st"
    "${cmd[@]}" > "$log" 2>&1
    local rc=$?
    tail -5 "$log"
    [ $rc -eq 0 ] || { echo "[cusum_reset] $s failed (exit $rc): stopping"; exit $rc; }
  done
}
case "$STAGE" in
  val)
    run_scenes val all 001 002 003 004
    [ "$DRY" = 1 ] && exit 0
    python3 scripts/cusum_reset_select.py select --prereg $PREREG --dir "$OUT/val" --out "$SEL" \
      | tee "$OUT/logs/select.$TS.log" ;;
  test)
    python3 scripts/cusum_reset_select.py check-test --prereg $PREREG --selection "$SEL" \
      --dir "$OUT/val" || exit 3
    CHOSEN=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['chosen'])" "$SEL")
    run_scenes test "$CHOSEN" 005 006 007 008 009 010 011 013 014 015 ;;
  table)
    python3 scripts/cusum_reset_select.py table --prereg $PREREG --selection "$SEL" \
      --test-dir "$OUT/test" --out-dir "$OUT/table" | tee "$OUT/logs/table.$TS.log" ;;
  *) fail "STAGE must be val, test or table" ;;
esac
echo "[cusum_reset] $(date '+%F %T') $STAGE done"
