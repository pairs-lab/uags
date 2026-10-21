#!/usr/bin/env bash
# video_gpu.sh — the GPU dumps of the supplementary video, through scripts/gpu_gate.sh.
#   scripts/video_gpu.sh [nor002|nor014|static]...   (default: all three, in this order)
# Stops at the first failure. Logs: $V/logs/<job>.<stamp>.log (header: time, git, sha256).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/venvs/uags/bin:$HOME/.local/cuda-12.8/bin:$PATH CUDA_HOME=$HOME/.local/cuda-12.8
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5
D=/home/thanhnc/gauss_ws/data; R=$D/uags_results; V=$R/video; Q=$R/qual/nor
stamp=$(date +%Y%m%d-%H%M%S)
jobs=("$@"); [ ${#jobs[@]} -eq 0 ] && jobs=(nor002 nor014 static)
hdr() { echo "# $(date '+%F %T') git $(git rev-parse --short HEAD) job $1"; sha256sum "$2"; }
for j in "${jobs[@]}"; do
  log=$V/logs/$j.$stamp.log
  case $j in
    nor002|nor014)
      s=${j#nor}; st=val; [ "$s" = 014 ] && st=test
      ref=(); [ -f "$Q/$s/curves.json" ] && ref=(--curves-ref "$Q/$s/curves.json")
      { hdr "$j" scripts/video_nor_timeline.py
        scripts/gpu_gate.sh 6500 python3 scripts/video_nor_timeline.py \
          --table "$R/nor/tab2_relmax/$st/$s.json" --dump "$Q/$s" \
          --tolerance "$Q/noise/tolerance.json" --vocab configs/vocab_replica.json \
          --text-cache "$V/text_cache_lseg_nor.pt" --betas "$R/tab1/eval_val.json" \
          "${ref[@]}" --out "$V/nor/$s"; } 2>&1 | tee "$log" ;;
    static)
      { hdr "$j" scripts/video_static_path.py
        scripts/gpu_gate.sh 4500 python3 scripts/video_static_path.py \
          --vocab configs/vocab_replica.json --extractor lseg --scenes configs/replica_test.json \
          --scene room_0 --betas "$R/tab1/eval_val.json" \
          --text-cache "$V/text_cache_lseg_replica.pt" \
          --kfield "$R/qual/kfields/room_0_lseg_k30.pt" --path 230:351 --path-downscale 1 \
          --check-frame 29 --check-meta "$R/qual/qual_room_0_f29.json" \
          --out "$V/static/room_0_path.npz"; } 2>&1 | tee "$log" ;;
    *) echo "unknown job $j"; exit 1 ;;
  esac
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "$j failed"; exit 1; }
done
