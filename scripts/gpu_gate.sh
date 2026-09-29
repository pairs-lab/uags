#!/usr/bin/env bash
# gpu_gate.sh — run a UAGS GPU job only when it cannot disturb the shared GPU.
#
#   scripts/gpu_gate.sh <need_mib> <command...>
#
# Waits (polling every UAGS_GPU_POLL_S seconds, default 60) until
#   * free GPU memory >= need_mib + headroom (UAGS_GPU_HEADROOM_MIB, default 2048), plus
#     UAGS_GPU_USGLOC_EXTRA_MIB (default 0) while a USGLoc job is running. The user allowed
#     UAGS jobs to run in PARALLEL with USGLoc (2026-09-29); set the extra to 3072 to be
#     conservative again (USGLoc's MASt3R stages peak at ~6.9 GB), and
#   * no other UAGS job holds the lock file (flock), and
#   * the free-memory condition still holds AFTER the lock is taken,
# then runs the command under the lock with UAGS_GPU_MEM_FRACTION = need/total exported
# (scripts/run_mapping.py applies it with torch.cuda.set_per_process_memory_fraction).
# It never kills or signals other processes. A failed nvidia-smi reading counts as
# "not enough memory" and the gate keeps waiting. The jobs reserve their memory at start
# (usplat/gpu_reserve.py) so a USGLoc batch cannot start in their CPU phases.
set -uo pipefail
need=${1:?need_mib}; shift
headroom=${UAGS_GPU_HEADROOM_MIB:-2048}
usgloc_extra=${UAGS_GPU_USGLOC_EXTRA_MIB:-0}
lock=${UAGS_GPU_LOCK:-/tmp/uags_gpu.lock}
poll=${UAGS_GPU_POLL_S:-60}
export CUDA_HOME=${CUDA_HOME:-$HOME/.local/cuda-12.8}
export PATH=$CUDA_HOME/bin:$PATH

num() { [[ $1 =~ ^[0-9]+$ ]] && echo "$1" || echo 0; }
free_mib() { num "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"; }
total_mib() { num "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"; }
# USGLoc jobs: any python process whose working directory is the USGLoc repo (covers every
# script it runs: run_7scenes, train_locfeat, train_map, ...), checked through /proc
usgloc_dir=${UAGS_USGLOC_DIR:-/home/thanhnc/gauss_ws/src/usgloc}
usgloc_running() {
  local p
  for p in $(pgrep -x python3) $(pgrep -x python); do
    [ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$usgloc_dir" ] && return 0
  done
  return 1
}
required() {
  local r=$((need + headroom))
  if usgloc_running; then r=$((r + usgloc_extra)); fi
  echo $r
}

exec 9>"$lock"
while true; do
  f=$(free_mib); r=$(required)
  if [ "$f" -ge "$r" ] && flock -n 9; then
    f=$(free_mib); r=$(required)                  # re-check under the lock
    if [ "$f" -ge "$r" ]; then
      t=$(total_mib)
      if [ "$t" -gt 0 ]; then
        export UAGS_GPU_MEM_FRACTION=$(python3 -c "print(min(1.0, $need / $t))")
      fi
      echo "[gpu_gate] $(date '+%F %T') free ${f} MiB >= ${r}; running: $*"
      rc=0; "$@" || rc=$?
      flock -u 9
      exit $rc
    fi
    flock -u 9
  fi
  echo "[gpu_gate] $(date '+%F %T') waiting: free ${f} MiB < ${r} or lock busy"
  sleep "$poll"
done
