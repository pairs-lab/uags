#!/usr/bin/env bash
# qual_compare.sh — the chain of fig:qual-compare and supp_compare_all (scripts/qual_compare.py).
#
#   scripts/qual_compare.sh <step>
#     select      pick_frames.py static on the test and validation scenes (CPU, rule check)
#     val-replica tokens + render of the validation scenes (development; GPU)
#     val-scannet
#     test-main   tokens + render of room_1 f66 and scene0140_00 f57 (GPU, once)
#     test-supp   tokens + render of the other 12 test scenes (GPU, once)
#     figures     the paper figure, the supplementary grid, the confident-error listing (CPU)
#
# Every GPU step runs behind scripts/gpu_gate.sh at 4500 MiB (headroom 1024, poll 5 s); the
# token step (LSeg) and the render step (gsplat) are separate gated processes. Each run writes
# one log under $C/logs with a header: time, git HEAD, sha256 of qual_compare.py, purpose.
# Text caches are copies (eval_seg writes new prompts into --text-cache), never the originals.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/venvs/uags/bin:$HOME/.local/cuda-12.8/bin:$PATH
export CUDA_HOME=$HOME/.local/cuda-12.8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
D=${UAGS_DATA:-/home/thanhnc/gauss_ws/data}
R=$D/uags_results
C=$R/qual/compare
GATE="env UAGS_GPU_HEADROOM_MIB=1024 UAGS_GPU_POLL_S=5 scripts/gpu_gate.sh ${QC_GATE_MIB:-4500}"
RF=usplat.baselines:row_factory,usplat.distill:row_factory
mkdir -p "$C/logs" "$C/val" "$C/test" "$R/qual/select"

header() { echo "[run] $(date '+%F %T') git $(git rev-parse --short HEAD) qual_compare.py sha256 $(sha256sum scripts/qual_compare.py | cut -c1-12) purpose=$1"; }

# dataset settings: <vocab> <scenes> <betas> <table> <legs> <cache> <label> <dataset> <stride>
replica_val=(configs/vocab_replica.json configs/replica_val.json $R/tab1/eval_val.json $R/tab1/eval_val.json legs_l4 $R/qual/text_cache_lseg_replica.pt Replica replica_semnerf 10)
replica_test=(configs/vocab_replica.json configs/replica_test.json $R/tab1/eval_val.json $R/tab1/eval_test.json legs_l4 $R/qual/text_cache_lseg_replica.pt Replica replica_semnerf 10)
scannet_val=(configs/vocab_scannet.json configs/scannet_val.json $R/scannet_tab1/eval_val.json $R/scannet_tab1/eval_val.json legs_l1 $R/qual/text_cache_lseg_scannet.pt ScanNet scannet 20)
scannet_test=(configs/vocab_scannet.json configs/scannet_test.json $R/scannet_tab1/eval_val.json $R/scannet_tab1/eval_test.json legs_l1 $R/qual/text_cache_lseg_scannet.pt ScanNet scannet 20)

tokens() {   # <split> <settings array name> <targets> <purpose>
  local -n s=$2; local log=$C/logs/$1_tokens_$(echo "$3" | tr ',:' '_-').log
  header "$4" > "$log"
  { time $GATE python3 scripts/qual_compare.py --stage tokens --vocab "${s[0]}" \
      --scenes "${s[1]}" --extractor lseg --dataset "${s[7]}" --stride "${s[8]}" --downscale 2 \
      --target "$3" --out-dir "$C/$1"; } >> "$log" 2>&1
  tail -3 "$log"
}

render() {   # <split> <settings array name> <scene> <frame> <select json> <purpose>
  local -n s=$2; local log=$C/logs/$1_render_$3.log
  header "$6" > "$log"
  { time $GATE python3 scripts/qual_compare.py --stage render --vocab "${s[0]}" \
      --scenes "${s[1]}" --extractor lseg --dataset "${s[7]}" --stride "${s[8]}" --downscale 2 \
      --tau-cov 0.5 --scene "$3" --frame "$4" --select "$5" --legs "${s[4]}" --betas "${s[2]}" \
      --table "${s[3]}" --text-cache "${s[5]}" --row-factory $RF --dataset-label "${s[6]}" \
      --out-dir "$C/$1"; } >> "$log" 2>&1
  tail -3 "$log"
}

case "${1:?step}" in
  select)
    python3 scripts/pick_frames.py static --check \
      --set replica=configs/replica_test.json,configs/vocab_replica.json,$R/tab1/eval_test.json \
      --set scannet=configs/scannet_test.json,configs/vocab_scannet.json,$R/scannet_tab1/eval_test.json \
      --out $R/qual/select/static.json 2>&1 | tee $C/logs/pick_frames_static_test.log
    python3 scripts/pick_frames.py static \
      --set replica_val=configs/replica_val.json,configs/vocab_replica.json \
      --set scannet_val=configs/scannet_val.json,configs/vocab_scannet.json \
      --out $R/qual/select/static_val.json 2>&1 | tee $C/logs/pick_frames_static_val.log ;;
  val-replica)
    V=$R/qual/select/static_val.json
    tokens val replica_val office_0:29,room_2:14 "validation tokens"
    render val replica_val office_0 29 $V "validation render"
    render val replica_val room_2 14 $V "validation render" ;;
  val-scannet)
    V=$R/qual/select/static_val.json
    tokens val scannet_val scene0000_00:150,scene0062_00:16 "validation tokens"
    render val scannet_val scene0000_00 150 $V "validation render"
    render val scannet_val scene0062_00 16 $V "validation render" ;;
  test-main)
    T=$R/qual/select/static.json
    tokens test replica_test room_1:66 "TEST tokens, main figure"
    tokens test scannet_test scene0140_00:57 "TEST tokens, main figure"
    render test replica_test room_1 66 $T "TEST render, main figure"
    render test scannet_test scene0140_00 57 $T "TEST render, main figure" ;;
  test-supp)
    T=$R/qual/select/static.json
    tokens test replica_test room_0:18,office_1:25,office_2:75,office_3:15,office_4:29 "TEST tokens, supplementary"
    tokens test scannet_test scene0070_00:46,scene0097_00:2,scene0200_00:36,scene0347_00:0,scene0400_00:0,scene0590_00:104,scene0645_00:156 "TEST tokens, supplementary"
    for sf in room_0:18 office_1:25 office_2:75 office_3:15 office_4:29; do
      render test replica_test ${sf%%:*} ${sf##*:} $T "TEST render, supplementary"; done
    for sf in scene0070_00:46 scene0097_00:2 scene0200_00:36 scene0347_00:0 scene0400_00:0 scene0590_00:104 scene0645_00:156; do
      render test scannet_test ${sf%%:*} ${sf##*:} $T "TEST render, supplementary"; done ;;
  figures)
    t=$C/test
    python3 scripts/qual_compare.py --stage figure --vocab configs/vocab_replica.json \
      --blocks $t/room_1_f66.pt,$t/scene0140_00_f57.pt --out papers/figs/qual_compare.pdf \
      --meta $C/qual_compare_figure.json 2>&1 | tee $C/logs/figure_main.log
    all=$t/room_0_f18.pt,$t/room_1_f66.pt,$t/office_1_f25.pt,$t/office_2_f75.pt,$t/office_3_f15.pt,$t/office_4_f29.pt,$t/scene0070_00_f46.pt,$t/scene0097_00_f2.pt,$t/scene0140_00_f57.pt,$t/scene0200_00_f36.pt,$t/scene0347_00_f0.pt,$t/scene0400_00_f0.pt,$t/scene0590_00_f104.pt,$t/scene0645_00_f156.pt
    python3 scripts/qual_compare.py --stage figure --vocab configs/vocab_replica.json \
      --blocks $all --per-page 2 --out papers/figs/supp_compare_all.pdf --png-dir $C/supp_png \
      --meta $C/supp_compare_all_figure.json 2>&1 | tee $C/logs/figure_supp.log
    python3 scripts/qual_compare.py --stage confident --vocab configs/vocab_replica.json \
      --metas "$(echo $all | sed 's/\.pt/.json/g')" --out $C/confident_errors.json \
      2>&1 | tee $C/logs/confident.log ;;
  *) echo "unknown step $1"; exit 1 ;;
esac
