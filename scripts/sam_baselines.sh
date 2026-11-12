#!/usr/bin/env bash
# sam_baselines.sh — configs/sam_baselines_prereg.json step by step: LangSplat and OpenGaussian
# re-implemented on SAM masks (usplat/sam_lift.py) for Tables I, II, III, IV and the error split.
#
#   scripts/sam_baselines.sh <step>
#     time             SAM (the registered settings) on 2 frames of office_0 on the GPU; writes the
#                      measured s/frame to results/sam_baselines/time/sam_time.json
#     masks            SAM masks of the mapping frames of every Replica / ScanNet val + test field
#                      and every NOR mapping-only field -> results/sam_masks/<field stem>_sam.npz
#     train            Replica + ScanNet: LangSplat (s, p, w) and OpenGaussian next to each field
#     val              eval_seg on the validation scenes (fits the temperatures) + control
#     select           the LangSplat level per dataset (NOR: Replica's) -> selection.json
#     test             eval_seg on the test scenes once, with the validation temperatures + control
#     nor-masks        SAM masks of the update keyframes of the Table IV scenes (val 002 + test)
#     nor-train        NOR mapping-only files (the selected LangSplat level, OpenGaussian)
#     nor-train-upd    NOR files re-trained on mapping + update keyframes (Table IV scenes)
#     nor-change-val | nor-change-test   Table III (scripts/sam_change_nor.py change)
#     nor-stale-val  | nor-stale-test    Table IV  (scripts/sam_change_nor.py stale)
#
# Resumable: a finished output is skipped (masks / trained files that exist, records with status
# ok); a record with another status is moved to the logs and its step reruns. Test steps refuse
# uncommitted changes in scripts/ usplat/ configs/ and never overwrite a finished test record.
# GPU work goes through scripts/gpu_gate.sh with UAGS_GPU_HEADROOM_MIB from the environment
# (default 1024); every need + headroom must stay <= 8900 MiB. Logs: results/sam_baselines/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/venvs/uags/bin:$HOME/.local/cuda-12.8/bin:$PATH OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_HEADROOM_MIB=${UAGS_GPU_HEADROOM_MIB:-1024} UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-30}
PREREG=configs/sam_baselines_prereg.json
R=results
OUT=$R/sam_baselines
LOGS=$OUT/logs
MASKS=$R/sam_masks
NEED_SAM=${NEED_SAM:-7800} NEED_TRAIN=${NEED_TRAIN:-5000} NEED_EVAL=${NEED_EVAL:-7000} NEED_NOR=${NEED_NOR:-6000}   # MiB; SAM vit_h fp32 OOMed at 6000 (2026-11-11)
NOR_VAL="001 002 003 004"
NOR_TEST="005 006 007 008 009 010 011 013 014 015"
NOR_STALE_VAL="002"
mkdir -p "$LOGS"
step=${1:-}

stamp() { echo "[$(date '+%F %T')] $*"; }
ok_rec() { python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status')=='ok' else 1)" "$1" 2>/dev/null; }
aside() {      # a record of another status is moved to the logs (never kept as a result)
  [ -f "$1" ] || return 0
  local to="$LOGS/$(basename "$1").notok.$(date +%Y%m%d-%H%M%S)"
  mv "$1" "$to"; stamp "moved a record that is not ok: $1 -> $to"
}
clean_tree() {
  if [ -n "$(git status --porcelain --untracked-files=no -- scripts usplat configs)" ]; then
    echo "refused: uncommitted changes in scripts/ usplat/ configs/ (the test run must name a commit)"; exit 2
  fi
}
gate() {       # gate <need> <command...>
  local need=$1; shift
  if [ $((need + UAGS_GPU_HEADROOM_MIB)) -gt 8900 ]; then
    echo "refused: need $need + headroom $UAGS_GPU_HEADROOM_MIB MiB > 8900"; exit 2
  fi
  scripts/gpu_gate.sh "$need" "$@"
}
retry() {      # retry <log> <need> <command...>: a CUDA launch time-out kills the job; it resumes
  local log=$1; shift
  for try in 1 2 3; do
    gate "$@" >> "$log" 2>&1 && return 0
    stamp "attempt $try failed ($(grep -oE 'CUDA error: [^.]*|Error: [^.]*' "$log" | tail -1)); see $log"
  done
  tail -20 "$log"; exit 1
}
fields() { python3 -c "import json,sys; [print(s['name'], s['field']) for f in sys.argv[1:] for s in json.load(open(f))]" "$@"; }
nor_field() { echo "$R/fields/nor_$1_lseg_relmax.pt"; }
nor_split() { case " $NOR_VAL " in *" $1 "*) echo val ;; *) echo test ;; esac; }
nor_record() { echo "$R/nor/tab2_relmax/$(nor_split "$1")/$1.json"; }
masks_of() { echo "$MASKS/$(basename "${1%.pt}")_sam.npz"; }
masks_upd_of() { echo "$MASKS/$(basename "${1%.pt}")_sam_upd.npz"; }
stem() { echo "${1%.pt}"; }
level() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]['level'])" "$OUT/selection.json" "$1"; }
need_selection() { [ -f "$OUT/selection.json" ] || { echo "refused: run the select step first"; exit 2; }; }

if [ -n "$(git status --porcelain -- "$PREREG")" ] || ! git ls-files --error-unmatch "$PREREG" >/dev/null 2>&1; then
  echo "refused: $PREREG is not committed"; exit 2
fi
stamp "sam_baselines $step, git $(git rev-parse --short HEAD), headroom $UAGS_GPU_HEADROOM_MIB MiB"

# ------------------------------------------------------------------------------- time
do_time() {
  local f=$R/fields/office_0_lseg.pt out=$OUT/time
  local npz=$out/office_0_lseg_sam_smoke2.npz
  if [ ! -f "$npz" ]; then
    stamp "time: SAM on 2 frames of office_0"
    gate "$NEED_SAM" python3 scripts/sam_masks.py --field "$f" --max-frames 2 --out-dir "$out" \
      > "$LOGS/time.log" 2>&1 || { tail -20 "$LOGS/time.log"; exit 1; }
  fi
  python3 - "$npz" "$out/sam_time.json" <<'EOF'
import json, sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as z:
    meta = json.loads(str(z["meta"]))
sec = [p["seconds"] for p in meta["per_frame"]]
frames = 1430 + 2110 + 638 + 637            # configs/sam_baselines_prereg.json budget.masks
per = sec[-1]                                 # the first frame includes the warm-up
out = dict(seconds_per_frame=sec, s_per_frame=per, frames=frames, est_total_h=per * frames / 3600,
           load_s=meta["load_s"], peak_alloc_mib=meta["peak_alloc_mib"], device=meta["device"],
           n_masks=[p["n_masks"] for p in meta["per_frame"]], settings=meta["settings"],
           identity=meta["identity"], masks=sys.argv[1])
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(f"SAM: {sec} s per frame -> {per:.1f} s/frame, about {out['est_total_h']:.1f} h for {frames} frames "
      f"(peak {meta['peak_alloc_mib']} MiB); {sys.argv[2]}")
EOF
}

# ------------------------------------------------------------------------------- masks
mask_one() {   # mask_one <field> <tag>
  local f=$1 out; out=$(masks_of "$f")
  [ -f "$out" ] && { stamp "skip masks $(basename "$f") (exists)"; return; }
  stamp "masks $(basename "$f")"
  retry "$LOGS/masks_$2.log" "$NEED_SAM" python3 scripts/sam_masks.py --field "$f" --out-dir "$MASKS"
  tail -1 "$LOGS/masks_$2.log"
}
do_masks() {
  while read -r name f; do mask_one "$f" "$name"; done < <(fields configs/replica_val.json \
    configs/replica_test.json configs/scannet_val.json configs/scannet_test.json)
  for s in $NOR_VAL $NOR_TEST; do mask_one "$(nor_field "$s")" "nor_$s"; done
}
do_nor_masks() {
  for s in $NOR_STALE_VAL $NOR_TEST; do
    local f; f=$(nor_field "$s")
    [ -f "$(masks_upd_of "$f")" ] && { stamp "skip update masks $s (exists)"; continue; }
    stamp "update masks $s"
    retry "$LOGS/masks_upd_$s.log" "$NEED_SAM" python3 scripts/sam_masks.py --table "$(nor_record "$s")" \
      --out-dir "$MASKS"
    tail -1 "$LOGS/masks_upd_$s.log"
  done
}

# ------------------------------------------------------------------------------- train
train_one() {  # train_one <tag> <outputs...> -- <train_sam_baselines.py arguments...>
  local tag=$1; shift
  local outs=() all=1
  while [ "$1" != "--" ]; do outs+=("$1"); shift; done; shift
  for o in "${outs[@]}"; do [ -f "$o" ] || all=0; done
  [ "$all" = 1 ] && { stamp "skip train $tag (exists)"; return; }
  stamp "train $tag"
  retry "$LOGS/train_$tag.log" "$NEED_TRAIN" python3 scripts/train_sam_baselines.py "$@"
  grep -E "^saved|exists" "$LOGS/train_$tag.log" | tail -4
}
do_train() {
  while read -r name f; do
    local s; s=$(stem "$f")
    train_one "${name}_langsplat" "${s}_langsplat_s.pt" "${s}_langsplat_p.pt" "${s}_langsplat_w.pt" -- \
      --field "$f" --masks "$(masks_of "$f")" --method langsplat --level all
    train_one "${name}_opengs" "${s}_opengs.pt" -- --field "$f" --masks "$(masks_of "$f")" --method opengs
  done < <(fields configs/replica_val.json configs/replica_test.json configs/scannet_val.json \
    configs/scannet_test.json)
}
do_nor_train() {
  need_selection
  local L; L=$(level nor)
  for s in $NOR_VAL $NOR_TEST; do
    local f st; f=$(nor_field "$s"); st=$(stem "$f")
    train_one "nor_${s}_langsplat_$L" "${st}_langsplat_$L.pt" -- --field "$f" --masks "$(masks_of "$f")" \
      --method langsplat --level "$L"
    train_one "nor_${s}_opengs" "${st}_opengs.pt" -- --field "$f" --masks "$(masks_of "$f")" --method opengs
  done
}
do_nor_train_upd() {
  need_selection
  local L; L=$(level nor)
  for s in $NOR_STALE_VAL $NOR_TEST; do
    local f st m; f=$(nor_field "$s"); st=$(stem "$f"); m="$(masks_of "$f"),$(masks_upd_of "$f")"
    train_one "nor_${s}_langsplat_${L}_upd" "${st}_langsplat_${L}_upd.pt" -- --table "$(nor_record "$s")" \
      --masks "$m" --method langsplat --level "$L"
    train_one "nor_${s}_opengs_upd" "${st}_opengs_upd.pt" -- --table "$(nor_record "$s")" --masks "$m" \
      --method opengs
  done
}

# ------------------------------------------------------------------------------- eval_seg
FACT=usplat.baselines:row_factory,usplat.distill:row_factory,usplat.sam_lift:row_factory
NEW_ROWS=langsplat_s,langsplat_p,langsplat_w,opengs
# --with-2d as Table I's commands: BetaFitter draws its pixel sample from one generator shared over
# the val scenes, and the 2D row draws from it after each scene, so without it every scene after the
# first gets another sample and the control rows' temperatures move (Replica: up to 4%, 2026-11-13).
ds_args() {    # ds_args <replica|scannet> <val|test>
  case "$1" in
    replica) [ -f "$OUT/text_cache_replica.pt" ] || cp "$R/fields/text_cache_lseg.pt" "$OUT/text_cache_replica.pt"
             echo "--vocab configs/vocab_replica.json --dataset replica_semnerf --stride 10" \
                  "--text-cache $OUT/text_cache_replica.pt --with-2d --rows $NEW_ROWS,$(control_rows replica),2d_features" ;;
    scannet) [ -f "$OUT/text_cache_scannet.pt" ] || cp "$R/fields/scannet/text_cache_lseg.pt" "$OUT/text_cache_scannet.pt"
             echo "--vocab configs/vocab_scannet.json --dataset scannet --stride 20" \
                  "--text-cache $OUT/text_cache_scannet.pt --with-2d --rows $NEW_ROWS,$(control_rows scannet),2d_features" ;;
  esac
}
control_rows() { case "$1" in replica) echo ours_P0,ours_P2,normlift,f3dgs,legs_l4 ;; scannet) echo ours_P0,ours_P2,normlift,f3dgs,legs_l1 ;; esac; }
table1() { case "$1" in replica) echo "$R/tab1" ;; scannet) echo "$R/scannet_tab1" ;; esac; }
eval_one() {   # eval_one <replica|scannet> <val|test>
  local ds=$1 st=$2 rec=$OUT/$1/eval_$2.json args
  mkdir -p "$OUT/$ds"
  if ok_rec "$rec"; then
    stamp "skip eval $ds $st (finished)"
  else
    [ "$st" = test ] && [ -f "$rec" ] && stamp "the test record $rec did not finish; rerunning it"
    aside "$rec"
    args=$(ds_args "$ds" "$st")
    local scen="--val-scenes configs/${ds}_val.json --scenes configs/${ds}_val.json"
    [ "$st" = test ] && scen="--betas $OUT/$ds/eval_val.json --scenes configs/${ds}_test.json"
    stamp "eval $ds $st"
    # shellcheck disable=SC2086
    retry "$LOGS/eval_${ds}_$st.log" "$NEED_EVAL" python3 scripts/eval_seg.py --row-factory "$FACT" \
      --extractor lseg --downscale 2 --tau-cov 0.5 $args $scen --out "$rec"
  fi
  python3 scripts/sam_baselines_check.py control "$rec" "$(table1 "$ds")/eval_$st.json" \
    --rows "$(control_rows "$ds")" --out "$OUT/$ds/control_$st.json" \
    || { echo "control failed: $OUT/$ds/control_$st.json (stop; report it)"; exit 2; }
}
do_val() { for ds in replica scannet; do eval_one "$ds" val; done; }
do_select() {
  for ds in replica scannet; do ok_rec "$OUT/$ds/eval_val.json" || { echo "refused: $ds validation not finished"; exit 2; }; done
  python3 scripts/sam_baselines_check.py select "$OUT/replica/eval_val.json" "$OUT/scannet/eval_val.json" \
    --out "$OUT/selection.json"
}
do_test() { clean_tree; need_selection; for ds in replica scannet; do eval_one "$ds" test; done; }

# ------------------------------------------------------------------------------- NOR tables
refopts() { python3 -c "import json,sys; c=json.load(open(sys.argv[1]))['config']; print(f\"--align {c['align']} --update-stride {c['update_stride']} --tau-cov {c['tau_cov']}\")" "$1"; }
do_nor_change() {   # do_nor_change <val|test>
  need_selection
  local split=$1 L scenes; L=$(level nor)
  [ "$split" = test ] && { clean_tree; scenes=$NOR_TEST; } || scenes=$NOR_VAL
  mkdir -p "$R/nor/sam/$split"
  for s in $scenes; do
    local f st rec ref; f=$(nor_field "$s"); st=$(stem "$f"); rec=$R/nor/sam/$split/$s.json; ref=$(nor_record "$s")
    if ok_rec "$rec"; then
      stamp "skip Table III $split $s (finished)"
    else
      aside "$rec"
      stamp "Table III $split $s"
      # shellcheck disable=SC2046
      retry "$LOGS/change_${split}_$s.log" "$NEED_NOR" python3 scripts/sam_change_nor.py change --field "$f" \
        $(refopts "$ref") --distilled "${st}_langsplat_$L.pt,${st}_opengs.pt" --reference "$ref" --out "$rec"
    fi
    # the run record keeps its results under "metrics"; a finished record is checked again on resume
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); c=d.get('metrics', d)['control_check']; print(f\"control {sys.argv[2]}: max |diff| {c['max_abs_diff']:.3g}\"); sys.exit(0 if c['max_abs_diff'] <= 1e-6 else 2)" "$rec" "$s" \
      || { echo "control failed for $s: feat_control differs from $ref (stop; report it)"; exit 2; }
  done
}
do_nor_stale() {    # do_nor_stale <val|test>
  need_selection
  local split=$1 L scenes; L=$(level nor)
  [ "$split" = test ] && { clean_tree; scenes=$NOR_TEST; } || scenes=$NOR_STALE_VAL
  mkdir -p "$R/stale_baselines_sam/$split"
  for s in $scenes; do
    local f st rec; f=$(nor_field "$s"); st=$(stem "$f"); rec=$R/stale_baselines_sam/$split/${s}_trained.json
    if ok_rec "$rec"; then stamp "skip Table IV $split $s (finished)"; continue; fi
    aside "$rec"
    stamp "Table IV $split $s"
    retry "$LOGS/stale_${split}_$s.log" "$NEED_NOR" python3 scripts/sam_change_nor.py stale \
      --table "$(nor_record "$s")" --dump "$R/qual/nor/$s" --vocab configs/vocab_replica.json \
      --text-cache "$R/stale_baselines/text_cache_lseg_nor.pt" --stage "$split" --no-2d \
      --distilled "${st}_langsplat_${L}_upd.pt,${st}_opengs_upd.pt,${st}_langsplat_$L.pt,${st}_opengs.pt" \
      --out "$rec"
    grep -E "stale answers|control" "$LOGS/stale_${split}_$s.log" | tail -6
  done
}

case "$step" in
  time) do_time ;;
  masks) do_masks ;;
  train) do_train ;;
  val) do_val ;;
  select) do_select ;;
  test) do_test ;;
  nor-masks) do_nor_masks ;;
  nor-train) do_nor_train ;;
  nor-train-upd) do_nor_train_upd ;;
  nor-change-val) do_nor_change val ;;
  nor-change-test) do_nor_change test ;;
  nor-stale-val) do_nor_stale val ;;
  nor-stale-test) do_nor_stale test ;;
  *) echo "usage: $0 time|masks|train|val|select|test|nor-masks|nor-train|nor-train-upd|nor-change-val|nor-change-test|nor-stale-val|nor-stale-test"; exit 2 ;;
esac
stamp "done: $step"
