#!/usr/bin/env bash
# Spatial smoothing (the BKI spatial kernel) as a read-out of the EXISTING Table I maps and
# fields: rows ours_P0s/ours_P2s/averaging_s of usplat/smoothing.py, 24 kernels fixed in
# configs/smooth_prereg.json, evaluated with scripts/eval_seg.py (no mapping, no distillation),
# selected and reported by scripts/smooth_select.py. One dataset per call:
#   DATASET=replica|scannet (the prereg's datasets.<d>: scenes, eval settings, text cache, OUT)
#   STAGE=val   : per batch (one sigma, 6 kernels; 4 batches in the prereg) fit one temperature
#                 per row on the validation scenes and evaluate them -> $OUT/eval_val_<b>.json
#                 (a finished batch is kept: a crash resumes), then the selection (@smooth: the
#                 lowest pooled validation NLL among ours_P2 and the 24 ours_P2s; @avg_own)
#                 -> $OUT/selection_val.json. Refused once the test stage started
#                 ($OUT/look_test.json or a finished test batch) and with uncommitted changes to
#                 tracked files (the records must name a commit).
#   STAGE=test  : the test scenes once per batch, with the validation betas of that batch.
#                 Refused with uncommitted changes in scripts/ usplat/ configs/, and unless
#                 scripts/smooth_select.py check-test passes (finished validation batches of
#                 the prereg's validation scenes at one commit equal to the tree; the
#                 selection made from them under this prereg file, reproduced by the rule;
#                 the selection of any earlier test attempt; a batch not finished yet).
#                 Finished test batches are never rerun. Appends the attempt to
#                 $OUT/look_test.json BEFORE the evaluation, then runs the table stage.
#   STAGE=table : $OUT/table_smooth.{json,tex} (every candidate, claim rule C1) and, when the
#                 other dataset's table exists, the family F3s (sign test + Holm) ->
#                 $OUT/claim_smooth.json.
# Gates: the prereg must be committed and unmodified (every stage); its "pipeline" must name
# this script; OUT never a Table I result directory; the text cache is a COPY in $OUT of the
# dataset's Table I cache (eval_seg.py writes missing prompts into the cache it is given).
# GPU: every evaluation goes through scripts/gpu_gate.sh at EVAL_NEED MiB (the prereg's
# gpu.eval_need_mib, 6000: ~4.0 GB estimated peak per batch at N = 1e6, see its "basis");
# never beside a USGLoc job; the measured peak of each batch is printed from its record.
# The kNN and the pooling run on the CPU inside eval_seg (the rows are frozen there).
# Overrides: PREREG, OUT, SCENES (same scene names as the prereg's split), BACKEND/DEVICE
# (mock/cpu: smoke tests), EVAL_NEED, EVAL_EXTRA (only --clip-model, --clip-pretrained,
# --clip-device, --channel-chunk), DRY_RUN=1 (print the GPU steps, run the checks).
# Logs: $OUT/eval_<stage>_<batch>.<timestamp>.log (never overwritten).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_USGLOC_EXTRA_MIB=${UAGS_GPU_USGLOC_EXTRA_MIB:-99999}   # never beside USGLoc
export UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-5}
PREREG=${PREREG:-configs/smooth_prereg.json}
DATASET=${DATASET:-}
STAGE=${STAGE:-val}
BACKEND=${BACKEND:-gsplat}
DEVICE=${DEVICE:-cuda}
DRY_RUN=${DRY_RUN:-0}
FIELDS_ROOT=/home/thanhnc/gauss_ws/src/uags/results/fields
TS=$(date +%Y%m%d-%H%M%S)
case "$STAGE" in
  val)   SPLIT=validation ;;
  test)  SPLIT=test ;;
  table) SPLIT=test ;;
  *) echo "STAGE must be val, test or table (got '$STAGE')"; exit 2 ;;
esac
[ -n "$DATASET" ] || { echo "set DATASET (replica or scannet)"; exit 2; }
if ! git ls-files --error-unmatch "$PREREG" > /dev/null 2>&1 || \
   [ -n "$(git status --porcelain -- "$PREREG")" ]; then
  echo "refused: $PREREG is not committed (the smoothing rules must be fixed before any run)"; exit 1
fi
for f in usplat/smoothing.py scripts/smooth_select.py scripts/smooth_tab1.sh; do
  if ! git ls-files --error-unmatch "$f" > /dev/null 2>&1; then
    echo "refused: $f is not committed (the records must name the code that ran)"; exit 1
  fi
done
# the dataset's settings, from the prereg only
eval "$(python3 - "$PREREG" "$DATASET" <<'PY'
import json, shlex, sys
from usplat.paths import moved_tree
p = moved_tree(json.load(open(sys.argv[1])))
if not str(p.get("pipeline") or "").startswith("scripts/smooth_tab1.sh"):
    sys.exit('echo "refused: the prereg names another pipeline"; exit 1')
d = (p.get("datasets") or {}).get(sys.argv[2])
if d is None:
    sys.exit(f'echo "refused: dataset {sys.argv[2]} not in the prereg"; exit 1')
e = d["eval"]
for k, v in dict(P_OUT=d["out"], P_VAL=d["scenes"]["validation"], P_TEST=d["scenes"]["test"],
                 VOCAB=e["vocab"], LOADER=e["dataset"], STRIDE=e["stride"],
                 DOWNSCALE=e["downscale"], TAU_COV=e["tau_cov"], EXTRACTOR=e["extractor"],
                 CACHE_SRC=d["text_cache_source"],
                 P_NEED=(p.get("gpu") or {}).get("eval_need_mib", 6000)).items():
    print(f"{k}={shlex.quote(str(v))}")
PY
)"
OUT=${OUT:-$P_OUT}
EVAL_NEED=${EVAL_NEED:-$P_NEED}
for d in tab1 tab1_v2 scannet_tab1 tab3 views views_repro; do
  if [ "$(realpath -m "$OUT")" = "$(realpath -m "/home/thanhnc/gauss_ws/src/uags/results/$d")" ]; then
    echo "refused: OUT is the Table I/III result directory $d (never touched)"; exit 1
  fi
done
case "$(realpath -m "$OUT")/" in
  "$(realpath -m "$FIELDS_ROOT")"/*) echo "refused: OUT inside $FIELDS_ROOT"; exit 1 ;;
esac
if [ "$STAGE" = val ]; then SCENES=${SCENES:-$P_VAL}; else SCENES=${SCENES:-$P_TEST}; fi
python3 - "$PREREG" "$DATASET" "$SCENES" "$SPLIT" <<'PY' || exit 1
import json, sys
from usplat.paths import moved_tree
sys.path.insert(0, "scripts")
import make_tables as mt
p = moved_tree(json.load(open(sys.argv[1])))
v = dict(p, scenes=p["datasets"][sys.argv[2]]["scenes"])
print(f"{sys.argv[4]} scenes: {mt.check_scenes(sys.argv[3], sys.argv[4], v)}")
PY
read -r -a EXTRA <<< "${EVAL_EXTRA:-}"
for a in ${EXTRA[@]+"${EXTRA[@]}"}; do   # resources and the text encoder only
  case "$a" in
    --clip-model|--clip-pretrained|--clip-device|--channel-chunk) ;;
    --clip-model=*|--clip-pretrained=*|--clip-device=*|--channel-chunk=*) ;;
    -*) echo "refused: EVAL_EXTRA option '$a' (allowed: --clip-model --clip-pretrained --clip-device --channel-chunk)"; exit 1 ;;
  esac
done
SEL="$OUT/selection_val.json"; LOOK="$OUT/look_test.json"
SS=(python3 scripts/smooth_select.py --prereg "$PREREG" --dataset "$DATASET" --dir "$OUT")
echo "[smooth $DATASET $STAGE] $TS git $(git rev-parse --short HEAD) dirty: $(git status --porcelain --untracked-files=no -- scripts usplat configs | wc -l) files; scenes $SCENES; prereg $PREREG sha256 $(sha256sum "$PREREG" | cut -c1-12); OUT $OUT"

table_stage() {
  "${SS[@]}" table
  local tabs=() d o
  for d in $(python3 -c "import json,sys; from usplat.paths import moved_tree; print(' '.join(moved_tree(json.load(open(sys.argv[1])))['datasets']))" "$PREREG"); do
    if [ "$d" = "$DATASET" ]; then o=$OUT; else
      o=$(python3 -c "import json,sys; from usplat.paths import moved_tree; print(moved_tree(json.load(open(sys.argv[1])))['datasets'][sys.argv[2]]['out'])" "$PREREG" "$d"); fi
    if [ -f "$o/table_smooth.json" ]; then tabs+=("$o/table_smooth.json"); fi
  done
  if [ "${#tabs[@]}" -eq "$(python3 -c "import json,sys; from usplat.paths import moved_tree; print(len(moved_tree(json.load(open(sys.argv[1])))['datasets']))" "$PREREG")" ]; then
    python3 scripts/smooth_select.py claim --prereg "$PREREG" --tables "${tabs[@]}" --out "$OUT/claim_smooth.json"
  else
    echo "claim (family F3s) waits for the other dataset's table"
  fi
}
if [ "$STAGE" = table ]; then table_stage; exit 0; fi

# the existing maps and fields only; the field must have been built on the scene's map
python3 - "$SCENES" <<'PY' || exit 1
import json, os, sys
from usplat.paths import moved
import torch, yaml
sc = yaml.safe_load(open(sys.argv[1]))
bad = []
for s in sc:
    miss = [f"{k} {s[k]}" for k in ("field", "ply") if not os.path.exists(s[k])]
    if miss:
        bad.append(f"{s['name']}: missing " + ", ".join(miss)); continue
    ck = torch.load(s["field"], map_location="cpu", mmap=True, weights_only=False)
    ply = moved(((ck.get("inputs") or {}).get("ply") or {}).get("path") or (ck.get("config") or {}).get("ply"))
    if not ply or os.path.realpath(ply) != os.path.realpath(s["ply"]):
        bad.append(f"{s['name']}: the field was built on {ply}, the scenes file names {s['ply']}")
    if not any(e.get("type") in ("Averaging", "NormLift") for e in ck.get("baselines") or []):
        bad.append(f"{s['name']}: the field holds no averaging sums (baselines)")
if bad:
    sys.exit("refused: " + "; ".join(bad))
print(f"scenes: {[s['name'] for s in sc]} (fields on their own maps, averaging sums present)")
PY

mkdir -p "$OUT"
TEXT_CACHE="$OUT/text_cache_${EXTRACTOR}.pt"
if [ ! -f "$TEXT_CACHE" ] && [ -f "$CACHE_SRC" ]; then
  cp "$CACHE_SRC" "$TEXT_CACHE"              # a copy: the Table I cache is never written
fi
finished() { [ -f "$1" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status','ok') == 'ok' else 1)" "$1"; }
mapfile -t BATCHES < <(python3 scripts/smooth_select.py batches --prereg "$PREREG")

if [ "$STAGE" = val ]; then
  if [ -e "$LOOK" ]; then echo "refused: $LOOK exists: the test stage has started, the selection is final"; exit 1; fi
  for line in "${BATCHES[@]}"; do
    if finished "$OUT/eval_test_${line%% *}.json"; then
      echo "refused: $OUT/eval_test_${line%% *}.json is a finished test batch; the selection is final"; exit 1
    fi
  done
  if ! python3 -c "import sys; from usplat.runlog import git_state; g = git_state('.'); sys.exit(1 if g.get('error') or g['dirty_tracked'] else 0)"; then
    git status --short --untracked-files=no
    echo "refused: uncommitted changes to tracked files (the validation records must name a commit)"; exit 1
  fi
  TODO=("${BATCHES[@]}")
else
  if [ -n "$(git status --porcelain --untracked-files=no -- scripts usplat configs)" ]; then
    echo "refused: uncommitted changes in scripts/ usplat/ configs/ (the test run must name a commit)"; exit 1
  fi
  todo=$("${SS[@]}" check-test) || exit 1
  TODO=()
  for line in "${BATCHES[@]}"; do
    if grep -qx "batch ${line%% *}" <<< "$todo"; then TODO+=("$line"); fi
  done
  echo "$(python3 -c "import json,sys; from usplat.paths import moved_tree; print(moved_tree(json.load(open(sys.argv[1])))['datasets'][sys.argv[2]]['test_look'])" "$PREREG" "$DATASET")"
  echo "choices fixed on validation ($SEL); test batches to run: ${#TODO[@]}"
  if [ "$DRY_RUN" != 1 ]; then
    "${SS[@]}" look --ts "$TS" --log "$OUT/eval_test_*.$TS.log"   # from here on: final
  fi
fi

COMMON=(--vocab "$VOCAB" --extractor "$EXTRACTOR" --dataset "$LOADER" --stride "$STRIDE"
        --downscale "$DOWNSCALE" --tau-cov "$TAU_COV" --text-cache "$TEXT_CACHE"
        --backend "$BACKEND" --device "$DEVICE" --no-p3
        --row-factory usplat.baselines:row_factory,usplat.smoothing:row_factory)
for line in "${TODO[@]}"; do
  b=${line%% *}; rows=${line#* }
  R="$OUT/eval_${STAGE}_${b}.json"
  if [ "$STAGE" = val ]; then
    if finished "$R"; then echo "batch $b: $R finished, kept"; continue; fi
    ARGS=(--val-scenes "$SCENES" --scenes "$SCENES" --out "$R")
  else
    ARGS=(--betas "$OUT/eval_val_${b}.json" --scenes "$SCENES" --out "$R")
  fi
  L="$OUT/eval_${STAGE}_${b}.$TS.log"
  if [ "$DRY_RUN" = 1 ]; then
    echo "[dry-run] scripts/gpu_gate.sh $EVAL_NEED python3 scripts/eval_seg.py ${EXTRA[*]+${EXTRA[*]}} ${COMMON[*]} --rows $rows ${ARGS[*]}"
    continue
  fi
  ok=0
  for try in 1 2 3; do                   # retried only after a crash (no finished record)
    scripts/gpu_gate.sh "$EVAL_NEED" python3 scripts/eval_seg.py ${EXTRA[@]+"${EXTRA[@]}"} \
      "${COMMON[@]}" --rows "$rows" "${ARGS[@]}" >> "$L" 2>&1 && { ok=1; break; }
    echo "batch $b $STAGE: attempt $try failed ($(grep -oE 'CUDA error: [^.]*|Error: [^.]*' "$L" | tail -1)); retrying"
  done
  [ "$ok" = 1 ] || { tail -30 "$L"; exit 1; }
  grep -E "^\[smoothing\]" "$L" | tail -40 || true
  python3 -c "import json,sys; m=json.load(open(sys.argv[1])).get('memory') or {}; print(f\"batch $b: peak GPU alloc {m.get('peak_gpu_alloc_gib', float('nan')):.2f} GiB (cap $EVAL_NEED MiB), peak RSS {m.get('peak_rss_bytes', 0)/2**30:.1f} GiB\")" "$R" || true
done
if [ "$DRY_RUN" = 1 ]; then echo "dry run: nothing evaluated"; exit 0; fi
if [ "$STAGE" = val ]; then
  "${SS[@]}" select --out "$SEL"
  echo "selection on validation -> $SEL"
else
  table_stage
fi
echo "done: smooth $DATASET $STAGE -> $OUT"
