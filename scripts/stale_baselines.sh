#!/usr/bin/env bash
# stale_baselines.sh — configs/stale_baselines_prereg.json: more baselines for Table IV (stale answers).
#   STAGE=val|test PASS=light|trained scripts/stale_baselines.sh
#   light   : per scene of the stage the rows before (control), 2d and the frozen trained maps
#             (reference rows) -> results/stale_baselines/<stage>/<scene>_light.json
#   trained : Feature 3DGS and LEGaussians (lambda_u 1) re-trained with the update keyframes
#             (scripts/distill_update_nor.py), then their rows
#             -> results/stale_baselines/<stage>/<scene>_trained.json
#             validation: the prereg's val_trained scenes only
# STAGE=test is refused with uncommitted changes in scripts/ usplat/ configs/ and until every
# validation record of the same pass is finished with its control passed. A finished test record
# is never overwritten; a trained file that exists is reused. GPU steps go through
# scripts/gpu_gate.sh. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/venvs/uags/bin:$HOME/.local/cuda-12.8/bin:$PATH OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_HEADROOM_MIB=${UAGS_GPU_HEADROOM_MIB:-1024} UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-5}
STAGE=${STAGE:-val}
PASS=${PASS:-light}
PREREG=configs/stale_baselines_prereg.json
OUT=results/stale_baselines
TRAIN_NEED=${TRAIN_NEED:-5000}
EVAL_NEED=${EVAL_NEED:-6000}
case "$STAGE" in val|test) ;; *) echo "STAGE must be val or test"; exit 2 ;; esac
case "$PASS" in light|trained) ;; *) echo "PASS must be light or trained"; exit 2 ;; esac
stamp() { echo "[$(date '+%F %T')] $*"; }
scenes() { python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['splits'][sys.argv[2]]))" "$PREREG" "$1"; }
field() { echo results/fields/nor_$1_lseg_relmax.pt; }
ok_rec() { python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status')=='ok' else 1)" "$1" 2>/dev/null; }

if [ -n "$(git status --porcelain -- "$PREREG")" ] || ! git ls-files --error-unmatch "$PREREG" >/dev/null 2>&1; then
  echo "refused: $PREREG is not committed"; exit 1
fi
VAL_KEY=val; [ "$PASS" = trained ] && VAL_KEY=val_trained
if [ "$STAGE" = test ]; then
  if [ -n "$(git status --porcelain --untracked-files=no -- scripts usplat configs)" ]; then
    echo "refused: uncommitted changes in scripts/ usplat/ configs/ (the test run must name a commit)"; exit 1
  fi
  for s in $(scenes "$VAL_KEY"); do
    ok_rec "$OUT/val/${s}_$PASS.json" || { echo "refused: validation record $OUT/val/${s}_$PASS.json is missing or its control failed"; exit 1; }
  done
  LIST=$(scenes test)
else
  LIST=$(scenes "$VAL_KEY")
fi
mkdir -p "$OUT/$STAGE" "$OUT/logs"
cp -n results/qual/text_cache_lseg_nor.pt "$OUT/text_cache_lseg_nor.pt"
stamp "stale baselines: stage $STAGE pass $PASS git $(git rev-parse --short HEAD) scenes: $LIST"

for s in $LIST; do
  REC="$OUT/$STAGE/${s}_$PASS.json"
  if ok_rec "$REC"; then stamp "skip $s $PASS (finished)"; continue; fi
  F=$(field "$s"); STEM="${F%.pt}"
  if [ "$PASS" = light ]; then
    DIST="${STEM}_f3dgs.pt,${STEM}_legs_l1.pt"; EXTRA=()
  else
    for mth in f3dgs legs; do
      name=$mth; [ "$mth" = legs ] && name=legs_l1
      if [ -f "${STEM}_${name}_upd.pt" ]; then stamp "skip train $s ${name}_upd (exists)"; continue; fi
      stamp "train $s ${name}_upd"
      scripts/gpu_gate.sh "$TRAIN_NEED" python3 scripts/distill_update_nor.py \
        --table "results/nor/tab2_relmax/$STAGE/$s.json" --method "$mth" --lam-u 1 \
        > "$OUT/logs/train_${s}_${name}_upd.log" 2>&1 || { tail -20 "$OUT/logs/train_${s}_${name}_upd.log"; exit 1; }
      tail -1 "$OUT/logs/train_${s}_${name}_upd.log"
    done
    DIST="${STEM}_f3dgs_upd.pt,${STEM}_legs_l1_upd.pt"; EXTRA=(--no-2d)
  fi
  stamp "eval $s $PASS"
  rc=0
  scripts/gpu_gate.sh "$EVAL_NEED" python3 scripts/stale_baselines_nor.py \
    --table "results/nor/tab2_relmax/$STAGE/$s.json" --dump "results/qual/nor/$s" \
    --vocab configs/vocab_replica.json --text-cache "$OUT/text_cache_lseg_nor.pt" \
    --distilled "$DIST" "${EXTRA[@]}" --stage "$STAGE" --out "$REC" \
    > "$OUT/logs/eval_${STAGE}_${s}_$PASS.log" 2>&1 || rc=$?
  grep -E "stale answers|control" "$OUT/logs/eval_${STAGE}_${s}_$PASS.log" || tail -20 "$OUT/logs/eval_${STAGE}_${s}_$PASS.log"
  [ "$rc" = 0 ] || { echo "stopped: $s $PASS exit $rc (2 = the control did not reproduce the dump)"; exit "$rc"; }
done
stamp "done: stage $STAGE pass $PASS -> $OUT/$STAGE"
