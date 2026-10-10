#!/usr/bin/env bash
# Replica semantic swap (usplat/swap.py, configs/swap_replica.json): labelled synthetic changes,
# supporting evidence for the change test and the reset next to RIO10. Per scene:
#   bank      scripts/make_swap_bank.py: resolve the spec's pair rule (labels + text only), cut
#             real target-class features from mapping frames (GPU: LSeg)
#   reset     run_mapping.py on Sequence_1, update pass on Sequence_2 through
#             --feature-transform usplat.swap:swap_factory (the default field: change test on),
#             with --baselines (averaging, NormLift: the rows averaging and normlift_drop) and
#             --change-baselines (the residual rows, saved as the update-pass-only
#             change_scores that usplat/change_eval.py reads). Replica's depth is rendered from
#             the scene mesh, so --oracle-depth: the depth rows are the oracle rows
#             (depth_oracle; depth_residual and fused stay missing). No --vocab: the closed-set
#             CSS rows (css_var, yager_conflict) are not applicable to the swap and stay missing
#   noreset   the same with --fixed-change-thr inf (the change test runs, the same c, never a
#             reset) and the reset field's PCA (--pca-from), so both fields see identical
#             features; eval_change.py checks it (code, arguments, PCA, centre, frames logs).
#             Only its mu* is read (segmentation block), so it is mapped without baselines
#   gt        scripts/eval_change.py gt-swap: update-pass splat mass on swapped pixels over the
#             frames the transform logged (GPU: gsplat)
#   eval      scripts/eval_change.py eval (CPU): change rows + accuracy on swapped / static
# then scripts/eval_change.py criterion over the stage's scenes.
#   STAGE=val  : office_0, room_2. The criterion is reported, not binding; nothing is tuned.
#   STAGE=test : the 6 test scenes, ONCE, with the criterion fixed in the spec. Refused with
#                uncommitted changes in scripts/ usplat/ configs/, without a finished validation
#                criterion record of the same spec ($OUT/criterion_val.json), when
#                $OUT/criterion_test.json holds a finished test run, and when a scene's finished
#                test record comes from another commit (FORCE_TEST=1 overrides the last two).
#                eval_change.py enforces the same rules itself (eval gets --reference) and
#                refuses test fields built with other mapping code than val's.
# Never writes into the Tab I outputs: the maps are READ from $MAPS (scripts/replica_tab1.sh);
# banks, fields and ground truth go to $SWAP, records and logs to $OUT. Every GPU step goes
# through scripts/gpu_gate.sh. Existing banks/fields/GT are reused. DRY_RUN=1 prints each
# step (shell-quoted) instead of running it; the refusal rules still run.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_USGLOC_EXTRA_MIB=${UAGS_GPU_USGLOC_EXTRA_MIB:-99999}   # never beside USGLoc
export UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-5}
ROOT=${ROOT:-/home/thanhnc/gauss_ws/data/replica_semnerf/Replica_Dataset}
MAPS=${MAPS:-/home/thanhnc/gauss_ws/data/uags_maps}
SWAP=${SWAP:-/home/thanhnc/gauss_ws/data/uags_swap}
OUT=${OUT:-/home/thanhnc/gauss_ws/data/uags_results/swap}
SPEC=${SPEC:-configs/swap_replica.json}
STAGE=${STAGE:-val}
DRY_RUN=${DRY_RUN:-0}
MAP_NEED=${MAP_NEED:-10000}      # as replica_static_fa.sh (mapping + baselines + residuals)
BANK_NEED=${BANK_NEED:-5000}     # LSeg fp32 on the bank frames
GT_NEED=${GT_NEED:-7000}         # one K-channel render per update frame
TS=$(date +%Y%m%d-%H%M%S)
case "$STAGE" in
  val|test) ;;
  *) echo "STAGE must be val or test (got '$STAGE')"; exit 2 ;;
esac
spec() {       # spec <key.key...>: a value of the frozen spec (lists space-separated)
  python3 - "$SPEC" "$1" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."):
    v = v[k]
print(" ".join(map(str, v)) if isinstance(v, list) else v)
PY
}
SCENES=$(spec "scenes.$STAGE")
DOWNSCALE=$(spec downscale)
MSEQ=$(spec mapping.sequence); MSTRIDE=$(spec mapping.stride)
USEQ=$(spec update.sequence); USTRIDE=$(spec update.stride)
EXTRACTOR=$(spec extractor)
HEAD=$(git rev-parse HEAD)
echo "[swap $STAGE] $TS git ${HEAD:0:7} dirty: $(git status --porcelain --untracked-files=no -- scripts usplat configs | wc -l) files; scenes: $SCENES"

# step <log> <command...>: run it with its output in <log> (tail on failure); DRY_RUN: print it
step() {
  local L=$1; shift
  if [ "$DRY_RUN" = 1 ]; then printf '+'; printf ' %q' "$@"; printf '\n'; return 0; fi
  "$@" > "$L" 2>&1 || { tail -20 "$L"; exit 1; }
  grep -vE "it/s|s/it|waiting" "$L" | tail -5
}
finished() {   # finished <record>: status "ok" (a crashed run leaves "running"/"failed")
  [ -f "$1" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status') == 'ok' else 1)" "$1"
}

if [ "$STAGE" = test ]; then
  if [ -n "$(git status --porcelain --untracked-files=no -- scripts usplat configs)" ]; then
    echo "refused: uncommitted changes in scripts/ usplat/ configs/ (the test run must name a commit)"; exit 1
  fi
  python3 - "$OUT/criterion_val.json" "$SPEC" <<'PY' || exit 1
import json, os, sys
sys.path.insert(0, os.getcwd())
from usplat.change_eval import load_config
path, spec = sys.argv[1], sys.argv[2]
if not os.path.isfile(path):
    sys.exit(f"refused: no {path}: run STAGE=val first (the test split runs after a finished validation)")
d = json.load(open(path))
m = d.get("metrics") if isinstance(d.get("metrics"), dict) else d
if d.get("status") != "ok" or m.get("split") != "val":
    sys.exit(f"refused: {path} is not a finished validation criterion record (status {d.get('status')!r})")
if m.get("config_sha256") != load_config(spec)[1]:
    sys.exit(f"refused: {path} was made with another {spec} (the spec is frozen)")
print(f"validation criterion record: {len(m.get('scenes', []))} scenes, spec {m['config_sha256'][:12]}")
PY
  if finished "$OUT/criterion_test.json" && [ "${FORCE_TEST:-0}" != 1 ]; then
    echo "refused: $OUT/criterion_test.json holds a finished test run (the test split runs once; FORCE_TEST=1 to override)"; exit 1
  fi
fi
for s in $SCENES; do
  [ -f "$MAPS/${s}_seq1.pt" ] || { echo "refused: no map $MAPS/${s}_seq1.pt (build it with scripts/replica_tab1.sh EVAL=0)"; exit 1; }
done
mkdir -p "$SWAP/banks" "$SWAP/fields" "$SWAP/gt" "$SWAP/scores" "$OUT"
FT=()                                   # eval_change.py refuses to overwrite a finished test record
[ "$STAGE" = test ] && [ "${FORCE_TEST:-0}" = 1 ] && FT=(--force-test)
REF=()                                  # test: scored after the finished validation criterion
[ "$STAGE" = test ] && REF=(--reference "$OUT/criterion_val.json")

RECORDS=()
for s in $SCENES; do
  BANK="$SWAP/banks/$s.pt"
  F="$SWAP/fields/${s}_swap.pt"
  F0="$SWAP/fields/${s}_swap_noreset.pt"
  GT="$SWAP/gt/$s.pt"
  REC="$OUT/eval_${STAGE}_${s}.json"
  RECORDS+=("$REC")
  if [ "$STAGE" = test ] && finished "$REC" && [ "${FORCE_TEST:-0}" != 1 ]; then
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d['git']['commit'] == sys.argv[2] else 1)" "$REC" "$HEAD" \
      || { echo "refused: $REC is a finished test record of another commit (FORCE_TEST=1 to override)"; exit 1; }
    echo "$s: finished test record of this commit, kept"; continue
  fi
  [ -f "$BANK" ] || step "$OUT/bank_${s}.$TS.log" scripts/gpu_gate.sh "$BANK_NEED" \
    python3 scripts/make_swap_bank.py --spec "$SPEC" --scene "$s" --root "$ROOT" \
    --extractor "$EXTRACTOR" --device cuda --out "$BANK" --record "$OUT/bank_${s}.json"
  COMMON=(--backend gsplat --dataset replica_semnerf --data "$ROOT/$s/$MSEQ" --stride "$MSTRIDE"
          --downscale "$DOWNSCALE" --ply "$MAPS/${s}_seq1.pt" --extractor "$EXTRACTOR"
          --feat-dim 64 --pca-frames 20 --seed 0 --use-depth
          --update-data "$ROOT/$s/$USEQ" --update-stride "$USTRIDE"
          --feature-transform usplat.swap:swap_factory)
  [ -f "$F" ] || step "$OUT/map_${s}.$TS.log" scripts/gpu_gate.sh "$MAP_NEED" \
    python3 scripts/run_mapping.py "${COMMON[@]}" --baselines --change-baselines --oracle-depth \
    --feature-transform-arg "bank=$BANK,spec=$SPEC,scene=$s,log=$SWAP/fields/${s}_swap.frames.json" \
    --out "$F"
  [ -f "$F0" ] || step "$OUT/map_${s}_noreset.$TS.log" scripts/gpu_gate.sh "$MAP_NEED" \
    python3 scripts/run_mapping.py "${COMMON[@]}" \
    --feature-transform-arg "bank=$BANK,spec=$SPEC,scene=$s,log=$SWAP/fields/${s}_swap_noreset.frames.json" \
    --fixed-change-thr inf --pca-from "$F" --out "$F0"
  [ -f "$GT" ] || step "$OUT/gt_${s}.$TS.log" scripts/gpu_gate.sh "$GT_NEED" \
    python3 scripts/eval_change.py gt-swap --ply "$MAPS/${s}_seq1.pt" --data "$ROOT/$s/$USEQ" \
    --dataset replica_semnerf --update-stride "$USTRIDE" \
    --frames-log "$SWAP/fields/${s}_swap.frames.json" --bank "$BANK" --spec "$SPEC" --scene "$s" \
    --downscale "$DOWNSCALE" --backend gsplat --device cuda --out "$GT" --record "$OUT/gt_${s}.json"
  step "$OUT/eval_${STAGE}_${s}.$TS.log" python3 scripts/eval_change.py eval --field "$F" --gt "$GT" \
    --config "$SPEC" --scene "$s" --split "$STAGE" --compare-field "ours_noreset=$F0" \
    --dump-scores "$SWAP/scores/${STAGE}_${s}.pt" "${REF[@]}" "${FT[@]}" --out "$REC"
done
step "$OUT/criterion_${STAGE}.$TS.log" python3 scripts/eval_change.py criterion --config "$SPEC" \
  --split "$STAGE" --records "${RECORDS[@]}" "${REF[@]}" "${FT[@]}" --out "$OUT/criterion_${STAGE}.json"
[ "$DRY_RUN" = 1 ] || cat "$OUT/criterion_${STAGE}.$TS.log"
echo "done: swap $STAGE -> $OUT"
