#!/usr/bin/env bash
# swap_trained.sh — configs/swap_trained_prereg.json: Feature 3DGS and LEGaussians (lambda_u 4)
# re-trained with the swapped update frames of the Replica semantic swap, scored with
# eval_change.py's segmentation block. STAGE=val|test scripts/swap_trained.sh. Per scene:
#   bank     scripts/make_swap_bank.py, as scripts/replica_swap.sh, when $SWAP/banks/<s>.pt is
#            missing; its pairs and segments must equal the kept record $OUT/bank_<s>.json
#   field    replica_swap.sh's reset field (run_mapping.py, its arguments and FIELD_CONFIG, the
#            paper run's change test) when missing; it logs to <s>_swap.rebuild.frames.json, the
#            kept paper-run log <s>_swap.frames.json is the reference (--frames-ref)
#   gt       eval_change.py gt-swap over the field's frames log when missing
#   control  swap_trained_eval.py --check-only: the recomputed 'averaging' must equal the eval
#            record $OUT/eval_<stage>_<s>.json within 1e-9; else stop (exit 2) before training
#   train    distill_update_swap.py, f3dgs and legs (lambda_u from the registration):
#            <field>_f3dgs_upd.pt, <field>_legs_l4_upd.pt; an existing file is reused (the
#            scorer refuses it if it was trained on another field)
#   score    swap_trained_eval.py -> $OUT/trained_<stage>_<s>.json
# The noreset field is not needed (ours_noreset is read from the eval record). Records never
# overwrite the kept ones: rebuild records go to $OUT/trained_rebuild, logs to $OUT/trained_logs.
# CLEAN=1 (default) deletes the bank, field and ground truth THIS driver rebuilt once the scene's
# record is finished (listed in $OUT/trained_rebuild/<s>.rebuilt); records, logs and the trained
# files stay. A bank is never rebuilt under an existing field (the field saw the old bank).
# STAGE=test is refused with uncommitted changes in scripts/ usplat/ configs/, without both
# finished validation records (control passed), and when every test record is finished; a
# finished test record of this commit is kept, one of another commit is refused.
# Every GPU step goes through scripts/gpu_gate.sh. DRY_RUN=1 prints each step (shell-quoted)
# instead of running it; the refusal rules still run.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/venvs/uags/bin:$HOME/.local/cuda-12.8/bin:$PATH OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_HEADROOM_MIB=${UAGS_GPU_HEADROOM_MIB:-1024} UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-5}
ROOT=${ROOT:-/home/thanhnc/gauss_ws/src/uags/data/replica_semnerf/Replica_Dataset}
MAPS=${MAPS:-/home/thanhnc/gauss_ws/src/uags/results/maps}
SWAP=${SWAP:-/home/thanhnc/gauss_ws/src/uags/results/swap_relmax}
OUT=${OUT:-$SWAP}
SPEC=${SPEC:-configs/swap_replica.json}
PREREG=${PREREG:-configs/swap_trained_prereg.json}
FIELD_CONFIG=${FIELD_CONFIG-'{"change_weight": "relmax", "fast_min": 0.5, "slow_min": 0.5}'}
STAGE=${STAGE:-val}
DRY_RUN=${DRY_RUN:-0}
CLEAN=${CLEAN:-1}
MAP_NEED=${MAP_NEED:-7000}       # reset field (mapping + baselines + residuals): 6.3 GiB reserved in the paper run
BANK_NEED=${BANK_NEED:-5000}
GT_NEED=${GT_NEED:-7000}
TRAIN_NEED=${TRAIN_NEED:-5000}   # Table I trainings peaked at 1.9 GiB allocated
DISK_FACTOR=${DISK_FACTOR:-1.5}
TS=$(date +%Y%m%d-%H%M%S)
case "$STAGE" in val|test) ;; *) echo "STAGE must be val or test (got '$STAGE')"; exit 2 ;; esac
stamp() { echo "[$(date '+%F %T')] $*"; }
jget() {       # jget <json> <key.key...>: a value (lists space-separated)
  python3 - "$1" "$2" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."):
    v = v[k]
print(" ".join(map(str, v)) if isinstance(v, list) else v)
PY
}
finished() {   # finished <record>: status "ok" (a scorer record: the control passed)
  [ -f "$1" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status') == 'ok' else 1)" "$1"
}
step() {       # step <log> <command...>: output in <log>, its tail on failure (exit code kept)
  local L=$1; shift
  if [ "$DRY_RUN" = 1 ]; then printf '+'; printf ' %q' "$@"; printf '\n'; return 0; fi
  local rc=0
  "$@" > "$L" 2>&1 || rc=$?
  if [ "$rc" != 0 ]; then tail -20 "$L"; echo "stopped: exit $rc ($L)"; exit "$rc"; fi
  grep -vE "it/s|s/it|waiting|iter [0-9]+/" "$L" | tail -4
}
space() {      # space <dir> <map> <bytes per Gaussian>
  [ "$DRY_RUN" = 1 ] && return 0
  python3 scripts/disk_guard.py "$@" --factor "$DISK_FACTOR" || exit 1
}

if [ -n "$(git status --porcelain -- "$PREREG")" ] || ! git ls-files --error-unmatch "$PREREG" >/dev/null 2>&1; then
  echo "refused: $PREREG is not committed"; exit 1
fi
SCENES=$(jget "$PREREG" "splits.$STAGE")
[ "$SCENES" = "$(jget "$SPEC" "scenes.$STAGE")" ] || { echo "refused: the $STAGE scenes of $PREREG are not the spec's"; exit 1; }
LAM=$(jget "$PREREG" training.settings.lam_u)
CHUNK=$(jget "$PREREG" training.settings.channel_chunk)
DOWNSCALE=$(jget "$SPEC" downscale)
MSEQ=$(jget "$SPEC" mapping.sequence); MSTRIDE=$(jget "$SPEC" mapping.stride)
USEQ=$(jget "$SPEC" update.sequence); USTRIDE=$(jget "$SPEC" update.stride)
EXTRACTOR=$(jget "$SPEC" extractor)
HEAD=$(git rev-parse HEAD)
REF=()
if [ "$STAGE" = test ]; then
  if [ -n "$(git status --porcelain --untracked-files=no -- scripts usplat configs)" ]; then
    echo "refused: uncommitted changes in scripts/ usplat/ configs/ (the test run must name a commit)"; exit 1
  fi
  for s in $(jget "$PREREG" splits.val); do
    finished "$OUT/trained_val_${s}.json" || { echo "refused: no finished validation record $OUT/trained_val_${s}.json (run STAGE=val first)"; exit 1; }
    REF+=("$OUT/trained_val_${s}.json")
  done
  n_done=0; n=0
  for s in $SCENES; do n=$((n + 1)); finished "$OUT/trained_test_${s}.json" && n_done=$((n_done + 1)); done
  [ "$n_done" -lt "$n" ] || { echo "refused: every test record is finished (the test split runs once)"; exit 1; }
fi
for s in $SCENES; do
  [ -f "$MAPS/${s}_seq1.pt" ] || { echo "refused: no map $MAPS/${s}_seq1.pt"; exit 1; }
  [ -f "$SWAP/fields/${s}_swap.frames.json" ] || { echo "refused: no kept frames log $SWAP/fields/${s}_swap.frames.json"; exit 1; }
  finished "$OUT/eval_${STAGE}_${s}.json" || { echo "refused: no finished eval record $OUT/eval_${STAGE}_${s}.json (scripts/replica_swap.sh)"; exit 1; }
done
LOGS="$OUT/trained_logs"; RB="$OUT/trained_rebuild"
mkdir -p "$SWAP/banks" "$SWAP/fields" "$SWAP/gt" "$LOGS" "$RB"
stamp "swap trained: stage $STAGE git ${HEAD:0:7} dirty: $(git status --porcelain --untracked-files=no -- scripts usplat configs | wc -l) files; lambda_u $LAM; scenes: $SCENES"
[ -z "$FIELD_CONFIG" ] || stamp "field config: $FIELD_CONFIG"

for s in $SCENES; do
  REC="$OUT/trained_${STAGE}_${s}.json"
  EREC="$OUT/eval_${STAGE}_${s}.json"
  BANK="$SWAP/banks/$s.pt"
  F="$SWAP/fields/${s}_swap.pt"
  GT="$SWAP/gt/$s.pt"
  LOG0="$SWAP/fields/${s}_swap.frames.json"          # the paper run's frames log (kept)
  LOGR="$SWAP/fields/${s}_swap.rebuild.frames.json"   # the log of a rebuilt field
  MARK="$RB/${s}.rebuilt"
  if finished "$REC"; then
    if [ "$STAGE" = test ]; then
      python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d['provenance']['git']['commit'] == sys.argv[2] else 1)" "$REC" "$HEAD" \
        || { echo "refused: $REC is a finished test record of another commit"; exit 1; }
    fi
    stamp "$s: finished record kept ($REC)"; continue
  fi
  if [ ! -f "$BANK" ]; then
    [ ! -f "$F" ] || { echo "refused: $F exists without its bank $BANK (delete the field or restore the bank)"; exit 1; }
    stamp "$s: rebuild bank"
    step "$LOGS/bank_${s}.$TS.log" scripts/gpu_gate.sh "$BANK_NEED" \
      python3 scripts/make_swap_bank.py --spec "$SPEC" --scene "$s" --root "$ROOT" \
      --extractor "$EXTRACTOR" --device cuda --out "$BANK" --record "$RB/bank_${s}.json"
    [ "$DRY_RUN" = 1 ] || echo "$BANK" >> "$MARK"
    [ "$DRY_RUN" = 1 ] || python3 - "$RB/bank_${s}.json" "$OUT/bank_${s}.json" <<'PY' || exit 1
import json, sys
new, old = (json.load(open(p))["metrics"] for p in sys.argv[1:])
key = lambda m: ([(p["source_id"], p["target_id"], p["kind"]) for p in m["pairs"]], m["classes"],
                 {b: [(s["frame"], s["component"], s["area"], s["bbox"]) for s in v]
                  for b, v in m["segments"].items()})
if key(new) != key(old):
    sys.exit(f"refused: the rebuilt bank {sys.argv[1]} has other pairs or segments than the kept record {sys.argv[2]}")
print(f"bank rebuilt: pairs and segments equal the kept record")
PY
  fi
  if [ ! -f "$F" ]; then
    stamp "$s: rebuild the reset field"
    space "$SWAP/fields" "$MAPS/${s}_seq1.pt" 1300
    COMMON=(--backend gsplat --dataset replica_semnerf --data "$ROOT/$s/$MSEQ" --stride "$MSTRIDE"
            --downscale "$DOWNSCALE" --ply "$MAPS/${s}_seq1.pt" --extractor "$EXTRACTOR"
            --feat-dim 64 --pca-frames 20 --seed 0 --use-depth
            --update-data "$ROOT/$s/$USEQ" --update-stride "$USTRIDE"
            --feature-transform usplat.swap:swap_factory)
    [ -z "$FIELD_CONFIG" ] || COMMON+=(--field-config "$FIELD_CONFIG")
    step "$LOGS/map_${s}.$TS.log" scripts/gpu_gate.sh "$MAP_NEED" \
      python3 scripts/run_mapping.py "${COMMON[@]}" --baselines --change-baselines --oracle-depth \
      --feature-transform-arg "bank=$BANK,spec=$SPEC,scene=$s,log=$LOGR" --out "$F"
    [ "$DRY_RUN" = 1 ] || echo "$F" >> "$MARK"
  fi
  if [ ! -f "$GT" ]; then
    FLOG=$LOGR
    [ "$DRY_RUN" = 1 ] || FLOG=$(python3 -c "
import sys, torch
c = torch.load(sys.argv[1], map_location='cpu', weights_only=False, mmap=True)['config']
print(dict(p.split('=', 1) for p in c['feature_transform_arg'].split(','))['log'])" "$F")
    stamp "$s: rebuild the ground truth"
    step "$LOGS/gt_${s}.$TS.log" scripts/gpu_gate.sh "$GT_NEED" \
      python3 scripts/eval_change.py gt-swap --ply "$MAPS/${s}_seq1.pt" --data "$ROOT/$s/$USEQ" \
      --dataset replica_semnerf --update-stride "$USTRIDE" --frames-log "$FLOG" --bank "$BANK" \
      --spec "$SPEC" --scene "$s" --downscale "$DOWNSCALE" --backend gsplat --device cuda \
      --out "$GT" --record "$RB/gt_${s}.json"
    [ "$DRY_RUN" = 1 ] || echo "$GT" >> "$MARK"
  fi
  EV=(--field "$F" --gt "$GT" --config "$SPEC" --scene "$s" --split "$STAGE" --record "$EREC")
  stamp "$s: control (averaging vs $EREC)"
  step "$LOGS/control_${STAGE}_${s}.$TS.log" python3 scripts/swap_trained_eval.py "${EV[@]}" --check-only
  TR=()
  for mth in f3dgs legs; do
    name=f3dgs; [ "$mth" = legs ] && name="legs_l$(python3 -c "import sys; print(f'{float(sys.argv[1]):g}')" "$LAM")"
    T="${F%.pt}_${name}_upd.pt"
    TR+=("$T")
    if [ -f "$T" ]; then stamp "$s: reuse $T"; continue; fi
    stamp "$s: train ${name}_upd"
    step "$LOGS/train_${s}_${name}_upd.$TS.log" scripts/gpu_gate.sh "$TRAIN_NEED" \
      python3 scripts/distill_update_swap.py --field "$F" --method "$mth" --lam-u "$LAM" \
      --frames-ref "$LOG0" --channel-chunk "$CHUNK"
  done
  stamp "$s: score"
  step "$LOGS/score_${STAGE}_${s}.$TS.log" python3 scripts/swap_trained_eval.py "${EV[@]}" \
    --trained "$(IFS=,; echo "${TR[*]}")" --prereg "$PREREG" ${REF[@]+--reference "${REF[@]}"} --out "$REC"
  if [ "$CLEAN" = 1 ] && [ "$DRY_RUN" != 1 ] && finished "$REC" && [ -f "$MARK" ]; then
    while read -r p; do [ -n "$p" ] && rm -f -- "$p"; done < "$MARK"
    rm -f -- "$MARK"
    stamp "$s: removed the rebuilt bank / field / ground truth (records and trained files kept)"
  fi
done
[ "$DRY_RUN" = 1 ] || python3 - "$OUT" "$STAGE" $SCENES <<'PY'
import json, os, sys
out, stage, scenes = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = ("ours", "ours_noreset", "averaging", "f3dgs_upd", "legs_l4_upd")
print(f"{'scene':>10} " + " ".join(f"{r:>22}" for r in rows) + "   (acc_changed / acc_static)")
for s in scenes:
    d = json.load(open(os.path.join(out, f"trained_{stage}_{s}.json")))
    a = dict(d["record_rows"], **d["rows"])
    print(f"{s:>10} " + " ".join(f"{a[r]['acc_changed']:>10.4f} / {a[r]['acc_static']:.4f}"
                                  if r in a and "acc_changed" in a[r] else f"{'-':>22}" for r in rows))
PY
stamp "done: swap trained $STAGE -> $OUT/trained_${STAGE}_<scene>.json"
