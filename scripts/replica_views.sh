#!/usr/bin/env bash
# Fig views on Replica (plan section 3): k-view fields (--max-keyframes k, evenly spaced over
# the Sequence_1 mapping frames, same 3DGS map) scored against the full Tab I field with
# scripts/eval_views.py. Needs the Tab I fields and $TAB1/eval_val.json (STAGE=val first).
#   STAGE=val : office_0, room_2      STAGE=test : the 6 test scenes (after Tab I test)
# One scene at a time: map its k-view fields, score them, then delete them (KEEP=1 keeps
# them; the disk is small and each field is ~0.3 GB). The k-view fields carry the view-hook
# baselines (NormLift, averaging, view count) but no CSS and no change test, and share the
# full field's PCA basis and centre (--pca-from), so e_red / e_irr compare like with like.
# v1 outputs are read, never written: the full fields and the text cache are READ from $FIELDS;
# the k-view fields go to $KFIELDS (default $OUT/kfields, never $FIELDS) and the text cache is
# copied into $OUT (eval_seg.py adds missing prompts to the cache it is given). OUT defaults to
# a new directory and is refused when it is the v1 views directory. A scene is skipped only when
# its per-scene file holds the reducibility analysis (eval_views.py since T4b); an older file
# is refused (move it aside).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_USGLOC_EXTRA_MIB=${UAGS_GPU_USGLOC_EXTRA_MIB:-99999}   # never beside USGLoc
export UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-5}
TS=$(date +%Y%m%d-%H%M%S)
ROOT=${ROOT:-/home/thanhnc/gauss_ws/src/uags/data/replica_semnerf/Replica_Dataset}
MAPS=${MAPS:-/home/thanhnc/gauss_ws/src/uags/results/maps}
FIELDS=${FIELDS:-/home/thanhnc/gauss_ws/src/uags/results/fields}           # read only
V1_FIELDS=/home/thanhnc/gauss_ws/src/uags/results/fields
V1_OUT=/home/thanhnc/gauss_ws/src/uags/results/views
TAB1=${TAB1:-/home/thanhnc/gauss_ws/src/uags/results/tab1}
OUT=${OUT:-/home/thanhnc/gauss_ws/src/uags/results/views_v2}
KFIELDS=${KFIELDS:-$OUT/kfields}
STAGE=${STAGE:-val}
KS=${KS:-"10 30 90"}
DOWNSCALE=${DOWNSCALE:-2}
TAU_COV=${TAU_COV:-0.5}
# GPU memory caps (MiB): mapping peaks at ~4.8 GB for 620k Gaussians and grows with N
# (room_2, 954k, went past 5.7 GB); the evaluation holds LSeg and the field together
MAP_NEED=${MAP_NEED:-9000}
EVAL_NEED=${EVAL_NEED:-7000}
KEEP=${KEEP:-0}
if [ "$(realpath -m "$OUT")" = "$(realpath -m "$V1_OUT")" ]; then
  echo "refused: OUT is the v1 views directory $V1_OUT (v1 outputs are never touched)"; exit 1
fi
for d in "$FIELDS" "$V1_FIELDS"; do
  if [ "$(realpath -m "$KFIELDS")" = "$(realpath -m "$d")" ]; then
    echo "refused: KFIELDS is the fields directory $d (k-view fields never go next to the v1 fields)"; exit 1
  fi
done
[ -f "$TAB1/eval_val.json" ] || { echo "no $TAB1/eval_val.json: run replica_tab1.sh STAGE=val first"; exit 1; }
mkdir -p "$OUT" "$KFIELDS"
TEXT_CACHE="$OUT/text_cache_lseg.pt"            # a copy: the v1 cache is never written
[ -f "$TEXT_CACHE" ] || [ ! -f "$FIELDS/text_cache_lseg.pt" ] || cp "$FIELDS/text_cache_lseg.pt" "$TEXT_CACHE"
has_reducibility() {   # has_reducibility <per-scene json>: eval_views.py output with the analysis
  python3 -c "import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if any('reducibility' in r for sc in d.get('scenes', {}).values() for kv in (sc.get('k') or {}).values() for r in (kv.get('rows') or {}).values()) else 1)" "$1"
}
case "$STAGE" in
  val)  SCENES="office_0 room_2" ;;
  test) SCENES="room_0 room_1 office_1 office_2 office_3 office_4" ;;
  *)    echo "STAGE must be val or test (got '$STAGE')"; exit 2 ;;
esac
echo "[views $STAGE] $TS git $(git rev-parse --short HEAD)"

for s in $SCENES; do
  if [ -f "$OUT/views_${STAGE}_${s}.json" ]; then
    has_reducibility "$OUT/views_${STAGE}_${s}.json" || {
      echo "refused: $OUT/views_${STAGE}_${s}.json has no reducibility analysis (older eval_views.py): move it aside"; exit 1; }
    echo "$s: done"; continue
  fi
  for k in $KS; do
    f="$KFIELDS/${s}_lseg_k${k}.pt"
    [ -f "$f" ] && continue
    L="$OUT/map_${s}_k${k}.$TS.log"
    scripts/gpu_gate.sh "$MAP_NEED" python3 scripts/run_mapping.py --backend gsplat \
      --dataset replica_semnerf --data "$ROOT/$s/Sequence_1" --stride 5 --downscale "$DOWNSCALE" \
      --ply "$MAPS/${s}_seq1.pt" --extractor lseg --baselines --feat-dim 64 --pca-frames 20 \
      --seed 0 --pca-from "$FIELDS/${s}_lseg.pt" \
      --field-config '{"change_test": false}' --max-keyframes "$k" --out "$f" \
      > "$L" 2>&1 || { tail -20 "$L"; exit 1; }
    grep -E "keyframes used|timing|saved" "$L"
  done
  python3 - "$s" "$KS" "$OUT/scenes_${STAGE}_${s}.json" "$ROOT" "$FIELDS" "$MAPS" "$KFIELDS" <<'PY'
import json, sys
s, ks, out, R, F, M, KF = sys.argv[1], sys.argv[2].split(), *sys.argv[3:]
json.dump([dict(name=s, field=f"{F}/{s}_lseg.pt", ply=f"{M}/{s}_seq1.pt",
                data=f"{R}/{s}/Sequence_2", dataset="replica_semnerf", stride=10,
                fields={k: f"{KF}/{s}_lseg_k{k}.pt" for k in ks})], open(out, "w"), indent=1)
PY
  scripts/gpu_gate.sh "$EVAL_NEED" python3 scripts/eval_views.py --vocab configs/vocab_replica.json \
    --row-factory usplat.baselines:row_factory --extractor lseg --dataset replica_semnerf \
    --stride 10 --downscale "$DOWNSCALE" --tau-cov "$TAU_COV" --text-cache "$TEXT_CACHE" \
    --rows ours_P0,ours_P2,averaging,normlift,viewcount \
    --betas "$TAB1/eval_val.json" --scenes "$OUT/scenes_${STAGE}_${s}.json" \
    --out "$OUT/views_${STAGE}_${s}.json" > "$OUT/eval_${STAGE}_${s}.$TS.log" 2>&1 \
    || { tail -30 "$OUT/eval_${STAGE}_${s}.$TS.log"; exit 1; }
  grep -E " k= " "$OUT/eval_${STAGE}_${s}.$TS.log" || true
  if [ "$KEEP" != 1 ]; then for k in $KS; do rm -f "$KFIELDS/${s}_lseg_k${k}.pt"; done; fi
done

python3 - "$OUT" "$STAGE" "$SCENES" <<'PY'
import json, sys
sys.path.insert(0, "scripts")
from eval_views import summarize, summarize_on_defined
out, stage, scenes = sys.argv[1], sys.argv[2], sys.argv[3].split()
per = {s: json.load(open(f"{out}/views_{stage}_{s}.json")) for s in scenes}
ks = per[scenes[0]]["ks"]
res = {s: per[s]["scenes"][s] for s in scenes}
json.dump(dict(ks=ks, scenes=res, summary=summarize(res, ks),
               summary_on_ale_defined=summarize_on_defined(res, ks),
               parts={s: f"{out}/views_{stage}_{s}.json" for s in scenes}),
          open(f"{out}/views_{stage}.json", "w"), indent=1)
print(f"merged {len(scenes)} scenes -> {out}/views_{stage}.json")
PY
echo "done: views $STAGE -> $OUT/views_$STAGE.json"
