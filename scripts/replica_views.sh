#!/usr/bin/env bash
# Fig views on Replica (plan section 3): k-view fields (--max-keyframes k, evenly spaced over
# the Sequence_1 mapping frames, same 3DGS map) scored against the full Tab I field with
# scripts/eval_views.py. Needs the Tab I fields and $TAB1/eval_val.json (STAGE=val first).
#   STAGE=val : office_0, room_2      STAGE=test : the 6 test scenes (after Tab I test)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
ROOT=/home/thanhnc/gauss_ws/data/replica_semnerf/Replica_Dataset
MAPS=/home/thanhnc/gauss_ws/data/uags_maps
FIELDS=/home/thanhnc/gauss_ws/data/uags_fields
TAB1=${TAB1:-/home/thanhnc/gauss_ws/data/uags_results/tab1}
OUT=${OUT:-/home/thanhnc/gauss_ws/data/uags_results/views}
STAGE=${STAGE:-val}
KS=${KS:-"10 30 90"}
DOWNSCALE=${DOWNSCALE:-2}
TAU_COV=${TAU_COV:-0.5}
mkdir -p "$OUT"
[ -f "$TAB1/eval_val.json" ] || { echo "no $TAB1/eval_val.json: run replica_tab1.sh STAGE=val first"; exit 1; }
if [ "$STAGE" = val ]; then SCENES="office_0 room_2"; else SCENES="room_0 room_1 office_1 office_2 office_3 office_4"; fi

for s in $SCENES; do
  for k in $KS; do
    f="$FIELDS/${s}_lseg_k${k}.pt"
    [ -f "$f" ] && continue
    scripts/gpu_gate.sh 6000 python3 scripts/run_mapping.py --backend gsplat \
      --dataset replica_semnerf --data "$ROOT/$s/Sequence_1" --stride 5 --downscale "$DOWNSCALE" \
      --ply "$MAPS/${s}_seq1.pt" --extractor lseg --baselines --vocab configs/vocab_replica.json \
      --css-eval-data "$ROOT/$s/Sequence_2" --css-eval-stride 10 --feat-dim 64 --pca-frames 20 \
      --max-keyframes "$k" --out "$f" > "$OUT/map_${s}_k${k}.log" 2>&1
    grep -E "keyframes used|timing|saved" "$OUT/map_${s}_k${k}.log" || { tail -20 "$OUT/map_${s}_k${k}.log"; exit 1; }
  done
done

python3 - "$STAGE" "$SCENES" "$KS" "$OUT/scenes_$STAGE.json" <<'PY'
import json, sys
stage, scenes, ks, out = sys.argv[1], sys.argv[2].split(), sys.argv[3].split(), sys.argv[4]
R = "/home/thanhnc/gauss_ws/data/replica_semnerf/Replica_Dataset"
F = "/home/thanhnc/gauss_ws/data/uags_fields"
M = "/home/thanhnc/gauss_ws/data/uags_maps"
json.dump([dict(name=s, field=f"{F}/{s}_lseg.pt", ply=f"{M}/{s}_seq1.pt",
                data=f"{R}/{s}/Sequence_2", dataset="replica_semnerf", stride=10,
                fields={k: f"{F}/{s}_lseg_k{k}.pt" for k in ks}) for s in scenes],
          open(out, "w"), indent=1)
PY
scripts/gpu_gate.sh 4000 python3 scripts/eval_views.py --vocab configs/vocab_replica.json \
  --row-factory usplat.baselines:row_factory --extractor lseg --dataset replica_semnerf \
  --stride 10 --downscale "$DOWNSCALE" --tau-cov "$TAU_COV" --text-cache "$FIELDS/text_cache_lseg.pt" \
  --betas "$TAB1/eval_val.json" --scenes "$OUT/scenes_$STAGE.json" --out "$OUT/views_$STAGE.json"
echo "done: views $STAGE -> $OUT/views_$STAGE.json"
