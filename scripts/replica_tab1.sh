#!/usr/bin/env bash
# Table I on Replica (Semantic-NeRF renders). Plan section 3:
#   map on Sequence_1 every 5th frame (180 frames; the near-duplicate pose filter of
#   run_mapping.py keeps 176-180), evaluate on Sequence_2 every 10th frame (90);
#   validation scenes office_0, room_2 fit the temperatures; test scenes run ONCE with them.
#   STAGE=val  : fit + map + evaluate the validation scenes, fit betas -> $OUT/eval_val.json
#   STAGE=test : fit + map the test scenes, evaluate them with the frozen betas (run ONCE:
#                refused when $OUT/eval_test.json exists, unless FORCE_TEST=1)
#   EVAL=0     : build the maps and fields only (label-free preparation), no evaluation
# Every GPU step goes through scripts/gpu_gate.sh (default here: never start beside a USGLoc
# job, poll every 5 s). Existing maps/fields are reused; eval_seg.py --betas refuses test
# fields built with other mapping code, options or protocol than the validation fields.
# Rows: ours P0/P1/P2/P3, ours_scalar, baselines (usplat.baselines, with the CSS prior
# read-outs css_a<alpha0>), the trained baselines Feature 3DGS and LEGaussians (usplat.distill,
# trained by scripts/distill_features.py on the same map, frames, features and PCA), and the
# 2D row (--with-2d: LSeg on each evaluation frame).
# Logs: $OUT/<step>_<scene>.<timestamp>.log (never overwritten); the stage log records the git
# state. Results: $OUT/eval_<stage>.json (+ .partial.json after every scene).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_USGLOC_EXTRA_MIB=${UAGS_GPU_USGLOC_EXTRA_MIB:-99999}   # never beside USGLoc
export UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-5}
ROOT=/home/thanhnc/gauss_ws/data/replica_semnerf/Replica_Dataset
MAPS=/home/thanhnc/gauss_ws/data/uags_maps
FIELDS=/home/thanhnc/gauss_ws/data/uags_fields
OUT=${OUT:-/home/thanhnc/gauss_ws/data/uags_results/tab1}
STAGE=${STAGE:-val}
EVAL=${EVAL:-1}
ITERS=${ITERS:-30000}
DOWNSCALE=${DOWNSCALE:-2}
TAU_COV=${TAU_COV:-0.5}          # coverage rule, fixed before any validation run
# GPU memory caps (MiB): mapping peaks at ~4.8 GB for 620k Gaussians and grows with N
# (room_2, 954k: 5.7 GB); the evaluation holds LSeg, then the frozen rows
MAP_NEED=${MAP_NEED:-9000}
EVAL_NEED=${EVAL_NEED:-7000}
DISTILL_NEED=${DISTILL_NEED:-7000}
DISTILL=${DISTILL:-"f3dgs legs"}
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p "$MAPS" "$FIELDS" "$OUT"
case "$STAGE" in
  val)  SCENES="office_0 room_2" ;;
  test) SCENES="room_0 room_1 office_1 office_2 office_3 office_4" ;;
  *)    echo "STAGE must be val or test (got '$STAGE')"; exit 2 ;;
esac
echo "[$STAGE] $TS git $(git rev-parse --short HEAD) dirty: $(git status --porcelain --untracked-files=no -- scripts usplat configs | wc -l) files"
if [ "$STAGE" = test ]; then
  if [ -n "$(git status --porcelain --untracked-files=no -- scripts usplat configs)" ]; then
    echo "refused: uncommitted changes in scripts/ usplat/ configs/ (the test run must name a commit)"; exit 1
  fi
  [ "$EVAL" = 1 ] && { python3 - "$OUT/eval_val.json" <<'PY' || exit 1; }
import json, sys
d = json.load(open(sys.argv[1]))
body = d.get("metrics") if isinstance(d.get("metrics"), dict) else d
if d.get("status", "ok") not in ("ok", "success", "finished") or not body.get("betas"):
    sys.exit(f"refused: {sys.argv[1]} is not a finished validation fit with betas "
             f"(status {d.get('status')}, betas {bool(body.get('betas'))})")
print(f"validation fit: {len(body['betas'])} rows, fields {[i[:12] for i in body['beta_fit_settings'].get('field_identities', [])]}")
PY
  if [ "$EVAL" = 1 ] && [ -f "$OUT/eval_test.json" ] && [ "${FORCE_TEST:-0}" != 1 ]; then
    echo "refused: $OUT/eval_test.json exists (the test split runs once; FORCE_TEST=1 to override)"; exit 1
  fi
fi

for s in $SCENES; do
  if [ ! -f "$MAPS/${s}_seq1.pt" ]; then
    L="$OUT/fit_${s}.$TS.log"
    scripts/gpu_gate.sh 3000 python3 scripts/fit_gs.py --dataset replica_semnerf \
      --data "$ROOT/$s/Sequence_1" --stride 5 --iters "$ITERS" --max-gaussians 1000000 \
      --out "$MAPS/${s}_seq1.pt" > "$L" 2>&1 || { tail -20 "$L"; exit 1; }
    grep -E "gpu_gate|PSNR|psnr|Gaussians|saved" "$L" | tail -5
  fi
  if [ ! -f "$FIELDS/${s}_lseg.pt" ]; then
    L="$OUT/map_${s}.$TS.log"
    scripts/gpu_gate.sh "$MAP_NEED" python3 scripts/run_mapping.py --backend gsplat \
      --dataset replica_semnerf --data "$ROOT/$s/Sequence_1" --stride 5 --downscale "$DOWNSCALE" \
      --ply "$MAPS/${s}_seq1.pt" --extractor lseg --baselines --vocab configs/vocab_replica.json \
      --css-eval-data "$ROOT/$s/Sequence_2" --css-eval-stride 10 --feat-dim 64 --pca-frames 20 \
      --seed 0 --out "$FIELDS/${s}_lseg.pt" > "$L" 2>&1 || { tail -20 "$L"; exit 1; }
    grep -E "gpu_gate|PCA|CSS|keyframes used|timing|saved" "$L"
    ls -la "$FIELDS/${s}_lseg.pt"
  fi
  for m in $DISTILL; do                  # trained baselines (Feature 3DGS, LEGaussians)
    [ -f "$FIELDS/${s}_lseg_${m}.pt" ] && continue
    L="$OUT/distill_${s}_${m}.$TS.log"
    scripts/gpu_gate.sh "$DISTILL_NEED" python3 scripts/distill_features.py \
      --field "$FIELDS/${s}_lseg.pt" --method "$m" > "$L" 2>&1 || { tail -20 "$L"; exit 1; }
    grep -E "gpu_gate|mapping frames|iter [0-9]+0000/|saved" "$L"
  done
done

[ "$EVAL" = 1 ] || { echo "done: $STAGE maps and fields only (EVAL=0)"; exit 0; }
COMMON=(--vocab configs/vocab_replica.json --row-factory usplat.baselines:row_factory,usplat.distill:row_factory
        --extractor lseg --dataset replica_semnerf --stride 10 --downscale "$DOWNSCALE"
        --tau-cov "$TAU_COV" --text-cache "$FIELDS/text_cache_lseg.pt" --with-2d)
L="$OUT/eval_${STAGE}.$TS.log"
if [ "$STAGE" = val ]; then
  scripts/gpu_gate.sh "$EVAL_NEED" python3 scripts/eval_seg.py "${COMMON[@]}" \
    --val-scenes configs/replica_val.json --scenes configs/replica_val.json \
    --out "$OUT/eval_val.json" > "$L" 2>&1 || { tail -30 "$L"; exit 1; }
else
  scripts/gpu_gate.sh "$EVAL_NEED" python3 scripts/eval_seg.py "${COMMON[@]}" \
    --betas "$OUT/eval_val.json" --scenes configs/replica_test.json \
    --out "$OUT/eval_test.json" > "$L" 2>&1 || { tail -30 "$L"; exit 1; }
fi
grep -vE "it/s|s/it|waiting" "$L" | tail -90
echo "done: $STAGE -> $OUT (log $L)"
