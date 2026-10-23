#!/usr/bin/env bash
# Stage C of configs/extractor_prereg.json: every Table I row again with the extractor W that
# won stage B (scripts/extractor_select.py winner; refused unless the gate passed). One dataset
# per call:
#   DATASET=replica|scannet
#   STAGE=val   : map the validation scenes with W on the EXISTING 3DGS maps (run_mapping.py as
#                 Table I, --extractor W, the prereg's vocabulary), distill Feature 3DGS and
#                 LEGaussians for every lambda_u, evaluate (eval_seg.py as Table I, betas fit on
#                 validation) -> $OUT/eval_val.json; then scripts/extractor_select.py select-c
#                 (Table I's choices: CSS alpha0, P3 temperature, LEGaussians lambda_u; the 2D
#                 consistency check against stage A) -> $OUT/selection_val.json. Refused once the
#                 test stage started, and with uncommitted changes to tracked files.
#   STAGE=test  : map + distill (f3dgs, the chosen lambda_u) the test scenes, then evaluate them
#                 ONCE with the validation betas -> $OUT/eval_test.json. Refused on a dirty tree,
#                 unless select-c's selection holds (check-test: same prereg, same eval_val.json,
#                 choices reproduce, consistency passed, scripts/ usplat/ configs/ equal the
#                 validation commit), and when a finished test record exists (no override). Each
#                 attempt is appended to $OUT/look_test.json before the evaluation.
#   STAGE=table : Table I with W (scripts/make_tables.py tab1) -> $OUT/table1_<W>_<dataset>.tex
#   STAGE=claim : (after both datasets' test) claim C1 per dataset vs Table I's LSeg test
#                 records -> <extractor out>/claim_<W>.json
# Paths come from the prereg only: fields in datasets.<d>.fields (a new directory per
# extractor; never a Table I field), results in datasets.<d>.out; the scene files given to
# eval_seg.py are derived from configs/<d>_{val,test}.json with only the field paths changed
# ($OUT/scenes_{val,test}.json). The 3DGS maps are never refitted (a missing map is refused).
# Text cache: a COPY of stage A's cache of W in $OUT. GPU: every step through
# scripts/gpu_gate.sh at the prereg's measured caps for W (gpu.map_need_mib, distill_need_mib,
# eval_need_mib), headroom UAGS_GPU_HEADROOM_MIB (default 1024), never beside USGLoc.
# Overrides: PREREG, DRY_RUN=1 (print the GPU steps after the checks), DISK_FACTOR (1.5);
# UAGS_FIELDS_ROOT / UAGS_RESULTS_ROOT move the two roots (tests only; the Table I result and
# field directories are refused whatever they are).
# Logs: $OUT/<step>_<scene>.<timestamp>.log (never overwritten).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_USGLOC_EXTRA_MIB=${UAGS_GPU_USGLOC_EXTRA_MIB:-99999}   # never beside USGLoc
export UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-5}
export UAGS_GPU_HEADROOM_MIB=${UAGS_GPU_HEADROOM_MIB:-1024}
PREREG=${PREREG:-configs/extractor_prereg.json}
DATASET=${DATASET:-}
STAGE=${STAGE:-val}
DRY_RUN=${DRY_RUN:-0}
DISK_FACTOR=${DISK_FACTOR:-1.5}
FIELDS_ROOT=${UAGS_FIELDS_ROOT:-/home/thanhnc/gauss_ws/data/uags_fields}
RESULTS_ROOT=${UAGS_RESULTS_ROOT:-/home/thanhnc/gauss_ws/data/uags_results}
TABLE1_DIRS="/home/thanhnc/gauss_ws/data/uags_results/tab1 /home/thanhnc/gauss_ws/data/uags_results/tab1_v2 /home/thanhnc/gauss_ws/data/uags_results/scannet_tab1 /home/thanhnc/gauss_ws/data/uags_results/tab3 /home/thanhnc/gauss_ws/data/uags_results/views /home/thanhnc/gauss_ws/data/uags_results/smooth /home/thanhnc/gauss_ws/data/uags_results/fusion"
TS=$(date +%Y%m%d-%H%M%S)
case "$STAGE" in val|test|table|claim) ;; *) echo "STAGE must be val, test, table or claim (got '$STAGE')"; exit 2 ;; esac
[ "$STAGE" = claim ] || [ -n "$DATASET" ] || { echo "set DATASET (replica or scannet)"; exit 2; }

for v in $(env | grep -oE '^UAGS_(LSEG|PROXYCLIP)_[A-Z_]+' || true); do
  echo "refused: $v is set (it changes an extractor; unset it)"; exit 1
done
if ! git ls-files --error-unmatch "$PREREG" > /dev/null 2>&1 || \
   [ -n "$(git status --porcelain -- "$PREREG")" ]; then
  echo "refused: $PREREG is not committed (the rules must be fixed before any run)"; exit 1
fi
for f in usplat/extractors/radseg.py usplat/extractors/trident.py usplat/extractors/proxyclip_h.py \
         usplat/extractors/_vfm.py scripts/extractor_2d.py scripts/extractor_select.py \
         scripts/extractor_select.sh scripts/extractor_tab1.sh configs/vocab_replica_extractors.json \
         configs/vocab_scannet_extractors.json; do
  if [ -e "$f" ] && ! git ls-files --error-unmatch "$f" > /dev/null 2>&1; then
    echo "refused: $f is not committed (the records must name the code that ran)"; exit 1
  fi
done
A_OUT=$(python3 -c "import json,sys; p=json.load(open(sys.argv[1])); print(p['stage_a']['out'])" "$PREREG")
python3 -c "import json,sys; p=json.load(open(sys.argv[1])); sys.exit(0 if str((p.get('pipeline') or {}).get('stage_c','')).startswith('scripts/extractor_tab1.sh') and str(p.get('registered') or '').strip() else 1)" "$PREREG" \
  || { echo "refused: the prereg is not registered or names another stage C pipeline"; exit 1; }
W=$(python3 scripts/extractor_select.py winner --prereg "$PREREG" --dir "$A_OUT") || exit 1
echo "[extractor stage C] $TS git $(git rev-parse --short HEAD); winner $W; prereg sha256 $(sha256sum "$PREREG" | cut -c1-12)"

if [ "$STAGE" = claim ]; then
  read -r -a DS <<< "$(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['datasets']))" "$PREREG")"
  TESTS=()
  for d in "${DS[@]}"; do
    o=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['datasets'][sys.argv[2]]['out'].format(extractor=sys.argv[3]))" "$PREREG" "$d" "$W")
    python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status') == 'ok' else 1)" "$o/eval_test.json" 2>/dev/null \
      || { echo "refused: no finished test record $o/eval_test.json ($d)"; exit 1; }
    TESTS+=("$o/eval_test.json")
  done
  CO="$(dirname "$o")/claim_${W}.json"
  python3 scripts/extractor_select.py claim --prereg "$PREREG" --datasets "${DS[@]}" --tests "${TESTS[@]}" --out "$CO"
  echo "done: claim -> $CO"; exit 0
fi

CFG=$(python3 - "$PREREG" "$DATASET" "$W" <<'PY'
import json, shlex, sys
p, d, w = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
s = (p.get("datasets") or {}).get(d)
if s is None:
    sys.exit(f"refused: dataset {d} not in the prereg")
g = p.get("gpu") or {}
need = {k: (g.get(k) or {}).get(w) for k in ("map_need_mib", "distill_need_mib", "eval_need_mib")}
if not all(isinstance(v, int) for v in need.values()):
    sys.exit(f"refused: the prereg has no measured GPU caps for {w}: {need}")
m, e = s["mapping"], s["eval"]
for k, v in dict(OUT=s["out"].format(extractor=w), FIELDS=s["fields"].format(extractor=w),
                 VAL_SRC=s["scenes"]["validation"], TEST_SRC=s["scenes"]["test"],
                 VOCAB=e["vocab"], LOADER=e["dataset"], ESTRIDE=e["stride"],
                 DOWNSCALE=e["downscale"], TAU_COV=e["tau_cov"],
                 MLOADER=m["dataset"], MDATA=m["data"], MSTRIDE=m["stride"],
                 CSSDATA=m["css_eval_data"], CSSSTRIDE=m["css_eval_stride"], MAPS=s["maps"],
                 MAP_NEED=need["map_need_mib"], DISTILL_NEED=need["distill_need_mib"],
                 EVAL_NEED=need["eval_need_mib"]).items():
    print(f"{k}={shlex.quote(str(v))}")
PY
) || exit 1
eval "$CFG"
for r in "$OUT" "$FIELDS"; do
  for t in $TABLE1_DIRS; do
    case "$(realpath -m "$r")/" in "$(realpath -m "$t")"/*) echo "refused: $r is inside the result directory $t (never touched)"; exit 1 ;; esac
  done
  case "$(realpath -m "$r")" in
    /home/thanhnc/gauss_ws/data/uags_fields|/home/thanhnc/gauss_ws/data/uags_fields/scannet) echo "refused: $r is a Table I field directory"; exit 1 ;;
  esac
  case "$(realpath -m "$r")/" in
    "$(realpath -m "$RESULTS_ROOT/extractor")"/*|"$(realpath -m "$FIELDS_ROOT/extractor")"/*) ;;
    *) echo "refused: $r is not under $RESULTS_ROOT/extractor or $FIELDS_ROOT/extractor (Table I files are never written)"; exit 1 ;;
  esac
done
VAL=$OUT/eval_val.json SEL=$OUT/selection_val.json TEST=$OUT/eval_test.json LOOK=$OUT/look_test.json
test_started() { [ -e "$LOOK" ] || [ -e "$TEST" ] || [ -e "$TEST.partial.json" ]; }
finished() { [ -f "$1" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status','ok') == 'ok' else 1)" "$1"; }
mkdir -p "$OUT"

if [ "$STAGE" = table ]; then
  finished "$TEST" || { echo "refused: $TEST holds no finished test run"; exit 1; }
  python3 - "$TEST" "$VAL" <<'PY' || exit 1
import json, os, sys
t = json.load(open(sys.argv[1])); b = t["metrics"] if isinstance(t.get("metrics"), dict) else t
if b.get("betas_path") != os.path.abspath(sys.argv[2]):
    sys.exit(f"refused: {sys.argv[1]} used the betas of {b.get('betas_path')}, not {sys.argv[2]}")
PY
  TAB="$OUT/table1_${W}_${DATASET}.tex"
  python3 scripts/make_tables.py tab1 "$TEST" --val "$VAL" > "$TAB.tmp" && mv "$TAB.tmp" "$TAB"
  cat "$TAB"; echo "done: table -> $TAB"; exit 0
fi

# weights of W (sha256 of the prereg) and the scene files with W's fields
python3 - "$PREREG" "$W" <<'PY' || exit 1
import hashlib, json, os, sys
p, w = json.load(open(sys.argv[1])), sys.argv[2]
bad = []
for path, e in p["weights"].items():
    if w not in e["used_by"]:
        continue
    if not os.path.exists(path):
        bad.append(f"missing {path}"); continue
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    if h.hexdigest() != e["sha256"]:
        bad.append(f"{path}: sha256 {h.hexdigest()[:12]} != {e['sha256'][:12]}")
if bad:
    sys.exit("refused: " + "; ".join(bad))
PY
python3 - "$VAL_SRC" "$TEST_SRC" "$OUT" "$FIELDS" "$W" "$MAPS" <<'PY' || exit 1
import json, os, sys
import yaml
vsrc, tsrc, out, fields, w, maps = sys.argv[1:]
for split, src in (("val", vsrc), ("test", tsrc)):
    sc = yaml.safe_load(open(src))
    new = []
    for s in sc:
        if os.path.realpath(s["ply"]) != os.path.realpath(maps.format(name=s["name"])):
            sys.exit(f"refused: {src}: {s['name']} map {s['ply']} is not the prereg's {maps.format(name=s['name'])}")
        new.append(dict(s, field=os.path.join(fields, f"{s['name']}_{w}.pt")))
    dst = os.path.join(out, f"scenes_{split}.json")
    text = json.dumps(new, indent=1)
    if os.path.exists(dst) and open(dst).read() != text:
        sys.exit(f"refused: {dst} exists with other content")
    open(dst, "w").write(text)
print("scene files: " + ", ".join(f"{os.path.join(out, f'scenes_{k}.json')}" for k in ("val", "test")))
PY

if [ "$STAGE" = val ]; then
  if test_started; then echo "refused: the test stage has started ($LOOK / $TEST); the validation selection is frozen"; exit 1; fi
  if ! python3 -c "import sys; from usplat.runlog import git_state; g = git_state('.'); sys.exit(1 if g.get('error') or g['dirty_tracked'] else 0)"; then
    git status --short --untracked-files=no
    echo "refused: uncommitted changes to tracked files (the validation records must name a commit)"; exit 1
  fi
  SC="$OUT/scenes_val.json"
  DISTILL_JOBS="f3dgs $(python3 -c "from usplat.distill import LEGS_LAMBDAS as L; print(' '.join(f'legs:{l:g}' for l in L))")"
else
  if [ -n "$(git status --porcelain --untracked-files=no -- scripts usplat configs)" ]; then
    echo "refused: uncommitted changes in scripts/ usplat/ configs/ (the test run must name a commit)"; exit 1
  fi
  python3 scripts/extractor_select.py check-test --prereg "$PREREG" --sel "$SEL" --val "$VAL" --test "$TEST" || exit 1
  chosen=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['choices']['legs'])" "$SEL")
  DISTILL_JOBS="f3dgs legs:${chosen#legs_l}"
  SC="$OUT/scenes_test.json"
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['datasets'][sys.argv[2]]['test_look'])" "$PREREG" "$DATASET"
fi
step() {
  local L=$1; shift
  if [ "$DRY_RUN" = 1 ]; then echo "[dry-run] $*"; return 0; fi
  "$@" > "$L" 2>&1 || { tail -20 "$L"; exit 1; }
}
for s in $(python3 -c "import json,sys; print(' '.join(x['name'] for x in json.load(open(sys.argv[1]))))" "$SC"); do
  ply=${MAPS//\{name\}/$s}
  [ -f "$ply" ] || { echo "refused: the 3DGS map $ply is missing (stage C never refits maps)"; exit 1; }
  F="$FIELDS/${s}_${W}.pt"
  if [ ! -f "$F" ]; then
    [ "$DRY_RUN" = 1 ] || { mkdir -p "$FIELDS"; python3 scripts/disk_guard.py "$FIELDS" "$ply" 1600 --factor "$DISK_FACTOR" || exit 1; }
    step "$OUT/map_${s}.$TS.log" scripts/gpu_gate.sh "$MAP_NEED" python3 scripts/run_mapping.py \
      --backend gsplat --dataset "$MLOADER" --data "${MDATA//\{name\}/$s}" --stride "$MSTRIDE" \
      --downscale "$DOWNSCALE" --ply "$ply" --extractor "$W" --baselines --vocab "$VOCAB" \
      --css-eval-data "${CSSDATA//\{name\}/$s}" --css-eval-stride "$CSSSTRIDE" --feat-dim 64 \
      --pca-frames 20 --seed 0 --out "$F"
    [ "$DRY_RUN" = 1 ] || grep -E "gpu_gate|PCA|CSS|keyframes used|timing|saved" "$OUT/map_${s}.$TS.log" || true
  fi
  for job in $DISTILL_JOBS; do
    m=${job%%:*}; lam=${job#*:}; [ "$m" = "$job" ] && lam=1
    name=$m; [ "$m" = legs ] && name="legs_l$(python3 -c "print(f'{float($lam):g}')")"
    [ -f "$FIELDS/${s}_${W}_${name}.pt" ] && continue
    L="$OUT/distill_${s}_${name}.$TS.log"
    if [ "$DRY_RUN" = 1 ]; then
      echo "[dry-run] scripts/gpu_gate.sh $DISTILL_NEED python3 scripts/distill_features.py --field $F --method $m --lam-u $lam"
      continue
    fi
    ok=0
    for try in 1 2 3; do                 # a CUDA launch time-out kills the process: resume
      scripts/gpu_gate.sh "$DISTILL_NEED" python3 scripts/distill_features.py --field "$F" \
        --method "$m" --lam-u "$lam" >> "$L" 2>&1 && { ok=1; break; }
      echo "distill $s $name: attempt $try failed; resuming"
    done
    [ "$ok" = 1 ] || { tail -20 "$L"; exit 1; }
    grep -E "gpu_gate|mapping frames|resumed|converged|saved" "$L" || true
  done
done

TEXT_CACHE="$OUT/text_cache_${W}.pt"
if [ ! -f "$TEXT_CACHE" ] && [ -f "$A_OUT/text_cache_${DATASET}_${W}.pt" ] && [ "$DRY_RUN" != 1 ]; then
  cp "$A_OUT/text_cache_${DATASET}_${W}.pt" "$TEXT_CACHE"        # a copy of stage A's cache
fi
COMMON=(--vocab "$VOCAB" --row-factory usplat.baselines:row_factory,usplat.distill:row_factory
        --extractor "$W" --dataset "$LOADER" --stride "$ESTRIDE" --downscale "$DOWNSCALE"
        --tau-cov "$TAU_COV" --text-cache "$TEXT_CACHE" --with-2d)
if [ "$STAGE" = val ]; then
  ARGS=(--val-scenes "$SC" --scenes "$SC" --out "$VAL")
  if finished "$VAL"; then echo "$VAL finished, kept"; ARGS=(); fi
else
  ARGS=(--betas "$VAL" --scenes "$SC" --out "$TEST")
fi
if [ "$DRY_RUN" = 1 ]; then
  [ ${#ARGS[@]} -gt 0 ] && echo "[dry-run] scripts/gpu_gate.sh $EVAL_NEED python3 scripts/eval_seg.py ${COMMON[*]} ${ARGS[*]}"
  [ "$STAGE" = val ] && echo "[dry-run] scripts/extractor_select.py select-c -> $SEL"
  [ "$STAGE" = test ] && echo "[dry-run] look record $LOOK: attempt"
  echo "done (dry run): $STAGE"; exit 0
fi
L="$OUT/eval_${STAGE}.$TS.log"
if [ ${#ARGS[@]} -gt 0 ]; then
  if [ "$STAGE" = test ]; then
    python3 scripts/extractor_select.py look --sel "$SEL" --look "$LOOK" --ts "$TS" --log "$L"
  fi
  ok=0
  for try in 1 2 3; do                   # retried only after a crash (no finished record)
    scripts/gpu_gate.sh "$EVAL_NEED" python3 scripts/eval_seg.py "${COMMON[@]}" "${ARGS[@]}" \
      >> "$L" 2>&1 && { ok=1; break; }
    echo "eval $STAGE: attempt $try failed ($(grep -oE 'CUDA error: [^.]*|Error: [^.]*' "$L" | tail -1)); retrying"
  done
  [ "$ok" = 1 ] || { tail -30 "$L"; exit 1; }
  grep -vE "it/s|s/it|waiting" "$L" | tail -60
fi
if [ "$STAGE" = val ]; then
  python3 scripts/extractor_select.py select-c --prereg "$PREREG" --dataset "$DATASET" \
    --extractor "$W" --val "$VAL" --adir "$A_OUT" --out "$SEL"
fi
echo "done: stage C $DATASET $STAGE -> $OUT"
