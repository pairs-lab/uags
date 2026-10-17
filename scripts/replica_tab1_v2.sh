#!/usr/bin/env bash
# Table I v2 on Replica: the read-out-only predictives of usplat.rows.v2_rows (P2 on the P1
# base, the P1 prior grid, the P4 vote predictive) scored on the EXISTING v1 maps and fields:
# no mapping, no distillation, eval_seg.py --v2-rows only. The rules were written down before
# any v2 run in configs/tab1_v2_prereg.json (dated 2026-10-10; this script refuses to run while
# that file has uncommitted changes) and are applied by scripts/make_tables.py.
#   STAGE=val  : fit one temperature per row (v1 rows and v2 rows) on the validation scenes,
#                evaluate them -> $OUT/eval_val.json, then the selection (@p1, @p4, @ours_v2:
#                lowest pooled validation NLL) -> $OUT/selection_val.json. Refused once the
#                test stage has started ($OUT/look_test.json exists, even after a crash: the
#                selection is then final) and with uncommitted changes to tracked files (the
#                record must name a commit: check-test refuses a dirty validation record).
#   STAGE=test : the SECOND look at the Replica test split (the first: Table I v1, code
#                224fe1e, the prereg's v1.test_commit). Refused with uncommitted changes in
#                scripts/ usplat/ configs/, and (make_tables.py check-test) without a finished
#                v2 validation record of the prereg's validation scenes holding a beta for
#                every v2 row, made at a commit whose scripts/ usplat/ configs/ are the ones in
#                the tree; without a finished selection of that record under the current
#                prereg file; with another selection than an earlier test attempt; or when
#                $OUT/eval_test.json holds a finished run (no override: a crashed run leaves
#                status "running"/"failed" and may be rerun, with the same selection).
#                Appends the attempt to $OUT/look_test.json (look number, selection, git
#                state) before the evaluation, then writes $OUT/eval_test.json and
#                $OUT/table1_v2_rows.tex.
# Scenes: the files the prereg names (scenes.validation, scenes.test; disjoint). SCENES may
# point at another file (e.g. moved fields) with the same scene names (make_tables.py
# check-scenes). Another dataset later, same rules: commit its own prereg (a copy naming its
# scene files, with "pipeline" naming this script; v1.test_commit only if its test split was
# looked at before) and run with PREREG=<that file> and its own OUT. A prereg of another
# pipeline (e.g. configs/scannet_v2_prereg.json, scripts/scannet_tab1.sh) is refused: its test
# split has its own once-gate there.
# Text cache: TEXT_CACHE defaults to $OUT/text_cache_<extractor>.pt, seeded with a copy of the v1
# cache ($FIELDS/text_cache_lseg.pt, read only; eval_seg.py adds missing prompts to the cache it
# is given); a TEXT_CACHE inside $FIELDS is refused.
# Every GPU step goes through scripts/gpu_gate.sh (never beside a USGLoc job, poll every 5 s).
# GPU memory: the v1 evaluation peaked at 2.5 GB allocated (6.6 GB reserved) for 1M
# Gaussians; the v2 rows add ~2.3 GB of frozen buffers (11 P4 rows (N, K) and 4 new P1 rows
# (N, 64) in float32), hence EVAL_NEED 10000 MiB. The P4 Monte Carlo runs on the CPU while the
# rows are frozen (eval_seg keeps the field there): ~17 s per run and scene load for 1M
# Gaussians on 4 threads (3 lambdas + the P4_rbar ablation; validation loads each scene twice:
# fit, then evaluate).
# Overrides: PREREG (needs OUT), SCENES, VOCAB, DATASET, STRIDE, EXTRACTOR, ROW_FACTORY,
# TEXT_CACHE, WITH_2D=0, BACKEND/DEVICE (mock/cpu: smoke runs), EVAL_EXTRA (only --clip-model,
# --clip-pretrained, --clip-device, --channel-chunk: every option that decides what is fit,
# evaluated or written is set here, and placed after EVAL_EXTRA).
# Logs: $OUT/eval_<stage>.<timestamp>.log (never overwritten).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_USGLOC_EXTRA_MIB=${UAGS_GPU_USGLOC_EXTRA_MIB:-99999}   # never beside USGLoc
export UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-5}
FIELDS=/home/thanhnc/gauss_ws/data/uags_fields
V1_OUT=/home/thanhnc/gauss_ws/data/uags_results/tab1
PREREG_REPLICA=configs/tab1_v2_prereg.json
PREREG=${PREREG:-$PREREG_REPLICA}
if [ -z "${OUT:-}" ] && [ "$(realpath -m "$PREREG")" != "$(realpath -m "$PREREG_REPLICA")" ]; then
  echo "refused: PREREG=$PREREG is not the Replica prereg; set OUT for its records"; exit 1
fi
OUT=${OUT:-/home/thanhnc/gauss_ws/data/uags_results/tab1_v2}
python3 -c "import json, sys; sys.exit(0 if str(json.load(open(sys.argv[1])).get('pipeline') or '').startswith('scripts/replica_tab1_v2.sh') else 1)" "$PREREG" || {
  echo "refused: $PREREG is not a prereg of scripts/replica_tab1_v2.sh (its \"pipeline\" names another script, whose test split has its own gate)"; exit 1; }
STAGE=${STAGE:-val}
DOWNSCALE=${DOWNSCALE:-2}
TAU_COV=${TAU_COV:-0.5}          # coverage rule of v1, unchanged
EVAL_NEED=${EVAL_NEED:-10000}
VOCAB=${VOCAB:-configs/vocab_replica.json}
DATASET=${DATASET:-replica_semnerf}
STRIDE=${STRIDE:-10}
EXTRACTOR=${EXTRACTOR:-lseg}
ROW_FACTORY=${ROW_FACTORY-usplat.baselines:row_factory,usplat.distill:row_factory}
V1_TEXT_CACHE=$FIELDS/text_cache_lseg.pt           # read only
TEXT_CACHE=${TEXT_CACHE:-$OUT/text_cache_${EXTRACTOR}.pt}
case "$(realpath -m "$TEXT_CACHE")" in
  "$(realpath -m "$FIELDS")"/*)
    echo "refused: TEXT_CACHE $TEXT_CACHE is inside the v1 fields directory (eval_seg.py writes missing prompts into its cache): leave it unset for a copy in OUT"; exit 1 ;;
esac
WITH_2D=${WITH_2D:-1}
BACKEND=${BACKEND:-gsplat}
DEVICE=${DEVICE:-cuda}
TS=$(date +%Y%m%d-%H%M%S)
case "$STAGE" in
  val)  SPLIT=validation ;;
  test) SPLIT=test ;;
  *)    echo "STAGE must be val or test (got '$STAGE')"; exit 2 ;;
esac
if [ "$(realpath -m "$OUT")" = "$(realpath -m "$V1_OUT")" ]; then
  echo "refused: OUT is the v1 result directory $V1_OUT (v1 outputs are never touched)"; exit 1
fi
if ! git ls-files --error-unmatch "$PREREG" > /dev/null 2>&1 || \
   [ -n "$(git status --porcelain -- "$PREREG")" ]; then
  echo "refused: $PREREG is not committed (the v2 rules must be fixed before any run)"; exit 1
fi
# the scenes of this stage are the prereg's; an earlier test look is the prereg's v1 entry
SCENES=${SCENES:-$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['scenes'][sys.argv[2]])" "$PREREG" "$SPLIT")} || exit 1
PRIOR_LOOK=$(python3 -c "import json,sys; print((json.load(open(sys.argv[1])).get('v1') or {}).get('test_commit') or '')" "$PREREG") || exit 1
python3 scripts/make_tables.py check-scenes "$SCENES" --split "$SPLIT" --prereg "$PREREG" || exit 1
read -r -a EXTRA <<< "${EVAL_EXTRA:-}"
for a in ${EXTRA[@]+"${EXTRA[@]}"}; do   # resources and the text encoder only
  case "$a" in
    --clip-model|--clip-pretrained|--clip-device|--channel-chunk) ;;
    --clip-model=*|--clip-pretrained=*|--clip-device=*|--channel-chunk=*) ;;
    -*) echo "refused: EVAL_EXTRA option '$a' (allowed: --clip-model --clip-pretrained --clip-device --channel-chunk)"; exit 1 ;;
  esac
done
mkdir -p "$OUT"
if [ ! -f "$TEXT_CACHE" ] && [ "$EXTRACTOR" = lseg ] && [ -f "$V1_TEXT_CACHE" ]; then
  cp "$V1_TEXT_CACHE" "$TEXT_CACHE"           # the v1 text embeddings, never the v1 file
fi
echo "[v2 $STAGE] $TS git $(git rev-parse --short HEAD) dirty: $(git status --porcelain --untracked-files=no -- scripts usplat configs | wc -l) files; scenes $SCENES; prereg $PREREG sha256 $(sha256sum "$PREREG" | cut -c1-12)"
# the existing fields only: v2 never maps (build them with scripts/replica_tab1.sh EVAL=0)
python3 - "$SCENES" <<'PY' || exit 1
import json, os, sys
import yaml
sc = yaml.safe_load(open(sys.argv[1]))
miss = [f"{s['name']}: {k} {s[k]}" for s in sc for k in ("field", "ply") if not os.path.exists(s[k])]
if miss:
    sys.exit("refused: v2 evaluates the EXISTING maps and fields; missing " + "; ".join(miss))
print(f"scenes: {[s['name'] for s in sc]}")
PY

VAL="$OUT/eval_val.json"; SEL="$OUT/selection_val.json"; TEST="$OUT/eval_test.json"
LOOK="$OUT/look_test.json"
L="$OUT/eval_${STAGE}.$TS.log"
finished() { [ -f "$1" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status','ok') == 'ok' else 1)" "$1"; }
if [ "$STAGE" = val ]; then
  if [ -e "$LOOK" ]; then
    echo "refused: $LOOK exists: the test stage has started, the validation selection is final"; exit 1
  fi
  if finished "$TEST"; then
    echo "refused: $TEST holds a finished v2 test run; the validation selection is final"; exit 1
  fi
  # the record's git state (usplat.runlog) must be clean: make_tables.py check-test reads it
  if ! python3 -c "import sys; from usplat.runlog import git_state; g = git_state('.'); sys.exit(1 if g.get('error') or g['dirty_tracked'] else 0)"; then
    git status --short --untracked-files=no
    echo "refused: uncommitted changes to tracked files outside docs/ papers/ outputs/ (the validation run must name a commit)"; exit 1
  fi
  ARGS=(--val-scenes "$SCENES" --scenes "$SCENES" --out "$VAL")
else
  if [ -n "$(git status --porcelain --untracked-files=no -- scripts usplat configs)" ]; then
    echo "refused: uncommitted changes in scripts/ usplat/ configs/ (the test run must name a commit)"; exit 1
  fi
  python3 scripts/make_tables.py check-test "$VAL" --selection "$SEL" --test-record "$TEST" \
    --look "$LOOK" --prereg "$PREREG" || exit 1
  if [ -n "$PRIOR_LOOK" ]; then
    echo "SECOND LOOK at the test split $SCENES: the first was Table I v1 (code $PRIOR_LOOK); the v2 choices were fixed on validation ($SEL) before this run"
  else
    echo "FIRST look at the test split $SCENES; choices fixed on validation ($SEL)"
  fi
  # one look record per OUT, one entry per attempt (check-test held each to the same selection)
  python3 - "$SEL" "$LOOK" "$PRIOR_LOOK" "$SCENES" "$TS" "$L" <<'PY' || exit 1
import json, os, subprocess, sys
sel_path, out, prior, scenes, ts, log = sys.argv[1:]
s = json.load(open(sel_path))
sel = (s["metrics"] if isinstance(s.get("metrics"), dict) else s)["selection"]
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
attempt = dict(started=ts, log=log, scenes=scenes, git=dict(commit=git("rev-parse", "HEAD"),
               branch=git("rev-parse", "--abbrev-ref", "HEAD")))
if os.path.exists(out):                  # an earlier attempt crashed or was killed
    rec = json.load(open(out))
else:
    rec = dict(look=2 if prior else 1, prior_looks=[prior] if prior else [], scenes=scenes,
               selection_record=sel_path, chosen=sel["chosen"],
               prereg_sha256=sel["prereg"]["sha256"],
               validation_sha256=sel["validation"]["sha256"], attempts=[])
rec.setdefault("attempts", []).append(attempt)
with open(out + ".tmp", "w") as f:
    json.dump(rec, f, indent=1)
os.replace(out + ".tmp", out)
print(f"look record {out}: attempt {len(rec['attempts'])}, chosen on validation {sel['chosen']}")
PY
  ARGS=(--betas "$VAL" --scenes "$SCENES" --out "$TEST")
fi

COMMON=(--vocab "$VOCAB" --extractor "$EXTRACTOR" --dataset "$DATASET" --stride "$STRIDE"
        --downscale "$DOWNSCALE" --tau-cov "$TAU_COV" --text-cache "$TEXT_CACHE"
        --backend "$BACKEND" --device "$DEVICE" --v2-rows)
[ -n "$ROW_FACTORY" ] && COMMON+=(--row-factory "$ROW_FACTORY")
[ "$WITH_2D" = 1 ] && COMMON+=(--with-2d)
ok=0
for try in 1 2 3; do                     # retried only after a crash (no finished record)
  scripts/gpu_gate.sh "$EVAL_NEED" python3 scripts/eval_seg.py ${EXTRA[@]+"${EXTRA[@]}"} \
    "${COMMON[@]}" "${ARGS[@]}" >> "$L" 2>&1 && { ok=1; break; }
  echo "eval v2 $STAGE: attempt $try failed ($(grep -oE 'CUDA error: [^.]*|Error: [^.]*' "$L" | tail -1)); retrying"
done
[ "$ok" = 1 ] || { tail -30 "$L"; exit 1; }
grep -vE "it/s|s/it|waiting" "$L" | tail -60
if [ "$STAGE" = val ]; then
  python3 scripts/make_tables.py select "$VAL" --prereg "$PREREG" --out "$SEL"
  echo "selection on validation -> $SEL"
else
  python3 scripts/make_tables.py tab1 "$TEST" --val "$VAL" --v2 --prereg "$PREREG" \
    > "$OUT/table1_v2_rows.tex.tmp"
  mv "$OUT/table1_v2_rows.tex.tmp" "$OUT/table1_v2_rows.tex"
  echo "Table I v2 rows -> $OUT/table1_v2_rows.tex"
fi
echo "done: v2 $STAGE -> $OUT (log $L)"
