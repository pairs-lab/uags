#!/usr/bin/env bash
# Stages A and B of configs/extractor_prereg.json (validation only; no test scene is read):
#   A  for every dataset of the prereg and every extractor (the incumbent, then the candidates
#      in the prereg's order): scripts/extractor_2d.py on the dataset's validation scenes
#      -> $OUT/a_<dataset>_<extractor>.json (a finished record is kept: a crash resumes)
#   B  scripts/extractor_select.py select -> $OUT/selection_a.json (R0 reproduction of Table I's
#      2D row by the incumbent, the winner, the gate). The selection is made once: a second
#      run is refused once selection_a.json exists or stage C has started.
# Gates (every run, DRY_RUN too): the prereg is committed, unmodified and 'registered'; it names
# this script; the extractor code, the drivers and the vocabularies are committed; no
# uncommitted change to tracked files (the records must name a commit); OUT is neither a
# Table I result directory nor inside uags_fields; the weight files match the prereg's sha256
# and the third-party code its commits; no UAGS_LSEG_* / UAGS_PROXYCLIP_* override is set (they
# change an extractor that the records would not show).
# GPU: each run through scripts/gpu_gate.sh at the prereg's gpu.stage_a_need_mib[<extractor>]
# (measured peak + margin), headroom UAGS_GPU_HEADROOM_MIB (default 1024 here), never beside a
# USGLoc job. Text caches: the incumbent gets a COPY of Table I's cache in $OUT; the
# candidates get new caches in $OUT. Nothing under uags_fields or a Table I directory is written.
# Overrides: PREREG, OUT, DEVICE (cpu: tests), DRY_RUN=1 (print the GPU steps after the checks).
# Logs: $OUT/a_<dataset>_<extractor>.<timestamp>.log (never overwritten).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_USGLOC_EXTRA_MIB=${UAGS_GPU_USGLOC_EXTRA_MIB:-99999}   # never beside USGLoc
export UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-5}
export UAGS_GPU_HEADROOM_MIB=${UAGS_GPU_HEADROOM_MIB:-1024}
PREREG=${PREREG:-configs/extractor_prereg.json}
DEVICE=${DEVICE:-cuda}
DRY_RUN=${DRY_RUN:-0}
FIELDS_ROOT=/home/thanhnc/gauss_ws/data/uags_fields
TS=$(date +%Y%m%d-%H%M%S)

for v in $(env | grep -oE '^UAGS_(LSEG|PROXYCLIP)_[A-Z_]+' || true); do
  echo "refused: $v is set (it changes an extractor; unset it)"; exit 1
done
if ! git ls-files --error-unmatch "$PREREG" > /dev/null 2>&1 || \
   [ -n "$(git status --porcelain -- "$PREREG")" ]; then
  echo "refused: $PREREG is not committed (the rules must be fixed before any run)"; exit 1
fi
for f in usplat/extractors/radseg.py usplat/extractors/trident.py usplat/extractors/proxyclip_h.py \
         usplat/extractors/_vfm.py scripts/extractor_2d.py scripts/extractor_select.py \
         scripts/extractor_select.sh configs/vocab_replica_extractors.json \
         configs/vocab_scannet_extractors.json; do
  if [ -e "$f" ] && ! git ls-files --error-unmatch "$f" > /dev/null 2>&1; then
    echo "refused: $f is not committed (the records must name the code that ran)"; exit 1
  fi
done
CFG=$(python3 - "$PREREG" <<'PY'
import json, shlex, sys
p = json.load(open(sys.argv[1]))
if not str((p.get("pipeline") or {}).get("stage_ab") or "").startswith("scripts/extractor_select.sh"):
    sys.exit("refused: the prereg names another pipeline")
if not str(p.get("registered") or "").strip():
    sys.exit("refused: the prereg is not registered yet (set registered and commit)")
names = [p["incumbent"]] + list(p["order"])
if sorted(p["order"]) != sorted(p["candidates"]):
    sys.exit("refused: the prereg order does not list exactly its candidates")
need = (p.get("gpu") or {}).get("stage_a_need_mib") or {}
miss = [n for n in names if not isinstance(need.get(n), int)]
if miss:
    sys.exit(f"refused: no measured gpu.stage_a_need_mib for {miss}")
print(f"P_OUT={shlex.quote(p['stage_a']['out'])}")
print(f"NAMES={shlex.quote(' '.join(names))}")
print(f"DATASETS={shlex.quote(' '.join(p['datasets']))}")
print("declare -A NEED=(" + " ".join(f"[{n}]={need[n]}" for n in names) + ")")
for d, s in p["datasets"].items():
    print(f"CACHE_{d}={shlex.quote(s['table1']['text_cache'])}")
PY
) || exit 1
eval "$CFG"
OUT=${OUT:-$P_OUT}
for d in tab1 tab1_v2 scannet_tab1 tab3 views views_repro smooth fusion; do
  case "$(realpath -m "$OUT")/" in
    "$(realpath -m "/home/thanhnc/gauss_ws/data/uags_results/$d")"/*)
      echo "refused: OUT is inside the result directory $d (never touched)"; exit 1 ;;
  esac
done
case "$(realpath -m "$OUT")/" in
  "$(realpath -m "$FIELDS_ROOT")"/*) echo "refused: OUT inside $FIELDS_ROOT"; exit 1 ;;
esac
if ! python3 -c "import sys; from usplat.runlog import git_state; g = git_state('.'); sys.exit(1 if g.get('error') or g['dirty_tracked'] else 0)"; then
  git status --short --untracked-files=no
  echo "refused: uncommitted changes to tracked files (the stage-A records must name a commit)"; exit 1
fi
python3 - "$PREREG" <<'PY' || exit 1
import hashlib, json, os, subprocess, sys
p = json.load(open(sys.argv[1]))
bad = []
for path, w in p["weights"].items():
    if not os.path.exists(path):
        bad.append(f"missing {path}")
        continue
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    if h.hexdigest() != w["sha256"]:
        bad.append(f"{path}: sha256 {h.hexdigest()[:12]} != {w['sha256'][:12]}")
for repo, commit in (p.get("third_party_code") or {}).items():
    r = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    if r.stdout.strip() != commit or dirty:
        bad.append(f"{repo}: at {r.stdout.strip()[:12] or '?'}{' (modified)' if dirty else ''}, the prereg names {commit[:12]}")
if bad:
    sys.exit("refused: " + "; ".join(bad))
print(f"weights ({len(p['weights'])} files) and third-party code: match the prereg")
PY
SEL="$OUT/selection_a.json"
if [ -e "$SEL" ]; then
  echo "refused: $SEL exists: stage A/B ran; the selection is final (stage C: scripts/extractor_tab1.sh)"; exit 1
fi
started=$(python3 - "$PREREG" <<'PY'
import glob, json, sys
p = json.load(open(sys.argv[1]))
hits = []
for d, s in p["datasets"].items():
    for x in p["order"]:
        hits += glob.glob(s["out"].format(extractor=x) + "/look_test.json")
print(" ".join(hits))
PY
)
if [ -n "$started" ]; then echo "refused: stage C has started ($started)"; exit 1; fi
mkdir -p "$OUT"
echo "[extractor stage A/B] $TS git $(git rev-parse --short HEAD); prereg $PREREG sha256 $(sha256sum "$PREREG" | cut -c1-12); OUT $OUT; extractors $NAMES; datasets $DATASETS"
finished() { [ -f "$1" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status') == 'ok' else 1)" "$1"; }
INC=${NAMES%% *}
for d in $DATASETS; do
  for x in $NAMES; do
    R="$OUT/a_${d}_${x}.json"
    if finished "$R"; then echo "$d $x: $R finished, kept"; continue; fi
    CACHE="$OUT/text_cache_${d}_${x}.pt"
    src_var="CACHE_${d}"
    if [ "$x" = "$INC" ] && [ ! -f "$CACHE" ] && [ -f "${!src_var}" ]; then
      cp "${!src_var}" "$CACHE"                 # a copy: Table I's cache is never written
    fi
    CMD=(python3 scripts/extractor_2d.py --prereg "$PREREG" --dataset "$d" --extractor "$x"
         --text-cache "$CACHE" --device "$DEVICE" --out "$R")
    if [ "$DRY_RUN" = 1 ]; then
      echo "[dry-run] scripts/gpu_gate.sh ${NEED[$x]} ${CMD[*]}"; continue
    fi
    L="$OUT/a_${d}_${x}.$TS.log"
    ok=0
    for try in 1 2; do                  # retried only after a crash (no finished record)
      scripts/gpu_gate.sh "${NEED[$x]}" "${CMD[@]}" >> "$L" 2>&1 && { ok=1; break; }
      echo "$d $x: attempt $try failed ($(grep -oE 'CUDA error: [^.]*|Error: [^.]*' "$L" | tail -1)); retrying"
    done
    [ "$ok" = 1 ] || { tail -30 "$L"; exit 1; }
    grep -E "2D mIoU|mean 2D" "$L" | tail -3
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); m=d.get('memory') or {}; b=d['metrics'] if isinstance(d.get('metrics'),dict) else d; print(f\"$d $x: peak GPU alloc {m.get('peak_gpu_alloc_gib', float('nan')):.2f} GiB (cap ${NEED[$x]} MiB), peak RSS {m.get('peak_rss_bytes', 0)/2**30:.1f} GiB\")" "$R" || true
  done
done
if [ "$DRY_RUN" = 1 ]; then echo "dry run: nothing extracted, no selection"; exit 0; fi
python3 scripts/extractor_select.py select --prereg "$PREREG" --dir "$OUT" --out "$SEL"
echo "done: stage A/B -> $SEL"
