#!/usr/bin/env bash
# lseg_inference_gate.sh — configs/lseg_inference_prereg.json: the validation-only 2D gate of LSeg's
# inference settings (token stride, flip, multi-scale). No map is rebuilt, no test scene is read.
#   per candidate (the prereg's order) and dataset: scripts/extractor_2d.py (unchanged) with the
#     candidate's UAGS_LSEG_* environment (usplat/extractors/lseg.py build()), through
#     scripts/gpu_gate.sh -> $OUT/a_<dataset>_<candidate>.json (a finished record is kept)
#   then the registered selection (R0, deltas, winner, gate) -> $OUT/selection.json and a table
# Refused: an uncommitted or modified prereg, extractor_2d.py or lseg.py (the records must name the
# code and the rules), a UAGS_LSEG_* variable set from outside, an OUT inside results/fields or a
# Table I directory. Text caches: COPIES of Table I's caches in $OUT.
# Overrides: PREREG, OUT, SELECT_ONLY=1 (no GPU step). Logs: $OUT/a_<dataset>_<candidate>.<time>.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export UAGS_GPU_POLL_S=${UAGS_GPU_POLL_S:-5}
export UAGS_GPU_HEADROOM_MIB=${UAGS_GPU_HEADROOM_MIB:-1024}
PREREG=${PREREG:-configs/lseg_inference_prereg.json}
TS=$(date +%Y%m%d-%H%M%S)

for v in $(env | grep -oE '^UAGS_LSEG_[A-Z_]+' || true); do
  echo "refused: $v is set from outside (the driver sets the candidate's variables itself)"; exit 1
done
for f in "$PREREG" scripts/extractor_2d.py usplat/extractors/lseg.py scripts/lseg_inference_gate.sh; do
  if ! git ls-files --error-unmatch "$f" > /dev/null 2>&1 || [ -n "$(git status --porcelain -- "$f")" ]; then
    echo "refused: $f is not committed or is modified"; exit 1
  fi
done
CFG=$(python3 - "$PREREG" <<'PY'
import json, shlex, sys
p = json.load(open(sys.argv[1]))
if not str(p.get("registered") or "").strip():
    sys.exit("refused: the prereg is not registered")
if sorted(p["order"]) != sorted(p["candidates"]) or p["order"][0] != "s8":
    sys.exit("refused: 'order' must list exactly the candidates, the incumbent s8 first")
need = p["gpu"]["need_mib"]
print(f"P_OUT={shlex.quote(p['stage_a']['out'])}")
print(f"NAMES={shlex.quote(' '.join(p['order']))}")
print(f"DATASETS={shlex.quote(' '.join(p['datasets']))}")
print("declare -A NEED=(" + " ".join(f"[{n}]={int(need[n])}" for n in p["order"]) + ")")
print("declare -A CENV=(" + " ".join(
    f"[{n}]={shlex.quote(' '.join(f'{k}={v}' for k, v in sorted(p['candidates'][n]['env'].items())))}"
    for n in p["order"]) + ")")
for d, s in p["datasets"].items():
    print(f"CACHE_{d}={shlex.quote(s['table1']['text_cache'])}")
PY
) || exit 1
eval "$CFG"
OUT=${OUT:-$P_OUT}
for d in fields tab1 tab1_v2 scannet_tab1 tab3 views smooth fusion extractor; do
  case "$(realpath -m "$OUT")/" in
    "$(realpath -m "results/$d")"/*) echo "refused: OUT is inside results/$d"; exit 1 ;;
  esac
done
mkdir -p "$OUT"
echo "[lseg gate] $TS git $(git rev-parse --short HEAD) candidates: $NAMES | datasets: $DATASETS -> $OUT"

finished() {   # finished <record>: status ok and not a smoke run
  [ -f "$1" ] && python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get('status') == 'ok' and not d['metrics'].get('smoke') else 1)" "$1"
}

if [ "${SELECT_ONLY:-0}" != 1 ]; then
  for d in $DATASETS; do
    src=CACHE_$d
    cp -n "${!src}" "$OUT/text_cache_${d}_lseg.pt"
  done
  for c in $NAMES; do
    for d in $DATASETS; do
      REC="$OUT/a_${d}_${c}.json"
      if finished "$REC"; then echo "$d $c: finished record kept"; continue; fi
      L="$OUT/a_${d}_${c}.$TS.log"
      # shellcheck disable=SC2086
      env ${CENV[$c]} scripts/gpu_gate.sh "${NEED[$c]}" python3 scripts/extractor_2d.py \
        --prereg "$PREREG" --dataset "$d" --extractor lseg \
        --text-cache "$OUT/text_cache_${d}_lseg.pt" --out "$REC" > "$L" 2>&1 \
        || { tail -20 "$L"; exit 1; }
      grep -E "2D mIoU" "$L" | tail -3
    done
  done
fi

python3 - "$PREREG" "$OUT" <<'PY' | tee "$OUT/selection.$TS.txt"
import json, os, sys
prereg_path, out = sys.argv[1], sys.argv[2]
p = json.load(open(prereg_path))
order, rule, tol_r0 = p["order"], p["rule"], float(p["reproduction"]["tol_points"])
per, ms, opts, missing = {}, {}, {}, []
for c in order:
    per[c], t = {}, []
    for d in p["datasets"]:
        path = os.path.join(out, f"a_{d}_{c}.json")
        if not os.path.exists(path):
            missing.append(path)
            continue
        r = json.load(open(path))
        m = r["metrics"]
        if r.get("status") != "ok" or m.get("smoke"):
            sys.exit(f"refused: {path} is not a finished full record")
        want = p["candidates"][c]["env"]
        o = m["extractor_options"]
        exp = (int(want.get("UAGS_LSEG_STRIDE", 8)), want.get("UAGS_LSEG_FLIP") == "1",
               [float(s) for s in want["UAGS_LSEG_SCALES"].split(",")] if want.get("UAGS_LSEG_SCALES") else None)
        got = (int(o["token_stride"]), bool(o["flip"]), list(o["scales"]) if o["scales"] else None)
        if exp != got:
            sys.exit(f"refused: {path} ran with {got}, the candidate is {exp}")
        opts[c] = o
        per[c][d] = {s: 100 * v["pca"]["miou"] for s, v in m["scenes"].items()}
        t += [v["ms_per_frame_extract"] for v in m["scenes"].values()]
    ms[c] = sum(t) / len(t) if t else None
if missing:
    sys.exit("refused: records missing: " + ", ".join(missing))
A = {c: {d: sum(v.values()) / len(v) for d, v in per[c].items()} for c in order}
delta = {c: {d: A[c][d] - A["s8"][d] for d in A[c]} for c in order}
r0 = {}
for d, spec in p["datasets"].items():
    t1 = json.load(open(spec["table1"]["val"]))
    t1 = t1.get("metrics", t1)["scenes"]
    for s, v in per["s8"][d].items():
        ref = 100 * t1[s]["rows"]["2d_features"]["all"]["miou"]
        r0[s] = dict(table1=ref, s8=v, abs_diff=abs(v - ref))
r0_ok = all(v["abs_diff"] <= tol_r0 for v in r0.values())
cands = [c for c in order if c != "s8"]
score = {c: min(delta[c].values()) for c in cands}
best = max(score.values())
winner = next(c for c in cands if score[c] >= best - float(rule["tie_tol_points"]))
gate = bool(r0_ok and score[winner] >= float(rule["margin_points"]))
sel = dict(prereg=os.path.abspath(prereg_path), per_scene=per, mean=A, delta=delta, min_delta=score,
           ms_per_frame=ms, options=opts, r0=r0, r0_ok=r0_ok, winner=winner if r0_ok else None,
           winner_min_delta=score[winner], margin_points=float(rule["margin_points"]), gate_pass=gate)
json.dump(sel, open(os.path.join(out, "selection.json"), "w"), indent=1)
scenes = [s for d in p["datasets"] for s in per["s8"][d]]
dsets = list(p["datasets"])
print("| candidate | " + " | ".join(scenes) + " | " + " | ".join(f"delta {d}" for d in dsets) + " | min delta | ms/frame |")
print("|---|" + "---|" * (len(scenes) + len(dsets) + 2))
for c in order:
    cells = [f"{per[c][d][s]:.2f}" for d in dsets for s in per[c][d]]
    print(f"| {c} | " + " | ".join(cells) + " | " + " | ".join(f"{delta[c][d]:+.2f}" for d in dsets)
          + f" | {min(delta[c].values()):+.2f} | {ms[c]:.0f} |")
print()
print("R0 (s8 vs Table I validation 2D row, points): "
      + ", ".join(f"{s} {v['s8']:.2f} vs {v['table1']:.2f}" for s, v in r0.items())
      + f" -> {'PASS' if r0_ok else 'FAIL'} (tolerance {tol_r0})")
print(f"winner: {winner} (min delta {score[winner]:+.2f}); gate (>= {rule['margin_points']} on both datasets): "
      f"{'PASS' if gate else 'FAIL'}")
PY
echo "done: lseg inference gate -> $OUT"
