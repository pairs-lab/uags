#!/usr/bin/env bash
# nor_distilled.sh — configs/nor_distilled_prereg.json: Feature 3DGS and LEGaussians as Table III rows.
#   scripts/nor_distilled.sh all        repro, val (train + eval), select, test (train + eval once), merge
# Steps skip what is already finished (a distilled file or a finished record); a test record is never
# overwritten. GPU steps go through scripts/gpu_gate.sh. Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/venvs/uags/bin:$HOME/.local/cuda-12.8/bin:$PATH OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
VAL="001 002 003 004"
TEST="005 006 007 008 009 010 011 013 014 015"
OUT=results/nor/distilled
LOGS=${LOGS:-docs/reports/2026-11-07-nor-trained-baselines/logs}
mkdir -p $OUT/val $OUT/test $OUT/repro $OUT/logs "$LOGS"
stamp() { echo "[$(date '+%F %T')] $*"; }
field() { echo results/fields/nor_$1_lseg_relmax.pt; }
done_rec() { python3 -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get('status')=='ok' else 1)" "$1" 2>/dev/null; }

train() {   # scene method lambda
  local f; f=$(field "$1")
  local name=$2; [ "$2" = legs ] && name="legs_l$3"
  local out="${f%.pt}_${name}.pt"
  if [ -f "$out" ]; then stamp "skip train $1 $name (exists)"; return; fi
  stamp "train $1 $name"
  scripts/gpu_gate.sh 5000 python scripts/distill_features.py --field "$f" --method "$2" \
      $([ "$2" = legs ] && echo --lam-u "$3") > "$OUT/logs/train_$1_$name.log" 2>&1
  tail -1 "$OUT/logs/train_$1_$name.log"
}

evaluate() {   # split scene files
  local rec="$OUT/$1/$2.json"
  if done_rec "$rec"; then stamp "skip eval $1 $2 (finished)"; return; fi
  stamp "eval $1 $2"
  scripts/gpu_gate.sh 6000 python scripts/eval_change_nor_distilled.py --field "$(field "$2")" --align none \
      --update-stride 1 --tau-cov 0.5 --distilled "$3" --reference "results/nor/tab2_relmax/$1/$2.json" \
      --out "$rec" > "$OUT/logs/eval_$1_$2.log" 2>&1
  tail -n 6 "$OUT/logs/eval_$1_$2.log"
}

repro() {
  local rec=$OUT/repro/001.json
  if ! done_rec "$rec"; then
    stamp "repro: eval_change_nor.py on val 001"
    scripts/gpu_gate.sh 6000 python scripts/eval_change_nor.py --field "$(field 001)" --align none \
        --update-stride 1 --tau-cov 0.5 --out "$rec" > "$OUT/logs/repro_001.log" 2>&1
  fi
  python3 - "$rec" results/nor/tab2_relmax/val/001.json "$OUT/repro/compare.json" <<'EOF'
import json, sys
a, b = (json.load(open(p))["metrics"] for p in sys.argv[1:3])
rows, out = a["rows"], {}
for r in rows:
    pa, pb = (a["pixel"]["rows"].get(r, {}), b["pixel"]["rows"].get(r, {}))
    oa, ob = a["operating_point"][r], b["operating_point"][r]
    vals = dict(g_ap=(a["gaussian"]["rows"][r]["ap"], b["gaussian"]["rows"][r]["ap"]),
                g_auroc=(a["gaussian"]["rows"][r]["auroc"], b["gaussian"]["rows"][r]["auroc"]),
                p_ap=(pa.get("ap"), pb.get("ap")), p_auroc=(pa.get("auroc"), pb.get("auroc")),
                fa=(oa.get("fa"), ob.get("fa")), rec=(oa.get("rec"), ob.get("rec")))
    out[r] = {k: dict(rerun=x, record=y, exact=x == y, abs_diff=None if x is None or y is None else abs(x - y))
              for k, (x, y) in vals.items()}
diffs = [v["abs_diff"] for r in out.values() for v in r.values() if v["abs_diff"] is not None]
out["all_exact"] = all(v["exact"] for r in rows for v in out[r].values())
out["max_abs_diff"] = max(diffs) if diffs else None
out["counts_equal"] = a["counts"]["n_observed"] == b["counts"]["n_observed"]
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(f"repro 001: all rows exact {out['all_exact']}, max |diff| {out['max_abs_diff']}, n_observed equal {out['counts_equal']}")
EOF
}

val() {
  for s in $VAL; do
    train "$s" f3dgs 0
    for l in 1 2 4; do train "$s" legs "$l"; done
    f=$(field "$s"); evaluate val "$s" "${f%.pt}_f3dgs.pt,${f%.pt}_legs_l1.pt,${f%.pt}_legs_l2.pt,${f%.pt}_legs_l4.pt"
  done
}

select_lambda() {
  stamp "select lambda_u on validation"
  python scripts/nor_distilled_table.py select --dir $OUT/val --scenes "${VAL// /,}" \
      --rows legs_l1_feat,legs_l2_feat,legs_l4_feat --out $OUT/selection.json
}

test_split() {
  local lam; lam=$(python3 -c "import json; print('%g' % json.load(open('$OUT/selection.json'))['lambda_u'])")
  stamp "test with lambda_u = $lam"
  for s in $TEST; do
    train "$s" f3dgs 0
    train "$s" legs "$lam"
    local rec="$OUT/test/$s.json"
    if [ -f "$rec" ]; then stamp "test $s: a record exists, not run again"; continue; fi
    f=$(field "$s"); evaluate test "$s" "${f%.pt}_f3dgs.pt,${f%.pt}_legs_l${lam}.pt"
  done
}

merge() {
  local lam; lam=$(python3 -c "import json; print('%g' % json.load(open('$OUT/selection.json'))['lambda_u'])")
  python scripts/nor_distilled_table.py merge --dir $OUT/val --scenes "${VAL// /,}" \
      --rows f3dgs_feat,legs_l1_feat,legs_l2_feat,legs_l4_feat,feat_control --out $OUT/merge_val.json
  python scripts/nor_distilled_table.py merge --dir $OUT/test --scenes "${TEST// /,}" \
      --rows "f3dgs_feat,legs_l${lam}_feat" --labels "Feature 3DGS, feature residual;LEGaussians, feature residual" \
      --out $OUT/merge_test.json --tex "$LOGS/rows.tex"
  python scripts/nor_distilled_table.py merge --dir $OUT/test --scenes "${TEST// /,}" \
      --rows feat_control --out $OUT/merge_test_control.json
}

case "${1:-all}" in
  repro) repro ;; val) val ;; select) select_lambda ;; test) test_split ;; merge) merge ;;
  all) stamp start; repro; val; select_lambda; test_split; merge; stamp done ;;
  *) echo "usage: $0 all|repro|val|select|test|merge" >&2; exit 2 ;;
esac
