#!/usr/bin/env bash
# waits for the label-free preparation, then STAGE=val (distill + eval) and STAGE=test (once)
set -uo pipefail
S=/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad
O=/home/thanhnc/gauss_ws/data/uags_results/tab1
echo "chain start $(date '+%F %T')"
until grep -q "^prep exit" $O/prep.log; do sleep 30; done
grep -q "^prep exit 0" $O/prep.log || { echo "prep failed: $(grep '^prep exit' $O/prep.log)"; exit 1; }
cd $S/wt_eval
for f in $(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -c "import sys; sys.path.insert(0,'scripts'); import run_mapping as r; print(' '.join(r.MAPPING_CODE))"); do
  cmp -s "$f" "$S/wt_val/$f" || { echo "mapping code differs: $f"; exit 1; }
done
echo "mapping code identical to the preparation; eval worktree $(git rev-parse --short HEAD)"
STAGE=val scripts/replica_tab1.sh || { echo "val stage failed"; exit 1; }
STAGE=test scripts/replica_tab1.sh || { echo "test stage failed"; exit 1; }
echo "chain done $(date '+%F %T')"
