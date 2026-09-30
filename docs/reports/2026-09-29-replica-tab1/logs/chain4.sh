#!/usr/bin/env bash
# views (val, with the defined-pixel analysis) -> Table III val -> views test -> Table III test
set -uo pipefail
S=/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad
cd $S/wt_eval
echo "chain4 start $(date '+%F %T') worktree $(git rev-parse --short HEAD)"
for f in $(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -c "import sys; sys.path.insert(0,'scripts'); import run_mapping as r; print(' '.join(r.MAPPING_CODE))"); do
  cmp -s "$f" "$S/wt_val/$f" || { echo "mapping code differs: $f"; exit 1; }
done
STAGE=val scripts/replica_views.sh || echo "views val failed"
STAGE=val scripts/replica_tab3.sh || echo "tab3 val failed"
STAGE=test scripts/replica_views.sh || echo "views test failed"
STAGE=test scripts/replica_tab3.sh || echo "tab3 test failed"
echo "chain4 done $(date '+%F %T')"
