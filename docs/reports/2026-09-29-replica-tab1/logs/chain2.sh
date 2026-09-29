#!/usr/bin/env bash
# after the Table I chain: views figure and Table III, validation then test (same clean worktree)
set -uo pipefail
S=/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad
O=/home/thanhnc/gauss_ws/data/uags_results/tab1
echo "chain2 start $(date '+%F %T')"
until grep -q "^chain exit" <(sed -n '/relaunch.*lambda_u grid/,$p' $O/chain.log); do sleep 60; done
echo "Table I chain finished: $(sed -n '/relaunch.*lambda_u grid/,$p' $O/chain.log | grep '^chain exit')"
cd $S/wt_eval
echo "worktree $(git rev-parse --short HEAD)"
STAGE=val scripts/replica_views.sh || echo "views val failed"
STAGE=val scripts/replica_tab3.sh || echo "tab3 val failed"
STAGE=test scripts/replica_views.sh || echo "views test failed"
STAGE=test scripts/replica_tab3.sh || echo "tab3 test failed"
echo "chain2 done $(date '+%F %T')"
