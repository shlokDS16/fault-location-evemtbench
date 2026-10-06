#!/bin/bash
# Design section 20 (3): re-run every token-based evaluation after the I0 guard. Sequential (HGB uses all cores).
cd "$(dirname "$0")/../.."
set -e
for step in "ha2_eval.py" "adapt_ingrid.py" "ablations.py" "zs_eval.py MV DL TG"; do
  echo "=== $step $(date +%T)"
  python src/c1/$step > "results/rerun_guard_${step%%.py*}.log" 2>&1
  tail -15 "results/rerun_guard_${step%%.py*}.log"
done
echo "=== ALL DONE $(date +%T)"
