#!/usr/bin/env bash
# Phase 6: reward shaping at a penalty magnitude prior work actually uses.
#
# The main study penalises an infeasible action at -10, inherited from v4. That
# is 10-30x our per-step cost, and it produced a pathological policy: feasibility
# learned almost perfectly, but the agent rushes high-value compromises and is
# caught by step 11 in 49 of 50 episodes.
#
# L-ARLPT uses a step penalty of -1 and a failure penalty of -3, i.e. roughly 3x
# the step cost. Reporting only the -10 result would be a strawman of reward
# shaping, so this runs the mechanism at a comparable magnitude before we draw
# any conclusion about it.
set -u
cd "$(dirname "$0")/.."
LOCK="analysis/.v5_run.lock"
if [ -e "$LOCK" ]; then
  oldpid=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$oldpid" ] && tasklist //FI "PID eq $oldpid" 2>/dev/null | grep -qi bash; then
    echo "main run still active (pid $oldpid); exiting"; exit 0
  fi
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

# 6a. Retrain every DQN run under the exploration schedule aligned with
# E-NASim's published settings. The phase-1 DQN runs used a 20% exploration
# fraction, below anything the comparator reports (they use 33-50%), so those
# checkpoints are discarded rather than reported.
echo "### 6a: DQN and A2C retrained with fairly-configured hyperparameters"
echo "###     DQN  exploration 0.20 -> 0.33 (E-NASim uses 0.33-0.50)"
echo "###     A2C  n_steps 5 -> 64 (our PPO control uses 2048)"
rm -f results/models/v5/dqn_enterprise_s*.zip results/models/v5/a2c_enterprise_s*.zip
rm -f analysis/v5_curves/dqn_enterprise_s*.json analysis/v5_curves/a2c_enterprise_s*.json
for seed in 0 1 2 3 4; do
  for cfg in dqn a2c; do
    tag="${cfg}_enterprise_s${seed}"
    [ -f "results/models/v5/${tag}.zip" ] && { echo "skip ${tag}"; continue; }
    echo "=== ${tag} (re-configured) ==="
    python analysis/v5_train.py --config "$cfg" --topology enterprise         --seed "$seed" --steps 400000 --max-steps 60
  done
done

echo "### 6b: reward shaping at penalties prior work actually uses"
for pen in -1 -3; do
  tagpen=${pen#-}
  for seed in 0 1 2; do
    tag="nomask_enterprise_s${seed}_shaped${tagpen}"
    [ -f "results/models/v5/${tag}.zip" ] && { echo "skip ${tag}"; continue; }
    echo "=== ${tag} (penalty ${pen}) ==="
    python analysis/v5_train.py --config nomask --topology enterprise \
        --seed "$seed" --steps 400000 --max-steps 60 \
        --illegal-penalty "$pen" --suffix "_shaped${tagpen}"
  done
done
echo "PHASE 6 COMPLETE"
