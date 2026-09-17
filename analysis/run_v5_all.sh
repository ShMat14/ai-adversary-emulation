#!/usr/bin/env bash
# The full v5 study, strictly sequential, ordered by scientific value so that an
# interrupted run still leaves the important results complete.
#
# Every run skips when its checkpoint exists, so this is stop/resume safe.
set -u
cd "$(dirname "$0")/.."

# Single-instance lock. An earlier version was launched twice and both copies
# wrote to the same log, interleaving two training streams into one unreadable
# file and halving the CPU available to each.
LOCK="analysis/.v5_run.lock"
if [ -e "$LOCK" ]; then
  oldpid=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$oldpid" ] && tasklist //FI "PID eq $oldpid" 2>/dev/null | grep -qi bash; then
    echo "another run is already active (pid $oldpid); exiting"; exit 0
  fi
  echo "stale lock from pid $oldpid; taking over"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

run () {  # cfg seed budget suffix penalty topology
  local cfg=$1 seed=$2 budget=$3 suffix=$4 pen=$5 topo=$6
  local tag="${cfg}_${topo}_s${seed}${suffix}"
  if [ -f "results/models/v5/${tag}.zip" ]; then echo "skip ${tag}"; return; fi
  echo "=== ${tag} (budget ${budget}, penalty ${pen}) ==="
  python analysis/v5_train.py --config "$cfg" --topology "$topo" \
      --seed "$seed" --steps 400000 --max-steps "$budget" \
      --suffix "$suffix" --illegal-penalty "$pen"
}

echo "### PHASE 1  main comparison, 12-host estate, 60-step budget, 5 seeds"
echo "###          masked | unmasked | REWARD-SHAPED (prior work's mechanism) | DQN | A2C"
for seed in 0 1 2 3 4; do
  run masked  "$seed" 60 ""       0    enterprise
  run nomask  "$seed" 60 ""       0    enterprise
  run nomask  "$seed" 60 "_shaped" -10 enterprise   # Chaudhary-style penalty
  run dqn     "$seed" 60 ""       0    enterprise
  run a2c     "$seed" 60 ""       0    enterprise
done

echo "### PHASE 2  fairness control at L-ARLPT's 500-step budget."
echo "###          If an unmasked agent succeeds here, masking buys efficiency"
echo "###          and not capability -- a weaker claim that must be reported."
for seed in 0 1 2; do
  for cfg in nomask dqn a2c; do run "$cfg" "$seed" 500 "_b500" 0 enterprise; done
  run nomask "$seed" 500 "_b500_shaped" -10 enterprise
done

echo "### PHASE 3  formulation vs. size. Same v5 rules on the 3-host network, so"
echo "###          any gain over v4 can be attributed to the action-space change"
echo "###          rather than simply to a larger estate."
for seed in 0 1 2; do
  run masked "$seed" 60 "" 0 v4compat
  run nomask "$seed" 60 "" 0 v4compat
done

echo "### PHASE 4  extreme control at E-NASim's 2000-step budget (slow: episodes"
echo "###          are 33x longer, so only the headline baseline, two seeds)"
for seed in 0 1; do run nomask "$seed" 2000 "_b2000" 0 enterprise; done

echo "### PHASE 5  extend the main comparison to ten seeds"
for seed in 5 6 7 8 9; do
  run masked  "$seed" 60 ""       0    enterprise
  run nomask  "$seed" 60 ""       0    enterprise
  run nomask  "$seed" 60 "_shaped" -10 enterprise
  run dqn     "$seed" 60 ""       0    enterprise
  run a2c     "$seed" 60 ""       0    enterprise
done

echo "ALL PHASES COMPLETE"
