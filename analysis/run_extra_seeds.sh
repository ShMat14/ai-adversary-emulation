#!/usr/bin/env bash
# Train the extra seeds needed to take the v4 headline comparison from 3 seeds
# to 5 (and later to 10), at exactly the protocol the paper reports:
# 2,000,000 timesteps, 40-step episodes, no illegal penalty, 25 techniques.
#
#   bash analysis/run_extra_seeds.sh 3 4        # -> 5 seeds total
#   bash analysis/run_extra_seeds.sh 5 6 7 8 9  # -> 10 seeds total
#
# Runs sequentially: this is a CPU-only box and parallel runs thrash the cache.
set -u
cd "$(dirname "$0")/.."
mkdir -p results/logs

STEPS=2000000
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && { echo "usage: $0 <seed> [seed...]"; exit 1; }

echo "=== extra-seed run started $(date) ==="
echo "seeds: ${SEEDS[*]}   steps: $STEPS"

for seed in "${SEEDS[@]}"; do
  for cfg in masked nomask dqn; do
    tag="${cfg}_s${seed}"
    if [ -f "results/models/v4/${tag}.zip" ]; then
      echo "[skip] ${tag} already exists"
      continue
    fi
    echo "[start] ${tag}  $(date +%H:%M:%S)"
    python analysis/v4_train.py --mode train --config "$cfg" --seed "$seed" \
        --steps "$STEPS" > "results/logs/extra_${tag}.log" 2>&1
    if [ $? -eq 0 ]; then
      echo "[done ] ${tag}  $(date +%H:%M:%S)  $(tail -1 results/logs/extra_${tag}.log)"
    else
      echo "[FAIL ] ${tag} -- see results/logs/extra_${tag}.log"
      tail -5 "results/logs/extra_${tag}.log"
    fi
  done
done

echo "=== training finished $(date) ==="
echo "next: python analysis/v4_train.py --mode eval --episodes 400 \\"
echo "        --seeds $(IFS=,; echo \"0,1,2,${SEEDS[*]}\" | tr ' ' ',') \\"
echo "        --out analysis/v4_results_5seed.json"
