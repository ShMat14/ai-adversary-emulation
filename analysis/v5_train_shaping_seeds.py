# -*- coding: utf-8 -*-
"""Train the reward-shaping (Chaudhary-style) baseline out to 10 seeds.

The paper's headline masked-vs-unmasked comparison runs at 3, 5 and 10 seeds,
but the reward-shaping control only ever had 3. To promote that control into a
first-class comparison against the prior state of the art, it has to be
evidenced at the same strength. This trains seeds 3-9 with the same settings
the original three used: unmasked PPO, -10 reward for an illegal action,
40-step budget, 600k timesteps. Skips any checkpoint already on disk, so it is
safe to stop and restart.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.v4_train import train, OUT, STEPS

PENALTY = -10.0
SUFFIX = "_fair"
SEEDS = [3, 4, 5, 6, 7, 8, 9]

if __name__ == "__main__":
    t0 = time.time()
    for s in SEEDS:
        tag = f"nomask_s{s}{SUFFIX}"
        if os.path.exists(os.path.join(OUT, tag + ".zip")):
            print(f"{tag}: already on disk, skipping", flush=True)
            continue
        print(f"--- training {tag} (penalty={PENALTY}, {STEPS:,} steps) ---", flush=True)
        train("nomask", s, steps=STEPS, illegal_penalty=PENALTY,
              suffix=SUFFIX, max_steps=40)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)
