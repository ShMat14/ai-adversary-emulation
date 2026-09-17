# -*- coding: utf-8 -*-
"""Re-evaluate the Table 7 budget/penalty control rows at 400 episodes, so the
whole masking matrix uses one eval protocol (the main rows are already at 400).

Loads the existing checkpoints (nomask/dqn *_big4m = 150-step 4M training;
nomask *_fair = -10 illegal penalty) and evaluates each at the stated eval
budget. Writes analysis/v4_table7_400.json. Does not retrain anything.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from analysis.v4_train import make_env, load

OUT = "results/models/v4"
SEEDS = [0, 1, 2]
EPISODES = 400

# (label, algo, checkpoint-suffix, eval max_steps)
CONFIGS = [
    ("PPO_penalty_40",   "ppo", "_fair",  40),
    ("PPO_150train_150", "ppo", "_big4m", 150),
    ("PPO_150train_40",  "ppo", "_big4m", 40),
    ("DQN_150train_150", "dqn", "_big4m", 150),
    ("DQN_150train_40",  "dqn", "_big4m", 40),
]


def evaluate(algo, name, suffix, max_steps, episodes=EPISODES):
    results = []
    for seed in SEEDS:
        path = os.path.join(OUT, f"{name}_s{seed}{suffix}")
        if not os.path.exists(path + ".zip"):
            print(f"  missing {path}.zip"); return None
        model = load(algo, path)
        env = make_env(max_steps=max_steps)
        succ = illegal = steps = 0
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed * 97 + ep + 500_000)
            done = False
            while not done:
                legal = env.action_masks()
                a, _ = model.predict(obs, deterministic=True)  # unmasked agents
                a = int(a)
                steps += 1
                if not legal[a]:
                    illegal += 1
                obs, r, term, trunc, _ = env.step(a)
                done = term or trunc
            if env.model.is_goal():
                succ += 1
        results.append((100.0 * succ / episodes, 100.0 * illegal / steps))
    return results


summary = {}
print(f"Re-evaluating Table 7 control rows at {EPISODES} episodes, 3 seeds\n")
print(f"{'row':20s}{'success %':>14}{'illegal %':>12}")
for label, algo, suffix, ms in CONFIGS:
    name = "nomask" if algo == "ppo" else "dqn"
    runs = evaluate(algo, name, suffix, ms)
    if not runs:
        continue
    succ = [s for s, _ in runs]; ill = [i for _, i in runs]
    summary[label] = {
        "n": len(runs), "eval_episodes": EPISODES, "eval_max_steps": ms,
        "success_mean": float(np.mean(succ)), "success_sd": float(np.std(succ)),
        "illegal_mean": float(np.mean(ill)), "illegal_sd": float(np.std(ill)),
        "per_seed_success": succ, "per_seed_illegal": ill,
    }
    s = summary[label]
    print(f"{label:20s}{s['success_mean']:>9.1f}±{s['success_sd']:<3.1f}"
          f"{s['illegal_mean']:>9.1f}±{s['illegal_sd']:<3.1f}")

with open("analysis/v4_table7_400.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nwritten to analysis/v4_table7_400.json")
