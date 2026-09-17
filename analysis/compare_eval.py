# -*- coding: utf-8 -*-
"""
Evaluate every checkpoint from the controlled comparison under one harness.

All agents face the same environment, the same episode budget and the same
evaluation seeds. The only difference is the algorithm and whether the action
mask is consulted:

    maskable  asks the environment for the legal action set before each choice
    ppo, dqn  choose from all 15 actions; the environment applies its ordinary
              penalties when a precondition is unmet

Success is defined exactly as in agents/ppo_eval.py -- dc01 reaching privileged
state -- so the numbers are comparable with those already reported in Table 3.

    python analysis/compare_eval.py --episodes 200
"""
import argparse, os, sys, json, statistics
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.adversary_env import AdversaryEnv
from env.attack_actions import ACTION_LIST

MODELS = "results/models/comparison"
ALGOS = ["maskable", "ppo", "dqn"]


class _NullTelemetry:
    def start_episode(self):
        pass

    def log_event(self, *a, **k):
        pass

    def end_episode(self):
        pass


def load(algo, path):
    if algo == "maskable":
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(path, device="cpu")
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(path, device="cpu")
    from stable_baselines3 import DQN
    return DQN.load(path, device="cpu")


def evaluate(algo, path, episodes, seed):
    model = load(algo, path)
    env = AdversaryEnv(config={"max_steps": 40, "real_mode": False})
    env.telemetry = _NullTelemetry()

    successes = detections = 0
    steps, rewards, uniques = [], [], []
    actions = Counter()

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 100000 + ep)   # identical episodes per algo
        done = False
        total = 0.0
        used = set()
        while not done:
            if algo == "maskable":
                a, _ = model.predict(obs, action_masks=env.action_masks(),
                                     deterministic=True)
            else:
                a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            actions[ACTION_LIST[a].name] += 1
            used.add(ACTION_LIST[a].name)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        uniques.append(len(used))
        rewards.append(total)
        hosts = env.state.hosts
        won = "dc01" in hosts and hosts["dc01"].privileged
        if won:
            successes += 1
            steps.append(env.current_step)
        if env.state.any_host_detected():
            detections += 1

    return {
        "success_rate": 100.0 * successes / episodes,
        "detection_rate": 100.0 * detections / episodes,
        "avg_steps": float(np.mean(steps)) if steps else float("nan"),
        "avg_reward": float(np.mean(rewards)),
        "avg_unique": float(np.mean(uniques)),
        "action_freq": dict(actions.most_common(10)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args()

    per_seed, summary = {}, {}
    for algo in ALGOS:
        runs = []
        for seed in (0, 1, 2):
            path = os.path.join(MODELS, f"{algo}_seed{seed}")
            if not os.path.exists(path + ".zip"):
                print(f"  missing: {path}.zip")
                continue
            r = evaluate(algo, path, args.episodes, seed)
            r["seed"] = seed
            runs.append(r)
            print(f"  {algo:9s} seed {seed}: success {r['success_rate']:5.1f}%  "
                  f"detect {r['detection_rate']:5.1f}%  reward {r['avg_reward']:8.2f}  "
                  f"tech {r['avg_unique']:.1f}")
        if not runs:
            continue
        per_seed[algo] = runs

        def ms(key):
            vals = [x[key] for x in runs if x[key] == x[key]]
            if not vals:
                return (float("nan"), float("nan"))
            return (statistics.mean(vals),
                    statistics.stdev(vals) if len(vals) > 1 else 0.0)

        summary[algo] = {k: ms(k) for k in
                         ("success_rate", "detection_rate", "avg_steps",
                          "avg_reward", "avg_unique")}

    print("\n" + "=" * 78)
    print(f"CONTROLLED COMPARISON — same environment, {args.episodes} episodes, "
          f"{len(per_seed.get('maskable', []))} seeds, mean +/- sd")
    print("=" * 78)
    hdr = f"{'agent':22s}{'success%':>12}{'detect%':>12}{'reward':>13}{'techniques':>13}"
    print(hdr)
    print("-" * 78)
    LABEL = {"maskable": "MaskablePPO (masked)",
             "ppo": "PPO (no mask)",
             "dqn": "DQN (no mask)"}
    for algo in ALGOS:
        if algo not in summary:
            continue
        s = summary[algo]
        print(f"{LABEL[algo]:22s}"
              f"{s['success_rate'][0]:8.1f}+/-{s['success_rate'][1]:<4.1f}"
              f"{s['detection_rate'][0]:8.1f}+/-{s['detection_rate'][1]:<4.1f}"
              f"{s['avg_reward'][0]:9.1f}+/-{s['avg_reward'][1]:<5.1f}"
              f"{s['avg_unique'][0]:8.1f}+/-{s['avg_unique'][1]:<4.1f}")

    out = {"episodes": args.episodes, "per_seed": per_seed, "summary": summary}
    with open("analysis/comparison_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwritten to analysis/comparison_results.json")


if __name__ == "__main__":
    main()
