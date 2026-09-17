# -*- coding: utf-8 -*-
"""
Train and evaluate masked and unmasked PPO across action-space sizes.

    python analysis/scale_experiment.py --mode train --actions 60 --algo ppo --seed 0
    python analysis/scale_experiment.py --mode eval
"""
import argparse, os, sys, json, time, statistics

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import torch
torch.set_num_threads(2)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from analysis.scaled_env import ScaledAdversaryEnv, REAL_N
from env.attack_actions import ACTION_LIST

SHARED = dict(learning_rate=2e-4, gamma=0.995)
PPO_ONLY = dict(n_steps=4096, batch_size=128, n_epochs=15, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.02, vf_coef=0.6, max_grad_norm=0.5)
TIMESTEPS = 800_000
OUT = "results/models/scale"
SIZES = [15, 60, 200]
SEEDS = [0, 1, 2]


class _NullTelemetry:
    def start_episode(self): pass
    def log_event(self, *a, **k): pass
    def end_episode(self): pass


def make_env(actions, seed=None):
    env = ScaledAdversaryEnv(total_actions=actions,
                             config={"max_steps": 40, "real_mode": False})
    env.telemetry = _NullTelemetry()
    if seed is not None:
        env.reset(seed=seed)
    return env


def build(algo, env, seed):
    if algo == "maskable":
        from sb3_contrib import MaskablePPO
        return MaskablePPO("MlpPolicy", env, device="cpu", verbose=0, seed=seed,
                           **SHARED, **PPO_ONLY)
    from stable_baselines3 import PPO
    return PPO("MlpPolicy", env, device="cpu", verbose=0, seed=seed,
               **SHARED, **PPO_ONLY)


def train(actions, algo, seed):
    os.makedirs(OUT, exist_ok=True)
    tag = f"{algo}_a{actions}_s{seed}"
    env = make_env(actions, seed)
    model = build(algo, env, seed)
    t0 = time.time()
    model.learn(total_timesteps=TIMESTEPS)
    model.save(os.path.join(OUT, tag))
    print(f"{tag}: {model.num_timesteps:,} steps in {(time.time()-t0)/60:.1f} min")


def load(algo, path):
    if algo == "maskable":
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(path, device="cpu")
    from stable_baselines3 import PPO
    return PPO.load(path, device="cpu")


def evaluate(algo, actions, seed, episodes=200):
    path = os.path.join(OUT, f"{algo}_a{actions}_s{seed}")
    if not os.path.exists(path + ".zip"):
        return None
    model = load(algo, path)
    env = make_env(actions)
    succ = det = illegal = steps_taken = 0
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 100000 + ep)
        done = False
        total = 0.0
        while not done:
            legal = env.action_masks()
            if algo == "maskable":
                a, _ = model.predict(obs, action_masks=legal, deterministic=True)
            else:
                a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            steps_taken += 1
            if not legal[a]:
                illegal += 1
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        rewards.append(total)
        h = env.state.hosts
        if "dc01" in h and h["dc01"].privileged:
            succ += 1
        if env.state.any_host_detected():
            det += 1
    return {"success_rate": 100.0 * succ / episodes,
            "detection_rate": 100.0 * det / episodes,
            "avg_reward": float(np.mean(rewards)),
            "illegal_pct": 100.0 * illegal / steps_taken}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--actions", type=int)
    ap.add_argument("--algo", choices=["maskable", "ppo"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--episodes", type=int, default=200)
    a = ap.parse_args()

    if a.mode == "train":
        train(a.actions, a.algo, a.seed)
        return

    results = {}
    for actions in SIZES:
        for algo in ("maskable", "ppo"):
            runs = [r for r in (evaluate(algo, actions, s, a.episodes) for s in SEEDS) if r]
            if runs:
                results[f"{algo}_{actions}"] = {
                    k: (statistics.mean(x[k] for x in runs),
                        statistics.stdev([x[k] for x in runs]) if len(runs) > 1 else 0.0)
                    for k in ("success_rate", "detection_rate", "avg_reward", "illegal_pct")}

    # how much of the action space is legal at a typical step
    fracs = {}
    for actions in SIZES:
        env = make_env(actions)
        env.reset(seed=0)
        vals = []
        for _ in range(60):
            vals.append(env.legal_fraction())
            m = env.action_masks()
            choice = int(np.flatnonzero(m)[0]) if m.any() else 0
            _, _, t, tr, _ = env.step(choice)
            if t or tr:
                env.reset()
        fracs[actions] = 100.0 * statistics.mean(vals)

    print("\n" + "=" * 82)
    print("ACTION-SPACE SCALING — does masking matter with a larger technique catalogue?")
    print("=" * 82)
    print(f"{'actions':>8}{'legal %':>10}{'agent':>14}{'success %':>14}{'reward':>12}{'illegal %':>12}")
    print("-" * 82)
    for actions in SIZES:
        for algo in ("maskable", "ppo"):
            k = f"{algo}_{actions}"
            if k not in results:
                continue
            r = results[k]
            label = "masked" if algo == "maskable" else "no mask"
            print(f"{actions:>8}{fracs[actions]:>9.1f}%{label:>14}"
                  f"{r['success_rate'][0]:>9.1f}±{r['success_rate'][1]:<4.1f}"
                  f"{r['avg_reward'][0]:>11.1f}"
                  f"{r['illegal_pct'][0]:>11.1f}%")
    with open("analysis/scale_results.json", "w") as f:
        json.dump({"results": results, "legal_fraction_pct": fracs}, f, indent=2)
    print("\nwritten to analysis/scale_results.json")


if __name__ == "__main__":
    main()
