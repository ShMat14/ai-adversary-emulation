# -*- coding: utf-8 -*-
"""
The full version of the comparison the supervisor asked for.

Two weaknesses in the first pass are addressed here, both of which a referee
would raise.

1.  DQN was given library defaults and scored zero. "Your baseline failed
    because you did not tune it" is a fair objection and would undermine the
    comparison against the DQN-based systems in the literature. Two further
    configurations are therefore trained: one with hyperparameters chosen for a
    long-horizon sparse-reward task, and one with three times the budget. If
    DQN still fails under both, the claim stands on evidence rather than on
    neglect.

2.  The collapse of unmasked runs at larger catalogues was observed in one seed
    of three. That is an observation, not a rate. Ten seeds per configuration
    are run so the frequency can be stated with an interval rather than as a
    fraction of three.

    python analysis/full_experiment.py --mode train --config dqn_tuned --seed 0
    python analysis/full_experiment.py --mode eval
"""
import argparse, os, sys, json, time, statistics, math

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import torch
torch.set_num_threads(2)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from analysis.scaled_env import ScaledAdversaryEnv, REAL_N

OUT = "results/models/full"
BASE_STEPS = 800_000

# PPO settings, taken from the released v3 checkpoint
PPO_KW = dict(learning_rate=2e-4, gamma=0.995, n_steps=4096, batch_size=128,
              n_epochs=15, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
              vf_coef=0.6, max_grad_norm=0.5)

# DQN as first run: library defaults but for the matched lr, gamma and network
DQN_DEFAULT = dict(learning_rate=2e-4, gamma=0.995)

# DQN configured for this task rather than left at defaults. The reward arrives
# only after a correctly ordered sequence, so exploration is extended and the
# replay buffer is large enough to retain successful chains once found.
DQN_TUNED = dict(learning_rate=1e-3, gamma=0.995, buffer_size=200_000,
                 learning_starts=10_000, batch_size=128, train_freq=4,
                 gradient_steps=1, target_update_interval=1_000,
                 exploration_fraction=0.30, exploration_initial_eps=1.0,
                 exploration_final_eps=0.05)

CONFIGS = {
    # name             algo        actions  kwargs        timesteps
    "dqn_default":   ("dqn",       REAL_N,  DQN_DEFAULT,  BASE_STEPS),
    "dqn_tuned":     ("dqn",       REAL_N,  DQN_TUNED,    BASE_STEPS),
    "dqn_tuned_long": ("dqn",      REAL_N,  DQN_TUNED,    BASE_STEPS * 3),
    "masked_200":    ("maskable",  200,     PPO_KW,       BASE_STEPS),
    "nomask_200":    ("ppo",       200,     PPO_KW,       BASE_STEPS),
}
SEEDS = list(range(10))


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


def build(algo, env, seed, kw):
    if algo == "maskable":
        from sb3_contrib import MaskablePPO
        return MaskablePPO("MlpPolicy", env, device="cpu", verbose=0, seed=seed, **kw)
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO("MlpPolicy", env, device="cpu", verbose=0, seed=seed, **kw)
    from stable_baselines3 import DQN
    return DQN("MlpPolicy", env, device="cpu", verbose=0, seed=seed,
               policy_kwargs=dict(net_arch=[64, 64]), **kw)


def train(name, seed):
    algo, actions, kw, steps = CONFIGS[name]
    os.makedirs(OUT, exist_ok=True)
    env = make_env(actions, seed)
    model = build(algo, env, seed, kw)
    t0 = time.time()
    model.learn(total_timesteps=steps)
    tag = f"{name}_s{seed}"
    model.save(os.path.join(OUT, tag))
    print(f"{tag}: {model.num_timesteps:,} steps in {(time.time()-t0)/60:.1f} min")


def load(algo, path):
    if algo == "maskable":
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(path, device="cpu")
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(path, device="cpu")
    from stable_baselines3 import DQN
    return DQN.load(path, device="cpu")


def evaluate(name, seed, episodes=200):
    algo, actions, _, _ = CONFIGS[name]
    path = os.path.join(OUT, f"{name}_s{seed}")
    if not os.path.exists(path + ".zip"):
        return None
    model = load(algo, path)
    env = make_env(actions)
    succ = det = illegal = steps_taken = 0
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 100000 + ep)
        done, total = False, 0.0
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


def wilson(k, n, z=1.96):
    """Wilson score interval, which behaves at k=0 where the normal one does not."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--episodes", type=int, default=200)
    a = ap.parse_args()

    if a.mode == "train":
        train(a.config, a.seed)
        return

    out = {}
    for name in CONFIGS:
        runs = []
        for s in SEEDS:
            r = evaluate(name, s, a.episodes)
            if r:
                r["seed"] = s
                runs.append(r)
        if runs:
            out[name] = runs
            print(f"\n{name}  ({len(runs)} seeds)")
            for r in runs:
                print(f"   seed {r['seed']}: success {r['success_rate']:5.1f}%  "
                      f"reward {r['avg_reward']:7.1f}  illegal {r['illegal_pct']:5.1f}%")

    print("\n" + "=" * 84)
    print("FULL COMPARISON")
    print("=" * 84)
    print(f"{'configuration':18s}{'seeds':>7}{'success %':>14}{'reward':>16}{'illegal %':>14}{'collapsed':>12}")
    print("-" * 84)
    summary = {}
    for name, runs in out.items():
        succ = [r["success_rate"] for r in runs]
        rew = [r["avg_reward"] for r in runs]
        ill = [r["illegal_pct"] for r in runs]
        collapsed = sum(1 for v in ill if v > 40)
        lo, hi = wilson(collapsed, len(runs))
        summary[name] = {
            "n": len(runs),
            "success_median": statistics.median(succ),
            "reward_median": statistics.median(rew),
            "reward_min": min(rew), "reward_max": max(rew),
            "illegal_median": statistics.median(ill),
            "illegal_min": min(ill), "illegal_max": max(ill),
            "collapsed": collapsed,
            "collapse_rate_pct": 100.0 * collapsed / len(runs),
            "collapse_ci95": [round(lo, 1), round(hi, 1)],
            "per_seed": runs,
        }
        s = summary[name]
        print(f"{name:18s}{s['n']:>7}"
              f"{s['success_median']:>13.1f}%"
              f"{s['reward_median']:>10.1f} [{s['reward_min']:.0f}-{s['reward_max']:.0f}]"
              f"{s['illegal_median']:>10.1f}%"
              f"{collapsed:>7}/{s['n']}")

    with open("analysis/full_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nwritten to analysis/full_results.json")


if __name__ == "__main__":
    main()
