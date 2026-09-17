# -*- coding: utf-8 -*-
"""
Train and evaluate agents on the v4 environment.

Trains three configurations on the identical v4 environment and reports the
statistics that matter for the thesis: mission success, the rule-based detection
rate, mean reward, and precondition violations. The detection rate here is
meaningful -- it is the fraction of episodes a named blue-team rule escalated to
incident response, not an abstract number.

    python analysis/v4_train.py --mode train --config masked --seed 0
    python analysis/v4_train.py --mode eval

Configurations:
    masked    MaskablePPO, the system as intended
    nomask    PPO with the mask removed -- the fidelity control
    (DQN can be added later; the point here is masked vs unmasked under the new
     detection model.)
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import torch
torch.set_num_threads(2)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env.adversary_env import AdversaryEnv
from env.kill_chain import ACTION_ORDER

OUT = "results/models/v4"
STEPS = 600_000

# The catalogue the initial environment exposed. Passing --catalogue15 runs the
# v4 method over exactly these, so technique count can be varied with the
# environment, detection model, reward and protocol all held fixed.
V3_CATALOGUE = [
    "PHISHING_EMAIL", "BRUTE_FORCE_SSH", "SQL_INJECTION", "NETWORK_SCAN",
    "VALID_ACCOUNTS_LOGIN", "INSTALL_BACKDOOR", "WEB_SHELL_UPLOAD", "CLEAR_LOGS",
    "LATERAL_MOVE_SMB", "PRIV_ESC_SUDO", "POWERSHELL_EXEC", "PASS_THE_HASH",
    "KERBEROASTING", "EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT",
]

PPO_KW = dict(learning_rate=3e-4, gamma=0.995, n_steps=2048, batch_size=128,
              n_epochs=10, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
              vf_coef=0.5, max_grad_norm=0.5)
# DQN: a value-based algorithm from a different family than PPO, so the
# comparison shows whether the masking result is specific to PPO. SB3's DQN has
# no action-masking variant, so DQN here is necessarily unmasked.
DQN_KW = dict(learning_rate=1e-3, gamma=0.995, buffer_size=100_000,
              learning_starts=5_000, batch_size=128, train_freq=4,
              target_update_interval=1_000, exploration_fraction=0.2,
              exploration_final_eps=0.05)

# A2C: the synchronous form of the A3C algorithm DeepExploit uses (Isozaki,
# 2018). Including it puts a third algorithm family in the comparison and, more
# to the point, tests the masking result against the specific algorithm the
# closest prior system was built on. Like DQN it has no masking variant in SB3,
# so it is necessarily unmasked.
A2C_KW = dict(learning_rate=7e-4, gamma=0.995, n_steps=5, gae_lambda=1.0,
              ent_coef=0.02, vf_coef=0.5, max_grad_norm=0.5)

CONFIGS = {
    "masked": "maskable",
    "nomask": "ppo",
    "dqn": "dqn",
    "a2c": "a2c",
}
SEEDS = [0, 1, 2]


class _Null:
    def start_episode(self): pass
    def log_event(self, *a, **k): pass
    def end_episode(self): pass


def make_env(seed=None, real_mode=False, illegal_penalty=0.0, max_steps=40,
             catalogue=None):
    env = AdversaryEnv(config={"max_steps": max_steps, "real_mode": real_mode,
                               "illegal_penalty": illegal_penalty,
                               "restrict_techniques": catalogue})
    env.telemetry = _Null()
    if seed is not None:
        env.reset(seed=seed)
    return env


def build(algo, env, seed):
    if algo == "maskable":
        from sb3_contrib import MaskablePPO
        return MaskablePPO("MlpPolicy", env, device="cpu", verbose=0, seed=seed,
                           policy_kwargs=dict(net_arch=[64, 64]), **PPO_KW)
    if algo == "dqn":
        from stable_baselines3 import DQN
        return DQN("MlpPolicy", env, device="cpu", verbose=0, seed=seed,
                   policy_kwargs=dict(net_arch=[64, 64]), **DQN_KW)
    if algo == "a2c":
        from stable_baselines3 import A2C
        return A2C("MlpPolicy", env, device="cpu", verbose=0, seed=seed,
                   policy_kwargs=dict(net_arch=[64, 64]), **A2C_KW)
    from stable_baselines3 import PPO
    return PPO("MlpPolicy", env, device="cpu", verbose=0, seed=seed,
               policy_kwargs=dict(net_arch=[64, 64]), **PPO_KW)


def load(algo, path):
    if algo == "maskable":
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(path, device="cpu")
    if algo == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(path, device="cpu")
    if algo == "a2c":
        from stable_baselines3 import A2C
        return A2C.load(path, device="cpu")
    from stable_baselines3 import PPO
    return PPO.load(path, device="cpu")


def train(name, seed, real_mode=False, steps=STEPS, illegal_penalty=0.0, suffix="",
          max_steps=40, catalogue=None):
    algo = CONFIGS[name]
    os.makedirs(OUT, exist_ok=True)
    env = make_env(seed, real_mode=real_mode, illegal_penalty=illegal_penalty,
                   max_steps=max_steps, catalogue=catalogue)
    model = build(algo, env, seed)
    t0 = time.time()
    model.learn(total_timesteps=steps)
    tag = f"{name}_s{seed}{suffix}" + ("_live" if real_mode else "")
    model.save(os.path.join(OUT, tag))
    print(f"{tag}: {model.num_timesteps:,} steps in {(time.time()-t0)/60:.1f} min "
          f"({'LIVE server' if real_mode else 'sim'}"
          f"{f', illegal_penalty={illegal_penalty}' if illegal_penalty else ''})")


def evaluate(name, seed, episodes=300):
    algo = CONFIGS[name]
    path = os.path.join(OUT, f"{name}_s{seed}")
    if not os.path.exists(path + ".zip"):
        return None
    model = load(algo, path)
    env = make_env()
    succ = det = illegal = steps = 0
    rewards = []
    fired = {}
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 97 + ep + 500_000)
        done, total = False, 0.0
        while not done:
            legal = env.action_masks()
            if algo == "maskable":
                a, _ = model.predict(obs, action_masks=legal, deterministic=True)
            else:
                a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            steps += 1
            if not legal[a]:
                illegal += 1
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
        rewards.append(total)
        if env.model.is_goal():
            succ += 1
        if env.model.state.any_host_detected():
            det += 1
        for rule in env.model.detection.fired_rules:
            fired[rule] = fired.get(rule, 0) + 1
    return {
        "success_rate": 100.0 * succ / episodes,
        "detection_rate": 100.0 * det / episodes,
        "avg_reward": float(np.mean(rewards)),
        "illegal_pct": 100.0 * illegal / steps,
        "top_rules": dict(sorted(fired.items(), key=lambda kv: -kv[1])[:5]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--live", action="store_true",
                    help="train directly against the running v4 server over HTTP")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--illegal-penalty", type=float, default=0.0,
                    help="v3-style reward penalty for illegal actions (teaches the unmasked baseline)")
    ap.add_argument("--suffix", type=str, default="",
                    help="appended to the saved model tag, so experiments don't overwrite the mains")
    ap.add_argument("--max-steps", type=int, default=40,
                    help="episode step budget during training (raise it to let an unmasked agent finish)")
    ap.add_argument("--catalogue15", action="store_true",
                    help="restrict the action set to the initial environment's 15 techniques")
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated seeds to aggregate in eval mode "
                         "(default: the three the paper reports)")
    ap.add_argument("--out", type=str, default="analysis/v4_results.json",
                    help="where eval writes its summary")
    a = ap.parse_args()
    catalogue = V3_CATALOGUE if a.catalogue15 else None
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else SEEDS

    if a.mode == "train":
        train(a.config, a.seed, real_mode=a.live, steps=a.steps,
              illegal_penalty=a.illegal_penalty, suffix=a.suffix,
              max_steps=a.max_steps, catalogue=catalogue)
        return

    out = {}
    for name in CONFIGS:
        pairs = [(s, r) for s in seeds if (r := evaluate(name, s, a.episodes))]
        for s, r in pairs:
            r["seed"] = s
        if pairs:
            out[name] = [r for _, r in pairs]
    summary = {}
    print("\n" + "=" * 78)
    print("v4 RESULTS")
    print("=" * 78)
    print(f"{'config':10s}{'seeds':>6}{'success %':>12}{'detection %':>13}"
          f"{'reward':>11}{'illegal %':>11}")
    for name, runs in out.items():
        succ = [r["success_rate"] for r in runs]
        det = [r["detection_rate"] for r in runs]
        rew = [r["avg_reward"] for r in runs]
        ill = [r["illegal_pct"] for r in runs]
        summary[name] = {
            "n": len(runs),
            "success_mean": float(np.mean(succ)), "success_sd": float(np.std(succ)),
            "detection_mean": float(np.mean(det)), "detection_sd": float(np.std(det)),
            "reward_mean": float(np.mean(rew)),
            "illegal_mean": float(np.mean(ill)), "illegal_sd": float(np.std(ill)),
            "per_seed": runs,
        }
        s = summary[name]
        print(f"{name:10s}{s['n']:>6}"
              f"{s['success_mean']:>10.1f}±{s['success_sd']:<3.1f}"
              f"{s['detection_mean']:>10.1f}±{s['detection_sd']:<3.1f}"
              f"{s['reward_mean']:>11.1f}"
              f"{s['illegal_mean']:>9.1f}±{s['illegal_sd']:<3.1f}")

    with open(a.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwritten to {a.out}")
    # show the detection rules that fired, for the write-up
    for name, runs in out.items():
        agg = {}
        for r in runs:
            for k, v in r["top_rules"].items():
                agg[k] = agg.get(k, 0) + v
        print(f"\n{name}: top detection rules that fired")
        for rule, n in sorted(agg.items(), key=lambda kv: -kv[1])[:5]:
            print(f"   {n:5d}  {rule}")


if __name__ == "__main__":
    main()
