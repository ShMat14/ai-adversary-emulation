"""
baselines/scripted_attacks.py

Runs a fixed scripted kill-chain as a deterministic baseline and then
runs the PPO model, printing a side-by-side comparison table.

This is the core comparison experiment required for the thesis
(Chapter 4: PPO vs Scripted Baseline).
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from env.adversary_env import AdversaryEnv
from env.attack_actions import ACTION_LIST


def get_action_index(name: str) -> int:
    for i, a in enumerate(ACTION_LIST):
        if a.name == name:
            return i
    raise ValueError(f"Unknown action name: '{name}'. "
                     f"Valid names: {[a.name for a in ACTION_LIST]}")


def run_baseline(chain_name: str, chain_actions: list, episodes: int = 200,
                 telemetry_dir: str = "results/telemetry/baseline"):
    """
    Runs a scripted kill-chain for `episodes` episodes and returns stats.
    Actions that go past the chain length are simply skipped (episode ends).
    """
    env = AdversaryEnv(config={
        "max_steps": 20,
        "telemetry_dir": telemetry_dir,
    })

    action_indices = [get_action_index(n) for n in chain_actions]

    successes, detections, steps_list, rewards_list = 0, 0, [], []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step_idx = 0

        while not done and step_idx < len(action_indices):
            action_idx = action_indices[step_idx]
            obs, reward, terminated, truncated, _ = env.step(action_idx)
            total_reward += reward
            step_idx += 1
            done = terminated or truncated

        # Success check
        if isinstance(env.state.hosts, dict):
            dc_won = env.state.hosts.get("dc01") and env.state.hosts["dc01"].privileged
        else:
            dc_won = any(h.name == "dc01" and h.privileged for h in env.state.hosts)

        if dc_won:
            successes += 1
            steps_list.append(env.current_step)

        if env.state.any_host_detected():
            detections += 1

        rewards_list.append(total_reward)

    return {
        "name": chain_name,
        "episodes": episodes,
        "success_rate": successes / episodes * 100,
        "detection_rate": detections / episodes * 100,
        "avg_steps": np.mean(steps_list) if steps_list else float("nan"),
        "avg_reward": np.mean(rewards_list),
    }


def print_comparison(results: list):
    """Print a formatted comparison table for the thesis."""
    header = f"{'Agent':<28} {'Success%':>9} {'Detect%':>8} {'AvgSteps':>9} {'AvgReward':>10}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("THESIS TABLE 1 — PPO vs Scripted Baselines")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(f"{r['name']:<28} {r['success_rate']:>8.1f}% {r['detection_rate']:>7.1f}% "
              f"{r['avg_steps']:>9.2f} {r['avg_reward']:>10.2f}")
    print(sep + "\n")


def main():
    EPISODES = 200

    # -----------------------------------------------------------------
    # Chain 1 — Standard Kill Chain (Sequential, no evasion)
    # -----------------------------------------------------------------
    chain_standard = [
        "PHISHING_EMAIL",
        "NETWORK_SCAN",
        "VALID_ACCOUNTS_LOGIN",
        "LATERAL_MOVE_SMB",
        "PRIV_ESC_SUDO",
        "EXFILTRATE_DATA",
    ]

    # -----------------------------------------------------------------
    # Chain 2 — Stealthy Chain (uses backdoor + log clearing)
    # -----------------------------------------------------------------
    chain_stealthy = [
        "PHISHING_EMAIL",
        "INSTALL_BACKDOOR",
        "CLEAR_LOGS",
        "NETWORK_SCAN",
        "VALID_ACCOUNTS_LOGIN",
        "LATERAL_MOVE_SMB",
        "PRIV_ESC_SUDO",
        "EXFILTRATE_DATA",
    ]

    # -----------------------------------------------------------------
    # Chain 3 — Aggressive Chain (brute force, fast but loud)
    # -----------------------------------------------------------------
    chain_aggressive = [
        "BRUTE_FORCE_SSH",
        "NETWORK_SCAN",
        "VALID_ACCOUNTS_LOGIN",
        "LATERAL_MOVE_SMB",
        "PRIV_ESC_SUDO",
        "RANSOMWARE_ENCRYPT",
    ]

    # -----------------------------------------------------------------
    # Chain 4 — SQL Injection Chain (OWASP-style web exploit entry)
    # SQLi replaces phishing: compromises host + dumps DB creds in step 1
    # -----------------------------------------------------------------
    chain_sqli = [
        "SQL_INJECTION",       # Initial Access + Credential Dump (T1190)
        "NETWORK_SCAN",        # Discovery
        "LATERAL_MOVE_SMB",    # Lateral Movement (creds already stolen)
        "PRIV_ESC_SUDO",       # Privilege Escalation
        "EXFILTRATE_DATA",     # Impact
    ]

    results = []
    print(f"\n>>> Running scripted baselines ({EPISODES} episodes each)...")

    results.append(run_baseline("Scripted-Standard",   chain_standard,   EPISODES,
                                telemetry_dir="results/telemetry/baseline_standard"))
    results.append(run_baseline("Scripted-Stealthy",   chain_stealthy,   EPISODES,
                                telemetry_dir="results/telemetry/baseline_stealthy"))
    results.append(run_baseline("Scripted-Aggressive",  chain_aggressive,  EPISODES,
                                telemetry_dir="results/telemetry/baseline_aggressive"))
    results.append(run_baseline("Scripted-SQLi",        chain_sqli,        EPISODES,
                                telemetry_dir="results/telemetry/baseline_sqli"))

    # -----------------------------------------------------------------
    # Compare with trained PPO v2 model (if it exists)
    # -----------------------------------------------------------------
    model_path = "results/models/ppo_adversary_v3"
    if os.path.exists(model_path + ".zip"):
        print("\n>>> PPO v2 model found — running evaluation...")
        from agents.ppo_eval import evaluate_agent
        ppo_stats = evaluate_agent(model_path, num_episodes=EPISODES, verbose=False)
        results.append(ppo_stats)
    else:
        print(f"\n[!] PPO v2 model not found at {model_path}.zip — skipping PPO comparison.")
        print("    Run agents/ppo_train.py first, then re-run this script.")

    print_comparison(results)


if __name__ == "__main__":
    main()
