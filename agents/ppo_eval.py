"""
agents/ppo_eval.py

Evaluates a trained MaskablePPO model in both simulation and real modes.
Produces a sim-vs-real comparison table for the thesis Chapter 4.
Can also be imported by baselines/scripted_attacks.py.
"""

import sys
import os
import numpy as np
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO
from env.adversary_env import AdversaryEnv
from env.attack_actions import ACTION_LIST


def evaluate_agent(model_path: str, num_episodes: int = 200,
                   verbose: bool = True, real_mode: bool = False) -> dict:
    """
    Evaluate a trained MaskablePPO model.

    Args:
        model_path:    Path to the saved model (without .zip)
        num_episodes:  Number of evaluation episodes
        verbose:       Print detailed results
        real_mode:     If True, hit the Flask mock server (realistic).
                       If False, use probabilistic simulation (fast).

    Returns a stats dict compatible with the comparison table:
        {name, episodes, success_rate, detection_rate, avg_steps, avg_reward}
    """
    mode_label = "Real" if real_mode else "Sim"

    env = AdversaryEnv(config={
        "max_steps": 40,
        "real_mode": real_mode,
        "telemetry_dir": f"results/telemetry/ppo_eval_{mode_label.lower()}",
    })

    if not os.path.exists(model_path + ".zip"):
        print(f"[ERROR] Model not found: {model_path}.zip")
        return {}

    model = MaskablePPO.load(model_path)
    if verbose:
        print(f"\n>>> [{mode_label} Mode] Evaluating over {num_episodes} episodes...")

    successes, detections = 0, 0
    steps_list, rewards_list = [], []
    action_counts = Counter()
    unique_per_episode = []   # track technique diversity per episode

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_actions = set()

        while not done:
            # Pass the action mask so only valid actions are considered
            action_masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            action_name = ACTION_LIST[action].name
            action_counts[action_name] += 1
            episode_actions.add(action_name)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        unique_per_episode.append(len(episode_actions))

        # Success: DC is fully compromised (privileged)
        dc_won = False
        hosts = env.state.hosts
        if isinstance(hosts, dict):
            dc_won = "dc01" in hosts and hosts["dc01"].privileged
        else:
            dc_won = any(h.name == "dc01" and h.privileged for h in hosts)

        if dc_won:
            successes += 1
            steps_list.append(env.current_step)

        if env.state.any_host_detected():
            detections += 1

        rewards_list.append(episode_reward)

    success_rate   = successes  / num_episodes * 100
    detection_rate = detections / num_episodes * 100
    avg_steps      = np.mean(steps_list) if steps_list else float("nan")
    avg_reward     = np.mean(rewards_list)
    avg_unique     = np.mean(unique_per_episode)

    total_actions = sum(action_counts.values())

    if verbose:
        print("\n" + "=" * 45)
        print(f"PPO EVALUATION RESULTS  [{mode_label} Mode]")
        print("=" * 45)
        print(f"  Episodes evaluated      : {num_episodes}")
        print(f"  Success Rate            : {success_rate:.2f}%")
        print(f"  Detection Rate          : {detection_rate:.2f}%")
        print(f"  Avg Steps to Goal       : {avg_steps:.2f}")
        print(f"  Avg Reward/Episode      : {avg_reward:.2f}")
        print(f"  Avg Unique Techniques   : {avg_unique:.1f} / {len(ACTION_LIST)}")
        print("=" * 45)
        print("\nACTION FREQUENCY (all episodes)")
        print(f"  {'Action':<25} {'Count':>6}  {'% of steps':>10}")
        print("  " + "-" * 45)
        for name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            pct = count / total_actions * 100
            bar = "█" * int(pct / 2)
            print(f"  {name:<25} {count:>6}   {pct:>5.1f}%  {bar}")
        print(f"  {'TOTAL':<25} {total_actions:>6}")
        print()
        print(f"  Techniques NEVER used: "
              f"{', '.join(a.name for a in ACTION_LIST if a.name not in action_counts) or 'None'}")
        print()

    return {
        "name": f"PPO-Agent ({mode_label})",
        "episodes": num_episodes,
        "success_rate": success_rate,
        "detection_rate": detection_rate,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "avg_unique_techniques": avg_unique,
    }


def print_comparison(results: list):
    """Print a formatted sim-vs-real comparison table for the thesis."""
    header = f"{'Mode':<22} {'Success%':>9} {'Detect%':>8} {'AvgSteps':>9} {'AvgReward':>10} {'AvgTech':>8}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("PPO SIM-TO-REAL TRANSFER  —  Thesis Table")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        steps_str = f"{r['avg_steps']:>9.2f}" if r['avg_steps'] == r['avg_steps'] else f"{'N/A':>9}"
        tech_str  = f"{r.get('avg_unique_techniques', 0):>7.1f}"
        print(f"{r['name']:<22} {r['success_rate']:>8.1f}% {r['detection_rate']:>7.1f}% "
              f"{steps_str} {r['avg_reward']:>10.2f} {tech_str}")
    print(sep + "\n")


if __name__ == "__main__":
    MODEL_FILE = "results/models/ppo_adversary_v3"
    EPISODES   = 200

    results = []

    # --- Simulation mode (fast, no mock server needed) ---
    results.append(evaluate_agent(MODEL_FILE, num_episodes=EPISODES,
                                  verbose=True, real_mode=False))

    # --- Real mode (mock server must be running on port 5000) ---
    import socket
    try:
        socket.create_connection(("127.0.0.1", 5000), timeout=1).close()
        results.append(evaluate_agent(MODEL_FILE, num_episodes=EPISODES,
                                      verbose=True, real_mode=True))
    except OSError:
        print("[!] Mock server not reachable on port 5000 — skipping real-mode eval.")
        print("    Start it with: .venv\\Scripts\\python.exe Target/mock_server.py")

    if len(results) > 1:
        print_comparison(results)