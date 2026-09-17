import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from stable_baselines3 import PPO
import env.adversary_env as env_config
from env.adversary_env import AdversaryEnv, ACTION_LIST

# ==========================================
# FINAL THESIS DATA (N=50 Real World Validation)
# ==========================================
SIM_SUCCESS_RATE = 90.00  # Baseline
REAL_SUCCESS_RATE = 94.00  # FINAL RESULT (N=50)
SIM_AVG_STEPS = 7.82  # Baseline
REAL_AVG_STEPS = 15.91  # FINAL RESULT (N=50)

MODEL_PATH = "results/models/ppo_adversary_v2"
NUM_EPISODES_FOR_DISTRIBUTION = 100


def generate_plots():
    print(">>> Generating Final Thesis Plots (N=50 Validation Data)...")

    # 1. GENERATE ATTACK DISTRIBUTION DATA
    env_config.REAL_MODE = False
    env = AdversaryEnv()
    model = PPO.load(MODEL_PATH)
    action_counts = Counter()

    for _ in range(NUM_EPISODES_FOR_DISTRIBUTION):
        obs, _ = env.reset()
        done = False
        while not done:
            action_idx, _ = model.predict(obs, deterministic=True)
            action_name = ACTION_LIST[int(action_idx)].name
            action_counts[action_name] += 1
            obs, _, done, _, _ = env.step(action_idx)

    # ==========================================
    # PLOT 1: THE KILL CHAIN (Technique Frequency)
    # ==========================================
    plt.figure(figsize=(14, 8))
    sorted_actions = action_counts.most_common()
    actions = [x[0].replace('_', '\n') for x in sorted_actions]
    counts = [x[1] for x in sorted_actions]

    colors = []
    for act in actions:
        act_flat = act.replace('\n', '_')
        if "EXFIL" in act_flat or "RANSOM" in act_flat:
            colors.append('#d62728')  # Red — Impact
        elif "PHISH" in act_flat or "BRUTE" in act_flat:
            colors.append('#ff7f0e')  # Orange — Initial Access (noisy)
        elif "SQL" in act_flat:
            colors.append('#9467bd')  # Purple — SQL Injection (OWASP T1190)
        elif "SCAN" in act_flat:
            colors.append('#2ca02c')  # Green — Discovery
        else:
            colors.append('#1f77b4')  # Blue — Everything else

    bars = plt.bar(actions, counts, color=colors, edgecolor='black', alpha=0.9)
    plt.title(f"Attack Technique Distribution (N={NUM_EPISODES_FOR_DISTRIBUTION})", fontsize=16, fontweight='bold')
    plt.ylabel("Total Executions", fontsize=12)
    plt.xlabel("MITRE ATT&CK Technique", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig("thesis_plot_1_kill_chain.png", dpi=300)
    print("✅ Saved 'thesis_plot_1_kill_chain.png'")

    # ==========================================
    # PLOT 2: SIM VS REAL SUCCESS (90 vs 94)
    # ==========================================
    plt.figure(figsize=(8, 6))
    modes = ['Simulation\n(Baseline)', 'Real World\n(N=50 Validation)']
    rates = [SIM_SUCCESS_RATE, REAL_SUCCESS_RATE]
    colors = ['#2ca02c', '#ff7f0e']

    bars2 = plt.bar(modes, rates, color=colors, width=0.5, edgecolor='black')
    plt.title("Sim-to-Real Transfer: Success Rate", fontsize=16, fontweight='bold')
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.ylim(0, 110)

    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 2,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.savefig("thesis_plot_2_success_comparison.png", dpi=300)
    print("✅ Saved 'thesis_plot_2_success_comparison.png'")

    # ==========================================
    # PLOT 3: THE COST OF FRICTION (7.8 vs 15.9)
    # ==========================================
    plt.figure(figsize=(8, 6))
    modes = ['Simulation', 'Real World']
    steps = [SIM_AVG_STEPS, REAL_AVG_STEPS]

    colors = ['#1f77b4', '#d62728']

    bars3 = plt.bar(modes, steps, color=colors, width=0.5, edgecolor='black')
    plt.title("Environmental Friction: Average Steps to Win", fontsize=16, fontweight='bold')
    plt.ylabel("Average Steps per Episode", fontsize=12)
    plt.ylim(0, max(steps) + 5)

    for bar in bars3:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.savefig("thesis_plot_3_friction_cost.png", dpi=300)
    print("✅ Saved 'thesis_plot_3_friction_cost.png'")

    plt.show()


if __name__ == "__main__":
    generate_plots()