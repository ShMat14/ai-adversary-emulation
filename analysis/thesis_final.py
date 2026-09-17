"""
analysis/thesis_final.py

Complete thesis evaluation pipeline:
  1. Runs scripted baselines (Standard, Stealthy, Aggressive, SQLi)
  2. Runs a Random masked-action agent
  3. Uses pre-computed PPO v3 results
  4. Generates 4 publication-quality figures
  5. Prints formatted comparison table
  6. Saves results JSON for the narrative script
"""

import sys, os, random, json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from math import pi

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.adversary_env import AdversaryEnv
from env.attack_actions import ACTION_LIST

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Known PPO v3 results (200-episode evaluation already completed) ──────────
PPO_SIM = {
    "name": "PPO (Sim)",
    "success_rate": 100.0, "detection_rate": 0.0,
    "avg_steps": 14.38,    "avg_reward": 815.38,
    "avg_unique_techniques": 9.0,
    "action_freq": {
        "POWERSHELL_EXEC": 17.9, "CLEAR_LOGS": 14.2,
        "LATERAL_MOVE_SMB": 13.9, "VALID_ACCOUNTS_LOGIN": 11.4,
        "KERBEROASTING": 9.4,    "WEB_SHELL_UPLOAD": 9.2,
        "EXFILTRATE_DATA": 8.3,  "NETWORK_SCAN": 7.4,
        "SQL_INJECTION": 4.3,    "PHISHING_EMAIL": 3.9,
    },
}
PPO_REAL = {
    "name": "PPO (Real)",
    "success_rate": 100.0, "detection_rate": 1.5,
    "avg_steps": 14.96,    "avg_reward": 807.76,
    "avg_unique_techniques": 9.0,
    "action_freq": {
        "POWERSHELL_EXEC": 16.4, "CLEAR_LOGS": 13.5,
        "LATERAL_MOVE_SMB": 13.4, "VALID_ACCOUNTS_LOGIN": 12.3,
        "EXFILTRATE_DATA": 9.9,  "WEB_SHELL_UPLOAD": 8.9,
        "KERBEROASTING": 7.9,    "NETWORK_SCAN": 7.7,
        "PHISHING_EMAIL": 6.2,   "SQL_INJECTION": 3.7,
    },
}

# ─── Colour palette ───────────────────────────────────────────────────────────
PALETTE = {
    "PPO (Sim)":           "#1565C0",
    "PPO (Real)":          "#00838F",
    "Scripted-Standard":   "#E65100",
    "Scripted-Stealthy":   "#6A1B9A",
    "Scripted-Aggressive": "#B71C1C",
    "Scripted-SQLi":       "#2E7D32",
    "Random Agent":        "#546E7A",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _action_idx(name):
    for i, a in enumerate(ACTION_LIST):
        if a.name == name:
            return i
    raise ValueError(name)

def _dc_won(env):
    return env.state.hosts["dc01"].privileged

# ─── Random agent ─────────────────────────────────────────────────────────────
def run_random_agent(episodes=200, max_steps=40, seed=42):
    print(f"  [Random Agent] running {episodes} episodes …")
    random.seed(seed); np.random.seed(seed)
    env = AdversaryEnv(config={"max_steps": max_steps})
    successes = detections = 0
    steps_list, rewards_list, unique_per_ep = [], [], []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False; ep_reward = 0.0; ep_acts = set()
        while not done:
            mask  = env.action_masks()
            valid = np.where(mask)[0]
            act   = int(np.random.choice(valid))
            ep_acts.add(ACTION_LIST[act].name)
            obs, reward, terminated, truncated, _ = env.step(act)
            ep_reward += reward
            done = terminated or truncated
        unique_per_ep.append(len(ep_acts))
        if _dc_won(env):
            successes += 1; steps_list.append(env.current_step)
        if env.state.any_host_detected():
            detections += 1
        rewards_list.append(ep_reward)

    return {
        "name": "Random Agent",
        "success_rate":  successes  / episodes * 100,
        "detection_rate": detections / episodes * 100,
        "avg_steps":     float(np.mean(steps_list)) if steps_list else float("nan"),
        "avg_reward":    float(np.mean(rewards_list)),
        "avg_unique_techniques": float(np.mean(unique_per_ep)),
    }

# ─── Scripted baselines ───────────────────────────────────────────────────────
CHAINS = {
    "Scripted-Standard": [
        "PHISHING_EMAIL","NETWORK_SCAN","VALID_ACCOUNTS_LOGIN",
        "LATERAL_MOVE_SMB","PRIV_ESC_SUDO","EXFILTRATE_DATA",
    ],
    "Scripted-Stealthy": [
        "PHISHING_EMAIL","WEB_SHELL_UPLOAD","CLEAR_LOGS",
        "NETWORK_SCAN","VALID_ACCOUNTS_LOGIN","LATERAL_MOVE_SMB",
        "KERBEROASTING","EXFILTRATE_DATA",
    ],
    "Scripted-Aggressive": [
        "BRUTE_FORCE_SSH","NETWORK_SCAN","VALID_ACCOUNTS_LOGIN",
        "LATERAL_MOVE_SMB","POWERSHELL_EXEC","RANSOMWARE_ENCRYPT",
    ],
    "Scripted-SQLi": [
        "SQL_INJECTION","NETWORK_SCAN","LATERAL_MOVE_SMB",
        "POWERSHELL_EXEC","EXFILTRATE_DATA",
    ],
}

def run_scripted(chain_name, chain, episodes=200, max_steps=40):
    print(f"  [{chain_name}] running {episodes} episodes …")
    env = AdversaryEnv(config={"max_steps": max_steps})
    indices = [_action_idx(n) for n in chain]
    successes = detections = 0
    steps_list, rewards_list, unique_per_ep = [], [], []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False; ep_reward = 0.0; ep_acts = set()
        for idx in indices:
            if done: break
            ep_acts.add(ACTION_LIST[idx].name)
            obs, reward, terminated, truncated, _ = env.step(idx)
            ep_reward += reward
            done = terminated or truncated
        unique_per_ep.append(len(ep_acts))
        if _dc_won(env):
            successes += 1; steps_list.append(env.current_step)
        if env.state.any_host_detected():
            detections += 1
        rewards_list.append(ep_reward)

    return {
        "name": chain_name,
        "success_rate":  successes  / episodes * 100,
        "detection_rate": detections / episodes * 100,
        "avg_steps":     float(np.mean(steps_list)) if steps_list else float("nan"),
        "avg_reward":    float(np.mean(rewards_list)),
        "avg_unique_techniques": float(np.mean(unique_per_ep)),
    }

# ─── PLOT 1: 4-panel metric comparison ───────────────────────────────────────
def plot_metric_comparison(all_results, out_path):
    metrics = [
        ("success_rate",         "Success Rate (%)",        True),
        ("detection_rate",       "Detection Rate (%)",       False),
        ("avg_steps",            "Avg Steps to Goal",        False),
        ("avg_unique_techniques","Avg Unique Techniques",    True),
    ]
    names   = [r["name"] for r in all_results]
    colours = [PALETTE.get(n, "#888") for n in names]
    x       = np.arange(len(names))
    width   = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Agent Performance Comparison — All Baselines vs PPO",
                 fontsize=15, fontweight="bold", y=1.01)

    for ax, (key, label, higher_better) in zip(axes.flat, metrics):
        vals = []
        for r in all_results:
            v = r.get(key, float("nan"))
            vals.append(v if v == v else 0)          # replace nan with 0

        bars = ax.bar(x, vals, width=width, color=colours, edgecolor="white",
                      linewidth=0.8, zorder=3)

        # Annotate bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.01,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.set_title(label, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, max(vals) * 1.20 + 1)

        arrow = "↑ better" if higher_better else "↓ better"
        ax.text(0.98, 0.97, arrow, transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#555",
                style="italic")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─── PLOT 2: Action frequency (Sim vs Real) ───────────────────────────────────
def plot_action_frequency(out_path):
    all_techs = sorted(
        set(PPO_SIM["action_freq"]) | set(PPO_REAL["action_freq"])
    )
    sim_vals  = [PPO_SIM["action_freq"].get(t, 0)  for t in all_techs]
    real_vals = [PPO_REAL["action_freq"].get(t, 0) for t in all_techs]

    # Sort by sim frequency descending
    order      = sorted(range(len(all_techs)), key=lambda i: -sim_vals[i])
    all_techs  = [all_techs[i]  for i in order]
    sim_vals   = [sim_vals[i]   for i in order]
    real_vals  = [real_vals[i]  for i in order]

    y     = np.arange(len(all_techs))
    h     = 0.38
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(y + h/2, sim_vals,  height=h, color=PALETTE["PPO (Sim)"],
            label="PPO (Sim)",  alpha=0.90, edgecolor="white")
    ax.barh(y - h/2, real_vals, height=h, color=PALETTE["PPO (Real)"],
            label="PPO (Real)", alpha=0.90, edgecolor="white")

    for i, (sv, rv) in enumerate(zip(sim_vals, real_vals)):
        ax.text(sv + 0.2, i + h/2, f"{sv:.1f}%", va="center", fontsize=8.5)
        ax.text(rv + 0.2, i - h/2, f"{rv:.1f}%", va="center", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(all_techs, fontsize=10)
    ax.set_xlabel("% of Total Steps", fontsize=11)
    ax.set_title("Technique Usage: PPO Sim vs Real (200 episodes each)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(0, max(max(sim_vals), max(real_vals)) * 1.22)

    # Shade "never used" region label
    unused = ["BRUTE_FORCE_SSH","INSTALL_BACKDOOR","PRIV_ESC_SUDO",
              "RANSOMWARE_ENCRYPT","PASS_THE_HASH"]
    ax.text(0.98, 0.01, f"Never used: {', '.join(unused)}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="#B71C1C", style="italic")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─── PLOT 3: Radar / Spider chart ─────────────────────────────────────────────
def plot_radar(all_results, out_path):
    DIMS = ["Success\nRate", "Stealth\n(1−Detect)", "Efficiency\n(1/Steps)",
            "Technique\nDiversity", "Reward\n(norm)"]

    max_reward = max(r["avg_reward"] for r in all_results if r["avg_reward"] == r["avg_reward"])
    max_tech   = max(r.get("avg_unique_techniques", 0) for r in all_results)
    max_steps  = max(r["avg_steps"] for r in all_results if r["avg_steps"] == r["avg_steps"])

    def to_radar(r):
        steps = r["avg_steps"] if r["avg_steps"] == r["avg_steps"] else max_steps
        return [
            r["success_rate"]  / 100,
            1 - r["detection_rate"] / 100,
            1 - (steps - 1) / (max_steps - 1 + 1e-9),   # fewer steps → higher
            r.get("avg_unique_techniques", 0) / max(max_tech, 1),
            max(r["avg_reward"], 0) / max(max_reward, 1),
        ]

    N    = len(DIMS)
    angles = [pi/2 + 2*pi*i/N for i in range(N)] + [pi/2]  # close polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks([2*pi*i/N for i in range(N)])
    ax.set_xticklabels(DIMS, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="gray")
    ax.spines["polar"].set_visible(False)

    for r in all_results:
        vals  = to_radar(r)
        vals += [vals[0]]
        plot_angles = [2*pi*i/N for i in range(N)] + [0]
        colour = PALETTE.get(r["name"], "#888")
        ax.plot(plot_angles, vals, "o-", linewidth=2, color=colour,
                label=r["name"])
        ax.fill(plot_angles, vals, alpha=0.07, color=colour)

    ax.set_title("Multi-Dimensional Agent Profile\n(Radar Chart)",
                 fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─── PLOT 4: Sim-to-Real transfer gap ─────────────────────────────────────────
def plot_sim_real_gap(out_path):
    metrics  = ["Success Rate\n(%)", "Detection Rate\n(%)",
                "Avg Steps", "Avg Reward", "Unique\nTechniques"]
    sim_vals = [100.0, 0.0,  14.38, 815.38, 9.0]
    rea_vals = [100.0, 1.5,  14.96, 807.76, 9.0]

    # Absolute delta (pp / raw units) — avoids divide-by-zero when sim=0
    deltas = [r - s for s, r in zip(sim_vals, rea_vals)]
    delta_labels = ["+0.0 pp", "+1.5 pp", "+0.58", "−7.62", "0.0"]
    # Colour: green = increased, red = decreased (context: detect↑ is bad)
    bad_increase = {1}   # Detection Rate: higher is worse
    colours = []
    for i, d in enumerate(deltas):
        if d == 0:
            colours.append("#90A4AE")
        elif i in bad_increase:
            colours.append("#C62828" if d > 0 else "#2E7D32")
        else:
            colours.append("#2E7D32" if d > 0 else "#C62828")

    x = np.arange(len(metrics))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left — raw values side-by-side
    ax = axes[0]
    w  = 0.35
    b1 = ax.bar(x - w/2, sim_vals, width=w, color=PALETTE["PPO (Sim)"],
                label="PPO (Sim)", edgecolor="white", zorder=3)
    b2 = ax.bar(x + w/2, rea_vals, width=w, color=PALETTE["PPO (Real)"],
                label="PPO (Real)", edgecolor="white", zorder=3)
    for bar, v in zip(list(b1)+list(b2), sim_vals+rea_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(sim_vals+rea_vals)*0.01,
                f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_title("Raw Metrics: Sim vs Real", fontweight="bold")
    ax.legend()

    # Right — absolute delta with clear labels
    ax2 = axes[1]
    bars = ax2.bar(x, deltas, color=colours, edgecolor="white", zorder=3, width=0.5)
    for bar, d, lbl in zip(bars, deltas, delta_labels):
        offset = max(abs(d) * 0.05, 0.3)
        ypos = bar.get_height() + offset if d >= 0 else bar.get_height() - offset * 3
        ax2.text(bar.get_x() + bar.get_width()/2, ypos,
                 lbl, ha="center", va="bottom" if d >= 0 else "top",
                 fontsize=10, fontweight="bold")
    ax2.axhline(0, color="black", linewidth=1.0)
    ax2.set_xticks(x); ax2.set_xticklabels(metrics, rotation=15, ha="right")
    ax2.set_ylabel("Δ  (Real − Sim,  absolute units)")
    ax2.set_title("Sim-to-Real Transfer Gap  (absolute change)", fontweight="bold")

    # Footnote
    ax2.text(0.99, 0.97,
             "Green = favourable  |  Red = unfavourable  |  Grey = no change",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=8, color="#555", style="italic")

    fig.suptitle("PPO Agent: Simulation-to-Real Transfer Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─── Print table ──────────────────────────────────────────────────────────────
def print_table(all_results):
    hdr = (f"{'Agent':<24} {'Success%':>9} {'Detect%':>8} "
           f"{'AvgSteps':>9} {'AvgReward':>10} {'AvgTech':>8}")
    sep = "─" * len(hdr)
    print(f"\n{sep}")
    print("THESIS TABLE — PPO vs All Baselines  (200 episodes each)")
    print(sep)
    print(hdr)
    print(sep)
    for r in all_results:
        steps = f"{r['avg_steps']:9.2f}" if r['avg_steps'] == r['avg_steps'] else "      N/A"
        tech  = f"{r.get('avg_unique_techniques', 0):7.1f}"
        print(f"{r['name']:<24} {r['success_rate']:8.1f}% "
              f"{r['detection_rate']:7.1f}% {steps} "
              f"{r['avg_reward']:10.2f} {tech}")
    print(sep + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    EPISODES = 200
    print("\n" + "="*60)
    print("  THESIS FINAL ANALYSIS")
    print("="*60)

    # 1. Run baselines
    print("\n[1/4] Running scripted baselines …")
    baseline_results = []
    for cname, chain in CHAINS.items():
        baseline_results.append(run_scripted(cname, chain, EPISODES))

    print("\n[2/4] Running random agent …")
    random_result = run_random_agent(EPISODES)

    # 2. Assemble ordered results table
    all_results = baseline_results + [random_result, PPO_SIM, PPO_REAL]
    print_table(all_results)

    # 3. Save JSON for narrative script
    json_path = os.path.join(ROOT, "analysis", "thesis_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved → {json_path}")

    # 4. Generate plots
    print("\n[3/4] Generating plots …")
    plot_metric_comparison(
        all_results,
        os.path.join(ROOT, "thesis_plot_5_metric_comparison.png"))
    plot_action_frequency(
        os.path.join(ROOT, "thesis_plot_5_action_freq.png"))
    plot_radar(
        all_results,
        os.path.join(ROOT, "thesis_plot_5_radar.png"))
    plot_sim_real_gap(
        os.path.join(ROOT, "thesis_plot_5_sim_real_gap.png"))

    print("\n[4/4] Done. All plots saved to project root.")
    print("="*60 + "\n")

    return all_results


if __name__ == "__main__":
    main()
