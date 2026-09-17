# -*- coding: utf-8 -*-
"""
The result figures for Section 4, regenerated from the verified numbers.

The previous versions hard-coded their own copies of the evaluation results and
were not updated when the evaluation was re-run, so the transfer-gap figure
disagreed with the text beside it. Everything here reads analysis/verified_results.py.

Two presentational decisions:

  * The transfer figure shows v3 alone. The earlier version set v3's gap beside
    v2's, but v2 comes from a development run that predates version control and
    cannot be reproduced; placing a verified bar next to an unverifiable one in
    a "which is better" framing invites a comparison the evidence does not
    support.
  * White surface throughout, matching the rest of the plot figures and the
    print edition.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.verified_results import V3_SIM, V3_REAL, BASELINES, transfer_gap, MEASURED

BLUE, VERM, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#8c8c8c"
INK, MUTED, GRID = "#1a1a1a", "#555555", "#d9d9d9"
NL = "\n"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "font.size": 9, "legend.frameon": False,
})


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)


# ══════════════════════════════════════════════════════════════════════
# every agent, on the metrics that matter
# ══════════════════════════════════════════════════════════════════════
def figure_all_agents(out):
    rows = [(k, v["success_rate"], v["detection_rate"], v["avg_reward"], GREY)
            for k, v in BASELINES.items()]
    rows.append(("PPO v3 (Sim)", V3_SIM["success_rate"], V3_SIM["detection_rate"],
                 V3_SIM["avg_reward"], BLUE))
    rows.append(("PPO v3 (Real)", V3_REAL["success_rate"], V3_REAL["detection_rate"],
                 V3_REAL["avg_reward"], GREEN))

    labels = [r[0].replace("Scripted-", "Scripted" + NL) for r in rows]
    colours = [r[4] for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (title, idx, unit) in zip(axes, [("Mission success", 1, "%"),
                                             ("Detection", 2, "%"),
                                             ("Mean episode reward", 3, "")]):
        vals = [r[idx] for r in rows]
        ax.bar(x, vals, width=0.66, color=colours, edgecolor="white", linewidth=1.3)
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.4, rotation=30, ha="right")
        ax.set_ylim(0, max(vals) * 1.24 if max(vals) else 1)
        tidy(ax)
        for xi, v in enumerate(vals):
            ax.text(xi, v + max(vals) * 0.03, f"{v:.1f}{unit}",
                    ha="center", fontsize=7.6, color=INK)

    fig.suptitle("All agents over 200 episodes each", fontsize=11, y=1.04)
    fig.text(0.5, -0.13, f"Scripted baselines and the random agent in grey; the trained policy "
             f"in colour. PPO figures re-verified {MEASURED}.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ══════════════════════════════════════════════════════════════════════
# simulation against the live deployment
# ══════════════════════════════════════════════════════════════════════
def figure_transfer(out):
    metrics = [("Success", "success_rate", "%"), ("Detection", "detection_rate", "%"),
               ("Steps to goal", "avg_steps", ""), ("Mean reward", "avg_reward", "")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    x = np.arange(2)
    for ax, (title, key, unit) in zip(axes, metrics):
        vals = [V3_SIM[key], V3_REAL[key]]
        ax.bar(x, vals, width=0.55, color=[BLUE, GREEN],
               edgecolor="white", linewidth=1.4)
        ax.set_title(title, fontsize=9.5, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["simulation", "live server"], fontsize=8.4)
        ax.set_ylim(0, max(vals) * 1.30 if max(vals) else 1)
        tidy(ax)
        for xi, v in enumerate(vals):
            ax.text(xi, v + max(vals) * 0.04, f"{v:g}{unit}",
                    ha="center", fontsize=8.4, color=INK)
        gap = abs(vals[0] - vals[1])
        if gap:
            ax.set_xlabel(f"gap {gap:.2f}{unit}", fontsize=8.2, color=MUTED)

    fig.suptitle("Simulation-to-real transfer: the same policy, unchanged, against a live target",
                 fontsize=10.5, y=1.04)
    fig.text(0.5, -0.10, f"200 episodes per mode, measured {MEASURED}. "
             f"Technique diversity is 9.0 of 15 in both modes.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    figure_all_agents("images/fig_all_agents.png")
    figure_transfer("images/fig_transfer.png")
