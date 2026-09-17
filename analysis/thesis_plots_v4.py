# -*- coding: utf-8 -*-
"""
Thesis-style plots for v4, and the comparison across all versions.

Matches the visual language of the v3 thesis figures (white panel, the same
purple/teal categorical hues, DejaVu Sans, no top/right spines) so the v4
figures sit beside the existing ones without a seam. Four figures:

  1. version_progression  success rate v1 -> v4, the evolution of the system
  2. masking_fidelity     v4 masked vs unmasked: the objective is reached either
                          way, but only masking keeps the telemetry executable
  3. detection_rules      which named blue-team rules fire, per 300 episodes --
                          the real, explainable detection v4 introduced
  4. sim_vs_live          the trained agent scores the same in simulation and
                          against the live server: transfer with no gap

    python analysis/thesis_plots_v4.py

A caveat encoded in the captions, not hidden: reward and detection are NOT
comparable across versions (v4 changed the reward scale and replaced the
abstract detection term with a rule-based one). Success rate is the metric that
carries the same meaning throughout, so the cross-version figure uses it.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── palette (identical to thesis_plots_v3_light) ──────────────────────────
BG = PANEL = "#ffffff"
ACCENT1 = "#5B4BD6"   # purple
ACCENT2 = "#009E73"   # teal
WARN = "#f97316"      # orange
RED = "#ef4444"
GRAY = "#d9d9d9"
TEXT = "#1a1a1a"
SUBTEXT = "#555555"
INK2 = "#0072B2"      # blue, for a second series

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRAY, "axes.labelcolor": TEXT,
    "xtick.color": SUBTEXT, "ytick.color": SUBTEXT, "text.color": TEXT,
    "grid.color": GRAY, "grid.alpha": 0.8,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

OUTDIR = "."


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", path)


# ── version numbers ───────────────────────────────────────────────────────
# v1/v2/v3 from the manuscript version-progression table; v4 from
# analysis/v4_results.json (masked, sim) and the live-server evaluation.
VERSIONS = [
    ("v1", "9 actions\nno masking",           62.0),
    ("v2", "15 actions\nmasking, fixed net",  94.0),
    ("v3", "15 actions\n+ randomisation",     100.0),
    ("v4", "18 actions\nshared model",        100.0),
]


def fig_version_progression():
    labels = [v[0] for v in VERSIONS]
    sub = [v[1] for v in VERSIONS]
    vals = [v[2] for v in VERSIONS]
    colors = [GRAY, "#b9b3ec", ACCENT1, ACCENT2]

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, width=0.62, color=colors, edgecolor="white", linewidth=1.5, zorder=3)
    ax.plot(x, vals, color=SUBTEXT, lw=1.3, marker="o", markersize=5,
            markerfacecolor="white", markeredgecolor=SUBTEXT, zorder=4)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 2, f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold", color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n{s}" for l, s in zip(labels, sub)], fontsize=9.5)
    # bold just the version token on the first line
    for tick, l in zip(ax.get_xticklabels(), labels):
        tick.set_fontsize(9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Mission success rate (%)", fontsize=10.5)
    ax.grid(axis="y", zorder=0)
    ax.set_title("System evolution: mission success across versions",
                 fontsize=12.5, fontweight="bold", pad=12)
    fig.subplots_adjust(bottom=0.26)
    fig.text(0.5, 0.015,
             "v4 reaches 100% in simulation and against the live server. Reward and "
             "detection are\nnot shown here — v4 changed both scales, so only success "
             "rate is comparable across versions.",
             ha="center", fontsize=8, color=SUBTEXT)
    save(fig, "thesis_plot_v4_version_progression.png")


def fig_masking_fidelity():
    res = json.load(open("analysis/v4_results.json"))
    m, n = res["masked"], res["nomask"]
    groups = ["Mission\nsuccess", "Impossible\nactions"]
    masked = [m["success_mean"], m["illegal_mean"]]
    nomask = [n["success_mean"], n["illegal_mean"]]

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    x = np.arange(len(groups))
    w = 0.36
    b1 = ax.bar(x - w/2, masked, w, label="MaskablePPO (masked)", color=ACCENT2,
                edgecolor="white", linewidth=1.4, zorder=3)
    b2 = ax.bar(x + w/2, nomask, w, label="PPO (no mask)", color=WARN,
                edgecolor="white", linewidth=1.4, zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2,
                    f"{b.get_height():.1f}%", ha="center", fontsize=9.5, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10.5)
    ax.set_ylim(0, 116)
    ax.set_ylabel("Percentage", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax.grid(axis="y", zorder=0)
    ax.set_title("Masking is fidelity, not performance (v4)",
                 fontsize=12.5, fontweight="bold", pad=28)
    fig.text(0.5, -0.05,
             "Both reach the objective every time. Without the mask, 95.5% of the "
             "agent's actions are impossible in the state it attempts them.",
             ha="center", fontsize=8, color=SUBTEXT)
    save(fig, "thesis_plot_v4_masking_fidelity.png")


def fig_detection_rules():
    # counts per 300-episode eval of the masked agent (from v4_train eval)
    rules = [
        ("Pass-the-Hash NTLM anomaly", "T1550.002", 382),
        ("Exfil outbound data volume", "T1041", 305),
        ("Kerberoast RC4 TGS", "T1558.003", 138),
        ("LDAP account enumeration", "T1087.002", 86),
        ("Valid-account logon", "T1078", 61),
    ]
    rules.reverse()
    names = [f"{r[0]}\n{r[1]}" for r in rules]
    vals = [r[2] for r in rules]

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    y = np.arange(len(names))
    ax.barh(y, vals, color=ACCENT1, edgecolor="white", linewidth=1.2, zorder=3, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 4, yi, str(v), va="center", fontsize=9.5, color=TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_xlabel("Times the rule fired (300 episodes)", fontsize=10.5)
    ax.grid(axis="x", zorder=0)
    ax.set_title("Detection is explainable: which blue-team rules fire (v4)",
                 fontsize=12.5, fontweight="bold", pad=12)
    fig.text(0.5, -0.05,
             "Each detection is a named rule on a real Windows/Sysmon event, not an "
             "abstract probability. The stealthy agent mostly trips the quiet rules.",
             ha="center", fontsize=8, color=SUBTEXT)
    save(fig, "thesis_plot_v4_detection_rules.png")


def fig_sim_vs_live():
    # trained MaskablePPO, sim vs live server (100-episode run)
    metrics = ["Success", "Detection", "Reward\n(÷10)"]
    sim = [100.0, 0.0, 64.3]
    live = [100.0, 1.0, 63.9]
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    x = np.arange(len(metrics))
    w = 0.36
    b1 = ax.bar(x - w/2, sim, w, label="Simulation", color=ACCENT1,
                edgecolor="white", linewidth=1.4, zorder=3)
    b2 = ax.bar(x + w/2, live, w, label="Live server (HTTP)", color=ACCENT2,
                edgecolor="white", linewidth=1.4, zorder=3)
    for bars, data in ((b1, sim), (b2, live)):
        for b, d in zip(bars, data):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.5, f"{d:.1f}",
                    ha="center", fontsize=9, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Value", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.grid(axis="y", zorder=0)
    ax.set_title("Sim-to-real transfer: the trained agent, evaluated live",
                 fontsize=12.5, fontweight="bold", pad=12)
    fig.text(0.5, -0.05,
             "The agent trained in simulation scores identically against the live "
             "server (100% success), because both run one model. Reward shown ÷10.",
             ha="center", fontsize=8, color=SUBTEXT)
    save(fig, "thesis_plot_v4_sim_vs_live.png")


if __name__ == "__main__":
    fig_version_progression()
    fig_masking_fidelity()
    fig_detection_rules()
    fig_sim_vs_live()
    print("\nv4 thesis-style figures written.")
