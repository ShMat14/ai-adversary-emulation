"""
Generate thesis-quality plots for PPO v3 evaluation results.
Run from project root: python analysis/thesis_plots_v3.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d27"
ACCENT1 = "#7c6af7"   # purple
ACCENT2 = "#38c9b0"   # teal
WARN    = "#f97316"   # orange
RED     = "#ef4444"
GRAY    = "#3a3f52"
TEXT    = "#e2e8f0"
SUBTEXT = "#94a3b8"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   GRAY,
    "axes.labelcolor":  TEXT,
    "xtick.color":      SUBTEXT,
    "ytick.color":      SUBTEXT,
    "text.color":       TEXT,
    "grid.color":       GRAY,
    "grid.alpha":       0.4,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── Data ────────────────────────────────────────────────────────────────
ALL_TECHNIQUES = [
    "PHISHING_EMAIL", "BRUTE_FORCE_SSH", "NETWORK_SCAN",
    "VALID_ACCOUNTS_LOGIN", "INSTALL_BACKDOOR", "CLEAR_LOGS",
    "LATERAL_MOVE_SMB", "PRIV_ESC_SUDO", "EXFILTRATE_DATA",
    "RANSOMWARE_ENCRYPT", "SQL_INJECTION", "PASS_THE_HASH",
    "POWERSHELL_EXEC", "KERBEROASTING", "WEB_SHELL_UPLOAD",
]

SIM_PCT = {
    "POWERSHELL_EXEC": 17.9, "CLEAR_LOGS": 14.2, "LATERAL_MOVE_SMB": 13.9,
    "VALID_ACCOUNTS_LOGIN": 11.4, "KERBEROASTING": 9.4, "WEB_SHELL_UPLOAD": 9.2,
    "EXFILTRATE_DATA": 8.3, "NETWORK_SCAN": 7.4, "SQL_INJECTION": 4.3,
    "PHISHING_EMAIL": 3.9,
}
REAL_PCT = {
    "POWERSHELL_EXEC": 16.4, "CLEAR_LOGS": 13.5, "LATERAL_MOVE_SMB": 13.4,
    "VALID_ACCOUNTS_LOGIN": 12.3, "EXFILTRATE_DATA": 9.9, "WEB_SHELL_UPLOAD": 8.9,
    "KERBEROASTING": 7.9, "NETWORK_SCAN": 7.7, "PHISHING_EMAIL": 6.2,
    "SQL_INJECTION": 3.7,
}

# MITRE tactic labels per technique
TACTIC = {
    "PHISHING_EMAIL":      "Initial Access",
    "BRUTE_FORCE_SSH":     "Initial Access",
    "SQL_INJECTION":       "Initial Access",
    "NETWORK_SCAN":        "Discovery",
    "VALID_ACCOUNTS_LOGIN":"Credential Access",
    "INSTALL_BACKDOOR":    "Persistence",
    "WEB_SHELL_UPLOAD":    "Persistence",
    "CLEAR_LOGS":          "Defense Evasion",
    "LATERAL_MOVE_SMB":    "Lateral Movement",
    "PASS_THE_HASH":       "Lateral Movement",
    "PRIV_ESC_SUDO":       "Privilege Escalation",
    "POWERSHELL_EXEC":     "Privilege Escalation",
    "KERBEROASTING":       "Privilege Escalation",
    "EXFILTRATE_DATA":     "Impact",
    "RANSOMWARE_ENCRYPT":  "Impact",
}

TACTIC_COLOR = {
    "Initial Access":       "#f97316",
    "Discovery":            "#38c9b0",
    "Credential Access":    "#7c6af7",
    "Persistence":          "#ec4899",
    "Defense Evasion":      "#eab308",
    "Lateral Movement":     "#3b82f6",
    "Privilege Escalation": "#ef4444",
    "Impact":               "#22c55e",
}

# ════════════════════════════════════════════════════════════════════════
# PLOT 1 — Technique Distribution: Sim vs Real (sorted by sim usage)
# ════════════════════════════════════════════════════════════════════════
sorted_techs = sorted(ALL_TECHNIQUES, key=lambda t: SIM_PCT.get(t, 0), reverse=True)
sim_vals  = [SIM_PCT.get(t, 0)  for t in sorted_techs]
real_vals = [REAL_PCT.get(t, 0) for t in sorted_techs]
colors    = [TACTIC_COLOR[TACTIC[t]] for t in sorted_techs]

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(sorted_techs))
w = 0.35

bars_sim  = ax.bar(x - w/2, sim_vals,  w, color=colors, alpha=0.95, label="Sim Mode",  zorder=3)
bars_real = ax.bar(x + w/2, real_vals, w, color=colors, alpha=0.55, label="Real Mode",
                   edgecolor=colors, linewidth=1.2, zorder=3)

# Hatching for "never used"
for i, (t, s, r) in enumerate(zip(sorted_techs, sim_vals, real_vals)):
    if s == 0:
        ax.bar(x[i] - w/2, 0.3, w, color=GRAY, alpha=0.5, zorder=3)
        ax.bar(x[i] + w/2, 0.3, w, color=GRAY, alpha=0.5, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels([t.replace("_", "\n") for t in sorted_techs],
                   fontsize=7.5, rotation=0, ha="center")
ax.set_ylabel("% of Total Steps", fontsize=11)
ax.set_title("PPO v3 — Technique Usage Distribution  (Sim vs Real Mode)",
             fontsize=14, fontweight="bold", pad=16)
ax.yaxis.grid(True, zorder=0)
ax.set_ylim(0, 22)

# Legend: tactic colours
tactic_patches = [mpatches.Patch(color=c, label=t)
                  for t, c in TACTIC_COLOR.items()]
mode_patches = [
    mpatches.Patch(color="white", alpha=0.9, label="■ Sim Mode (solid)"),
    mpatches.Patch(color="white", alpha=0.5, label="□ Real Mode (faded)"),
]
leg1 = ax.legend(handles=tactic_patches, loc="upper right",
                 fontsize=8, title="MITRE Tactic", title_fontsize=9,
                 framealpha=0.2, ncol=2)
ax.add_artist(leg1)

# "Never used" annotation
ax.axhline(0.5, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.6)
ax.text(13.5, 1.0, "Never used →", fontsize=8, color=SUBTEXT, ha="right")

plt.tight_layout()
plt.savefig("thesis_plot_v3_technique_distribution.png", dpi=600,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("✓ Saved: thesis_plot_v3_technique_distribution.png")


# ════════════════════════════════════════════════════════════════════════
# PLOT 2 — v2 vs v3 Key Metrics Comparison
# ════════════════════════════════════════════════════════════════════════
metrics     = ["Success\nRate (%)", "Detection\nRate (%)", "Avg Steps\nto Goal",
               "Avg Reward\n(÷10)", "Unique\nTechniques"]
v2_sim_vals = [100.0, 0.50,  6.50, 62.89, 5.0]
v3_sim_vals = [100.0, 0.00, 14.38, 81.54, 9.0]
v2_rl_vals  = [99.5,  4.00,  7.57, 61.07, 5.0]
v3_rl_vals  = [100.0, 1.50, 14.96, 80.78, 9.0]

fig, axes = plt.subplots(1, 5, figsize=(16, 5))
fig.suptitle("PPO Agent — v2 vs v3 Performance Comparison",
             fontsize=15, fontweight="bold", y=1.02)

bar_colors = [GRAY, ACCENT1, GRAY, ACCENT2]
labels     = ["v2 Sim", "v3 Sim", "v2 Real", "v3 Real"]

for ax, metric, vals in zip(axes, metrics,
        zip(v2_sim_vals, v3_sim_vals, v2_rl_vals, v3_rl_vals)):
    bars = ax.bar(labels, vals, color=bar_colors, alpha=0.9, width=0.6, zorder=3)
    ax.yaxis.grid(True, zorder=0, alpha=0.5)
    ax.set_title(metric, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)

    # Value labels on bars
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.02,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25)

plt.tight_layout()
plt.savefig("thesis_plot_v3_v2_comparison.png", dpi=600,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("✓ Saved: thesis_plot_v3_v2_comparison.png")


# ════════════════════════════════════════════════════════════════════════
# PLOT 3 — Kill Chain Phase Coverage (Radar / Spider chart)
# ════════════════════════════════════════════════════════════════════════
phases   = ["Initial\nAccess", "Discovery", "Credential\nAccess",
            "Persistence", "Defense\nEvasion", "Lateral\nMovement",
            "Privilege\nEscalation", "Impact"]
v2_cover = [1, 0, 1, 1, 0, 0, 0, 1]   # phases covered by v2 (1=yes,0=no)
v3_cover = [1, 1, 1, 1, 1, 1, 1, 1]   # v3 covers all 8 phases

N = len(phases)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

v2_vals = v2_cover + v2_cover[:1]
v3_vals = v3_cover + v3_cover[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.set_facecolor(PANEL)
fig.patch.set_facecolor(BG)

ax.plot(angles, v2_vals, color=GRAY,    linewidth=2, label="v2 (5 techniques)")
ax.fill(angles, v2_vals, color=GRAY,    alpha=0.25)
ax.plot(angles, v3_vals, color=ACCENT1, linewidth=2.5, label="v3 (10 techniques)")
ax.fill(angles, v3_vals, color=ACCENT1, alpha=0.30)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(phases, fontsize=10, color=TEXT)
ax.set_yticks([0, 1])
ax.set_yticklabels(["", "Covered"], fontsize=8, color=SUBTEXT)
ax.set_ylim(0, 1.2)
ax.grid(color=GRAY, alpha=0.5)
ax.spines["polar"].set_color(GRAY)

ax.set_title("MITRE ATT&CK Phase Coverage\nv2 vs v3",
             fontsize=14, fontweight="bold", pad=24, color=TEXT)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15),
          fontsize=10, framealpha=0.2)

plt.tight_layout()
plt.savefig("thesis_plot_v3_phase_coverage.png", dpi=600,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("✓ Saved: thesis_plot_v3_phase_coverage.png")


# ════════════════════════════════════════════════════════════════════════
# PLOT 4 — Sim-to-Real Transfer Gap (v2 vs v3)
# ════════════════════════════════════════════════════════════════════════
transfer_metrics = ["Success %", "Detection %", "Avg Steps", "Avg Reward"]
v2_gaps = [abs(100.0-99.5), abs(0.5-4.0),  abs(6.50-7.57), abs(628.86-610.69)]
v3_gaps = [abs(100.0-100.0),abs(0.0-1.5),  abs(14.38-14.96),abs(815.38-807.76)]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(transfer_metrics))
w = 0.35

ax.bar(x - w/2, v2_gaps, w, color=WARN,    alpha=0.9, label="v2 Sim→Real gap", zorder=3)
ax.bar(x + w/2, v3_gaps, w, color=ACCENT2, alpha=0.9, label="v3 Sim→Real gap", zorder=3)

for i, (g2, g3) in enumerate(zip(v2_gaps, v3_gaps)):
    ax.text(i - w/2, g2 + max(v2_gaps)*0.02, f"{g2:.2f}",
            ha="center", fontsize=9, fontweight="bold", color=WARN)
    ax.text(i + w/2, g3 + max(v2_gaps)*0.02, f"{g3:.2f}",
            ha="center", fontsize=9, fontweight="bold", color=ACCENT2)

ax.set_xticks(x)
ax.set_xticklabels(transfer_metrics, fontsize=11)
ax.set_ylabel("Absolute Gap  (Sim − Real)", fontsize=11)
ax.set_title("Sim-to-Real Transfer Gap  —  v2 vs v3", fontsize=13,
             fontweight="bold", pad=14)
ax.yaxis.grid(True, zorder=0, alpha=0.5)
ax.legend(fontsize=10, framealpha=0.2)
ax.set_ylim(0, max(max(v2_gaps), max(v3_gaps)) * 1.35)

plt.tight_layout()
plt.savefig("thesis_plot_v3_transfer_gap.png", dpi=600,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("✓ Saved: thesis_plot_v3_transfer_gap.png")

print("\nAll 4 thesis plots generated successfully.")
