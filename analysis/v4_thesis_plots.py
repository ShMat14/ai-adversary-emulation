# -*- coding: utf-8 -*-
"""
Thesis-quality figures for the v4 results, matching the figure TYPES used in the
v3 thesis plots (analysis/thesis_plots_v3.py) and reproducing each relevant one
with v4 data, plus the figures the v4 story adds (scenario adaptation, technique
breadth over all 25, rule-based detection, sim-to-real transfer under one shared
model).

Reads the result JSONs so it always plots the current numbers:
    analysis/v4_results.json           masked vs nomask (success/detection/reward/illegal)
    analysis/v4_thorough_result.json   per-scenario breakdown + technique usage + adaptation
    analysis/v4_transfer_result.json   sim vs live (sim-to-real transfer)

Run from the project root:
    python analysis/v4_thesis_plots.py
Writes thesis_plot_v4_*.png (600 dpi) to the project root.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── palette (matches thesis_plots_v3.py) ────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d27"
ACCENT1 = "#7c6af7"   # purple  -> masked / sim
ACCENT2 = "#38c9b0"   # teal    -> nomask / live
WARN    = "#f97316"   # orange
RED     = "#ef4444"
GRAY    = "#3a3f52"
TEXT    = "#e2e8f0"
SUBTEXT = "#94a3b8"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": GRAY,
    "axes.labelcolor": TEXT, "xtick.color": SUBTEXT, "ytick.color": SUBTEXT,
    "text.color": TEXT, "grid.color": GRAY, "grid.alpha": 0.4,
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# kill-chain phases, for ordering techniques and measuring phase coverage
PHASES = {
    "Entry": ["PHISHING_EMAIL", "SQL_INJECTION", "PASSWORD_SPRAYING", "BRUTE_FORCE_SSH"],
    "Discovery": ["NETWORK_SCAN", "DOMAIN_ACCT_DISCOVERY", "DOMAIN_TRUST_DISCOVERY"],
    "Cred access": ["VALID_ACCOUNTS_LOGIN", "AS_REP_ROASTING", "CRED_DUMP_LSASS"],
    "Lateral": ["LATERAL_MOVE_SMB", "PASS_THE_HASH", "PASS_THE_TICKET"],
    "DC dominance": ["KERBEROASTING", "DCSYNC", "GPO_MODIFICATION", "GOLDEN_TICKET"],
    "Persistence": ["INSTALL_BACKDOOR", "WEB_SHELL_UPLOAD"],
    "Host priv-esc": ["PRIV_ESC_SUDO", "POWERSHELL_EXEC", "ACCOUNT_MANIPULATION"],
    "Evasion": ["CLEAR_LOGS"],
    "Impact": ["EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT"],
}


def _load(name):
    with open(os.path.join(ROOT, "analysis", name)) as f:
        return json.load(f)


def _save(fig, name):
    path = os.path.join(ROOT, name)
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  wrote {name}")


# ── 1. masking fidelity: the headline (success held, illegal eliminated) ─────
def fig_masking_fidelity(res):
    m, n = res["masked"], res["nomask"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))

    # left: success rate is matched
    labels = ["Masked\n(MaskablePPO)", "Unmasked\n(PPO)"]
    succ = [m["success_mean"], n["success_mean"]]
    err = [m["success_sd"], n["success_sd"]]
    bars = axL.bar(labels, succ, yerr=err, capsize=6,
                   color=[ACCENT1, ACCENT2], width=0.6, ecolor=SUBTEXT)
    axL.set_ylim(0, 109)
    axL.set_ylabel("Mission success (%)")
    axL.set_title("Success collapses without it", color=TEXT, fontweight="bold")
    for b, v in zip(bars, succ):
        axL.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center",
                 color=TEXT, fontweight="bold")

    # right: illegal-action rate is the difference
    ill = [m["illegal_mean"], n["illegal_mean"]]
    ierr = [m["illegal_sd"], n["illegal_sd"]]
    bars = axR.bar(labels, ill, yerr=ierr, capsize=6,
                   color=[ACCENT1, ACCENT2], width=0.6, ecolor=SUBTEXT)
    axR.set_ylim(0, max(ill) * 1.25 + 5)
    axR.set_ylabel("Precondition-violating actions (%)")
    axR.set_title("Legality is not", color=TEXT, fontweight="bold")
    for b, v in zip(bars, ill):
        axR.text(b.get_x() + b.get_width() / 2, v + max(ill) * 0.03 + 0.5,
                 f"{v:.1f}%", ha="center", color=TEXT, fontweight="bold")

    fig.suptitle("Action masking is decisive at v4 scale  (25 techniques, 3 seeds)",
                 fontsize=13, fontweight="bold", color=TEXT)
    fig.tight_layout()
    _save(fig, "thesis_plot_v4_masking_fidelity.png")


# ── 2. technique usage across all 25 (breadth) ──────────────────────────────
def fig_technique_breadth(thorough):
    combined = thorough["combined"]
    order, colors = [], []
    phase_color = {
        "Entry": ACCENT1, "Discovery": "#5b8def", "Cred access": ACCENT2,
        "Lateral": "#22d3ee", "DC dominance": WARN, "Persistence": "#eab308",
        "Host priv-esc": "#f472b6", "Evasion": RED, "Impact": "#a78bfa",
    }
    for ph, techs in PHASES.items():
        for t in techs:
            order.append(t)
            colors.append(phase_color[ph])
    vals = [combined.get(t, 0) for t in order]
    total = sum(vals) or 1
    pct = [100 * v / total for v in vals]

    fig, ax = plt.subplots(figsize=(11, 9))
    y = np.arange(len(order))
    ax.barh(y, pct, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Share of technique selections (%)")
    used = sum(1 for v in vals if v > 0)
    ax.set_title(f"Technique usage across the full ATT&CK kill chain\n"
                 f"v4 adaptive agent exercises {used}/25 techniques",
                 color=TEXT, fontweight="bold")
    # phase legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=ph) for ph, c in phase_color.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.2,
              facecolor=PANEL, edgecolor=GRAY, labelcolor=TEXT)
    for yi, v in zip(y, pct):
        if v > 0:
            ax.text(v + 0.15, yi, f"{v:.1f}", va="center", fontsize=7, color=SUBTEXT)
    _save(fig, "thesis_plot_v4_technique_breadth.png")


# ── 3. scenario adaptation: the new v4 result ───────────────────────────────
def fig_scenario_adaptation(thorough):
    scen = thorough["scenarios"]
    names = list(scen.keys())
    entries = ["SQL_INJECTION", "PHISHING_EMAIL", "PASSWORD_SPRAYING", "BRUTE_FORCE_SSH"]
    # fraction of episodes each entry opened the chain, per scenario
    fracs = {}
    for nm in names:
        e = scen[nm]["entries"]
        tot = sum(e.values()) or 1
        fracs[nm] = {k: 100 * e.get(k, 0) / tot for k in entries}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(entries))
    w = 0.38
    colors = [ACCENT1, ACCENT2]
    for i, nm in enumerate(names):
        vals = [fracs[nm][k] for k in entries]
        b = ax.bar(x + (i - 0.5) * w, vals, w, label=nm, color=colors[i % 2])
        for xi, v in zip(x + (i - 0.5) * w, vals):
            if v > 1:
                ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=8, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("_", "\n") for e in entries], fontsize=9)
    ax.set_ylabel("Entry technique chosen (% of episodes)")
    ax.set_ylim(0, 115)
    div = None
    # recompute divergence from the two scenarios if present
    if len(names) == 2:
        a, b2 = fracs[names[0]], fracs[names[1]]
        div = sum(abs(a[k] - b2[k]) for k in entries) / 100.0
    title = "Scenario adaptation: the agent switches its way in"
    if div is not None:
        title += f"  (entry divergence {div:.2f}/2.00)"
    ax.set_title(title, color=TEXT, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.2, facecolor=PANEL, edgecolor=GRAY, labelcolor=TEXT)
    fig.tight_layout()
    _save(fig, "thesis_plot_v4_scenario_adaptation.png")


# ── 4. sim-to-real transfer under one shared model ──────────────────────────
def fig_transfer(tr):
    sim, live = tr["sim"], tr["live"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
    metrics = ["success", "detection"]
    labels = ["Success", "Detection"]
    smean = [sim["success_mean"], sim["detection_mean"]]
    ssd = [sim["success_sd"], sim["detection_sd"]]
    lmean = [live["success_mean"], live["detection_mean"]]
    lsd = [live["success_sd"], live["detection_sd"]]
    x = np.arange(len(metrics)); w = 0.38
    axL.bar(x - w / 2, smean, w, yerr=ssd, capsize=5, label="Simulation",
            color=ACCENT1, ecolor=SUBTEXT)
    axL.bar(x + w / 2, lmean, w, yerr=lsd, capsize=5, label="Live server (HTTP)",
            color=ACCENT2, ecolor=SUBTEXT)
    axL.set_xticks(x); axL.set_xticklabels(labels)
    axL.set_ylabel("%")
    axL.set_title("Sim vs live", color=TEXT, fontweight="bold")
    axL.legend(fontsize=8, framealpha=0.2, facecolor=PANEL, edgecolor=GRAY, labelcolor=TEXT)
    for xi, v in zip(x - w / 2, smean):
        axL.text(xi, v + 2, f"{v:.1f}", ha="center", fontsize=8, color=TEXT)
    for xi, v in zip(x + w / 2, lmean):
        axL.text(xi, v + 2, f"{v:.1f}", ha="center", fontsize=8, color=TEXT)

    gap = [abs(smean[i] - lmean[i]) for i in range(len(metrics))]
    axR.bar(labels, gap, color=[WARN, RED], width=0.5)
    axR.set_ylabel("Absolute sim-to-real gap (pts)")
    axR.set_ylim(0, max(gap) * 1.6 + 1)
    axR.set_title("Transfer gap is transport-only", color=TEXT, fontweight="bold")
    for i, v in enumerate(gap):
        axR.text(i, v + 0.1, f"{v:.1f}", ha="center", color=TEXT, fontweight="bold")

    fig.suptitle("One shared world model: the agent trained in simulation transfers to the live target",
                 fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    _save(fig, "thesis_plot_v4_transfer.png")


# ── 5. phase coverage radar ─────────────────────────────────────────────────
def fig_phase_coverage(thorough):
    combined = thorough["combined"]
    phases = list(PHASES.keys())
    # coverage = fraction of a phase's techniques the agent actually used
    cover = []
    for ph, techs in PHASES.items():
        used = sum(1 for t in techs if combined.get(t, 0) > 0)
        cover.append(used / len(techs))
    angles = np.linspace(0, 2 * np.pi, len(phases), endpoint=False).tolist()
    cover += cover[:1]; angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor(PANEL)
    ax.plot(angles, cover, color=ACCENT2, linewidth=2)
    ax.fill(angles, cover, color=ACCENT2, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(phases, fontsize=9, color=TEXT)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color=SUBTEXT, fontsize=8)
    ax.set_title("ATT&CK phase coverage — fraction of each phase's techniques used\n"
                 "v4 adaptive agent (every phase exercised)",
                 color=TEXT, fontweight="bold", pad=24)
    _save(fig, "thesis_plot_v4_phase_coverage.png")


# ── 6. rule-based detection: which named rules catch the agent ──────────────
def fig_detection_rules(res):
    # aggregate the named rules that fired across the masked seeds
    rules = {}
    for run in res["masked"].get("per_seed", []):
        for rule, c in run.get("top_rules", {}).items():
            rules[rule] = rules.get(rule, 0) + c
    if not rules:
        print("  (skipping detection-rules figure: no per-seed top_rules in results)")
        return
    def _short(s, n=52):
        s = s.split("(")[0].strip()          # drop the trailing "(Txxxx)"
        if len(s) <= n:
            return s
        return s[:s.rfind(" ", 0, n)].rstrip(" ,;") + "…"   # cut at a word boundary
    items = sorted(rules.items(), key=lambda kv: kv[1])[-10:]
    labels = [_short(k) for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(12.5, 6))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=WARN)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Times the rule escalated (masked agent, 3 seeds)")
    ax.set_title("Detection is explainable: named blue-team rules, real Windows/Sysmon event IDs",
                 color=TEXT, fontweight="bold")
    fig.tight_layout()
    _save(fig, "thesis_plot_v4_detection_rules.png")


# ── 7. multi-metric masked vs nomask ────────────────────────────────────────
def fig_metric_comparison(res):
    m, n = res["masked"], res["nomask"]
    metrics = [("success_mean", "Success %", 100),
               ("detection_mean", "Detection %", None),
               ("illegal_mean", "Illegal %", None),
               ("reward_mean", "Mean reward", None)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, (key, label, top) in zip(axes, metrics):
        vals = [m[key], n[key]]
        bars = ax.bar(["Masked", "Unmasked"], vals, color=[ACCENT1, ACCENT2], width=0.6)
        ax.set_title(label, color=TEXT, fontweight="bold", fontsize=11)
        # give every subplot headroom above the tallest bar so the value label
        # sits inside the axes instead of overflowing into the title
        hi = max(vals) if max(vals) > 0 else 1
        ax.set_ylim(0, (top if top else hi) * 1.18)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + hi * 0.03,
                    f"{v:.1f}", ha="center", color=TEXT, fontsize=9, fontweight="bold")
    fig.suptitle("Masked vs unmasked on the v4 environment (25 techniques, 3 seeds)",
                 fontsize=13, fontweight="bold", color=TEXT)
    fig.tight_layout()
    _save(fig, "thesis_plot_v4_metric_comparison.png")


# ── 9. algorithm comparison: masking across PPO and DQN ─────────────────────
def fig_algorithm_comparison(alg):
    """MaskablePPO vs unmasked PPO vs unmasked DQN, at the 40-step budget:
    the masking gap is not specific to PPO."""
    masked = alg["reference_masked"]
    # pick the 40-step, 40-eval rows for PPO and DQN
    def row(name):
        for r in alg["matrix"]:
            if r["algo"] == name and r["train_budget"] == 40 and r["eval_budget"] == 40:
                return r
        return None
    ppo, dqn = row("PPO"), row("DQN")
    groups = ["MaskablePPO\n(masked)", "PPO\n(unmasked)", "DQN\n(unmasked)"]
    succ = [masked["success_mean"], ppo["success_mean"], dqn["success_mean"]]
    ill = [masked["illegal_mean"], ppo["illegal_mean"], dqn["illegal_mean"]]
    colors = [ACCENT1, ACCENT2, "#22d3ee"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
    for ax, vals, title, ylab in ((axL, succ, "Mission success", "Success (%)"),
                                  (axR, ill, "Illegal actions", "Illegal actions (%)")):
        bars = ax.bar(groups, vals, color=colors, width=0.62)
        ax.set_ylim(0, 112)
        ax.set_ylabel(ylab)
        ax.set_title(title, color=TEXT, fontweight="bold")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}", ha="center",
                    color=TEXT, fontweight="bold", fontsize=10)
    fig.suptitle("Masking is not PPO-specific: an unmasked DQN fails just like unmasked PPO",
                 fontsize=12.5, fontweight="bold", color=TEXT)
    fig.tight_layout()
    _save(fig, "thesis_plot_v4_algorithm_comparison.png")


# ── 8. cross-version success progression ────────────────────────────────────
def fig_version_progression(res):
    # v1/v2/v3 from the manuscript version-progression table; v4 = masked, sim
    # (only mission success is comparable across versions -- reward and detection
    # scales changed, and v4 has a realistic detection model)
    v4 = res["masked"]["success_mean"]
    versions = [
        ("v1", "9 actions", 62.0, GRAY),
        ("v2", "15 actions", 94.0, "#b9b3ec"),
        ("v3", "15 actions", 100.0, ACCENT1),
        ("v4", "25 actions", v4, ACCENT2),
    ]
    labels = [v[0] for v in versions]
    vals = [v[2] for v in versions]
    colors = [v[3] for v in versions]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(labels))
    ax.bar(x, vals, width=0.62, color=colors, edgecolor="white", linewidth=1.4, zorder=3)
    ax.plot(x, vals, color=SUBTEXT, lw=1.3, marker="o", markersize=5,
            markerfacecolor="white", markeredgecolor=SUBTEXT, zorder=4)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 2, f"{v:.0f}%" if v == int(v) else f"{v:.1f}%",
                ha="center", fontsize=11, fontweight="bold", color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n{s}" for l, s in zip(labels, [v[1] for v in versions])], fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Mission success rate (%)")
    ax.grid(axis="y", zorder=0)
    ax.set_title("System evolution: mission success across versions",
                 color=TEXT, fontweight="bold", pad=12)
    fig.tight_layout()
    _save(fig, "thesis_plot_v4_version_progression.png")


def main():
    res = _load("v4_results.json")
    thorough = _load("v4_thorough_result.json")
    print("generating v4 thesis figures...")
    fig_masking_fidelity(res)
    fig_metric_comparison(res)
    fig_technique_breadth(thorough)
    fig_scenario_adaptation(thorough)
    fig_phase_coverage(thorough)
    fig_detection_rules(res)
    fig_version_progression(res)
    try:
        alg = _load("v4_algorithm_comparison.json")
        fig_algorithm_comparison(alg)
    except FileNotFoundError:
        print("  (skipping algorithm-comparison figure: no v4_algorithm_comparison.json)")
    try:
        tr = _load("v4_transfer_result.json")
        fig_transfer(tr)
    except FileNotFoundError:
        print("  (skipping transfer figure: run the live transfer eval first)")
    print("done.")


if __name__ == "__main__":
    main()
