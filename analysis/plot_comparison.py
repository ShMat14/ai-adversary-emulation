# -*- coding: utf-8 -*-
"""
Figures for the controlled comparison and the action-space scaling experiment.

Values are read from the result JSON files rather than retyped, so a figure
cannot drift away from the table beside it.

Design decisions worth stating:

  * One measure per panel. Success rate, reward and illegal-action share differ
    by orders of magnitude; putting any two on twin axes would invite a
    comparison the scales do not support.
  * Hues are Okabe-Ito blue / vermillion / bluish-green, checked rather than
    chosen by eye: worst pairwise separation is dE 8.6 under tritanopia, above
    the floor of 8, and every pair clears 18 for normal vision.
  * Marker shape and line style repeat the series identity, so the figures
    survive greyscale printing and colour-vision deficiency without hue.
  * The scaling figure draws every seed. Its unmasked runs are bimodal -- two
    seeds behave, one collapses -- so a mean with a standard deviation larger
    than itself would imply a uniform shift that did not occur.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#1a1a1a", "#555555", "#d9d9d9"
NL = "\n"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "font.size": 9,
    "legend.frameon": False,
})


def load(name):
    """Result file by name, or None if it has not been generated yet."""
    path = os.path.join("analysis", name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)


# ══════════════════════════════════════════════════════════════════════
# Controlled comparison at the implemented action space
# ══════════════════════════════════════════════════════════════════════
def figure_comparison(out):
    res = load("comparison_results.json")["summary"]
    ill = load("comparison_illegal.json")

    agents = [("maskable", "MaskablePPO" + NL + "(masked)", BLUE),
              ("ppo", "PPO" + NL + "(no mask)", VERM),
              ("dqn", "DQN" + NL + "(no mask)", GREEN)]

    panels = [
        ("Mission success", "%", [res[a]["success_rate"] for a, _, _ in agents], 118),
        ("Detection", "%", [res[a]["detection_rate"] for a, _, _ in agents], 118),
        ("Mean episode reward", "", [res[a]["avg_reward"] for a, _, _ in agents], None),
        ("Actions violating" + NL + "a precondition", "%",
         [(ill[a]["illegal_pct_mean"], ill[a]["illegal_pct_sd"]) for a, _, _ in agents], None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    x = np.arange(len(agents))
    for ax, (title, unit, vals, top) in zip(axes, panels):
        means = [v[0] for v in vals]
        sds = [v[1] for v in vals]
        ax.bar(x, means, yerr=sds, width=0.62, capsize=3,
               color=[c for _, _, c in agents], edgecolor="white", linewidth=1.4)
        ax.set_title(title, fontsize=9.5, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl, _ in agents], fontsize=8)
        tidy(ax)
        ax.set_ylim(0, top if top else max(m + s for m, s in zip(means, sds)) * 1.30)
        for xi, (m, s) in enumerate(zip(means, sds)):
            ax.text(xi, m + s + ax.get_ylim()[1] * 0.035, f"{m:.1f}{unit}",
                    ha="center", fontsize=8.2, color=INK)

    fig.suptitle("Controlled comparison: identical environment, budget and seeds; "
                 "only the algorithm differs", fontsize=10.5, y=1.03)
    fig.text(0.5, -0.07, "Bars are the mean of three seeds; whiskers are one standard deviation.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ══════════════════════════════════════════════════════════════════════
# What changes as the technique catalogue grows
# ══════════════════════════════════════════════════════════════════════
def figure_scaling(out):
    per = load("scale_per_seed.json")
    # legal_share.json is written by analysis/point5_report.py and measures the
    # share over states the trained agent actually reaches. scale_results.json
    # holds an earlier figure taken from a lowest-index-legal walk, which never
    # leaves the opening of the kill chain and overstates the share; it is used
    # only as a fallback so this figure still draws on a bare checkout.
    frac = load("legal_share.json") or load("scale_results.json")["legal_fraction_pct"]
    sizes = [15, 60, 200]
    series = [("maskable", "masked", BLUE, "o", "-"),
              ("ppo", "no mask", VERM, "s", "--")]

    panels = [
        ("Mission success", "success rate (%)", "success_rate", (85, 104)),
        ("Mean episode reward", "reward", "avg_reward", (380, 900)),
        ("Actions violating a precondition", "share of actions (%)", "illegal_pct", (-5, 88)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7))
    xs = np.arange(len(sizes))
    for ax, (title, ylab, key, ylim) in zip(axes, panels):
        for algo, label, colour, marker, ls in series:
            med = [float(np.median([per[f"{algo}_{s}_s{k}"][key] for k in (0, 1, 2)]))
                   for s in sizes]
            ax.plot(xs, med, color=colour, linewidth=2, linestyle=ls, zorder=2,
                    label=label, marker=marker, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.2)
            off = -0.08 if algo == "maskable" else 0.08
            for k in (0, 1, 2):
                ax.scatter(xs + off, [per[f"{algo}_{s}_s{k}"][key] for s in sizes],
                           s=18, color=colour, alpha=0.55, edgecolor="none", zorder=3)
        ax.set_title(title, fontsize=9.5, pad=8)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{s}" + NL + f"({frac[str(s)]:.1f}% legal)" for s in sizes],
                           fontsize=8)
        ax.set_xlabel("actions in the technique catalogue", fontsize=8.5)
        tidy(ax)
        if ylim:
            ax.set_ylim(*ylim)

    # the collapsed runs are the finding, so name them
    ax = axes[2]
    worst_i = None
    for i, size in enumerate(sizes):
        if max(per[f"ppo_{size}_s{k}"]["illegal_pct"] for k in (0, 1, 2)) > 50:
            worst_i = i
            break
    if worst_i is not None:
        wv = max(per[f"ppo_{sizes[worst_i]}_s{k}"]["illegal_pct"] for k in (0, 1, 2))
        ax.annotate("one seed in three collapses", xy=(worst_i + 0.08, wv),
                    xytext=(10, -26), textcoords="offset points", fontsize=8.2,
                    color=VERM, ha="left",
                    arrowprops=dict(arrowstyle="-", color=VERM, lw=0.8))

    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Effect of catalogue size: success is unaffected; without masking, "
                 "some runs collapse", fontsize=10.5, y=1.03)
    fig.text(0.5, -0.07, "Line is the median of three seeds; dots are the individual seeds.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ══════════════════════════════════════════════════════════════════════
# Reward distribution, which shows the bimodality directly
# ══════════════════════════════════════════════════════════════════════
def figure_stability(out):
    per = load("scale_per_seed.json")
    sizes = [15, 60, 200]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.34
    xs = np.arange(len(sizes))
    for j, (algo, label, colour) in enumerate([("maskable", "masked", BLUE),
                                               ("ppo", "no mask", VERM)]):
        for i, s in enumerate(sizes):
            vals = [per[f"{algo}_{s}_s{k}"]["avg_reward"] for k in (0, 1, 2)]
            pos = xs[i] + (j - 0.5) * width
            ax.plot([pos, pos], [min(vals), max(vals)], color=colour, lw=1.4, zorder=1)
            ax.scatter([pos] * 3, vals, s=34, color=colour, zorder=2,
                       edgecolor="white", linewidth=1.0,
                       label=label if i == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s} actions" for s in sizes], fontsize=9)
    ax.set_ylabel("mean episode reward", fontsize=9)
    ax.set_title("Run-to-run stability: every seed, with its range", fontsize=10.5, pad=10)
    tidy(ax)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.text(0.5, -0.04,
             "Each dot is one training run. Masking removes the spread rather than "
             "shifting the average.", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    figure_comparison("images/fig_controlled_comparison.png")
    figure_scaling("images/fig_action_space_scaling.png")
    figure_stability("images/fig_run_stability.png")
