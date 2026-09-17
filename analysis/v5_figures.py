# -*- coding: utf-8 -*-
"""Figures for the v5 experiments.

Design decisions here are deliberate responses to how the two closest papers
present their results.

Koo et al. (E-NASim, ETRI Journal 2026, Fig. 6) plot episode length and
cumulative return only. Their v2 return panel spans 340 to 348 -- a 2.3% change
rendered to fill the frame -- and no figure in the paper states what fraction of
episodes reached the objective. Reward is not self-interpreting; success is.

Zhan et al. (L-ARLPT, Applied Sciences 2026, Fig. 6) do report variance bands
over ten runs, which is good practice and matched here. But they give each method
its own panel with its own y-axis, so a reader cannot compare them visually: the
best and worst methods differ by roughly 1800 reward while occupying identically
shaped frames.

So: one shared axis, every configuration on it, success rate on a full 0-100%
scale, with a mean and a variance band across seeds. Plus the panel neither paper
has -- the proportion of proposed actions the environment refuses.
"""
import glob
import re
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Elsevier: 300 dpi halftone, 500 combination, 1000 line art.
# These are combination art, so 500 binds; 600 leaves margin.
FIG_DPI = 600


def _save_both(fig, path, **kw):
    """Raster at FIG_DPI for the manuscript, vector for production."""
    fig.savefig(path, dpi=FIG_DPI, **kw)
    if path.lower().endswith(".png"):
        fig.savefig(path[:-4] + ".pdf", **kw)

CURVES = "analysis/v5_curves"
FIGDIR = "results/figures_v5"

LABELS = {
    "masked":        "MaskablePPO (prerequisite mask)",
    "nomask":        "PPO, unmasked",
    "nomask_shaped": "PPO + invalid-action penalty",
    "dqn":           "DQN, unmasked",
    "a2c":           "A2C, unmasked",
}
COLORS = {"masked": "#1b6ca8", "nomask": "#d1495b",
          "nomask_shaped": "#a13d4c", "dqn": "#7d4f9c", "a2c": "#e08b30"}
ORDER = ["masked", "nomask", "nomask_shaped", "dqn", "a2c"]
# the reward-shaped baseline is a nomask run carrying its own suffix
SUFFIX_OF = {"nomask_shaped": "_shaped"}
BASE_OF = {"nomask_shaped": "nomask"}


def load_curves(topology="enterprise", suffix="", max_steps=None):
    """config -> {timesteps: [...], success: 2-D [seed, point], ...}

    `max_steps` filters to runs sharing one episode budget. Mixing budgets in a
    single band would be meaningless -- success rate is not comparable across
    them -- and an early run of this study did exactly that before the guard was
    added.
    """
    out = {}
    for cfg in ORDER:
        base = BASE_OF.get(cfg, cfg)
        sfx = suffix + SUFFIX_OF.get(cfg, "")
        # Match the tag exactly. A bare glob on "nomask_enterprise_s*.json"
        # also matches "nomask_enterprise_s0_shaped.json", which would average
        # the unmasked baseline together with the reward-shaped one and hide the
        # difference between them in a single band.
        want = re.compile(rf"^{re.escape(base)}_{re.escape(topology)}_s\d+"
                          rf"{re.escape(sfx)}\.json$")
        files = sorted(f for f in glob.glob(os.path.join(CURVES, "*.json"))
                       if want.match(os.path.basename(f)))
        runs = []
        for f in files:
            with open(f) as fh:
                d = json.load(fh)
            if max_steps is not None and d.get("max_steps") != max_steps:
                continue
            if d.get("points"):
                runs.append(d["points"])
        if not runs:
            continue
        n = min(len(r) for r in runs)
        out[cfg] = {
            "timesteps": [p["timesteps"] for p in runs[0][:n]],
            "seeds": len(runs),
            "success": np.array([[p["success"] for p in r[:n]] for r in runs]),
            "illegal": np.array([[p["illegal"] for p in r[:n]] for r in runs]),
            "detection": np.array([[p["detection"] for p in r[:n]] for r in runs]),
            "ep_len": np.array([[p["ep_len"] for p in r[:n]] for r in runs]),
        }
    return out


def _band(ax, x, arr, color, label):
    """Mean line with a +/-1 SD band and a fainter min-max band."""
    mean, sd = arr.mean(axis=0), arr.std(axis=0)
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0)
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.25, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=2.0, label=label)


def figure_learning(curves, path):
    """Success and refused-action rate, all configurations on shared axes."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for cfg in ORDER:
        if cfg not in curves:
            continue
        c = curves[cfg]
        x = np.array(c["timesteps"]) / 1000.0
        lbl = f"{LABELS[cfg]} (n={c['seeds']})"
        _band(axes[0], x, c["success"], COLORS[cfg], lbl)
        _band(axes[1], x, c["illegal"], COLORS[cfg], lbl)

    axes[0].set_ylabel("Episodes reaching the objective (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("(a) Mission success", loc="left", fontsize=11)
    axes[1].set_ylabel("Proposed actions refused by the environment (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("(b) Infeasible action selection", loc="left", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Training timesteps (thousands)")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    _save_both(fig, path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_cost_of_success(curves, path):
    """What the campaign cost: detection and episode length."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for cfg in ORDER:
        if cfg not in curves:
            continue
        c = curves[cfg]
        x = np.array(c["timesteps"]) / 1000.0
        _band(axes[0], x, c["detection"], COLORS[cfg], LABELS[cfg])
        _band(axes[1], x, c["ep_len"], COLORS[cfg], LABELS[cfg])
    axes[0].set_ylabel("Episodes ending in incident response (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("(a) Exposure", loc="left", fontsize=11)
    axes[1].set_ylabel("Mean episode length (steps)")
    axes[1].set_title("(b) Campaign length", loc="left", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Training timesteps (thousands)")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    _save_both(fig, path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_topology(path, topology="enterprise"):
    """The estate as security zones, with reachability drawn between them.

    Layout rules that matter: arrows sit in the gaps between zone boxes at the
    vertical centre, so a connection is never mistaken for a wire threading
    through a machine; and the workstation-to-admin shortcut is routed beneath
    the zones, where it cannot collide with the main path or its own label.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from env.topology_v5 import Topology
    topo = Topology(topology)

    zones = {0: "external", 1: "DMZ", 2: "workstations", 3: "servers",
             4: "admin tier", 5: "domain core"}
    cols = [0] + sorted(topo.subnets)
    nmax = max(len(v) for v in topo.subnets.values())

    HW, HH = 0.62, 0.40          # host box
    GAP = 0.60                   # vertical spacing between hosts
    ZW = 0.82                     # zone box width
    half = (nmax - 1) * GAP / 2 + HH / 2 + 0.28
    ytop, ybot = half, -half

    fig, ax = plt.subplots(figsize=(12.0, 5.0))

    def host_ys(n):
        return [(n - 1) * GAP / 2 - i * GAP for i in range(n)]

    for s in cols:
        # zone container, sized to the tallest column so all zones align
        ax.add_patch(FancyBboxPatch(
            (s - ZW / 2, ybot), ZW, ytop - ybot,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor="#f5f7f9", edgecolor="#d9e1e8", linewidth=1.0, zorder=0))
        ax.text(s, ybot - 0.30, zones.get(s, f"subnet {s}"), ha="center",
                va="top", fontsize=10, color="#33475b")

        members = ["attacker"] if s == 0 else topo.subnets[s]
        for y, h in zip(host_ys(len(members)), members):
            external = (s == 0)
            win = (not external) and topo.is_windows(h)
            face = "#f7dcdc" if external else ("#dbe7f4" if win else "#f7ead6")
            edge = "#a4444a" if external else "#41576c"
            ax.add_patch(FancyBboxPatch(
                (s - HW / 2, y - HH / 2), HW, HH,
                boxstyle="round,pad=0.01,rounding_size=0.04",
                facecolor=face, edgecolor=edge, linewidth=1.1, zorder=3))
            ax.text(s, y + 0.055, h, ha="center", va="center", fontsize=8.8, zorder=4)
            if not external:
                ax.text(s, y - 0.115, "Windows" if win else "Linux", ha="center",
                        va="center", fontsize=6.5, color="#5d6f7e", zorder=4)

    # adjacent reachability: arrows in the gaps, at the vertical centre
    for a in range(0, max(cols)):
        ax.add_patch(FancyArrowPatch(
            (a + ZW / 2 + 0.03, 0), (a + 1 - ZW / 2 - 0.03, 0),
            arrowstyle="-|>", mutation_scale=13, color="#7f96a9",
            linewidth=1.6, zorder=2))

    # the one non-adjacent edge, routed below so nothing overlaps
    ax.add_patch(FancyArrowPatch(
        (2, ybot - 0.02), (4, ybot - 0.02),
        connectionstyle="arc3,rad=0.32", arrowstyle="-|>", mutation_scale=13,
        color="#c07f45", linewidth=1.6, linestyle=(0, (5, 2.5)), zorder=2))
    ax.text(3, ybot - 1.02, "workstations also reach the admin tier directly:\n"
                            "the route to the domain core is a choice, not a chain",
            ha="center", va="top", fontsize=8, color="#a06a3f", style="italic")

    ax.set_xlim(-0.75, max(cols) + 0.75)
    ax.set_ylim(ybot - 1.85, ytop + 0.35)
    ax.axis("off")
    fig.tight_layout()
    _save_both(fig, path, bbox_inches="tight")
    plt.close(fig)
    return path


def main(topo="enterprise"):
    """Render every figure for one topology.

    Filenames carry the topology for anything but the enterprise estate. Without
    that, rendering the three-host network second silently overwrote the
    twelve-host figures with the three-host ones, and the paper would have
    carried a learning curve labelled 540 actions that was drawn from 135.
    """
    os.makedirs(FIGDIR, exist_ok=True)
    sfx = "" if topo == "enterprise" else f"_{topo}"
    made = [figure_topology(os.path.join(FIGDIR, f"topology{sfx}.png"), topo)]
    from analysis.v5_train import DEFAULT_MAX_STEPS
    curves = load_curves(topo, max_steps=DEFAULT_MAX_STEPS)
    if curves:
        made.append(figure_learning(curves, os.path.join(FIGDIR, f"learning{sfx}.png")))
        made.append(figure_cost_of_success(
            curves, os.path.join(FIGDIR, f"cost{sfx}.png")))
        for cfg, c in curves.items():
            print(f"  {cfg:8s} {c['seeds']} seed(s), "
                  f"{len(c['timesteps'])} points, final success "
                  f"{c['success'][:, -1].mean():.1f}%")
    else:
        print("  no curves yet in", CURVES)
    for m in made:
        print("  wrote", m)
    return made


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "enterprise")
