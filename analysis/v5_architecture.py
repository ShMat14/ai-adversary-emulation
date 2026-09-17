# -*- coding: utf-8 -*-
"""Figure 1: the v5 platform, drawn to show what the paper actually evaluates.

The figure this replaces was carried over from earlier work. It showed one
shared world model with a simulator and an HTTP target hanging off it, which was
accurate then and is misleading now for two reasons. It does not show the
vulnerable target that Section 4.8 uses to measure calibration, and by drawing
the HTTP target beside the simulator it implies the two can disagree, which is
exactly the confusion that made our first transfer experiment worthless.

So this version separates the three things by what they share with the world
model, because that is the distinction the results turn on:

  shares the model   the Gymnasium environment, and the HTTP target. Neither can
                     disagree with it. The HTTP target therefore tests transport
                     and not transfer.
  shares nothing     the vulnerable web application. It has its own database,
                     files and authentication, so it can disagree, and the seven
                     techniques executed against it are what calibration means.

    python analysis/v5_architecture.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Elsevier: 300 dpi halftone, 500 combination, 1000 line art.
# These are combination art, so 500 binds; 600 leaves margin.
FIG_DPI = 600


def _save_both(fig, path, **kw):
    """Raster at FIG_DPI for the manuscript, vector for production."""
    fig.savefig(path, dpi=FIG_DPI, **kw)
    if path.lower().endswith(".png"):
        fig.savefig(path[:-4] + ".pdf", **kw)

OUT = "results/figures_v5"

INK = "#1d2a35"
MUTED = "#5d6f7e"
SHARED = "#dbe7f4"      # shares the world model
SHARED_E = "#41576c"
INDEP = "#f7dcdc"       # shares nothing
INDEP_E = "#a4444a"
CORE = "#e8eef5"
CORE_E = "#31518f"
ART = "#e4f0e6"
ART_E = "#1d6f4a"


def box(ax, x, y, w, h, face, edge, title, lines, tsize=10.5, lsize=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 facecolor=face, edgecolor=edge, linewidth=1.4, zorder=3))
    ax.text(x + w / 2, y + h - 0.20, title, ha="center", va="top",
            fontsize=tsize, color=INK, weight="semibold", zorder=4)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.46 - i * 0.20, ln, ha="center", va="top",
                fontsize=lsize, color=MUTED, zorder=4)


def arrow(ax, p, q, color=SHARED_E, style="-|>", dashed=False, lw=1.5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12,
                 color=color, linewidth=lw, zorder=2,
                 linestyle=(0, (5, 2.5)) if dashed else "solid"))


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.4, 6.6))

    # ---- the shared world model, centre ------------------------------------
    box(ax, 4.05, 3.55, 4.3, 1.65, CORE, CORE_E,
        "shared world model  (env/kill_chain_v5.py)",
        ["45 ATT&CK techniques x 12 hosts = 540 actions",
         "preconditions, typed credential material, effects",
         "rule-based detection on real event identifiers",
         "written once; nothing below reimplements it"], 11, 8.6)

    # ---- things that share the model ---------------------------------------
    box(ax, 0.35, 1.30, 3.25, 1.55, SHARED, SHARED_E,
        "Gymnasium environment",
        ["computes the mask each step", "assigns the reward",
         "trains and evaluates the policy"])
    box(ax, 4.20, 1.30, 4.0, 1.55, SHARED, SHARED_E,
        "HTTP target  (Target/server_v5.py)",
        ["serves the SAME model per session",
         "cannot disagree with it, so it tests",
         "transport and not transfer  (Sec. 4.8)"])

    # ---- the thing that shares nothing -------------------------------------
    box(ax, 8.60, 1.30, 3.60, 1.55, INDEP, INDEP_E,
        "vulnerable target",
        ["Target/vulnerable_app.py",
         "own database, files, authentication",
         "genuinely vulnerable; shares NO code",
         "7 techniques executed for real  (Sec. 4.8)"])

    # ---- the agent ---------------------------------------------------------
    box(ax, 0.35, 4.05, 3.25, 1.15, "#ffffff", MUTED,
        "MaskablePPO agent",
        ["sees only masked logits", "never addresses a target directly"])

    # ---- the artefact ------------------------------------------------------
    box(ax, 8.70, 3.85, 3.35, 1.35, ART, ART_E,
        "ATT&CK-labelled telemetry",
        ["emitted as each technique executes",
         "label known at emission, not inferred",
         "31,925 events, 45/45 techniques"])

    # ---- edges -------------------------------------------------------------
    arrow(ax, (1.97, 4.05), (1.97, 2.85))                     # agent -> env
    arrow(ax, (1.97, 2.85), (1.97, 4.05))                     # env -> agent
    ax.text(2.10, 3.45, "masked action\nobservation, reward", fontsize=7.8,
            color=MUTED, va="center")

    arrow(ax, (3.60, 2.10), (4.20, 2.10))                     # env -> http
    arrow(ax, (2.30, 2.85), (4.55, 3.55))                     # env -> model
    arrow(ax, (6.20, 2.85), (6.20, 3.55))                     # http -> model
    arrow(ax, (8.35, 4.45), (8.70, 4.45), color=ART_E)        # model -> telemetry

    # the one edge that does not touch the model
    arrow(ax, (8.20, 2.10), (8.60, 2.10), color=INDEP_E, dashed=True)
    ax.text(8.40, 1.05, "executed, not simulated", fontsize=7.6,
            color=INDEP_E, ha="center", style="italic")

    # ---- legend ------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.35, 0.18), 0.30, 0.22,
                 boxstyle="round,pad=0.01", facecolor=SHARED,
                 edgecolor=SHARED_E, linewidth=1.2))
    ax.text(0.78, 0.29, "shares the world model — cannot disagree with it",
            fontsize=8.4, color=MUTED, va="center")
    ax.add_patch(FancyBboxPatch((6.60, 0.18), 0.30, 0.22,
                 boxstyle="round,pad=0.01", facecolor=INDEP,
                 edgecolor=INDEP_E, linewidth=1.2))
    ax.text(7.03, 0.29, "shares nothing — can disagree, and does",
            fontsize=8.4, color=MUTED, va="center")

    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    fig.tight_layout()
    path = os.path.join(OUT, "architecture.png")
    _save_both(fig, path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
