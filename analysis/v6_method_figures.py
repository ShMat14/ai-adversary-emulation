"""Method figures for Section 3.

The JNCA corpus profile (analysis/jnca_style_profile.py) shows the journal's
median paper carries 12 figures against 8 tables, and its two reinforcement
learning papers run at 77-80% figures. Ours was 7 against 14. The deficit was
not in the results, which Figs 3-7 cover, but in the method: the corpus papers
draw the artefact before reporting on it, and Section 3 described the mask, the
detection model and the corpus almost entirely in prose.

These five close that gap. Three are drawn from the released corpus and the
catalogue rather than composed by hand, so they report rather than illustrate.

    python analysis/v6_method_figures.py

Writes results/figures_v5/{mask,detection,corpus,killchain,scenario}.{png,pdf}
"""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures_v5"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, RED, GREEN, ORANGE, GREY = "#2b6cb0", "#c0392b", "#2e7d5b", "#d68910", "#4a5568"
LIGHT, PALE = "#dce6f2", "#f4f6f8"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "legend.fontsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "pdf.fonttype": 42,  # TrueType, not Type 3, in the PDFs
})

# Elsevier: 300 dpi halftone, 500 combination, 1000 line art. These are
# combination art, so 500 binds and 600 leaves margin.
DPI = 600


def save(fig, name):
    w, small = printed_size_check(fig, name)
    print(f"  {name}: {w:.2f} in, smallest label {small:.1f} pt")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def box(ax, x, y, w, h, text, fc=LIGHT, ec=BLUE, fs=8, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, linespacing=1.45)


def arrow(ax, p, q, color=GREY, style="-|>", lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=2, shrinkB=2))


def printed_size_check(fig, name, width_in=5.8, min_pt=7.0):
    """Refuse a chart whose smallest label would print below min_pt.

    Word places every figure at the text width, 5.8 in. A chart drawn wider
    is shrunk, and its labels with it; that is how Figs. 8 and 9 came to print
    at half size.
    """
    from matplotlib.text import Text
    # Lay everything out first; before a draw, legend entries still sit at
    # their default position and every one of them appears to overlap.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w = fig.get_tightbbox(r).width
    scale = min(1.0, width_in / w)
    sizes = [t.get_fontsize() for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    small = min(sizes) * scale
    # Labels that run into one another (Fig. 6, panel c, before this check).
    # Slanted tick labels are skipped: their axis-aligned boxes overlap even
    # when the text does not, so the test would only report false positives.
    # Tick labels for ticks outside the view limits exist but are never drawn.
    hidden = set()
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    hidden.update({id(tick.label1), id(tick.label2)})
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
             and t.get_rotation() % 90 == 0 and id(t) not in hidden]
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    for i, (ta, a) in enumerate(boxes):
        for tb, b in boxes[i + 1:]:
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 1.5 and oy > 1.5:
                raise SystemExit(f"{name}: labels overlap: {ta.get_text()!r} / {tb.get_text()!r}")
    if small < min_pt - 0.05:
        raise SystemExit(f"{name}: {w:.2f} in wide, smallest label prints at "
                         f"{small:.1f} pt (< {min_pt})")
    return w, small


def blank(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    return fig, ax


# ------------------------------------------------------------------ 1. the mask
def figure_mask():
    """What the environment does between observing a state and the agent acting.

    The contrast at the bottom is the point of the paper: the same illegal
    action is unreachable under a mask and merely discouraged under shaping.
    """
    fig, ax = blank((7.2, 3.5))

    box(ax, 0.01, 0.56, 0.15, 0.22, "state $s$\nand scenario $\\omega$", PALE, GREY)
    box(ax, 0.20, 0.56, 0.21, 0.22,
        "evaluate preconditions\nfor all 540 actions\n(45 techniques × 12 hosts)")
    box(ax, 0.45, 0.56, 0.17, 0.22, "mask $\\mu(s,\\omega)$\nbit per action")
    box(ax, 0.66, 0.56, 0.15, 0.22, "policy logits\n$\\rightarrow -\\infty$\nwhere masked")
    box(ax, 0.85, 0.56, 0.14, 0.22, "action\nsampled", "#e8f3ee", GREEN)

    for a, b in (((0.16, 0.67), (0.19, 0.67)), ((0.44, 0.67), (0.47, 0.67)),
                 ((0.63, 0.67), (0.66, 0.67)), ((0.81, 0.67), (0.85, 0.67))):
        arrow(ax, a, b)

    ax.text(0.315, 0.50, "computed every step, never learned",
            ha="center", fontsize=7.5, style="italic", color=GREY)

    # The two mechanisms, side by side.
    ax.plot([0.01, 0.99], [0.42, 0.42], color="#d8dee4", lw=0.8)
    ax.text(0.01, 0.345, "An action whose preconditions do not hold:",
            fontsize=8.5, weight="bold")

    box(ax, 0.04, 0.10, 0.40, 0.19,
        "under the prerequisite mask\nreceives no probability mass and\n"
        "cannot be proposed at all", "#e8f3ee", GREEN, 8)
    box(ax, 0.54, 0.10, 0.42, 0.19,
        "under reward shaping\nis proposed, executed, refused, and\n"
        "discouraged by a penalty afterwards", "#fbeceb", RED, 8)
    ax.text(0.24, 0.045, "0.0% of actions selected", ha="center", fontsize=8,
            color=GREEN, weight="bold")
    ax.text(0.75, 0.045, "10.3% after training, from 99%", ha="center",
            fontsize=8, color=RED, weight="bold")
    save(fig, "mask")


# --------------------------------------------------------------- 2. detection
def figure_detection():
    """How a technique becomes an alert, and how an engagement ends."""
    fig, ax = blank((7.2, 2.9))

    box(ax, 0.01, 0.60, 0.16, 0.26, "technique\nexecutes\non a host", PALE, GREY)
    box(ax, 0.21, 0.60, 0.20, 0.26,
        "emits the records it\nwould really leave\n(9 channels)")
    box(ax, 0.45, 0.60, 0.18, 0.26, "a named rule\nmatches the\nevent identifier")
    box(ax, 0.67, 0.60, 0.15, 0.26, "suspicion\naccumulates")
    box(ax, 0.86, 0.60, 0.13, 0.26, "incident\ndeclared", "#fbeceb", RED)

    for a, b in (((0.17, 0.73), (0.21, 0.73)), ((0.41, 0.73), (0.45, 0.73)),
                 ((0.63, 0.73), (0.67, 0.73)), ((0.82, 0.73), (0.86, 0.73))):
        arrow(ax, a, b)
    ax.text(0.925, 0.55, "ends the engagement", ha="center", fontsize=7.5,
            style="italic", color=RED)

    ax.text(0.01, 0.44, "Platform correctness is enforced, not assumed:",
            fontsize=8.5, weight="bold")
    box(ax, 0.04, 0.12, 0.42, 0.24,
        "Windows hosts\nSecurity, Sysmon, PowerShell,\nTerminal Services, WMI-Activity", LIGHT, BLUE, 8)
    box(ax, 0.54, 0.12, 0.42, 0.24,
        "Linux hosts\nauditd, syslog\n(0 Windows identifiers on Linux)", "#e8f3ee", GREEN, 8)
    save(fig, "detection")


# ------------------------------------------------------------------ 3. corpus
def figure_corpus():
    """The released corpus described by its own contents.

    Everything here is counted from results/telemetry/v5/, not asserted.
    """
    man = json.loads((RESULTS / "telemetry/v5/manifest.json").read_text())
    tactic, channel, platform = (collections.Counter() for _ in range(3))
    with open(RESULTS / "telemetry/v5/telemetry.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tactic[row["tactic"]] += 1
            channel[row["channel"]] += 1
            platform[row["platform"]] += 1

    fig, axes = plt.subplots(1, 3, figsize=(5.8, 2.6),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 0.8]})

    items = tactic.most_common()
    y = np.arange(len(items))
    axes[0].barh(y, [c for _, c in items], color=BLUE)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([t for t, _ in items], fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("records")
    axes[0].set_title("(a) By ATT&CK tactic", loc="left")

    ch = channel.most_common()
    y = np.arange(len(ch))
    colours = [BLUE if c not in ("auditd", "syslog") else GREEN for c, _ in ch]
    axes[1].barh(y, [c for _, c in ch], color=colours)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([c for c, _ in ch], fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("records")
    axes[1].set_title("(b) By log channel", loc="left")

    reached = man["objective_reached"]
    incident = man["incidents_declared"]
    other = man["episodes"] - reached - incident
    # Horizontal like (a) and (b): as vertical bars the three category names
    # ran into one another at print width.
    outcome = [("objective reached", reached, GREEN), ("incident declared", incident, RED),
               ("neither", other, GREY)]
    y = np.arange(len(outcome))
    axes[2].barh(y, [v for _, v, _ in outcome], color=[c for _, _, c in outcome])
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([n for n, _, _ in outcome], fontsize=7)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("episodes")
    axes[2].set_title("(c) By outcome", loc="left")
    axes[2].set_xlim(0, max(v for _, v, _ in outcome) * 1.35)
    for yi, (_, v, _) in zip(y, outcome):
        axes[2].text(v + 8, yi, str(v), va="center", fontsize=7, color=GREY)

    fig.suptitle(
        f"{man['events']:,} records over {man['episodes']} episodes, "
        f"{man['distinct_techniques']}/45 techniques and "
        f"{man['distinct_attack_ids']}/45 ATT&CK identifiers",
        fontsize=8, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "corpus")


# --------------------------------------------------------------- 4. kill chain
def figure_killchain():
    """The catalogue as a chain: how many techniques serve each tactic, and
    when a rule first has anything to fire on."""
    cov = json.loads((RESULTS / "v5_coverage.json").read_text())
    first = {r["tactic"]: r["median_step"] for r in cov["per_tactic"]}

    # Technique counts per tactic, from the catalogue in Section 3.6.
    catalogue = {
        "Initial Access": 3, "Discovery": 6, "Credential Access": 9,
        "Privilege Escalation": 5, "Execution": 2, "Lateral Movement": 4,
        "Persistence": 4, "Defense Evasion": 5, "Command and Control": 1,
        "Collection": 2, "Exfiltration": 1, "Impact": 3,
    }
    order = list(catalogue)
    counts = [catalogue[t] for t in order]
    steps = [first.get(t, np.nan) for t in order]

    fig, ax1 = plt.subplots(figsize=(5.8, 2.9))
    x = np.arange(len(order))
    ax1.bar(x, counts, color=LIGHT, edgecolor=BLUE, lw=1.0, width=0.66)
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=38, ha="right", fontsize=7.5)
    ax1.set_ylabel("techniques in the catalogue", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_ylim(0, max(counts) * 1.35)
    for xi, c in zip(x, counts):
        ax1.text(xi, c + 0.15, str(c), ha="center", fontsize=7.5, color=BLUE)

    ax2 = ax1.twinx()
    ax2.plot(x, steps, "o-", color=RED, lw=1.4, ms=4.5)
    ax2.set_ylabel("median step of first detection", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.grid(False)
    ax2.set_ylim(0, max(s for s in steps if s == s) * 1.25)

    ax1.set_title("Catalogue coverage against detection latency, by tactic",
                  loc="left")
    save(fig, "killchain")


# ------------------------------------------------------------- 5. scenario (w)
def figure_scenario():
    """Why the mask is a function of the scenario and not of the state alone."""
    fig, ax = blank((7.2, 2.7))

    box(ax, 0.01, 0.58, 0.19, 0.28,
        "engagement drawn\nat episode start", PALE, GREY)
    box(ax, 0.25, 0.58, 0.25, 0.28,
        "a technique is withdrawn\nfrom each substitution group\nand extra requirements set")
    box(ax, 0.55, 0.58, 0.19, 0.28, "scenario $\\omega$\nfixed for the episode")
    box(ax, 0.79, 0.58, 0.20, 0.28,
        "mask becomes\n$\\mu(s,\\omega)$,\nnot $\\mu(s)$", "#e8f3ee", GREEN)
    for a, b in (((0.20, 0.72), (0.23, 0.72)), ((0.53, 0.72), (0.56, 0.72)),
                 ((0.74, 0.72), (0.78, 0.72))):
        arrow(ax, a, b)

    ax.text(0.01, 0.45, "Why this matters for the ablation:",
            fontsize=8.5, weight="bold")
    box(ax, 0.04, 0.09, 0.42, 0.28,
        "legality a fixed function of state\n$\\mu(s)$\n"
        "an agent can learn it from experience\nmasked 99.7% vs unmasked 99.8%",
        "#fbeceb", RED, 8)
    box(ax, 0.54, 0.09, 0.42, 0.28,
        "legality depends on the engagement\n$\\mu(s,\\omega)$\n"
        "there is nothing stable to learn\nmasked 69.7% vs unmasked 0.0%",
        "#e8f3ee", GREEN, 8)
    save(fig, "scenario")


if __name__ == "__main__":
    print("method figures for Section 3\n")
    # The mask, detection and scenario diagrams are drawn by v7_diagrams.py,
    # which measures every label against its box. The versions in this file
    # overflowed their boxes and are kept only for the record; running them
    # would overwrite the corrected figures.
    figure_corpus()
    figure_killchain()
    print(f"\nwritten to {OUT}")
