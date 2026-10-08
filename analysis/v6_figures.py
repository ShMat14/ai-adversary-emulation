"""Figures added for the v6 (article-format) manuscript.

Three charts that let dense tables come out of the main text, drawn from the
same JSON the tables were built from. Elsevier wants combination art at 500 dpi
or better, so every figure is written twice: 600 dpi PNG and vector PDF.

    python analysis/v6_figures.py

Writes results/figures_v5/{comparison,latency,transfer}.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures_v5"
OUT.mkdir(parents=True, exist_ok=True)

# One palette for the whole paper. Masked configurations are blue, unmasked are
# warm, so a reader can tell the two families apart without reading the legend.
BLUE, RED, PURPLE, ORANGE, DARKRED = (
    "#2b6cb0",
    "#c0392b",
    "#6b46c1",
    "#d68910",
    "#7b241c",
)
GREY = "#4a5568"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,  # TrueType, not Type 3, in the PDFs
    }
)


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


def save(fig, name: str) -> None:
    w, small = printed_size_check(fig, name)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png / .pdf  ({w:.2f} in, smallest label {small:.1f} pt)")


def load(name: str):
    return json.loads((RESULTS / name).read_text())


# --------------------------------------------------------------- Figure: comparison
def figure_comparison() -> None:
    """Table 6 as a chart: three masked/unmasked pairs across two families.

    Success alone hides the finding. Infeasible selection separates masked from
    unmasked, success separates the families, and exposure separates all three
    masked configurations even where success does not -- so all three are drawn.
    """
    main = load("v5_report.json")["main"]
    mdqn = load("v5_maskdqn.json")
    ma2c = load("v5_maska2c.json")

    def row(d):
        """Pull one row from either JSON shape.

        v5_report.json flattens its fields (`success_mean`); the two maskable
        variants were written later with a nested `mean`/`sd` block.
        """
        if "success_mean" in d:
            return (
                d["success_mean"],
                d.get("success_sd", 0.0),
                d["illegal_mean"],
                d["detection_mean"],
            )
        return (
            d["mean"]["success"],
            d["sd"]["success"],
            d["mean"]["illegal"],
            d["mean"]["detection"],
        )

    configs = [
        ("MaskablePPO", row(main["masked"]), True),
        ("PPO", row(main["nomask"]), False),
        ("MaskableA2C", row(ma2c), True),
        ("A2C", row(main["a2c"]), False),
        ("MaskableDQN", row(mdqn), True),
        ("DQN", row(main["dqn"]), False),
    ]

    labels = [c[0] for c in configs]
    success = np.array([c[1][0] for c in configs])
    sd = np.array([c[1][1] for c in configs])
    illegal = np.array([c[1][2] for c in configs])
    detected = np.array([c[1][3] for c in configs])
    masked = [c[2] for c in configs]

    # Shared y axis: all three panels are percentages of engagements, so one
    # scale and one label keep the titles from colliding at column width.
    fig, axes = plt.subplots(1, 3, figsize=(5.8, 2.7), sharey=True)
    x = np.arange(len(labels))
    colours = [BLUE if m else RED for m in masked]

    for ax, vals, err, title in (
        (axes[0], success, sd, "(a) Mission success"),
        (axes[1], illegal, None, "(b) Infeasible actions"),
        (axes[2], detected, None, "(c) Exposure"),
    ):
        ax.bar(x, vals, color=colours, yerr=err, capsize=2.5, error_kw={"lw": 0.8})
        ax.set_title(title, loc="left", pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("% of engagements")

    # The zero bars are the result in (b); label them so they are not read as
    # missing data.
    for i, v in enumerate(illegal):
        if v < 1.0:
            axes[1].text(i, 2.0, "0.0", ha="center", fontsize=7, color=BLUE)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color=RED),
    ]
    fig.legend(
        handles,
        ["prerequisite mask", "no mask"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
    )
    save(fig, "comparison")


# ----------------------------------------------------------------- Figure: latency
def figure_latency() -> None:
    """When each tactic is first detected, over the released corpus.

    The ordering is the practitioner finding: the tactics a rule set sees first
    are not the ones most monitoring effort is spent on.
    """
    cov = load("v5_coverage.json")
    rows = sorted(cov["per_tactic"], key=lambda r: r["median_step"])
    names = [r["tactic"] for r in rows]
    steps = [r["median_step"] for r in rows]

    # Coloured by the Cyber Kill Chain phase of Table 4,
    # so the interleaving that Section 4.10 reports is visible: the phases do
    # not arrive in their own order.
    kc = load("v7_killchain.json")
    phase_of = kc["mapping"]
    PHASE_COLOUR = {"Delivery": "#c0392b", "Exploitation": "#d68910",
                    "Installation": "#6b46c1", "Command and Control": "#2f855a",
                    "Actions on Objectives": "#2b6cb0"}
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    y = np.arange(len(names))
    ax.barh(y, steps, color=[PHASE_COLOUR[phase_of[n]] for n in names])
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("median step at which the tactic is first detected")
    ax.set_title(
        f"First detection by tactic, {cov['campaigns']} campaigns, "
        f"{cov['rules']} rules",
        loc="left",
    )
    for i, s_ in enumerate(steps):
        ax.text(s_ + 0.6, i, f"{s_:g}", va="center", fontsize=7.5, color=GREY)
    ax.set_xlim(0, max(steps) * 1.12)
    used = [p for p in PHASE_COLOUR if p in {phase_of[n] for n in names}]
    # Under the plot, not over it: inside the axes it covered the three
    # longest bars and their values.
    fig.tight_layout()
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=PHASE_COLOUR[p]) for p in used],
               used, title="Cyber Kill Chain phase", loc="upper center",
               bbox_to_anchor=(0.55, 0.0), ncol=3, frameon=False, fontsize=7,
               title_fontsize=7.5)
    save(fig, "latency")


# ---------------------------------------------------------------- Figure: transfer
def figure_transfer() -> None:
    """Simulated vs real execution, per configuration.

    Retention is the y axis rather than the raw gap: a configuration that
    barely succeeds in simulation has little to lose, so its small gap flatters
    it. Seeds below the minimum simulated base rate are excluded, as in the
    table, because a retention percentage computed on two successful episodes
    is noise.
    """
    sim = load("v5_sim_to_real.json")
    order = [("masked", "MaskablePPO"), ("maska2c", "MaskableA2C"), ("maskdqn", "MaskableDQN")]

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.6))
    width = 0.36
    x = np.arange(len(order))

    sim_means, real_means, retentions, errs = [], [], [], []
    for key, _ in order:
        block = sim[key]
        floor = block.get("min_base_rate", 20.0)
        s = [r["simulated"]["success"] for r in block["rows"]]
        r = [r["real"]["success"] for r in block["rows"]]
        sim_means.append(float(np.mean(s)))
        real_means.append(float(np.mean(r)))
        keep = [
            100.0 * rr / ss
            for ss, rr in zip(s, r)
            if ss >= floor  # a retention on a near-zero base rate is meaningless
        ]
        retentions.append(float(np.mean(keep)))
        errs.append(float(np.std(keep)))

    labels = [lbl for _, lbl in order]
    axes[0].bar(x - width / 2, sim_means, width, label="simulated", color=BLUE)
    axes[0].bar(x + width / 2, real_means, width, label="real execution", color=ORANGE)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("mission success (%)")
    axes[0].set_title("(a) Simulated and real", loc="left")
    axes[0].set_ylim(0, 105)
    # Under the panels: inside (a) it sat on top of the MaskableA2C bar.
    fig.tight_layout()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)

    axes[1].bar(x, retentions, 0.55, yerr=errs, capsize=3,
                color=[BLUE, BLUE, PURPLE], error_kw={"lw": 0.8})
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("capability retained (%)")
    axes[1].set_title("(b) Retention under real execution", loc="left")
    axes[1].set_ylim(0, 105)
    for i, (v, e) in enumerate(zip(retentions, errs)):
        # Clear the error-bar cap, not just the bar top.
        axes[1].text(i, v + e + 3, f"{v:.0f}%", ha="center", fontsize=7.5, color=GREY)

    save(fig, "transfer")


if __name__ == "__main__":
    figure_comparison()
    figure_latency()
    figure_transfer()
    print(f"\nfigures written to {OUT}")
