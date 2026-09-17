# -*- coding: utf-8 -*-
"""Regenerate the four system/method diagrams for the v4 environment.

The v3 originals describe the superseded design: two independent
implementations, a 13-dimensional state, the v3 reward scale, the
`alert > 1.0` bust rule and the free CLEAR_LOGS loop. Each of those is
contradicted by the v4 system, so the diagrams are redrawn here rather
than reused.

Outputs (300 dpi, results/figures/):
    v4_fig_architecture.png   shared world model
    v4_fig_topology.png       three-host network, four DC routes
    v4_fig_masking.png        mask as a function of state AND scenario
    v4_fig_statemachine.png   phases, v4 rewards, rule-based detection
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "figures")

INK = "#1f2933"
MUTED = "#5c6b7a"
BLUE = "#1f6fb2"
GREEN = "#2e7d4f"
RED = "#b3352c"
PURPLE = "#6b3fa0"
AMBER = "#b5761d"
TEAL = "#12726e"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def box(ax, x, y, w, h, title, body="", edge=BLUE, face="#ffffff",
        title_size=11, body_size=8.5, lw=1.8, ls="solid"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=lw, edgecolor=edge, facecolor=face,
                                linestyle=ls, zorder=2))
    if body:
        ax.text(x + w / 2, y + h * 0.82, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=edge, zorder=3)
        ax.text(x + w / 2, y + h * 0.36, body, ha="center", va="center",
                fontsize=body_size, color=MUTED, zorder=3, linespacing=1.5)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=edge, zorder=3)


def arrow(ax, p, q, color=INK, lw=1.5, style="-|>", rad=0.0, ls="solid"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14,
                                 linewidth=lw, color=color, zorder=1,
                                 linestyle=ls,
                                 connectionstyle="arc3,rad=%.2f" % rad,
                                 shrinkA=2, shrinkB=2))


def label(ax, x, y, s, color=MUTED, size=8.2, style="italic", weight="normal"):
    ax.text(x, y, s, ha="center", va="center", fontsize=size, color=color,
            style=style, fontweight=weight, zorder=4,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="none", alpha=0.92))


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=14)
    return fig, ax


def save(fig, name):
    # PNG for drafting, PDF for submission: these are vector line drawings, and
    # Elsevier asks for vector artwork as EPS or PDF rather than a bitmap.
    png = os.path.join(OUT, name)
    fig.savefig(png, dpi=600, bbox_inches="tight")
    pdf = os.path.join(OUT, name.replace(".png", ".pdf"))
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png, "+ pdf")


# ── 1. architecture: one shared world model ────────────────────────────────
def architecture():
    fig, ax = canvas(11.5, 6.4, "System architecture: one shared world model")

    box(ax, 0.02, 0.58, 0.20, 0.23, "MaskablePPO agent",
        "sb3-contrib\nMLP policy (64-64)", edge=GREEN, face="#eef7f1")

    box(ax, 0.29, 0.58, 0.23, 0.23, "AdversaryEnv",
        "Gymnasium wrapper\nreward · telemetry", edge=BLUE, face="#eaf2fa")

    box(ax, 0.60, 0.58, 0.23, 0.23, "mock_server.py",
        "HTTP target · sessions\ntoken auth · SIEM feed", edge=AMBER,
        face="#fdf4e7")

    # the shared core
    box(ax, 0.29, 0.23, 0.54, 0.26, "kill_chain.py — shared world model",
        "25-technique registry (probability · noise · ATT&CK id · precondition · effect)\n"
        "NetworkState  ·  per-episode scenario ω  -  legal-action mask  ·  state transition",
        edge=PURPLE, face="#f3eefa", title_size=12, body_size=8.8, lw=2.6)

    box(ax, 0.29, 0.01, 0.25, 0.17, "detection.py",
        "Windows / Sysmon events\nnamed rules · suspicion σ", edge=RED,
        face="#fbeceb")
    box(ax, 0.58, 0.01, 0.25, 0.17, "Telemetry logger",
        "ATT&CK-labelled records\nECS export", edge=TEAL, face="#e9f5f4")

    arrow(ax, (0.22, 0.72), (0.29, 0.72), GREEN)
    label(ax, 0.255, 0.765, "action")
    arrow(ax, (0.29, 0.665), (0.22, 0.665), GREEN)
    label(ax, 0.255, 0.625, "obs + mask")

    arrow(ax, (0.52, 0.72), (0.60, 0.72), AMBER)
    label(ax, 0.56, 0.765, "HTTP\n(real mode)", size=7.6)

    # both wrap the same module
    arrow(ax, (0.405, 0.58), (0.405, 0.49), PURPLE, lw=2.2)
    arrow(ax, (0.715, 0.58), (0.715, 0.49), PURPLE, lw=2.2)
    label(ax, 0.405, 0.535, "imports", color=PURPLE, weight="bold")
    label(ax, 0.715, 0.535, "imports", color=PURPLE, weight="bold")

    arrow(ax, (0.415, 0.23), (0.415, 0.18), RED)
    arrow(ax, (0.70, 0.23), (0.70, 0.18), TEAL)

    ax.text(0.862, 0.355,
            "One definition of the world.\nThe simulator and the live\ntarget call the same\n"
            "transition function, so they\ncannot disagree by\nconstruction (Section 3.3).",
            ha="left", va="center", fontsize=8.6, color=PURPLE, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f3eefa",
                      edgecolor=PURPLE, linewidth=1.2))
    save(fig, "v4_fig_architecture.png")


# ── 2. topology ────────────────────────────────────────────────────────────
def topology():
    fig, ax = canvas(11.5, 5.9, "Target network: three hosts, four routes to domain dominance")

    hosts = [(0.05, BLUE, "user01", "Workstation\n192.168.1.10", "#eaf2fa"),
             (0.38, GREEN, "srv01", "Application server\n192.168.1.20", "#eef7f1"),
             (0.71, RED, "dc01", "Domain controller\n192.168.1.1", "#fbeceb")]
    for x, c, name, sub, face in hosts:
        box(ax, x, 0.44, 0.24, 0.26, name, sub + "\n\ncompromised · privileged\nhas_credentials",
            edge=c, face=face, title_size=13, body_size=8.4)

    arrow(ax, (0.29, 0.555), (0.38, 0.555), GREEN, lw=2.0)
    label(ax, 0.335, 0.615, "lateral\nmovement", size=7.8)
    arrow(ax, (0.62, 0.555), (0.71, 0.555), RED, lw=2.0)
    label(ax, 0.665, 0.615, "lateral\nmovement", size=7.8)

    box(ax, 0.05, 0.02, 0.24, 0.26, "Entry vectors",
        "phishing (T1566)\nSQL injection (T1190)\npassword spraying (T1110.003)\nbrute force (T1110)",
        edge=PURPLE, face="#f3eefa", title_size=9.5, body_size=8.0)
    arrow(ax, (0.17, 0.28), (0.17, 0.40), PURPLE, lw=1.6)

    box(ax, 0.38, 0.02, 0.24, 0.26, "Credential access",
        "LSASS dump (T1003.001)\nKerberoasting (T1558.003)\nAS-REP roast (T1558.004)\n"
        "pass-the-hash / -ticket",
        edge=TEAL, face="#e9f5f4", title_size=9.5, body_size=8.0)
    arrow(ax, (0.50, 0.28), (0.50, 0.40), TEAL, lw=1.6)

    box(ax, 0.71, 0.02, 0.24, 0.26, "Domain dominance",
        "Kerberoasting · DCSync\nGPO modification\ngolden ticket (T1558.001)",
        edge=RED, face="#fbeceb", title_size=9.5, body_size=8.0)
    arrow(ax, (0.83, 0.28), (0.83, 0.40), RED, lw=1.6)

    ax.text(0.5, 0.83,
            "Each episode draws a scenario ω: a random non-empty subset of the substitutable "
            "techniques is offered at every phase,\nand the objective may additionally require "
            "persistence or host-level elevation. The agent observes what is on offer.",
            ha="center", va="center", fontsize=8.8, color=INK, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="#fffaf0",
                      edgecolor=AMBER, linewidth=1.3))
    save(fig, "v4_fig_topology.png")


# ── 3. masking ─────────────────────────────────────────────────────────────
def masking():
    fig, ax = canvas(11.5, 5.6,
                     "Action masking: the legal set depends on state and on the episode scenario")

    box(ax, 0.02, 0.54, 0.21, 0.29, "Environment state  s",
        "9 host flags\nnetwork scanned · backdoor\nsuspicion σ / Θ",
        edge=BLUE, face="#eaf2fa", title_size=10.5, body_size=8.3)
    box(ax, 0.02, 0.14, 0.21, 0.29, "Episode scenario  ω",
        "24 availability flags\n2 objective-requirement\nflags",
        edge=AMBER, face="#fdf4e7", title_size=10.5, body_size=8.3)

    box(ax, 0.31, 0.32, 0.20, 0.33, "m = μ(s, ω)",
        "evaluate every\nprecondition against s,\nintersect with what ω\noffers this episode",
        edge=PURPLE, face="#f3eefa", title_size=12, body_size=8.3, lw=2.4)

    arrow(ax, (0.23, 0.68), (0.31, 0.56), BLUE, rad=-0.12)
    arrow(ax, (0.23, 0.28), (0.31, 0.41), AMBER, rad=0.12)

    box(ax, 0.585, 0.32, 0.17, 0.33, "logits",
        "m = 0  →  −∞\nbefore the softmax\n(Equation 4)",
        edge=GREEN, face="#eef7f1", title_size=10.5, body_size=8.3)
    arrow(ax, (0.51, 0.485), (0.585, 0.485), PURPLE, lw=2.0)

    box(ax, 0.815, 0.32, 0.165, 0.33, "policy",
        "samples only among\ntechniques whose\npreconditions hold",
        edge=GREEN, face="#eef7f1", title_size=10.5, body_size=8.3)
    arrow(ax, (0.755, 0.485), (0.815, 0.485), GREEN, lw=2.0)

    ax.text(0.5, 0.90,
            "Action space: 25 ATT&CK-aligned techniques.  Observation: 38 dimensions, of which 24 are "
            "the availability flags of ω.",
            ha="center", va="center", fontsize=9, color=INK)

    ax.text(0.5, 0.12,
            "In the initial environment the mask had the form μ(s): legality was a fixed function of state, and an unmasked agent could induce it.\n"
            "Here legality is conditional on the episode as well, which Section 4.9 shows an unmasked agent does not induce -- although ω is fully observable.",
            ha="center", va="center", fontsize=8.5, color=MUTED, linespacing=1.7,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="#f7f8fa",
                      edgecolor="#c8d0d8", linewidth=1.1))
    save(fig, "v4_fig_masking.png")


# ── 4. state machine ───────────────────────────────────────────────────────
def statemachine():
    fig, ax = canvas(11.5, 6.2, "Environment state machine: phases, rewards and termination")

    nodes = [
        (0.075, 0.60, "start", "", "#e7eaee", INK),
        (0.245, 0.60, "initial\naccess", "+10", "#eaf2fa", BLUE),
        (0.415, 0.60, "discovery\n+ credentials", "+15 / +20", "#e9f5f4", TEAL),
        (0.585, 0.60, "lateral\nmovement", "+30", "#eef7f1", GREEN),
        (0.755, 0.60, "DC root", "+100", "#f3eefa", PURPLE),
        (0.915, 0.60, "impact", "+400", "#eef7f1", GREEN),
    ]
    for x, y, name, rew, face, edge in nodes:
        ax.add_patch(Circle((x, y), 0.062, facecolor=face, edgecolor=edge,
                            linewidth=1.9, zorder=2))
        ax.text(x, y + 0.008, name, ha="center", va="center", fontsize=8.6,
                fontweight="bold", color=edge, zorder=3, linespacing=1.3)
        if rew:
            ax.text(x, y + 0.093, rew, ha="center", va="bottom", fontsize=8.6,
                    color=edge, fontweight="bold", zorder=3)

    for i in range(len(nodes) - 1):
        arrow(ax, (nodes[i][0] + 0.062, 0.60), (nodes[i + 1][0] - 0.062, 0.60),
              INK, lw=1.5)

    # detection branch
    ax.add_patch(Circle((0.50, 0.20), 0.070, facecolor="#fbeceb",
                        edgecolor=RED, linewidth=2.2, zorder=2))
    ax.text(0.50, 0.215, "incident\nresponse", ha="center", va="center",
            fontsize=8.8, fontweight="bold", color=RED, zorder=3, linespacing=1.3)
    ax.text(0.50, 0.105, "-50, episode ends", ha="center", va="center",
            fontsize=8.4, color=RED, fontweight="bold", zorder=3)

    for x in (0.245, 0.415, 0.585, 0.755):
        arrow(ax, (x, 0.536), (0.50, 0.274), RED, lw=1.0, ls=(0, (4, 3)))

    label(ax, 0.215, 0.365, "every technique emits events;\nnamed rules raise suspicion σ",
          color=RED, size=8.2)
    label(ax, 0.755, 0.325, "terminates when  σ ≥ Θ = 1.0", color=RED, size=8.4,
          weight="bold")

    ax.text(0.5, 0.86,
            "Log clearing (T1070) raises Event 1102, may fail, only partly reduces suspicion and earns no reward,\n"
            "so it is taken only when the risk it removes exceeds the noise it creates.",
            ha="center", va="center", fontsize=8.6, color=AMBER, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffaf0",
                      edgecolor=AMBER, linewidth=1.2))

    ax.text(0.5, 0.02,
            "Absent mission success or detection, an episode ends at the 40-step limit. "
            "Actions whose preconditions are unmet are inert: no state change, no telemetry.",
            ha="center", va="center", fontsize=8.4, color=MUTED)
    save(fig, "v4_fig_statemachine.png")


if __name__ == "__main__":
    architecture()
    topology()
    masking()
    statemachine()
