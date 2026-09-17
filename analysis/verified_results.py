# -*- coding: utf-8 -*-
"""
Single source of truth for the v3 evaluation figures.

Several plotting scripts previously hard-coded their own copies of these
numbers. When the evaluation was re-run on 2026-07-28 the manuscript text was
corrected but the figures were not, so the transfer-gap figure showed a
detection gap of 1.50 while the surrounding text said 8.0. Every plot now reads
from here instead.

Values are the fresh 200-episode re-runs of results/models/ppo_adversary_v3.zip,
the checkpoint released with the paper.

    Sim   100.0% success, 0.0% detection, 14.21 steps, 811.40 reward, 9.0/15
    Real   99.5% success, 8.0% detection, 16.90 steps, 769.85 reward, 9.0/15

v1 and v2 come from development runs predating version control. They cannot be
reproduced from the repository and are carried unchanged, flagged as such.
"""

MEASURED = "2026-07-28"

V3_SIM = {"success_rate": 100.0, "detection_rate": 0.0,
          "avg_steps": 14.21, "avg_reward": 811.40, "avg_unique_techniques": 9.0}

V3_REAL = {"success_rate": 99.5, "detection_rate": 8.0,
           "avg_steps": 16.90, "avg_reward": 769.85, "avg_unique_techniques": 9.0}

# unverifiable, pre-version-control; retained for the progression only
V2_SIM = {"success_rate": 100.0, "detection_rate": 0.5,
          "avg_steps": 6.50, "avg_reward": 628.86, "avg_unique_techniques": 5.0}

V2_REAL = {"success_rate": 99.5, "detection_rate": 4.0,
           "avg_steps": 7.57, "avg_reward": 610.69, "avg_unique_techniques": 5.0}

BASELINES = {
    "Scripted-Standard":   {"success_rate": 0.0, "detection_rate": 0.0, "avg_reward": 61.12},
    "Scripted-Stealthy":   {"success_rate": 0.0, "detection_rate": 0.0, "avg_reward": 58.46},
    "Scripted-Aggressive": {"success_rate": 0.0, "detection_rate": 1.0, "avg_reward": 21.25},
    "Scripted-SQLi":       {"success_rate": 0.0, "detection_rate": 0.0, "avg_reward": 23.01},
    "Random Agent":        {"success_rate": 87.0, "detection_rate": 19.5, "avg_reward": 616.44},
}


def transfer_gap(sim, real):
    """Absolute Sim - Real gap for each reported metric."""
    return {k: abs(sim[k] - real[k]) for k in sim}


def summary_line():
    g = transfer_gap(V3_SIM, V3_REAL)
    drop = 100 * (V3_SIM["avg_reward"] - V3_REAL["avg_reward"]) / V3_SIM["avg_reward"]
    return (f"v3 sim-to-real: {g['detection_rate']:.1f} pp detection, "
            f"{drop:.1f}% reward reduction (measured {MEASURED})")


if __name__ == "__main__":
    print(summary_line())
    for name, d in (("v3 Sim", V3_SIM), ("v3 Real", V3_REAL)):
        print(f"  {name:8s} " + "  ".join(f"{k}={v}" for k, v in d.items()))
