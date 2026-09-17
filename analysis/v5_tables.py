# -*- coding: utf-8 -*-
"""Result tables for the v5 study.

The main table is deliberately laid out in the same shape as Table 3 of Zhan et
al. (L-ARLPT, Applied Sciences 2026), so our numbers can be read directly
against theirs rather than requiring a reader to translate between metrics.
Their columns are failed %, redundant %, successful %, average steps and average
penetration depth; ours adds mission success and detection, which neither
comparator reports.

    python analysis/v5_tables.py                # main comparison
    python analysis/v5_tables.py --budgets      # the fairness controls
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from analysis.v5_train import evaluate_all, CONFIGS

# Zhan et al., Applied Sciences 2026, Table 3. Their environment is not ours, so
# these are context for the reader, never a like-for-like score.
LARLPT_TABLE3 = [
    ("BDQ",                       10.04, 89.76, 0.20, 500.0,  1.00),
    ("Wolpertinger",              89.50,  9.71, 0.79, 500.0,  1.00),
    ("Discrete-SAC",              92.78,  5.82, 1.40, 500.0,  1.27),
    ("Knowledge-Driven (LLM)",    92.77,  5.82, 1.42, 182.0,  1.12),
    ("L-ARLPT (their best)",      74.33, 19.02, 6.66, 128.43, 3.33),
]

LABELS = {
    "masked": "MaskablePPO + prerequisite mask",
    "nomask": "PPO, unmasked",
    "nomask_shaped": "PPO + invalid-action penalty",
    "dqn": "DQN, unmasked",
    "a2c": "A2C, unmasked",
}


def collect(topology="enterprise", suffix="", episodes=200, max_steps=60):
    res = evaluate_all(topology, episodes, None, max_steps, suffix)
    # the reward-shaped variant is a nomask checkpoint with its own suffix
    shaped = evaluate_all(topology, episodes, None, max_steps, suffix + "_shaped")
    if "nomask" in shaped:
        res["nomask_shaped"] = shaped["nomask"]
    return res


def print_main(res, title):
    print()
    print(title)
    print("=" * len(title))
    hdr = (f"{'configuration':32s}{'success %':>13}{'failed %':>12}"
           f"{'redund %':>10}{'advanced %':>12}{'APD':>7}{'steps':>8}{'detected %':>12}")
    print(hdr); print("-" * len(hdr))
    for k in ["masked", "nomask", "nomask_shaped", "dqn", "a2c"]:
        v = res.get(k)
        if not v:
            continue
        print(f"{LABELS[k]:32s}"
              f"{v['success_mean']:8.1f} ±{v['success_sd']:4.1f}"
              f"{v['illegal_mean']:8.1f} ±{v['illegal_sd']:3.1f}"
              f"{v['redundant_mean']:10.1f}{v['advanced_mean']:12.1f}"
              f"{v['apd_mean']:7.2f}{v['ep_len_mean']:8.1f}"
              f"{v['detection_mean']:12.1f}")
    n = max((v["n"] for v in res.values()), default=0)
    print(f"\n  seeds: {n}   episodes per seed: "
          f"{next(iter(res.values()))['episodes'] if res else 0}")
    print("  APD = average penetration depth: deepest subnet reached, "
          "on the zone hierarchy")


def print_requirements(res, config="masked"):
    """Success split by what the engagement demanded.

    The headline mean is a poor description of this policy. A seed that solves
    every episode with no extra requirement and none that demands local
    elevation reports as roughly 48%, which is true of no scenario it faces. The
    split is the honest statement and it is also the more interesting one: it
    says where scenario adaptation succeeds and where it does not.
    """
    v = res.get(config)
    if not v or not v.get("by_requirement"):
        return
    title = f"Success by engagement requirement — {LABELS[config]}"
    print(); print(title); print("=" * len(title))
    hdr = f"{'additional requirement':40s}{'success %':>12}{'sd':>8}{'seeds':>8}"
    print(hdr); print("-" * len(hdr))
    for k, d in sorted(v["by_requirement"].items(),
                       key=lambda kv: -kv[1]["mean"]):
        print(f"{k:40s}{d['mean']:12.1f}{d['sd']:8.1f}{d['seeds']:8d}")
    print('')
    print("  Every episode requires initial access, lateral movement to the "
          "objective host and the objective action; the rows above are what "
          "it demands on top.")


def print_larlpt_context():
    print()
    print("For context — Zhan et al. (L-ARLPT, Applied Sciences 2026), their Table 3")
    print("Their environment, their metrics; not a like-for-like score against ours.")
    hdr = f"{'method':32s}{'failed %':>12}{'redund %':>10}{'success %':>12}{'steps':>8}{'APD':>7}"
    print(hdr); print("-" * len(hdr))
    for name, f, r, s, st, apd in LARLPT_TABLE3:
        print(f"{name:32s}{f:12.2f}{r:10.2f}{s:12.2f}{st:8.1f}{apd:7.2f}")
    print("\n  Their best configuration prunes a 60,374,160-action space with two"
          "\n  LLMs to five candidates per step, and still refuses 74.33% of what"
          "\n  it proposes. One training run took approximately nine days.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="enterprise")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--budgets", action="store_true",
                    help="report the fairness controls at 500 and 2000 steps")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    out = {}
    main = collect(a.topology, "", a.episodes, 60)
    out["main"] = main
    print_main(main, f"Main comparison — {a.topology}, 60-step budget")
    print_requirements(main)
    print_larlpt_context()

    if a.budgets:
        for b in (500, 2000):
            r = collect(a.topology, f"_b{b}", a.episodes, b)
            if r:
                out[f"budget_{b}"] = r
                print_main(r, f"Fairness control — unmasked agents at a "
                              f"{b}-step budget")
                print("  If an unmasked agent succeeds here, masking buys "
                      "efficiency rather than\n  capability, and that is the "
                      "claim that must be reported.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {a.json}")
