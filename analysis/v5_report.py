# -*- coding: utf-8 -*-
"""Everything the paper needs, from the checkpoints, in one command.

    python analysis/v5_report.py                  # tables + figures + corpus
    python analysis/v5_report.py --no-corpus      # skip the slow part
    python analysis/v5_report.py --episodes 300   # tighter error bars

Writes:
    results/v5_report.md          every table, in the order the paper uses them
    results/v5_report.json        the same numbers, machine-readable
    results/figures_v5/*.png      the figures
    results/telemetry/v5/         the released corpus

Nothing here recomputes a result from a training curve. The curve reports a
policy one gradient update older than the saved checkpoint, and with a policy
this variable that difference reached forty points in one run. Tables come from
re-evaluating checkpoints; curves are for showing the shape of learning.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from analysis.v5_train import evaluate_all, OUT

LABELS = {
    "masked":        "MaskablePPO + prerequisite mask",
    "nomask":        "PPO, unmasked",
    "nomask_shaped": "PPO + invalid-action penalty",
    "dqn":           "DQN, unmasked",
    "a2c":           "A2C, unmasked",
}
ORDER = ["masked", "nomask", "nomask_shaped", "dqn", "a2c"]

# Zhan et al., Applied Sciences 2026, Table 3. Their environment is not ours, so
# this is context for the reader and never a like-for-like score.
LARLPT = [
    ("BDQ",                    10.04, 89.76, 0.20, 500.0,  1.00),
    ("Wolpertinger",           89.50,  9.71, 0.79, 500.0,  1.00),
    ("Discrete-SAC",           92.78,  5.82, 1.40, 500.0,  1.27),
    ("Knowledge-Driven (LLM)", 92.77,  5.82, 1.42, 182.0,  1.12),
    ("L-ARLPT (their best)",   74.33, 19.02, 6.66, 128.43, 3.33),
]


def seeds_present(cfg, topology, suffix=""):
    pat = os.path.join(OUT, f"{cfg}_{topology}_s*{suffix}.zip")
    out = []
    for f in glob.glob(pat):
        tag = os.path.basename(f)[:-4]
        rest = tag[len(f"{cfg}_{topology}_s"):]
        if suffix:
            if not rest.endswith(suffix):
                continue
            rest = rest[:-len(suffix)]
        if rest.isdigit():
            out.append(int(rest))
    return sorted(out)


def collect(topology, suffix, episodes, max_steps):
    res = evaluate_all(topology, episodes, None, max_steps, suffix)
    shaped = evaluate_all(topology, episodes, None, max_steps, suffix + "_shaped")
    if "nomask" in shaped:
        res["nomask_shaped"] = shaped["nomask"]
    return res


def table_main(res, caption):
    L = [f"", f"**{caption}**", "",
         "| Configuration | Mission success % | Infeasible actions % | "
         "Redundant % | Advanced % | APD | Steps | Detected % | Seeds |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k in ORDER:
        v = res.get(k)
        if not v:
            continue
        L.append(
            f"| {LABELS[k]} | {v['success_mean']:.1f} ± {v['success_sd']:.1f} "
            f"| {v['illegal_mean']:.1f} ± {v['illegal_sd']:.1f} "
            f"| {v['redundant_mean']:.1f} | {v['advanced_mean']:.1f} "
            f"| {v['apd_mean']:.2f} | {v['ep_len_mean']:.1f} "
            f"| {v['detection_mean']:.1f} | {v['n']} |")
    L += ["", "Infeasible, redundant and advanced are shares of all actions "
              "taken, and they do not sum to 100: the remainder is actions that "
              "were feasible and executed but missed their success probability, "
              "which is stochastic failure rather than wasted choice. Separating "
              "those out matters — counting them as redundant overstated our own "
              "redundancy figure by seven points before it was corrected. "
              "APD is average penetration depth, the deepest zone reached on the "
              "five-tier hierarchy, counting a host as reached if it is held or "
              "the agent holds privilege on it."]
    return L


def table_requirements(res, cfg="masked"):
    v = res.get(cfg)
    if not v or not v.get("by_requirement"):
        return []
    # a configuration that never succeeds has nothing to break down
    if all(d["mean"] == 0 for d in v["by_requirement"].values()):
        return ["", f"*{LABELS[cfg]} reached the objective in no engagement, so "
                    f"there is no breakdown to report.*"]
    L = ["", f"**Success by what the engagement additionally demanded — "
             f"{LABELS[cfg]}**", "",
         "The headline mean is a poor description of this policy. A run that "
         "solves every episode without a given requirement and none of the "
         "episodes with it reports as a middling average that describes no "
         "scenario it actually faces.", "",
         "| Additional requirement | Success % | SD | Seeds |", "|---|---|---|---|"]
    for k, d in sorted(v["by_requirement"].items(), key=lambda kv: -kv[1]["mean"]):
        L.append(f"| {k} | {d['mean']:.1f} | {d['sd']:.1f} | {d['seeds']} |")
    L += ["", "Every episode requires initial access, movement to the objective "
              "and elevation on it; the rows above are what it demands on top. "
              "Sixteen groups share the evaluation episodes, so each row rests on "
              "roughly a sixteenth of them per seed; read the spread across seeds "
              "rather than the ordering of the rows."]
    return L


def table_larlpt():
    L = ["", "**For context — Zhan et al. (L-ARLPT, Applied Sciences 2026), "
             "their Table 3**", "",
         "Their environment, their metrics. Not a like-for-like score against "
         "ours, and reported here only so the reader can see the shape of the "
         "difference without translating between metrics.", "",
         "| Method | Failed % | Redundant % | Successful % | Steps | APD |",
         "|---|---|---|---|---|---|"]
    for n, f, r, s, st, a in LARLPT:
        L.append(f"| {n} | {f:.2f} | {r:.2f} | {s:.2f} | {st:.1f} | {a:.2f} |")
    L += ["", "Their best configuration prunes a 60,374,160-action space with "
              "two language models to five candidates per step, and still "
              "refuses 74.33% of what it proposes. One training run took "
              "approximately nine days."]
    return L


def main():
    ap = argparse.ArgumentParser()
    # 400 episodes, not 200: the per-requirement breakdown splits them sixteen ways,
    # so 200 leaves about a dozen episodes per group per seed and the group means
    # are dominated by sampling noise.
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--no-corpus", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--corpus-episodes", type=int, default=600)
    a = ap.parse_args()

    out, md = {}, ["# v5 results", "",
                   "Generated by `analysis/v5_report.py`. Every figure here "
                   "comes from re-evaluating a saved checkpoint, never from the "
                   "last point of a training curve.", ""]

    print("main comparison, 60-step budget ...", flush=True)
    main_res = collect("enterprise", "", a.episodes, 60)
    out["main"] = main_res
    md += ["## Main comparison — 12-host estate, 60-step budget"]
    md += table_main(main_res, "Decision behaviour and mission outcome")
    for cfg in ORDER:
        md += table_requirements(main_res, cfg)
    md += table_larlpt()

    for budget, tag in ((500, "_b500"), (2000, "_b2000"),
                        (500, "_b500_epmatch")):
        print(f"fairness control at {budget} steps ...", flush=True)
        r = collect("enterprise", tag, a.episodes, budget)
        if not r:
            continue
        out[f"budget_{budget}"] = r
        md += ["", f"## Fairness control — {budget}-step episode budget"]
        md += ["", "A short budget disadvantages an unmasked agent, which has "
                   "to absorb rejected attempts before reaching a reward. This "
                   "control gives the unmasked baselines the budget the "
                   "comparators use. **If an unmasked agent succeeds here, "
                   "masking buys efficiency rather than capability, and that "
                   "weaker claim is the one that must be reported.**"]
        md += table_main(r, f"Unmasked baselines at a {budget}-step budget"
                            + (", episode-matched (3.3M timesteps)"
                               if "epmatch" in tag else ""))
        if "epmatch" in tag:
            md += ["", "Phase 2 matches the comparators' episode budget at the "
                       "same timestep budget, which leaves those runs with 800 "
                       "episodes against 6,667 at a 60-step budget. This control "
                       "removes that objection by matching episodes instead, at "
                       "eight times the training cost."]

    print("formulation vs. size, 3-host network ...", flush=True)
    r = collect("v4compat", "", a.episodes, 60)
    if r:
        out["v4compat"] = r
        md += ["", "## Formulation against size — the 3-host network under v5 rules"]
        md += ["", "The same rules on the original three-host network, so any "
                   "gain over the previous environment can be attributed to the "
                   "action-space change rather than to a larger estate."]
        md += table_main(r, "Three-host network, 60-step budget")
        md += table_requirements(r)

    with open("results/v5_report.json", "w") as f:
        json.dump(out, f, indent=1)
    with open("results/v5_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("wrote results/v5_report.md and results/v5_report.json")

    if not a.no_figures:
        print("figures ...", flush=True)
        from analysis import v5_figures
        for topo in ("enterprise", "v4compat"):
            try:
                v5_figures.main(topo)
            except Exception as e:
                print(f"  figures for {topo} failed: {e}")

    if not a.no_corpus:
        print("telemetry corpus ...", flush=True)
        models = sorted(glob.glob(os.path.join(OUT, "masked_enterprise_s*.zip")))
        models = [m[:-4] for m in models
                  if m[:-4].split("_s")[-1].isdigit()]
        if models:
            from analysis.v5_corpus import generate, write
            rows, meta = generate(models, "enterprise", a.corpus_episodes, 60)
            man, files = write(rows, meta, "results/telemetry/v5",
                               "enterprise", 60)
            print(f"  {man['events']:,} events, {man['episodes']} episodes, "
                  f"{man['catalogue_coverage']}% of the catalogue")
        else:
            print("  no masked checkpoints yet")


if __name__ == "__main__":
    main()
