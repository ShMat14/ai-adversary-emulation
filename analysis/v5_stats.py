# -*- coding: utf-8 -*-
"""Confidence intervals and significance tests for every comparison we make.

The paper reports means with standard deviations and, in places, reads an
ordering off them. With ten seeds and standard deviations as large as 24 points
that ordering is not always meaningful, and saying so requires a test rather
than an eyebrow.

Everything here is resampling-based and implemented directly, because SciPy is
not a dependency of this project and adding one for two tests is not worth it.

  bootstrap_ci   percentile bootstrap over seeds, 20,000 resamples
  perm_test      two-sided exact-style permutation test on the difference of
                 means, 50,000 relabellings; with n=10 per group the exact test
                 has 184,756 arrangements, so the sampled version is
                 indistinguishable from exact at three decimal places

We report the permutation test rather than a t-test because success rates over
ten seeds are bounded, skewed and clearly not normal, and a permutation test
assumes only exchangeability under the null.

    python analysis/v5_stats.py
"""
import argparse
import itertools
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

RNG = np.random.default_rng(20260906)


def bootstrap_ci(xs, reps=20000, alpha=0.05):
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return float(xs.mean()), float("nan"), float("nan")
    idx = RNG.integers(0, len(xs), size=(reps, len(xs)))
    means = xs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(xs.mean()), float(lo), float(hi)


def perm_test(a, b, reps=50000):
    """Two-sided permutation test on the difference of means."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    n = len(a)
    # exact when the arrangement count is small enough to enumerate
    total = len(pool)
    from math import comb
    if comb(total, n) <= 200_000:
        hits = tot = 0
        for combo in itertools.combinations(range(total), n):
            mask = np.zeros(total, dtype=bool)
            mask[list(combo)] = True
            d = abs(pool[mask].mean() - pool[~mask].mean())
            hits += d >= obs - 1e-12
            tot += 1
        return obs, hits / tot, "exact"
    hits = 0
    for _ in range(reps):
        p = RNG.permutation(pool)
        if abs(p[:n].mean() - p[n:].mean()) >= obs - 1e-12:
            hits += 1
    return obs, (hits + 1) / (reps + 1), "sampled"


def cliffs_delta(a, b):
    """Non-parametric effect size: P(a>b) - P(a<b), in [-1, 1]."""
    a, b = list(a), list(b)
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def label(d):
    d = abs(d)
    return ("negligible" if d < 0.147 else "small" if d < 0.33
            else "medium" if d < 0.474 else "large")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/v5_stats.json")
    a = ap.parse_args()
    out = {}

    R = json.load(open("results/v5_report.json"))
    per = {k: [s["success"] for s in v["per_seed"]]
           for k, v in R["main"].items()}
    ill = {k: [s["illegal"] for s in v["per_seed"]]
           for k, v in R["main"].items()}

    print("Main comparison, mission success, bootstrap 95% CI over seeds\n")
    print(f"{'configuration':32}{'mean':>8}{'95% CI':>20}{'n':>4}")
    print("-" * 64)
    out["main_ci"] = {}
    for k, xs in per.items():
        m, lo, hi = bootstrap_ci(xs)
        out["main_ci"][k] = {"mean": m, "lo": lo, "hi": hi, "n": len(xs)}
        print(f"{k:32}{m:>7.1f}%{f'[{lo:.1f}, {hi:.1f}]':>20}{len(xs):>4}")

    print("\nPairwise tests against the masked agent (success)\n")
    print(f"{'comparison':34}{'diff':>8}{'p':>10}{'kind':>9}{'effect':>22}")
    print("-" * 83)
    out["main_tests"] = {}
    for k in per:
        if k == "masked":
            continue
        d, p, kind = perm_test(per["masked"], per[k])
        cd = cliffs_delta(per["masked"], per[k])
        out["main_tests"][k] = {"diff": d, "p": p, "kind": kind, "cliffs": cd}
        print(f"{'masked vs ' + k:34}{d:>7.1f}{p:>10.4f}{kind:>9}"
              f"{f'{cd:+.2f} ({label(cd)})':>22}")

    # ---- the comparison that actually needed testing -----------------------
    G = json.load(open("results/v5_generalisation.json"))
    gper = {k: [s["success"] for s in v["per_seed"]] for k, v in G.items()}
    print("\nShape transfer: is the ordering of the unseen estates real?\n")
    print(f"{'comparison':34}{'diff':>8}{'p':>10}{'kind':>9}{'effect':>22}")
    print("-" * 83)
    out["shape_tests"] = {}
    pairs = [("enterprise", "flat12"), ("enterprise", "deep12"),
             ("enterprise", "hub12"), ("deep12", "hub12"),
             ("flat12", "deep12")]
    for x, y in pairs:
        d, p, kind = perm_test(gper[x], gper[y])
        cd = cliffs_delta(gper[x], gper[y])
        out["shape_tests"][f"{x} vs {y}"] = {"diff": d, "p": p, "kind": kind,
                                             "cliffs": cd}
        print(f"{f'{x} vs {y}':34}{d:>7.1f}{p:>10.4f}{kind:>9}"
              f"{f'{cd:+.2f} ({label(cd)})':>22}")

    print("\nShape transfer, bootstrap 95% CI\n")
    out["shape_ci"] = {}
    for k, xs in gper.items():
        m, lo, hi = bootstrap_ci(xs)
        out["shape_ci"][k] = {"mean": m, "lo": lo, "hi": hi, "n": len(xs)}
        print(f"  {k:12}{m:>6.1f}%  [{lo:.1f}, {hi:.1f}]  n={len(xs)}")

    os.makedirs("results", exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
