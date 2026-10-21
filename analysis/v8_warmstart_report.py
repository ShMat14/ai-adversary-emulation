# -*- coding: utf-8 -*-
"""Summarise the warm-start runs: does a twelve-host policy give a head start?

Reads results/v8_warmstart/*.json (written by v8_warmstart.py) and reports,
per target estate and condition, the four quantities transfer studies usually
separate:

  jump-start    success at step 0, before any training on the target
  time to 50%   the first evaluation at which success reaches 50%
  area          mean success over the whole curve, which rewards being good early
  final         success of the saved model over 200 held-out engagements

Warm and scratch are compared with the exact two-sided permutation test on the
difference of means (5 against 5 seeds, all 252 splits enumerated), a
percentile bootstrap interval on each mean, and Cliff's delta, the same
statistics the paper uses elsewhere. Warm runs are also scored back on the
twelve-host estate they came from, against the 71.2% those policies reach there
before any further training (Supplementary Table S1, same 120-step budget).

    python analysis/v8_warmstart_report.py          # table + JSON
    python analysis/v8_warmstart_report.py --figure # also draw Fig. S1
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "v8_warmstart"
OUT = ROOT / "results" / "v8_warmstart.json"
FLOOR = {"mid16": 4.3, "large20": 2.3}          # random-legal %, Table S1
ENTERPRISE_BEFORE = 71.2                         # Table S1, trained-on row


def perm_test(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    n, hits, total = len(a), 0, 0
    for idx in itertools.combinations(range(len(pooled)), n):
        mask = np.zeros(len(pooled), bool)
        mask[list(idx)] = True
        d = abs(pooled[mask].mean() - pooled[~mask].mean())
        hits += d >= obs - 1e-12
        total += 1
    return hits / total


def cliffs(a, b):
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def boot(a, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    means = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def first_reach(curve, level):
    for p in curve:
        if p["success"] >= level:
            return p["timesteps"]
    return None


def load():
    runs = {}
    for f in sorted(RUNS.glob("*.json")):
        r = json.loads(f.read_text())
        runs.setdefault((r["target"], r["cond"]), []).append(r)
    return runs


def summarise(runs):
    out = {}
    for target in ("mid16", "large20"):
        warm = sorted(runs.get((target, "warm"), []), key=lambda r: r["seed"])
        scratch = sorted(runs.get((target, "scratch"), []), key=lambda r: r["seed"])
        if not warm or not scratch:
            continue
        block = {"seeds": {"warm": [r["seed"] for r in warm],
                           "scratch": [r["seed"] for r in scratch]},
                 "floor": FLOOR[target], "conditions": {}, "tests": {}}
        metrics = {}
        for cond, rs in (("warm", warm), ("scratch", scratch)):
            steps = [p["timesteps"] for p in rs[0]["curve"]]
            curves = np.array([[p["success"] for p in r["curve"]] for r in rs])
            m = {
                "jumpstart": [r["curve"][0]["success"] for r in rs],
                "area": list(curves.mean(axis=1)),
                "final": [r["final"]["success"] for r in rs],
                "final_detection": [r["final"]["detection"] for r in rs],
                "final_illegal": [r["final"]["illegal"] for r in rs],
                "to50": [first_reach(r["curve"], 50.0) for r in rs],
            }
            metrics[cond] = m
            c = {
                "n": len(rs),
                "curve_steps": steps,
                "curve_mean": [float(x) for x in curves.mean(axis=0)],
                "curve_min": [float(x) for x in curves.min(axis=0)],
                "curve_max": [float(x) for x in curves.max(axis=0)],
                "minutes_mean": float(np.mean([r["minutes"] for r in rs])),
            }
            for k in ("jumpstart", "area", "final", "final_detection", "final_illegal"):
                v = np.asarray(m[k], float)
                c[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                        "ci95": boot(v), "per_seed": [float(x) for x in v]}
            reached = [x for x in m["to50"] if x is not None]
            c["to50"] = {"per_seed": m["to50"], "reached": len(reached),
                         "median": float(np.median(reached)) if reached else None}
            if cond == "warm":
                back = [r["enterprise_after"]["success"] for r in rs if r.get("enterprise_after")]
                c["enterprise_after"] = {"mean": float(np.mean(back)), "per_seed": back,
                                         "before": ENTERPRISE_BEFORE}
            block["conditions"][cond] = c
        for k in ("jumpstart", "area", "final", "final_detection"):
            a, b = metrics["warm"][k], metrics["scratch"][k]
            if len(a) >= 2 and len(b) >= 2:
                block["tests"][k] = {"diff": float(np.mean(a) - np.mean(b)),
                                     "p": perm_test(a, b), "cliffs_delta": cliffs(a, b)}
        out[target] = block
    return out


def show(summary):
    for target, b in summary.items():
        w, s = b["conditions"]["warm"], b["conditions"]["scratch"]
        print(f"\n{target}  (random-legal floor {b['floor']}%, "
              f"{w['n']} warm / {s['n']} scratch seeds)")
        print(f"  {'':18}{'warm':>22}{'scratch':>22}   p      delta")
        for k, label in (("jumpstart", "success at step 0"), ("area", "mean over curve"),
                         ("final", "final success"), ("final_detection", "final detected")):
            t = b["tests"].get(k, {})
            print(f"  {label:18}{w[k]['mean']:>14.1f} ± {w[k]['sd']:<5.1f}"
                  f"{s[k]['mean']:>14.1f} ± {s[k]['sd']:<5.1f}"
                  f"  {t.get('p', float('nan')):.3f}  {t.get('cliffs_delta', float('nan')):+.2f}")
        print(f"  first eval >= 50%  warm {w['to50']['per_seed']}  scratch {s['to50']['per_seed']}")
        print(f"  infeasible final   warm {w['final_illegal']['mean']:.1f}%  scratch {s['final_illegal']['mean']:.1f}%")
        e = w["enterprise_after"]
        print(f"  warm back on enterprise: {e['mean']:.1f}% (was {e['before']}% before)  {e['per_seed']}")
        print("  mean curve  " + "  ".join(f"{t // 1000}k:{wm:.0f}/{sm:.0f}" for t, wm, sm in
                                           zip(w["curve_steps"], w["curve_mean"], s["curve_mean"])))


def figure(summary):
    import sys
    sys.path.insert(0, str(ROOT))
    from analysis.v6_figures import BLUE, ORANGE, plt, save
    targets = [t for t in ("mid16", "large20") if t in summary]
    fig, axes = plt.subplots(1, len(targets), figsize=(4.69, 2.6), sharey=True, squeeze=False)
    for ax, target in zip(axes[0], targets):
        b = summary[target]
        for cond, colour, label in (("warm", BLUE, "continued from twelve hosts"),
                                    ("scratch", ORANGE, "trained from scratch")):
            c = b["conditions"][cond]
            x = np.array(c["curve_steps"]) / 1000
            ax.plot(x, c["curve_mean"], color=colour, lw=1.4, label=label)
            ax.fill_between(x, c["curve_min"], c["curve_max"], color=colour, alpha=0.15, lw=0)
        ax.axhline(b["floor"], color="#4a5568", lw=0.8, ls="--", label="random legal play")
        hosts = {"mid16": "sixteen", "large20": "twenty"}[target]
        ax.set_title(f"({'ab'[targets.index(target)]}) {hosts}-host estate")
        ax.set_xlabel("Training timesteps (thousands)")
        ax.set_ylim(0, 100)
    axes[0][0].set_ylabel("Mission success (%)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, "warmstart")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    summary = summarise(load())
    if not summary:
        raise SystemExit("no complete warm/scratch pair yet")
    OUT.write_text(json.dumps(summary, indent=1))
    show(summary)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    if a.figure:
        figure(summary)


if __name__ == "__main__":
    main()
