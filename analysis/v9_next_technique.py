# -*- coding: utf-8 -*-
"""How well the next technique of a campaign can be predicted from the corpus.

Alert-correlation systems predict an intruder's next step from a graph of
which step followed which, weighted by how often (the construction of Fig. 13).
This runs the simplest such predictors on the released corpus, so that the
claim that the corpus can serve as labelled test data for them is measured
rather than asserted.

Data: results/telemetry/v5/episodes.jsonl, all 600 campaigns, 60 from each of
the ten trained policies. Evaluation holds policies out, not episodes: five
folds, each training on the campaigns of eight policies and predicting every
step of the other two, so a predictor never sees the agent it is tested on.

Predictors, each giving a ranked list of techniques:
  frequency    the techniques most used in the training campaigns
  first-order  the techniques that most often followed the current one
               (the transition counts of Fig. 13); frequency where unseen
  second-order the same, conditioned on the last two techniques; first-order
               where that pair was never seen

Targets: the next technique that differs from the current one (the sense of
Fig. 13, which does not draw back-to-back repeats), and the next action
including repeats. Reports top-1 and top-3 accuracy, mean and standard
deviation over the five folds.

    python analysis/v9_next_technique.py
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPISODES = ROOT / "results" / "telemetry" / "v5" / "episodes.jsonl"
OUT = ROOT / "results" / "v9_next_technique.json"


def load():
    eps = [json.loads(line) for line in EPISODES.open(encoding="utf-8")]
    for e in eps:
        e["seq"] = [s.split("@")[0] for s in e["technique_sequence"]]
    return eps


def collapse(seq):
    return [t for i, t in enumerate(seq) if i == 0 or t != seq[i - 1]]


def ranked(counter, k, fallback):
    out = [t for t, _ in counter.most_common(k)]
    for t in fallback:
        if len(out) >= k:
            break
        if t not in out:
            out.append(t)
    return out


def evaluate(train, test, k=3):
    freq = Counter(t for s in train for t in s)
    first, second = defaultdict(Counter), defaultdict(Counter)
    for s in train:
        for a, b in zip(s, s[1:]):
            first[a][b] += 1
        for a, b, c in zip(s, s[1:], s[2:]):
            second[(a, b)][c] += 1
    base = [t for t, _ in freq.most_common(k)]
    hits = {m: [0, 0] for m in ("frequency", "first-order", "second-order")}
    n = 0
    for s in test:
        for i in range(len(s) - 1):
            cur, nxt = s[i], s[i + 1]
            p1 = ranked(first[cur], k, base)
            p2 = ranked(second[(s[i - 1], cur)], k, p1) if i > 0 else p1
            for m, p in (("frequency", base), ("first-order", p1), ("second-order", p2)):
                hits[m][0] += p[0] == nxt
                hits[m][1] += nxt in p
            n += 1
    return n, {m: (100 * h1 / n, 100 * h3 / n) for m, (h1, h3) in hits.items()}


def main():
    eps = load()
    policies = sorted({e["policy"] for e in eps})
    assert len(policies) == 10 and all(sum(e["policy"] == p for e in eps) == 60 for p in policies)
    techniques = {t for e in eps for t in e["seq"]}
    result = {"campaigns": len(eps), "policies": len(policies), "techniques": len(techniques),
              "folds": 5, "targets": {}}
    for target, prep in (("next_different_technique", collapse), ("next_action", lambda s: s)):
        per_fold, total = [], 0
        for f in range(5):
            held = set(policies[2 * f:2 * f + 2])
            train = [prep(e["seq"]) for e in eps if e["policy"] not in held]
            test = [prep(e["seq"]) for e in eps if e["policy"] in held]
            n, acc = evaluate(train, test)
            per_fold.append(acc)
            total += n
        summary = {}
        for m in ("frequency", "first-order", "second-order"):
            t1 = [a[m][0] for a in per_fold]
            t3 = [a[m][1] for a in per_fold]
            summary[m] = {"top1_mean": round(statistics.mean(t1), 1), "top1_sd": round(statistics.stdev(t1), 1),
                          "top3_mean": round(statistics.mean(t3), 1), "top3_sd": round(statistics.stdev(t3), 1)}
        result["targets"][target] = {"predictions": total, "models": summary}

    # Which current techniques make the next one predictable: first-order top-1
    # on all campaigns pooled, for techniques followed at least 200 times.
    seqs = [collapse(e["seq"]) for e in eps]
    follow = defaultdict(Counter)
    for s in seqs:
        for a, b in zip(s, s[1:]):
            follow[a][b] += 1
    share = {a: (c.most_common(1)[0][0], round(100 * c.most_common(1)[0][1] / sum(c.values()), 1),
                 sum(c.values()), len(c)) for a, c in follow.items() if sum(c.values()) >= 200}
    result["most_predictable"] = sorted(([a, *v] for a, v in share.items()), key=lambda r: -r[2])[:5]
    result["least_predictable"] = sorted(([a, *v] for a, v in share.items()), key=lambda r: r[2])[:5]
    OUT.write_text(json.dumps(result, indent=1))

    print(f"{result['campaigns']} campaigns from {result['policies']} policies, "
          f"{result['techniques']} techniques; five folds, two policies held out per fold")
    for target, block in result["targets"].items():
        print(f"\n{target}: {block['predictions']} predictions")
        for m, s in block["models"].items():
            print(f"  {m:13} top-1 {s['top1_mean']:5.1f} ± {s['top1_sd']:.1f}   "
                  f"top-3 {s['top3_mean']:5.1f} ± {s['top3_sd']:.1f}")
    print("\nmost predictable (technique -> most frequent next, share, times followed, distinct next):")
    for r in result["most_predictable"]:
        print("  ", r)
    print("least predictable:")
    for r in result["least_predictable"]:
        print("  ", r)
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
