# -*- coding: utf-8 -*-
"""Turn the corpus into a detection-latency instrument.

A coverage matrix records that a rule *exists* for a technique. It cannot say
how much of a campaign runs before that rule has anything to fire on, because
answering that needs the techniques in the order an attacker actually used them.
That ordering is what the corpus carries and what a matrix throws away.

WHAT THIS MEASURES, AND WHAT IT DOES NOT

A first version of this analysis asked what fraction of campaigns a rule set
"sees". That question is saturated here and answers nothing: the catalogue maps
one named rule to each technique, so any reasonable rule set eventually sees
every campaign. Reporting 100% would have been true and useless.

The question that discriminates is *when*. For a deployed rule set we report the
step at which it first has something to fire on, and -- more usefully -- how far
the intrusion has already progressed by that point: what fraction of the
campaign's steps have run, and which security zone the attacker already holds.
A rule that first fires once the attacker is in the domain core is a rule that
fires too late, whatever the coverage matrix says.

Nothing here is stochastic. We ask what a rule set *covers*, not whether a given
roll of the detection dice fired, because a defender choosing what to deploy
wants the first question.

    python analysis/v5_coverage.py
"""
import argparse
import collections
import json
import re
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.detection import PROFILES
from env.kill_chain_v5 import TECHNIQUES_V5
from env.topology_v5 import Topology

ZONE_NAME = {0: "external", 1: "DMZ", 2: "workstations", 3: "servers",
             4: "admin tier", 5: "domain core"}


def load_episodes(d, topology="enterprise"):
    topo = Topology(topology)
    eps = []
    with open(os.path.join(d, "episodes.jsonl"), encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            raw = e.get("technique_sequence", [])
            seq = []
            for s in raw:
                tech, _, host = s.partition("@")
                seq.append((tech, topo.subnet_of.get(host, 0)))
            if seq and e.get("objective_reached"):
                eps.append({"seq": seq, "steps": int(e.get("steps", len(seq)))})
    return eps


def first_hit(ep, rules):
    """(step, deepest zone already held) when `rules` first has something to fire on."""
    deepest = 0
    for i, (tech, zone) in enumerate(ep["seq"]):
        deepest = max(deepest, zone)
        if PROFILES[tech].rule in rules:
            return i + 1, deepest
    return None, deepest


def latency(eps, rules):
    steps, fracs, zones, missed = [], [], [], 0
    for e in eps:
        s, z = first_hit(e, rules)
        if s is None:
            missed += 1
            continue
        steps.append(s)
        fracs.append(s / max(1, e["steps"]))
        zones.append(z)
    if not steps:
        return None
    return {"median_step": statistics.median(steps),
            "mean_step": statistics.mean(steps),
            "pct_campaign_elapsed": 100.0 * statistics.mean(fracs),
            "median_zone_reached": int(statistics.median(zones)),
            "never_covered": 100.0 * missed / max(1, len(eps))}


def greedy_earliest(eps, k, all_rules):
    """Choose rules one at a time to minimise the median first-hit step."""
    chosen, remaining, out = [], set(all_rules), []
    for _ in range(k):
        best, best_v = None, None
        for r in sorted(remaining):
            v = latency(eps, set(chosen) | {r})
            if v is None:
                continue
            key = (v["median_step"], v["pct_campaign_elapsed"])
            if best_v is None or key < best_v:
                best, best_v, best_full = r, key, v
        if best is None:
            break
        chosen.append(best)
        remaining.discard(best)
        out.append((len(chosen), best, best_full))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="results/telemetry/v5")
    ap.add_argument("--topology", default="enterprise")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json", default="results/v5_coverage.json")
    a = ap.parse_args()

    eps = load_episodes(a.corpus, a.topology)
    all_rules = sorted({PROFILES[t].rule for t in TECHNIQUES_V5})
    print(f"{len(eps)} successful campaigns in the corpus, "
          f"mean length {statistics.mean(e['steps'] for e in eps):.1f} steps")
    print(f"{len(all_rules)} named detection rules\n")

    full = latency(eps, set(all_rules))
    print("every rule deployed")
    print(f"   first firing opportunity at step {full['median_step']:.0f} (median)")
    print(f"   by then {full['pct_campaign_elapsed']:.1f}% of the campaign has run "
          f"and the attacker holds the {ZONE_NAME[full['median_zone_reached']]}\n")

    print(f"the {a.top} rules that see a campaign earliest, each deployed alone")
    print(f"   {'median step':>11}  {'% elapsed':>9}  {'zone reached':>13}  rule")
    singles = []
    for r in all_rules:
        v = latency(eps, {r})
        if v:
            singles.append((v, r))
    singles.sort(key=lambda x: (x[0]["median_step"], x[0]["pct_campaign_elapsed"]))
    for v, r in singles[:a.top]:
        print(f"   {v['median_step']:>11.0f}  {v['pct_campaign_elapsed']:>8.1f}%  "
              f"{ZONE_NAME[v['median_zone_reached']]:>13}  {r[:60]}")

    print(f"\nthe {a.top} rules that see a campaign LATEST, each deployed alone")
    for v, r in singles[-a.top:][::-1]:
        print(f"   {v['median_step']:>11.0f}  {v['pct_campaign_elapsed']:>8.1f}%  "
              f"{ZONE_NAME[v['median_zone_reached']]:>13}  {r[:60]}")

    print("\nbuilding a deployment greedily, choosing the earliest-firing rule each time")
    greedy = greedy_earliest(eps, a.top, all_rules)
    for k, rule, v in greedy:
        print(f"   {k} rule{'s' if k > 1 else ' '}: step {v['median_step']:>4.0f}   "
              f"{v['pct_campaign_elapsed']:5.1f}% elapsed   "
              f"{ZONE_NAME[v['median_zone_reached']]:>13}   + {rule[:46]}")

    by_tactic = collections.defaultdict(list)
    for v, r in singles:
        tac = next(TECHNIQUES_V5[t].tactic for t in TECHNIQUES_V5
                   if PROFILES[t].rule == r)
        by_tactic[tac].append(v["median_step"])
    print("\nwhen each tactic first offers a firing opportunity (median step)")
    for tac, xs in sorted(by_tactic.items(), key=lambda kv: statistics.median(kv[1])):
        print(f"   {statistics.median(xs):>5.0f}  {tac}")

    with open(a.json, "w") as f:
        # Table 19: the same numbers grouped by tactic. Derivable from per_rule
        # by parsing each rule's ATT&CK id, but recorded so it need not be.
        _by_tac = {}
        for _v, _r in singles:
            _m = re.search(r"\((T\d+(?:\.\d+)?)\)", _r)
            if not _m:
                continue
            for _n, _tech in TECHNIQUES_V5.items():
                if _tech.mitre_id == _m.group(1):
                    _by_tac.setdefault(_tech.tactic, []).append(_v["median_step"])
                    break
        per_tactic = sorted(
            ({"tactic": _k, "median_step": statistics.median(_vals),
              "rules": len(_vals)} for _k, _vals in _by_tac.items()),
            key=lambda d: d["median_step"])

        json.dump({"corpus": a.corpus, "campaigns": len(eps),
                   "per_tactic": per_tactic,
                   "rules": len(all_rules), "full_deployment": full,
                   "per_rule": [{"rule": r, **v} for v, r in singles],
                   "greedy": [{"k": k, "rule": r, **v} for k, r, v in greedy]},
                  f, indent=1)
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
