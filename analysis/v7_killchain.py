# -*- coding: utf-8 -*-
"""Read the detection-latency instrument through the Cyber Kill Chain.

The paper used "kill chain" only in its generic sense, a sequence of techniques
whose preconditions must be met in order. This relates the catalogue to the
seven phases of Hutchins, Cloppert and Amin (2011) and asks one question the
corpus can answer: does a learned campaign move through those phases in order?

THE MAPPING, AND WHY IT IS AT THE TACTIC LEVEL

The Cyber Kill Chain predates ATT&CK and no official correspondence exists. We
use the tactic-level assignment most often used in practice, so that a reader
can check every row against the ATT&CK matrix rather than against our judgement
technique by technique. Its known weakness is stated in the paper rather than
patched here: two Credential Access techniques (brute force and password
spraying) are used in this environment to gain entry, which in kill-chain terms
is Delivery, but the tactic-level rule files them under Actions on Objectives.

WHAT IS MEASURED

For each phase, every rule watching a technique of that phase is deployed
together, and we report the median step at which that rule set first has
something to fire on, over the campaigns that reached the objective. This is
`latency()` from v5_coverage.py unchanged, so the numbers are directly
comparable with Table 14 and Fig. 11. Nothing here is stochastic.

    python analysis/v7_killchain.py
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.detection import PROFILES
from env.kill_chain_v5 import TECHNIQUES_V5
from analysis.v5_coverage import ZONE_NAME, latency, load_episodes

# Hutchins et al. (2011), in their order.
PHASES = [
    "Reconnaissance",
    "Weaponization",
    "Delivery",
    "Exploitation",
    "Installation",
    "Command and Control",
    "Actions on Objectives",
]

TACTIC_TO_PHASE = {
    "Initial Access": "Delivery",
    "Execution": "Exploitation",
    "Persistence": "Installation",
    "Privilege Escalation": "Installation",
    "Defense Evasion": "Installation",
    "Command and Control": "Command and Control",
    "Discovery": "Actions on Objectives",
    "Credential Access": "Actions on Objectives",
    "Lateral Movement": "Actions on Objectives",
    "Collection": "Actions on Objectives",
    "Exfiltration": "Actions on Objectives",
    "Impact": "Actions on Objectives",
}

# Why the first two phases are empty, in the environment's own terms.
NOT_MODELLED = {
    "Reconnaissance": "network scanning needs a compromised host, so no "
                      "reconnaissance from outside the estate is modelled",
    "Weaponization": "precedes the engagement; nothing to observe on the estate",
}


def main():
    eps = load_episodes("results/telemetry/v5", "enterprise")

    # Every tactic in the catalogue must have a phase, or the table would
    # silently drop techniques.
    tactics = {t.tactic for t in TECHNIQUES_V5.values()}
    unmapped = tactics - set(TACTIC_TO_PHASE)
    assert not unmapped, f"tactics without a phase: {unmapped}"

    rows = []
    for phase in PHASES:
        tacs = [t for t, p in TACTIC_TO_PHASE.items() if p == phase]
        techs = [n for n, t in TECHNIQUES_V5.items() if t.tactic in tacs]
        per_tactic = {t: sum(1 for x in TECHNIQUES_V5.values() if x.tactic == t)
                      for t in tacs}
        row = {"phase": phase, "tactics": per_tactic, "techniques": len(techs)}
        if techs:
            rules = {PROFILES[n].rule for n in techs}
            v = latency(eps, rules)
            row.update(rules=len(rules), **v)
        else:
            row["not_modelled"] = NOT_MODELLED[phase]
        rows.append(row)

    assert sum(r["techniques"] for r in rows) == len(TECHNIQUES_V5) == 45

    print(f"{len(eps)} campaigns that reached the objective\n")
    print(f"{'phase':23} {'tech':>4}  {'median step':>11}  {'% elapsed':>9}  zone held")
    for r in rows:
        if "median_step" in r:
            print(f"{r['phase']:23} {r['techniques']:>4}  {r['median_step']:>11.1f}  "
                  f"{r['pct_campaign_elapsed']:>8.1f}%  "
                  f"{ZONE_NAME[r['median_zone_reached']]}")
        else:
            print(f"{r['phase']:23} {r['techniques']:>4}  {'—':>11}  {'—':>9}  "
                  f"({r['not_modelled']})")

    # Does the learned campaign respect the phase order? Compare each phase's
    # first opportunity with the phase before it.
    seen = [r for r in rows if "median_step" in r]
    inversions = [(a["phase"], b["phase"]) for a, b in zip(seen, seen[1:])
                  if b["median_step"] < a["median_step"]]
    print("\nphase-order inversions (a later phase first seen earlier):")
    for a, b in inversions:
        print(f"   {b} before {a}")
    if not inversions:
        print("   none")

    out = {"campaigns": len(eps), "mapping": TACTIC_TO_PHASE,
           "phases": rows, "inversions": inversions}
    with open("results/v7_killchain.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nwrote results/v7_killchain.json")


if __name__ == "__main__":
    main()
