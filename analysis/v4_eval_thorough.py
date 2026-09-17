# -*- coding: utf-8 -*-
"""
Thorough evaluation: is the model great on ALL kill chains?

Overall success rate can hide a weak path. This breaks the evaluation down by
the scenario the agent faces and by the route it takes, so "great accuracy on
all kill chains" can be checked rather than assumed:

  * by entry scenario  — episodes where a SQL-injectable web app is present vs
    episodes where it is not and the agent must phish or spray in. The agent
    must be reliable in both.
  * detection          — the stealthy agent should rarely be caught.
  * route coverage      — which domain-dominance route (Kerberoast / DCSync /
    GPO / Golden Ticket) and which entry the agent actually exercises, so we can
    see the AD kill chain is reachable, not just one lucky path.

Exit status is 0 only if every scenario clears the bar (default: success >= 99%
in each), so this doubles as the gate for the train-until-great loop.

    python analysis/v4_eval_thorough.py --model results/models/v4/masked_s0
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sb3_contrib import MaskablePPO
from env.adversary_env import AdversaryEnv
from env.kill_chain import ACTION_ORDER

# The kill chain grouped by phase, so we can report which route the model takes
# at each stage and compare them.
PHASES = {
    "Entry": ["PHISHING_EMAIL", "SQL_INJECTION", "PASSWORD_SPRAYING", "BRUTE_FORCE_SSH"],
    "Discovery": ["NETWORK_SCAN", "DOMAIN_ACCT_DISCOVERY", "DOMAIN_TRUST_DISCOVERY"],
    "Credential access": ["VALID_ACCOUNTS_LOGIN", "AS_REP_ROASTING", "CRED_DUMP_LSASS"],
    "Lateral movement": ["LATERAL_MOVE_SMB", "PASS_THE_HASH", "PASS_THE_TICKET"],
    "DC dominance": ["KERBEROASTING", "DCSYNC", "GPO_MODIFICATION", "GOLDEN_TICKET"],
    "Persistence": ["INSTALL_BACKDOOR", "WEB_SHELL_UPLOAD"],
    "Host priv-esc": ["PRIV_ESC_SUDO", "POWERSHELL_EXEC", "ACCOUNT_MANIPULATION"],
    "Evasion": ["CLEAR_LOGS"],
    "Impact": ["EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT"],
}
DC_ROUTES = set(PHASES["DC dominance"])
ENTRIES = set(PHASES["Entry"])


class _Null:
    def start_episode(self): pass
    def log_event(self, *a, **k): pass
    def end_episode(self): pass


def eval_scenario(model, sqli, episodes, seed0):
    """Force sqli_available to a fixed value; report success/detection and the
    full per-technique and per-phase usage so every kill chain can be compared."""
    env = AdversaryEnv(config={"max_steps": 40, "sqli_available": sqli})
    env.telemetry = _Null()
    succ = det = steps = 0
    tech = Counter()          # every technique, how many times chosen
    dc_route = Counter()      # which DC-dominance route finished the chain
    entry_used = Counter()    # which entry opened the chain
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        used = []
        while not done:
            legal = env.action_masks()
            a, _ = model.predict(obs, action_masks=legal, deterministic=True)
            a = int(a)
            name = ACTION_ORDER[a]
            used.append(name)
            tech[name] += 1
            steps += 1
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
        if env.model.is_goal():
            succ += 1
        if env.model.state.any_host_detected():
            det += 1
        for t in used:
            if t in DC_ROUTES and env.model.host("dc01").privileged:
                dc_route[t] += 1
                break
        for t in used:
            if t in ENTRIES:
                entry_used[t] += 1
                break
    # roll technique counts up into phases
    phase_counts = {ph: {t: tech.get(t, 0) for t in techs if tech.get(t, 0)}
                    for ph, techs in PHASES.items()}
    return {
        "episodes": episodes,
        "success": 100.0 * succ / episodes,
        "detection": 100.0 * det / episodes,
        "avg_steps": steps / episodes,
        "dc_routes": dict(dc_route),
        "entries": dict(entry_used),
        "phase_counts": phase_counts,
        "tech": dict(tech),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/models/v4/masked_s0")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--bar", type=float, default=99.0)
    a = ap.parse_args()

    model = MaskablePPO.load(a.model, device="cpu")
    print(f"thorough evaluation of {a.model}")
    print("=" * 66)
    scenarios = {
        "SQLi available (web entry)": eval_scenario(model, True, a.episodes, 600_000),
        "no SQLi (phish/spray entry)": eval_scenario(model, False, a.episodes, 700_000),
    }

    all_ok = True
    for name, r in scenarios.items():
        ok = r["success"] >= a.bar
        all_ok = all_ok and ok
        flag = "ok  " if ok else "WEAK"
        print(f"\n[{flag}] {name}")
        print(f"     success {r['success']:.1f}%   detection {r['detection']:.1f}%   "
              f"avg steps {r['avg_steps']:.1f}   ({r['episodes']} eps)")

    # ── per-phase kill-chain comparison across both scenarios ──
    print("\n" + "=" * 66)
    print("KILL-CHAIN USAGE — technique picks per phase (both scenarios combined)")
    print("=" * 66)
    combined = Counter()
    for r in scenarios.values():
        combined.update(r["tech"])
    total_actions = sum(combined.values()) or 1
    for ph, techs in PHASES.items():
        rows = [(t, combined.get(t, 0)) for t in techs if combined.get(t, 0)]
        if not rows:
            continue
        print(f"\n  {ph}:")
        for t, c in sorted(rows, key=lambda kv: -kv[1]):
            bar = "#" * int(40 * c / max(combined.values()))
            print(f"    {t:24s} {c:6d}  {100*c/total_actions:4.1f}%  {bar}")

    # ── which route completed the chain, per scenario ──
    print("\n" + "=" * 66)
    print("ROUTE COMPARISON — how each scenario opened and finished the chain")
    print("=" * 66)
    print(f"  {'scenario':30s}{'entry mix':>0}")
    for name, r in scenarios.items():
        print(f"\n  {name}")
        print(f"     entry     : {r['entries']}")
        print(f"     DC route  : {r['dc_routes']}")

    # ── adaptation: does the model change its kill chain with the scenario? ──
    # A model that "thinks" uses the web-app entry when SQLi is present and
    # switches to phishing/spraying when it is not. If the entry mix is the same
    # in both, the policy is rigid (a memorised script), not adaptive.
    print("\n" + "=" * 66)
    print("ADAPTATION — does the model pick the kill chain for the scenario?")
    print("=" * 66)
    sc = list(scenarios.items())
    (n1, r1), (n2, r2) = sc[0], sc[1]

    def frac(d):
        tot = sum(d.values()) or 1
        return {k: v / tot for k, v in d.items()}

    e1, e2 = frac(r1["entries"]), frac(r2["entries"])
    keys = sorted(set(e1) | set(e2))
    print(f"  {'entry technique':24s}{n1[:18]:>20}{n2[:18]:>20}")
    for k in keys:
        print(f"  {k:24s}{100*e1.get(k,0):>19.0f}%{100*e2.get(k,0):>19.0f}%")
    # L1 distance between the two entry distributions: 0 = identical (rigid),
    # ~2 = completely different (fully adaptive)
    div = sum(abs(e1.get(k, 0) - e2.get(k, 0)) for k in keys)
    sqli_when_avail = e1.get("SQL_INJECTION", 0) if "SQLi" in n1 else e2.get("SQL_INJECTION", 0)
    sqli_when_absent = e2.get("SQL_INJECTION", 0) if "SQLi" in n1 else e1.get("SQL_INJECTION", 0)
    print(f"\n  entry-distribution divergence between scenarios: {div:.2f} / 2.00")
    print(f"  uses SQL injection when web app present : {100*sqli_when_avail:.0f}%")
    print(f"  uses SQL injection when it is absent    : {100*sqli_when_absent:.0f}%")
    adaptive = div > 0.3 and sqli_when_absent < 0.02
    print("  -> " + ("ADAPTIVE: the model switches its entry to fit the scenario."
                     if adaptive else
                     "RIGID: the model uses the same entry regardless — investigate."))

    print("\n" + "=" * 66)
    worst = min(s["success"] for s in scenarios.values())
    print(f"worst-scenario success: {worst:.1f}%   bar: {a.bar:.1f}%")
    if all_ok:
        print("GREAT — every kill chain clears the bar.")
    else:
        print("NOT YET — at least one kill chain is below the bar; train more.")

    # persist for tracking / plots
    import json
    with open("analysis/v4_thorough_result.json", "w") as f:
        json.dump({"scenarios": scenarios, "combined": dict(combined),
                   "worst_success": worst, "bar": a.bar, "great": all_ok}, f, indent=2)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
