# -*- coding: utf-8 -*-
"""The scripted-playbook baseline: what an authoring-time tool achieves here.

The introduction claims that Caldera, Atomic Red Team and Metasploit playbooks
are fixed at authoring time, so a patched vulnerability, a rotated credential or
a withdrawn technique silently invalidates them. That is the standard criticism
and we had asserted it rather than measured it, which left the paper comparing
itself only against other learned agents and never against the tool a red team
actually uses.

This measures it. A playbook here is an ordered list of (technique, host)
intentions written once, against the estate as the author understood it. Three
variants, in increasing order of how much engineering effort an author puts in:

  fixed        one hand-written chain, executed in order. If a step is not
               currently legal it is attempted anyway and fails, exactly as a
               script that does not check state would.
  guarded      the same chain, but each step is skipped when its preconditions
               do not hold. This is a playbook whose author added state checks,
               which is the best case for an authoring-time tool.
  retry        guarded, plus up to three retries per step, since a real operator
               reruns a step that failed on a dice roll.

None of the three adapts: the ordering is fixed before the episode begins and
does not depend on what the engagement offers. That is the property under test.

    python analysis/v5_playbook.py --episodes 300
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.v5_train import make_env
from env.kill_chain_v5 import TECHNIQUES_V5

# A competent hand-written chain for the twelve-host estate, of the kind an
# operator would author after reconnaissance: break in at the DMZ, enumerate,
# escalate, harvest, then work inward zone by zone to the domain controller.
PLAYBOOK = [
    ("PHISHING_EMAIL", "mail01"),
    ("SYSTEM_INFO_DISCOVERY", "mail01"),
    ("PRIV_ESC_SUDO", "mail01"),
    ("CREDS_IN_FILES", "mail01"),
    ("NETWORK_SCAN", "mail01"),
    ("LATERAL_MOVE_SMB", "user01"),
    ("SYSTEM_INFO_DISCOVERY", "user01"),
    ("POWERSHELL_EXEC", "user01"),
    ("CREDS_IN_FILES", "user01"),
    ("DOMAIN_ACCT_DISCOVERY", "user01"),
    ("LATERAL_MOVE_SMB", "srv01"),
    ("SYSTEM_INFO_DISCOVERY", "srv01"),
    ("POWERSHELL_EXEC", "srv01"),
    ("CRED_DUMP_LSASS", "srv01"),
    ("LATERAL_MOVE_SMB", "admin01"),
    ("SYSTEM_INFO_DISCOVERY", "admin01"),
    ("POWERSHELL_EXEC", "admin01"),
    ("CREDS_IN_FILES", "admin01"),
    ("DATA_FROM_LOCAL_SYSTEM", "admin01"),
    ("ARCHIVE_COLLECTED_DATA", "admin01"),
    ("C2_CHANNEL_ESTABLISH", "admin01"),
    ("INSTALL_BACKDOOR", "admin01"),
    ("LATERAL_MOVE_SMB", "dc01"),
    ("SYSTEM_INFO_DISCOVERY", "dc01"),
    ("POWERSHELL_EXEC", "dc01"),
    ("RANSOMWARE_ENCRYPT", "dc01"),
]


def run(mode, episodes, max_steps, topology="enterprise", base_seed=900_000):
    env = make_env(topology, seed=0, max_steps=max_steps)
    mm = env.model
    order = list(TECHNIQUES_V5)
    hosts = list(mm.topo.hosts)

    def encode(tech, host):
        return order.index(tech) * len(hosts) + hosts.index(host)

    wins = caught = illegal = steps = skipped = 0
    lens, offered = [], []
    for ep in range(episodes):
        env.reset(seed=base_seed + ep)
        # how much of the playbook this engagement even offers
        offered.append(100.0 * sum(1 for tech, _ in PLAYBOOK
                                   if mm.offers(tech)) / len(PLAYBOOK))
        done, n = False, 0
        for tech, host in PLAYBOOK:
            if done:
                break
            tries = 3 if mode == "retry" else 1
            for _ in range(tries):
                if done:
                    break
                a = encode(tech, host)
                mask = env.action_masks()
                if mode in ("guarded", "retry") and not mask[a]:
                    skipped += 1
                    break                      # author checked state; move on
                if not mask[a]:
                    illegal += 1
                _, _, term, trunc, info = env.step(a)
                n += 1
                steps += 1
                done = term or trunc
                if info.get("advanced"):
                    break                      # step worked, go to the next
        lens.append(n)
        wins += mm.is_goal()
        caught += mm.caught()
    return {
        "mode": mode,
        "success": 100.0 * wins / episodes,
        "detected": 100.0 * caught / episodes,
        "illegal": 100.0 * illegal / max(1, steps),
        "skipped_per_episode": skipped / episodes,
        "steps": statistics.mean(lens),
        "playbook_offered": statistics.mean(offered),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--topology", default="enterprise")
    ap.add_argument("--json", default="results/v5_playbook.json")
    a = ap.parse_args()

    rows = [run(m, a.episodes, a.max_steps, a.topology)
            for m in ("fixed", "guarded", "retry")]

    hdr = (f"{'playbook variant':18}{'success %':>11}{'infeasible %':>14}"
           f"{'detected %':>12}{'steps':>8}{'skipped/ep':>12}")
    print(f"\nscripted playbook on {a.topology}, {a.episodes} engagements, "
          f"{a.max_steps}-step budget")
    print(f"the chain has {len(PLAYBOOK)} steps; on average "
          f"{rows[0]['playbook_offered']:.1f}% of them are offered by the engagement\n")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['mode']:18}{r['success']:>10.1f}%{r['illegal']:>13.1f}%"
              f"{r['detected']:>11.1f}%{r['steps']:>8.1f}{r['skipped_per_episode']:>12.1f}")

    os.makedirs("results", exist_ok=True)
    with open(a.json, "w") as f:
        json.dump({"episodes": a.episodes, "topology": a.topology,
                   "max_steps": a.max_steps, "playbook_length": len(PLAYBOOK),
                   "rows": rows}, f, indent=1)
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
