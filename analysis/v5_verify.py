# -*- coding: utf-8 -*-
"""Verification of the environment's structural claims.

The paper's central claim is that the mask is *exactly* the precondition: it
permits every feasible action and no infeasible one. That is checked here over
every action in every state visited across many episodes -- roughly four million
(state, action) pairs -- rather than sampled, because a single disagreement in
either direction would invalidate the result. A mask that were merely
conservative would block feasible actions and quietly cap what any policy could
achieve; one that were permissive would let an infeasible action through and
make the 0% infeasible-selection figure meaningless.

Also checks the catalogue integrity that the comparison with E-NASim rests on:
45 techniques with 45 distinct ATT&CK identifiers, every one carrying a detection
profile, and no profile without a technique.

    python analysis/v5_verify.py
"""
import collections
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.v5_train import make_env
from env.detection import PROFILES
from env.kill_chain_v5 import TECHNIQUES_V5, TECHNIQUE_ORDER_V5
from env.topology_v5 import Topology


def check_catalogue():
    T = TECHNIQUES_V5
    ids = [t.mitre_id for t in T.values()]
    dup = [k for k, v in collections.Counter(ids).items() if v > 1]
    missing = [k for k in T if k not in PROFILES]
    ok = not dup and not missing and len(set(ids)) == len(T)
    print(f"  techniques                {len(T)}")
    print(f"  distinct ATT&CK ids       {len(set(ids))}")
    print(f"  duplicate ids             {dup or 'none'}")
    print(f"  ATT&CK tactics covered    {len(set(t.tactic for t in T.values()))}")
    print(f"  without detection profile {missing or 'none'}")
    print(f"  runtime-gated techniques  {sorted(k for k, v in T.items() if v.runtime_gate)}")
    return ok


def check_mask(topology="enterprise", episodes=300, seed=0):
    env = make_env(topology, seed=seed, max_steps=60)
    mm = env.model
    rng = random.Random(7)
    states = legal_offers = gated = false_pos = false_neg = 0
    n_actions = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=500_000 + ep)
        done = False
        while not done:
            # the raw mask, not the one padded for SB3 when nothing is legal
            mask = mm.action_mask_raw()
            n_actions = len(mask)
            states += 1
            for a in range(n_actions):
                tech, tgt = mm.decode(a)
                # the mask is mu(s, omega): the precondition intersected with
                # what this engagement offers. Comparing against the bare
                # precondition would report every withdrawn technique as a
                # wrongly blocked action.
                truth = (mm.offers(tech)
                         and bool(TECHNIQUES_V5[tech].precondition(mm, tgt)))
                if mask[a] and not truth:
                    false_pos += 1
                elif truth and not mask[a]:
                    false_neg += 1
                if mask[a]:
                    legal_offers += 1
                    g = TECHNIQUES_V5[tech].runtime_gate
                    if g is not None and g(mm, tgt):
                        gated += 1
            legal = [i for i, m in enumerate(mask) if m]
            obs, r, term, trunc, _ = env.step(rng.choice(legal) if legal else 0)
            done = term or trunc

    print(f"  states inspected               {states:,}")
    print(f"  (state, action) pairs checked  {states * n_actions:,}")
    print(f"  mask permits an infeasible act {false_pos}   <- must be 0")
    print(f"  mask blocks a feasible act     {false_neg}   <- must be 0")
    print(f"  mean legal actions per state   {legal_offers/states:.1f} of "
          f"{n_actions}  ({100*legal_offers/(states*n_actions):.1f}%)")
    print(f"  legal but runtime-denied       {gated:,} "
          f"({100*gated/max(1, legal_offers):.2f}% of legal offers)")
    print("    that last row is the PermErr gate working as designed: the action")
    print("    is feasible and selectable, and is refused at execution, so the")
    print("    agent can learn the requirement instead of watching it disappear")
    return false_pos == 0 and false_neg == 0


def check_solvable(topology="enterprise", engagements=150, walks=6, budget=1500):
    """Can every engagement still be won at all?

    Each episode withdraws a random subset of each substitution group. If a draw
    can leave the agent with no usable route, some episodes are unwinnable for
    reasons that have nothing to do with the policy, and every success rate
    reported is silently capped by the environment rather than by the agent.

    This is not hypothetical. The first implementation of the availability draw
    left 50 of 150 engagements unsolvable, because the only discovery and
    credential techniques that survived some draws were Windows-only while the
    foothold in the DMZ is Linux. The defender is switched off here and the
    budget is generous, so a failure means no path exists, not that the walk was
    unlucky.
    """
    import random
    import env.detection as D
    import env.kill_chain_v5 as K
    # kill_chain_v5 binds INCIDENT_THRESHOLD by value at import, so setting it
    # on the detection module alone leaves caught() using the original and the
    # walks die to the defender instead of exhausting the search.
    saved = (D.INCIDENT_THRESHOLD, K.INCIDENT_THRESHOLD)
    D.INCIDENT_THRESHOLD = K.INCIDENT_THRESHOLD = 1e9
    try:
        env = make_env(topology, seed=0, max_steps=budget)
        mm = env.model
        unsolved = []
        for ep in range(engagements):
            solved = False
            for attempt in range(walks):
                obs, _ = env.reset(seed=int(ep))
                rng = random.Random(ep * 100 + attempt)
                done = False
                while not done:
                    legal = [i for i, m in enumerate(env.action_masks()) if m]
                    if not legal:
                        break
                    obs, r, term, trunc, _ = env.step(rng.choice(legal))
                    if mm.is_goal():
                        solved = True
                        break
                    done = term or trunc
                if solved:
                    break
            if not solved:
                unsolved.append((ep, sorted(set(TECHNIQUE_ORDER_V5) - mm.available)))
    finally:
        D.INCIDENT_THRESHOLD, K.INCIDENT_THRESHOLD = saved
    print(f"  engagements tested             {engagements}")
    print(f"  no winning path exists         {len(unsolved)}   <- must be 0")
    for ep, w in unsolved[:5]:
        print(f"     episode {ep}: withdrawn {w}")
    return not unsolved


def check_scenarios(topology="enterprise", episodes=300):
    """The scenario really does vary what is on offer."""
    env = make_env(topology, seed=0, max_steps=60)
    sizes, seen = [], set()
    for i in range(episodes):
        env.reset(seed=i)
        sizes.append(len(env.model.available))
        seen.add(frozenset(env.model.available))
    print(f"  techniques offered per episode {min(sizes)}-{max(sizes)} of "
          f"{len(TECHNIQUES_V5)}")
    print(f"  distinct engagements seen      {len(seen)} in {episodes} episodes")
    return min(sizes) < len(TECHNIQUES_V5)


if __name__ == "__main__":
    print("catalogue")
    a = check_catalogue()
    print("\nmask exactness")
    b = check_mask()
    print("")
    print("scenario variation")
    c = check_scenarios()
    print("")
    print("solvability")
    d = check_solvable()
    topo = Topology("enterprise")
    print(f"\n  enterprise estate: {len(topo.hosts)} hosts, "
          f"{len(topo.subnets)} internal zones, "
          f"{sum(topo.is_windows(h) for h in topo.hosts)} Windows / "
          f"{sum(not topo.is_windows(h) for h in topo.hosts)} Linux, "
          f"{len(TECHNIQUES_V5)*len(topo.hosts)} actions")
    ok = a and b and c and d
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
