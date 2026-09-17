# -*- coding: utf-8 -*-
"""What is the policy actually doing? Aggregate metrics do not say.

Twice now a failure here was diagnosable only from behaviour. A policy sitting at
43% looked merely mediocre; the trace showed it doing real work for nineteen
steps and then farming one repeatable no-op for the remaining forty. A policy at
0% success with 0% detection is not failing to find the objective -- it is
declining to take any action loud enough to matter. Those two need opposite
fixes, and no summary statistic separates them.

    python analysis/v5_trace.py results/models/v5/masked_enterprise_s0
    python analysis/v5_trace.py <model> --episodes 40 --show 2
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.v5_train import load, make_env
from env.kill_chain_v5 import TECHNIQUES_V5


def trace(model_path, topology="enterprise", episodes=40, max_steps=60,
          show=2, algo="maskable", base_seed=900_000):
    from sb3_contrib.common.maskable.utils import get_action_masks
    model = load(algo, model_path)
    env = make_env(topology, seed=0, max_steps=max_steps)
    mm = env.model

    picks = collections.Counter()
    tactics = collections.Counter()
    chain_pos = collections.defaultdict(list)
    results = collections.Counter()
    wins = 0
    first_repeat_step = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        req = [k for k, v in (("persist", mm.need_persist),
                              ("hostpriv", mm.need_hostpriv),
                              ("collect", mm.need_collect),
                              ("c2", mm.need_c2)) if v]
        lines, seen, done, n, stall_from = [], set(), False, 0, None
        seq = []
        while not done:
            a, _ = model.predict(obs, action_masks=get_action_masks(env),
                                 deterministic=True)
            tech, tgt = mm.decode(int(a))
            obs, r, term, trunc, info = env.step(int(a))
            n += 1
            picks[f"{tech}@{tgt}"] += 1
            tactics[TECHNIQUES_V5[tech].tactic] += 1
            results[info["result"]] += 1
            key = (tech, tgt)
            seq.append(key)
            if key in seen and stall_from is None:
                stall_from = n
            seen.add(key)
            lines.append(f"   {n:>3}  {tech:<28} {tgt:<9} "
                         f"{info['result']:<9} adv={str(info['advanced']):<5} "
                         f"r={r:+7.2f}  susp={mm.detection.suspicion:.2f}")
            done = term or trunc
        won = mm.is_goal()
        wins += won
        if won and n > 1:
            for j, (tk, _) in enumerate(seq):
                chain_pos[TECHNIQUES_V5[tk].tactic].append(j / (n - 1))
        if stall_from:
            first_repeat_step.append(stall_from)
        if ep < show:
            print(f"\n--- episode {ep}  requires {req or ['nothing extra']}  "
                  f"-> {'REACHED OBJECTIVE' if won else 'failed'} in {n} steps"
                  f"{'  (incident declared)' if mm.caught() else ''}")
            for L in lines:
                print(L)

    print(f"\nover {episodes} episodes: {100*wins/episodes:.1f}% reached the objective")

    # A catalogue can be well-formed and still be used incoherently, so this
    # reports WHERE in a winning episode each tactic is actually used: the mean
    # position of every action of that tactic, scaled to [0, 1] across the
    # episode. A real kill chain reads top to bottom.
    if chain_pos:
        print("")
        print("where each tactic falls in a winning episode (0 = first step, 1 = last)")
        for tac, xs in sorted(chain_pos.items(),
                              key=lambda kv: sum(kv[1]) / len(kv[1])):
            m = sum(xs) / len(xs)
            bar = "-" * int(round(m * 40))
            print(f"   {m:4.2f}  {tac:22s} |{bar}o  ({len(xs)} actions)")
    print("\nmost-selected (technique @ host)")
    for k, v in picks.most_common(12):
        print(f"   {v:>5}  {k}")
    print("\nby tactic")
    total = sum(tactics.values())
    for k, v in tactics.most_common():
        print(f"   {100*v/total:5.1f}%  {k}")
    print("\nexecution result (E-NASim Eq. 7)")
    for k, v in results.most_common():
        print(f"   {100*v/total:5.1f}%  {k}")
    if first_repeat_step:
        print(f"\nfirst repeated (technique, host) pair at step "
              f"{sum(first_repeat_step)/len(first_repeat_step):.1f} on average "
              f"-- a policy that starts repeating early is stalling, not searching")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--topology", default="enterprise")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--algo", default="maskable")
    a = ap.parse_args()
    trace(a.model, a.topology, a.episodes, a.max_steps, a.show, a.algo)
