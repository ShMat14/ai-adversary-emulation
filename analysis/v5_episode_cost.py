# -*- coding: utf-8 -*-
"""Where does a winning episode spend its steps?

The nearest comparator, E-NASim, converges to roughly ten-step episodes; our
masked policy takes about thirty-three. Reported side by side that reads as an
efficiency deficit, and a reviewer will read it that way unless we decompose it.
This script does the decomposition, and it does it on three axes:

  1. What a step *is*. Every step in a winning episode is classified as
     productive (it executed and changed the world), a failed roll, an
     execution-time refusal, or redundant (it executed and changed nothing).

  2. What tactic a step serves. Defense Evasion has no analogue in an
     environment with no defender, so any share spent there is not comparable
     work -- it is a cost our task carries and theirs does not.

  3. What the task itself costs. A random-legal walk that reaches the same
     objective gives a floor: the number of steps the *engagement* requires
     before any question of policy quality arises.

Together those say whether thirty-three is a slow policy or a long task.

    python analysis/v5_episode_cost.py
    python analysis/v5_episode_cost.py --episodes 200 --topology enterprise
"""
import argparse
import collections
import io
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.v5_train import load, make_env
from env.kill_chain_v5 import TECHNIQUES_V5

OUT = "results/v5_episode_cost.json"


def classify(info):
    """One step of a winning episode, in the four ways it can be spent."""
    if not info["success"]:
        return "refused" if info["result"] == "perm_err" else "failed"
    return "productive" if info["advanced"] else "redundant"


def decompose(models, topology="enterprise", episodes=200, max_steps=60,
              algo="maskable", base_seed=770_000):
    """Roll out each policy and account for every step of every win."""
    from sb3_contrib.common.maskable.utils import get_action_masks

    kinds = collections.Counter()
    causes = collections.Counter()
    tactics = collections.Counter()
    lengths, evasion_shares = [], []
    wins = attempts = 0

    for mp in models:
        model = load(algo, mp)
        env = make_env(topology, seed=0, max_steps=max_steps)
        for ep in range(episodes):
            obs, _ = env.reset(seed=base_seed + ep)
            steps, done = [], False
            while not done:
                a, _ = model.predict(obs, action_masks=get_action_masks(env),
                                     deterministic=True)
                obs, _r, term, trunc, info = env.step(int(a))
                steps.append(info)
                done = term or trunc
            attempts += 1
            if not env.model.is_goal():
                continue                      # only winning episodes are costed
            wins += 1
            lengths.append(len(steps))
            ev = 0
            for info in steps:
                kinds[classify(info)] += 1
                if not info["success"]:
                    causes[info["result"]] += 1
                tac = TECHNIQUES_V5[info["technique"]].tactic
                tactics[tac] += 1
                if tac == "Defense Evasion":
                    ev += 1
            evasion_shares.append(ev / len(steps))

    total = sum(kinds.values())
    return {
        "topology": topology,
        "models": len(models),
        "episodes_per_model": episodes,
        "attempts": attempts,
        "wins": wins,
        "mean_winning_length": statistics.mean(lengths) if lengths else 0.0,
        "median_winning_length": statistics.median(lengths) if lengths else 0.0,
        "steps_costed": total,
        "kinds": {k: {"count": v, "share": v / total} for k, v in kinds.most_common()},
        "failure_causes": {k: {"count": v, "share": v / total}
                           for k, v in causes.most_common()},
        "tactics": {k: {"count": v, "share": v / total} for k, v in tactics.most_common()},
        "mean_evasion_share": statistics.mean(evasion_shares) if evasion_shares else 0.0,
    }


def random_legal_floor(topology="enterprise", trials=40, max_steps=400, seed=31,
                       defender=True):
    """How many steps does the engagement itself take, policy quality aside?

    A uniform draw from the legal set is the weakest agent that never selects an
    infeasible action. Where it first reaches the objective is a ceiling on how
    hard the task is to *find*, not a lower bound on the optimal path, so we
    report it as the floor a policy must beat rather than as an optimum.
    """
    import numpy as np
    env = make_env(topology, seed=0, max_steps=max_steps)
    if not defender:
        # Every other mechanic is untouched -- the detector still logs, suspicion
        # still accumulates, the episode simply never ends on an incident. So the
        # two measurements differ in the defender and in nothing else.
        env.model.caught = lambda: False
    rng = random.Random(seed)
    solved = []
    for t in range(trials):
        env.reset(seed=seed * 1000 + t)
        for n in range(1, max_steps + 1):
            mask = np.asarray(env.action_masks(), dtype=bool)
            legal = [i for i, ok in enumerate(mask) if ok]
            if not legal:
                break
            _o, _r, term, trunc, _i = env.step(rng.choice(legal))
            if env.model.is_goal():
                solved.append(n)
                break
            if term or trunc:
                break
    return {"trials": trials, "solved": len(solved),
            "median": statistics.median(solved) if solved else None,
            "min": min(solved) if solved else None,
            "max": max(solved) if solved else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="enterprise")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--models", default="results/models/v5")
    ap.add_argument("--trials", type=int, default=40)
    a = ap.parse_args()

    models = sorted(os.path.join(a.models, f[:-4])
                    for f in os.listdir(a.models)
                    if f.startswith(f"masked_{a.topology}_s") and f.endswith(".zip"))
    if not models:
        raise SystemExit(f"no masked_{a.topology}_s* checkpoints under {a.models}")

    print(f"costing {len(models)} policies x {a.episodes} episodes on {a.topology}")
    d = decompose(models, a.topology, a.episodes)
    d["random_legal_floor"] = random_legal_floor(a.topology, a.trials)
    d["random_legal_floor_no_defender"] = random_legal_floor(
        a.topology, a.trials, defender=False)

    print(f"\nwins {d['wins']}/{d['attempts']}   "
          f"mean winning length {d['mean_winning_length']:.1f} steps")
    print("\n  how a step is spent")
    for k, v in d["kinds"].items():
        print(f"    {k:<12} {v['share']*100:5.1f}%   ({v['count']})")
    print("\n  what a step serves")
    for k, v in list(d["tactics"].items())[:8]:
        print(f"    {k:<22} {v['share']*100:5.1f}%   ({v['count']})")
    print("\n  why a step failed")
    for k, v in d["failure_causes"].items():
        print(f"    {k:<12} {v['share']*100:5.1f}%   ({v['count']})")
    for label, key in (("with defender", "random_legal_floor"),
                       ("no defender  ", "random_legal_floor_no_defender")):
        f = d[key]
        print(f"\n  random-legal floor, {label}: solved {f['solved']}/{f['trials']}, "
              f"median {f['median']} steps, min {f['min']}")

    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
