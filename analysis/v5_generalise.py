# -*- coding: utf-8 -*-
"""Zero-shot generalisation across estates of the same size but different shape.

Every result in Section 4 comes from one twelve-host estate, which is the
paper's narrowest limitation. This measures whether a policy trained on that
estate does anything sensible on estates it has never seen.

The three test estates hold exactly twelve hosts each, which is deliberate. The
observation is 196 dimensions for any twelve-host estate, so a policy transfers
without retraining or reshaping, and whatever changes between them is *shape*
rather than size. A drop in success therefore cannot be blamed on the network
simply being bigger.

  enterprise  5 zones, tiered, one lateral shortcut   (trained on)
  flat12      2 zones; once inside, nearly all adjacent
  hub12       hub and spoke through a single jump host
  deep12      6 zones of two hosts in a line

    python analysis/v5_generalise.py --episodes 150
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from analysis.v5_train import load, make_env, rollout, OUT

ESTATES = ["enterprise", "flat12", "hub12", "deep12"]
NOTE = {
    "enterprise": "trained on; 5 zones, tiered",
    "flat12": "2 zones, almost flat",
    "hub12": "hub and spoke, one chokepoint",
    "deep12": "6 zones in a line, deepest",
}


def random_legal(topology, episodes, max_steps, seed=2026):
    """The floor: uniform choice among legal actions, on this estate."""
    import random
    env = make_env(topology, seed=0, max_steps=max_steps)
    mm = env.model
    rng = random.Random(seed)
    wins = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=700_000 + ep)
        done = False
        while not done:
            legal = [i for i, m in enumerate(env.action_masks()) if m]
            obs, r, term, trunc, _ = env.step(rng.choice(legal))
            done = term or trunc
        wins += mm.is_goal()
    return 100.0 * wins / episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--json", default="results/v5_generalisation.json")
    a = ap.parse_args()

    ckpts = sorted(glob.glob(os.path.join(OUT, "masked_enterprise_s*.zip")))
    ckpts = [c[:-4] for c in ckpts if c[:-4].split("_s")[-1].isdigit()]
    print(f"{len(ckpts)} policies trained on 'enterprise', evaluated zero-shot "
          f"on {len(ESTATES)} estates, {a.episodes} episodes each\n")

    out = {}
    for est in ESTATES:
        rows = []
        for c in ckpts:
            model = load("maskable", c)
            env = make_env(est, seed=0, max_steps=a.max_steps)
            s, i, d, L, red, adv, apd, wl, _ = rollout(
                model, env, "maskable", a.episodes, full=True)
            rows.append({"seed": int(c.split("_s")[-1]), "success": s,
                         "illegal": i, "detection": d, "ep_len": L, "apd": apd})
        base = random_legal(est, min(400, a.episodes * 2), a.max_steps)
        out[est] = {
            "note": NOTE[est],
            "n_policies": len(rows),
            "success_mean": float(np.mean([r["success"] for r in rows])),
            "success_sd": float(np.std([r["success"] for r in rows])),
            "illegal_mean": float(np.mean([r["illegal"] for r in rows])),
            "detection_mean": float(np.mean([r["detection"] for r in rows])),
            "ep_len_mean": float(np.mean([r["ep_len"] for r in rows])),
            "apd_mean": float(np.mean([r["apd"] for r in rows])),
            "random_legal": base,
            "per_seed": rows,
        }
        v = out[est]
        print(f"{est:11s} {v['success_mean']:5.1f} ± {v['success_sd']:4.1f}%   "
              f"illegal {v['illegal_mean']:4.1f}%   detected {v['detection_mean']:5.1f}%   "
              f"steps {v['ep_len_mean']:5.1f}   random-legal {base:4.1f}%")

    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.json}")

    trained = out["enterprise"]["success_mean"]
    print("\nretention against the estate the policies were trained on:")
    for est in ESTATES[1:]:
        v = out[est]
        print(f"   {est:9s} {100*v['success_mean']/max(1e-9,trained):5.1f}% of "
              f"trained-estate success, against a random-legal floor of "
              f"{v['random_legal']:.1f}%")


if __name__ == "__main__":
    main()
