# -*- coding: utf-8 -*-
"""Zero-shot transfer across estate SIZE, not just shape.

Section 4.7 transfers across shape while holding the host count at twelve,
because the observation width otherwise follows the host count and a policy is
tied to the size it trained on. This study removes that constraint using the
size-invariant mode of the environment: the estate is laid into a fixed number
of slots, each carrying a present/absent flag, and every action addressing an
absent slot is masked. One policy then runs on estates of eight, twelve, sixteen
and twenty hosts without retraining or reshaping.

Policies are trained on the twelve-host estate only. Everything else is unseen.

    python analysis/v5_scale.py --episodes 150
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from analysis.v5_train import load, rollout
from env.adversary_env_v5 import AdversaryEnvV5

PAD = 20
ESTATES = [("small8", 8), ("enterprise", 12), ("mid16", 16), ("large20", 20)]


def mk(topo, seed=0, max_steps=60):
    return AdversaryEnvV5({"topology": topo, "max_steps": max_steps,
                           "pad_hosts": PAD, "seed": seed})


def random_legal(topo, episodes, max_steps=60, seed=2026):
    env = mk(topo, 0, max_steps)
    mm = env.model
    rng = random.Random(seed)
    wins = 0
    for ep in range(episodes):
        env.reset(seed=700_000 + ep)
        done = False
        while not done:
            legal = [i for i, m in enumerate(env.action_masks()) if m]
            _, _, term, trunc, _ = env.step(rng.choice(legal))
            done = term or trunc
        wins += mm.is_goal()
    return 100.0 * wins / episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--json", default="results/v5_scale.json")
    a = ap.parse_args()

    ck = sorted(glob.glob("results/models/v5_pad/masked_pad_s*.zip"))
    ck = [c[:-4] for c in ck]
    if not ck:
        print("no size-invariant checkpoints in results/models/v5_pad/")
        raise SystemExit(1)
    print(f"{len(ck)} policies trained on the 12-host estate, evaluated on "
          f"{len(ESTATES)} sizes, {a.episodes} engagements each\n")

    out = {}
    for topo, n in ESTATES:
        rows = []
        for c in ck:
            model = load("maskable", c)
            s, i, d, L, red, adv, apd, wl, _ = rollout(
                model, mk(topo, 0, a.max_steps), "maskable", a.episodes, full=True)
            rows.append({"seed": int(c.split("_s")[-1]), "success": s,
                         "illegal": i, "detection": d, "ep_len": L, "apd": apd})
        base = random_legal(topo, min(300, a.episodes * 2), a.max_steps)
        out[topo] = {
            "hosts": n, "n_policies": len(rows),
            "success_mean": float(np.mean([r["success"] for r in rows])),
            "success_sd": float(np.std([r["success"] for r in rows])),
            "illegal_mean": float(np.mean([r["illegal"] for r in rows])),
            "detection_mean": float(np.mean([r["detection"] for r in rows])),
            "ep_len_mean": float(np.mean([r["ep_len"] for r in rows])),
            "random_legal": base, "per_seed": rows,
        }
        v = out[topo]
        tag = " (trained on)" if topo == "enterprise" else ""
        print(f"{topo:11s} {n:>2} hosts  {v['success_mean']:5.1f} ± "
              f"{v['success_sd']:4.1f}%   illegal {v['illegal_mean']:4.1f}%   "
              f"detected {v['detection_mean']:5.1f}%   steps {v['ep_len_mean']:5.1f}   "
              f"random-legal {base:4.1f}%{tag}")

    os.makedirs("results", exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.json}")

    trained = out["enterprise"]["success_mean"]
    print("\nretention against the size trained on:")
    for topo, n in ESTATES:
        if topo == "enterprise":
            continue
        v = out[topo]
        print(f"   {topo:9s} ({n:>2} hosts) {100*v['success_mean']/max(1e-9,trained):5.1f}% "
              f"of trained-size success, floor {v['random_legal']:.1f}%")


if __name__ == "__main__":
    main()
