# -*- coding: utf-8 -*-
"""Compare the v4 method over a 15-technique and a 25-technique catalogue.

Everything except the technique count is held fixed: the same shared world
model, the same rule-based detection, the same per-episode scenario mechanism,
the same reward, the same budget, seeds and evaluation protocol. This isolates
catalogue size, which is what the supervisor asked to see.

    python analysis/v4_catalogue_compare.py
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from env.adversary_env import AdversaryEnv
from env.kill_chain import ACTION_ORDER
from analysis.v4_train import V3_CATALOGUE

OUT = "results/models/v4"
EPISODES = 400
SEEDS = [0, 1, 2]


class _Null:
    def __getattr__(self, _):
        return lambda *a, **k: None


def evaluate(path, masked, catalogue, seed, episodes=EPISODES):
    model = (MaskablePPO if masked else PPO).load(path, device="cpu")
    env = AdversaryEnv(config={"restrict_techniques": catalogue})
    env.telemetry = _Null()
    succ = det = illegal = steps = 0
    rewards, used = [], set()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 97 + ep + 500_000)
        done, total = False, 0.0
        while not done:
            legal = env.action_masks()
            if masked:
                a, _ = model.predict(obs, action_masks=legal, deterministic=True)
            else:
                a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            steps += 1
            if not legal[a]:
                illegal += 1
            else:
                used.add(ACTION_ORDER[a])
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
        rewards.append(total)
        if env.model.is_goal():
            succ += 1
        if env.model.state.any_host_detected():
            det += 1
    return {
        "success": 100.0 * succ / episodes,
        "detection": 100.0 * det / episodes,
        "illegal": 100.0 * illegal / steps,
        "reward": float(np.mean(rewards)),
        "techniques_used": len(used),
    }


def agg(rows, key):
    v = [r[key] for r in rows]
    return float(np.mean(v)), float(np.std(v))


def main():
    out = {"episodes": EPISODES, "seeds": SEEDS, "catalogues": {}}
    plans = [
        ("15", V3_CATALOGUE, "_cat15"),
        ("25", None, ""),
    ]
    for label, catalogue, suffix in plans:
        out["catalogues"][label] = {}
        for cfg, masked in (("masked", True), ("nomask", False)):
            rows = []
            for s in SEEDS:
                p = os.path.join(OUT, f"{cfg}_s{s}{suffix}.zip")
                if not os.path.exists(p):
                    print("missing", p)
                    continue
                rows.append(evaluate(p, masked, catalogue, s))
            if not rows:
                continue
            r = {}
            for k in ("success", "detection", "illegal", "reward", "techniques_used"):
                m, sd = agg(rows, k)
                r[k] = round(m, 2)
                r[k + "_sd"] = round(sd, 2)
            r["per_seed_success"] = [round(x["success"], 1) for x in rows]
            out["catalogues"][label][cfg] = r
            print("%s techniques | %-7s success %5.1f ± %4.1f | illegal %5.1f | detection %4.1f | techniques used %.1f"
                  % (label, cfg, r["success"], r["success_sd"], r["illegal"],
                     r["detection"], r["techniques_used"]))
    json.dump(out, open("analysis/v4_catalogue_comparison.json", "w"), indent=1)
    print("\nwrote analysis/v4_catalogue_comparison.json")


if __name__ == "__main__":
    main()
