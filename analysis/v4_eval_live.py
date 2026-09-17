# -*- coding: utf-8 -*-
"""
Evaluate the trained agent against the LIVE server (real mode).

The main eval runs in simulation for speed. This runs the trained MaskablePPO
policy over HTTP against the running v4 target, so the number is the policy's
score in deployment, not in the simulator. Because the two share one model the
result should match the sim eval within sampling noise -- that is the point.

    python Target/mock_server.py             # terminal 1
    python analysis/v4_eval_live.py          # terminal 2
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import requests

from sb3_contrib import MaskablePPO
from env.adversary_env import AdversaryEnv

SERVER = "http://127.0.0.1:5000"


class _Null:
    def start_episode(self): pass
    def log_event(self, *a, **k): pass
    def end_episode(self): pass


def evaluate(real_mode, episodes, model):
    env = AdversaryEnv(config={"max_steps": 40, "real_mode": real_mode})
    env.telemetry = _Null()
    succ = det = 0
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=500_000 + ep)
        done, total = False, 0.0
        while not done:
            legal = env.action_masks()
            a, _ = model.predict(obs, action_masks=legal, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(a))
            total += r
            done = term or trunc
        rewards.append(total)
        if env.model.is_goal():
            succ += 1
        if env.model.state.any_host_detected():
            det += 1
    return {"success": 100.0 * succ / episodes,
            "detection": 100.0 * det / episodes,
            "reward": float(np.mean(rewards))}


def main(episodes=100):
    model = MaskablePPO.load("results/models/v4/masked_s0", device="cpu")
    print(f"trained MaskablePPO (masked_s0) — {episodes} episodes")
    print("-" * 52)
    print("evaluating in simulation ...")
    sim = evaluate(False, episodes, model)
    try:
        requests.get(f"{SERVER}/health", timeout=1)
    except Exception:
        print("server down — start Target/mock_server.py"); sys.exit(1)
    print("evaluating against the live server ...")
    real = evaluate(True, episodes, model)

    print(f"\n  {'metric':12s}{'sim':>10}{'live server':>14}")
    for k, u in (("success", "%"), ("detection", "%"), ("reward", "")):
        print(f"  {k:12s}{sim[k]:>9.1f}{u}{real[k]:>13.1f}{u}")


if __name__ == "__main__":
    main()
