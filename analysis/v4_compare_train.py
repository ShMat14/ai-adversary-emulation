# -*- coding: utf-8 -*-
"""
Compare a sim-trained agent with a live-trained agent.

Both are MaskablePPO trained for the full budget on the v4 environment with the
same seed; one learned in the fast simulator, the other by talking to the live
server over HTTP for every step. Each is then evaluated in both modes. If the
shared model does its job, the four cells agree: how the agent trained does not
change what it learned, because the two worlds are one.

    python Target/mock_server.py            # terminal 1
    python analysis/v4_compare_train.py     # terminal 2

Reads results/models/v4/masked_s0.zip (sim) and masked_s0_live.zip (live).
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


def evaluate(model, real_mode, episodes):
    env = AdversaryEnv(config={"max_steps": 40, "real_mode": real_mode})
    env.telemetry = _Null()
    succ = det = 0
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=800_000 + ep)
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


def server_up():
    try:
        requests.get(f"{SERVER}/health", timeout=1)
        return True
    except Exception:
        return False


def main(sim_eps=300, live_eps=30):
    sim_model = MaskablePPO.load("results/models/v4/masked_s0", device="cpu")
    live_path = "results/models/v4/masked_s0_live"
    if not os.path.exists(live_path + ".zip"):
        print("live-trained model not found — run training with --live first"); sys.exit(1)
    live_model = MaskablePPO.load(live_path, device="cpu")

    rows = []
    # sim-mode evaluation is fast; run the full episode count
    rows.append(("sim-trained", "sim", evaluate(sim_model, False, sim_eps)))
    rows.append(("live-trained", "sim", evaluate(live_model, False, sim_eps)))
    if server_up():
        # live-mode evaluation is slow (HTTP); a smaller count still confirms parity
        rows.append(("sim-trained", "live", evaluate(sim_model, True, live_eps)))
        rows.append(("live-trained", "live", evaluate(live_model, True, live_eps)))

    lines = ["full-budget training: sim vs live, cross-evaluated", "",
             f"{'agent':14s}{'eval':>7}{'success':>11}{'detection':>12}{'reward':>10}"]
    for who, mode, r in rows:
        lines.append(f"{who:14s}{mode:>7}{r['success']:>10.1f}%"
                     f"{r['detection']:>11.1f}%{r['reward']:>10.1f}")
    txt = "\n".join(lines)
    with open("analysis/v4_train_compare_result.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
