# -*- coding: utf-8 -*-
"""Sim-to-real transfer: the same policy, in process and over HTTP.

SUPERSEDED as a transfer experiment. This measures transport only, because
the server imports the same world model the simulator drives -- the point the
docstring below already makes, and which Section 4.8 of the manuscript reports
as a mistake. The experiment that actually tests transfer is
`analysis/v5_sim_to_real.py`, which runs the policy against a target sharing no
code with the simulator. This file is kept because the paper describes it.

The claim being tested is narrow and worth stating precisely. Because the server
imports the same world model the simulator uses, a difference between the two
columns cannot be model disagreement. It can only be transport: serialisation of
the observation and the mask, the network round trip, session handling, and
float round-tripping through JSON. So this experiment answers "does the policy
survive deployment over a real interface", and it deliberately does not answer
"is the model right", which is Section 3's job.

The earlier version of this work had two hand-written implementations and
measured a 32.5-point gap on a single technique. That gap measured a bug.

    python Target/server_v5.py --port 5001          # in one terminal
    python analysis/v5_transfer.py --episodes 100   # in another
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import requests

from analysis.v5_train import load, make_env


def run_local(model, topology, episodes, max_steps, base_seed):
    from sb3_contrib.common.maskable.utils import get_action_masks
    env = make_env(topology, seed=0, max_steps=max_steps)
    mm = env.model
    wins = caught = steps = 0
    lens, events = [], []
    t0 = time.time()
    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        n0 = len(mm.detection.log)
        done, n = False, 0
        while not done:
            a, _ = model.predict(obs, action_masks=get_action_masks(env),
                                 deterministic=True)
            obs, r, term, trunc, _ = env.step(int(a))
            n += 1
            done = term or trunc
        lens.append(n)
        steps += n
        wins += mm.is_goal()
        caught += mm.caught()
        events.append(len(mm.detection.log) - n0)
    return {"success": 100.0 * wins / episodes,
            "detected": 100.0 * caught / episodes,
            "steps": statistics.mean(lens),
            "events": statistics.mean(events),
            "wall": time.time() - t0}


def run_remote(model, url, topology, episodes, max_steps, base_seed):
    s = requests.Session()
    r = s.post(f"{url}/session", json={"topology": topology,
                                       "max_steps": max_steps}, timeout=30)
    r.raise_for_status()
    tok = r.json()["session"]
    wins = caught = 0
    lens, events = [], []
    t0 = time.time()
    for ep in range(episodes):
        r = s.post(f"{url}/reset", json={"session": tok,
                                         "seed": base_seed + ep}, timeout=30).json()
        obs = np.asarray(r["observation"], dtype=np.float32)
        mask = np.asarray(r["action_mask"], dtype=bool)
        before = s.get(f"{url}/siem/events", params={"session": tok},
                       timeout=30).json()["total"]
        done, n, won, inc = False, 0, False, False
        while not done:
            a, _ = model.predict(obs, action_masks=mask, deterministic=True)
            r = s.post(f"{url}/attempt", json={"session": tok, "action": int(a)},
                       timeout=30).json()
            obs = np.asarray(r["observation"], dtype=np.float32)
            mask = np.asarray(r["action_mask"], dtype=bool)
            n += 1
            won, inc = r["objective_reached"], r["incident_declared"]
            done = r["terminated"] or r["truncated"]
        after = s.get(f"{url}/siem/events", params={"session": tok},
                      timeout=30).json()["total"]
        lens.append(n)
        events.append(after - before)
        wins += won
        caught += inc
    return {"success": 100.0 * wins / episodes,
            "detected": 100.0 * caught / episodes,
            "steps": statistics.mean(lens),
            "events": statistics.mean(events),
            "wall": time.time() - t0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/models/v5/masked_enterprise_s0")
    ap.add_argument("--url", default="http://127.0.0.1:5001")
    ap.add_argument("--topology", default="enterprise")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--base-seed", type=int, default=910_000)
    a = ap.parse_args()

    model = load("maskable", a.model)
    print(f"policy: {os.path.basename(a.model)}   estate: {a.topology}   "
          f"episodes: {a.episodes}\n")

    loc = run_local(model, a.topology, a.episodes, a.max_steps, a.base_seed)
    rem = run_remote(model, a.url, a.topology, a.episodes, a.max_steps, a.base_seed)

    hdr = f"{'metric':26}{'in process':>14}{'over HTTP':>14}{'gap':>12}"
    print(hdr); print("-" * len(hdr))
    for key, name, unit in (("success", "mission success", "%"),
                            ("detected", "incident declared", "%"),
                            ("steps", "mean episode length", ""),
                            ("events", "telemetry events/episode", "")):
        g = rem[key] - loc[key]
        print(f"{name:26}{loc[key]:>13.1f}{unit}{rem[key]:>13.1f}{unit}{g:>+12.1f}")
    print(f"\n{'wall-clock seconds':26}{loc['wall']:>14.1f}{rem['wall']:>14.1f}"
          f"{rem['wall']/max(1e-9, loc['wall']):>11.0f}x")
    print("\nBoth columns drive the same world model, so any gap above is "
          "transport,\nnot model disagreement. Episode seeds are identical "
          "across the two runs.")
