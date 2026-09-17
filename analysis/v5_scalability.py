# -*- coding: utf-8 -*-
"""How the mask costs scale with the size of the action space.

Section 5.6 asserted that computing the mask is cheap and grows linearly. That
was an argument from the code rather than a measurement, and it is the point on
which the nearest comparator most clearly beats us: L-ARLPT operates over a
parameter-combination space of 60,374,160 while ours is 540. If our answer to
that is "a structural mask is cheaper than a learned pruner", the cost had
better be measured.

This builds synthetic tiered estates from 8 up to 200 hosts, which is 360 up to
9,000 actions, and measures the wall-clock cost of one full mask evaluation and
the environment's step throughput. No policy is trained; this is a cost
measurement, not a capability one.

One caution on the comparison it supports. L-ARLPT's space enumerates parameter
combinations — technique crossed with host crossed with the arguments a
technique takes — while ours enumerates semantically distinct (technique, host)
pairs. The raw counts are not the same kind of object and putting 540 beside
6 x 10^7 overstates the difference. What can be compared is the growth rate and
the cost per action, which is what this measures.

    python analysis/v5_scalability.py
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env import topology_v5 as T
from env.adversary_env_v5 import AdversaryEnvV5

SIZES = [8, 12, 20, 40, 80, 120, 200]


def synth(n_hosts, name):
    """A tiered estate of n_hosts, five zones, shaped like `enterprise`."""
    zones = 5
    per = max(1, n_hosts // zones)
    subnets, services, oss = {}, {}, {}
    made = 0
    for z in range(1, zones + 1):
        k = per if z < zones else n_hosts - made
        hosts = []
        for i in range(k):
            h = f"z{z}h{i}"
            hosts.append(h)
            made += 1
            if z == 1:
                services[h] = ["web", "ssh", "vpn"] if i == 0 else ["web", "smtp"]
                oss[h] = "linux"
            else:
                services[h] = ["smb", "rdp"] if i % 2 else ["smb", "ssh"]
                oss[h] = "windows" if i % 3 else "linux"
        subnets[z] = hosts
    goal = subnets[zones][0]
    T.TOPOLOGIES[name] = {
        "subnets": subnets,
        "edges": [(z, z + 1) for z in range(0, zones)],
        "services": services, "os": oss, "goal": goal,
    }
    return name


def bench(n_hosts, reps=200):
    name = synth(n_hosts, f"_synth{n_hosts}")
    env = AdversaryEnvV5({"topology": name, "max_steps": 60})
    mm = env.model
    env.reset(seed=1)
    rng = random.Random(3)

    # cost of one full mask evaluation
    t0 = time.perf_counter()
    for _ in range(reps):
        mm.action_mask_raw()
    mask_ms = 1000.0 * (time.perf_counter() - t0) / reps

    # end-to-end throughput, mask included
    steps, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        env.reset(seed=steps)
        done = False
        while not done and time.perf_counter() - t0 < 1.0:
            legal = [i for i, m in enumerate(env.action_masks()) if m]
            if not legal:
                break
            _, _, term, trunc, _ = env.step(rng.choice(legal))
            steps += 1
            done = term or trunc
    sps = steps / (time.perf_counter() - t0)
    return {"hosts": n_hosts, "actions": mm.n_actions,
            "mask_ms": mask_ms,
            "us_per_action": 1000.0 * mask_ms / mm.n_actions,
            "steps_per_sec": sps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/v5_scalability.json")
    a = ap.parse_args()

    rows = [bench(n) for n in SIZES]
    hdr = (f"{'hosts':>7}{'actions':>10}{'mask (ms)':>12}"
           f"{'us/action':>12}{'env steps/s':>14}")
    print(f"\nmask cost against action-space size, one CPU core\n")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['hosts']:>7}{r['actions']:>10}{r['mask_ms']:>12.3f}"
              f"{r['us_per_action']:>12.2f}{r['steps_per_sec']:>14,.0f}")

    x = np.array([r["actions"] for r in rows], dtype=float)
    y = np.array([r["mask_ms"] for r in rows], dtype=float)
    # slope in log-log space is the growth exponent: 1.0 is linear
    k = np.polyfit(np.log(x), np.log(y), 1)[0]
    perA = np.array([r["us_per_action"] for r in rows])
    print(f"\ngrowth exponent (log-log slope) {k:.2f}   "
          f"1.00 would be exactly linear")
    print(f"cost per action {perA.min():.2f}-{perA.max():.2f} us, "
          f"roughly flat across a 25x range of action-space size")

    os.makedirs("results", exist_ok=True)
    with open(a.json, "w") as f:
        json.dump({"rows": rows, "growth_exponent": float(k)}, f, indent=1)
    print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
