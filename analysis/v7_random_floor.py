# -*- coding: utf-8 -*-
"""Re-measure Table 3: random play restricted to legal actions.

Rebuilding the supplement's provenance table (Section S10) showed that no script
in the repository produces Table 3. The numbers are internally consistent (each
95% interval is the Wilson interval of its rate at n = 2,000) but the run that
made them was not kept. This re-runs the measurement as the paper describes it,
2,000 episodes per environment with Wilson intervals, each environment at its
own default step budget and with its defender on, and records the result.

It does not overwrite Table 3. It reports whether a fresh run agrees with the
published rows: if every fresh rate falls inside the published interval, this
script is a faithful producer and S10 can name it; if not, the difference is
reported rather than papered over.

    python analysis/v7_random_floor.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N = 2000
PUBLISHED = {   # Table 3 as printed: rate %, interval low, interval high
    "previous": (51.1, 48.9, 53.3),
    "v4compat": (20.00, 18.31, 21.81),
    "enterprise": (1.35, 0.93, 1.96),
}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


class _NoTelemetry:
    def __getattr__(self, _):
        return lambda *a, **k: None


def run_v5(topology: str, seed0: int):
    from env.adversary_env_v5 import AdversaryEnvV5
    env = AdversaryEnvV5({"topology": topology, "seed": seed0})
    rng = random.Random(seed0)
    wins, fracs = 0, []
    for ep in range(N):
        env.reset(seed=seed0 + ep)
        done = False
        while not done:
            m = env.action_masks()
            legal = [i for i, x in enumerate(m) if x]
            fracs.append(len(legal) / len(m))
            _, _, term, trunc, _ = env.step(rng.choice(legal) if legal else 0)
            done = term or trunc
        wins += bool(env.model.is_goal())
    return wins, sum(fracs) / len(fracs), int(env.action_space.n), int(env.max_steps)


def run_v4(seed0: int):
    from env.adversary_env import AdversaryEnv
    env = AdversaryEnv({})
    env.telemetry = _NoTelemetry()       # 2,000 episodes need not write 2,000 logs
    rng = random.Random(seed0)
    wins, fracs = 0, []
    for ep in range(N):
        env.reset(seed=seed0 + ep)
        env.telemetry = _NoTelemetry()
        done = False
        while not done:
            m = env.action_masks()
            legal = [i for i, x in enumerate(m) if x]
            fracs.append(len(legal) / len(m))
            _, _, term, trunc, _ = env.step(rng.choice(legal) if legal else 0)
            done = term or trunc
        wins += bool(env.model.is_goal())
    return wins, sum(fracs) / len(fracs), int(env.action_space.n), int(env.max_steps)


def main():
    rows = {}
    for key, fn in (("previous", lambda s: run_v4(s)),
                    ("v4compat", lambda s: run_v5("v4compat", s)),
                    ("enterprise", lambda s: run_v5("enterprise", s))):
        k, legal, nact, budget = fn(10_000)
        lo, hi = wilson(k, N)
        pub, plo, phi = PUBLISHED[key]
        rate = 100 * k / N
        agrees = plo <= rate <= phi
        rows[key] = {"episodes": N, "objective_reached": k, "rate_pct": rate,
                     "wilson_95": [lo, hi], "mean_legal_fraction_pct": 100 * legal,
                     "actions": nact, "step_budget": budget,
                     "published": {"rate_pct": pub, "wilson_95": [plo, phi]},
                     "inside_published_interval": agrees}
        print(f"{key:11} {nact:4} actions, budget {budget:3}: {k:4}/{N} = {rate:5.2f}% "
              f"[{lo:5.2f}, {hi:5.2f}], legal {100 * legal:4.1f}%   "
              f"published {pub}% [{plo}, {phi}]  {'agrees' if agrees else 'DIFFERS'}")
    out = ROOT / "results" / "v7_random_floor.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
