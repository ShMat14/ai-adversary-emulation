# -*- coding: utf-8 -*-
"""Run a trained policy against a live target, and against nothing else.

WHY THIS IS THE THIRD ATTEMPT

The first transfer experiment in this project served the simulator's own world
model over HTTP and measured a gap of exactly zero. It could not have measured
anything else: both arms imported the same class and called the same function.
The second, in `v5_calibration.py`, fixed the independence problem but changed
the question -- it calibrates *parameters*, executing each technique in isolation
against a live target to ask whether our assumed success probabilities are right.
Neither runs a *policy* against anything real.

This does. The agent chooses its own actions, in its own order, and every choice
is executed against `Target/vulnerable_app.py`, which shares no code with the
simulator. The only difference between the two arms is where the outcome of an
attempt comes from:

    simulated   succeeded = rng.random() <= tech.success_prob
    real        succeeded = <the technique executed against the live target>

Everything else -- preconditions, the mask, runtime gates, the detection model,
state effects, the reward -- is the same object in both arms. That is enforced by
construction: `KillChainModelV5._outcome` is the single seam, and this module
overrides that one method and nothing else. So a difference between the arms is a
difference in execution, not in bookkeeping.

THE MISSION HAD TO CHANGE, AND THAT IS WORTH READING

The paper's mission is domain admin. On a single web host it is unreachable, and
not by oversight: ACCOUNT_MANIPULATION requires domain enumeration and an
already-privileged host elsewhere, and exfiltration requires admin plus an
established C2 channel. Every one of those is a technique a web application
cannot host. The dependency structure that makes the catalogue honest is exactly
what puts the headline mission out of reach here.

So this arm uses the mission the live target can actually support: break in,
stage the document store, package it -- `objective="archived"`. Three steps, all
executable for real, and the target enforces the same ordering we do, returning a
conflict rather than a coin flip when an unstaged token is archived. The claim
this experiment supports is therefore narrower than "our agent transfers": it is
that a policy trained against our assumed probabilities executes a real chain
against a real system, and by how much its outcomes differ when it does.

    python Target/vulnerable_app.py --port 5005 &
    python analysis/v5_sim_to_real.py --config masked --train
    python analysis/v5_sim_to_real.py --config masked --episodes 100
"""
import argparse
import importlib.util
import io
import json
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.calibration import MEASURED
from env.kill_chain_v5 import TECHNIQUE_ORDER_V5

OUT = "results/v5_sim_to_real.json"
DEFAULT_URL = "http://127.0.0.1:5005"

# The mission for this arm, and the estate that matches the live target.
TOPOLOGY = "webhost"
OBJECTIVE = "archived"
# Simulated success below this cannot support a retention ratio: a seed at
# 3% is two episodes in sixty, and its "retention" is noise as a percentage.
MIN_BASE_RATE = 20.0
# Only techniques we can actually execute; the rest are withdrawn so the policy
# is never trained to rely on something the live target cannot perform.
KEEP = set(MEASURED)
WITHDRAW = tuple(n for n in TECHNIQUE_ORDER_V5 if n not in KEEP)


def _executors():
    """Load the live-target executors from the calibration harness."""
    spec = importlib.util.spec_from_file_location(
        "_cal", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "v5_calibration.py"))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    by_tech = {}
    for tech, _variant, fn in mod.EXECUTORS:
        # A technique with two variants (SQL injection) keeps both, and one is
        # drawn per attempt, exactly as in the calibration.
        by_tech.setdefault(tech, []).append(fn)
    return by_tech


class RealOutcome:
    """Replaces the success roll with a real execution against the live target.

    Installed on a model instance rather than subclassed, so that the object the
    policy interacts with is literally the same class in both arms.
    """

    def __init__(self, url, rng, executors):
        self.url = url
        self.rng = rng
        self.executors = executors
        self.calls = 0
        self.per_technique = {}

    def __call__(self, tech, technique, target):
        fns = self.executors.get(technique)
        if not fns:
            # No live analogue: fall back to the assumed probability and record
            # it, so the reported gap is never quietly credited to real execution.
            self.per_technique.setdefault(technique, {"real": 0, "simulated": 0,
                                                      "hits": 0})["simulated"] += 1
            return self.rng.random() <= tech.success_prob
        fn = self.rng.choice(fns)
        try:
            ok = bool(fn(self.url, self.rng))
        except Exception:
            ok = False                      # a failed attempt is a failed attempt
        self.calls += 1
        rec = self.per_technique.setdefault(technique, {"real": 0, "simulated": 0,
                                                        "hits": 0})
        rec["real"] += 1
        rec["hits"] += int(ok)
        return ok


def make(seed, max_steps, real=None):
    """The environment both arms use. `real` installs the live-execution seam."""
    from analysis.v5_train import make_env
    env = make_env(TOPOLOGY, seed=seed, max_steps=max_steps,
                   exclude_techniques=WITHDRAW, objective=OBJECTIVE)
    if real is not None:
        env.model._outcome = real
    return env


def evaluate(model, algo, seed, episodes, max_steps, real=None):
    from analysis.v5_train import rollout
    env = make(seed, max_steps, real=real)
    succ, ill, det, steps = rollout(model, env, algo, episodes=episodes,
                                    base_seed=500_000)
    return {"success": succ, "illegal": ill, "detected": det, "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="masked",
                    choices=["masked", "maskdqn", "maska2c"])
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()

    from analysis.v5_train import CONFIGS, OUT as MODELS, build, load
    algo = CONFIGS[a.config]
    seeds = [int(s) for s in a.seeds.split(",")]
    execs = _executors()

    import requests
    try:
        requests.get(a.url + "/health", timeout=4).raise_for_status()
    except Exception:
        raise SystemExit(
            f"no live target at {a.url}.  start it with:\n"
            f"    python Target/vulnerable_app.py --port {a.url.rsplit(':', 1)[-1]}")

    def _reset_target():
        """Each seed starts from the same target state.

        The target accumulates: cleared audit logs, staged tokens, uploaded
        shells and accounts created through the injection flaw all persist. Left
        alone that makes a seed's result depend on which seeds ran before it.
        """
        try:
            requests.post(a.url + "/reset", timeout=10).raise_for_status()
        except Exception as exc:
            raise SystemExit(f"could not reset the live target: {exc}")

    rows = []
    for s in seeds:
        _reset_target()
        tag = f"{a.config}_s2r_s{s}"
        path = os.path.join(MODELS, tag)
        if a.train or not os.path.exists(path + ".zip"):
            env = make(s, a.max_steps)
            model = build(algo, env, s)
            print(f"--- {tag}: {a.steps:,} steps in simulation ---", flush=True)
            model.learn(total_timesteps=a.steps)
            os.makedirs(MODELS, exist_ok=True)
            model.save(path)
        model = load(algo, path)

        sim = evaluate(model, algo, s, a.episodes, a.max_steps)
        real_hook = RealOutcome(a.url, random.Random(9000 + s), execs)
        real = evaluate(model, algo, s, a.episodes, a.max_steps, real=real_hook)
        rows.append({"seed": s, "simulated": sim, "real": real,
                     "live_calls": real_hook.calls,
                     "per_technique": real_hook.per_technique})
        print(f"RESULT {tag}: simulated success {sim['success']:5.1f}%  "
              f"real {real['success']:5.1f}%   "
              f"(gap {real['success'] - sim['success']:+.1f}, "
              f"{real_hook.calls} live executions)", flush=True)

    def agg(arm, k):
        return st.mean([r[arm][k] for r in rows])

    def sd(arm, k):
        xs = [r[arm][k] for r in rows]
        return st.stdev(xs) if len(xs) > 1 else 0.0

    summary = {m: {k: {"mean": agg(m, k), "sd": sd(m, k)}
                   for k in ("success", "illegal", "detected", "steps")}
               for m in ("simulated", "real")}
    print(f"\n=== {a.config}, {len(seeds)} seeds x {a.episodes} episodes ===")
    for k in ("success", "illegal", "detected", "steps"):
        s_, r_ = summary["simulated"][k], summary["real"][k]
        print(f"  {k:<9} simulated {s_['mean']:6.1f} +/- {s_['sd']:4.1f}   "
              f"real {r_['mean']:6.1f} +/- {r_['sd']:4.1f}   "
              f"gap {r_['mean'] - s_['mean']:+6.1f}")

    prev = {}
    if os.path.exists(OUT):
        prev = json.load(io.open(OUT, encoding="utf-8"))
    # Merge by seed rather than replace. A second invocation with further seeds
    # must extend the record, not silently discard the first run's.
    kept = [r for r in prev.get(a.config, {}).get("rows", [])
            if r["seed"] not in {x["seed"] for x in rows}]
    allrows = sorted(kept + rows, key=lambda r: r["seed"])

    def _agg(arm, k):
        xs = [r[arm][k] for r in allrows]
        return {"mean": st.mean(xs),
                "sd": st.stdev(xs) if len(xs) > 1 else 0.0}

    summary = {m: {k: _agg(m, k)
                   for k in ("success", "illegal", "detected", "steps")}
               for m in ("simulated", "real")}
    # Retention is only interpretable when the simulated arm succeeds often
    # enough to form a ratio; below that a seed contributes noise as a
    # percentage. Table 16 uses this threshold and reports how many seeds clear
    # it, so both numbers are recorded here.
    usable = [r for r in allrows
              if r["simulated"]["success"] >= MIN_BASE_RATE]
    retention = [100.0 * r["real"]["success"] / r["simulated"]["success"]
                 for r in usable]
    retention_all = [100.0 * r["real"]["success"] / r["simulated"]["success"]
                     for r in allrows if r["simulated"]["success"]]
    prev[a.config] = {"topology": TOPOLOGY, "objective": OBJECTIVE,
                      "episodes": a.episodes, "max_steps": a.max_steps,
                      "withdrawn": len(WITHDRAW), "rows": allrows,
                      "n_seeds": len(allrows),
                      # retention is the fair cross-configuration measure: a
                      # configuration that barely succeeds in simulation has a
                      # small absolute gap for trivial reasons.
                      "min_base_rate": MIN_BASE_RATE,
                      "retention_seeds_used": len(usable),
                      "retention_mean": st.mean(retention) if retention else None,
                      # every seed, including ones too near the floor to be
                      # meaningful; kept only so the exclusion is auditable
                      "retention_mean_all_seeds":
                          st.mean(retention_all) if retention_all else None,
                      "summary": summary}
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(prev, f, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
