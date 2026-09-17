# -*- coding: utf-8 -*-
"""The full v5 study, run as a bounded pool of single-threaded workers.

Ordered by scientific value, so an interrupted run still leaves the important
results complete, and phase-by-phase, so a later phase never starts before the
one it depends on has finished.

Why a pool rather than the sequential shell script this replaces: PyTorch on CPU
will happily take every core for one small MLP and gain almost nothing, so the
old script left thirteen of fourteen cores idle for eight hours. Each worker here
is pinned to a single thread and several run at once, which is both faster and
more reproducible -- thread count no longer varies with what else is running.

Every job skips when its checkpoint already exists, so this is stop/resume safe.

    python analysis/run_v5_all.py                 # everything
    python analysis/run_v5_all.py --phases 1 2    # selected phases
    python analysis/run_v5_all.py --jobs 4        # narrower pool
    python analysis/run_v5_all.py --dry-run
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "models", "v5")
LOGS = os.path.join(ROOT, "logs", "v5")
STEPS = 600_000

# cfg, seed, budget, suffix, illegal_penalty, topology
PHASES = {}

PHASES[1] = ("main comparison, 12-host estate, 60-step budget, 5 seeds: "
             "masked | unmasked | reward-shaped (prior work's mechanism) | DQN | A2C",
             [(c, s, 60, sx, p, "enterprise")
              for s in range(5)
              for c, sx, p in [("masked", "", 0), ("nomask", "", 0),
                               ("nomask", "_shaped", -10),
                               ("dqn", "", 0), ("a2c", "", 0)]])

PHASES[2] = ("fairness control at L-ARLPT's 500-step budget. If an unmasked "
             "agent succeeds here, masking buys efficiency and not capability, "
             "and that weaker claim is the one that must be reported",
             [(c, s, 500, "_b500", 0, "enterprise")
              for s in range(3) for c in ("nomask", "dqn", "a2c")]
             + [("nomask", s, 500, "_b500_shaped", -10, "enterprise")
                for s in range(3)])

PHASES[3] = ("formulation vs. size: the same v5 rules on the 3-host network, so "
             "any gain over v4 can be attributed to the action-space change "
             "rather than to a larger estate",
             [(c, s, 60, "", 0, "v4compat")
              for s in range(3) for c in ("masked", "nomask")])

PHASES[4] = ("extreme control at E-NASim's 2000-step budget; episodes are 33x "
             "longer so only the headline baseline, two seeds",
             [("nomask", s, 2000, "_b2000", 0, "enterprise") for s in range(2)])

# Phase 6 is defined before 5 only so the dict reads in dependency order; the
# runner still executes them in numeric order.
PHASES[6] = ("episode-matched fairness control. Phase 2 matches the comparators' "
             "500-step budget at the same TIMESTEP budget, which gives those "
             "runs 800 episodes against 6,667 at 60 steps -- so a reviewer could "
             "object that the baseline was starved of episodes rather than of "
             "steps. This repeats it at 3.3M timesteps, matching episodes "
             "instead. Two seeds, unmasked only, because it is 8x the cost",
             [("nomask", s, 500, "_b500_epmatch", 0, "enterprise")
              for s in range(2)])

PHASES[5] = ("extend the main comparison to ten seeds",
             [(c, s, 60, sx, p, "enterprise")
              for s in range(5, 10)
              for c, sx, p in [("masked", "", 0), ("nomask", "", 0),
                               ("nomask", "_shaped", -10),
                               ("dqn", "", 0), ("a2c", "", 0)]])


def tag_of(job):
    cfg, seed, budget, suffix, pen, topo = job
    return f"{cfg}_{topo}_s{seed}{suffix}"


def done(job):
    return os.path.exists(os.path.join(OUT, tag_of(job) + ".zip"))


# Phase 6 trains far longer on purpose; every other job uses the common budget.
STEPS_OVERRIDE = {"_b500_epmatch": 3_300_000}


def launch(job, steps):
    cfg, seed, budget, suffix, pen, topo = job
    steps = STEPS_OVERRIDE.get(suffix, steps)
    tag = tag_of(job)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log = open(os.path.join(LOGS, tag + ".log"), "w")
    cmd = [sys.executable, os.path.join(ROOT, "analysis", "v5_train.py"),
           "--config", cfg, "--topology", topo, "--seed", str(seed),
           "--steps", str(steps), "--max-steps", str(budget),
           "--suffix", suffix, "--illegal-penalty", str(pen)]
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                         stderr=subprocess.STDOUT)
    return p, log, tag


def run_phase(n, jobs, pool, steps, dry):
    todo = [j for j in jobs if not done(j)]
    skipped = len(jobs) - len(todo)
    print(f"\n### PHASE {n}  {PHASES[n][0]}", flush=True)
    print(f"###   {len(jobs)} jobs, {skipped} already complete, "
          f"{len(todo)} to run, {pool} at a time", flush=True)
    if dry:
        for j in todo:
            print(f"      would run {tag_of(j)}", flush=True)
        return
    active, queue, failed = [], list(todo), []
    t0 = time.time()
    while queue or active:
        while queue and len(active) < pool:
            job = queue.pop(0)
            p, log, tag = launch(job, steps)
            active.append((p, log, tag, time.time()))
            print(f"      start {tag}", flush=True)
        time.sleep(5)
        for entry in list(active):
            p, log, tag, ts = entry
            if p.poll() is None:
                continue
            active.remove(entry)
            log.close()
            mins = (time.time() - ts) / 60
            if p.returncode == 0:
                print(f"      done  {tag}  ({mins:.1f} min)", flush=True)
            else:
                failed.append(tag)
                print(f"      FAIL  {tag}  exit {p.returncode}  "
                      f"see logs/v5/{tag}.log", flush=True)
    print(f"### PHASE {n} finished in {(time.time()-t0)/60:.1f} min"
          + (f", FAILURES: {failed}" if failed else ""), flush=True)
    return failed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6,
                    help="workers in the pool; each is pinned to one thread")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--phases", type=int, nargs="*",
                    default=sorted(PHASES), help="phases to run, in order")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    all_failed = []
    for n in a.phases:
        f = run_phase(n, PHASES[n][1], a.jobs, a.steps, a.dry_run)
        all_failed += (f or [])
    if not a.dry_run:
        print(f"\nALL REQUESTED PHASES COMPLETE in "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
        if all_failed:
            print(f"FAILURES ({len(all_failed)}): {all_failed}", flush=True)
