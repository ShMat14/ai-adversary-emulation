# -*- coding: utf-8 -*-
"""Run every warm-start run, a few at a time, skipping ones already finished.

Twenty runs: {warm, scratch} x {mid16, large20} x seeds 0-4. Memory, not
cores, limits concurrency on this machine, hence --jobs. Paired runs (same
target and seed, both conditions) are queued next to each other so that a
partial set is still a usable paired comparison.

    python analysis/v8_warmstart_all.py --jobs 4 --steps 400000
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "v8_warmstart")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400_000)
    a = ap.parse_args()

    os.makedirs(os.path.join(RUNS, "logs"), exist_ok=True)
    queue = [(c, t, s) for t in ("mid16", "large20") for s in range(5)
             for c in ("warm", "scratch")]
    queue = [q for q in queue
             if not os.path.exists(os.path.join(RUNS, f"{q[0]}_{q[1]}_s{q[2]}.json"))]
    print(f"{len(queue)} runs to do, {a.jobs} at a time", flush=True)

    running = []
    t0 = time.time()
    while queue or running:
        while queue and len(running) < a.jobs:
            c, t, s = queue.pop(0)
            tag = f"{c}_{t}_s{s}"
            log = open(os.path.join(RUNS, "logs", tag + ".log"), "w")
            p = subprocess.Popen([sys.executable, "analysis/v8_warmstart.py",
                                  "--cond", c, "--target", t, "--seed", str(s),
                                  "--steps", str(a.steps)],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running.append((p, tag, log))
            print(f"[{(time.time() - t0) / 60:6.1f} min] start {tag}", flush=True)
        time.sleep(10)
        for p, tag, log in running[:]:
            if p.poll() is not None:
                log.close()
                running.remove((p, tag, log))
                state = "done" if p.returncode == 0 else f"FAILED ({p.returncode})"
                print(f"[{(time.time() - t0) / 60:6.1f} min] {state} {tag}", flush=True)
    print(f"all finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
