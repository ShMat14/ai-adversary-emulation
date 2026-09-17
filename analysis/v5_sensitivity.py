# -*- coding: utf-8 -*-
"""Sensitivity of a configuration to the assumed success probabilities.

Section 4.8 asks whether the paper survives a pessimistic environment, and it has
to ask that question twice for each configuration, because two different things
are easily confused:

  trained elsewhere  take a policy trained under the shipped probabilities and
                     evaluate it under the corrected ones. This measures the
                     environment being harder AND the policy having grown up
                     somewhere else, and reporting it alone overstates fragility.

  trained here       train a fresh policy under the corrected probabilities and
                     evaluate it there. This measures the difficulty alone.

The first version of this experiment reported only the first column and drew the
wrong conclusion from it, which is why both are computed here rather than left to
whoever runs it next.

The calibration mutates the module-level technique table before the environment
is built, so every episode in this process runs under it. Each run is therefore a
separate process, which is also why the launcher below trains one seed at a time.

    python analysis/v5_sensitivity.py --config maskdqn --mode measured --seed 0
    python analysis/v5_sensitivity.py --config maskdqn --mode shipped --eval-only
"""
import argparse
import io
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.kill_chain_v5 import TECHNIQUES_V5
from env import calibration

OUT_DIR = "results/sensitivity"


def apply_mode(mode):
    """Mutate the technique table for this process. Returns a label."""
    if mode == "shipped":
        return "as shipped"
    if mode == "measured":
        calibration.apply(TECHNIQUES_V5, mode="measured")
        return f"{len(calibration.MEASURED)} measured rates substituted"
    if mode == "scaled":
        calibration.apply(TECHNIQUES_V5, mode="scaled")
        return f"all 45 scaled by {calibration.MEAN_RATIO}"
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="maskdqn")
    ap.add_argument("--mode", default="measured",
                    choices=["shipped", "measured", "scaled"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; evaluate the SHIPPED checkpoint under "
                         "this calibration, i.e. the 'trained elsewhere' column")
    a = ap.parse_args()

    label = apply_mode(a.mode)
    # imported after the mutation so nothing captured the old probabilities
    from analysis.v5_train import CONFIGS, OUT, build, load, make_env, rollout

    algo = CONFIGS[a.config]
    shipped_path = os.path.join(OUT, f"{a.config}_enterprise_s{a.seed}")
    tag = f"{a.config}_{a.mode}_s{a.seed}"

    if a.eval_only:
        if not os.path.exists(shipped_path + ".zip"):
            raise SystemExit(f"no shipped checkpoint at {shipped_path}.zip")
        model = load(algo, shipped_path)
        arm = "trained_elsewhere"
        print(f"--- {tag}: evaluating the shipped checkpoint under '{label}' ---",
              flush=True)
    else:
        env = make_env("enterprise", seed=a.seed, max_steps=a.max_steps)
        model = build(algo, env, a.seed)
        arm = "trained_here"
        print(f"--- {tag}: {a.steps:,} steps under '{label}' ---", flush=True)
        model.learn(total_timesteps=a.steps)
        os.makedirs(OUT, exist_ok=True)
        model.save(os.path.join(OUT, tag))

    env = make_env("enterprise", seed=a.seed, max_steps=a.max_steps)
    succ, ill, det, L = rollout(model, env, algo, episodes=a.episodes)
    print(f"RESULT {tag} [{arm}]: success {succ:5.1f}%  illegal {ill:4.1f}%  "
          f"detected {det:5.1f}%  steps {L:.1f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{a.config}_{a.mode}_{arm}_s{a.seed}.json")
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump({"config": a.config, "mode": a.mode, "label": label,
                   "arm": arm, "seed": a.seed, "episodes": a.episodes,
                   "success": succ, "illegal": ill, "detected": det,
                   "steps": L}, f, indent=1)
    print(f"wrote {path}")


def collect(config):
    """Aggregate whatever runs have completed, into the shape of Table 15."""
    rows = {}
    if not os.path.isdir(OUT_DIR):
        return rows
    for fn in sorted(os.listdir(OUT_DIR)):
        if not fn.startswith(config + "_") or not fn.endswith(".json"):
            continue
        d = json.load(io.open(os.path.join(OUT_DIR, fn), encoding="utf-8"))
        rows.setdefault((d["mode"], d["arm"]), []).append(d)
    out = {}
    for (mode, arm), rs in sorted(rows.items()):
        f = lambda k: [r[k] for r in rs]
        out[f"{mode}/{arm}"] = {
            "n": len(rs),
            "success_mean": st.mean(f("success")),
            "success_sd": st.stdev(f("success")) if len(rs) > 1 else 0.0,
            "detected_mean": st.mean(f("detected")),
            "illegal_mean": st.mean(f("illegal")),
            "steps_mean": st.mean(f("steps")),
        }
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--collect":
        cfg = sys.argv[2] if len(sys.argv) > 2 else "maskdqn"
        agg = collect(cfg)
        print(f"{'condition':<34}{'n':>3}{'success':>12}{'detected':>10}{'illegal':>9}")
        for k, v in agg.items():
            print(f"{k:<34}{v['n']:>3}{v['success_mean']:>8.1f} +/-{v['success_sd']:<4.1f}"
                  f"{v['detected_mean']:>10.1f}{v['illegal_mean']:>9.1f}")
        print()
        print(json.dumps(agg, indent=1))
    else:
        main()
