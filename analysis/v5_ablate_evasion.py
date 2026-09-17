# -*- coding: utf-8 -*-
"""Does Defense Evasion buy the exposure the agent enjoys, or only cost steps?

Section 4.12 finds that 18.1% of a winning episode goes to Defense Evasion, a
tactic the nearest comparator's environment has no analogue for. That is a fact
about where steps go. It is not yet a claim that those steps *do* anything, and
the paper should not make the claim without the experiment.

This is the experiment. Withdraw every Defense Evasion technique from the
catalogue, retrain from scratch under that catalogue, and evaluate there. Fresh
training matters for the same reason it mattered in Section 4.8: re-evaluating a
policy trained with evasion in an environment without it measures distribution
shift as well as the ablation, and reports the sum as though it were the part.

Every engagement remains winnable without Defense Evasion -- checked by
random-legal search with the defender off, 60/60 solved at a median of 99.5
steps against 105.5 with the tactic present -- so a drop here cannot be an
artefact of engagements that no policy could have won.

    python analysis/v5_ablate_evasion.py --seeds 0,1,2
"""
import argparse
import io
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.v5_train import CONFIGS, OUT, build, load, make_env, rollout

# Two arms, because the first one alone does not separate two things.
#
# The tactic arm withdraws all five Defense Evasion techniques. One of them,
# reusing a valid stolen account, is a route as well as a concealment, so part
# of any drop in mission success there is lost access rather than lost cover.
#
# The technique arm withdraws only the four that do nothing but conceal, leaving
# every route intact. If exposure still rises, concealment is what was buying it.
ABLATED = ("Defense Evasion",)
CONCEALMENT_ONLY = ("CLEAR_LOGS", "DISABLE_SECURITY_TOOLS", "MODIFY_REGISTRY",
                    "OBFUSCATED_FILES")
ARMS = {
    "tactic": dict(tag="masked_enterprise_noevasion_s",
                   tactics=ABLATED, techniques=(),
                   label="all 5 Defense Evasion techniques withdrawn",
                   out="results/v5_ablate_evasion.json"),
    "concealment": dict(tag="masked_enterprise_noconceal_s",
                        tactics=(), techniques=CONCEALMENT_ONLY,
                        label="4 concealment-only techniques withdrawn",
                        out="results/v5_ablate_conceal.json"),
}


def train_one(seed, steps, max_steps, arm):
    a = ARMS[arm]
    env = make_env("enterprise", seed=seed, max_steps=max_steps,
                   exclude_tactics=a["tactics"],
                   exclude_techniques=a["techniques"])
    model = build(CONFIGS["masked"], env, seed)
    print(f"--- {a['tag']}{seed}: {steps:,} steps ---", flush=True)
    model.learn(total_timesteps=steps)
    os.makedirs(OUT, exist_ok=True)
    model.save(os.path.join(OUT, a["tag"] + str(seed)))
    return os.path.join(OUT, a["tag"] + str(seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--arm", default="tactic", choices=sorted(ARMS))
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    arm = ARMS[a.arm]

    rows = []
    for s in seeds:
        path = os.path.join(OUT, arm["tag"] + str(s))
        if not a.skip_train:
            train_one(s, a.steps, a.max_steps, a.arm)
        model = load(CONFIGS["masked"], path)
        env = make_env("enterprise", seed=0, max_steps=a.max_steps,
                       exclude_tactics=arm["tactics"],
                       exclude_techniques=arm["techniques"])
        succ, illegal, det, steps = rollout(model, env, CONFIGS["masked"],
                                            episodes=a.episodes)
        rows.append({"seed": s, "success": succ, "illegal": illegal,
                     "detected": det, "steps": steps})
        print(f"RESULT {arm['tag']}{s}: success {succ:5.1f}%  illegal {illegal:4.1f}%  "
              f"detected {det:5.1f}%  steps {steps:.1f}", flush=True)

    def agg(k):
        xs = [r[k] for r in rows]
        return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)

    d = {"arm": a.arm, "label": arm["label"],
         "ablated_tactics": list(arm["tactics"]),
         "ablated_techniques": list(arm["techniques"]), "seeds": seeds,
         "episodes": a.episodes, "rows": rows,
         "mean": {k: agg(k)[0] for k in ("success", "illegal", "detected", "steps")},
         "sd": {k: agg(k)[1] for k in ("success", "illegal", "detected", "steps")}}
    with io.open(arm["out"], "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    m, sd = d["mean"], d["sd"]
    print(f"\n{arm['label']}, {len(seeds)} seeds:")
    print(f"  success  {m['success']:.1f} +/- {sd['success']:.1f}")
    print(f"  detected {m['detected']:.1f} +/- {sd['detected']:.1f}")
    print(f"  steps    {m['steps']:.1f} +/- {sd['steps']:.1f}")
    print(f"  illegal  {m['illegal']:.1f}")
    print(f"\nwrote {arm['out']}")


if __name__ == "__main__":
    main()
