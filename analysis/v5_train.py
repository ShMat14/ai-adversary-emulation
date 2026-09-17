# -*- coding: utf-8 -*-
"""Training and evaluation for the v5 environment.

Two things this does that the closest prior work does not.

First, it records the **success rate** through training, not only episode length
and return. Koo et al. (E-NASim, ETRI Journal 2026) report the latter two and
never state what fraction of episodes reached the objective, so their learning
curves cannot be read as competence. Ours can.

Second, it records the **illegal-action rate** through training. That is the
quantity their design makes unobservable: feasibility is enforced inside the
transition function, so an agent may propose infeasible actions indefinitely and
nothing in their reported metrics would show it. They assert their agent "learns
to avoid invalid action sequences" without measuring it. This measures it.

Curves are written per seed so figures can carry a mean and a variance band
across seeds rather than a single run.

    python analysis/v5_train.py --config masked --topology enterprise --seed 0
    python analysis/v5_train.py --mode eval --topology enterprise
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from env.adversary_env_v5 import AdversaryEnvV5

OUT = "results/models/v5"
CURVES = "analysis/v5_curves"
# 400,000, which is what the study in the paper actually ran. The default was
# 600,000 while the runner passed --steps 400000 explicitly, so anyone re-running
# with defaults would have got a different budget from the published one.
STEPS = 400_000
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
EVAL_EPISODES = 200
EVAL_EVERY = 25_000

CONFIGS = {
    "masked":  "maskable",   # the system as intended
    "nomask":  "ppo",        # PPO with the mask removed
    "dqn":     "dqn",        # a value-based family
    "maskdqn": "maskdqn",    # the same family, masked: isolates mask from algorithm
    "maska2c": "maska2c",    # closes the masked/unmasked grid on the second PG method
    "a2c":     "a2c",        # E-NASim future-work item 1
}

PPO_KW = dict(learning_rate=3e-4, gamma=0.995, n_steps=2048, batch_size=128,
              n_epochs=10, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
              vf_coef=0.5, max_grad_norm=0.5)
# DQN settings are aligned with the closest published comparator rather than
# chosen by us, so the baseline cannot be accused of being handicapped. Koo et al.
# (E-NASim, ETRI Journal 2026, Table 3) use learning rate 0.001, batch 256-384,
# hidden width 128, replay 100,000, target update 1,000-3,000, and -- importantly
# -- anneal epsilon over 50,000 of 100,000 steps in their small scenario and
# 100,000 of 300,000 in their large one, i.e. a third to a half of training. Our
# first setting explored for only 20%, which is less than any of theirs; it is
# raised to 0.33 to sit inside their range.
DQN_KW = dict(learning_rate=1e-3, gamma=0.995, buffer_size=100_000,
              learning_starts=5_000, batch_size=256, train_freq=4,
              target_update_interval=3_000, exploration_fraction=0.33,
              exploration_final_eps=0.05)
# A2C: stable-baselines3 defaults to n_steps=5, i.e. an update every five
# transitions. With 60-step episodes and a reward concentrated at the objective
# that is very poor credit assignment, and it collapsed to a single infeasible
# action. Since our PPO baseline gets n_steps=2048, leaving A2C at 5 would mean
# the two on-policy methods were not comparably configured, and A2C's failure
# could be attributed to the rollout length rather than to the absence of a mask.
# Raised to 64 with GAE enabled, which is a standard setting for sparse,
# long-horizon tasks and brings it into the same regime as the PPO control.
A2C_KW = dict(learning_rate=7e-4, gamma=0.995, n_steps=64, gae_lambda=0.95,
              ent_coef=0.02, vf_coef=0.5, max_grad_norm=0.5)


# Episode budget. The shortest winning trajectory on the enterprise estate is 20
# steps and a trained masked policy converges to a mean of about 27, so 60 gives
# roughly twice what a competent policy needs. A budget of 100 was tried first,
# sized from random play rather than learned play, and trained materially worse:
# it oscillated between 0% and 43% success while 60 reached 81%. The longer
# horizon dilutes credit assignment and lets a policy stall without penalty.
#
# It is also far tighter than the comparators -- L-ARLPT allows 500 and E-NASim
# 2000 -- which is defensible here because every attempt is observed by a
# defender and suspicion accumulates, so success cannot be bought with volume.
DEFAULT_MAX_STEPS = 60


def make_env(topology="enterprise", seed=None, max_steps=DEFAULT_MAX_STEPS,
             illegal_penalty=0.0, exclude_tactics=(), exclude_techniques=(),
             objective="admin"):
    return AdversaryEnvV5({"topology": topology, "max_steps": max_steps,
                           "exclude_tactics": exclude_tactics,
                           "exclude_techniques": exclude_techniques,
                           "objective": objective,
                           "illegal_penalty": illegal_penalty, "seed": seed})


# Network width. Widened from [128,128] -- the width E-NASim reports -- to
# [256,256], which is a reasonable default for a 540-action space and is applied
# to every configuration, masked and unmasked alike, so no baseline is
# disadvantaged by it.
#
# It is NOT the fix for the conditional-requirement failure, although an earlier
# note here said it was. That was concluded from one seed: seed 2 went from 0% to
# 80.8% on host-privilege episodes when widened, and seed 0 at the same width
# then gave 0% on that half while seed 0 at the narrow width had managed
# 83-100%. Width does not determine the outcome. The actual cause was a modelling
# error -- the credential action was excluded from the mask instead of being
# denied at execution -- and the fix is the runtime PermErr gate in
# env/kill_chain_v5.py. Training length is separately ruled out: seed 2 was
# continued to 802,000 steps at the narrow width and plateaued.
NET_ARCH = [256, 256]


def build(algo, env, seed):
    net = dict(net_arch=NET_ARCH)
    if algo == "maskable":
        from sb3_contrib import MaskablePPO
        return MaskablePPO("MlpPolicy", env, device="cpu", verbose=0, seed=seed,
                           policy_kwargs=net, **PPO_KW)
    if algo == "dqn":
        from stable_baselines3 import DQN
        return DQN("MlpPolicy", env, device="cpu", verbose=0, seed=seed,
                   policy_kwargs=net, **DQN_KW)
    if algo == "maska2c":
        # MaskablePPO's masking machinery with A2C's update rule and A2C's own
        # RMSprop optimiser, so the pair against the unmasked A2C varies the mask
        # alone. See analysis/maskable_a2c.py.
        from analysis.maskable_a2c import MaskableA2C
        return MaskableA2C("MlpPolicy", env, device="cpu", verbose=0, seed=seed,
                           batch_size=A2C_KW["n_steps"], policy_kwargs=net,
                           **A2C_KW)
    if algo == "maskdqn":
        # Same hyperparameters as the unmasked DQN above, deliberately. The two
        # rows differ in the mask and in nothing else, which is what makes the
        # pair an ablation of the mechanism rather than of the algorithm.
        from analysis.maskable_dqn import MaskableDQN
        return MaskableDQN("MlpPolicy", env, device="cpu", verbose=0, seed=seed,
                           policy_kwargs=net, **DQN_KW)
    if algo == "a2c":
        from stable_baselines3 import A2C
        return A2C("MlpPolicy", env, device="cpu", verbose=0, seed=seed,
                   policy_kwargs=net, **A2C_KW)
    from stable_baselines3 import PPO
    return PPO("MlpPolicy", env, device="cpu", verbose=0, seed=seed,
               policy_kwargs=net, **PPO_KW)


def load(algo, path):
    if algo == "maskable":
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(path, device="cpu")
    if algo == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(path, device="cpu")
    if algo == "maskdqn":
        from analysis.maskable_dqn import MaskableDQN
        return MaskableDQN.load(path, device="cpu")
    if algo == "maska2c":
        from analysis.maskable_a2c import MaskableA2C
        return MaskableA2C.load(path, device="cpu")
    if algo == "a2c":
        from stable_baselines3 import A2C
        return A2C.load(path, device="cpu")
    from stable_baselines3 import PPO
    return PPO.load(path, device="cpu")


def rollout(model, env, algo, episodes=EVAL_EPISODES, base_seed=900_000,
            full=False):
    """Evaluate a policy.

    Returns (success %, illegal %, detection %, mean episode length) by default.
    With `full=True` also returns the decision-behaviour breakdown used by
    Zhan et al. (L-ARLPT, Applied Sciences 2026, Table 3), so results here can be
    read directly against theirs:

      failed      - the action was refused: its preconditions did not hold
      redundant   - the action executed but changed nothing in the world state
      successful  - the action executed and advanced the state
      apd         - average penetration depth: the deepest subnet reached, on the
                    zone hierarchy, averaged over episodes (their Eq. 16)

    `failed` is the quantity E-NASim leaves unmeasured and that L-ARLPT reports at
    74.33% for its best configuration. For a masked agent it is zero by
    construction, because an infeasible pair is never selectable.
    """
    if algo in ("maskable", "maska2c"):
        from sb3_contrib.common.maskable.utils import get_action_masks
    wins = caught = illegal = steps = redundant = advanced = missed = 0
    lengths, depths, win_lengths = [], [], []
    # success split by what the engagement additionally demanded. A single
    # mean hides the failure mode that actually occurs here: a policy that
    # is perfect on the episodes with no extra requirement and absent on
    # the ones that impose host privilege reads as a mediocre 48%, which
    # describes nothing that is true of it.
    by_req = {}
    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        req = (env.model.need_persist, env.model.need_hostpriv,
               env.model.need_collect, env.model.need_c2)
        done, n = False, 0
        while not done:
            legal = env.action_masks()
            if algo in ("maskable", "maska2c"):
                a, _ = model.predict(obs, action_masks=get_action_masks(env),
                                     deterministic=True)
            elif algo == "maskdqn":
                a, _ = model.predict(obs, action_masks=legal, deterministic=True)
            else:
                a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            steps += 1
            n += 1
            if not legal[a]:
                illegal += 1
            obs, r, term, trunc, info = env.step(a)
            if legal[a]:
                if info.get("advanced"):
                    advanced += 1
                elif info.get("success"):
                    # executed cleanly but changed nothing: L-ARLPT's "redundant"
                    redundant += 1
                else:
                    # the technique simply missed its probability roll. That is
                    # stochastic failure, not redundancy, and counting it as
                    # redundant overstates wasted effort -- it inflated our figure
                    # from 26% to 40% before this was separated out.
                    missed += 1
            done = term or trunc
        lengths.append(n)
        slot = by_req.setdefault(req, [0, 0])
        slot[0] += 1
        if env.model.is_goal():
            wins += 1
            win_lengths.append(n)
            slot[1] += 1
        caught += env.model.caught()
        # Penetration depth counts a host as reached if it is held OR the agent
        # holds privilege on it. A golden ticket confers administrative control
        # of the domain controller without ever compromising it, so counting
        # only `compromised` would report the deepest possible outcome as
        # shallower than a lateral move onto a workstation.
        held = [env.model.topo.subnet_of[h] for h in env.model.topo.hosts
                if env.model.host(h).compromised
                or env.model.host(h).privilege > 0]
        depths.append(max(held) if held else 0)
    base = (100.0 * wins / episodes, 100.0 * illegal / max(1, steps),
            100.0 * caught / episodes, float(np.mean(lengths)))
    if not full:
        return base
    return base + (100.0 * redundant / max(1, steps),
                   100.0 * advanced / max(1, steps),
                   float(np.mean(depths)),
                   float(np.mean(win_lengths)) if win_lengths else float("nan"),
                   by_req)


class CurveCallback(BaseCallback):
    """Evaluate periodically and record the curve E-NASim does not report."""

    def __init__(self, algo, topology, seed, every=EVAL_EVERY,
                 max_steps=DEFAULT_MAX_STEPS):
        super().__init__()
        self.algo, self.topology, self.seed, self.every = algo, topology, seed, every
        # must match the training budget: evaluating a 500-step policy under a
        # 60-step cap silently reports the wrong curve
        self.max_steps = max_steps
        self.points = []
        self._next = every

    def _on_step(self):
        if self.num_timesteps >= self._next:
            self._next += self.every
            env = make_env(self.topology, seed=self.seed,
                           max_steps=self.max_steps)
            s, i, d, L = rollout(self.model, env, self.algo, episodes=100,
                                 base_seed=800_000)
            self.points.append({"timesteps": int(self.num_timesteps),
                                "success": s, "illegal": i,
                                "detection": d, "ep_len": L})
            print(f"      {self.num_timesteps:>7,}  success {s:5.1f}%  "
                  f"illegal {i:5.1f}%  detected {d:5.1f}%  len {L:4.1f}", flush=True)
        return True


def train(name, seed, topology="enterprise", steps=STEPS, illegal_penalty=0.0,
          suffix="", max_steps=DEFAULT_MAX_STEPS):
    algo = CONFIGS[name]
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CURVES, exist_ok=True)
    env = make_env(topology, seed=seed, max_steps=max_steps,
                   illegal_penalty=illegal_penalty)
    model = build(algo, env, seed)
    cb = CurveCallback(algo, topology, seed, max_steps=max_steps)
    tag = f"{name}_{topology}_s{seed}{suffix}"
    print(f"--- {tag}: {steps:,} steps ---", flush=True)
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=cb)
    model.save(os.path.join(OUT, tag))
    with open(os.path.join(CURVES, tag + ".json"), "w") as f:
        json.dump({"tag": tag, "algo": algo, "topology": topology, "seed": seed,
                   "illegal_penalty": illegal_penalty, "max_steps": max_steps,
                   "points": cb.points}, f, indent=1)
    print(f"{tag}: done in {(time.time()-t0)/60:.1f} min", flush=True)


REQ_NAMES = ("persistence", "host_privilege", "collection", "c2")


def _nanmean(xs):
    vals = [x for x in xs if x == x]      # drop NaN without warning
    return float(np.mean(vals)) if vals else float("nan")


def _merge_requirements(rows):
    """Mean success per requirement group, with the spread across seeds.

    Reported because the headline mean is not the informative number. See the
    comment in `rollout`.
    """
    groups = {}
    for r in rows:
        for k, v in r["by_requirement"].items():
            groups.setdefault(k, []).append(v["success"])
    return {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                "seeds": len(v)}
            for k, v in sorted(groups.items())}


def evaluate_all(topology="enterprise", episodes=EVAL_EPISODES, seeds=None,
                 max_steps=DEFAULT_MAX_STEPS, suffix=""):
    seeds = seeds or SEEDS
    out = {}
    for name, algo in CONFIGS.items():
        rows = []
        for s in seeds:
            p = os.path.join(OUT, f"{name}_{topology}_s{s}{suffix}")
            if not os.path.exists(p + ".zip"):
                continue
            model = load(algo, p)
            env = make_env(topology, seed=s, max_steps=max_steps)
            succ, ill, det, L, red, adv, apd, wl, by_req = rollout(
                model, env, algo, episodes, full=True)
            rows.append({"seed": s, "success": succ, "illegal": ill,
                         "detection": det, "ep_len": L, "redundant": red,
                         "advanced": adv, "apd": apd, "win_len": wl,
                         "by_requirement": {",".join(
                             n for n, on in zip(REQ_NAMES, k) if on) or "none":
                             {"episodes": v[0], "wins": v[1],
                              "success": 100.0 * v[1] / max(1, v[0])}
                          for k, v in by_req.items()}})
        if rows:
            out[name] = {
                "n": len(rows), "episodes": episodes, "topology": topology,
                "success_mean": float(np.mean([r["success"] for r in rows])),
                "success_sd": float(np.std([r["success"] for r in rows])),
                "illegal_mean": float(np.mean([r["illegal"] for r in rows])),
                "illegal_sd": float(np.std([r["illegal"] for r in rows])),
                "detection_mean": float(np.mean([r["detection"] for r in rows])),
                "ep_len_mean": float(np.mean([r["ep_len"] for r in rows])),
                "redundant_mean": float(np.mean([r["redundant"] for r in rows])),
                "advanced_mean": float(np.mean([r["advanced"] for r in rows])),
                "apd_mean": float(np.mean([r["apd"] for r in rows])),
                # a configuration that never reaches the objective has no winning
                # episodes to average, which is a fact about the result and not
                # an error; report it as NaN rather than warn
                "win_len_mean": _nanmean([r["win_len"] for r in rows]),
                "by_requirement": _merge_requirements(rows),
                "per_seed": rows,
            }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    ap.add_argument("--config", default="masked", choices=list(CONFIGS))
    ap.add_argument("--topology", default="enterprise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--illegal-penalty", type=float, default=0.0)
    ap.add_argument("--suffix", type=str, default="")
    ap.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    if a.mode == "train":
        train(a.config, a.seed, a.topology, a.steps, a.illegal_penalty,
              a.suffix, a.max_steps)
    else:
        seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
        res = evaluate_all(a.topology, a.episodes, seeds, a.max_steps, a.suffix)
        path = a.out or f"analysis/v5_results_{a.topology}.json"
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
        hdr = (f"\n{'config':10s}{'success %':>16}{'failed %':>15}"
               f"{'redundant %':>13}{'advanced %':>12}{'APD':>7}{'steps/win':>11}")
        print(hdr); print("-" * len(hdr))
        for k, v in res.items():
            print(f"{k:10s}{v['success_mean']:9.1f} ± {v['success_sd']:4.1f}"
                  f"{v['illegal_mean']:10.1f} ± {v['illegal_sd']:3.1f}"
                  f"{v['redundant_mean']:13.1f}{v['advanced_mean']:12.1f}"
                  f"{v['apd_mean']:7.2f}{v['win_len_mean']:11.1f}")
        print(f"\nwrote {path}")
