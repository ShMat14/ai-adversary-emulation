# -*- coding: utf-8 -*-
"""Does a twelve-host policy give a head start on a larger estate?

Supplementary Section S1 found that policies trained on the twelve-host estate
do not transfer zero-shot to larger ones: 0.9x the random-legal floor on
sixteen hosts and 0.0x on twenty. Two systems we compare against use transfer
differently, as a starting point rather than a finished policy: INNES (Li et
al., 2023b) migrates a trained model to a new scenario and finds it beats 100
episodes of fresh training there, and GORGO (Giapantzis et al., 2025) carries
weights from one 5G component to the next and cuts training time by 75-81%.
This experiment asks the same question of our policies.

Design, one run per (condition, target, seed):

  warm     load masked_pad_s{seed}, the size-invariant policy trained on the
           twelve-host estate, and continue training on the target estate
  scratch  a fresh MaskablePPO with identical hyperparameters, trained on the
           target estate from nothing

Both use the size-invariant mode (20 slots), so the observation and action
spaces are identical and the only difference is the starting weights. Both get
the same number of training steps and the same 120-step episode budget that S1
uses, chosen there so that random legal play succeeds at a non-zero rate on
every estate. Five seeds, because five size-invariant source policies exist.

Each run evaluates every 25,000 steps on 100 held-out engagements, including
step 0 (zero-shot for warm, untrained for scratch), then a final 200-engagement
evaluation on the target. Warm runs are also evaluated back on the twelve-host
estate, to see whether the head start costs the policy what it knew.

    python analysis/v8_warmstart.py --cond warm --target mid16 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.set_num_threads(1)

from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from analysis.v5_train import NET_ARCH, PPO_KW, build, rollout  # noqa: E402
from env.adversary_env_v5 import AdversaryEnvV5  # noqa: E402

PAD = 20
SOURCE = "results/models/v5_pad/masked_pad_s{}"
OUT_MODELS = "results/models/v8_warm"
OUT_RUNS = "results/v8_warmstart"


def mk(topo, seed, max_steps):
    return AdversaryEnvV5({"topology": topo, "max_steps": max_steps,
                           "pad_hosts": PAD, "seed": seed})


class Curve(BaseCallback):
    def __init__(self, target, seed, max_steps, every, episodes):
        super().__init__()
        self.target, self.seed, self.max_steps = target, seed, max_steps
        self.every, self.episodes = every, episodes
        self.points, self._next = [], every

    def evaluate(self, t):
        s, i, d, L = rollout(self.model, mk(self.target, self.seed, self.max_steps),
                             "maskable", episodes=self.episodes, base_seed=800_000)
        self.points.append({"timesteps": int(t), "success": s, "illegal": i,
                            "detection": d, "ep_len": L})
        print(f"   {t:>7,}  success {s:5.1f}%  illegal {i:4.1f}%  "
              f"detected {d:5.1f}%  len {L:5.1f}", flush=True)

    def _on_step(self):
        if self.num_timesteps >= self._next:
            self._next += self.every
            self.evaluate(self.num_timesteps)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["warm", "scratch"], required=True)
    ap.add_argument("--target", choices=["mid16", "large20"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--every", type=int, default=25_000)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--final-episodes", type=int, default=200)
    a = ap.parse_args()

    from sb3_contrib import MaskablePPO

    tag = f"{a.cond}_{a.target}_s{a.seed}"
    os.makedirs(OUT_MODELS, exist_ok=True)
    os.makedirs(OUT_RUNS, exist_ok=True)
    env = mk(a.target, a.seed, a.max_steps)
    if a.cond == "warm":
        model = MaskablePPO.load(SOURCE.format(a.seed), env=env, device="cpu")
        # Warm and scratch must differ only in the starting weights, so the
        # checkpoint has to carry the study's own hyperparameters. They are
        # checked rather than overwritten: SB3 turns clip_range into a schedule
        # at setup, and n_steps sizes a buffer that already exists.
        for k in ("gamma", "n_steps", "batch_size", "n_epochs", "gae_lambda",
                  "ent_coef", "vf_coef", "max_grad_norm"):
            if getattr(model, k) != PPO_KW[k]:
                raise SystemExit(f"{tag}: checkpoint has {k}={getattr(model, k)}, "
                                 f"study uses {PPO_KW[k]}")
        if model.clip_range(1.0) != PPO_KW["clip_range"] or \
                model.lr_schedule(1.0) != PPO_KW["learning_rate"]:
            raise SystemExit(f"{tag}: checkpoint clip/learning rate differ")
        if list(model.policy.net_arch["pi"] if isinstance(model.policy.net_arch, dict)
                else model.policy.net_arch) != NET_ARCH:
            raise SystemExit(f"{tag}: checkpoint network is {model.policy.net_arch}")
        model.set_random_seed(a.seed)
    else:
        model = build("maskable", env, a.seed)

    cb = Curve(a.target, a.seed, a.max_steps, a.every, a.episodes)
    print(f"--- {tag}: {a.steps:,} steps, budget {a.max_steps} ---", flush=True)
    cb.model = model
    cb.evaluate(0)
    t0 = time.time()
    model.learn(total_timesteps=a.steps, callback=cb, reset_num_timesteps=True)
    minutes = (time.time() - t0) / 60
    model.save(os.path.join(OUT_MODELS, tag))

    s, i, d, L = rollout(model, mk(a.target, a.seed, a.max_steps), "maskable",
                         episodes=a.final_episodes, base_seed=900_000)
    final = {"success": s, "illegal": i, "detection": d, "ep_len": L}
    back = None
    if a.cond == "warm":
        s2, i2, d2, L2 = rollout(model, mk("enterprise", a.seed, a.max_steps),
                                 "maskable", episodes=100, base_seed=900_000)
        back = {"success": s2, "illegal": i2, "detection": d2, "ep_len": L2}
    with open(os.path.join(OUT_RUNS, tag + ".json"), "w") as f:
        json.dump({"tag": tag, "cond": a.cond, "target": a.target, "seed": a.seed,
                   "steps": a.steps, "max_steps": a.max_steps, "pad": PAD,
                   "minutes": minutes, "curve": cb.points, "final": final,
                   "enterprise_after": back}, f, indent=1)
    print(f"{tag}: final {s:.1f}% on {a.target}"
          + (f", {back['success']:.1f}% back on enterprise" if back else "")
          + f"  ({minutes:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
