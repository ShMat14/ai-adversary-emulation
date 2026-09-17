# -*- coding: utf-8 -*-
"""
Train one (algorithm, seed) combination for the controlled comparison.

The supervisor asked whether the results could be compared against a study
proposing a different RL algorithm under the same parameters. Rather than port
an external system into this environment, or this agent into theirs -- either of
which changes what is being measured -- the comparison is run here, holding the
environment, the budget and the seed fixed and varying only the algorithm:

    maskable  MaskablePPO with prerequisite masking   (the paper's method)
    ppo       PPO, identical hyperparameters, no mask (isolates masking)
    dqn       DQN, no mask                            (what the literature uses:
                                                       Chaudhary et al. 2020,
                                                       Ghanem & Chen 2020,
                                                       Li et al. 2024)

Hyperparameters are taken from the saved v3 checkpoint rather than from the
manuscript, because the two disagree; the checkpoint is the ground truth.

Telemetry is disabled during training. The original run left 73,164 JSON files
behind, and nine runs would produce well over half a million.

    python analysis/compare_train.py --algo ppo --seed 0
"""
import argparse, os, sys, time, json

# Nine of these run concurrently; without a cap each process would try to use
# every core and they would contend rather than parallelise.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import torch
torch.set_num_threads(2)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.adversary_env import AdversaryEnv

# taken from results/models/ppo_adversary_v3.zip
SHARED = dict(learning_rate=2e-4, gamma=0.995)
PPO_ONLY = dict(n_steps=4096, batch_size=128, n_epochs=15, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.02, vf_coef=0.6, max_grad_norm=0.5)
TIMESTEPS = 800_000
OUTDIR = "results/models/comparison"


class _NullTelemetry:
    """Training does not need per-episode logs, and they are expensive."""
    def start_episode(self):
        pass

    def log_event(self, *a, **k):
        pass

    def end_episode(self):
        pass


def make_env(seed):
    env = AdversaryEnv(config={"max_steps": 40, "real_mode": False})
    env.telemetry = _NullTelemetry()
    env.reset(seed=seed)
    return env


def build(algo, env, seed):
    if algo == "maskable":
        from sb3_contrib import MaskablePPO
        return MaskablePPO("MlpPolicy", env, device="cpu", verbose=0, seed=seed,
                           **SHARED, **PPO_ONLY)
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO("MlpPolicy", env, device="cpu", verbose=0, seed=seed,
                   **SHARED, **PPO_ONLY)
    if algo == "dqn":
        from stable_baselines3 import DQN
        # PPO's on-policy settings have no DQN equivalent. Learning rate,
        # discount, network shape and budget are matched; the remainder are
        # library defaults and this is stated in the write-up.
        return DQN("MlpPolicy", env, device="cpu", verbose=0, seed=seed,
                   **SHARED, policy_kwargs=dict(net_arch=[64, 64]))
    raise SystemExit(f"unknown algorithm: {algo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["maskable", "ppo", "dqn"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--timesteps", type=int, default=TIMESTEPS)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    tag = f"{args.algo}_seed{args.seed}"
    env = make_env(args.seed)
    model = build(args.algo, env, args.seed)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps)
    secs = time.time() - t0

    path = os.path.join(OUTDIR, tag)
    model.save(path)
    with open(os.path.join(OUTDIR, tag + ".meta.json"), "w") as f:
        json.dump({"algo": args.algo, "seed": args.seed,
                   "requested_timesteps": args.timesteps,
                   "actual_timesteps": int(model.num_timesteps),
                   "train_seconds": round(secs, 1)}, f, indent=2)
    print(f"{tag}: {model.num_timesteps:,} steps in {secs/60:.1f} min -> {path}.zip")


if __name__ == "__main__":
    main()
