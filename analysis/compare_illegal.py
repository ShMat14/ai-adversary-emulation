# -*- coding: utf-8 -*-
"""
Count how often each agent selects an action whose preconditions are unmet.

The outcome metrics show masked and unmasked PPO performing alike, which raises
the obvious question of what the mask is actually doing. This measures it
directly: at every step the environment's own action_masks() is consulted for
the ground truth of which actions are legal, and the agent's choice is checked
against it. The mask is never used to constrain the unmasked agents, only to
score them.

    python analysis/compare_illegal.py --episodes 200
"""
import argparse, os, sys, json, statistics

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.adversary_env import AdversaryEnv
from env.attack_actions import ACTION_LIST

MODELS = "results/models/comparison"


class _NullTelemetry:
    def start_episode(self): pass
    def log_event(self, *a, **k): pass
    def end_episode(self): pass


def load(algo, path):
    if algo == "maskable":
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(path, device="cpu")
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(path, device="cpu")
    from stable_baselines3 import DQN
    return DQN.load(path, device="cpu")


def run(algo, path, episodes, seed):
    model = load(algo, path)
    env = AdversaryEnv(config={"max_steps": 40, "real_mode": False})
    env.telemetry = _NullTelemetry()

    illegal = total = 0
    eps_with_illegal = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 100000 + ep)
        done = False
        bad_here = 0
        while not done:
            legal = env.action_masks()          # ground truth, not a constraint
            if algo == "maskable":
                a, _ = model.predict(obs, action_masks=legal, deterministic=True)
            else:
                a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            total += 1
            if not legal[a]:
                illegal += 1
                bad_here += 1
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
        if bad_here:
            eps_with_illegal += 1

    return {
        "illegal_actions": illegal,
        "total_actions": total,
        "illegal_pct": 100.0 * illegal / total if total else 0.0,
        "episodes_with_illegal_pct": 100.0 * eps_with_illegal / episodes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args()

    out = {}
    for algo in ("maskable", "ppo", "dqn"):
        rows = []
        for seed in (0, 1, 2):
            p = os.path.join(MODELS, f"{algo}_seed{seed}")
            if not os.path.exists(p + ".zip"):
                continue
            r = run(algo, p, args.episodes, seed)
            rows.append(r)
            print(f"  {algo:9s} seed {seed}: {r['illegal_pct']:5.2f}% of actions illegal, "
                  f"{r['episodes_with_illegal_pct']:5.1f}% of episodes affected")
        if rows:
            out[algo] = {
                "illegal_pct_mean": statistics.mean(x["illegal_pct"] for x in rows),
                "illegal_pct_sd": statistics.stdev(x["illegal_pct"] for x in rows) if len(rows) > 1 else 0.0,
                "episodes_affected_mean": statistics.mean(x["episodes_with_illegal_pct"] for x in rows),
                "per_seed": rows,
            }

    print("\n" + "=" * 72)
    print("ILLEGAL ACTION SELECTION (mask used only to score, not to constrain)")
    print("=" * 72)
    LBL = {"maskable": "MaskablePPO (masked)", "ppo": "PPO (no mask)", "dqn": "DQN (no mask)"}
    print(f"{'agent':24s}{'illegal actions':>18}{'episodes affected':>20}")
    print("-" * 72)
    for a, v in out.items():
        print(f"{LBL[a]:24s}{v['illegal_pct_mean']:12.2f} +/-{v['illegal_pct_sd']:<4.2f}"
              f"{v['episodes_affected_mean']:16.1f}%")

    with open("analysis/comparison_illegal.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwritten to analysis/comparison_illegal.json")


if __name__ == "__main__":
    main()
