# agents/ppo_train.py
import sys
import os

# Ensure the project root is on the path when running this script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.adversary_env import AdversaryEnv
from sb3_contrib import MaskablePPO


def main():
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/telemetry/rl", exist_ok=True)

    env = AdversaryEnv(config={
        "telemetry_dir": "results/telemetry/rl",
        "max_steps": 40,
        "real_mode": False,   # Sim mode for training — fast, no HTTP calls
    })

    model = MaskablePPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        # ---------------------------------------------------------------
        # REALISTIC MULTI-TECHNIQUE HYPERPARAMETERS (v3)
        # - Higher entropy: keeps agent exploring all valid techniques
        # - Longer rollout: needed for the now-longer kill chain (~12 steps)
        # - Action masking: only valid actions presented at each state
        # ---------------------------------------------------------------
        learning_rate=2e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=15,
        gamma=0.995,
        ent_coef=0.02,          # Raised from 0.008 — prevents premature technique collapse
        vf_coef=0.6,
        max_grad_norm=0.5,
        tensorboard_log="./results/runs/",
    )

    print(">>> Starting Training v3 (800K timesteps — realistic multi-technique)...")
    model.learn(total_timesteps=800_000)
    model.save("results/models/ppo_adversary_v3")
    print(">>> Training Complete. Model saved as: results/models/ppo_adversary_v3")


if __name__ == "__main__":
    main()