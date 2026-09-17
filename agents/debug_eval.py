import os
import sys
from stable_baselines3 import PPO
from env.adversary_env import AdversaryEnv, ACTION_LIST


def debug_agent():
    # 1. Load Environment & Model
    env = AdversaryEnv()
    model_path = "results/models/ppo_adversary_model"

    if not os.path.exists(model_path + ".zip"):
        print("❌ Model not found! Did you finish training?")
        return

    model = PPO.load(model_path)
    print("\n>>> 🕵️‍♂️ ANALYZING ADVANCED AGENT BEHAVIOR <<<\n")

    # 2. Run ONE Episode
    obs, _ = env.reset()
    done = False
    step_count = 0

    print(f"{'STEP':<5} | {'ACTION':<25} | {'REWARD':<8} | {'NOTES'}")
    print("-" * 60)

    while not done:
        step_count += 1
        action_idx, _ = model.predict(obs, deterministic=True)
        action_name = ACTION_LIST[int(action_idx)].name

        obs, reward, done, truncated, info = env.step(action_idx)

        # Add commentary based on the move
        note = ""
        if action_name == "PHISHING_EMAIL":
            note = "🎣 Initial Access"
        elif action_name == "INSTALL_BACKDOOR":
            note = "💾 Persistence (Smart!)"
        elif action_name == "CLEAR_LOGS":
            note = "🧹 Defense Evasion (Stealth)"
        elif action_name == "NETWORK_SCAN":
            note = "📡 Discovery"
        elif action_name == "EXFILTRATE_DATA":
            note = "💰 MISSION ACCOMPLISHED"
        elif "RANSOM" in action_name:
            note = "🔥 DESTRUCTIVE WIN"

        print(f"{step_count:<5} | {action_name:<25} | {reward:5.1f}    | {note}")

        if done or truncated:
            break

    # 3. Final Report
    print("-" * 60)
    if "dc01" in env.state.hosts and env.state.hosts["dc01"].privileged:
        print("🏆 RESULT: DOMAIN CONTROLLER COMPROMISED")
    else:
        print("❌ RESULT: FAILED TO ROOT DC")
    print("=" * 30)


if __name__ == "__main__":
    debug_agent()