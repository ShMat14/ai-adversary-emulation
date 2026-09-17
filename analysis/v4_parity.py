# -*- coding: utf-8 -*-
"""
Sim-vs-server parity: the whole point of the shared model.

Runs the identical stealthy policy in simulation and against the live v4 server
and compares the outcomes. Because both call env.kill_chain, the success and
detection rates must agree within sampling noise. In v3 the same measurement
showed phishing landing 86.5% in sim against 54.0% on the server; here the gap
should be indistinguishable from zero.

    python Target/mock_server.py        # terminal 1
    python analysis/v4_parity.py        # terminal 2
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import requests

from env.adversary_env import AdversaryEnv
from env.kill_chain import ACTION_ORDER

SERVER = "http://127.0.0.1:5000"
NAME = {n: i for i, n in enumerate(ACTION_ORDER)}
STEALTH = [
    "EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT", "KERBEROASTING",
    "LATERAL_MOVE_SMB", "PASS_THE_HASH", "NETWORK_SCAN", "VALID_ACCOUNTS_LOGIN",
    "SQL_INJECTION", "PHISHING_EMAIL", "POWERSHELL_EXEC", "PRIV_ESC_SUDO",
    "BRUTE_FORCE_SSH", "INSTALL_BACKDOOR", "WEB_SHELL_UPLOAD", "CLEAR_LOGS",
]


class _Null:
    def start_episode(self): pass
    def log_event(self, *a, **k): pass
    def end_episode(self): pass


def greedy(env):
    mask = env.action_masks()
    legal = {ACTION_ORDER[i] for i in np.flatnonzero(mask)}
    win = legal & {"EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT"}
    if not win and "CLEAR_LOGS" in legal and env.alert_level >= 0.45:
        return NAME["CLEAR_LOGS"]
    return NAME[next(n for n in STEALTH if n in legal)]


def run(real_mode, episodes):
    env = AdversaryEnv(config={"max_steps": 40, "real_mode": real_mode})
    env.telemetry = _Null()
    wins = caught = steps = 0
    rewards = []
    for ep in range(episodes):
        env.reset(seed=20_000 + ep)
        total, done, n = 0.0, False, 0
        while not done:
            _, r, term, trunc, _ = env.step(greedy(env))
            total += r
            n += 1
            done = term or trunc
        rewards.append(total)
        steps += n
        if env.model.is_goal():
            wins += 1
        if env.model.state.any_host_detected():
            caught += 1
    return {
        "success": 100.0 * wins / episodes,
        "detection": 100.0 * caught / episodes,
        "reward": float(np.mean(rewards)),
        "steps": steps / episodes,
    }


def server_up():
    try:
        requests.get(f"{SERVER}/health", timeout=1)
        return True
    except Exception:
        return False


def main():
    episodes = 20
    print("=" * 68)
    print(f"sim vs server parity — stealthy policy, {episodes} episodes each")
    print("=" * 68)
    sim = run(False, episodes)
    if not server_up():
        print("server down — start Target/mock_server.py"); sys.exit(1)
    real = run(True, episodes)

    print(f"  {'metric':14s}{'sim':>10}{'server':>10}{'gap':>9}")
    for k, unit in (("success", "%"), ("detection", "%"), ("reward", ""), ("steps", "")):
        g = abs(sim[k] - real[k])
        print(f"  {k:14s}{sim[k]:>9.1f}{unit}{real[k]:>9.1f}{unit}{g:>8.1f}")

    # the headline: success and detection must agree within sampling noise
    ok = abs(sim["success"] - real["success"]) < 8 and abs(sim["detection"] - real["detection"]) < 8
    print("\n  " + ("PARITY OK — sim and server agree within noise"
                    if ok else "PARITY FAILED — investigate"))
    # show a sample of the server's SIEM feed: the labelled telemetry
    env = AdversaryEnv(config={"real_mode": True})
    env.telemetry = _Null()
    env.reset(seed=1)
    for _ in range(6):
        try:
            env.step(greedy(env))
        except Exception:
            break
    try:
        feed = requests.get(f"{SERVER}/siem/events",
                            headers={"Authorization": f"Bearer {env._session}"},
                            timeout=1).json()
        print(f"\n  sample SIEM feed after 6 actions: {feed['count']} events, "
              f"suspicion {feed['suspicion']:.2f}/{feed['incident_threshold']}")
        for e in feed["events"][:6]:
            print(f"    [{e['channel']:>10}] Event {e['event_id']:<5} {e['mitre_id']:<10} "
                  f"{e['host']:<7} {e['name']}")
    except Exception as ex:
        print("  (could not fetch SIEM feed:", ex, ")")


if __name__ == "__main__":
    main()
