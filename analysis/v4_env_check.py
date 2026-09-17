# -*- coding: utf-8 -*-
"""
Env-level smoke test for the v4 AdversaryEnv (sim mode).

Checks the Gym integration around the shared model: observation and action
shapes, the reward's key properties, and that a stealthy greedy policy can solve
the environment at a sensible rate. Prints a difficulty read before any RL is
trained, so training results have a baseline to sit against.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env.adversary_env import AdversaryEnv
from env.kill_chain import ACTION_ORDER

NAME = {n: i for i, n in enumerate(ACTION_ORDER)}
FAILS = []


def check(cond, msg):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILS.append(msg)


def _null(env):
    class N:
        def start_episode(self): pass
        def log_event(self, *a, **k): pass
        def end_episode(self): pass
    env.telemetry = N()
    return env


# full stealth-ordered priority over all 25 techniques, so the greedy fallback
# always has a legal pick whatever the scenario makes available
STEALTH = [
    "EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT",
    "KERBEROASTING", "DCSYNC", "GPO_MODIFICATION", "GOLDEN_TICKET",
    "LATERAL_MOVE_SMB", "PASS_THE_TICKET", "PASS_THE_HASH",
    "DOMAIN_TRUST_DISCOVERY", "DOMAIN_ACCT_DISCOVERY", "NETWORK_SCAN",
    "VALID_ACCOUNTS_LOGIN", "AS_REP_ROASTING", "CRED_DUMP_LSASS",
    "SQL_INJECTION", "PHISHING_EMAIL",
    "POWERSHELL_EXEC", "PRIV_ESC_SUDO", "ACCOUNT_MANIPULATION",
    "PASSWORD_SPRAYING", "BRUTE_FORCE_SSH",
    "INSTALL_BACKDOOR", "WEB_SHELL_UPLOAD", "CLEAR_LOGS",
]


def greedy_action(env):
    mask = env.action_masks()
    legal = {ACTION_ORDER[i] for i in np.flatnonzero(mask)}
    win = legal & {"EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT"}
    # clear logs when suspicion is genuinely high and no win is available
    if not win and "CLEAR_LOGS" in legal and env.alert_level >= 0.45:
        return NAME["CLEAR_LOGS"]
    pick = next((n for n in STEALTH if n in legal), None)
    return NAME[pick]


def test_shapes():
    print("1. gym integration")
    env = _null(AdversaryEnv(config={"max_steps": 40}))
    obs, _ = env.reset(seed=0)
    check(obs.shape == env.observation_space.shape, f"observation shape {obs.shape} matches space {env.observation_space.shape}")
    check(env.action_space.n == 25, "action space has 25 actions")
    m = env.action_masks()
    check(m.shape == (25,) and m.dtype == bool, "mask is 25 bools")


def test_reward_properties():
    print("2. reward properties")
    # CLEAR_LOGS earns no positive reward (the anti-exploit)
    env = _null(AdversaryEnv(config={"max_steps": 40}))
    env.reset(seed=1)
    env.model.host("user01").compromised = True
    env.model.alert = 0.6
    _, r, _, _, _ = env.step(NAME["CLEAR_LOGS"])
    check(r <= 0.0, f"CLEAR_LOGS reward {r:.2f} is not positive")

    # reaching the DC root pays the objective reward; winning pays big
    env.reset(seed=2)
    env.model.host("dc01").compromised = True
    env.model.host("dc01").privileged = True
    env.sqli_available = False
    _, r, term, _, _ = env.step(NAME["EXFILTRATE_DATA"])
    check(r > 300 and term, f"successful exfiltration pays {r:.1f} and terminates")


def test_difficulty(episodes=400):
    print(f"3. stealthy greedy policy over {episodes} episodes")
    env = _null(AdversaryEnv(config={"max_steps": 40}))
    wins = caught = 0
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        total, done = 0.0, False
        while not done:
            a = greedy_action(env)
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
        rewards.append(total)
        if env.model.is_goal():
            wins += 1
        if env.model.state.any_host_detected():
            caught += 1
    print(f"       success rate : {100*wins/episodes:5.1f}%")
    print(f"       detection    : {100*caught/episodes:5.1f}%")
    print(f"       mean reward  : {np.mean(rewards):7.1f}")
    # a competent scripted operator should solve most episodes, but not all --
    # detection makes some runs unwinnable, which is the point
    check(0.55 <= wins / episodes <= 0.98,
          f"stealthy success rate {100*wins/episodes:.1f}% is in a learnable band")


def main():
    print("=" * 62)
    print("v4 AdversaryEnv smoke test (sim mode)")
    print("=" * 62)
    test_shapes()
    test_reward_properties()
    test_difficulty()
    print("=" * 62)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s)")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
