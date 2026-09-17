# -*- coding: utf-8 -*-
"""
Baseline diagnostic for the v4 rebuild.

Two independent tests, because "does every action work" and "do sim and server
agree" are different questions and conflating them produced a misleading first
draft of this file.

TEST 1 - capability, per action, deterministic.
    For each implemented action, construct exactly the state its precondition
    requires, force the technique to succeed (success probability 1, zero alert
    so nothing blocks), fire it once, and assert the documented state change
    happened. This isolates the environment's *logic* from its randomness: it
    answers "given that the technique lands, does the effect apply" with no luck
    involved. Runs in sim mode only, since it manipulates internal state.

TEST 2 - a full kill chain, greedy, both modes.
    Play toward a domain-controller compromise by always taking the highest
    priority legal (masked) action, retrying a stochastic failure a bounded
    number of times. Report the clean sequence and whether the objective was
    reached, in sim and against the live server. Divergence between the two is
    the specification for the server rewrite.

    python Target/mock_server.py            # terminal 1, for TEST 2 real mode
    python analysis/v4_diagnose.py          # terminal 2
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import requests

from env.adversary_env import AdversaryEnv
from env.attack_actions import ACTION_LIST

NAME_TO_IDX = {a.name: i for i, a in enumerate(ACTION_LIST)}
SERVER = "http://127.0.0.1:5000"


class _Null:
    def start_episode(self): pass
    def log_event(self, *a, **k): pass
    def end_episode(self): pass


def server_up():
    try:
        requests.get(f"{SERVER}/status", timeout=1)
        return True
    except Exception:
        return False


def fresh(real_mode=False, max_steps=80):
    env = AdversaryEnv(config={"max_steps": max_steps, "real_mode": real_mode})
    env.telemetry = _Null()
    env.reset(seed=0)
    env.sqli_available = False
    return env


def host(env, name):
    return env.state.hosts[name]


# ─────────────────────────────────────────────────────────────────────────
# TEST 1 — every action, given success, produces its effect
# ─────────────────────────────────────────────────────────────────────────
# Each case: set up the precondition, then assert a predicate on the resulting
# state. `setup` mutates a fresh env into the state the action needs; `effect`
# returns True if the action did what it claims.
CASES = [
    ("PHISHING_EMAIL",
     lambda e: None,
     lambda e: host(e, "user01").compromised),

    ("SQL_INJECTION",
     lambda e: setattr(e, "sqli_available", True),
     lambda e: host(e, "user01").compromised and host(e, "user01").has_credentials),

    ("BRUTE_FORCE_SSH",
     lambda e: None,
     lambda e: any(host(e, n).compromised for n in e.host_names)),

    ("NETWORK_SCAN",
     lambda e: setattr(host(e, "user01"), "compromised", True),
     lambda e: e.network_scanned),

    ("VALID_ACCOUNTS_LOGIN",
     lambda e: setattr(host(e, "user01"), "compromised", True),
     lambda e: host(e, "user01").has_credentials),

    ("INSTALL_BACKDOOR",
     lambda e: setattr(host(e, "user01"), "compromised", True),
     lambda e: e.backdoor_installed),

    ("WEB_SHELL_UPLOAD",
     lambda e: setattr(host(e, "user01"), "compromised", True),
     lambda e: e.backdoor_installed),

    ("PRIV_ESC_SUDO",
     lambda e: setattr(host(e, "user01"), "compromised", True),
     lambda e: host(e, "user01").privileged),

    ("POWERSHELL_EXEC",
     lambda e: setattr(host(e, "user01"), "compromised", True),
     lambda e: host(e, "user01").privileged),

    ("CLEAR_LOGS",
     lambda e: setattr(e, "alert_level", 2.0),
     lambda e: e.alert_level < 2.0),

    # lateral movement: user01 owned with creds, network scanned -> take srv01
    ("LATERAL_MOVE_SMB",
     lambda e: (setattr(host(e, "user01"), "compromised", True),
                setattr(host(e, "user01"), "has_credentials", True),
                setattr(e, "network_scanned", True)),
     lambda e: host(e, "srv01").compromised),

    ("PASS_THE_HASH",
     lambda e: (setattr(host(e, "user01"), "compromised", True),
                setattr(host(e, "user01"), "has_credentials", True)),
     lambda e: host(e, "srv01").compromised),

    # DC endgame reached: dc compromised, creds + scan present
    ("KERBEROASTING",
     lambda e: (setattr(host(e, "user01"), "compromised", True),
                setattr(host(e, "user01"), "has_credentials", True),
                setattr(e, "network_scanned", True),
                setattr(host(e, "dc01"), "compromised", True)),
     lambda e: host(e, "dc01").privileged),

    ("EXFILTRATE_DATA",
     lambda e: (setattr(host(e, "dc01"), "compromised", True),
                setattr(host(e, "dc01"), "privileged", True)),
     lambda e: True),   # terminates with reward; effect is the win, checked via reward

    ("RANSOMWARE_ENCRYPT",
     lambda e: (setattr(host(e, "dc01"), "compromised", True),
                setattr(host(e, "dc01"), "privileged", True)),
     lambda e: True),
]


def test_capabilities():
    print("=" * 74)
    print("TEST 1 — every action produces its effect when it succeeds (sim, deterministic)")
    print("=" * 74)
    results = []
    for name, setup, effect in CASES:
        idx = NAME_TO_IDX[name]
        env = fresh()
        setup(env)
        # isolate the logic from the stochastic layer entirely: force the
        # technique to land (prob 1) AND remove its noise, so the alert stays 0
        # and the block chance (alert * 0.2 in sim) is exactly zero. Firing once
        # is then a clean test of "given success, does the effect apply".
        act = ACTION_LIST[idx]
        saved_p, saved_n = act.success_prob, act.noise_level
        act.success_prob, act.noise_level = 1.0, 0.0
        if name != "CLEAR_LOGS":
            env.alert_level = 0.0

        offered = bool(env.action_masks()[idx])
        obs, reward, term, trunc, _ = env.step(idx)
        act.success_prob, act.noise_level = saved_p, saved_n

        ok = effect(env)
        # for the two win actions, the signal is a large positive terminal reward
        if name in ("EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT"):
            ok = reward > 100 and term

        flag = "ok  " if (ok and offered) else "FAIL"
        note = ""
        if not offered:
            note += " [not offered by mask in its own precondition state]"
        if not ok:
            note += " [effect did NOT apply]"
        print(f"  [{flag}] {name:22s} reward={reward:7.1f}{note}")
        results.append((name, ok and offered))
    n_ok = sum(1 for _, ok in results if ok)
    print(f"\n  {n_ok}/{len(results)} actions pass the capability test")
    return results


# ─────────────────────────────────────────────────────────────────────────
# TEST 2 — greedy kill chain, both modes
# ─────────────────────────────────────────────────────────────────────────
# Priority order an ideal operator follows. Among legal actions, the earliest
# in this list is taken. CLEAR_LOGS is deliberately low so we can see whether
# the agent needs it, not have the probe lean on it.
PRIORITY = [
    "EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT",
    "KERBEROASTING", "POWERSHELL_EXEC", "PRIV_ESC_SUDO",
    "LATERAL_MOVE_SMB", "PASS_THE_HASH",
    "NETWORK_SCAN", "VALID_ACCOUNTS_LOGIN",
    "PHISHING_EMAIL", "SQL_INJECTION", "BRUTE_FORCE_SSH",
    "INSTALL_BACKDOOR", "WEB_SHELL_UPLOAD",
    "CLEAR_LOGS",
]


def greedy_chain(real_mode):
    label = "REAL (live server)" if real_mode else "SIM (probabilistic)"
    print("\n" + "=" * 74)
    print(f"TEST 2 — greedy kill chain — {label}")
    print("=" * 74)
    env = fresh(real_mode=real_mode, max_steps=120)

    sequence = []
    for _ in range(120):
        mask = env.action_masks()
        legal = {ACTION_LIST[i].name for i in np.flatnonzero(mask)}
        pick = next((n for n in PRIORITY if n in legal), None)
        if pick is None:
            break
        idx = NAME_TO_IDX[pick]
        obs, reward, term, trunc, _ = env.step(idx)
        sequence.append((pick, round(reward, 1)))
        if term or trunc:
            break

    dc = host(env, "dc01")
    reached = dc.privileged
    print("  sequence:")
    for i, (a, r) in enumerate(sequence, 1):
        print(f"    {i:2d}. {a:22s} r={r:+7.1f}")
    print(f"\n  steps={len(sequence)}  DC compromised={dc.compromised}  "
          f"DC root={dc.privileged}  objective_reached={reached}  "
          f"detected={env.state.any_host_detected()}")
    return reached, sequence


# ─────────────────────────────────────────────────────────────────────────
# TEST 3 — per-action success rate, sim vs live server
# ─────────────────────────────────────────────────────────────────────────
# The greedy chain hides probability drift because it retries until success.
# Here we fire each entry technique from a clean, precondition-satisfied state
# many times and record how often it lands, in each mode. This is where the
# sim and the server are expected to disagree, and by how much.
PROBE = [
    ("PHISHING_EMAIL",  lambda e: None,                                    "/phish"),
    ("BRUTE_FORCE_SSH", lambda e: None,                                    "/brute"),
    ("SQL_INJECTION",   lambda e: setattr(e, "sqli_available", True),      "/sqli"),
]


def _reset_server():
    try:
        requests.post(f"{SERVER}/reset", json={"sqli_available": True}, timeout=1)
    except Exception:
        pass


def success_rates(real_mode, trials=200):
    rates = {}
    for name, setup, _ in PROBE:
        idx = NAME_TO_IDX[name]
        landed = 0
        for _ in range(trials):
            env = fresh(real_mode=real_mode)
            if real_mode:
                _reset_server()
            setup(env)
            env.alert_level = 0.0
            before = any(host(env, n).compromised for n in env.host_names) or host(env, "user01").has_credentials
            env.step(idx)
            after = any(host(env, n).compromised for n in env.host_names) or host(env, "user01").has_credentials
            if after and not before:
                landed += 1
        rates[name] = 100.0 * landed / trials
    return rates


def test_probabilities():
    print("\n" + "=" * 74)
    print("TEST 3 — per-action success rate, sim vs live server (200 trials each)")
    print("=" * 74)
    sim = success_rates(real_mode=False)
    real = success_rates(real_mode=True) if server_up() else None
    print(f"  {'action':22s}{'declared':>10}{'sim %':>9}{'server %':>10}{'gap':>8}")
    for name, _, _ in PROBE:
        idx = NAME_TO_IDX[name]
        decl = ACTION_LIST[idx].success_prob * 100
        s = sim[name]
        r = real[name] if real else float("nan")
        gap = abs(s - r) if real else float("nan")
        print(f"  {name:22s}{decl:>9.0f}%{s:>8.1f}%{r:>9.1f}%{gap:>7.1f}")
    if real is None:
        print("  (server down — server column omitted)")


def main():
    print("Baseline diagnostic — current environment, before the v4 rebuild\n")
    test_capabilities()
    test_probabilities()
    sim_ok, _ = greedy_chain(real_mode=False)
    if server_up():
        real_ok, _ = greedy_chain(real_mode=True)
    else:
        print("\n(server down — skipping real mode; start Target/mock_server.py)")
        real_ok = None

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  SIM  objective reached: {sim_ok}")
    if real_ok is not None:
        print(f"  REAL objective reached: {real_ok}")


if __name__ == "__main__":
    main()
