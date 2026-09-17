# -*- coding: utf-8 -*-
"""
Generate a labelled attack-telemetry dataset from the v4 target.

Drives the trained agent (or a scripted stealthy operator) against the live
server and captures the SIEM feed -- the stream of ATT&CK-mapped Windows/Sysmon
security events every action emits. This is the artefact the whole platform
exists to produce: a labelled dataset a defender could load into a SIEM and use
to build or test detections.

The events are generated in simulation, which is fast and -- because the
simulator and the server share one model (proven by analysis/v4_parity.py) --
produces exactly the telemetry the live server serves at /siem/events. The
server route is the same data over HTTP; this is the same data in bulk.

Writes two files under results/v4_dataset/:
  events.jsonl    one security event per line, with host, Event ID, ATT&CK id,
                  the technique that caused it, and the episode it belongs to
  summary.json    per-technique event counts, which detection rules fired, and
                  the detection outcome of each episode

    python analysis/v4_generate_dataset.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env.adversary_env import AdversaryEnv
from env.kill_chain import ACTION_ORDER

OUT = "results/v4_dataset"


def event_dict(e):
    return {"event_id": e.event_id, "channel": e.channel, "name": e.name,
            "host": e.host, "mitre_id": e.mitre_id, "technique": e.technique,
            "suspicious": e.suspicious}
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


def main(episodes=60):
    os.makedirs(OUT, exist_ok=True)
    env = AdversaryEnv(config={"max_steps": 40})   # sim mode: fast, identical feed
    env.telemetry = _Null()

    events_path = os.path.join(OUT, "events.jsonl")
    ev_count = 0
    per_technique = {}
    per_event_id = {}
    rules_fired = {}
    episodes_meta = []

    with open(events_path, "w", encoding="utf-8") as fh:
        for ep in range(episodes):
            env.reset(seed=70_000 + ep)
            done = False
            while not done:
                _, _, term, trunc, _ = env.step(greedy(env))
                done = term or trunc
            det = env.model.detection
            for e in det.log:
                rec = {"episode": ep, **event_dict(e)}
                fh.write(json.dumps(rec) + "\n")
                ev_count += 1
                per_technique[e.technique] = per_technique.get(e.technique, 0) + 1
                key = f"{e.event_id} ({e.channel})"
                per_event_id[key] = per_event_id.get(key, 0) + 1
            for rule in det.fired_rules:
                rules_fired[rule] = rules_fired.get(rule, 0) + 1
            episodes_meta.append({
                "episode": ep,
                "events": len(det.log),
                "final_suspicion": round(det.suspicion, 3),
                "caught": env.model.state.any_host_detected(),
                "objective": env.model.is_goal(),
            })

    summary = {
        "episodes": episodes,
        "total_events": ev_count,
        "events_per_episode": round(ev_count / episodes, 1),
        "objective_rate_pct": round(100 * sum(m["objective"] for m in episodes_meta) / episodes, 1),
        "detection_rate_pct": round(100 * sum(m["caught"] for m in episodes_meta) / episodes, 1),
        "events_by_technique": dict(sorted(per_technique.items(), key=lambda kv: -kv[1])),
        "events_by_event_id": dict(sorted(per_event_id.items(), key=lambda kv: -kv[1])),
        "detection_rules_fired": dict(sorted(rules_fired.items(), key=lambda kv: -kv[1])),
    }
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("=" * 70)
    print("v4 attack-telemetry dataset generated")
    print("=" * 70)
    print(f"  episodes           : {episodes}")
    print(f"  total events       : {ev_count}  ({summary['events_per_episode']}/episode)")
    print(f"  objective reached  : {summary['objective_rate_pct']}%")
    print(f"  detected by SOC    : {summary['detection_rate_pct']}%")
    print(f"\n  events by ATT&CK technique:")
    for t, n in list(summary["events_by_technique"].items())[:8]:
        print(f"    {n:5d}  {t}")
    print(f"\n  events by Windows/Sysmon Event ID:")
    for e, n in list(summary["events_by_event_id"].items())[:8]:
        print(f"    {n:5d}  Event {e}")
    print(f"\n  written: {events_path}")
    print(f"           {os.path.join(OUT, 'summary.json')}")


if __name__ == "__main__":
    main()
