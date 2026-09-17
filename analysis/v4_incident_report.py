# -*- coding: utf-8 -*-
"""
Blue-team incident report from an episode's telemetry.

Runs one attack against the target, then reconstructs it the way a defender
would after the fact: a step-by-step timeline of the security events, the hosts
that were touched and how far each was compromised, the ATT&CK techniques
observed (mapped to tactics), which detection rules fired and at which step, and
concrete recommendations. This is the defender-facing companion to the attack
telemetry -- it turns the raw event stream into an account of what happened and
what to do about it.

    python analysis/v4_incident_report.py                 # markdown to stdout + file
    python analysis/v4_incident_report.py --seed 123      # a specific episode

Writes results/v4_dataset/incident_report.md.
"""
import argparse
import os
import sys
from collections import defaultdict, OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env.adversary_env import AdversaryEnv
from env.kill_chain import ACTION_ORDER, TECHNIQUES
from env.detection import PROFILES

OUT = "results/v4_dataset/incident_report.md"
NAME = {n: i for i, n in enumerate(ACTION_ORDER)}
STEALTH = [
    "EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT", "KERBEROASTING", "DCSYNC",
    "LATERAL_MOVE_SMB", "PASS_THE_HASH", "DOMAIN_ACCT_DISCOVERY", "NETWORK_SCAN",
    "VALID_ACCOUNTS_LOGIN", "CRED_DUMP_LSASS", "SQL_INJECTION", "PHISHING_EMAIL",
    "POWERSHELL_EXEC", "PRIV_ESC_SUDO", "BRUTE_FORCE_SSH",
    "INSTALL_BACKDOOR", "WEB_SHELL_UPLOAD", "CLEAR_LOGS",
]


class _Null:
    def start_episode(self): pass
    def log_event(self, *a, **k): pass
    def end_episode(self): pass


# a loud operator: prefers the noisiest route (brute force, LSASS dumping,
# DCSync, ransomware) and never manages its footprint. Used to show the SOC
# catching an unsophisticated intrusion, the contrast to the stealthy one.
LOUD = [
    "RANSOMWARE_ENCRYPT", "EXFILTRATE_DATA",
    "DCSYNC", "POWERSHELL_EXEC", "PRIV_ESC_SUDO",
    "PASS_THE_HASH", "LATERAL_MOVE_SMB",
    "NETWORK_SCAN", "CRED_DUMP_LSASS",
    "BRUTE_FORCE_SSH", "PHISHING_EMAIL", "SQL_INJECTION",
    "DOMAIN_ACCT_DISCOVERY", "VALID_ACCOUNTS_LOGIN",
    "INSTALL_BACKDOOR", "WEB_SHELL_UPLOAD", "KERBEROASTING",
]


def greedy(env, loud=False):
    mask = env.action_masks()
    legal = {ACTION_ORDER[i] for i in np.flatnonzero(mask)}
    order = LOUD if loud else STEALTH
    if not loud:
        win = legal & {"EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT"}
        if not win and "CLEAR_LOGS" in legal and env.alert_level >= 0.45:
            return NAME["CLEAR_LOGS"]
    return NAME[next(n for n in order if n in legal)]


# ATT&CK tactic names are already on each Technique; recommendations are keyed
# by technique so the report can suggest a concrete mitigation per observation.
RECS = {
    "PHISHING_EMAIL": "Harden the mail gateway and user-report workflow; alert on mail-client child processes (T1566).",
    "SQL_INJECTION": "Patch the web application and put it behind a WAF; parameterise queries (T1190).",
    "BRUTE_FORCE_SSH": "Enforce account lockout and rate-limiting; alert on 4625 bursts (T1110).",
    "NETWORK_SCAN": "Segment the network and alert on high connection-fan-out from a single host (T1046).",
    "DOMAIN_ACCT_DISCOVERY": "Monitor 4661 directory queries for anomalous enumeration volume (T1087.002).",
    "VALID_ACCOUNTS_LOGIN": "Enforce MFA and flag logons at anomalous hours or from new hosts (T1078).",
    "CRED_DUMP_LSASS": "Enable Credential Guard and alert on handle opens to lsass.exe, Sysmon Event 10 (T1003.001).",
    "LATERAL_MOVE_SMB": "Restrict admin-share access and alert on inter-workstation type-3 logons (T1021).",
    "PASS_THE_HASH": "Deploy Credential Guard and monitor NTLM logon anomalies (T1550.002).",
    "KERBEROASTING": "Use long service-account passwords / gMSAs; alert on RC4 4769 requests (T1558.003).",
    "DCSYNC": "Restrict replication rights; alert on 4662 replication from non-DC hosts (T1003.006).",
    "POWERSHELL_EXEC": "Enable script-block logging and constrained language mode (T1059.001).",
    "PRIV_ESC_SUDO": "Patch privilege-escalation vectors and monitor for exploit processes (T1068).",
    "INSTALL_BACKDOOR": "Alert on new service installs, Event 7045 (T1543).",
    "WEB_SHELL_UPLOAD": "Restrict web-root writes and alert on web-server-spawned shells (T1505.003).",
    "CLEAR_LOGS": "Forward logs off-host in real time; Event 1102 must page the SOC immediately (T1070).",
    "EXFILTRATE_DATA": "Apply DLP and egress filtering; alert on anomalous outbound volume (T1041).",
    "RANSOMWARE_ENCRYPT": "Maintain offline backups and alert on mass file modification (T1486).",
}


def run_episode(seed, loud=False):
    """Play one episode, recording a per-step trace of what happened."""
    env = AdversaryEnv(config={"max_steps": 40})
    env.telemetry = _Null()
    env.reset(seed=seed)
    trace = []
    ev_before = 0
    while True:
        a = greedy(env, loud=loud)
        name = ACTION_ORDER[a]
        suspicion_before = env.model.alert
        _, r, term, trunc, info = env.step(a)
        det = env.model.detection
        new_events = det.log[ev_before:]
        ev_before = len(det.log)
        # did a rule fire on this step?
        fired = len(det.fired_rules)
        trace.append({
            "step": env.current_step,
            "technique": name,
            "mitre": TECHNIQUES[name].mitre_id,
            "tactic": TECHNIQUES[name].tactic,
            "target": env.model.target_host(name) if False else None,
            "success": info["success"],
            "detected": info["detected"],
            "events": new_events,
            "suspicion": round(env.model.alert, 3),
        })
        if term or trunc:
            break
    return env, trace


def build_report(seed, loud=False):
    env, trace = run_episode(seed, loud=loud)
    m = env.model
    caught = m.state.any_host_detected()
    objective = m.is_goal()
    fired_rules = list(OrderedDict.fromkeys(m.detection.fired_rules))

    lines = []
    lines.append(f"# Incident Report — Episode {seed}")
    lines.append("")
    lines.append("_Generated by the v4 adversary-emulation platform from the target's "
                 "SIEM telemetry. Every event below is a real Windows/Sysmon event ID "
                 "mapped to the ATT&CK technique that produced it._")
    lines.append("")

    # ── executive summary ──
    outcome = ("**contained** — the SOC escalated to incident response before the "
               "objective was met" if caught else
               "**not contained** — the intrusion reached its objective undetected"
               if objective else "**incomplete** — the intrusion stalled")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Outcome: {outcome}")
    lines.append(f"- Objective (domain-controller compromise + exfiltration): "
                 f"{'REACHED' if objective else 'not reached'}")
    lines.append(f"- Detected by SOC: {'yes' if caught else 'no'}")
    lines.append(f"- Attacker actions observed: {len(trace)}")
    lines.append(f"- Security events generated: {len(m.detection.log)}")
    lines.append(f"- Final SOC suspicion: {round(m.detection.suspicion, 2)} "
                 f"(incident threshold 1.0)")
    lines.append("")

    # ── host impact ──
    lines.append("## Host impact")
    lines.append("")
    lines.append("| Host | Compromised | Credentials stolen | Privilege escalated | Detected on |")
    lines.append("|---|---|---|---|---|")
    for n in m.HOSTS:
        h = m.host(n)
        lines.append(f"| `{n}` | {'yes' if h.compromised else '—'} | "
                     f"{'yes' if h.has_credentials else '—'} | "
                     f"{'yes' if h.privileged else '—'} | "
                     f"{'yes' if h.detected else '—'} |")
    lines.append("")

    # ── attack timeline ──
    lines.append("## Attack timeline")
    lines.append("")
    lines.append("| Step | Tactic | Technique | ATT&CK | Result | Events | Suspicion |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in trace:
        evids = ", ".join(f"{e.event_id}" for e in t["events"]) or "—"
        result = "detected → IR" if t["detected"] else ("success" if t["success"] else "failed")
        lines.append(f"| {t['step']} | {t['tactic']} | {t['technique']} | "
                     f"{t['mitre']} | {result} | {evids} | {t['suspicion']} |")
    lines.append("")

    # ── techniques observed ──
    by_tactic = defaultdict(list)
    for t in trace:
        if t["technique"] not in [x.split(" ")[0] for x in by_tactic[t["tactic"]]]:
            by_tactic[t["tactic"]].append(f"{t['technique']} ({t['mitre']})")
    lines.append("## ATT&CK techniques observed, by tactic")
    lines.append("")
    for tactic, techs in by_tactic.items():
        lines.append(f"- **{tactic}**: {', '.join(sorted(set(techs)))}")
    lines.append("")

    # ── detections ──
    lines.append("## Detection rules that fired")
    lines.append("")
    if fired_rules:
        for rule in fired_rules:
            lines.append(f"- {rule}")
    else:
        lines.append("- None. The intrusion stayed below every rule threshold — "
                     "a gap the recommendations below address.")
    lines.append("")

    # ── recommendations ──
    seen = OrderedDict()
    for t in trace:
        if t["technique"] in RECS:
            seen[t["technique"]] = RECS[t["technique"]]
    lines.append("## Recommendations")
    lines.append("")
    lines.append("Prioritised by the techniques actually observed in this intrusion:")
    lines.append("")
    for i, (tech, rec) in enumerate(seen.items(), 1):
        lines.append(f"{i}. {rec}")
    lines.append("")
    lines.append("---")
    lines.append("_This report is reproducible: rerun "
                 "`python analysis/v4_incident_report.py --seed "
                 f"{seed}` to regenerate it from the same episode._")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--loud", action="store_true",
                    help="use a noisy operator, to show the SOC catching the attack")
    ap.add_argument("--both", action="store_true",
                    help="write both a stealthy (evaded) and a loud (caught) report")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    if a.both:
        stealth = build_report(a.seed, loud=False)
        loud = build_report(a.seed, loud=True)
        with open(OUT, "w", encoding="utf-8") as f:
            f.write(stealth + "\n")
        loud_path = OUT.replace(".md", "_loud.md")
        with open(loud_path, "w", encoding="utf-8") as f:
            f.write(loud + "\n")
        print(stealth)
        print("\n\n" + "=" * 70 + "\n\n")
        print(loud)
        print(f"\n[written {OUT} and {loud_path}]")
        return

    report = build_report(a.seed, loud=a.loud)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\n[written to {OUT}]")


if __name__ == "__main__":
    main()
