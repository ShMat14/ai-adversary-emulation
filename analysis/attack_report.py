"""
analysis/attack_report.py

Generates a structured post-attack analysis report from a telemetry episode JSON.
Similar in style to ANY.RUN sandbox reports, but tailored to the RL agent kill chain.

Usage:
    python analysis/attack_report.py                         # report latest episode
    python analysis/attack_report.py --episode <file.json>   # report specific episode
    python analysis/attack_report.py --all                   # summarise all episodes
"""

import sys, os, json, argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── MITRE & WEAKNESS KNOWLEDGE BASE ───────────────────────────────────────

ATTACK_META = {
    "PHISHING_EMAIL": {
        "tactic": "Initial Access", "technique": "Spearphishing Link", "id": "T1566",
        "description": "Malicious email delivered to user endpoint to establish initial foothold.",
        "weakness": {
            "host": "user01",
            "finding": "Email gateway failed to quarantine malicious payload.",
            "recommendation": "Deploy sandboxed email filtering (e.g. Defender for Office 365).",
        },
    },
    "BRUTE_FORCE_SSH": {
        "tactic": "Credential Access", "technique": "Brute Force: Password Guessing", "id": "T1110",
        "description": "Repeated SSH login attempts against exposed service port.",
        "weakness": {
            "host": "target",
            "finding": "SSH service exposed with no rate-limiting or account lockout policy.",
            "recommendation": "Enforce fail2ban / account lockout after 5 failed attempts.",
        },
    },
    "NETWORK_SCAN": {
        "tactic": "Discovery", "technique": "Network Service Discovery", "id": "T1046",
        "description": "Internal network topology mapped after gaining initial foothold.",
        "weakness": {
            "host": "network",
            "finding": "No internal east-west traffic inspection or NDR alerting in place.",
            "recommendation": "Deploy Network Detection & Response (NDR) for lateral traffic.",
        },
    },
    "VALID_ACCOUNTS_LOGIN": {
        "tactic": "Defense Evasion", "technique": "Valid Accounts", "id": "T1078",
        "description": "Harvested credentials used to authenticate as a legitimate user.",
        "weakness": {
            "host": "srv01",
            "finding": "No MFA enforced. Credential reuse from compromised host went undetected.",
            "recommendation": "Enforce MFA on all privileged accounts and monitor for anomalous logins.",
        },
    },
    "INSTALL_BACKDOOR": {
        "tactic": "Persistence", "technique": "Create/Modify System Process", "id": "T1543",
        "description": "Persistent backdoor installed to survive reboots and maintain re-entry.",
        "weakness": {
            "host": "target",
            "finding": "No integrity monitoring (FIM) on system service directories.",
            "recommendation": "Deploy auditd / Sysmon with FIM rules on critical paths.",
        },
    },
    "CLEAR_LOGS": {
        "tactic": "Defense Evasion", "technique": "Indicator Removal: Clear Linux Logs", "id": "T1070",
        "description": "System and security logs wiped to erase evidence of intrusion.",
        "weakness": {
            "host": "target",
            "finding": "Logs stored locally only. Attacker wiped evidence without triggering alert.",
            "recommendation": "Forward logs to a remote SIEM (Splunk/ELK) in real time.",
        },
    },
    "LATERAL_MOVE_SMB": {
        "tactic": "Lateral Movement", "technique": "Remote Services: SMB/Windows Admin Shares", "id": "T1021",
        "description": "SMB used to propagate from compromised host to Domain Controller.",
        "weakness": {
            "host": "dc01",
            "finding": "SMB signing not enforced. No segmentation between user and DC subnet.",
            "recommendation": "Enable SMB signing (RequireSecuritySignature) and segment DC traffic.",
        },
    },
    "PRIV_ESC_SUDO": {
        "tactic": "Privilege Escalation", "technique": "Abuse Elevation Control: Sudo", "id": "T1068",
        "description": "Misconfigured sudo rules exploited to obtain root/SYSTEM privileges.",
        "weakness": {
            "host": "dc01",
            "finding": "Sudoers file grants excessive permissions without requiring a password.",
            "recommendation": "Audit /etc/sudoers. Apply principle of least privilege. Enable PAM logging.",
        },
    },
    "EXFILTRATE_DATA": {
        "tactic": "Exfiltration", "technique": "Exfiltration Over C2 Channel", "id": "T1041",
        "description": "Sensitive data exfiltrated over the established command-and-control channel.",
        "weakness": {
            "host": "dc01",
            "finding": "No DLP policy. Outbound traffic from DC not inspected or rate-limited.",
            "recommendation": "Deploy DLP and inspect/block unexpected outbound DC connections.",
        },
    },
    "RANSOMWARE_ENCRYPT": {
        "tactic": "Impact", "technique": "Data Encrypted for Impact", "id": "T1486",
        "description": "Files on compromised hosts encrypted to deny availability (ransomware).",
        "weakness": {
            "host": "dc01",
            "finding": "No endpoint backup or immutable snapshot policy in place.",
            "recommendation": "Enable VSS shadow copies and immutable offsite backups.",
        },
    },
}

# ─── REMEDIATION DATABASE ────────────────────────────────────────────────────
# Priority: CRITICAL (root/data loss) → HIGH (lateral/credential) → MEDIUM (initial/recon)
# Each entry has: priority, host, impact, steps[], effort, countermeasure

REMEDIATION_DB = {
    "PHISHING_EMAIL": {
        "priority": "MEDIUM",
        "host": "user01 / email gateway",
        "impact": "Initial foothold established — attacker gained entry via end-user deception.",
        "steps": [
            "1. Deploy sandboxed email filtering (Microsoft Defender for O365 / Proofpoint).",
            "2. Enable link-time URL detonation — rewrite and detonate all clicked links.",
            "3. Configure DMARC / DKIM / SPF to block spoofed sender domains.",
            "4. Run mandatory phishing-awareness training quarterly with simulated campaigns.",
            "5. Block macro execution in Office documents arriving via email.",
        ],
        "effort": "Low-Medium (1-3 days to configure gateway policies)",
        "countermeasure": "D3FEND: Email Filtering (D3-EF), User Training (D3-UT)",
    },
    "BRUTE_FORCE_SSH": {
        "priority": "MEDIUM",
        "host": "target (SSH service)",
        "impact": "SSH exposed with no lockout — brute-force succeeded in obtaining credentials.",
        "steps": [
            "1. Install and configure fail2ban: ban IPs after 5 failed SSH attempts.",
            "2. Enforce account lockout policy (PAM: pam_tally2 / pam_faillock).",
            "3. Disable password-based SSH authentication — require key pairs only.",
            "4. Move SSH to a non-standard port and restrict via firewall to known IPs.",
            "5. Enable SSH login alerts via auditd or SIEM ingestion of /var/log/auth.log.",
        ],
        "effort": "Low (2-4 hours)",
        "countermeasure": "D3FEND: Account Locking (D3-AL), Network Traffic Filtering (D3-NTF)",
    },
    "NETWORK_SCAN": {
        "priority": "MEDIUM",
        "host": "network (internal segments)",
        "impact": "Internal topology exposed — attacker mapped srv01 and dc01 from initial foothold.",
        "steps": [
            "1. Deploy a Network Detection & Response (NDR) sensor (Zeek / Darktrace / ExtraHop).",
            "2. Implement micro-segmentation: isolate user VLAN from server and DC subnets.",
            "3. Block ICMP sweeps and port-scan patterns at internal firewall/ACLs.",
            "4. Alert on rapid sequential connection attempts (IDS rule: ET SCAN threshold).",
            "5. Enforce a zero-trust posture — require explicit allow-list per host pair.",
        ],
        "effort": "Medium (1-2 weeks for full segmentation)",
        "countermeasure": "D3FEND: Network Segmentation (D3-NS), Network Traffic Analysis (D3-NTA)",
    },
    "VALID_ACCOUNTS_LOGIN": {
        "priority": "HIGH",
        "host": "srv01 / AD domain",
        "impact": "Stolen credentials accepted without challenge — attacker moved as legitimate user.",
        "steps": [
            "1. IMMEDIATELY enforce MFA on all domain accounts (Microsoft Authenticator / Duo).",
            "2. Enable Conditional Access policies: block logins from unexpected IPs/devices.",
            "3. Deploy Azure AD Identity Protection or on-prem AD anomalous-login detection.",
            "4. Rotate ALL credentials from hosts confirmed compromised in this episode.",
            "5. Audit service accounts — remove interactive login rights and enforce PAM vaulting.",
            "6. Enable Windows Event 4624/4625/4768 forwarding to SIEM for login anomalies.",
        ],
        "effort": "Medium (MFA rollout: 1-5 days depending on user count)",
        "countermeasure": "D3FEND: Multi-factor Authentication (D3-MFA), Credential Hardening (D3-CH)",
    },
    "INSTALL_BACKDOOR": {
        "priority": "HIGH",
        "host": "target (compromised host)",
        "impact": "Persistent backdoor installed — attacker survives reboots and retains re-entry.",
        "steps": [
            "1. Deploy File Integrity Monitoring (FIM): auditd / Sysmon / Wazuh on all hosts.",
            "2. Alert on writes to: /etc/systemd/system/, /etc/cron.*, /usr/local/bin/.",
            "3. Implement application allowlisting (AppArmor / SELinux / Carbon Black).",
            "4. Run a full endpoint forensic scan — use chkrootkit / rkhunter to find implants.",
            "5. Re-image confirmed compromised hosts before returning to production.",
        ],
        "effort": "Medium (FIM deployment: 1-2 days; re-imaging: immediate)",
        "countermeasure": "D3FEND: File Integrity Monitoring (D3-FIM), System Call Analysis (D3-SCA)",
    },
    "CLEAR_LOGS": {
        "priority": "MEDIUM",
        "host": "target (local logging)",
        "impact": "Local logs wiped — forensic evidence destroyed, investigation severely impaired.",
        "steps": [
            "1. Forward all logs in real time to a remote SIEM (Splunk / Elastic / Graylog).",
            "2. Make log destination write-once / append-only from the endpoint's perspective.",
            "3. Alert on: auditd log truncation, journalctl wipe, /var/log/ mass-delete events.",
            "4. Restrict log directory permissions — only root + syslog service should write.",
            "5. Enable immutable log flags: chattr +a /var/log/auth.log.",
        ],
        "effort": "Low (SIEM forwarding: 1 day; permissions: 1 hour)",
        "countermeasure": "D3FEND: Remote Logging (D3-RL), Log Analysis (D3-LA)",
    },
    "LATERAL_MOVE_SMB": {
        "priority": "HIGH",
        "host": "dc01 / internal network",
        "impact": "Attacker pivoted to Domain Controller via SMB — highest-value asset now reachable.",
        "steps": [
            "1. Enable SMB signing on ALL hosts: Set-SmbServerConfiguration -RequireSecuritySignature $true.",
            "2. Block SMB (445/tcp) between workstation subnets and DC subnet at firewall.",
            "3. Disable SMBv1 entirely: Set-SmbServerConfiguration -EnableSMB1Protocol $false.",
            "4. Deploy LAPS (Local Administrator Password Solution) to prevent pass-the-hash.",
            "5. Alert on DC admin-share access from non-admin workstations (Event 5140).",
            "6. Apply Microsoft AD Tiering — workstations must not directly access DC admin shares.",
        ],
        "effort": "Medium (SMB signing: 2 hours; network segmentation: 1-2 weeks)",
        "countermeasure": "D3FEND: Network Segmentation (D3-NS), Credential Hardening (D3-CH)",
    },
    "PRIV_ESC_SUDO": {
        "priority": "CRITICAL",
        "host": "dc01 (root obtained)",
        "impact": "Root/SYSTEM privileges obtained — attacker has FULL control of the Domain Controller.",
        "steps": [
            "1. IMMEDIATELY audit /etc/sudoers and /etc/sudoers.d/: remove NOPASSWD entries.",
            "2. Apply principle of least privilege — no account should have unrestricted sudo.",
            "3. Enable PAM sudo logging: sudo logs → /var/log/sudo.log + forward to SIEM.",
            "4. Use 'sudo -l' auditing — alert on any account granted new sudo privileges.",
            "5. Consider replacing sudo with a PAM-vaulted privilege escalation tool (CyberArk / BeyondTrust).",
            "6. Deploy auditd rule: -a always,exit -F arch=b64 -S execve -F euid=0 to catch all root executions.",
        ],
        "effort": "Low-immediate (sudoers audit: 30 mins; PAM config: 1-2 hours)",
        "countermeasure": "D3FEND: Privilege Restriction (D3-PR), Execution Isolation (D3-EI)",
    },
    "EXFILTRATE_DATA": {
        "priority": "CRITICAL",
        "host": "dc01 / perimeter",
        "impact": "Sensitive data left the network — regulatory breach notification may be required.",
        "steps": [
            "1. Deploy Data Loss Prevention (DLP): inspect and block unexpected outbound DC traffic.",
            "2. Restrict outbound internet access from DC to an explicit allowlist of destinations.",
            "3. Enable full TLS inspection at the perimeter — C2 over HTTPS will be visible.",
            "4. Alert on large outbound transfers from DC (e.g. > 10 MB to non-corporate IPs).",
            "5. Encrypt sensitive data at rest — exfiltrated files become useless without keys.",
            "6. Review and classify all data stored on dc01 — apply need-to-know access controls.",
        ],
        "effort": "Medium-High (DLP deployment: 1-2 weeks; data classification: ongoing)",
        "countermeasure": "D3FEND: Data Loss Prevention (D3-DLP), Outbound Traffic Filtering (D3-OTF)",
    },
    "RANSOMWARE_ENCRYPT": {
        "priority": "CRITICAL",
        "host": "dc01 / all connected hosts",
        "impact": "Files encrypted for ransom — potential full business disruption, data unrecoverable without keys.",
        "steps": [
            "1. IMMEDIATELY isolate all affected hosts from the network.",
            "2. Enable VSS shadow copies on all hosts: vssadmin create shadow /for=C:\\ (scheduled daily).",
            "3. Set up immutable offsite backups (AWS S3 Object Lock / Azure Immutable Blob).",
            "4. Deploy EDR with behavioural ransomware detection (CrowdStrike / SentinelOne).",
            "5. Restrict write permissions on shared drives — users should only write to their own folders.",
            "6. Test backup restoration quarterly — confirm you can recover without paying ransom.",
        ],
        "effort": "High (backup infrastructure: 1 week; EDR: 1-3 days)",
        "countermeasure": "D3FEND: Backup Strategy (D3-BS), Decoy File (D3-DF), Process Termination (D3-PT)",
    },
}

_PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}

KILL_CHAIN_ORDER = [
    "PHISHING_EMAIL", "BRUTE_FORCE_SSH", "NETWORK_SCAN",
    "VALID_ACCOUNTS_LOGIN", "INSTALL_BACKDOOR", "CLEAR_LOGS",
    "LATERAL_MOVE_SMB", "PRIV_ESC_SUDO", "EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT",
]


# ─── HELPERS ───────────────────────────────────────────────────────────────

def _verdict(events):
    total_reward = sum(e.get("reward", 0) for e in events)
    last_state   = events[-1].get("state", {}) if events else {}
    dc           = last_state.get("dc01", {})
    detected     = any(e.get("detection_triggered") for e in events)
    # True mission success = exfil or ransomware ACTUALLY completed (reward > 400)
    mission_done = any(
        e.get("action") in ("EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT")
        and e.get("reward", 0) > 400
        for e in events
    )
    if mission_done:
        return "SUCCESS", total_reward, detected
    if dc.get("privileged"):
        return "PARTIAL", total_reward, detected   # Root obtained but exfil not completed
    if detected:
        return "DETECTED", total_reward, detected
    return "FAILED", total_reward, detected


def _risk_score(events, detected):
    dc = events[-1].get("state", {}).get("dc01", {}) if events else {}
    score = 0
    if dc.get("compromised"):  score += 30
    if dc.get("privileged"):   score += 40
    if dc.get("has_credentials"): score += 15
    if not detected:           score += 15
    return min(score, 100)


def _stealth_score(events):
    """Score based on actual peak alert accumulated."""
    from env.attack_actions import ACTION_MAP
    alert = 0.0
    peak  = 0.0
    for e in events:
        if e.get("detection_triggered"):
            return 0
        name = e.get("action", "")
        if name == "CLEAR_LOGS":
            alert = 0.0
        elif name in ACTION_MAP:
            alert = max(0.0, alert + ACTION_MAP[name].noise_level)
        peak = max(peak, alert)
    if peak <= 0.5:  return 95
    if peak <= 1.0:  return 85
    if peak <= 1.5:  return 70
    if peak <= 2.0:  return 50
    if peak <= 2.5:  return 30
    return 10


def _alert_progression(events):
    """Reconstruct alert level per step from noise metadata."""
    from env.attack_actions import ACTION_MAP
    alert = 0.0
    prog  = [0.0]
    for e in events:
        name = e.get("action", "")
        if name == "CLEAR_LOGS":
            alert = 0.0
        elif name in ACTION_MAP:
            alert = max(0.0, alert + ACTION_MAP[name].noise_level)
        prog.append(round(alert, 2))
    return prog


# Precondition descriptions — explains WHY an out-of-order action failed
_PRECONDITIONS = {
    "VALID_ACCOUNTS_LOGIN": "Requires PHISHING_EMAIL to succeed first (no credentials available yet)",
    "NETWORK_SCAN":         "Requires initial foothold via PHISHING_EMAIL",
    "LATERAL_MOVE_SMB":     "Requires valid credentials + network scan to locate DC",
    "PRIV_ESC_SUDO":        "Requires lateral movement to DC (dc01 not yet reached)",
    "EXFILTRATE_DATA":      "Requires root/admin privileges on dc01",
    "RANSOMWARE_ENCRYPT":   "Requires root/admin privileges on dc01",
    "INSTALL_BACKDOOR":     "Requires initial compromise of a host",
    "CLEAR_LOGS":           "No elevated noise — clearing logs is unnecessary at this stage",
}


def _action_result(reward, action, attempt_num, alert_at_step, noise_added):
    """Context-aware outcome using exact reward bands.

    reward <= -49      : hard lockdown (-50)
    -49 < reward <= -24: probabilistic detection (-25 from step())
    -23 < reward <= -15: wrong-phase heavy penalty (-20 from logic phase)
    -14 < reward < -1  : precondition miss (-2 to -10 extra penalty)
    -0.15 <= reward <= -0.05 : probabilistic retry (base -0.1 only, no extra)
    reward > 0         : success
    """
    noise_str   = f"+{noise_added:.2f} alert" if noise_added > 0 else "no noise"
    attempt_tag = f"Attempt #{attempt_num}" if attempt_num > 1 else ""

    if reward <= -49:
        return f"BLOCKED — Hard lockdown at alert {alert_at_step:.2f}  ->  episode terminated"
    if reward <= -24:
        return f"DETECTED — Probabilistic blue-team catch at alert {alert_at_step:.2f}  ->  episode terminated"
    if reward <= -15:
        pre = _PRECONDITIONS.get(action, "wrong phase for this action")
        tag = f"  [{attempt_tag}]" if attempt_tag else ""
        return f"WRONG PHASE{tag} -- {pre}"
    if reward < -1.0:
        pre = _PRECONDITIONS.get(action, "condition not satisfied")
        tag = f"  [{attempt_tag}]" if attempt_tag else ""
        return f"x  MISSED{tag} -- {pre}"

    # Base-only penalty (-0.1): action ran but probabilistic failure (no extra penalty)
    if -0.15 <= reward <= -0.05:
        if attempt_num > 1:
            return f"x  FAILED  {attempt_tag}  ({noise_str})  -- retrying (probabilistic miss)"
        return f"x  FAILED -- probabilistic miss  ({noise_str})  -- retrying"

    # Successes
    if action == "PHISHING_EMAIL":
        return f"HIT -- Phishing payload accepted by user01  ({noise_str})"
    if action == "NETWORK_SCAN":
        return f"HIT -- Internal topology mapped: srv01, dc01 discovered  ({noise_str})"
    if action == "VALID_ACCOUNTS_LOGIN":
        return f"HIT -- Credentials accepted; authenticated as domain user  ({noise_str})"
    if action == "LATERAL_MOVE_SMB":
        hop = "dc01" if reward > 45 else "srv01"
        return f"HIT -- Moved to {hop} via SMB admin share  ({noise_str})"
    if action == "PRIV_ESC_SUDO" and reward > 50:
        tag = f"  after {attempt_num} attempts" if attempt_num > 1 else ""
        return f"HIT -- Root obtained on dc01 via sudo misconfiguration{tag}  ({noise_str})"
    if action == "PRIV_ESC_SUDO":
        return f"HIT -- Privileges escalated on intermediate host  ({noise_str})"
    if action in ("EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT") and reward > 400:
        label = "exfiltrated" if action == "EXFILTRATE_DATA" else "encrypted"
        return f"MISSION COMPLETE -- Data {label}"
    if action == "CLEAR_LOGS":
        return "HIT -- Security logs wiped; alert meter reset to 0.00"
    if action == "INSTALL_BACKDOOR":
        return f"HIT -- Persistent backdoor installed  ({noise_str})"
    return f"HIT -- Action succeeded  ({noise_str})"


# ─── CORE REPORT FUNCTION ──────────────────────────────────────────────────

def generate_report(episode_file: Path, out_file: Path = None):
    with open(episode_file, "r", encoding="utf-8") as f:
        events = json.load(f)

    if isinstance(events, dict):
        events = events.get("events", events.get("episode_events", [events]))

    if not events:
        print("[!] Empty telemetry file."); return

    episode_id  = events[0].get("episode_id", episode_file.stem)
    ts          = datetime.fromtimestamp(events[0].get("timestamp", 0))
    verdict, total_reward, detected = _verdict(events)
    risk        = _risk_score(events, detected)
    stealth     = _stealth_score(events)
    alert_prog  = _alert_progression(events)
    actions_used = [e.get("action") for e in events]
    unique_actions = list(dict.fromkeys(actions_used))

    lines = []
    W = 68

    def sep(char="━"): lines.append(char * W)
    def hdr(txt):
        lines.append("")
        lines.append(txt)
        sep()

    # Build retry counter and noise per step upfront (needed by VERDICT section below)
    from env.attack_actions import ACTION_MAP
    _attempt_counts = {}   # action -> current attempt number
    _retry_stats    = {}   # action -> {attempts, successes}
    _step_noise     = []   # noise added per step
    _alert_at_step  = []   # alert level AFTER each step
    _running_alert  = 0.0
    for ev in events:
        a = ev.get("action", "")
        r = ev.get("reward", 0.0)
        _attempt_counts[a] = _attempt_counts.get(a, 0) + 1
        if a not in _retry_stats:
            _retry_stats[a] = {"attempts": 0, "successes": 0}
        _retry_stats[a]["attempts"] += 1
        if r > 0:
            _retry_stats[a]["successes"] += 1
        noise = 0.0
        if a == "CLEAR_LOGS":
            _running_alert = 0.0
        elif a in ACTION_MAP:
            noise = ACTION_MAP[a].noise_level
            _running_alert = max(0.0, _running_alert + noise)
        _step_noise.append(round(noise, 3))
        _alert_at_step.append(round(_running_alert, 2))

    # ── HEADER ────────────────────────────────────────────────────────────
    lines.append("╔" + "═" * (W - 2) + "╗")
    lines.append("║" + "  AI ADVERSARY EMULATION — ATTACK ANALYSIS REPORT  ".center(W - 2) + "║")
    lines.append("║" + f"  Episode : {episode_id}".ljust(W - 2) + "║")
    lines.append("║" + f"  Date    : {ts.strftime('%Y-%m-%d  %H:%M:%S')}".ljust(W - 2) + "║")
    lines.append("╚" + "═" * (W - 2) + "╝")

    # ── VERDICT ───────────────────────────────────────────────────────────
    icon = {"SUCCESS": "[SUCCESS]", "PARTIAL": "[PARTIAL]", "DETECTED": "[DETECTED]", "FAILED": "[FAILED]"}[verdict]
    lines.append("")
    lines.append(f"  VERDICT      :  {icon}")
    if verdict == "PARTIAL":
        exfil_att = _retry_stats.get("EXFILTRATE_DATA", {}).get("attempts", 0)
        att_str   = f" after {exfil_att} attempt{'s' if exfil_att != 1 else ''}" if exfil_att > 0 else ""
        lines.append(f"  NOTE         :  Root access obtained via sudo — exfiltration failed{att_str}, mission incomplete.")
    lines.append(f"  Risk Score   :  {risk}/100  {'X' * (risk // 10)}{'-' * (10 - risk // 10)}")
    lines.append(f"  Stealth Score:  {stealth}/100  {'X' * (stealth // 10)}{'-' * (10 - stealth // 10)}")
    lines.append(f"  Steps        :  {len(events)}")
    lines.append(f"  Total Reward :  {total_reward:.1f} pts")

    # ── ATTACK TIMELINE ───────────────────────────────────────────────────
    hdr("  ATTACK TIMELINE")
    lines.append(f"  {'Step':<5} {'Action':<25} {'Reward':>7}   Outcome")
    lines.append("  " + "-" * (W - 2))

    _seen_counts = {}
    for i, ev in enumerate(events):
        step   = ev.get("step", "?")
        action = ev.get("action", "?")
        reward = ev.get("reward", 0.0)
        _seen_counts[action] = _seen_counts.get(action, 0) + 1
        attempt_num = _seen_counts[action]
        noise       = _step_noise[i]
        alert_val   = _alert_at_step[i]
        outcome     = _action_result(reward, action, attempt_num, alert_val, noise)
        reward_str  = f"{reward:+.1f}"
        lines.append(f"  [{step:<3}] {action:<25} {reward_str:>7}   {outcome}")

    # ── RETRY ANALYSIS ────────────────────────────────────────────────────
    multi_attempt = {a: s for a, s in _retry_stats.items() if s["attempts"] > 1}
    if multi_attempt:
        hdr("  RETRY ANALYSIS  (persistence behaviour)")
        lines.append(f"  {'Action':<25}  {'Attempts':>8}  {'Successes':>9}  {'Fail%':>6}  Note")
        lines.append("  " + "-" * (W - 2))
        for a, s in sorted(multi_attempt.items(), key=lambda x: -x[1]["attempts"]):
            att  = s["attempts"]
            suc  = s["successes"]
            fail = (att - suc) / att * 100
            noise_total = round(ACTION_MAP[a].noise_level * att, 2) if a in ACTION_MAP else 0
            note = ""
            if att >= 5:
                note = f"⚠ HIGH RETRIES — generated {noise_total} total alert"
            elif att >= 3:
                note = f"Retried {att}×, added {noise_total} noise"
            lines.append(f"  {a:<25}  {att:>8}  {suc:>9}  {fail:>5.0f}%  {note}")
        # Flag the highest-risk retrier
        worst = max(multi_attempt, key=lambda a: multi_attempt[a]["attempts"])
        ws = multi_attempt[worst]
        if ws["attempts"] >= 5:
            noise_total = round(ACTION_MAP[worst].noise_level * ws["attempts"], 2) if worst in ACTION_MAP else 0
            lines.append("")
            lines.append(f"  ⚠ ROOT CAUSE OF ELEVATED ALERT: {worst}")
            lines.append(f"     Attempted {ws['attempts']}× with only {ws['successes']} success(es).")
            lines.append(f"     Each failed attempt added +{ACTION_MAP[worst].noise_level if worst in ACTION_MAP else '?'} noise")
            lines.append(f"     → Total noise contribution: {noise_total}  (primary detection risk)")

    # ── KILL CHAIN COVERAGE ───────────────────────────────────────────────
    hdr("  KILL CHAIN COVERAGE  (MITRE ATT&CK)")
    tactic_order = [
        "Initial Access", "Discovery", "Credential Access",
        "Persistence", "Defense Evasion", "Lateral Movement",
        "Privilege Escalation", "Exfiltration", "Impact"
    ]
    # Only mark EXECUTED if the action had at least one successful step (reward > 0)
    successful_actions = {e.get("action") for e in events if e.get("reward", 0) > 0}
    seen_tactics = {ATTACK_META[a]["tactic"] for a in successful_actions if a in ATTACK_META}
    attempted_tactics = {ATTACK_META[a]["tactic"] for a in unique_actions if a in ATTACK_META}

    for tactic in tactic_order:
        if tactic in seen_tactics:
            status = "[OK] SUCCEEDED "
        elif tactic in attempted_tactics:
            status = "[~] ATTEMPTED  "   # tried but all attempts failed
        else:
            status = "[ ] NOT USED   "
        acts = [a for a in unique_actions if ATTACK_META.get(a, {}).get("tactic") == tactic]
        tech = ", ".join(f"{a} ({ATTACK_META[a]['id']})" for a in acts) if acts else "--"
        lines.append(f"  {status}  {tactic:<25}  {tech}")


    # ── WEAK POINTS ANALYSIS ──────────────────────────────────────────────
    hdr("  WEAK POINTS EXPLOITED")
    reported_hosts = set()
    for action in unique_actions:
        meta = ATTACK_META.get(action)
        if not meta: continue
        w = meta.get("weakness", {})
        host = w.get("host", "unknown")
        if host in reported_hosts: continue
        if action not in actions_used: continue
        reward_sum = sum(e["reward"] for e in events if e.get("action") == action)
        if reward_sum <= 0: continue   # only report if action actually succeeded
        reported_hosts.add(host)
        lines.append(f"")
        lines.append(f"  HOST  :  {host.upper()}")
        lines.append(f"  ├─ Exploited via  : {meta['technique']} ({meta['id']})")
        lines.append(f"  ├─ Finding        : {w.get('finding','—')}")
        lines.append(f"  └─ Recommendation : {w.get('recommendation','—')}")

    # ── ALERT LEVEL PROGRESSION ───────────────────────────────────────────
    DETECT_THRESHOLD = 1.0
    hdr("  ALERT LEVEL PROGRESSION")
    lines.append("  Step  0 : 0.00  (baseline)")
    lines.append(f"  ─── Detection threshold: {DETECT_THRESHOLD:.2f} — ⚠ marks steps at or above this level ───")
    _threshold_crossed = False
    for i, ev in enumerate(events):
        al = alert_prog[i + 1] if i + 1 < len(alert_prog) else "—"
        marker = " ⚠" if isinstance(al, float) and al >= DETECT_THRESHOLD else ""
        lines.append(f"  Step {ev.get('step','?'):>2} : {str(al):<5}  [{ev.get('action','?')}]{marker}")

    # ── DETECTION STATUS ──────────────────────────────────────────────────
    hdr("  DETECTION & BLUE TEAM RESPONSE")
    peak_alert = max(alert_prog)
    if detected:
        det_steps = [e.get("step") for e in events if e.get("detection_triggered")]
        det_action = events[det_steps[0]-1].get("action", "unknown") if det_steps else "unknown"
        stats = _retry_stats.get(det_action, {})
        lines.append(f"  🚨 DETECTION TRIGGERED")
        lines.append(f"     ├─ At step           : {det_steps}")
        lines.append(f"     ├─ Triggering action : {det_action}")
        if stats.get("attempts", 0) > 1:
            lines.append(f"     ├─ Repeated {stats['attempts']}× with only {stats['successes']} success")
            lines.append(f"        → Each failure added noise until alert crossed threshold")
        lines.append(f"     ├─ Peak alert level  : {peak_alert:.2f} / 3.00")
        lines.append(f"     └─ Blue team response: LOCKDOWN initiated — all connections blocked")
    else:
        lines.append(f"  ✅ UNDETECTED — Agent operated below blue team detection threshold.")
        lines.append(f"     ├─ Peak alert level  : {peak_alert:.2f} / 3.00")
        lines.append(f"     ├─ Detection threshold: 1.00")
        lines.append(f"     └─ No SIEM alerts, no lockdown events triggered.")

    # ── TACTICAL SUMMARY ──────────────────────────────────────────────────
    hdr("  TACTICAL SUMMARY")
    lines.append(f"  The agent executed a {len(events)}-step kill chain targeting the domain")
    lines.append(f"  controller (dc01). Entry was established via {actions_used[0].replace('_',' ').title()},")
    if "LATERAL_MOVE_SMB" in unique_actions:
        lines.append(f"  followed by SMB-based lateral movement to the DC subnet.")
    # Only claim root access if PRIV_ESC actually succeeded
    priv_stats  = _retry_stats.get("PRIV_ESC_SUDO", {})
    exfil_stats = _retry_stats.get("EXFILTRATE_DATA", {})
    if verdict == "PARTIAL":
        # Single merged line — covers both root success and exfil failure
        exfil_att = exfil_stats.get("attempts", 0)
        att_str   = f" after {exfil_att} attempt{'s' if exfil_att != 1 else ''}" if exfil_att > 0 else ""
        lines.append(f"  Root access obtained via sudo — exfiltration failed{att_str}, mission incomplete.")
    elif priv_stats.get("successes", 0) > 0:
        lines.append(f"  Root access was obtained through sudo privilege escalation.")
    elif "PRIV_ESC_SUDO" in unique_actions:
        lines.append(f"  Privilege escalation attempted but ALL {priv_stats.get('attempts',0)} attempts failed -- dc01 NOT rooted.")
    if verdict == "SUCCESS":
        win_action = next((a for a in reversed(actions_used)
                          if a in ("EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT")
                          and any(e.get("action") == a and e.get("reward", 0) > 400 for e in events)), None)
        if win_action:
            lines.append(f"  Mission completed via {win_action.replace('_',' ').title()}.")

    lines.append("")
    lines.append("  Key observations:")
    lines.append(f"    - {len(unique_actions)} distinct attack techniques used across {len(events)} steps")
    n_succeeded = len(seen_tactics)
    n_attempted = len(attempted_tactics)
    lines.append(f"    - {n_succeeded} MITRE ATT&CK tactics succeeded / {n_attempted} attempted")
    if detected:
        culprit = max(_retry_stats, key=lambda a: _retry_stats[a]["attempts"]) if _retry_stats else "unknown"
        lines.append(f"    - DETECTED -- {culprit} retried too many times, noise accumulated above threshold")
    elif "CLEAR_LOGS" in unique_actions:
        lines.append("    - Agent evaded detection by wiping security logs mid-operation")
    else:
        lines.append("    - Agent maintained stealth -- stayed below noise floor, no logs wiped")

    # ── REMEDIATION ROADMAP ───────────────────────────────────────────────
    hdr("  REMEDIATION ROADMAP  (based on this episode's successful attacks)")
    lines.append("  Priority legend:  [!!!] CRITICAL   [!! ] HIGH   [!  ] MEDIUM")
    lines.append("")

    # Collect successful actions + any attempted-but-not-succeeded (add at lower priority)
    remediation_items = []
    for action in unique_actions:
        if action not in REMEDIATION_DB:
            continue
        succeeded = action in successful_actions
        attempted = action in unique_actions
        if succeeded:
            remediation_items.append((action, REMEDIATION_DB[action]))

    # Sort: CRITICAL first, then HIGH, then MEDIUM
    remediation_items.sort(key=lambda x: _PRIORITY_ORDER.get(x[1]["priority"], 99))

    if not remediation_items:
        lines.append("  No successful exploits detected — no immediate remediations required.")
    else:
        priority_icons = {"CRITICAL": "[!!!]", "HIGH": "[!! ]", "MEDIUM": "[!  ]"}
        priority_labels = {"CRITICAL": "CRITICAL", "HIGH": "HIGH   ", "MEDIUM": "MEDIUM "}
        prev_priority = None
        item_num = 0
        for action, rem in remediation_items:
            p = rem["priority"]
            if p != prev_priority:
                if prev_priority is not None:
                    lines.append("")
                lines.append(f"  {'─' * 64}")
                lines.append(f"  {priority_icons[p]} {priority_labels[p]}  ──────────────────────────────────────────")
                lines.append(f"  {'─' * 64}")
                prev_priority = p
            item_num += 1
            meta = ATTACK_META.get(action, {})
            lines.append("")
            lines.append(f"  [{item_num}] {action.replace('_', ' ')}  ({meta.get('id','—')})  |  Host: {rem['host']}")
            lines.append(f"      Impact  : {rem['impact']}")
            lines.append(f"      Effort  : {rem['effort']}")
            lines.append(f"      Defence : {rem['countermeasure']}")
            lines.append(f"      Actions :")
            for step in rem["steps"]:
                lines.append(f"        {step}")

        lines.append("")
        lines.append(f"  {'─' * 64}")
        n_crit = sum(1 for _, r in remediation_items if r["priority"] == "CRITICAL")
        n_high = sum(1 for _, r in remediation_items if r["priority"] == "HIGH")
        n_med  = sum(1 for _, r in remediation_items if r["priority"] == "MEDIUM")
        lines.append(f"  TOTAL: {len(remediation_items)} gaps found  |  "
                     f"Critical: {n_crit}  High: {n_high}  Medium: {n_med}")

    lines.append("")
    sep("═")
    lines.append("  Generated by: AI Adversary Emulation Platform  |  MITRE ATT&CK® aligned")
    sep("═")

    report = "\n".join(lines)

    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        existed = out_file.exists()
        out_file.write_text(report, encoding="utf-8")   # always overwrites
        tag = "[~] Updated" if existed else "[+] Saved "
        print(f"{tag} → {out_file}")
    else:
        print(report)

    return report


# ─── BATCH MODE ────────────────────────────────────────────────────────────

def summarise_all(telemetry_dir: Path):
    files = sorted(telemetry_dir.glob("episode_*.json"))
    if not files:
        print(f"[!] No episode files in {telemetry_dir}"); return

    verdicts = {"SUCCESS": 0, "DETECTED": 0, "FAILED": 0}
    risks, stealths, steps_list = [], [], []

    for f in files:
        try:
            events = json.load(open(f, encoding="utf-8"))
            if isinstance(events, dict):
                events = events.get("events", [events])
            if not events: continue
            v, _, det = _verdict(events)
            verdicts[v] += 1
            risks.append(_risk_score(events, det))
            stealths.append(_stealth_score(events))
            steps_list.append(len(events))
        except Exception:
            continue

    total = sum(verdicts.values())
    print("\n" + "═" * 60)
    print("  CAMPAIGN SUMMARY — ALL EPISODES")
    print("═" * 60)
    print(f"  Episodes analysed : {total}")
    print(f"  Success Rate      : {verdicts['SUCCESS']/total*100:.1f}%  ({verdicts['SUCCESS']} eps)")
    print(f"  Detection Rate    : {verdicts['DETECTED']/total*100:.1f}%  ({verdicts['DETECTED']} eps)")
    print(f"  Fail Rate         : {verdicts['FAILED']/total*100:.1f}%  ({verdicts['FAILED']} eps)")
    if risks:
        print(f"  Avg Risk Score    : {sum(risks)/len(risks):.0f}/100")
        print(f"  Avg Stealth Score : {sum(stealths)/len(stealths):.0f}/100")
        print(f"  Avg Steps/Episode : {sum(steps_list)/len(steps_list):.1f}")
    print("═" * 60 + "\n")

    # Generate individual reports
    out_dir = telemetry_dir.parent / "reports"
    for f in files:
        try:
            events = json.load(open(f, encoding="utf-8"))
            if isinstance(events, dict):
                events = events.get("events", [events])
            generate_report(f, out_dir / f"{f.stem}_report.txt")
        except Exception as e:
            print(f"[!] Skipped {f.name}: {e}")


# ─── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Adversary Attack Report Generator")
    parser.add_argument("--episode", type=str, default=None,
                        help="Path to a specific episode JSON file")
    parser.add_argument("--dir", type=str, default="results/telemetry/ppo_eval",
                        help="Telemetry directory (default: results/telemetry/ppo_eval)")
    parser.add_argument("--all",  action="store_true",
                        help="Summarise all episodes and generate individual reports")
    parser.add_argument("--out",  type=str, default=None,
                        help="Save report to file instead of printing")
    args = parser.parse_args()

    if args.all:
        summarise_all(Path(args.dir))
        return

    if args.episode:
        ep_file = Path(args.episode)
    else:
        # Pick the most recent episode in the directory
        files = sorted(Path(args.dir).glob("episode_*.json"))
        if not files:
            print(f"[!] No episode files found in {args.dir}")
            print("    Run ppo_eval.py first to generate telemetry.")
            return
        ep_file = files[-1]
        print(f"[*] Using latest episode: {ep_file.name}\n")

    out = Path(args.out) if args.out else None
    generate_report(ep_file, out)


if __name__ == "__main__":
    main()
