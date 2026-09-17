# -*- coding: utf-8 -*-
"""
The detection model — a rule-based blue team, shared by the simulator and the
server so both detect identically.

WHY THIS REPLACES `alert * 0.2`
-------------------------------
v3 decided detection with one line: block with probability proportional to an
abstract "alert level". That is not defensible in a thesis, because the number
has no referent. A reviewer asks "detection by what?" and there is no answer.

Here detection is what it is in a real SOC: techniques leave observable events in
the logs (Windows Security / Sysmon event IDs), and detection rules fire on
those events. Some techniques are intrinsically loud and are caught often
(clearing the audit log raises Event 1102, which is a five-alarm signal);
some are quiet by design (logging in with valid stolen accounts looks like a
normal user). The agent that learns to prefer the quiet path is learning real
tradecraft, and the detection rate it is measured against is explainable rule by
rule.

WHAT THIS PRODUCES
------------------
Two things, from one pass:
  * a stream of ATT&CK-mapped security events -- the labelled telemetry the whole
    platform exists to generate, the artefact a defender would load into a SIEM;
  * a rising SOC suspicion score; when a detection rule fires with enough
    confidence, or suspicion crosses an incident threshold, the blue team
    responds and the intrusion is caught.

This module is pure and deterministic given its RNG, so the simulator and the
server get identical behaviour by both importing it.
"""
from dataclasses import dataclass, field
from typing import List
import random


# ── security events ────────────────────────────────────────────────────────
# The Windows Security / Sysmon events each technique leaves behind. Event IDs
# are the real ones, so the emitted telemetry reads like a genuine log.
@dataclass
class SecurityEvent:
    event_id: int          # Windows/Sysmon event ID
    channel: str           # which log it lands in
    name: str
    host: str
    mitre_id: str
    technique: str
    suspicious: bool       # would a tuned rule consider this event notable?


# For each technique: the events it emits, and a detection rule.
#   emits:        list of (event_id, channel, name, suspicious)
#   base_detect:  probability the SOC rule fires when the technique is attempted,
#                 before modifiers. This is the technique's intrinsic loudness.
#   rule:         the human-readable detection rule, for the write-up.
@dataclass
class DetectionProfile:
    emits: list
    base_detect: float
    rule: str


# Detection probabilities are grounded in how observable each technique is.
# Loud, high-signal techniques (log clear, ransomware, brute force) are near the
# top; living-off-the-land and valid-credential techniques are near the bottom.
PROFILES = {
    "PHISHING_EMAIL": DetectionProfile(
        emits=[(4688, "Security", "Process Create: outlook child process", True)],
        base_detect=0.05,
        rule="Suspicious child process of a mail client (T1566)"),

    "BRUTE_FORCE_SSH": DetectionProfile(
        emits=[(4625, "Security", "Failed logon", True),
               (4625, "Security", "Failed logon", True),
               (4624, "Security", "Successful logon after failures", True)],
        base_detect=0.45,
        rule="Multiple 4625 failed logons then a 4624 success within a window (T1110)"),

    "NETWORK_SCAN": DetectionProfile(
        emits=[(5156, "Security", "Windows Filtering Platform connection", True)],
        base_detect=0.20,
        rule="High rate of distinct connection attempts from one host (T1046)"),

    "VALID_ACCOUNTS_LOGIN": DetectionProfile(
        emits=[(4624, "Security", "Successful logon (valid account)", False)],
        base_detect=0.03,
        rule="Valid-account logon looks legitimate; only anomalous hours flag (T1078)"),

    "INSTALL_BACKDOOR": DetectionProfile(
        emits=[(7045, "System", "A new service was installed", True)],
        base_detect=0.30,
        rule="New service / persistence mechanism installed (T1543)"),

    "CLEAR_LOGS": DetectionProfile(
        emits=[(1102, "Security", "The audit log was cleared", True)],
        base_detect=0.50,
        rule="Event ID 1102 audit-log-cleared is itself the alarm (T1070)"),

    "LATERAL_MOVE_SMB": DetectionProfile(
        emits=[(4624, "Security", "Network logon type 3", True),
               (5140, "Security", "Network share accessed (ADMIN$)", True)],
        base_detect=0.25,
        rule="Admin-share access with a network logon between workstations (T1021)"),

    "PRIV_ESC_SUDO": DetectionProfile(
        emits=[(4688, "Security", "Process Create: privilege escalation", True)],
        base_detect=0.30,
        rule="Exploitation-driven privilege escalation process (T1068)"),

    "EXFILTRATE_DATA": DetectionProfile(
        emits=[(5156, "Security", "Large outbound transfer over C2 channel", True)],
        base_detect=0.35,
        rule="Anomalous outbound data volume to an external host (T1041)"),

    "RANSOMWARE_ENCRYPT": DetectionProfile(
        emits=[(4688, "Security", "Process Create: mass file modification", True)],
        base_detect=0.70,
        rule="High-rate file modification consistent with encryption (T1486)"),

    "SQL_INJECTION": DetectionProfile(
        emits=[(1116, "WAF", "Web request matched injection signature", True)],
        base_detect=0.30,
        rule="WAF signature match on a union/boolean injection pattern (T1190)"),

    "PASS_THE_HASH": DetectionProfile(
        emits=[(4624, "Security", "Network logon type 3 with NTLM", True)],
        base_detect=0.20,
        rule="NTLM network logon anomaly consistent with pass-the-hash (T1550.002)"),

    "POWERSHELL_EXEC": DetectionProfile(
        emits=[(4104, "PowerShell", "Script block logging: remote execution", True)],
        base_detect=0.30,
        rule="Suspicious PowerShell script block / remote execution (T1059.001)"),

    "KERBEROASTING": DetectionProfile(
        emits=[(4769, "Security", "Kerberos service ticket (RC4) requested", True)],
        base_detect=0.15,
        rule="Anomalous 4769 TGS requests with RC4 encryption (T1558.003)"),

    "WEB_SHELL_UPLOAD": DetectionProfile(
        emits=[(4688, "Security", "Process Create: web server spawned shell", True)],
        base_detect=0.25,
        rule="Web server process spawning a command shell (T1505.003)"),

    # ── extended techniques (v4) ──────────────────────────────────────────
    "CRED_DUMP_LSASS": DetectionProfile(
        emits=[(10, "Sysmon", "ProcessAccess: handle to lsass.exe", True)],
        base_detect=0.45,
        rule="Suspicious process opening a handle to LSASS memory (T1003.001)"),

    "DOMAIN_ACCT_DISCOVERY": DetectionProfile(
        emits=[(4661, "Security", "A handle to a directory object was requested", True)],
        base_detect=0.08,
        rule="LDAP/directory enumeration of domain accounts, low-signal (T1087.002)"),

    "DCSYNC": DetectionProfile(
        emits=[(4662, "Security", "Directory replication (DS-Replication-Get-Changes)", True)],
        base_detect=0.35,
        rule="Replication request from a non-DC host, consistent with DCSync (T1003.006)"),

    # ── Active Directory attack techniques (Event IDs per the supervisor's material) ──
    "DOMAIN_TRUST_DISCOVERY": DetectionProfile(
        emits=[(4661, "Security", "A handle to a directory object (trust) was requested", True)],
        base_detect=0.07,
        rule="BloodHound/PowerView-style enumeration of domain trusts, low-signal (T1482)"),

    "AS_REP_ROASTING": DetectionProfile(
        emits=[(4768, "Security", "Kerberos AS-REQ for an account without pre-authentication", True)],
        base_detect=0.22,
        rule="AS-REQ for a pre-auth-disabled account, returning a crackable hash (T1558.004)"),

    "PASSWORD_SPRAYING": DetectionProfile(
        emits=[(4625, "Security", "Failed logon", True),
               (4625, "Security", "Failed logon (different account)", True),
               (4624, "Security", "Successful logon after spray", True)],
        base_detect=0.30,
        rule="Many 4625 failures across distinct accounts from one source (T1110.003)"),

    "PASS_THE_TICKET": DetectionProfile(
        emits=[(4769, "Security", "Kerberos service ticket presented for reuse", True)],
        base_detect=0.20,
        rule="Ticket reuse / anomalous TGS presentation, consistent with pass-the-ticket (T1550.003)"),

    "ACCOUNT_MANIPULATION": DetectionProfile(
        emits=[(4728, "Security", "Member added to a security-enabled global group", True),
               (4732, "Security", "Member added to a security-enabled local group", True)],
        base_detect=0.35,
        rule="Account added to a privileged group (e.g. Backup Operators, Admins) (T1098)"),

    "GPO_MODIFICATION": DetectionProfile(
        emits=[(5136, "Security", "A directory service object (GPO) was modified", True)],
        base_detect=0.30,
        rule="Group Policy Object modified to gain SYSTEM or disable controls (T1484.001)"),

    "GOLDEN_TICKET": DetectionProfile(
        emits=[(4769, "Security", "Kerberos TGS with anomalous TGT lifetime / krbtgt use", True)],
        base_detect=0.40,
        rule="Forged TGT (Golden Ticket): anomalous ticket lifetime and encryption (T1558.001)"),
}

# Suspicion at or above this ends the intrusion: the SOC has escalated to
# incident response and locks the attacker out.
INCIDENT_THRESHOLD = 1.0


@dataclass
class DetectionResult:
    events: List[SecurityEvent] = field(default_factory=list)
    rule_fired: bool = False
    suspicion_added: float = 0.0
    caught: bool = False        # this attempt tripped a detection outright


class DetectionEngine:
    """Accumulates SOC suspicion and emits ATT&CK-mapped events per attempt."""

    def __init__(self):
        self.suspicion = 0.0
        self.log: List[SecurityEvent] = []
        self.fired_rules: List[str] = []

    def reset(self):
        self.suspicion = 0.0
        self.log = []
        self.fired_rules = []

    def observe(self, technique: str, host: str, succeeded: bool,
                has_backdoor: bool, rng: random.Random) -> DetectionResult:
        """Run the blue team against one technique attempt.

        `has_backdoor` lowers detection slightly: an established foothold lets
        the operator route through trusted channels. A *failed* attempt is often
        louder for the noisy techniques (failed logons are the brute-force
        signature), so failure does not reduce detection.
        """
        prof = PROFILES[technique]
        res = DetectionResult()

        for eid, channel, name, suspicious in prof.emits:
            ev = SecurityEvent(eid, channel, name, host,
                               _mitre(technique), technique, suspicious)
            res.events.append(ev)
            self.log.append(ev)

        p = prof.base_detect
        if has_backdoor:
            p *= 0.8                     # trusted channel, marginally quieter
        p = max(0.0, min(1.0, p))

        if rng.random() < p:
            res.rule_fired = True
            self.fired_rules.append(prof.rule)
            # a fired rule adds suspicion equal to the rule's own loudness, so a
            # quiet technique that trips is a small blip and a loud one is not.
            # Catching therefore requires several loud events or many quiet ones,
            # which is what accumulating SOC evidence looks like.
            res.suspicion_added = prof.base_detect
        else:
            # an un-fired notable event still leaves a faint correlatable trace
            res.suspicion_added = 0.03 if any(e[3] for e in prof.emits) else 0.0

        self.suspicion += res.suspicion_added
        res.caught = self.suspicion >= INCIDENT_THRESHOLD
        return res


def _mitre(technique):
    # imported lazily to avoid a cycle with kill_chain
    from env.kill_chain import TECHNIQUES
    return TECHNIQUES[technique].mitre_id
