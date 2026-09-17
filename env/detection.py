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
from typing import Dict, List
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

    # ── v5 additions ──────────────────────────────────────────────────────
    # Event identifiers below are the real ones a defender would key on, in the
    # channel each actually lands in, so the emitted corpus stays usable as
    # detection-engineering material rather than as abstract alert levels.

    "EXTERNAL_REMOTE_SERVICES": DetectionProfile(
        emits=[(4624, "Security", "Successful logon, type 10 (RemoteInteractive)", True)],
        base_detect=0.18,
        rule="RDP/VPN logon from an external address outside business hours (T1133)"),

    "REMOTE_SYSTEM_DISCOVERY": DetectionProfile(
        emits=[(4688, "Security", "Process Create: net view / nltest", True)],
        base_detect=0.12,
        rule="Host enumeration utilities executed from a user context (T1018)"),

    "DOMAIN_GROUP_DISCOVERY": DetectionProfile(
        emits=[(4688, "Security", "Process Create: net group /domain", True),
               (4661, "Security", "A handle to an object was requested", False)],
        base_detect=0.10,
        rule="Domain group enumeration from a non-administrative workstation (T1069.002)"),

    "SYSTEM_INFO_DISCOVERY": DetectionProfile(
        emits=[(4688, "Security", "Process Create: systeminfo", False)],
        base_detect=0.05,
        rule="Host fingerprinting; individually benign, notable in sequence (T1082)"),

    "CREDS_IN_FILES": DetectionProfile(
        emits=[(4663, "Security", "Attempt to access an object (credential file)", True),
               (11, "Sysmon", "FileCreate: copy of a configuration file", False)],
        base_detect=0.15,
        rule="Access to files matching credential patterns (unattend.xml, .kdbx) (T1552.001)"),

    "CREDENTIALS_FROM_BROWSER": DetectionProfile(
        emits=[(10, "Sysmon", "ProcessAccess: handle opened on a browser process", True)],
        base_detect=0.22,
        rule="Non-browser process reading browser credential stores (T1555.003)"),

    "PROCESS_INJECTION": DetectionProfile(
        emits=[(8, "Sysmon", "CreateRemoteThread", True),
               (10, "Sysmon", "ProcessAccess with PROCESS_VM_WRITE", True)],
        base_detect=0.38,
        rule="Remote thread creation into another process address space (T1055)"),

    "RDP_HIJACK": DetectionProfile(
        emits=[(4624, "Security", "Successful logon, type 10 (RemoteInteractive)", True),
               (1149, "TerminalServices", "Remote Desktop user authentication succeeded", True)],
        base_detect=0.24,
        rule="Internal RDP between hosts with no prior session history (T1021.001)"),

    "SCHEDULED_TASK": DetectionProfile(
        emits=[(4698, "Security", "A scheduled task was created", True)],
        base_detect=0.28,
        rule="Scheduled task registered outside a change window (T1053.005)"),

    "REGISTRY_RUN_KEYS": DetectionProfile(
        emits=[(13, "Sysmon", "RegistryEvent: value set under a Run key", True)],
        base_detect=0.26,
        rule="Autorun registry key written by a non-installer process (T1547.001)"),

    "DISABLE_SECURITY_TOOLS": DetectionProfile(
        emits=[(4719, "Security", "System audit policy was changed", True),
               (7036, "System", "Service entered the stopped state (Defender)", True)],
        base_detect=0.60,
        rule="Security service stopped or audit policy altered; loud by design (T1562.001)"),

    "MODIFY_REGISTRY": DetectionProfile(
        emits=[(4657, "Security", "A registry value was modified", True),
               (13, "Sysmon", "RegistryEvent: value set", False)],
        base_detect=0.20,
        rule="Registry modification affecting logging or execution policy (T1112)"),

    "DATA_FROM_LOCAL_SYSTEM": DetectionProfile(
        emits=[(4663, "Security", "Attempt to access an object (document store)", True),
               (5145, "Security", "Network share object checked for access", False)],
        base_detect=0.16,
        rule="Bulk read of document stores by a single process (T1005)"),

    "ARCHIVE_COLLECTED_DATA": DetectionProfile(
        emits=[(11, "Sysmon", "FileCreate: archive written to a staging path", True),
               (4688, "Security", "Process Create: archiver utility", True)],
        base_detect=0.25,
        rule="Archive created in a staging directory before egress (T1560.001)"),

    "INHIBIT_SYSTEM_RECOVERY": DetectionProfile(
        emits=[(4688, "Security", "Process Create: vssadmin delete shadows", True),
               (524, "System", "The backup was deleted", True)],
        base_detect=0.68,
        rule="Shadow copy or backup deletion; near-certain ransomware precursor (T1490)"),

    "SERVICE_STOP": DetectionProfile(
        emits=[(7036, "System", "Service entered the stopped state", True),
               (4688, "Security", "Process Create: net stop", True)],
        base_detect=0.42,
        rule="Business-critical service stopped by an interactive process (T1489)"),

    "C2_CHANNEL_ESTABLISH": DetectionProfile(
        emits=[(3, "Sysmon", "Network connection to an uncategorised external host", True),
               (5156, "Security", "Windows Filtering Platform permitted a connection", False)],
        base_detect=0.22,
        rule="Beaconing: periodic outbound HTTPS to a low-reputation destination (T1071.001)"),

    "WMI_EXECUTION": DetectionProfile(
        emits=[(4688, "Security", "Process Create: child of WmiPrvSE.exe", True),
               (5861, "WMI-Activity", "Permanent WMI event consumer registered", True)],
        base_detect=0.30,
        rule="Process spawned by WmiPrvSE on a remote host (T1047)"),

    "NTLM_RELAY": DetectionProfile(
        emits=[(4624, "Security", "Network logon type 3 with NTLM package", True),
               (4776, "Security", "Domain controller attempted credential validation", True)],
        base_detect=0.34,
        rule="NTLM authentication relayed between hosts; mismatched source workstation (T1557.001)"),

    "OBFUSCATED_FILES": DetectionProfile(
        emits=[(4104, "PowerShell", "Script block logging: encoded command", True),
               (1, "Sysmon", "Process Create with base64 command line", True)],
        base_detect=0.28,
        rule="Base64 or otherwise encoded command line executed (T1027)"),
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


# A refused or failed attempt is evidence, and repeated ones are much stronger
# evidence. Without this an agent can hammer one denied action for the whole
# episode at the cost of the step charge alone, which is both an exploit and
# wrong about security: Event 4625 repeated is the brute-force signature, and
# a denied handle-open against LSASS is louder than one that succeeds, not
# quieter. The first failure adds FAILED_ATTEMPT_BASE; each repetition of the
# same (technique, host) adds another increment, capped so that a policy that
# retries a few times is not treated as certainly caught.
# Measured against random legal play: with the charge removed entirely, 90.0% of
# episodes still end in an incident and the mean episode is 20.3 steps; with it,
# 90.3% and 19.6. The charge closes the repetition exploit without changing how
# hard the environment is to explore, which is the property that had to hold.
FAILED_ATTEMPT_BASE = 0.02
FAILED_ATTEMPT_REPEAT = 0.07
FAILED_ATTEMPT_CAP = 8


class DetectionEngine:
    """Accumulates SOC suspicion and emits ATT&CK-mapped events per attempt."""

    def __init__(self):
        self.suspicion = 0.0
        self.log: List[SecurityEvent] = []
        self.fired_rules: List[str] = []
        # Failed attempts per (technique, host). Real correlation rules key on
        # repetition, not on a single record: account lockout, brute-force and
        # impossible-travel all fire on N failures within a window. Counting
        # them is what makes a refused attempt cost something.
        self.failures: Dict[tuple, int] = {}

    def reset(self):
        self.suspicion = 0.0
        self.log = []
        self.fired_rules = []
        self.failures = {}

    @staticmethod
    def _channel_for(host_os):
        return "Security" if host_os == "windows" else "auditd"

    # Channels that exist only on Windows. On a Linux host the same technique
    # is still observable, but through a different source, so an event on the
    # wrong platform is translated rather than silently emitted. Without this a
    # released corpus would contain Windows Security 4624 records for Linux
    # hosts, which no defender could reconcile with their own telemetry.
    _WINDOWS_ONLY = ("Security", "System", "Sysmon", "PowerShell",
                     "TerminalServices", "WMI-Activity")
    # Linux equivalents carry real auditd record types, not the Windows numeric
    # identifier. An auditd record has no event ID 4688; keeping the Windows
    # number while relabelling the channel produces a record that looks plausible
    # and is wrong, which is worse for a released corpus than an obvious gap.
    # auditd types: 1100 USER_AUTH, 1112 USER_LOGIN, 1300 SYSCALL, 1302 PATH,
    # 1305 CONFIG_CHANGE. Text logs (auth.log, syslog) have no numeric type, so
    # they carry 0.
    _LINUX_EQUIV = {
        4624: (1112, "auditd",   "USER_LOGIN: authentication accepted"),
        4625: (1100, "auditd",   "USER_AUTH: authentication failure"),
        4688: (1300, "auditd",   "SYSCALL: execve"),
        4663: (1302, "auditd",   "PATH: open on a watched file"),
        4657: (1305, "auditd",   "CONFIG_CHANGE: watched configuration written"),
        4719: (1305, "auditd",   "CONFIG_CHANGE: audit rules altered"),
        1102: (1305, "auditd",   "CONFIG_CHANGE: audit log truncated"),
        4698: (1300, "auditd",   "SYSCALL: cron/systemd-timer unit written"),
        7036: (0,    "syslog",   "systemd unit entered a stopped state"),
        7045: (0,    "syslog",   "systemd unit file installed"),
        4104: (0,    "syslog",   "shell invoked with an encoded command"),
        1149: (1112, "auditd",   "USER_LOGIN: remote session established"),
        4776: (1100, "auditd",   "USER_AUTH: credential validation"),
        4661: (1302, "auditd",   "PATH: object handle requested"),
        5140: (1302, "auditd",   "PATH: network share accessed"),
        5145: (1302, "auditd",   "PATH: share object checked"),
        5156: (0,    "syslog",   "connection permitted by the host firewall"),
        524:  (0,    "syslog",   "backup snapshot removed"),
        5861: (0,    "syslog",   "management agent registered a subscription"),
    }

    def _translate(self, eid, channel, name, host_os):
        """Map a Windows event onto its Linux-equivalent source where needed."""
        if host_os == "windows" or channel not in self._WINDOWS_ONLY:
            return eid, channel, name
        if eid in self._LINUX_EQUIV:
            return self._LINUX_EQUIV[eid]
        # Sysmon has no Linux counterpart in this model; fall back to a generic
        # syslog line rather than inventing an identifier.
        return 0, "syslog", name

    def observe(self, technique: str, host: str, succeeded: bool,
                has_backdoor: bool, rng: random.Random,
                host_os: str = "windows",
                denied: bool = False) -> DetectionResult:
        """Run the blue team against one technique attempt.

        `has_backdoor` lowers detection slightly: an established foothold lets
        the operator route through trusted channels. A *failed* attempt is often
        louder for the noisy techniques (failed logons are the brute-force
        signature), so failure does not reduce detection.
        """
        prof = PROFILES[technique]
        res = DetectionResult()

        for eid, channel, name, suspicious in prof.emits:
            eid, channel, name = self._translate(eid, channel, name, host_os)
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

        if not succeeded:
            # The attempt did not do what it was meant to -- it was refused by
            # the host (denied) or it simply missed. Either way the host logged
            # an error, and a rule that correlates repeated errors from one
            # source is the most ordinary detection there is. Repetition is what
            # is scored, so a single retry is cheap and a hundred are not.
            key = (technique, host)
            n = self.failures.get(key, 0)
            self.failures[key] = n + 1
            add = FAILED_ATTEMPT_BASE + FAILED_ATTEMPT_REPEAT * min(n, FAILED_ATTEMPT_CAP)
            res.suspicion_added += add
            if denied:
                # every refusal is logged, not only the first: the repetition is
                # the signal, and a corpus that recorded one attempt out of six
                # would misrepresent what the defender actually sees
                res.events.append(SecurityEvent(
                    4673, self._channel_for(host_os), "Privileged service called: "
                    "operation refused", host, _mitre(technique), technique, True))
                self.log.append(res.events[-1])

        self.suspicion += res.suspicion_added
        res.caught = self.suspicion >= INCIDENT_THRESHOLD
        return res


# Additional technique -> ATT&CK id mappings, registered by newer catalogues at
# import time. Kept as a hook rather than an import so that env.kill_chain_v5
# (which imports this module) does not create a cycle.
_EXTRA_MITRE = {}


def register_mitre(mapping):
    """Let another catalogue declare its ATT&CK ids to the detection engine."""
    _EXTRA_MITRE.update(mapping)


def _mitre(technique):
    # imported lazily to avoid a cycle with kill_chain
    from env.kill_chain import TECHNIQUES
    if technique in TECHNIQUES:
        return TECHNIQUES[technique].mitre_id
    return _EXTRA_MITRE.get(technique, "unknown")
