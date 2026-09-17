# -*- coding: utf-8 -*-
"""
The ATT&CK Enterprise techniques used to pad the action space in the scaling
experiment, and the reason each one cannot apply in the modelled network.

WHY THIS FILE EXISTS
--------------------
The scaling experiment widens the action space from fifteen techniques to sixty
and then two hundred, to vary the share of the action space that is legal at any
given step. In the first version of that experiment the added actions were
anonymous: index 15 to index 199 with no identity. That invites the obvious
objection that the result is an artefact of invented actions.

The padding here is drawn from the real ATT&CK Enterprise catalogue instead, and
every entry carries the reason it is inapplicable to the environment under test.
The environment is three Windows hosts in a single Active Directory domain
(user01, srv01, dc01) with an optional web-application surface. It has no cloud
tenancy, no container runtime, no network appliances, no macOS or Linux hosts,
and no physical access. Techniques that require any of those cannot be executed
here as a property of the network, not as a property of our construction.

The two hundred figure is likewise no longer arbitrary: ATT&CK Enterprise
carries on the order of two hundred top-level techniques, so the widest
condition tested corresponds to the size of the real catalogue.

WHAT IS AND IS NOT CLAIMED
--------------------------
These techniques are catalogued, not instantiated. Naming them does not mean the
environment implements them, and no result in the paper depends on their
semantics -- they are never legal, and selecting one costs the same small
penalty and alert increase regardless of which one it is. The naming exists so
the action space is a defensible sample of ATT&CK rather than an invented set.

Two tiers are distinguished, because they support different strengths of claim:

  STRUCTURAL   the required asset does not exist in the network. Impossible here
               for any agent, under any policy. This is a property of the target.
  UNMODELLED   the technique is Windows-applicable in principle, but this
               environment does not model it. Inapplicable because of the
               abstraction level, not because of the target.

PROVENANCE
----------
Every identifier and name below is verified against MITRE's official STIX bundle
for **ATT&CK Enterprise v19.2**. Re-run

    python analysis/validate_catalogue.py

after any change here, and again before submission if ATT&CK has published a new
version. The check is not cosmetic: v19 revoked the whole T1562 "Impair
Defenses" sub-tree in favour of T1685 and T1686, and merged DLL Side-Loading
into T1574.001, so a list curated against an earlier version carries dead
identifiers.

One caveat the validator cannot see. Two implemented actions carry Unix-flavoured
labels -- BRUTE_FORCE_SSH and PRIV_ESC_SUDO -- although the modelled hosts are a
Windows domain. Both map to platform-neutral ATT&CK techniques (T1110 Brute
Force, T1068 Exploitation for Privilege Escalation), so the mapping is sound, but
the labels sit awkwardly beside the claim that this network has no Linux or macOS
hosts. Renaming them would invalidate the trained checkpoints and the telemetry
already collected, so they are left alone and the point is stated in the write-up
instead.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.attack_actions import ACTION_LIST

STRUCTURAL = "structural"
UNMODELLED = "unmodelled"

# reason codes, and the sentence each one becomes in the paper
REASONS = {
    "no_cloud":     "no cloud tenancy, cloud storage or SaaS service in the network",
    "no_container": "no container runtime or orchestrator in the network",
    "no_netdev":    "no routers, switches or other managed network appliances",
    "no_macos":     "no macOS hosts",
    "no_linux":     "no Linux hosts",
    "unmodelled":   "Windows-applicable, but outside the abstraction of this environment",
}

# (technique id, name, tier, reason)
#
# Tier 1 -- STRUCTURAL. The asset the technique acts on does not exist here.
_STRUCTURAL = [
    # --- cloud, SaaS and identity-provider surfaces ------------------------
    ("T1078.004", "Valid Accounts: Cloud Accounts",                              "no_cloud"),
    ("T1087.004", "Account Discovery: Cloud Account",                            "no_cloud"),
    ("T1098.001", "Account Manipulation: Additional Cloud Credentials",          "no_cloud"),
    ("T1098.003", "Account Manipulation: Additional Cloud Roles",                "no_cloud"),
    ("T1069.003", "Permission Groups Discovery: Cloud Groups",                   "no_cloud"),
    ("T1136.003", "Create Account: Cloud Account",                               "no_cloud"),
    ("T1526",     "Cloud Service Discovery",                                     "no_cloud"),
    ("T1530",     "Data from Cloud Storage",                                     "no_cloud"),
    ("T1535",     "Unused/Unsupported Cloud Regions",                            "no_cloud"),
    ("T1537",     "Transfer Data to Cloud Account",                              "no_cloud"),
    ("T1538",     "Cloud Service Dashboard",                                     "no_cloud"),
    ("T1550.001", "Use Alternate Authentication Material: Application Access Token", "no_cloud"),
    ("T1552.005", "Unsecured Credentials: Cloud Instance Metadata API",          "no_cloud"),
    ("T1686.001", "Disable or Modify System Firewall: Cloud Firewall",           "no_cloud"),
    ("T1685.002", "Disable or Modify Tools: Disable or Modify Cloud Log",        "no_cloud"),
    ("T1578",     "Modify Cloud Compute Infrastructure",                         "no_cloud"),
    ("T1578.001", "Modify Cloud Compute Infrastructure: Create Snapshot",        "no_cloud"),
    ("T1578.002", "Modify Cloud Compute Infrastructure: Create Cloud Instance",  "no_cloud"),
    ("T1578.003", "Modify Cloud Compute Infrastructure: Delete Cloud Instance",  "no_cloud"),
    ("T1578.004", "Modify Cloud Compute Infrastructure: Revert Cloud Instance",  "no_cloud"),
    ("T1580",     "Cloud Infrastructure Discovery",                              "no_cloud"),
    ("T1619",     "Cloud Storage Object Discovery",                              "no_cloud"),
    ("T1651",     "Cloud Administration Command",                                "no_cloud"),

    # --- containers -------------------------------------------------------
    ("T1609",     "Container Administration Command",                            "no_container"),
    ("T1610",     "Deploy Container",                                            "no_container"),
    ("T1612",     "Build Image on Host",                                         "no_container"),
    ("T1613",     "Container and Resource Discovery",                            "no_container"),
    ("T1525",     "Implant Internal Image",                                      "no_container"),
    ("T1552.007", "Unsecured Credentials: Container API",                        "no_container"),
    ("T1053.007", "Scheduled Task/Job: Container Orchestration Job",             "no_container"),
    ("T1098.006", "Account Manipulation: Additional Container Cluster Roles",    "no_container"),

    # --- network appliances ------------------------------------------------
    ("T1059.008", "Command and Scripting Interpreter: Network Device CLI",       "no_netdev"),
    ("T1542.005", "Pre-OS Boot: TFTP Boot",                                      "no_netdev"),
    ("T1556.004", "Modify Authentication Process: Network Device Authentication", "no_netdev"),
    ("T1599",     "Network Boundary Bridging",                                   "no_netdev"),
    ("T1599.001", "Network Boundary Bridging: Network Address Translation Traversal", "no_netdev"),
    ("T1600",     "Weaken Encryption",                                           "no_netdev"),
    ("T1600.001", "Weaken Encryption: Reduce Key Space",                         "no_netdev"),
    ("T1600.002", "Weaken Encryption: Disable Crypto Hardware",                  "no_netdev"),
    ("T1601",     "Modify System Image",                                         "no_netdev"),
    ("T1601.001", "Modify System Image: Patch System Image",                     "no_netdev"),
    ("T1601.002", "Modify System Image: Downgrade System Image",                 "no_netdev"),
    ("T1602",     "Data from Configuration Repository",                          "no_netdev"),
    ("T1602.001", "Data from Configuration Repository: SNMP (MIB Dump)",         "no_netdev"),
    ("T1602.002", "Data from Configuration Repository: Network Device Configuration Dump", "no_netdev"),

    # --- macOS-only --------------------------------------------------------
    ("T1059.002", "Command and Scripting Interpreter: AppleScript",              "no_macos"),
    ("T1543.001", "Create or Modify System Process: Launch Agent",               "no_macos"),
    ("T1543.004", "Create or Modify System Process: Launch Daemon",              "no_macos"),
    ("T1546.006", "Event Triggered Execution: LC_LOAD_DYLIB Addition",           "no_macos"),
    ("T1546.014", "Event Triggered Execution: Emond",                            "no_macos"),
    ("T1547.007", "Boot or Logon Autostart Execution: Re-opened Applications",   "no_macos"),
    ("T1547.015", "Boot or Logon Autostart Execution: Login Items",              "no_macos"),
    ("T1548.004", "Abuse Elevation Control Mechanism: Elevated Execution with Prompt", "no_macos"),
    ("T1553.001", "Subvert Trust Controls: Gatekeeper Bypass",                   "no_macos"),
    ("T1555.001", "Credentials from Password Stores: Keychain",                  "no_macos"),
    ("T1574.004", "Hijack Execution Flow: Dylib Hijacking",                      "no_macos"),
    ("T1647",     "Plist File Modification",                                     "no_macos"),
    ("T1037.002", "Boot or Logon Initialization Scripts: Login Hook",            "no_macos"),

    # --- Linux-only --------------------------------------------------------
    ("T1059.004", "Command and Scripting Interpreter: Unix Shell",               "no_linux"),
    ("T1053.003", "Scheduled Task/Job: Cron",                                    "no_linux"),
    ("T1543.002", "Create or Modify System Process: Systemd Service",            "no_linux"),
    ("T1546.004", "Event Triggered Execution: Unix Shell Configuration Modification", "no_linux"),
    ("T1546.005", "Event Triggered Execution: Trap",                             "no_linux"),
    ("T1547.006", "Boot or Logon Autostart Execution: Kernel Modules and Extensions", "no_linux"),
    ("T1548.001", "Abuse Elevation Control Mechanism: Setuid and Setgid",        "no_linux"),
    ("T1556.003", "Modify Authentication Process: Pluggable Authentication Modules", "no_linux"),
    ("T1574.006", "Hijack Execution Flow: Dynamic Linker Hijacking",             "no_linux"),
    ("T1222.002", "File and Directory Permissions Modification: Linux and Mac Permissions", "no_linux"),
    ("T1037.004", "Boot or Logon Initialization Scripts: RC Scripts",            "no_linux"),
]

# Tier 2 -- UNMODELLED. Windows-applicable, but this environment abstracts the
# kill chain to fifteen steps and does not represent these.
_UNMODELLED = [
    # credential access
    ("T1003.001", "OS Credential Dumping: LSASS Memory"),
    ("T1003.002", "OS Credential Dumping: Security Account Manager"),
    ("T1003.003", "OS Credential Dumping: NTDS"),
    ("T1003.004", "OS Credential Dumping: LSA Secrets"),
    ("T1003.005", "OS Credential Dumping: Cached Domain Credentials"),
    ("T1003.006", "OS Credential Dumping: DCSync"),
    ("T1552.001", "Unsecured Credentials: Credentials In Files"),
    ("T1552.002", "Unsecured Credentials: Credentials in Registry"),
    ("T1552.004", "Unsecured Credentials: Private Keys"),
    ("T1555.003", "Credentials from Password Stores: Credentials from Web Browsers"),
    ("T1555.004", "Credentials from Password Stores: Windows Credential Manager"),
    ("T1558.001", "Steal or Forge Kerberos Tickets: Golden Ticket"),
    ("T1558.002", "Steal or Forge Kerberos Tickets: Silver Ticket"),
    ("T1550.003", "Use Alternate Authentication Material: Pass the Ticket"),
    ("T1187",     "Forced Authentication"),
    ("T1557.001", "Adversary-in-the-Middle: Name Resolution Poisoning and SMB Relay"),
    ("T1040",     "Network Sniffing"),
    ("T1056.001", "Input Capture: Keylogging"),
    ("T1056.002", "Input Capture: GUI Input Capture"),
    ("T1212",     "Exploitation for Credential Access"),

    # execution
    ("T1047",     "Windows Management Instrumentation"),
    ("T1059.003", "Command and Scripting Interpreter: Windows Command Shell"),
    ("T1059.005", "Command and Scripting Interpreter: Visual Basic"),
    ("T1059.006", "Command and Scripting Interpreter: Python"),
    ("T1059.007", "Command and Scripting Interpreter: JavaScript"),
    ("T1106",     "Native API"),
    ("T1129",     "Shared Modules"),
    ("T1203",     "Exploitation for Client Execution"),
    ("T1204.001", "User Execution: Malicious Link"),
    ("T1204.002", "User Execution: Malicious File"),
    ("T1569.002", "System Services: Service Execution"),
    ("T1053.005", "Scheduled Task/Job: Scheduled Task"),

    # persistence
    ("T1136.001", "Create Account: Local Account"),
    ("T1136.002", "Create Account: Domain Account"),
    ("T1197",     "BITS Jobs"),
    ("T1505.001", "Server Software Component: SQL Stored Procedures"),
    ("T1505.002", "Server Software Component: Transport Agent"),
    ("T1543.003", "Create or Modify System Process: Windows Service"),
    ("T1546.002", "Event Triggered Execution: Screensaver"),
    ("T1546.003", "Event Triggered Execution: Windows Management Instrumentation Event Subscription"),
    ("T1546.008", "Event Triggered Execution: Accessibility Features"),
    ("T1546.010", "Event Triggered Execution: AppInit DLLs"),
    ("T1546.012", "Event Triggered Execution: Image File Execution Options Injection"),
    ("T1547.001", "Boot or Logon Autostart Execution: Registry Run Keys / Startup Folder"),
    ("T1547.002", "Boot or Logon Autostart Execution: Authentication Package"),
    ("T1547.004", "Boot or Logon Autostart Execution: Winlogon Helper DLL"),
    ("T1547.005", "Boot or Logon Autostart Execution: Security Support Provider"),
    ("T1547.009", "Boot or Logon Autostart Execution: Shortcut Modification"),
    ("T1037.001", "Boot or Logon Initialization Scripts: Logon Script (Windows)"),
    ("T1574.001", "Hijack Execution Flow: DLL"),
    ("T1574.011", "Hijack Execution Flow: Services Registry Permissions Weakness"),
    ("T1133",     "External Remote Services"),

    # privilege escalation
    ("T1134",     "Access Token Manipulation"),
    ("T1134.001", "Access Token Manipulation: Token Impersonation/Theft"),
    ("T1134.002", "Access Token Manipulation: Create Process with Token"),
    ("T1548.002", "Abuse Elevation Control Mechanism: Bypass User Account Control"),
    ("T1055",     "Process Injection"),
    ("T1055.001", "Process Injection: Dynamic-link Library Injection"),
    ("T1055.002", "Process Injection: Portable Executable Injection"),
    ("T1055.003", "Process Injection: Thread Execution Hijacking"),
    ("T1055.012", "Process Injection: Process Hollowing"),
    ("T1484.001", "Domain or Tenant Policy Modification: Group Policy Modification"),
    ("T1207",     "Rogue Domain Controller"),

    # defence evasion
    ("T1027",     "Obfuscated Files or Information"),
    ("T1027.002", "Obfuscated Files or Information: Software Packing"),
    ("T1036",     "Masquerading"),
    ("T1036.005", "Masquerading: Match Legitimate Resource Name or Location"),
    ("T1070.004", "Indicator Removal: File Deletion"),
    ("T1070.006", "Indicator Removal: Timestomp"),
    ("T1112",     "Modify Registry"),
    ("T1140",     "Deobfuscate/Decode Files or Information"),
    ("T1202",     "Indirect Command Execution"),
    ("T1218.001", "System Binary Proxy Execution: Compiled HTML File"),
    ("T1218.005", "System Binary Proxy Execution: Mshta"),
    ("T1218.010", "System Binary Proxy Execution: Regsvr32"),
    ("T1218.011", "System Binary Proxy Execution: Rundll32"),
    ("T1220",     "XSL Script Processing"),
    ("T1497",     "Virtualization/Sandbox Evasion"),
    ("T1685",     "Disable or Modify Tools"),
    ("T1685.001", "Disable or Modify Tools: Disable or Modify Windows Event Log"),
    ("T1686",     "Disable or Modify System Firewall"),
    ("T1564.001", "Hide Artifacts: Hidden Files and Directories"),
    ("T1564.003", "Hide Artifacts: Hidden Window"),
    ("T1014",     "Rootkit"),
    ("T1553.002", "Subvert Trust Controls: Code Signing"),

    # discovery
    ("T1007",     "System Service Discovery"),
    ("T1012",     "Query Registry"),
    ("T1016",     "System Network Configuration Discovery"),
    ("T1018",     "Remote System Discovery"),
    ("T1033",     "System Owner/User Discovery"),
    ("T1049",     "System Network Connections Discovery"),
    ("T1057",     "Process Discovery"),
    ("T1082",     "System Information Discovery"),
    ("T1083",     "File and Directory Discovery"),
    ("T1087.001", "Account Discovery: Local Account"),
    ("T1087.002", "Account Discovery: Domain Account"),
    ("T1069.001", "Permission Groups Discovery: Local Groups"),
    ("T1069.002", "Permission Groups Discovery: Domain Groups"),
    ("T1120",     "Peripheral Device Discovery"),
    ("T1135",     "Network Share Discovery"),
    ("T1201",     "Password Policy Discovery"),
    ("T1482",     "Domain Trust Discovery"),
    ("T1518",     "Software Discovery"),
    ("T1518.001", "Software Discovery: Security Software Discovery"),
    ("T1614",     "System Location Discovery"),

    # lateral movement
    ("T1021.001", "Remote Services: Remote Desktop Protocol"),
    ("T1021.003", "Remote Services: Distributed Component Object Model"),
    ("T1021.004", "Remote Services: SSH"),
    ("T1021.006", "Remote Services: Windows Remote Management"),
    ("T1210",     "Exploitation of Remote Services"),
    ("T1534",     "Internal Spearphishing"),
    ("T1570",     "Lateral Tool Transfer"),
    ("T1080",     "Taint Shared Content"),

    # collection
    ("T1005",     "Data from Local System"),
    ("T1039",     "Data from Network Shared Drive"),
    ("T1074.001", "Data Staged: Local Data Staging"),
    ("T1074.002", "Data Staged: Remote Data Staging"),
    ("T1113",     "Screen Capture"),
    ("T1115",     "Clipboard Data"),
    ("T1119",     "Automated Collection"),
    ("T1123",     "Audio Capture"),
    ("T1125",     "Video Capture"),
    ("T1213",     "Data from Information Repositories"),
    ("T1560",     "Archive Collected Data"),
    ("T1560.001", "Archive Collected Data: Archive via Utility"),
    ("T1114.001", "Email Collection: Local Email Collection"),
    ("T1114.002", "Email Collection: Remote Email Collection"),

    # command and control
    ("T1001",     "Data Obfuscation"),
    ("T1008",     "Fallback Channels"),
    ("T1071.001", "Application Layer Protocol: Web Protocols"),
    ("T1071.004", "Application Layer Protocol: DNS"),
    ("T1090",     "Proxy"),
    ("T1090.003", "Proxy: Multi-hop Proxy"),
    ("T1095",     "Non-Application Layer Protocol"),
    ("T1102",     "Web Service"),
    ("T1104",     "Multi-Stage Channels"),
    ("T1105",     "Ingress Tool Transfer"),
    ("T1132",     "Data Encoding"),
    ("T1205",     "Traffic Signaling"),
    ("T1219",     "Remote Access Tools"),
    ("T1568",     "Dynamic Resolution"),
    ("T1571",     "Non-Standard Port"),
    ("T1573",     "Encrypted Channel"),
    ("T1573.001", "Encrypted Channel: Symmetric Cryptography"),
    ("T1573.002", "Encrypted Channel: Asymmetric Cryptography"),

    # exfiltration and impact
    ("T1020",     "Automated Exfiltration"),
    ("T1029",     "Scheduled Transfer"),
    ("T1030",     "Data Transfer Size Limits"),
    ("T1048",     "Exfiltration Over Alternative Protocol"),
    ("T1048.003", "Exfiltration Over Alternative Protocol: Exfiltration Over Unencrypted Non-C2 Protocol"),
    ("T1485",     "Data Destruction"),
    ("T1489",     "Service Stop"),
    ("T1490",     "Inhibit System Recovery"),
    ("T1491.001", "Defacement: Internal Defacement"),
    ("T1499",     "Endpoint Denial of Service"),
    ("T1529",     "System Shutdown/Reboot"),
    ("T1561.001", "Disk Wipe: Disk Content Wipe"),
    ("T1561.002", "Disk Wipe: Disk Structure Wipe"),
    ("T1565.001", "Data Manipulation: Stored Data Manipulation"),

    # initial access not modelled as a separate step
    ("T1189",     "Drive-by Compromise"),
    ("T1195.002", "Supply Chain Compromise: Compromise Software Supply Chain"),
    ("T1199",     "Trusted Relationship"),
    ("T1566.001", "Phishing: Spearphishing Attachment"),
    ("T1566.003", "Phishing: Spearphishing via Service"),

    # The following were curated as structural -- firmware, removable media,
    # physical access, cloud-hosted mail -- on the grounds that this network has
    # no such asset. ATT&CK v19.2 nonetheless lists Windows among their
    # platforms, so the stronger claim cannot be made from the published data
    # and they are demoted here. They remain inapplicable, but as a consequence
    # of the abstraction rather than of the target.
    ("T1114.003", "Email Collection: Email Forwarding Rule"),
    ("T1564.008", "Hide Artifacts: Email Hiding Rules"),
    ("T1567.002", "Exfiltration Over Web Service: Exfiltration to Cloud Storage"),
    ("T1567.004", "Exfiltration Over Web Service: Exfiltration Over Webhook"),
    ("T1611",     "Escape to Host"),
    ("T1552.003", "Unsecured Credentials: Shell History"),
    ("T1205.001", "Traffic Signaling: Port Knocking"),
    ("T1542.001", "Pre-OS Boot: System Firmware"),
    ("T1542.002", "Pre-OS Boot: Component Firmware"),
    ("T1542.003", "Pre-OS Boot: Bootkit"),
    ("T1495",     "Firmware Corruption"),
    ("T1200",     "Hardware Additions"),
    ("T1052.001", "Exfiltration Over Physical Medium: Exfiltration over USB"),
    ("T1091",     "Replication Through Removable Media"),
    ("T1025",     "Data from Removable Media"),
]


class CataloguedTechnique:
    """A technique that exists in ATT&CK but cannot be executed in this network."""

    __slots__ = ("id", "name", "tier", "reason")

    def __init__(self, tid, name, tier, reason):
        self.id = tid
        self.name = name
        self.tier = tier
        self.reason = reason

    @property
    def explanation(self):
        return REASONS[self.reason]

    def __repr__(self):
        return f"<{self.id} {self.name} ({self.tier})>"


# Structural techniques come first, so that the smaller conditions in the
# scaling experiment are padded with the strongest case -- at sixty actions the
# padding is entirely "impossible against this target", and only the two
# hundred condition reaches into the unmodelled tier.
INAPPLICABLE = (
    [CataloguedTechnique(t, n, STRUCTURAL, r) for t, n, r in _STRUCTURAL]
    + [CataloguedTechnique(t, n, UNMODELLED, "unmodelled") for t, n in _UNMODELLED]
)

# identifiers already implemented as real actions, which must never appear as
# padding -- an implemented technique offered twice would be a genuine error
IMPLEMENTED_IDS = {a.mitre_technique for a in ACTION_LIST}

_seen = set()
for _t in INAPPLICABLE:
    if _t.id in _seen:
        raise ValueError(f"duplicate padding technique {_t.id}")
    if _t.id in IMPLEMENTED_IDS:
        raise ValueError(f"padding technique {_t.id} is already implemented")
    _seen.add(_t.id)
del _seen, _t


def padding_for(n):
    """The first `n` inapplicable techniques, in catalogue order.

    Raises if the catalogue is too small, rather than silently falling back to
    anonymous actions -- a partly-named action space would be worse than either
    alternative, because the paper could no longer describe it in one sentence.
    """
    if n > len(INAPPLICABLE):
        raise ValueError(
            f"need {n} inapplicable techniques but the catalogue holds "
            f"{len(INAPPLICABLE)}; extend analysis/attack_catalogue.py"
        )
    return INAPPLICABLE[:n]


def composition(total_actions, real_n=len(ACTION_LIST)):
    """How an action space of `total_actions` breaks down, for reporting."""
    pad = padding_for(total_actions - real_n)
    structural = sum(1 for t in pad if t.tier == STRUCTURAL)
    return {
        "total": total_actions,
        "implemented": real_n,
        "structural": structural,
        "unmodelled": len(pad) - structural,
    }


if __name__ == "__main__":
    print(f"catalogue holds {len(INAPPLICABLE)} inapplicable techniques "
          f"({sum(1 for t in INAPPLICABLE if t.tier == STRUCTURAL)} structural, "
          f"{sum(1 for t in INAPPLICABLE if t.tier == UNMODELLED)} unmodelled)")
    for n in (15, 60, 200):
        print(f"  {n:>3} actions: {composition(n)}")
