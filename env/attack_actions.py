from enum import Enum, auto


class AttackAction:
    def __init__(self, name, success_prob, noise_level=0,
                 mitre_tactic="", mitre_technique=""):
        self.name = name
        self.success_prob = success_prob
        self.noise_level = noise_level  # How much it raises the Alert Meter
        self.mitre_tactic = mitre_tactic
        self.mitre_technique = mitre_technique


# THE 10-STEP ADVANCED KILL CHAIN (MITRE ATT&CK Aligned)
ACTION_LIST = [
    # --- 1. INITIAL ACCESS ---
    AttackAction("PHISHING_EMAIL",       0.90, noise_level=0.05,
                 mitre_tactic="Initial Access",       mitre_technique="T1566"),
    AttackAction("BRUTE_FORCE_SSH",      0.60, noise_level=0.5,   # still loud
                 mitre_tactic="Credential Access",    mitre_technique="T1110"),

    # --- 2. DISCOVERY ---
    AttackAction("NETWORK_SCAN",         0.95, noise_level=0.1,
                 mitre_tactic="Discovery",            mitre_technique="T1046"),

    # --- 3. CREDENTIAL ACCESS ---
    AttackAction("VALID_ACCOUNTS_LOGIN", 0.95, noise_level=0.05,
                 mitre_tactic="Defense Evasion",      mitre_technique="T1078"),

    # --- 4. PERSISTENCE ---
    AttackAction("INSTALL_BACKDOOR",     0.80, noise_level=0.15,
                 mitre_tactic="Persistence",          mitre_technique="T1543"),

    # --- 5. DEFENSE EVASION ---
    AttackAction("CLEAR_LOGS",           1.00, noise_level=-1.0,
                 mitre_tactic="Defense Evasion",      mitre_technique="T1070"),

    # --- 6. LATERAL MOVEMENT ---
    AttackAction("LATERAL_MOVE_SMB",     0.85, noise_level=0.15,
                 mitre_tactic="Lateral Movement",     mitre_technique="T1021"),

    # --- 7. PRIVILEGE ESCALATION ---
    AttackAction("PRIV_ESC_SUDO",        0.70, noise_level=0.3,
                 mitre_tactic="Privilege Escalation", mitre_technique="T1068"),

    # --- 8. IMPACT — Win Type A: Exfiltration ---
    AttackAction("EXFILTRATE_DATA",      1.00, noise_level=0.25,
                 mitre_tactic="Exfiltration",         mitre_technique="T1041"),

    # --- 9. IMPACT — Win Type B: Ransomware ---
    AttackAction("RANSOMWARE_ENCRYPT",   0.90, noise_level=0.35,
                 mitre_tactic="Impact",               mitre_technique="T1486"),

    # --- 10. INITIAL ACCESS — SQL Injection ---
    AttackAction("SQL_INJECTION",        0.85, noise_level=0.12,
                 mitre_tactic="Initial Access",       mitre_technique="T1190"),

    # --- 11. LATERAL MOVEMENT — Pass the Hash ---
    # Use stolen credential hashes to authenticate without cracking.
    # Prerequisite: credentials already stolen. Quieter than brute force.
    AttackAction("PASS_THE_HASH",        0.80, noise_level=0.20,
                 mitre_tactic="Lateral Movement",     mitre_technique="T1550.002"),

    # --- 12. EXECUTION — PowerShell Remote Execution ---
    # Execute commands remotely via PowerShell on a compromised host.
    # Enables privilege escalation or C2 establishment.
    AttackAction("POWERSHELL_EXEC",      0.85, noise_level=0.25,
                 mitre_tactic="Execution",            mitre_technique="T1059.001"),

    # --- 13. CREDENTIAL ACCESS — Kerberoasting ---
    # Request Kerberos service tickets and crack offline.
    # Quiet (no brute force noise), targets Active Directory / DC.
    AttackAction("KERBEROASTING",        0.75, noise_level=0.15,
                 mitre_tactic="Credential Access",   mitre_technique="T1558.003"),

    # --- 14. PERSISTENCE — Web Shell Upload ---
    # Upload persistent web shell after SQLi compromise.
    # Combines backdoor + scanning capability in one step.
    AttackAction("WEB_SHELL_UPLOAD",     0.80, noise_level=0.20,
                 mitre_tactic="Persistence",          mitre_technique="T1505.003"),
]

# Convenience lookup by name
ACTION_MAP = {a.name: a for a in ACTION_LIST}
