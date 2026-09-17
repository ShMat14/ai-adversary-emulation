# -*- coding: utf-8 -*-
"""
The service layer — realistic, technique-specific artifacts the target returns.

The shared model (env.kill_chain) decides *whether* a technique succeeds and how
the network state changes; this module decides *what a real service would return*
when it does. That separation is deliberate: the model stays the single source of
truth for state (so the simulator and the server never diverge), while the server
gains the depth of a genuine multi-service network -- a mailbox with a lure, a web
app that reflects an injection and dumps a database, a Kerberos service that hands
back a crackable ticket, a domain controller that yields NTDS secrets.

Every artifact here is synthetic and self-contained. The credentials, hashes and
records are fabricated fixtures for an emulation range; nothing is a real secret.

Each function takes the session's model (post-transition) and returns a dict the
endpoint serialises. Artifacts are deterministic per session where it matters
(a stolen hash stays the same once dumped), so a manual operator can carry a
credential from one call to the next like a real intrusion.
"""
import hashlib


def _rng_hex(seed: str, n: int) -> str:
    """Deterministic hex string from a seed, for stable fake secrets per session."""
    h = hashlib.sha256(seed.encode()).hexdigest()
    while len(h) < n:
        h += hashlib.sha256(h.encode()).hexdigest()
    return h[:n]


# ── fabricated fixtures for the range ──────────────────────────────────────
DOMAIN = "NORTHWIND.LOCAL"
USERS = [
    {"user": "jsmith",   "role": "Sales",   "email": "jsmith@northwind.local"},
    {"user": "achen",    "role": "Finance", "email": "achen@northwind.local"},
    {"user": "svc_sql",  "role": "Service", "email": "svc_sql@northwind.local"},
    {"user": "administrator", "role": "Domain Admin", "email": "admin@northwind.local"},
]
SHARES = ["ADMIN$", "C$", "IPC$", "Finance", "IT-Backups"]


def mail_inbox(model, token):
    """user01 mailbox — the phishing lure the operator would click (T1566)."""
    return {
        "mailbox": "jsmith@northwind.local",
        "messages": [
            {"from": "it-support@northwind-it.com",
             "subject": "[Action Required] Mailbox quota exceeded",
             "has_link": True,
             "link": "http://northwind-it.com/reset?u=jsmith",
             "lure": "credential-harvest"},
            {"from": "payroll@northwind.local",
             "subject": "March payslip", "has_link": False},
        ],
        "note": "Clicking the flagged link triggers the phishing foothold.",
    }


def webapp_sqli(model, token, payload):
    """Web app search — reflects the payload; on a real injection, dumps creds (T1190)."""
    injected = any(p in (payload or "").lower()
                   for p in ["' or", "union select", "--", "1=1", "or '1'"])
    if not injected:
        return {"reflected": payload, "rows": [], "note": "query returned no rows"}
    rows = []
    for u in USERS:
        nthash = _rng_hex(f"{token}:{u['user']}", 32)
        rows.append({"username": u["user"], "role": u["role"],
                     "nt_hash": nthash})
    return {"reflected": payload, "vulnerable": True,
            "database": f"{DOMAIN} / webapp_users",
            "rows": rows,
            "note": "boolean/UNION injection succeeded — credential table dumped"}


def lsass_dump(model, token):
    """LSASS memory dump — plaintext-ish creds + NTLM hashes from the host (T1003.001)."""
    creds = []
    for u in USERS[:2]:
        creds.append({"username": u["user"], "domain": DOMAIN,
                      "ntlm": _rng_hex(f"{token}:lsass:{u['user']}", 32),
                      "logon": "Interactive"})
    return {"source": "lsass.exe", "sysmon_event": 10,
            "credentials": creds,
            "note": "credentials recovered from LSASS process memory"}


def smb_shares(model, token):
    """SMB share enumeration on the file server (T1021 recon)."""
    return {"host": "srv01", "shares": SHARES,
            "accessible": ["ADMIN$", "C$", "Finance"] if model.host("srv01").compromised
            else ["IPC$"],
            "note": "admin-share access indicates lateral movement capability"}


def kerberos_tgs(model, token):
    """Kerberos service ticket — a crackable RC4 roast hash (T1558.003)."""
    spn = "MSSQLSvc/srv01.northwind.local:1433"
    roast = "$krb5tgs$23$*svc_sql$NORTHWIND.LOCAL$" + spn + "*$" + \
            _rng_hex(f"{token}:krb", 32) + "$" + _rng_hex(f"{token}:krb2", 160)
    return {"spn": spn, "account": "svc_sql", "enctype": "RC4-HMAC (23)",
            "ticket": roast,
            "note": "service ticket requested with RC4 — crackable offline"}


def dcsync_dump(model, token):
    """DCSync — replicate the domain's secrets, NTDS style (T1003.006)."""
    accounts = []
    for u in USERS + [{"user": "krbtgt", "role": "KDC"}]:
        accounts.append({"account": u["user"], "domain": DOMAIN,
                         "nt_hash": _rng_hex(f"{token}:ntds:{u['user']}", 32)})
    return {"method": "DS-Replication-Get-Changes", "directory_event": 4662,
            "accounts": accounts,
            "note": "domain credential material replicated from the DC (incl. krbtgt)"}


def exfil_records(model, token):
    """Exfiltration — the sensitive records leaving the network (T1041)."""
    return {"channel": "C2 / HTTPS", "bytes": 4_820_331,
            "sample_records": [
                {"table": "Finance.Payroll", "rows": 1284},
                {"table": "HR.Employees", "rows": 342},
                {"table": "Finance.Accounts", "rows": 5120}],
            "note": "sensitive datasets transferred to an external host"}


def ransomware(model, token):
    """Ransomware impact — mass file encryption (T1486)."""
    return {"encrypted_hosts": [h for h in model.HOSTS if model.host(h).compromised],
            "extension": ".nwlock", "files_encrypted": 18734,
            "ransom_note": "NORTHWIND_RECOVERY.txt",
            "note": "files encrypted across compromised hosts"}


def domain_accounts(model, token):
    """LDAP account enumeration (T1087.002)."""
    return {"domain": DOMAIN, "accounts": [u["user"] for u in USERS],
            "directory_event": 4661,
            "note": "domain account list enumerated via LDAP"}


# ── Active Directory attack artifacts ──────────────────────────────────────
def asrep_roast(model, token):
    """AS-REP roast hash for a pre-auth-disabled account (T1558.004)."""
    roast = "$krb5asrep$23$svc-admin@" + DOMAIN + ":" + \
            _rng_hex(f"{token}:asrep", 32) + "$" + _rng_hex(f"{token}:asrep2", 200)
    return {"account": "svc-admin", "preauth": "disabled",
            "hash": roast, "kerberos_event": 4768,
            "note": "AS-REP hash recovered — no pre-auth required, crack offline"}


def golden_ticket(model, token):
    """A forged TGT signed with the krbtgt hash (T1558.001)."""
    return {"forged_for": "administrator@" + DOMAIN,
            "krbtgt_hash": _rng_hex(f"{token}:krbtgt", 32),
            "ticket_lifetime": "10 years (anomalous)", "kerberos_event": 4769,
            "note": "Golden Ticket forged — domain-wide impersonation, survives password resets"}


def domain_trusts(model, token):
    """Domain trust map, BloodHound-style (T1482)."""
    return {"domain": DOMAIN,
            "trusts": [{"target": "PARTNER.LOCAL", "type": "External", "direction": "Bidirectional"}],
            "high_value_targets": ["Domain Admins", "Backup Operators", "krbtgt"],
            "directory_event": 4661,
            "note": "trust relationships and privileged groups mapped"}


def group_manip(model, token):
    """Adding an account to a privileged group (T1098)."""
    return {"account": "jsmith", "added_to": "Backup Operators",
            "grants": "Replicating Directory Changes (enables DCSync)",
            "events": [4728, 4732],
            "note": "account escalated via privileged group membership"}


def password_spray(model, token):
    """Result of a password-spray across the domain (T1110.003)."""
    return {"password_tried": "Spring2026!", "accounts_tested": len(USERS) + 20,
            "hits": [{"user": "jsmith", "password": "Spring2026!"}],
            "logon_event": 4625,
            "note": "one weak password matched across the account base"}


# technique -> the artifact function that describes its service-level result.
# Techniques without a rich artifact (scan, backdoor, clear logs, etc.) simply
# return the state; the model already handles those.
ARTIFACTS = {
    "PHISHING_EMAIL": mail_inbox,
    "SQL_INJECTION": lambda m, t: webapp_sqli(m, t, "' OR '1'='1' --"),
    "CRED_DUMP_LSASS": lsass_dump,
    "LATERAL_MOVE_SMB": smb_shares,
    "KERBEROASTING": kerberos_tgs,
    "DCSYNC": dcsync_dump,
    "EXFILTRATE_DATA": exfil_records,
    "RANSOMWARE_ENCRYPT": ransomware,
    "DOMAIN_ACCT_DISCOVERY": domain_accounts,
    "AS_REP_ROASTING": asrep_roast,
    "GOLDEN_TICKET": golden_ticket,
    "DOMAIN_TRUST_DISCOVERY": domain_trusts,
    "ACCOUNT_MANIPULATION": group_manip,
    "PASSWORD_SPRAYING": password_spray,
}


def artifact_for(technique, model, token):
    """Realistic service result for a technique, or None if it has no rich artifact."""
    fn = ARTIFACTS.get(technique)
    if fn is None:
        return None
    try:
        return fn(model, token)
    except Exception as exc:  # never let an artifact break the transition
        return {"error": f"artifact generation failed: {exc}"}
