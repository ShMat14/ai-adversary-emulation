import random
import secrets
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- AUTH TOKEN ---
# A single shared Bearer token that the RL agent / scripts must send.
# In production you would load this from an env variable.
VALID_TOKEN = "rl-agent-secret-token-2026"

def _check_auth():
    """Returns True if the request carries the correct Bearer token."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    return secrets.compare_digest(auth_header[7:], VALID_TOKEN)

# --- SERVER STATE ---
server_state = {
    "compromised":        False,
    "creds_stolen":       False,
    "admin_privileges":   False,
    "scanned":            False,
    "backdoor_installed": False,
    "data_stolen":        False,
    "ransomed":           False,
    "sql_injected":       False,
    "sqli_available":     True,   # Randomized per episode by the env
    "alert_level":        0.0,
    "is_banned":          False
}

# --- SQL INJECTION SIMULATION DATA ---
# Fake database records returned when SQLi succeeds (credential dump)
FAKE_DB_USERS = [
    {"id": 1, "username": "admin",    "password_hash": "sup3r$ecret"},
    {"id": 2, "username": "jsmith",   "password_hash": "pass1234"},
    {"id": 3, "username": "svc_acct", "password_hash": "Adm1n!2026"},
]

# Patterns that a WAF/IDS would flag as SQL injection attempts
SQLI_PATTERNS = ["'", "--", "OR 1=1", "UNION SELECT", "DROP TABLE", "1'='1", ";--", "OR '1'"]


# --- ACTIVE DEFENSE LOGIC ---
def active_defense_check(noise_added=0):
    """Returns True (blocked) if defense triggers, False otherwise."""
    # 1. Already banned → reject immediately
    if server_state["is_banned"]:
        return True

    # 2. Accumulate noise
    server_state["alert_level"] = round(server_state["alert_level"] + noise_added, 3)

    # 3. Hard threshold → lockdown
    if server_state["alert_level"] >= 3.0:
        print(f"!!! CRITICAL ALERT: Level {server_state['alert_level']:.1f} — INITIATING LOCKDOWN !!!")
        server_state["is_banned"] = True
        return True

    # 4. Probabilistic block (risk scales with alert level)
    block_chance = 0.1 + (server_state["alert_level"] * 0.2)
    if random.random() < block_chance:
        print(f"[DEFENSE] Random block triggered (risk: {block_chance:.2f})")
        return True

    return False


# --- ENDPOINTS ---

@app.route('/reset', methods=['POST'])
def reset():
    """Reset all server state between episodes."""
    data = request.get_json(force=True) or {}
    for k in list(server_state.keys()):
        v = server_state[k]
        if isinstance(v, bool):
            server_state[k] = False
        elif isinstance(v, float):
            server_state[k] = 0.0
    # Restore sqli_available from env's randomization decision
    server_state["sqli_available"] = bool(data.get("sqli_available", True))
    return jsonify({"msg": "Reset OK", "sqli_available": server_state["sqli_available"]}), 200


# 1. PHISHING (Quiet Entry)
@app.route('/phish', methods=['POST'])
def phish():
    if active_defense_check(0.1):
        return jsonify({"status": "blocked"}), 403
    if random.random() > 0.4:
        server_state['compromised'] = True
        return jsonify({"status": "success"}), 200
    return jsonify({"status": "failed"}), 400


# 2. BRUTE FORCE (Loud Entry)
@app.route('/brute', methods=['POST'])
def brute():
    if active_defense_check(0.8):
        return jsonify({"status": "blocked"}), 403
    server_state['compromised'] = True
    return jsonify({"status": "success"}), 200


# 3. SCAN
@app.route('/scan', methods=['POST'])
def scan():
    if active_defense_check(0.2):
        return jsonify({"status": "blocked"}), 403
    if server_state['compromised'] or server_state['backdoor_installed']:
        server_state['scanned'] = True
        return jsonify({"hosts": ["srv01", "dc01"]}), 200
    return jsonify({"status": "failed"}), 401


# 4. LOGIN — requires auth token (OWASP A01 fix)
@app.route('/login', methods=['POST'])
def login():
    if not _check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    if active_defense_check(0.1):
        return jsonify({"status": "blocked"}), 403
    if server_state['compromised']:
        server_state['creds_stolen'] = True
        return jsonify({"creds": "admin:p@ssw0rd!"}), 200
    return jsonify({"status": "failed"}), 401


# 5. BACKDOOR — requires compromised state (OWASP A01 fix)
@app.route('/backdoor', methods=['POST'])
def backdoor():
    if not _check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    if active_defense_check(0.3):
        return jsonify({"status": "blocked"}), 403
    if server_state['compromised']:
        server_state['backdoor_installed'] = True
        return jsonify({"msg": "Persistence Established"}), 200
    return jsonify({"status": "failed"}), 401


# 6. CLEAR LOGS (Defense Evasion)
@app.route('/clean', methods=['POST'])
def clean():
    server_state["alert_level"] = 0.0
    return jsonify({"msg": "Logs Wiped", "alert_level": 0.0}), 200


# 7. STATUS (Lateral Movement prerequisite)
@app.route('/status', methods=['GET'])
def status():
    active_defense_check(0.05)
    return jsonify({k: v for k, v in server_state.items()}), 200


# 8. PRIV ESC — requires auth + stolen creds (OWASP A01 fix)
@app.route('/privesc', methods=['POST'])
def privesc():
    if not _check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    if active_defense_check(0.5):
        return jsonify({"status": "blocked"}), 403
    if server_state['creds_stolen']:
        server_state['admin_privileges'] = True
        return jsonify({"status": "root"}), 200
    return jsonify({"status": "failed"}), 403


# 9. EXFIL (Win Type A) — requires admin + auth
@app.route('/exfil', methods=['POST'])
def exfil():
    if not _check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    if active_defense_check(0.5):
        return jsonify({"status": "blocked"}), 403
    if server_state['admin_privileges']:
        server_state['data_stolen'] = True
        return jsonify({"status": "WIN — Data Exfiltrated"}), 200
    return jsonify({"status": "failed"}), 403


# 10. RANSOMWARE (Win Type B) — previously missing endpoint
@app.route('/ransom', methods=['POST'])
def ransom():
    if not _check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    if active_defense_check(0.6):
        return jsonify({"status": "blocked"}), 403
    if server_state['admin_privileges']:
        server_state['ransomed'] = True
        return jsonify({"status": "WIN — Files Encrypted"}), 200
    return jsonify({"status": "failed"}), 403


# 11. SQL INJECTION (Initial Access + Credential Dump — no auth required)
# Simulates a vulnerable public-facing login or search form.
# The WAF raises alert level and may block based on pattern detection.
@app.route('/sqli', methods=['POST'])
def sqli():
    # Check if this target is even vulnerable to SQLi this episode
    if not server_state["sqli_available"]:
        return jsonify({"status": "failed",
                        "reason": "No vulnerable web endpoint found"}), 404

    data = request.get_json(force=True) or {}
    query = str(data.get("query", ""))

    if active_defense_check(0.25):
        return jsonify({"status": "blocked",
                        "reason": "WAF: Suspicious query pattern detected"}), 403

    is_sqli = any(pat.lower() in query.lower() for pat in SQLI_PATTERNS)
    if not is_sqli:
        return jsonify({"status": "failed",
                        "reason": "Query looks clean — no injection detected"}), 400

    if server_state["sql_injected"]:
        return jsonify({"status": "already_exploited", "msg": "DB already dumped"}), 200

    server_state["sql_injected"] = True
    server_state["compromised"]   = True
    server_state["creds_stolen"]  = True
    return jsonify({
        "status":  "success",
        "msg":     "SQL Injection successful — DB records exfiltrated",
        "records": FAKE_DB_USERS,
    }), 200


# 12. PASS THE HASH (T1550.002) — Lateral Movement using credential hashes
@app.route('/pth', methods=['POST'])
def pth():
    if not _check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    if active_defense_check(0.20):
        return jsonify({"status": "blocked"}), 403
    # Requires stolen credentials (hash source)
    if server_state['creds_stolen']:
        server_state['compromised'] = True
        return jsonify({"status": "success", "msg": "Pass-the-Hash auth accepted"}), 200
    return jsonify({"status": "failed", "reason": "No credential hashes available"}), 403


# 13. POWERSHELL EXEC (T1059.001) — Remote execution on compromised host
@app.route('/psexec', methods=['POST'])
def psexec():
    if not _check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    if active_defense_check(0.25):
        return jsonify({"status": "blocked"}), 403
    if server_state['compromised']:
        server_state['admin_privileges'] = True
        return jsonify({"status": "success", "msg": "PowerShell session established — elevated"}), 200
    return jsonify({"status": "failed", "reason": "No foothold on target"}), 403


# 14. KERBEROASTING (T1558.003) — Request + crack service tickets offline
@app.route('/kerberoast', methods=['POST'])
def kerberoast():
    if not _check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    if active_defense_check(0.15):
        return jsonify({"status": "blocked"}), 403
    # Requires creds + host scanned (inside the network)
    if server_state['creds_stolen'] and server_state['scanned']:
        server_state['admin_privileges'] = True
        return jsonify({
            "status": "success",
            "msg":    "Service tickets obtained — cracked offline",
            "hash":   "$krb5tgs$23$*svc_acct$DOMAIN.LOCAL$...<hash>",
        }), 200
    return jsonify({"status": "failed", "reason": "Need valid creds + internal network access"}), 403


# 15. WEB SHELL UPLOAD (T1505.003) — Persistent shell after web compromise
@app.route('/webshell', methods=['POST'])
def webshell():
    if active_defense_check(0.20):
        return jsonify({"status": "blocked"}), 403
    # Requires SQLi compromise (web app already pwned)
    if server_state['sql_injected'] or server_state['compromised']:
        server_state['backdoor_installed'] = True
        server_state['scanned'] = True   # Web shell gives internal visibility
        return jsonify({"status": "success", "msg": "Web shell planted — persistent access established"}), 200
    return jsonify({"status": "failed", "reason": "No web compromise found"}), 403


if __name__ == '__main__':
    print(f">>> TARGET SERVER (15-ACTION MODE) RUNNING ON PORT 5000")
    print(f">>> Auth token: {VALID_TOKEN}")
    app.run(port=5000)