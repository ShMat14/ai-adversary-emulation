# -*- coding: utf-8 -*-
"""
The v4 emulation target — a multi-service network on the shared kill-chain model.

WHAT CHANGED FROM v3
--------------------
v3's server was a single flat state dict with its own attack logic that
disagreed with the simulator (phishing landed 54% here against 86.5% in sim). It
had one global state, so two episodes at once corrupted each other, and a free
/clean endpoint the agent learned to exploit.

v4 is a real target:
  * It runs the SAME env.kill_chain.KillChainModel and env.detection engine the
    simulator uses, so success probabilities and detection are identical by
    construction. The transfer gap is now transport, not model disagreement.
  * State is per session, not global -- each client gets its own network, so
    episodes run in parallel without corrupting one another.
  * A session token is issued and required for actions against compromised
    hosts, so authentication actually gates capability.
  * It presents a multi-service surface: a mail service and web app on the
    workstation, SMB on the file server, Kerberos and admin on the domain
    controller. Each service endpoint drives the one shared model.
  * It exposes a SIEM feed (/siem/events) of the ATT&CK-mapped security events
    every action emits -- the labelled attack telemetry the platform exists to
    produce.

    python Target/mock_server.py          # serves on :5000

The RL environment drives it through /session, /reset and /attempt; the
per-service endpoints exist so the target can also be driven like a real network
and so the surface is honestly multi-service rather than a single RPC call.
"""
import os
import sys
import secrets
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from dataclasses import asdict

from env.kill_chain import KillChainModel, TECHNIQUES, ACTION_ORDER
from env.detection import INCIDENT_THRESHOLD
import services
import random

app = Flask(__name__)

# ── sessions ────────────────────────────────────────────────────────────────
# Each session owns a fresh model + RNG, so parallel episodes never share state.
# Sessions are evicted after TTL to bound memory.
_lock = threading.Lock()
_sessions = {}
SESSION_TTL = 3600


class Session:
    def __init__(self, token):
        self.token = token
        self.model = KillChainModel()
        self.rng = random.Random()
        self.created = time.time()
        self.touched = time.time()

    def touch(self):
        self.touched = time.time()


def _gc():
    now = time.time()
    for t in [t for t, s in _sessions.items() if now - s.touched > SESSION_TTL]:
        _sessions.pop(t, None)


def _auth_session():
    """Return the session for the request's bearer token, or None."""
    hdr = request.headers.get("Authorization", "")
    if not hdr.startswith("Bearer "):
        return None
    return _sessions.get(hdr[7:])


# ── state serialisation (env mirrors this) ────────────────────────────────
def _state_dict(m: KillChainModel):
    return {
        "hosts": {n: asdict(m.host(n)) for n in m.HOSTS},
        "scanned": m.scanned,
        "backdoor": dict(m.backdoor),
        "suspicion": round(m.detection.suspicion, 4),
        "sqli_available": m.sqli_available,
        "need_persist": m.need_persist,
        "need_hostpriv": m.need_hostpriv,
        "locked_out": m.locked_out,
        "goal": m.is_goal(),
    }


def _apply(session: Session, technique: str):
    """Run one technique through the shared model and package the outcome."""
    m = session.model
    before_events = len(m.detection.log)
    out = m.attempt(technique, session.rng)
    new_events = m.detection.log[before_events:]
    return out, new_events


# ── session lifecycle ─────────────────────────────────────────────────────
@app.route("/session", methods=["POST"])
def new_session():
    with _lock:
        _gc()
        token = secrets.token_hex(16)
        _sessions[token] = Session(token)
    return jsonify({"session": token, "msg": "session created"}), 201


@app.route("/reset", methods=["POST"])
def reset():
    s = _auth_session()
    if s is None:
        # convenience: allow reset to also create a session if none supplied
        with _lock:
            token = secrets.token_hex(16)
            s = _sessions[token] = Session(token)
    data = request.get_json(silent=True) or {}
    scenario = data.get("scenario", None)
    sqli = data.get("sqli_available", None)
    # the client (env) rolls the per-episode scenario and sends it here so both
    # sides run the identical world; if none is sent, fall back to a local roll
    s.model.reset(sqli_available=sqli, rng=s.rng, scenario=scenario)
    s.touch()
    return jsonify({"session": s.token, "state": _state_dict(s.model)}), 200


@app.route("/state", methods=["GET"])
def state():
    s = _auth_session()
    if s is None:
        return jsonify({"error": "no session"}), 401
    return jsonify({"state": _state_dict(s.model)}), 200


# ── the RL interface ───────────────────────────────────────────────────────
@app.route("/attempt", methods=["POST"])
def attempt():
    s = _auth_session()
    if s is None:
        return jsonify({"error": "no session; POST /session first"}), 401
    data = request.get_json(force=True) or {}
    technique = data.get("technique")
    if technique not in TECHNIQUES:
        return jsonify({"error": f"unknown technique {technique!r}"}), 400
    s.touch()
    with _lock:
        out, events = _apply(s, technique)
    # on success, the service returns the realistic artifact a real one would
    artifact = services.artifact_for(technique, s.model, s.token) if out.success else None
    return jsonify({
        "technique": technique,
        "success": out.success,
        "advanced": out.advanced,
        "blocked": out.blocked,
        "detected": out.detected,
        "terminated": out.terminated,
        "events": [asdict_event(e) for e in events],
        "artifact": artifact,
        "state": _state_dict(s.model),
    }), 200


# ── the SIEM feed: the labelled attack telemetry ──────────────────────────
def asdict_event(e):
    return {"event_id": e.event_id, "channel": e.channel, "name": e.name,
            "host": e.host, "mitre_id": e.mitre_id, "technique": e.technique,
            "suspicious": e.suspicious}


@app.route("/siem/events", methods=["GET"])
def siem_events():
    s = _auth_session()
    if s is None:
        return jsonify({"error": "no session"}), 401
    return jsonify({
        "count": len(s.model.detection.log),
        "suspicion": round(s.model.detection.suspicion, 4),
        "incident_threshold": INCIDENT_THRESHOLD,
        "fired_rules": s.model.detection.fired_rules,
        "events": [asdict_event(e) for e in s.model.detection.log],
    }), 200


@app.route("/siem/export", methods=["GET"])
def siem_export():
    """The event log in an ECS-style schema a SIEM could ingest directly.

    Elastic Common Schema is the lingua franca of modern SOC tooling; emitting it
    means the telemetry this range produces is not a bespoke format but something
    a real detection pipeline could consume unchanged.
    """
    s = _auth_session()
    if s is None:
        return jsonify({"error": "no session"}), 401
    docs = []
    for e in s.model.detection.log:
        docs.append({
            "@timestamp": None,   # emulation range: relative ordering only
            "event": {"code": str(e.event_id), "provider": e.channel,
                      "action": e.name, "kind": "alert" if e.suspicious else "event"},
            "host": {"name": e.host},
            "threat": {"technique": {"id": e.mitre_id, "name": e.technique},
                       "framework": "MITRE ATT&CK"},
            "message": f"{e.name} on {e.host}",
        })
    return jsonify({"format": "ecs-like", "count": len(docs), "events": docs}), 200


@app.route("/network/hosts", methods=["GET"])
def network_hosts():
    """Inventory the attacker can see. The DC and file server are only revealed
    once the network has been scanned from inside, as a real discovery would."""
    s = _auth_session()
    if s is None:
        return jsonify({"error": "no session"}), 401
    m = s.model
    visible = ["user01"]
    if m.scanned:
        visible = list(m.HOSTS)
    services = {
        "user01": ["mail", "webapp", "workstation-login"],
        "srv01": ["smb", "fileshare"],
        "dc01": ["kerberos", "ldap", "admin"],
    }
    return jsonify({"visible_hosts": visible,
                    "services": {h: services[h] for h in visible}}), 200


# ── realistic per-service endpoints ────────────────────────────────────────
# Each drives the one shared model via the technique it represents, so the
# multi-service surface stays consistent with the RL interface and the sim.
SERVICE_ROUTES = {
    ("user01", "mail", "click"):        "PHISHING_EMAIL",
    ("user01", "webapp", "query"):      "SQL_INJECTION",
    ("user01", "webapp", "upload"):     "WEB_SHELL_UPLOAD",
    ("auth", "login", "spray"):         "BRUTE_FORCE_SSH",
    ("auth", "login", "valid"):         "VALID_ACCOUNTS_LOGIN",
    ("network", "scan", "sweep"):       "NETWORK_SCAN",
    ("srv01", "smb", "connect"):        "LATERAL_MOVE_SMB",
    ("srv01", "smb", "pth"):            "PASS_THE_HASH",
    ("srv01", "service", "install"):    "INSTALL_BACKDOOR",
    ("host", "priv", "sudo"):           "PRIV_ESC_SUDO",
    ("host", "priv", "powershell"):     "POWERSHELL_EXEC",
    ("dc01", "kerberos", "tgs"):        "KERBEROASTING",
    ("dc01", "admin", "exfil"):         "EXFILTRATE_DATA",
    ("dc01", "admin", "ransom"):        "RANSOMWARE_ENCRYPT",
    ("host", "logs", "clear"):          "CLEAR_LOGS",
}


@app.route("/svc/<host>/<service>/<action>", methods=["POST"])
def service(host, service, action):
    s = _auth_session()
    if s is None:
        return jsonify({"error": "no session; POST /session first"}), 401
    technique = SERVICE_ROUTES.get((host, service, action))
    if technique is None:
        return jsonify({"error": "no such service action"}), 404
    # is the technique even available in the current state?
    idx = ACTION_ORDER.index(technique)
    if not s.model.legal_mask()[idx]:
        return jsonify({"status": "unavailable",
                        "reason": "preconditions not met for this action"}), 409
    s.touch()
    with _lock:
        out, events = _apply(s, technique)
    artifact = services.artifact_for(technique, s.model, s.token) if out.success else None
    status = 200 if out.success else 400
    if out.detected:
        status = 403   # incident response engaged
    return jsonify({
        "service": f"{host}/{service}/{action}",
        "technique": technique,
        "success": out.success,
        "detected": out.detected,
        "terminated": out.terminated,
        "events": [asdict_event(e) for e in events],
        "artifact": artifact,
        "state": _state_dict(s.model),
    }), status


@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"target": "v4 emulation network",
                    "hosts": list(KillChainModel.HOSTS),
                    "techniques": len(TECHNIQUES),
                    "sessions": len(_sessions),
                    "shared_model": "env.kill_chain",
                    "status": "ok"}), 200


def create_app():
    """WSGI factory, so the target can run under a production server:
        waitress-serve --port=5000 --call Target.mock_server:create_app
    """
    return app


if __name__ == "__main__":
    print(">>> v4 EMULATION TARGET on :5000")
    print(f">>> {len(TECHNIQUES)} techniques, per-session state, SIEM feed at /siem/events")
    app.run(port=5000, threaded=True)
