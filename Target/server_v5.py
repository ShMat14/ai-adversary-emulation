# -*- coding: utf-8 -*-
"""The v5 emulation target: the shared world model served over HTTP.

WHY THIS EXISTS

A trained policy that only ever runs inside the process that trained it has not
been shown to survive contact with anything. The usual way to check is to
reimplement the target and compare, and that is exactly what we did in an
earlier version of this work, with the result that a phishing attempt succeeded
86.5% of the time in simulation and 54.0% against the server. The gap measured
disagreement between two pieces of code, not any property of deployment.

So this server does not reimplement anything. It imports `KillChainModelV5` and
`AdversaryEnvV5` and drives them, which is the same module the simulator drives.
A success probability, a precondition or a detection rule is written once. What
a transfer experiment measures here is therefore transport -- serialisation, the
network round trip, session handling -- and not model drift, because model drift
is impossible by construction.

State is per session. Each client gets its own estate, so several agents can be
evaluated at once without corrupting one another, and a session token is
required for every action.

    python Target/server_v5.py                 # serves on :5001
    python Target/server_v5.py --port 8080 --topology hub12

SAFETY. This target is a simulation of a network. It executes no real attack,
touches no host outside the process, and binds to the loopback interface by
default. It exists so a policy can be evaluated over a real transport, not so
that anything can be attacked.
"""
import argparse
import os
import secrets
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request

from env.adversary_env_v5 import AdversaryEnvV5
from env.kill_chain_v5 import TECHNIQUE_ORDER_V5, TECHNIQUES_V5

app = Flask(__name__)

SESSIONS = {}
LOCK = threading.Lock()
SESSION_TTL = 3600.0
DEFAULT_TOPOLOGY = "enterprise"
DEFAULT_MAX_STEPS = 60


def _reap():
    """Drop sessions no one has touched for an hour."""
    now = time.time()
    with LOCK:
        for k in [k for k, v in SESSIONS.items() if now - v["touched"] > SESSION_TTL]:
            del SESSIONS[k]


def _session(token):
    with LOCK:
        s = SESSIONS.get(token)
        if s:
            s["touched"] = time.time()
        return s


@app.post("/session")
def open_session():
    """Open a session and get its own estate."""
    _reap()
    body = request.get_json(silent=True) or {}
    topo = body.get("topology", DEFAULT_TOPOLOGY)
    max_steps = int(body.get("max_steps", DEFAULT_MAX_STEPS))
    try:
        env = AdversaryEnvV5({"topology": topo, "max_steps": max_steps,
                              "illegal_penalty": 0.0, "seed": body.get("seed")})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    token = secrets.token_urlsafe(16)
    with LOCK:
        SESSIONS[token] = {"env": env, "touched": time.time(),
                           "topology": topo, "max_steps": max_steps}
    return jsonify({
        "session": token,
        "topology": topo,
        "hosts": list(env.model.topo.hosts),
        "techniques": list(TECHNIQUE_ORDER_V5),
        "n_actions": env.model.n_actions,
        "obs_dim": int(env.observation_space.shape[0]),
        "max_steps": max_steps,
    })


@app.post("/reset")
def reset():
    s = _session((request.get_json(silent=True) or {}).get("session"))
    if not s:
        return jsonify({"error": "unknown or expired session"}), 401
    seed = (request.get_json(silent=True) or {}).get("seed")
    obs, _ = s["env"].reset(seed=None if seed is None else int(seed))
    return jsonify({"observation": obs.tolist(),
                    "action_mask": [bool(b) for b in s["env"].action_masks()]})


@app.post("/attempt")
def attempt():
    """Execute one (technique, host) action against this session's estate."""
    body = request.get_json(silent=True) or {}
    s = _session(body.get("session"))
    if not s:
        return jsonify({"error": "unknown or expired session"}), 401
    if "action" not in body:
        return jsonify({"error": "no action given"}), 400
    env = s["env"]
    a = int(body["action"])
    if not 0 <= a < env.model.n_actions:
        return jsonify({"error": f"action out of range 0..{env.model.n_actions-1}"}), 400
    technique, target = env.model.decode(a)
    obs, reward, terminated, truncated, info = env.step(a)
    return jsonify({
        "observation": obs.tolist(),
        "action_mask": [bool(b) for b in env.action_masks()],
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "technique": technique,
        "attack_id": TECHNIQUES_V5[technique].mitre_id,
        "tactic": TECHNIQUES_V5[technique].tactic,
        "target": target,
        "result": info["result"],
        "success": bool(info["success"]),
        "advanced": bool(info["advanced"]),
        "objective_reached": bool(env.model.is_goal()),
        "incident_declared": bool(env.model.caught()),
        "suspicion": float(env.model.detection.suspicion),
    })


@app.get("/siem/events")
def siem():
    """The ATT&CK-labelled telemetry this session has produced so far.

    This is the artefact the platform exists to generate, served the way a
    defender would actually consume it.
    """
    s = _session(request.args.get("session"))
    if not s:
        return jsonify({"error": "unknown or expired session"}), 401
    mm = s["env"].model
    since = int(request.args.get("since", 0))
    out = []
    for ev in mm.detection.log[since:]:
        out.append({"event_id": ev.event_id, "channel": ev.channel,
                    "name": ev.name, "host": ev.host,
                    "platform": mm.topo.os_of(ev.host),
                    "technique": ev.technique, "attack_id": ev.mitre_id,
                    "suspicious": bool(ev.suspicious)})
    return jsonify({"events": out, "total": len(mm.detection.log),
                    "suspicion": float(mm.detection.suspicion)})


@app.get("/health")
def health():
    return jsonify({"ok": True, "sessions": len(SESSIONS),
                    "techniques": len(TECHNIQUE_ORDER_V5)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--host", default="127.0.0.1",
                    help="loopback by default; this is a simulation target")
    ap.add_argument("--topology", default=DEFAULT_TOPOLOGY)
    a = ap.parse_args()
    DEFAULT_TOPOLOGY = a.topology
    app.run(host=a.host, port=a.port, threaded=True)
