# -*- coding: utf-8 -*-
"""
The shared kill-chain model — one definition of the world, used by both the
simulator and the live server.

WHY THIS EXISTS
---------------
In v3 the simulator (env/adversary_env.py) and the mock server
(Target/mock_server.py) each carried their own copy of the attack logic, and
they disagreed. Measured on the v3 code, phishing succeeded 86.5% of the time in
simulation and 54.0% against the server; brute force 53.5% against 74.5%. The
paper's sim-to-real "transfer gap" was therefore partly a gap between two
hand-written models rather than a property of deployment.

v4 removes that possibility. This module owns:
  * the technique registry (success probability, noise, ATT&CK mapping);
  * the network state (per host);
  * the precondition mask;
  * the state transition for attempting a technique.

AdversaryEnv wraps it to present a Gym environment for training. The server wraps
it to answer HTTP. Both call the same `attempt`, so a success probability or a
precondition can only ever be defined once.

WHAT IS AND IS NOT HERE
-----------------------
Here: everything about what the world is and how it changes -- state,
preconditions, success rolls, effects, detection. This is the part that must be
identical in both modes.

Not here: reward shaping. Reward is a training signal with no meaning to the
server, so it lives in AdversaryEnv, computed from the outcome this module
returns. Keeping it out means the server cannot accidentally depend on a reward
constant, and the reward can be retuned without touching the world model.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional
import random

from env.state_models import NetworkState
from env.detection import DetectionEngine, INCIDENT_THRESHOLD


# ── detection model ──────────────────────────────────────────────────────
# Detection is delegated to env/detection.py, a rule-based blue team shared by
# both modes. The "alert level" the agent observes is the SOC's accumulated
# suspicion (0 .. INCIDENT_THRESHOLD). CLEAR_LOGS can lower it, at the cost of
# raising the loud Event 1102. The floor below is only the point past which
# clearing logs becomes a legal move -- there is something worth clearing.
CLEAR_FLOOR = 0.25         # suspicion above this makes CLEAR_LOGS available
CLEAR_REDUCTION = 0.5      # how much a successful clear removes


@dataclass
class Technique:
    """One attack action: its identity, its cost, and its world effect.

    precondition(model) -> bool   is the action available in this state?
    effect(model) -> bool         apply the state change; return True if it did
                                  something (used to distinguish a real advance
                                  from a no-op). Only called on a successful roll.
    """
    name: str
    mitre_id: str
    tactic: str
    success_prob: float
    noise: float
    precondition: Callable
    effect: Callable
    terminal: bool = False       # does a successful use end the episode (a win)?


@dataclass
class Outcome:
    """What attempting a technique did, for whichever caller wants it."""
    attempted: bool = True
    blocked: bool = False
    success: bool = False
    advanced: bool = False
    detected: bool = False
    terminated: bool = False


class KillChainModel:
    """The mutable world: a three-host domain and the attacker's progress in it."""

    HOSTS = ("user01", "srv01", "dc01")

    # The substitutable choice at each kill-chain phase. Every episode a random
    # non-empty subset of each group is made available, so no single path is
    # always open: the agent must read what the scenario permits and pick
    # accordingly. This is what forces the whole breadth of the 25 techniques
    # into use rather than one memorised chain, and what makes the policy adapt.
    AVAIL_GROUPS = {
        "entry":     ["PHISHING_EMAIL", "SQL_INJECTION", "PASSWORD_SPRAYING", "BRUTE_FORCE_SSH"],
        "discovery": ["NETWORK_SCAN", "DOMAIN_ACCT_DISCOVERY", "DOMAIN_TRUST_DISCOVERY"],
        "cred":      ["VALID_ACCOUNTS_LOGIN", "AS_REP_ROASTING", "CRED_DUMP_LSASS"],
        "lateral":   ["LATERAL_MOVE_SMB", "PASS_THE_HASH", "PASS_THE_TICKET"],
        "dc":        ["KERBEROASTING", "DCSYNC", "GPO_MODIFICATION", "GOLDEN_TICKET"],
        "persist":   ["INSTALL_BACKDOOR", "WEB_SHELL_UPLOAD"],
        "hostpriv":  ["PRIV_ESC_SUDO", "POWERSHELL_EXEC", "ACCOUNT_MANIPULATION"],
        "impact":    ["EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT"],
    }
    AVAIL_ORDER = [n for grp in AVAIL_GROUPS.values() for n in grp]

    def __init__(self):
        self.state = NetworkState(list(self.HOSTS))
        self.detection = DetectionEngine()
        self.scanned = False                     # network discovery (global)
        self.backdoor = {h: False for h in self.HOSTS}   # persistence, per host
        self.sqli_available = True
        self.locked_out = False
        # per-episode availability: True = this technique is viable this scenario
        self.avail = {n: True for n in self.AVAIL_ORDER}
        # per-episode objective requirements, off the default critical path.
        # need_persist:  the objective requires established persistence -- impact
        #                (exfil/ransomware) is only reachable once a backdoor or
        #                web shell is planted, so persistence is forced into use.
        # need_hostpriv: reading credentials requires local elevation first, so
        #                the agent must escalate a foothold (sudo / PowerShell /
        #                account manipulation) before it can harvest and pivot.
        # Both are observed, so the agent adapts: it performs these steps only
        # when the scenario demands them, and skips them when it does not.
        self.need_persist = False
        self.need_hostpriv = False
        # Optional catalogue restriction: when set, only these techniques are
        # ever offered. Used to run the v4 method over the original 15-technique
        # catalogue, so catalogue size can be varied with everything else held
        # fixed (the supervisor's 15-vs-25 question).
        self.restrict = None

    # the observable "alert level" is the SOC's suspicion score
    @property
    def alert(self):
        return self.detection.suspicion

    @alert.setter
    def alert(self, value):
        self.detection.suspicion = max(0.0, float(value))

    # ── lifecycle ────────────────────────────────────────────────────────
    def reset(self, sqli_available: Optional[bool] = None,
              rng: Optional[random.Random] = None,
              randomize_scenario: bool = True,
              scenario: Optional[dict] = None):
        self.state.reset()
        self.detection.reset()
        self.scanned = False
        self.backdoor = {h: False for h in self.HOSTS}
        self.locked_out = False
        r = rng or random

        # An explicit scenario (real mode: the env rolls it and sends it to the
        # server verbatim) is applied as-is, so both sides run the identical
        # world and cannot drift. This is the one source of truth for a run.
        if scenario is not None:
            self.avail = {n: bool(scenario["avail"].get(n, True)) for n in self.AVAIL_ORDER}
            self.need_persist = bool(scenario["need_persist"])
            self.need_hostpriv = bool(scenario["need_hostpriv"])
            self.sqli_available = self.avail["SQL_INJECTION"]
            return

        if randomize_scenario:
            # enable a random non-empty subset of each phase group; the agent
            # observes this and must pick an available technique at every stage
            self.avail = {}
            for techs in self.AVAIL_GROUPS.values():
                pool = [t for t in techs if self.restrict is None or t in self.restrict]
                if not pool:
                    raise RuntimeError(
                        "catalogue restriction empties a phase group: %s" % techs)
                self.avail.update(self._rand_subset(pool, r))
                # anything outside the restriction is never offered
                for t in techs:
                    if t not in pool:
                        self.avail[t] = False
            # objective requirements that pull persistence and host privilege
            # escalation onto the critical path in some scenarios (see __init__)
            self.need_persist = r.random() < 0.5
            self.need_hostpriv = r.random() < 0.5
        else:
            self.avail = {n: (self.restrict is None or n in self.restrict)
                          for n in self.AVAIL_ORDER}
            self.need_persist = False
            self.need_hostpriv = False

        # sqli availability is just the SQL-injection entry being open; an
        # explicit argument (reproducible probes) overrides the roll
        if sqli_available is not None:
            self.avail["SQL_INJECTION"] = bool(sqli_available)
            # Forcing SQL injection off must not leave the entry phase empty --
            # that scenario has no way in and would raise on the first mask. If
            # the roll left no other entry open, open a random non-SQLi one, so
            # the forced-off case still varies its entry rather than always
            # falling back to the same technique (which would bias adaptation).
            entry = self.AVAIL_GROUPS["entry"]
            if not any(self.avail[e] for e in entry):
                alt = [e for e in entry if e != "SQL_INJECTION"]
                self.avail[r.choice(alt)] = True
        self.sqli_available = self.avail["SQL_INJECTION"]

    def scenario_dict(self):
        """The full per-episode scenario, for sending to the server (real mode)."""
        return {"avail": dict(self.avail),
                "need_persist": self.need_persist,
                "need_hostpriv": self.need_hostpriv,
                "sqli_available": self.sqli_available}

    @staticmethod
    def _rand_subset(keys, rng):
        """A random non-empty availability dict over `keys`."""
        while True:
            avail = {k: rng.random() < 0.6 for k in keys}
            if any(avail.values()):
                return avail

    # ── convenience accessors ────────────────────────────────────────────
    def host(self, name):
        return self.state.hosts[name]

    def any_compromised(self):
        return any(h.compromised for h in self.state.hosts.values())

    def any_creds(self):
        return any(h.has_credentials for h in self.state.hosts.values())

    def next_hop(self):
        """The host the attacker would take next, enforcing user01→srv01→dc01."""
        if not self.host("srv01").compromised:
            return "srv01"
        if not self.host("dc01").compromised:
            return "dc01"
        return None

    def is_goal(self):
        return self.host("dc01").privileged

    def dc_endgame(self):
        """True once the DC is compromised but not yet rooted."""
        return self.host("dc01").compromised and not self.host("dc01").privileged

    def target_host(self, name: str) -> str:
        """Which host a technique acts on, for event attribution."""
        if name in ("PHISHING_EMAIL", "SQL_INJECTION"):
            return "user01"
        if name == "BRUTE_FORCE_SSH":
            return next((h for h in self.HOSTS if not self.host(h).compromised), "user01")
        if name in ("LATERAL_MOVE_SMB", "PASS_THE_HASH"):
            return self.next_hop() or "dc01"
        if name in ("KERBEROASTING", "EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT"):
            return "dc01"
        if name in ("VALID_ACCOUNTS_LOGIN", "INSTALL_BACKDOOR", "WEB_SHELL_UPLOAD",
                    "PRIV_ESC_SUDO", "POWERSHELL_EXEC"):
            for h in self.HOSTS:
                if self.host(h).compromised:
                    return h
            return "user01"
        # NETWORK_SCAN, CLEAR_LOGS: attribute to the most recently owned host
        for h in reversed(self.HOSTS):
            if self.host(h).compromised:
                return h
        return "user01"

    # ── the shared transition ────────────────────────────────────────────
    def attempt(self, name: str, rng: random.Random) -> Outcome:
        """Attempt a technique. This is the one code path both modes use.

        Rolls the technique's own success, then runs the detection engine over
        the events it emits. Detection and success are independent: a technique
        can succeed and still be caught (the SOC escalates to incident response
        and the run ends), which is how real intrusions fail. CLEAR_LOGS is
        special only in that a success lowers accumulated suspicion.
        """
        tech = TECHNIQUES[name]
        host = self.target_host(name)
        out = Outcome()

        if self.locked_out:
            out.blocked = out.detected = out.terminated = True
            return out

        # Full legality = the technique's precondition holds AND it is available
        # this scenario. This is the same test legal_mask() uses. Enforcing it
        # here, in the one transition both modes share, makes the mask a true
        # reflection of the world. An action whose preconditions are unmet is not
        # executed against the target: the operator's tooling rejects it locally
        # or it cannot connect, so it neither advances the intrusion nor leaves
        # target-side telemetry. Without this an unmasked agent could bypass a
        # scenario requirement -- exfiltrate without the persistence the objective
        # needs, or read credentials without the elevation it needs -- which the
        # mask forbids but the transition would otherwise allow.
        if not (tech.precondition(self) and self.avail.get(name, True)):
            return out   # attempted, but rejected before execution: inert

        succeeded = rng.random() < tech.success_prob

        # the blue team observes the attempt whether or not it succeeded --
        # failed loud techniques (failed logons) are often the louder signal
        det = self.detection.observe(
            name, host, succeeded,
            has_backdoor=any(self.backdoor.values()), rng=rng)

        if det.caught:
            out.detected = out.terminated = True
            self.locked_out = True
            self.state.detection_triggered = True
            self.host(host).detected = True
            # a technique that succeeded still changed state before the SOC
            # escalated to incident response
            if succeeded and name != "CLEAR_LOGS":
                out.success = True
                out.advanced = bool(tech.effect(self))
            return out

        if not succeeded:
            return out

        out.success = True
        if name == "CLEAR_LOGS":
            self.detection.suspicion = max(0.0, self.detection.suspicion - CLEAR_REDUCTION)
            out.advanced = True
            return out

        out.advanced = bool(tech.effect(self))
        if tech.terminal and out.advanced:
            out.terminated = True
        return out

    # ── the precondition mask ────────────────────────────────────────────
    def legal_mask(self):
        """Boolean list over ACTION_ORDER: which techniques are available now.

        Raises if it would be empty. v3 silently opened every action in that
        case, which would quietly void the masking guarantee the paper rests on;
        a raise turns a logic hole into a loud failure instead.
        """
        # a technique is legal iff its preconditions hold AND it is available
        # this scenario (availability gates the substitutable phase choices)
        mask = [TECHNIQUES[n].precondition(self) and self.avail.get(n, True)
                for n in ACTION_ORDER]
        if not any(mask):
            raise RuntimeError(
                "empty action mask: no technique is legal in state "
                f"{self.snapshot()} -- prerequisite logic has a hole"
            )
        return mask

    def avail_vector(self):
        """Availability flags in AVAIL_ORDER, for the agent's observation."""
        return [1.0 if self.avail.get(n, True) else 0.0 for n in self.AVAIL_ORDER]

    def requirement_vector(self):
        """Per-episode objective requirements, for the agent's observation, so
        it can tell when persistence / host elevation are on the critical path."""
        return [1.0 if self.need_persist else 0.0,
                1.0 if self.need_hostpriv else 0.0]

    def snapshot(self):
        h = self.state.hosts
        return {
            n: (int(h[n].compromised), int(h[n].has_credentials), int(h[n].privileged))
            for n in self.HOSTS
        } | {"scan": int(self.scanned),
             "backdoor": {k: int(v) for k, v in self.backdoor.items()},
             "alert": round(self.alert, 3)}


# ── preconditions ─────────────────────────────────────────────────────────
# Each returns True if the technique is legal in the given model state. The DC
# endgame narrows the legal set to privilege escalation, then to impact, so the
# agent cannot wander once it is on the domain controller.
def _pre_phish(m):   return not m.dc_endgame() and not m.is_goal() and not m.host("user01").compromised
def _pre_sqli(m):    return not m.dc_endgame() and not m.is_goal() and m.sqli_available and not m.host("user01").compromised
def _pre_brute(m):
    if m.dc_endgame() or m.is_goal():
        return False
    return not m.host("user01").compromised or not m.host("srv01").compromised
def _pre_scan(m):    return not m.dc_endgame() and not m.is_goal() and m.any_compromised() and not m.scanned
def _has_elevated_foothold(m):
    """A compromised non-DC host that has been locally escalated -- the
    precondition for reading credentials when the scenario requires elevation."""
    return any(h.compromised and h.privileged and h.name != "dc01"
               for h in m.state.hosts.values())
def _pre_login(m):
    if m.dc_endgame() or m.is_goal():
        return False
    if not any(h.compromised and not h.has_credentials for h in m.state.hosts.values()):
        return False
    # when the scenario demands elevation to read credentials, a foothold must
    # be escalated first -- this forces host privilege escalation onto the path
    if m.need_hostpriv and not _has_elevated_foothold(m):
        return False
    return True
def _pre_backdoor(m):
    if m.is_goal():
        # persistence may still be required to enable impact after the DC falls;
        # keep it available in that case so the objective stays reachable
        return m.need_persist and not any(m.backdoor.values())
    if m.dc_endgame():
        return False
    return m.any_compromised() and any(not m.backdoor[h] for h in m.HOSTS if m.host(h).compromised)
def _pre_clear(m):   return m.alert > CLEAR_FLOOR
def _pre_lateral(m): return not m.is_goal() and m.any_creds() and m.scanned and m.next_hop() is not None and not m.dc_endgame()
def _pre_pth(m):     return not m.is_goal() and m.any_creds() and m.next_hop() is not None and not m.dc_endgame()
def _pre_privesc_host(m):
    if m.dc_endgame():   # non-DC privesc, offered only outside the DC endgame
        return False
    return any(h.compromised and not h.privileged and h.name != "dc01" for h in m.state.hosts.values())
def _pre_dc_privesc(m):  return m.dc_endgame()
def _pre_impact(m):
    # the objective (exfiltration / ransomware) is reachable once the DC is
    # rooted, unless the scenario requires established persistence first, which
    # forces a backdoor / web shell onto the path before impact
    if not m.is_goal():
        return False
    return (not m.need_persist) or any(m.backdoor.values())
# extended techniques (v4): louder or quieter alternatives to existing steps,
# so the agent has genuine tradecraft choices rather than one forced path
def _pre_lsass(m):       return _pre_login(m)          # dump creds from a foothold
def _pre_domain_disc(m): return _pre_scan(m)           # quiet LDAP enumeration
def _pre_dcsync(m):      return _pre_dc_privesc(m)      # replicate the DC's secrets
# ── Active Directory attack techniques ────────────────────────────────────
# Grounded in the AD kill chain from the supervisor's material: enum -> AS-REP
# roast -> lateral -> DCSync -> domain dominance, plus password spraying, GPO
# abuse and group-membership escalation. Event IDs come from that material.
def _pre_trust_disc(m):  return _pre_scan(m)            # BloodHound/PowerView trust enum (a recon path)
def _pre_asrep(m):
    # roast accounts with Kerberos pre-auth disabled: needs the domain enumerated
    # (scanned) and a compromised host still lacking credentials
    if m.dc_endgame() or m.is_goal():
        return False
    if not (m.scanned and any(h.compromised and not h.has_credentials
                              for h in m.state.hosts.values())):
        return False
    if m.need_hostpriv and not _has_elevated_foothold(m):
        return False
    return True
def _pre_pwspray(m):     return _pre_brute(m)           # spray one password across accounts (entry)
def _pre_ptt(m):         return _pre_pth(m)             # pass-the-ticket: like PtH, uses a stolen ticket
def _pre_acct_manip(m):  return _pre_privesc_host(m)    # add self to a privileged local/domain group
def _pre_gpo(m):         return _pre_dc_privesc(m)      # abuse a GPO for SYSTEM on the DC
def _pre_golden(m):      return _pre_dc_privesc(m)      # forge a TGT with the krbtgt hash


# ── effects ────────────────────────────────────────────────────────────────
# Applied only on a successful roll. Return True if the state actually changed.
def _eff_phish(m):
    m.host("user01").compromised = True
    return True

def _eff_sqli(m):
    u = m.host("user01")
    u.compromised = True
    u.has_credentials = True     # SQLi dumps credentials directly
    return True

def _eff_brute(m):
    for n in m.HOSTS:
        if not m.host(n).compromised:
            m.host(n).compromised = True
            return True
    return False

def _eff_scan(m):
    m.scanned = True
    return True

def _eff_login(m):
    for n in m.HOSTS:
        h = m.host(n)
        if h.compromised and not h.has_credentials:
            h.has_credentials = True
            return True
    return False

def _eff_backdoor(m):
    for n in m.HOSTS:
        if m.host(n).compromised and not m.backdoor[n]:
            m.backdoor[n] = True
            return True
    return False

def _eff_lateral(m):
    hop = m.next_hop()
    if hop:
        m.host(hop).compromised = True
        return True
    return False

def _eff_host_privesc(m):
    for n in m.HOSTS:
        h = m.host(n)
        if h.compromised and not h.privileged and n != "dc01":
            h.privileged = True
            return True
    return False

def _eff_dc_privesc(m):
    dc = m.host("dc01")
    if dc.compromised and not dc.privileged:
        dc.privileged = True
        return True
    return False

def _eff_impact(m):
    return m.is_goal()   # win condition already met; the action realises it


# ── the registry ───────────────────────────────────────────────────────────
# Success probabilities are the single declared values; both modes read these.
# ATT&CK ids match env/attack_actions.py so the two stay aligned.
def _t(name, mid, tactic, prob, noise, pre, eff, terminal=False):
    return Technique(name, mid, tactic, prob, noise, pre, eff, terminal)

_TECHS = [
    _t("PHISHING_EMAIL",       "T1566",     "Initial Access",       0.90, 0.05, _pre_phish,        _eff_phish),
    _t("BRUTE_FORCE_SSH",      "T1110",     "Credential Access",    0.60, 0.50, _pre_brute,        _eff_brute),
    _t("NETWORK_SCAN",         "T1046",     "Discovery",            0.95, 0.10, _pre_scan,         _eff_scan),
    _t("VALID_ACCOUNTS_LOGIN", "T1078",     "Defense Evasion",      0.95, 0.05, _pre_login,        _eff_login),
    _t("INSTALL_BACKDOOR",     "T1543",     "Persistence",          0.80, 0.15, _pre_backdoor,     _eff_backdoor),
    _t("CLEAR_LOGS",           "T1070",     "Defense Evasion",      0.90, 0.05, _pre_clear,        lambda m: True),
    _t("LATERAL_MOVE_SMB",     "T1021",     "Lateral Movement",     0.85, 0.15, _pre_lateral,      _eff_lateral),
    _t("PRIV_ESC_SUDO",        "T1068",     "Privilege Escalation", 0.70, 0.30, _pre_privesc_host, _eff_host_privesc),
    _t("EXFILTRATE_DATA",      "T1041",     "Exfiltration",         1.00, 0.25, _pre_impact,       _eff_impact, terminal=True),
    _t("RANSOMWARE_ENCRYPT",   "T1486",     "Impact",               0.90, 0.35, _pre_impact,       _eff_impact, terminal=True),
    _t("SQL_INJECTION",        "T1190",     "Initial Access",       0.85, 0.12, _pre_sqli,         _eff_sqli),
    _t("PASS_THE_HASH",        "T1550.002", "Lateral Movement",     0.80, 0.20, _pre_pth,          _eff_lateral),
    _t("POWERSHELL_EXEC",      "T1059.001", "Execution",            0.85, 0.25, _pre_privesc_host, _eff_host_privesc),
    _t("KERBEROASTING",        "T1558.003", "Credential Access",    0.75, 0.15, _pre_dc_privesc,   _eff_dc_privesc),
    _t("WEB_SHELL_UPLOAD",     "T1505.003", "Persistence",          0.80, 0.20, _pre_backdoor,     _eff_backdoor),
    # ── extended techniques (v4) ──────────────────────────────────────────
    # a loud credential path (dump LSASS) vs the quiet VALID_ACCOUNTS_LOGIN,
    # a quiet discovery path (LDAP enum) vs the loud NETWORK_SCAN, and a second
    # DC-escalation route (DCSync) alongside Kerberoasting.
    _t("CRED_DUMP_LSASS",      "T1003.001", "Credential Access",    0.85, 0.20, _pre_lsass,        _eff_login),
    _t("DOMAIN_ACCT_DISCOVERY","T1087.002", "Discovery",            0.95, 0.05, _pre_domain_disc,  _eff_scan),
    _t("DCSYNC",               "T1003.006", "Credential Access",    0.85, 0.15, _pre_dcsync,       _eff_dc_privesc),
    # ── Active Directory attack techniques (from the supervisor's material) ──
    _t("DOMAIN_TRUST_DISCOVERY","T1482",    "Discovery",            0.95, 0.05, _pre_trust_disc,   _eff_scan),
    _t("AS_REP_ROASTING",      "T1558.004", "Credential Access",    0.80, 0.15, _pre_asrep,        _eff_login),
    _t("PASSWORD_SPRAYING",    "T1110.003", "Credential Access",    0.55, 0.30, _pre_pwspray,      _eff_brute),
    _t("PASS_THE_TICKET",      "T1550.003", "Lateral Movement",     0.80, 0.20, _pre_ptt,          _eff_lateral),
    _t("ACCOUNT_MANIPULATION", "T1098",     "Privilege Escalation", 0.75, 0.25, _pre_acct_manip,   _eff_host_privesc),
    _t("GPO_MODIFICATION",     "T1484.001", "Privilege Escalation", 0.80, 0.20, _pre_gpo,          _eff_dc_privesc),
    _t("GOLDEN_TICKET",        "T1558.001", "Privilege Escalation", 0.85, 0.25, _pre_golden,       _eff_dc_privesc),
]

# The domain controller must be escalated through a real AD route (Kerberoast,
# DCSync, GPO abuse or Golden Ticket), not by sudo/PowerShell — you do not
# "sudo" a domain controller. The host-privesc techniques therefore stay
# non-DC (their registry preconditions already enforce this), which forces the
# agent to use whichever AD route the scenario makes available.

TECHNIQUES = {t.name: t for t in _TECHS}
ACTION_ORDER = [t.name for t in _TECHS]   # fixed index order for the RL action space
