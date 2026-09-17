# -*- coding: utf-8 -*-
"""v5 shared kill-chain model: the agent chooses the target as well as the technique.

WHY THIS EXISTS
---------------
v4 asked the agent to choose a technique and then chose the target for it: the
progression user01 -> srv01 -> dc01 was hard-coded in `next_hop()`, every
precondition asked "is there ANY host where this applies?" and every effect
"find the FIRST such host". That is an easier problem than the one the field
poses. Koo et al. (E-NASim, ETRI Journal 2026) state the standard formulation
directly -- an action is a (host, technique) pair, so A = H x T.

v5 adopts that formulation and, deliberately, three further elements of their
design, so the comparison is against their environment rather than a weaker
strawman:

  * per-host `discovered` and `reachable` flags distinct from `compromised`,
    so reconnaissance is a real prerequisite rather than a global switch;
  * graded privilege (none / user / admin) instead of a boolean;
  * an execution-result vector carrying the REASON an action failed
    (ConnErr / PermErr / UndefErr), their Eq. 7.

That last one matters for fairness. An unmasked agent in v4 learned only that an
action failed. E-NASim's agent is told why. If masking still beats an unmasked
agent receiving the same diagnostic feedback, the result cannot be dismissed as
an artefact of starving the baseline of information.

What v5 keeps that no system in this lineage has: the rule-based detection engine
keyed to real Windows Security, Sysmon and WAF event identifiers, and the
labelled telemetry it emits.

v4 is untouched, so its published numbers stay reproducible. The `v4compat`
topology reproduces the original three-host network under the new action space,
isolating the formulation change from the change in network size.
"""
from dataclasses import dataclass
from typing import Callable, Optional
import random

from env.detection import (DetectionEngine, INCIDENT_THRESHOLD,
                           register_mitre as _register_mitre)
from env.topology_v5 import Topology

CLEAR_FLOOR = 0.25
CLEAR_REDUCTION = 0.5

# privilege ladder
P_NONE, P_USER, P_ADMIN = 0, 1, 2

# execution-result codes (E-NASim Eq. 7), reported to the agent every step
R_SUCCESS, R_CONNERR, R_PERMERR, R_UNDEF = 0, 1, 2, 3
RESULT_NAMES = ["success", "conn_err", "perm_err", "undef_err"]


# Kinds of credential material. Modelling these separately is what stops
# pass-the-hash, pass-the-ticket and an SMB logon from being three names for one
# action: they consume different stolen material, and the material has to be
# obtained by a technique that actually produces it. A golden ticket needs the
# krbtgt hash, which needs DCSync, which needs local administrator somewhere and
# a domain enumeration first. That chain is the point of the environment.
C_PASSWORD, C_NTLM, C_TICKET = "password", "ntlm", "ticket"


@dataclass
class HostV5:
    name: str
    subnet: int
    asset_value: float = 1.0
    compromised: bool = False
    discovered: bool = False
    privilege: int = P_NONE
    backdoored: bool = False
    # what has been enumerated on this host. Local escalation needs it: an
    # operator finds the vulnerable service or driver before abusing it, and
    # without this SYSTEM_INFO_DISCOVERY is a no-op the agent can farm.
    enumerated: bool = False
    staged: bool = False        # data collected on this host
    archived: bool = False      # and packaged for egress
    creds: set = None           # subset of {password, ntlm, ticket}

    def __post_init__(self):
        if self.creds is None:
            self.creds = set()

    @property
    def has_credentials(self):
        """Any material at all. Kept so callers that only ask "are there creds
        here?" -- the observation, the lateral-movement source search -- read
        unchanged."""
        return bool(self.creds)

    def holds(self, kind):
        return kind in self.creds

    def reset(self):
        self.compromised = self.discovered = False
        self.privilege = P_NONE
        self.backdoored = self.enumerated = self.staged = self.archived = False
        self.creds = set()


@dataclass
class TechniqueV5:
    """One attack action; `scope` fixes how its target argument is read.

    entry  : target is an internet-exposed host being broken into
    recon  : target is a compromised host the agent operates *from*
    local  : target is a compromised host acted on in place
    remote : target is a destination reached from some compromised source
    impact : target is the objective, once it has fallen
    """
    name: str
    mitre_id: str
    tactic: str
    scope: str
    success_prob: float
    noise: float
    precondition: Callable      # (model, target) -> bool
    effect: Callable            # (model, target) -> bool
    fail_code: int = R_UNDEF    # reason reported when the precondition fails
    terminal: bool = False
    # Optional runtime gate: (model, target) -> True when the attempt should be
    # refused at execution time rather than excluded from the mask. This is the
    # difference between an action that cannot be taken and one that can be
    # taken and denied. Dumping credentials without local administrator is the
    # second kind: the attempt is made and the operating system refuses it, which
    # is what E-NASim's PermErr encodes (their Eq. 7). Modelling it as a mask
    # exclusion instead gave the agent no signal at all -- the action simply
    # vanished -- and some seeds never discovered the requirement.
    runtime_gate: Optional[Callable] = None
    # Substitution group. Techniques sharing a group are alternative routes to
    # the same outcome, and each episode offers a random non-empty subset of the
    # group. This is what makes the mask depend on the scenario as well as the
    # state -- mu(s, omega) rather than mu(s) -- and it is the reason a
    # memorised path cannot survive here: an agent that always reaches for
    # Kerberoasting must use AS-REP roasting in the episodes that only offer it.
    group: Optional[str] = None


@dataclass
class OutcomeV5:
    attempted: bool = True
    blocked: bool = False
    success: bool = False
    advanced: bool = False
    detected: bool = False
    terminated: bool = False
    result_code: int = R_SUCCESS


# ---------------------------------------------------------------------------
# preconditions and effects.  Every one takes (model, target) and answers about
# THAT host -- the inversion of v4, where preconditions asked "is there any host"
# and effects picked the first match themselves.
# ---------------------------------------------------------------------------

# -- entry: break in from outside -------------------------------------------
# Each entry vector needs the service it actually abuses, so the DMZ's exposed
# surface decides which ones exist. Without this every entry technique is the
# same action under four ATT&CK identifiers.
def _pre_entry(svc=None, need_stolen_creds=False):
    def f(m, t):
        h = m.host(t)
        if h.compromised or not m.topo.exposed(t):
            return False
        if svc and not m.topo.has_service(t, svc):
            return False
        if need_stolen_creds and not any(o.holds(C_PASSWORD) for o in m.hosts.values()):
            return False
        return True
    return f


def _pre_entry_any(svcs):
    """An authentication portal of any kind. Password spraying is not specific
    to one protocol, and tying it to a single service made it legal in exactly
    the same places as SSH brute force."""
    def f(m, t):
        h = m.host(t)
        if h.compromised or not m.topo.exposed(t):
            return False
        return any(m.topo.has_service(t, s) for s in svcs)
    return f


def _eff_entry(m, t):
    h = m.host(t)
    h.compromised = True
    h.discovered = True
    h.privilege = max(h.privilege, P_USER)
    return True


def _eff_entry_with_creds(m, t):
    _eff_entry(m, t)
    m.host(t).creds.add(C_PASSWORD)
    return True


# -- recon: run from a foothold ---------------------------------------------
def _pre_recon(m, t):
    return m.host(t).compromised


def _pre_recon_windows(m, t):
    return m.host(t).compromised and m.topo.is_windows(t)


def _pre_enumerate_host(m, t):
    """Host enumeration is worth doing once per host, and it gates escalating on
    that host. Previously this technique could be repeated forever for a reward,
    which is exactly what a trained policy learned to do."""
    return m.host(t).compromised and not m.host(t).enumerated


def _eff_enumerate_host(m, t):
    m.host(t).enumerated = True
    return True


def _pre_domain_enum(m, t):
    """Domain enumeration needs a Windows foothold and is done once."""
    return (m.host(t).compromised and m.topo.is_windows(t)
            and not m.domain_enumerated)


def _eff_scan(m, t):
    """Discover every host reachable from this foothold."""
    found = False
    for name in m.topo.hosts:
        if name != t and m.topo.reachable_from(t, name) and not m.host(name).discovered:
            m.host(name).discovered = True
            found = True
    m.scanned = True
    return found


def _eff_domain_enum(m, t):
    """Reveals the domain, and unlocks every Kerberos-based technique."""
    m.domain_enumerated = True
    m.scanned = True
    for name in m.topo.hosts:
        m.host(name).discovered = True
    return True


# -- credential access: each technique yields a specific kind of material ----
def _pre_creds_files(m, t):
    """Credentials left in configuration files. No elevation required: they are
    readable by the account already running."""
    h = m.host(t)
    return h.compromised and not h.holds(C_PASSWORD)


def _pre_creds_browser(m, t):
    """A browser credential store, which is per-user and Windows-specific here."""
    h = m.host(t)
    return h.compromised and m.topo.is_windows(t) and not h.holds(C_PASSWORD)


def _pre_creds_reuse(m, t):
    """Reusing a password already stolen elsewhere. Needing prior material is
    what separates this from harvesting."""
    h = m.host(t)
    if not h.compromised or h.holds(C_PASSWORD):
        return False
    return any(o.holds(C_PASSWORD) for o in m.hosts.values() if o.name != t)


def _grant(kind):
    def f(m, t):
        m.host(t).creds.add(kind)
        return True
    return f


def _gate_hostpriv(m, t):
    """Deny at execution when the engagement requires local elevation first.

    The action stays in the mask deliberately. Refusing it here returns PermErr
    (E-NASim Eq. 7), which the agent can learn from; removing it from the mask
    made the requirement invisible and some seeds never discovered it.
    """
    return m.need_hostpriv and m.host(t).privilege < P_ADMIN


def _pre_lsass(m, t):
    """Dumping LSASS needs local administrator, and yields the NTLM hash that
    pass-the-hash consumes."""
    h = m.host(t)
    return (h.compromised and m.topo.is_windows(t)
            and h.privilege >= P_ADMIN and not h.holds(C_NTLM))


def _eff_lsass(m, t):
    m.host(t).creds.update({C_NTLM, C_PASSWORD})
    return True


def _pre_asrep(m, t):
    """AS-REP roasting needs the domain enumerated. It needs no elevation, which
    is precisely why it is attractive."""
    h = m.host(t)
    return (h.compromised and m.topo.is_windows(t) and m.domain_enumerated
            and not h.holds(C_TICKET))


def _pre_kerberoast(m, t):
    """Kerberoasting needs the domain enumerated and a domain account to ask
    with, i.e. a password already in hand somewhere."""
    h = m.host(t)
    if not h.compromised or h.holds(C_TICKET) or not m.domain_enumerated:
        return False
    return any(o.holds(C_PASSWORD) for o in m.hosts.values())


def _pre_dcsync(m, t):
    """DCSync replicates the directory, so it needs replication rights: local
    administrator on a held Windows host, and the domain enumerated. It yields
    the krbtgt hash, which nothing else in the catalogue produces."""
    h = m.host(t)
    return (h.compromised and h.privilege >= P_ADMIN and m.domain_enumerated
            and m.topo.is_windows(t) and not m.krbtgt_hash)


def _eff_dcsync(m, t):
    m.krbtgt_hash = True
    return True


def _pre_relay(m, t):
    """Poison name resolution and relay the authentication, obtaining an NTLM
    hash for a host never touched."""
    h = m.host(t)
    if not h.discovered or h.holds(C_NTLM) or not m.topo.is_windows(t):
        return False
    return any(o.compromised and m.topo.reachable_from(o.name, t)
               for o in m.hosts.values() if o.name != t)


def _eff_relay(m, t):
    m.host(t).creds.add(C_NTLM)
    return True


# -- privilege escalation ----------------------------------------------------
def _pre_privesc(m, t):
    """Local escalation, and it needs the host enumerated first: an operator
    finds the vulnerable service or driver before abusing it."""
    h = m.host(t)
    return h.compromised and h.privilege < P_ADMIN and h.enumerated


def _pre_privesc_windows(m, t):
    return _pre_privesc(m, t) and m.topo.is_windows(t)


def _pre_privesc_linux(m, t):
    return _pre_privesc(m, t) and not m.topo.is_windows(t)


def _pre_token_theft(m, t):
    """Stealing a token needs a privileged session to steal from, so another
    host in the same zone must already be held at administrator."""
    h = m.host(t)
    if not (h.compromised and h.privilege < P_ADMIN and m.topo.is_windows(t)):
        return False
    return any(o.privilege >= P_ADMIN and o.name != t
               and m.topo.subnet_of[o.name] == m.topo.subnet_of[t]
               for o in m.hosts.values())


def _eff_privesc(m, t):
    m.host(t).privilege = P_ADMIN
    return True


def _pre_account_manip(m, t):
    """Creating a privileged domain account is a domain action and leaves
    persistence behind, which is what separates it from a local exploit."""
    h = m.host(t)
    if not h.compromised or not m.domain_enumerated or h.backdoored:
        return False
    return any(o.privilege >= P_ADMIN for o in m.hosts.values())


def _eff_account_manip(m, t):
    h = m.host(t)
    h.privilege = P_ADMIN
    h.backdoored = True
    return True


# -- the two routes to domain dominance that never touch the controller -----
def _pre_golden(m, t):
    """A golden ticket forges any identity in the domain, so it needs the krbtgt
    hash and nothing else. It is the pay-off of the DCSync chain."""
    return (t == m.topo.goal and m.krbtgt_hash
            and m.host(t).privilege < P_ADMIN)


def _pre_gpo(m, t):
    """Abusing group policy needs administrator inside the admin tier."""
    if t != m.topo.goal or m.host(t).privilege >= P_ADMIN:
        return False
    if not m.domain_enumerated:
        return False
    tier = max(m.topo.subnets) - 1
    return any(o.privilege >= P_ADMIN and m.topo.subnet_of[o.name] >= tier
               for o in m.hosts.values())


def _eff_dominate(m, t):
    h = m.host(t)
    h.discovered = True
    h.privilege = P_ADMIN
    return True


# -- lateral movement: each vector consumes a specific credential ------------
def _pre_move(kind, svc=None, windows=False, need_admin_source=False):
    def f(m, t):
        h = m.host(t)
        if h.compromised or not h.discovered:
            return False
        if windows and not m.topo.is_windows(t):
            return False
        if svc and not m.topo.has_service(t, svc):
            return False
        for name in m.topo.hosts:
            o = m.host(name)
            if not o.compromised or name == t:
                continue
            if kind is not None and not o.holds(kind):
                continue
            if need_admin_source and o.privilege < P_ADMIN:
                continue
            if m.topo.reachable_from(name, t):
                return True
        return False
    return f


def _pre_move_rdp(m, t):
    """An RDP session hijack takes either a password or a hash, but it does need
    the service to be there."""
    h = m.host(t)
    if h.compromised or not h.discovered or not m.topo.has_service(t, "rdp"):
        return False
    return any(o.compromised and o.name != t
               and (o.holds(C_PASSWORD) or o.holds(C_NTLM))
               and m.topo.reachable_from(o.name, t)
               for o in m.hosts.values())


def _eff_lateral(m, t):
    h = m.host(t)
    h.compromised = True
    h.privilege = max(h.privilege, P_USER)
    return True


# -- persistence -------------------------------------------------------------
def _pre_persist_service(m, t):
    """A system service needs administrator."""
    h = m.host(t)
    return h.compromised and h.privilege >= P_ADMIN and not h.backdoored


def _pre_persist_webshell(m, t):
    """A web shell needs a web server to drop it into."""
    h = m.host(t)
    return h.compromised and m.topo.has_service(t, "web") and not h.backdoored


def _pre_persist_task(m, t):
    """A scheduled task running as SYSTEM needs administrator."""
    h = m.host(t)
    return (h.compromised and m.topo.is_windows(t)
            and h.privilege >= P_ADMIN and not h.backdoored)


def _pre_persist_runkey(m, t):
    """A per-user Run key needs no elevation. That is the trade: quieter to
    obtain and available earlier, at a weaker foothold."""
    h = m.host(t)
    return h.compromised and m.topo.is_windows(t) and not h.backdoored


def _eff_persist(m, t):
    m.host(t).backdoored = True
    return True


# -- defence evasion ---------------------------------------------------------
def _pre_clear(m, t):
    return m.host(t).compromised and m.alert > CLEAR_FLOOR


def _pre_disable_def(m, t):
    h = m.host(t)
    return h.compromised and h.privilege >= P_ADMIN and not m.defences_disabled


def _eff_disable_def(m, t):
    """Tampering with the security stack dampens later detection.

    A real tradecraft choice rather than a free win: the act itself is among the
    loudest events in the catalogue, so an agent reaching for it early pays more
    than it saves.
    """
    m.defences_disabled = True
    return True


def _pre_audit_tamper(m, t):
    """Editing audit policy through the registry needs administrator on Windows."""
    h = m.host(t)
    return (h.compromised and m.topo.is_windows(t)
            and h.privilege >= P_ADMIN and not m.obfuscated)


def _pre_obfuscate(m, t):
    """Obfuscating tooling needs nothing but a foothold, on any platform."""
    return m.host(t).compromised and not m.obfuscated


def _eff_obfuscate(m, t):
    m.obfuscated = True
    return True


# -- command and control ----------------------------------------------------
def _pre_c2(m, t):
    return m.host(t).compromised and not m.c2_established


def _eff_c2(m, t):
    m.c2_established = True
    return True


# -- collection: staged, then packaged, and only then can it leave -----------
def _pre_stage(m, t):
    h = m.host(t)
    return h.compromised and not h.staged


def _eff_stage(m, t):
    m.host(t).staged = True
    return True


def _pre_archive(m, t):
    h = m.host(t)
    return h.staged and not h.archived


def _eff_archive(m, t):
    m.host(t).archived = True
    return True


# -- impact ------------------------------------------------------------------
def _pre_service_stop(m, t):
    h = m.host(t)
    return h.compromised and h.privilege >= P_ADMIN and not m.services_stopped


def _eff_service_stop(m, t):
    m.services_stopped = True
    return True


def _pre_inhibit_recovery(m, t):
    h = m.host(t)
    return (h.compromised and m.topo.is_windows(t)
            and h.privilege >= P_ADMIN and not m.recovery_inhibited)


def _eff_inhibit_recovery(m, t):
    m.recovery_inhibited = True
    return True


def _requirements_met(m):
    """The conditions this particular engagement additionally imposes."""
    if m.need_persist and not any(h.backdoored for h in m.hosts.values()):
        return False
    if m.need_collect and not m.collected:
        return False
    if m.need_c2 and not m.c2_established:
        return False
    return True


def _pre_ransomware(m, t):
    """Encryption is executed on the objective once it is fully held."""
    return (t == m.topo.goal and m.host(t).privilege >= P_ADMIN
            and _requirements_met(m))


def _pre_exfiltrate(m, t):
    """Data cannot leave that was never packaged, and cannot leave without a
    channel. This dependency is what makes the impact techniques four different
    actions rather than four names for one."""
    return (t == m.topo.goal and m.host(t).privilege >= P_ADMIN
            and any(h.archived for h in m.hosts.values())
            and m.c2_established and _requirements_met(m))


def _eff_impact(m, t):
    return True


# ---------------------------------------------------------------------------
# the registry.  40 techniques, matching the catalogue size E-NASim uses in its
# largest scenario.  The first 25 are the v4 catalogue re-scoped to a target
# host; the remainder extend coverage across discovery, persistence, defence
# evasion, collection and impact, which v4 under-represented.
# ---------------------------------------------------------------------------
def _t(name, mid, tactic, scope, prob, noise, pre, eff, fail=R_UNDEF,
       terminal=False, gate=None, group=None):
    return TechniqueV5(name, mid, tactic, scope, prob, noise, pre, eff, fail,
                       terminal, gate, group)


_TECHS_V5 = [
    # ---- initial access: each vector abuses a service the DMZ actually runs
    _t("PHISHING_EMAIL",          "T1566",     "Initial Access",       "entry",  0.90, 0.05,
       _pre_entry("smtp"),                 _eff_entry,            R_CONNERR,
       group="entry"),
    _t("SQL_INJECTION",           "T1190",     "Initial Access",       "entry",  0.85, 0.12,
       _pre_entry("web"),                  _eff_entry_with_creds, R_CONNERR,
       group="entry"),
    _t("BRUTE_FORCE_SSH",         "T1110",     "Credential Access",    "entry",  0.60, 0.50,
       _pre_entry("ssh"),                  _eff_entry,            R_CONNERR,
       group="entry"),
    _t("PASSWORD_SPRAYING",       "T1110.003", "Credential Access",    "entry",  0.55, 0.30,
       _pre_entry_any(("web", "vpn")),     _eff_entry_with_creds, R_CONNERR,
       group="entry"),
    # T1133 is re-entry with credentials already stolen, which is what the
    # technique describes and what makes it distinct from spraying the same portal
    _t("EXTERNAL_REMOTE_SERVICES","T1133",     "Initial Access",       "entry",  0.70, 0.15,
       _pre_entry("vpn", need_stolen_creds=True), _eff_entry,     R_CONNERR),

    # ---- discovery --------------------------------------------------------
    _t("NETWORK_SCAN",            "T1046",     "Discovery",            "recon",  0.95, 0.10,
       _pre_recon,                         _eff_scan,             R_PERMERR),
    _t("REMOTE_SYSTEM_DISCOVERY", "T1018",     "Discovery",            "recon",  0.95, 0.05,
       _pre_recon_windows,                 _eff_scan,             R_PERMERR),
    # enumerating a host is now a prerequisite for escalating on it, so this is
    # a step in the chain rather than a repeatable no-op
    _t("SYSTEM_INFO_DISCOVERY",   "T1082",     "Discovery",            "recon",  0.98, 0.03,
       _pre_enumerate_host,                _eff_enumerate_host,   R_PERMERR),
    # three substitutable routes to the same domain view. The per-episode
    # availability vector offers a subset, so the agent must use whichever it is
    # given -- that substitutability is the point, not padding
    _t("DOMAIN_ACCT_DISCOVERY",   "T1087.002", "Discovery",            "recon",  0.95, 0.05,
       _pre_domain_enum,                   _eff_domain_enum,      R_PERMERR,
       group="domain_enum"),
    _t("DOMAIN_TRUST_DISCOVERY",  "T1482",     "Discovery",            "recon",  0.95, 0.05,
       _pre_domain_enum,                   _eff_domain_enum,      R_PERMERR,
       group="domain_enum"),
    _t("DOMAIN_GROUP_DISCOVERY",  "T1069.002", "Discovery",            "recon",  0.95, 0.05,
       _pre_domain_enum,                   _eff_domain_enum,      R_PERMERR,
       group="domain_enum"),

    # ---- credential access: each yields a specific kind of material -------
    _t("CREDS_IN_FILES",          "T1552.001", "Credential Access",    "local",  0.75, 0.10,
       _pre_creds_files,                   _grant(C_PASSWORD),    R_PERMERR),
    _t("CREDENTIALS_FROM_BROWSER","T1555.003", "Credential Access",    "local",  0.70, 0.15,
       _pre_creds_browser,                 _grant(C_PASSWORD),    R_PERMERR,
       gate=_gate_hostpriv, group="password"),
    _t("VALID_ACCOUNTS_LOGIN",    "T1078",     "Defense Evasion",      "local",  0.95, 0.05,
       _pre_creds_reuse,                   _grant(C_PASSWORD),    R_PERMERR,
       gate=_gate_hostpriv, group="password"),
    _t("CRED_DUMP_LSASS",         "T1003.001", "Credential Access",    "local",  0.85, 0.20,
       _pre_lsass,                         _eff_lsass,            R_PERMERR),
    _t("AS_REP_ROASTING",         "T1558.004", "Credential Access",    "local",  0.80, 0.15,
       _pre_asrep,                         _grant(C_TICKET),      R_PERMERR,
       group="ticket"),
    _t("KERBEROASTING",           "T1558.003", "Credential Access",    "local",  0.75, 0.15,
       _pre_kerberoast,                    _grant(C_TICKET),      R_PERMERR,
       group="ticket"),
    _t("DCSYNC",                  "T1003.006", "Credential Access",    "local",  0.85, 0.15,
       _pre_dcsync,                        _eff_dcsync,           R_PERMERR),
    _t("NTLM_RELAY",              "T1557.001", "Credential Access",    "remote", 0.72, 0.28,
       _pre_relay,                         _eff_relay,            R_CONNERR),

    # ---- privilege escalation ---------------------------------------------
    _t("PRIV_ESC_SUDO",           "T1068",     "Privilege Escalation", "local",  0.70, 0.30,
       _pre_privesc_linux,                 _eff_privesc,          R_PERMERR),
    _t("POWERSHELL_EXEC",         "T1059.001", "Execution",            "local",  0.85, 0.25,
       _pre_privesc_windows,               _eff_privesc,          R_PERMERR),
    _t("PROCESS_INJECTION",       "T1055",     "Privilege Escalation", "local",  0.75, 0.30,
       _pre_token_theft,                   _eff_privesc,          R_PERMERR),
    _t("ACCOUNT_MANIPULATION",    "T1098",     "Privilege Escalation", "local",  0.75, 0.25,
       _pre_account_manip,                 _eff_account_manip,    R_PERMERR),
    # two routes to domain dominance that never touch the controller
    _t("GOLDEN_TICKET",           "T1558.001", "Privilege Escalation", "impact", 0.85, 0.25,
       _pre_golden,                        _eff_dominate,         R_PERMERR,
       group="dominance"),
    _t("GPO_MODIFICATION",        "T1484.001", "Privilege Escalation", "impact", 0.80, 0.20,
       _pre_gpo,                           _eff_dominate,         R_PERMERR,
       group="dominance"),

    # ---- lateral movement: each consumes different stolen material --------
    _t("LATERAL_MOVE_SMB",        "T1021",     "Lateral Movement",     "remote", 0.85, 0.15,
       _pre_move(C_PASSWORD, svc="smb"),   _eff_lateral,          R_CONNERR),
    _t("PASS_THE_HASH",           "T1550.002", "Lateral Movement",     "remote", 0.80, 0.20,
       _pre_move(C_NTLM, windows=True),    _eff_lateral,          R_CONNERR),
    _t("PASS_THE_TICKET",         "T1550.003", "Lateral Movement",     "remote", 0.80, 0.20,
       _pre_move(C_TICKET, windows=True),  _eff_lateral,          R_CONNERR),
    _t("RDP_HIJACK",              "T1021.001", "Lateral Movement",     "remote", 0.78, 0.22,
       _pre_move_rdp,                      _eff_lateral,          R_CONNERR),
    _t("WMI_EXECUTION",           "T1047",     "Execution",            "remote", 0.82, 0.22,
       _pre_move(C_PASSWORD, windows=True, need_admin_source=True),
       _eff_lateral,                                              R_CONNERR),

    # ---- persistence: the trade is elevation required against noise --------
    _t("INSTALL_BACKDOOR",        "T1543",     "Persistence",          "local",  0.80, 0.15,
       _pre_persist_service,               _eff_persist,          R_PERMERR,
       group="persist"),
    _t("WEB_SHELL_UPLOAD",        "T1505.003", "Persistence",          "local",  0.80, 0.20,
       _pre_persist_webshell,              _eff_persist,          R_PERMERR,
       group="persist"),
    _t("SCHEDULED_TASK",          "T1053.005", "Persistence",          "local",  0.85, 0.18,
       _pre_persist_task,                  _eff_persist,          R_PERMERR,
       group="persist"),
    _t("REGISTRY_RUN_KEYS",       "T1547.001", "Persistence",          "local",  0.88, 0.15,
       _pre_persist_runkey,                _eff_persist,          R_PERMERR,
       group="persist"),

    # ---- defence evasion ---------------------------------------------------
    _t("CLEAR_LOGS",              "T1070",     "Defense Evasion",      "local",  0.90, 0.05,
       _pre_clear,                         lambda m, t: True,     R_PERMERR),
    _t("DISABLE_SECURITY_TOOLS",  "T1562.001", "Defense Evasion",      "local",  0.70, 0.45,
       _pre_disable_def,                   _eff_disable_def,      R_PERMERR),
    _t("MODIFY_REGISTRY",         "T1112",     "Defense Evasion",      "local",  0.85, 0.20,
       _pre_audit_tamper,                  _eff_obfuscate,        R_PERMERR,
       group="quiet"),
    _t("OBFUSCATED_FILES",        "T1027",     "Defense Evasion",      "local",  0.90, 0.10,
       _pre_obfuscate,                     _eff_obfuscate,        R_PERMERR,
       group="quiet"),

    # ---- command and control ----------------------------------------------
    _t("C2_CHANNEL_ESTABLISH",    "T1071.001", "Command and Control",  "local",  0.88, 0.20,
       _pre_c2,                            _eff_c2,               R_PERMERR),

    # ---- collection: stage it, package it, and only then can it leave -----
    _t("DATA_FROM_LOCAL_SYSTEM",  "T1005",     "Collection",           "local",  0.90, 0.10,
       _pre_stage,                         _eff_stage,            R_PERMERR),
    _t("ARCHIVE_COLLECTED_DATA",  "T1560.001", "Collection",           "local",  0.92, 0.12,
       _pre_archive,                       _eff_archive,          R_PERMERR),

    # ---- impact: two preparatory actions, two that end the engagement ------
    _t("SERVICE_STOP",            "T1489",     "Impact",               "local",  0.90, 0.30,
       _pre_service_stop,                  _eff_service_stop,     R_PERMERR),
    _t("INHIBIT_SYSTEM_RECOVERY", "T1490",     "Impact",               "local",  0.88, 0.40,
       _pre_inhibit_recovery,              _eff_inhibit_recovery, R_PERMERR),
    _t("RANSOMWARE_ENCRYPT",      "T1486",     "Impact",               "impact", 0.90, 0.35,
       _pre_ransomware,                    _eff_impact,           R_PERMERR, terminal=True),
    _t("EXFILTRATE_DATA",         "T1041",     "Exfiltration",         "impact", 1.00, 0.25,
       _pre_exfiltrate,                    _eff_impact,           R_PERMERR, terminal=True),
]

TECHNIQUES_V5 = {t.name: t for t in _TECHS_V5}
TECHNIQUE_ORDER_V5 = [t.name for t in _TECHS_V5]

# tell the detection engine the ATT&CK id of every technique it will be asked
# to log, so emitted events stay correctly labelled for the released corpus
_register_mitre({n: t.mitre_id for n, t in TECHNIQUES_V5.items()})


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class KillChainModelV5:
    """The mutable world: a multi-subnet estate and the attacker progress in it.

    An action is the pair (technique index, host index), flattened to a single
    integer so it presents as a Discrete space to any SB3 algorithm.
    """

    def __init__(self, topology="enterprise", rng=None, exclude_tactics=(),
                 exclude_techniques=()):
        self.topo = Topology(topology)
        self.rng = rng or random
        self.detection = DetectionEngine()
        # Tactics and techniques this engagement will not offer. Both empty in
        # every configuration reported outside the ablation of Section 4.12.
        self.exclude_tactics = frozenset(exclude_tactics or ())
        self.exclude_techniques = frozenset(exclude_techniques or ())
        # Mission definition. Everything in the paper uses "admin"; see is_goal.
        self.objective = "admin"
        self.hosts = {
            n: HostV5(name=n, subnet=self.topo.subnet_of[n],
                      asset_value=3.0 if n == self.topo.goal else 1.0)
            for n in self.topo.hosts
        }
        self.n_tech = len(TECHNIQUE_ORDER_V5)
        self.n_host = len(self.topo.hosts)
        self.n_actions = self.n_tech * self.n_host
        self.scanned = False
        self.need_persist = False
        self.need_hostpriv = False
        self.need_collect = False
        self.need_c2 = False
        self.domain_enumerated = False
        self.krbtgt_hash = False
        self.services_stopped = False
        self.recovery_inhibited = False
        self.defences_disabled = False
        self.c2_established = False
        self.obfuscated = False
        self.last_result = R_SUCCESS
        self.stuck = False
        self.reset()

    # -- plumbing -----------------------------------------------------------
    def host(self, name):
        return self.hosts[name]

    @property
    def alert(self):
        return self.detection.suspicion

    @alert.setter
    def alert(self, v):
        self.detection.suspicion = max(0.0, float(v))

    def decode(self, action):
        """flat action index -> (technique name, host name)"""
        return TECHNIQUE_ORDER_V5[action // self.n_host], self.topo.hosts[action % self.n_host]

    def encode(self, technique, host):
        return TECHNIQUE_ORDER_V5.index(technique) * self.n_host + self.topo.hosts.index(host)

    def reset(self):
        for h in self.hosts.values():
            h.reset()
        self.detection.reset()
        self.scanned = False
        self.domain_enumerated = False
        self.krbtgt_hash = False
        self.services_stopped = False
        self.recovery_inhibited = False
        self.defences_disabled = False
        self.c2_established = False
        self.obfuscated = False
        self.last_result = R_SUCCESS
        self.stuck = False
        # every exposed host starts discovered -- an attacker can see the edge
        for n in self.topo.hosts:
            if self.topo.exposed(n):
                self.hosts[n].discovered = True
        self.available = self._draw_availability()
        # per-episode objective requirements, as in v4
        self.need_persist = self.rng.random() < 0.5
        self.need_hostpriv = self.rng.random() < 0.5
        self.need_collect = self.rng.random() < 0.5
        self.need_c2 = self.rng.random() < 0.5

    @property
    def collected(self):
        """The engagement's collection requirement is met by packaged data, not
        by a single flag. Staging and archiving are separate techniques with a
        real dependency between them, so this asks whether the chain completed."""
        return any(h.archived for h in self.hosts.values())

    # -- goal / status ------------------------------------------------------
    def is_goal(self):
        """Has the engagement's mission been met?

        "admin" is the mission everywhere in the paper. "archived" exists only
        for the sim-to-real arm of Section 4.8, where the live target is a single
        web host on which the domain-admin chain is unreachable by construction.
        """
        if self.objective == "archived":
            return any(h.archived for h in self.hosts.values())
        return self.host(self.topo.goal).privilege >= P_ADMIN

    def caught(self):
        return self.detection.suspicion >= INCIDENT_THRESHOLD

    # -- the mask -----------------------------------------------------------
    def _draw_availability(self):
        """Which techniques this engagement offers.

        Every ungrouped technique is always available. Within a substitution
        group a random non-empty subset is offered, so an agent cannot rely on
        one preferred route being present. Drawing a non-empty subset matters:
        an episode that offered no way to enumerate the domain would be
        unwinnable for reasons that have nothing to do with the policy.
        """
        # NETWORK_SCAN and CREDS_IN_FILES are deliberately ungrouped. They are
        # the only discovery and credential primitives that work on any
        # platform, and withdrawing them stranded the agent: with a Linux
        # foothold in the DMZ and the Windows-only variants as the survivors,
        # nothing could be discovered and no password could be obtained. That
        # made 50 of 150 engagements unwinnable for reasons unrelated to the
        # policy, which would have silently capped every success rate reported.
        avail = set()
        groups = {}
        for name in TECHNIQUE_ORDER_V5:
            g = TECHNIQUES_V5[name].group
            if g is None:
                avail.add(name)
            else:
                groups.setdefault(g, []).append(name)
        for members in groups.values():
            # Only draw among members that could actually be used in this
            # topology. An entry technique whose service no exposed host runs is
            # not a substitute for one that works, and treating it as one left
            # the three-host network with 19 of 100 engagements unwinnable --
            # every entry vector withdrawn except ones needing a service that
            # network does not have.
            viable = [n for n in members if self._usable_opener(n)]
            pool = viable or members
            k = self.rng.randint(1, len(pool))
            avail.update(self.rng.sample(pool, k))
        if self.exclude_tactics:
            # Used only by the ablation in Section 4.12. Empty everywhere else,
            # so no result outside that experiment is affected by its presence.
            avail = {n for n in avail
                     if TECHNIQUES_V5[n].tactic not in self.exclude_tactics}
        if self.exclude_techniques:
            avail = {n for n in avail if n not in self.exclude_techniques}
        if self.exclude_tactics or self.exclude_techniques:
            # Exclusion runs after the substitution-group draw, so a draw that
            # picked only excluded members of a group loses all of them. That has
            # never stranded an episode in practice, but it is not structurally
            # prevented, and an engagement with no way in is unwinnable for
            # reasons unrelated to the policy -- the exact fault Section 3.10
            # exists to catch. Fail loudly rather than silently cap a result.
            if not any(TECHNIQUES_V5[n].scope == "entry" for n in avail):
                raise RuntimeError(
                    "exclusion left this engagement with no entry technique; "
                    "the withdrawn set removes every way into the estate")
        return avail

    def _usable_opener(self, name):
        """For entry techniques, whether some exposed host admits it right now.

        Evaluated on the freshly reset state, before anything is compromised, so
        it answers "could this open the engagement?" rather than "is it legal
        later". Non-entry techniques are not filtered.
        """
        tech = TECHNIQUES_V5[name]
        if tech.scope != "entry":
            return True
        for h in self.topo.hosts:
            if self.topo.exposed(h):
                try:
                    if tech.precondition(self, h):
                        return True
                except Exception:
                    pass
        return False

    def offers(self, technique):
        return technique in self.available

    def action_mask_raw(self):
        """Boolean vector over every (technique, host) pair.

        This is the object E-NASim declines to compute -- they note that a
        uniform H x T space avoids "the need for per-host action masking" and
        enforce feasibility only inside the transition function. Computing it
        costs one precondition evaluation per pair and is what keeps an emitted
        trace executable.
        """
        mask = [False] * self.n_actions
        for ti, tname in enumerate(TECHNIQUE_ORDER_V5):
            tech = TECHNIQUES_V5[tname]
            base = ti * self.n_host
            if tname not in self.available:
                continue          # withdrawn for this engagement
            for hi, hname in enumerate(self.topo.hosts):
                try:
                    mask[base + hi] = bool(tech.precondition(self, hname))
                except Exception:
                    mask[base + hi] = False
        return mask

    def action_mask(self):
        """The mask handed to the policy.

        Stable-baselines3 cannot sample from an all-false mask, so when nothing
        is legal one bit is set and `stuck` is raised. The environment ends the
        episode on that flag rather than letting the agent take an action the
        preconditions forbid -- fabricating a legal action here would put a
        genuine infeasible selection into the very number the paper reports as
        zero. It is rare but it does happen: on the three-host network the mean
        legal set is nine actions of 135.
        """
        mask = self.action_mask_raw()
        self.stuck = not any(mask)
        if self.stuck:
            mask[0] = True
        return mask

    def legal(self, technique, host):
        return bool(TECHNIQUES_V5[technique].precondition(self, host))

    # -- the transition -----------------------------------------------------
    def _outcome(self, tech, technique, target):
        """Did the attempt succeed? Overridden by the real-execution arm."""
        return self.rng.random() <= tech.success_prob

    def attempt(self, technique, target):
        """Try `technique` against `target`; return an OutcomeV5.

        Mirrors v4's transition semantics, with one addition taken from
        E-NASim: when the precondition fails the world does not change and a
        *typed* failure signal goes back to the agent (their Eq. 7) rather than
        a bare boolean. That keeps an unmasked baseline here as well informed as
        it is in their environment, so the masking comparison cannot be
        dismissed as an artefact of starving the baseline of feedback.
        """
        tech = TECHNIQUES_V5[technique]
        out = OutcomeV5()

        # 1. feasibility. Unmasked agents reach here often; masked ones never.
        if not tech.precondition(self, target):
            out.blocked = True
            out.result_code = tech.fail_code
            self.last_result = tech.fail_code
            return out

        # 2. runtime denial. The action was feasible and was attempted; the
        #    system refused it. This is reported as a typed failure so the agent
        #    can learn the requirement, which it cannot do when the action is
        #    merely absent from the mask.
        if tech.runtime_gate is not None and tech.runtime_gate(self, target):
            out.result_code = R_PERMERR
            self.last_result = R_PERMERR
            det = self.detection.observe(
                technique, target, False,
                has_backdoor=any(h.backdoored for h in self.hosts.values()),
                rng=self.rng, host_os=self.topo.os_of(target), denied=True)
            out.detected = det.rule_fired
            out.terminated = det.caught
            return out

        # 3. the outcome. `_outcome` is the only seam between this world
        #    model and one that executes for real: the simulated arm draws
        #    against the assumed probability, the sim-to-real arm of
        #    analysis/v5_sim_to_real.py runs the technique against a live
        #    target and reports what actually happened. Everything else on
        #    this path -- preconditions, runtime gates, detection, effects --
        #    is shared, so a difference between the two arms is a difference
        #    in execution and not in bookkeeping.
        succeeded = self._outcome(tech, technique, target)

        # 3. the blue team observes the attempt either way -- a failed loud
        #    technique (failed logons) is often the louder signal
        det = self.detection.observe(
            technique, target, succeeded,
            has_backdoor=any(h.backdoored for h in self.hosts.values()),
            rng=self.rng, host_os=self.topo.os_of(target))

        if self.defences_disabled:
            self.detection.suspicion = max(0.0, self.detection.suspicion * 0.85)
        if self.obfuscated:
            self.detection.suspicion = max(0.0, self.detection.suspicion * 0.93)

        # 4. escalation to incident response ends the intrusion
        if det.caught:
            out.detected = out.terminated = True
            out.result_code = R_UNDEF
            self.last_result = R_UNDEF
            if succeeded and technique != "CLEAR_LOGS":
                out.success = True
                out.advanced = bool(tech.effect(self, target))
            return out

        out.detected = det.rule_fired

        if not succeeded:
            out.result_code = R_UNDEF
            self.last_result = R_UNDEF
            return out

        out.success = True
        out.result_code = R_SUCCESS
        self.last_result = R_SUCCESS

        if technique == "CLEAR_LOGS":
            self.detection.suspicion = max(0.0, self.detection.suspicion - CLEAR_REDUCTION)
            out.advanced = True
            return out

        out.advanced = bool(tech.effect(self, target))
        if tech.terminal and out.advanced:
            out.terminated = True
        return out
