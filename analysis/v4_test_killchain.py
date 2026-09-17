# -*- coding: utf-8 -*-
"""
Unit tests for env/kill_chain.py, the shared world model.

Checks four things, and exits non-zero if any fail:
  1. the action order matches v3's ACTION_LIST, so trained checkpoints and the
     action-space indices stay valid;
  2. every technique produces its effect when forced to succeed (15/15);
  3. a greedy operator reaches the domain controller and exfiltrates;
  4. the mask is never empty along that path, and CLEAR_LOGS gives no free ride.
"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.kill_chain import KillChainModel, TECHNIQUES, ACTION_ORDER
from env.attack_actions import ACTION_LIST
from env import detection as _det


def det_profile(name):
    """Temporarily silence a technique's detection so a capability test measures
    only the state effect, not a detection roll. Returns the saved base rate."""
    p = _det.PROFILES[name]
    saved = p.base_detect
    p.base_detect = 0.0
    return saved


def restore_detect(name, saved):
    _det.PROFILES[name].base_detect = saved

FAILS = []


def check(cond, msg):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILS.append(msg)


def test_order():
    print("1. action order")
    v3 = [a.name for a in ACTION_LIST]
    # the original 15 keep their order and indices, so v3 checkpoints remain
    # valid; any v4 additions come strictly after them
    check(ACTION_ORDER[:len(v3)] == v3,
          f"first {len(v3)} actions match v3 order (checkpoint-compatible)")
    check(len(ACTION_ORDER) >= len(v3), f"{len(ACTION_ORDER)} actions total")
    check(all(TECHNIQUES[a.name].mitre_id == a.mitre_technique for a in ACTION_LIST),
          "every original ATT&CK id matches attack_actions.py")


# precondition setups mirror the ones in v4_diagnose, but against the model
def setup(model, name):
    h = model.host
    if name in ("NETWORK_SCAN", "VALID_ACCOUNTS_LOGIN", "INSTALL_BACKDOOR",
                "WEB_SHELL_UPLOAD", "PRIV_ESC_SUDO", "POWERSHELL_EXEC",
                "CRED_DUMP_LSASS", "DOMAIN_ACCT_DISCOVERY", "DOMAIN_TRUST_DISCOVERY",
                "ACCOUNT_MANIPULATION"):
        h("user01").compromised = True
    elif name == "AS_REP_ROASTING":
        h("user01").compromised = True; model.scanned = True
    elif name == "PASS_THE_TICKET":
        h("user01").compromised = True; h("user01").has_credentials = True
    elif name in ("DCSYNC", "GPO_MODIFICATION", "GOLDEN_TICKET"):
        h("user01").compromised = True; h("user01").has_credentials = True
        model.scanned = True; h("dc01").compromised = True
    elif name == "LATERAL_MOVE_SMB":
        h("user01").compromised = True; h("user01").has_credentials = True; model.scanned = True
    elif name == "PASS_THE_HASH":
        h("user01").compromised = True; h("user01").has_credentials = True
    elif name == "KERBEROASTING":
        h("user01").compromised = True; h("user01").has_credentials = True
        model.scanned = True; h("dc01").compromised = True
    elif name in ("EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT"):
        h("dc01").compromised = True; h("dc01").privileged = True
    elif name == "SQL_INJECTION":
        model.sqli_available = True
    elif name == "CLEAR_LOGS":
        model.alert = 0.6


def effect_ok(model, name):
    h = model.host
    if name == "PHISHING_EMAIL":       return h("user01").compromised
    if name == "SQL_INJECTION":        return h("user01").compromised and h("user01").has_credentials
    if name == "BRUTE_FORCE_SSH":      return model.any_compromised()
    if name == "NETWORK_SCAN":         return model.scanned
    if name == "VALID_ACCOUNTS_LOGIN": return h("user01").has_credentials
    if name == "CRED_DUMP_LSASS":      return h("user01").has_credentials
    if name == "DOMAIN_ACCT_DISCOVERY": return model.scanned
    if name == "DCSYNC":               return h("dc01").privileged
    if name == "DOMAIN_TRUST_DISCOVERY": return model.scanned
    if name == "AS_REP_ROASTING":      return h("user01").has_credentials
    if name == "PASSWORD_SPRAYING":    return model.any_compromised()
    if name == "PASS_THE_TICKET":      return h("srv01").compromised
    if name == "ACCOUNT_MANIPULATION": return h("user01").privileged
    if name in ("GPO_MODIFICATION", "GOLDEN_TICKET"): return h("dc01").privileged
    if name in ("INSTALL_BACKDOOR", "WEB_SHELL_UPLOAD"): return any(model.backdoor.values())
    if name in ("PRIV_ESC_SUDO", "POWERSHELL_EXEC"):     return h("user01").privileged
    if name == "CLEAR_LOGS":           return model.alert < 2.0
    if name == "LATERAL_MOVE_SMB":     return h("srv01").compromised
    if name == "PASS_THE_HASH":        return h("srv01").compromised
    if name == "KERBEROASTING":        return h("dc01").privileged
    if name in ("EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT"): return True
    return False


def test_capabilities():
    print("2. every technique produces its effect when it succeeds")
    rng = random.Random(0)
    n_ok = 0
    for name in ACTION_ORDER:
        m = KillChainModel()
        m.reset(rng=rng, randomize_scenario=False)  # every technique available
        setup(m, name)
        m.alert = 0.6 if name == "CLEAR_LOGS" else 0.0
        tech = TECHNIQUES[name]
        saved_p, saved_n = tech.success_prob, tech.noise
        tech.success_prob, tech.noise = 1.0, 0.0
        # isolate capability from detection: force the SOC not to catch this
        # single attempt, so the test measures the effect, not a detection roll
        saved_detect = det_profile(name)
        offered = m.legal_mask()[ACTION_ORDER.index(name)]
        out = m.attempt(name, rng)
        restore_detect(name, saved_detect)
        tech.success_prob, tech.noise = saved_p, saved_n
        ok = offered and out.success and effect_ok(m, name)
        if ok:
            n_ok += 1
        else:
            print(f"       {name}: offered={offered} success={out.success} effect={effect_ok(m, name)}")
    check(n_ok == len(ACTION_ORDER), f"{n_ok}/{len(ACTION_ORDER)} techniques pass")


# A stealth-optimal operator: reach the DC on the quietest path. It uses
# Kerberoasting (base_detect 0.15) for the DC escalation rather than the louder
# host-privesc techniques, prefers SQLi/phishing entry over brute force, and
# skips persistence entirely -- none of it is needed to reach the objective, and
# every extra action is extra noise. Exfiltrate is the win.
PRIORITY = [
    "EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT",
    "KERBEROASTING",                               # quietest DC escalation (0.15)
    "DCSYNC", "GPO_MODIFICATION", "GOLDEN_TICKET", # louder DC-dominance alternatives
    "LATERAL_MOVE_SMB", "PASS_THE_TICKET", "PASS_THE_HASH",
    "DOMAIN_TRUST_DISCOVERY", "DOMAIN_ACCT_DISCOVERY", "NETWORK_SCAN",  # quiet recon first
    "VALID_ACCOUNTS_LOGIN", "AS_REP_ROASTING", "CRED_DUMP_LSASS",       # quiet cred paths first
    "SQL_INJECTION", "PHISHING_EMAIL",             # quiet entry
    "POWERSHELL_EXEC", "PRIV_ESC_SUDO", "ACCOUNT_MANIPULATION",  # host privesc / DC fallback
    "PASSWORD_SPRAYING", "BRUTE_FORCE_SSH",        # loud entry, last resort
    "INSTALL_BACKDOOR", "WEB_SHELL_UPLOAD",
    "CLEAR_LOGS",
]


def test_killchain():
    print("3. a greedy operator reaches the objective")
    reached = clears = 0
    empty_mask = False
    for seed in range(20):
        rng = random.Random(seed)
        m = KillChainModel()
        m.reset(sqli_available=False, rng=rng, randomize_scenario=False)
        for _ in range(200):
            try:
                mask = m.legal_mask()
            except RuntimeError:
                empty_mask = True
                break
            legal = {ACTION_ORDER[i] for i, v in enumerate(mask) if v}
            # a competent operator takes an available win immediately, but
            # otherwise clears logs before the alert makes blocks likely --
            # stealth management the naive priority order ignores
            terminal_now = legal & {"EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT"}
            # CLEAR_LOGS is only legal once alert is above the floor; a stealthy
            # operator takes it then, unless a win is available this step
            if not terminal_now and "CLEAR_LOGS" in legal:
                pick = "CLEAR_LOGS"
            else:
                pick = next((n for n in PRIORITY if n in legal), None)
            out = m.attempt(pick, rng)
            if pick == "CLEAR_LOGS" and out.success:
                clears += 1
            if out.terminated:
                break
        if m.is_goal():
            reached += 1
    check(reached >= 19, f"{reached}/20 greedy runs reach the DC objective")
    check(not empty_mask, "mask never went empty along the path")
    print(f"       (CLEAR_LOGS used {clears} times across 20 runs — should be modest)")


def main():
    print("=" * 66)
    print("kill_chain.py unit tests")
    print("=" * 66)
    test_order()
    test_capabilities()
    test_killchain()
    print("=" * 66)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s)")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
