# -*- coding: utf-8 -*-
"""Produce worked examples of successful and failed attack chains.

For each episode this records what the scenario offered, which route the agent
chose, which offered routes it declined, which techniques the scenario withheld,
and -- for a failure -- exactly where and why it went wrong. Written for the
supervisor's request to see a successful chain and a failed one side by side.

    python analysis/v4_attack_chains.py
"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO

from env.adversary_env import AdversaryEnv
from env.kill_chain import ACTION_ORDER, TECHNIQUES, KillChainModel
from env.detection import PROFILES, INCIDENT_THRESHOLD

MODEL = "results/models/v4/dqn_s0.zip"      # replaced below
OUT = "results/attack_chain_examples.md"
PHASE_OF = {t: g for g, techs in KillChainModel.AVAIL_GROUPS.items() for t in techs}
PHASE_LABEL = {
    "entry": "Initial access", "discovery": "Discovery", "cred": "Credential access",
    "lateral": "Lateral movement", "dc": "Domain dominance", "persist": "Persistence",
    "hostpriv": "Host privilege escalation", "impact": "Impact",
}


def run_episode(model, env, seed):
    """Roll one episode, capturing a full step-by-step trace."""
    obs, _ = env.reset(seed=seed)
    m = env.model
    scenario = {
        "avail": dict(m.avail),
        "need_persist": m.need_persist,
        "need_hostpriv": m.need_hostpriv,
    }
    trace = []
    outcome = "step budget exhausted"
    for step in range(1, env.max_steps + 1):
        mask = env.action_masks()
        act, _ = model.predict(obs, action_masks=mask, deterministic=True)
        name = ACTION_ORDER[int(act)]
        before = m.alert
        obs, reward, term, trunc, info = env.step(int(act))
        prof = PROFILES[name]
        trace.append({
            "step": step, "technique": name, "mitre": TECHNIQUES[name].mitre_id,
            "phase": PHASE_LABEL.get(PHASE_OF.get(name), "Defence evasion"),
            "event": "%s %d" % (prof.emits[0][1], prof.emits[0][0]),
            "rule_rate": prof.base_detect,
            "suspicion": m.alert, "delta": m.alert - before,
            "reward": reward,
        })
        if term or trunc:
            if info.get("success"):
                outcome = "objective reached"
            elif info.get("caught"):
                outcome = "caught by incident response"
            break
    return scenario, trace, outcome


def render(fh, title, scenario, trace, outcome, note):
    used = [t["technique"] for t in trace]
    offered = [n for n, v in scenario["avail"].items() if v]
    withheld = [n for n, v in scenario["avail"].items() if not v]

    fh.write("## %s\n\n" % title)
    fh.write("**Outcome: %s** after %d steps.  %s\n\n" % (outcome, len(trace), note))

    fh.write("### The scenario the agent was given\n\n")
    fh.write("| Phase | Offered this episode | Withheld this episode |\n|---|---|---|\n")
    for g, techs in KillChainModel.AVAIL_GROUPS.items():
        on = [t for t in techs if scenario["avail"].get(t)]
        off = [t for t in techs if not scenario["avail"].get(t)]
        fh.write("| %s | %s | %s |\n" % (PHASE_LABEL[g],
                                         ", ".join(on) or "--",
                                         ", ".join(off) or "--"))
    fh.write("\nObjective requirements: persistence %s, host elevation %s.\n\n" % (
        "REQUIRED" if scenario["need_persist"] else "not required",
        "REQUIRED" if scenario["need_hostpriv"] else "not required"))

    fh.write("### The chain the agent executed\n\n")
    fh.write("| # | Phase | Technique | ATT&CK | Event emitted | Rule rate | Suspicion | Reward |\n")
    fh.write("|---|---|---|---|---|---|---|---|\n")
    for t in trace:
        fh.write("| %d | %s | %s | %s | %s | %.2f | %.2f%s | %+.0f |\n" % (
            t["step"], t["phase"], t["technique"], t["mitre"], t["event"],
            t["rule_rate"], t["suspicion"],
            " **+%.2f**" % t["delta"] if t["delta"] > 0.1 else "", t["reward"]))
    fh.write("\n")

    declined = [n for n in offered if n not in used]
    fh.write("**Offered but declined:** %s\n\n" % (", ".join(declined) or "none"))
    fh.write("**Withheld by the scenario (never legal here):** %s\n\n" % (", ".join(withheld) or "none"))
    fh.write("**Suspicion at the end:** %.2f of the %.1f incident threshold.\n\n"
             % (trace[-1]["suspicion"], INCIDENT_THRESHOLD))

    # What to read from this episode, derived from the trace itself.
    fh.write("### How to read this episode\n\n")
    fired = [t for t in trace if t["delta"] > 0.1]
    dc = [t for t in trace if t["phase"] == "Domain dominance"]
    entry = trace[0]
    alts = [n for n in KillChainModel.AVAIL_GROUPS["entry"]
            if scenario["avail"].get(n) and n != entry["technique"]]
    bits = ["The scenario offered %d of the 25 techniques. The agent entered through %s, "
            "declining %s, and reached domain dominance via %s."
            % (len(offered), entry["technique"], ", ".join(alts) or "no alternative",
               dc[0]["technique"] if dc else "no domain-dominance technique")]

    if scenario["need_persist"] or scenario["need_hostpriv"]:
        req = ([ "persistence" ] if scenario["need_persist"] else []) + \
              ([ "host elevation" ] if scenario["need_hostpriv"] else [])
        done = sorted({t["technique"] for t in trace
                       if t["phase"] in ("Persistence", "Host privilege escalation")})
        bits.append("The objective additionally required %s, which the agent satisfied with %s."
                    % (" and ".join(req), ", ".join(done) or "nothing"))

    if fired:
        bits.append("Detection rules fired %d time(s): %s."
                    % (len(fired), "; ".join(
                        "step %d, %s (rate %.2f, +%.2f suspicion)"
                        % (t["step"], t["technique"], t["rule_rate"], t["delta"])
                        for t in fired)))

    if outcome.startswith("caught"):
        last = fired[-1] if fired else trace[-1]
        bits.append("**The run ended because %s at step %d took suspicion to %.2f, at or above "
                    "the %.1f threshold.**" % (last["technique"], last["step"],
                                               last["suspicion"], INCIDENT_THRESHOLD))
        if last["technique"] == "CLEAR_LOGS":
            bits.append("Note what crossed the line: the agent's own attempt to reduce exposure. "
                        "Clearing the audit log raises Event 1102, the second-loudest rule in the "
                        "catalogue, so evasion is a gamble that can cost more than it saves. In the "
                        "initial environment this action was free and reset the alert to zero, and "
                        "the agent exploited it; here it can and does backfire.")
    else:
        clears = [t for t in trace if t["technique"] == "CLEAR_LOGS"]
        if clears:
            bits.append("The agent cleared logs %d time(s), each after suspicion rose above the "
                        "0.25 floor at which clearing becomes legal. A successful clear removes a "
                        "fixed 0.5 of suspicion, so it is a partial remedy, not the unconditional "
                        "reset the initial environment allowed -- and it raises Event 1102 in "
                        "doing so." % len(clears))
        bits.append("Suspicion never reached the threshold, so the intrusion completed quietly.")

    for b in bits:
        fh.write("- %s\n" % b)
    fh.write("\n---\n\n")


def main():
    model_path = "results/models/v4/masked_s0.zip"
    model = MaskablePPO.load(model_path, device="cpu")
    env = AdversaryEnv()
    class _Null:
        def __getattr__(self, _):
            return lambda *a, **k: None
    env.telemetry = _Null()

    wins, losses = [], []
    for seed in range(400):
        sc, tr, oc = run_episode(model, env, seed)
        if oc == "objective reached" and len(wins) < 2:
            wins.append((seed, sc, tr, oc))
        elif oc != "objective reached" and len(losses) < 2:
            losses.append((seed, sc, tr, oc))
        if len(wins) >= 2 and len(losses) >= 2:
            break

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("# Worked examples: successful and failed attack chains\n\n")
        fh.write("Episodes produced by the released masked agent (`%s`) in the refined "
                 "environment, chosen as the first successes and first failures over "
                 "seeds 0-399 so they are representative rather than selected.\n\n"
                 "Each episode draws a scenario: a random non-empty subset of the "
                 "techniques at every phase, plus objective requirements. The agent sees "
                 "what is offered and must pick a route through it.\n\n---\n\n" % model_path)
        for i, (seed, sc, tr, oc) in enumerate(wins, 1):
            render(fh, "Successful chain %d (episode seed %d)" % (i, seed), sc, tr, oc,
                   "The agent reached privileged access on the domain controller and "
                   "executed an impact action without the blue team escalating.")
        for i, (seed, sc, tr, oc) in enumerate(losses, 1):
            render(fh, "Failed chain %d (episode seed %d)" % (i, seed), sc, tr, oc,
                   "The blue team escalated to incident response before the objective "
                   "was reached." if oc.startswith("caught") else
                   "The step budget ran out before the objective was reached.")
    print("wrote", OUT, "| wins:", len(wins), "losses:", len(losses))


if __name__ == "__main__":
    main()
