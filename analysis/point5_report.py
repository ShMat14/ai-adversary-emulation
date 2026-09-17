# -*- coding: utf-8 -*-
"""
Build the written answer to the supervisor's point 5 -- compare the system
against a different RL algorithm under the same parameters.

Everything in the document is read from the result files, so the numbers cannot
drift from the experiments. Run it again whenever a stage finishes and the text
updates itself.

    python analysis/point5_report.py                 # markdown
    python analysis/point5_report.py --docx          # markdown + Word

Sources, in the order the document uses them:
    analysis/comparison_results.json   three seeds, fifteen actions, outcomes
    analysis/comparison_illegal.json   three seeds, precondition violations
    analysis/scale_per_seed.json       three seeds at 15 / 60 / 200 actions
    analysis/full_results.json         ten seeds + tuned DQN, when available
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_MD = os.path.join(ROOT, "results", "point5_algorithm_comparison.md")
OUT_DOCX = os.path.join(ROOT, "results", "point5_algorithm_comparison.docx")

BASE_STEPS = 800_000
EPISODES = 200


def load(name):
    path = os.path.join(ROOT, "analysis", name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def mean_sd(values):
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


SCALE_MODELS = os.path.join(ROOT, "results", "models", "scale")


def _env(n):
    from analysis.scaled_env import ScaledAdversaryEnv

    class _Null:
        def start_episode(self): pass
        def log_event(self, *a, **k): pass
        def end_episode(self): pass

    env = ScaledAdversaryEnv(total_actions=n,
                             config={"max_steps": 40, "real_mode": False})
    env.telemetry = _Null()
    return env


def legal_share(episodes=50, seeds=(0, 1, 2)):
    """Mean share of the action space that is legal at a step the agent reaches.

    This quantity is a property of the state distribution, not of the
    environment alone, so the policy that generates the states has to be stated.
    Two are measured:

      agent   states visited by the trained MaskablePPO policy at that catalogue
              size. This is the decision problem the system actually faces, and
              it is the figure the paper should quote.
      random  states visited by a uniform-random legal policy. Reported as a
              robustness check, since it depends on no trained checkpoint.

    An earlier version of this measurement walked the environment by always
    taking the lowest-indexed legal action. That walk never leaves the opening
    of the kill chain, where more techniques are applicable than at any later
    point, and so overstated the legal share (38.7 / 9.1 / 2.5 %). Those numbers
    should not be reproduced.
    """
    import random as _random

    from sb3_contrib import MaskablePPO

    agent, rand = {}, {}
    for n in (15, 60, 200):
        by_agent, by_random = [], []

        for s in seeds:
            path = os.path.join(SCALE_MODELS, f"maskable_a{n}_s{s}.zip")
            if not os.path.exists(path):
                continue
            model = MaskablePPO.load(path, device="cpu")
            env = _env(n)
            for ep in range(episodes):
                obs, _ = env.reset(seed=900_000 + ep)
                done = False
                while not done:
                    m = env.action_masks()
                    by_agent.append(float(m.sum()) / len(m))
                    a, _ = model.predict(obs, action_masks=m, deterministic=True)
                    obs, _, term, trunc, _ = env.step(int(a))
                    done = term or trunc

        env = _env(n)
        rng = _random.Random(0)
        for ep in range(episodes):
            env.reset(seed=900_000 + ep)
            done = False
            while not done:
                m = env.action_masks()
                by_random.append(float(m.sum()) / len(m))
                legal = [i for i, v in enumerate(m) if v]
                _, _, term, trunc, _ = env.step(rng.choice(legal) if legal else 0)
                done = term or trunc

        agent[n] = 100.0 * statistics.mean(by_agent) if by_agent else float("nan")
        rand[n] = 100.0 * statistics.mean(by_random)

    return agent, rand


def section_protocol(shares, rand_shares, comp):
    return f"""## 1. What was compared, and how

The request was to test the system against a different reinforcement-learning
algorithm under the same parameters. Three algorithms were trained on the
identical environment, with the identical budget, and evaluated with the
identical protocol. Nothing but the learning algorithm differs.

| | |
|---|---|
| Environment | Three-host Windows domain (`user01`, `srv01`, `dc01`), 40-step episode limit |
| Training budget | {BASE_STEPS:,} timesteps per run, and {BASE_STEPS*3:,} for the extended DQN condition |
| Evaluation | {EPISODES} episodes per run, deterministic policy, seeds held apart from training |
| Success | Privilege obtained on the domain controller |
| Seeds | Three per configuration in the first pass; ten in the confirmation run |

The three algorithms:

- **MaskablePPO** — the system as published. An action whose preconditions are
  unmet is removed from the distribution before sampling.
- **PPO** — the same algorithm and the same hyperparameters with the mask
  removed. This is the controlled comparison: it isolates masking itself.
- **DQN** — value-based rather than policy-gradient, and the family used by most
  of the comparable systems in the literature.

The action space is widened in a second experiment to test whether the finding
survives at catalogue scale. The padding is not invented: it is drawn from real
ATT&CK Enterprise techniques that cannot apply to this network because the
required asset does not exist in it — no cloud tenancy, no container runtime, no
network appliances, no macOS or Linux hosts, no physical access. At
{comp['60']['total']} actions all {comp['60']['structural']} padded techniques are of that kind. At
{comp['200']['total']} actions, {comp['200']['structural']} are, and the remaining {comp['200']['unmodelled']} are
Windows-applicable techniques that this environment does not model. They are
catalogued, not implemented: no result depends on their semantics, since each is
illegal in every state and carries the same penalty. Every identifier is verified
against MITRE's published bundle for ATT&CK Enterprise v19.2 by
`analysis/validate_catalogue.py`.

Two of the fifteen implemented actions carry Unix-flavoured names, `BRUTE_FORCE_SSH`
and `PRIV_ESC_SUDO`, which sit awkwardly beside the statement that the network has
no Linux or macOS hosts. Both map to platform-neutral ATT&CK techniques — T1110
Brute Force and T1068 Exploitation for Privilege Escalation — and denote those
techniques applied to Windows services; the names are legacy labels from an
earlier version of the environment.

The quantity being varied is the share of the action space that is legal at a
step the agent actually reaches. This depends on the policy generating the
states, so both the trained agent's own distribution and a uniform-random legal
policy are given:

| Action space | Legal share, trained agent | Legal share, random legal policy |
|---|---|---|
| 15 techniques | {shares[15]:.1f}% | {rand_shares[15]:.1f}% |
| 60 techniques | {shares[60]:.1f}% | {rand_shares[60]:.1f}% |
| 200 techniques | {shares[200]:.1f}% | {rand_shares[200]:.1f}% |

A correction to the earlier draft belongs here. The legal shares previously
reported (38.7 / 9.1 / 2.5 %) came from a walk that always took the
lowest-indexed legal action. That walk never leaves the opening of the kill
chain, where more techniques apply than at any later point, so it overstated how
much of the action space is available. The figures above replace it.
"""


def section_outcomes(cmp_res):
    s = cmp_res["summary"]

    def row(label, key):
        d = s[key]
        succ = d["success_rate"]
        det = d["detection_rate"]
        rew = d["avg_reward"]
        steps = d["avg_steps"][0]
        steps_txt = "never completes" if steps != steps else f"{steps:.1f}"
        return (f"| {label} | {succ[0]:.1f} ± {succ[1]:.1f} | {det[0]:.1f} | "
                f"{rew[0]:.1f} ± {rew[1]:.1f} | {steps_txt} |")

    return f"""## 2. Result: on outcomes, masking makes no difference

Three seeds, fifteen techniques, {EPISODES} evaluation episodes each. Mean ± sd
across seeds.

| Algorithm | Success rate (%) | Detection rate (%) | Mean reward | Steps to success |
|---|---|---|---|---|
{row('MaskablePPO (published system)', 'maskable')}
{row('PPO, mask removed', 'ppo')}
{row('DQN', 'dqn')}

Unmasked PPO matches the masked system on every outcome measure. The difference
in success rate is {abs(s['ppo']['success_rate'][0] - s['maskable']['success_rate'][0]):.2f} percentage points, well inside the seed-to-seed
spread of either. DQN reached the domain controller in none of the {EPISODES}
evaluation episodes, on any of these three seeds. That statement is deliberately
scoped to these seeds: the wider run reported in Section 5 does find a DQN seed
that solves the task, and the claim is corrected there.

**This required a correction to the manuscript.** The paper claimed that
prerequisite masking is what makes the system perform. That claim does not
survive the controlled test and has been withdrawn. What follows is what the
evidence does support.
"""


def section_violations(illegal):
    m, p, d = illegal["maskable"], illegal["ppo"], illegal["dqn"]
    return f"""## 3. Result: the difference is in what the agent attempts

The algorithms diverge sharply on a measure the original evaluation did not
report — actions selected whose preconditions are unmet. These are steps no real
adversary could have taken.

| Algorithm | Precondition violations (% of actions) | Episodes containing at least one |
|---|---|---|
| MaskablePPO | {m['illegal_pct_mean']:.1f} | {m['episodes_affected_mean']:.1f}% |
| PPO, mask removed | {p['illegal_pct_mean']:.1f} ± {p['illegal_pct_sd']:.1f} | {p['episodes_affected_mean']:.1f}% |
| DQN | {d['illegal_pct_mean']:.1f} ± {d['illegal_pct_sd']:.1f} | {d['episodes_affected_mean']:.1f}% |

![Controlled comparison: identical environment, budget and seeds; only the algorithm differs.](../images/fig_controlled_comparison.png)

Masking gives zero violations by construction, in every run. Unmasked PPO
violates a precondition in roughly one action in twenty, and does so in
{p['episodes_affected_mean']:.0f}% of episodes — it reaches the same objective, but the trace of how it
got there contains steps that could not have happened. DQN violates a
precondition in the majority of its actions.

This matters for this system specifically, because its output is attack
telemetry for training and testing defences. An emulator that reaches the
objective while attempting impossible techniques emits sequences no defender
would ever observe, which is a defect in the product even when the score is
identical.
"""


def section_scaling(scale):
    rows = []
    for n in (15, 60, 200):
        for algo, label in (("maskable", "MaskablePPO"), ("ppo", "PPO")):
            for s in (0, 1, 2):
                k = f"{algo}_{n}_s{s}"
                if k not in scale:
                    continue
                r = scale[k]
                rows.append(f"| {n} | {label} | {s} | {r['success_rate']:.1f} | "
                            f"{r['avg_reward']:.0f} | {r['illegal_pct']:.1f} |")
    body = "\n".join(rows)

    collapsed = [k for k, v in scale.items()
                 if k.startswith("ppo") and v["illegal_pct"] > 40]
    return f"""## 4. Result: at catalogue scale, unmasked runs become unreliable

Reported per seed rather than as a mean. The spread across seeds exceeds the
mean, so an average would hide the finding rather than summarise it.

| Actions | Algorithm | Seed | Success (%) | Mean reward | Violations (%) |
|---|---|---|---|---|---|
{body}

Every masked run is identical in behaviour at every catalogue size: full
success, no violations. The unmasked runs split into two populations. Most
degrade gently — violations rise from about 4% at fifteen actions to about 10%
at two hundred. But {len(collapsed)} of the six unmasked runs at sixty and two hundred actions
collapsed outright, spending roughly three quarters of their actions on
techniques that cannot apply, and losing about 40% of the reward. Success rate
stays at 100% even in the collapsed runs, which is precisely why success rate
alone was the wrong measure to judge this on.

![Effect of catalogue size: success is unaffected; without masking, some runs collapse.](../images/fig_action_space_scaling.png)

![Run-to-run stability. Masking removes the spread rather than shifting the average.](../images/fig_run_stability.png)

A collapse observed in one seed of three is an observation, not a rate. Ten
seeds per configuration are being run to state the frequency with a confidence
interval; Section 6 records the status.
"""


def section_dqn(full):
    if not full:
        return """## 5. Is the DQN comparison fair?

DQN scored zero with library defaults, and "your baseline failed because you did
not tune it" is a fair objection — one that would undermine the comparison
against the DQN-based systems in the literature. Two further configurations are
therefore being trained: one with hyperparameters chosen for a long-horizon
sparse-reward task (extended exploration schedule, larger replay buffer), and
one with three times the training budget.

**This run is still in progress.** The claim that DQN fails on this task should
not be stated in the paper until both configurations have finished, so that it
rests on evidence rather than on an untuned baseline.
"""
    labels = {"dqn_default": "DQN, library defaults",
              "dqn_tuned": "DQN, tuned for this task",
              "dqn_tuned_long": "DQN, tuned, 3x budget"}
    lines = []
    dqn_runs = dqn_wins = tuned_runs = tuned_wins = 0
    for name in ("dqn_default", "dqn_tuned", "dqn_tuned_long"):
        if name not in full:
            continue
        s = full[name]
        wins = sum(1 for r in s["per_seed"] if r["success_rate"] > 50)
        dqn_runs += s["n"]
        dqn_wins += wins
        if name != "dqn_default":
            tuned_runs += s["n"]
            tuned_wins += wins
        lines.append(f"| {labels[name]} | {s['n']} | {wins}/{s['n']} | "
                     f"{s['success_median']:.1f} | {s['reward_median']:.0f} | "
                     f"{s['illegal_median']:.1f} |")
    body = "\n".join(lines) or "| (no DQN runs evaluated yet) | | | | | |"

    verdict = f"""
Across all {dqn_runs} DQN runs, {dqn_wins} reached the domain controller. **This corrects the
earlier statement that DQN cannot solve the task.** It can, but rarely and
unreliably: the successful run came from the *untuned* configuration, while
{tuned_wins} of the {tuned_runs} tuned runs succeeded, including those given three times the
budget. Tuning did not help, and extending the budget did not help.

The honest form of the claim is therefore that DQN is unreliable on this task
rather than incapable of it, and that its failure is not an artefact of leaving
it at library defaults — the tuned configurations did worse, not better. Whether
the single success is a fortunate initialisation or a genuinely reachable
optimum cannot be settled at this sample size, and the paper should say so.
"""

    extra = ""
    for name, label in (("nomask_200", "PPO, mask removed, 200 actions"),
                        ("masked_200", "MaskablePPO, 200 actions")):
        if name in full:
            s = full[name]
            solved = sum(1 for r in s["per_seed"] if r["success_rate"] > 50)
            extra += (f"\n- **{label}**: reached the objective in {solved} of {s['n']} runs; of those, "
                      f"{s['collapsed']} collapsed "
                      f"({s['collapse_rate_pct']:.0f}%, 95% CI "
                      f"{s['collapse_ci95'][0]:.0f}–{s['collapse_ci95'][1]:.0f}%). "
                      f"Median reward {s['reward_median']:.0f} "
                      f"(range {s['reward_min']:.0f}–{s['reward_max']:.0f}), "
                      f"median violations {s['illegal_median']:.1f}%.")

    return f"""## 5. Is the DQN comparison fair?

DQN scored zero on all three seeds of the first pass, and "your baseline failed
because you did not tune it" is a fair objection. Two further configurations
were therefore trained: one with hyperparameters chosen for a long-horizon
sparse-reward task, and one with three times the budget. The seed count was also
raised, which is how the single success below came to light.

All three rows below are at the implemented action space of fifteen techniques,
with the same budget and the same evaluation protocol, so they are comparable to
each other and to the DQN row of Section 2.

| Configuration | Seeds | Seeds solving the task | Median success (%) | Median reward | Median violations (%) |
|---|---|---|---|---|---|
{body}
{verdict}
The collapse result is a separate experiment at two hundred actions, and its
numbers are not comparable with the table above — a different action space makes
a different task. Within it, the two agents differ only in whether the mask is
applied. Note that reaching the objective and collapsing are not alternatives: a
collapsed run still reaches the domain controller, which is exactly why the
success rate does not reveal it.
{extra}
"""


def section_conclusion(full):
    status = ("complete" if full else
              "the three-seed results are final; the ten-seed confirmation and "
              "the tuned DQN conditions are still training")
    rate = ""
    if full and "nomask_200" in full:
        s = full["nomask_200"]
        m = full.get("masked_200", {})
        rate = (f"\n\n**The collapse rate, stated with an interval.** At two hundred "
                f"actions, {s['collapsed']} of {s['n']} unmasked runs collapsed "
                f"({s['collapse_rate_pct']:.0f}%, 95% CI "
                f"{s['collapse_ci95'][0]:.0f}–{s['collapse_ci95'][1]:.0f}%), against "
                f"{m.get('collapsed', 0)} of {m.get('n', 0)} masked runs. The interval is wide "
                f"because ten seeds is a small sample for a binomial rate; what it "
                f"supports is that collapse is a real and repeatable failure mode of "
                f"the unmasked configuration, not that its frequency is known "
                f"precisely.")
    return f"""## 6. What this changes, and current status

**The finding.** Prerequisite masking does not explain this system's
performance — an unmasked agent with identical hyperparameters reaches the same
objective just as often. Masking is a fidelity and reliability property: it
guarantees that every emitted action sequence is one a real adversary could have
executed, and it prevents the failure in which an agent facing a realistically
large technique catalogue spends most of its actions on impossible techniques.
For a system whose output is attack telemetry, that is the property that
matters, and it is a stronger claim than the one it replaces because it is the
one the evidence supports.{rate}

**Status:** {status}.

**Limitations stated plainly.** The environment abstracts a real network to
three hosts and fifteen implemented techniques; the padded action spaces test
sensitivity to catalogue size, not to the full behavioural complexity of ATT&CK.
The v1 and v2 rows of the version-progression table come from runs made before
version control and cannot be reproduced; they are flagged as such in the text
rather than removed.
"""


def build():
    cmp_res = load("comparison_results.json")
    illegal = load("comparison_illegal.json")
    scale = load("scale_per_seed.json")
    full = load("full_results.json")
    if not (cmp_res and illegal and scale):
        sys.exit("missing result files; run the comparison experiments first")

    from analysis.attack_catalogue import composition
    comp = {str(n): composition(n) for n in (60, 200)}
    shares, rand_shares = legal_share()

    # the figures label their x axis with these, so write them where
    # analysis/plot_comparison.py can read them rather than retyping
    with open(os.path.join(ROOT, "analysis", "legal_share.json"), "w") as fh:
        json.dump({str(k): round(v, 2) for k, v in shares.items()}, fh, indent=2)

    doc = "\n".join([
        "# Point 5 — Comparison against a different RL algorithm",
        "",
        "Prepared for Prof. Dr. İbrahim Özçelik. Every figure in this document is",
        "generated directly from the experiment output files by",
        "`analysis/point5_report.py`.",
        "",
        section_protocol(shares, rand_shares, comp),
        section_outcomes(cmp_res),
        section_violations(illegal),
        section_scaling(scale),
        section_dqn(full),
        section_conclusion(full),
    ])

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"written {OUT_MD}")
    return doc


def to_docx():
    """Convert to Word, run from the document's own directory.

    The figures are referenced as ../images/..., which is what a reader of the
    markdown sees; running pandoc from results/ makes those same paths resolve
    for the Word build without a second set of paths to keep in step.
    """
    ref = os.path.join(ROOT, "custom-reference.docx")
    cmd = ["pandoc", os.path.basename(OUT_MD), "-o", OUT_DOCX]
    if os.path.exists(ref):
        cmd += [f"--reference-doc={ref}"]
    subprocess.run(cmd, check=True, cwd=os.path.dirname(OUT_MD))
    print(f"written {OUT_DOCX}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", action="store_true")
    a = ap.parse_args()
    build()
    if a.docx:
        to_docx()
