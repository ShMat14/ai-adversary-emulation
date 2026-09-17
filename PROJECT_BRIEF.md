# Project Brief — Reinforcement-Learning Adversary Emulation

*A summary of the current state of the work for a collaborator or supervisor.
Prepared 2026-08-17. Companion to the submitted manuscript.*

---

## 1. Summary

The project trains a reinforcement-learning agent to autonomously execute MITRE ATT&CK-aligned
attack kill chains against a simulated three-host Windows Active Directory network, and to emit
labelled attack telemetry that a blue team can use for detection engineering and training data.

There are two systems:

- **v3** — the system whose results appear in the **submitted manuscript**. 15 techniques, a
  largely linear kill chain, an abstract detection model.
- **v4** — a substantial rebuild of the environment, undertaken after submission to make the
  paper's claims more defensible. 25 techniques, per-episode scenario randomization, a rule-based
  detection engine grounded in real Windows/Sysmon event IDs, and a single shared world model used
  by both the training simulator and the deployable live target.

v4 is complete: built, trained, evaluated, and written up. The remaining work is editorial and
requires a human decision — see §6.

---

## 2. Why v4 was built

Three weaknesses in v3 motivated the rebuild.

**The simulator and the live target disagreed.** In v3 they were two independently hand-written
implementations of the same world. They drifted: phishing succeeded 86.5% of the time in
simulation but 54.0% against the HTTP server — a 32.5-point gap that undermined the paper's
sim-to-real claim. In v4 both import one shared module, so they cannot disagree by construction;
the residual gap is transport only.

**Detection was abstract.** v3 used a scalar `alert × 0.2` term. v4 replaces it with a rule-based
blue team that maps each technique to specific Windows and Sysmon event IDs and named detection
rules, accumulates a suspicion score, and ends the episode when the score crosses an incident
threshold. Detection outcomes are now interpretable and arguable in security terms.

**The environment was too easy and too static.** 15 techniques on a near-linear chain meant the
agent had little to adapt to. v4 has 25 techniques and randomizes, per episode, which are
available and what the objective requires — so the agent must actually respond to the environment
in front of it rather than replay one memorized sequence.

---

## 3. The system

The training simulator (a Gym environment) and the deployable target (an HTTP server with token
auth, per-session state, a SIEM feed, and ECS export) are thin wrappers over one shared world
model. That model owns the technique registry — each technique carries a success probability, a
noise level, its ATT&CK identifier, its precondition and its effect — along with the network
state, the per-episode scenario randomization, the legal-action mask, and the state transition.

The agent is **MaskablePPO**: the environment computes, at every step, which techniques are
currently legal given the state and the episode's scenario, and the policy's action distribution
is restricted to that set. Comparison baselines are plain PPO and DQN, neither of which has a
masking variant.

Actions whose preconditions are not met are **inert** — they neither change state nor emit
telemetry. This matters for the comparison in §5: it prevents an unmasked agent from bypassing
scenario requirements by spamming illegal actions.

---

## 4. Results

All figures below: 25 techniques, 3 seeds, 400-episode evaluation, fixed 40-step budget.

| Configuration | Success | Detection | Illegal actions |
|---|---|---|---|
| **Masked (MaskablePPO)** | **96.1 ± 0.5%** | 7.0% | **0.0%** |
| Unmasked (PPO) | 47.9 ± 3.6% | 2.6% | 87.5% |
| Unmasked (DQN) | 52.2 ± 7.5% | 7.7% | 82.4% |

**Adaptation.** Entry-distribution divergence reaches the maximum of 2.00/2.00: the agent uses SQL
injection 100% of the time when a web application is present and 0% when it is absent, selecting
a different entry vector instead. Over evaluation it exercises all 25 techniques.

**Sim-to-live transfer.** 96.1% in simulation against 94.7 ± 2.2% on the live HTTP target — a
1.4-point gap, compared with v3's 32.5-point discrepancy. Both modes share the detection engine
and both detect at approximately 7%.

**On the 96%.** Success is deliberately not 100%. With realistic detection, some randomized
scenarios force techniques that are legitimately noisy and get caught. A ceiling below 100% is
more credible than v3's perfect score, not less.

---

## 5. The masking finding

This is the scientifically interesting part, and it **changed** between v3 and v4.

**What the submitted paper says.** *Masking is fidelity, not performance.* In v3, an unmasked PPO
agent still reached the objective — roughly 100% success with only about 5% illegal actions.
Masking guaranteed zero illegal actions, which matters because it means every emitted telemetry
sequence is actually executable, but it was not required in order to succeed.

**What v4 shows.** The unmasked agent collapses: 47.9% success, 87.5% illegal actions. In v4,
masking is **competence-critical**, not merely a fidelity property.

**Why the change is real.** In v3, legality is a stable function of state — for example,
Kerberoasting is legal exactly when the domain controller is compromised but not yet privileged.
An unmasked agent can memorize a fixed rule of that shape. In v4, legality additionally depends on
the random per-episode scenario, so there is no fixed rule available to learn. Masking supplies
the correct legal set for *this* episode, structurally.

**Three controls rule out an unfair baseline.**

1. *Reward teaching.* v3 penalized illegal actions, teaching the unmasked agent to avoid them;
   v4's default reward does not. Adding a v3-style −10 penalty moved success only from 47.9% to
   54.7%, and illegal actions from 87.5% to 82.2%. This is not a motivation problem.
2. *Step budget.* Perhaps 40 steps is too short for a wasteful unmasked agent to ever finish and
   observe the terminal reward. Retrained at a 150-step budget for up to 4M timesteps: 50.8%,
   essentially the 40-step result. Budget is neutral.
3. *Algorithm family.* DQN, a value-based method with no masking variant, also fails: 52.2%
   success, 82.4% illegal. This is not specific to PPO.

Across the full matrix (PPO and DQN × 40- and 150-step budgets × with and without the illegal
penalty), every unmasked configuration lands at 48–55% success and 82–95% illegal actions, while
the masked configuration is at 96%. The determinant is the mask.

**Caveats that belong in any writeup.** The property that makes v4 hard for the unmasked agent —
per-episode scenario randomization — was introduced to obtain *adaptation*, which was an
independent design goal. Masking becoming competence-critical is a downstream consequence of that
choice, not an environment engineered to flatter masking, and it should always be presented in
that order. The headline numbers are also under a fixed 40-step budget; control (2) above is what
establishes that the collapse is not an artifact of that constraint.

---

## 6. The open question for the supervisor

Two defensible ways to present this, and the choice is a judgment call rather than a technical one:

**(a) Lead with the evolution.** Report v4's competence-critical result as the headline: as the
environment grows in size and variability, masking moves from a fidelity property to a competence
requirement. Both statements remain true of their respective environments. This is what the
combined paper currently does.

**(b) Preserve the v3 headline.** Measure the masking comparison on the *base* task without
per-episode requirements, which should reproduce the "fidelity, not performance" finding, and
report breadth and adaptation separately as independent contributions of v4.

Option (a) is the more interesting scientific claim; option (b) keeps continuity with what was
submitted. The alternative is documented in `results/v4_paper_outline.md` §6, and switching to it
is an editorial change, not a re-run — the measurements for both already exist.

---

## 7. Deliverables

- **`results/combined_paper.md`** — the main document: the submitted manuscript's full text with
  v4 folded in as a refinement. Extended abstract and contributions, a new methods section on the
  shared model and rule-based detection, results sections on breadth/adaptation/transfer and on
  masking, and a rewritten conclusion. Approximately 9,900 words, 7 figures, 11 tables.
- **`results/point5_v4_vs_v3.md`** and **`results/point5_v4_vs_v3_TR.md`** — a focused v4-versus-v3
  comparison, in English and in academic Turkish.
- **Figures** — nine, covering masking, metric comparison, technique breadth, scenario adaptation,
  phase coverage, detection rules, version progression, the DQN comparison, and transfer. All
  regenerated from the final result files.
- **Blue-team outputs** — generated incident reports and the labelled SIEM telemetry dataset.
- **The deployable target** — HTTP server with a Dockerfile, for anyone wanting to run the
  environment live.

---

## 8. Reproducibility

Every number in §4 and §5 comes from a JSON result file written by the evaluation scripts, and
every figure is regenerated from those files by a single plotting script. Nothing is
hand-transcribed. The exact code state behind the submitted manuscript is preserved on a frozen
git tag, separate from the v4 branch, so the original results remain reproducible independently of
the rebuild.

---

## 9. Status

**Complete:** the v4 environment, training, and evaluation; verification of all 25 techniques
against MITRE ATT&CK v19.2; adaptation and breadth results; sim-to-live transfer; the masking
finding with its three controls; all figures; the comparison documents in both languages; the
combined paper.

**Awaiting a decision:**

1. The framing question in §6.
2. Whether to revise the submitted manuscript with v4 or publish v4 as a separate follow-up.
3. Outstanding items from the original submission: the supervisor's biography and author
   photographs, and the journal scope question.

**Optional, not required:** training the agent directly against the live HTTP target rather than in
simulation. The transfer evaluation already demonstrates the point; this would be a robustness
addition, costing hours of compute.
