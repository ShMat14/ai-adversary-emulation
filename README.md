# Reinforcement-Learning Adversary Emulation

> ATT&CK-aligned adversary emulation with a shared-model environment, explainable
> rule-based detection, and prerequisite action masking.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/RL-MaskablePPO-green.svg)](https://sb3-contrib.readthedocs.io/)
[![MITRE ATT&CK](https://img.shields.io/badge/MITRE-ATT%26CK-red.svg)](https://attack.mitre.org/)

This repository accompanies the paper *"Reinforcement Learning-Based Adversary Emulation with a
Shared-Model, Explainable-Detection Environment: A MaskablePPO Approach Aligned with the MITRE
ATT&CK Framework."* It contains the environment, the training and evaluation code, the trained
checkpoints, and the figures reported in the paper.

---

## Overview

A MaskablePPO agent learns to sequence MITRE ATT&CK-aligned kill chains against a simulated
three-host Windows Active Directory network, and emits labelled attack telemetry that a blue team
can use for detection engineering.

Three properties distinguish the environment from the simulators this field usually reuses.

- **Prerequisites are structural, not reward-shaped.** Each of the 25 techniques carries a real
  precondition. The environment computes the legal set at every step and the policy can only sample
  from it, so an emitted sequence is one an adversary could actually have executed.
- **Detection is explainable.** Every technique emits the Windows Security, Sysmon or WAF events it
  would leave in reality. Named rules watch those events and accumulate a suspicion score, so a
  reported detection decomposes into the specific rules that produced it rather than an abstract
  alert level.
- **Simulator and live target share one world model.** The Gymnasium environment and the deployable
  HTTP target are thin wrappers over the same module, so sim-to-real transfer is a measurement
  rather than a comparison between two implementations that were written to agree and do not.

---

## Results

Three seeds, 400 evaluation episodes, 40-step budget, 25 techniques.

| Configuration | Success | Detection | Illegal actions |
|---|---|---|---|
| **MaskablePPO (masked)** | **96.1 ± 0.5%** | 7.0% | **0.0%** |
| PPO (no mask) | 47.9 ± 3.6% | 2.6% | 87.5% |
| DQN (no mask) | 51.7 ± 7.0% | 7.6% | 82.4% |

Repeated at more seeds, the gap widens rather than closing:

| Configuration | 3 seeds | 5 seeds | 10 seeds |
|---|---|---|---|
| MaskablePPO (masked) | 96.1 ± 0.5% | 96.0 ± 0.4% | 95.5 ± 1.3% |
| PPO (no mask) | 47.9 ± 3.6% | 45.5 ± 5.4% | 43.2 ± 5.5% |
| DQN (no mask) | 51.7 ± 7.0% | 48.5 ± 8.7% | 44.8 ± 14.5% |

**Sim-to-live transfer.** 96.1% in simulation against 94.7 ± 2.2% on the live HTTP target, a
1.4-point gap. Both modes run the same detection engine because both wrap the same module.

**Adaptation.** When a SQL-injectable web application is present the agent enters through it in
every episode; when it is absent it never attempts SQL injection and switches to another entry
vector. Over evaluation it exercises all 25 techniques and all four routes to domain dominance.

### What masking does, and what it does not

This is the paper's central finding, and it is more qualified than an earlier version of this work
claimed.

In an initial 15-technique environment where legality is a fixed function of state, an unmasked
agent **matched** the masked one (99.8% against 99.7%). Masking there bought fidelity — every
emitted action was executable when proposed — but not success. The earlier claim that masking
explained the performance is withdrawn.

Once the legal action set varies per episode, masking becomes competence-critical: unmasked agents
of two algorithm families fall to 43–52% while the masked agent holds ~96% and never proposes an
inexecutable technique. Which regime applies is a property of the target, not of the algorithm.

A related correction: in the initial environment the agent learned to clear logs whenever its alert
level approached the threshold, which looked like emergent tradecraft. It was partly an artefact,
because log clearing was free there. Under the rule-based model the same technique raises Windows
Event 1102 at one of the highest rule rates in the catalogue, and the behaviour largely disappears.

---

## Architecture

```
env/            shared world model: technique registry, network state,
                per-episode scenario, legal-action mask, state transition
Target/         deployable HTTP target (Flask + Dockerfile), same world model
agents/         policy wrappers
baselines/      scripted attack sequences used as non-learning comparators
telemetry/      ATT&CK-labelled telemetry logger and exporters
analysis/       training, evaluation and figure generation
results/        trained checkpoints and the figures reported in the paper
```

**Observation:** 38 dimensions — three Boolean flags per host, two global flags, a normalised
suspicion score, a 24-dimensional technique-availability vector and two scenario requirement flags.

**Action space:** 25 ATT&CK-aligned techniques spanning initial access, discovery, credential
access, lateral movement, execution, privilege escalation, persistence, defense evasion,
exfiltration and impact.

---

## Setup

```bash
git clone https://github.com/ShMat14/ai-adversary-emulation.git
cd ai-adversary-emulation

python -m venv .venv
# Windows:      .venv\Scripts\Activate.ps1
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
```

Trained on CPU; no GPU required.

---

## Usage

### Train

```bash
# masked agent (the proposed system)
python analysis/v4_train.py --mode train --config masked --seed 0 --steps 2000000

# unmasked baselines
python analysis/v4_train.py --mode train --config nomask --seed 0 --steps 2000000
python analysis/v4_train.py --mode train --config dqn    --seed 0 --steps 2000000
```

`--config` is one of `masked` (MaskablePPO), `nomask` (PPO, no mask) or `dqn` (DQN, necessarily
unmasked since the standard implementations have no masked variant).

### Evaluate

```bash
# the three seeds reported in the paper
python analysis/v4_train.py --mode eval --episodes 400

# extend to more seeds
python analysis/v4_train.py --mode eval --episodes 400 \
    --seeds 0,1,2,3,4 --out analysis/v4_results_5seed.json
```

### Reproduce the extra-seed runs

```bash
bash analysis/run_extra_seeds.sh 3 4          # -> 5 seeds
bash analysis/run_extra_seeds.sh 5 6 7 8 9    # -> 10 seeds
```

The driver skips any seed already trained, so it is safe to stop and resume.

### Live target

```bash
cd Target && python mock_server.py     # or: docker build -t adversary-target .
```

Then evaluate against it with `--live`. Because the server and the simulator import the same world
model, the agent runs unchanged in both modes.

### Figures

```bash
python analysis/v4_system_figures.py
```

---

## Reproducibility

Every number in the paper is written to a JSON file by the evaluation scripts and every figure is
regenerated from those files:

- `analysis/v4_results.json` — main comparison, 3 seeds
- `analysis/v4_results_5seed.json`, `analysis/v4_results_10seed.json` — extended seeds
- `analysis/v4_catalogue_comparison.json` — 15 vs 25 technique catalogue
- `analysis/v4_table7_400.json` — budget and reward-penalty controls
- `analysis/v4_failure_modes.json` — where caught episodes end

Evaluation uses 400 episodes throughout. Training checkpoints are in `results/models/v4/`.

---

## Citation

The paper is under review; citation details will be added on publication.

## License

MIT — see `LICENSE`.
