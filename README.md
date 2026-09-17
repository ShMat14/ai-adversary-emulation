# AI-Based Adversary Emulation Platform

> **Advanced Cybersecurity Testing using Reinforcement Learning and MITRE ATT&CK Framework**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/RL-MaskablePPO-green.svg)](https://sb3-contrib.readthedocs.io/)
[![MITRE ATT&CK](https://img.shields.io/badge/MITRE-ATT%26CK-red.svg)](https://attack.mitre.org/)

---

## 🎯 Overview

This platform leverages **Reinforcement Learning (RL)**, specifically **Maskable Proximal Policy Optimization (MaskablePPO)**, to train autonomous AI agents that simulate sophisticated cyberattacks. Unlike traditional rule-based adversary emulation tools, this system learns optimal attack strategies through trial-and-error, dynamically adapting to defensive responses and environment prerequisites.

### Key Capabilities

- ✅ **Autonomous Attack Path Discovery**: The RL agent learns multi-stage kill chains without pre-programmed scripts
- ✅ **MITRE ATT&CK Alignment**: All 15 attack actions map to specific MITRE ATT&CK tactics and techniques
- ✅ **Action Masking**: Environment-enforced prerequisites (T1021 requires scan + credentials, etc.) prevent illegal moves and force realistic sequencing
- ✅ **Target Randomisation**: 50/50 SQLi availability per episode forces the agent to generalise across both phishing and web-exploit initial access paths
- ✅ **Adaptive Evasion**: Agent learns proactive log-clearing and alert-level management as emergent behaviour
- ✅ **Dual-Mode Evaluation**: Sim mode (fast, probabilistic) and Real mode (Flask mock server) with verified sim-to-real transfer

---

## 📊 Final Results (PPO v3 — 400 episodes)

| Mode             | Success% | Detect% | Avg Steps | Avg Reward | Techniques Used |
|------------------|----------|---------|-----------|------------|-----------------|
| PPO-Agent (Sim)  | **100.0%** | **0.5%** | 15.03   | 811.75     | 10.0 / 15       |
| PPO-Agent (Real) | **98.5%**  | **7.0%** | 15.78   | 776.80     | 8.4 / 15        |

### vs. Baselines (200 episodes each)

| Agent               | Success% | Detect% | Avg Steps | Avg Reward | Techniques |
|---------------------|----------|---------|-----------|------------|------------|
| Scripted-Standard   | 0.0%     | 0.0%    | N/A       | 61.00      | 6.0        |
| Scripted-Stealthy   | 0.0%     | 0.0%    | N/A       | 53.46      | 8.0        |
| Scripted-Aggressive | 0.0%     | 1.0%    | N/A       | 23.15      | 6.0        |
| Scripted-SQLi       | 0.0%     | 0.0%    | N/A       | 25.32      | 5.0        |
| Random Agent        | 87.0%    | 19.5%   | 15.05     | 616.44     | 9.6        |
| **PPO-Agent (Sim)** | **100.0%** | **0.5%** | **15.03** | **811.75** | **10.0** |
| **PPO-Agent (Real)**| **98.5%**  | **7.0%** | **15.78** | **776.80** | **8.4**  |

---

## 📐 Architecture

### System Components

```
┌──────────────────────────────────────────────────────┐
│                   Training Pipeline                   │
│                                                      │
│  MaskablePPO ──► AdversaryEnv ──► Action Masking     │
│       ▲               │                              │
│       │           ┌───┴──────────────────┐           │
│    reward      Sim Mode          Real Mode           │
│                (probabilistic)   (Flask API)         │
│                               http://localhost:5000  │
└──────────────────────────────────────────────────────┘

┌─────────────────────────┐    ┌──────────────────────┐
│   Network State (obs)   │    │    Telemetry Logger   │
│  user01 · srv01 · dc01  │    │  results/telemetry/  │
│  9 host flags + 4 global│    │  JSON episode logs   │
└─────────────────────────┘    └──────────────────────┘
```

### Observation Space (13 inputs)

| Index | Feature | Description |
|-------|---------|-------------|
| 0–2 | user01 flags | compromised, privileged, has_credentials |
| 3–5 | srv01 flags | compromised, privileged, has_credentials |
| 6–8 | dc01 flags | compromised, privileged, has_credentials |
| 9 | network_scanned | 1 if NETWORK_SCAN completed |
| 10 | backdoor_installed | 1 if WEB_SHELL_UPLOAD/INSTALL_BACKDOOR succeeded |
| 11 | alert_level | normalised 0–1 (raw / 3.0) |
| 12 | sqli_available | 1 if target has SQLi vuln this episode |

---

## 🗺️ MITRE ATT&CK Mapping (15 Techniques)

| # | Action | MITRE ID | Tactic | Reward | Noise |
|---|--------|----------|--------|--------|-------|
| 1 | PHISHING_EMAIL | T1566 | Initial Access | +10 | 0.05 |
| 2 | BRUTE_FORCE_SSH | T1110 | Credential Access | +5 | **0.50** |
| 3 | NETWORK_SCAN | T1046 | Discovery | +15 | 0.10 |
| 4 | VALID_ACCOUNTS_LOGIN | T1078 | Defense Evasion | +20 | 0.05 |
| 5 | INSTALL_BACKDOOR | T1543 | Persistence | +15 | 0.15 |
| 6 | CLEAR_LOGS | T1070 | Defense Evasion | +10 | −1.0 |
| 7 | LATERAL_MOVE_SMB | T1021 | Lateral Movement | +30/+50 | 0.15 |
| 8 | PRIV_ESC_SUDO | T1068 | Privilege Escalation | +10/+100 | 0.30 |
| 9 | EXFILTRATE_DATA | T1041 | Exfiltration | **+500** | 0.25 |
| 10 | RANSOMWARE_ENCRYPT | T1486 | Impact | +500 | 0.35 |
| 11 | SQL_INJECTION | T1190 | Initial Access | +30 | 0.12 |
| 12 | PASS_THE_HASH | T1550.002 | Lateral Movement | +25/+40 | 0.20 |
| 13 | POWERSHELL_EXEC | T1059.001 | Execution | +15/+100 | 0.25 |
| 14 | KERBEROASTING | T1558.003 | Credential Access | +100 (DC) | 0.15 |
| 15 | WEB_SHELL_UPLOAD | T1505.003 | Persistence | +25 | 0.20 |

> **Notes**: LATERAL_MOVE_SMB / PASS_THE_HASH give higher reward (+50/+40) when reaching dc01. PRIV_ESC_SUDO / POWERSHELL_EXEC / KERBEROASTING give +100 at DC endgame. EXFILTRATE_DATA/RANSOMWARE_ENCRYPT terminate episode as WIN (+500).

### Learned Kill Chain (emergent — not hard-coded)

```
PHISHING_EMAIL  or  SQL_INJECTION  (episode-randomised)
       ↓
VALID_ACCOUNTS_LOGIN  →  WEB_SHELL_UPLOAD  →  NETWORK_SCAN
       ↓
CLEAR_LOGS  (reactive, when alert > 0.5)
       ↓
LATERAL_MOVE_SMB → srv01  →  LATERAL_MOVE_SMB → dc01
       ↓
POWERSHELL_EXEC  or  KERBEROASTING  (DC privilege escalation)
       ↓
EXFILTRATE_DATA  ←  WIN  (+500 reward, episode terminates)
```

---

## 🚀 Setup Instructions

### Prerequisites

- **Python**: 3.8 or higher
- **OS**: Windows, Linux, or macOS
- **Hardware**: CPU sufficient for training; GPU optional

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ShMat14/ai-adversary-emulation.git
cd ai-adversary-emulation

# 2. Create virtual environment
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install stable-baselines3 sb3-contrib gymnasium numpy matplotlib \
            scipy flask requests tensorboard
```

---

## 🎮 Usage Guide

### 1. Train a New Agent

```bash
# Optional: start mock server (needed only if real_mode=True in config)
python Target/mock_server.py

# Train PPO v3
python agents/ppo_train.py
```

Saved to `results/models/ppo_adversary_v3.zip`. Training ~15–20 min on CPU for 1M steps.

### 2. Evaluate (Sim + Real)

```bash
# Requires mock server running on port 5000 for real-mode eval
python agents/ppo_eval.py
```

Expected output (v3 model, 200 episodes):
```
=============================================
PPO EVALUATION RESULTS  [Sim Mode]
=============================================
  Episodes evaluated      : 200
  Success Rate            : 100.00%
  Detection Rate          : 0.50%
  Avg Steps to Goal       : 15.03
  Avg Reward/Episode      : 811.75
  Avg Unique Techniques   : 10.0 / 15

PPO EVALUATION RESULTS  [Real Mode]
=============================================
  Episodes evaluated      : 200
  Success Rate            : 98.50%
  Detection Rate          : 7.00%
  Avg Steps to Goal       : 15.78
  Avg Reward/Episode      : 776.80
  Avg Unique Techniques   : 8.4 / 15
```

### 3. Run All Baselines + Generate Thesis Plots

```bash
python analysis/thesis_final.py
```

Generates 4 publication-quality plots saved to the project root:
- `thesis_plot_5_metric_comparison.png` — all agents, 4 metrics
- `thesis_plot_5_action_freq.png` — technique usage Sim vs Real
- `thesis_plot_5_radar.png` — spider/radar multi-dimension profile
- `thesis_plot_5_sim_real_gap.png` — transfer gap analysis

### 4. Generate Training Curves

```bash
# From TensorBoard event files (MaskablePPO_1 run)
python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# ... or simply run the standalone curve script:
"
# Training curves saved as thesis_plot_6_training_curves.png
```

### 5. View TensorBoard Live

```bash
tensorboard --logdir=./results/runs/
# Navigate to http://localhost:6006
```

### 6. Compare with Scripted Baselines

```bash
python baselines/scripted_attacks.py
```

---

## 📂 Repository Structure

```
ai-adversary-emulation_4/
├── agents/
│   ├── ppo_train.py              # MaskablePPO training script (1M steps)
│   └── ppo_eval.py               # Evaluation (Sim + Real), sim-to-real table
├── env/
│   ├── adversary_env.py          # Gymnasium env: 15 actions, masking, rewards
│   ├── attack_actions.py         # AttackAction definitions (15 MITRE techniques)
│   ├── state_models.py           # NetworkState, HostState
│   └── __init__.py
├── Target/
│   └── mock_server.py            # Flask mock target (HTTP endpoints for all 15 actions)
├── baselines/
│   └── scripted_attacks.py       # 4 scripted kill-chains + PPO comparison table
├── analysis/
│   ├── thesis_final.py           # Full pipeline: baselines + random agent + 4 plots
│   ├── thesis_results.json       # Cached baseline results
│   ├── compute_metrics.py        # Metric utilities
│   └── generate_thesis_plots.py  # Legacy plot script
├── telemetry/
│   └── telemetry_logger.py       # JSON episode event logger
├── results/
│   ├── models/
│   │   ├── ppo_adversary_v3.zip  # ← FINAL trained model
│   │   ├── ppo_adversary_v2.zip
│   │   └── ppo_adversary_model.zip
│   ├── runs/
│   │   └── MaskablePPO_1/        # TensorBoard logs (~800K steps)
│   └── telemetry/                # Per-episode JSON logs
├── thesis_results_narrative.md   # Chapter 4 results write-up (§4.1–4.6)
├── thesis_plot_5_metric_comparison.png
├── thesis_plot_5_action_freq.png
├── thesis_plot_5_radar.png
├── thesis_plot_5_sim_real_gap.png
├── thesis_plot_6_training_curves.png
└── README.md
```

---

## 🧪 Key Design Decisions

### Why MaskablePPO?

Standard PPO cannot enforce action prerequisites (e.g., lateral movement requires both credentials AND a network scan). `MaskablePPO` from `sb3-contrib` applies a boolean mask at each step — invalid actions receive −∞ logit, making them impossible to select. This eliminates illegal moves without requiring the reward function to penalise every impossible action.

### Why 5 Techniques Are Never Used

The PPO agent converged to never using: `BRUTE_FORCE_SSH`, `INSTALL_BACKDOOR`, `PRIV_ESC_SUDO`, `RANSOMWARE_ENCRYPT`, `PASS_THE_HASH`. Each is strictly dominated by an alternative in reward, success probability, or noise. See `thesis_results_narrative.md` §4.5 for the full per-technique analysis.

### Target Randomisation (50/50 SQLi)

Each episode randomly enables or disables SQL injection vulnerability (`sqli_available` flag). This forces the agent to learn *both* initial-access strategies (phishing when SQLi is off, SQL injection when it's on), preventing overfitting to a single entry path.

---

## 🛡️ Security Considerations

> [!WARNING]
> This platform is for **ETHICAL SECURITY RESEARCH ONLY**. Unauthorized use against systems you do not own or have explicit permission to test is illegal.

- All attack simulation is contained within the mock environment
- The Flask mock server uses a hardcoded bearer token (`rl-agent-secret-token-2026`)
- No real network traffic is generated in simulation mode
- Deploy in an isolated network when using real-mode evaluation

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **MITRE ATT&CK** — comprehensive adversary behaviour knowledge base
- **Stable-Baselines3 / sb3-contrib** — MaskablePPO implementation
- **OpenAI Gym / Gymnasium** — standardised RL environment interface
- **CAGE Challenge** — inspiration for autonomous cyber agents

---

> **Disclaimer**: This tool is intended for authorised security testing and academic research only.
