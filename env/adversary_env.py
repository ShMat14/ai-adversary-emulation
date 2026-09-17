import gymnasium as gym
import numpy as np
import random
import requests
from gymnasium import spaces

from env.state_models import NetworkState
from env.attack_actions import ACTION_LIST
from telemetry.telemetry_logger import TelemetryLogger

# Bearer token must match VALID_TOKEN in Target/mock_server.py
AUTH_HEADERS = {"Authorization": "Bearer rl-agent-secret-token-2026"}

# Valid DC privilege-escalation techniques (used in endgame logic + mask)
VALID_DC_PRIVESC = {"PRIV_ESC_SUDO", "POWERSHELL_EXEC", "KERBEROASTING"}


class AdversaryEnv(gym.Env):
    def __init__(self, config=None):
        super(AdversaryEnv, self).__init__()
        config = config or {}

        self.host_names = ["user01", "srv01", "dc01"]
        self.state = NetworkState(self.host_names)

        # Action Space: 15 MITRE ATT&CK techniques
        self.action_space = spaces.Discrete(len(ACTION_LIST))

        # --- OBSERVATION SHAPE: 13 ---
        # 9 (Hosts: 3×3 flags) + 1 (Scan) + 1 (Backdoor) + 1 (Alert Level) + 1 (SQLi Available)
        self.observation_space = spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32)

        self.max_steps = config.get("max_steps", 40)
        self.current_step = 0

        # real_mode=True  → hit the Flask mock server (realistic, for eval)
        # real_mode=False → use probabilistic simulation (fast, for training)
        self.real_mode = config.get("real_mode", False)

        # Memory
        self.network_scanned = False
        self.backdoor_installed = False
        self.alert_level = 0.0
        self.sqli_available = False   # Randomised each episode in reset()

        # --- TELEMETRY ---
        telemetry_dir = config.get("telemetry_dir", "results/telemetry/rl")
        self.telemetry = TelemetryLogger(output_dir=telemetry_dir)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = NetworkState(self.host_names)

        # Reset Memory
        self.network_scanned = False
        self.backdoor_installed = False
        self.alert_level = 0.0

        # --- TARGET RANDOMIZATION ---
        # 50% of episodes the target has a SQLi-vulnerable web app.
        # 50% it doesn't — agent must fall back to phishing.
        # This forces the agent to learn BOTH entry strategies.
        self.sqli_available = random.random() < 0.5

        # Start a fresh telemetry episode
        self.telemetry.start_episode()

        if self.real_mode:
            try:
                requests.post("http://localhost:5000/reset",
                              json={"sqli_available": self.sqli_available},
                              timeout=0.5)
            except Exception:
                pass

        return self._get_obs(), {}

    def step(self, action_idx):
        self.current_step += 1
        reward, terminated = self._apply_action(action_idx)
        truncated = self.current_step >= self.max_steps

        # --- Probabilistic mid-episode detection ---
        # Realistic noise floor: below alert=1.0 the attacker blends into background traffic.
        # Detection only kicks in above the noise floor, scaling up to 8% at alert=3.0.
        if not terminated and self.alert_level > 1.0:
            detection_chance = ((self.alert_level - 1.0) / 2.0) * 0.08
            if random.random() < detection_chance:
                self._trigger_detection()
                reward -= 25.0
                terminated = True

        # Log every step to telemetry
        action_name = ACTION_LIST[action_idx].name
        self.telemetry.log_event(
            action=action_name,
            state_snapshot=self.state,
            reward=reward,
            step=self.current_step,
        )

        # Flush episode JSON when episode ends
        if terminated or truncated:
            self.telemetry.end_episode()

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        obs = []
        # 1. Host States (9 inputs: 3 hosts × 3 binary flags)
        for name in self.host_names:
            h = self.state.hosts[name]
            obs.extend([
                1.0 if h.compromised else 0.0,
                1.0 if h.privileged else 0.0,
                1.0 if h.has_credentials else 0.0
            ])

        # 2. Global Memory (2 inputs)
        obs.append(1.0 if self.network_scanned else 0.0)    # index 9
        obs.append(1.0 if self.backdoor_installed else 0.0)  # index 10

        # 3. Normalised Alert Level (1 input)
        obs.append(min(self.alert_level / 3.0, 1.0))         # index 11

        # 4. SQLi Available flag (1 input) — agent knows if target is vulnerable
        obs.append(1.0 if self.sqli_available else 0.0)      # index 12

        return np.array(obs, dtype=np.float32)

    def action_masks(self):
        """
        Returns a boolean mask of currently valid actions for the agent's state.
        Called automatically by MaskablePPO (sb3_contrib) during training and eval.

        This enforces realistic prerequisites:
        - Initial Access  → must compromise user01 first
        - Discovery       → NETWORK_SCAN requires being inside the network
        - Lateral Move    → requires credentials + scan; multi-hop: user01→srv01→dc01
        - DC Endgame      → only PRIV_ESC_SUDO / POWERSHELL_EXEC / KERBEROASTING accepted
        - Impact          → only after DC is fully compromised (privileged)
        """
        mask = np.zeros(len(ACTION_LIST), dtype=bool)
        hosts   = self.state.hosts
        user01  = hosts["user01"]
        srv01   = hosts["srv01"]
        dc01    = hosts["dc01"]

        any_compromised = any(h.compromised for h in hosts.values())
        any_creds       = any(h.has_credentials for h in hosts.values())

        # ── DC ENDGAME OVERRIDES ──────────────────────────────────────────────
        # Once on the DC, ONLY privesc or impact actions are valid.
        # This prevents the agent from wasting steps and forces realism.

        if dc01.privileged:
            # Final phase: exfiltrate or deploy ransomware
            for i, a in enumerate(ACTION_LIST):
                if a.name in {"EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT"}:
                    mask[i] = True
            return mask

        if dc01.compromised and not dc01.privileged:
            # DC privilege escalation phase — three valid paths
            for i, a in enumerate(ACTION_LIST):
                if a.name in VALID_DC_PRIVESC:
                    mask[i] = True
                elif a.name == "CLEAR_LOGS" and self.alert_level > 0.5:
                    mask[i] = True
            # Safety: always allow at least one privesc
            if not any(mask):
                for i, a in enumerate(ACTION_LIST):
                    if a.name in VALID_DC_PRIVESC:
                        mask[i] = True
            return mask

        # ── STANDARD KILL CHAIN PHASES ────────────────────────────────────────

        for i, a in enumerate(ACTION_LIST):
            name = a.name

            # --- Initial Access ---
            if name == "PHISHING_EMAIL":
                mask[i] = not user01.compromised

            elif name == "SQL_INJECTION":
                mask[i] = self.sqli_available and not user01.compromised

            elif name == "BRUTE_FORCE_SSH":
                # Noisy fallback — valid when user01 or srv01 not yet compromised
                mask[i] = (not user01.compromised) or (not srv01.compromised)

            # --- Discovery ---
            elif name == "NETWORK_SCAN":
                mask[i] = any_compromised and not self.network_scanned

            # --- Credential Access ---
            elif name == "VALID_ACCOUNTS_LOGIN":
                mask[i] = any(h.compromised and not h.has_credentials
                               for h in hosts.values())

            # --- Persistence ---
            elif name in {"INSTALL_BACKDOOR", "WEB_SHELL_UPLOAD"}:
                mask[i] = any_compromised and not self.backdoor_installed

            # --- Defense Evasion ---
            elif name == "CLEAR_LOGS":
                mask[i] = self.alert_level > 0.5

            # --- Lateral Movement ---
            elif name == "LATERAL_MOVE_SMB":
                # Requires creds + scan; enforces multi-hop (srv01 before dc01)
                next_hop = (None if srv01.compromised and dc01.compromised
                            else ("srv01" if not srv01.compromised else "dc01"))
                mask[i] = any_creds and self.network_scanned and next_hop is not None

            elif name == "PASS_THE_HASH":
                # No scan required — blind hop using stolen hashes (faster but noisier)
                next_hop = (None if srv01.compromised and dc01.compromised
                            else ("srv01" if not srv01.compromised else "dc01"))
                mask[i] = any_creds and next_hop is not None

            # --- Privilege Escalation (non-DC hosts) ---
            elif name in {"PRIV_ESC_SUDO", "POWERSHELL_EXEC"}:
                mask[i] = any(h.compromised and not h.privileged and h.name != "dc01"
                               for h in hosts.values())

            # --- KERBEROASTING is only valid at DC endgame (handled above) ---
            # --- Impact actions only valid at endgame (handled above) ---
            # Everything else stays False (not valid in this phase)

        # Safety fallback — should never trigger with correct logic
        if not any(mask):
            mask[:] = True

        return mask

    # ------------------------------------------------------------------
    # Internal helper: mark host as detected and flag network detection
    # ------------------------------------------------------------------
    def _trigger_detection(self, host_name=None):
        """Called whenever an action is blocked by the defense."""
        self.state.detection_triggered = True
        if host_name and host_name in self.state.hosts:
            self.state.hosts[host_name].detected = True
        else:
            # Generic detection: mark the most recently compromised host
            for name in reversed(self.host_names):
                if self.state.hosts[name].compromised:
                    self.state.hosts[name].detected = True
                    break

    def _apply_action(self, action_idx):
        reward = -0.1
        terminated = False
        action = ACTION_LIST[action_idx]
        TARGET_URL = "http://localhost:5000"
        request_success = False

        # Track noise (CLEAR_LOGS resets, others add)
        if action.name != "CLEAR_LOGS":
            self.alert_level = max(0.0, self.alert_level + action.noise_level)

        # --- EXECUTION PHASE ---
        if self.real_mode:
            try:
                resp = None
                if action.name == "PHISHING_EMAIL":
                    resp = requests.post(f"{TARGET_URL}/phish", json={"p": "1"}, timeout=1)
                elif action.name == "BRUTE_FORCE_SSH":
                    resp = requests.post(f"{TARGET_URL}/brute", timeout=1)
                elif action.name == "NETWORK_SCAN":
                    resp = requests.post(f"{TARGET_URL}/scan", timeout=1)
                elif action.name == "VALID_ACCOUNTS_LOGIN":
                    resp = requests.post(f"{TARGET_URL}/login", headers=AUTH_HEADERS, timeout=1)
                elif action.name == "INSTALL_BACKDOOR":
                    resp = requests.post(f"{TARGET_URL}/backdoor", headers=AUTH_HEADERS, timeout=1)
                elif action.name == "CLEAR_LOGS":
                    resp = requests.post(f"{TARGET_URL}/clean", timeout=1)
                elif action.name == "LATERAL_MOVE_SMB":
                    resp = requests.get(f"{TARGET_URL}/status", timeout=1)
                elif action.name == "PRIV_ESC_SUDO":
                    resp = requests.post(f"{TARGET_URL}/privesc", json={"exploit": "1"}, headers=AUTH_HEADERS, timeout=1)
                elif action.name == "EXFILTRATE_DATA":
                    resp = requests.post(f"{TARGET_URL}/exfil", headers=AUTH_HEADERS, timeout=1)
                elif action.name == "RANSOMWARE_ENCRYPT":
                    resp = requests.post(f"{TARGET_URL}/ransom", headers=AUTH_HEADERS, timeout=1)
                elif action.name == "SQL_INJECTION":
                    if not self.sqli_available:
                        return -3.0, False
                    resp = requests.post(f"{TARGET_URL}/sqli",
                                         json={"query": "' OR '1'='1' --"},
                                         timeout=1)
                elif action.name == "PASS_THE_HASH":
                    resp = requests.post(f"{TARGET_URL}/pth", headers=AUTH_HEADERS, timeout=1)
                elif action.name == "POWERSHELL_EXEC":
                    resp = requests.post(f"{TARGET_URL}/psexec", headers=AUTH_HEADERS, timeout=1)
                elif action.name == "KERBEROASTING":
                    resp = requests.post(f"{TARGET_URL}/kerberoast", headers=AUTH_HEADERS, timeout=1)
                elif action.name == "WEB_SHELL_UPLOAD":
                    resp = requests.post(f"{TARGET_URL}/webshell", timeout=1)

                if resp and resp.status_code == 403:
                    self._trigger_detection()
                    return -50.0, True
                if resp and resp.status_code == 200:
                    request_success = True
                if action.name == "LATERAL_MOVE_SMB" and resp and resp.status_code != 403:
                    request_success = True
            except Exception:
                request_success = False
        else:
            # Sim mode: check sqli_available first, then probabilistic defense
            if action.name == "SQL_INJECTION" and not self.sqli_available:
                return -3.0, False
            defense_block_chance = self.alert_level * 0.2
            if random.random() > defense_block_chance:
                if random.random() < action.success_prob:
                    request_success = True
            else:
                request_success = False
                if self.alert_level > 3.0:
                    self._trigger_detection()
                    return -50.0, True

        # --- LOGIC PHASE ---

        dc_host = self.state.hosts["dc01"]
        is_dc_compromised = dc_host.compromised
        is_dc_root = dc_host.privileged

        # === DC ENDGAME — Privilege Escalation ===
        # Three valid paths: PRIV_ESC_SUDO, POWERSHELL_EXEC, KERBEROASTING
        if is_dc_compromised and not is_dc_root:
            if action.name in VALID_DC_PRIVESC:
                if request_success:
                    dc_host.privileged = True
                    reward += 100.0
                # else: no penalty — let the agent retry
            else:
                reward -= 20.0   # Heavy penalty for non-privesc action at DC
            return reward, terminated

        elif is_dc_root:
            if action.name in {"EXFILTRATE_DATA", "RANSOMWARE_ENCRYPT"}:
                if request_success:
                    reward += 500.0
                    terminated = True
            elif action.name == "CLEAR_LOGS" and self.alert_level > 0.5:
                # Smart CLEAR_LOGS before exfil — agent is reducing noise before the final step
                if self.real_mode:
                    try:
                        requests.post(f"{TARGET_URL}/clean", timeout=1)
                    except Exception:
                        pass
                self.alert_level = 0.0
                reward += 15.0   # Rewarded more than in early phase — stealth before exfil is critical
            else:
                reward -= 20.0   # Force exit action
            return reward, terminated


        # === STANDARD ACTION LOGIC ===

        if action.name == "PHISHING_EMAIL":
            if not self.state.hosts["user01"].compromised:
                if request_success:
                    self.state.hosts["user01"].compromised = True
                    reward += 10.0
            else:
                reward -= 2.0

        elif action.name == "BRUTE_FORCE_SSH":
            target = None
            for name in self.host_names:
                if not self.state.hosts[name].compromised:
                    target = name
                    break
            if target and request_success:
                self.state.hosts[target].compromised = True
                reward += 5.0
            else:
                reward -= 2.0   # Reduced penalty (was -5) — makes it a viable noisy option

        elif action.name == "NETWORK_SCAN":
            is_inside = any(h.compromised for h in self.state.hosts.values())
            if is_inside and not self.network_scanned:
                if request_success:
                    self.network_scanned = True
                    reward += 15.0
            else:
                reward -= 10.0   # Penalise spamming scan

        elif action.name == "VALID_ACCOUNTS_LOGIN":
            target = None
            for name in self.host_names:
                h = self.state.hosts[name]
                if h.compromised and not h.has_credentials:
                    target = name
                    break
            if target and request_success:
                self.state.hosts[target].has_credentials = True
                reward += 20.0
            elif not target:
                reward -= 2.0

        elif action.name == "INSTALL_BACKDOOR":
            is_inside = any(h.compromised for h in self.state.hosts.values())
            if is_inside and not self.backdoor_installed:
                if request_success:
                    self.backdoor_installed = True
                    reward += 15.0
            else:
                reward -= 10.0   # Already installed

        elif action.name == "CLEAR_LOGS":
            if self.alert_level > 0.5:
                if self.real_mode:
                    try:
                        requests.post(f"{TARGET_URL}/clean", timeout=1)
                    except Exception:
                        pass
                reward += 10.0
                self.alert_level = 0.0
            else:
                reward -= 8.0   # Heavy penalty for useless clears

        elif action.name == "LATERAL_MOVE_SMB":
            # --- FIX: Enforce multi-hop lateral movement (user01 → srv01 → dc01) ---
            has_creds = any(h.has_credentials for h in self.state.hosts.values())
            if has_creds and self.network_scanned:
                new_victim = None
                if not self.state.hosts["srv01"].compromised:
                    new_victim = "srv01"    # First hop: must take srv01 before dc01
                elif not self.state.hosts["dc01"].compromised:
                    new_victim = "dc01"     # Second hop: only after srv01 is taken
                if new_victim:
                    self.state.hosts[new_victim].compromised = True
                    reward += 50.0 if new_victim == "dc01" else 30.0
                else:
                    reward -= 2.0
            else:
                reward -= 5.0

        elif action.name == "PRIV_ESC_SUDO":
            # Non-DC PrivEsc (DC endgame handled above)
            target = None
            for name in self.host_names:
                h = self.state.hosts[name]
                if h.compromised and not h.privileged and name != "dc01":
                    target = name
                    break
            if target and request_success:
                self.state.hosts[target].privileged = True
                reward += 10.0
            else:
                reward -= 5.0

        elif action.name == "SQL_INJECTION":
            user01 = self.state.hosts["user01"]
            if not user01.compromised:
                if request_success:
                    user01.compromised = True
                    user01.has_credentials = True   # SQLi gives immediate creds — better than phishing
                    reward += 30.0
                else:
                    reward -= 3.0
            else:
                reward -= 2.0

        elif action.name == "PASS_THE_HASH":
            # No scan required — differentiates it from LATERAL_MOVE_SMB
            has_creds = any(h.has_credentials for h in self.state.hosts.values())
            if has_creds:
                new_victim = None
                if not self.state.hosts["srv01"].compromised:
                    new_victim = "srv01"
                elif not self.state.hosts["dc01"].compromised:
                    new_victim = "dc01"
                if new_victim and request_success:
                    self.state.hosts[new_victim].compromised = True
                    reward += 40.0 if new_victim == "dc01" else 25.0
                elif not new_victim:
                    reward -= 2.0
                else:
                    reward -= 3.0
            else:
                reward -= 5.0

        elif action.name == "POWERSHELL_EXEC":
            # Non-DC remote execution (DC endgame handled above)
            target = None
            for name in self.host_names:
                h = self.state.hosts[name]
                if h.compromised and not h.privileged and name != "dc01":
                    target = name
                    break
            if target and request_success:
                self.state.hosts[target].privileged = True
                reward += 15.0
            else:
                reward -= 4.0

        elif action.name == "KERBEROASTING":
            # --- FIX: Kerberoasting is ONLY valid at DC endgame (handled above) ---
            # This branch is only reached if somehow called outside endgame — penalise.
            reward -= 5.0

        elif action.name == "WEB_SHELL_UPLOAD":
            # --- FIX: Web shell is ONLY a backdoor — it does NOT grant free network scan ---
            # Agent must still use NETWORK_SCAN explicitly to discover the internal network.
            is_compromised = any(h.compromised for h in self.state.hosts.values())
            if is_compromised and not self.backdoor_installed:
                if request_success:
                    self.backdoor_installed = True
                    # NOTE: network_scanned is intentionally NOT set here (removed shortcut)
                    reward += 25.0
                else:
                    reward -= 3.0
            else:
                reward -= 4.0

        return reward, terminated