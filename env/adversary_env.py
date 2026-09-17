import os
import random

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces

from env.kill_chain import KillChainModel, TECHNIQUES, ACTION_ORDER, INCIDENT_THRESHOLD
from telemetry.telemetry_logger import TelemetryLogger

# Bearer token must match the v4 server. Only used in real mode.
AUTH_HEADERS = {"Authorization": "Bearer rl-agent-secret-token-2026"}
TARGET_URL = os.environ.get("V4_TARGET_URL", "http://127.0.0.1:5000")

# Milestone rewards, keyed by the state change an action produces. Reward lives
# here, not in the shared model, because it is a training signal with no meaning
# to the server. The scale mirrors v3 so the two are broadly comparable, with
# one deliberate change: CLEAR_LOGS earns nothing. Its only value is lowering
# suspicion to protect the path to the +400 win, so a rational agent still
# clears when the risk warrants it -- but it is never paid to do so, which
# removes the reward-hacking exploit v3's free /clean invited.
R_STEP = -0.1            # per-step time pressure
R_FAIL = -1.0           # an attempt that neither advanced nor was caught
R_CAUGHT = -50.0        # incident response: the intrusion is burned
R_COMPROMISE = {"user01": 10.0, "srv01": 30.0, "dc01": 50.0}
R_CREDS = 20.0
R_SCAN = 15.0
R_BACKDOOR = 8.0
R_HOST_PRIV = 5.0
R_DC_ROOT = 100.0       # the objective state
R_WIN = 400.0           # successful exfiltration / ransomware


class AdversaryEnv(gym.Env):
    """Gym wrapper over the shared KillChainModel.

    In sim mode the model is stepped in process. In real mode the same actions
    are sent to the v4 server, which owns an identical model, and this env
    mirrors the returned state so the mask and observation are computed the same
    way in both modes. The transition logic itself is defined once, in
    env/kill_chain.py, so the two modes cannot drift.
    """

    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        config = config or {}

        self.host_names = list(KillChainModel.HOSTS)
        self.model = KillChainModel()
        # restrict the technique catalogue (e.g. to the original 15) so that
        # catalogue size can be varied with the method held fixed
        restrict = config.get("restrict_techniques")
        self.model.restrict = set(restrict) if restrict else None

        self.action_space = spaces.Discrete(len(ACTION_ORDER))
        # 9 host flags (3x3) + scan + backdoor(any) + alert + per-episode
        # availability of every substitutable technique + 2 objective-requirement
        # flags (persistence / host-elevation needed), so the agent can see what
        # the scenario permits and requires, and adapt its kill chain to it
        self._obs_dim = 12 + len(KillChainModel.AVAIL_ORDER) + 2
        self.observation_space = spaces.Box(low=0, high=1, shape=(self._obs_dim,), dtype=np.float32)

        self.max_steps = config.get("max_steps", 40)
        self.current_step = 0
        self.real_mode = config.get("real_mode", False)
        self._fixed_sqli = config.get("sqli_available", None)  # for reproducible probes
        # Optional v3-style reward penalty for taking an action the mask would
        # forbid. The masked agent never takes one (so it is unaffected); this
        # exists to TEACH an unmasked baseline to avoid illegal moves, the way v3
        # did, so the masked-vs-unmasked comparison can be made fair across
        # versions. 0.0 (default) = no such teaching, the unmasked agent only
        # learns legality implicitly.
        self.illegal_penalty = float(config.get("illegal_penalty", 0.0))

        self.rng = random.Random()
        self._session = None   # server session token, real mode only
        # a persistent HTTP session: keep-alive connection reuse turns real-mode
        # steps from ~1s each (new connection per call) into ~2ms, which is what
        # makes training directly against the live server feasible
        self._http = requests.Session() if self.real_mode else None
        telemetry_dir = config.get("telemetry_dir", "results/telemetry/rl")
        self.telemetry = TelemetryLogger(output_dir=telemetry_dir)

    # ── compatibility surface (external code reads these) ─────────────────
    @property
    def state(self):
        return self.model.state

    @property
    def alert_level(self):
        return self.model.alert

    @alert_level.setter
    def alert_level(self, v):
        self.model.alert = v

    @property
    def network_scanned(self):
        return self.model.scanned

    @property
    def backdoor_installed(self):
        return any(self.model.backdoor.values())

    @property
    def sqli_available(self):
        return self.model.sqli_available

    @sqli_available.setter
    def sqli_available(self, v):
        self.model.sqli_available = bool(v)

    # ── gym API ───────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        self.current_step = 0
        self.model.reset(sqli_available=self._fixed_sqli, rng=self.rng)
        self.telemetry.start_episode()

        if self.real_mode:
            self._server_reset()

        return self._get_obs(), {}

    def step(self, action_idx):
        self.current_step += 1
        name = ACTION_ORDER[int(action_idx)]
        # whether this action was legal (mask would have allowed it), computed
        # from the state before the action -- used only for the optional
        # v3-style illegal-action penalty
        was_legal = bool(self.model.legal_mask()[int(action_idx)])
        before = self._reward_snapshot()

        if self.real_mode:
            out = self._server_attempt(name)
        else:
            out = self.model.attempt(name, self.rng)

        after = self._reward_snapshot()
        reward = self._reward(name, out, before, after, was_legal)

        terminated = out.terminated
        truncated = self.current_step >= self.max_steps

        self.telemetry.log_event(
            action=name, state_snapshot=self.model.state,
            reward=reward, step=self.current_step)
        if terminated or truncated:
            self.telemetry.end_episode()

        info = {"detected": out.detected, "caught": out.detected,
                "success": out.success, "suspicion": self.model.alert}
        return self._get_obs(), reward, terminated, truncated, info

    def action_masks(self):
        return np.array(self.model.legal_mask(), dtype=bool)

    def _get_obs(self):
        obs = []
        for name in self.host_names:
            h = self.model.host(name)
            obs += [1.0 if h.compromised else 0.0,
                    1.0 if h.privileged else 0.0,
                    1.0 if h.has_credentials else 0.0]
        obs.append(1.0 if self.model.scanned else 0.0)
        obs.append(1.0 if any(self.model.backdoor.values()) else 0.0)
        obs.append(min(self.model.alert / INCIDENT_THRESHOLD, 1.0))
        # per-episode availability of every substitutable technique, then the
        # objective-requirement flags (persistence / host elevation needed)
        obs.extend(self.model.avail_vector())
        obs.extend(self.model.requirement_vector())
        return np.array(obs, dtype=np.float32)

    # ── reward ────────────────────────────────────────────────────────────
    def _reward_snapshot(self):
        h = self.model.host
        return {
            n: (h(n).compromised, h(n).has_credentials, h(n).privileged)
            for n in self.host_names
        } | {"scan": self.model.scanned,
             "backdoor": any(self.model.backdoor.values())}

    def _reward(self, name, out, before, after, was_legal=True):
        if out.detected:
            return R_CAUGHT
        # v3-style teaching signal (only when enabled): an action the mask would
        # forbid is penalised, so an unmasked baseline learns to avoid illegal
        # moves. The masked agent never triggers this branch.
        if not was_legal and self.illegal_penalty:
            return R_STEP + self.illegal_penalty
        r = R_STEP
        if not out.success:
            return r + R_FAIL
        # award by what actually changed this step
        for n in self.host_names:
            was_c, was_cr, was_p = before[n]
            now_c, now_cr, now_p = after[n]
            if now_c and not was_c:
                r += R_COMPROMISE[n]
            if now_cr and not was_cr:
                r += R_CREDS
            if now_p and not was_p:
                r += R_DC_ROOT if n == "dc01" else R_HOST_PRIV
        if after["scan"] and not before["scan"]:
            r += R_SCAN
        if after["backdoor"] and not before["backdoor"]:
            r += R_BACKDOOR
        if out.terminated and TECHNIQUES[name].terminal:
            r += R_WIN
        return r

    # ── real mode: talk to the v4 server, mirror its state ────────────────
    def _headers(self):
        return {"Authorization": f"Bearer {self._session}"} if self._session else {}

    def _ensure_session(self):
        if self._session is None:
            resp = self._http.post(f"{TARGET_URL}/session", timeout=5.0)
            self._session = resp.json()["session"]

    def _server_reset(self):
        try:
            self._ensure_session()
            # send the full per-episode scenario the env just rolled; the server
            # applies it verbatim so both run the identical world (availability,
            # objective requirements and sqli), making drift impossible
            resp = self._http.post(f"{TARGET_URL}/reset",
                                   json={"scenario": self.model.scenario_dict(),
                                         "sqli_available": self.model.sqli_available},
                                   headers=self._headers(), timeout=5.0)
            self._sync(resp.json().get("state"))
        except Exception:
            self._session = None   # force a fresh session next episode

    def _server_attempt(self, name):
        from env.kill_chain import Outcome
        try:
            resp = self._http.post(f"{TARGET_URL}/attempt",
                                   json={"technique": name},
                                   headers=self._headers(), timeout=5.0)
            data = resp.json()
            self._sync(data.get("state"))
            return Outcome(attempted=True, blocked=data.get("blocked", False),
                           success=data.get("success", False),
                           advanced=data.get("advanced", False),
                           detected=data.get("detected", False),
                           terminated=data.get("terminated", False))
        except Exception:
            return Outcome(attempted=True)

    def _sync(self, state):
        """Overwrite the local model mirror from a server state snapshot."""
        if not state:
            return
        for n in self.host_names:
            hs = state["hosts"][n]
            h = self.model.host(n)
            h.compromised = hs["compromised"]
            h.has_credentials = hs["has_credentials"]
            h.privileged = hs["privileged"]
            h.detected = hs["detected"]
        self.model.scanned = state["scanned"]
        self.model.backdoor = dict(state["backdoor"])
        self.model.alert = state["suspicion"]
        self.model.sqli_available = state["sqli_available"]
        self.model.locked_out = state.get("locked_out", False)
