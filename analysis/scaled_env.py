# -*- coding: utf-8 -*-
"""
Widen the action space to test whether masking matters at ATT&CK scale.

The controlled comparison in Section 4.6 found masking made no difference to
outcomes over fifteen techniques. The obvious question is whether that holds
when the catalogue is the size of ATT&CK Enterprise rather than a fifteen-item
subset, since roughly seven of the fifteen are legal at any step here, whereas
a full catalogue would leave the great majority inapplicable in any given state.

This wrapper leaves AdversaryEnv untouched and presents a larger action space
on top of it. The first fifteen actions are the real ones and are delegated
unchanged. The remainder are real ATT&CK Enterprise techniques that cannot
apply to a three-host Windows domain -- see analysis/attack_catalogue.py, which
names each one and records why. They are never legal, and selecting one costs a
small penalty and raises the alert level, which is what attempting an
inapplicable technique against a real target would do.

These padding actions are catalogued, not instantiated: naming them does not
mean the environment implements them. No result depends on their semantics,
since every one of them is illegal in every state and carries the same penalty.
They exist to vary one quantity -- the proportion of the action space that is
legal at a given step -- while holding the environment, the reward structure and
the kill chain identical.

Naming them does change what can be claimed about the experiment. At sixty
actions the padding is drawn entirely from techniques that are impossible
against this target because the required asset does not exist (no cloud, no
containers, no network appliances, no macOS or Linux hosts, no physical access).
Only the two-hundred condition reaches into techniques that are Windows-
applicable but outside this environment's abstraction.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.adversary_env import AdversaryEnv
from env.attack_actions import ACTION_LIST
from analysis.attack_catalogue import padding_for, composition

REAL_N = len(ACTION_LIST)

# cost of attempting a technique that cannot apply in the current state:
# a small negative reward, and the alert increase any noisy attempt would cause
INAPPLICABLE_REWARD = -1.0
INAPPLICABLE_NOISE = 0.15


class ScaledAdversaryEnv(gym.Env):
    """AdversaryEnv with an action space padded to `total_actions`."""

    metadata = {"render_modes": []}

    def __init__(self, total_actions=REAL_N, config=None):
        super().__init__()
        if total_actions < REAL_N:
            raise ValueError(f"total_actions must be >= {REAL_N}")
        self.inner = AdversaryEnv(config=config or {})
        self.total_actions = total_actions
        self.action_space = spaces.Discrete(total_actions)
        self.observation_space = self.inner.observation_space
        # identities only: the padding techniques are reported, never executed
        self.padding = padding_for(total_actions - REAL_N)

    # ── gym API ──────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        return self.inner.reset(seed=seed, options=options)

    def step(self, action):
        action = int(action)
        if action < REAL_N:
            return self.inner.step(action)

        # an inapplicable technique: costs alert and a small penalty, and
        # consumes a step, but changes no host state
        self.inner.current_step += 1
        self.inner.alert_level = max(0.0, self.inner.alert_level + INAPPLICABLE_NOISE)
        truncated = self.inner.current_step >= self.inner.max_steps
        return self.inner._get_obs(), INAPPLICABLE_REWARD, False, truncated, {}

    def action_masks(self):
        real = np.asarray(self.inner.action_masks(), dtype=bool)
        if self.total_actions == REAL_N:
            return real
        pad = np.zeros(self.total_actions - REAL_N, dtype=bool)
        return np.concatenate([real, pad])

    # ── passthrough for the evaluation harness ───────────────────────
    @property
    def state(self):
        return self.inner.state

    @property
    def current_step(self):
        return self.inner.current_step

    @property
    def telemetry(self):
        return self.inner.telemetry

    @telemetry.setter
    def telemetry(self, value):
        self.inner.telemetry = value

    def legal_fraction(self):
        """Share of the action space currently legal, for reporting."""
        m = self.action_masks()
        return float(m.sum()) / len(m)

    # ── identity of the action space, for the write-up ───────────────
    def technique_at(self, action):
        """The ATT&CK technique an action index stands for.

        Returns (id, name) for both halves of the action space, so an episode
        log reads as techniques rather than as integers.
        """
        action = int(action)
        if action < REAL_N:
            a = ACTION_LIST[action]
            return a.mitre_technique, a.name
        t = self.padding[action - REAL_N]
        return t.id, t.name

    def composition(self):
        """Counts of implemented / structurally impossible / unmodelled actions."""
        return composition(self.total_actions, REAL_N)
