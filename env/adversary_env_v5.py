# -*- coding: utf-8 -*-
"""Gymnasium wrapper for the v5 kill-chain model.

Differences from v4 that matter for the experiments:

  * the action is a (technique, host) pair flattened to one Discrete index, so
    the agent chooses where to act as well as what to do;
  * the observation carries per-host state rather than a global summary, plus
    the execution-result vector from Koo et al. (E-NASim, ETRI Journal 2026,
    Eq. 7) which reports *why* the previous action failed. Giving that signal to
    every agent, masked or not, is what makes the masking ablation fair: the
    unmasked baseline here is as well informed as an agent in their environment;
  * reward follows their decomposition, R = R_step(a) + R_goal(s') - cost(a),
    rather than v4's flat milestone table, so an agent is rewarded for strategic
    value and charged for operational cost.
"""
import os
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.kill_chain_v5 import (KillChainModelV5, TECHNIQUES_V5, TECHNIQUE_ORDER_V5,
                               P_ADMIN, RESULT_NAMES,
                               C_PASSWORD, C_NTLM, C_TICKET)

# ---- reward -----------------------------------------------------------------
# R_step: the strategic value of the action type itself, before its effect.
R_STEP_BY_TACTIC = {
    "Discovery": 1.0, "Initial Access": 4.0, "Credential Access": 4.0,
    "Lateral Movement": 5.0, "Privilege Escalation": 5.0, "Execution": 3.0,
    "Persistence": 3.0, "Defense Evasion": 1.0, "Command and Control": 3.0,
    "Collection": 4.0, "Exfiltration": 6.0, "Impact": 6.0,
}
# cost: operational expense / exposure risk, subtracted every attempt.
COST_BY_TACTIC = {
    "Discovery": 1.0, "Initial Access": 2.0, "Credential Access": 3.0,
    "Lateral Movement": 3.0, "Privilege Escalation": 4.0, "Execution": 2.0,
    "Persistence": 2.0, "Defense Evasion": 2.0, "Command and Control": 2.0,
    "Collection": 2.0, "Exfiltration": 3.0, "Impact": 3.0,
}
# Balancing this needed two corrections. Setting the milestones too high (12/7/6)
# let the agent earn as much by harvesting the estate as by finishing, and the
# policy oscillated. Cutting them hard while raising the capture penalty to -100
# was worse: doing nothing became the safest policy, because idling out the clock
# cost about -12 while a capture cost -100, so the agent stopped attacking
# entirely and success collapsed to zero.
#
# The settings below make inaction expensive. A full idle episode now costs
# roughly -40, comparable to a capture, so waiting is not a refuge; the whole
# estate is worth about 0.7 of the objective, so finishing still dominates
# harvesting; and the milestones remain large enough to guide exploration toward
# the objective in a 540-action space where random legal play succeeds 2.5% of
# the time.
R_GOAL_COMPROMISE = 8.0      # a host fell
R_GOAL_PRIVESC = 5.0         # privilege raised on a host
R_GOAL_CREDS = 4.0           # credentials obtained
R_GOAL_DISCOVER = 0.5        # per newly discovered host
R_WIN = 300.0
R_CAUGHT = -50.0             # costly, but not so costly that idling is safer
STEP_COST_SCALE = 0.3        # per-attempt cost multiplier; makes stalling expensive
R_ILLEGAL = 0.0              # set by config; the reward-shaping control uses <0


class AdversaryEnvV5(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        cfg = config or {}
        self.topology = cfg.get("topology", "enterprise")
        self.max_steps = int(cfg.get("max_steps", 60))
        self.illegal_penalty = float(cfg.get("illegal_penalty", 0.0))
        self._py_rng = random.Random(cfg.get("seed"))

        self.model = KillChainModelV5(self.topology, rng=self._py_rng)
        self.n_host = self.model.n_host
        self.n_tech = self.model.n_tech

        # Size-invariant mode. Without it the observation width is a function of
        # the host count, so a policy is tied to estates of exactly the size it
        # trained on. With pad_hosts=N the estate is laid into N slots, each
        # slot carries a present/absent flag, and every action addressing an
        # absent slot is masked out. The policy then sees one width whatever the
        # estate, which is what makes transfer across scale possible at all.
        # `padded` follows whether the caller ASKED for slots, not whether the
        # estate happens to fill them. An estate that exactly fills the slots
        # would otherwise fall back to the unpadded layout and present a
        # different observation width from its siblings, which silently breaks
        # the transfer it exists to support.
        self.model.exclude_tactics = frozenset(cfg.get("exclude_tactics") or ())
        self.model.exclude_techniques = frozenset(cfg.get("exclude_techniques") or ())
        self.model.objective = cfg.get("objective", "admin")
        requested = int(cfg.get("pad_hosts", 0))
        self.pad_hosts = requested or self.n_host
        if self.pad_hosts < self.n_host:
            raise ValueError(f"pad_hosts={self.pad_hosts} is smaller than the "
                             f"{self.n_host} hosts in {self.topology!r}")
        self.padded = bool(requested)

        self.action_space = spaces.Discrete(self.n_tech * self.pad_hosts)
        # 12 per slot when padding: the 11 host features plus a presence flag
        per_host = 12 if self.padded else 11
        self._obs_dim = (self.pad_hosts * per_host + 15 + self.n_tech
                         + len(RESULT_NAMES))
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(self._obs_dim,), dtype=np.float32)
        self.steps = 0

    # -- observation ----------------------------------------------------------
    def _obs(self):
        m = self.model
        v = []
        max_subnet = max(m.topo.subnets) or 1
        for name in m.topo.hosts:
            h = m.host(name)
            # Credential material is reported by kind. A policy that cannot see
            # whether it holds a hash, a ticket or a password cannot choose
            # between pass-the-hash, pass-the-ticket and an SMB logon, and those
            # are three different actions here rather than three names for one.
            v += [float(h.compromised), float(h.discovered),
                  float(h.holds(C_PASSWORD)), float(h.holds(C_NTLM)),
                  float(h.holds(C_TICKET)),
                  h.privilege / 2.0, float(h.backdoored),
                  float(h.enumerated), float(h.staged), float(h.archived),
                  h.subnet / max_subnet]
        if self.padded:
            # a present flag for every real slot, then empty slots
            v = [x for i in range(self.n_host)
                 for x in v[i * 11:(i + 1) * 11] + [1.0]]
            v += [0.0] * ((self.pad_hosts - self.n_host) * 12)
        v += [
            min(1.0, m.alert),
            float(m.scanned), float(m.domain_enumerated), float(m.krbtgt_hash),
            float(m.need_persist), float(m.need_hostpriv),
            float(m.need_collect), float(m.need_c2),
            float(m.collected), float(m.c2_established),
            float(m.defences_disabled), float(m.obfuscated),
            float(m.services_stopped), float(m.recovery_inhibited),
            self.steps / self.max_steps,
        ]
        # The availability vector v(omega): which techniques this engagement
        # offers. Without it in the observation the scenario is unobservable and
        # the agent could only discover a withdrawn technique by finding it
        # missing from the mask, which is the mistake that made the
        # host-privilege requirement unlearnable.
        v += [float(name in m.available) for name in TECHNIQUE_ORDER_V5]
        r = [0.0] * len(RESULT_NAMES)
        r[m.last_result] = 1.0          # E-NASim Eq. 7
        v += r
        return np.asarray(v, dtype=np.float32)

    # -- masking --------------------------------------------------------------
    def _pad_index(self, a):
        """Map a padded action index onto the model's own indexing."""
        ti, hi = divmod(int(a), self.pad_hosts)
        if hi >= self.n_host:
            return None                      # addresses an absent slot
        return ti * self.n_host + hi

    def action_masks(self):
        base = np.asarray(self.model.action_mask(), dtype=bool)
        if not self.padded:
            return base
        out = np.zeros(self.n_tech * self.pad_hosts, dtype=bool)
        for ti in range(self.n_tech):
            s = ti * self.pad_hosts
            out[s:s + self.n_host] = base[ti * self.n_host:(ti + 1) * self.n_host]
        if not out.any():
            out[0] = True
        return out

    # -- gym API --------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._py_rng.seed(seed)
        self.model.rng = self._py_rng
        self.model.reset()
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        m = self.model
        self.steps += 1
        if self.padded:
            mapped = self._pad_index(action)
            if mapped is None:
                # an action against a slot no host occupies. It is masked, so a
                # masked agent never reaches here; an unmasked one is charged
                # the step and told the target does not exist.
                self.model.last_result = 1          # ConnErr
                truncated = self.steps >= self.max_steps
                return (self._obs(), -1.0, False, truncated,
                        {"technique": "-", "target": "-", "legal": False,
                         "result": "conn_err", "success": False,
                         "advanced": False})
            action = mapped
        technique, target = m.decode(int(action))
        tech = TECHNIQUES_V5[technique]

        legal = m.legal(technique, target)
        before = self._snapshot()
        out = m.attempt(technique, target)
        reward = -COST_BY_TACTIC.get(tech.tactic, 2.0) * STEP_COST_SCALE

        if out.blocked:
            reward += self.illegal_penalty
        else:
            # Pay the strategic value of an action only when it actually changed
            # the world. Paying it on a successful dice roll alone creates a
            # free-money loop: a repeatable no-op -- re-running discovery on a
            # host already enumerated -- earns R_step and costs only the step
            # charge, so an agent can farm it indefinitely instead of finishing.
            # A trained policy found exactly that: it did genuine work for about
            # nineteen steps, then repeated SYSTEM_INFO_DISCOVERY on one host for
            # the remaining forty and never reached the objective. This mirrors
            # the CLEAR_LOGS reward-hacking exploit removed in v4, and matches
            # E-NASim's Eq. 10, where the goal term is granted on a state
            # transition rather than on execution.
            if out.success and out.advanced:
                reward += R_STEP_BY_TACTIC.get(tech.tactic, 1.0) * 0.5
            reward += self._goal_reward(before)

        terminated = False
        if m.caught():
            reward += R_CAUGHT
            terminated = True
        elif out.terminated and m.is_goal():
            reward += R_WIN
            terminated = True

        # An intrusion with no legal move left is over. Ending it here is what
        # keeps the mask honest: the alternative is handing the policy a
        # fabricated legal action, which would appear in the infeasible-action
        # rate the paper reports as zero.
        m.action_mask()
        truncated = self.steps >= self.max_steps or m.stuck
        info = {"technique": technique, "target": target, "legal": legal,
                "result": RESULT_NAMES[out.result_code], "success": out.success,
                # `advanced` distinguishes an action that changed the world from
                # one that executed but did nothing -- L-ARLPT calls the latter a
                # redundant action and reports it at 19% for its best method.
                "advanced": out.advanced}
        return self._obs(), float(reward), terminated, truncated, info

    # -- reward helpers -------------------------------------------------------
    def _snapshot(self):
        m = self.model
        return {n: (m.host(n).compromised, m.host(n).privilege,
                    m.host(n).has_credentials, m.host(n).discovered)
                for n in m.topo.hosts}

    def _goal_reward(self, before):
        m = self.model
        r = 0.0
        for n in m.topo.hosts:
            c0, p0, k0, d0 = before[n]
            h = m.host(n)
            if h.compromised and not c0:
                r += R_GOAL_COMPROMISE * h.asset_value
            if h.privilege > p0:
                r += R_GOAL_PRIVESC * h.asset_value
            if h.has_credentials and not k0:
                r += R_GOAL_CREDS
            if h.discovered and not d0:
                r += R_GOAL_DISCOVER
        return r
