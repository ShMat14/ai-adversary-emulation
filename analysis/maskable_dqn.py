# -*- coding: utf-8 -*-
"""A maskable DQN, because the literature does not ship one.

WHY THIS EXISTS

Every masked result in this paper is produced by MaskablePPO, and every unmasked
baseline is a different algorithm. That leaves a question we could not answer:
is the benefit we measure a property of *masking*, or a property of *PPO*? The
comparison as it stood confounded the two, and our own text admitted it by
explaining that DQN and A2C are unmasked "because no masked variant exists in
the standard implementations". That is a statement about Stable-Baselines3, not
about the mechanism, and it is not a good enough reason.

So we implement one. Masking a value-based method is not the same operation as
masking a policy-gradient one, and the difference is worth stating because it is
where an implementation usually goes wrong:

  * A policy-gradient method masks the *distribution*. Setting the logits of
    illegal actions to -inf removes them from the categorical, and the gradient
    of an action that cannot be sampled is zero, so nothing else is needed.

  * A value-based method has no distribution to mask. It has a greedy argmax and
    a bootstrap target, and BOTH must be masked. Masking only action selection is
    the mistake: the target still bootstraps through
    max_a Q(s', a) over *all* actions, so the network keeps learning the value of
    a state from an action that state does not admit, and the illegal action goes
    on contaminating every value that backs up through it.

The second point is why the replay buffer here stores the legality mask of the
*next* state alongside the transition. Without it the target cannot be masked at
all, because by sample time the environment is long past that state.

WHAT IS MASKED

  selection   epsilon-greedy draws uniformly from the legal set, and the greedy
              branch takes the argmax over legal Q-values only
  target      max_a' Q_target(s', a') is taken over the legal set of s'
  gradient    unchanged: the Huber loss is on the action actually taken, which
              was legal by construction

Everything else is Stable-Baselines3 DQN, unmodified, so a difference against the
unmasked baseline is attributable to the mask and not to a second change made at
the same time.

    from analysis.maskable_dqn import MaskableDQN
    model = MaskableDQN("MlpPolicy", env, ...)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from torch.nn import functional as F

NEG = -1e8          # a finite stand-in for -inf, so masked_fill cannot produce NaN


class MaskableReplayBuffer(ReplayBuffer):
    """A replay buffer that also remembers which actions the next state allowed.

    The mask arrives through `infos`, which is the one channel Stable-Baselines3
    passes from the algorithm to `add()` without a signature change.

    Restricted to a single environment on purpose. The parent's `_get_samples`
    draws its own random `env_indices` and does not return them, so with more
    than one environment there is no way to line the masks up with the rows that
    were sampled. Rather than duplicate the parent's sampling logic to recover
    them, we assert the case we actually use.
    """

    def __init__(self, buffer_size, observation_space, action_space,
                 device="auto", n_envs=1, optimize_memory_usage=False,
                 handle_timeout_termination=True):
        super().__init__(buffer_size, observation_space, action_space, device,
                         n_envs=n_envs, optimize_memory_usage=optimize_memory_usage,
                         handle_timeout_termination=handle_timeout_termination)
        assert n_envs == 1, "MaskableReplayBuffer supports a single environment"
        assert isinstance(action_space, spaces.Discrete)
        self.n_actions = int(action_space.n)
        # Default True: a transition stored before any mask was seen is treated
        # as fully legal, which is the parent's behaviour and cannot mask out a
        # legal action by accident.
        self.next_masks = np.ones((self.buffer_size, self.n_envs, self.n_actions),
                                  dtype=bool)
        self.last_next_masks: th.Tensor | None = None

    def add(self, obs, next_obs, action, reward, done, infos):
        rows = []
        for info in infos:
            m = info.pop("next_action_mask", None)
            rows.append(np.ones(self.n_actions, dtype=bool) if m is None
                        else np.asarray(m, dtype=bool))
        self.next_masks[self.pos] = np.stack(rows)
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        data = super()._get_samples(batch_inds, env=env)
        # Stashed rather than returned, so ReplayBufferSamples keeps its shape.
        # Safe because sampling and the update that consumes it happen back to
        # back on one thread, in `MaskableDQN.train`.
        self.last_next_masks = th.as_tensor(
            self.next_masks[batch_inds, 0], device=self.device)
        return data


class MaskableDQN(DQN):
    """DQN with action masking applied to selection and to the bootstrap target."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("replay_buffer_class", MaskableReplayBuffer)
        super().__init__(*args, **kwargs)

    # -- masks from the environment -------------------------------------------
    def _env_masks(self) -> np.ndarray:
        """Legality of the state the training environment is currently in."""
        return np.asarray(self.env.env_method("action_masks"), dtype=bool)

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward,
                          dones, infos):
        # Called immediately after the environment step, so the environment now
        # stands in the successor state and its mask is the one the target needs.
        # On a terminal step the vectorised environment has already auto-reset and
        # this is the next episode's mask; that row is multiplied by (1 - done) in
        # the target, so it never contributes.
        masks = self._env_masks()
        for i, info in enumerate(infos):
            info["next_action_mask"] = masks[i]
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward,
                                  dones, infos)

    # -- masked action selection ----------------------------------------------
    def predict(self, observation, state=None, episode_start=None,
                deterministic: bool = False, action_masks: np.ndarray | None = None):
        """Epsilon-greedy over the legal set only.

        `action_masks` mirrors MaskablePPO's signature so evaluation code can
        pass the mask explicitly. During training it is omitted and read from the
        environment.
        """
        if action_masks is None:
            # Only the training loop may omit the mask: there `self.env` is by
            # definition the environment being stepped. At evaluation time the
            # model is usually loaded without an env, and even when one is
            # attached it is the training environment rather than the one being
            # evaluated -- reading masks from it would be silently wrong, which
            # is worse than failing. So fail, and say what to pass.
            if self.env is None:
                raise ValueError(
                    "MaskableDQN.predict needs action_masks when the model has no "
                    "attached environment. Pass action_masks=env.action_masks().")
            action_masks = self._env_masks()
        masks = np.asarray(action_masks, dtype=bool)
        if masks.ndim == 1:
            masks = masks[None, :]

        if not deterministic and np.random.rand() < self.exploration_rate:
            # Uniform over what is legal. Sampling the full action space here
            # would reintroduce exactly the illegal selections the mask exists to
            # remove, and would do it at the exploration rate.
            actions = np.array([
                np.random.choice(np.flatnonzero(m)) if m.any() else 0
                for m in masks
            ])
        else:
            obs_tensor, _ = self.policy.obs_to_tensor(observation)
            with th.no_grad():
                q = self.q_net(obs_tensor).cpu().numpy()
            q = np.where(masks, q, NEG)
            actions = q.argmax(axis=1)

        if not self.policy.is_vectorized_observation(observation):
            actions = actions.squeeze(axis=0)
        return actions, state

    # -- masked bootstrap target ----------------------------------------------
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)
            next_masks = self.replay_buffer.last_next_masks
            discounts = (replay_data.discounts
                         if getattr(replay_data, "discounts", None) is not None
                         else self.gamma)

            with th.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                # The whole point. Without this the target bootstraps through
                # actions the successor state does not admit.
                next_q_values = next_q_values.masked_fill(~next_masks, NEG)
                next_q_values, _ = next_q_values.max(dim=1)
                # A state with nothing legal ends the episode, so its row is
                # zeroed by (1 - done); clamping keeps the sentinel out of the
                # loss if a buffer row ever disagrees.
                next_q_values = th.clamp(next_q_values, min=NEG / 2)
                next_q_values = th.where(next_q_values <= NEG / 2,
                                         th.zeros_like(next_q_values),
                                         next_q_values)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = (replay_data.rewards
                                   + (1 - replay_data.dones) * discounts * next_q_values)

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1,
                                         index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
