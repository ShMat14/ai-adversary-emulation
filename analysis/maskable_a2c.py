# -*- coding: utf-8 -*-
"""A maskable A2C, to close the masked/unmasked grid over both algorithm families.

WHY THIS IS THE EASY ONE

Masking A2C is not the same kind of exercise as masking DQN, and the paper should
not pretend it is. A2C and PPO are both on-policy policy-gradient methods, and in
both the mask is applied to the *distribution*: the logits of illegal actions go
to negative infinity, they cannot be sampled, and an action that cannot be sampled
contributes no gradient. That is the same mechanism MaskablePPO already uses, so
this configuration probes the same idea in a second implementation rather than a
second idea. The value-based case in `maskable_dqn.py` is where masking actually
had to be reasoned about again, because a bootstrap target has no distribution.

We include it anyway for one reason: without it the paper masks one of its two
policy-gradient baselines and not the other, which invites the question of what
the unmasked one would have done. Now it is answered rather than argued.

WHAT IS INHERITED AND WHAT IS NOT

We inherit sb3-contrib's MaskablePPO for the masking machinery -- the maskable
rollout buffer, the maskable policy and the masked action distribution are all
already correct there, and reimplementing them would add risk without adding
anything. We override only `train`, replacing PPO's clipped multi-epoch update
with A2C's single full-batch update, which is A2C's actual defining difference.

ONE TRAP WORTH NAMING

Stable-Baselines3's A2C uses **RMSprop**, while PPO and MaskablePPO use Adam. A
subclass that inherited the optimiser as well as the masking would differ from
the unmasked A2C baseline in two things at once, and the comparison would no
longer isolate the mask. We therefore reproduce SB3's A2C optimiser setup exactly,
so that MaskableA2C against A2C varies the mask and nothing else.

    from analysis.maskable_a2c import MaskableA2C
"""
from __future__ import annotations

import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F


class MaskableA2C(MaskablePPO):
    """A2C's update rule with MaskablePPO's masking machinery.

    `n_epochs`, `batch_size` and `clip_range` are inherited from the parent's
    signature but are not used: `train` below takes a single pass over the whole
    rollout with no clipping, which is what makes this A2C and not PPO.
    """

    def __init__(self, *args, normalize_advantage: bool = False,
                 use_rms_prop: bool = True, rms_prop_eps: float = 1e-5, **kwargs):
        # Match Stable-Baselines3's A2C optimiser, not PPO's. Without this the
        # masked and unmasked A2C rows would differ in the optimiser as well as
        # the mask, and the pair would stop being an ablation.
        if use_rms_prop:
            pk = dict(kwargs.get("policy_kwargs") or {})
            pk.setdefault("optimizer_class", th.optim.RMSprop)
            pk.setdefault("optimizer_kwargs",
                          dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0))
            kwargs["policy_kwargs"] = pk
        # A2C takes one pass over the rollout; the parent needs a legal batch
        # size, and `train` ignores it.
        kwargs.setdefault("n_epochs", 1)
        super().__init__(*args, **kwargs)
        self.normalize_advantage = normalize_advantage

    def train(self) -> None:
        """A2C: one gradient step over the whole rollout, no clipping."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        # batch_size=None yields the entire rollout in one go, which is the
        # difference that makes this A2C.
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()

            # The masks travel with the rollout, so the evaluated distribution is
            # the masked one and illegal actions carry no probability mass.
            values, log_prob, entropy = self.policy.evaluate_actions(
                rollout_data.observations, actions,
                action_masks=rollout_data.action_masks)
            values = values.flatten()

            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -(advantages * log_prob).mean()
            value_loss = F.mse_loss(rollout_data.returns, values)
            entropy_loss = (-th.mean(-log_prob) if entropy is None
                            else -th.mean(entropy))

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                           self.rollout_buffer.returns.flatten())
        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
