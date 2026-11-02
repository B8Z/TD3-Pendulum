"""TD3 for Pendulum-v1: replay, bounded actions, twin critics, and delayed updates."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn


def tensor(values):
    """Use the same CPU float32 representation throughout the agent."""
    return torch.as_tensor(values, dtype=torch.float32, device="cpu")


def optimize(loss, optimizer, parameters):
    """Clear accumulated gradients, differentiate, and reject invalid updates."""
    if not torch.isfinite(loss):
        raise FloatingPointError("The training loss is non-finite.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("A training gradient is non-finite.")
    optimizer.step()


class ReplayBuffer:
    """A fixed-size ring buffer of (state, action, reward, next state, end flags)."""

    def __init__(self, capacity=50_000, seed=0):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive.")
        self.capacity, self.position, self.size = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        widths = {
            "observations": 3, "actions": 1, "rewards": 1,
            "next_observations": 3, "terminated": 1, "truncated": 1,
        }
        self.data = {
            name: np.empty((capacity, width), dtype=np.float32)
            for name, width in widths.items()
        }

    def __len__(self):
        return self.size

    def add(self, observation, action, reward, next_observation, terminated, truncated):
        observation = np.asarray(observation, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        if observation.shape != (3,) or action.shape != (1,) or next_observation.shape != (3,):
            raise ValueError("Expected observation (3,), action (1,), next observation (3,).")
        values = {
            "observations": observation, "actions": action, "rewards": float(reward),
            "next_observations": next_observation,
            "terminated": float(terminated), "truncated": float(truncated),
        }
        if not all(np.isfinite(value).all() for value in values.values()):
            raise FloatingPointError("A transition contains a non-finite value.")
        # Assignment copies the values, including the endpoint before an episode reset.
        for name, value in values.items():
            self.data[name][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size == 0 or batch_size < 1:
            raise ValueError("Sampling requires stored experience and a positive batch size.")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return {name: values[indices] for name, values in self.data.items()}


class Actor(nn.Module):
    """Map three observation features to one torque in [-2, 2] N m."""

    def __init__(self, hidden_size=64):
        super().__init__()
        # Layer names and order match the published actor checkpoint.
        self.network = nn.Sequential(
            nn.Linear(3, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1), nn.Tanh(),
        )

    def forward(self, observations):
        return 2.0 * self.network(observations)

    @torch.no_grad()
    def act(self, observation):
        """Convert one observation into an independent, noise-free action array."""
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("Expected a finite observation with shape (3,).")
        action = self(tensor(values).unsqueeze(0)).squeeze(0).numpy().copy()
        if not np.isfinite(action).all():
            raise FloatingPointError("The actor produced a non-finite action.")
        return action


class Critic(nn.Module):
    """Estimate discounted return from three state features and one action."""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, observations, actions):
        return self.network(torch.cat((observations, actions), dim=1))


class TD3:
    """Own online/target networks and apply the three TD3 learning mechanisms."""

    def __init__(
        self, hidden_size=64, gamma=0.99, target_noise_std=0.4, noise_clip=1.0,
        critic_learning_rate=0.0003, actor_learning_rate=0.0003,
        policy_delay=2, tau=0.005,
    ):
        if not 0 <= gamma < 1 or not 0 < tau <= 1:
            raise ValueError("Require 0 <= gamma < 1 and 0 < tau <= 1.")
        if type(policy_delay) is not int or policy_delay < 1:
            raise ValueError("Policy delay must be a positive integer.")
        if (
            not np.isfinite([target_noise_std, noise_clip]).all()
            or min(target_noise_std, noise_clip) < 0
        ):
            raise ValueError("Target noise settings must be finite and nonnegative.")
        if (
            not np.isfinite([critic_learning_rate, actor_learning_rate]).all()
            or min(critic_learning_rate, actor_learning_rate) <= 0
        ):
            raise ValueError("Learning rates must be finite and positive.")
        self.gamma, self.tau = gamma, tau
        self.target_noise_std, self.noise_clip = target_noise_std, noise_clip
        self.policy_delay = policy_delay

        # Preserve initialization order and give targets separate, frozen storage.
        self.actor = Actor(hidden_size).to("cpu")
        self.critic_one = Critic(hidden_size).to("cpu")
        self.critic_two = Critic(hidden_size).to("cpu")
        self.actor_target = deepcopy(self.actor).requires_grad_(False).eval()
        self.critic_one_target = deepcopy(self.critic_one).requires_grad_(False).eval()
        self.critic_two_target = deepcopy(self.critic_two).requires_grad_(False).eval()
        self.critic_parameters = (
            list(self.critic_one.parameters()) + list(self.critic_two.parameters())
        )
        self.actor_parameters = list(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=critic_learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=actor_learning_rate)
        self.critic_updates = self.actor_updates = self.target_updates = 0
        self.networks = {
            "actor": self.actor, "critic_one": self.critic_one, "critic_two": self.critic_two,
            "actor_target": self.actor_target,
            "critic_one_target": self.critic_one_target,
            "critic_two_target": self.critic_two_target,
        }

    def select_action(self, observation):
        return self.actor.act(observation)

    @torch.no_grad()
    def calculate_targets(self, batch, *, diagnostics=False):
        """Build fixed Bellman supervision; target noise never reaches the simulator."""
        next_observations = tensor(batch["next_observations"])
        rewards, terminated = tensor(batch["rewards"]), tensor(batch["terminated"])
        next_actor_actions = self.actor_target(next_observations)
        if rewards.shape != next_actor_actions.shape or terminated.shape != rewards.shape:
            raise ValueError("Rewards, termination flags, and target actions must be (B, 1).")
        raw_noise = torch.randn_like(next_actor_actions) * self.target_noise_std
        clipped_noise = raw_noise.clamp(-self.noise_clip, self.noise_clip)
        next_actions = (next_actor_actions + clipped_noise).clamp(-2.0, 2.0)
        q_one = self.critic_one_target(next_observations, next_actions)
        q_two = self.critic_two_target(next_observations, next_actions)
        minimum_q = torch.minimum(q_one, q_two)
        # A time limit ends interaction, but only true termination removes continuation.
        continuation = 1.0 - terminated
        discounted_future = self.gamma * continuation * minimum_q
        result = {"targets": rewards + discounted_future}
        if diagnostics:
            result.update(
                rewards=rewards, terminated=terminated, truncated=tensor(batch["truncated"]),
                next_actor_actions=next_actor_actions, raw_noise=raw_noise,
                clipped_noise=clipped_noise, next_actions=next_actions,
                target_q_one=q_one, target_q_two=q_two, minimum_target_q=minimum_q,
                continuation_mask=continuation, discounted_future=discounted_future,
            )
        return result

    def _update_critics(self, observations, actions, targets, diagnostics):
        self.critic_one.train()
        self.critic_two.train()
        q_one = self.critic_one(observations, actions)
        q_two = self.critic_two(observations, actions)
        if q_one.shape != targets.shape or q_two.shape != targets.shape:
            raise ValueError("Critic predictions and targets must have matching (B, 1) shapes.")
        loss_one, loss_two = ((q_one - targets) ** 2).mean(), ((q_two - targets) ** 2).mean()
        loss = loss_one + loss_two
        optimize(loss, self.critic_optimizer, self.critic_parameters)
        self.critic_updates += 1
        result = {"loss_before": loss.item()}
        if diagnostics:
            with torch.no_grad():
                after_one = self.critic_one(observations, actions)
                after_two = self.critic_two(observations, actions)
                after_loss = (
                    ((after_one - targets) ** 2).mean() + ((after_two - targets) ** 2).mean()
                )
            if not torch.isfinite(after_loss):
                raise FloatingPointError("Post-update critic loss is non-finite.")
            result.update(
                loss_after=after_loss.item(), q_one_before=q_one.detach(),
                q_two_before=q_two.detach(), q_one_after=after_one, q_two_after=after_two,
            )
        return result

    def _update_actor(self, observations, diagnostics):
        before = [p.detach().clone() for p in self.critic_parameters] if diagnostics else None
        flags = [p.requires_grad for p in self.critic_parameters]
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor.train()
        # Freeze critic parameters, but differentiate through Q1's action input.
        for parameter in self.critic_parameters:
            parameter.requires_grad_(False)
        try:
            actions = self.actor(observations)
            loss = -self.critic_one(observations, actions).mean()
            optimize(loss, self.actor_optimizer, self.actor_parameters)
            self.actor_updates += 1
            result = {"loss_before": loss.item()}
            if diagnostics:
                with torch.no_grad():
                    after_actions = self.actor(observations)
                    after_loss = -self.critic_one(observations, after_actions).mean()
                if not torch.isfinite(after_loss):
                    raise FloatingPointError("Post-update actor loss is non-finite.")
                result.update(
                    loss_after=after_loss.item(), actions_before=actions.detach(),
                    actions_after=after_actions,
                    critics_unchanged=all(
                        torch.equal(old, new)
                        for old, new in zip(before, self.critic_parameters, strict=True)
                    ),
                )
        finally:
            for parameter, enabled in zip(self.critic_parameters, flags, strict=True):
                parameter.requires_grad_(enabled)
        return result

    @torch.no_grad()
    def _soft_update_targets(self):
        for online, target in (
            (self.actor, self.actor_target),
            (self.critic_one, self.critic_one_target),
            (self.critic_two, self.critic_two_target),
        ):
            for source, destination in zip(online.parameters(), target.parameters(), strict=True):
                destination.lerp_(source, self.tau)
        self.target_updates += 1

    def update(self, batch, *, fixed_targets=None, diagnostics=False):
        """Update both critics, then the actor/targets when due. Diagnostics add no learning."""
        snapshots = None
        if diagnostics:
            snapshots = {
                name: {key: value.detach().clone() for key, value in model.state_dict().items()}
                for name, model in self.networks.items()
            }
        target_info = (
            self.calculate_targets(batch, diagnostics=diagnostics)
            if fixed_targets is None else {"targets": tensor(fixed_targets).detach()}
        )
        observations, actions = tensor(batch["observations"]), tensor(batch["actions"])
        critics = self._update_critics(observations, actions, target_info["targets"], diagnostics)
        actor = None
        if self.critic_updates % self.policy_delay == 0:
            actor = self._update_actor(observations, diagnostics)
            self._soft_update_targets()
        # Normal training returns only the objectives used for backpropagation.
        result = {"critic_loss": critics["loss_before"],
                  "actor_loss": None if actor is None else actor["loss_before"]}
        if diagnostics:
            changes = {}
            for name, model in self.networks.items():
                current = model.state_dict()
                changes[name] = {
                    "changed": any(not torch.equal(snapshots[name][key], value)
                                   for key, value in current.items()),
                    "max_change": max((value - snapshots[name][key]).abs().max().item()
                                      for key, value in current.items()),
                }
            result["diagnostics"] = {
                "critics": critics, "actor": actor, "targets": target_info,
                "weights": changes, "targets_updated": actor is not None,
            }
        return result
