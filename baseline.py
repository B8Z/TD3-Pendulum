"""Compare fixed baseline policies; optionally inspect two TD3 learning updates."""

import argparse
from contextlib import closing

import gymnasium as gym
import numpy as np
import torch

from td3 import ReplayBuffer, TD3

STARTING_STATE_SEEDS = (42, 43, 44, 45, 46)
ACTION_SEEDS = (123, 124, 125)


def run_episode(state_seed, replay=None, action_seed=None):
    """Collect zero torque or a seeded uniform random policy, without learning."""
    with closing(gym.make("Pendulum-v1")) as environment:
        observation, _ = environment.reset(seed=state_seed)
        if action_seed is not None:
            environment.action_space.seed(action_seed)
        episode_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = (
                np.array([0.0], dtype=np.float32)
                if action_seed is None else environment.action_space.sample()
            )
            next_observation, reward, terminated, truncated, _ = environment.step(action)
            if replay is not None:
                replay.add(observation, action, reward, next_observation, terminated, truncated)
            observation = next_observation
            episode_return += float(reward)
        return episode_return


def same_weights(left, right):
    right_state = right.state_dict()
    return all(
        torch.equal(value, right_state[key])
        for key, value in left.state_dict().items()
    )


def inspect_learning(replay):
    """Use a fixed batch to demonstrate learning and check diagnostic equivalence."""
    batch = replay.sample(4)
    original_batch = {key: value.copy() for key, value in batch.items()}
    torch.set_num_threads(1)
    torch.manual_seed(11)
    reference = TD3()
    torch.manual_seed(11)
    agent = TD3()

    # Reuse the same target-noise draw to check that diagnostics add no randomness.
    rng_before = torch.get_rng_state()
    plain_targets = reference.calculate_targets(batch)["targets"]
    rng_after = torch.get_rng_state()
    torch.set_rng_state(rng_before)
    details = agent.calculate_targets(batch, diagnostics=True)
    assert torch.equal(plain_targets, details["targets"]), "Target calculations differ."
    assert torch.equal(rng_after, torch.get_rng_state()), "Diagnostics changed randomness."
    fixed_targets = details["targets"].clone()
    print(f"\nReplay: {len(replay)} transitions; next write position: {replay.position}")
    print("Stored truncated transitions:", int(replay.data["truncated"][:len(replay)].sum()))
    print("First sampled state:", batch["observations"][0])
    print("First sampled action:", batch["actions"][0])
    print("First sampled next state:", batch["next_observations"][0])
    print("Target components for the first row:")
    for name in (
        "rewards", "terminated", "truncated", "next_actor_actions", "raw_noise",
        "clipped_noise", "next_actions", "target_q_one", "target_q_two",
        "minimum_target_q", "continuation_mask", "discounted_future", "targets",
    ):
        print(f"  {name}: {details[name][0].tolist()}")

    for call in (1, 2):
        rng_before = torch.get_rng_state()
        plain = reference.update(batch, fixed_targets=fixed_targets)
        measured = agent.update(batch, fixed_targets=fixed_targets, diagnostics=True)
        diagnostic = measured["diagnostics"]
        assert torch.equal(rng_before, torch.get_rng_state()), "Fixed-target updates drew noise."
        assert plain["critic_loss"] == measured["critic_loss"]
        assert plain["actor_loss"] == measured["actor_loss"]
        assert all(same_weights(model, reference.networks[name])
                   for name, model in agent.networks.items()), "Diagnostic modes changed weights."
        assert torch.equal(fixed_targets, details["targets"]), "Fixed targets changed."
        assert all(np.array_equal(value, original_batch[key])
                   for key, value in batch.items()), "Sampled data changed."
        print(f"\nLEARNING CALL {call}: same batch and fixed targets")
        critics = diagnostic["critics"]
        print(f"Critic loss: {critics['loss_before']:.6f} -> {critics['loss_after']:.6f}")
        actor = diagnostic["actor"]
        if actor is None:
            print("Actor update: skipped.")
        else:
            assert actor["critics_unchanged"], "Critics changed during the actor step."
            print(f"Actor loss: {actor['loss_before']:.6f} -> {actor['loss_after']:.6f}")
            print("Actor actions before:", actor["actions_before"].flatten().tolist())
            print("Actor actions after: ", actor["actions_after"].flatten().tolist())
            print("Critics unchanged during the actor step:", actor["critics_unchanged"])
        print("Soft target update:", diagnostic["targets_updated"])
        print("Critic / actor / target updates:",
              agent.critic_updates, agent.actor_updates, agent.target_updates)
        for name, change in diagnostic["weights"].items():
            print(f"  {name}: changed={change['changed']}; max change={change['max_change']:.3e}")
        print("Diagnostic and normal updates match; batch and fixed targets are unchanged.")
    print("\nThese checks demonstrate learning mechanics, not improved pendulum control.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", action="store_true",
                        help="Inspect replay and two updates; compare diagnostics on/off.")
    args = parser.parse_args()
    replay = ReplayBuffer(capacity=5_000, seed=2026) if args.diagnostics else None
    differences = []
    print("State       Zero    Random 123  Random 124  Random 125   Random mean   Difference")
    for state_seed in STARTING_STATE_SEEDS:
        zero = run_episode(state_seed, replay)
        random = [run_episode(state_seed, replay, seed) for seed in ACTION_SEEDS]
        mean_random = float(np.mean(random))
        differences.append(mean_random - zero)
        print(f"{state_seed:5d} {zero:10.3f} "
              + " ".join(f"{value:11.3f}" for value in random)
              + f" {mean_random:13.3f} {mean_random - zero:+12.3f}")
    print(f"\nMean paired difference (random - zero): {np.mean(differences):+.3f}")
    print("Higher return is better. These are 5 zero and 15 random episodes, not training runs.")
    print("Random sequences are reused across states; these cases differ from TD3 monitoring.")
    if args.diagnostics:
        inspect_learning(replay)


if __name__ == "__main__":
    main()
