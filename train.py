"""Train a fresh TD3 agent with random warm-up and separate noise-free evaluation."""

import argparse
import csv
import json
import platform
import re
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch

from td3 import ReplayBuffer, TD3

# Experiment settings are explicit here and saved with every run.
WARMUP_STEPS = 1_000
BATCH_SIZE = 128
REPLAY_CAPACITY = 50_000
EXPLORATION_STD = 0.2
EVALUATE_EVERY = 2_000
EVALUATION_SEEDS = (201, 202, 203, 204, 205)
TD3_SETTINGS = {
    "hidden_size": 64, "gamma": 0.99, "target_noise_std": 0.4, "noise_clip": 1.0,
    "critic_learning_rate": 0.0003, "actor_learning_rate": 0.0003,
    "policy_delay": 2, "tau": 0.005,
}
PROJECT = Path(__file__).resolve().parent


def evaluate(agent, environment, step, writer):
    """Measure the current actor on fixed cases without noise, replay writes, or updates."""
    was_training = agent.actor.training
    agent.actor.eval()
    returns = []
    try:
        for seed in EVALUATION_SEEDS:
            observation, _ = environment.reset(seed=seed)
            episode_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.select_action(observation)
                observation, reward, terminated, truncated, _ = environment.step(action)
                episode_return += float(reward)
            writer.writerow((step, seed, episode_return))
            returns.append(episode_return)
    finally:
        agent.actor.train(was_training)
    print(f"EVAL step={step:6d}: mean={np.mean(returns):.2f}; "
          f"case SD={np.std(returns, ddof=1):.2f}")
    print("  Individual returns:", [round(value, 2) for value in returns])


def train(args):
    if not 1 <= BATCH_SIZE <= WARMUP_STEPS < args.steps:
        raise ValueError("Require 1 <= batch size <= warm-up steps < total steps.")
    if EVALUATE_EVERY < 1 or not np.isfinite(EXPLORATION_STD) or EXPLORATION_STD < 0:
        raise ValueError("Evaluation interval must be positive; noise SD finite and nonnegative.")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    exploration_rng = np.random.default_rng(args.seed + 2)
    agent = TD3(**TD3_SETTINGS)
    replay = ReplayBuffer(capacity=REPLAY_CAPACITY, seed=args.seed + 3)
    run_name = args.run_name or datetime.now(timezone.utc).strftime("train_%Y%m%dT%H%M%S_%fZ")
    run_directory = PROJECT / "runs" / run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    settings = {
        "environment": "Pendulum-v1", "seed": args.seed, "total_steps": args.steps,
        "warmup_steps": WARMUP_STEPS, "batch_size": BATCH_SIZE,
        "exploration_std": EXPLORATION_STD, "evaluate_every": EVALUATE_EVERY,
        "evaluation_seeds": EVALUATION_SEEDS, "replay_capacity": REPLAY_CAPACITY,
        "action_seed": args.seed + 1, "exploration_seed": args.seed + 2,
        "replay_seed": args.seed + 3, "td3": TD3_SETTINGS,
        "python": platform.python_version(), "gymnasium": gym.__version__,
        "numpy": np.__version__, "torch": torch.__version__,
        "diagnostics": args.diagnostics, "log_format_version": 2,
        "implementation": "consolidated-v2",
    }
    (run_directory / "settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    print("Saving this run to:", run_directory)
    print(f"First {WARMUP_STEPS} steps: random actions, without learning.")
    print("Then: actor plus exploration noise; one joint critic update per step.")
    start_time = perf_counter()
    with (
        closing(gym.make("Pendulum-v1")) as training_environment,
        closing(gym.make("Pendulum-v1")) as evaluation_environment,
        (run_directory / "episodes.csv").open("w", newline="", encoding="utf-8") as episode_file,
        (run_directory / "evaluations.csv").open(
            "w", newline="", encoding="utf-8"
        ) as evaluation_file,
    ):
        training_environment.action_space.seed(args.seed + 1)
        observation, _ = training_environment.reset(seed=args.seed)
        episode, episode_steps, episode_return = 1, 0, 0.0
        episode_writer, evaluation_writer = csv.writer(episode_file), csv.writer(evaluation_file)
        episode_writer.writerow((
            "episode", "train_step", "episode_return", "episode_steps",
            "terminated", "truncated", "critic_loss_before_update", "actor_loss_before_update",
        ))
        evaluation_writer.writerow(("train_step", "starting_state_seed", "episode_return"))
        evaluate(agent, evaluation_environment, 0, evaluation_writer)
        evaluation_file.flush()
        for step in range(1, args.steps + 1):
            if step <= WARMUP_STEPS:
                action = training_environment.action_space.sample()
            else:
                proposal = agent.select_action(observation)
                noise = exploration_rng.normal(0.0, EXPLORATION_STD, size=proposal.shape)
                action = np.clip(
                    proposal + noise,
                    training_environment.action_space.low,
                    training_environment.action_space.high,
                ).astype(np.float32)
            next_observation, reward, terminated, truncated, _ = training_environment.step(action)
            # Record the applied action and true endpoint before resetting either state.
            replay.add(observation, action, reward, next_observation, terminated, truncated)
            observation = next_observation
            episode_return += float(reward)
            episode_steps += 1
            metrics = None
            if step > WARMUP_STEPS and len(replay) >= BATCH_SIZE:
                metrics = agent.update(replay.sample(BATCH_SIZE), diagnostics=args.diagnostics)
            if terminated or truncated:
                critic_loss = "" if metrics is None else metrics["critic_loss"]
                actor_loss = (
                    "" if metrics is None or metrics["actor_loss"] is None
                    else metrics["actor_loss"]
                )
                episode_writer.writerow((
                    episode, step, episode_return, episode_steps, terminated, truncated,
                    critic_loss, actor_loss,
                ))
                episode_file.flush()
                print(f"TRAIN step={step:6d}; episode={episode:3d}; return={episode_return:.2f}")
                if args.diagnostics and metrics is not None:
                    detail = metrics["diagnostics"]
                    print("  Critic loss before/after:",
                          detail["critics"]["loss_before"], detail["critics"]["loss_after"])
                    if detail["actor"] is not None:
                        print("  Actor loss before/after:",
                              detail["actor"]["loss_before"], detail["actor"]["loss_after"])
                observation, _ = training_environment.reset()
                episode, episode_steps, episode_return = episode + 1, 0, 0.0
            if step % EVALUATE_EVERY == 0 or step == args.steps:
                evaluate(agent, evaluation_environment, step, evaluation_writer)
                evaluation_file.flush()
        if episode_steps:
            print("Incomplete final episode excluded from episodes.csv; steps:", episode_steps)
        torch.save(agent.actor.state_dict(), run_directory / "actor_final.pt")
    elapsed = perf_counter() - start_time
    summary = {
        "completed": True, "seed": args.seed, "diagnostics": args.diagnostics,
        "training_steps": args.steps, "critic_updates": agent.critic_updates,
        "actor_updates": agent.actor_updates, "target_updates": agent.target_updates,
        "elapsed_seconds": elapsed,
    }
    (run_directory / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Completed critic / actor / target updates:",
          agent.critic_updates, agent.actor_updates, agent.target_updates)
    print(f"Elapsed time: {elapsed:.1f} seconds")
    print("Completed. Results:", run_directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11, help="Base training seed.")
    parser.add_argument("--steps", type=int, default=20_000, help="Training interaction budget.")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Measure before/after losses and weight changes on every update.")
    parser.add_argument(
        "--run-name", help="Optional new directory name under runs; never overwritten."
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative.")
    if args.run_name is not None and (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name)
        or args.run_name.endswith(".")
    ):
        parser.error(
            "--run-name must be a simple directory name using letters, digits, _, -, or ."
        )
    train(args)


if __name__ == "__main__":
    main()
