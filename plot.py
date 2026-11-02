"""Plot recorded returns and replay a saved actor without exploration or learning."""

import argparse
import csv
import json
from contextlib import closing
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from td3 import Actor

PROJECT = Path(__file__).resolve().parent
INCLUDED_RUN = "train_20261102T020701_889706Z"
NAVY, TEAL, ORANGE = "#163B50", "#007F86", "#B96816"


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return rows


def plot_returns(episodes, evaluations, settings, output):
    """Separate noisy training episodes from fixed-case, noise-free evaluations."""
    train_steps = np.array([int(row["train_step"]) for row in episodes])
    train_returns = np.array([float(row["episode_return"]) for row in episodes])
    steps = sorted({int(row["train_step"]) for row in evaluations})
    seeds = settings["evaluation_seeds"]
    scores = {
        (int(row["train_step"]), int(row["starting_state_seed"])): float(row["episode_return"])
        for row in evaluations
    }
    expected = {(step, seed) for step in steps for seed in seeds}
    if set(scores) != expected or len(scores) != len(evaluations):
        raise ValueError("Require exactly one evaluation per checkpoint and starting-state seed.")
    values = np.array([[scores[(step, seed)] for seed in seeds] for step in steps])
    if not np.isfinite(train_returns).all() or not np.isfinite(values).all():
        raise ValueError("A return log contains a non-finite value.")
    if steps[-1] != settings["total_steps"]:
        raise ValueError("Final evaluation does not match the configured training budget.")
    figure, axes = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, sharey=True, layout="constrained"
    )
    figure.suptitle(f"TD3 learning | training seed {settings['seed']}")
    axes[0].plot(train_steps, train_returns, color=NAVY, alpha=0.55,
                 marker=".", linewidth=0.8, label="Training episodes")
    if len(train_returns) >= 5:
        smoothed = np.convolve(train_returns, np.ones(5) / 5, mode="valid")
        axes[0].plot(train_steps[4:], smoothed, color=TEAL, linewidth=2,
                     label="Trailing 5-episode mean")
    axes[0].axvspan(0, settings["warmup_steps"], color="0.8", alpha=0.4,
                   label="Random warm-up")
    axes[0].set_title("Training returns: random warm-up, then actor plus exploration noise")
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    for column, seed in enumerate(seeds):
        axes[1].plot(steps, values[:, column], marker=".", linewidth=1,
                     alpha=0.75, label=f"Start seed {seed}")
    axes[1].plot(steps, values.mean(axis=1), color=NAVY, marker="o",
                 markersize=4, linewidth=2.5, label="Mean across cases")
    axes[1].set_title("Noise-free evaluation: one trained agent, fixed starting states")
    axes[1].legend(loc="lower right", ncol=2, frameon=False, fontsize=9)
    axes[1].set_xlabel("Training environment steps")
    for axis in axes:
        axis.set_ylabel("Episode return (sum of rewards)")
        axis.axhline(0, color="0.5", linewidth=0.8, linestyle=":")
        axis.set_ylim(top=50)
        axis.grid(alpha=0.2)
    axes[1].set_xlim(0, settings["total_steps"])
    figure.savefig(output / "learning_curves.png", dpi=300, facecolor="white")
    return {seed: scores[(steps[-1], seed)] for seed in seeds}


def replay_actor(actor, environment_name, seed):
    """Store N transitions and N+1 states, including the state before the first action."""
    with closing(gym.make(environment_name)) as environment:
        observation, _ = environment.reset(seed=seed)
        states, actions, rewards = [observation.copy()], [], []
        terminated = truncated = False
        while not (terminated or truncated):
            action = actor.act(observation)
            observation, reward, terminated, truncated, _ = environment.step(action)
            states.append(observation.copy())
            actions.append(float(action[0]))
            rewards.append(float(reward))
        states = np.array(states)
        return {
            "times": np.arange(len(states)) * float(environment.unwrapped.dt),
            "angles": np.arctan2(states[:, 1], states[:, 0]).astype(float),
            "velocities": states[:, 2].astype(float),
            "torques": np.array(actions), "rewards": np.array(rewards),
            "return": float(sum(rewards)), "terminated": terminated, "truncated": truncated,
        }


def plot_trajectory(trajectory, seed, output):
    """Align state samples with the intervals over which actions were applied."""
    times, angles = trajectory["times"], trajectory["angles"]
    degrees = np.degrees(angles)
    degrees[np.flatnonzero(np.abs(np.diff(angles)) > np.pi) + 1] = np.nan
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
    figure.suptitle(
        f"Final actor | start seed {seed} | return {trajectory['return']:.2f} | no exploration"
    )
    # Gaps avoid a false sweep across the angle's +/-180-degree wrapping boundary.
    axes[0].plot(times, degrees, color=NAVY, linewidth=1.5)
    axes[0].set_ylabel("Angle from upright (deg)")
    axes[0].set_ylim(-190, 190)
    axes[0].set_yticks([-180, -90, 0, 90, 180])
    axes[1].plot(times, trajectory["velocities"], color=TEAL, linewidth=1.5)
    axes[1].set_ylabel("Angular velocity (rad/s)")
    axes[1].set_ylim(-8.5, 8.5)
    axes[2].stairs(trajectory["torques"], times, baseline=None, color=ORANGE, linewidth=1.2)
    axes[2].set_ylabel("Applied torque (N m)")
    axes[2].set_ylim(-2.2, 2.2)
    axes[2].axhline(2, color="0.5", linewidth=0.8, linestyle="--", label="Torque limits")
    axes[2].axhline(-2, color="0.5", linewidth=0.8, linestyle="--")
    axes[2].legend(loc="upper right", frameon=False, fontsize=9)
    axes[2].set_xlabel("Simulated time (s)")
    for axis in axes:
        axis.axhline(0, color="0.5", linewidth=0.8, linestyle=":")
        axis.set_xlim(times[0], times[-1])
        axis.grid(alpha=0.2)
    figure.savefig(output / f"trajectory_seed_{seed}.png", dpi=300, facecolor="white")
    with (output / f"trajectory_seed_{seed}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "time_s", "wrapped_angle_rad", "angular_velocity_rad_s",
            "torque_for_next_interval_nm", "reward_for_next_transition",
        ))
        # The final state has no subsequent transition; leave its action/reward blank.
        writer.writerows(zip(
            times, angles, trajectory["velocities"],
            trajectory["torques"].tolist() + [""],
            trajectory["rewards"].tolist() + [""], strict=True,
        ))


def analyze_run(run, *, show=False):
    """Replay a completed run and save fresh analysis separately from its archived figures."""
    run = Path(run)
    for name in ("settings.json", "episodes.csv", "evaluations.csv", "actor_final.pt"):
        if not (run / name).is_file():
            raise FileNotFoundError(f"Missing {run / name}")
    settings = json.loads((run / "settings.json").read_text(encoding="utf-8"))
    if settings["environment"] != "Pendulum-v1":
        raise ValueError("This script expects Pendulum-v1.")
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "figure.titlesize": 13,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    references = plot_returns(
        read_rows(run / "episodes.csv"), read_rows(run / "evaluations.csv"), settings, output
    )
    torch.set_num_threads(1)
    actor = Actor(hidden_size=settings["td3"]["hidden_size"]).to("cpu")
    actor.load_state_dict(
        torch.load(run / "actor_final.pt", map_location="cpu", weights_only=True), strict=True
    )
    actor.eval()
    print("Reading run:", run)
    cases = []
    for seed in settings["evaluation_seeds"]:
        trajectory = replay_actor(actor, settings["environment"], seed)
        difference = trajectory["return"] - references[seed]
        print(f"Seed {seed}: replay={trajectory['return']:.6f}; "
              f"recorded={references[seed]:.6f}; difference={difference:+.3e}")
        print(f"  Initial angle={np.degrees(trajectory['angles'][0]):.2f} deg; "
              f"initial velocity={trajectory['velocities'][0]:.3f} rad/s")
        print(f"  Steps={len(trajectory['torques'])}; terminated={trajectory['terminated']}; "
              f"truncated={trajectory['truncated']}")
        if abs(difference) > 0.001:
            print("  Replay differs: check the checkpoint, logs, code, and software versions.")
        tail = trajectory["times"] >= trajectory["times"][-1] - 2.0
        action_tail = tail[:-1]
        cases.append({
            "seed": seed, "recorded_return": references[seed],
            "replay_return": trajectory["return"], "return_difference": difference,
            "angle_mae_last_2s_deg": float(np.abs(
                np.degrees(trajectory["angles"][tail])
            ).mean()),
            "max_speed_last_2s_rad_s": float(np.abs(
                trajectory["velocities"][tail]
            ).max()),
            "mean_torque_last_2s_nm": float(trajectory["torques"][action_tail].mean()),
        })
        plot_trajectory(trajectory, seed, output)
    final_returns = list(references.values())
    summary = {
        "run_name": run.name, "seed": settings["seed"],
        "training_steps": settings["total_steps"],
        "mean_evaluation_return": float(np.mean(final_returns)),
        "case_sd": float(np.std(final_returns, ddof=1)),
        "mean_angle_mae_last_2s_deg": float(np.mean([
            case["angle_mae_last_2s_deg"] for case in cases
        ])),
        "replay_matches_log": all(abs(case["return_difference"]) <= 0.001 for case in cases),
        "cases": cases,
    }
    with (output / "case_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Final mean return: {summary['mean_evaluation_return']:.3f}; "
          f"last-two-second angle MAE: {summary['mean_angle_mae_last_2s_deg']:.3f} deg")
    print("Figures and trajectory CSVs:", output)
    if show:
        plt.show()
    else:
        plt.close("all")
    return summary


def compare_learning_curves(runs, output, *, filename="run_comparison.png"):
    """Compare case-mean returns; separate curves still represent individual trained agents."""
    figure, axis = plt.subplots(figsize=(10, 5), layout="constrained")
    for run in runs:
        settings = json.loads((run / "settings.json").read_text(encoding="utf-8"))
        rows = read_rows(run / "evaluations.csv")
        steps = sorted({int(row["train_step"]) for row in rows})
        means = [
            np.mean([float(row["episode_return"]) for row in rows
                     if int(row["train_step"]) == step])
            for step in steps
        ]
        mode = "diagnostics" if settings.get("diagnostics") else "normal"
        if "diagnostics" not in settings:
            mode = "original instrumented"
        axis.plot(steps, means, marker="o", linewidth=1.8,
                  label=f"Seed {settings['seed']}, {mode}")
    axis.set(
        xlabel="Training environment steps", ylabel="Mean evaluation episode return",
        title="Fixed monitoring cases across training runs",
    )
    axis.axhline(0, color="0.5", linewidth=0.8, linestyle=":")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(output / filename, dpi=300, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=INCLUDED_RUN, help="Run directory name or absolute path.")
    parser.add_argument(
        "--no-show", action="store_true", help="Save without opening plot windows."
    )
    args = parser.parse_args()
    selected = Path(args.run)
    run = selected if selected.is_absolute() else PROJECT / "runs" / selected
    analyze_run(run, show=not args.no_show)


if __name__ == "__main__":
    main()
