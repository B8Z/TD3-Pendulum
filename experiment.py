"""Run local validation and one new training-seed experiment, with saved reports."""

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from plot import INCLUDED_RUN, analyze_run, compare_learning_curves, read_rows

PROJECT = Path(__file__).resolve().parent
PROTOCOL_KEYS = (
    "environment", "seed", "warmup_steps", "batch_size", "exploration_std",
    "evaluate_every", "evaluation_seeds", "replay_capacity", "action_seed",
    "exploration_seed", "replay_seed", "td3",
)


def settings(run):
    return json.loads((run / "settings.json").read_text(encoding="utf-8"))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def run_script(name, *arguments):
    """Use this same interpreter and show child-process progress in the terminal."""
    subprocess.run(
        [sys.executable, "-X", "utf8", "-u", str(PROJECT / name), *map(str, arguments)],
        cwd=PROJECT, check=True,
    )


def analyze_checked(run):
    summary = analyze_run(run)
    if not summary["replay_matches_log"]:
        raise RuntimeError(f"Saved-actor replay differs from the log in {run.name}.")
    return summary


def write_presentation_update(report, path):
    """Fill the repeat-experiment paragraph only from completed measurements."""
    if "replication" not in report:
        path.write_text(
            "# Presentation update\n\nValidation completed; no new training seed was run.\n"
            "Keep the repeat experiment labelled as pending.\n", encoding="utf-8",
        )
        return
    pair = report["replication"]
    original, repeat = pair["reference"], pair["repeat"]
    validation = report.get("reference_reproduction")
    scope = (
        f"Reference validation covered {validation['training_steps_checked']:,} steps "
        f"({validation['scope']})."
        if validation else
        "This invocation skipped validation; use an earlier report if available."
    )
    text = (
        "# Measured presentation update\n\n"
        f"{scope}\n\n"
        "Suggested Slide 7 wording:\n\n"
        f"I repeated training with seed {repeat['seed']} for "
        f"{repeat['training_steps']:,} interactions. The mean evaluation return was "
        f"{repeat['mean_evaluation_return']:.2f}, compared with "
        f"{original['mean_evaluation_return']:.2f} in the recorded seed-11 run. "
        "The mean absolute angle during the final two seconds changed from "
        f"{original['mean_angle_mae_last_2s_deg']:.2f} to "
        f"{repeat['mean_angle_mae_last_2s_deg']:.2f} degrees. "
        "This adds one independent training seed. Same-seed diagnostic checks validate "
        "instrumentation and do not increase that count.\n\n"
        "Discuss the measured differences without claiming an established root cause "
        "or broad reliability from two seeds.\n"
    )
    if not pair["same_training_budget"]:
        text += "\nThe training budget also changed, so this was not a seed-only comparison.\n"
    path.write_text(text, encoding="utf-8")


def compare_rows(left, right, filename, key_fields, numeric_fields, cutoff):
    """Compare behavior columns shared by old and new log formats."""
    def index(run):
        rows = [
            row for row in read_rows(run / filename)
            if int(row["train_step"]) <= cutoff
        ]
        indexed = {tuple(row[field] for field in key_fields): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"Duplicate records in {run / filename}")
        return indexed

    a, b = index(left), index(right)
    same_keys = bool(a) and a.keys() == b.keys()
    max_difference = 0.0
    flags_equal = True
    if same_keys:
        for key in a:
            for field in numeric_fields:
                x, y = float(a[key][field]), float(b[key][field])
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("A compared log contains a non-finite value.")
                max_difference = max(max_difference, abs(x - y))
            for field in ("terminated", "truncated"):
                if field in a[key] and field in b[key]:
                    flags_equal = flags_equal and a[key][field] == b[key][field]
    return {
        "same_records": same_keys, "records_checked": len(a) if same_keys else 0,
        "max_numeric_difference": max_difference if same_keys else None,
        "matching_end_flags": flags_equal if same_keys else False,
        "passed": same_keys and flags_equal and max_difference <= 1e-8,
    }


def compare_equivalence(left, right, *, prefix=False):
    """Validate same-seed behavior. A short prefix cannot validate final actor weights."""
    a, b = settings(left), settings(right)
    mismatches = [key for key in PROTOCOL_KEYS if a.get(key) != b.get(key)]
    same_budget = a["total_steps"] == b["total_steps"]
    cutoff = min(a["total_steps"], b["total_steps"])
    if not prefix and not same_budget:
        mismatches.append("total_steps")
    episodes = compare_rows(
        left, right, "episodes.csv", ("train_step",),
        ("episode", "train_step", "episode_return", "episode_steps"), cutoff,
    )
    evaluations = compare_rows(
        left, right, "evaluations.csv", ("train_step", "starting_state_seed"),
        ("episode_return",), cutoff,
    )
    weights_equal = None
    if same_budget:
        x = torch.load(left / "actor_final.pt", map_location="cpu", weights_only=True)
        y = torch.load(right / "actor_final.pt", map_location="cpu", weights_only=True)
        weights_equal = x.keys() == y.keys() and all(
            torch.equal(x[key], y[key]) for key in x
        )
    return {
        "left": left.name, "right": right.name,
        "scope": "common training prefix" if prefix else "complete same-seed runs",
        "training_steps_checked": cutoff, "protocol_mismatches": mismatches,
        "episodes": episodes, "evaluations": evaluations,
        "final_actor_exactly_equal": weights_equal,
        "software_versions": {
            key: {"left": a.get(key), "right": b.get(key)}
            for key in ("python", "gymnasium", "numpy", "torch")
        },
        "passed": (
            not mismatches and episodes["passed"] and evaluations["passed"]
            and (weights_equal is True or (prefix and not same_budget))
        ),
    }


def execute(args):
    baseline = PROJECT / "runs" / INCLUDED_RUN
    if not (baseline / "actor_final.pt").is_file():
        raise FileNotFoundError("Keep the included reference run in the project.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    report_directory = PROJECT / "comparisons" / stamp
    report_directory.mkdir(parents=True, exist_ok=False)
    report_path = report_directory / "experiment_report.json"
    report = {
        "status": "running", "reference_run": INCLUDED_RUN,
        "new_training_seed": args.seed, "new_training_steps": args.steps,
        "new_results_pending": True,
    }
    save_json(report_path, report)
    try:
        if not args.skip_checks:
            run_script("baseline.py", "--diagnostics")
            report["learning_diagnostics_passed"] = True
            save_json(report_path, report)
            normal_name = f"check_{stamp}_seed11_normal"
            diagnostic_name = f"check_{stamp}_seed11_diagnostics"
            common = ("--seed", 11, "--steps", args.check_steps)
            run_script("train.py", *common, "--run-name", normal_name)
            run_script("train.py", *common, "--run-name", diagnostic_name, "--diagnostics")
            normal = PROJECT / "runs" / normal_name
            diagnostic = PROJECT / "runs" / diagnostic_name
            mode_check = compare_equivalence(normal, diagnostic)
            save_json(report_directory / "diagnostic_equivalence.json", mode_check)
            report["diagnostic_equivalence"] = mode_check
            save_json(report_path, report)
            if not mode_check["passed"]:
                raise RuntimeError("Diagnostic and normal runs differ. Inspect the saved report.")
            reproduction = compare_equivalence(
                baseline, normal, prefix=args.check_steps < 20_000
            )
            save_json(report_directory / "reference_reproduction.json", reproduction)
            report["reference_reproduction"] = reproduction
            save_json(report_path, report)
            if not reproduction["passed"]:
                raise RuntimeError("Reference reproduction differs. Inspect the saved report.")
            # Both modes receive their own plots and physical measurements.
            report["normal_analysis"] = analyze_checked(normal)
            report["diagnostic_analysis"] = analyze_checked(diagnostic)
            compare_learning_curves(
                [normal, diagnostic], report_directory,
                filename="diagnostics_learning_curves.png",
            )
        if not args.checks_only:
            repeat_name = f"repeat_{stamp}_seed{args.seed}_normal"
            run_script(
                "train.py", "--seed", args.seed, "--steps", args.steps,
                "--run-name", repeat_name,
            )
            repeat = PROJECT / "runs" / repeat_name
            original_summary = analyze_checked(baseline)
            repeat_summary = analyze_checked(repeat)
            report["replication"] = {
                "reference": original_summary, "repeat": repeat_summary,
                "mean_return_difference": (
                    repeat_summary["mean_evaluation_return"]
                    - original_summary["mean_evaluation_return"]
                ),
                "angle_mae_difference_deg": (
                    repeat_summary["mean_angle_mae_last_2s_deg"]
                    - original_summary["mean_angle_mae_last_2s_deg"]
                ),
                "same_training_budget": args.steps == 20_000,
                "independent_training_seeds_including_reference": 2,
            }
            report["new_results_pending"] = False
            compare_learning_curves(
                [baseline, repeat], report_directory,
                filename="seed_learning_curves.png",
            )
            save_json(report_directory / "seed_comparison.json", report["replication"])
        report["status"] = "complete"
        write_presentation_update(report, report_directory / "presentation_update.md")
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["error"] = "The user interrupted the workflow."
        raise
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        save_json(report_path, report)
    print("\nWorkflow complete. Reports:", report_directory)
    print("Same-seed diagnostic runs are validation, not additional independent experiments.")
    if args.check_steps < 20_000 and not args.skip_checks:
        print("Reference validation covered only the configured training prefix.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=22, help="One new training seed.")
    parser.add_argument("--steps", type=int, default=20_000, help="New experiment's budget.")
    parser.add_argument("--check-steps", type=int, default=20_000,
                        help="Same-seed check budget: a multiple of 2000, up to 20000.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--checks-only", action="store_true", help="Run validation without a new seed."
    )
    modes.add_argument(
        "--skip-checks", action="store_true", help="Use only after validation passed."
    )
    args = parser.parse_args()
    if args.seed < 0 or args.seed == 11:
        parser.error("--seed must be nonnegative and different from the reference seed 11.")
    if args.steps <= 1_000:
        parser.error("--steps must exceed the 1000-step warm-up.")
    if not 2_000 <= args.check_steps <= 20_000 or args.check_steps % 2_000:
        parser.error("--check-steps must be a multiple of 2000 between 2000 and 20000.")
    execute(args)


if __name__ == "__main__":
    main()
