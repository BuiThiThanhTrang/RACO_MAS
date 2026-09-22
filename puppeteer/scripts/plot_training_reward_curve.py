"""Plot a report-ready reward learning curve from an audit-enabled training run.

The upper panel shows the mean terminal path reward used by one policy update.
The lower panel shows task accuracy for the same updates.  Plotting both avoids
mistaking a reward change caused by the step/action-cost term for an accuracy
improvement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _rolling_mean(values: list[float], window: int) -> list[float]:
    result: list[float] = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def _committed_trace_dirs(audit_dir: Path) -> set[Path]:
    """Keep only traces whose predictions were committed to a result file."""
    run_dir = audit_dir.parent if audit_dir.name == "audit" else audit_dir
    committed: set[Path] = set()
    for result_file in sorted((run_dir / "results").glob("*.jsonl")):
        if ".uncommitted_" in result_file.name:
            continue
        for line in result_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                trace_path = json.loads(line).get("trace_path")
                if trace_path:
                    committed.add(Path(trace_path).resolve())
    return committed


def _records(audit_dir: Path) -> list[dict]:
    committed = _committed_trace_dirs(audit_dir)
    records: list[dict] = []
    for trace in sorted(audit_dir.rglob("events.jsonl")):
        if committed and trace.parent.resolve() not in committed:
            continue
        events = [
            json.loads(line)
            for line in trace.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not any(event.get("event_type") == "task_finished" for event in events):
            continue
        updates = [event for event in events if event.get("event_type") == "policy_update"]
        rewards = [event for event in events if event.get("event_type") == "reward_assigned"]
        if not updates or not rewards:
            continue
        update = updates[-1]
        path_rewards = [float(event["path_reward"]) for event in rewards if event.get("path_reward") is not None]
        mean_reward = update.get("mean_reward")
        if mean_reward is None:
            mean_reward = sum(path_rewards) / len(path_rewards) if path_rewards else None
        if mean_reward is None:
            continue
        records.append(
            {
                "task_id": str(events[0].get("task_id", "")),
                "mean_reward": float(mean_reward),
                "task_return": float(update["task_return"]) if update.get("task_return") is not None else None,
                "accuracy": sum(float(event.get("task_reward", -1.0)) > 0 for event in rewards) / len(rewards),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=20, help="Moving-average window in policy updates (default: 20).")
    parser.add_argument("--title", default="Training learning curve")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be at least 1")

    records = _records(args.audit_dir)
    if not records:
        parser.error("No completed, committed policy-update records were found.")

    steps = list(range(1, len(records) + 1))
    rewards = [record["mean_reward"] for record in records]
    accuracies = [record["accuracy"] for record in records]
    reward_trend = _rolling_mean(rewards, args.window)
    accuracy_trend = _rolling_mean(accuracies, args.window)

    figure, (reward_axis, accuracy_axis) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    figure.suptitle(args.title, fontsize=14, fontweight="bold")

    reward_axis.plot(steps, rewards, color="#9ecae1", linewidth=1, alpha=0.65, label="Mean path reward / update")
    reward_axis.plot(steps, reward_trend, color="#08519c", linewidth=2.4, label=f"Moving average ({args.window})")
    reward_axis.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9)
    reward_axis.set_ylabel("Path reward")
    reward_axis.grid(alpha=0.25)
    reward_axis.legend(loc="best")

    accuracy_axis.plot(steps, accuracies, color="#fcbba1", linewidth=1, alpha=0.65, label="Task accuracy / update")
    accuracy_axis.plot(steps, accuracy_trend, color="#cb181d", linewidth=2.4, label=f"Moving average ({args.window})")
    accuracy_axis.set_ylim(-0.03, 1.03)
    accuracy_axis.set_xlabel("Policy update")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.grid(alpha=0.25)
    accuracy_axis.legend(loc="best")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    print(json.dumps({"output": str(args.output), "updates": len(records), "window": args.window}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
