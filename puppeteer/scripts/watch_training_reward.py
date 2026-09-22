"""Watch committed training rewards from a running audit-enabled experiment.

The script reads only result rows already committed by the runner, then follows
their audit ``reward_assigned`` events in dataset-offset order.  This avoids
counting an interrupted or partially written task as training evidence.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import fmean


def jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def committed_rows(run_dir: Path) -> list[dict]:
    rows = []
    for result_path in sorted((run_dir / "results").glob("*.jsonl")):
        if ".uncommitted_" in result_path.name:
            continue
        rows.extend(jsonl(result_path))
    return sorted(enumerate(rows), key=lambda pair: (pair[1].get("offset", pair[0]), pair[0]))


def task_rewards(row: dict) -> list[dict]:
    trace_path = Path(row.get("trace_path", ""))
    if not trace_path.is_dir():
        return []
    events_path = trace_path / "events.jsonl"
    if not events_path.is_file():
        return []
    events = list(jsonl(events_path))
    if not any(event.get("event_type") == "task_finished" for event in events):
        return []
    return [event for event in events if event.get("event_type") == "reward_assigned"]


def snapshot(run_dir: Path, window: int) -> dict:
    tasks = []
    all_paths = []
    for _, row in committed_rows(run_dir):
        rewards = task_rewards(row)
        if not rewards:
            continue
        path_rewards = [float(event["path_reward"]) for event in rewards]
        task_rewards_binary = [float(event["task_reward"]) for event in rewards]
        all_paths.extend(zip(path_rewards, task_rewards_binary))
        tasks.append(
            {
                "offset": row.get("offset"),
                "path_reward": fmean(path_rewards),
                "candidate_accuracy": fmean(value > 0 for value in task_rewards_binary),
            }
        )
    recent = tasks[-window:]
    return {
        "tasks": len(tasks),
        "paths": len(all_paths),
        "mean_path_reward": fmean(value[0] for value in all_paths) if all_paths else None,
        "candidate_accuracy": fmean(value[1] > 0 for value in all_paths) if all_paths else None,
        "rolling_window": len(recent),
        "rolling_mean_path_reward": fmean(item["path_reward"] for item in recent) if recent else None,
        "rolling_candidate_accuracy": fmean(item["candidate_accuracy"] for item in recent) if recent else None,
        "last_offset": tasks[-1]["offset"] if tasks else None,
    }


def render(metrics: dict) -> str:
    if not metrics["tasks"]:
        return "Waiting for the first committed task reward..."
    return (
        f"tasks={metrics['tasks']} paths={metrics['paths']} last_offset={metrics['last_offset']} | "
        f"cumulative: path_reward={metrics['mean_path_reward']:.4f}, "
        f"candidate_accuracy={metrics['candidate_accuracy']:.3f} | "
        f"last-{metrics['rolling_window']}: path_reward={metrics['rolling_mean_path_reward']:.4f}, "
        f"candidate_accuracy={metrics['rolling_candidate_accuracy']:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.window < 1 or args.interval_seconds <= 0:
        parser.error("--window and --interval-seconds must be positive")

    previous_count = None
    while True:
        metrics = snapshot(args.run_dir, args.window)
        if args.once or metrics["tasks"] != previous_count:
            print(render(metrics), flush=True)
            previous_count = metrics["tasks"]
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
