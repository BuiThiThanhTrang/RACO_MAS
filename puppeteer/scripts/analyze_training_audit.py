"""Summarize optimizer, reward, and routing health from training audit traces."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def _numbers(events: list[dict], key: str) -> list[float]:
    values = []
    for event in events:
        value = event.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "mean": _mean(values),
        "median": statistics.median(values) if values else None,
        "max": max(values) if values else None,
    }


def _committed_trace_dirs(audit_dir: Path) -> set[Path]:
    """Return trace directories referenced by the run's committed result files."""
    run_dir = audit_dir.parent if audit_dir.name == "audit" else audit_dir
    results_dir = run_dir / "results"
    committed = set()
    if not results_dir.is_dir():
        return committed
    for result_file in sorted(results_dir.glob("*.jsonl")):
        if ".uncommitted_" in result_file.name:
            continue
        for line in result_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            trace_path = row.get("trace_path")
            if trace_path:
                committed.add(Path(trace_path).resolve())
    return committed


def _committed_result_rows(audit_dir: Path) -> dict[Path, dict]:
    """Map each committed audit trace to its result row and dataset offset."""
    run_dir = audit_dir.parent if audit_dir.name == "audit" else audit_dir
    rows: dict[Path, dict] = {}
    results_dir = run_dir / "results"
    if not results_dir.is_dir():
        return rows
    for result_file in sorted(results_dir.glob("*.jsonl")):
        if ".uncommitted_" in result_file.name:
            continue
        for line in result_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            trace_path = row.get("trace_path")
            if trace_path:
                rows[Path(trace_path).resolve()] = row
    return rows


def _window_summary(records: list[dict]) -> dict:
    """Summarize a contiguous part of the actual training sequence."""
    return {
        "items": len(records),
        "accuracy": _mean(record["correct"] for record in records),
        "path_reward": _summary(
            [record["path_reward"] for record in records if record["path_reward"] is not None]
        ),
        "token_cost_penalty": _summary(
            [
                record["token_cost_penalty"]
                for record in records
                if record["token_cost_penalty"] is not None
            ]
        ),
        "normalized_entropy": _summary(
            [record["normalized_entropy"] for record in records if record["normalized_entropy"] is not None]
        ),
        "top1_top2_margin": _summary(
            [record["top1_top2_margin"] for record in records if record["top1_top2_margin"] is not None]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    traces = sorted(args.audit_dir.rglob("events.jsonl"))
    if not traces:
        parser.error(f"No events.jsonl traces found below {args.audit_dir}")

    committed_trace_dirs = _committed_trace_dirs(args.audit_dir)
    committed_result_rows = _committed_result_rows(args.audit_dir)
    completed_records = []
    interrupted_traces = 0
    interruption_types = {}
    for trace in traces:
        events = [
            json.loads(line)
            for line in trace.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not any(event.get("event_type") == "task_finished" for event in events):
            interrupted_traces += 1
            interrupted = next(
                (
                    event
                    for event in reversed(events)
                    if event.get("event_type") == "task_interrupted"
                ),
                None,
            )
            error_type = (
                str(interrupted.get("error_type")) if interrupted else "unknown"
            )
            interruption_types[error_type] = interruption_types.get(error_type, 0) + 1
            continue
        completed_records.append((trace, events))

    if committed_trace_dirs:
        selected_records = [
            (trace, events)
            for trace, events in completed_records
            if trace.parent.resolve() in committed_trace_dirs
        ]
    else:
        # Evaluation runs may not write result rows. In that case retain the
        # latest completed attempt for each task directory.
        latest_by_task_dir = {}
        for trace, events in completed_records:
            latest_by_task_dir[trace.parent.parent.resolve()] = (trace, events)
        selected_records = list(latest_by_task_dir.values())

    updates = []
    rewards = []
    decisions = []
    task_ids = set()
    trajectory_records = []
    selected_agents = Counter()
    selected_backbones = Counter()
    selected_roles = Counter()
    for _trace, events in selected_records:
        rewards_for_trace = []
        decisions_for_trace = []
        for event in events:
            task_ids.add(str(event.get("task_id")))
            event_type = event.get("event_type")
            if event_type == "policy_update":
                updates.append(event)
            elif event_type == "reward_assigned":
                rewards.append(event)
                rewards_for_trace.append(event)
            elif event_type == "routing_decision":
                decisions.append(event)
                decisions_for_trace.append(event)
            elif event_type == "action_started":
                teammate_id = event.get("teammate_id")
                backbone = event.get("backbone")
                role = event.get("role")
                if teammate_id:
                    selected_agents[str(teammate_id)] += 1
                if backbone:
                    selected_backbones[str(backbone)] += 1
                if role:
                    selected_roles[str(role)] += 1

        result_row = committed_result_rows.get(_trace.parent.resolve(), {})
        trace_entropies = []
        trace_margins = []
        for decision in decisions_for_trace:
            probabilities = [
                float(probability)
                for probability, available in zip(
                    decision["probabilities"], decision["mask"], strict=True
                )
                if available
            ]
            if probabilities:
                ordered = sorted(probabilities, reverse=True)
                trace_entropies.append(
                    float(decision["entropy"]) / math.log(len(ordered))
                )
                trace_margins.append(
                    ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
                )
        reward = rewards_for_trace[-1] if rewards_for_trace else {}
        trajectory_records.append(
            {
                "offset": result_row.get("offset"),
                "correct": float(reward.get("task_reward", -1.0)) > 0,
                "path_reward": reward.get("path_reward"),
                "token_cost_penalty": reward.get("token_cost_penalty"),
                "normalized_entropy": _mean(trace_entropies),
                "top1_top2_margin": _mean(trace_margins),
            }
        )

    gradient_norms = _numbers(updates, "gradient_norm")
    finite_gradients = [value for value in gradient_norms if math.isfinite(value)]
    optimizer_steps = sum(event.get("optimizer_stepped") is True for event in updates)
    candidate_counts = [
        len(event.get("threshold_candidates") or []) for event in decisions
    ]
    normalized_entropy = []
    top_margin = []
    probability_range = []
    p_stop = []
    for decision in decisions:
        probabilities = [
            float(probability)
            for probability, available in zip(
                decision["probabilities"], decision["mask"], strict=True
            )
            if available
        ]
        if not probabilities:
            continue
        ordered = sorted(probabilities, reverse=True)
        normalized_entropy.append(float(decision["entropy"]) / math.log(len(ordered)))
        top_margin.append(ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0])
        probability_range.append(ordered[0] - ordered[-1])
        if decision.get("allow_stop"):
            p_stop.append(float(decision.get("p_stop", 0.0)))

    warnings = []
    if not updates:
        warnings.append("No policy_update events were recorded.")
    if updates and optimizer_steps == 0:
        warnings.append("No policy update reports optimizer_stepped=true.")
    if updates and len(gradient_norms) < len(updates):
        warnings.append("Some policy updates are missing gradient_norm.")
    if gradient_norms and len(finite_gradients) != len(gradient_norms):
        warnings.append("At least one gradient norm is NaN or infinite.")
    if finite_gradients and max(finite_gradients) == 0:
        warnings.append("All recorded gradient norms are zero.")
    if decisions and not any(count > 0 for count in candidate_counts):
        warnings.append("No routing decision has an agent above the legacy threshold.")
    if interrupted_traces:
        warnings.append(
            f"Excluded {interrupted_traces} interrupted trace(s) from training metrics."
        )
    uncommitted_completed = len(completed_records) - len(selected_records)
    if uncommitted_completed:
        warnings.append(
            f"Excluded {uncommitted_completed} completed but uncommitted trace(s) "
            "from training metrics."
        )

    correct = sum(float(event.get("task_reward", -1.0)) > 0 for event in rewards)
    ordered_trajectories = sorted(
        trajectory_records,
        key=lambda record: (
            record["offset"] is None,
            record["offset"] if record["offset"] is not None else 0,
        ),
    )
    midpoint = len(ordered_trajectories) // 2
    result = {
        "audit_dir": str(args.audit_dir),
        "traces": {
            "total": len(traces),
            "completed_analyzed": len(selected_records),
            "completed_uncommitted_excluded": uncommitted_completed,
            "interrupted_excluded": interrupted_traces,
            "interruption_types": interruption_types,
            "committed_result_traces": len(committed_trace_dirs),
        },
        "tasks": len(task_ids),
        "updates": {
            "count": len(updates),
            "optimizer_steps": optimizer_steps,
            "optimizer_step_rate": optimizer_steps / len(updates) if updates else None,
            "gradient_norm": _summary(gradient_norms),
            "finite_nonzero_gradient_count": sum(value > 0 for value in finite_gradients),
            "policy_loss": _summary(_numbers(updates, "policy_loss")),
            "task_return": _summary(_numbers(updates, "task_return")),
            "return_variance": _summary(_numbers(updates, "return_variance")),
            "threshold_margin_mean": _summary(_numbers(updates, "threshold_margin_mean")),
            "threshold_margin_loss": _summary(_numbers(updates, "threshold_margin_loss")),
        },
        "rewards": {
            "count": len(rewards),
            "correct": correct,
            "accuracy": correct / len(rewards) if rewards else None,
            "task_reward": _summary(_numbers(rewards, "task_reward")),
            "path_reward": _summary(_numbers(rewards, "path_reward")),
            "step_penalty": _summary(_numbers(rewards, "step_penalty")),
            "token_cost_penalty": _summary(_numbers(rewards, "token_cost_penalty")),
        },
        "routing": {
            "decisions": len(decisions),
            "normalized_entropy": _summary(normalized_entropy),
            "top1_top2_margin": _summary(top_margin),
            "probability_range": _summary(probability_range),
            "p_stop": _summary(p_stop),
            "decisions_with_threshold_candidates": sum(
                count > 0 for count in candidate_counts
            ),
            "selected_agents": dict(sorted(selected_agents.items())),
            "selected_backbones": dict(sorted(selected_backbones.items())),
            "selected_roles": dict(sorted(selected_roles.items())),
        },
        "training_sequence": {
            "ordering": "committed result offset",
            "first_half": _window_summary(ordered_trajectories[:midpoint]),
            "last_half": _window_summary(ordered_trajectories[midpoint:]),
        },
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tasks": len(task_ids),
                "updates": len(updates),
                "optimizer_steps": optimizer_steps,
                "warnings": len(warnings),
            },
            ensure_ascii=False,
        )
    )
    return 1 if any("NaN or infinite" in warning for warning in warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
