"""Compare two routing audit runs on the same tasks without model calls."""

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


def _load_events(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _find_manifest(audit_dir: Path) -> dict:
    candidates = [audit_dir / "manifest.json", audit_dir.parent / "manifest.json"]
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_run(audit_dir: Path) -> tuple[dict[str, dict], list[dict], dict]:
    tasks: dict[str, dict] = {}
    for path in sorted(audit_dir.rglob("events.jsonl")):
        events = _load_events(path)
        if not events:
            continue
        task_id = str(events[0]["task_id"])
        finished = next(
            (event for event in events if event.get("event_type") == "task_finished"),
            None,
        )
        if finished is None:
            continue
        task_decisions = [
            event for event in events if event.get("event_type") == "routing_decision"
        ]
        reward = next(
            (event for event in events if event.get("event_type") == "reward_assigned"),
            None,
        )
        tasks[task_id] = {
            "correct": bool(reward and float(reward.get("task_reward", -1.0)) > 0),
            "prediction": finished.get("prediction"),
            "decisions": task_decisions,
            "source": str(path),
        }
    if not tasks:
        raise ValueError(f"No completed events.jsonl traces found below {audit_dir}")
    decisions = [
        decision for task in tasks.values() for decision in task["decisions"]
    ]
    return tasks, decisions, _find_manifest(audit_dir)


def distribution_stats(decisions: list[dict]) -> dict:
    entropy = []
    normalized_entropy = []
    kl_uniform = []
    maximum = []
    probability_range = []
    top_margin = []
    candidate_counts = []
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
        count = len(probabilities)
        observed_entropy = float(decision["entropy"])
        entropy.append(observed_entropy)
        normalized_entropy.append(observed_entropy / math.log(count))
        kl_uniform.append(math.log(count) - observed_entropy)
        maximum.append(ordered[0])
        probability_range.append(ordered[0] - ordered[-1])
        top_margin.append(ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0])
        candidate_counts.append(len(decision.get("threshold_candidates") or []))
        if decision.get("allow_stop"):
            p_stop.append(float(decision.get("p_stop", 0.0)))

    return {
        "decisions": len(decisions),
        "entropy_mean": _mean(entropy),
        "normalized_entropy_mean": _mean(normalized_entropy),
        "kl_from_uniform_mean": _mean(kl_uniform),
        "max_probability_mean": _mean(maximum),
        "probability_range_mean": _mean(probability_range),
        "top1_top2_margin_mean": _mean(top_margin),
        "threshold_candidates_mean": _mean(candidate_counts),
        "decisions_with_threshold_candidates": sum(
            count > 0 for count in candidate_counts
        ),
        "p_stop_mean": _mean(p_stop),
        "p_stop_min": min(p_stop) if p_stop else None,
        "p_stop_max": max(p_stop) if p_stop else None,
    }


def task_sensitivity(tasks: dict[str, dict]) -> dict | None:
    vectors = []
    for task in tasks.values():
        root = next(
            (
                decision
                for decision in task["decisions"]
                if int(decision.get("steps_completed", 0)) == 0
            ),
            None,
        )
        if root is None:
            continue
        vectors.append(
            [
                float(probability)
                for probability, available in zip(
                    root["probabilities"], root["mask"], strict=True
                )
                if available
            ]
        )
    if not vectors or len({len(vector) for vector in vectors}) != 1:
        return None
    center = [_mean(vector[index] for vector in vectors) for index in range(len(vectors[0]))]
    distances = [
        sum(abs(value - average) for value, average in zip(vector, center, strict=True))
        for vector in vectors
    ]
    per_action_std = [
        statistics.pstdev(vector[index] for vector in vectors)
        for index in range(len(center))
    ]
    return {
        "root_tasks": len(vectors),
        "mean_l1_from_mean_distribution": _mean(distances),
        "max_l1_from_mean_distribution": max(distances),
        "mean_per_action_std": _mean(per_action_std),
    }


def summarize_run(tasks: dict[str, dict], decisions: list[dict]) -> dict:
    roots = [
        decision
        for decision in decisions
        if int(decision.get("steps_completed", 0)) == 0
    ]
    post_step = [
        decision
        for decision in decisions
        if int(decision.get("steps_completed", 0)) > 0
    ]
    return {
        "tasks": len(tasks),
        "correct": sum(task["correct"] for task in tasks.values()),
        "accuracy": _mean(task["correct"] for task in tasks.values()),
        "root": distribution_stats(roots),
        "post_step": distribution_stats(post_step),
        "task_sensitivity": task_sensitivity(tasks),
        "selected_action_counts": dict(
            Counter(
                action
                for decision in decisions
                for action in (decision.get("selected") or [])
            )
        ),
    }


def _manifest_value(manifest: dict, dotted_path: str):
    value = manifest
    for part in dotted_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def compare_manifests(left: dict, right: dict) -> dict:
    invariants = [
        "profile_hash",
        "pool_hash",
        "split_hash",
        "routing_mode",
        "config.policy.policy_mode",
        "config.policy.routing.mode",
        "config.policy.routing.threshold_multiplier",
    ]
    values = {}
    mismatches = []
    for key in invariants:
        left_value = _manifest_value(left, key)
        right_value = _manifest_value(right, key)
        values[key] = {"left": left_value, "right": right_value}
        if left_value != right_value:
            mismatches.append(key)
    return {
        "comparable": not mismatches,
        "mismatches": mismatches,
        "values": values,
        "checkpoint_hashes": {
            "left": left.get("checkpoint_hash"),
            "right": right.get("checkpoint_hash"),
        },
    }


def paired_comparison(left: dict[str, dict], right: dict[str, dict]) -> dict:
    common = sorted(set(left) & set(right), key=lambda value: (len(value), value))
    transitions = Counter()
    root_same = 0
    sequence_same = 0
    step_same = 0
    step_total = 0
    probability_l1 = []
    discordant = []

    for task_id in common:
        left_task = left[task_id]
        right_task = right[task_id]
        before = "correct" if left_task["correct"] else "wrong"
        after = "correct" if right_task["correct"] else "wrong"
        transitions[f"{before}->{after}"] += 1

        left_decisions = left_task["decisions"]
        right_decisions = right_task["decisions"]
        left_sequence = [tuple(item.get("selected") or []) for item in left_decisions]
        right_sequence = [tuple(item.get("selected") or []) for item in right_decisions]
        root_same += bool(
            left_sequence and right_sequence and left_sequence[0] == right_sequence[0]
        )
        sequence_same += left_sequence == right_sequence
        for left_decision, right_decision in zip(
            left_decisions, right_decisions, strict=False
        ):
            step_total += 1
            step_same += tuple(left_decision.get("selected") or []) == tuple(
                right_decision.get("selected") or []
            )
            if len(left_decision["probabilities"]) == len(
                right_decision["probabilities"]
            ):
                probability_l1.append(
                    sum(
                        abs(float(a) - float(b))
                        for a, b in zip(
                            left_decision["probabilities"],
                            right_decision["probabilities"],
                            strict=True,
                        )
                    )
                )
        if left_task["correct"] != right_task["correct"]:
            discordant.append(
                {
                    "task_id": task_id,
                    "outcome": "left_only" if left_task["correct"] else "right_only",
                    "left_prediction": left_task["prediction"],
                    "right_prediction": right_task["prediction"],
                    "left_selected": [list(value) for value in left_sequence],
                    "right_selected": [list(value) for value in right_sequence],
                }
            )

    return {
        "common_tasks": len(common),
        "left_only_tasks": sorted(set(left) - set(right)),
        "right_only_tasks": sorted(set(right) - set(left)),
        "outcome_transitions": dict(transitions),
        "root_selection_same": root_same,
        "full_sequence_same": sequence_same,
        "selected_step_agreement": step_same / step_total if step_total else None,
        "mean_probability_l1_between_runs": _mean(probability_l1),
        "max_probability_l1_between_runs": max(probability_l1)
        if probability_l1
        else None,
        "discordant_tasks": discordant,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="First run's audit directory")
    parser.add_argument("right", type=Path, help="Second run's audit directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    left_tasks, left_decisions, left_manifest = load_run(args.left)
    right_tasks, right_decisions, right_manifest = load_run(args.right)
    result = {
        "left": str(args.left),
        "right": str(args.right),
        "manifest_check": compare_manifests(left_manifest, right_manifest),
        "runs": {
            "left": summarize_run(left_tasks, left_decisions),
            "right": summarize_run(right_tasks, right_decisions),
        },
        "paired": paired_comparison(left_tasks, right_tasks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "comparable": result["manifest_check"]["comparable"],
                "left_accuracy": result["runs"]["left"]["accuracy"],
                "right_accuracy": result["runs"]["right"]["accuracy"],
                "selected_step_agreement": result["paired"][
                    "selected_step_agreement"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["manifest_check"]["comparable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
