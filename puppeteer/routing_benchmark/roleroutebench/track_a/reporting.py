from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from .canonicalize import read_jsonl, write_json


def _usage_value(usage: dict[str, Any], *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _all_prelabels(root: Path) -> dict[str, dict[str, Any]]:
    prelabels: dict[str, dict[str, Any]] = {}
    for cohort in sorted(root.glob("labeling*")):
        if not cohort.is_dir() or not (cohort / "sample_manifest.json").exists():
            continue
        for row in read_jsonl(cohort / "llm_prelabels.jsonl"):
            prelabels[row["decision_id"]] = row
    return prelabels


def build_prelabel_report(root: Path, labeling_dir: str = "labeling") -> dict[str, Any]:
    label_root = root / labeling_dir
    states = {row["decision_id"]: row for row in read_jsonl(label_root / "sample_states.jsonl")}
    prelabels = read_jsonl(label_root / "llm_prelabels.jsonl")
    attempts = read_jsonl(label_root / "llm_attempts.jsonl")
    latest_prelabel = {row["decision_id"]: row for row in prelabels}
    input_values: list[int] = []
    output_values: list[int] = []
    reasoning_values: list[int] = []
    provider_cost_values: list[float] = []
    latency_values: list[float] = []
    phase_counts = Counter()
    confidence_counts = Counter()
    domain_counts = Counter()
    classification_counts = Counter()
    for decision_id, row in latest_prelabel.items():
        usage = row.get("usage") or {}
        input_values.append(_usage_value(usage, "input_tokens", "prompt_tokens"))
        output_values.append(_usage_value(usage, "output_tokens", "completion_tokens"))
        reasoning_values.append(
            _usage_value(usage, "reasoning_output_tokens", "reasoning_tokens")
            or _usage_value(usage.get("output_tokens_details") or {}, "reasoning_tokens")
            or _usage_value(usage.get("completion_tokens_details") or {}, "reasoning_tokens")
        )
        if isinstance(usage.get("cost"), (int, float)):
            provider_cost_values.append(float(usage["cost"]))
        latency_values.append(float(row.get("latency_ms") or 0))
        phase_counts[row.get("state_phase")] += 1
        confidence_counts[(row.get("classification") or {}).get("confidence")] += 1
        domain_counts[states.get(decision_id, {}).get("domain", "unknown")] += 1
        for action in (row.get("classification") or {}).get("actions", []):
            classification_counts[action.get("classification")] += 1
    total_input = sum(input_values)
    total_output = sum(output_values)
    report = {
        "sample_count": len(states),
        "successful_prelabels": len(latest_prelabel),
        "unlabeled": len(states) - len(latest_prelabel),
        "attempts": len(attempts),
        "attempt_status_counts": dict(Counter(row.get("status") for row in attempts)),
        "domain_counts": dict(domain_counts),
        "phase_counts": dict(phase_counts),
        "confidence_counts": dict(confidence_counts),
        "action_classification_counts": dict(classification_counts),
        "usage": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "reasoning_tokens": sum(reasoning_values),
            "total_tokens": total_input + total_output,
            "estimated_standard_usd": total_input / 1_000_000 + total_output * 5 / 1_000_000,
            "provider_reported_usd": sum(provider_cost_values) if provider_cost_values else None,
            "input_per_state_mean": mean(input_values) if input_values else None,
            "input_per_state_median": median(input_values) if input_values else None,
            "output_per_state_mean": mean(output_values) if output_values else None,
            "reasoning_per_state_mean": mean(reasoning_values) if reasoning_values else None,
            "latency_ms_mean": mean(latency_values) if latency_values else None,
            "latency_ms_median": median(latency_values) if latency_values else None,
        },
        "provenance": {
            "model_ids": sorted({row.get("model_id") for row in latest_prelabel.values()}),
            "providers": sorted({row.get("model_provider") for row in latest_prelabel.values()}),
            "reasoning_efforts": sorted({row.get("reasoning_effort") for row in latest_prelabel.values()}),
            "prompt_versions": sorted({row.get("prompt_version") for row in latest_prelabel.values()}),
        },
    }
    write_json(label_root / "prelabel_report.json", report)
    return report


def build_human_review_report(root: Path) -> dict[str, Any]:
    prelabels = _all_prelabels(root)
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(root / "reviews" / "human_reviews.jsonl"):
        latest[row["decision_id"]] = row
    decisions = Counter(row.get("decision") for row in latest.values())
    exact = 0
    action_changes = 0
    stop_changes = 0
    unsafe_set_changes = 0
    for decision_id, review in latest.items():
        prelabel = prelabels.get(decision_id)
        if not prelabel:
            continue
        before = prelabel["classification"]
        after = review["classification"]
        if before == after:
            exact += 1
        before_classes = {item["action"]: item["classification"] for item in before["actions"]}
        after_classes = {item["action"]: item["classification"] for item in after["actions"]}
        action_changes += sum(before_classes.get(action) != value for action, value in after_classes.items())
        stop_changes += before_classes.get("STOP") != after_classes.get("STOP")
        unsafe_set_changes += set(prelabel.get("unsafe_actions", [])) != set(review.get("unsafe_actions", []))
    report = {
        "reviewed": len(latest),
        "decision_counts": dict(decisions),
        "exact_prelabel_acceptance": exact,
        "exact_prelabel_acceptance_rate": exact / len(latest) if latest else None,
        "action_class_changes": action_changes,
        "stop_class_changes": stop_changes,
        "unsafe_set_changes": unsafe_set_changes,
    }
    write_json(root / "reviews" / "human_review_report.json", report)
    return report

