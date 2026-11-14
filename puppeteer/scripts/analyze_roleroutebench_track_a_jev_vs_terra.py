from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare live JEV actions with Terra set-valued Track A pre-labels"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT / "data" / "RoleRouteBench" / "track_a_4d_live_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "report" / "roleroutebench_track_a_jev_vs_terra.json",
    )
    args = parser.parse_args()

    provenance = {row["decision_id"]: row for row in read_jsonl(args.root / "track_a_provenance.jsonl")}
    states = {row["decision_id"]: row for row in read_jsonl(args.root / "track_a_states.jsonl")}
    labels: dict[str, dict[str, Any]] = {}
    cohorts: dict[str, str] = {}
    for cohort in sorted(args.root.glob("labeling*")):
        if not (cohort / "sample_manifest.json").exists():
            continue
        for row in read_jsonl(cohort / "llm_prelabels.jsonl"):
            labels[row["decision_id"]] = row
            cohorts[row["decision_id"]] = cohort.name

    ids = sorted(set(states) & set(labels) & set(provenance))
    classification_counts: Counter[str] = Counter()
    valid_set_sizes: Counter[int] = Counter()
    selected_role: dict[str, Counter[str]] = defaultdict(Counter)
    domain_stats: dict[str, Counter[str]] = defaultdict(Counter)
    cohort_stats: dict[str, Counter[str]] = defaultdict(Counter)
    confidence_stats: dict[str, Counter[str]] = defaultdict(Counter)
    packet_stats: dict[str, Counter[str]] = defaultdict(Counter)
    correct = 0
    singleton = 0
    singleton_correct = 0
    executor_total = 0
    executor_correct = 0
    stop_total = 0
    stop_correct = 0

    for decision_id in ids:
        state = states[decision_id]
        label = labels[decision_id]
        source = provenance[decision_id]
        selected = source["source_action_as_run"]
        valid_actions = set(label.get("valid_actions", []))
        action_classes = {
            item["action"]: item["classification"]
            for item in label["classification"]["actions"]
        }
        selected_class = action_classes.get(selected, "missing_from_label")
        is_correct = selected in valid_actions
        domain = state["domain"]
        cohort = cohorts[decision_id]
        confidence = label["classification"]["confidence"]
        packet = source["packet_id"]

        classification_counts[selected_class] += 1
        valid_set_sizes[len(valid_actions)] += 1
        selected_role[selected]["selected"] += 1
        selected_role[selected]["correct"] += int(is_correct)
        selected_role[selected][selected_class] += 1
        for bucket, key in (
            (domain_stats, domain),
            (cohort_stats, cohort),
            (confidence_stats, confidence),
            (packet_stats, packet),
        ):
            bucket[key]["states"] += 1
            bucket[key]["correct"] += int(is_correct)

        correct += int(is_correct)
        if len(valid_actions) == 1:
            singleton += 1
            singleton_correct += int(is_correct)
        if selected == "STOP":
            stop_total += 1
            stop_correct += int(is_correct)
        else:
            executor_total += 1
            executor_correct += int(is_correct)

    def summarize(groups: dict[str, Counter[str]]) -> dict[str, Any]:
        return {
            key: {
                **dict(counts),
                "accuracy": ratio(counts["correct"], counts["states"]),
            }
            for key, counts in sorted(groups.items())
        }

    roles = {
        role: {
            **dict(counts),
            "accuracy": ratio(counts["correct"], counts["selected"]),
        }
        for role, counts in sorted(selected_role.items())
    }
    report = {
        "metric_definition": {
            "primary_accuracy": "JEV source_action_as_run is a member of Terra valid_actions",
            "singleton_exact_agreement": "same metric restricted to states where Terra valid_actions has size 1",
            "note": "Terra supplies a set-valued action label, not one preferred executor.",
        },
        "coverage": {
            "primary_states": len(states),
            "matched_states": len(ids),
            "missing_labels": len(set(states) - set(labels)),
            "missing_provenance": len(set(states) - set(provenance)),
        },
        "overall": {
            "states": len(ids),
            "correct": correct,
            "accuracy": ratio(correct, len(ids)),
            "singleton_states": singleton,
            "singleton_correct": singleton_correct,
            "singleton_exact_agreement": ratio(singleton_correct, singleton),
            "executor_selections": executor_total,
            "executor_correct": executor_correct,
            "executor_accuracy": ratio(executor_correct, executor_total),
            "stop_selections": stop_total,
            "stop_correct": stop_correct,
            "stop_accuracy": ratio(stop_correct, stop_total),
        },
        "jev_selected_action_classification": dict(classification_counts),
        "terra_valid_action_set_size": {str(k): v for k, v in sorted(valid_set_sizes.items())},
        "by_domain": summarize(domain_stats),
        "by_labeling_cohort": summarize(cohort_stats),
        "by_terra_confidence": summarize(confidence_stats),
        "by_selected_role": roles,
        "by_packet": summarize(packet_stats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
