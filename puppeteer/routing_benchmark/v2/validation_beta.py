from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from .environment import RoutingEnvironmentV2
from .generator import generate_decision_rows
from .generator_beta import build_beta_bundle
from .loader import BenchmarkBundleV2, validate_bundle
from .schemas import STOP_ACTION


FORBIDDEN_PUBLIC_TERMS = (
    "backbone",
    "capability_score",
    "empirical_accuracy",
    "uncertainty",
    "distractor_only",
    "progress_actions",
    "oracle_efficient_actions",
    "valid_actions",
    "admissible_actions",
)


def _forbidden_term(value: Any) -> str | None:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True).casefold()
    return next((term for term in FORBIDDEN_PUBLIC_TERMS if term in serialized), None)


def _public_signature(row: dict[str, Any]) -> str:
    public = {
        "task": row["task"],
        "task_context": row.get("task_context", {}),
        "observed_state": row["observed_state"],
        "execution_memory": row.get("execution_memory", {}),
        "recent_events": row.get("recent_events", []),
        "budget": row["budget"],
        "agent_pool": row["agent_pool"],
        "candidate_actions": row["candidate_actions"],
    }
    return json.dumps(public, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_beta(bundle: BenchmarkBundleV2) -> dict[str, Any]:
    validate_bundle(bundle)
    errors: list[str] = []
    warnings: list[str] = []
    family_counts = Counter(item.family for item in bundle.scenarios)
    split_counts = Counter(item.split for item in bundle.scenarios)
    if len(bundle.scenarios) != 120:
        errors.append(f"Expected 120 beta workflows, got {len(bundle.scenarios)}")
    if set(family_counts) != {
        "sequential_dependency",
        "parallel_join",
        "conditional_branch",
        "conflict_resolution",
        "tool_routing",
        "failure_recovery",
        "terminal_control",
        "budget_sensitive",
    } or set(family_counts.values()) != {15}:
        errors.append(f"Expected 8 families x 15 workflows, got {dict(family_counts)}")
    if dict(split_counts) != {"development": 24, "validation": 32, "test": 64}:
        errors.append(f"Unexpected beta split counts: {dict(split_counts)}")

    group_splits: dict[str, set[str]] = defaultdict(set)
    for scenario in bundle.scenarios:
        group_splits[scenario.split_group_id].add(scenario.split)
        env = RoutingEnvironmentV2(scenario, bundle.agents)
        if not env.action_outcomes():
            errors.append(f"Initial state is unsolvable: {scenario.scenario_id}")
        if not env.admissible_actions():
            errors.append(f"Initial state has no admissible action: {scenario.scenario_id}")
        forbidden = _forbidden_term(env.observation())
        if forbidden:
            errors.append(
                f"Public observation leaks evaluator label {forbidden!r}: "
                f"{scenario.scenario_id}"
            )
    leaking_groups = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaking_groups:
        errors.append(f"Split groups cross releases: {sorted(leaking_groups)}")

    rows = generate_decision_rows(bundle)
    signatures: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    counterfactual: dict[str, list[set[str]]] = defaultdict(list)
    for row in rows:
        admissible = set(row["admissible_actions"])
        progress = set(row["progress_actions"])
        efficient = set(row["oracle_efficient_actions"])
        if not admissible:
            errors.append(f"No admissible action: {row['decision_id']}")
        if not progress.issubset(admissible):
            errors.append(f"Progress action is not admissible: {row['decision_id']}")
        if not efficient.issubset(progress):
            errors.append(f"Efficient action is not a progress action: {row['decision_id']}")
        if row["stop_ready"] and admissible != {STOP_ACTION}:
            errors.append(f"STOP-ready state has another action: {row['decision_id']}")
        public = {
            key: row.get(key)
            for key in (
                "task",
                "task_context",
                "observed_state",
                "execution_memory",
                "recent_events",
                "budget",
                "agent_pool",
                "candidate_actions",
            )
        }
        forbidden = _forbidden_term(public)
        if forbidden:
            errors.append(
                f"Decision input leaks evaluator label {forbidden!r}: "
                f"{row['decision_id']}"
            )
        signatures[_public_signature(row)].add(tuple(sorted(admissible)))
        group = row.get("counterfactual_group_id")
        if group:
            counterfactual[str(group)].append(admissible)

    collisions = [
        signature for signature, label_sets in signatures.items() if len(label_sets) > 1
    ]
    if collisions:
        errors.append(
            f"Found {len(collisions)} observationally identical inputs with conflicting "
            "admissible-action labels"
        )
    for group, label_sets in counterfactual.items():
        if len(label_sets) != 2:
            errors.append(f"Counterfactual group {group} has {len(label_sets)} rows")
        elif label_sets[0] == label_sets[1]:
            errors.append(f"Counterfactual group {group} does not change admissible actions")

    provenance = Counter(row["state_provenance"] for row in rows)
    if provenance.get("post_failure", 0) == 0:
        errors.append("Beta has no post-execution-failure state")
    if provenance.get("post_routing_rejection", 0) == 0:
        errors.append("Beta has no post-routing-rejection state")
    if not any(len(row["admissible_actions"]) > 1 for row in rows):
        errors.append("Beta has no set-valued admissible-action state")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "workflow_count": len(bundle.scenarios),
        "decision_state_count": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "counterfactual_group_count": len(counterfactual),
        "observability_collision_count": len(collisions),
        "provenance_counts": dict(sorted(provenance.items())),
    }


def validate_beta_determinism(seed: int = 42) -> bool:
    left = build_beta_bundle(seed)
    right = build_beta_bundle(seed)
    return (
        [item.to_dict() for item in left.scenarios]
        == [item.to_dict() for item in right.scenarios]
        and generate_decision_rows(left) == generate_decision_rows(right)
    )

