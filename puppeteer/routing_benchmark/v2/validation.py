from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from typing import Any

from .environment import RoutingEnvironmentV2
from .generator import build_alpha_bundle, generate_decision_rows
from .loader import BenchmarkBundleV2, validate_bundle
from .schemas import STOP_ACTION


FORBIDDEN_PUBLIC_TERMS = (
    "backbone",
    "capability_score",
    "empirical_accuracy",
    "uncertainty",
    "distractor_only",
)


def _public_has_forbidden_term(value: Any) -> str | None:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True).casefold()
    return next((term for term in FORBIDDEN_PUBLIC_TERMS if term in serialized), None)


def validate_alpha(bundle: BenchmarkBundleV2) -> dict[str, Any]:
    validate_bundle(bundle)
    errors: list[str] = []
    warnings: list[str] = []
    family_counts = Counter(item.family for item in bundle.scenarios)
    if len(bundle.scenarios) != 16:
        errors.append(f"Expected 16 alpha workflows, got {len(bundle.scenarios)}")
    if len(family_counts) != 8 or set(family_counts.values()) != {2}:
        errors.append(f"Expected 8 families x 2 workflows, got {dict(family_counts)}")

    for scenario in bundle.scenarios:
        env = RoutingEnvironmentV2(scenario, bundle.agents)
        if not env.action_outcomes():
            errors.append(f"Initial state is unsolvable: {scenario.scenario_id}")
        if "agent_" in scenario.task.casefold():
            errors.append(f"Task leaks an opaque action ID: {scenario.scenario_id}")
        forbidden = _public_has_forbidden_term(env.observation())
        if forbidden:
            errors.append(
                f"Public observation leaks forbidden field {forbidden!r}: "
                f"{scenario.scenario_id}"
            )

        queue = deque([env])
        seen = {env.state.fingerprint()}
        checked = 0
        while queue and checked < 128:
            current = queue.popleft()
            checked += 1
            valid = set(current.valid_actions())
            efficient = set(current.efficient_actions())
            if not valid:
                errors.append(
                    f"Reachable state has no successful continuation: "
                    f"{scenario.scenario_id}:{current.state.fingerprint()}"
                )
                continue
            if not efficient.issubset(valid):
                errors.append(
                    f"Efficient actions are not a valid subset: {scenario.scenario_id}"
                )
            if current.goal_reached and valid != {STOP_ACTION}:
                errors.append(
                    f"Goal state has non-STOP valid actions: {scenario.scenario_id}"
                )
            for action in valid:
                if action == STOP_ACTION:
                    continue
                child = current.clone()
                child.step(action)
                fingerprint = child.state.fingerprint()
                if fingerprint not in seen and not child.done:
                    seen.add(fingerprint)
                    queue.append(child)

    rows = generate_decision_rows(bundle)
    for row in rows:
        if not row["valid_actions"]:
            errors.append(f"Decision row has no valid action: {row['decision_id']}")
        if not set(row["efficient_actions"]).issubset(row["valid_actions"]):
            errors.append(f"Invalid efficient-action label: {row['decision_id']}")
        public_row = {
            key: row[key]
            for key in (
                "task",
                "observed_state",
                "budget",
                "agent_pool",
                "candidate_actions",
            )
        }
        forbidden = _public_has_forbidden_term(public_row)
        if forbidden:
            errors.append(
                f"Decision public input leaks forbidden field {forbidden!r}: "
                f"{row['decision_id']}"
            )

    counterfactual_sets: dict[str, list[set[str]]] = defaultdict(list)
    for row in rows:
        group = row.get("counterfactual_group_id")
        if group:
            counterfactual_sets[str(group)].append(set(row["valid_actions"]))
    for group, valid_sets in counterfactual_sets.items():
        if len(valid_sets) != 2:
            errors.append(
                f"Counterfactual group {group} must have exactly two initial rows"
            )
        elif valid_sets[0] == valid_sets[1]:
            errors.append(
                f"Counterfactual group {group} does not change valid actions"
            )

    if not any(row["state_provenance"] == "post_failure" for row in rows):
        errors.append("Alpha contains no post-failure decision state")
    if not any(row["state_provenance"] == "post_invalid" for row in rows):
        warnings.append("Alpha contains no post-invalid decision state")
    if not any(len(row["valid_actions"]) > 1 for row in rows):
        errors.append("Alpha contains no set-valued valid-action state")
    if not any(
        len(row["valid_actions"]) > len(row["efficient_actions"]) for row in rows
    ):
        errors.append("Alpha contains no valid-vs-efficient tradeoff state")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "workflow_count": len(bundle.scenarios),
        "decision_state_count": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "counterfactual_group_count": len(counterfactual_sets),
    }


def validate_deterministic_generation(seed: int = 42) -> bool:
    left = build_alpha_bundle(seed)
    right = build_alpha_bundle(seed)
    left_payload = [item.to_dict() for item in left.scenarios]
    right_payload = [item.to_dict() for item in right.scenarios]
    return left_payload == right_payload and generate_decision_rows(
        left
    ) == generate_decision_rows(right)
