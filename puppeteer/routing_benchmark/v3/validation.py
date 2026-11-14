from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from routing_benchmark.v2.loader import validate_bundle

from .generator import PROJECT_DIR, quota_report_v3
from .loader import BenchmarkBundleV3
from .schemas import STOP_ACTION, V3_SCHEMA_VERSION


FORBIDDEN_PUBLIC_TERMS = (
    "valid_actions",
    "preferred_actions",
    "action_utilities",
    "capability_score",
    "backbone",
    "failure_origin",
    "distractor_only",
)


def _public_input(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": row["task"],
        "task_context": row.get("task_context", {}),
        "observed_state": row["observed_state"],
        "execution_memory": row.get("execution_memory", {}),
        "recent_events": row.get("recent_events", []),
        "budget": row["budget"],
        "agent_pool": row["agent_pool"],
        "candidate_actions": row["candidate_actions"],
    }


def _signature(row: dict[str, Any]) -> str:
    return json.dumps(
        _public_input(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _read_source_ids(path: Path) -> set[str]:
    values: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if value.get("scenario_id"):
                    values.add(str(value["scenario_id"]))
    return values


def validate_v3(bundle: BenchmarkBundleV3) -> dict[str, Any]:
    validate_bundle(bundle.as_v2_bundle())
    errors: list[str] = []
    warnings: list[str] = []
    if bundle.metadata.get("version") != V3_SCHEMA_VERSION:
        errors.append(f"Expected {V3_SCHEMA_VERSION}, got {bundle.metadata.get('version')}")
    if bundle.metadata.get("preferred_actions") is not False:
        errors.append("V3 metadata must explicitly disable preferred_actions")
    if bundle.metadata.get("action_utilities") is not False:
        errors.append("V3 metadata must explicitly disable action_utilities")

    scenario_ids = {item.scenario_id for item in bundle.scenarios}
    provenance_scenarios = {item.scenario_id for item in bundle.provenance.values()}
    if scenario_ids != provenance_scenarios:
        errors.append(
            "Scenario/provenance mismatch: "
            f"missing={sorted(scenario_ids - provenance_scenarios)}, "
            f"extra={sorted(provenance_scenarios - scenario_ids)}"
        )

    source_cache: dict[Path, tuple[str, set[str]]] = {}
    for item in bundle.provenance.values():
        source_path = PROJECT_DIR / item.source_path
        if not source_path.exists():
            errors.append(f"Missing provenance source: {source_path}")
            continue
        if source_path not in source_cache:
            source_cache[source_path] = (
                hashlib.sha256(source_path.read_bytes()).hexdigest(),
                _read_source_ids(source_path),
            )
        digest, source_ids = source_cache[source_path]
        if digest != item.source_sha256:
            errors.append(f"Source hash mismatch for {item.provenance_id}")
        if item.source_item_id not in source_ids:
            errors.append(
                f"Unknown source item {item.source_item_id} for {item.provenance_id}"
            )
        if item.human_review_status == "pending":
            warnings.append(f"Human review pending: {item.provenance_id}")

    signatures: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    decision_ids: set[str] = set()
    for row in bundle.decision_rows:
        decision_id = str(row.get("decision_id"))
        if decision_id in decision_ids:
            errors.append(f"Duplicate decision ID: {decision_id}")
        decision_ids.add(decision_id)
        if row.get("scenario_id") not in scenario_ids:
            errors.append(f"Unknown scenario in {decision_id}")
        if row.get("provenance_id") not in bundle.provenance:
            errors.append(f"Unknown provenance in {decision_id}")
        for forbidden in ("preferred_actions", "action_utilities", "capability_scores"):
            if forbidden in row:
                errors.append(f"Forbidden label {forbidden!r} in {decision_id}")
        public_serialized = json.dumps(
            _public_input(row), ensure_ascii=False, sort_keys=True
        ).casefold()
        for forbidden in FORBIDDEN_PUBLIC_TERMS:
            if forbidden in public_serialized:
                errors.append(f"Public input leaks {forbidden!r}: {decision_id}")
        candidates = set(map(str, row.get("candidate_actions") or ()))
        valid = set(map(str, row.get("valid_actions") or ()))
        if not valid:
            errors.append(f"No valid action in {decision_id}")
        if not valid.issubset(candidates):
            errors.append(f"Valid action is not a candidate in {decision_id}")
        rationales = row.get("action_rationales") or {}
        if set(rationales) != candidates:
            errors.append(f"Rationale coverage mismatch in {decision_id}")
        for action in candidates:
            rationale = rationales.get(action) or {}
            expected = "valid" if action in valid else "invalid"
            if rationale.get("status") != expected:
                errors.append(
                    f"Rationale status mismatch for {decision_id}/{action}: "
                    f"expected {expected}"
                )
            if rationale.get("rationale_source") != "formal_transition_preconditions":
                errors.append(f"Untraceable rationale for {decision_id}/{action}")
            if expected == "valid" and rationale.get("violated_preconditions"):
                errors.append(f"Valid action has violated preconditions: {decision_id}/{action}")
            if expected == "invalid" and not rationale.get("violated_preconditions"):
                errors.append(f"Invalid action lacks violated preconditions: {decision_id}/{action}")
        if bool(row.get("terminal_valid")) != (valid == {STOP_ACTION}):
            errors.append(f"Terminal label mismatch in {decision_id}")
        signatures[_signature(row)].add(tuple(sorted(valid)))

    collisions = [key for key, labels in signatures.items() if len(labels) > 1]
    if collisions:
        errors.append(
            f"Found {len(collisions)} observationally identical states with conflicting labels"
        )

    split_groups: dict[str, set[str]] = defaultdict(set)
    for item in bundle.scenarios:
        split_groups[item.split_group_id].add(item.split)
    crossing = [key for key, values in split_groups.items() if len(values) > 1]
    if crossing:
        errors.append(f"Split groups cross partitions: {sorted(crossing)}")

    quota = quota_report_v3(list(bundle.decision_rows), bundle.scenarios)
    failed_targets = [key for key, value in quota["targets"].items() if not value]
    if failed_targets:
        errors.append(f"V3-alpha quota targets failed: {failed_targets}")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "workflow_count": len(bundle.scenarios),
        "decision_state_count": len(bundle.decision_rows),
        "family_counts": dict(
            sorted(Counter(item.family for item in bundle.scenarios).items())
        ),
        "split_counts": dict(
            sorted(Counter(item.split for item in bundle.scenarios).items())
        ),
        "provenance_count": len(bundle.provenance),
        "human_review_pending_count": sum(
            item.human_review_status == "pending" for item in bundle.provenance.values()
        ),
        "observability_collision_count": len(collisions),
        "quota_report": quota,
    }
