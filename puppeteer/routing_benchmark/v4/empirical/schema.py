from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


EMPIRICAL_SCHEMA_VERSION = "4.3-empirical-state-history-v1"
OUTCOME_STATUSES = {
    "success",
    "partial",
    "no_progress",
    "invalid_output",
    "tool_failure",
    "transport_failure",
    "failed",
    "rejected",
    "timeout",
}
ACTION_LABELS = {"valid", "invalid", "uncertain"}


def canonical_hash(value: Any, length: int = 24) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def output_id_for(
    *,
    workflow_id: str,
    pre_state_fingerprint: str,
    action: str,
    executor_profile_id: str,
    replicate_id: int,
) -> str:
    digest = canonical_hash(
        {
            "workflow_id": workflow_id,
            "pre_state_fingerprint": pre_state_fingerprint,
            "action": action,
            "executor_profile_id": executor_profile_id,
            "replicate_id": int(replicate_id),
        }
    )
    return f"emp-{digest}"


def outcome_id_for(output_id: str, prompt_hash: str) -> str:
    return f"outcome-{canonical_hash([output_id, prompt_hash])}"


def empirical_decision_id(output_id: str, memory_view: str = "state") -> str:
    return f"empirical-decision-{canonical_hash([output_id, memory_view])}"


def require_fields(record: Mapping[str, Any], fields: Sequence[str], kind: str) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(f"{kind} is missing required fields: {missing}")


def validate_empirical_output(record: Mapping[str, Any]) -> None:
    require_fields(
        record,
        (
            "schema_version",
            "output_id",
            "scenario_id",
            "workflow_id",
            "pre_state_fingerprint",
            "selected_action",
            "executor_profile_id",
            "output_origin",
            "raw_output",
            "generation",
            "transport_status",
        ),
        "empirical output",
    )
    if record["schema_version"] != EMPIRICAL_SCHEMA_VERSION:
        raise ValueError("Unexpected empirical output schema version")
    if record["output_origin"] != "empirical":
        raise ValueError("Empirical output must have output_origin='empirical'")
    if not isinstance(record["raw_output"], str):
        raise ValueError("raw_output must be a string")


def validate_outcome(record: Mapping[str, Any]) -> None:
    require_fields(
        record,
        (
            "schema_version",
            "outcome_id",
            "output_id",
            "execution_status",
            "progress_status",
            "produced_artifacts",
            "observable_facts_added",
            "resolved_requirements",
            "new_open_requirements",
            "observable_issues",
            "completion_claimed",
            "state_delta_summary",
            "judge_provenance",
            "review_status",
        ),
        "outcome interpretation",
    )
    if record["execution_status"] not in OUTCOME_STATUSES:
        raise ValueError(f"Unknown execution status: {record['execution_status']}")
    for key in (
        "produced_artifacts",
        "observable_facts_added",
        "resolved_requirements",
        "new_open_requirements",
        "observable_issues",
    ):
        if not isinstance(record[key], list):
            raise ValueError(f"{key} must be a list")


def validate_action_annotation(record: Mapping[str, Any]) -> None:
    require_fields(
        record,
        (
            "schema_version",
            "decision_id",
            "candidate_actions",
            "action_annotations",
            "proposed_valid_actions",
            "judge_provenance",
            "review_status",
        ),
        "action annotation",
    )
    candidates = set(map(str, record["candidate_actions"]))
    seen: set[str] = set()
    for item in record["action_annotations"]:
        action = str(item.get("action", ""))
        label = str(item.get("label", ""))
        if action not in candidates:
            raise ValueError(f"Annotated action outside candidate set: {action}")
        if action in seen:
            raise ValueError(f"Duplicate action annotation: {action}")
        if label not in ACTION_LABELS:
            raise ValueError(f"Unknown action label {label!r} for {action}")
        seen.add(action)
    if seen != candidates:
        raise ValueError("Every candidate action must receive exactly one label")
    proposed = set(map(str, record["proposed_valid_actions"]))
    labeled_valid = {
        str(item["action"])
        for item in record["action_annotations"]
        if item["label"] == "valid"
    }
    if proposed != labeled_valid:
        raise ValueError("proposed_valid_actions disagrees with per-action labels")


def schema_definitions() -> dict[str, dict[str, Any]]:
    """Return release schemas without embedding provider-specific prompt text."""

    common_string_array = {"type": "array", "items": {"type": "string"}}
    definitions = {
        "empirical_output.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": [
                "schema_version",
                "output_id",
                "scenario_id",
                "workflow_id",
                "pre_state_fingerprint",
                "selected_action",
                "executor_profile_id",
                "output_origin",
                "raw_output",
                "generation",
                "transport_status",
            ],
            "properties": {
                "schema_version": {"const": EMPIRICAL_SCHEMA_VERSION},
                "output_id": {"type": "string"},
                "scenario_id": {"type": "string"},
                "workflow_id": {"type": "string"},
                "pre_state_fingerprint": {"type": "string"},
                "selected_action": {"type": "string"},
                "executor_profile_id": {"type": "string"},
                "output_origin": {"const": "empirical"},
                "raw_output": {"type": "string"},
                "generation": {"type": "object"},
                "transport_status": {"type": "string"},
            },
        },
        "outcome_interpretation.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": [
                "schema_version",
                "outcome_id",
                "output_id",
                "execution_status",
                "progress_status",
                "produced_artifacts",
                "observable_facts_added",
                "resolved_requirements",
                "new_open_requirements",
                "observable_issues",
                "completion_claimed",
                "state_delta_summary",
                "judge_provenance",
                "review_status",
            ],
            "properties": {
                "schema_version": {"const": EMPIRICAL_SCHEMA_VERSION},
                "outcome_id": {"type": "string"},
                "output_id": {"type": "string"},
                "execution_status": {"enum": sorted(OUTCOME_STATUSES)},
                "progress_status": {"type": "string"},
                "produced_artifacts": {"type": "array", "items": {"type": "object"}},
                "observable_facts_added": common_string_array,
                "resolved_requirements": common_string_array,
                "new_open_requirements": common_string_array,
                "observable_issues": {"type": "array", "items": {"type": "object"}},
                "completion_claimed": {"type": "boolean"},
                "state_delta_summary": {"type": "string"},
                "judge_provenance": {"type": "object"},
                "review_status": {"type": "string"},
            },
        },
        "action_annotation.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": [
                "schema_version",
                "decision_id",
                "candidate_actions",
                "action_annotations",
                "proposed_valid_actions",
                "judge_provenance",
                "review_status",
            ],
            "properties": {
                "schema_version": {"const": EMPIRICAL_SCHEMA_VERSION},
                "decision_id": {"type": "string"},
                "candidate_actions": common_string_array,
                "action_annotations": {"type": "array", "items": {"type": "object"}},
                "proposed_valid_actions": common_string_array,
                "judge_provenance": {"type": "object"},
                "review_status": {"type": "string"},
            },
        },
    }
    try:
        from routing_benchmark.v4.execution.schema import schema_definitions as execution_schemas

        definitions.update(execution_schemas())
    except ImportError:
        pass
    return definitions
