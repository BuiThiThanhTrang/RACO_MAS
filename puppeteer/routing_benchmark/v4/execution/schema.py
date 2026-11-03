from __future__ import annotations

from typing import Any, Mapping, Sequence


EXECUTION_SCHEMA_VERSION = "4.2-execution-rc1"
COUNTERFACTUAL_SCHEMA_VERSION = "4.2-counterfactual-rc1"


def require_fields(record: Mapping[str, Any], fields: Sequence[str], kind: str) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(f"{kind} is missing required fields: {missing}")


def validate_tool_event(record: Mapping[str, Any]) -> None:
    require_fields(record, ("tool_call", "tool_result"), "tool event")
    call = record["tool_call"]
    result = record["tool_result"]
    if not isinstance(call, Mapping) or not str(call.get("name", "")):
        raise ValueError("tool_call must contain a non-empty name")
    if not isinstance(result, Mapping) or not str(result.get("status", "")):
        raise ValueError("tool_result must contain a status")


def validate_counterfactual_record(record: Mapping[str, Any]) -> None:
    require_fields(
        record,
        (
            "schema_version",
            "counterfactual_group_id",
            "snapshot_id",
            "pre_state_fingerprint",
            "branch_type",
            "factual_action",
            "branch_action",
            "pre_state_equivalent",
            "execution_status",
            "post_state_fingerprint",
        ),
        "counterfactual record",
    )
    if record["schema_version"] != COUNTERFACTUAL_SCHEMA_VERSION:
        raise ValueError("Unexpected counterfactual schema version")
    if record["branch_type"] not in {"action", "executor", "fault"}:
        raise ValueError("Unknown counterfactual branch type")


def schema_definitions() -> dict[str, dict[str, Any]]:
    return {
        "tool_event.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["tool_call", "tool_result"],
            "properties": {
                "tool_call": {"type": "object"},
                "tool_result": {"type": "object"},
            },
        },
        "artifact_diff.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["added", "modified", "removed"],
            "properties": {
                "added": {"type": "array"},
                "modified": {"type": "array"},
                "removed": {"type": "array"},
            },
        },
        "environment_snapshot.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["snapshot_id", "tree_hash", "public_state_fingerprint", "budget"],
        },
        "counterfactual_record.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": [
                "schema_version",
                "counterfactual_group_id",
                "snapshot_id",
                "pre_state_fingerprint",
                "branch_type",
                "factual_action",
                "branch_action",
                "pre_state_equivalent",
                "execution_status",
                "post_state_fingerprint",
            ],
            "properties": {
                "schema_version": {"const": COUNTERFACTUAL_SCHEMA_VERSION},
                "branch_type": {"enum": ["action", "executor", "fault"]},
                "pre_state_equivalent": {"type": "boolean"},
            },
        },
    }
