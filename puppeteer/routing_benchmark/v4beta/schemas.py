from __future__ import annotations

from copy import deepcopy

from routing_benchmark.v4.schemas import v4_schema_definitions


V4_BETA_SCHEMA_VERSION = "4.0-beta"


def v4_beta_schema_definitions() -> dict[str, dict]:
    """Return the V4 schemas with the beta version and migration fields."""

    result = deepcopy(v4_schema_definitions())
    for schema in result.values():
        properties = schema.get("properties") or {}
        version = properties.get("schema_version")
        if isinstance(version, dict) and "const" in version:
            version["const"] = V4_BETA_SCHEMA_VERSION
    private_label = result.get("private_label.schema.json", {})
    private_label.setdefault("properties", {}).update(
        {
            "task_domain": {"type": "string"},
            "workflow_pattern": {"type": "string"},
        }
    )
    for name in (
        "post_fault_input.schema.json",
        "post_fault_runtime.schema.json",
        "post_fault_label.schema.json",
        "post_fault_pair.schema.json",
    ):
        result.get(name, {}).setdefault("properties", {}).update(
            {
                "task_domain": {"type": ["string", "null"]},
                "workflow_pattern": {"type": ["string", "null"]},
            }
        )
    return result
