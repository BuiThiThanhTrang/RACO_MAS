from __future__ import annotations

from typing import Any

from routing_benchmark.v3.schemas import v3_schema_definitions


V3_BETA_SCHEMA_VERSION = "3.0-beta"


def v3_beta_schema_definitions() -> dict[str, dict[str, Any]]:
    definitions = v3_schema_definitions()
    decision = dict(definitions["decision.schema.json"])
    decision["title"] = "RoutingBench V3-beta set-valued routing decision"
    decision["properties"] = dict(decision["properties"])
    decision["properties"]["schema_version"] = {
        "const": V3_BETA_SCHEMA_VERSION
    }
    task_input = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "routingbench-v3-beta-task-input.schema.json",
        "title": "Private-to-actor task input; never exposed to the router",
        "type": "object",
        "required": ["scenario_id", "task_input"],
        "properties": {
            "scenario_id": {"type": "string"},
            "task_input": {"type": "object"},
        },
        "additionalProperties": False,
    }
    evaluation = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "routingbench-v3-beta-evaluation.schema.json",
        "title": "Gold evaluation sidecar; never exposed to router or actor",
        "type": "object",
        "required": ["scenario_id", "metric", "answer", "source_item_id"],
        "properties": {
            "scenario_id": {"type": "string"},
            "metric": {
                "enum": ["normalized_exact_match", "exact_choice"]
            },
            "answer": {"type": "string"},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "supporting_ids": {
                "type": "array",
                "items": {"type": ["string", "integer"]},
            },
            "source_item_id": {"type": "string"},
        },
        "additionalProperties": False,
    }
    return {
        "decision.schema.json": decision,
        "provenance.schema.json": definitions["provenance.schema.json"],
        "task_input.schema.json": task_input,
        "evaluation.schema.json": evaluation,
    }
