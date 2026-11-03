from __future__ import annotations

from typing import Any

from routing_benchmark.v3.schemas import v3_schema_definitions


V4_SCHEMA_VERSION = "4.0-alpha.2"
MEMORY_VIEWS = ("state", "structured", "raw")


def v4_schema_definitions() -> dict[str, dict[str, Any]]:
    definitions = v3_schema_definitions()
    decision = dict(definitions["decision.schema.json"])
    decision["title"] = "RoutingBench V4 history-aware routing decision"
    decision["properties"] = dict(decision["properties"])
    decision["properties"]["schema_version"] = {"const": V4_SCHEMA_VERSION}
    decision["properties"]["memory_views"] = {
        "type": "object",
        "required": ["state", "structured"],
        "properties": {
            "state": {"type": "object"},
            "structured": {"type": "object"},
            "raw": {"type": "object"},
        },
        "additionalProperties": False,
    }
    decision["required"] = list(decision["required"]) + ["memory_views"]
    executor_output = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "routingbench-v4-executor-output.schema.json",
        "title": "Deterministic executor output fixture",
        "type": "object",
        "required": [
            "output_id",
            "scenario_id",
            "transition_id",
            "action",
            "status",
            "structured_event",
            "raw_output",
        ],
        "properties": {
            "output_id": {"type": "string"},
            "scenario_id": {"type": "string"},
            "transition_id": {"type": "string"},
            "action": {"type": "string"},
            "status": {"type": "string"},
            "structured_event": {"type": "object"},
            "raw_output": {"type": "string"},
            "contains_fault_signal": {"type": "boolean"},
        },
        "additionalProperties": False,
    }
    recovery = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "routingbench-v4-recovery.schema.json",
        "title": "Private recovery evaluation sidecar",
        "type": "object",
        "required": [
            "scenario_id",
            "pair_id",
            "variant",
            "clean_scenario_id",
            "faults",
            "unsafe_transition_ids",
            "decision_labels",
        ],
        "properties": {
            "scenario_id": {"type": "string"},
            "pair_id": {"type": "string"},
            "variant": {"type": "string"},
            "clean_scenario_id": {"type": "string"},
            "faults": {"type": "array", "items": {"type": "object"}},
            "unsafe_transition_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "inefficient_transition_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "detector_actions": {"type": "array", "items": {"type": "string"}},
            "repair_actions": {"type": "array", "items": {"type": "string"}},
            "recovery_actions": {"type": "array", "items": {"type": "string"}},
            "decision_labels": {"type": "array", "items": {"type": "object"}},
        },
        "additionalProperties": False,
    }
    public_input = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "routingbench-v4-public-input.schema.json",
        "title": "Leakage-safe router input views",
        "type": "object",
        "required": [
            "decision_id",
            "split",
            "workflow_family",
            "observations",
        ],
        "properties": {
            "decision_id": {"type": "string"},
            "split": {"type": "string"},
            "workflow_family": {"type": "string"},
            "observations": {
                "type": "object",
                "required": ["state", "structured"],
                "properties": {
                    "state": {"type": "object"},
                    "structured": {"type": "object"},
                    "raw": {"type": "object"},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }
    private_label = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "routingbench-v4-private-label.schema.json",
        "title": "Private routing and recovery labels",
        "type": "object",
        "required": [
            "decision_id",
            "internal_scenario_id",
            "public_decision_id",
            "valid_actions",
            "terminal_valid",
            "phase",
            "fault_variant",
        ],
        "properties": {
            "decision_id": {"type": "string"},
            "internal_scenario_id": {"type": "string"},
            "public_decision_id": {"type": "string"},
            "valid_actions": {"type": "array", "items": {"type": "string"}},
            "terminal_valid": {"type": "boolean"},
            "phase": {"type": "string"},
            "unsafe_actions": {"type": "array", "items": {"type": "string"}},
            "fault_variant": {"type": "string"},
            "action_rationales": {"type": "object"},
        },
        "additionalProperties": False,
    }
    post_fault_input = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "routingbench-v4-post-fault-input.schema.json",
        "title": "Track C leakage-safe checkpoint views",
        "type": "object",
        "required": ["checkpoint_id", "split", "workflow_family", "observations"],
        "properties": {
            "checkpoint_id": {"type": "string", "pattern": "^pfc-"},
            "split": {"type": "string"},
            "workflow_family": {"type": "string"},
            "observations": public_input["properties"]["observations"],
        },
        "additionalProperties": False,
    }
    post_fault_runtime = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "routingbench-v4-post-fault-runtime.schema.json",
        "title": "Private Track C simulator checkpoint",
        "type": "object",
        "required": [
            "checkpoint_id",
            "split",
            "workflow_family",
            "internal_scenario_id",
            "state",
            "history",
        ],
        "properties": {
            "checkpoint_id": {"type": "string"},
            "split": {"type": "string"},
            "workflow_family": {"type": "string"},
            "internal_scenario_id": {"type": "string"},
            "state": {"type": "object"},
            "history": {"type": "array", "items": {"type": "object"}},
        },
        "additionalProperties": False,
    }
    post_fault_label = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "routingbench-v4-post-fault-label.schema.json",
        "title": "Private Track C recovery label",
        "type": "object",
        "required": [
            "checkpoint_id",
            "pair_id",
            "condition",
            "internal_scenario_id",
            "fault_class",
            "valid_actions",
            "unsafe_actions",
            "oracle_min_remaining_calls",
        ],
        "properties": {
            "checkpoint_id": {"type": "string"},
            "pair_id": {"type": "string"},
            "condition": {"enum": ["fault", "control"]},
            "internal_scenario_id": {"type": "string"},
            "fault_class": {"type": "string"},
            "injection_position": {"type": ["string", "null"]},
            "requires_detection_agent": {"type": "boolean"},
            "unsafe_transition_ids": {"type": "array", "items": {"type": "string"}},
            "inefficient_transition_ids": {"type": "array", "items": {"type": "string"}},
            "detector_actions": {"type": "array", "items": {"type": "string"}},
            "repair_actions": {"type": "array", "items": {"type": "string"}},
            "recovery_actions": {"type": "array", "items": {"type": "string"}},
            "valid_actions": {"type": "array", "items": {"type": "string"}},
            "unsafe_actions": {"type": "array", "items": {"type": "string"}},
            "detector_actions_available": {"type": "array", "items": {"type": "string"}},
            "oracle_min_remaining_calls": {"type": "integer", "minimum": 0},
            "checkpoint_state_fingerprint": {"type": "string"},
        },
        "additionalProperties": False,
    }
    post_fault_pair = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "routingbench-v4-post-fault-pair.schema.json",
        "title": "Private Track C fault-control pair manifest",
        "type": "object",
        "required": [
            "pair_id",
            "split",
            "workflow_family",
            "fault_class",
            "fault_checkpoint_id",
            "control_checkpoint_id",
        ],
        "properties": {
            "pair_id": {"type": "string"},
            "split": {"type": "string"},
            "workflow_family": {"type": "string"},
            "fault_class": {"type": "string"},
            "injection_position": {"type": ["string", "null"]},
            "fault_checkpoint_id": {"type": "string"},
            "control_checkpoint_id": {"type": "string"},
        },
        "additionalProperties": False,
    }
    return {
        "decision.schema.json": decision,
        "provenance.schema.json": definitions["provenance.schema.json"],
        "executor_output.schema.json": executor_output,
        "recovery.schema.json": recovery,
        "public_input.schema.json": public_input,
        "private_label.schema.json": private_label,
        "post_fault_input.schema.json": post_fault_input,
        "post_fault_runtime.schema.json": post_fault_runtime,
        "post_fault_label.schema.json": post_fault_label,
        "post_fault_pair.schema.json": post_fault_pair,
    }
