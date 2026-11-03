from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


V3_SCHEMA_VERSION = "3.0-alpha"
STOP_ACTION = "STOP"


def _strings(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in (value or ()))


@dataclass(frozen=True)
class ProvenanceRecordV3:
    provenance_id: str
    scenario_id: str
    source_type: str
    source_dataset: str
    source_version: str
    source_split: str
    source_item_id: str
    source_path: str
    source_sha256: str
    license: str
    derivation_script: str
    derivation_version: str
    transformation_steps: tuple[str, ...]
    validation_method: str
    human_review_status: str = "pending"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ProvenanceRecordV3":
        return cls(
            provenance_id=str(raw["provenance_id"]),
            scenario_id=str(raw["scenario_id"]),
            source_type=str(raw["source_type"]),
            source_dataset=str(raw["source_dataset"]),
            source_version=str(raw["source_version"]),
            source_split=str(raw["source_split"]),
            source_item_id=str(raw["source_item_id"]),
            source_path=str(raw["source_path"]),
            source_sha256=str(raw["source_sha256"]),
            license=str(raw["license"]),
            derivation_script=str(raw["derivation_script"]),
            derivation_version=str(raw["derivation_version"]),
            transformation_steps=_strings(raw.get("transformation_steps")),
            validation_method=str(raw["validation_method"]),
            human_review_status=str(raw.get("human_review_status", "pending")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance_id": self.provenance_id,
            "scenario_id": self.scenario_id,
            "source_type": self.source_type,
            "source_dataset": self.source_dataset,
            "source_version": self.source_version,
            "source_split": self.source_split,
            "source_item_id": self.source_item_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "license": self.license,
            "derivation_script": self.derivation_script,
            "derivation_version": self.derivation_version,
            "transformation_steps": list(self.transformation_steps),
            "validation_method": self.validation_method,
            "human_review_status": self.human_review_status,
        }


def v3_schema_definitions() -> dict[str, dict[str, Any]]:
    rationale = {
        "type": "object",
        "required": ["status", "rationale_source"],
        "properties": {
            "status": {"enum": ["valid", "invalid"]},
            "rationale_source": {"const": "formal_transition_preconditions"},
            "satisfied_preconditions": {
                "type": "array",
                "items": {"type": "string"},
            },
            "violated_preconditions": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "additionalProperties": False,
    }
    decision = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "routingbench-v3-decision.schema.json",
        "title": "RoutingBench V3 set-valued routing decision",
        "type": "object",
        "required": [
            "decision_id",
            "scenario_id",
            "provenance_id",
            "candidate_actions",
            "valid_actions",
            "action_rationales",
            "schema_version",
        ],
        "properties": {
            "decision_id": {"type": "string"},
            "scenario_id": {"type": "string"},
            "provenance_id": {"type": "string"},
            "candidate_actions": {"type": "array", "items": {"type": "string"}},
            "valid_actions": {"type": "array", "items": {"type": "string"}},
            "action_rationales": {
                "type": "object",
                "additionalProperties": rationale,
            },
            "schema_version": {"const": V3_SCHEMA_VERSION},
        },
        "not": {
            "anyOf": [
                {"required": ["preferred_actions"]},
                {"required": ["action_utilities"]},
                {"required": ["capability_scores"]},
            ]
        },
        "additionalProperties": True,
    }
    provenance = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "routingbench-v3-provenance.schema.json",
        "title": "RoutingBench V3 workflow provenance",
        "type": "object",
        "required": [
            "provenance_id",
            "scenario_id",
            "source_type",
            "source_dataset",
            "source_item_id",
            "source_path",
            "source_sha256",
            "license",
            "derivation_script",
            "transformation_steps",
            "validation_method",
            "human_review_status",
        ],
        "properties": {
            "provenance_id": {"type": "string"},
            "scenario_id": {"type": "string"},
            "source_type": {"type": "string"},
            "source_dataset": {"type": "string"},
            "source_version": {"type": "string"},
            "source_split": {"type": "string"},
            "source_item_id": {"type": "string"},
            "source_path": {"type": "string"},
            "source_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "license": {"type": "string"},
            "derivation_script": {"type": "string"},
            "derivation_version": {"type": "string"},
            "transformation_steps": {"type": "array", "items": {"type": "string"}},
            "validation_method": {"type": "string"},
            "human_review_status": {"enum": ["pending", "reviewed", "adjudicated"]},
        },
        "additionalProperties": False,
    }
    return {
        "decision.schema.json": decision,
        "provenance.schema.json": provenance,
    }
