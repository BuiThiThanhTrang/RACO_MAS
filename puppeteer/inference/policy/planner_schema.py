from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


PLANNER_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["stop", "task_signature", "state_gaps", "selections"],
    "properties": {
        "stop": {"type": "boolean"},
        "task_signature": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "state_gaps": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "selections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidate_id", "subtask", "expected_contribution"],
                "properties": {
                    "candidate_id": {"type": "integer", "minimum": 0},
                    "subtask": {"type": "string"},
                    "expected_contribution": {"type": "string"},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class PlannerAssignment:
    candidate_id: int
    subtask: str
    expected_contribution: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlannerAssignment":
        candidate_id = int(data["candidate_id"])
        subtask = str(data.get("subtask", "")).strip()
        contribution = str(data.get("expected_contribution", "")).strip()
        if candidate_id < 0:
            raise ValueError("candidate_id must be nonnegative")
        if not subtask or not contribution:
            raise ValueError("Planner assignments require subtask and expected_contribution")
        return cls(candidate_id, subtask, contribution)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "subtask": self.subtask,
            "expected_contribution": self.expected_contribution,
        }


@dataclass(frozen=True)
class PlannerDecision:
    stop: bool
    task_signature: tuple[str, ...]
    state_gaps: tuple[str, ...]
    selections: tuple[PlannerAssignment, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlannerDecision":
        stop = data.get("stop")
        if not isinstance(stop, bool):
            raise ValueError("Planner decision stop must be boolean")
        signature = tuple(str(item).strip() for item in data.get("task_signature", ()))
        gaps = tuple(str(item).strip() for item in data.get("state_gaps", ()))
        selections = tuple(
            PlannerAssignment.from_dict(item) for item in data.get("selections", ())
        )
        if stop and selections:
            raise ValueError("A STOP decision cannot include selections")
        if not stop and not selections:
            raise ValueError("A non-STOP decision requires at least one selection")
        return cls(stop, signature, gaps, selections)

