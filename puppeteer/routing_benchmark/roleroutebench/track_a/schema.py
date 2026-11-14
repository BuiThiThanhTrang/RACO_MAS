from __future__ import annotations

from typing import Any


LABEL_CONTRACT_VERSION = "roleroutebench.track_a.labels.v1"
PROMPT_VERSION = "roleroutebench.track_a.terra.v1"

ACTION_CLASSES = (
    "valid_progress",
    "valid_recovery",
    "valid_stop",
    "redundant_no_progress",
    "invalid_precondition",
    "unsafe",
    "uncertain",
)

STATE_PHASES = (
    "initial",
    "evidence_collection",
    "candidate_authoring",
    "execution",
    "failure_diagnosis",
    "repair_or_recovery",
    "validation",
    "terminal",
)


def prelabel_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "classification": {"type": "string", "enum": list(ACTION_CLASSES)},
                        "rationale": {"type": "string"},
                    },
                    "required": ["action", "classification", "rationale"],
                    "additionalProperties": False,
                },
            },
            "terminal_valid": {"type": "boolean"},
            "state_phase": {"type": "string", "enum": list(STATE_PHASES)},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "summary": {"type": "string"},
        },
        "required": ["actions", "terminal_valid", "state_phase", "confidence", "summary"],
        "additionalProperties": False,
    }


def validate_prelabel(payload: dict[str, Any], candidate_actions: list[str]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["prelabel must be an object"]
    actions = payload.get("actions")
    if not isinstance(actions, list):
        return ["actions must be an array"]
    seen: list[str] = []
    by_action: dict[str, str] = {}
    for index, item in enumerate(actions):
        if not isinstance(item, dict):
            errors.append(f"actions[{index}] must be an object")
            continue
        action = item.get("action")
        classification = item.get("classification")
        rationale = item.get("rationale")
        if not isinstance(action, str):
            errors.append(f"actions[{index}].action must be a string")
            continue
        seen.append(action)
        if action in by_action:
            errors.append(f"duplicate action: {action}")
        if classification not in ACTION_CLASSES:
            errors.append(f"invalid classification for {action}: {classification}")
        else:
            by_action[action] = classification
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append(f"missing rationale for {action}")
    if seen != candidate_actions:
        errors.append("actions must exactly match candidate_actions in source order")
    if payload.get("state_phase") not in STATE_PHASES:
        errors.append("invalid state_phase")
    if payload.get("confidence") not in {"low", "medium", "high"}:
        errors.append("invalid confidence")
    if not isinstance(payload.get("terminal_valid"), bool):
        errors.append("terminal_valid must be boolean")
    stop_class = by_action.get("STOP")
    if stop_class == "valid_stop" and payload.get("terminal_valid") is not True:
        errors.append("STOP=valid_stop requires terminal_valid=true")
    if payload.get("terminal_valid") is True and stop_class != "valid_stop":
        errors.append("terminal_valid=true requires STOP=valid_stop")
    return errors


def derived_action_sets(payload: dict[str, Any]) -> dict[str, list[str]]:
    grouped = {classification: [] for classification in ACTION_CLASSES}
    for item in payload["actions"]:
        grouped[item["classification"]].append(item["action"])
    return {
        "valid_actions": grouped["valid_progress"] + grouped["valid_recovery"] + grouped["valid_stop"],
        "unsafe_actions": grouped["unsafe"],
        "redundant_actions": grouped["redundant_no_progress"],
        "uncertain_actions": grouped["uncertain"],
    }

