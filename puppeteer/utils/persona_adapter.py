"""Normalize persona pools into the flat schema used by the original runner."""

import json
import re
from pathlib import Path


def _nonempty_string(value, field, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {field} must be a non-empty string")
    return value.strip()


def _string_list(value, field, context):
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ValueError(f"{context}: {field} must be a non-empty string list")
    return [item.strip() for item in value]


def _slug(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _join(values):
    if not isinstance(values, list):
        return ""
    return "; ".join(str(value).strip() for value in values if str(value).strip())


def _design_prompt(persona, context):
    role_card = persona.get("role_card")
    if not isinstance(role_card, dict):
        raise ValueError(f"{context}: role_card must be an object")

    role_name = _nonempty_string(role_card.get("role_name"), "role_card.role_name", context)
    role_goal = _nonempty_string(role_card.get("role_goal"), "role_card.role_goal", context)
    routing = persona.get("routing_profile") or {}

    parts = [f"You are the {role_name}.", role_goal]
    core_functions = _join(role_card.get("core_functions"))
    use_when = _join(routing.get("use_when"))
    avoid_when = _join(routing.get("avoid_when"))
    expected_contribution = _join(routing.get("expected_contribution"))

    if core_functions:
        parts.append(f"Core functions: {core_functions}.")
    if use_when:
        parts.append(f"Use this specialist when: {use_when}.")
    if avoid_when:
        parts.append(f"Avoid using this specialist when: {avoid_when}.")
    if expected_contribution:
        parts.append(f"Expected contribution: {expected_contribution}.")
    return " ".join(parts), role_name, role_card


def normalize_persona(persona, context="persona"):
    """Convert either a legacy or schema-1.0 persona to the runtime shape."""
    if not isinstance(persona, dict):
        raise ValueError(f"{context}: persona must be an object")

    if persona.get("available") is False:
        return None

    legacy_fields = {"name", "role_prompt", "model_type", "actions", "agent_type"}
    if legacy_fields.issubset(persona):
        normalized = {
            "name": _nonempty_string(persona.get("name"), "name", context),
            "role_prompt": _nonempty_string(
                persona.get("role_prompt"), "role_prompt", context
            ),
            "model_type": _nonempty_string(
                persona.get("model_type"), "model_type", context
            ),
            "actions": _string_list(persona.get("actions"), "actions", context),
            "agent_type": _nonempty_string(
                persona.get("agent_type"), "agent_type", context
            ),
            "policy": persona.get("policy") or "autonomous",
        }
    elif {"backbone", "role_card"}.issubset(persona):
        schema_version = str(persona.get("schema_version", ""))
        if schema_version != "1.0":
            raise ValueError(
                f"{context}: unsupported persona schema_version {schema_version!r}"
            )
        backbone = _nonempty_string(persona.get("backbone"), "backbone", context)
        role_prompt, role_name, role_card = _design_prompt(persona, context)
        normalized = {
            "name": f"{_slug(role_name)}_{_slug(backbone)}",
            "role_prompt": role_prompt,
            "model_type": backbone,
            "actions": _string_list(
                role_card.get("allowed_actions"),
                "role_card.allowed_actions",
                context,
            ),
            "agent_type": "reasoning",
            "policy": "autonomous",
        }
    else:
        raise ValueError(
            f"{context}: expected either runtime persona fields or "
            "schema-1.0 backbone/role_card fields"
        )

    if normalized["agent_type"] != "reasoning":
        raise ValueError(
            f"{context}: unsupported agent_type {normalized['agent_type']!r}"
        )
    return normalized


def load_runtime_personas(personas_path, add_terminator=True):
    """Load JSONL personas and return a validated runtime persona list."""
    path = Path(personas_path)
    personas = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            context = f"{path}:{line_number}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{context}: invalid JSON: {error.msg}") from error
            normalized = normalize_persona(raw, context)
            if normalized is not None:
                personas.append(normalized)

    if not personas:
        raise ValueError(f"{path}: no available personas")

    names = [persona["name"] for persona in personas]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"{path}: duplicate persona names: {duplicate_names}")

    terminators = [
        persona for persona in personas if "terminate" in persona["actions"]
    ]
    if len(terminators) > 1:
        raise ValueError(f"{path}: expected at most one termination persona")
    if not terminators and add_terminator:
        personas.append(
            {
                "name": "AutomaticStopController",
                "role_prompt": (
                    "You control reasoning termination. End a path when its evidence "
                    "is sufficient or another step is unlikely to improve the answer."
                ),
                # Termination does not call the provider, but Agent still requires a
                # registered model alias during construction.
                "model_type": personas[0]["model_type"],
                "actions": ["terminate"],
                "agent_type": "reasoning",
                "policy": "autonomous",
            }
        )
    return personas
