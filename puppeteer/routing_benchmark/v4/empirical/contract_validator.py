from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .schema import OUTCOME_STATUSES


ROUTING_HINT_PATTERNS = (
    re.compile(r"\b(next|then)\s+(call|route|select|use)\b", re.I),
    re.compile(r"\broute\s+(this|the task)\s+to\b", re.I),
    re.compile(r"\bnext\s+agent\b", re.I),
)


def task_input_is_concrete(task_input: Mapping[str, Any]) -> tuple[bool, str]:
    manifest = task_input.get("artifact_manifest")
    if isinstance(manifest, Mapping) and manifest.get("manifest_path"):
        return True, "artifact_manifest"
    source = task_input.get("source_task_input")
    if isinstance(source, Mapping) and source:
        meaningful = [
            value
            for value in source.values()
            if value not in (None, "", [], {})
        ]
        if meaningful:
            return True, "source_task_input"
    payload = task_input.get("concrete_payload")
    if payload not in (None, "", [], {}):
        return True, "concrete_payload"
    attachments = task_input.get("attachments")
    if attachments:
        return True, "attachments"
    return False, "only abstract contract fields are available"


def validate_interpreted_outcome(
    *,
    raw_output: str,
    outcome: Mapping[str, Any],
    pre_state: Mapping[str, Any],
    role_profile: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    status = str(outcome.get("execution_status", ""))
    if status not in OUTCOME_STATUSES:
        errors.append(f"unknown_execution_status:{status}")
    if not raw_output.strip() and status not in {"transport_failure", "tool_failure"}:
        errors.append("empty_output_requires_failure_status")
    if status == "success" and not (
        outcome.get("produced_artifacts")
        or outcome.get("observable_facts_added")
        or outcome.get("resolved_requirements")
    ):
        errors.append("success_without_observable_delta")

    current_requirements = set(map(str, pre_state.get("open_requirements") or ()))
    unknown_resolved = set(map(str, outcome.get("resolved_requirements") or ())) - current_requirements
    if unknown_resolved:
        warnings.append("resolved_requirement_not_in_pre_state")

    expected_text = " ".join(map(str, role_profile.get("expected_outputs") or ())).casefold()
    for artifact in outcome.get("produced_artifacts") or ():
        if not isinstance(artifact, Mapping):
            errors.append("artifact_is_not_object")
            continue
        artifact_type = str(artifact.get("artifact_type", "")).strip()
        summary = str(artifact.get("content_summary", "")).strip()
        if not artifact_type or not summary:
            errors.append("artifact_missing_type_or_summary")
        elif expected_text and not any(
            token in expected_text
            for token in re.findall(r"[a-z0-9]+", artifact_type.casefold())
            if len(token) > 3
        ):
            warnings.append("artifact_type_not_obviously_declared_by_role")

    if any(pattern.search(raw_output) for pattern in ROUTING_HINT_PATTERNS):
        warnings.append("raw_output_contains_possible_routing_hint")
    if bool(outcome.get("completion_claimed")) and outcome.get("new_open_requirements"):
        warnings.append("completion_claim_conflicts_with_open_requirements")
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "requires_review": bool(errors or warnings),
    }


def validate_action_labels(
    annotations: Sequence[Mapping[str, Any]],
    candidate_actions: Sequence[str],
) -> dict[str, Any]:
    errors: list[str] = []
    candidates = set(map(str, candidate_actions))
    seen: set[str] = set()
    for item in annotations:
        action = str(item.get("action", ""))
        if action not in candidates:
            errors.append(f"action_outside_candidate_set:{action}")
        if action in seen:
            errors.append(f"duplicate_action:{action}")
        seen.add(action)
        if item.get("label") not in {"valid", "invalid", "uncertain"}:
            errors.append(f"unknown_label:{action}")
        if not str(item.get("reason_code", "")).strip():
            errors.append(f"missing_reason_code:{action}")
    if seen != candidates:
        errors.append("candidate_coverage_mismatch")
    return {"valid": not errors, "errors": sorted(set(errors))}
