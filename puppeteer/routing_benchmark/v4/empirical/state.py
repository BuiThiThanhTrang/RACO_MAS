from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

from routing_benchmark.v4.execution.state_history import normalize_current_state

from .schema import canonical_hash


FAILURE_STATUSES = {"no_progress", "invalid_output", "tool_failure", "transport_failure", "failed", "timeout"}


def _legacy_to_current(state: Mapping[str, Any]) -> dict[str, Any]:
    facts = [
        {"key": f"fact-{canonical_hash(str(value), 12)}", "value": str(value), "evidence_refs": []}
        for value in state.get("facts") or ()
    ]
    artifacts = []
    for index, item in enumerate(state.get("artifact_inventory") or state.get("artifacts") or ()):
        value = dict(item) if isinstance(item, Mapping) else {"description": str(item)}
        artifacts.append(
            {
                "artifact_id": str(value.get("artifact_id") or f"artifact-{index:03d}"),
                "type": str(value.get("artifact_type") or value.get("type") or "workflow_artifact"),
                "status": str(value.get("status") or "available"),
                "summary": str(value.get("content_summary") or value.get("description") or "available artifact"),
                "content_ref": str(value.get("content_ref") or value.get("path") or ""),
                "evidence_refs": list(value.get("evidence_refs") or ()),
            }
        )
    requirements = [
        {"requirement_id": f"requirement-{canonical_hash(str(value), 12)}", "description": str(value), "evidence_needed": "Observable completion evidence."}
        for value in state.get("open_requirements") or ()
    ]
    issues = [
        {"issue_id": f"issue-{canonical_hash(str(value), 12)}", "type": "observable_execution_issue", "summary": str(value), "evidence_refs": []}
        for value in state.get("issues") or state.get("observable_issues") or ()
    ]
    goal = str(state.get("goal_status") or "")
    return {
        "task_status": "complete" if goal in {"complete", "claimed_complete"} else "active",
        "facts": facts,
        "artifacts": artifacts,
        "open_requirements": requirements,
        "issues": issues,
    }


def normalize_public_state(state: Mapping[str, Any]) -> dict[str, Any]:
    if all(key in state for key in ("task_status", "facts", "artifacts", "open_requirements", "issues")):
        return normalize_current_state(state)
    return _legacy_to_current(deepcopy(dict(state)))


def state_fingerprint(state: Mapping[str, Any]) -> str:
    return canonical_hash(normalize_public_state(state), length=20)


def apply_outcome(pre_state: Mapping[str, Any], outcome: Mapping[str, Any], *, action: str) -> dict[str, Any]:
    """Deterministically fold a reviewed outcome into the public state schema."""
    state = normalize_public_state(pre_state)
    facts = list(state["facts"])
    existing_fact_values = {str(item.get("value")) for item in facts}
    for value in map(str, outcome.get("observable_facts_added") or ()):
        if value and value not in existing_fact_values:
            facts.append({"key": f"fact-{canonical_hash(value, 12)}", "value": value, "evidence_refs": []})
            existing_fact_values.add(value)
    state["facts"] = facts

    artifacts = list(state["artifacts"])
    existing_ids = {str(item.get("artifact_id")) for item in artifacts}
    for artifact in outcome.get("produced_artifacts") or ():
        if not isinstance(artifact, Mapping):
            continue
        summary = str(artifact.get("content_summary") or artifact.get("summary") or "produced artifact")
        artifact_id = str(artifact.get("artifact_id") or f"artifact-{canonical_hash([action, summary], 12)}")
        if artifact_id not in existing_ids:
            artifacts.append(
                {
                    "artifact_id": artifact_id,
                    "type": str(artifact.get("artifact_type") or "workflow_artifact"),
                    "status": str(artifact.get("status") or "available"),
                    "summary": summary,
                    "content_ref": str(artifact.get("content_ref") or ""),
                    "evidence_refs": list(map(str, artifact.get("evidence_refs") or ())),
                }
            )
            existing_ids.add(artifact_id)
    state["artifacts"] = artifacts

    requirements = list(state["open_requirements"])
    resolved = set(map(str, outcome.get("resolved_requirements") or ()))
    requirements = [item for item in requirements if str(item.get("requirement_id")) not in resolved and str(item.get("description")) not in resolved]
    existing_descriptions = {str(item.get("description")) for item in requirements}
    for description in map(str, outcome.get("new_open_requirements") or ()):
        if description and description not in existing_descriptions:
            requirements.append({"requirement_id": f"requirement-{canonical_hash(description, 12)}", "description": description, "evidence_needed": "Observable completion evidence."})
    state["open_requirements"] = requirements

    state["issues"] = [
        {"issue_id": f"issue-{canonical_hash(str(value), 12)}", "type": "observable_execution_issue", "summary": str(value), "evidence_refs": []}
        for value in outcome.get("observable_issues") or ()
    ]
    if bool(outcome.get("completion_claimed")) and not requirements:
        state["task_status"] = "ready_to_finish"
    return state


def build_empirical_memory_views(post_state: Mapping[str, Any], events: Iterable[Mapping[str, Any]], *, recent_limit: int = 5) -> dict[str, dict[str, Any]]:
    state = normalize_public_state(post_state)
    event_list = [
        {
            "step": item.get("step"),
            "agent_id": item.get("agent_id"),
            "intent": item.get("intent", ""),
            "execution_status": item.get("execution_status", item.get("status")),
            "state_delta": deepcopy(item.get("state_delta") or {}),
            "output_ref": item.get("output_ref") or item.get("output_id"),
        }
        for item in events
    ]
    recent = event_list[-recent_limit:]
    raw = [
        {"output_id": item.get("output_id"), "agent_id": item.get("agent_id"), "raw_output": item.get("raw_output", "")}
        for item in events
    ]
    return {
        "state": {"observed_state": state},
        "structured": {"observed_state": state, "structured_history": recent},
        "raw_empirical": {"observed_state": state, "structured_history": recent, "raw_executor_outputs": raw},
    }
