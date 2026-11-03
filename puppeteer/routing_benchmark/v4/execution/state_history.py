from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .actor import READ_ONLY_TOOLS


STATE_KEYS = (
    "task_status",
    "facts",
    "artifacts",
    "open_requirements",
    "issues",
)


def normalize_current_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete router-visible state contract and nothing private."""
    facts = [
        dict(item) if isinstance(item, Mapping) else {"key": f"legacy-fact-{index:03d}", "value": str(item), "evidence_refs": []}
        for index, item in enumerate(state.get("facts") or ())
    ]
    artifact_source = state.get("artifacts") or state.get("artifact_inventory") or ()
    artifacts = [
        dict(item) if isinstance(item, Mapping) else {"artifact_id": f"legacy-artifact-{index:03d}", "type": "workflow_artifact", "status": "available", "summary": str(item), "content_ref": "", "evidence_refs": []}
        for index, item in enumerate(artifact_source)
    ]
    requirements = [
        dict(item) if isinstance(item, Mapping) else {"requirement_id": f"legacy-requirement-{index:03d}", "description": str(item), "evidence_needed": "Observable completion evidence."}
        for index, item in enumerate(state.get("open_requirements") or ())
    ]
    issues = [
        dict(item) if isinstance(item, Mapping) else {"issue_id": f"legacy-issue-{index:03d}", "type": "observable_execution_issue", "summary": str(item), "evidence_refs": []}
        for index, item in enumerate(state.get("issues") or state.get("observable_issues") or ())
    ]
    return {
        "task_status": str(state.get("task_status") or ("complete" if state.get("goal_status") in {"complete", "claimed_complete"} else "active")),
        "facts": facts,
        "artifacts": artifacts,
        "open_requirements": requirements,
        "issues": issues,
    }


def _indexed(items: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(item[key]): dict(item) for item in items if item.get(key) not in (None, "")}


def state_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Compute a deterministic structured delta between two public states."""

    pre = normalize_current_state(before)
    post = normalize_current_state(after)
    pre_facts, post_facts = _indexed(pre["facts"], "key"), _indexed(post["facts"], "key")
    pre_artifacts, post_artifacts = _indexed(pre["artifacts"], "artifact_id"), _indexed(post["artifacts"], "artifact_id")
    pre_requirements = _indexed(pre["open_requirements"], "requirement_id")
    post_requirements = _indexed(post["open_requirements"], "requirement_id")
    pre_issues, post_issues = _indexed(pre["issues"], "issue_id"), _indexed(post["issues"], "issue_id")
    return {
        "facts_added": [post_facts[key] for key in sorted(post_facts.keys() - pre_facts.keys())],
        "facts_updated": [
            {"key": key, "before": pre_facts[key], "after": post_facts[key]}
            for key in sorted(pre_facts.keys() & post_facts.keys())
            if pre_facts[key] != post_facts[key]
        ],
        "artifacts_added": [post_artifacts[key] for key in sorted(post_artifacts.keys() - pre_artifacts.keys())],
        "artifacts_updated": [
            {"artifact_id": key, "before": pre_artifacts[key], "after": post_artifacts[key]}
            for key in sorted(pre_artifacts.keys() & post_artifacts.keys())
            if pre_artifacts[key] != post_artifacts[key]
        ],
        "requirements_satisfied": [
            pre_requirements[key] for key in sorted(pre_requirements.keys() - post_requirements.keys())
        ],
        "issues_added": [post_issues[key] for key in sorted(post_issues.keys() - pre_issues.keys())],
        "issues_resolved": [pre_issues[key] for key in sorted(pre_issues.keys() - post_issues.keys())],
    }


def delta_has_change(delta: Mapping[str, Any]) -> bool:
    return any(bool(delta.get(key)) for key in (
        "facts_added",
        "facts_updated",
        "artifacts_added",
        "artifacts_updated",
        "requirements_satisfied",
        "issues_added",
        "issues_resolved",
    ))


def normalize_execution_status(
    actor_result: Mapping[str, Any],
    delta: Mapping[str, Any] | None = None,
) -> str:
    """Classify an executor call using both tool outcome and public progress.

    A successful repository search or file inspection is not by itself workflow
    progress.  When a public state delta is available, a call that produced no
    durable fact, artifact, satisfied requirement, or issue change is reported as
    ``no_progress`` unless a more specific failure status applies.
    """

    transport = str(actor_result.get("transport_status") or "success")
    if transport in {"transport_failure", "timeout", "pre_execution_rejection", "invalid_protocol"}:
        return {
            "pre_execution_rejection": "rejected",
            "invalid_protocol": "invalid_output",
            "transport_failure": "failed",
        }.get(transport, transport)
    tool_events = list(actor_result.get("tool_events") or ())
    tool_results = [dict(item.get("tool_result") or {}) for item in tool_events]
    state_changing = [
        dict(item.get("tool_result") or {})
        for item in tool_events
        if str((item.get("tool_call") or {}).get("name", ""))
        not in READ_ONLY_TOOLS
    ]
    last_result = state_changing[-1] if state_changing else (
        tool_results[-1] if tool_results else {}
    )
    status = str(last_result.get("status") or "no_progress")
    if status in {"invalid_output", "rejected"}:
        return "invalid_output"
    if status in {"tool_failure", "transport_failure"}:
        return "failed"
    if status == "timeout":
        return "timeout"
    if status == "no_progress":
        return "no_progress"
    if delta is not None:
        return "success" if delta_has_change(delta) else "no_progress"
    if not state_changing:
        return "no_progress"
    return {
        "tool_failure": "failed",
        "pre_execution_rejection": "rejected",
        "no_progress": "failed",
    }.get(status, status)


class RawOutputStore:
    """Persist executor text outside router state and structured history."""

    def __init__(self, root: str | Path, *, namespace: str | None = None) -> None:
        self.root = Path(root)
        if namespace:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", namespace).strip(".-")
            if not safe:
                raise ValueError("Raw output namespace must contain a safe character")
            self.root = self.root / safe
        self.root.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        step: int,
        agent_id: str,
        actor_result: Mapping[str, Any],
        artifact_refs: Sequence[str],
    ) -> dict[str, Any]:
        output_id = f"output-{step:03d}"
        tool_names = [
            str((item.get("tool_call") or {}).get("name"))
            for item in actor_result.get("tool_events") or ()
            if (item.get("tool_call") or {}).get("name")
        ]
        record = {
            "output_id": output_id,
            "agent_id": agent_id,
            "model": (actor_result.get("generation") or {}).get("model"),
            "raw_output": str(actor_result.get("raw_output") or ""),
            "raw_model_responses": list(actor_result.get("raw_model_responses") or ()),
            "work_product": str(actor_result.get("work_product") or ""),
            "tool": tool_names[-1] if tool_names else None,
            "tools": tool_names,
            "tool_events": list(actor_result.get("tool_events") or ()),
            "artifact_refs": list(artifact_refs),
        }
        path = self.root / f"{output_id}.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite executor output: {path}")
        path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return {**record, "output_ref": path.as_posix()}


def derive_private_phase(state: Mapping[str, Any]) -> str:
    """Derive smoke labels from public facts without exposing a workflow stage."""

    current = normalize_current_state(state)
    if current["task_status"] == "complete":
        return "complete"
    facts = {str(item.get("key")): item.get("value") for item in current["facts"]}
    if bool(facts.get("final_artifact_ready")):
        return "complete" if current["task_status"] == "complete" else "downstream"
    if bool(facts.get("downstream_artifact_ready")):
        return "downstream"
    if facts.get("validation_outcome") == "pass":
        return "verified"
    if facts.get("validation_outcome") == "fail":
        return "candidate_invalid"
    if bool(facts.get("candidate_available")):
        return "candidate"
    if facts.get("candidate_test_outcome") == "pass":
        return "verified"
    if facts.get("candidate_test_outcome") == "fail" or facts.get("candidate_patch_outcome") == "rejected":
        return "candidate_invalid"
    # An applied patch is newer and more actionable than a diagnosis artifact
    # retained from the preceding phase. Verification must take precedence over
    # asking the repair role to author the same patch again.
    if facts.get("candidate_patch_outcome") == "applied":
        return "candidate"
    if any(item.get("type") == "diagnosis" for item in current["artifacts"]):
        return "diagnosed"
    if facts.get("baseline_test_outcome") == "fail":
        return "candidate_invalid"
    return "initial"
