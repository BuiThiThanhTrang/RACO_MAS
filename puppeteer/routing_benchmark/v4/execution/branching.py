from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from routing_benchmark.v4.artifacts.schema import canonical_hash


BRANCH_SCHEMA_VERSION = "4.3-human-intervention-branch-v1"


def validate_intervention(
    intervention: Mapping[str, Any],
    *,
    candidate_actions: Sequence[str],
) -> None:
    required = {
        "source_run_id",
        "parent_episode_id",
        "fork_step",
        "forced_action",
        "intervention_source",
        "rationale",
    }
    missing = sorted(required - set(intervention))
    if missing:
        raise ValueError(f"Intervention is missing required fields: {missing}")
    if str(intervention["intervention_source"]) != "human":
        raise ValueError("Only explicitly human-authored interventions are supported")
    if int(intervention["fork_step"]) < 0:
        raise ValueError("fork_step must be non-negative")
    action = str(intervention["forced_action"])
    if action not in set(map(str, candidate_actions)):
        raise ValueError(f"Forced action is outside the candidate pool: {action}")
    if not str(intervention["rationale"]).strip():
        raise ValueError("A human rationale is required")


def _load_output_reference(reference: str | Path) -> dict[str, Any] | None:
    path = Path(reference)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else None


def _recover_prefix_outputs(
    episode: Mapping[str, Any], prefix_steps: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    references = list(episode.get("raw_output_records") or ())
    for index, step in enumerate(prefix_steps):
        loaded = (
            _load_output_reference(str(references[index]))
            if index < len(references)
            else None
        )
        if loaded is not None:
            loaded["output_ref"] = str(references[index])
            recovered.append(loaded)
            continue
        result = dict(step.get("executor_result") or {})
        if not result:
            continue
        action = str(step.get("selected_action") or "")
        recovered.append(
            {
                "output_id": f"inherited-output-{index + 1:03d}",
                "agent_id": action,
                "model": (result.get("generation") or {}).get("model"),
                "raw_output": str(result.get("raw_output") or ""),
                "raw_model_responses": list(result.get("raw_model_responses") or ()),
                "work_product": str(result.get("work_product") or ""),
                "tool_events": list(result.get("tool_events") or ()),
                "artifact_refs": [],
                "output_ref": None,
                "recovered_from_episode": True,
            }
        )
    return recovered


def build_branch_checkpoint(
    episode: Mapping[str, Any],
    *,
    fork_step: int,
) -> dict[str, Any]:
    steps = list(episode.get("steps") or ())
    if fork_step < 0 or fork_step >= len(steps):
        raise IndexError(
            f"fork_step {fork_step} is outside the episode step range 0..{len(steps) - 1}"
        )
    target = dict(steps[fork_step])
    observation = dict(target.get("pre_action_observation") or {})
    if not observation:
        raise ValueError("Fork step does not contain pre_action_observation")
    snapshot_dir = Path(str(target.get("snapshot_dir") or ""))
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"Fork snapshot is unavailable: {snapshot_dir}")
    prefix_steps = [dict(item) for item in steps[:fork_step]]
    complete_history = list(episode.get("structured_history") or ())
    prefix_history = [dict(item) for item in complete_history[:fork_step]]
    if len(prefix_history) != len(prefix_steps):
        prefix_history = [
            dict(item.get("history_event") or {})
            for item in prefix_steps
            if item.get("history_event")
        ]
    calls_remaining = int(
        ((observation.get("budget") or {}).get("calls_remaining", 0))
    )
    checkpoint = {
        "schema_version": BRANCH_SCHEMA_VERSION,
        "parent_schema_version": str(episode.get("schema_version") or ""),
        "snapshot_id": str(target["snapshot_id"]),
        "snapshot_dir": str(snapshot_dir),
        "pre_state_fingerprint": str(target["pre_state_fingerprint"]),
        "pre_action_observation": observation,
        "prefix_steps": prefix_steps,
        "prefix_history": prefix_history,
        "prefix_raw_outputs": _recover_prefix_outputs(episode, prefix_steps),
        "calls_remaining": calls_remaining,
        "manifest_path": str(episode["manifest_path"]),
        "parent_episode_id": str(
            episode.get("base_workflow_id") or episode.get("scenario_id") or ""
        ),
        "parent_scenario_id": episode.get("scenario_id"),
        "fork_step": fork_step,
        "original_action": target.get("selected_action"),
        "candidate_actions": list(target.get("candidate_actions") or ()),
        "memory_view": str(episode.get("memory_view") or "structured"),
        "private_execution_variant": str(
            episode.get("private_execution_variant") or "clean"
        ),
    }
    checkpoint["checkpoint_hash"] = canonical_hash(checkpoint, 64)
    return checkpoint


def branch_metrics(
    *,
    parent_episode: Mapping[str, Any],
    branch_episode: Mapping[str, Any],
    fork_step: int,
) -> dict[str, Any]:
    suffix = list(branch_episode.get("steps") or ())[fork_step:]
    autonomous = [
        step
        for step in suffix
        if str(step.get("decision_source")) == "router"
    ]
    first_progress = next(
        (
            index
            for index, step in enumerate(suffix, start=1)
            if any(
                bool((step.get("history_event") or {}).get("state_delta", {}).get(key))
                for key in (
                    "facts_added",
                    "facts_updated",
                    "artifacts_added",
                    "artifacts_updated",
                    "requirements_satisfied",
                    "issues_added",
                    "issues_resolved",
                )
            )
            or str(step.get("execution_status")) == "valid_stop"
        ),
        None,
    )
    branch_execution_calls = sum(
        str(item.get("selected_action")) != "STOP"
        and str(item.get("execution_status")) != "call_limit"
        for item in suffix
    )
    factual_suffix = list(parent_episode.get("steps") or ())[fork_step:]
    factual_execution_calls = sum(
        str(item.get("selected_action")) != "STOP"
        and str(item.get("execution_status")) != "call_limit"
        for item in factual_suffix
    )
    return {
        "intervention_count": 1,
        "parent_completed": bool(
            (parent_episode.get("terminal_status") or {}).get("terminal")
        ),
        "post_intervention_completion": bool(
            (branch_episode.get("terminal_status") or {}).get("terminal")
        ),
        "steps_to_first_substantive_progress": first_progress,
        "post_intervention_router_accuracy": (
            sum(bool(item.get("selected_action_valid")) for item in autonomous)
            / len(autonomous)
            if autonomous
            else None
        ),
        "invalid_actions_after_intervention": sum(
            not bool(item.get("selected_action_valid")) for item in autonomous
        ),
        "additional_executor_failures": sum(
            str(item.get("execution_status")) in {"failed", "invalid_output", "timeout"}
            for item in suffix
        ),
        "branch_vs_factual_completion_delta": int(
            bool((branch_episode.get("terminal_status") or {}).get("terminal"))
        )
        - int(bool((parent_episode.get("terminal_status") or {}).get("terminal"))),
        "branch_vs_factual_suffix_step_delta": len(suffix)
        - len(factual_suffix),
        "branch_vs_factual_execution_call_delta": branch_execution_calls
        - factual_execution_calls,
    }
