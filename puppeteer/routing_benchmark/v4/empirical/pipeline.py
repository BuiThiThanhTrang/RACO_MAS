from __future__ import annotations

from typing import Any, Mapping

from routing_benchmark.v4.execution.state_history import state_delta

from .schema import EMPIRICAL_SCHEMA_VERSION, empirical_decision_id
from .state import apply_outcome, build_empirical_memory_views


def materialize_empirical_decision(
    output: Mapping[str, Any], outcome: Mapping[str, Any]
) -> dict[str, Any]:
    context = dict(output.get("collection_context") or {})
    pre_state = dict(context.get("pre_state") or {})
    action = str(output["selected_action"])
    # Imported live trajectories already contain the exact state produced by the
    # execution adapter. The judge interprets the raw output for annotation and
    # review, but must never rewrite empirical state. Legacy collections without
    # that state continue to use the deterministic outcome fold.
    empirical_post_state = context.get("empirical_post_state")
    post_state = (
        dict(empirical_post_state)
        if isinstance(empirical_post_state, Mapping) and empirical_post_state
        else apply_outcome(pre_state, outcome, action=action)
    )
    artifacts = outcome.get("produced_artifacts") or ()
    evidence_refs = sorted(
        {
            str(ref)
            for artifact in artifacts
            if isinstance(artifact, Mapping)
            for ref in artifact.get("evidence_refs") or ()
        }
    )
    event = {
        "step": post_state.get("step_index"),
        "agent_id": action,
        "intent": str((context.get("role_profile") or {}).get("role_goal", "")),
        "execution_status": outcome.get("execution_status"),
        "state_delta": state_delta(pre_state, post_state),
        "raw_output": output.get("raw_output", ""),
        "output_id": output.get("output_id"),
        "output_ref": output.get("output_ref") or output.get("output_id"),
    }
    prior_events = list(context.get("prior_empirical_events") or ())
    events = [*prior_events, event]
    views = build_empirical_memory_views(post_state, events)
    output_id = str(output["output_id"])
    return {
        "schema_version": EMPIRICAL_SCHEMA_VERSION,
        "decision_id": empirical_decision_id(output_id),
        "source_decision_id": context.get("source_decision_id"),
        "scenario_id": output.get("scenario_id"),
        "workflow_id": output.get("workflow_id"),
        "split": context.get("split"),
        "workflow_family": context.get("workflow_family"),
        "task": context.get("task"),
        "task_context": context.get("task_context", {}),
        "post_state": post_state,
        "memory_views": views,
        "budget": {
            **dict(context.get("budget") or {}),
            "calls_remaining": max(
                int((context.get("budget") or {}).get("calls_remaining", 1)) - 1,
                0,
            ),
        },
        "agent_pool": list(context.get("agent_pool") or ()),
        "candidate_actions": list(context.get("candidate_actions") or ()),
        "empirical_output_id": output_id,
        "outcome_id": outcome.get("outcome_id"),
        "review_status": "pending",
    }


def public_observation_for(
    decision: Mapping[str, Any], memory_view: str = "structured"
) -> dict[str, Any]:
    view = dict((decision.get("memory_views") or {})[memory_view])
    result = {
        "benchmark": "RoutingBench",
        "benchmark_version": EMPIRICAL_SCHEMA_VERSION,
        "scenario_id": decision.get("workflow_id"),
        "workflow_family": "dynamic_workflow",
        "task": decision.get("task"),
        "task_context": decision.get("task_context", {}),
        "state": view.get("observed_state", {}),
        "budget": decision.get("budget", {}),
        "agent_profiles": decision.get("agent_pool", []),
        "constraints": {
            "choose_exactly_one": True,
            "allowed_actions": decision.get("candidate_actions", []),
            "stop_only_when_task_complete": True,
            "set_valued_ground_truth": True,
            "preferred_action_provided": False,
            "quality_rankings_provided": False,
            "reserved_terminal_stop": True,
            "memory_view": memory_view,
        },
    }
    if memory_view in {"structured", "raw_empirical"}:
        result["structured_history"] = view.get("structured_history", [])
    if memory_view == "raw_empirical":
        result["raw_executor_outputs"] = view.get("raw_executor_outputs", [])
    return result


def public_decision_record(
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    view_names = ["state", "structured"]
    if decision.get("split") == "development":
        view_names.append("raw_empirical")
    return {
        "schema_version": EMPIRICAL_SCHEMA_VERSION,
        "decision_id": decision["decision_id"],
        "split": decision.get("split"),
        "workflow_family": "dynamic_workflow",
        "observations": {
            name: public_observation_for(decision, name)
            for name in view_names
        },
    }


def private_label_record(
    decision: Mapping[str, Any], annotation: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": EMPIRICAL_SCHEMA_VERSION,
        "decision_id": decision["decision_id"],
        "source_decision_id": decision.get("source_decision_id"),
        "scenario_id": decision.get("scenario_id"),
        "workflow_id": decision.get("workflow_id"),
        "proposed_valid_actions": list(annotation.get("proposed_valid_actions") or ()),
        "human_valid_actions": [],
        "action_annotations": list(annotation.get("action_annotations") or ()),
        "judge_provenance": dict(annotation.get("judge_provenance") or {}),
        "review_status": annotation.get("review_status", "pending"),
    }
