from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

from routing_benchmark.v2.environment import RoutingEnvironmentV2
from routing_benchmark.v2.schemas import AgentProfileV2

from .memory import build_memory_views


def public_scenario_id(workflow_family: str) -> str:
    """Return an identifier that is invariant across hidden counterfactuals."""

    return f"v4-{workflow_family}-case"


def build_public_observation(
    row: Mapping[str, Any], memory_view: str
) -> dict[str, Any]:
    """Materialize the exact Track A object that a router is allowed to see."""

    views = row.get("memory_views") or {}
    if memory_view not in views:
        raise ValueError(
            f"Memory view {memory_view!r} is unavailable for {row.get('decision_id')}"
        )
    view = dict(views[memory_view])
    beta = str(row.get("schema_version", "")) == "4.0-beta"
    public_family = "dynamic_workflow" if beta else row["workflow_family"]
    observation = {
        "benchmark": "RoutingBench",
        "benchmark_version": row["schema_version"],
        "scenario_id": str(
            row.get(
                "router_scenario_id",
                public_scenario_id(str(row["workflow_family"])),
            )
        ),
        "workflow_family": public_family,
        "task": row["task"],
        "task_context": row.get("task_context", {}),
        "state": view["observed_state"],
        "execution_memory": view.get("execution_memory", {}),
        "recent_events": view.get("recent_events", []),
        "budget": row["budget"],
        "agent_profiles": row["agent_pool"],
        "constraints": {
            "choose_exactly_one": True,
            "allowed_actions": row["candidate_actions"],
            "stop_only_when_task_complete": True,
            "set_valued_ground_truth": True,
            "preferred_action_provided": False,
            "quality_rankings_provided": False,
            "reserved_terminal_stop": True,
            "memory_view": memory_view,
        },
    }
    if memory_view == "raw":
        observation["raw_executor_outputs"] = view.get(
            "raw_executor_outputs", []
        )
    return observation


def build_online_observation(
    *,
    env: RoutingEnvironmentV2,
    agents: Mapping[str, AgentProfileV2],
    fixtures: Iterable[Mapping[str, Any]],
    benchmark_version: str,
    memory_view: str,
) -> dict[str, Any]:
    """Materialize the public payload for an interactive routing state."""

    views = build_memory_views(
        env,
        fixtures,
        include_raw=memory_view == "raw",
    )
    if memory_view not in views:
        raise ValueError(f"Memory view {memory_view!r} is unavailable")
    view = views[memory_view]
    beta = str(benchmark_version) == "4.0-beta"
    public_family = "dynamic_workflow" if beta else env.scenario.family
    scenario_id = (
        "v4b-"
        + hashlib.sha256(env.scenario.template_id.encode("utf-8")).hexdigest()[:16]
        if beta
        else public_scenario_id(env.scenario.family)
    )
    observation = {
        "benchmark": "RoutingBench",
        "benchmark_version": benchmark_version,
        "scenario_id": scenario_id,
        "workflow_family": public_family,
        "task": env.scenario.task,
        "task_context": dict(env.scenario.public_context),
        "state": view["observed_state"],
        "execution_memory": view.get("execution_memory", {}),
        "recent_events": view.get("recent_events", []),
        "budget": {
            "calls_remaining": env.remaining_steps,
            "reserved_terminal_stop": True,
        },
        "agent_profiles": [
            agents[item].public_profile() for item in env.scenario.agent_pool
        ],
        "constraints": {
            "choose_exactly_one": True,
            "allowed_actions": list(env.candidate_actions()),
            "stop_only_when_task_complete": True,
            "set_valued_ground_truth": True,
            "preferred_action_provided": False,
            "quality_rankings_provided": False,
            "reserved_terminal_stop": True,
            "memory_view": memory_view,
        },
    }
    if memory_view == "raw":
        observation["raw_executor_outputs"] = view.get(
            "raw_executor_outputs", []
        )
    return observation
