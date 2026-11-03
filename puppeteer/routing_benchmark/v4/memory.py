from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from routing_benchmark.v2.environment import RoutingEnvironmentV2
from routing_benchmark.v2.schemas import ROUTING_REJECTION_STATUSES, STOP_ACTION


def _fixture_index(
    fixtures: Iterable[Mapping[str, Any]], scenario_id: str
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (
            str(item["action"]),
            str(item["status"]),
            str((item.get("structured_event") or {}).get("effect", "")),
        ): item
        for item in fixtures
        if str(item.get("scenario_id")) == scenario_id
    }


def build_memory_views(
    env: RoutingEnvironmentV2,
    fixtures: Iterable[Mapping[str, Any]],
    *,
    include_raw: bool,
) -> dict[str, dict[str, Any]]:
    """Create leak-free state, structured-history, and optional raw-history views."""

    observation = env.observation()
    state = dict(observation["state"])
    recent_events = [
        {
            "step": item.step,
            "agent_id": item.action,
            "status": item.status,
            "effect": item.effect,
            "semantic_delta": (
                "state_changed" if item.substantive_state_changed else "none"
            ),
        }
        for item in env.history[-5:]
    ]
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    failed_approaches: list[dict[str, Any]] = []
    for event in env.history:
        normalized = {
            "step": event.step,
            "agent_id": event.action,
            "status": event.status,
            "state_delta": "state_changed" if event.substantive_state_changed else "none",
            "effect": event.effect,
        }
        attempts[event.action].append(normalized)
        if event.action != STOP_ACTION and event.substantive_state_changed:
            completed.append(normalized)
        if event.status.startswith("EXECUTED_") and event.status != "EXECUTED_SUCCESS":
            failures.append(normalized)
            failed_approaches.append(
                {"agent_id": event.action, "status": event.status, "step": event.step}
            )
        elif event.status in ROUTING_REJECTION_STATUSES:
            failures.append(normalized)
    persistent = {
        "attempts_by_agent": dict(attempts),
        "completed_contributions": completed,
        "observed_failures": failures,
        "failed_approaches": failed_approaches,
        "available_artifacts": list(state.get("artifact_inventory") or ()),
        "open_requirements": list(state.get("open_requirements") or ()),
    }
    views: dict[str, dict[str, Any]] = {
        "state": {
            "observed_state": state,
            "execution_memory": {},
            "recent_events": [],
        },
        "structured": {
            "observed_state": state,
            "execution_memory": persistent,
            "recent_events": recent_events,
        },
    }
    if include_raw:
        index = _fixture_index(fixtures, env.scenario.scenario_id)
        raw_events: list[dict[str, Any]] = []
        for event in env.history:
            fixture = index.get((event.action, event.status, event.effect))
            raw_events.append(
                {
                    "step": event.step,
                    "agent_id": event.action,
                    "status": event.status,
                    "raw_output": (
                        str(fixture.get("raw_output", ""))
                        if fixture is not None
                        else event.effect
                    ),
                }
            )
        views["raw"] = {
            **views["structured"],
            "raw_executor_outputs": raw_events,
        }
    return views
