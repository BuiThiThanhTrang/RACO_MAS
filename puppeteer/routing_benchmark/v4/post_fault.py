from __future__ import annotations

import hashlib
from collections import deque
from typing import Any, Iterable, Mapping

from routing_benchmark.v2.environment import RoutingEnvironmentV2
from routing_benchmark.v2.schemas import (
    AgentProfileV2,
    ScenarioSpecV2,
    StepEventV2,
    WorkflowStateV2,
)

from .public import build_online_observation


def _base_safe_actions(
    env: RoutingEnvironmentV2, annotation: Mapping[str, Any]
) -> tuple[str, ...]:
    if env.goal_reached:
        return ("STOP",)
    excluded = set(map(str, annotation.get("unsafe_transition_ids") or ()))
    excluded.update(
        map(str, annotation.get("inefficient_transition_ids") or ())
    )
    result: list[str] = []
    for action in env.applicable_actions():
        if action == "STOP":
            continue
        transition = env._transition_for(action)
        if transition is not None and transition.transition_id not in excluded:
            result.append(action)
    return tuple(result)


def _checkpoint_id(pair_id: str, condition: str) -> str:
    digest = hashlib.sha256(
        f"routingbench-v4-track-c|{pair_id}|{condition}".encode("utf-8")
    ).hexdigest()[:20]
    return f"pfc-{digest}"


def _state_to_dict(state: WorkflowStateV2) -> dict[str, Any]:
    return {
        "facts": sorted(state.facts),
        "artifacts": sorted(state.artifacts),
        "last_action": state.last_action,
        "last_status": state.last_status,
        "last_effect": state.last_effect,
        "failure_count": state.failure_count,
        "routing_rejection_count": state.routing_rejection_count,
        "actor_failure_count": state.actor_failure_count,
        "steps_used": state.steps_used,
        "cost_used": state.cost_used,
    }


def _state_from_dict(raw: Mapping[str, Any]) -> WorkflowStateV2:
    return WorkflowStateV2(
        facts=set(map(str, raw.get("facts") or ())),
        artifacts=set(map(str, raw.get("artifacts") or ())),
        last_action=(str(raw["last_action"]) if raw.get("last_action") else None),
        last_status=(str(raw["last_status"]) if raw.get("last_status") else None),
        last_effect=(str(raw["last_effect"]) if raw.get("last_effect") else None),
        failure_count=int(raw.get("failure_count", 0)),
        routing_rejection_count=int(raw.get("routing_rejection_count", 0)),
        actor_failure_count=int(raw.get("actor_failure_count", 0)),
        steps_used=int(raw.get("steps_used", 0)),
        cost_used=float(raw.get("cost_used", 0.0)),
    )


def _event_from_dict(raw: Mapping[str, Any]) -> StepEventV2:
    tuple_fields = {
        "applicable_before",
        "valid_before",
        "efficient_before",
        "admissible_before",
        "progress_actions_before",
    }
    values = dict(raw)
    for field in tuple_fields:
        values[field] = tuple(map(str, values.get(field) or ()))
    return StepEventV2(**values)


def restore_checkpoint_environment(
    runtime: Mapping[str, Any],
    scenarios: Mapping[str, ScenarioSpecV2],
    agents: dict[str, AgentProfileV2],
) -> RoutingEnvironmentV2:
    scenario_id = str(runtime["internal_scenario_id"])
    scenario = scenarios[scenario_id]
    env = RoutingEnvironmentV2(
        scenario,
        agents,
        _state_from_dict(runtime["state"]),
    )
    env.history = [
        _event_from_dict(item) for item in runtime.get("history") or ()
    ]
    env.done = False
    env.success = False
    env.terminal_reason = None
    return env


def checkpoint_safe_actions(
    env: RoutingEnvironmentV2, label: Mapping[str, Any]
) -> tuple[str, ...]:
    if env.goal_reached:
        return ("STOP",)
    excluded = set(map(str, label.get("unsafe_transition_ids") or ()))
    excluded.update(map(str, label.get("inefficient_transition_ids") or ()))
    result: list[str] = []
    for action in env.applicable_actions():
        if action == "STOP":
            continue
        transition = env._transition_for(action)
        if transition is not None and transition.transition_id not in excluded:
            result.append(action)
    return tuple(result)


def _find_pre_transition_environment(
    scenario: ScenarioSpecV2,
    agents: dict[str, AgentProfileV2],
    annotation: Mapping[str, Any],
    transition_id: str,
    *,
    fast: bool = False,
) -> RoutingEnvironmentV2:
    target = next(
        item for item in scenario.transitions if item.transition_id == transition_id
    )
    initial = RoutingEnvironmentV2(scenario, agents)
    queue = deque([initial])
    seen = {initial.state.fingerprint()}
    while queue:
        env = queue.popleft()
        matched = env._transition_for(target.action)
        if matched is not None and matched.transition_id == transition_id:
            return env
        for action in _base_safe_actions(env, annotation):
            if action == "STOP":
                continue
            child = env.clone()
            _step_for_checkpoint(child, action, fast=fast)
            if child.done or child.goal_reached:
                continue
            fingerprint = child.state.fingerprint()
            if fingerprint not in seen:
                seen.add(fingerprint)
                queue.append(child)
    raise ValueError(
        f"Cannot replay a prefix for post-fault transition {transition_id}"
    )


def _matched_clean_environment(
    scenario: ScenarioSpecV2,
    agents: dict[str, AgentProfileV2],
    target_steps: int,
    *,
    fast: bool = False,
) -> RoutingEnvironmentV2:
    env = RoutingEnvironmentV2(scenario, agents)
    primary = next(
        item for item in scenario.transitions if ":execute-primary:" in item.transition_id
    )
    _step_for_checkpoint(env, primary.action, fast=fast)
    while env.state.steps_used < target_steps:
        candidates = [
            item
            for item in scenario.transitions
            if ":optional-clean-check:" in item.transition_id
            and item.matches(env.state)
        ]
        if not candidates:
            raise ValueError(
                f"No matched clean continuation at depth {target_steps} for "
                f"{scenario.scenario_id}"
            )
        _step_for_checkpoint(
            env,
            sorted(candidates, key=lambda item: item.transition_id)[0].action,
            fast=fast,
        )
    if env.state.steps_used != target_steps:
        raise ValueError(f"Clean checkpoint depth mismatch: {scenario.scenario_id}")
    return env


def _shortest_safe_continuation(
    initial: RoutingEnvironmentV2,
    label: Mapping[str, Any],
    *,
    fast: bool = False,
) -> int | None:
    queue = deque([(initial.clone(), 0)])
    seen = {initial.state.fingerprint()}
    while queue:
        env, calls = queue.popleft()
        if env.goal_reached:
            return calls
        for action in checkpoint_safe_actions(env, label):
            if action == "STOP":
                continue
            child = env.clone()
            _step_for_checkpoint(child, action, fast=fast)
            if child.done and not child.goal_reached:
                continue
            fingerprint = child.state.fingerprint()
            if fingerprint not in seen:
                seen.add(fingerprint)
                queue.append((child, calls + 1))
    return None


def _step_for_checkpoint(
    env: RoutingEnvironmentV2, action: str, *, fast: bool
) -> StepEventV2:
    """Replay a checkpoint transition, optionally skipping unused V2 utilities."""

    if not fast:
        return env.step(action)
    before = env.state.fingerprint()
    before_progress = env.state.progress_fingerprint()
    applicable_before = env.applicable_actions()
    redundant = any(
        item.action == action and item.state_before == before for item in env.history
    )
    transition = env._transition_for(action)
    cost = env._cost_for(action, transition)
    env._mutate(action, transition, cost)
    after = env.state.fingerprint()
    changed = before_progress != env.state.progress_fingerprint()
    stagnant_repeat = (
        not changed
        and any(
            item.action == action and not item.substantive_state_changed
            for item in reversed(env.history)
        )
    )
    event = StepEventV2(
        step=env.state.steps_used,
        action=action,
        status=str(env.state.last_status),
        applicable=action in applicable_before,
        valid=action in applicable_before,
        efficient=False,
        admissible=action in applicable_before,
        progress_action=changed,
        applicable_before=applicable_before,
        valid_before=applicable_before,
        efficient_before=(),
        admissible_before=applicable_before,
        progress_actions_before=applicable_before,
        state_before=before,
        state_after=after,
        substantive_state_changed=changed,
        redundant=redundant,
        stagnant_repeat=stagnant_repeat,
        cost=cost,
        effect=str(env.state.last_effect or ""),
    )
    env.history.append(event)
    threshold = env.scenario.max_stagnant_repeats
    if threshold and not env.done:
        stagnant_tail = 0
        for previous in reversed(env.history):
            if previous.action != action or previous.substantive_state_changed:
                break
            stagnant_tail += 1
        if stagnant_tail >= threshold:
            env.done = True
            env.success = False
            env.terminal_reason = "LOOP_LIMIT"
    return event


def _unsafe_actions(
    env: RoutingEnvironmentV2, transition_ids: Iterable[str]
) -> list[str]:
    ids = set(map(str, transition_ids))
    return sorted(
        {
            transition.action
            for transition in env.scenario.transitions
            if transition.transition_id in ids and transition.matches(env.state)
        }
    )


def generate_post_fault_track(
    *,
    scenarios: Iterable[ScenarioSpecV2],
    agents: dict[str, AgentProfileV2],
    fixtures: Iterable[Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
    benchmark_version: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Create paired post-fault and clean-control checkpoints for Track C."""

    scenario_by_id = {item.scenario_id: item for item in scenarios}
    fixtures = tuple(fixtures)
    public_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    fast_replay = str(benchmark_version) == "4.0-beta"
    for scenario in sorted(scenario_by_id.values(), key=lambda item: item.scenario_id):
        annotation = annotations[scenario.scenario_id]
        if annotation.get("variant") == "clean":
            continue
        fault = (annotation.get("faults") or [None])[0]
        if not fault:
            raise ValueError(f"Missing fault annotation: {scenario.scenario_id}")
        transition_id = str(fault["transition_ids"][0])
        fault_env = _find_pre_transition_environment(
            scenario, agents, annotation, transition_id, fast=fast_replay
        )
        target = next(
            item for item in scenario.transitions if item.transition_id == transition_id
        )
        executed = _step_for_checkpoint(
            fault_env, target.action, fast=fast_replay
        )
        if executed.state_after != fault_env.state.fingerprint():
            raise ValueError(f"Fault replay mismatch: {transition_id}")

        clean_scenario = scenario_by_id[str(annotation["clean_scenario_id"])]
        clean_annotation = annotations[clean_scenario.scenario_id]
        clean_env = _matched_clean_environment(
            clean_scenario,
            agents,
            fault_env.state.steps_used,
            fast=fast_replay,
        )
        if clean_env.remaining_steps != fault_env.remaining_steps:
            raise ValueError(f"Checkpoint budget mismatch: {scenario.scenario_id}")

        pair_scope = str(
            annotation.get("base_workflow_id") or scenario.family
        )
        pair_id = f"track-c:{pair_scope}:{annotation['variant']}"
        pair_entry = {
            "pair_id": pair_id,
            "split": scenario.split,
            "workflow_family": scenario.family,
            "fault_class": annotation["variant"],
            "injection_position": fault.get("injection_position"),
        }
        for condition, env, active_annotation in (
            ("fault", fault_env, annotation),
            ("control", clean_env, clean_annotation),
        ):
            checkpoint_id = _checkpoint_id(pair_id, condition)
            fault_transition_ids = (
                list(map(str, fault.get("transition_ids") or ()))
                if condition == "fault"
                else []
            )
            unsafe_transition_ids = sorted(
                set(
                    map(
                        str,
                        active_annotation.get("unsafe_transition_ids") or (),
                    )
                ).union(fault_transition_ids)
            )
            label = {
                "checkpoint_id": checkpoint_id,
                "pair_id": pair_id,
                "condition": condition,
                "internal_scenario_id": env.scenario.scenario_id,
                "fault_class": annotation["variant"],
                "injection_position": fault.get("injection_position"),
                "requires_detection_agent": bool(
                    fault.get("requires_detection_agent")
                )
                if condition == "fault"
                else False,
                "unsafe_transition_ids": unsafe_transition_ids,
                "inefficient_transition_ids": list(
                    map(
                        str,
                        active_annotation.get("inefficient_transition_ids") or (),
                    )
                ),
                "detector_actions": list(
                    map(str, active_annotation.get("detector_actions") or ())
                ),
                "repair_actions": list(
                    map(str, active_annotation.get("repair_actions") or ())
                ),
                "recovery_actions": list(
                    map(str, active_annotation.get("recovery_actions") or ())
                ),
            }
            valid = checkpoint_safe_actions(env, label)
            label["valid_actions"] = list(valid)
            label["unsafe_actions"] = _unsafe_actions(
                env, unsafe_transition_ids
            )
            label["detector_actions_available"] = sorted(
                set(valid).intersection(label["detector_actions"])
            )
            minimum = _shortest_safe_continuation(
                env, label, fast=fast_replay
            )
            if minimum is None:
                raise ValueError(f"Unrecoverable checkpoint: {checkpoint_id}")
            label["oracle_min_remaining_calls"] = minimum
            label["checkpoint_state_fingerprint"] = env.state.fingerprint()

            runtime_rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "split": scenario.split,
                    "workflow_family": scenario.family,
                    "internal_scenario_id": env.scenario.scenario_id,
                    "state": _state_to_dict(env.state),
                    "history": [item.to_dict() for item in env.history],
                }
            )
            views = (
                ("state", "structured", "raw")
                if scenario.split == "development"
                else ("state", "structured")
            )
            public_rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "split": scenario.split,
                    "workflow_family": scenario.family,
                    "observations": {
                        view: build_online_observation(
                            env=env,
                            agents=agents,
                            fixtures=fixtures,
                            benchmark_version=benchmark_version,
                            memory_view=view,
                        )
                        for view in views
                    },
                }
            )
            label_rows.append(label)
            pair_entry[f"{condition}_checkpoint_id"] = checkpoint_id
        pairs.append(pair_entry)
    return public_rows, runtime_rows, label_rows, pairs
