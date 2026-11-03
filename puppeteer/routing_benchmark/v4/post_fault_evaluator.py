from __future__ import annotations

from statistics import median
from typing import Any

from routing_benchmark.routers import Router
from routing_benchmark.v2.schemas import FAILURE_STATUSES, STOP_ACTION
from routing_benchmark.v3.evaluator import (
    LiveActorExecutor,
    LiveActorResult,
    _record_live_actor_failure,
)

from .evaluator import _group_rate, _mean, _online_observation, _rate
from .loader import BenchmarkBundleV4
from .post_fault import checkpoint_safe_actions, restore_checkpoint_environment


def evaluate_track_c_v4(
    bundle: BenchmarkBundleV4,
    router: Router,
    *,
    memory_view: str = "state",
    seed: int = 42,
    actor: LiveActorExecutor | None = None,
) -> dict[str, Any]:
    """Evaluate controlled recovery from paired post-fault checkpoints."""

    if not bundle.post_fault_runtime or not bundle.post_fault_labels:
        raise ValueError(
            "Track C requires post_fault_runtime.jsonl and post_fault_labels.jsonl"
        )
    scenarios = {item.scenario_id: item for item in bundle.scenarios}
    labels = {
        str(item["checkpoint_id"]): item for item in bundle.post_fault_labels
    }
    fixture_by_transition = {
        str(item["transition_id"]): item for item in bundle.executor_outputs
    }
    episodes: list[dict[str, Any]] = []
    for index, runtime in enumerate(bundle.post_fault_runtime):
        checkpoint_id = str(runtime["checkpoint_id"])
        label = labels[checkpoint_id]
        env = restore_checkpoint_environment(runtime, scenarios, bundle.agents)
        if env.state.fingerprint() != label["checkpoint_state_fingerprint"]:
            raise ValueError(f"Checkpoint replay mismatch: {checkpoint_id}")
        router.reset(seed + index)
        condition = str(label["condition"])
        detector_actions = set(map(str, label.get("detector_actions") or ()))
        repair_actions = set(map(str, label.get("repair_actions") or ()))
        unsafe_ids = set(map(str, label.get("unsafe_transition_ids") or ()))
        events: list[dict[str, Any]] = []
        live_contributions: list[dict[str, Any]] = []
        first_action: str | None = None
        first_action_valid = False
        first_action_unsafe = False
        first_action_detector = False
        detection_call: int | None = None
        repair_call: int | None = None
        repair_started = False
        unsafe_call: int | None = None
        unnecessary_checks = 0
        calls_after_goal_path_available = 0
        router_error = False
        executor_error = False
        while not env.done:
            observation = _online_observation(env, bundle, memory_view)
            valid = checkpoint_safe_actions(env, label)
            decision = router.select(
                observation, env.candidate_actions(), valid
            )
            if first_action is None:
                first_action = decision.action
                first_action_valid = decision.action in valid
                first_action_detector = (
                    decision.action in detector_actions and first_action_valid
                )
            transition = (
                env._transition_for(decision.action)
                if decision.action != STOP_ACTION
                else None
            )
            transition_id = (
                transition.transition_id if transition is not None else None
            )
            selected_valid = decision.action in valid
            selected_unsafe = bool(transition_id and transition_id in unsafe_ids)
            if len(events) == 0:
                first_action_unsafe = selected_unsafe
            if not selected_valid:
                router_error = True
            goal_path_available_before = "downstream_ready" in env.state.facts
            if (
                goal_path_available_before
                and decision.action != STOP_ACTION
                and not (
                    transition is not None
                    and "final_committed" in transition.add_facts
                )
            ):
                calls_after_goal_path_available += 1

            actor_result: LiveActorResult | None = None
            if actor is not None and selected_valid and decision.action != STOP_ACTION:
                profile = next(
                    item
                    for item in observation["agent_profiles"]
                    if item["agent_id"] == decision.action
                )
                execution_profile = dict(profile)
                if transition is not None and transition.add_artifacts:
                    execution_profile["expected_outputs"] = [
                        env.scenario.artifact_descriptions.get(item, item)
                        for item in sorted(transition.add_artifacts)
                    ]
                    execution_profile["role_goal"] = (
                        f"{profile.get('role_goal', '')} Current operation: "
                        f"{transition.public_effect}"
                    ).strip()
                actor_result = actor.execute(
                    task=env.scenario.task,
                    observation={
                        **observation,
                        "task_input": bundle.task_inputs.get(
                            env.scenario.scenario_id, {}
                        ),
                        "prior_live_contributions": list(live_contributions),
                    },
                    role_profile=execution_profile,
                )
                if actor_result.success:
                    event = env.step(decision.action)
                else:
                    event = _record_live_actor_failure(
                        env, decision.action, actor_result
                    )
                    executor_error = True
            else:
                event = env.step(decision.action)

            post_call = sum(
                item["action"] != STOP_ACTION for item in events
            ) + int(decision.action != STOP_ACTION)
            if (
                decision.action in detector_actions
                and selected_valid
                and detection_call is None
            ):
                detection_call = post_call
            if (
                "corrected_candidate_artifact" in env.state.artifacts
                and repair_call is None
            ):
                repair_call = post_call
            if decision.action in repair_actions and selected_valid:
                repair_started = True
            if selected_unsafe and unsafe_call is None:
                unsafe_call = post_call
            if (
                condition == "control"
                and decision.action in detector_actions
                and selected_valid
            ):
                unnecessary_checks += 1
            fixture = (
                fixture_by_transition.get(str(transition_id))
                if transition_id is not None
                else None
            )
            if actor_result is not None:
                live_contributions.append(
                    {
                        "post_call": post_call,
                        "agent_id": decision.action,
                        "status": actor_result.status,
                        "contribution": actor_result.contribution,
                        "produced_outputs": list(actor_result.produced_outputs),
                        "candidate_answer": actor_result.candidate_answer,
                    }
                )
            events.append(
                {
                    **event.to_dict(),
                    "post_call": post_call,
                    "selected_action_valid": selected_valid,
                    "selected_action_unsafe": selected_unsafe,
                    "transition_id": transition_id,
                    "raw_executor_output": (
                        actor_result.contribution
                        if actor_result is not None
                        else fixture.get("raw_output") if fixture is not None else None
                    ),
                    "controlled_fixture_output": (
                        fixture.get("raw_output") if fixture is not None else None
                    ),
                    "failure_origin": (
                        "executor"
                        if actor_result is not None and not actor_result.success
                        else "router"
                        if not selected_valid
                        else None
                    ),
                    "live_actor": (
                        {
                            "status": actor_result.status,
                            "tokens": actor_result.tokens,
                            "latency_ms": actor_result.latency_ms,
                            "model": actor_result.model,
                            "attempts": actor_result.attempts,
                        }
                        if actor_result is not None
                        else None
                    ),
                    "router_metadata": dict(decision.metadata),
                }
            )
            if not env.done and not env.applicable_actions():
                env.done = True
                env.success = False
                env.terminal_reason = "UNRECOVERABLE_STATE"

        calls = sum(item["action"] != STOP_ACTION for item in events)
        excess_calls = (
            calls - int(label["oracle_min_remaining_calls"])
            if env.success
            else None
        )
        episodes.append(
            {
                "checkpoint_id": checkpoint_id,
                "pair_id": label["pair_id"],
                "condition": condition,
                "fault_class": label["fault_class"],
                "injection_position": label.get("injection_position"),
                "workflow_family": runtime["workflow_family"],
                "split": runtime["split"],
                "success": env.success,
                "terminal_reason": env.terminal_reason,
                "calls": calls,
                "first_action": first_action,
                "first_action_valid": first_action_valid,
                "first_action_unsafe": first_action_unsafe,
                "first_action_detector": first_action_detector,
                "requires_detection_agent": bool(
                    label.get("requires_detection_agent")
                ),
                "fault_detected": detection_call is not None,
                "calls_to_detection": detection_call,
                "repaired_artifact": repair_call is not None,
                "repair_started": repair_started,
                "calls_to_repair": repair_call,
                "unsafe_consumption": unsafe_call is not None,
                "unsafe_call": unsafe_call,
                "unnecessary_check_count": unnecessary_checks,
                "loop_after_checkpoint": env.terminal_reason == "LOOP_LIMIT"
                or any(item["stagnant_repeat"] for item in events),
                "budget_exhausted": env.terminal_reason
                in {"STEP_LIMIT", "COST_LIMIT"},
                "premature_stop": env.terminal_reason == "PREMATURE_STOP",
                "oracle_min_remaining_calls": label[
                    "oracle_min_remaining_calls"
                ],
                "recovery_excess_calls": excess_calls,
                "calls_after_goal_path_available": calls_after_goal_path_available,
                "failure_attribution": (
                    "success"
                    if env.success
                    else "mixed"
                    if router_error and executor_error
                    else "router"
                    if router_error
                    else "executor"
                    if executor_error
                    else "controlled_environment"
                ),
                "events": events,
            }
        )

    fault_episodes = [item for item in episodes if item["condition"] == "fault"]
    control_episodes = [
        item for item in episodes if item["condition"] == "control"
    ]
    detectable = [
        item for item in fault_episodes if item["requires_detection_agent"]
    ]
    repair_attempts = [item for item in fault_episodes if item["repair_started"]]
    all_events = [event for item in episodes for event in item["events"]]
    successful = [item for item in episodes if item["success"]]
    pairs_by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for episode in episodes:
        pairs_by_id.setdefault(str(episode["pair_id"]), {})[
            str(episode["condition"])
        ] = episode
    paired: list[dict[str, Any]] = []
    for pair_id, members in sorted(pairs_by_id.items()):
        if set(members) != {"fault", "control"}:
            continue
        fault = members["fault"]
        control = members["control"]
        paired.append(
            {
                "pair_id": pair_id,
                "workflow_family": fault["workflow_family"],
                "fault_class": fault["fault_class"],
                "injection_position": fault["injection_position"],
                "control_success": control["success"],
                "fault_success": fault["success"],
                "success_degradation": int(bool(control["success"]))
                - int(bool(fault["success"])),
                "recovery_overhead_calls": fault["calls"] - control["calls"],
            }
        )
    metrics = {
        "track": "C",
        "execution_mode": (
            "live_llm_actor_from_controlled_checkpoint"
            if actor is not None
            else "deterministic_checkpoint_simulator"
        ),
        "simulated_actor_outputs": actor is None,
        "actor_model": actor.model if actor is not None else None,
        "benchmark_version": bundle.metadata["version"],
        "router": router.name,
        "memory_view": memory_view,
        "checkpoint_episode_count": len(episodes),
        "fault_checkpoint_count": len(fault_episodes),
        "control_checkpoint_count": len(control_episodes),
        "post_fault_goal_completion_rate": _rate(
            sum(item["success"] for item in fault_episodes), len(fault_episodes)
        ),
        "clean_continuation_success_rate": _rate(
            sum(item["success"] for item in control_episodes),
            len(control_episodes),
        ),
        "first_action_valid_accuracy": _rate(
            sum(item["first_action_valid"] for item in episodes), len(episodes)
        ),
        "first_action_unsafe_rate": _rate(
            sum(item["first_action_unsafe"] for item in episodes), len(episodes)
        ),
        "first_action_detector_rate": _rate(
            sum(item["first_action_detector"] for item in detectable),
            len(detectable),
        ),
        "post_checkpoint_valid_transition_rate": _rate(
            sum(item["selected_action_valid"] for item in all_events),
            len(all_events),
        ),
        "fault_detection_recall": _rate(
            sum(item["fault_detected"] for item in detectable), len(detectable)
        ),
        "repair_success_rate": _rate(
            sum(item["repaired_artifact"] for item in repair_attempts),
            len(repair_attempts),
        ),
        "unsafe_consumption_rate": _rate(
            sum(item["unsafe_consumption"] for item in fault_episodes),
            len(fault_episodes),
        ),
        "clean_unnecessary_check_rate": _rate(
            sum(item["unnecessary_check_count"] > 0 for item in control_episodes),
            len(control_episodes),
        ),
        "loop_after_checkpoint_rate": _rate(
            sum(item["loop_after_checkpoint"] for item in episodes), len(episodes)
        ),
        "budget_exhaustion_rate": _rate(
            sum(item["budget_exhausted"] for item in episodes), len(episodes)
        ),
        "premature_stop_rate": _rate(
            sum(item["premature_stop"] for item in episodes), len(episodes)
        ),
        "median_calls_to_detection": (
            median(
                item["calls_to_detection"]
                for item in detectable
                if item["calls_to_detection"] is not None
            )
            if any(item["calls_to_detection"] is not None for item in detectable)
            else None
        ),
        "median_calls_to_repair": (
            median(
                item["calls_to_repair"]
                for item in fault_episodes
                if item["calls_to_repair"] is not None
            )
            if any(item["calls_to_repair"] is not None for item in fault_episodes)
            else None
        ),
        "median_calls_to_goal": (
            median(item["calls"] for item in successful) if successful else None
        ),
        "mean_recovery_excess_calls": _mean(
            [
                float(item["recovery_excess_calls"])
                for item in successful
                if item["recovery_excess_calls"] is not None
            ]
        ),
        "mean_calls_after_goal_path_available": _mean(
            [float(item["calls_after_goal_path_available"]) for item in episodes]
        ),
        "paired_counterfactual_count": len(paired),
        "mean_success_degradation": _mean(
            [float(item["success_degradation"]) for item in paired]
        ),
        "mean_recovery_overhead_calls": _mean(
            [float(item["recovery_overhead_calls"]) for item in paired]
        ),
        "by_domain": _group_rate(
            fault_episodes, "workflow_family", "success"
        ),
        "by_fault_class": _group_rate(
            fault_episodes, "fault_class", "success"
        ),
        "by_injection_position": _group_rate(
            fault_episodes, "injection_position", "success"
        ),
        "combined_weighted_score": None,
        "metric_policy": (
            "post-fault recovery, clean specificity, safety, and efficiency are "
            "reported separately"
        ),
    }
    if actor is not None:
        metrics.update(
            {
                "live_actor_call_count": actor.calls,
                "live_actor_tokens": actor.total_tokens,
                "live_actor_latency_ms": actor.total_latency_ms,
            }
        )
    return {"metrics": metrics, "episodes": episodes, "paired": paired}
