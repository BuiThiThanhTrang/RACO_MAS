from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Mapping

from routing_benchmark.routers import Router
from routing_benchmark.v2.environment import RoutingEnvironmentV2
from routing_benchmark.v2.schemas import FAILURE_STATUSES, STOP_ACTION
from routing_benchmark.v3.evaluator import (
    LiveActorExecutor,
    LiveActorResult,
    _record_live_actor_failure,
)

from .generator import _phase, _safe_actions
from .loader import BenchmarkBundleV4
from .public import build_online_observation, build_public_observation


def _rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _group_rate(
    records: list[dict[str, Any]], key: str, score: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        groups[str(item.get(key, "unknown"))].append(item)
    return {
        name: {
            "count": len(items),
            "score": _rate(sum(bool(item.get(score)) for item in items), len(items)),
        }
        for name, items in sorted(groups.items())
    }


def _view_observation(
    *,
    row: Mapping[str, Any],
    memory_view: str,
) -> dict[str, Any]:
    return build_public_observation(row, memory_view)


def _label_index(bundle: BenchmarkBundleV4) -> dict[str, dict[str, Any]]:
    return {
        str(item["decision_id"]): item
        for annotation in bundle.recovery_annotations.values()
        for item in annotation.get("decision_labels") or ()
    }


def evaluate_track_a_v4(
    bundle: BenchmarkBundleV4,
    router: Router,
    *,
    memory_view: str = "state",
    seed: int = 42,
) -> dict[str, Any]:
    router.reset(seed)
    labels = _label_index(bundle)
    records: list[dict[str, Any]] = []
    for row in bundle.decision_rows:
        valid = tuple(map(str, row["valid_actions"]))
        decision = router.select(
            _view_observation(row=row, memory_view=memory_view),
            tuple(map(str, row["candidate_actions"])),
            valid,
        )
        private = labels.get(str(row["decision_id"]), {})
        unsafe = set(map(str, private.get("unsafe_actions") or ()))
        records.append(
            {
                "decision_id": row["decision_id"],
                "scenario_id": row["scenario_id"],
                "workflow_family": row["workflow_family"],
                "phase": private.get("phase", row.get("state_provenance")),
                "selected_action": decision.action,
                "valid_actions": list(valid),
                "valid": decision.action in valid,
                "unsafe_selection": decision.action in unsafe,
                "terminal_valid": bool(row["terminal_valid"]),
                "premature_stop": decision.action == STOP_ACTION
                and not row["terminal_valid"],
                "detector_available": bool(private.get("detector_actions_available")),
                "selected_detector": decision.action
                in set(private.get("detector_actions_available") or ()),
                "router_metadata": dict(decision.metadata),
            }
        )
    recovery_phases = {"verification", "recovery", "repair", "contaminated"}
    recovery = [item for item in records if item["phase"] in recovery_phases]
    detector_states = [item for item in records if item["detector_available"]]
    nonterminal = [item for item in records if not item["terminal_valid"]]
    metrics = {
        "track": "A",
        "benchmark_version": bundle.metadata["version"],
        "router": router.name,
        "memory_view": memory_view,
        "decision_count": len(records),
        "valid_action_accuracy": _rate(sum(item["valid"] for item in records), len(records)),
        "recovery_state_accuracy": _rate(
            sum(item["valid"] for item in recovery), len(recovery)
        ),
        "unsafe_selection_rate": _rate(
            sum(item["unsafe_selection"] for item in records), len(records)
        ),
        "detector_selection_rate_when_available": _rate(
            sum(item["selected_detector"] for item in detector_states),
            len(detector_states),
        ),
        "premature_stop_rate": _rate(
            sum(item["premature_stop"] for item in nonterminal), len(nonterminal)
        ),
        "by_domain": _group_rate(records, "workflow_family", "valid"),
        "by_phase": _group_rate(records, "phase", "valid"),
        "combined_weighted_score": None,
        "metric_policy": "routing and recovery metrics are reported separately without utility weights",
    }
    return {"metrics": metrics, "records": records}


def _online_observation(
    env: RoutingEnvironmentV2,
    bundle: BenchmarkBundleV4,
    memory_view: str,
) -> dict[str, Any]:
    return build_online_observation(
        env=env,
        agents=bundle.agents,
        fixtures=bundle.executor_outputs,
        benchmark_version=str(bundle.metadata["version"]),
        memory_view=memory_view,
    )


def evaluate_track_b_v4(
    bundle: BenchmarkBundleV4,
    router: Router,
    *,
    memory_view: str = "state",
    seed: int = 42,
    actor: LiveActorExecutor | None = None,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    check_actions = {"agent_07", "agent_18", "agent_19", "agent_27", "agent_39"}
    fixture_by_transition = {
        str(item["transition_id"]): item for item in bundle.executor_outputs
    }
    for index, scenario in enumerate(bundle.scenarios):
        if memory_view == "raw" and scenario.split != "development":
            continue
        router.reset(seed + index)
        env = RoutingEnvironmentV2(scenario, bundle.agents)
        annotation = bundle.recovery_annotations[scenario.scenario_id]
        unsafe_ids = set(map(str, annotation.get("unsafe_transition_ids") or ()))
        fault_ids = {
            str(transition_id)
            for fault in annotation.get("faults") or ()
            for transition_id in fault.get("transition_ids") or ()
        }
        detector_actions = set(map(str, annotation.get("detector_actions") or ()))
        events: list[dict[str, Any]] = []
        fault_step: int | None = None
        detection_step: int | None = None
        recovery_step: int | None = None
        unsafe_step: int | None = None
        unnecessary_checks = 0
        downstream_fault_events = 0
        calls_after_goal_path_available = 0
        repair_started = False
        live_contributions: list[dict[str, Any]] = []
        router_error = False
        executor_error = False
        while not env.done:
            observation = _online_observation(env, bundle, memory_view)
            valid = _safe_actions(env, annotation)
            decision = router.select(observation, env.candidate_actions(), valid)
            transition = (
                env._transition_for(decision.action)
                if decision.action != STOP_ACTION
                else None
            )
            transition_id = transition.transition_id if transition is not None else None
            selected_valid = decision.action in valid
            selected_unsafe = bool(transition_id and transition_id in unsafe_ids)
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
                        scenario.artifact_descriptions.get(item, item)
                        for item in sorted(transition.add_artifacts)
                    ]
                    execution_profile["role_goal"] = (
                        f"{profile.get('role_goal', '')} Current operation: "
                        f"{transition.public_effect}"
                    ).strip()
                actor_result = actor.execute(
                    task=scenario.task,
                    observation={
                        **observation,
                        "task_input": bundle.task_inputs.get(scenario.scenario_id, {}),
                        "prior_live_contributions": list(live_contributions),
                    },
                    role_profile=execution_profile,
                )
                if actor_result.success:
                    # The controlled transition remains authoritative for fault
                    # injection; the live actor determines whether execution
                    # reaches that post-execution check at all.
                    event = env.step(decision.action)
                else:
                    event = _record_live_actor_failure(
                        env, decision.action, actor_result
                    )
                    executor_error = True
            else:
                event = env.step(decision.action)

            if not selected_valid:
                router_error = True
            transition_executed = actor_result is None or actor_result.success
            fixture = (
                fixture_by_transition.get(str(transition_id))
                if transition_id is not None
                else None
            )
            if (
                transition_executed
                and transition_id in fault_ids
                and fault_step is None
            ):
                fault_step = event.step
            if (
                fault_step is not None
                and decision.action in detector_actions
                and selected_valid
                and detection_step is None
            ):
                detection_step = event.step
            if selected_unsafe:
                unsafe_step = unsafe_step or event.step
                downstream_fault_events += 1
            if (
                fault_step is not None
                and "corrected_candidate_artifact" in env.state.artifacts
                and recovery_step is None
            ):
                recovery_step = event.step
            if fault_step is not None and decision.action in set(
                annotation.get("repair_actions") or ()
            ):
                repair_started = True
            if (
                annotation["variant"] == "clean"
                and decision.action in check_actions
                and _phase(env) not in {"verification", "repair", "recovery", "contaminated"}
            ):
                unnecessary_checks += 1
            if (
                fault_step is not None
                and event.status in FAILURE_STATUSES
                and transition_id not in fault_ids
            ):
                downstream_fault_events += 1
            if actor_result is not None:
                live_contributions.append(
                    {
                        "step": event.step,
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
                        else "controlled_post_execution_fault"
                        if transition_id in fault_ids and transition_executed
                        else None
                    ),
                    "live_actor": (
                        {
                            "status": actor_result.status,
                            "contribution": actor_result.contribution,
                            "produced_outputs": list(actor_result.produced_outputs),
                            "candidate_answer": actor_result.candidate_answer,
                            "tokens": actor_result.tokens,
                            "latency_ms": actor_result.latency_ms,
                            "model": actor_result.model,
                            "attempts": actor_result.attempts,
                            "contract_validation_errors": list(
                                actor_result.raw.get("contract_validation_errors", ())
                            ),
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
        repeated_after_fault = any(
            item["stagnant_repeat"] and fault_step is not None and item["step"] > fault_step
            for item in events
        )
        fault_injected = fault_step is not None
        non_stop_events = [
            item for item in events if item["action"] != STOP_ACTION
        ]
        first_progress_index = next(
            (
                index
                for index, item in enumerate(non_stop_events)
                if item["substantive_state_changed"]
            ),
            len(non_stop_events),
        )
        budget_wasted_before_first_progress = sum(
            not item["substantive_state_changed"]
            for item in non_stop_events[:first_progress_index]
        )
        task_executor_failure_observed = any(
            item["status"] in FAILURE_STATUSES for item in non_stop_events
        )
        episodes.append(
            {
                "scenario_id": scenario.scenario_id,
                "pair_id": annotation["pair_id"],
                "variant": annotation["variant"],
                "workflow_family": scenario.family,
                "split": scenario.split,
                "success": env.success,
                "terminal_reason": env.terminal_reason,
                "calls": calls,
                "fault_injected": fault_injected,
                "fault_step": fault_step,
                "fault_detected": detection_step is not None,
                "detection_step": detection_step,
                "recovered_artifact": recovery_step is not None,
                "repair_started": repair_started,
                "recovery_step": recovery_step,
                "unsafe_consumption": unsafe_step is not None,
                "unsafe_step": unsafe_step,
                "contained_before_contamination": fault_injected
                and unsafe_step is None,
                "injection_position": (
                    (annotation.get("faults") or [{}])[0].get("injection_position")
                    if annotation.get("faults")
                    else None
                ),
                "unnecessary_check_count": unnecessary_checks,
                "loop_after_failure": repeated_after_fault,
                "downstream_fault_events": downstream_fault_events,
                "budget_wasted_before_first_progress": budget_wasted_before_first_progress,
                "calls_after_goal_path_available": calls_after_goal_path_available,
                "task_executor_failure_observed": task_executor_failure_observed,
                "recovered_after_task_executor_failure": (
                    task_executor_failure_observed and env.success
                ),
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

    fault_episodes = [item for item in episodes if item["fault_injected"]]
    detectable_ids = {
        scenario_id
        for scenario_id, annotation in bundle.recovery_annotations.items()
        if any(
            bool(item.get("requires_detection_agent"))
            for item in annotation.get("faults") or ()
        )
    }
    detectable = [item for item in fault_episodes if item["scenario_id"] in detectable_ids]
    repair_started = [item for item in fault_episodes if item["repair_started"]]
    clean_by_pair = {
        item["pair_id"]: item for item in episodes if item["variant"] == "clean"
    }
    paired: list[dict[str, Any]] = []
    for item in fault_episodes:
        clean = clean_by_pair.get(item["pair_id"])
        if clean is None:
            continue
        paired.append(
            {
                "scenario_id": item["scenario_id"],
                "pair_id": item["pair_id"],
                "variant": item["variant"],
                "clean_success": clean["success"],
                "fault_success": item["success"],
                "success_degradation": int(bool(clean["success"]))
                - int(bool(item["success"])),
                "excess_calls": item["calls"] - clean["calls"],
                "excess_tokens": None,
                "failure_amplification": (
                    item["downstream_fault_events"] / 1.0
                ),
            }
        )
    detection_delays = [
        item["detection_step"] - item["fault_step"]
        for item in fault_episodes
        if item["detection_step"] is not None and item["fault_step"] is not None
    ]
    recovery_delays = [
        item["recovery_step"] - item["fault_step"]
        for item in fault_episodes
        if item["recovery_step"] is not None and item["fault_step"] is not None
    ]
    all_events = [event for item in episodes for event in item["events"]]
    non_stop_events = [
        item for item in all_events if item["action"] != STOP_ACTION
    ]
    executed_applicable_events = [
        item
        for item in non_stop_events
        if item.get("applicable") and item["status"] != "ACTION_REJECTED"
    ]
    pre_execution_rejections = [
        item for item in non_stop_events if item["status"] == "ACTION_REJECTED"
    ]
    applicable_no_progress = [
        item
        for item in executed_applicable_events
        if item["status"] == "EXECUTED_NO_PROGRESS"
    ]
    inefficient_executions = [
        item
        for item in executed_applicable_events
        if item["status"] == "EXECUTED_INEFFICIENT"
    ]
    task_executor_failures = [
        item
        for item in executed_applicable_events
        if item["status"] in FAILURE_STATUSES
    ]
    task_failure_episodes = [
        item for item in episodes if item["task_executor_failure_observed"]
    ]
    clean_events = [
        event
        for item in episodes
        if item["variant"] == "clean"
        for event in item["events"]
    ]
    metrics = {
        "track": "B",
        "execution_mode": (
            "live_llm_actor_with_controlled_fault_injection"
            if actor is not None
            else "deterministic_recovery_simulator"
        ),
        "simulated_actor_outputs": actor is None,
        "actor_model": actor.model if actor is not None else None,
        "benchmark_version": bundle.metadata["version"],
        "router": router.name,
        "memory_view": memory_view,
        "episode_count": len(episodes),
        "clean_control_success_rate": _rate(
            sum(item["success"] for item in episodes if item["variant"] == "clean"),
            sum(item["variant"] == "clean" for item in episodes),
        ),
        "fault_scheduled_episode_count": sum(
            item["variant"] != "clean" for item in episodes
        ),
        "fault_injected_episode_count": len(fault_episodes),
        "fault_injection_coverage": _rate(
            len(fault_episodes),
            sum(item["variant"] != "clean" for item in episodes),
        ),
        "goal_completion_rate": _rate(sum(item["success"] for item in episodes), len(episodes)),
        "valid_transition_rate": _rate(
            sum(item["selected_action_valid"] for item in all_events), len(all_events)
        ),
        "pre_execution_rejection_rate": _rate(
            len(pre_execution_rejections), len(non_stop_events)
        ),
        "applicable_no_progress_rate": _rate(
            len(applicable_no_progress), len(executed_applicable_events)
        ),
        "inefficient_execution_rate": _rate(
            len(inefficient_executions), len(executed_applicable_events)
        ),
        "task_executor_failure_rate": _rate(
            len(task_executor_failures), len(executed_applicable_events)
        ),
        "recovery_after_executor_failure_rate": _rate(
            sum(
                item["recovered_after_task_executor_failure"]
                for item in task_failure_episodes
            ),
            len(task_failure_episodes),
        ),
        "task_executor_failure_episode_count": len(task_failure_episodes),
        "mean_budget_wasted_before_first_progress": _mean(
            [
                float(item["budget_wasted_before_first_progress"])
                for item in episodes
            ]
        ),
        "mean_calls_after_goal_path_available": _mean(
            [
                float(item["calls_after_goal_path_available"])
                for item in episodes
            ]
        ),
        "recovery_success_rate": _rate(
            sum(item["success"] for item in fault_episodes), len(fault_episodes)
        ),
        "fault_detection_recall": _rate(
            sum(item["fault_detected"] for item in detectable), len(detectable)
        ),
        "repair_success_rate": _rate(
            sum(item["recovered_artifact"] for item in repair_started),
            len(repair_started),
        ),
        "unsafe_consumption_rate": _rate(
            sum(item["unsafe_consumption"] for item in fault_episodes),
            len(fault_episodes),
        ),
        "cascade_containment_rate": _rate(
            sum(item["contained_before_contamination"] for item in fault_episodes),
            len(fault_episodes),
        ),
        "loop_after_failure_rate": _rate(
            sum(item["loop_after_failure"] for item in fault_episodes),
            len(fault_episodes),
        ),
        "unnecessary_check_rate": _rate(
            sum(item["unnecessary_check_count"] for item in episodes),
            sum(event["action"] in check_actions for event in clean_events),
        ),
        "median_calls_to_detection": median(detection_delays) if detection_delays else None,
        "median_calls_to_recovery": median(recovery_delays) if recovery_delays else None,
        "paired_counterfactual_count": len(paired),
        "mean_success_degradation": _mean(
            [float(item["success_degradation"]) for item in paired]
        ),
        "mean_excess_calls": _mean([float(item["excess_calls"]) for item in paired]),
        "mean_excess_tokens": None,
        "mean_failure_amplification": _mean(
            [float(item["failure_amplification"]) for item in paired]
        ),
        "by_domain": _group_rate(episodes, "workflow_family", "success"),
        "by_fault_class": _group_rate(
            [item for item in episodes if item["variant"] != "clean"],
            "variant",
            "success",
        ),
        "by_injection_position": _group_rate(
            fault_episodes, "injection_position", "success"
        ),
        "combined_weighted_score": None,
        "metric_policy": "routing, detection, repair, propagation, and paired overhead are reported separately",
    }
    if actor is not None:
        live_events = [item for item in all_events if item.get("live_actor")]
        metrics.update(
            {
                "live_actor_call_count": len(live_events),
                "actor_contract_failure_count": sum(
                    item["live_actor"]["status"] == "INVALID_OUTPUT"
                    for item in live_events
                ),
                "actor_no_progress_count": sum(
                    item["live_actor"]["status"] == "NO_PROGRESS"
                    for item in live_events
                ),
                "router_caused_failure_rate": _rate(
                    sum(item["failure_attribution"] == "router" for item in episodes),
                    len(episodes),
                ),
                "executor_caused_failure_rate": _rate(
                    sum(item["failure_attribution"] == "executor" for item in episodes),
                    len(episodes),
                ),
                "mixed_failure_rate": _rate(
                    sum(item["failure_attribution"] == "mixed" for item in episodes),
                    len(episodes),
                ),
            }
        )
    return {"metrics": metrics, "episodes": episodes, "paired": paired}
