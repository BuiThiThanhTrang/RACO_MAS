from __future__ import annotations

from collections import defaultdict
from typing import Any

from routing_benchmark.routers import Router

from .environment import RoutingEnvironmentV2
from .generator import generate_decision_rows
from .loader import BenchmarkBundleV2
from .schemas import STOP_ACTION


def _rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _family_scores(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["workflow_family"])].append(record)
    return {
        family: {
            "count": len(items),
            "score": _rate(sum(bool(item[field]) for item in items), len(items)),
        }
        for family, items in sorted(groups.items())
    }


def _group_scores(
    records: list[dict[str, Any]], group_field: str, score_field: str
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[group_field])].append(record)
    return {
        name: {
            "count": len(items),
            "score": _rate(
                sum(bool(item[score_field]) for item in items), len(items)
            ),
        }
        for name, items in sorted(groups.items())
    }


def _macro_group_score(
    records: list[dict[str, Any]], group_field: str, score_field: str
) -> float | None:
    grouped = _group_scores(records, group_field, score_field)
    scores = [item["score"] for item in grouped.values() if item["score"] is not None]
    return _rate(sum(scores), len(scores))


def _pool_mode(tags: list[str] | tuple[str, ...]) -> str:
    return next(
        (item.split(":", 1)[1] for item in tags if item.startswith("pool:")),
        "unspecified",
    )


def _router_observation(row: dict[str, Any]) -> dict[str, Any]:
    observation = {
        "benchmark": "RoutingBench",
        "benchmark_version": row["schema_version"],
        "scenario_id": row["scenario_id"],
        "workflow_family": row["workflow_family"],
        "task": row["task"],
        "state": row["observed_state"],
        "budget": row["budget"],
        "agent_profiles": row["agent_pool"],
        "constraints": {
            "choose_exactly_one": True,
            "allowed_actions": row["candidate_actions"],
            "stop_only_when_task_complete": True,
            "quality_rankings_provided": False,
        },
    }
    if str(row["schema_version"]).startswith("2.0-beta"):
        observation["task_context"] = row.get("task_context", {})
        observation["execution_memory"] = row.get("execution_memory", {})
        observation["recent_events"] = row.get("recent_events", [])
    return observation


def _confidence_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [
        item
        for item in records
        if isinstance((item.get("router_metadata") or {}).get("confidence"), (int, float))
    ]
    if not scored:
        return {
            "expected_calibration_error": None,
            "high_confidence_error_rate": None,
        }
    bins: list[list[dict[str, Any]]] = [[] for _ in range(10)]
    for item in scored:
        confidence = min(max(float(item["router_metadata"]["confidence"]), 0.0), 1.0)
        bins[min(int(confidence * 10), 9)].append(item)
    ece = 0.0
    for items in bins:
        if not items:
            continue
        mean_confidence = sum(
            float(item["router_metadata"]["confidence"]) for item in items
        ) / len(items)
        accuracy = sum(bool(item["admissible"]) for item in items) / len(items)
        ece += len(items) / len(scored) * abs(mean_confidence - accuracy)
    high = [
        item
        for item in scored
        if float(item["router_metadata"]["confidence"]) >= 0.8
    ]
    return {
        "expected_calibration_error": ece,
        "high_confidence_error_rate": _rate(
            sum(not item["admissible"] for item in high), len(high)
        ),
    }


def evaluate_track_a_v2(
    bundle: BenchmarkBundleV2, router: Router, seed: int = 42
) -> dict[str, Any]:
    router.reset(seed)
    records: list[dict[str, Any]] = []
    for row in generate_decision_rows(bundle):
        is_beta = str(row["schema_version"]).startswith("2.0-beta")
        admissible = tuple(
            row.get("admissible_actions", row["valid_actions"])
            if is_beta
            else row["valid_actions"]
        )
        progress = tuple(row.get("progress_actions", row["valid_actions"]))
        efficient = tuple(
            row.get("oracle_efficient_actions", row["efficient_actions"])
        )
        oracle_order = efficient + tuple(
            item for item in admissible if item not in efficient
        )
        decision = router.select(
            _router_observation(row), row["candidate_actions"], oracle_order
        )
        records.append(
            {
                "decision_id": row["decision_id"],
                "scenario_id": row["scenario_id"],
                "template_id": row.get("template_id", row["scenario_id"]),
                "split_group_id": row.get(
                    "split_group_id", row.get("template_id", row["scenario_id"])
                ),
                "instance_id": row.get("instance_id"),
                "workflow_family": row["workflow_family"],
                "state_provenance": row["state_provenance"],
                "counterfactual_group_id": row["counterfactual_group_id"],
                "selected_action": decision.action,
                "valid_actions": list(admissible),
                "admissible_actions": list(admissible),
                "progress_actions": list(progress),
                "efficient_actions": list(efficient),
                "valid": decision.action in admissible,
                "admissible": decision.action in admissible,
                "progress_action": decision.action in progress,
                "efficient": decision.action in efficient,
                "selected_distractor": decision.action in row["distractor_actions"],
                "stop_ready": row["stop_ready"],
                "difficulty_tags": list(row.get("difficulty_tags") or ()),
                "pool_mode": _pool_mode(row.get("difficulty_tags") or ()),
                "router_metadata": dict(decision.metadata),
            }
        )

    predicted_stop = sum(item["selected_action"] == STOP_ACTION for item in records)
    gold_stop = sum(item["stop_ready"] for item in records)
    true_stop = sum(
        item["selected_action"] == STOP_ACTION and item["stop_ready"]
        for item in records
    )
    pair_groups: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        if record["counterfactual_group_id"]:
            pair_groups[str(record["counterfactual_group_id"])].append(record["valid"])
    post_failure = [
        item for item in records if item["state_provenance"] == "post_failure"
    ]
    post_rejection = [
        item
        for item in records
        if item["state_provenance"] in {"post_invalid", "post_routing_rejection"}
    ]
    ambiguous = [item for item in records if len(item["valid_actions"]) > 1]
    epistemic = [
        item
        for item in records
        if "epistemic_exploration" in item["difficulty_tags"]
    ]
    for item in records:
        item["admissible_cardinality"] = (
            "singleton" if len(item["valid_actions"]) == 1 else "set_valued"
        )
    metrics = {
        "track": "A",
        "benchmark_version": bundle.metadata["version"],
        "router": router.name,
        "decision_count": len(records),
        "valid_next_action_accuracy": _rate(
            sum(item["valid"] for item in records), len(records)
        ),
        "admissible_next_action_accuracy": _rate(
            sum(item["admissible"] for item in records), len(records)
        ),
        "hidden_progress_action_rate": _rate(
            sum(item["progress_action"] for item in records), len(records)
        ),
        "efficient_action_rate": _rate(
            sum(item["efficient"] for item in records), len(records)
        ),
        "ambiguous_state_accuracy": _rate(
            sum(item["valid"] for item in ambiguous), len(ambiguous)
        ),
        "post_failure_routing_accuracy": _rate(
            sum(item["valid"] for item in post_failure), len(post_failure)
        ),
        "post_routing_rejection_accuracy": _rate(
            sum(item["valid"] for item in post_rejection), len(post_rejection)
        ),
        "counterfactual_pair_accuracy": _rate(
            sum(len(values) == 2 and all(values) for values in pair_groups.values()),
            len(pair_groups),
        ),
        "epistemic_exploration_accuracy": _rate(
            sum(item["valid"] for item in epistemic), len(epistemic)
        ),
        "distractor_selection_rate": _rate(
            sum(item["selected_distractor"] for item in records), len(records)
        ),
        "stop_precision": _rate(true_stop, predicted_stop),
        "stop_recall": _rate(true_stop, gold_stop),
        "by_family": _family_scores(records, "valid"),
        "by_state_provenance": _group_scores(
            records, "state_provenance", "valid"
        ),
        "by_admissible_cardinality": _group_scores(
            records, "admissible_cardinality", "valid"
        ),
        "by_pool_mode": _group_scores(records, "pool_mode", "valid"),
        "macro_template_admissible_accuracy": _macro_group_score(
            records, "split_group_id", "admissible"
        ),
        "by_template": _group_scores(records, "split_group_id", "admissible"),
    }
    metrics.update(_confidence_metrics(records))
    return {"metrics": metrics, "records": records}


def evaluate_track_b_v2(
    bundle: BenchmarkBundleV2, router: Router, seed: int = 42
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    distractors_by_scenario = {
        item.scenario_id: set(item.distractor_ids) for item in bundle.scenarios
    }
    for index, scenario in enumerate(bundle.scenarios):
        router.reset(seed + index)
        env = RoutingEnvironmentV2(scenario, bundle.agents)
        initial_outcomes = env._success_outcomes()
        oracle_steps = min((item[0] for item in initial_outcomes), default=None)
        oracle_cost = min((item[1] for item in initial_outcomes), default=None)
        episode_events: list[dict[str, Any]] = []
        while not env.done:
            is_beta = scenario.schema_version.startswith("2.0-beta")
            valid = env.admissible_actions() if is_beta else env.valid_actions()
            efficient = env.efficient_actions()
            oracle_order = efficient + tuple(item for item in valid if item not in efficient)
            decision = router.select(
                env.observation(), env.candidate_actions(), oracle_order
            )
            event = env.step(decision.action)
            event_row = event.to_dict()
            event_row.update(
                {
                    "scenario_id": scenario.scenario_id,
                    "workflow_family": scenario.family,
                    "router_metadata": dict(decision.metadata),
                }
            )
            episode_events.append(event_row)
            events.append(event_row)
            if not env.done and not env.applicable_actions():
                env.done = True
                env.success = False
                env.terminal_reason = "UNRECOVERABLE_STATE"
        episodes.append(
            {
                "scenario_id": scenario.scenario_id,
                "template_id": scenario.template_id,
                "split_group_id": scenario.split_group_id,
                "instance_id": scenario.instance_id,
                "workflow_family": scenario.family,
                "success": env.success,
                "terminal_reason": env.terminal_reason,
                "steps": env.state.steps_used,
                "cost": env.state.cost_used,
                "oracle_steps": oracle_steps,
                "oracle_cost": oracle_cost,
                "step_regret": (
                    env.state.steps_used - oracle_steps
                    if env.success and oracle_steps is not None
                    else None
                ),
                "cost_regret": (
                    round(env.state.cost_used - oracle_cost, 6)
                    if env.success and oracle_cost is not None
                    else None
                ),
                "difficulty_tags": list(scenario.difficulty_tags),
                "pool_mode": _pool_mode(scenario.difficulty_tags),
                "events": episode_events,
            }
        )

    successful = [item for item in episodes if item["success"]]
    recovery = [
        item for item in episodes if item["workflow_family"] == "failure_recovery"
    ]
    non_stop = [item for item in events if item["action"] != STOP_ACTION]
    metrics = {
        "track": "B",
        "benchmark_version": bundle.metadata["version"],
        "router": router.name,
        "episode_count": len(episodes),
        "goal_completion_rate": _rate(
            sum(item["success"] for item in episodes), len(episodes)
        ),
        "valid_action_rate": _rate(sum(item["valid"] for item in events), len(events)),
        "admissible_action_rate": _rate(
            sum(item.get("admissible", item["valid"]) for item in events),
            len(events),
        ),
        "hidden_progress_action_rate": _rate(
            sum(item.get("progress_action", item["valid"]) for item in events),
            len(events),
        ),
        "efficient_action_rate": _rate(
            sum(item["efficient"] for item in events), len(events)
        ),
        "applicable_but_invalid_rate": _rate(
            sum(item["applicable"] and not item["valid"] for item in events),
            len(events),
        ),
        "no_progress_action_rate": _rate(
            sum(
                item["status"]
                in {
                    "NO_PROGRESS",
                    "NOT_APPLICABLE",
                    "ACTION_REJECTED",
                    "EXECUTED_NO_PROGRESS",
                }
                and not item["valid"]
                for item in non_stop
            ),
            len(non_stop),
        ),
        "scheduled_failure_event_rate": _rate(
            sum(
                item["status"]
                in {
                    "NO_PROGRESS",
                    "RETRYABLE_FAILURE",
                    "EXECUTED_NO_PROGRESS",
                    "EXECUTED_RETRYABLE_FAILURE",
                }
                and item["valid"]
                for item in non_stop
            ),
            len(non_stop),
        ),
        "premature_stop_rate": _rate(
            sum(item["status"] == "PREMATURE_STOP" for item in events),
            len(episodes),
        ),
        "distractor_selection_rate": _rate(
            sum(
                event["action"]
                in distractors_by_scenario.get(event["scenario_id"], set())
                for event in events
            ),
            len(events),
        ),
        "recovery_success_rate": _rate(
            sum(item["success"] for item in recovery), len(recovery)
        ),
        "average_step_regret_on_success": _rate(
            sum(item["step_regret"] for item in successful), len(successful)
        ),
        "average_cost_regret_on_success": _rate(
            sum(item["cost_regret"] for item in successful), len(successful)
        ),
        "action_rejection_rate": _rate(
            sum(
                item["status"] in {"NOT_APPLICABLE", "ACTION_REJECTED"}
                for item in non_stop
            ),
            len(non_stop),
        ),
        "stagnant_repeat_rate": _rate(
            sum(bool(item.get("stagnant_repeat")) for item in non_stop),
            len(non_stop),
        ),
        "loop_termination_rate": _rate(
            sum(item["terminal_reason"] == "LOOP_LIMIT" for item in episodes),
            len(episodes),
        ),
        "by_family": _family_scores(episodes, "success"),
        "by_pool_mode": _group_scores(episodes, "pool_mode", "success"),
        "macro_template_completion_rate": _macro_group_score(
            episodes, "split_group_id", "success"
        ),
        "by_template": _group_scores(episodes, "split_group_id", "success"),
    }
    return {"metrics": metrics, "episodes": episodes}
