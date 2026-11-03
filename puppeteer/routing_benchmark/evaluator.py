from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable

from .environment import RoutingEnvironment
from .loader import BenchmarkBundle
from .routers import Router
from .schemas import ScenarioSpec, STOP_ACTION, state_fingerprint


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _reachable_environments(
    scenario: ScenarioSpec,
    bundle: BenchmarkBundle,
) -> list[RoutingEnvironment]:
    initial = RoutingEnvironment(scenario, bundle.agents)
    queue = deque([initial])
    seen = {state_fingerprint(initial.facts)}
    states: list[RoutingEnvironment] = []
    while queue:
        env = queue.popleft()
        states.append(env)
        if env.goal_reached:
            continue
        for action in env.valid_actions():
            if action == STOP_ACTION:
                continue
            child = env.clone()
            child.step(action)
            if child.done and not child.goal_reached:
                continue
            fingerprint = state_fingerprint(child.facts)
            if fingerprint not in seen:
                seen.add(fingerprint)
                queue.append(child)
    return states


def _oracle_steps_from_env(initial: RoutingEnvironment) -> int | None:
    """Shortest valid action count from an environment, including final STOP."""
    queue = deque([(initial.clone(), 0)])
    seen = {state_fingerprint(initial.facts)}
    while queue:
        env, depth = queue.popleft()
        if env.goal_reached:
            return depth + 1
        for action in env.valid_actions():
            if action == STOP_ACTION:
                continue
            child = env.clone()
            child.step(action)
            fingerprint = state_fingerprint(child.facts)
            if fingerprint in seen or not child.history[-1].state_changed:
                continue
            seen.add(fingerprint)
            queue.append((child, depth + 1))
    return None


def _oracle_steps(scenario: ScenarioSpec, bundle: BenchmarkBundle) -> int | None:
    return _oracle_steps_from_env(RoutingEnvironment(scenario, bundle.agents))


def _oracle_action_order(env: RoutingEnvironment) -> tuple[str, ...]:
    """Rank valid actions by shortest remaining successful trajectory."""
    ranked: list[tuple[float, int, str]] = []
    for index, action in enumerate(env.valid_actions()):
        if action == STOP_ACTION:
            distance = 1.0
        else:
            child = env.clone()
            child.step(action)
            remaining = _oracle_steps_from_env(child)
            distance = float("inf") if remaining is None else 1.0 + remaining
        ranked.append((distance, index, action))
    return tuple(action for _, _, action in sorted(ranked))


def _family_metrics(records: Iterable[dict[str, Any]], correct_key: str) -> dict:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["family"])].append(record)
    return {
        family: {
            "count": len(items),
            "score": _safe_rate(sum(bool(item[correct_key]) for item in items), len(items)),
        }
        for family, items in sorted(grouped.items())
    }


def evaluate_track_a(
    bundle: BenchmarkBundle,
    router: Router,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate isolated next-agent decisions over all reachable V1 states."""
    router.reset(seed)
    records: list[dict[str, Any]] = []
    for scenario in bundle.scenarios:
        for case_index, env in enumerate(_reachable_environments(scenario, bundle)):
            valid_actions = env.valid_actions()
            decision = router.select(
                env.observation(), env.candidate_actions(), _oracle_action_order(env)
            )
            correct = decision.action in valid_actions
            record = {
                "decision_id": f"{scenario.scenario_id}:state-{case_index:03d}",
                "scenario_id": scenario.scenario_id,
                "family": scenario.family,
                "state_fingerprint": state_fingerprint(env.facts),
                "visible_facts": sorted(env.facts),
                "candidate_actions": list(env.candidate_actions()),
                "valid_actions": list(valid_actions),
                "selected_action": decision.action,
                "correct": correct,
                "selected_distractor": bool(
                    decision.action in bundle.agents
                    and bundle.agents[decision.action].distractor
                ),
                "router_metadata": dict(decision.metadata),
            }
            records.append(record)

    predicted_stop = sum(item["selected_action"] == STOP_ACTION for item in records)
    gold_stop = sum(STOP_ACTION in item["valid_actions"] for item in records)
    true_stop = sum(
        item["selected_action"] == STOP_ACTION
        and STOP_ACTION in item["valid_actions"]
        for item in records
    )
    stop_precision = _safe_rate(true_stop, predicted_stop)
    stop_recall = _safe_rate(true_stop, gold_stop)
    stop_f1 = (
        2 * stop_precision * stop_recall / (stop_precision + stop_recall)
        if stop_precision is not None
        and stop_recall is not None
        and stop_precision + stop_recall
        else None
    )
    nonterminal = [item for item in records if STOP_ACTION not in item["valid_actions"]]
    metrics = {
        "track": "A",
        "benchmark_version": bundle.version,
        "router": router.name,
        "decision_count": len(records),
        "valid_next_agent_accuracy": _safe_rate(
            sum(item["correct"] for item in records), len(records)
        ),
        "nonterminal_routing_accuracy": _safe_rate(
            sum(item["correct"] for item in nonterminal), len(nonterminal)
        ),
        "distractor_selection_rate": _safe_rate(
            sum(item["selected_distractor"] for item in records), len(records)
        ),
        "stop_precision": stop_precision,
        "stop_recall": stop_recall,
        "stop_f1": stop_f1,
        "by_family": _family_metrics(records, "correct"),
    }
    return {"metrics": metrics, "records": records}


def evaluate_track_b(
    bundle: BenchmarkBundle,
    router: Router,
    seed: int = 42,
) -> dict[str, Any]:
    """Run complete interactive trajectories in the deterministic environment."""
    episodes: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(bundle.scenarios):
        router.reset(seed + scenario_index)
        env = RoutingEnvironment(scenario, bundle.agents)
        decision_audit: list[dict[str, Any]] = []
        while not env.done:
            observation = env.observation()
            valid_actions = env.valid_actions()
            decision = router.select(
                observation, env.candidate_actions(), _oracle_action_order(env)
            )
            event = env.step(decision.action)
            event_dict = event.to_dict()
            event_dict.update(
                {
                    "scenario_id": scenario.scenario_id,
                    "family": scenario.family,
                    "router_metadata": dict(decision.metadata),
                }
            )
            all_events.append(event_dict)
            decision_audit.append(event_dict)

        oracle_steps = _oracle_steps(scenario, bundle)
        excess_steps = (
            len(env.history) - oracle_steps
            if env.success and oracle_steps is not None
            else None
        )
        episodes.append(
            {
                "scenario_id": scenario.scenario_id,
                "family": scenario.family,
                "tags": list(scenario.tags),
                "success": env.success,
                "goal_reached": env.goal_reached,
                "terminal_reason": env.terminal_reason,
                "steps": len(env.history),
                "total_cost": env.total_cost,
                "oracle_steps": oracle_steps,
                "excess_steps": excess_steps,
                "normalized_step_regret": (
                    excess_steps / oracle_steps
                    if excess_steps is not None and oracle_steps
                    else None
                ),
                "events": decision_audit,
            }
        )

    recovery = [item for item in episodes if "recovery" in item["tags"]]
    successful_regrets = [
        item["normalized_step_regret"]
        for item in episodes
        if item["normalized_step_regret"] is not None
    ]
    non_stop_events = [item for item in all_events if item["action"] != STOP_ACTION]
    metrics = {
        "track": "B",
        "benchmark_version": bundle.version,
        "router": router.name,
        "episode_count": len(episodes),
        "goal_completion_rate": _safe_rate(
            sum(item["success"] for item in episodes), len(episodes)
        ),
        "valid_action_rate": _safe_rate(
            sum(item["valid_action"] for item in all_events), len(all_events)
        ),
        "invalid_action_rate": _safe_rate(
            sum(not item["valid_action"] for item in all_events), len(all_events)
        ),
        "no_progress_action_rate": _safe_rate(
            sum(not item["state_changed"] for item in non_stop_events),
            len(non_stop_events),
        ),
        "redundant_action_rate": _safe_rate(
            sum(item["redundant"] for item in all_events), len(all_events)
        ),
        "distractor_selection_rate": _safe_rate(
            sum(item["distractor"] for item in all_events), len(all_events)
        ),
        "premature_stop_rate": _safe_rate(
            sum(item["status"] == "PREMATURE_STOP" for item in all_events),
            len(episodes),
        ),
        "recovery_success_rate": _safe_rate(
            sum(item["success"] for item in recovery), len(recovery)
        ),
        "average_steps": _safe_rate(
            sum(item["steps"] for item in episodes), len(episodes)
        ),
        "average_cost": _safe_rate(
            sum(item["total_cost"] for item in episodes), len(episodes)
        ),
        "average_normalized_step_regret_on_success": _safe_rate(
            sum(successful_regrets), len(successful_regrets)
        ),
        "by_family": _family_metrics(episodes, "success"),
    }
    return {"metrics": metrics, "episodes": episodes}
