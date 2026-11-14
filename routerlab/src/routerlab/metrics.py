from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Iterable

from .contracts import EpisodeResult


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_metrics(results: Iterable[EpisodeResult]) -> dict:
    episodes = list(results)
    initial_base = [
        bool(item.initial_evaluation and item.initial_evaluation.base_passed)
        for item in episodes
    ]
    final_base = [
        bool(item.final_evaluation and item.final_evaluation.base_passed)
        for item in episodes
    ]
    initial_plus = [item.initial_success for item in episodes]
    final_plus = [item.final_success for item in episodes]
    initial_plus_rate = _rate(initial_plus)
    final_plus_rate = _rate(final_plus)
    failed_initial = sum(not value for value in initial_plus)
    recovered = sum(
        (not before) and after for before, after in zip(initial_plus, final_plus)
    )
    actions = Counter(
        step.action_id
        for item in episodes
        for step in item.steps
        if not step.router_decision
    )
    router_steps = [
        step for item in episodes for step in item.steps if step.router_decision
    ]
    executor_steps = [
        step for item in episodes for step in item.steps if not step.router_decision
    ]
    invalid_router = sum(not bool(step.metadata.get("valid", True)) for step in router_steps)
    no_op_actions = sum(bool(step.metadata.get("no_op", False)) for step in executor_steps)
    no_submissions = sum(not item.has_submission for item in episodes)
    premature_stops = sum(
        item.termination_reason == "router_stop_no_submission" for item in episodes
    )
    routing_steps_to_first_code = []
    calls_to_first_code = []
    for item in episodes:
        first_index = next(
            (
                index
                for index, step in enumerate(item.steps)
                if step.code_version_before == 0 and step.code_version_after > 0
            ),
            None,
        )
        if first_index is None:
            continue
        prefix = item.steps[: first_index + 1]
        routing_steps_to_first_code.append(sum(step.router_decision for step in prefix))
        calls_to_first_code.append(
            sum(
                (not step.router_decision)
                or bool(step.metadata.get("counts_model_call", False))
                for step in prefix
            )
        )
    solved = sum(final_plus)
    total_cost = sum(item.total_cost_usd for item in episodes)
    denominator = 1.0 - initial_plus_rate
    return {
        "task_count": len(episodes),
        "quality": {
            "initial_base_pass_rate": _rate(initial_base),
            "final_base_pass_rate": _rate(final_base),
            "initial_plus_pass_rate": initial_plus_rate,
            "final_plus_pass_rate": final_plus_rate,
            "absolute_plus_improvement": final_plus_rate - initial_plus_rate,
            "normalized_plus_improvement": (
                (final_plus_rate - initial_plus_rate) / denominator
                if denominator > 0
                else 0.0
            ),
            "repair_recovery_rate": recovered / failed_initial if failed_initial else 0.0,
        },
        "efficiency": {
            "mean_model_calls": mean([x.total_model_calls for x in episodes])
            if episodes
            else 0.0,
            "mean_total_tokens": mean(
                [x.total_input_tokens + x.total_output_tokens for x in episodes]
            )
            if episodes
            else 0.0,
            "mean_base_executions": mean(
                [x.total_base_executions for x in episodes]
            )
            if episodes
            else 0.0,
            "mean_wall_time_ms": mean([x.wall_time_ms for x in episodes])
            if episodes
            else 0.0,
            "total_cost_usd": total_cost,
            "cost_per_solved_task_usd": total_cost / solved if solved else None,
        },
        "routing": {
            "executor_utilization": dict(sorted(actions.items())),
            "router_decisions": len(router_steps),
            "invalid_router_decisions": invalid_router,
            "invalid_router_rate": invalid_router / len(router_steps)
            if router_steps
            else 0.0,
            "no_op_actions": no_op_actions,
            "no_op_action_rate": no_op_actions / len(executor_steps)
            if executor_steps
            else 0.0,
            "no_submission_rate": no_submissions / len(episodes) if episodes else 0.0,
            "premature_stop_rate": premature_stops / len(episodes)
            if episodes
            else 0.0,
            "mean_routing_steps_to_first_code": mean(routing_steps_to_first_code)
            if routing_steps_to_first_code
            else None,
            "mean_model_calls_to_first_code": mean(calls_to_first_code)
            if calls_to_first_code
            else None,
        },
        "termination_reasons": dict(
            sorted(Counter(item.termination_reason for item in episodes).items())
        ),
    }
