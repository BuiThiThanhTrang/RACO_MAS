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
    final_plus = [item.final_success for item in episodes]
    final_plus_rate = _rate(final_plus)
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
    budget_violations = sum(
        bool(step.metadata.get("budget_violation", False)) for step in router_steps
    )
    no_op_actions = sum(bool(step.metadata.get("no_op", False)) for step in executor_steps)
    action_rejections = sum(
        bool(step.metadata.get("action_rejected", False)) for step in executor_steps
    )
    duplicate_candidates = sum(
        bool(step.metadata.get("duplicate_candidate", False)) for step in executor_steps
    )
    semantic_progress = sum(
        bool(step.metadata.get("semantic_delta", False)) for step in executor_steps
    )
    context_budget_failures = sum(
        step.metadata.get("status") == "context_budget_exceeded"
        for step in executor_steps
    )
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
    base_recovery_opportunities = sum(
        bool(item.initial_evaluation and not item.initial_evaluation.base_passed)
        for item in episodes
    )
    base_recoveries = sum(
        bool(
            item.initial_evaluation
            and not item.initial_evaluation.base_passed
            and item.final_evaluation
            and item.final_evaluation.base_passed
        )
        for item in episodes
    )
    base_regressions = sum(
        bool(
            item.initial_evaluation
            and item.initial_evaluation.base_passed
            and item.final_evaluation
            and not item.final_evaluation.base_passed
        )
        for item in episodes
    )
    return {
        "task_count": len(episodes),
        "quality": {
            "initial_base_pass_rate": _rate(initial_base),
            "final_base_pass_rate": _rate(final_base),
            # Plus Tests are intentionally evaluated only for the final
            # submission. Keep the legacy keys as null rather than fabricating
            # an initial failure rate from unobserved data.
            "initial_plus_pass_rate": None,
            "final_plus_pass_rate": final_plus_rate,
            "absolute_plus_improvement": None,
            "normalized_plus_improvement": None,
            "repair_recovery_rate": None,
            "base_recovery_opportunities": base_recovery_opportunities,
            "base_recoveries": base_recoveries,
            "base_recovery_rate": (
                base_recoveries / base_recovery_opportunities
                if base_recovery_opportunities
                else None
            ),
            "base_regressions": base_regressions,
        },
        "efficiency": {
            "mean_executor_steps": mean([x.total_steps for x in episodes])
            if episodes
            else 0.0,
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
            "budget_violation_decisions": budget_violations,
            "budget_violation_rate": budget_violations / len(router_steps)
            if router_steps
            else 0.0,
            "no_op_actions": no_op_actions,
            "no_op_action_rate": no_op_actions / len(executor_steps)
            if executor_steps
            else 0.0,
            "action_rejections": action_rejections,
            "action_rejection_rate": action_rejections / len(executor_steps)
            if executor_steps
            else 0.0,
            "duplicate_candidates": duplicate_candidates,
            "duplicate_candidate_rate": duplicate_candidates / len(executor_steps)
            if executor_steps
            else 0.0,
            "semantic_progress_actions": semantic_progress,
            "semantic_progress_rate": semantic_progress / len(executor_steps)
            if executor_steps
            else 0.0,
            "context_budget_failures": context_budget_failures,
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
