from __future__ import annotations

from typing import Any, Mapping, Sequence

from routing_benchmark.v2.schemas import STOP_ACTION
from routing_benchmark.v4.execution.state_history import state_delta

from .pipeline import public_observation_for
from .state import apply_outcome, build_empirical_memory_views


def _decision(
    *,
    row: Mapping[str, Any],
    state: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    calls_remaining: int,
) -> dict[str, Any]:
    return {
        "decision_id": f"live:{row['decision_id']}:{len(events):03d}",
        "scenario_id": row.get("scenario_id"),
        "workflow_id": row.get("template_id") or row.get("scenario_id"),
        "split": row.get("split"),
        "workflow_family": row.get("workflow_family"),
        "task": row.get("task"),
        "task_context": row.get("task_context", {}),
        "post_state": dict(state),
        "memory_views": build_empirical_memory_views(state, events),
        "budget": {
            "calls_remaining": max(int(calls_remaining), 0),
            "reserved_terminal_stop": True,
        },
        "agent_pool": list(row.get("agent_pool") or ()),
        "candidate_actions": list(row.get("candidate_actions") or ()),
    }


def _reference(task_input: Mapping[str, Any]) -> tuple[str | None, tuple[str, ...]]:
    source = task_input.get("source_task_input") or {}
    answer = source.get("answer") or task_input.get("answer")
    aliases = source.get("answer_aliases") or source.get("aliases") or task_input.get("answer_aliases") or ()
    return (str(answer) if answer not in (None, "") else None, tuple(map(str, aliases)))


def evaluate_live_empirical_track_b(
    *,
    rows: Sequence[Mapping[str, Any]],
    task_inputs: Mapping[str, Mapping[str, Any]],
    router,
    actor,
    outcome_judge,
    action_judge,
    post_task_judge,
    memory_view: str = "structured",
    seed: int = 42,
    max_calls: int | None = None,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        router.reset(seed + index)
        state = dict(row["observed_state"])
        events: list[dict[str, Any]] = []
        outputs: list[dict[str, Any]] = []
        labels: list[dict[str, Any]] = []
        calls_left = int(max_calls or (row.get("budget") or {}).get("calls_remaining", 1))
        terminal_reason = "CALL_LIMIT"
        success = False
        profiles = {str(item["agent_id"]): dict(item) for item in row.get("agent_pool") or ()}
        task_input = dict(task_inputs.get(str(row["scenario_id"]), {}))
        while calls_left >= 0:
            decision = _decision(row=row, state=state, events=events, calls_remaining=calls_left)
            observation = public_observation_for(decision, memory_view)
            annotation = action_judge.label_actions(
                decision_id=str(decision["decision_id"]),
                public_observation=observation,
                candidate_actions=decision["candidate_actions"],
            )
            labels.append(annotation)
            valid = tuple(annotation["proposed_valid_actions"])
            selected = router.select(observation, tuple(decision["candidate_actions"]), valid)
            if selected.action == STOP_ACTION:
                success = STOP_ACTION in valid
                terminal_reason = "VALID_STOP" if success else "PREMATURE_STOP"
                events.append(
                    {
                        "step": len(events) + 1,
                        "agent_id": STOP_ACTION,
                        "status": terminal_reason.lower(),
                        "state_delta": "",
                        "raw_output": "",
                        "selected_action_valid": success,
                    }
                )
                break
            if calls_left <= 0:
                terminal_reason = "CALL_LIMIT"
                break
            profile = profiles.get(selected.action)
            if profile is None:
                terminal_reason = "UNKNOWN_ACTION"
                break
            output = actor.collect(
                scenario_id=str(row["scenario_id"]),
                workflow_id=str(row.get("template_id") or row["scenario_id"]),
                task=str(row["task"]),
                task_input=task_input,
                pre_state=state,
                action=selected.action,
                role_profile=profile,
                replicate_id=len(outputs),
            )
            output["collection_context"].update(
                {
                    "candidate_actions": decision["candidate_actions"],
                    "agent_pool": decision["agent_pool"],
                    "budget": decision["budget"],
                }
            )
            outcome = outcome_judge.interpret_outcome(output)
            new_state = apply_outcome(state, outcome, action=selected.action)
            event = {
                "step": len(events) + 1,
                "agent_id": selected.action,
                "intent": profile.get("role_goal", ""),
                "execution_status": outcome["execution_status"],
                "state_delta": state_delta(state, new_state),
                "raw_output": output["raw_output"],
                "output_id": output["output_id"],
                "selected_action_valid": selected.action in valid,
            }
            outputs.append({"output": output, "outcome": outcome})
            events.append(event)
            state = new_state
            calls_left -= 1
        answer, aliases = _reference(task_input)
        semantic = post_task_judge.judge_post_task(
            task=str(row["task"]),
            task_input=task_input,
            final_state=state,
            trajectory_outputs=[
                {
                    "agent_id": item["agent_id"],
                    "status": item["status"],
                    "raw_output": item["raw_output"],
                }
                for item in events
                if item["agent_id"] != STOP_ACTION
            ],
            reference_answer=answer,
            answer_aliases=aliases,
        )
        episodes.append(
            {
                "scenario_id": row["scenario_id"],
                "success": success,
                "terminal_reason": terminal_reason,
                "final_state": state,
                "events": events,
                "judge_action_labels": labels,
                "post_task_judgment": semantic,
                "executor_outputs": outputs,
            }
        )
    return {
        "metrics": {
            "track": "B-live-empirical",
            "episode_count": len(episodes),
            "completion_rate": (
                sum(bool(item["success"]) for item in episodes) / len(episodes)
                if episodes
                else None
            ),
            "semantic_correctness_rate": (
                sum(
                    item["post_task_judgment"].get("semantic_correctness") is True
                    for item in episodes
                    if item["post_task_judgment"].get("semantic_correctness") is not None
                )
                / sum(
                    item["post_task_judgment"].get("semantic_correctness") is not None
                    for item in episodes
                )
                if any(
                    item["post_task_judgment"].get("semantic_correctness") is not None
                    for item in episodes
                )
                else None
            ),
        },
        "episodes": episodes,
    }
