from __future__ import annotations

import json
import re
import string
from collections import defaultdict
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence

from routing_benchmark.routers import Router
from routing_benchmark.v2.environment import RoutingEnvironmentV2
from routing_benchmark.v2.schemas import FAILURE_STATUSES, StepEventV2

from .loader import BenchmarkBundleV3
from .schemas import STOP_ACTION


def _rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _groups(
    records: list[dict[str, Any]], field: str, score: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(field, "unknown"))].append(record)
    return {
        key: {
            "count": len(items),
            "score": _rate(sum(bool(item[score]) for item in items), len(items)),
        }
        for key, items in sorted(grouped.items())
    }


def _pool_mode(tags: Sequence[str]) -> str:
    return next(
        (item.split(":", 1)[1] for item in tags if item.startswith("pool:")),
        "unspecified",
    )


def _canonical_output_id(value: Any) -> str:
    """Return a stable machine-facing identifier for an artifact contract."""

    text = str(value or "").strip().casefold()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _router_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": "RoutingBench",
        "benchmark_version": row["schema_version"],
        "scenario_id": row["scenario_id"],
        "workflow_family": row["workflow_family"],
        "task": row["task"],
        "task_context": row.get("task_context", {}),
        "state": row["observed_state"],
        "execution_memory": row.get("execution_memory", {}),
        "recent_events": row.get("recent_events", []),
        "budget": row["budget"],
        "agent_profiles": row["agent_pool"],
        "constraints": {
            "choose_exactly_one": True,
            "allowed_actions": row["candidate_actions"],
            "stop_only_when_task_complete": True,
            "set_valued_ground_truth": True,
            "preferred_action_provided": False,
            "quality_rankings_provided": False,
            "stop_uses_execution_budget": not str(
                row["schema_version"]
            ).startswith("3.0-beta"),
            "reserved_terminal_stop": str(row["schema_version"]).startswith(
                "3.0-beta"
            ),
        },
    }


def evaluate_track_a_v3(
    bundle: BenchmarkBundleV3, router: Router, seed: int = 42
) -> dict[str, Any]:
    router.reset(seed)
    records: list[dict[str, Any]] = []
    for row in bundle.decision_rows:
        valid = tuple(row["valid_actions"])
        decision = router.select(
            _router_observation(row), row["candidate_actions"], valid
        )
        accepted = decision.action in valid
        records.append(
            {
                "decision_id": row["decision_id"],
                "scenario_id": row["scenario_id"],
                "provenance_id": row["provenance_id"],
                "workflow_family": row["workflow_family"],
                "state_provenance": row["state_provenance"],
                "selected_action": decision.action,
                "valid_actions": list(valid),
                "valid": accepted,
                "terminal_valid": bool(row["terminal_valid"]),
                "premature_stop": decision.action == STOP_ACTION and not row["terminal_valid"],
                "pool_mode": _pool_mode(row.get("difficulty_tags") or ()),
                "valid_cardinality": len(valid),
                "router_metadata": dict(decision.metadata),
            }
        )
    terminal = [item for item in records if item["terminal_valid"]]
    nonterminal = [item for item in records if not item["terminal_valid"]]
    multi = [item for item in nonterminal if item["valid_cardinality"] >= 2]
    selected_stop = [item for item in records if item["selected_action"] == STOP_ACTION]
    true_stop = [item for item in selected_stop if item["terminal_valid"]]
    metrics = {
        "track": "A",
        "benchmark_version": bundle.metadata["version"],
        "router": router.name,
        "decision_count": len(records),
        "valid_action_accuracy": _rate(sum(item["valid"] for item in records), len(records)),
        "invalid_action_rate": _rate(sum(not item["valid"] for item in records), len(records)),
        "multi_valid_state_accuracy": _rate(sum(item["valid"] for item in multi), len(multi)),
        "premature_stop_rate": _rate(sum(item["premature_stop"] for item in records), len(nonterminal)),
        "stop_precision": _rate(len(true_stop), len(selected_stop)),
        "stop_recall": _rate(len(true_stop), len(terminal)),
        "by_family": _groups(records, "workflow_family", "valid"),
        "by_state_provenance": _groups(records, "state_provenance", "valid"),
        "by_pool_mode": _groups(records, "pool_mode", "valid"),
        "combined_weighted_score": None,
        "metric_policy": "report metrics separately; no subjective weighted utility",
    }
    return {"metrics": metrics, "records": records}


@dataclass(frozen=True)
class LiveActorResult:
    status: str
    contribution: str
    produced_outputs: tuple[str, ...]
    candidate_answer: str = ""
    tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    attempts: int = 1
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "SUCCESS"


class LiveActorExecutor(Protocol):
    calls: int
    total_tokens: int
    total_latency_ms: float
    model: str

    def execute(
        self,
        *,
        task: str,
        observation: Mapping[str, Any],
        role_profile: Mapping[str, Any],
    ) -> LiveActorResult: ...


class ModelLiveActorExecutor:
    """Call a real actor model and validate its observable output contract."""

    OUTPUT_SCHEMA = {
        "type": "object",
        "required": ["status", "contribution", "produced_outputs", "candidate_answer"],
        "properties": {
            "status": {
                "type": "string",
                "enum": ["SUCCESS", "NO_PROGRESS", "INVALID_OUTPUT"],
            },
            "contribution": {"type": "string"},
            "produced_outputs": {
                "type": "array",
                "items": {"type": "string"},
            },
            "candidate_answer": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    }

    def __init__(self, model: str, max_attempts: int = 2) -> None:
        # Imported only after the runner loads .env; model.api_config resolves
        # credentials at import time.
        from model.query_manager import query_manager

        self.query_manager = query_manager
        self.model = model
        self.max_attempts = max(1, int(max_attempts))
        self.calls = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0

    def execute(self, *, task, observation, role_profile) -> LiveActorResult:
        import time

        expected = tuple(map(str, role_profile.get("expected_outputs") or ()))
        expected_ids = tuple(
            output_id
            for output_id in (_canonical_output_id(item) for item in expected)
            if output_id
        )
        output_schema = {
            **self.OUTPUT_SCHEMA,
            "properties": {
                **self.OUTPUT_SCHEMA["properties"],
                "produced_outputs": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        **({"enum": list(expected_ids)} if expected_ids else {}),
                    },
                },
            },
        }
        prompt = {
            "task": task,
            "task_input": observation.get("task_input", {}),
            "current_observable_state": observation.get("state", {}),
            "execution_memory": observation.get("execution_memory", {}),
            "recent_events": observation.get("recent_events", []),
            "prior_live_contributions": observation.get(
                "prior_live_contributions", []
            ),
            "role_contract": {
                "role_name": role_profile.get("role_name"),
                "role_goal": role_profile.get("role_goal"),
                "supported_inputs": role_profile.get("supported_inputs", []),
                "expected_outputs": list(expected),
                "expected_output_ids": list(expected_ids),
            },
            "instructions": (
                "Execute this role now using only visible information. This workflow "
                "workflow is intentionally represented at the artifact-contract level: "
                "the visible state descriptions are authoritative inputs, and concrete "
                "entity values are not required unless the task explicitly supplies "
                "them. Produce a useful workflow artifact at that same abstraction "
                "level instead of refusing because names or raw documents are abstract. "
                "Return SUCCESS only when you produced a substantive contribution and "
                "name at least one exact machine-facing ID from expected_output_ids in "
                "produced_outputs. Do not invent an output type or return the artifact "
                "content itself as an output type. "
                "Return NO_PROGRESS only when a visible role precondition is actually "
                "missing. Put a short answer value in candidate_answer when your role "
                "derives one. A role whose expected output is a committed final artifact "
                "must return the final answer in candidate_answer. Otherwise it may be "
                "null. Do not claim that another hidden source, answer, or capability "
                "score exists."
            ),
            "output_schema": output_schema,
        }
        messages = [
            {
                "role": "system",
                "content": "You are a live workflow actor. Follow the role contract and output schema exactly.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        last_error = ""
        started = time.perf_counter()
        tokens = 0
        for attempt in range(1, self.max_attempts + 1):
            try:
                payload, used = self.query_manager.query_structured(
                    self.model,
                    messages,
                    output_schema,
                    schema_name="routingbench_v3_live_actor",
                )
                tokens += int(used or 0)
                status = str(payload.get("status", "INVALID_OUTPUT")).strip().upper()
                contribution = str(payload.get("contribution", "")).strip()
                raw_produced = tuple(map(str, payload.get("produced_outputs") or ()))
                produced = tuple(
                    output_id
                    for output_id in (
                        _canonical_output_id(item) for item in raw_produced
                    )
                    if output_id
                )
                candidate_answer = str(payload.get("candidate_answer") or "").strip()
                matched = (
                    bool(set(produced).intersection(expected_ids))
                    if expected_ids
                    else bool(produced)
                )
                refusal = any(
                    marker in contribution.casefold()
                    for marker in ("i cannot", "unable to", "do not have access")
                )
                validation_errors: list[str] = []
                if len(contribution) < 20:
                    validation_errors.append("contribution is not substantive")
                if not matched:
                    validation_errors.append(
                        "produced_outputs must contain one of: "
                        + ", ".join(expected_ids or ("<non-empty output ID>",))
                    )
                if refusal:
                    validation_errors.append("contribution contains a refusal")
                if (
                    "committed_final_artifact" in expected_ids
                    and not candidate_answer
                ):
                    validation_errors.append(
                        "candidate_answer is required for a committed final artifact"
                    )
                if status == "INVALID_OUTPUT":
                    validation_errors.append("actor returned INVALID_OUTPUT")
                if status == "SUCCESS" and validation_errors:
                    status = "INVALID_OUTPUT"
                if status == "INVALID_OUTPUT" and attempt < self.max_attempts:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The response violated the observable role contract: "
                                + "; ".join(validation_errors)
                                + ". Retry the same role once. Return only one JSON "
                                "object matching the schema, use produced_outputs IDs "
                                "exactly as listed in expected_output_ids, and do not "
                                "change the task or role."
                            ),
                        }
                    )
                    continue
                elapsed = (time.perf_counter() - started) * 1000.0
                self.calls += 1
                self.total_tokens += tokens
                self.total_latency_ms += elapsed
                return LiveActorResult(
                    status=status,
                    contribution=contribution,
                    produced_outputs=produced,
                    candidate_answer=candidate_answer,
                    tokens=tokens,
                    latency_ms=elapsed,
                    model=self.model,
                    attempts=attempt,
                    raw={
                        **payload,
                        "raw_produced_outputs": list(raw_produced),
                        "contract_validation_errors": validation_errors,
                    },
                )
            except Exception as error:  # provider/parser failures are actor failures
                last_error = f"{type(error).__name__}: {error}"
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous response could not be validated. Return only "
                            "one JSON object matching the schema."
                        ),
                    }
                )
        elapsed = (time.perf_counter() - started) * 1000.0
        self.calls += 1
        self.total_tokens += tokens
        self.total_latency_ms += elapsed
        return LiveActorResult(
            status="INVALID_OUTPUT",
            contribution=last_error,
            produced_outputs=(),
            candidate_answer="",
            tokens=tokens,
            latency_ms=elapsed,
            model=self.model,
            attempts=self.max_attempts,
            raw={"error": last_error},
        )


def _live_observation(
    env: RoutingEnvironmentV2, bundle: BenchmarkBundleV3
) -> dict[str, Any]:
    observation = env.observation()
    observation["budget"] = {
        "calls_remaining": env.remaining_steps,
        "reserved_terminal_stop": str(env.scenario.schema_version).startswith(
            "3.0-beta"
        ),
    }
    observation["agent_profiles"] = [
        {
            key: value
            for key, value in profile.items()
            if key not in {"cost_class", "action_cost", "backbone"}
        }
        for profile in observation["agent_profiles"]
    ]
    observation["constraints"].update(
        {
            "set_valued_ground_truth": True,
            "preferred_action_provided": False,
            "quality_rankings_provided": False,
            "execution_mode": "live_llm_actor",
        }
    )
    observation["benchmark_version"] = bundle.metadata["version"]
    return observation


def _record_live_actor_failure(
    env: RoutingEnvironmentV2, action: str, result: LiveActorResult
) -> StepEventV2:
    before = env.state.fingerprint()
    applicable = env.applicable_actions()
    env.state.steps_used += 1
    env.state.cost_used = round(env.state.cost_used + 1.0, 6)
    env.state.last_action = action
    env.state.last_status = (
        "EXECUTED_NO_PROGRESS"
        if result.status == "NO_PROGRESS"
        else "EXECUTED_INVALID_OUTPUT"
    )
    env.state.last_effect = (
        "The live actor executed but did not satisfy its observable output contract."
    )
    env.state.failure_count += 1
    env.state.actor_failure_count += 1
    after = env.state.fingerprint()
    event = StepEventV2(
        step=env.state.steps_used,
        action=action,
        status=str(env.state.last_status),
        applicable=True,
        valid=True,
        efficient=False,
        admissible=True,
        progress_action=False,
        applicable_before=applicable,
        valid_before=applicable,
        efficient_before=(),
        admissible_before=applicable,
        progress_actions_before=(),
        state_before=before,
        state_after=after,
        substantive_state_changed=False,
        redundant=any(
            item.action == action and item.state_before == before for item in env.history
        ),
        stagnant_repeat=any(
            item.action == action and not item.substantive_state_changed
            for item in env.history
        ),
        cost=1.0,
        effect=str(env.state.last_effect),
    )
    env.history.append(event)
    if env.state.steps_used >= env.scenario.max_steps:
        env.done = True
        env.success = False
        env.terminal_reason = "STEP_LIMIT"
    return event


def _step_after_live_success(
    env: RoutingEnvironmentV2, action: str
) -> StepEventV2:
    transition = env._transition_for(action)
    if transition is None or transition.status not in FAILURE_STATUSES:
        return env.step(action)
    alternatives = [
        item
        for item in env.scenario.transitions
        if item.action == action and item.status not in FAILURE_STATUSES
    ]
    if not alternatives:
        # The live actor succeeded, so a fixed scheduled failure may not override
        # it. Treat the call as an invalid environment contract instead.
        return _record_live_actor_failure(
            env,
            action,
            LiveActorResult(
                status="INVALID_OUTPUT",
                contribution="No live-success transition is defined for this action.",
                produced_outputs=(),
            ),
        )
    success = alternatives[0]
    replacement_transition = replace(
        transition,
        add_facts=success.add_facts,
        remove_facts=success.remove_facts,
        add_artifacts=success.add_artifacts,
        remove_artifacts=success.remove_artifacts,
        status="EXECUTED_SUCCESS",
        public_effect="The live actor satisfied the role contract and advanced the workflow state.",
    )
    original = env.scenario
    env.scenario = replace(
        original,
        transitions=tuple(
            replacement_transition if item.transition_id == transition.transition_id else item
            for item in original.transitions
        ),
    )
    try:
        return env.step(action)
    finally:
        env.scenario = original


def _normalize_answer(value: Any) -> str:
    text = str(value or "").casefold()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _token_f1(prediction: str, gold: str) -> float:
    predicted = _normalize_answer(prediction).split()
    expected = _normalize_answer(gold).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = Counter(predicted) & Counter(expected)
    shared = sum(overlap.values())
    if not shared:
        return 0.0
    precision = shared / len(predicted)
    recall = shared / len(expected)
    return 2 * precision * recall / (precision + recall)


def _choice_value(value: str) -> str:
    text = str(value or "").strip().upper()
    match = re.fullmatch(r"(?:FINAL\s+ANSWER\s*[:=]\s*)?[\(\[]?([A-J])[\)\]]?[\.!]?", text)
    return match.group(1) if match else text


def _score_end_task(
    prediction: str, evaluation: Mapping[str, Any] | None
) -> dict[str, Any]:
    if not evaluation:
        return {"correct": None, "metric": None}
    metric = str(evaluation.get("metric", "normalized_exact_match"))
    golds = [
        str(item)
        for item in [evaluation.get("answer"), *(evaluation.get("aliases") or [])]
        if item is not None and str(item).strip()
    ]
    if not golds:
        return {"correct": None, "metric": metric}
    if metric == "exact_choice":
        predicted = _choice_value(prediction)
        exact = max(float(predicted == _choice_value(gold)) for gold in golds)
        return {"correct": bool(exact), "metric": metric, "exact_match": exact}
    normalized = _normalize_answer(prediction)
    exact = max(float(normalized == _normalize_answer(gold)) for gold in golds)
    f1 = max(_token_f1(prediction, gold) for gold in golds)
    return {
        "correct": bool(exact),
        "metric": metric,
        "exact_match": exact,
        "token_f1": f1,
    }


def evaluate_track_b_live_v3(
    bundle: BenchmarkBundleV3,
    router: Router,
    actor: LiveActorExecutor,
    seed: int = 42,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    for index, scenario in enumerate(bundle.scenarios):
        router.reset(seed + index)
        env = RoutingEnvironmentV2(scenario, bundle.agents)
        episode_events: list[dict[str, Any]] = []
        router_error = False
        executor_error = False
        live_contributions: list[dict[str, Any]] = []
        committed_answer = ""
        while not env.done:
            observation = _live_observation(env, bundle)
            valid = env.admissible_actions()
            decision = router.select(observation, env.candidate_actions(), valid)
            actor_result: LiveActorResult | None = None
            selected_valid = decision.action in valid
            if decision.action == STOP_ACTION or not selected_valid:
                event = env.step(decision.action)
                router_error = router_error or not selected_valid
                failure_origin = "router" if not selected_valid else None
            else:
                profile = next(
                    item
                    for item in observation["agent_profiles"]
                    if item["agent_id"] == decision.action
                )
                actor_result = actor.execute(
                    task=scenario.task,
                    observation={
                        **observation,
                        "task_input": bundle.task_inputs.get(
                            scenario.scenario_id, {}
                        ),
                        "prior_live_contributions": list(live_contributions),
                    },
                    role_profile=profile,
                )
                if actor_result.success:
                    event = _step_after_live_success(env, decision.action)
                    if event.status in {
                        "EXECUTED_INVALID_OUTPUT",
                        "EXECUTED_NO_PROGRESS",
                    }:
                        executor_error = True
                        failure_origin = "executor"
                    else:
                        failure_origin = None
                        if env.goal_reached and actor_result.candidate_answer:
                            committed_answer = actor_result.candidate_answer
                else:
                    event = _record_live_actor_failure(env, decision.action, actor_result)
                    executor_error = True
                    failure_origin = "executor"
            event_row = event.to_dict()
            if actor_result is not None:
                live_contributions.append(
                    {
                        "step": event.step,
                        "agent_id": decision.action,
                        "role_name": profile.get("role_name"),
                        "status": actor_result.status,
                        "contribution": actor_result.contribution,
                        "candidate_answer": actor_result.candidate_answer,
                        "produced_outputs": list(actor_result.produced_outputs),
                    }
                )
            event_row.update(
                {
                    "scenario_id": scenario.scenario_id,
                    "workflow_family": scenario.family,
                    "selected_action_valid": selected_valid,
                    "failure_origin": failure_origin,
                    "router_metadata": dict(decision.metadata),
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
                            "raw_produced_outputs": list(
                                actor_result.raw.get(
                                    "raw_produced_outputs",
                                    actor_result.produced_outputs,
                                )
                            ),
                            "contract_validation_errors": list(
                                actor_result.raw.get(
                                    "contract_validation_errors", ()
                                )
                            ),
                        }
                        if actor_result is not None
                        else None
                    ),
                }
            )
            episode_events.append(event_row)
            all_events.append(event_row)
            if not env.done and not env.admissible_actions():
                env.done = True
                env.success = False
                env.terminal_reason = "UNRECOVERABLE_STATE"
        semantic = _score_end_task(
            committed_answer, bundle.evaluation.get(scenario.scenario_id)
        )
        semantic_correct = semantic.get("correct")
        if env.success and semantic_correct is False:
            executor_error = True
        if env.success and semantic_correct is not False:
            attribution = "success"
        elif router_error and executor_error:
            attribution = "mixed"
        elif router_error:
            attribution = "router"
        elif executor_error:
            attribution = "executor"
        else:
            attribution = "indeterminate"
        episodes.append(
            {
                "scenario_id": scenario.scenario_id,
                "provenance_id": f"prov-{scenario.scenario_id}",
                "workflow_family": scenario.family,
                "split": scenario.split,
                "success": env.success,
                "model_answer": committed_answer,
                "end_task_correct": semantic_correct,
                "semantic_metrics": semantic,
                "terminal_reason": env.terminal_reason,
                "failure_attribution": attribution,
                "steps": env.state.steps_used,
                "events": episode_events,
            }
        )

    non_stop = [item for item in all_events if item["action"] != STOP_ACTION]
    executed_actor_events = [
        item for item in non_stop if item.get("live_actor") is not None
    ]
    rejected_actions = [
        item for item in non_stop if item["status"] == "ACTION_REJECTED"
    ]
    actor_contract_failures = [
        item
        for item in executed_actor_events
        if item["status"] == "EXECUTED_INVALID_OUTPUT"
    ]
    actor_no_progress = [
        item
        for item in executed_actor_events
        if item["status"] == "EXECUTED_NO_PROGRESS"
    ]
    failed = [
        item
        for item in episodes
        if not item["success"] or item.get("end_task_correct") is False
    ]
    semantically_scored = [
        item for item in episodes if item.get("end_task_correct") is not None
    ]
    metrics = {
        "track": "B",
        "execution_mode": "live_llm_actor",
        "simulated_actor_outputs": False,
        "benchmark_version": bundle.metadata["version"],
        "router": router.name,
        "actor_model": actor.model,
        "episode_count": len(episodes),
        "goal_completion_rate": _rate(sum(item["success"] for item in episodes), len(episodes)),
        "end_task_accuracy": _rate(
            sum(bool(item["end_task_correct"]) for item in semantically_scored),
            len(semantically_scored),
        ),
        "end_task_scored_count": len(semantically_scored),
        "end_task_accuracy_note": (
            "Computed only for scenarios with a private evaluation sidecar; "
            "gold answers are never router or actor inputs."
            if semantically_scored
            else "No semantic evaluation sidecar is present; completion is contract-based."
        ),
        "valid_transition_rate": _rate(sum(item["selected_action_valid"] for item in all_events), len(all_events)),
        "invalid_action_rate": _rate(sum(not item["selected_action_valid"] for item in all_events), len(all_events)),
        "premature_stop_rate": _rate(sum(item["status"] == "PREMATURE_STOP" for item in all_events), len(episodes)),
        "router_caused_failure_rate": _rate(sum(item["failure_attribution"] == "router" for item in failed), len(episodes)),
        "executor_caused_failure_rate": _rate(sum(item["failure_attribution"] == "executor" for item in failed), len(episodes)),
        "mixed_failure_rate": _rate(sum(item["failure_attribution"] == "mixed" for item in failed), len(episodes)),
        "indeterminate_failure_rate": _rate(sum(item["failure_attribution"] == "indeterminate" for item in failed), len(episodes)),
        "action_rejection_count": len(rejected_actions),
        "action_rejection_rate": _rate(len(rejected_actions), len(non_stop)),
        "executed_actor_call_count": len(executed_actor_events),
        "actor_contract_failure_count": len(actor_contract_failures),
        "actor_contract_failure_rate": _rate(
            len(actor_contract_failures), len(executed_actor_events)
        ),
        "actor_no_progress_count": len(actor_no_progress),
        "actor_no_progress_rate": _rate(
            len(actor_no_progress), len(executed_actor_events)
        ),
        # Backward-compatible alias. The denominator is now actor executions,
        # not all routing decisions, so the name matches the measured event.
        "live_actor_invalid_output_rate": _rate(
            len(actor_contract_failures), len(executed_actor_events)
        ),
        "average_agent_calls": _rate(actor.calls, len(episodes)),
        "total_actor_tokens": actor.total_tokens,
        "total_actor_latency_ms": actor.total_latency_ms,
        "by_family": _groups(episodes, "workflow_family", "success"),
        "combined_weighted_score": None,
        "metric_policy": "routing, execution, completion, and attribution are reported separately",
    }
    return {"metrics": metrics, "episodes": episodes}
