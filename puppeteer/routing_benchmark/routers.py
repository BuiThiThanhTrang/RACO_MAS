from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from inference.policy.decision_model_client import SystemOneDecisionClient

from .schemas import STOP_ACTION


@dataclass(frozen=True)
class RoutingDecision:
    action: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Router(Protocol):
    name: str

    def reset(self, seed: int | None = None) -> None: ...

    def select(
        self,
        observation: Mapping[str, Any],
        candidate_actions: Sequence[str],
        oracle_valid_actions: Sequence[str] | None = None,
    ) -> RoutingDecision: ...


class RandomRouter:
    name = "random"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._random = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        self._random = random.Random(self.seed if seed is None else seed)

    def select(self, observation, candidate_actions, oracle_valid_actions=None):
        del observation, oracle_valid_actions
        return RoutingDecision(self._random.choice(tuple(candidate_actions)))


class FixedSequenceRouter:
    name = "fixed_sequence"

    def __init__(self, sequence: Sequence[str]):
        if not sequence:
            raise ValueError("FixedSequenceRouter requires a non-empty sequence")
        self.sequence = tuple(sequence)

    def reset(self, seed: int | None = None) -> None:
        del seed

    def select(self, observation, candidate_actions, oracle_valid_actions=None):
        del oracle_valid_actions
        step = int((observation.get("state") or {}).get("step_index", 0))
        action = self.sequence[min(step, len(self.sequence) - 1)]
        if action not in candidate_actions:
            action = STOP_ACTION if STOP_ACTION in candidate_actions else candidate_actions[0]
        return RoutingDecision(action)


class OracleRouter:
    """Upper-bound router. It is the only router allowed to inspect valid actions."""

    name = "oracle"

    def reset(self, seed: int | None = None) -> None:
        del seed

    def select(self, observation, candidate_actions, oracle_valid_actions=None):
        del observation, candidate_actions
        if not oracle_valid_actions:
            return RoutingDecision(STOP_ACTION, {"oracle_fallback": True})
        non_stop = [item for item in oracle_valid_actions if item != STOP_ACTION]
        return RoutingDecision(non_stop[0] if non_stop else STOP_ACTION)


class DecisionModelRouter:
    """Adapter for Jev/Clef-compatible OpenRouter Decisions endpoints."""

    name = "decision_model"

    ORCHESTRATION_PROFILES = frozenset({"naive", "structured"})
    STRUCTURED_PROMPT_VERSION = "routingbench_structured_v1"
    NAIVE_PROMPT_VERSION = "routingbench_naive_v1"
    STRUCTURED_HARNESS_VERSION = "observable_rejection_guard_v1"
    NAIVE_HARNESS_VERSION = "none"

    def __init__(
        self,
        decision_config: Mapping[str, Any],
        orchestration_profile: str = "structured",
        observable_rejection_guard_enabled: bool = True,
    ):
        orchestration_profile = str(orchestration_profile).strip().casefold()
        if orchestration_profile not in self.ORCHESTRATION_PROFILES:
            raise ValueError(
                "orchestration_profile must be one of "
                f"{sorted(self.ORCHESTRATION_PROFILES)}"
            )
        self.client = SystemOneDecisionClient(decision_config)
        self.orchestration_profile = orchestration_profile
        self.observable_rejection_guard_enabled = bool(
            observable_rejection_guard_enabled
        )
        self.total_input_tokens = 0
        self.total_latency_ms = 0.0
        self.calls = 0
        self.guard_activations = 0
        self.blocked_action_count = 0

    def reset(self, seed: int | None = None) -> None:
        del seed

    @staticmethod
    def _criteria(
        observation: Mapping[str, Any],
        candidate_actions: Sequence[str],
        orchestration_profile: str = "structured",
    ):
        profiles = {
            str(item.get("agent_id")): item
            for item in observation.get("agent_profiles") or ()
        }
        criteria: dict[str, str] = {}
        for action in candidate_actions:
            if action == STOP_ACTION:
                criteria[action] = (
                    "Stop only when the visible state contains a committed final artifact "
                    "that completes the task."
                )
                continue
            profile = profiles.get(action, {})
            if orchestration_profile == "naive":
                criteria[action] = (
                    f"{profile.get('role_name', action)}: "
                    f"{profile.get('role_goal', '')}"
                ).strip()
            else:
                criteria[action] = (
                    f"{profile.get('role_name', action)}: {profile.get('role_goal', '')} "
                    f"Use when: {'; '.join(profile.get('use_when') or ())}. "
                    f"Avoid when: {'; '.join(profile.get('avoid_when') or ())}. "
                    f"Supported inputs: {'; '.join(profile.get('supported_inputs') or ())}. "
                    f"Expected outputs: {'; '.join(profile.get('expected_outputs') or ())}."
                ).strip()
        return criteria

    @classmethod
    def _instructions(
        cls, orchestration_profile: str, beta: bool, benchmark_version: str = ""
    ) -> str:
        if orchestration_profile == "naive":
            return (
                "Select exactly one next agent using the task, current state, and role "
                "descriptions. Select STOP only when the task is complete."
            )
        if str(benchmark_version).startswith(("3.", "4.")):
            return (
                "Select exactly one next agent using only observable task state, "
                "execution memory, recent events, remaining call budget, and the "
                "qualitative role contracts. Select an agent only when its visible "
                "input preconditions are satisfied. Several agents may be equally "
                "valid; do not invent a hidden preference, capability score, backbone "
                "ranking, or source-location hint. ACTION_REJECTED is a pre-execution "
                "routing error. EXECUTED_INEFFICIENT means an applicable actor ran "
                "but added no useful state and is not by itself a task-executor "
                "failure. EXECUTED_INVALID_OUTPUT, EXECUTED_NO_PROGRESS, and "
                "EXECUTED_RETRYABLE_FAILURE are post-execution actor outcomes. Retry or "
                "recover only when the visible state justifies it. Select STOP only "
                "when a committed final artifact visibly completes the task."
            )
        if beta:
            return (
                "Select exactly one next agent using only the observable task, current "
                "state, compact execution memory, recent events, remaining budget, and "
                "qualitative role contracts. Choose an action that is reasonable now; do "
                "not guess hidden simulator outcomes or hidden source locations. Apply this "
                "failure taxonomy before choosing: ACTION_REJECTED is a pre-execution "
                "routing rejection, so the actor did not run and Recovery Strategist must "
                "not be selected solely because of that rejection. EXECUTED_NO_PROGRESS, "
                "EXECUTED_RETRYABLE_FAILURE, and EXECUTED_INVALID_OUTPUT are post-execution "
                "actor outcomes; use recovery only when the failed method must be materially "
                "revised. After recovery succeeds, retry the still-unresolved operation "
                "instead of calling recovery again. Before composing or finalizing, verify "
                "from visible facts and artifacts that every required branch is complete. "
                "When several actions are reasonable, prefer visible progress and lower "
                "resource use. Do not infer or rank hidden backbone quality. Select STOP only "
                "when a committed final artifact visibly completes the task."
            )
        return (
            "Select exactly one next agent from the qualitative role contracts and "
            "current task state. Choose the action with the greatest state-specific "
            "marginal value. Do not infer or rank hidden backbone quality. Select STOP "
            "only when the task is complete."
        )

    @staticmethod
    def _observable_rejection_guard(
        observation: Mapping[str, Any], candidate_actions: Sequence[str]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Mask rejected actions until a substantive public state change occurs.

        The guard reads only ``recent_events`` supplied to the router. It never reads
        applicable, admissible, progress, or oracle-efficient action labels.
        """

        blocked: set[str] = set()
        for event in reversed(tuple(observation.get("recent_events") or ())):
            if str(event.get("semantic_delta", "")).casefold() == "state_changed":
                break
            if str(event.get("status", "")) in {"ACTION_REJECTED", "NOT_APPLICABLE"}:
                action = str(event.get("agent_id", "")).strip()
                if action and action != STOP_ACTION:
                    blocked.add(action)
        filtered = tuple(action for action in candidate_actions if action not in blocked)
        if not filtered:
            return tuple(candidate_actions), ()
        return filtered, tuple(
            action for action in candidate_actions if action in blocked
        )

    def select(self, observation, candidate_actions, oracle_valid_actions=None):
        del oracle_valid_actions
        question_name = "next_agent"
        benchmark_version = str(observation.get("benchmark_version", ""))
        beta = benchmark_version.startswith(("2.0-beta", "3.", "4."))
        selected_candidates = tuple(candidate_actions)
        blocked_actions: tuple[str, ...] = ()
        if (
            self.orchestration_profile == "structured"
            and beta
            and self.observable_rejection_guard_enabled
        ):
            selected_candidates, blocked_actions = self._observable_rejection_guard(
                observation, selected_candidates
            )
            if blocked_actions:
                self.guard_activations += 1
                self.blocked_action_count += len(blocked_actions)
        decision_observation = dict(observation)
        if blocked_actions:
            constraints = dict(decision_observation.get("constraints") or {})
            constraints["allowed_actions"] = list(selected_candidates)
            constraints["observable_guard"] = {
                "rule": "do_not_repeat_pre_execution_rejection_without_state_change",
                "blocked_actions": list(blocked_actions),
            }
            decision_observation["constraints"] = constraints
        questions = {
            question_name: {
                "type": "choice",
                "instructions": self._instructions(
                    self.orchestration_profile, beta, benchmark_version
                ),
                "criteria": self._criteria(
                    decision_observation,
                    selected_candidates,
                    self.orchestration_profile,
                ),
            }
        }
        result = self.client.decide(
            decision_observation, questions, question_name
        )
        self.calls += 1
        self.total_input_tokens += result.input_tokens
        self.total_latency_ms += result.latency_ms
        structured = self.orchestration_profile == "structured"
        return RoutingDecision(
            result.choice,
            {
                "confidence": result.confidence,
                "probabilities": dict(result.probabilities),
                "input_tokens": result.input_tokens,
                "latency_ms": result.latency_ms,
                "model": result.model,
                "model_version": result.model_version,
                "orchestration_profile": self.orchestration_profile,
                "prompt_version": (
                    self.STRUCTURED_PROMPT_VERSION
                    if structured
                    else self.NAIVE_PROMPT_VERSION
                ),
                "role_card_mode": "full_contract" if structured else "name_and_goal",
                "harness_version": (
                    self.STRUCTURED_HARNESS_VERSION
                    if structured and self.observable_rejection_guard_enabled
                    else "guard_disabled_diagnostic"
                    if structured
                    else self.NAIVE_HARNESS_VERSION
                ),
                "observable_rejection_guard_enabled": self.observable_rejection_guard_enabled,
                "guard_blocked_actions": list(blocked_actions),
            },
        )


class StructuredLLMRouter:
    """Structured-output LLM router for models exposed through chat completion.

    This adapter intentionally mirrors :class:`DecisionModelRouter` so Jev and
    general LLM orchestrators receive the same public observation, role-card
    treatment, and observable rejection guard.  It never receives oracle valid
    actions.
    """

    name = "structured_llm"

    def __init__(
        self,
        model: str,
        orchestration_profile: str = "structured",
        reasoning_effort: str = "low",
        max_repair_attempts: int = 1,
        observable_rejection_guard_enabled: bool = True,
    ) -> None:
        profile = str(orchestration_profile).strip().casefold()
        if profile not in DecisionModelRouter.ORCHESTRATION_PROFILES:
            raise ValueError(
                "orchestration_profile must be one of "
                f"{sorted(DecisionModelRouter.ORCHESTRATION_PROFILES)}"
            )
        # Delay this import until after the V3 runner has loaded .env.  The
        # model registry creates provider clients at import time.
        from model.query_manager import query_manager

        self.query_manager = query_manager
        self.model = str(model).strip()
        if not self.model:
            raise ValueError("StructuredLLMRouter requires a model")
        self.orchestration_profile = profile
        self.reasoning_effort = str(reasoning_effort or "low")
        self.max_repair_attempts = max(0, int(max_repair_attempts))
        self.observable_rejection_guard_enabled = bool(
            observable_rejection_guard_enabled
        )
        self.calls = 0
        self.total_input_tokens = 0
        self.total_latency_ms = 0.0
        self.guard_activations = 0
        self.blocked_action_count = 0

    def reset(self, seed: int | None = None) -> None:
        del seed

    def select(
        self,
        observation: Mapping[str, Any],
        candidate_actions: Sequence[str],
        oracle_valid_actions: Sequence[str] | None = None,
    ) -> RoutingDecision:
        del oracle_valid_actions
        benchmark_version = str(observation.get("benchmark_version", ""))
        beta = benchmark_version.startswith(("2.0-beta", "3.", "4."))
        selected_candidates = tuple(map(str, candidate_actions))
        blocked_actions: tuple[str, ...] = ()
        if (
            self.orchestration_profile == "structured"
            and beta
            and self.observable_rejection_guard_enabled
        ):
            selected_candidates, blocked_actions = (
                DecisionModelRouter._observable_rejection_guard(
                    observation, selected_candidates
                )
            )
            if blocked_actions:
                self.guard_activations += 1
                self.blocked_action_count += len(blocked_actions)

        decision_observation = dict(observation)
        if blocked_actions:
            constraints = dict(decision_observation.get("constraints") or {})
            constraints["allowed_actions"] = list(selected_candidates)
            constraints["observable_guard"] = {
                "rule": "do_not_repeat_pre_execution_rejection_without_state_change",
                "blocked_actions": list(blocked_actions),
            }
            decision_observation["constraints"] = constraints

        criteria = DecisionModelRouter._criteria(
            decision_observation,
            selected_candidates,
            self.orchestration_profile,
        )
        instructions = DecisionModelRouter._instructions(
            self.orchestration_profile,
            beta,
            benchmark_version,
        )
        schema = {
            "type": "object",
            "required": ["action", "rationale"],
            "properties": {
                "action": {"type": "string", "enum": list(selected_candidates)},
                "rationale": {"type": "string"},
            },
            "additionalProperties": False,
        }
        payload = {
            "instructions": instructions,
            "observation": decision_observation,
            "candidate_contracts": criteria,
            "output_schema": schema,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a state-aware multi-agent router. Select exactly one "
                    "listed action. Use only the supplied observable state and role "
                    "contracts. Return one JSON object matching the schema."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        started = time.perf_counter()
        last_error: Exception | None = None
        used_tokens = 0
        result: Mapping[str, Any] | None = None
        for _ in range(self.max_repair_attempts + 1):
            try:
                result, used = self.query_manager.query_structured(
                    self.model,
                    messages,
                    schema,
                    schema_name="routingbench_next_agent",
                    reasoning_effort=self.reasoning_effort,
                )
                used_tokens += int(used or 0)
                action = str(result.get("action", "")).strip()
                if action not in selected_candidates:
                    raise ValueError("LLM router returned an action outside the candidate set")
                break
            except Exception as error:
                last_error = error
                result = None
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous output was invalid. Return only a corrected "
                            "JSON object whose action is one of the listed candidates."
                        ),
                    }
                )
        if result is None:
            raise RuntimeError(
                f"Structured LLM router failed after repair attempts: {last_error}"
            ) from last_error

        latency_ms = (time.perf_counter() - started) * 1000.0
        self.calls += 1
        self.total_input_tokens += used_tokens
        self.total_latency_ms += latency_ms
        structured = self.orchestration_profile == "structured"
        return RoutingDecision(
            str(result["action"]),
            {
                "confidence": None,
                "probabilities": {},
                "input_tokens": used_tokens,
                "latency_ms": latency_ms,
                "model": self.model,
                "model_version": None,
                "orchestration_profile": self.orchestration_profile,
                "prompt_version": (
                    DecisionModelRouter.STRUCTURED_PROMPT_VERSION
                    if structured
                    else DecisionModelRouter.NAIVE_PROMPT_VERSION
                ),
                "role_card_mode": "full_contract" if structured else "name_and_goal",
                "harness_version": (
                    DecisionModelRouter.STRUCTURED_HARNESS_VERSION
                    if structured and self.observable_rejection_guard_enabled
                    else "guard_disabled_diagnostic"
                    if structured
                    else DecisionModelRouter.NAIVE_HARNESS_VERSION
                ),
                "observable_rejection_guard_enabled": self.observable_rejection_guard_enabled,
                "guard_blocked_actions": list(blocked_actions),
                "rationale": str(result.get("rationale", "")),
            },
        )
