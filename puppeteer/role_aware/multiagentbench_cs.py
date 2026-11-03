"""Router collaboration scoring with an optional MuSiQue semantic verdict.

The combined contract deliberately keeps two independent judgements in one LLM
call: collaboration is scored only from the routing trace, while answer
equivalence is decided only from the already committed prediction and accepted
answers.  The judge never repairs or replaces the committed answer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from model.query_manager import query_manager


PROMPT_VERSION = "router_cs_v1"
COMBINED_PROMPT_VERSION = "router_cs_semantic_v2"

ROUTER_CS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "routing_planning_score",
        "state_handoff_communication_score",
        "planning_rationale",
        "communication_rationale",
        "failure_tags",
        "critical_decision_ids",
    ],
    "properties": {
        "routing_planning_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "state_handoff_communication_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 5,
        },
        "planning_rationale": {"type": "string"},
        "communication_rationale": {"type": "string"},
        "failure_tags": {
            "type": "array",
            "items": {"type": "string"},
        },
        "critical_decision_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

ROUTER_CS_SEMANTIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        *ROUTER_CS_SCHEMA["required"],
        "answer_verdict",
        "answer_confidence",
        "answer_rationale",
    ],
    "properties": {
        **ROUTER_CS_SCHEMA["properties"],
        "answer_verdict": {
            "type": "string",
            "enum": ["EQUIVALENT", "NOT_EQUIVALENT", "UNCERTAIN"],
        },
        "answer_confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "answer_rationale": {"type": "string"},
    },
}

SYSTEM_PROMPT = """You are an impartial evaluator of a multi-agent ROUTER.
Score only collaboration quality visible in the trace. Do not score factual
correctness of the final answer and do not reward a run merely because its final
answer looks plausible.

Routing Planning Score (1-5):
1 = incoherent or harmful routing; 2 = major omissions/redundancy; 3 = reasonable
but generic sequence; 4 = strong state-adaptive choices with little waste; 5 =
excellent minimal routing that reacts precisely to progress, conflict, or failure.

State-Handoff Communication Score (0-5):
0 = no cross-role handoff occurred; 1 = the next role effectively ignores prior
state; 2 = partial/ambiguous transfer; 3 = usable transfer; 4 = clear transfer that
advances the task; 5 = precise, lossless state transfer with the next role building
directly on unresolved dependencies and evidence.

Judge the router, not the actor backbone. A weak actor output should reduce the
router score only when the router fails to react appropriately. Return only the
requested JSON object."""

COMBINED_SYSTEM_PROMPT = """You perform two strictly separated evaluations for
a completed multi-agent MuSiQue run in one structured response.

TASK A — ROUTER COLLABORATION
Score only collaboration quality visible in collaboration_trace. Do not use the
committed prediction, accepted answers, or factual correctness to raise or lower
either collaboration score.

Routing Planning Score (1-5):
1 = incoherent or harmful routing; 2 = major omissions/redundancy; 3 = reasonable
but generic sequence; 4 = strong state-adaptive choices with little waste; 5 =
excellent minimal routing that reacts precisely to progress, conflict, or failure.

State-Handoff Communication Score (0-5):
0 = no cross-role handoff occurred; 1 = the next role effectively ignores prior
state; 2 = partial/ambiguous transfer; 3 = usable transfer; 4 = clear transfer that
advances the task; 5 = precise, lossless state transfer with the next role building
directly on unresolved dependencies and evidence.

Judge the router, not the actor backbone. A weak actor output reduces a router
score only when the router fails to react appropriately.

TASK B — COMMITTED ANSWER EQUIVALENCE
Use only answer_evaluation. Do not solve the question, search the trace for a
better answer, repair the prediction, or copy an intermediate answer. Decide
whether committed_prediction and at least one accepted answer denote the same
answer to the original question. Harmless articles, punctuation, acronym
expansions, parenthetical aliases, and non-contradictory qualifiers are
equivalent. Different entities, contradictory qualifiers, incomplete lists, or
answers to another relation are not equivalent. Use UNCERTAIN when equivalence
cannot be established reliably.

Return only the requested JSON object."""


def _clip(value: Any, limit: int = 2400) -> Any:
    if isinstance(value, str):
        compact = " ".join(value.split())
        return compact if len(compact) <= limit else compact[: limit - 1] + "…"
    if isinstance(value, Mapping):
        return {str(key): _clip(item, limit) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clip(item, limit) for item in value]
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
        if isinstance(value, dict):
            records.append(value)
    return records


def build_router_cs_trace(
    candidates: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a bounded, answer-blind judge input from one audited attempt."""

    routing_decisions = []
    for event in events:
        if event.get("event_type") != "routing_decision":
            continue
        routing_decisions.append(
            _clip(
                {
                    "decision_id": event.get("decision_id"),
                    "path_uid": event.get("path_uid"),
                    "selected": event.get("selected") or [],
                    "assignments": event.get("assignments") or [],
                    "remaining_depth": event.get("remaining_depth"),
                    "allow_stop": event.get("allow_stop"),
                    "fallback": event.get("fallback"),
                    "routing_state": event.get("routing_state") or {},
                    "eligible_candidate_ids": event.get("eligible_candidate_ids") or [],
                    "masked_candidates": event.get("masked_candidates") or [],
                    "decision_model": event.get("decision_model") or {},
                },
                1800,
            )
        )

    paths = []
    handoffs = []
    for path in candidates.get("paths") or []:
        compact_steps = []
        steps = path.get("steps") or []
        for index, step in enumerate(steps):
            role = str(step.get("agent") or "")
            result = step.get("result") or {}
            output = result.get("step_data") if isinstance(result, Mapping) else result
            compact_steps.append(
                {
                    "step": index + 1,
                    "role": role,
                    "action": (step.get("action") or {}).get("action"),
                    "success": step.get("success"),
                    "output": _clip(output, 1600),
                }
            )
            if index == 0:
                continue
            previous = steps[index - 1]
            previous_role = str(previous.get("agent") or "")
            if previous_role == role:
                continue
            previous_result = previous.get("result") or {}
            previous_output = (
                previous_result.get("step_data")
                if isinstance(previous_result, Mapping)
                else previous_result
            )
            handoffs.append(
                {
                    "path_uid": path.get("path_uid"),
                    "from_role": previous_role,
                    "to_role": role,
                    "sender_output": _clip(previous_output, 1800),
                    "receiver_output": _clip(output, 1800),
                }
            )
        paths.append(
            {
                "path_uid": path.get("path_uid"),
                "stop_reason": path.get("stop_reason"),
                "steps": compact_steps,
            }
        )

    return {
        "prompt_version": PROMPT_VERSION,
        "task_id": candidates.get("task_id"),
        "task": _clip(candidates.get("question") or "", 7000),
        "routing_decisions": routing_decisions,
        "paths": paths,
        "cross_role_handoffs": handoffs,
        "handoff_count": len(handoffs),
        "evaluation_instruction": (
            "Evaluate routing and state handoffs only; do not use final-answer "
            "correctness as a scoring signal."
        ),
    }


def build_router_cs_semantic_trace(
    candidates: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
    *,
    prediction: Any,
    accepted_answers: Iterable[Any],
    question: Any = "",
) -> dict[str, Any]:
    """Build a combined trace without allowing the judge to change prediction."""

    collaboration_trace = build_router_cs_trace(candidates, events)
    collaboration_trace["prompt_version"] = COMBINED_PROMPT_VERSION
    answers = [
        str(value).strip()
        for value in accepted_answers
        if str(value or "").strip()
    ]
    return {
        "prompt_version": COMBINED_PROMPT_VERSION,
        "collaboration_trace": collaboration_trace,
        "answer_evaluation": {
            "original_question": str(question or "").strip(),
            "committed_prediction": str(prediction or "").strip(),
            "accepted_answers": answers,
        },
        "separation_rule": (
            "Collaboration scores must ignore answer_evaluation. The answer verdict "
            "must judge only the committed prediction and must not select another "
            "answer from collaboration_trace."
        ),
    }


def validate_router_cs_payload(
    payload: Mapping[str, Any], *, has_handoffs: bool
) -> dict[str, Any]:
    planning = int(payload["routing_planning_score"])
    communication = int(payload["state_handoff_communication_score"])
    if not 1 <= planning <= 5:
        raise ValueError("routing_planning_score must be between 1 and 5")
    if not 0 <= communication <= 5:
        raise ValueError("state_handoff_communication_score must be between 0 and 5")
    if not has_handoffs:
        communication = 0
    raw = (planning + communication) / 2.0
    return {
        "prompt_version": PROMPT_VERSION,
        "routing_planning_score": planning,
        "state_handoff_communication_score": communication,
        "collaboration_score_raw": raw,
        "collaboration_score_100": raw * 20.0,
        "planning_rationale": str(payload.get("planning_rationale") or "").strip(),
        "communication_rationale": str(
            payload.get("communication_rationale") or ""
        ).strip(),
        "failure_tags": [str(value) for value in payload.get("failure_tags") or []],
        "critical_decision_ids": [
            str(value) for value in payload.get("critical_decision_ids") or []
        ],
    }


def validate_router_cs_semantic_payload(
    payload: Mapping[str, Any],
    *,
    has_handoffs: bool,
    official_success: bool = False,
) -> dict[str, Any]:
    result = validate_router_cs_payload(payload, has_handoffs=has_handoffs)
    verdict = str(payload.get("answer_verdict") or "").strip().upper()
    if verdict not in {"EQUIVALENT", "NOT_EQUIVALENT", "UNCERTAIN"}:
        raise ValueError("answer_verdict must be EQUIVALENT, NOT_EQUIVALENT, or UNCERTAIN")
    try:
        confidence = float(payload.get("answer_confidence"))
    except (TypeError, ValueError) as error:
        raise ValueError("answer_confidence must be numeric") from error
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("answer_confidence must be between 0 and 1")
    rationale = " ".join(str(payload.get("answer_rationale") or "").split())
    if not rationale:
        raise ValueError("answer_rationale must be non-empty")
    equivalent = verdict == "EQUIVALENT"
    effective = bool(official_success or equivalent)
    if verdict == "UNCERTAIN":
        effective = bool(official_success)
    result.update(
        {
            "prompt_version": COMBINED_PROMPT_VERSION,
            "answer_verdict": verdict,
            "answer_equivalent": equivalent,
            "semantic_correct": effective,
            "answer_confidence": confidence,
            "answer_rationale": rationale,
        }
    )
    return result


def judge_router_cs(
    trace: Mapping[str, Any],
    *,
    model: str,
    reasoning_effort: str = "low",
    query_func: Callable[..., tuple[dict[str, Any], int]] | None = None,
) -> tuple[dict[str, Any], int]:
    query = query_func or query_manager.query_structured
    payload, tokens = query(
        model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(trace, ensure_ascii=False),
            },
        ],
        ROUTER_CS_SCHEMA,
        schema_name="router_collaboration_score",
        reasoning_effort=reasoning_effort,
    )
    result = validate_router_cs_payload(
        payload,
        has_handoffs=bool(trace.get("handoff_count")),
    )
    result.update({"judge_model": model, "judge_tokens": int(tokens)})
    return result, int(tokens)


def judge_router_cs_semantic(
    trace: Mapping[str, Any],
    *,
    model: str,
    reasoning_effort: str = "low",
    max_repair_attempts: int = 1,
    official_success: bool = False,
    query_func: Callable[..., tuple[dict[str, Any], int]] | None = None,
) -> tuple[dict[str, Any], int]:
    query = query_func or query_manager.query_structured
    messages = [
        {"role": "system", "content": COMBINED_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(trace, ensure_ascii=False)},
    ]
    total_tokens = 0
    repairs = 0
    result = None
    last_error: ValueError | None = None
    collaboration_trace = trace.get("collaboration_trace") or {}
    for attempt in range(max(0, int(max_repair_attempts)) + 1):
        payload, tokens = query(
            model,
            messages,
            ROUTER_CS_SEMANTIC_SCHEMA,
            schema_name="router_collaboration_and_semantic_outcome",
            reasoning_effort=reasoning_effort,
        )
        total_tokens += int(tokens)
        try:
            result = validate_router_cs_semantic_payload(
                payload,
                has_handoffs=bool(collaboration_trace.get("handoff_count")),
                official_success=bool(official_success),
            )
            break
        except ValueError as error:
            last_error = error
            if attempt >= max_repair_attempts:
                raise
            repairs += 1
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The structured judgement was invalid: {error}. Return a "
                        "corrected object without changing the trace, committed "
                        "prediction, or accepted answers."
                    ),
                }
            )
    if result is None:
        raise last_error or ValueError("Combined router and semantic judging failed")
    result.update(
        {
            "judge_model": model,
            "judge_tokens": int(total_tokens),
            "repair_attempts": repairs,
        }
    )
    return result, int(total_tokens)
