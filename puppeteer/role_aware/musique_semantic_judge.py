"""Post-task semantic equivalence judging for MuSiQue answers.

The judge is evaluation-only: it receives the committed prediction and accepted
gold answers after routing, reasoning, and aggregation have finished.  Official
MuSiQue EM/F1 remain untouched; the semantic verdict is an optional learning
signal for route experience, profile evidence, or policy reward.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Mapping

from model.query_manager import query_manager


PROMPT_VERSION = "musique_semantic_equivalence_v1"

SEMANTIC_EQUIVALENCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "confidence", "reason"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["EQUIVALENT", "NOT_EQUIVALENT", "UNCERTAIN"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
    },
}

SYSTEM_PROMPT = """You judge semantic equivalence for a completed MuSiQue QA
task. Do not solve the question and do not repair either answer. Decide only
whether the committed prediction and at least one accepted answer denote the
same answer to the supplied question.

Treat harmless articles, punctuation, abbreviations, acronym expansions,
parenthetical aliases, and non-contradictory disambiguating qualifiers as
equivalent. Treat a different entity, contradictory qualifier, incomplete list,
or answer to a different relation as not equivalent. Use UNCERTAIN when the
strings and question do not establish equivalence reliably. Keep the reason
short and return only the requested JSON object."""


def _clean_answers(values: Iterable[Any]) -> list[str]:
    return [str(value).strip() for value in values if str(value or "").strip()]


def build_semantic_judge_input(
    prediction: Any,
    accepted_answers: Iterable[Any],
    *,
    question: Any = "",
) -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "question": str(question or "").strip(),
        "prediction": str(prediction or "").strip(),
        "accepted_answers": _clean_answers(accepted_answers),
    }


def validate_semantic_verdict(payload: Mapping[str, Any]) -> dict[str, Any]:
    verdict = str(payload.get("verdict") or "").strip().upper()
    if verdict not in {"EQUIVALENT", "NOT_EQUIVALENT", "UNCERTAIN"}:
        raise ValueError("Semantic judge returned an invalid verdict")
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError) as error:
        raise ValueError("Semantic judge confidence must be numeric") from error
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Semantic judge confidence must be between 0 and 1")
    reason = " ".join(str(payload.get("reason") or "").split())
    if not reason:
        raise ValueError("Semantic judge must provide a short reason")
    return {
        "verdict": verdict,
        "equivalent": verdict == "EQUIVALENT",
        "confidence": confidence,
        "reason": reason,
    }


def judge_semantic_equivalence(
    prediction: Any,
    accepted_answers: Iterable[Any],
    *,
    question: Any = "",
    model: str,
    reasoning_effort: str = "low",
    max_repair_attempts: int = 1,
    query_func: Callable[..., tuple[dict[str, Any], int]] | None = None,
) -> tuple[dict[str, Any], int]:
    judge_input = build_semantic_judge_input(
        prediction, accepted_answers, question=question
    )
    if not judge_input["prediction"] or not judge_input["accepted_answers"]:
        result = {
            "verdict": "NOT_EQUIVALENT",
            "equivalent": False,
            "confidence": 1.0,
            "reason": "The prediction or accepted-answer set is empty.",
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "tokens": 0,
            "repair_attempts": 0,
        }
        return result, 0

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(judge_input, ensure_ascii=False)},
    ]
    query = query_func or query_manager.query_structured
    total_tokens = 0
    result = None
    last_error: ValueError | None = None
    repairs = 0
    for attempt in range(max(0, int(max_repair_attempts)) + 1):
        payload, tokens = query(
            model,
            messages,
            SEMANTIC_EQUIVALENCE_SCHEMA,
            schema_name="musique_semantic_equivalence",
            reasoning_effort=reasoning_effort,
        )
        total_tokens += int(tokens)
        try:
            result = validate_semantic_verdict(payload)
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
                        f"The verdict object was invalid: {error}. Return a corrected "
                        "object without changing the prediction or accepted answers."
                    ),
                }
            )
    if result is None:
        raise last_error or ValueError("Semantic equivalence judging failed")
    result.update(
        {
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "tokens": total_tokens,
            "repair_attempts": repairs,
        }
    )
    return result, total_tokens
