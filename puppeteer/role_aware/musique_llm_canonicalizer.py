"""Gold-blind LLM answer extraction and path selection for MuSiQue logs."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping

from model.query_manager import query_manager


PROMPT_VERSION = "musique_llm_canonicalizer_v1"
RUNTIME_PROMPT_VERSION = "musique_runtime_extractor_v1"

ANSWER_EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status",
        "source_path_index",
        "answer",
        "answer_quote",
    ],
    "properties": {
        "status": {"type": "string", "enum": ["ANSWER", "NO_ANSWER"]},
        "source_path_index": {"type": "integer", "minimum": 0},
        "answer": {"type": "string"},
        "answer_quote": {"type": "string"},
    },
}

RUNTIME_EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "answer", "answer_quote"],
    "properties": {
        "status": {
            "type": "string",
            "enum": ["FINAL", "CANDIDATE", "NO_ANSWER"],
        },
        "answer": {"type": "string"},
        "answer_quote": {"type": "string"},
    },
}

SYSTEM_PROMPT = """You are a conservative MuSiQue answer canonicalizer and
candidate selector. You are not allowed to solve the question yourself or add a
fact absent from the supplied path-final outputs.

Choose the path-final output that most directly and defensibly answers the
original question. Extract the shortest sufficient answer span from that output.
Remove explanatory framing, duplicated entities, articles, citations, and
location qualifiers only by selecting a shorter contiguous span already present
in the selected output. Do not copy an answer from the question/context.

Treat null, null followed by punctuation, empty JSON fields, code-fence language
tags, NEEDS_EVIDENCE, CONFLICT, refusals, and statements that the answer cannot be
determined as NO_ANSWER unless another path-final output contains an explicit
usable answer.

For ANSWER, answer_quote must be an exact contiguous quote from the selected raw
output and answer must equal that quote after trimming surrounding whitespace.
For NO_ANSWER, use source_path_index 0 and empty strings for answer and
answer_quote. Return only the requested JSON object."""

RUNTIME_SYSTEM_PROMPT = """You are a conservative, gold-blind MuSiQue answer
extractor used while a multi-agent task is still running. You never see the gold
answer and must not solve the question, correct an answer, or add a fact absent
from the supplied agent output.

Return FINAL only when the output explicitly presents a final answer to the
original question. Return CANDIDATE when it presents a possible answer but does
not explicitly commit it as final. Return NO_ANSWER for evidence-only text,
plans, intermediate entities, null values, refusals, unresolved conflicts, or
missing information. For FINAL or CANDIDATE, answer_quote must be an exact
contiguous span from agent_output and answer must equal that quote after
whitespace normalization. Return only the requested JSON object."""


def original_question(task_text: Any) -> str:
    text = str(task_text or "").strip()
    match = re.search(
        r"(?is)(?:^|\n)Question:\s*(.*?)\s*\n\s*Candidate paragraphs:",
        text,
    )
    return match.group(1).strip() if match else text


def _path_final_output(path: Mapping[str, Any]) -> str:
    steps = list(path.get("steps") or ())
    if not steps:
        return ""
    result = steps[-1].get("result") or {}
    if isinstance(result, Mapping):
        return json.dumps(result, ensure_ascii=False, default=str)
    return str(result)


def build_llm_canonicalization_input(
    snapshot: Mapping[str, Any], *, include_candidate_context: bool = False
) -> dict[str, Any]:
    task_text = str(snapshot.get("question") or "")
    paths = []
    for index, path in enumerate(snapshot.get("paths") or (), start=1):
        steps = list(path.get("steps") or ())
        paths.append(
            {
                "path_index": index,
                "final_role": steps[-1].get("agent") if steps else None,
                "stop_reason": path.get("stop_reason"),
                "raw_final_output": _path_final_output(path),
            }
        )
    payload = {
        "prompt_version": PROMPT_VERSION,
        "original_question": original_question(task_text),
        "path_final_outputs": paths,
    }
    if include_candidate_context:
        payload["candidate_context"] = task_text
    return payload


def _compact_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split())


def validate_llm_extraction(
    payload: Mapping[str, Any], judge_input: Mapping[str, Any]
) -> dict[str, Any]:
    status = str(payload.get("status") or "").strip().upper()
    if status not in {"ANSWER", "NO_ANSWER"}:
        raise ValueError("LLM canonicalizer returned an invalid status")
    paths = list(judge_input.get("path_final_outputs") or ())
    if status == "NO_ANSWER":
        return {
            "status": status,
            "source_path_index": 0,
            "answer": "",
            "answer_quote": "",
        }

    try:
        source_index = int(payload.get("source_path_index"))
    except (TypeError, ValueError) as error:
        raise ValueError("ANSWER requires a valid source_path_index") from error
    if source_index < 1 or source_index > len(paths):
        raise ValueError("source_path_index is outside the saved path range")
    answer = _compact_whitespace(payload.get("answer"))
    quote = _compact_whitespace(payload.get("answer_quote"))
    source = _compact_whitespace(paths[source_index - 1].get("raw_final_output"))
    if not answer or not quote:
        raise ValueError("ANSWER requires non-empty answer and answer_quote")
    if answer != quote:
        raise ValueError("answer must equal answer_quote after whitespace normalization")
    if quote.casefold() not in source.casefold():
        raise ValueError("answer_quote is not a contiguous span of the selected output")
    if re.fullmatch(r"(?i)null\s*[,.;:]?|none\s*[,.;:]?", answer):
        raise ValueError("null/none cannot be accepted as an answer")
    return {
        "status": status,
        "source_path_index": source_index,
        "answer": answer,
        "answer_quote": quote,
    }


def canonicalize_with_llm(
    snapshot: Mapping[str, Any],
    *,
    model: str,
    reasoning_effort: str = "low",
    include_candidate_context: bool = False,
    max_repair_attempts: int = 1,
    query_func: Callable[..., tuple[dict[str, Any], int]] | None = None,
) -> tuple[dict[str, Any], int]:
    judge_input = build_llm_canonicalization_input(
        snapshot,
        include_candidate_context=include_candidate_context,
    )
    query = query_func or query_manager.query_structured
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(judge_input, ensure_ascii=False)},
    ]
    total_tokens = 0
    last_error: ValueError | None = None
    result = None
    for attempt in range(max_repair_attempts + 1):
        payload, tokens = query(
            model,
            messages,
            ANSWER_EXTRACTION_SCHEMA,
            schema_name="musique_answer_canonicalization",
            reasoning_effort=reasoning_effort,
        )
        total_tokens += int(tokens)
        try:
            result = validate_llm_extraction(payload, judge_input)
            break
        except ValueError as error:
            last_error = error
            if attempt >= max_repair_attempts:
                raise
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The extraction violated the candidate-preserving contract: "
                        f"{error}. Return a corrected object. Do not invent an answer."
                    ),
                }
            )
    if result is None:
        raise last_error or ValueError("LLM canonicalization failed")
    result.update(
        {
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "tokens": total_tokens,
            "repair_attempts": max(0, len(messages) - 2),
            "include_candidate_context": bool(include_candidate_context),
        }
    )
    return result, total_tokens


def validate_runtime_extraction(
    payload: Mapping[str, Any], raw_output: Any
) -> dict[str, Any]:
    status = str(payload.get("status") or "").strip().upper()
    if status not in {"FINAL", "CANDIDATE", "NO_ANSWER"}:
        raise ValueError("Runtime extractor returned an invalid status")
    if status == "NO_ANSWER":
        return {"status": status, "answer": "", "answer_quote": ""}
    answer = _compact_whitespace(payload.get("answer"))
    quote = _compact_whitespace(payload.get("answer_quote"))
    source = _compact_whitespace(raw_output)
    if not answer or not quote:
        raise ValueError(f"{status} requires non-empty answer and answer_quote")
    if answer != quote:
        raise ValueError("answer must equal answer_quote after whitespace normalization")
    if quote.casefold() not in source.casefold():
        raise ValueError("answer_quote is not a contiguous span of agent_output")
    if re.fullmatch(r"(?i)null\s*[,.;:]?|none\s*[,.;:]?", answer):
        raise ValueError("null/none cannot be accepted as an answer")
    return {"status": status, "answer": answer, "answer_quote": quote}


def extract_runtime_answer_with_llm(
    raw_output: Any,
    *,
    question: Any,
    role: Any,
    model: str,
    reasoning_effort: str = "low",
    max_repair_attempts: int = 1,
    query_func: Callable[..., tuple[dict[str, Any], int]] | None = None,
) -> tuple[dict[str, Any], int]:
    """Extract an existing answer span without exposing gold or rewriting text."""

    source = str(raw_output or "").strip()
    if not source:
        result = {
            "status": "NO_ANSWER",
            "answer": "",
            "answer_quote": "",
            "prompt_version": RUNTIME_PROMPT_VERSION,
            "model": model,
            "tokens": 0,
            "repair_attempts": 0,
        }
        return result, 0
    judge_input = {
        "prompt_version": RUNTIME_PROMPT_VERSION,
        "original_question": original_question(question),
        "agent_role": str(role or "unknown"),
        "agent_output": source,
    }
    messages = [
        {"role": "system", "content": RUNTIME_SYSTEM_PROMPT},
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
            RUNTIME_EXTRACTION_SCHEMA,
            schema_name="musique_runtime_answer_extraction",
            reasoning_effort=reasoning_effort,
        )
        total_tokens += int(tokens)
        try:
            result = validate_runtime_extraction(payload, source)
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
                        "The extraction violated the quote-only contract: "
                        f"{error}. Return a corrected object without adding text."
                    ),
                }
            )
    if result is None:
        raise last_error or ValueError("Runtime answer extraction failed")
    result.update(
        {
            "prompt_version": RUNTIME_PROMPT_VERSION,
            "model": model,
            "tokens": total_tokens,
            "repair_attempts": repairs,
        }
    )
    return result, total_tokens
