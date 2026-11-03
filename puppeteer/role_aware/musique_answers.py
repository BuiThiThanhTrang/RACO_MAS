"""Candidate-preserving MuSiQue answer extraction.

This module never asks a model to rewrite an answer.  It only reads explicit
answer fields or markers already present in an actor output and removes
presentation-only wrappers such as paragraph citations.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable, Mapping


_FINAL_KEYS = ("final_answer",)
_CANDIDATE_KEYS = ("corrected_answer", "candidate_answer")
_INTERMEDIATE_KEYS = ("intermediate_answer",)
_INVALID_ANSWERS = {
    "",
    "null",
    "none",
    "n/a",
    "unknown",
    "not available",
    "not provided",
    # Markdown fence language tags are presentation syntax, never answers.
    "json",
    "yaml",
    "yml",
    "python",
    "text",
    "markdown",
}
_REFUSAL_MARKERS = (
    "cannot be determined",
    "cannot be answered",
    "information not provided",
    "not specified in the provided",
    "does not contain information",
    "insufficient information",
)
_SCHEMA_MARKERS = (
    "supporting_paragraph_ids",
    "evidence_quotes",
    "supported_claims",
    "hop_checks",
)


@dataclass(frozen=True)
class MusiqueAnswerFields:
    final_answer: str = ""
    candidate_answer: str = ""
    intermediate_answer: str = ""
    final_source: str | None = None
    candidate_source: str | None = None

    @property
    def has_final(self) -> bool:
        return bool(self.final_answer)

    @property
    def has_candidate(self) -> bool:
        return bool(self.candidate_answer or self.final_answer)


def _scalar(value: Any) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set, bool)):
        return ""
    return str(value).strip()


def _strip_citations(value: str) -> str:
    citation_group = r"(?:P\s*\d+\s*[,;]?\s*)+"
    value = re.sub(
        rf"\s*(?:\(\s*{citation_group}\)|\[\s*{citation_group}\])\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.split(
        r"(?im)\s+(?:supporting\s+paragraph(?:s|\s+ids?)?|sources?|evidence)\s*:",
        value,
        maxsplit=1,
    )[0]
    return value.strip()


def canonicalize_answer_value(value: Any, *, strict_final: bool = False) -> str:
    """Return a short existing answer value, or empty when it is not answer-like."""
    text = _scalar(value)
    if not text:
        return ""
    text = text.strip().strip("`").strip()
    text = re.sub(
        r"(?is)^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*",
        "",
        text,
        count=1,
    )
    text = re.sub(r"(?is)^\s*(?:it|this|that)\s+(?:is|was)\s+", "", text)
    text = re.sub(r"(?is)^\s*(?:therefore|thus|hence)\s*[:,]?\s*", "", text)
    text = re.split(
        r"(?i)\s+(?:because|according\s+to|as\s+(?:shown|stated|reported))\b",
        text,
        maxsplit=1,
    )[0]
    text = re.sub(r"(?is)\s+(?:is|was)\s+the\s+answer\.?\s*$", "", text)
    text = re.split(
        r"(?im)^\s*(?:supporting\s+paragraph(?:s|\s+ids?)?|sources?|evidence)\s*:",
        text,
        maxsplit=1,
    )[0].strip()
    text = _strip_citations(text)
    text = text.strip().strip('"\'').strip()
    normalized = " ".join(text.casefold().split())
    if normalized in _INVALID_ANSWERS:
        return ""
    if any(marker in normalized for marker in _REFUSAL_MARKERS):
        return ""
    if text.startswith(("{", "[")) or any(
        marker in normalized for marker in _SCHEMA_MARKERS
    ):
        return ""
    # MuSiQue answers are short spans. Long prose is evidence or explanation,
    # not a value that can be shortened safely without generating a new answer.
    word_limit = 12 if strict_final else 32
    character_limit = 180 if strict_final else 300
    if len(text.split()) > word_limit or len(text) > character_limit:
        return ""
    return text


def _json_payloads(text: str) -> Iterable[Any]:
    stripped = text.strip()
    candidates = [stripped]
    candidates.extend(
        match.strip()
        for match in re.findall(
            r"(?is)```(?:json)?\s*(.*?)\s*```", stripped
        )
    )
    seen: set[str] = set()
    decoder = json.JSONDecoder()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            yield json.loads(candidate)
            continue
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        for index, character in enumerate(candidate):
            if character not in "{[":
                continue
            try:
                payload, _ = decoder.raw_decode(candidate[index:])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            yield payload
            break


def _find_values(payload: Any, keys: tuple[str, ...]) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(payload, Mapping):
        normalized = {str(key).casefold(): value for key, value in payload.items()}
        for key in keys:
            if key in normalized:
                found.append((key, normalized[key]))
        for value in payload.values():
            found.extend(_find_values(value, keys))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            found.extend(_find_values(value, keys))
    return found


def _labeled_values(text: str, keys: tuple[str, ...]) -> list[tuple[str, str]]:
    values = []
    for key in keys:
        pattern = rf"(?im)^\s*[\"']?{re.escape(key)}[\"']?\s*:\s*(.+?)\s*$"
        values.extend((key, match) for match in re.findall(pattern, text))
    return values


def _first_valid(
    values: Iterable[tuple[str, Any]], *, strict_final: bool = False
) -> tuple[str, str | None]:
    for source, value in values:
        canonical = canonicalize_answer_value(value, strict_final=strict_final)
        if canonical:
            return canonical, source
    return "", None


def parse_musique_output(text: Any) -> MusiqueAnswerFields:
    raw = "" if text is None else str(text).strip()
    if not raw:
        return MusiqueAnswerFields()

    payloads = list(_json_payloads(raw))
    final_values: list[tuple[str, Any]] = []
    candidate_values: list[tuple[str, Any]] = []
    intermediate_values: list[tuple[str, Any]] = []
    for payload in payloads:
        final_values.extend(_find_values(payload, _FINAL_KEYS))
        candidate_values.extend(_find_values(payload, _CANDIDATE_KEYS))
        intermediate_values.extend(_find_values(payload, _INTERMEDIATE_KEYS))
    final_values.extend(_labeled_values(raw, _FINAL_KEYS))
    candidate_values.extend(_labeled_values(raw, _CANDIDATE_KEYS))
    intermediate_values.extend(_labeled_values(raw, _INTERMEDIATE_KEYS))

    marker_values = re.findall(
        r"(?is)final\s+answer\s*:\s*(.*)", raw
    )
    for marker in reversed(marker_values):
        nested = parse_musique_output(marker) if marker != raw else MusiqueAnswerFields()
        if nested.final_answer:
            final_values.insert(0, ("final_answer_marker", nested.final_answer))
        elif nested.candidate_answer:
            final_values.insert(0, ("final_answer_marker", nested.candidate_answer))
        else:
            lines = [line.strip() for line in marker.splitlines() if line.strip()]
            # A fenced JSON object containing only intermediate/evidence fields
            # is not a final answer.  Previously the opening ```json token was
            # canonicalized to the literal answer "json".
            if not (lines and lines[0].startswith("```")):
                first_line = lines[0] if lines else ""
                final_values.insert(0, ("final_answer_marker", first_line))

    final_answer, final_source = _first_valid(final_values, strict_final=True)
    # An overlong FINAL ANSWER is still an existing candidate, but it is not a
    # terminal short answer. Preserve it for Answer Integrator instead of
    # silently truncating or generating a replacement.
    if not final_answer:
        candidate_values = [*final_values, *candidate_values]
    candidate_answer, candidate_source = _first_valid(candidate_values)
    intermediate_answer, _ = _first_valid(intermediate_values)
    if final_answer:
        candidate_answer = final_answer
        candidate_source = final_source
    return MusiqueAnswerFields(
        final_answer=final_answer,
        candidate_answer=candidate_answer,
        intermediate_answer=intermediate_answer,
        final_source=final_source,
        candidate_source=candidate_source,
    )


def parse_musique_result(result: Any) -> MusiqueAnswerFields:
    if not isinstance(result, Mapping):
        return parse_musique_output(result)
    direct = {
        key: result.get(key)
        for key in (*_FINAL_KEYS, *_CANDIDATE_KEYS, *_INTERMEDIATE_KEYS)
        if result.get(key) is not None
    }
    text = "\n".join(
        str(result.get(key)).strip()
        for key in ("step_data", "evidence", "output")
        if result.get(key) is not None and str(result.get(key)).strip()
    )
    combined = json.dumps(direct, ensure_ascii=False) + "\n" + text if direct else text
    return parse_musique_output(combined)


def canonicalize_musique_candidate(text: Any) -> str:
    """Canonicalize one existing candidate without inventing missing content."""
    fields = parse_musique_output(text)
    if fields.final_answer:
        return fields.final_answer
    if fields.candidate_answer:
        return fields.candidate_answer
    raw = "" if text is None else str(text).strip()
    if not raw or raw.startswith(("{", "[", "```")):
        return ""
    if re.match(r"(?is)^\s*final\s+answer\s*:\s*```", raw):
        return ""
    return canonicalize_answer_value(raw, strict_final=True)
