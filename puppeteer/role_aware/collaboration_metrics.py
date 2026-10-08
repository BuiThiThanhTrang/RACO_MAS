"""Deterministic collaboration metrics for closed-context multi-hop QA traces."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from role_aware.musique_answers import parse_musique_output, parse_musique_result
from tasks.evaluator import BenchmarkEvaluator


_REFUSAL_MARKERS = (
    "i cannot provide",
    "i cannot access",
    "i do not have access",
    "unable to answer",
    "cannot complete",
    "no relevant evidence",
)

_SUPPORT_FIELD_KEYS = {
    "supporting_paragraph_ids",
    "supporting_paragraph_indices",
    "support_paragraph_ids",
    "support_ids",
}


def _step_text(step: Mapping[str, Any]) -> str:
    result = step.get("result") or {}
    if not isinstance(result, Mapping):
        return str(result)
    values = []
    for key in (
        "step_data",
        "answer",
        "evidence",
        "candidate_answer",
        "corrected_answer",
        "final_answer",
        "output",
    ):
        value = result.get(key)
        if value is not None and str(value).strip():
            values.append(str(value).strip())
    return "\n".join(values)


def _support_ids_from_declared_fields(value: Any) -> set[int]:
    """Read only explicitly declared support fields from structured output."""
    identifiers: set[int] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
            if normalized_key in _SUPPORT_FIELD_KEYS:
                if isinstance(child, (list, tuple, set)):
                    candidates = child
                else:
                    candidates = re.findall(r"\d+", str(child or ""))
                for candidate in candidates:
                    try:
                        identifiers.add(int(candidate))
                    except (TypeError, ValueError):
                        continue
            elif isinstance(child, (Mapping, list, tuple)):
                identifiers.update(_support_ids_from_declared_fields(child))
        return identifiers
    if isinstance(value, (list, tuple)):
        for child in value:
            identifiers.update(_support_ids_from_declared_fields(child))
    return identifiers


def _json_payloads(text: str) -> list[Any]:
    candidates = [text.strip()]
    candidates.extend(
        match.strip()
        for match in re.findall(r"(?is)```(?:json)?\s*(.*?)\s*```", text)
    )
    payloads = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payloads.append(json.loads(candidate))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return payloads


def _paper_support_ids_from_result(result: Any) -> set[int]:
    """Extract the support prediction emitted with one terminal answer."""
    identifiers = _support_ids_from_declared_fields(result)
    text = "" if result is None else str(result)
    if isinstance(result, Mapping):
        text = "\n".join(
            str(result.get(key)).strip()
            for key in (
                "step_data",
                "answer",
                "evidence",
                "candidate_answer",
                "corrected_answer",
                "final_answer",
                "output",
            )
            if result.get(key) is not None and str(result.get(key)).strip()
        )
    for payload in _json_payloads(text):
        identifiers.update(_support_ids_from_declared_fields(payload))
    if not identifiers:
        identifiers.update(BenchmarkEvaluator.extract_musique_support_ids(text))
    return identifiers


def _paper_compatible_support_prediction(
    paths: list[Mapping[str, Any]], selected_candidate_index: int | None
) -> tuple[set[int], dict[str, Any]]:
    """Return support IDs from the answer-bearing output of the selected path.

    Unlike the trace diagnostic metric, this deliberately does not union support
    citations across paths or earlier steps. It approximates the single support
    set emitted by a MuSiQue model at task completion.
    """
    if selected_candidate_index is None or not (
        0 <= int(selected_candidate_index) < len(paths)
    ):
        return set(), {
            "source": "missing_selected_path",
            "selected_candidate_index": selected_candidate_index,
            "path_uid": None,
            "step_index": None,
        }

    index = int(selected_candidate_index)
    path = paths[index]
    steps = list(path.get("steps") or ())
    for step_index in range(len(steps) - 1, -1, -1):
        step = steps[step_index]
        result = step.get("result") or {}
        if not (
            parse_musique_result(result).has_candidate
            or parse_musique_output(_step_text(step)).has_candidate
        ):
            continue
        return _paper_support_ids_from_result(result), {
            "source": "selected_path_terminal_answer_step",
            "selected_candidate_index": index,
            "path_uid": path.get("path_uid"),
            "step_index": step_index,
            "role": _role(step),
        }

    raw_prediction = path.get("raw_prediction") or path.get("prediction") or ""
    return _paper_support_ids_from_result(raw_prediction), {
        "source": "selected_path_aggregation",
        "selected_candidate_index": index,
        "path_uid": path.get("path_uid"),
        "step_index": None,
    }


def _role(step: Mapping[str, Any]) -> str:
    return str(step.get("agent") or step.get("role") or "unknown")


def _step_failed(step: Mapping[str, Any], text: str) -> bool:
    declared = str(step.get("success", "Success")).strip().casefold()
    if declared not in {"success", "ok", "true", "1"}:
        return True
    normalized = " ".join(text.casefold().split())
    return not normalized or any(marker in normalized for marker in _REFUSAL_MARKERS)


def _contains_answer(text: str, answer: str) -> bool:
    normalized_text = BenchmarkEvaluator.normalize_qa_answer(text)
    normalized_answer = BenchmarkEvaluator.normalize_qa_answer(answer)
    if not normalized_text or not normalized_answer:
        return False
    padded_text = f" {normalized_text} "
    padded_answer = f" {normalized_answer} "
    return padded_answer in padded_text or BenchmarkEvaluator._qa_f1(
        text, answer
    ) >= 0.80


def _milestones_for_text(
    text: str,
    decomposition: list[Mapping[str, Any]],
    gold_support_ids: set[int],
    final_golds: list[str],
) -> set[str]:
    milestones: set[str] = set()
    support_ids = BenchmarkEvaluator.extract_musique_support_ids(text)
    for support_id in support_ids & gold_support_ids:
        milestones.add(f"support:{support_id}")
    for index, hop in enumerate(decomposition):
        answer = hop.get("answer")
        if answer is not None and _contains_answer(text, str(answer)):
            milestones.add(f"hop_answer:{index}")
    scores = BenchmarkEvaluator.musique_answer_scores(text, final_golds)
    if scores["answer_em"]:
        milestones.add("final_answer")
    return milestones


def _is_verification(step: Mapping[str, Any], text: str, final_golds: list[str]) -> bool:
    if _role(step) != "Evidence Verifier":
        return False
    if not BenchmarkEvaluator.musique_answer_scores(text, final_golds)["answer_em"]:
        return False
    lowered = text.casefold()
    return any(token in lowered for token in ("verif", "support", "consistent", "correct"))


def evaluate_musique_collaboration(
    paths: Iterable[Mapping[str, Any]],
    decomposition: Iterable[Mapping[str, Any]],
    gold_support_ids: Iterable[int],
    final_answer: str,
    answer_aliases: Iterable[str] = (),
    selected_candidate_index: int | None = None,
) -> dict[str, Any]:
    """Score progress, handoffs, recovery, and redundant transitions.

    Gold decomposition is used only here, after execution. It must never be
    included in the public task state passed to the planner or actors.
    """

    paths = [dict(path) for path in paths]
    decomposition = [dict(item) for item in decomposition]
    gold_support = {int(value) for value in gold_support_ids}
    final_golds = [str(final_answer), *(str(value) for value in answer_aliases)]
    expected_milestones = {
        *(f"support:{value}" for value in gold_support),
        *(f"hop_answer:{index}" for index in range(len(decomposition))),
        "final_answer",
    }

    achieved: set[str] = set()
    predicted_support_ids: set[int] = set()
    contribution_by_role: dict[str, int] = {}
    total_calls = 0
    useful_calls = 0
    handoffs = 0
    useful_handoffs = 0
    transitions = 0
    redundant_transitions = 0
    recovery_opportunities = 0
    recovered_failures = 0
    path_reports = []

    for path in paths:
        steps = list(path.get("steps") or ())
        path_achieved: set[str] = set()
        previous_role = None
        previous_digest = None
        pending_failures = 0
        path_handoffs = 0
        path_useful_handoffs = 0
        path_redundant = 0

        for step in steps:
            total_calls += 1
            role = _role(step)
            text = _step_text(step)
            text_digest = hashlib.sha256(
                " ".join(text.casefold().split()).encode("utf-8")
            ).hexdigest()
            predicted_support_ids.update(
                BenchmarkEvaluator.extract_musique_support_ids(text)
            )
            current = _milestones_for_text(
                text, decomposition, gold_support, final_golds
            )
            new_milestones = current - path_achieved
            verified = _is_verification(step, text, final_golds)
            failed = _step_failed(step, text)
            useful = bool(new_milestones or verified)
            if useful:
                useful_calls += 1
                contribution_by_role[role] = (
                    contribution_by_role.get(role, 0) + len(new_milestones) + int(verified)
                )
                if pending_failures:
                    recovered_failures += pending_failures
                    pending_failures = 0
            if failed:
                recovery_opportunities += 1
                pending_failures += 1

            if previous_role is not None:
                transitions += 1
                if role != previous_role:
                    handoffs += 1
                    path_handoffs += 1
                    if useful:
                        useful_handoffs += 1
                        path_useful_handoffs += 1
                if (
                    not useful
                    and (role == previous_role or text_digest == previous_digest)
                ):
                    redundant_transitions += 1
                    path_redundant += 1

            path_achieved.update(current)
            achieved.update(current)
            previous_role = role
            previous_digest = text_digest

        path_reports.append(
            {
                "path_uid": path.get("path_uid"),
                "milestones_achieved": sorted(path_achieved),
                "handoffs": path_handoffs,
                "useful_handoffs": path_useful_handoffs,
                "redundant_transitions": path_redundant,
            }
        )

    milestone_rate = len(achieved & expected_milestones) / len(expected_milestones)
    useful_handoff_rate = useful_handoffs / handoffs if handoffs else None
    collaboration_effectiveness = (
        milestone_rate * useful_handoff_rate
        if useful_handoff_rate is not None
        else None
    )
    recovery_rate = (
        recovered_failures / recovery_opportunities
        if recovery_opportunities
        else None
    )
    redundancy_rate = (
        redundant_transitions / transitions if transitions else None
    )
    support_scores = BenchmarkEvaluator.musique_support_scores(
        predicted_support_ids, gold_support
    )
    paper_support_ids, paper_support_metadata = _paper_compatible_support_prediction(
        paths, selected_candidate_index
    )
    paper_support_scores = BenchmarkEvaluator.musique_support_scores(
        paper_support_ids, gold_support
    )
    return {
        "milestone_achievement_rate": milestone_rate,
        "milestones_achieved": sorted(achieved & expected_milestones),
        "milestones_expected": sorted(expected_milestones),
        "useful_handoff_rate": useful_handoff_rate,
        "collaboration_effectiveness": collaboration_effectiveness,
        "recovery_success_rate": recovery_rate,
        "redundant_transition_rate": redundancy_rate,
        "useful_call_ratio": useful_calls / total_calls if total_calls else 0.0,
        "total_calls": total_calls,
        "handoff_count": handoffs,
        "useful_handoff_count": useful_handoffs,
        "recovery_opportunities": recovery_opportunities,
        "recovered_failures": recovered_failures,
        "predicted_supporting_paragraphs": sorted(predicted_support_ids),
        "paper_compatible_supporting_paragraphs": sorted(paper_support_ids),
        "paper_compatible_support_source": paper_support_metadata,
        "paper_compatible_support_precision": paper_support_scores[
            "support_precision"
        ],
        "paper_compatible_support_recall": paper_support_scores[
            "support_recall"
        ],
        "paper_compatible_support_f1": paper_support_scores["support_f1"],
        "contribution_by_role": contribution_by_role,
        "path_reports": path_reports,
        **support_scores,
    }


def evaluate_musique_snapshot(snapshot: Mapping[str, Any], evaluation: Mapping[str, Any]):
    """Offline convenience wrapper for candidates.json + evaluation metadata."""
    return evaluate_musique_collaboration(
        snapshot.get("paths") or (),
        evaluation.get("gold_decomposition") or (),
        evaluation.get("supporting_paragraph_indices") or (),
        evaluation.get("answer") or evaluation.get("gold") or "",
        evaluation.get("answer_aliases") or (),
        selected_candidate_index=(snapshot.get("aggregation") or {}).get(
            "selected_candidate_index"
        ),
    )


def stable_metric_digest(metrics: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(metrics, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
