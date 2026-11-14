from __future__ import annotations

import json
import hashlib
import re
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from role_aware.musique_answers import MusiqueAnswerFields, parse_musique_result
from role_aware.musique_llm_canonicalizer import (
    extract_runtime_answer_with_llm,
    finalize_path_end_candidate_with_llm,
)


_URL_RE = re.compile(r"https?://[^\s\]\[<>{}\"']+", re.IGNORECASE)
_COMPUTE_RE = re.compile(
    r"\b(calculate|compute|count|sum|average|median|percentage|ratio|convert|"
    r"difference|how many|spreadsheet|csv|xlsx|table|dataset|python)\b",
    re.IGNORECASE,
)
_DEPENDENCY_RE = re.compile(
    r"\b(first .* then|after finding|use .* to find|multiple sources|compare .* and|"
    r"cross[- ]?reference|several steps)\b",
    re.IGNORECASE,
)
_TOOL_ACTIONS = {
    "read_file",
    "search_web",
    "search_bing",
    "search_arxiv",
    "access_website",
    "run_python",
    "inspect_media",
    "inspect_spreadsheet",
}
_NON_ANSWER_ROLES = {"Task Decomposer", "Recovery Strategist"}
_TERMINAL_ROLES = {"Evidence Verifier", "Answer Integrator"}
_MUSIQUE_ROOT_ROLES = {
    "Task Decomposer",
    "Evidence Retriever",
    "Comparison & Composition Reasoner",
}
_MUSIQUE_TERMINAL_ROLES = {
    "Evidence Verifier",
    "Answer Integrator",
}
_MMLU_ROOT_ROLES = {
    "Planner / Decomposer",
    "General Reasoner",
    "Problem Decomposer",
}
_MMLU_ANSWER_ROLES = {
    "General Reasoner",
    "Critic / Verifier",
    "Reflector",
    "Integrator / Concluder",
    "Modifier / Repair",
}
_MMLU_SPECIALIST_SOLVER_ROLES = {
    "Quantitative & Formal Reasoner",
    "Natural & Life Science Specialist",
    "Computing & Engineering Specialist",
    "Social, Legal & Business Specialist",
    "Humanities & Behavioral Specialist",
    "Generalist Independent Solver",
}
_MMLU_SPECIALIST_VERIFIER_ROLE = "Adversarial Verifier"
_MMLU_TOOL_ROLES = {
    "File Analyst",
    "Academic Researcher",
    "Web Researcher",
    "Website Reader",
    "Python Tool Agent",
}
_MMLU_CHOICE_RE = re.compile(
    r"(?im)(?:final\s+answer|answer|option|choice)\s*(?:is|:|-)?\s*[\[(]?([A-J])[\])]?\b"
)
_MMLU_BARE_CHOICE_RE = re.compile(r"^\s*[\[(]?([A-J])[\])]?\s*[.!]?\s*$", re.I)
_MMLU_DEFECT_RE = re.compile(
    r"\b(?:incorrect|wrong|flaw|contradiction|misread|calculation error|"
    r"corrected answer|must be corrected|should instead be)\b",
    re.I,
)
_MEDIA_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".mp3",
    ".m4a",
    ".wav",
    ".flac",
    ".ogg",
    ".mov",
    ".mp4",
    ".mkv",
    ".avi",
    ".webm",
}
_SPREADSHEET_EXTENSIONS = {".csv", ".xlsx"}
_PYTHON_FRIENDLY_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".json",
    ".jsonld",
    ".xml",
    ".pdb",
    ".py",
    ".txt",
    ".zip",
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _task_text(task: Mapping[str, Any]) -> str:
    return "\n".join(
        _text(task.get(key))
        for key in ("Question", "question", "prompt", "task")
        if task.get(key) is not None
    )


def _result_answer(result: Any) -> str:
    if not isinstance(result, Mapping):
        return ""
    for key in ("answer", "final_answer", "candidate_answer", "corrected_answer"):
        value = result.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _result_payload_text(result: Any) -> str:
    if not isinstance(result, Mapping):
        return _text(result).strip()
    values = []
    for key in ("step_data", "evidence", "output", "answer", "final_answer"):
        value = result.get(key)
        if value is not None and str(value).strip():
            values.append(_text(value).strip())
    return "\n".join(values)


def _is_substantive(result_text: str) -> bool:
    text = " ".join(result_text.split()).casefold()
    if len(text) < 8:
        return False
    empty_markers = {
        "[no speech detected]",
        "no results found",
        "no relevant results found",
        "file not exists",
    }
    refusal_markers = (
        "i cannot provide",
        "i cannot access",
        "i do not have access",
        "unable to access",
        "cannot complete the task",
        "no file-capable agent",
    )
    return text not in empty_markers and not any(
        marker in text for marker in refusal_markers
    )


def _state_fingerprint(seed: Mapping[str, Any], evidence: list[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        {"seed": dict(seed), "evidence": evidence},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _normalized_span(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _musique_semantic_fingerprint(
    seed: Mapping[str, Any],
    paragraph_ids: Iterable[int],
    candidate_answers: Iterable[str],
    final_answers: Iterable[str],
    intermediate_answers: Iterable[str],
    output_statuses: Iterable[str] = (),
    missing_subquestions: Iterable[str] = (),
    conflicts: Iterable[str] = (),
) -> str:
    """Fingerprint meaningful MuSiQue progress, not another verbose rendering.

    Raw output hashes made every retry look like a new state.  Dynamic routing
    instead tracks only evidence references and canonical answer values, so a
    repeated Integrator call is allowed only after the usable state changed.
    """

    payload = {
        "seed": dict(seed),
        "paragraph_ids": sorted({int(value) for value in paragraph_ids}),
        "candidate_answers": sorted(
            {_normalized_span(value) for value in candidate_answers if value}
        ),
        "final_answers": sorted(
            {_normalized_span(value) for value in final_answers if value}
        ),
        "intermediate_answers": sorted(
            {_normalized_span(value) for value in intermediate_answers if value}
        ),
        "output_statuses": list(output_statuses)[-3:],
        "missing_subquestions": sorted(
            {_normalized_span(value) for value in missing_subquestions if value}
        ),
        "conflicts": sorted(
            {_normalized_span(value) for value in conflicts if value}
        ),
    }
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]


def _error_type(action_name: str, result_text: str, declared_success: bool) -> str | None:
    lowered = result_text.lower()
    if not declared_success:
        return "declared_failure"
    if action_name == "access_website":
        if re.search(r"(?:error|status|http)?\s*403\b|access denied|forbidden", lowered):
            return "http_403"
        if re.search(r"(?:error|status|http)?\s*404\b|page not found", lowered):
            return "http_404"
        if any(marker in lowered for marker in ("captcha", "too many requests", "blocked request")):
            return "access_blocked"
    if action_name == "read_file" and any(
        marker in lowered
        for marker in (
            "object has no attribute 'text_content'",
            'object has no attribute "text_content"',
            "unsupported file",
            "failed to read",
            "could not read",
            "converter error",
        )
    ):
        return "file_conversion_error"
    if any(marker in lowered for marker in ("traceback (most recent call last)", "tool error:")):
        return "tool_error"
    return None


@dataclass(frozen=True)
class RoutingSnapshot:
    state: dict[str, Any]
    eligible_candidate_ids: tuple[int, ...]
    masked_candidates: tuple[dict[str, Any], ...]

    def eligible_views(
        self, views: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        allowed = set(self.eligible_candidate_ids)
        return [
            dict(view)
            for view in views
            if int(view["candidate_id"]) in allowed and bool(view.get("available"))
        ]


class StageRoutingGuard:
    """Build GAIA path state and apply deterministic stage eligibility rules.

    The decision model ranks only valid roles. This keeps the Jev/Sol comparison
    focused on planning quality instead of their willingness to follow soft role
    descriptions.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.profile = str(self.config.get("profile", "gaia_stage_v1"))
        self.max_role_calls = int(self.config.get("max_role_calls_per_path", 3))
        self.prevent_same_state_repeat = bool(
            self.config.get("prevent_role_repeat_same_state", True)
        )
        if self.max_role_calls < 1:
            raise ValueError("routing_guard.max_role_calls_per_path must be positive")
        extraction = dict(self.config.get("answer_extraction") or {})
        self.llm_extraction_enabled = bool(extraction.get("enabled", False))
        self.llm_extraction_model = str(extraction.get("model") or "")
        self.llm_extraction_effort = str(
            extraction.get("reasoning_effort", "low")
        )
        self.llm_extraction_repairs = int(extraction.get("max_repair_attempts", 1))
        self.path_end_finalization_enabled = bool(
            extraction.get("path_end_finalization", True)
        )
        self.llm_extraction_roles = {
            str(value)
            for value in extraction.get(
                "roles",
                (
                    "Comparison & Composition Reasoner",
                    "General Evidence Solver",
                    "Evidence Verifier",
                    "Answer Integrator",
                ),
            )
        }
        if self.llm_extraction_enabled and not self.llm_extraction_model:
            raise ValueError(
                "routing_guard.answer_extraction.model is required when enabled"
            )
        if self.llm_extraction_repairs < 0:
            raise ValueError(
                "routing_guard.answer_extraction.max_repair_attempts must be nonnegative"
            )
        self.audit = None
        self._llm_extraction_cache: dict[str, MusiqueAnswerFields] = {}
        self._path_end_finalization_cache: dict[str, MusiqueAnswerFields] = {}

    def begin_task(self, audit=None) -> None:
        self.audit = audit
        self._llm_extraction_cache = {}
        self._path_end_finalization_cache = {}

    def end_task(self) -> None:
        self.audit = None
        self._llm_extraction_cache = {}
        self._path_end_finalization_cache = {}

    def parse_musique_result(
        self,
        result: Any,
        *,
        question: Any,
        role: Any,
    ) -> MusiqueAnswerFields:
        """Parse structured output, then use a cached gold-blind LLM fallback."""

        parsed = parse_musique_result(result)
        result_text = _result_payload_text(result)
        role_name = str(role or "unknown")
        if (
            parsed.has_candidate
            or parsed.explicit_nonfinal
            or not self.llm_extraction_enabled
            or role_name not in self.llm_extraction_roles
            or not _is_substantive(result_text)
        ):
            return parsed
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "question": str(question or ""),
                    "role": role_name,
                    "output": result_text,
                    "model": self.llm_extraction_model,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        cached = self._llm_extraction_cache.get(cache_key)
        if cached is not None:
            return cached
        scope = (
            self.audit.call_scope(
                path_uid="routing-guard",
                purpose="musique_runtime_answer_extraction",
            )
            if self.audit is not None
            else nullcontext()
        )
        try:
            with scope:
                extraction, tokens = extract_runtime_answer_with_llm(
                    result_text,
                    question=question,
                    role=role_name,
                    model=self.llm_extraction_model,
                    reasoning_effort=self.llm_extraction_effort,
                    max_repair_attempts=self.llm_extraction_repairs,
                )
            answer = str(extraction.get("answer") or "").strip()
            status = str(extraction.get("status") or "NO_ANSWER")
            resolved = MusiqueAnswerFields(
                final_answer=answer if status == "FINAL" else "",
                candidate_answer=answer if status in {"FINAL", "CANDIDATE"} else "",
                final_source=("llm_runtime_fallback" if status == "FINAL" else None),
                candidate_source=(
                    "llm_runtime_fallback"
                    if status in {"FINAL", "CANDIDATE"}
                    else None
                ),
                status=status,
                status_source="llm_runtime_fallback",
            )
            if self.audit is not None:
                self.audit.emit(
                    "musique_runtime_answer_extracted",
                    role=role_name,
                    output_digest=cache_key,
                    status=status,
                    answer=answer,
                    tokens=int(tokens),
                    model=self.llm_extraction_model,
                )
        except Exception as error:
            resolved = parsed
            if self.audit is not None:
                self.audit.emit(
                    "musique_runtime_answer_extraction_failed",
                    role=role_name,
                    output_digest=cache_key,
                    error_type=type(error).__name__,
                    model=self.llm_extraction_model,
                )
        self._llm_extraction_cache[cache_key] = resolved
        return resolved

    def finalize_musique_path_candidate(
        self,
        result: Any,
        *,
        question: Any,
        role: Any,
    ) -> MusiqueAnswerFields:
        """Gold-blind final gate used only after a path has exhausted execution."""

        parsed = self.parse_musique_result(
            result,
            question=question,
            role=role,
        )
        if parsed.final_answer or parsed.explicit_nonfinal or not parsed.candidate_answer:
            return parsed
        if not self.llm_extraction_enabled or not self.path_end_finalization_enabled:
            return parsed
        result_text = _result_payload_text(result)
        role_name = str(role or "unknown")
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "purpose": "path_end_candidate_finalization",
                    "question": str(question or ""),
                    "role": role_name,
                    "output": result_text,
                    "model": self.llm_extraction_model,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        cached = self._path_end_finalization_cache.get(cache_key)
        if cached is not None:
            return cached
        scope = (
            self.audit.call_scope(
                path_uid="routing-guard",
                purpose="musique_path_end_candidate_finalization",
            )
            if self.audit is not None
            else nullcontext()
        )
        try:
            with scope:
                extraction, tokens = finalize_path_end_candidate_with_llm(
                    result_text,
                    question=question,
                    role=role_name,
                    model=self.llm_extraction_model,
                    reasoning_effort=self.llm_extraction_effort,
                    max_repair_attempts=self.llm_extraction_repairs,
                )
            answer = str(extraction.get("answer") or "").strip()
            if extraction.get("status") == "FINAL" and answer:
                resolved = MusiqueAnswerFields(
                    final_answer=answer,
                    candidate_answer=answer,
                    final_source="llm_path_end_gate",
                    candidate_source="llm_path_end_gate",
                    status="FINAL",
                    status_source="llm_path_end_gate",
                    supporting_paragraph_ids=parsed.supporting_paragraph_ids,
                )
            else:
                resolved = MusiqueAnswerFields(
                    status="NO_ANSWER",
                    status_source="llm_path_end_gate",
                    supporting_paragraph_ids=parsed.supporting_paragraph_ids,
                )
            if self.audit is not None:
                self.audit.emit(
                    "musique_path_end_candidate_finalized",
                    role=role_name,
                    output_digest=cache_key,
                    status=resolved.status,
                    answer=resolved.final_answer,
                    tokens=int(tokens),
                    model=self.llm_extraction_model,
                )
        except Exception as error:
            resolved = parsed
            if self.audit is not None:
                self.audit.emit(
                    "musique_path_end_candidate_finalization_failed",
                    role=role_name,
                    output_digest=cache_key,
                    error_type=type(error).__name__,
                    model=self.llm_extraction_model,
                )
        self._path_end_finalization_cache[cache_key] = resolved
        return resolved

    @staticmethod
    def _is_gaia(global_info) -> bool:
        return str(global_info.task.get("type", "")).strip().lower() == "gaia"

    @staticmethod
    def _is_musique(global_info) -> bool:
        return str(global_info.task.get("type", "")).strip().lower() == "musique"

    @staticmethod
    def _is_mmlu(global_info) -> bool:
        return str(global_info.task.get("type", "")).strip().lower() in {
            "mmlu",
            "mmlu-pro",
            "mmlu_pro",
        }

    def evaluate(
        self, global_info, views: Iterable[Mapping[str, Any]]
    ) -> RoutingSnapshot:
        views = [dict(view) for view in views]
        available = [view for view in views if bool(view.get("available"))]
        if self.enabled and self._is_musique(global_info):
            return self._evaluate_musique(global_info, available)
        if self.enabled and self._is_mmlu(global_info) and self.profile.startswith(
            "mmlu_"
        ):
            return self._evaluate_mmlu(global_info, available)
        if not self.enabled or not self._is_gaia(global_info):
            return RoutingSnapshot(
                state={"enabled": False, "profile": self.profile},
                eligible_candidate_ids=tuple(
                    int(view["candidate_id"]) for view in available
                ),
                masked_candidates=(),
            )

        task = global_info.task
        question = _task_text(task)
        workflow = list(global_info.workflow.workflow)
        root = global_info.path_id == -1
        task_urls = _URL_RE.findall(question)
        discovered_urls = list(task_urls)
        role_history: list[str] = []
        successful_actions: list[str] = []
        failed_actions: list[str] = []
        evidence_count = 0
        substantive_output_count = 0
        candidate_answers: list[str] = []
        attachment_read_success = False
        media_inspection_completed = False
        web_search_completed = False
        last_error_type = None
        last_effective_success = None
        evidence_events: list[dict[str, Any]] = []
        role_state_fingerprints: dict[str, set[str]] = {}
        used_search_queries: list[str] = []
        state_seed = {
            "question": question,
            "attachment": task.get("file_name") or getattr(global_info, "file_name", None),
        }

        for action in workflow:
            role = str(action.agent_role)
            action_name = str(action.action.get("action") or "")
            parameter = str(action.action.get("parameter") or "").strip()
            result_text = _result_payload_text(action.result)
            answer = _result_answer(action.result)
            declared_success = str(action.success).strip().lower() == "success"
            error_type = _error_type(action_name, result_text, declared_success)
            if (
                declared_success
                and error_type is None
                and not answer
                and not _is_substantive(result_text)
            ):
                error_type = "non_substantive_output"
            effective_success = declared_success and error_type is None
            pre_action_fingerprint = _state_fingerprint(state_seed, evidence_events)
            role_state_fingerprints.setdefault(role, set()).add(pre_action_fingerprint)
            role_history.append(role)
            discovered_urls.extend(_URL_RE.findall(result_text))
            if action_name in {"search_web", "search_bing", "search_arxiv"} and parameter:
                used_search_queries.append(parameter)
            if effective_success:
                successful_actions.append(action_name)
                if _is_substantive(result_text):
                    substantive_output_count += 1
                if action_name in _TOOL_ACTIONS and _is_substantive(result_text):
                    evidence_count += 1
                if action_name == "read_file":
                    attachment_read_success = True
                if action_name == "inspect_media" and _is_substantive(result_text):
                    media_inspection_completed = True
                if action_name in {"search_web", "search_bing", "search_arxiv"}:
                    web_search_completed = True
                if answer and role not in _NON_ANSWER_ROLES:
                    candidate_answers.append(answer)
                if _is_substantive(result_text):
                    evidence_events.append(
                        {
                            "role": role,
                            "action": action_name,
                            "parameter": " ".join(parameter.casefold().split()),
                            "result_digest": hashlib.sha256(
                                result_text.encode("utf-8")
                            ).hexdigest()[:16],
                            "answer": answer,
                        }
                    )
            else:
                failed_actions.append(action_name)
            last_error_type = error_type
            last_effective_success = effective_success

        discovered_urls = list(dict.fromkeys(url.rstrip(".,);:") for url in discovered_urls))
        role_counts = Counter(role_history)
        same_role_repeat_count = 0
        if role_history:
            last_role = role_history[-1]
            for role in reversed(role_history):
                if role != last_role:
                    break
                same_role_repeat_count += 1

        has_attachment = bool(
            task.get("has_attachment")
            or task.get("file_name")
            or getattr(global_info, "file_name", None)
        )
        attachment_extension = str(
            getattr(global_info, "file_extension", "") or ""
        ).casefold()
        try:
            level = int(task.get("level") or 1)
        except (TypeError, ValueError):
            level = 1
        computation_likely = bool(_COMPUTE_RE.search(question))
        multi_step_likely = level > 1 or bool(_DEPENDENCY_RE.search(question))
        candidate_answer_exists = bool(candidate_answers)
        last_role = role_history[-1] if role_history else None
        last_action = (
            str(workflow[-1].action.get("action") or "") if workflow else None
        )
        if workflow and last_effective_success and not candidate_answer_exists and last_role in _TERMINAL_ROLES:
            last_error_type = "missing_expected_answer"
            last_effective_success = False

        current_state_fingerprint = _state_fingerprint(state_seed, evidence_events)
        role_state_changed = {
            role: current_state_fingerprint not in fingerprints
            for role, fingerprints in role_state_fingerprints.items()
        }

        state = {
            "enabled": True,
            "profile": self.profile,
            "stage": "root" if root else "path",
            "has_attachment": has_attachment,
            "attachment_extension": attachment_extension,
            "attachment_read_success": attachment_read_success,
            "media_inspection_completed": media_inspection_completed,
            "question_has_url": bool(task_urls),
            "has_url": bool(discovered_urls),
            "discovered_urls": discovered_urls[:10],
            "web_search_completed": web_search_completed,
            "used_search_queries": list(dict.fromkeys(used_search_queries)),
            "evidence_count": evidence_count,
            "substantive_output_count": substantive_output_count,
            "candidate_answer_exists": candidate_answer_exists,
            "candidate_answer_count": len(candidate_answers),
            "last_action": last_action,
            "last_role": last_role,
            "last_action_status": (
                None
                if last_effective_success is None
                else "success" if last_effective_success else "failure"
            ),
            "last_error_type": last_error_type,
            "successful_actions": successful_actions,
            "failed_actions": failed_actions,
            "role_history": role_history,
            "role_call_counts": dict(role_counts),
            "state_fingerprint": current_state_fingerprint,
            "role_state_changed": role_state_changed,
            "same_role_repeat_count": same_role_repeat_count,
            "remaining_depth": getattr(global_info, "remaining_depth", None),
            "path_stalled": bool(last_effective_success is False),
            "has_conflicting_evidence": False,
            "computation_likely": computation_likely,
            "multi_step_likely": multi_step_likely,
        }

        eligible_ids: list[int] = []
        masked: list[dict[str, Any]] = []
        for view in available:
            candidate_id = int(view["candidate_id"])
            role = str(view.get("role_name") or "")
            reason = self._mask_reason(
                role=role,
                root=root,
                state=state,
                role_counts=role_counts,
                level=level,
            )
            if reason is None:
                eligible_ids.append(candidate_id)
            else:
                masked.append(
                    {"candidate_id": candidate_id, "role_name": role, "reason": reason}
                )

        if root and not eligible_ids:
            # A valid root must always exist. Prefer the general solver rather
            # than silently bypassing all masking rules.
            for view in available:
                if str(view.get("role_name")) == "General Evidence Solver":
                    candidate_id = int(view["candidate_id"])
                    eligible_ids.append(candidate_id)
                    masked = [item for item in masked if item["candidate_id"] != candidate_id]
                    break

        state["eligible_candidate_ids"] = list(eligible_ids)
        state["masked_candidates"] = masked
        return RoutingSnapshot(state, tuple(eligible_ids), tuple(masked))

    @staticmethod
    def _mmlu_choice(*values: Any) -> str:
        for value in values:
            rendered = str(value or "").strip()
            if not rendered:
                continue
            match = _MMLU_CHOICE_RE.search(rendered) or _MMLU_BARE_CHOICE_RE.match(
                rendered
            )
            if match:
                return match.group(1).upper()
        return ""

    def _evaluate_mmlu(
        self, global_info, available: list[Mapping[str, Any]]
    ) -> RoutingSnapshot:
        """Expose only observable MMLU progress and mask stage-infeasible roles.

        This profile intentionally uses no capability or backbone scores.  It
        enforces role preconditions while leaving the decision model free to
        select among all currently feasible baseline roles.
        """

        task = global_info.task
        question = _task_text(task)
        workflow = list(global_info.workflow.workflow)
        root = global_info.path_id == -1
        role_history: list[str] = []
        evidence_events: list[dict[str, Any]] = []
        role_state_fingerprints: dict[str, set[str]] = {}
        candidate_answers: list[str] = []
        explicit_choices: list[str] = []
        substantive_output_count = 0
        last_effective_success = None
        last_error_type = None
        last_result_text = ""
        state_seed = {
            "question": question,
            "category": task.get("category"),
        }
        specialist_profile = self.profile == "mmlu_specialist_structured_v1"
        answer_roles = (
            _MMLU_SPECIALIST_SOLVER_ROLES | {_MMLU_SPECIALIST_VERIFIER_ROLE}
            if specialist_profile
            else _MMLU_ANSWER_ROLES
        )

        for action in workflow:
            role = str(action.agent_role)
            action_name = str(action.action.get("action") or "")
            result_text = _result_payload_text(action.result)
            answer = _result_answer(action.result)
            declared_success = str(action.success).strip().lower() == "success"
            substantive = _is_substantive(result_text)
            effective_success = declared_success and substantive
            pre_action_fingerprint = _state_fingerprint(state_seed, evidence_events)
            role_state_fingerprints.setdefault(role, set()).add(
                pre_action_fingerprint
            )
            role_history.append(role)
            if effective_success:
                substantive_output_count += 1
                choice = self._mmlu_choice(answer, result_text)
                if role in answer_roles:
                    candidate_answers.append(answer or result_text)
                    if choice:
                        explicit_choices.append(choice)
                evidence_events.append(
                    {
                        "role": role,
                        "action": action_name,
                        "choice": choice,
                        "result_digest": hashlib.sha256(
                            result_text.encode("utf-8")
                        ).hexdigest()[:16],
                    }
                )
                last_error_type = None
            else:
                last_error_type = (
                    "declared_failure"
                    if not declared_success
                    else "non_substantive_output"
                )
            last_effective_success = effective_success
            last_result_text = result_text

        role_counts = Counter(role_history)
        current_state_fingerprint = _state_fingerprint(state_seed, evidence_events)
        role_state_changed = {
            role: current_state_fingerprint not in fingerprints
            for role, fingerprints in role_state_fingerprints.items()
        }
        candidate_answer_exists = bool(candidate_answers)
        explicit_choice_exists = bool(explicit_choices)
        last_role = role_history[-1] if role_history else None
        defect_identified = bool(
            last_role in {"Critic / Verifier", "Reflector"}
            and _MMLU_DEFECT_RE.search(last_result_text)
        )
        state = {
            "enabled": True,
            "profile": self.profile,
            "stage": "root" if root else "path",
            "closed_book": True,
            "candidate_answer_exists": candidate_answer_exists,
            "candidate_answer_count": len(candidate_answers),
            "candidate_answers": candidate_answers[-3:],
            "explicit_choice_exists": explicit_choice_exists,
            "explicit_choices": explicit_choices[-3:],
            "terminal_ready": bool(explicit_choice_exists and last_effective_success),
            "substantive_output_count": substantive_output_count,
            "multiple_substantive_outputs": substantive_output_count >= 2,
            "long_context_available": len(last_result_text) >= 1600,
            "defect_identified": defect_identified,
            "last_role": last_role,
            "last_action": (
                str(workflow[-1].action.get("action") or "") if workflow else None
            ),
            "last_action_status": (
                None
                if last_effective_success is None
                else "success" if last_effective_success else "failure"
            ),
            "last_error_type": last_error_type,
            "path_stalled": bool(last_effective_success is False),
            "role_history": role_history,
            "role_call_counts": dict(role_counts),
            "state_fingerprint": current_state_fingerprint,
            "role_state_changed": role_state_changed,
            "remaining_depth": getattr(global_info, "remaining_depth", None),
        }

        eligible_ids: list[int] = []
        masked: list[dict[str, Any]] = []
        for view in available:
            candidate_id = int(view["candidate_id"])
            role = str(view.get("role_name") or "")
            if specialist_profile:
                reason = self._mmlu_specialist_mask_reason(
                    role=role,
                    root=root,
                    state=state,
                    role_counts=role_counts,
                )
            else:
                reason = self._mmlu_mask_reason(
                    role=role,
                    root=root,
                    state=state,
                    role_counts=role_counts,
                )
            if reason is None:
                eligible_ids.append(candidate_id)
            else:
                masked.append(
                    {
                        "candidate_id": candidate_id,
                        "role_name": role,
                        "reason": reason,
                    }
                )

        if root and not eligible_ids:
            fallback_role = (
                "Generalist Independent Solver"
                if specialist_profile
                else "General Reasoner"
            )
            for view in available:
                if str(view.get("role_name") or "") == fallback_role:
                    candidate_id = int(view["candidate_id"])
                    eligible_ids.append(candidate_id)
                    masked = [
                        item
                        for item in masked
                        if int(item["candidate_id"]) != candidate_id
                    ]
                    break
        state["eligible_candidate_ids"] = list(eligible_ids)
        state["masked_candidates"] = masked
        return RoutingSnapshot(state, tuple(eligible_ids), tuple(masked))

    def _mmlu_specialist_mask_reason(
        self,
        *,
        role: str,
        root: bool,
        state: Mapping[str, Any],
        role_counts: Counter,
    ) -> str | None:
        """Apply only state preconditions for the specialist MMLU pool.

        Domain matching remains a Jev decision over the question, category and
        role cards.  The guard prevents an invalid root verifier call and exact
        same-state repetition without introducing capability/backbone scores.
        """

        known_roles = _MMLU_SPECIALIST_SOLVER_ROLES | {
            _MMLU_SPECIALIST_VERIFIER_ROLE
        }
        if role not in known_roles:
            return "unknown_mmlu_specialist_role"
        if root:
            return (
                "requires_candidate_answer"
                if role == _MMLU_SPECIALIST_VERIFIER_ROLE
                else None
            )
        if role_counts.get(role, 0) >= self.max_role_calls:
            return "role_call_limit_reached"
        if (
            self.prevent_same_state_repeat
            and role_counts.get(role, 0) > 0
            and not bool(state["role_state_changed"].get(role, False))
        ):
            return "role_already_called_without_observable_progress"
        if role == _MMLU_SPECIALIST_VERIFIER_ROLE:
            return (
                None
                if state["candidate_answer_exists"]
                else "requires_candidate_answer"
            )
        return None

    def _mmlu_mask_reason(
        self,
        *,
        role: str,
        root: bool,
        state: Mapping[str, Any],
        role_counts: Counter,
    ) -> str | None:
        if role in _MMLU_TOOL_ROLES:
            return "closed_book_mmlu_disallows_external_tools"
        if root:
            return None if role in _MMLU_ROOT_ROLES else "role_not_valid_at_root"
        if role_counts.get(role, 0) >= self.max_role_calls:
            return "role_call_limit_reached"
        if (
            self.prevent_same_state_repeat
            and role_counts.get(role, 0) > 0
            and not bool(state["role_state_changed"].get(role, False))
        ):
            return "role_already_called_without_observable_progress"
        if role in {"Planner / Decomposer", "Problem Decomposer"}:
            return "decomposition_only_allowed_at_root"
        if role == "Critic / Verifier":
            return (
                None
                if state["candidate_answer_exists"]
                else "requires_candidate_answer"
            )
        if role == "Reflector":
            return (
                None
                if state["candidate_answer_exists"] or state["path_stalled"]
                else "requires_prior_attempt"
            )
        if role == "Modifier / Repair":
            return None if state["defect_identified"] else "requires_identified_defect"
        if role == "Integrator / Concluder":
            return (
                None
                if state["multiple_substantive_outputs"]
                else "requires_multiple_substantive_outputs"
            )
        if role == "Summarizer":
            return (
                None
                if state["multiple_substantive_outputs"]
                or state["long_context_available"]
                else "requires_long_or_multiple_prior_outputs"
            )
        if role == "General Reasoner":
            return None
        return "unknown_mmlu_baseline_role"

    def _evaluate_musique(
        self, global_info, available: list[Mapping[str, Any]]
    ) -> RoutingSnapshot:
        task = global_info.task
        workflow = list(global_info.workflow.workflow)
        root = global_info.path_id == -1
        role_history: list[str] = []
        evidence_events: list[dict[str, Any]] = []
        role_state_fingerprints: dict[str, set[str]] = {}
        candidate_answers: list[str] = []
        final_answers: list[str] = []
        intermediate_answers: list[str] = []
        output_statuses: list[str] = []
        missing_subquestions: list[str] = []
        conflict_descriptions: list[str] = []
        final_answer_lineage: list[dict[str, Any]] = []
        paragraph_ids: set[int] = set()
        intermediate_answer_count = 0
        substantive_output_count = 0
        last_error_type = None
        last_effective_success = None
        last_output_status = "NO_ANSWER"
        last_output_status_explicit = False
        state_seed = {
            "question": _task_text(task),
            "paragraph_ids": task.get("paragraph_ids") or (),
        }
        dynamic = self.profile == "musique_dynamic_v2"

        for action in workflow:
            role = str(action.agent_role)
            action_name = str(action.action.get("action") or "")
            result_text = _result_payload_text(action.result)
            parsed_answer = self.parse_musique_result(
                action.result,
                question=_task_text(task),
                role=role,
            )
            answer = parsed_answer.candidate_answer
            declared_success = str(action.success).strip().lower() == "success"
            substantive = _is_substantive(result_text)
            effective_success = declared_success and substantive
            pre_action_fingerprint = (
                _musique_semantic_fingerprint(
                    state_seed,
                    paragraph_ids,
                    candidate_answers,
                    final_answers,
                    intermediate_answers,
                    output_statuses,
                    missing_subquestions,
                    conflict_descriptions,
                )
                if dynamic
                else _state_fingerprint(state_seed, evidence_events)
            )
            role_state_fingerprints.setdefault(role, set()).add(
                pre_action_fingerprint
            )
            role_history.append(role)
            references = {
                int(value)
                for value in re.findall(
                    r"(?i)(?<![A-Za-z0-9])P\s*(\d+)\b", result_text
                )
            }
            references.update(parsed_answer.supporting_paragraph_ids)
            paragraph_ids.update(references)
            if effective_success:
                substantive_output_count += 1
                if parsed_answer.status_explicit:
                    output_statuses.append(parsed_answer.status)
                if parsed_answer.missing_subquestion:
                    missing_subquestions.append(parsed_answer.missing_subquestion)
                if parsed_answer.conflict_description:
                    conflict_descriptions.append(parsed_answer.conflict_description)
                if role == "Bridge Entity Reasoner" and parsed_answer.intermediate_answer:
                    intermediate_answer_count += 1
                    intermediate_answers.append(parsed_answer.intermediate_answer)
                if answer:
                    candidate_answers.append(answer)
                if parsed_answer.final_answer:
                    final_answers.append(parsed_answer.final_answer)
                    final_answer_lineage.append(
                        {
                            "answer": parsed_answer.final_answer,
                            "source_role": role,
                            "source_action": action_name,
                            "evidence_paragraph_ids": sorted(paragraph_ids),
                            "source_event_index": len(evidence_events),
                        }
                    )
                evidence_events.append(
                    {
                        "event_index": len(evidence_events),
                        "role": role,
                        "action": action_name,
                        "references": sorted(references),
                        "inherited_references": sorted(paragraph_ids),
                        "result_digest": hashlib.sha256(
                            result_text.encode("utf-8")
                        ).hexdigest()[:16],
                        "answer": answer,
                        "final_answer": parsed_answer.final_answer,
                        "status": parsed_answer.status,
                        "status_explicit": parsed_answer.status_explicit,
                        "missing_subquestion": parsed_answer.missing_subquestion,
                        "conflict_description": parsed_answer.conflict_description,
                    }
                )
                last_error_type = None
            else:
                last_error_type = (
                    "declared_failure" if not declared_success else "non_substantive_output"
                )
            last_effective_success = effective_success
            last_output_status = parsed_answer.status
            last_output_status_explicit = parsed_answer.status_explicit

        role_counts = Counter(role_history)
        current_state_fingerprint = (
            _musique_semantic_fingerprint(
                state_seed,
                paragraph_ids,
                candidate_answers,
                final_answers,
                intermediate_answers,
                output_statuses,
                missing_subquestions,
                conflict_descriptions,
            )
            if dynamic
            else _state_fingerprint(state_seed, evidence_events)
        )
        role_state_changed = {
            role: current_state_fingerprint not in fingerprints
            for role, fingerprints in role_state_fingerprints.items()
        }
        last_role = role_history[-1] if role_history else None
        candidate_answer_exists = bool(candidate_answers)
        final_answer_exists = bool(final_answers)
        explicit_nonfinal_status = bool(
            last_output_status_explicit
            and last_output_status
            in {"NEEDS_EVIDENCE", "CONFLICT", "NO_ANSWER", "MALFORMED"}
        )
        unresolved_dependency = bool(
            explicit_nonfinal_status or last_output_status == "CANDIDATE"
        )
        integrator_required = bool(
            candidate_answer_exists
            and not final_answer_exists
            and not explicit_nonfinal_status
            and last_role
            in {"Evidence Retriever", "Evidence Verifier", "Answer Integrator"}
        )
        state = {
            "enabled": True,
            "profile": self.profile,
            "stage": "root" if root else "path",
            # Dynamic routing deliberately does not expose the benchmark's
            # oracle hop count.  The legacy profile retains it for exact run
            # reproducibility.
            **({} if dynamic else {"hop_count": int(task.get("hop_count") or 0)}),
            "paragraph_reference_count": len(paragraph_ids),
            "referenced_paragraph_ids": sorted(paragraph_ids),
            "evidence_lineage_exists": bool(paragraph_ids),
            "evidence_lineage": evidence_events[-8:],
            "final_answer_lineage": final_answer_lineage[-3:],
            "intermediate_answer_count": intermediate_answer_count,
            "substantive_output_count": substantive_output_count,
            "candidate_answer_exists": candidate_answer_exists,
            "candidate_answer_count": len(candidate_answers),
            "candidate_answers": list(dict.fromkeys(candidate_answers))[-5:],
            "final_answer_exists": final_answer_exists,
            "final_answer_count": len(final_answers),
            "final_answers": list(dict.fromkeys(final_answers))[-3:],
            "integrator_required": integrator_required,
            "current_output_status": last_output_status,
            "output_status_explicit": last_output_status_explicit,
            "unresolved_dependency": unresolved_dependency,
            "missing_subquestion": (
                missing_subquestions[-1]
                if explicit_nonfinal_status and missing_subquestions
                else ""
            ),
            "conflict_description": (
                conflict_descriptions[-1]
                if last_output_status == "CONFLICT" and conflict_descriptions
                else ""
            ),
            "has_conflicting_evidence": bool(
                last_output_status == "CONFLICT" and last_output_status_explicit
            ),
            "last_role": last_role,
            "last_action_status": (
                None
                if last_effective_success is None
                else "success" if last_effective_success else "failure"
            ),
            "last_error_type": last_error_type,
            "role_history": role_history,
            "role_call_counts": dict(role_counts),
            "state_fingerprint": current_state_fingerprint,
            "semantic_state_fingerprint": current_state_fingerprint,
            "role_state_changed": role_state_changed,
            "repeat_without_progress": bool(
                last_role
                and role_counts.get(last_role, 0) > 0
                and not role_state_changed.get(last_role, True)
            ),
            "remaining_depth": getattr(global_info, "remaining_depth", None),
            "recovery_depth_active": bool(
                getattr(global_info, "recovery_depth_active", False)
            ),
            "recovery_steps_remaining": int(
                getattr(global_info, "recovery_steps_remaining", 0) or 0
            ),
            "path_stalled": bool(
                last_effective_success is False
                or (
                    last_output_status_explicit
                    and last_output_status in {"MALFORMED", "NO_ANSWER"}
                )
            ),
        }

        eligible_ids: list[int] = []
        masked: list[dict[str, Any]] = []
        for view in available:
            role = str(view.get("role_name") or "")
            candidate_id = int(view["candidate_id"])
            reason = (
                self._musique_dynamic_mask_reason(
                    role=role,
                    root=root,
                    state=state,
                    role_counts=role_counts,
                )
                if dynamic
                else self._musique_mask_reason(
                    role=role,
                    root=root,
                    state=state,
                    role_counts=role_counts,
                )
            )
            if reason is None:
                eligible_ids.append(candidate_id)
            else:
                masked.append(
                    {
                        "candidate_id": candidate_id,
                        "role_name": role,
                        "reason": reason,
                    }
                )

        if root and not eligible_ids and available:
            candidate_id = int(available[0]["candidate_id"])
            eligible_ids.append(candidate_id)
            masked = [
                item for item in masked if item["candidate_id"] != candidate_id
            ]
        state["eligible_candidate_ids"] = list(eligible_ids)
        state["masked_candidates"] = masked
        return RoutingSnapshot(state, tuple(eligible_ids), tuple(masked))

    def _musique_dynamic_mask_reason(
        self,
        *,
        role: str,
        root: bool,
        state: Mapping[str, Any],
        role_counts: Counter,
    ) -> str | None:
        """Apply only feasibility, budget, and genuine no-progress constraints.

        Unlike ``musique_stage_v1``, this does not encode a fixed role order.
        Decomposition, retrieval, bridge reasoning, and composition remain
        available whenever their role contracts are technically satisfiable.
        """

        if role_counts.get(role, 0) >= self.max_role_calls:
            return "role_call_limit_reached"
        recovery_finalization_slot = bool(
            state.get("recovery_depth_active")
            and int(state.get("recovery_steps_remaining") or 0) <= 1
            and state.get("candidate_answer_exists")
        )
        if recovery_finalization_slot and role != "Answer Integrator":
            return "recovery_finalization_slot_requires_integrator"
        if (
            self.prevent_same_state_repeat
            and role_counts.get(role, 0) > 0
            and not bool(state["role_state_changed"].get(role, False))
        ):
            return "role_already_called_without_semantic_progress"
        if role in _MUSIQUE_TERMINAL_ROLES and not state["candidate_answer_exists"]:
            return "requires_candidate_answer"
        return None

    def _musique_mask_reason(
        self,
        *,
        role: str,
        root: bool,
        state: Mapping[str, Any],
        role_counts: Counter,
    ) -> str | None:
        if root:
            return None if role in _MUSIQUE_ROOT_ROLES else "role_not_valid_at_root"
        if role_counts.get(role, 0) >= self.max_role_calls:
            return "role_call_limit_reached"
        if (
            self.prevent_same_state_repeat
            and role_counts.get(role, 0) > 0
            and not bool(state["role_state_changed"].get(role, False))
        ):
            return "role_already_called_for_current_state"
        if state["final_answer_exists"]:
            return "terminal_answer_ready"
        if state["integrator_required"]:
            return None if role == "Answer Integrator" else "answer_integrator_required"
        if state["candidate_answer_exists"]:
            if role == "Evidence Verifier":
                return None
            if role == "Answer Integrator":
                return None
            return "candidate_ready_for_verification"
        if role == "Task Decomposer":
            return "decomposition_only_allowed_at_root"
        if role == "Evidence Retriever":
            return None
        if role == "Bridge Entity Reasoner":
            return (
                None
                if state["paragraph_reference_count"] > 0
                else "requires_retrieved_evidence"
            )
        if role == "Comparison & Composition Reasoner":
            return (
                None
                if state["paragraph_reference_count"] >= 2
                or state["intermediate_answer_count"] > 0
                else "requires_multiple_facts_or_intermediate_answer"
            )
        if role == "Evidence Verifier":
            return "requires_candidate_answer"
        if role == "Answer Integrator":
            return "requires_candidate_answer"
        return "unknown_musique_role"

    def _mask_reason(
        self,
        *,
        role: str,
        root: bool,
        state: Mapping[str, Any],
        role_counts: Counter,
        level: int,
    ) -> str | None:
        if root:
            if role == "General Evidence Solver":
                return None
            if role == "Web Researcher":
                return None
            if role == "File Analyst":
                if not state["has_attachment"]:
                    return "requires_attachment"
                if state["attachment_extension"] in _MEDIA_EXTENSIONS:
                    return "media_attachment_requires_media_analyst"
                if state["attachment_extension"] in _SPREADSHEET_EXTENSIONS:
                    return "spreadsheet_requires_spreadsheet_analyst"
                return None
            if role == "Media Analyst":
                return (
                    None
                    if state["attachment_extension"] in _MEDIA_EXTENSIONS
                    else "requires_media_attachment"
                )
            if role == "Spreadsheet Analyst":
                return (
                    None
                    if state["attachment_extension"] in _SPREADSHEET_EXTENSIONS
                    else "requires_spreadsheet_attachment"
                )
            if role == "Python Data Analyst":
                return (
                    None
                    if state["computation_likely"]
                    or state["attachment_extension"] in _PYTHON_FRIENDLY_EXTENSIONS
                    else "no_computation_or_attachment_signal"
                )
            if role == "Website Reader":
                return None if state["question_has_url"] else "requires_known_url"
            if role == "Task Decomposer":
                return (
                    None
                    if level > 1 or state["multi_step_likely"]
                    else "level_1_task_does_not_need_decomposition"
                )
            return "role_not_valid_at_root"

        if role_counts.get(role, 0) >= self.max_role_calls:
            return "role_call_limit_reached"

        if (
            self.prevent_same_state_repeat
            and role_counts.get(role, 0) > 0
            and not bool(state["role_state_changed"].get(role, False))
        ):
            return "role_already_called_for_current_state"

        if state["last_role"] in _TERMINAL_ROLES and state["candidate_answer_exists"]:
            return "terminal_answer_ready"

        if state["candidate_answer_exists"] and role not in _TERMINAL_ROLES:
            return "candidate_ready_for_verification"

        if role == "Task Decomposer":
            return "decomposition_only_allowed_at_root"
        if role == "File Analyst":
            if state["candidate_answer_exists"]:
                return "candidate_ready_for_verification"
            if not state["has_attachment"]:
                return "requires_attachment"
            if state["attachment_extension"] in _MEDIA_EXTENSIONS:
                return "media_attachment_requires_media_analyst"
            if state["attachment_extension"] in _SPREADSHEET_EXTENSIONS:
                return "spreadsheet_requires_spreadsheet_analyst"
            if state["attachment_read_success"]:
                return "attachment_already_read"
            return None
        if role == "Media Analyst":
            if state["candidate_answer_exists"]:
                return "candidate_ready_for_verification"
            if state["media_inspection_completed"]:
                return "media_already_inspected"
            return (
                None
                if state["attachment_extension"] in _MEDIA_EXTENSIONS
                else "requires_media_attachment"
            )
        if role == "Spreadsheet Analyst":
            if state["candidate_answer_exists"]:
                return "candidate_ready_for_verification"
            return (
                None
                if state["attachment_extension"] in _SPREADSHEET_EXTENSIONS
                else "requires_spreadsheet_attachment"
            )
        if role == "Web Researcher":
            if state["candidate_answer_exists"]:
                return "candidate_ready_for_verification"
            return None
        if role == "Website Reader":
            if state["candidate_answer_exists"]:
                return "candidate_ready_for_verification"
            return None if state["has_url"] else "requires_known_url"
        if role == "Python Data Analyst":
            if state["candidate_answer_exists"]:
                return "candidate_ready_for_verification"
            if not (
                state["computation_likely"]
                or state["attachment_extension"] in _PYTHON_FRIENDLY_EXTENSIONS
                or state["evidence_count"] > 0
            ):
                return "no_computation_input"
            return None
        if role == "General Evidence Solver":
            return None
        if role == "Evidence Verifier":
            return None if state["candidate_answer_exists"] else "requires_candidate_answer"
        if role == "Answer Integrator":
            if not state["candidate_answer_exists"]:
                return "requires_candidate_answer"
            if state["substantive_output_count"] < 2:
                return "requires_multiple_substantive_outputs"
            return None
        if role == "Recovery Strategist":
            return None if state["last_action_status"] == "failure" else "requires_failed_or_stalled_step"
        return None
