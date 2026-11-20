from __future__ import annotations

import ast
import copy
import hashlib
import json
import tokenize
from dataclasses import asdict, dataclass, field
from io import StringIO
from typing import Any, Callable, Mapping

from .capabilities import router_capability_payload
from .config import ProtocolConfig
from .contracts import (
    BaseTestFeedback,
    BudgetSnapshot,
    CodeArtifact,
    ROUTER_ACTIONS,
    TaskSpec,
)


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def python_presentation_fingerprint(source: str) -> tuple[str, str]:
    """Return a conservative presentation identity, never semantic equivalence."""

    try:
        tree = ast.parse(source)
        value = ast.dump(tree, annotate_fields=True, include_attributes=False)
        return hashlib.sha256(value.encode("utf-8")).hexdigest(), "python_ast_v1"
    except SyntaxError:
        pass
    try:
        values = [
            (item.type, item.string)
            for item in tokenize.generate_tokens(StringIO(source).readline)
            if item.type
            not in {
                tokenize.ENCODING,
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            }
        ]
        return _stable_hash(values), "python_tokens_v1"
    except (IndentationError, tokenize.TokenError):
        return hashlib.sha256(source.encode("utf-8")).hexdigest(), "exact_source_v1"


def validate_python_candidate(source: str, entry_point: str) -> tuple[bool, str]:
    if not source.strip():
        return False, "empty_candidate"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"syntax_error:{exc.lineno or 0}:{exc.offset or 0}"
    definitions = {
        item.name
        for item in ast.walk(tree)
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if entry_point not in definitions:
        return False, "missing_entry_point"
    return True, ""


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    kind: str
    version: int
    sha256: str
    producer_action: str
    producer_step: int
    input_versions: Mapping[str, int]
    input_refs: Mapping[str, str]
    visibility: str
    created_at_event: int
    content: Any
    scope: str = "candidate"

    def is_fresh(self, code: CodeArtifact | None) -> bool:
        if self.scope == "task_specification":
            return True
        expected_version = self.input_versions.get("candidate")
        expected_hash = self.input_refs.get("candidate_sha256")
        if code is None:
            return expected_version in {None, 0} and expected_hash in {None, ""}
        return expected_version == code.version and expected_hash == code.sha256

    def public(self, code: CodeArtifact | None, *, include_content: bool = True) -> dict:
        value = {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "version": self.version,
            "sha256": self.sha256,
            "producer_action": self.producer_action,
            "producer_step": self.producer_step,
            "input_versions": dict(self.input_versions),
            "input_refs": dict(self.input_refs),
            "visibility": self.visibility,
            "created_at_event": self.created_at_event,
            "scope": self.scope,
            "fresh": self.is_fresh(code),
        }
        if include_content and self.visibility == "public":
            value["content"] = self.content
        return value


class ArtifactLedger:
    def __init__(self) -> None:
        self.history: list[ArtifactRecord] = []
        self.latest: dict[str, ArtifactRecord] = {}
        self._versions: dict[str, int] = {}

    def add(
        self,
        kind: str,
        content: Any,
        *,
        producer_action: str,
        producer_step: int,
        code: CodeArtifact | None,
        event_index: int,
        visibility: str = "public",
        scope: str = "candidate",
    ) -> ArtifactRecord:
        version = self._versions.get(kind, 0) + 1
        self._versions[kind] = version
        input_versions = {"candidate": code.version if code else 0}
        input_refs = {"candidate_sha256": code.sha256 if code else ""}
        digest = _stable_hash(content)
        record = ArtifactRecord(
            artifact_id=f"{len(self.history) + 1:04d}_{kind}",
            kind=kind,
            version=version,
            sha256=digest,
            producer_action=producer_action,
            producer_step=producer_step,
            input_versions=input_versions,
            input_refs=input_refs,
            visibility=visibility,
            created_at_event=event_index,
            content=content,
            scope=scope,
        )
        self.history.append(record)
        self.latest[kind] = record
        return record

    def fresh(self, kind: str, code: CodeArtifact | None) -> ArtifactRecord | None:
        record = self.latest.get(kind)
        return record if record and record.is_fresh(code) else None

    def public_latest(self, code: CodeArtifact | None) -> dict[str, dict]:
        return {
            kind: record.public(code)
            for kind, record in sorted(self.latest.items())
            if kind != "candidate"
        }

    def export(self, code: CodeArtifact | None) -> list[dict]:
        return [record.public(code) for record in self.history]


def base_feedback_artifact(
    feedback: BaseTestFeedback,
    code: CodeArtifact,
    policy: str,
) -> dict[str, Any]:
    status = feedback.status.strip().lower()
    if feedback.passed:
        failure_kind = "none"
    elif "timeout" in status:
        failure_kind = "timeout"
    elif any(word in status for word in ("error", "exception")):
        failure_kind = "exception"
    elif status in {"fail", "failed"}:
        failure_kind = "assertion"
    else:
        failure_kind = "unknown"
    signature_payload = {
        "status": status,
        "failure_kind": failure_kind,
        "passed_tests": feedback.passed_tests,
        "total_tests": feedback.total_tests,
    }
    value: dict[str, Any] = {
        "candidate_version": code.version,
        "candidate_sha256": code.sha256,
        "status": feedback.status,
        "passed": feedback.passed,
        "passed_tests": feedback.passed_tests,
        "total_tests": feedback.total_tests,
        "failure_count": feedback.failure_count,
        "summary": feedback.summary,
    }
    if policy == "sanitized_diagnostic_v2":
        value.update(
            failure_kind=failure_kind,
            failure_signature=_stable_hash(signature_payload),
            diagnostic={
                "exception_class": None,
                "generated_code_line": None,
                "message": feedback.summary[:500],
                "redacted": True,
            },
        )
    return value


def finding_records(
    claims: tuple[str, ...],
    records: tuple[Mapping[str, Any], ...],
    *,
    kind: str,
    step: int,
) -> list[dict[str, Any]]:
    if records:
        return [dict(item) for item in records]
    category = "logic" if kind == "logic_review" else "edge_case"
    return [
        {
            "finding_id": f"{kind}-{step}-{index}",
            "category": category,
            "claim": claim,
            "code_span": None,
            "confidence": "unknown",
        }
        for index, claim in enumerate(claims, start=1)
    ]


def _bounded_preview(text: str, limit: int) -> dict[str, Any]:
    return {
        "preview": text[:limit],
        "visible_chars": min(len(text), limit),
        "total_chars": len(text),
        "truncated": len(text) > limit,
    }


@dataclass(frozen=True)
class RoutingStateV6:
    task: TaskSpec
    current_code: str | None
    code_version: int
    code_sha256: str | None
    presentation_fingerprint: str | None
    presentation_method: str | None
    base_feedback: BaseTestFeedback | None
    logic_review: tuple[str, ...]
    edge_cases: tuple[str, ...]
    action_history: tuple[str, ...]
    artifacts: Mapping[str, Any]
    role_progress: Mapping[str, Any]
    recent_events: tuple[Mapping[str, Any], ...]
    latest_generation_failure: Mapping[str, Any] | None
    budget: BudgetSnapshot
    code_preview_chars: int
    action_space: tuple[str, ...] = ROUTER_ACTIONS

    def __post_init__(self) -> None:
        if self.action_space != ROUTER_ACTIONS:
            raise ValueError("action_space must contain the full, fixed router action set")

    @property
    def task_id(self) -> str:
        return self.task.task_id

    @property
    def task_prompt(self) -> str:
        return self.task.prompt

    @property
    def entry_point(self) -> str:
        return self.task.entry_point

    def to_router_payload(self) -> dict[str, Any]:
        candidate = None
        if self.current_code is not None:
            candidate = {
                "exists": True,
                "version": self.code_version,
                "raw_sha256": self.code_sha256,
                "presentation_fingerprint": self.presentation_fingerprint,
                "presentation_method": self.presentation_method,
                **_bounded_preview(self.current_code, self.code_preview_chars),
            }
        return {
            "protocol_version": "v6",
            "task": asdict(self.task),
            "current_state": {
                "candidate": candidate,
                "latest_base_execution": (
                    self.artifacts.get("base_execution")
                    if self.artifacts
                    else None
                ),
                "artifacts": dict(self.artifacts),
                "latest_generation_failure": self.latest_generation_failure,
                "role_progress": dict(self.role_progress),
                "recent_events": [dict(item) for item in self.recent_events],
                "terminal": False,
            },
            "agent_profiles": router_capability_payload(),
            "budget": asdict(self.budget),
            "constraints": {"allowed_actions": list(self.action_space)},
            # Compatibility fields keep deterministic/legacy router policies usable.
            "task_id": self.task.task_id,
            "task_prompt": self.task.prompt,
            "entry_point": self.task.entry_point,
            "current_code": candidate["preview"] if candidate else None,
            "code_version": self.code_version,
            "base_feedback": asdict(self.base_feedback) if self.base_feedback else None,
            "logic_review": list(self.logic_review),
            "edge_cases": list(self.edge_cases),
            "action_history": list(self.action_history),
            "action_space": list(self.action_space),
        }


class ContextBudgetExceeded(RuntimeError):
    pass


def _iter_text_paths(value: Any, path: tuple[Any, ...] = ()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_text_paths(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_text_paths(item, path + (index,))
    elif isinstance(value, str):
        yield path, value


def _set_path(value: Any, path: tuple[Any, ...], replacement: Any) -> None:
    target = value
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = replacement


def pack_executor_messages(
    *,
    system_prompt: str,
    payload: Mapping[str, Any],
    input_cap: int,
    count_tokens: Callable[[list[dict[str, str]]], tuple[int, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Pack executor input deterministically while preserving critical metadata."""

    packed = copy.deepcopy(dict(payload))

    def messages() -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(packed, ensure_ascii=False, sort_keys=True),
            },
        ]

    current_messages = messages()
    count, method = count_tokens(current_messages)
    trimmed: list[dict[str, Any]] = []
    while count > input_cap:
        recent = packed.get("current_state", {}).get("recent_events")
        if isinstance(recent, list) and len(recent) > 2:
            recent.pop(0)
            trimmed.append({"field": "recent_events", "operation": "drop_oldest"})
        else:
            candidates = [
                (path, text)
                for path, text in _iter_text_paths(packed)
                if len(text) > 512
                and path[-1]
                in {"preview", "source", "claim", "message", "summary", "prompt"}
            ]
            if not candidates:
                raise ContextBudgetExceeded(
                    f"Executor context requires {count} tokens; cap is {input_cap}"
                )
            path, text = max(candidates, key=lambda item: len(item[1]))
            minimum = 2000 if path[-1] in {"preview", "source"} else 512
            new_length = max(minimum, len(text) // 2)
            if new_length >= len(text):
                raise ContextBudgetExceeded(
                    f"Executor context requires {count} tokens; cap is {input_cap}"
                )
            _set_path(packed, path, text[:new_length])
            trimmed.append(
                {
                    "field": ".".join(str(x) for x in path),
                    "operation": "truncate",
                    "original_chars": len(text),
                    "visible_chars": new_length,
                }
            )
        current_messages = messages()
        count, method = count_tokens(current_messages)
    return current_messages, {
        "protocol_version": "v6",
        "input_cap_tokens": input_cap,
        "packed_input_tokens": count,
        "token_count_method": method,
        "trimmed": trimmed,
        "delivered_evidence": packed.get("delivered_evidence", []),
        "packed_payload": packed,
        "messages": current_messages,
    }


def build_executor_payload(
    *,
    task: TaskSpec,
    code: CodeArtifact | None,
    ledger: ArtifactLedger,
    role_progress: Mapping[str, Any],
    recent_events: list[Mapping[str, Any]],
    config: ProtocolConfig,
) -> dict[str, Any]:
    candidate = None
    if code is not None:
        fingerprint, method = python_presentation_fingerprint(code.source)
        candidate = {
            "version": code.version,
            "raw_sha256": code.sha256,
            "presentation_fingerprint": fingerprint,
            "presentation_method": method,
            "source": code.source,
        }
    delivered = []
    evidence: dict[str, Any] = {}
    for kind in ("base_execution", "logic_review", "edge_case_analysis"):
        record = ledger.fresh(kind, code)
        if record is None:
            continue
        evidence[kind] = record.public(code)
        delivered.append(
            {
                "artifact_id": record.artifact_id,
                "kind": kind,
                "sha256": record.sha256,
                "fresh": True,
            }
        )
    return {
        "protocol_version": "v6",
        "task": asdict(task),
        "current_state": {
            "candidate": candidate,
            "evidence": evidence,
            "role_progress": copy.deepcopy(dict(role_progress)),
            "recent_events": copy.deepcopy(
                recent_events[-config.recent_event_limit :]
            ),
        },
        "delivered_evidence": delivered,
    }
