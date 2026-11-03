from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence

from .contract_validator import (
    validate_action_labels,
    validate_interpreted_outcome,
)
from .schema import (
    EMPIRICAL_SCHEMA_VERSION,
    canonical_hash,
    empirical_decision_id,
    outcome_id_for,
    utc_now,
    validate_action_annotation,
    validate_outcome,
)


OUTCOME_PROMPT_VERSION = "routingbench_v4_outcome_interpreter_v1"
ACTION_PROMPT_VERSION = "routingbench_v4_valid_action_labeler_v2_applicability"
POST_TASK_PROMPT_VERSION = "routingbench_v4_post_task_semantic_judge_v1"


OUTCOME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "execution_status",
        "progress_status",
        "produced_artifacts",
        "observable_facts_added",
        "resolved_requirements",
        "new_open_requirements",
        "candidate_answer",
        "completion_claimed",
        "observable_issues",
        "state_delta_summary",
    ],
    "properties": {
        "execution_status": {
            "type": "string",
            "enum": [
                "success",
                "partial",
                "no_progress",
                "invalid_output",
                "tool_failure",
                "transport_failure",
            ],
        },
        "progress_status": {
            "type": "string",
            "enum": ["substantive_progress", "partial_progress", "no_progress", "failure"],
        },
        "produced_artifacts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["artifact_type", "content_summary", "evidence_refs"],
                "properties": {
                    "artifact_type": {"type": "string"},
                    "content_summary": {"type": "string"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
        },
        "observable_facts_added": {"type": "array", "items": {"type": "string"}},
        "resolved_requirements": {"type": "array", "items": {"type": "string"}},
        "new_open_requirements": {"type": "array", "items": {"type": "string"}},
        "candidate_answer": {"type": ["string", "null"]},
        "completion_claimed": {"type": "boolean"},
        "observable_issues": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "evidence"],
                "properties": {
                    "type": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "state_delta_summary": {"type": "string"},
    },
    "additionalProperties": False,
}


def _action_schema(candidate_actions: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["actions"],
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "action",
                        "label",
                        "reason_code",
                        "state_evidence",
                        "confidence",
                    ],
                    "properties": {
                        "action": {"type": "string", "enum": list(candidate_actions)},
                        "label": {
                            "type": "string",
                            "enum": ["valid", "invalid", "uncertain"],
                        },
                        "reason_code": {"type": "string"},
                        "state_evidence": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "additionalProperties": False,
                },
            }
        },
        "additionalProperties": False,
    }


class StructuredJudgeClient:
    def __init__(self, model: str, reasoning_effort: str = "low") -> None:
        # Import after .env loading.
        from model.query_manager import query_manager

        self.query_manager = query_manager
        self.model = model
        self.reasoning_effort = reasoning_effort

    def _provenance(
        self, *, prompt_version: str, prompt_hash: str, tokens: int, latency_ms: float
    ) -> dict[str, Any]:
        info = self.query_manager.get_model_info(self.model) or {}
        return {
            "model": self.model,
            "api_model_name": info.get("api_model_name"),
            "provider": info.get("provider"),
            "prompt_version": prompt_version,
            "prompt_hash": prompt_hash,
            "reasoning_effort": self.reasoning_effort,
            "total_tokens": int(tokens or 0),
            "latency_ms": latency_ms,
            "judged_at": utc_now(),
        }

    def interpret_outcome(self, output: Mapping[str, Any]) -> dict[str, Any]:
        context = dict(output.get("collection_context") or {})
        payload = {
            "task": context.get("task"),
            "task_input": context.get("task_input", {}),
            "pre_execution_observable_state": context.get("pre_state", {}),
            "executed_role_contract": context.get("role_profile", {}),
            "transport_status": output.get("transport_status"),
            "executor_output": output.get("raw_output", ""),
            "rules": [
                "Use only information observable in the supplied state and executor output.",
                "Do not infer a gold answer, hidden fault class, future transition, or next agent.",
                "A plausible but unsupported claim is an observable issue, not a verified fact.",
                "Mark success only when the output contains a substantive role contribution.",
                "Resolved requirements must quote an existing open requirement exactly.",
            ],
        }
        prompt_hash = canonical_hash(
            {"version": OUTCOME_PROMPT_VERSION, "payload": payload}, length=64
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a gold-blind workflow outcome interpreter. Convert an executor "
                    "response into observable state changes. Never recommend routing actions."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        started = time.perf_counter()
        parsed, tokens = self.query_manager.query_structured(
            self.model,
            messages,
            OUTCOME_SCHEMA,
            schema_name="routingbench_v4_empirical_outcome",
            reasoning_effort=self.reasoning_effort,
        )
        latency = (time.perf_counter() - started) * 1000.0
        record = {
            "schema_version": EMPIRICAL_SCHEMA_VERSION,
            "outcome_id": outcome_id_for(str(output["output_id"]), prompt_hash),
            "output_id": str(output["output_id"]),
            **parsed,
            "judge_provenance": self._provenance(
                prompt_version=OUTCOME_PROMPT_VERSION,
                prompt_hash=prompt_hash,
                tokens=tokens,
                latency_ms=latency,
            ),
            "review_status": "pending",
        }
        validation = validate_interpreted_outcome(
            raw_output=str(output.get("raw_output", "")),
            outcome=record,
            pre_state=context.get("pre_state") or {},
            role_profile=context.get("role_profile") or {},
        )
        record["contract_validation"] = validation
        if validation["requires_review"]:
            record["review_status"] = "priority_review"
        validate_outcome(record)
        return record

    def label_actions(
        self,
        *,
        decision_id: str | None,
        public_observation: Mapping[str, Any],
        candidate_actions: Sequence[str],
    ) -> dict[str, Any]:
        candidates = tuple(map(str, candidate_actions))
        payload = {
            "public_observation": dict(public_observation),
            "candidate_actions": list(candidates),
            "label_definition": {
                "valid": (
                    "The role has a real capability applicable to the visible state and its visible "
                    "preconditions are satisfied. An applicable action remains valid even when it is "
                    "inefficient, redundant, or may make no progress; those are execution/efficiency "
                    "properties, not pre-execution rejection reasons. STOP is valid only when visible "
                    "terminal conditions are satisfied."
                ),
                "invalid": (
                    "The role cannot perform the needed operation, a required visible precondition "
                    "is absent, executing it would be unsafe, or STOP is premature."
                ),
                "uncertain": "The public observation is insufficient to decide fairly.",
            },
            "rules": [
                "Judge each action independently; multiple actions may be valid.",
                "Use only the public observation and role cards.",
                "Do not infer hidden gold correctness, executor model identity, or fault labels.",
                "Do not rank actions and do not create utilities.",
                "Do not mark an applicable action invalid merely because another action is better or more efficient.",
                "Distinguish pre-execution inapplicability from post-execution failure or no progress.",
                "Return exactly one annotation for every candidate action.",
            ],
        }
        prompt_hash = canonical_hash(
            {"version": ACTION_PROMPT_VERSION, "payload": payload}, length=64
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a gold-blind benchmark annotation assistant. Determine the set of "
                    "acceptable next actions from observable workflow state."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        started = time.perf_counter()
        parsed, tokens = self.query_manager.query_structured(
            self.model,
            messages,
            _action_schema(candidates),
            schema_name="routingbench_v4_empirical_actions",
            reasoning_effort=self.reasoning_effort,
        )
        latency = (time.perf_counter() - started) * 1000.0
        annotations = list(parsed.get("actions") or ())
        validation = validate_action_labels(annotations, candidates)
        if not validation["valid"]:
            raise ValueError(
                "Invalid action-labeler response: " + ", ".join(validation["errors"])
            )
        proposed = [
            str(item["action"]) for item in annotations if item["label"] == "valid"
        ]
        record = {
            "schema_version": EMPIRICAL_SCHEMA_VERSION,
            "decision_id": decision_id or empirical_decision_id(prompt_hash),
            "candidate_actions": list(candidates),
            "action_annotations": annotations,
            "proposed_valid_actions": proposed,
            "judge_provenance": self._provenance(
                prompt_version=ACTION_PROMPT_VERSION,
                prompt_hash=prompt_hash,
                tokens=tokens,
                latency_ms=latency,
            ),
            "review_status": (
                "priority_review"
                if any(item["label"] == "uncertain" for item in annotations)
                else "pending"
            ),
        }
        validate_action_annotation(record)
        return record

    def judge_post_task(
        self,
        *,
        task: str,
        task_input: Mapping[str, Any],
        final_state: Mapping[str, Any],
        trajectory_outputs: Sequence[Mapping[str, Any]],
        reference_answer: str | None = None,
        answer_aliases: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Judge the committed result after routing; gold never feeds back into state."""

        schema = {
            "type": "object",
            "required": [
                "task_completed",
                "semantic_correctness",
                "answer_supported",
                "committed_answer",
                "rationale",
            ],
            "properties": {
                "task_completed": {"type": "boolean"},
                "semantic_correctness": {"type": ["boolean", "null"]},
                "answer_supported": {"type": "boolean"},
                "committed_answer": {"type": ["string", "null"]},
                "rationale": {"type": "string"},
            },
            "additionalProperties": False,
        }
        payload = {
            "task": task,
            "task_input": task_input,
            "final_observable_state": final_state,
            "trajectory_outputs": list(trajectory_outputs),
            "reference_answer": reference_answer,
            "answer_aliases": list(answer_aliases),
            "rules": [
                "This judgment occurs only after the trajectory has terminated.",
                "Set semantic_correctness to null when no reference or sufficient deterministic evidence exists.",
                "Do not change the trajectory, state, or routing labels.",
                "Judge the committed answer, not an intermediate answer that was never committed.",
            ],
        }
        prompt_hash = canonical_hash(
            {"version": POST_TASK_PROMPT_VERSION, "payload": payload}, length=64
        )
        messages = [
            {
                "role": "system",
                "content": "You are a post-task semantic evaluator. Your judgment cannot affect execution.",
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        started = time.perf_counter()
        parsed, tokens = self.query_manager.query_structured(
            self.model,
            messages,
            schema,
            schema_name="routingbench_v4_post_task_judgment",
            reasoning_effort=self.reasoning_effort,
        )
        latency = (time.perf_counter() - started) * 1000.0
        return {
            **parsed,
            "judge_provenance": self._provenance(
                prompt_version=POST_TASK_PROMPT_VERSION,
                prompt_hash=prompt_hash,
                tokens=tokens,
                latency_ms=latency,
            ),
        }
