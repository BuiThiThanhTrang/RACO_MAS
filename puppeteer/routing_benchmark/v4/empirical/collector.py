from __future__ import annotations

import json
import time
from typing import Any, Mapping

from .schema import (
    EMPIRICAL_SCHEMA_VERSION,
    canonical_hash,
    output_id_for,
    utc_now,
    validate_empirical_output,
)


ACTOR_PROMPT_VERSION = "routingbench_v4_empirical_actor_v1"


class EmpiricalActorClient:
    """Collect natural-language work products without asking for routing advice."""

    def __init__(self, model: str) -> None:
        # Import after the caller loads .env. model.api_config resolves secrets at import.
        from model.query_manager import query_manager

        self.query_manager = query_manager
        self.model = model

    def collect(
        self,
        *,
        scenario_id: str,
        workflow_id: str,
        task: str,
        task_input: Mapping[str, Any],
        pre_state: Mapping[str, Any],
        action: str,
        role_profile: Mapping[str, Any],
        replicate_id: int = 0,
    ) -> dict[str, Any]:
        pre_fingerprint = str(pre_state.get("fingerprint") or canonical_hash(pre_state))
        executor_profile_id = f"{self.model}:actor-prompt-v1"
        output_id = output_id_for(
            workflow_id=workflow_id,
            pre_state_fingerprint=pre_fingerprint,
            action=action,
            executor_profile_id=executor_profile_id,
            replicate_id=replicate_id,
        )
        payload = {
            "task": task,
            "task_input": task_input,
            "current_observable_state": pre_state,
            "your_role": {
                "role_name": role_profile.get("role_name"),
                "role_goal": role_profile.get("role_goal"),
                "use_when": role_profile.get("use_when", []),
                "avoid_when": role_profile.get("avoid_when", []),
                "supported_inputs": role_profile.get("supported_inputs", []),
                "expected_outputs": role_profile.get("expected_outputs", []),
            },
            "instructions": (
                "Perform this role on the supplied task and state. Return the actual work "
                "product or finding in natural language. Preserve concrete evidence, values, "
                "citations, identifiers, and observable failure details. Do not recommend which "
                "agent should be called next, do not discuss routing, and do not invent missing "
                "records or tool results. If required input is absent, state exactly what is "
                "missing and what you were unable to produce."
            ),
        }
        prompt_hash = canonical_hash(
            {"version": ACTOR_PROMPT_VERSION, "payload": payload}, length=64
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an executor inside a multi-agent workflow. Produce only your "
                    "task contribution, never orchestration advice."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        started = time.perf_counter()
        raw_output = ""
        tokens = 0
        status = "success"
        error: str | None = None
        try:
            raw_output, tokens = self.query_manager.query(self.model, messages)
            raw_output = str(raw_output or "")
            if not raw_output.strip():
                status = "empty_response"
        except Exception as exc:  # retained as an empirical transport outcome
            status = "transport_failure"
            error = f"{type(exc).__name__}: {exc}"
        elapsed = (time.perf_counter() - started) * 1000.0
        info = self.query_manager.get_model_info(self.model) or {}
        record = {
            "schema_version": EMPIRICAL_SCHEMA_VERSION,
            "output_id": output_id,
            "scenario_id": scenario_id,
            "workflow_id": workflow_id,
            "pre_state_fingerprint": pre_fingerprint,
            "selected_action": action,
            "executor_profile_id": executor_profile_id,
            "replicate_id": int(replicate_id),
            "output_origin": "empirical",
            "raw_output": raw_output,
            "generation": {
                "model": self.model,
                "api_model_name": info.get("api_model_name"),
                "provider": info.get("provider"),
                "prompt_version": ACTOR_PROMPT_VERSION,
                "prompt_hash": prompt_hash,
                "collected_at": utc_now(),
            },
            "transport_status": status,
            "transport_error": error,
            "token_usage": {"total_tokens": int(tokens or 0)},
            "latency_ms": elapsed,
            # Inputs are included in the private collection bank so interpretation is
            # reproducible. Release packaging strips executor identity and private fields.
            "collection_context": {
                "task": task,
                "task_input": dict(task_input),
                "pre_state": dict(pre_state),
                "role_profile": dict(role_profile),
            },
        }
        validate_empirical_output(record)
        return record
