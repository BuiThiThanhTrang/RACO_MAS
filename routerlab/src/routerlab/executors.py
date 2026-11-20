from __future__ import annotations

import json
from dataclasses import asdict

from .config import ProtocolConfig
from .contracts import (
    BaseTestFeedback,
    CodeArtifact,
    ExecutorResult,
    ModelUsage,
    TaskSpec,
)
from .model import ModelClient
from .parsing import extract_code, parse_findings, parse_structured_findings
from .pool import executor_map
from .prompts import load_prompt
from .protocol_v6 import ContextBudgetExceeded, pack_executor_messages


def _input_cap(client: ModelClient, protocol: ProtocolConfig) -> int:
    config = getattr(client, "config", None)
    if config is None:
        return 6912
    return max(
        256,
        int(config.context_limit)
        - int(config.max_output_tokens)
        - protocol.framing_reserve_tokens,
    )


def _messages_and_audit(
    client: ModelClient,
    *,
    prompt: str,
    payload: dict,
    protocol: ProtocolConfig | None,
) -> tuple[list[dict[str, str]], dict]:
    if protocol is None or protocol.version != "v6":
        return (
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            {},
        )
    return pack_executor_messages(
        system_prompt=prompt,
        payload=payload,
        input_cap=_input_cap(client, protocol),
        count_tokens=client.count_tokens,
    )


def _context_failure(executor_id: str, exc: ContextBudgetExceeded) -> ExecutorResult:
    return ExecutorResult(
        executor_id=executor_id,
        raw_output="",
        usage=ModelUsage(),
        latency_ms=0.0,
        no_op=True,
        no_op_reason="context_budget_exceeded",
        metadata={
            "protocol_version": "v6",
            "failure_origin": "context_budget",
            "failure_type": type(exc).__name__,
            "error": str(exc),
        },
    )


class ExecutorPool:
    def __init__(self, client: ModelClient) -> None:
        self.client = client
        self.specs = executor_map()

    def run(
        self,
        executor_id: str,
        task: TaskSpec,
        code: CodeArtifact | None,
        base_feedback: BaseTestFeedback | None = None,
        logic_review: tuple[str, ...] = (),
        edge_cases: tuple[str, ...] = (),
        *,
        protocol_context: dict | None = None,
        protocol_config: ProtocolConfig | None = None,
    ) -> ExecutorResult:
        if executor_id not in self.specs:
            raise ValueError(f"Unknown executor: {executor_id}")
        use_v6 = bool(protocol_config and protocol_config.version == "v6")
        payload = protocol_context or {
            "task": asdict(task),
            "current_code": code.source if code else None,
            "code_version": code.version if code else None,
            "base_feedback": asdict(base_feedback) if base_feedback else None,
            "logic_review": list(logic_review),
            "edge_cases": list(edge_cases),
        }
        prompt = load_prompt(f"{executor_id}_v6" if use_v6 else executor_id)
        try:
            messages, request_metadata = _messages_and_audit(
                self.client,
                prompt=prompt,
                payload=payload,
                protocol=protocol_config,
            )
        except ContextBudgetExceeded as exc:
            return _context_failure(executor_id, exc)
        response = self.client.generate(messages)
        output_kind = self.specs[executor_id].output_kind
        no_code_to_repair = executor_id == "repair_agent" and code is None
        claims: tuple[str, ...] = ()
        records = ()
        if output_kind == "findings":
            if use_v6:
                claims, records = parse_structured_findings(response.text)
            else:
                claims = parse_findings(response.text)
        return ExecutorResult(
            executor_id=executor_id,
            raw_output=response.text,
            usage=response.usage,
            latency_ms=response.latency_ms,
            code=(
                extract_code(response.text)
                if output_kind == "code" and not no_code_to_repair
                else None
            ),
            findings=claims,
            no_op=no_code_to_repair,
            no_op_reason="no_code_to_repair" if no_code_to_repair else "",
            finding_records=records,
            metadata=request_metadata,
        )


class SingleAgent:
    def __init__(self, client: ModelClient) -> None:
        self.client = client

    def run(
        self,
        task: TaskSpec,
        code: CodeArtifact | None,
        base_feedback: BaseTestFeedback | None,
        *,
        protocol_context: dict | None = None,
        protocol_config: ProtocolConfig | None = None,
    ) -> ExecutorResult:
        use_v6 = bool(protocol_config and protocol_config.version == "v6")
        payload = protocol_context or {
            "task": asdict(task),
            "current_code": code.source if code else None,
            "code_version": code.version if code else None,
            "base_feedback": asdict(base_feedback) if base_feedback else None,
        }
        prompt = load_prompt("single_agent_v6" if use_v6 else "single_agent")
        try:
            messages, request_metadata = _messages_and_audit(
                self.client,
                prompt=prompt,
                payload=payload,
                protocol=protocol_config,
            )
        except ContextBudgetExceeded as exc:
            return _context_failure("single_agent", exc)
        response = self.client.generate(messages)
        return ExecutorResult(
            executor_id="single_agent",
            raw_output=response.text,
            usage=response.usage,
            latency_ms=response.latency_ms,
            code=extract_code(response.text),
            metadata=request_metadata,
        )
