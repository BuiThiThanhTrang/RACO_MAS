from __future__ import annotations

import json
from dataclasses import asdict

from .contracts import (
    BaseTestFeedback,
    CodeArtifact,
    ExecutorResult,
    TaskSpec,
)
from .model import ModelClient
from .parsing import extract_code, parse_findings
from .pool import executor_map
from .prompts import load_prompt


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
    ) -> ExecutorResult:
        if executor_id not in self.specs:
            raise ValueError(f"Unknown executor: {executor_id}")
        payload = {
            "task": asdict(task),
            "current_code": code.source if code else None,
            "code_version": code.version if code else None,
            "base_feedback": asdict(base_feedback) if base_feedback else None,
            "logic_review": list(logic_review),
            "edge_cases": list(edge_cases),
        }
        response = self.client.generate(
            [
                {"role": "system", "content": load_prompt(executor_id)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
        output_kind = self.specs[executor_id].output_kind
        no_code_to_repair = executor_id == "repair_agent" and code is None
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
            findings=parse_findings(response.text) if output_kind == "findings" else (),
            no_op=no_code_to_repair,
            no_op_reason="no_code_to_repair" if no_code_to_repair else "",
        )


class SingleAgent:
    def __init__(self, client: ModelClient) -> None:
        self.client = client

    def run(
        self,
        task: TaskSpec,
        code: CodeArtifact | None,
        base_feedback: BaseTestFeedback | None,
    ) -> ExecutorResult:
        payload = {
            "task": asdict(task),
            "current_code": code.source if code else None,
            "code_version": code.version if code else None,
            "base_feedback": asdict(base_feedback) if base_feedback else None,
        }
        response = self.client.generate(
            [
                {"role": "system", "content": load_prompt("single_agent")},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
        return ExecutorResult(
            executor_id="single_agent",
            raw_output=response.text,
            usage=response.usage,
            latency_ms=response.latency_ms,
            code=extract_code(response.text),
        )
