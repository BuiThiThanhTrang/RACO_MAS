from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .contracts import EpisodeResult, StepRecord


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class RunLogger:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.public_path = self.run_dir / "public_trace.jsonl"
        self.private_path = self.run_dir / "private_eval.jsonl"
        self.solutions_path = self.run_dir / "solutions.jsonl"

    def log_episode(self, result: EpisodeResult) -> None:
        public = {
            "task_id": result.task_id,
            "method": result.method,
            "steps": [self._public_step(step) for step in result.steps],
            "termination_reason": result.termination_reason,
            "final_code_sha256": (
                result.final_code.sha256 if result.final_code else None
            ),
            "usage": {
                "steps": result.total_steps,
                "model_calls": result.total_model_calls,
                "input_tokens": result.total_input_tokens,
                "output_tokens": result.total_output_tokens,
                "base_executions": result.total_base_executions,
                "cost_usd": result.total_cost_usd,
                "wall_time_ms": result.wall_time_ms,
            },
        }
        private = {
            "task_id": result.task_id,
            "initial_evaluation": asdict(result.initial_evaluation)
            if result.initial_evaluation
            else None,
            "final_evaluation": asdict(result.final_evaluation)
            if result.final_evaluation
            else None,
        }
        self._append(self.public_path, public)
        self._append(self.private_path, private)
        self._append(
            self.solutions_path,
            {
                "task_id": result.task_id,
                "solution": result.final_code.source if result.final_code else None,
                "code_sha256": result.final_code.sha256 if result.final_code else None,
                "status": "submitted" if result.final_code else "no_submission",
            },
        )

    @staticmethod
    def _public_step(step: StepRecord) -> dict[str, Any]:
        value = asdict(step)
        # Step metadata is generated only from public workflow state.
        return value

    @staticmethod
    def _append(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=_json_default, ensure_ascii=False))
            handle.write("\n")
