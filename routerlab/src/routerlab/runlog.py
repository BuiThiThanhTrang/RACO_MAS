from __future__ import annotations

import json
import re
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
        if result.protocol_data:
            self._log_protocol_episode(result, public, private)

    def _log_protocol_episode(
        self,
        result: EpisodeResult,
        public: dict[str, Any],
        private: dict[str, Any],
    ) -> None:
        """Write the auditable v6 layout without exposing final Plus results."""

        safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", result.task_id).strip("_")
        episode_dir = self.run_dir / "episodes" / (safe_task_id or "task")
        artifact_dir = episode_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        data = dict(result.protocol_data)
        self._write_jsonl(
            episode_dir / "state_snapshots.jsonl",
            list(data.get("state_snapshots", [])),
        )
        self._write_jsonl(
            episode_dir / "trajectory.jsonl",
            list(data.get("events", [])),
        )
        self._write_jsonl(
            episode_dir / "requests.jsonl",
            list(data.get("requests", [])),
        )
        self._write_jsonl(
            episode_dir / "decisions.jsonl",
            [
                self._public_step(step)
                for step in result.steps
                if step.router_decision
            ],
        )
        self._write_json(episode_dir / "result.json", public)
        self._write_json(episode_dir / "private_final_eval.json", private)
        self._write_json(
            episode_dir / "role_progress.json",
            dict(data.get("role_progress", {})),
        )
        for artifact in data.get("artifacts", []):
            if not isinstance(artifact, dict):
                continue
            artifact_id = re.sub(
                r"[^A-Za-z0-9_.-]+", "_", str(artifact.get("artifact_id", "artifact"))
            )
            content = artifact.get("content")
            if artifact.get("kind") == "candidate" and isinstance(content, dict):
                source = content.get("source")
                if isinstance(source, str):
                    (artifact_dir / f"{artifact_id}.py").write_text(
                        source, encoding="utf-8"
                    )
            self._write_json(artifact_dir / f"{artifact_id}.json", artifact)

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

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(
                payload,
                default=_json_default,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_jsonl(path: Path, payloads: list[Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(
                    json.dumps(payload, default=_json_default, ensure_ascii=False)
                )
                handle.write("\n")
