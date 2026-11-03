from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from routing_benchmark.v2.loader import BenchmarkBundleV2, validate_bundle
from routing_benchmark.v2.schemas import AgentProfileV2, ScenarioSpecV2

from .schemas import ProvenanceRecordV3


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return rows


@dataclass(frozen=True)
class BenchmarkBundleV3:
    metadata: dict[str, Any]
    agents: dict[str, AgentProfileV2]
    scenarios: tuple[ScenarioSpecV2, ...]
    provenance: dict[str, ProvenanceRecordV3]
    decision_rows: tuple[dict[str, Any], ...]
    task_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    evaluation: dict[str, dict[str, Any]] = field(default_factory=dict)
    root: Path | None = None

    def as_v2_bundle(self) -> BenchmarkBundleV2:
        return BenchmarkBundleV2(
            metadata=dict(self.metadata),
            agents=self.agents,
            scenarios=self.scenarios,
            root=self.root,
        )


def load_benchmark_v3(root: str | Path) -> BenchmarkBundleV3:
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    agent_rows = _read_jsonl(root / "agents.jsonl")
    scenario_rows = _read_jsonl(root / "scenarios.jsonl")
    provenance_rows = _read_jsonl(root / "provenance.jsonl")
    decisions = tuple(_read_jsonl(root / "decision_states.jsonl"))
    task_input_path = root / "task_inputs.jsonl"
    evaluation_path = root / "evaluation.jsonl"
    task_inputs = {
        str(item["scenario_id"]): dict(item.get("task_input") or {})
        for item in (_read_jsonl(task_input_path) if task_input_path.exists() else ())
    }
    evaluation = {
        str(item["scenario_id"]): dict(item)
        for item in (_read_jsonl(evaluation_path) if evaluation_path.exists() else ())
    }
    agents = {item.agent_id: item for item in map(AgentProfileV2.from_dict, agent_rows)}
    scenarios = tuple(ScenarioSpecV2.from_dict(item) for item in scenario_rows)
    provenance = {
        item.provenance_id: item
        for item in map(ProvenanceRecordV3.from_dict, provenance_rows)
    }
    bundle = BenchmarkBundleV3(
        metadata=metadata,
        agents=agents,
        scenarios=scenarios,
        provenance=provenance,
        decision_rows=decisions,
        task_inputs=task_inputs,
        evaluation=evaluation,
        root=root,
    )
    validate_bundle(bundle.as_v2_bundle())
    return bundle
