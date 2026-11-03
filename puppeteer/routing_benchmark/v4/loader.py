from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from routing_benchmark.v2.loader import BenchmarkBundleV2, validate_bundle
from routing_benchmark.v2.schemas import AgentProfileV2, ScenarioSpecV2
from routing_benchmark.v3.schemas import ProvenanceRecordV3


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return rows


@dataclass(frozen=True)
class BenchmarkBundleV4:
    metadata: dict[str, Any]
    agents: dict[str, AgentProfileV2]
    scenarios: tuple[ScenarioSpecV2, ...]
    provenance: dict[str, ProvenanceRecordV3]
    decision_rows: tuple[dict[str, Any], ...]
    executor_outputs: tuple[dict[str, Any], ...]
    recovery_annotations: dict[str, dict[str, Any]]
    task_inputs: dict[str, dict[str, Any]]
    post_fault_inputs: tuple[dict[str, Any], ...] = ()
    post_fault_runtime: tuple[dict[str, Any], ...] = ()
    post_fault_labels: tuple[dict[str, Any], ...] = ()
    post_fault_pairs: tuple[dict[str, Any], ...] = ()
    root: Path | None = None

    def as_v2_bundle(self) -> BenchmarkBundleV2:
        return BenchmarkBundleV2(
            metadata=dict(self.metadata),
            agents=self.agents,
            scenarios=self.scenarios,
            root=self.root,
        )


def load_benchmark_v4(root: str | Path) -> BenchmarkBundleV4:
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    agents = {
        item.agent_id: item
        for item in map(AgentProfileV2.from_dict, _read_jsonl(root / "agents.jsonl"))
    }
    scenarios = tuple(
        ScenarioSpecV2.from_dict(item) for item in _read_jsonl(root / "scenarios.jsonl")
    )
    provenance = {
        item.provenance_id: item
        for item in map(
            ProvenanceRecordV3.from_dict, _read_jsonl(root / "provenance.jsonl")
        )
    }
    recovery = {
        str(item["scenario_id"]): item
        for item in _read_jsonl(root / "recovery_annotations.jsonl")
    }
    task_inputs = {
        str(item["scenario_id"]): dict(item.get("task_input") or {})
        for item in _read_jsonl(root / "task_inputs.jsonl")
    }
    optional_rows = {
        name: tuple(_read_jsonl(root / name)) if (root / name).exists() else ()
        for name in (
            "post_fault_inputs.jsonl",
            "post_fault_runtime.jsonl",
            "post_fault_labels.jsonl",
            "post_fault_pair_manifest.jsonl",
        )
    }
    bundle = BenchmarkBundleV4(
        metadata=metadata,
        agents=agents,
        scenarios=scenarios,
        provenance=provenance,
        decision_rows=tuple(_read_jsonl(root / "decision_states.jsonl")),
        executor_outputs=tuple(_read_jsonl(root / "executor_outputs.jsonl")),
        recovery_annotations=recovery,
        task_inputs=task_inputs,
        post_fault_inputs=optional_rows["post_fault_inputs.jsonl"],
        post_fault_runtime=optional_rows["post_fault_runtime.jsonl"],
        post_fault_labels=optional_rows["post_fault_labels.jsonl"],
        post_fault_pairs=optional_rows["post_fault_pair_manifest.jsonl"],
        root=root,
    )
    validate_bundle(bundle.as_v2_bundle())
    return bundle
