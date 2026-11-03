from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import AgentProfileV2, ScenarioSpecV2, STOP_ACTION


@dataclass(frozen=True)
class BenchmarkBundleV2:
    metadata: dict[str, Any]
    agents: dict[str, AgentProfileV2]
    scenarios: tuple[ScenarioSpecV2, ...]
    root: Path | None = None


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


def validate_bundle(bundle: BenchmarkBundleV2) -> None:
    if STOP_ACTION in bundle.agents:
        raise ValueError("STOP is reserved and cannot be an agent ID")
    scenario_ids: set[str] = set()
    for scenario in bundle.scenarios:
        if scenario.scenario_id in scenario_ids:
            raise ValueError(f"Duplicate scenario ID: {scenario.scenario_id}")
        scenario_ids.add(scenario.scenario_id)
        unknown = set(scenario.agent_pool).difference(bundle.agents)
        transition_agents = {
            item.action for item in scenario.transitions
        }.difference(scenario.agent_pool)
        if unknown or transition_agents:
            raise ValueError(
                f"Scenario {scenario.scenario_id} references unavailable agents: "
                f"{sorted(unknown.union(transition_agents))}"
            )
        missing_fact_descriptions = (
            set(scenario.initial_facts)
            .union(scenario.goal_facts)
            .union(
                fact
                for transition in scenario.transitions
                for fact in (
                    set(transition.requires_all)
                    | set(transition.requires_any)
                    | set(transition.forbids)
                    | set(transition.add_facts)
                    | set(transition.remove_facts)
                )
            )
            .difference(scenario.fact_descriptions)
        )
        if missing_fact_descriptions:
            raise ValueError(
                f"Scenario {scenario.scenario_id} lacks public descriptions for facts: "
                f"{sorted(missing_fact_descriptions)}"
            )
        if scenario.max_steps < 2 or scenario.max_cost <= 0:
            raise ValueError(f"Invalid budget in {scenario.scenario_id}")


def load_benchmark_v2(root: str | Path) -> BenchmarkBundleV2:
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    agent_rows = _read_jsonl(root / "agents.jsonl")
    scenario_rows = _read_jsonl(root / "scenarios.jsonl")
    agents = {item.agent_id: item for item in map(AgentProfileV2.from_dict, agent_rows)}
    if len(agents) != len(agent_rows):
        raise ValueError("Agent IDs must be unique")
    bundle = BenchmarkBundleV2(
        metadata=metadata,
        agents=agents,
        scenarios=tuple(ScenarioSpecV2.from_dict(row) for row in scenario_rows),
        root=root,
    )
    validate_bundle(bundle)
    return bundle
