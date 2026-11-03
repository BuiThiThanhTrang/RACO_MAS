from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .schemas import AgentContract, ScenarioSpec, STOP_ACTION


@dataclass(frozen=True)
class BenchmarkBundle:
    version: str
    agents: dict[str, AgentContract]
    scenarios: tuple[ScenarioSpec, ...]
    root: Path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def load_benchmark(root: str | Path) -> BenchmarkBundle:
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    agent_rows = _read_jsonl(root / "agents.jsonl")
    scenario_rows = _read_jsonl(root / "scenarios.jsonl")
    agents = {item.agent_id: item for item in map(AgentContract.from_dict, agent_rows)}
    if len(agents) != len(agent_rows):
        raise ValueError("Agent IDs must be unique")
    if STOP_ACTION in agents:
        raise ValueError(f"{STOP_ACTION!r} is reserved and cannot be an agent ID")

    scenarios = tuple(map(ScenarioSpec.from_dict, scenario_rows))
    scenario_ids = [item.scenario_id for item in scenarios]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Scenario IDs must be unique")
    for scenario in scenarios:
        unknown_pool = set(scenario.agent_pool).difference(agents)
        unknown_actions = {
            transition.action for transition in scenario.transitions
        }.difference(agents)
        if unknown_pool or unknown_actions:
            raise ValueError(
                f"Scenario {scenario.scenario_id} references unknown agents: "
                f"{sorted(unknown_pool.union(unknown_actions))}"
            )
        outside_pool = {
            transition.action for transition in scenario.transitions
        }.difference(scenario.agent_pool)
        if outside_pool:
            raise ValueError(
                f"Scenario {scenario.scenario_id} transitions use agents outside pool: "
                f"{sorted(outside_pool)}"
            )
        if not scenario.goal_facts:
            raise ValueError(f"Scenario {scenario.scenario_id} has no goal facts")
        if scenario.max_steps < 1 or scenario.max_cost <= 0:
            raise ValueError(f"Scenario {scenario.scenario_id} has invalid budgets")
    return BenchmarkBundle(
        version=str(metadata.get("version", "unknown")),
        agents=agents,
        scenarios=scenarios,
        root=root,
    )
