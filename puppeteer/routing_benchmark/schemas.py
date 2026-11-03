from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


STOP_ACTION = "STOP"


def state_fingerprint(facts: set[str] | frozenset[str]) -> str:
    payload = json.dumps(sorted(facts), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class AgentContract:
    agent_id: str
    role_name: str
    role_goal: str
    use_when: tuple[str, ...]
    avoid_when: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    cost: float = 1.0
    distractor: bool = False

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AgentContract":
        return cls(
            agent_id=str(raw["agent_id"]),
            role_name=str(raw["role_name"]),
            role_goal=str(raw["role_goal"]),
            use_when=tuple(map(str, raw.get("use_when") or ())),
            avoid_when=tuple(map(str, raw.get("avoid_when") or ())),
            expected_outputs=tuple(map(str, raw.get("expected_outputs") or ())),
            cost=float(raw.get("cost", 1.0)),
            distractor=bool(raw.get("distractor", False)),
        )

    def public_profile(self) -> dict[str, Any]:
        """Return a qualitative role card; hidden distractor labels stay private."""
        return {
            "agent_id": self.agent_id,
            "role_name": self.role_name,
            "role_goal": self.role_goal,
            "use_when": list(self.use_when),
            "avoid_when": list(self.avoid_when),
            "expected_outputs": list(self.expected_outputs),
            "cost": self.cost,
        }


@dataclass(frozen=True)
class TransitionSpec:
    transition_id: str
    action: str
    requires_all: frozenset[str] = field(default_factory=frozenset)
    requires_any: frozenset[str] = field(default_factory=frozenset)
    forbids: frozenset[str] = field(default_factory=frozenset)
    add: frozenset[str] = field(default_factory=frozenset)
    remove: frozenset[str] = field(default_factory=frozenset)
    status: str = "SUCCESS"
    cost_override: float | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TransitionSpec":
        override = raw.get("cost_override")
        return cls(
            transition_id=str(raw["transition_id"]),
            action=str(raw["action"]),
            requires_all=frozenset(map(str, raw.get("requires_all") or ())),
            requires_any=frozenset(map(str, raw.get("requires_any") or ())),
            forbids=frozenset(map(str, raw.get("forbids") or ())),
            add=frozenset(map(str, raw.get("add") or ())),
            remove=frozenset(map(str, raw.get("remove") or ())),
            status=str(raw.get("status", "SUCCESS")),
            cost_override=float(override) if override is not None else None,
        )

    def matches(self, facts: set[str] | frozenset[str]) -> bool:
        return (
            self.requires_all.issubset(facts)
            and (not self.requires_any or bool(self.requires_any.intersection(facts)))
            and not self.forbids.intersection(facts)
        )


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    family: str
    task: str
    agent_pool: tuple[str, ...]
    initial_facts: frozenset[str]
    goal_facts: frozenset[str]
    milestone_facts: frozenset[str]
    transitions: tuple[TransitionSpec, ...]
    max_steps: int
    max_cost: float
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ScenarioSpec":
        config = raw.get("config") or {}
        return cls(
            scenario_id=str(raw["scenario_id"]),
            family=str(raw["family"]),
            task=str(raw["task"]),
            agent_pool=tuple(map(str, raw["agent_pool"])),
            initial_facts=frozenset(map(str, raw.get("initial_facts") or ())),
            goal_facts=frozenset(map(str, raw.get("goal_facts") or ())),
            milestone_facts=frozenset(map(str, raw.get("milestone_facts") or ())),
            transitions=tuple(
                TransitionSpec.from_dict(item) for item in raw.get("transitions") or ()
            ),
            max_steps=int(config.get("max_steps", 8)),
            max_cost=float(config.get("max_cost", 20.0)),
            tags=tuple(map(str, raw.get("tags") or ())),
        )


@dataclass(frozen=True)
class StepEvent:
    step: int
    action: str
    status: str
    valid_action: bool
    valid_actions_before: tuple[str, ...]
    state_before: str
    state_after: str
    facts_added: tuple[str, ...]
    facts_removed: tuple[str, ...]
    state_changed: bool
    redundant: bool
    distractor: bool
    cost: float
    newly_achieved_milestones: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "status": self.status,
            "valid_action": self.valid_action,
            "valid_actions_before": list(self.valid_actions_before),
            "state_before": self.state_before,
            "state_after": self.state_after,
            "facts_added": list(self.facts_added),
            "facts_removed": list(self.facts_removed),
            "state_changed": self.state_changed,
            "redundant": self.redundant,
            "distractor": self.distractor,
            "cost": self.cost,
            "newly_achieved_milestones": list(self.newly_achieved_milestones),
        }
