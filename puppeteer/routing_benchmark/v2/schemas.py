from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


STOP_ACTION = "STOP"
FAILURE_STATUSES = frozenset(
    {
        "NO_PROGRESS",
        "RETRYABLE_FAILURE",
        "INVALID_OUTPUT",
        "HARD_FAILURE",
        "EXECUTED_NO_PROGRESS",
        "EXECUTED_RETRYABLE_FAILURE",
        "EXECUTED_INVALID_OUTPUT",
        "EXECUTED_HARD_FAILURE",
    }
)
ROUTING_REJECTION_STATUSES = frozenset({"NOT_APPLICABLE", "ACTION_REJECTED"})


def _tuple(raw: Any) -> tuple[str, ...]:
    return tuple(map(str, raw or ()))


@dataclass(frozen=True)
class AgentProfileV2:
    agent_id: str
    role_name: str
    role_goal: str
    use_when: tuple[str, ...]
    avoid_when: tuple[str, ...]
    supported_inputs: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    cost_class: str
    action_cost: float
    semantic_family: str
    distractor_only: bool = False

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AgentProfileV2":
        return cls(
            agent_id=str(raw["agent_id"]),
            role_name=str(raw["role_name"]),
            role_goal=str(raw["role_goal"]),
            use_when=_tuple(raw.get("use_when")),
            avoid_when=_tuple(raw.get("avoid_when")),
            supported_inputs=_tuple(raw.get("supported_inputs")),
            expected_outputs=_tuple(raw.get("expected_outputs")),
            cost_class=str(raw.get("cost_class", "standard")),
            action_cost=float(raw.get("action_cost", 1.0)),
            semantic_family=str(raw.get("semantic_family", "general")),
            distractor_only=bool(raw.get("distractor_only", False)),
        )

    def public_profile(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role_name": self.role_name,
            "role_goal": self.role_goal,
            "use_when": list(self.use_when),
            "avoid_when": list(self.avoid_when),
            "supported_inputs": list(self.supported_inputs),
            "expected_outputs": list(self.expected_outputs),
            "cost_class": self.cost_class,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.public_profile()
        result.update(
            {
                "action_cost": self.action_cost,
                "semantic_family": self.semantic_family,
                "distractor_only": self.distractor_only,
            }
        )
        return result


@dataclass
class WorkflowStateV2:
    facts: set[str] = field(default_factory=set)
    artifacts: set[str] = field(default_factory=set)
    last_action: str | None = None
    last_status: str | None = None
    last_effect: str | None = None
    failure_count: int = 0
    routing_rejection_count: int = 0
    actor_failure_count: int = 0
    steps_used: int = 0
    cost_used: float = 0.0

    def clone(self) -> "WorkflowStateV2":
        return WorkflowStateV2(
            facts=set(self.facts),
            artifacts=set(self.artifacts),
            last_action=self.last_action,
            last_status=self.last_status,
            last_effect=self.last_effect,
            failure_count=self.failure_count,
            routing_rejection_count=self.routing_rejection_count,
            actor_failure_count=self.actor_failure_count,
            steps_used=self.steps_used,
            cost_used=self.cost_used,
        )

    def fingerprint(self) -> str:
        payload = {
            "facts": sorted(self.facts),
            "artifacts": sorted(self.artifacts),
            "last_action": self.last_action,
            "last_status": self.last_status,
            "failure_count": self.failure_count,
            "routing_rejection_count": self.routing_rejection_count,
            "actor_failure_count": self.actor_failure_count,
            "steps_used": self.steps_used,
            "cost_used": round(self.cost_used, 6),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]

    def semantic_fingerprint(self) -> str:
        """Legacy semantic identity retained for V2-alpha compatibility."""
        payload = {
            "facts": sorted(self.facts),
            "artifacts": sorted(self.artifacts),
            "last_status": self.last_status,
            "failure_count": self.failure_count,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]

    def progress_fingerprint(self) -> str:
        """Hash durable task progress, excluding counters and event wording."""
        payload = {
            "facts": sorted(self.facts),
            "artifacts": sorted(self.artifacts),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class TransitionSpecV2:
    transition_id: str
    action: str
    requires_all: frozenset[str] = field(default_factory=frozenset)
    requires_any: frozenset[str] = field(default_factory=frozenset)
    forbids: frozenset[str] = field(default_factory=frozenset)
    requires_artifacts: frozenset[str] = field(default_factory=frozenset)
    requires_last_statuses: frozenset[str] = field(default_factory=frozenset)
    requires_last_actions: frozenset[str] = field(default_factory=frozenset)
    add_facts: frozenset[str] = field(default_factory=frozenset)
    remove_facts: frozenset[str] = field(default_factory=frozenset)
    add_artifacts: frozenset[str] = field(default_factory=frozenset)
    remove_artifacts: frozenset[str] = field(default_factory=frozenset)
    status: str = "SUCCESS"
    public_effect: str = "State updated."
    cost_override: float | None = None
    minimum_steps_remaining: int = 1
    minimum_cost_remaining: float = 0.0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TransitionSpecV2":
        override = raw.get("cost_override")
        return cls(
            transition_id=str(raw["transition_id"]),
            action=str(raw["action"]),
            requires_all=frozenset(_tuple(raw.get("requires_all"))),
            requires_any=frozenset(_tuple(raw.get("requires_any"))),
            forbids=frozenset(_tuple(raw.get("forbids"))),
            requires_artifacts=frozenset(_tuple(raw.get("requires_artifacts"))),
            requires_last_statuses=frozenset(
                _tuple(raw.get("requires_last_statuses"))
            ),
            requires_last_actions=frozenset(
                _tuple(raw.get("requires_last_actions"))
            ),
            add_facts=frozenset(_tuple(raw.get("add_facts"))),
            remove_facts=frozenset(_tuple(raw.get("remove_facts"))),
            add_artifacts=frozenset(_tuple(raw.get("add_artifacts"))),
            remove_artifacts=frozenset(_tuple(raw.get("remove_artifacts"))),
            status=str(raw.get("status", "SUCCESS")),
            public_effect=str(raw.get("public_effect", "State updated.")),
            cost_override=float(override) if override is not None else None,
            minimum_steps_remaining=int(raw.get("minimum_steps_remaining", 1)),
            minimum_cost_remaining=float(raw.get("minimum_cost_remaining", 0.0)),
        )

    def matches(self, state: WorkflowStateV2) -> bool:
        return (
            self.requires_all.issubset(state.facts)
            and (
                not self.requires_any
                or bool(self.requires_any.intersection(state.facts))
            )
            and not self.forbids.intersection(state.facts)
            and self.requires_artifacts.issubset(state.artifacts)
            and (
                not self.requires_last_statuses
                or state.last_status in self.requires_last_statuses
            )
            and (
                not self.requires_last_actions
                or state.last_action in self.requires_last_actions
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "action": self.action,
            "requires_all": sorted(self.requires_all),
            "requires_any": sorted(self.requires_any),
            "forbids": sorted(self.forbids),
            "requires_artifacts": sorted(self.requires_artifacts),
            "requires_last_statuses": sorted(self.requires_last_statuses),
            "requires_last_actions": sorted(self.requires_last_actions),
            "add_facts": sorted(self.add_facts),
            "remove_facts": sorted(self.remove_facts),
            "add_artifacts": sorted(self.add_artifacts),
            "remove_artifacts": sorted(self.remove_artifacts),
            "status": self.status,
            "public_effect": self.public_effect,
            "cost_override": self.cost_override,
            "minimum_steps_remaining": self.minimum_steps_remaining,
            "minimum_cost_remaining": self.minimum_cost_remaining,
        }


@dataclass(frozen=True)
class ScenarioSpecV2:
    scenario_id: str
    template_id: str
    instance_id: str
    split: str
    split_group_id: str
    counterfactual_group_id: str | None
    family: str
    task: str
    agent_pool: tuple[str, ...]
    distractor_ids: tuple[str, ...]
    initial_facts: frozenset[str]
    initial_artifacts: frozenset[str]
    fact_descriptions: Mapping[str, str]
    artifact_descriptions: Mapping[str, str]
    goal_facts: frozenset[str]
    goal_artifacts: frozenset[str]
    transitions: tuple[TransitionSpecV2, ...]
    max_steps: int
    max_cost: float
    difficulty_tags: tuple[str, ...]
    schema_version: str = "2.0-alpha"
    public_context: Mapping[str, Any] = field(default_factory=dict)
    max_stagnant_repeats: int | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ScenarioSpecV2":
        return cls(
            scenario_id=str(raw["scenario_id"]),
            template_id=str(raw["template_id"]),
            instance_id=str(raw["instance_id"]),
            split=str(raw.get("split", "validation")),
            split_group_id=str(raw["split_group_id"]),
            counterfactual_group_id=(
                str(raw["counterfactual_group_id"])
                if raw.get("counterfactual_group_id")
                else None
            ),
            family=str(raw["family"]),
            task=str(raw["task"]),
            agent_pool=_tuple(raw.get("agent_pool")),
            distractor_ids=_tuple(raw.get("distractor_ids")),
            initial_facts=frozenset(_tuple(raw.get("initial_facts"))),
            initial_artifacts=frozenset(_tuple(raw.get("initial_artifacts"))),
            fact_descriptions={
                str(key): str(value)
                for key, value in (raw.get("fact_descriptions") or {}).items()
            },
            artifact_descriptions={
                str(key): str(value)
                for key, value in (raw.get("artifact_descriptions") or {}).items()
            },
            goal_facts=frozenset(_tuple(raw.get("goal_facts"))),
            goal_artifacts=frozenset(_tuple(raw.get("goal_artifacts"))),
            transitions=tuple(
                TransitionSpecV2.from_dict(item)
                for item in raw.get("transitions") or ()
            ),
            max_steps=int(raw.get("max_steps", 8)),
            max_cost=float(raw.get("max_cost", 12.0)),
            difficulty_tags=_tuple(raw.get("difficulty_tags")),
            schema_version=str(raw.get("schema_version", "2.0-alpha")),
            public_context=dict(raw.get("public_context") or {}),
            max_stagnant_repeats=(
                int(raw["max_stagnant_repeats"])
                if raw.get("max_stagnant_repeats") is not None
                else None
            ),
        )

    def initial_state(self) -> WorkflowStateV2:
        return WorkflowStateV2(
            facts=set(self.initial_facts), artifacts=set(self.initial_artifacts)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "template_id": self.template_id,
            "instance_id": self.instance_id,
            "split": self.split,
            "split_group_id": self.split_group_id,
            "counterfactual_group_id": self.counterfactual_group_id,
            "family": self.family,
            "task": self.task,
            "agent_pool": list(self.agent_pool),
            "distractor_ids": list(self.distractor_ids),
            "initial_facts": sorted(self.initial_facts),
            "initial_artifacts": sorted(self.initial_artifacts),
            "fact_descriptions": dict(self.fact_descriptions),
            "artifact_descriptions": dict(self.artifact_descriptions),
            "goal_facts": sorted(self.goal_facts),
            "goal_artifacts": sorted(self.goal_artifacts),
            "transitions": [item.to_dict() for item in self.transitions],
            "max_steps": self.max_steps,
            "max_cost": self.max_cost,
            "difficulty_tags": list(self.difficulty_tags),
            "schema_version": self.schema_version,
            "public_context": dict(self.public_context),
            "max_stagnant_repeats": self.max_stagnant_repeats,
        }


@dataclass(frozen=True)
class StepEventV2:
    step: int
    action: str
    status: str
    applicable: bool
    valid: bool
    efficient: bool
    admissible: bool
    progress_action: bool
    applicable_before: tuple[str, ...]
    valid_before: tuple[str, ...]
    efficient_before: tuple[str, ...]
    admissible_before: tuple[str, ...]
    progress_actions_before: tuple[str, ...]
    state_before: str
    state_after: str
    substantive_state_changed: bool
    redundant: bool
    stagnant_repeat: bool
    cost: float
    effect: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "status": self.status,
            "applicable": self.applicable,
            "valid": self.valid,
            "efficient": self.efficient,
            "admissible": self.admissible,
            "progress_action": self.progress_action,
            "applicable_before": list(self.applicable_before),
            "valid_before": list(self.valid_before),
            "efficient_before": list(self.efficient_before),
            "admissible_before": list(self.admissible_before),
            "progress_actions_before": list(self.progress_actions_before),
            "state_before": self.state_before,
            "state_after": self.state_after,
            "substantive_state_changed": self.substantive_state_changed,
            "redundant": self.redundant,
            "stagnant_repeat": self.stagnant_repeat,
            "cost": self.cost,
            "effect": self.effect,
        }
