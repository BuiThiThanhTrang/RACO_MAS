from __future__ import annotations

import copy
from typing import Any

from .schemas import AgentContract, ScenarioSpec, StepEvent, STOP_ACTION, state_fingerprint


class RoutingEnvironment:
    """Deterministic actor executor used by the interactive routing track.

    The router chooses an agent. The environment applies that agent's state contract;
    no language model is called and no hidden capability score is consulted.
    """

    def __init__(self, scenario: ScenarioSpec, agents: dict[str, AgentContract]):
        self.scenario = scenario
        self.agents = agents
        self.facts: set[str] = set(scenario.initial_facts)
        self.history: list[StepEvent] = []
        self.total_cost = 0.0
        self.done = False
        self.success = False
        self.terminal_reason: str | None = None

    def clone(self) -> "RoutingEnvironment":
        return copy.deepcopy(self)

    @property
    def goal_reached(self) -> bool:
        return self.scenario.goal_facts.issubset(self.facts)

    def candidate_actions(self) -> tuple[str, ...]:
        return tuple(self.scenario.agent_pool) + (STOP_ACTION,)

    def _matching_transitions(self, action: str):
        return tuple(
            transition
            for transition in self.scenario.transitions
            if transition.action == action and transition.matches(self.facts)
        )

    def valid_actions(self) -> tuple[str, ...]:
        if self.done:
            return ()
        if self.goal_reached:
            return (STOP_ACTION,)
        return tuple(
            action
            for action in self.scenario.agent_pool
            if self._matching_transitions(action)
        )

    def observation(self) -> dict[str, Any]:
        last = self.history[-1] if self.history else None
        return {
            "benchmark": "RoutingBench",
            "benchmark_version": "1.0",
            "scenario_id": self.scenario.scenario_id,
            "workflow_family": self.scenario.family,
            "task": self.scenario.task,
            "state": {
                "facts": sorted(self.facts),
                "fingerprint": state_fingerprint(self.facts),
                "step_index": len(self.history),
                "last_action": last.action if last else None,
                "last_status": last.status if last else None,
                "last_state_changed": last.state_changed if last else None,
                "recent_history": [
                    {
                        "action": item.action,
                        "status": item.status,
                        "state_changed": item.state_changed,
                        "facts_added": list(item.facts_added),
                    }
                    for item in self.history[-4:]
                ],
            },
            "budget": {
                "steps_used": len(self.history),
                "steps_remaining": max(self.scenario.max_steps - len(self.history), 0),
                "cost_used": self.total_cost,
                "cost_remaining": max(self.scenario.max_cost - self.total_cost, 0.0),
            },
            "agent_profiles": [
                self.agents[agent_id].public_profile()
                for agent_id in self.scenario.agent_pool
            ],
            "constraints": {
                "choose_exactly_one": True,
                "allowed_actions": list(self.candidate_actions()),
                "stop_only_when_task_complete": True,
                "capability_scores_available": False,
            },
        }

    def step(self, action: str) -> StepEvent:
        if self.done:
            raise RuntimeError("Cannot step a completed routing episode")
        action = str(action).strip()
        before_facts = set(self.facts)
        before_fp = state_fingerprint(before_facts)
        valid_before = self.valid_actions()
        valid = action in valid_before
        status = "NOT_APPLICABLE"
        added: set[str] = set()
        removed: set[str] = set()
        cost = 0.0
        distractor = False

        if action == STOP_ACTION:
            status = "GOAL_COMPLETE" if self.goal_reached else "PREMATURE_STOP"
            self.done = True
            self.success = self.goal_reached
            self.terminal_reason = status
        elif action not in self.scenario.agent_pool:
            status = "UNKNOWN_AGENT"
        else:
            agent = self.agents[action]
            cost = agent.cost
            distractor = agent.distractor
            matches = self._matching_transitions(action)
            if matches:
                transition = matches[0]
                if transition.cost_override is not None:
                    cost = transition.cost_override
                removed = set(transition.remove).intersection(self.facts)
                self.facts.difference_update(transition.remove)
                added = set(transition.add).difference(self.facts)
                self.facts.update(transition.add)
                status = transition.status

        self.total_cost += cost
        after_fp = state_fingerprint(self.facts)
        redundant = any(
            previous.action == action and previous.state_before == before_fp
            for previous in self.history
        )
        newly_achieved = sorted(
            self.scenario.milestone_facts.intersection(added)
        )
        event = StepEvent(
            step=len(self.history) + 1,
            action=action,
            status=status,
            valid_action=valid,
            valid_actions_before=valid_before,
            state_before=before_fp,
            state_after=after_fp,
            facts_added=tuple(sorted(added)),
            facts_removed=tuple(sorted(removed)),
            state_changed=before_fp != after_fp,
            redundant=redundant,
            distractor=distractor,
            cost=cost,
            newly_achieved_milestones=tuple(newly_achieved),
        )
        self.history.append(event)

        if not self.done and len(self.history) >= self.scenario.max_steps:
            self.done = True
            self.success = False
            self.terminal_reason = "STEP_LIMIT"
        elif not self.done and self.total_cost >= self.scenario.max_cost:
            self.done = True
            self.success = False
            self.terminal_reason = "COST_LIMIT"
        return event
