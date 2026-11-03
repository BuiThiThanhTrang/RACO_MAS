from __future__ import annotations

from typing import Any, Iterable

from .schemas import (
    FAILURE_STATUSES,
    ROUTING_REJECTION_STATUSES,
    AgentProfileV2,
    ScenarioSpecV2,
    StepEventV2,
    STOP_ACTION,
    WorkflowStateV2,
)


Outcome = tuple[int, float]


def _uses_observable_contract(schema_version: str) -> bool:
    """Return whether a scenario uses the compact failure-aware observation."""

    value = str(schema_version)
    return (
        value.startswith("2.0-beta")
        or value.startswith("3.")
        or value.startswith("4.")
    )


def _reserves_terminal_stop(schema_version: str) -> bool:
    """Return whether STOP has a zero-cost decision outside execution budget."""

    value = str(schema_version)
    return value.startswith("3.0-beta") or value.startswith("4.")


def _pareto(points: Iterable[Outcome]) -> tuple[Outcome, ...]:
    unique = sorted({(int(steps), round(float(cost), 6)) for steps, cost in points})
    frontier: list[Outcome] = []
    for point in unique:
        dominated = any(
            other[0] <= point[0]
            and other[1] <= point[1]
            and other != point
            for other in unique
        )
        if not dominated:
            frontier.append(point)
    return tuple(frontier)


class RoutingEnvironmentV2:
    """Deterministic V2 environment with validity and Pareto efficiency labels."""

    def __init__(
        self,
        scenario: ScenarioSpecV2,
        agents: dict[str, AgentProfileV2],
        state: WorkflowStateV2 | None = None,
    ) -> None:
        self.scenario = scenario
        self.agents = agents
        self.state = state.clone() if state is not None else scenario.initial_state()
        self.history: list[StepEventV2] = []
        self.done = False
        self.success = False
        self.terminal_reason: str | None = None
        self._action_outcomes_cache: tuple[
            str, dict[str, tuple[Outcome, ...]]
        ] | None = None

    def clone(self) -> "RoutingEnvironmentV2":
        child = RoutingEnvironmentV2(self.scenario, self.agents, self.state)
        child.history = list(self.history)
        child.done = self.done
        child.success = self.success
        child.terminal_reason = self.terminal_reason
        child._action_outcomes_cache = self._action_outcomes_cache
        return child

    @property
    def goal_reached(self) -> bool:
        return self.scenario.goal_facts.issubset(
            self.state.facts
        ) and self.scenario.goal_artifacts.issubset(self.state.artifacts)

    @property
    def remaining_steps(self) -> int:
        return max(self.scenario.max_steps - self.state.steps_used, 0)

    @property
    def remaining_cost(self) -> float:
        return max(self.scenario.max_cost - self.state.cost_used, 0.0)

    def candidate_actions(self) -> tuple[str, ...]:
        return tuple(self.scenario.agent_pool) + (STOP_ACTION,)

    def _transition_for(self, action: str):
        matches = [
            transition
            for transition in self.scenario.transitions
            if transition.action == action and transition.matches(self.state)
            and transition.minimum_steps_remaining <= self.remaining_steps
            and transition.minimum_cost_remaining <= self.remaining_cost + 1e-9
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous transitions for {self.scenario.scenario_id} action "
                f"{action}: {[item.transition_id for item in matches]}"
            )
        return matches[0] if matches else None

    def _cost_for(self, action: str, transition=None) -> float:
        if action == STOP_ACTION:
            return 0.0
        if transition is not None and transition.cost_override is not None:
            return transition.cost_override
        profile = self.agents.get(action)
        return profile.action_cost if profile is not None else 0.0

    def applicable_actions(self) -> tuple[str, ...]:
        if self.done:
            return ()
        if self.goal_reached:
            return (
                (STOP_ACTION,)
                if _reserves_terminal_stop(self.scenario.schema_version)
                or self.remaining_steps >= 1
                else ()
            )
        if self.remaining_steps < 1:
            return ()
        actions: list[str] = []
        for action in self.scenario.agent_pool:
            transition = self._transition_for(action)
            if transition is None:
                continue
            if self._cost_for(action, transition) <= self.remaining_cost + 1e-9:
                actions.append(action)
        return tuple(actions)

    def _apply_for_search(self, action: str) -> tuple["RoutingEnvironmentV2", float]:
        child = self.clone()
        child.history = []
        transition = child._transition_for(action)
        cost = child._cost_for(action, transition)
        child._mutate(action, transition, cost)
        return child, cost

    def _success_outcomes(
        self,
        cache: dict[str, tuple[Outcome, ...]] | None = None,
        visiting: set[str] | None = None,
    ) -> tuple[Outcome, ...]:
        cache = cache if cache is not None else {}
        visiting = visiting if visiting is not None else set()
        key = self.state.fingerprint()
        if key in cache:
            return cache[key]
        if self.done:
            return ((0, 0.0),) if self.success else ()
        if self.goal_reached:
            result = (
                ((0, 0.0),)
                if _reserves_terminal_stop(self.scenario.schema_version)
                else (((1, 0.0),) if self.remaining_steps >= 1 else ())
            )
            cache[key] = result
            return result
        if key in visiting:
            return ()
        visiting.add(key)
        outcomes: list[Outcome] = []
        for action in self.applicable_actions():
            if action == STOP_ACTION:
                continue
            child, cost = self._apply_for_search(action)
            for remaining_steps, remaining_cost in child._success_outcomes(
                cache, visiting
            ):
                outcomes.append((1 + remaining_steps, cost + remaining_cost))
        visiting.remove(key)
        result = _pareto(outcomes)
        cache[key] = result
        return result

    def action_outcomes(self) -> dict[str, tuple[Outcome, ...]]:
        cache_key = self.state.fingerprint()
        if (
            self._action_outcomes_cache is not None
            and self._action_outcomes_cache[0] == cache_key
        ):
            return self._action_outcomes_cache[1]
        if self.done:
            return {}
        if self.goal_reached:
            result = (
                {STOP_ACTION: ((0, 0.0),)}
                if _reserves_terminal_stop(self.scenario.schema_version)
                else ({STOP_ACTION: ((1, 0.0),)} if self.remaining_steps >= 1 else {})
            )
            self._action_outcomes_cache = (cache_key, result)
            return result
        cache: dict[str, tuple[Outcome, ...]] = {}
        results: dict[str, tuple[Outcome, ...]] = {}
        for action in self.applicable_actions():
            child, cost = self._apply_for_search(action)
            action_points = [
                (1 + remaining_steps, cost + remaining_cost)
                for remaining_steps, remaining_cost in child._success_outcomes(cache)
            ]
            frontier = _pareto(action_points)
            if frontier:
                results[action] = frontier
        self._action_outcomes_cache = (cache_key, results)
        return results

    def valid_actions(self) -> tuple[str, ...]:
        """Actions with at least one hidden successful continuation.

        This is the V2-alpha label. V2-beta uses ``admissible_actions`` as its
        primary routing target so a router is not punished for failing to know
        a hidden tool outcome.
        """
        outcomes = self.action_outcomes()
        return tuple(
            action for action in self.candidate_actions() if action in outcomes
        )

    def progress_actions(self) -> tuple[str, ...]:
        """Hidden diagnostic actions that can still reach the goal."""
        return self.valid_actions()

    def admissible_actions(self) -> tuple[str, ...]:
        """Actions justified by the public state and qualitative contracts."""
        return self.applicable_actions()

    def efficient_actions(self) -> tuple[str, ...]:
        outcomes = self.action_outcomes()
        global_frontier = set(
            _pareto(point for points in outcomes.values() for point in points)
        )
        return tuple(
            action
            for action in self.candidate_actions()
            if action in outcomes
            and any(point in global_frontier for point in outcomes[action])
        )

    def _mutate(self, action: str, transition, cost: float) -> None:
        self._action_outcomes_cache = None
        reserved_terminal = _reserves_terminal_stop(self.scenario.schema_version)
        goal_before = self.goal_reached
        terminal_decision = reserved_terminal and goal_before
        if not terminal_decision:
            self.state.steps_used += 1
            self.state.cost_used = round(self.state.cost_used + cost, 6)
        self.state.last_action = action
        if action == STOP_ACTION:
            if self.goal_reached:
                self.state.last_status = "GOAL_COMPLETE"
                self.state.last_effect = "The committed artifact satisfies the workflow goal."
                self.done = True
                self.success = True
                self.terminal_reason = "GOAL_COMPLETE"
            else:
                self.state.last_status = "PREMATURE_STOP"
                self.state.last_effect = "The router stopped before the workflow goal was met."
                self.done = True
                self.success = False
                self.terminal_reason = "PREMATURE_STOP"
            return

        if terminal_decision:
            self.state.last_status = "ACTION_REJECTED"
            self.state.last_effect = (
                "The workflow goal was already complete; the reserved terminal "
                "decision accepts only STOP."
            )
            self.state.failure_count += 1
            self.state.routing_rejection_count += 1
            self.done = True
            self.success = False
            self.terminal_reason = "TERMINAL_ACTION_ERROR"
            return

        if transition is None:
            beta = _uses_observable_contract(self.scenario.schema_version)
            self.state.last_status = "ACTION_REJECTED" if beta else "NOT_APPLICABLE"
            self.state.last_effect = (
                "The action was rejected before execution because the role contract "
                "does not match the current state."
                if beta
                else "The selected role could not act on the current state."
            )
            self.state.failure_count += 1
            self.state.routing_rejection_count += 1
        else:
            self.state.facts.difference_update(transition.remove_facts)
            self.state.facts.update(transition.add_facts)
            self.state.artifacts.difference_update(transition.remove_artifacts)
            self.state.artifacts.update(transition.add_artifacts)
            self.state.last_status = transition.status
            self.state.last_effect = transition.public_effect
            if transition.status in FAILURE_STATUSES:
                self.state.failure_count += 1
                self.state.actor_failure_count += 1
            if transition.status in {"HARD_FAILURE", "EXECUTED_HARD_FAILURE"}:
                self.done = True
                self.success = False
                self.terminal_reason = "HARD_FAILURE"

        if self.done:
            return
        if (
            self.state.steps_used >= self.scenario.max_steps
            and not (reserved_terminal and self.goal_reached)
        ):
            self.done = True
            self.success = False
            self.terminal_reason = "STEP_LIMIT"
        elif self.state.cost_used > self.scenario.max_cost + 1e-9:
            self.done = True
            self.success = False
            self.terminal_reason = "COST_LIMIT"

    def step(self, action: str) -> StepEventV2:
        if self.done:
            raise RuntimeError("Cannot step a completed V2 routing episode")
        action = str(action).strip()
        applicable_before = self.applicable_actions()
        valid_before = self.valid_actions()
        admissible_before = self.admissible_actions()
        progress_actions_before = self.progress_actions()
        efficient_before = self.efficient_actions()
        before = self.state.fingerprint()
        before_progress = self.state.progress_fingerprint()
        terminal_decision = (
            _reserves_terminal_stop(self.scenario.schema_version)
            and self.goal_reached
        )
        transition = (
            self._transition_for(action)
            if action in self.agents and not terminal_decision
            else None
        )
        applicable = action in applicable_before
        cost = 0.0 if terminal_decision else self._cost_for(action, transition)
        redundant = any(
            item.action == action and item.state_before == before for item in self.history
        )
        self._mutate(action, transition, cost)
        after = self.state.fingerprint()
        progress_changed = before_progress != self.state.progress_fingerprint()
        stagnant_repeat = (
            not progress_changed
            and any(
                item.action == action
                and not item.substantive_state_changed
                for item in reversed(self.history)
            )
        )
        event = StepEventV2(
            step=(len(self.history) + 1 if terminal_decision else self.state.steps_used),
            action=action,
            status=str(self.state.last_status),
            applicable=applicable,
            valid=(
                action in admissible_before
                if _uses_observable_contract(self.scenario.schema_version)
                else action in valid_before
            ),
            efficient=action in efficient_before,
            admissible=action in admissible_before,
            progress_action=action in progress_actions_before,
            applicable_before=applicable_before,
            valid_before=valid_before,
            efficient_before=efficient_before,
            admissible_before=admissible_before,
            progress_actions_before=progress_actions_before,
            state_before=before,
            state_after=after,
            substantive_state_changed=progress_changed,
            redundant=redundant,
            stagnant_repeat=stagnant_repeat,
            cost=cost,
            effect=str(self.state.last_effect or ""),
        )
        self.history.append(event)
        threshold = self.scenario.max_stagnant_repeats
        if threshold and not self.done:
            stagnant_tail = 0
            for previous in reversed(self.history):
                if previous.action != action or previous.substantive_state_changed:
                    break
                stagnant_tail += 1
            if stagnant_tail >= threshold:
                self.done = True
                self.success = False
                self.terminal_reason = "LOOP_LIMIT"
        return event

    def _execution_memory(self) -> dict[str, Any]:
        attempts: dict[str, list[dict[str, Any]]] = {}
        rejected: list[dict[str, Any]] = []
        contributions: list[dict[str, Any]] = []
        for event in self.history:
            item = {
                "step": event.step,
                "status": event.status,
                "effect": event.effect,
                "state_changed": event.substantive_state_changed,
            }
            attempts.setdefault(event.action, []).append(item)
            if event.status in ROUTING_REJECTION_STATUSES:
                rejected.append({"step": event.step, "agent_id": event.action})
            elif event.substantive_state_changed and event.action != STOP_ACTION:
                contributions.append(
                    {
                        "step": event.step,
                        "agent_id": event.action,
                        "contribution": event.effect,
                    }
                )
        return {
            "attempts_by_agent": attempts,
            "rejected_actions": rejected,
            "completed_contributions": contributions,
        }

    def _recent_events(self, limit: int = 3) -> list[dict[str, Any]]:
        return [
            {
                "step": item.step,
                "agent_id": item.action,
                "status": item.status,
                "effect": item.effect,
                "semantic_delta": (
                    "state_changed" if item.substantive_state_changed else "none"
                ),
            }
            for item in self.history[-limit:]
        ]

    def observation(self) -> dict[str, Any]:
        facts = sorted(
            self.scenario.fact_descriptions.get(item, item)
            for item in self.state.facts
        )
        artifacts = sorted(
            self.scenario.artifact_descriptions.get(item, item)
            for item in self.state.artifacts
        )
        state = {
            "facts": facts,
            "available_artifacts": artifacts,
            "last_event": {
                "action": self.state.last_action,
                "status": self.state.last_status,
                "effect": self.state.last_effect,
            },
            "failure_count": self.state.failure_count,
            "fingerprint": self.state.fingerprint(),
            "step_index": self.state.steps_used,
        }
        if _uses_observable_contract(self.scenario.schema_version):
            state.update(
                {
                    "artifact_inventory": [
                        {"description": item, "status": "available"}
                        for item in artifacts
                    ],
                    "open_requirements": [
                        self.scenario.fact_descriptions.get(item, item)
                        for item in sorted(self.state.facts)
                        if any(
                            marker in item
                            for marker in (
                                "pending",
                                "unresolved",
                                "blocked",
                                "stalled",
                                "unmapped",
                            )
                        )
                    ],
                    "goal_status": "complete" if self.goal_reached else "incomplete",
                    "routing_rejection_count": self.state.routing_rejection_count,
                    "actor_failure_count": self.state.actor_failure_count,
                }
            )
        observation = {
            "benchmark": "RoutingBench",
            "benchmark_version": self.scenario.schema_version,
            "scenario_id": self.scenario.scenario_id,
            "workflow_family": self.scenario.family,
            "task": self.scenario.task,
            "state": state,
            "budget": {
                "steps_remaining": self.remaining_steps,
                "cost_remaining": self.remaining_cost,
                "cost_class_only": True,
            },
            "agent_profiles": [
                self.agents[agent_id].public_profile()
                for agent_id in self.scenario.agent_pool
            ],
            "constraints": {
                "choose_exactly_one": True,
                "allowed_actions": list(self.candidate_actions()),
                "stop_only_when_task_complete": True,
                "stop_uses_execution_budget": not _reserves_terminal_stop(
                    self.scenario.schema_version
                ),
                "reserved_terminal_stop": _reserves_terminal_stop(
                    self.scenario.schema_version
                ),
                "quality_rankings_provided": False,
            },
        }
        if _uses_observable_contract(self.scenario.schema_version):
            observation["task_context"] = dict(self.scenario.public_context)
            observation["execution_memory"] = self._execution_memory()
            observation["recent_events"] = self._recent_events()
        return observation
