from __future__ import annotations

import copy
import json

from inference.policy.base_policy import Policy
from inference.policy.capability_harness import CapabilityRoutingHarness
from inference.policy.planner_schema import (
    PLANNER_RESPONSE_SCHEMA,
    PlannerAssignment,
    PlannerDecision,
)
from inference.policy.stage_routing import RoutingSnapshot, StageRoutingGuard
from model.query_manager import StructuredResponseError, query_manager
from role_aware.audit_trace import digest


ORCHESTRATOR_STOP = "__orchestrator_stop__"


class FrozenLLMPlannerPolicy(Policy):
    """A non-learning LLM router over static, identity-safe role profiles."""

    def __init__(
        self,
        agent_graph,
        action_graph,
        config,
        runtime_config,
        experience_store=None,
    ) -> None:
        super().__init__(agent_graph, action_graph)
        self.config = copy.deepcopy(dict(config or {}))
        self.runtime_config = copy.deepcopy(dict(runtime_config or {}))
        planner = self.config.get("planner") or {}
        self.model = str(planner.get("model", "")).strip()
        if not self.model:
            raise ValueError("FrozenLLMPlannerPolicy requires planner.model")
        self.reasoning_effort = planner.get("reasoning_effort", "medium")
        self.max_repair_attempts = int(planner.get("max_repair_attempts", 1))
        self.experience_limit = int(planner.get("experience_limit", 20))
        self.audit_planner_input = bool(planner.get("audit_input", False))
        routing_input = self.config.get("routing_input") or {}
        self.exclude_task_fields = {
            str(value) for value in routing_input.get("exclude_task_fields", ())
        }
        self.routing_guard = StageRoutingGuard(self.config.get("routing_guard"))
        self.routing_harness = CapabilityRoutingHarness(
            self.config.get("routing_harness")
        )
        self.experience_store = experience_store
        self.agent_hash_list = list(agent_graph.hash_nodes)
        self.optimizer_updates_enabled = False
        self.training = False
        self.routing_mode = "frozen_llm_planner"
        self.audit = None

    def begin_task(self, audit=None):
        self.audit = audit
        self.routing_guard.begin_task(audit)

    def abort_task(self):
        self.routing_guard.end_task()
        self.audit = None

    @staticmethod
    def _previous_outputs(global_info) -> list[dict]:
        outputs = []
        for action in global_info.workflow.workflow:
            outputs.append(
                {
                    "role": action.agent_role,
                    "action": action.action.get("action"),
                    "result": action.result,
                    "success": action.success,
                }
            )
        return outputs

    def _planner_input(
        self,
        global_info,
        capacity: int,
        routing_snapshot: RoutingSnapshot | None = None,
    ) -> dict:
        inherited_signature = tuple(
            getattr(global_info, "task_signature", ()) or ()
        )
        signature = (
            inherited_signature
            if not self.exclude_task_fields
            else (f"task_type:{global_info.task.get('type', 'unknown')}",)
        )
        experience = (
            self.experience_store.snapshot(signature, self.experience_limit)
            if self.experience_store is not None
            else []
        )
        all_views = self.agent_graph.public_agent_views()
        routing_snapshot = routing_snapshot or self.routing_guard.evaluate(
            global_info, all_views
        )
        candidates = routing_snapshot.eligible_views(all_views)
        require_candidate_before_stop = (
            bool(routing_snapshot.state.get("enabled", False))
            and str(routing_snapshot.state.get("profile", "")).startswith(
                ("gaia_", "musique_")
            )
        )
        if "terminal_ready" in routing_snapshot.state:
            stop_ready = bool(routing_snapshot.state.get("terminal_ready"))
        else:
            stop_ready = bool(
                routing_snapshot.state.get(
                    "final_answer_exists"
                    if str(routing_snapshot.state.get("profile", "")).startswith(
                        "musique_"
                    )
                    else "candidate_answer_exists",
                    False,
                )
            )
        previous_outputs = self._previous_outputs(global_info)
        if self.routing_harness.enabled:
            previous_outputs = previous_outputs[
                -self.routing_harness.max_previous_outputs :
            ]
        planner_input = {
            "task": {
                str(key): value
                for key, value in global_info.task.items()
                if str(key) not in self.exclude_task_fields
            },
            "task_type": global_info.task.get("type"),
            "path_id": getattr(global_info, "path_uid", "root"),
            "completed_steps": len(global_info.workflow.workflow),
            "remaining_depth": getattr(global_info, "remaining_depth", None),
            "remaining_width": capacity,
            "previous_outputs": previous_outputs,
            "routing_state": dict(routing_snapshot.state),
            "candidate_profiles": candidates,
            "route_experience": experience,
            "constraints": {
                "stop_allowed": (
                    global_info.path_id != -1
                    and (
                        not require_candidate_before_stop
                        or stop_ready
                    )
                ),
                "width_is_upper_bound": True,
                "minimum_selections_when_not_stopping": 1,
                "maximum_selections": capacity,
                "eligible_candidate_ids": [
                    int(view["candidate_id"]) for view in candidates
                ],
                "only_listed_candidates_are_valid": True,
            },
        }
        harness_state = routing_snapshot.state.get("routing_harness") or {}
        planner_input["constraints"]["branch_allowed"] = bool(
            harness_state.get("branch_allowed", False)
        )
        if self.routing_harness.enabled and self.routing_harness.typed_blackboard:
            planner_input["blackboard"] = self.routing_harness.blackboard(
                task=global_info.task,
                routing_state=routing_snapshot.state,
                previous_outputs=previous_outputs,
                remaining_depth=getattr(global_info, "remaining_depth", None),
                remaining_width=capacity,
                dominant_need=str(harness_state.get("dominant_need", "")),
            )
        return planner_input

    def _system_prompt(self) -> str:
        prompt = (
            "You are a frozen multi-agent orchestration planner. Select roles by matching "
            "the current task state to their static routing profiles. Do not infer hidden "
            "model identities. At the root, create independent useful paths. On an existing "
            "path, either choose the next contributor or stop if the answer/artifact is ready. "
            "Width is an upper bound, not a target: select only as many paths as have distinct "
            "expected value, from one through maximum_selections. Never add a redundant path "
            "just to fill capacity. Respect the depth constraint. Return only the requested "
            "JSON object. Candidate profiles have already been stage-masked; only candidate_id "
            "values listed in candidate_profiles are valid."
        )
        if self.routing_harness.enabled:
            prompt += (
                " Use the typed blackboard as the authoritative compact state. "
                "The candidate list may include generic cross-benchmark roles but excludes "
                "explicit scope mismatches. STOP is valid only when terminal_ready is true. "
                "Select more than one role on an existing path only when branch_allowed is "
                "true; additional selections create state-sharing recovery/check forks."
            )
        if self.routing_guard.profile == "musique_dynamic_v2":
            prompt += (
                " For dynamic MuSiQue routing, diagnose the semantic state before choosing: "
                "whether the current output answers the original question, whether evidence "
                "is sufficient, whether dependencies introduced by the generated plan are "
                "resolved, whether candidates conflict, and which capability has the greatest "
                "marginal value. Candidate masking represents feasibility only, not a fixed "
                "stage graph. A role may be selected again only after substantive evidence or "
                "candidate-state progress. Do not infer a hidden gold hop count."
            )
        return prompt

    def _validate(
        self,
        decision: PlannerDecision,
        global_info,
        capacity: int,
        eligible_candidate_ids: set[int],
        stop_allowed: bool,
    ) -> PlannerDecision:
        if decision.stop:
            if not stop_allowed:
                raise ValueError("STOP is not allowed before a terminal answer exists")
            return decision
        seen = set()
        accepted = []
        views = self.agent_graph.public_agent_views()
        for assignment in decision.selections:
            index = assignment.candidate_id
            if index >= len(views):
                raise ValueError(f"Unknown candidate_id {index}")
            if not views[index]["available"]:
                raise ValueError(f"Candidate {index} is unavailable")
            if index not in eligible_candidate_ids:
                raise ValueError(f"Candidate {index} is masked for the current stage")
            if index in seen:
                raise ValueError(f"Duplicate candidate_id {index}")
            seen.add(index)
            accepted.append(assignment)
        if not accepted:
            raise ValueError("At least one eligible selection is required when stop=false")
        if len(accepted) > capacity:
            accepted = accepted[:capacity]
        return PlannerDecision(
            stop=False,
            task_signature=decision.task_signature,
            state_gaps=decision.state_gaps,
            selections=tuple(accepted),
        )

    def _fallback(
        self,
        global_info,
        capacity: int,
        routing_snapshot: RoutingSnapshot,
    ) -> PlannerDecision:
        available = list(routing_snapshot.eligible_candidate_ids)
        if not available:
            if global_info.path_id != -1:
                return PlannerDecision(True, (), ("no available candidate",), ())
            raise RuntimeError("No available agent for frozen planner fallback")
        # Invalid planner output should fail closed on cost: use one viable path
        # instead of silently filling the maximum width.
        count = 1
        assignments = tuple(
            PlannerAssignment(
                candidate_id=index,
                subtask="Solve the current task state according to your role and output contract.",
                expected_contribution="Provide a valid role-specific candidate result.",
            )
            for index in available[:count]
        )
        return PlannerDecision(False, ("unclassified",), ("planner fallback",), assignments)

    def propose(self, global_info, capacity):
        if capacity < 1:
            raise ValueError("Routing requires positive capacity")
        all_views = self.agent_graph.public_agent_views()
        routing_snapshot = self.routing_guard.evaluate(global_info, all_views)
        harness_result = self.routing_harness.apply(
            task_type=global_info.task.get("type"),
            root=global_info.path_id == -1,
            capacity=capacity,
            views=all_views,
            snapshot=routing_snapshot,
        )
        routing_snapshot = harness_result.snapshot
        route_capacity = harness_result.route_capacity
        planner_input = self._planner_input(
            global_info, route_capacity, routing_snapshot
        )
        stop_allowed = bool(planner_input["constraints"]["stop_allowed"])
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": json.dumps(planner_input, ensure_ascii=False)},
        ]
        decision = None
        error_type = None
        tokens = 0
        guard_stop = (
            global_info.path_id != -1
            and not routing_snapshot.eligible_candidate_ids
        )
        if guard_stop:
            decision = PlannerDecision(
                True,
                tuple(getattr(global_info, "task_signature", ()) or ()),
                ("routing guard marked the path terminal",),
                (),
            )
        for _ in range(0 if guard_stop else self.max_repair_attempts + 1):
            try:
                payload, used = query_manager.query_structured(
                    self.model,
                    messages,
                    PLANNER_RESPONSE_SCHEMA,
                    schema_name="planner_decision",
                    reasoning_effort=self.reasoning_effort,
                )
            except StructuredResponseError as error:
                error_type = type(error).__name__
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous response was invalid. Return a corrected JSON object "
                            f"that obeys every constraint. Validation error: {error_type}"
                        ),
                    }
                )
                continue
            tokens += used
            try:
                decision = self._validate(
                    PlannerDecision.from_dict(payload),
                    global_info,
                    route_capacity,
                    set(routing_snapshot.eligible_candidate_ids),
                    stop_allowed,
                )
                break
            except (KeyError, TypeError, ValueError) as error:
                error_type = type(error).__name__
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous response was invalid. Return a corrected JSON object "
                            f"that obeys every constraint. Validation error: {error_type}"
                        ),
                    }
                )
        fallback = decision is None
        decision = decision or self._fallback(
            global_info, route_capacity, routing_snapshot
        )
        if decision.stop:
            actions = [ORCHESTRATOR_STOP]
            assignments = [None]
        else:
            actions = [self.agent_hash_list[item.candidate_id] for item in decision.selections]
            assignments = [item.to_dict() for item in decision.selections]
        decision_id = self.audit.new_id("decision") if self.audit else "planner-decision"
        proposal = {
            "decision_id": decision_id,
            "actions": actions,
            "assignments": assignments,
            "task_signature": list(decision.task_signature),
            "state_gaps": list(decision.state_gaps),
            "planner_tokens": tokens,
            "fallback": fallback,
            "routing_state": dict(routing_snapshot.state),
        }
        if self.audit:
            audit_payload = {
                "decision_id": decision_id,
                "path_uid": getattr(global_info, "path_uid", "root"),
                "mode": self.routing_mode,
                "selected": actions,
                "assignments": assignments,
                "task_signature": list(decision.task_signature),
                "state_gaps": list(decision.state_gaps),
                "capacity": capacity,
                "route_capacity": route_capacity,
                "allow_stop": stop_allowed,
                "remaining_depth": getattr(global_info, "remaining_depth", None),
                "planner_tokens": tokens,
                "fallback": fallback,
                "error_type": error_type,
                "routing_state": dict(routing_snapshot.state),
                "eligible_candidate_ids": list(
                    routing_snapshot.eligible_candidate_ids
                ),
                "masked_candidates": list(routing_snapshot.masked_candidates),
                "planner_input_hash": digest(planner_input),
            }
            if self.audit_planner_input:
                audit_payload["planner_input"] = planner_input
            self.audit.emit(
                "routing_decision",
                **audit_payload,
            )
        return proposal

    def accept(self, proposal, parent_uid, allocations):
        return None

    def forward(self, global_info):
        return self.propose(global_info, 1)["actions"]

    def finalize_task(self, transition, global_info):
        return None

    def update(self):
        return {}

    def state_fingerprint(self):
        return digest(
            {
                "config": self.config,
                "profiles": [dict(view) for view in self.agent_graph.public_agent_views()],
            }
        )

    def training_state_dict(self):
        return {"type": "frozen_llm_planner", "fingerprint": self.state_fingerprint()}

    def save_model(self, path=None, tag=None):
        return None
