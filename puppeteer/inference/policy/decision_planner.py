from __future__ import annotations

import copy
import re
from typing import Any, Mapping

from inference.policy.base_policy import Policy
from inference.policy.decision_model_client import (
    DecisionModelResponseError,
    SystemOneDecisionClient,
)
from inference.policy.planner_schema import PlannerAssignment, PlannerDecision
from inference.policy.route_bundle import RouteBundleBuilder, RouteOption
from inference.policy.stage_routing import StageRoutingGuard
from role_aware.audit_trace import digest


ORCHESTRATOR_STOP = "__orchestrator_stop__"


def _slug(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return normalized[:80]


class DecisionPlannerPolicy(Policy):
    """Non-learning role router backed by a typed System One decision model."""

    def __init__(
        self,
        agent_graph,
        action_graph,
        config,
        runtime_config,
        experience_store=None,
        decision_client=None,
    ) -> None:
        super().__init__(agent_graph, action_graph)
        self.config = copy.deepcopy(dict(config or {}))
        self.runtime_config = copy.deepcopy(dict(runtime_config or {}))
        decision_config = self.config.get("decision") or {}
        if not decision_config:
            raise ValueError("DecisionPlannerPolicy requires policy.decision")
        self.experience_limit = int(decision_config.get("experience_limit", 20))
        semantic_config = decision_config.get("semantic_diagnosis") or {}
        self.semantic_diagnosis_enabled = bool(semantic_config.get("enabled", False))
        self.semantic_diagnosis_on_root = bool(
            semantic_config.get("include_root", False)
        )
        self.audit_planner_input = bool(decision_config.get("audit_input", False))
        routing_input = self.config.get("routing_input") or {}
        self.exclude_task_fields = {
            str(value) for value in routing_input.get("exclude_task_fields", ())
        }
        self.root_selection_mode = str(
            decision_config.get("root_selection_mode", "flat_bundle")
        ).strip().lower()
        if self.root_selection_mode not in {"flat_bundle", "sequential"}:
            raise ValueError(
                "decision.root_selection_mode must be flat_bundle or sequential"
            )
        self.bundle_builder = RouteBundleBuilder(
            max_options=int(decision_config.get("max_options", 255))
        )
        self.client = decision_client or SystemOneDecisionClient(decision_config)
        self.routing_guard = StageRoutingGuard(self.config.get("routing_guard"))
        self.experience_store = experience_store
        self.agent_hash_list = list(agent_graph.hash_nodes)
        self.optimizer_updates_enabled = False
        self.training = False
        self.routing_mode = "decision_model_planner"
        self.audit = None

    def begin_task(self, audit=None):
        self.audit = audit
        self.routing_guard.begin_task(audit)

    def abort_task(self):
        self.routing_guard.end_task()
        self.audit = None

    @staticmethod
    def _previous_outputs(global_info) -> list[dict]:
        return [
            {
                "role": action.agent_role,
                "action": action.action.get("action"),
                "result": action.result,
                "success": action.success,
            }
            for action in global_info.workflow.workflow
        ]

    def _router_task(self, task: Mapping[str, Any]) -> dict[str, Any]:
        return {
            str(key): value
            for key, value in task.items()
            if str(key) not in self.exclude_task_fields
        }

    def _task_signature(self, global_info) -> tuple[str, ...]:
        inherited = tuple(getattr(global_info, "task_signature", ()) or ())
        if inherited and not self.exclude_task_fields:
            return inherited
        task = global_info.task
        signature = [f"task_type:{_slug(task.get('type', 'unknown'))}"]
        for key in ("category", "subject", "domain", "req", "level", "has_attachment"):
            if key in self.exclude_task_fields:
                continue
            value = task.get(key)
            if value is not None and str(value).strip():
                signature.append(f"{key}:{_slug(value)}")
        return tuple(signature)

    def _state(
        self,
        global_info,
        capacity: int,
        signature: tuple[str, ...],
        candidate_views: list[dict],
        routing_state: Mapping[str, Any],
    ) -> dict:
        experience = (
            self.experience_store.snapshot(signature, self.experience_limit)
            if self.experience_store is not None
            else []
        )
        guarded = bool(routing_state.get("enabled", False))
        profile = str(routing_state.get("profile", ""))
        if profile.startswith("musique_"):
            terminal_ready = bool(routing_state.get("final_answer_exists", False))
        elif profile.startswith("gaia_"):
            terminal_ready = bool(
                routing_state.get("candidate_answer_exists", False)
            )
        else:
            terminal_ready = True
        return {
            "task": self._router_task(global_info.task),
            "task_type": global_info.task.get("type"),
            "task_signature": list(signature),
            "path_id": getattr(global_info, "path_uid", "root"),
            "completed_steps": len(global_info.workflow.workflow),
            "remaining_depth": getattr(global_info, "remaining_depth", None),
            "remaining_width": capacity,
            "previous_outputs": self._previous_outputs(global_info),
            "routing_state": dict(routing_state),
            "candidate_profiles": [dict(view) for view in candidate_views],
            "route_experience": experience,
            "constraints": {
                "stop_allowed": (
                    global_info.path_id != -1
                    and (not guarded or terminal_ready)
                ),
                "width_is_upper_bound": True,
                "maximum_selections": capacity,
                "probabilities_are_audit_only": True,
                "eligible_candidate_ids": [
                    int(view["candidate_id"]) for view in candidate_views
                ],
                "only_listed_candidates_are_valid": True,
            },
        }

    @staticmethod
    def _semantic_questions() -> dict[str, Any]:
        return {
            "answer_addresses_original_question": {
                "type": "noul",
                "instructions": (
                    "Does the current candidate directly answer the original question, "
                    "rather than only an intermediate subquestion?"
                ),
            },
            "evidence_is_sufficient": {
                "type": "noul",
                "instructions": (
                    "Does the accumulated evidence support a final answer without an "
                    "unstated or missing fact?"
                ),
            },
            "planned_dependencies_resolved": {
                "type": "noul",
                "instructions": (
                    "Are all dependencies explicitly introduced by the current generated "
                    "plan resolved in the accumulated state? Do not infer the benchmark's "
                    "hidden gold hop count."
                ),
            },
            "candidate_conflict_present": {
                "type": "noul",
                "instructions": (
                    "Do the current candidates or evidence fragments materially conflict?"
                ),
            },
            "progress_since_previous_step": {
                "type": "score",
                "instructions": (
                    "How much usable progress did the most recent action add to the task state?"
                ),
                "criteria": [
                    "No meaningful change",
                    "Minor new information",
                    "Useful partial progress",
                    "A required dependency was resolved",
                    "The task is ready for finalization",
                ],
            },
            "dominant_next_need": {
                "type": "choice",
                "instructions": (
                    "Which capability would add the greatest marginal value to the current "
                    "state? This is a diagnosis, not a fixed role transition."
                ),
                "criteria": {
                    "revise_plan": "The current decomposition is missing or inadequate.",
                    "retrieve_evidence": "A required fact or paragraph is missing.",
                    "resolve_bridge": "An intermediate entity or relation must be resolved.",
                    "compose_evidence": "Available facts must be combined.",
                    "verify_candidate": "A candidate exists but needs evidence checking.",
                    "integrate_answer": (
                        "Evidence is sufficient and only concise finalization remains."
                    ),
                    "recover_failure": "The previous action failed or produced unusable output.",
                    "ready_to_stop": "A supported direct final answer is already ready.",
                },
            },
        }

    @staticmethod
    def _normalize_semantic_answers(answers: Mapping[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for name, answer in answers.items():
            if not isinstance(answer, Mapping):
                normalized[str(name)] = answer
                continue
            item: dict[str, Any] = {}
            for key in ("noul", "score", "choice", "confidence"):
                value = answer.get(key)
                if isinstance(value, (str, int, float, bool)):
                    item[key] = value
            probabilities = answer.get("probabilities")
            if isinstance(probabilities, Mapping):
                item["probabilities"] = {
                    str(key): float(value)
                    for key, value in probabilities.items()
                    if isinstance(value, (int, float))
                }
            normalized[str(name)] = item
        return normalized

    def _diagnose(self, state: Mapping[str, Any]):
        questions = self._semantic_questions()
        if not hasattr(self.client, "judge"):
            raise TypeError(
                "semantic_diagnosis requires a decision client with judge(state, questions)"
            )
        judgment = self.client.judge(state, questions)
        return (
            self._normalize_semantic_answers(judgment.answers),
            judgment,
            questions,
        )

    @staticmethod
    def _question(
        root: bool, options: tuple[RouteOption, ...]
    ) -> tuple[str, dict[str, Any]]:
        name = "route_bundle" if root else "next_action"
        instructions = (
            "Choose the smallest non-redundant bundle whose roles add distinct value "
            "for solving the task. Width is an upper bound, not a target."
            if root
            else "Choose exactly one useful next role, or stop when the current path is ready."
        )
        return name, {
            name: {
                "type": "choice",
                "instructions": instructions,
                "criteria": {
                    option.option_id: option.description for option in options
                },
            }
        }

    @staticmethod
    def _sequential_root_question(
        options: tuple[RouteOption, ...],
        selection_number: int,
        selected_views: tuple[Mapping[str, Any], ...],
    ) -> tuple[str, dict[str, Any]]:
        name = f"root_agent_{selection_number}"
        if not selected_views:
            instructions = (
                "Choose the single best role to open the first reasoning path for "
                "this task. Every listed role is currently available."
            )
        else:
            selected_roles = ", ".join(
                str(view.get("role_name", view.get("candidate_id")))
                for view in selected_views
            )
            instructions = (
                "The root paths already include: "
                f"{selected_roles}. Choose one remaining role that adds the greatest "
                "distinct, complementary, non-redundant value, or finish the bundle "
                "when another path is unnecessary."
            )
        return name, {
            name: {
                "type": "choice",
                "instructions": instructions,
                "criteria": {
                    option.option_id: option.description for option in options
                },
            }
        }

    @staticmethod
    def _assignment(view: Mapping[str, Any]) -> PlannerAssignment:
        profile = view.get("routing_profile") or {}
        expected = tuple(profile.get("expected_contribution") or ())
        contribution = "; ".join(map(str, expected)).strip()
        if not contribution:
            contribution = str(view.get("role_goal", "Provide a role-specific result"))
        role_name = str(view.get("role_name", "selected role"))
        subtask = (
            f"Act as {role_name}. Review the task and any previous output, follow the "
            "routing profile, and return a valid result under the role's output contract."
        )
        return PlannerAssignment(
            candidate_id=int(view["candidate_id"]),
            subtask=subtask,
            expected_contribution=contribution,
        )

    def _decision_from_option(
        self,
        option: RouteOption,
        signature: tuple[str, ...],
    ) -> PlannerDecision:
        if option.stop:
            return PlannerDecision(True, signature, (), ())
        views = self.agent_graph.public_agent_views()
        assignments = tuple(
            self._assignment(views[index]) for index in option.candidate_ids
        )
        return PlannerDecision(False, signature, (), assignments)

    def _fallback(
        self,
        global_info,
        signature: tuple[str, ...],
        candidate_views: list[dict],
    ) -> PlannerDecision:
        if global_info.path_id != -1 and self._previous_outputs(global_info):
            return PlannerDecision(True, signature, ("decision model fallback",), ())
        for view in candidate_views:
            if view.get("available"):
                return PlannerDecision(
                    False,
                    signature,
                    ("decision model fallback",),
                    (self._assignment(view),),
                )
        if global_info.path_id != -1:
            return PlannerDecision(True, signature, ("no available candidate",), ())
        raise RuntimeError("No available agent for decision planner fallback")

    def _sequential_root_decision(
        self,
        global_info,
        capacity: int,
        signature: tuple[str, ...],
        candidate_views: list[dict],
        routing_state: Mapping[str, Any],
        semantic_diagnostics: Mapping[str, Any] | None = None,
    ) -> tuple[PlannerDecision, list, list[dict], list[dict], str | None]:
        all_views = self.agent_graph.public_agent_views()
        views_by_id = {
            int(view["candidate_id"]): view for view in all_views
        }
        views = candidate_views
        selected_ids: list[int] = []
        results = []
        states: list[dict] = []
        question_sets: list[dict] = []
        error_type = None

        while len(selected_ids) < capacity:
            selected_set = set(selected_ids)
            remaining = [
                view
                for view in views
                if view.get("available")
                and int(view["candidate_id"]) not in selected_set
            ]
            if not remaining:
                break
            options = self.bundle_builder.sequential_root_options(
                views,
                selected_ids=selected_ids,
                allow_finish=bool(selected_ids),
            )
            selected_views = tuple(views_by_id[index] for index in selected_ids)
            question_name, questions = self._sequential_root_question(
                options,
                selection_number=len(selected_ids) + 1,
                selected_views=selected_views,
            )
            question_sets.append(questions)
            state = self._state(
                global_info,
                capacity - len(selected_ids),
                signature,
                candidate_views,
                routing_state,
            )
            if semantic_diagnostics:
                state["semantic_diagnostics"] = dict(semantic_diagnostics)
            state["root_selection"] = {
                "mode": "sequential",
                "selection_number": len(selected_ids) + 1,
                "selected_candidate_ids": list(selected_ids),
                "selected_roles": [
                    view.get("role_name") for view in selected_views
                ],
            }
            state["constraints"].update(
                {
                    "finish_bundle_allowed": bool(selected_ids),
                    "duplicate_selection_allowed": False,
                }
            )
            states.append(state)

            try:
                result = self.client.decide(state, questions, question_name)
                results.append(result)
                by_id = {option.option_id: option for option in options}
                if result.choice not in by_id:
                    raise DecisionModelResponseError(
                        f"Unknown decision option {result.choice!r}"
                    )
                option = by_id[result.choice]
            except DecisionModelResponseError as error:
                error_type = type(error).__name__
                if not selected_ids:
                    selected_ids.append(int(remaining[0]["candidate_id"]))
                break

            if option.stop:
                break
            selected_ids.append(option.candidate_ids[0])

        if not selected_ids:
            raise RuntimeError("Sequential root selection produced no agent")
        assignments = tuple(
            self._assignment(views_by_id[index]) for index in selected_ids
        )
        return (
            PlannerDecision(False, signature, (), assignments),
            results,
            states,
            question_sets,
            error_type,
        )

    def propose(self, global_info, capacity):
        if capacity < 1:
            raise ValueError("Routing requires positive capacity")
        root = global_info.path_id == -1
        signature = self._task_signature(global_info)
        all_views = self.agent_graph.public_agent_views()
        routing_snapshot = self.routing_guard.evaluate(global_info, all_views)
        views = routing_snapshot.eligible_views(all_views)
        sequential_root = root and self.root_selection_mode == "sequential"
        require_candidate_before_stop = (
            bool(routing_snapshot.state.get("enabled", False))
            and str(routing_snapshot.state.get("profile", "")).startswith(
                ("gaia_", "musique_")
            )
        )
        stop_ready = bool(
            routing_snapshot.state.get(
                "final_answer_exists"
                if str(routing_snapshot.state.get("profile", "")).startswith("musique_")
                else "candidate_answer_exists",
                False,
            )
        )
        allow_stop = (
            not root
            and (
                not require_candidate_before_stop
                or stop_ready
            )
        )
        guard_stop = not root and not views
        semantic_diagnostics: dict[str, Any] = {}
        diagnosis_judgment = None
        diagnosis_questions = None
        diagnosis_state = None
        diagnosis_error_type = None
        if (
            self.semantic_diagnosis_enabled
            and not guard_stop
            and (not root or self.semantic_diagnosis_on_root)
        ):
            diagnosis_state = self._state(
                global_info,
                capacity,
                signature,
                views,
                routing_snapshot.state,
            )
            try:
                (
                    semantic_diagnostics,
                    diagnosis_judgment,
                    diagnosis_questions,
                ) = self._diagnose(diagnosis_state)
            except DecisionModelResponseError as error:
                # Diagnosis is advisory. A malformed diagnosis must not prevent the
                # router from making its ordinary typed route decision.
                diagnosis_error_type = type(error).__name__
        if guard_stop:
            decision = PlannerDecision(
                True,
                signature,
                ("routing guard marked the path terminal",),
                (),
            )
            results = []
            route_question_sets = []
            states = [
                self._state(
                    global_info,
                    capacity,
                    signature,
                    views,
                    routing_snapshot.state,
                )
            ]
            error_type = None
        elif sequential_root:
            (
                decision,
                results,
                states,
                route_question_sets,
                error_type,
            ) = self._sequential_root_decision(
                global_info,
                capacity,
                signature,
                views,
                routing_snapshot.state,
                semantic_diagnostics,
            )
        else:
            options = (
                self.bundle_builder.root_options(views, capacity)
                if root
                else self.bundle_builder.next_options(views, allow_stop=allow_stop)
            )
            question_name, questions = self._question(root, options)
            state = self._state(
                global_info,
                capacity,
                signature,
                views,
                routing_snapshot.state,
            )
            if semantic_diagnostics:
                state["semantic_diagnostics"] = dict(semantic_diagnostics)
            states = [state]
            route_question_sets = [questions]
            results = []
            error_type = None
            try:
                result = self.client.decide(state, questions, question_name)
                results.append(result)
                by_id = {option.option_id: option for option in options}
                if result.choice not in by_id:
                    raise DecisionModelResponseError(
                        f"Unknown decision option {result.choice!r}"
                    )
                decision = self._decision_from_option(by_id[result.choice], signature)
            except DecisionModelResponseError as error:
                error_type = type(error).__name__
                decision = self._fallback(global_info, signature, views)
        result = results[-1] if results else None
        fallback = (not guard_stop) and (not results or error_type is not None)
        if decision.stop:
            actions = [ORCHESTRATOR_STOP]
            assignments = [None]
        else:
            actions = [
                self.agent_hash_list[item.candidate_id]
                for item in decision.selections
            ]
            assignments = [item.to_dict() for item in decision.selections]
        decision_id = self.audit.new_id("decision") if self.audit else "decision-model"
        decision_calls = [
            {
                "selected_option": item.choice,
                "confidence": item.confidence,
                "probabilities": dict(item.probabilities),
                "input_tokens": item.input_tokens,
                "latency_ms": item.latency_ms,
            }
            for item in results
        ]
        metadata = {
            "provider": getattr(self.client, "provider", "injected"),
            "model": result.model if result else None,
            "model_version": result.model_version if result else None,
            "selected_option": result.choice if result else None,
            "selected_options": [item.choice for item in results],
            "confidence": result.confidence if result else None,
            "probabilities": dict(result.probabilities) if result else {},
            "latency_ms": sum(item.latency_ms for item in results),
            "root_selection_mode": (
                self.root_selection_mode if root else "next_action"
            ),
            "calls": decision_calls,
            "guard_stop": guard_stop,
        }
        if diagnosis_judgment is not None:
            metadata["semantic_diagnosis"] = {
                "answers": dict(semantic_diagnostics),
                "model": diagnosis_judgment.model,
                "model_version": diagnosis_judgment.model_version,
                "input_tokens": diagnosis_judgment.input_tokens,
                "latency_ms": diagnosis_judgment.latency_ms,
            }
        elif diagnosis_error_type is not None:
            metadata["semantic_diagnosis"] = {
                "answers": {},
                "error_type": diagnosis_error_type,
            }
        planner_tokens = sum(item.input_tokens for item in results) + (
            diagnosis_judgment.input_tokens
            if diagnosis_judgment is not None
            else 0
        )
        proposal = {
            "decision_id": decision_id,
            "actions": actions,
            "assignments": assignments,
            "task_signature": list(decision.task_signature),
            "state_gaps": list(decision.state_gaps),
            "planner_tokens": planner_tokens,
            "fallback": fallback,
            "routing_state": dict(routing_snapshot.state),
            "decision_model": metadata,
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
                "allow_stop": allow_stop,
                "remaining_depth": getattr(global_info, "remaining_depth", None),
                "planner_tokens": planner_tokens,
                "fallback": fallback,
                "error_type": error_type,
                "diagnosis_error_type": diagnosis_error_type,
                "routing_state": dict(routing_snapshot.state),
                "eligible_candidate_ids": list(
                    routing_snapshot.eligible_candidate_ids
                ),
                "masked_candidates": list(routing_snapshot.masked_candidates),
                "planner_input_hash": digest(
                    {
                        "diagnosis": diagnosis_state,
                        "routing": states,
                    }
                ),
                "decision_model": metadata,
            }
            if self.audit_planner_input:
                audit_payload["planner_inputs"] = {
                    "diagnosis": diagnosis_state,
                    "routing": states,
                }
                audit_payload["planner_questions"] = {
                    "diagnosis": diagnosis_questions,
                    "routing": route_question_sets,
                }
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
                "profiles": [
                    dict(view) for view in self.agent_graph.public_agent_views()
                ],
            }
        )

    def training_state_dict(self):
        return {
            "type": "decision_model_planner",
            "fingerprint": self.state_fingerprint(),
        }

    def save_model(self, path=None, tag=None):
        return None
