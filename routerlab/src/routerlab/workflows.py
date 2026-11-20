from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol

from .budget import BudgetTracker
from .config import ProtocolConfig, RunConfig
from .contracts import (
    BaseTestFeedback,
    CodeArtifact,
    EpisodeResult,
    EvaluationSnapshot,
    ExecutorResult,
    ModelUsage,
    RouterDecision,
    ROUTER_ACTIONS,
    RoutingState,
    STOP_ACTION,
    StepRecord,
    TaskSpec,
)
from .evaluator import CodeEvaluator
from .executors import ExecutorPool, SingleAgent
from .protocol_v6 import (
    ArtifactLedger,
    RoutingStateV6,
    base_feedback_artifact,
    build_executor_payload,
    finding_records,
    python_presentation_fingerprint,
    validate_python_candidate,
)


class Router(Protocol):
    def select(self, state: Any) -> RouterDecision: ...


def _empty_progress() -> dict[str, Any]:
    return {
        "calls": 0,
        "action_rejections": 0,
        "generation_failures": 0,
        "duplicate_candidates": 0,
        "consecutive_no_progress": 0,
        "last_result": None,
    }


@dataclass
class _Session:
    task: TaskSpec
    method: str
    evaluator: CodeEvaluator
    budget: BudgetTracker
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    code: CodeArtifact | None = None
    base_feedback: BaseTestFeedback | None = None
    logic_review: tuple[str, ...] = ()
    edge_cases: tuple[str, ...] = ()
    action_history: list[str] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    initial_evaluation: EvaluationSnapshot | None = None
    final_evaluation: EvaluationSnapshot | None = None
    ledger: ArtifactLedger = field(default_factory=ArtifactLedger)
    role_progress: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    state_snapshots: list[dict[str, Any]] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_v6(self) -> bool:
        return self.protocol.version == "v6"

    def _event(self, event: str, **payload: Any) -> int:
        value = {
            "event_index": len(self.events),
            "event": event,
            "code_version": self.code.version if self.code else 0,
            **payload,
        }
        self.events.append(value)
        return value["event_index"]

    def _progress(self, role: str) -> dict[str, Any]:
        if role not in self.role_progress:
            self.role_progress[role] = _empty_progress()
        return self.role_progress[role]

    def _fresh_claims(self, kind: str) -> tuple[str, ...]:
        record = self.ledger.fresh(kind, self.code)
        if record is None or not isinstance(record.content, list):
            return ()
        return tuple(
            str(item.get("claim", "")).strip()
            for item in record.content
            if isinstance(item, dict) and str(item.get("claim", "")).strip()
        )

    def executor_context(self) -> dict | None:
        if not self.is_v6:
            return None
        return build_executor_payload(
            task=self.task,
            code=self.code,
            ledger=self.ledger,
            role_progress=self.role_progress,
            recent_events=self.events,
            config=self.protocol,
        )

    def evaluate_current(self, *, include_plus: bool) -> EvaluationSnapshot:
        if self.code is None:
            raise RuntimeError("Cannot evaluate before code exists")
        feedback, snapshot = self.evaluator.evaluate(
            self.task, self.code, include_plus=include_plus
        )
        self.budget.record_base()
        self.base_feedback = feedback
        if self.is_v6 and not include_plus:
            content = base_feedback_artifact(
                feedback,
                self.code,
                self.protocol.feedback_policy,
            )
            event_index = self._event(
                "base_execution",
                status=feedback.status,
                passed=feedback.passed,
                passed_tests=feedback.passed_tests,
                total_tests=feedback.total_tests,
            )
            self.ledger.add(
                "base_execution",
                content,
                producer_action="automatic_base_evaluator",
                producer_step=self.budget.steps,
                code=self.code,
                event_index=event_index,
            )
        return snapshot

    def _record_request(self, result: ExecutorResult) -> dict[str, Any]:
        metadata = dict(result.metadata)
        if not self.is_v6 or not metadata:
            return {}
        audit = {
            "request_index": len(self.requests),
            "executor_id": result.executor_id,
            **metadata,
        }
        self.requests.append(audit)
        return {
            "packed_input_tokens": metadata.get("packed_input_tokens"),
            "token_count_method": metadata.get("token_count_method"),
            "trimmed": metadata.get("trimmed", []),
            "delivered_evidence": metadata.get("delivered_evidence", []),
            "failure_origin": metadata.get("failure_origin"),
        }

    def apply_executor(self, result: ExecutorResult) -> EvaluationSnapshot | None:
        if self.is_v6:
            return self._apply_executor_v6(result)
        return self._apply_executor_v5(result)

    def _apply_executor_v5(self, result: ExecutorResult) -> EvaluationSnapshot | None:
        self.budget.record_step()
        self.budget.record_model(result.usage)
        before = self.code.version if self.code else 0
        changed = False
        evaluation = None
        is_first_code = self.code is None and result.code is not None
        if result.no_op:
            pass
        elif result.code is not None:
            source = result.code.strip()
            if self.code is None or source != self.code.source.strip():
                if not self.budget.can_run_base(reserve_final=True):
                    raise RuntimeError("Code changed without an available automatic Base Test")
                self.code = CodeArtifact(source=source, version=before + 1)
                changed = True
                self.logic_review = ()
                self.edge_cases = ()
                evaluation = self.evaluate_current(include_plus=False)
                if is_first_code:
                    self.initial_evaluation = evaluation
        elif result.executor_id == "logic_reviewer":
            self.logic_review = result.findings
        elif result.executor_id == "edge_case_analyst":
            self.edge_cases = result.findings
        after = self.code.version if self.code else before
        self.action_history.append(result.executor_id)
        self.steps.append(
            StepRecord(
                step_index=len(self.steps),
                action_id=result.executor_id,
                code_version_before=before,
                code_version_after=after,
                model_usage=result.usage,
                latency_ms=result.latency_ms,
                base_test_ran=changed,
                metadata={
                    "code_changed": changed,
                    "finding_count": len(result.findings),
                    "no_op": result.no_op,
                    "no_op_reason": result.no_op_reason or None,
                    "base_status": self.base_feedback.status
                    if changed and self.base_feedback
                    else None,
                    "base_passed": self.base_feedback.passed
                    if changed and self.base_feedback
                    else None,
                    "base_passed_tests": self.base_feedback.passed_tests
                    if changed and self.base_feedback
                    else None,
                    "base_total_tests": self.base_feedback.total_tests
                    if changed and self.base_feedback
                    else None,
                },
            )
        )
        return evaluation

    def _apply_executor_v6(self, result: ExecutorResult) -> EvaluationSnapshot | None:
        self.budget.record_step()
        self.budget.record_model(result.usage)
        before = self.code.version if self.code else 0
        before_fingerprint = (
            python_presentation_fingerprint(self.code.source)[0] if self.code else None
        )
        before_base_signature = (
            (
                self.base_feedback.status,
                self.base_feedback.passed,
                self.base_feedback.passed_tests,
                self.base_feedback.total_tests,
                self.base_feedback.failure_count,
            )
            if self.base_feedback
            else None
        )
        changed = False
        candidate_changed = False
        duplicate_candidate = False
        evaluation = None
        artifact_ids: list[str] = []
        meaningful_evidence_delta = False
        status = "ok_no_change"
        failure_reason = result.no_op_reason or None
        request_summary = self._record_request(result)
        is_first_code = self.code is None and result.code is not None

        if result.no_op:
            status = (
                "context_budget_exceeded"
                if result.no_op_reason == "context_budget_exceeded"
                else "ok_no_change"
            )
            if result.no_op_reason == "context_budget_exceeded":
                event_index = self._event(
                    "generation_failure",
                    role=result.executor_id,
                    status=status,
                )
                record = self.ledger.add(
                    "generation_failure",
                    {
                        "role": result.executor_id,
                        "failure_origin": "context_budget",
                        "failure_type": result.metadata.get("failure_type"),
                        "message": result.metadata.get("error"),
                    },
                    producer_action=result.executor_id,
                    producer_step=self.budget.steps,
                    code=self.code,
                    event_index=event_index,
                )
                artifact_ids.append(record.artifact_id)
        elif result.code is not None:
            source = result.code.strip()
            valid, reason = validate_python_candidate(source, self.task.entry_point)
            if not valid:
                status = "contract_failure"
                failure_reason = reason
                event_index = self._event(
                    "candidate_rejected",
                    role=result.executor_id,
                    reason=reason,
                )
                record = self.ledger.add(
                    "rejected_candidate",
                    {
                        "reason": reason,
                        "source_sha256": hashlib.sha256(
                            source.encode("utf-8")
                        ).hexdigest(),
                        "source": source,
                    },
                    producer_action=result.executor_id,
                    producer_step=self.budget.steps,
                    code=self.code,
                    event_index=event_index,
                )
                artifact_ids.append(record.artifact_id)
            elif self.code is not None and source == self.code.source.strip():
                duplicate_candidate = True
                status = "ok_no_change"
                self._event(
                    "duplicate_candidate",
                    role=result.executor_id,
                    presentation_duplicate=True,
                )
            else:
                if not self.budget.can_run_base(reserve_final=True):
                    raise RuntimeError("Code changed without an available automatic Base Test")
                self.code = CodeArtifact(source=source, version=before + 1)
                changed = True
                new_fingerprint, method = python_presentation_fingerprint(source)
                candidate_changed = before_fingerprint != new_fingerprint
                event_index = self._event(
                    "candidate_registered",
                    role=result.executor_id,
                    source_changed=True,
                    candidate_changed=candidate_changed,
                )
                record = self.ledger.add(
                    "candidate",
                    {
                        "source": source,
                        "raw_sha256": self.code.sha256,
                        "presentation_fingerprint": new_fingerprint,
                        "presentation_method": method,
                    },
                    producer_action=result.executor_id,
                    producer_step=self.budget.steps,
                    code=self.code,
                    event_index=event_index,
                )
                artifact_ids.append(record.artifact_id)
                evaluation = self.evaluate_current(include_plus=False)
                after_base_signature = (
                    self.base_feedback.status,
                    self.base_feedback.passed,
                    self.base_feedback.passed_tests,
                    self.base_feedback.total_tests,
                    self.base_feedback.failure_count,
                )
                meaningful_evidence_delta = (
                    before_base_signature != after_base_signature
                )
                base_record = self.ledger.latest.get("base_execution")
                if base_record:
                    artifact_ids.append(base_record.artifact_id)
                if is_first_code:
                    self.initial_evaluation = evaluation
                status = "ok_artifact"
        elif result.executor_id in {"logic_reviewer", "edge_case_analyst"}:
            kind = (
                "logic_review"
                if result.executor_id == "logic_reviewer"
                else "edge_case_analysis"
            )
            records = finding_records(
                result.findings,
                result.finding_records,
                kind=kind,
                step=self.budget.steps,
            )
            if not records:
                status = "contract_failure"
                failure_reason = "empty_findings"
                self._event(
                    "finding_rejected",
                    role=result.executor_id,
                    reason="empty_findings",
                )
            else:
                scope = "candidate" if self.code else "task_specification"
                previous = self.ledger.latest.get(kind)
                event_index = self._event(
                    "evidence_created",
                    role=result.executor_id,
                    kind=kind,
                    finding_count=len(records),
                    scope=scope,
                )
                record = self.ledger.add(
                    kind,
                    records,
                    producer_action=result.executor_id,
                    producer_step=self.budget.steps,
                    code=self.code,
                    event_index=event_index,
                    scope=scope,
                )
                artifact_ids.append(record.artifact_id)
                meaningful_evidence_delta = (
                    previous is None or previous.sha256 != record.sha256
                )
                status = "ok_artifact"

        self.logic_review = self._fresh_claims("logic_review")
        self.edge_cases = self._fresh_claims("edge_case_analysis")
        after = self.code.version if self.code else before
        semantic_delta = bool(
            candidate_changed
            or meaningful_evidence_delta
        )
        progress = self._progress(result.executor_id)
        progress["calls"] += 1
        if duplicate_candidate:
            progress["duplicate_candidates"] += 1
        if status in {"contract_failure", "context_budget_exceeded"}:
            progress["generation_failures"] += 1
        if semantic_delta:
            progress["consecutive_no_progress"] = 0
        else:
            progress["consecutive_no_progress"] += 1
        progress["last_result"] = {
            "status": status,
            "source_changed": changed,
            "candidate_changed": candidate_changed,
            "duplicate_candidate": duplicate_candidate,
            "artifact_ids": artifact_ids,
            "input_candidate_version": before,
            "output_candidate_version": after,
            "semantic_delta": semantic_delta,
            "failure_reason": failure_reason,
        }
        self.action_history.append(result.executor_id)
        self._event(
            "role_result",
            role=result.executor_id,
            status=status,
            semantic_delta=semantic_delta,
            artifact_ids=artifact_ids,
        )
        self.steps.append(
            StepRecord(
                step_index=len(self.steps),
                action_id=result.executor_id,
                code_version_before=before,
                code_version_after=after,
                model_usage=result.usage,
                latency_ms=result.latency_ms,
                base_test_ran=changed,
                metadata={
                    "protocol_version": "v6",
                    "status": status,
                    "source_changed": changed,
                    "code_changed": changed,
                    "candidate_changed": candidate_changed,
                    "duplicate_candidate": duplicate_candidate,
                    "semantic_delta": semantic_delta,
                    "artifact_ids": artifact_ids,
                    "finding_count": len(result.findings),
                    "no_op": result.no_op,
                    "no_op_reason": failure_reason,
                    "base_status": self.base_feedback.status
                    if changed and self.base_feedback
                    else None,
                    "base_passed": self.base_feedback.passed
                    if changed and self.base_feedback
                    else None,
                    "base_passed_tests": self.base_feedback.passed_tests
                    if changed and self.base_feedback
                    else None,
                    "base_total_tests": self.base_feedback.total_tests
                    if changed and self.base_feedback
                    else None,
                    **request_summary,
                },
            )
        )
        return evaluation

    def record_action_rejection(self, action_id: str, reason: str) -> None:
        self.budget.record_step()
        before = self.code.version if self.code else 0
        progress = self._progress(action_id)
        progress["calls"] += 1
        progress["action_rejections"] += 1
        progress["consecutive_no_progress"] += 1
        progress["last_result"] = {
            "status": "action_rejected",
            "reason": reason,
            "input_candidate_version": before,
            "output_candidate_version": before,
            "semantic_delta": False,
        }
        self.action_history.append(action_id)
        self._event(
            "action_rejected",
            role=action_id,
            status="action_rejected",
            reason=reason,
            semantic_delta=False,
        )
        self.steps.append(
            StepRecord(
                step_index=len(self.steps),
                action_id=action_id,
                code_version_before=before,
                code_version_after=before,
                model_usage=ModelUsage(),
                metadata={
                    "protocol_version": "v6",
                    "status": "action_rejected",
                    "action_rejected": True,
                    "reason": reason,
                    "no_op": True,
                    "no_op_reason": reason,
                    "semantic_delta": False,
                    "counts_model_call": False,
                },
            )
        )

    def routing_state(self) -> RoutingState | RoutingStateV6:
        if not self.is_v6:
            return RoutingState(
                task_id=self.task.task_id,
                task_prompt=self.task.prompt,
                entry_point=self.task.entry_point,
                current_code=self.code.source if self.code else None,
                code_version=self.code.version if self.code else 0,
                base_feedback=self.base_feedback,
                logic_review=self.logic_review,
                edge_cases=self.edge_cases,
                action_history=tuple(self.action_history),
                budget=self.budget.snapshot(),
                action_space=ROUTER_ACTIONS,
            )
        fingerprint = method = None
        if self.code is not None:
            fingerprint, method = python_presentation_fingerprint(self.code.source)
        latest_failure = self.ledger.latest.get("generation_failure")
        state = RoutingStateV6(
            task=self.task,
            current_code=self.code.source if self.code else None,
            code_version=self.code.version if self.code else 0,
            code_sha256=self.code.sha256 if self.code else None,
            presentation_fingerprint=fingerprint,
            presentation_method=method,
            base_feedback=self.base_feedback,
            logic_review=self._fresh_claims("logic_review"),
            edge_cases=self._fresh_claims("edge_case_analysis"),
            action_history=tuple(self.action_history),
            artifacts=self.ledger.public_latest(self.code),
            role_progress=copy.deepcopy(self.role_progress),
            recent_events=tuple(
                copy.deepcopy(self.events[-self.protocol.recent_event_limit :])
            ),
            latest_generation_failure=(
                latest_failure.public(self.code) if latest_failure else None
            ),
            budget=self.budget.snapshot(),
            code_preview_chars=self.protocol.router_code_preview_chars,
            action_space=ROUTER_ACTIONS,
        )
        self.state_snapshots.append(state.to_router_payload())
        return state

    def record_router(
        self,
        decision: RouterDecision,
        *,
        counts_model_call: bool,
        executable: bool,
        budget_violation: bool = False,
        non_execution_reason: str | None = None,
    ) -> None:
        if counts_model_call:
            self.budget.record_model(decision.usage)
        version = self.code.version if self.code else 0
        if self.is_v6:
            self._event(
                "router_decision",
                action=decision.action_id,
                valid=decision.valid,
                executable=executable,
                non_execution_reason=non_execution_reason,
            )
        self.steps.append(
            StepRecord(
                step_index=len(self.steps),
                action_id=decision.action_id,
                code_version_before=version,
                code_version_after=version,
                model_usage=decision.usage,
                latency_ms=decision.latency_ms,
                router_decision=True,
                metadata={
                    "valid": decision.valid,
                    "executable": executable,
                    "budget_violation": budget_violation,
                    "non_execution_reason": non_execution_reason,
                    "remaining_steps_before_action": self.budget.remaining_steps,
                    "counts_model_call": counts_model_call,
                    "protocol_version": self.protocol.version,
                    **dict(decision.metadata),
                },
            )
        )

    def _protocol_data(self) -> dict[str, Any]:
        if not self.is_v6:
            return {}
        return {
            "protocol_version": "v6",
            "artifacts": self.ledger.export(self.code),
            "role_progress": copy.deepcopy(self.role_progress),
            "events": copy.deepcopy(self.events),
            "state_snapshots": copy.deepcopy(self.state_snapshots),
            "requests": copy.deepcopy(self.requests),
        }

    def finalize(self, termination_reason: str) -> EpisodeResult:
        if self.code is None:
            return EpisodeResult(
                task_id=self.task.task_id,
                method=self.method,
                initial_evaluation=None,
                final_evaluation=None,
                final_code=None,
                steps=tuple(self.steps),
                termination_reason=termination_reason,
                total_steps=self.budget.steps,
                total_model_calls=self.budget.model_calls,
                total_input_tokens=self.budget.input_tokens,
                total_output_tokens=self.budget.output_tokens,
                total_base_executions=self.budget.base_executions,
                total_cost_usd=self.budget.cost_usd,
                wall_time_ms=self.budget.elapsed_ms,
                protocol_data=self._protocol_data(),
            )
        self.final_evaluation = self.evaluate_current(include_plus=True)
        return EpisodeResult(
            task_id=self.task.task_id,
            method=self.method,
            initial_evaluation=self.initial_evaluation,
            final_evaluation=self.final_evaluation,
            final_code=self.code,
            steps=tuple(self.steps),
            termination_reason=termination_reason,
            total_steps=self.budget.steps,
            total_model_calls=self.budget.model_calls,
            total_input_tokens=self.budget.input_tokens,
            total_output_tokens=self.budget.output_tokens,
            total_base_executions=self.budget.base_executions,
            total_cost_usd=self.budget.cost_usd,
            wall_time_ms=self.budget.elapsed_ms,
            protocol_data=self._protocol_data(),
        )


class WorkflowRunner:
    def __init__(
        self,
        config: RunConfig,
        evaluator: CodeEvaluator,
        executors: ExecutorPool,
        single_agent: SingleAgent,
        router: Router | None = None,
        *,
        router_counts_model_call: bool = False,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.executors = executors
        self.single_agent = single_agent
        self.router = router
        self.router_counts_model_call = router_counts_model_call
        if config.budget.max_base_executions < 2:
            raise ValueError("At least two Base Test executions are required")

    def run(self, task: TaskSpec) -> EpisodeResult:
        method = self.config.workflow.method
        session = _Session(
            task=task,
            method=method,
            evaluator=self.evaluator,
            budget=BudgetTracker(self.config.budget),
            protocol=self.config.protocol,
        )
        if method == "e1":
            reason = self._run_e1(session)
        elif method == "e2":
            reason = self._run_e2(session)
        elif method == "e3":
            reason = self._run_e3(session)
        else:
            raise ValueError(f"Unknown workflow method: {method}")
        return session.finalize(reason)

    def _run_executor(self, session: _Session, action: str) -> ExecutorResult:
        return self.executors.run(
            action,
            session.task,
            session.code,
            session.base_feedback,
            session.logic_review,
            session.edge_cases,
            protocol_context=session.executor_context(),
            protocol_config=self.config.protocol,
        )

    def _bootstrap(self, session: _Session, *, single: bool) -> bool:
        if not session.budget.can_execute_step():
            return False
        if single:
            result = self.single_agent.run(
                session.task,
                session.code,
                session.base_feedback,
                protocol_context=session.executor_context(),
                protocol_config=self.config.protocol,
            )
        else:
            result = self._run_executor(session, "coder")
        initial = session.apply_executor(result)
        if initial is None:
            if session.is_v6:
                return False
            raise RuntimeError("Initial generator did not produce valid new code")
        return True

    def _can_edit(self, session: _Session) -> bool:
        return session.budget.can_execute_step() and session.budget.can_run_base(
            reserve_final=True
        )

    def _run_e1(self, session: _Session) -> str:
        if not self._bootstrap(session, single=True) and not session.is_v6:
            raise RuntimeError("Budget does not permit mandatory initial generation")
        while session.code is None or (
            session.base_feedback is not None and not session.base_feedback.passed
        ):
            if not self._can_edit(session):
                return (
                    "step_budget_exhausted"
                    if session.code is not None
                    else "step_budget_exhausted_no_submission"
                )
            result = self.single_agent.run(
                session.task,
                session.code,
                session.base_feedback,
                protocol_context=session.executor_context(),
                protocol_config=self.config.protocol,
            )
            session.apply_executor(result)
        return "base_passed"

    def _run_e2(self, session: _Session) -> str:
        if not self._bootstrap(session, single=False):
            if session.is_v6:
                return "initial_generation_failed_no_submission"
            raise RuntimeError("Budget does not permit mandatory initial generation")
        if session.base_feedback and session.base_feedback.passed:
            return "base_passed"
        for _ in range(self.config.workflow.fixed_rounds):
            for action in ("logic_reviewer", "edge_case_analyst", "repair_agent"):
                if not session.budget.can_execute_step():
                    return "step_budget_exhausted"
                if action == "repair_agent" and not session.budget.can_run_base(
                    reserve_final=True
                ):
                    return "base_test_budget_exhausted"
                result = self._run_executor(session, action)
                session.apply_executor(result)
                if (
                    action == "repair_agent"
                    and session.base_feedback
                    and session.base_feedback.passed
                ):
                    return "base_passed"
        return "fixed_rounds_complete"

    def _run_e3(self, session: _Session) -> str:
        if self.router is None:
            raise ValueError("E3 requires a router")
        while True:
            if not session.budget.can_route():
                return (
                    "wall_time_exhausted"
                    if session.code is not None
                    else "wall_time_exhausted_no_submission"
                )
            state = session.routing_state()
            decision = self.router.select(state)
            budget_violation = (
                decision.action_id != STOP_ACTION
                and session.budget.remaining_steps == 0
            )
            wall_time_exhausted = (
                decision.action_id != STOP_ACTION
                and not session.budget.within_wall_time()
            )
            precondition_rejected = (
                session.is_v6
                and decision.action_id == "repair_agent"
                and session.code is None
            )
            executable = (
                decision.action_id == STOP_ACTION
                or (
                    not budget_violation
                    and not wall_time_exhausted
                    and not precondition_rejected
                )
            )
            non_execution_reason = None
            if budget_violation:
                non_execution_reason = "no_remaining_steps"
            elif wall_time_exhausted:
                non_execution_reason = "wall_time_exhausted_after_router"
            elif precondition_rejected:
                non_execution_reason = "candidate_required"
            session.record_router(
                decision,
                counts_model_call=self.router_counts_model_call,
                executable=executable,
                budget_violation=budget_violation,
                non_execution_reason=non_execution_reason,
            )
            if decision.action_id == STOP_ACTION:
                return (
                    "router_stop"
                    if session.code is not None
                    else "router_stop_no_submission"
                )
            if budget_violation:
                return (
                    "suppressed_by_budget"
                    if session.is_v6
                    else "executor_selected_with_zero_steps"
                )
            if wall_time_exhausted:
                return "wall_time_exhausted_after_router"
            if precondition_rejected:
                session.record_action_rejection(
                    decision.action_id,
                    "candidate_required",
                )
                continue
            result = self._run_executor(session, decision.action_id)
            session.apply_executor(result)
