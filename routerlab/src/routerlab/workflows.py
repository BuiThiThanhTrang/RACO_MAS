from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .budget import BudgetTracker
from .config import RunConfig
from .contracts import (
    BaseTestFeedback,
    CodeArtifact,
    EpisodeResult,
    EvaluationSnapshot,
    ExecutorResult,
    RouterDecision,
    RoutingState,
    STOP_ACTION,
    StepRecord,
    TaskSpec,
)
from .evaluator import CodeEvaluator
from .executors import ExecutorPool, SingleAgent


class Router(Protocol):
    def select(self, state: RoutingState) -> RouterDecision: ...


@dataclass
class _Session:
    task: TaskSpec
    method: str
    evaluator: CodeEvaluator
    budget: BudgetTracker
    code: CodeArtifact | None = None
    base_feedback: BaseTestFeedback | None = None
    logic_review: tuple[str, ...] = ()
    edge_cases: tuple[str, ...] = ()
    action_history: list[str] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    initial_evaluation: EvaluationSnapshot | None = None
    final_evaluation: EvaluationSnapshot | None = None
    best_code: CodeArtifact | None = None
    best_score: float = -1.0

    def _base_score(self, feedback: BaseTestFeedback) -> float:
        if feedback.passed:
            return 1.0
        if feedback.passed_tests is None or not feedback.total_tests:
            return 0.0
        return feedback.passed_tests / feedback.total_tests

    def evaluate_current(self, *, include_plus: bool) -> EvaluationSnapshot:
        if self.code is None:
            raise RuntimeError("Cannot evaluate before code exists")
        feedback, snapshot = self.evaluator.evaluate(
            self.task, self.code, include_plus=include_plus
        )
        self.budget.record_base()
        self.base_feedback = feedback
        score = self._base_score(feedback)
        if score > self.best_score:
            self.best_score = score
            self.best_code = self.code
        return snapshot

    def apply_executor(
        self,
        result: ExecutorResult,
        *,
        include_plus: bool = False,
    ) -> EvaluationSnapshot | None:
        self.budget.record_model(result.usage)
        before = self.code.version if self.code else 0
        changed = False
        evaluation = None
        if result.code is not None:
            source = result.code.strip()
            if self.code is None or source != self.code.source.strip():
                if not self.budget.can_run_base(reserve_final=not include_plus):
                    raise RuntimeError("Code changed without an available automatic Base Test")
                self.code = CodeArtifact(source=source, version=before + 1)
                changed = True
                # Reviews refer to a particular code version and become stale after edits.
                self.logic_review = ()
                self.edge_cases = ()
                evaluation = self.evaluate_current(include_plus=include_plus)
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

    def record_router(self, decision: RouterDecision, *, counts_model_call: bool) -> None:
        if counts_model_call:
            self.budget.record_model(decision.usage)
        version = self.code.version if self.code else 0
        self.steps.append(
            StepRecord(
                step_index=len(self.steps),
                action_id=decision.action_id,
                code_version_before=version,
                code_version_after=version,
                model_usage=decision.usage,
                latency_ms=decision.latency_ms,
                router_decision=True,
                metadata={"valid": decision.valid},
            )
        )

    def finalize(self, termination_reason: str) -> EpisodeResult:
        if self.code is None:
            raise RuntimeError("Episode ended before producing code")
        # Return the best version chosen only by public Base Test performance.
        if self.best_code is not None:
            self.code = self.best_code
        if (
            self.initial_evaluation is not None
            and self.initial_evaluation.code_version == self.code.version
        ):
            self.final_evaluation = self.initial_evaluation
        else:
            self.final_evaluation = self.evaluate_current(include_plus=True)
        return EpisodeResult(
            task_id=self.task.task_id,
            method=self.method,
            initial_evaluation=self.initial_evaluation,
            final_evaluation=self.final_evaluation,
            final_code=self.code,
            steps=tuple(self.steps),
            termination_reason=termination_reason,
            total_model_calls=self.budget.model_calls,
            total_input_tokens=self.budget.input_tokens,
            total_output_tokens=self.budget.output_tokens,
            total_base_executions=self.budget.base_executions,
            total_cost_usd=self.budget.cost_usd,
            wall_time_ms=self.budget.elapsed_ms,
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

    def _bootstrap(self, session: _Session, *, single: bool) -> bool:
        if not session.budget.can_call_model():
            return False
        if single:
            result = self.single_agent.run(session.task, None, None)
        else:
            result = self.executors.run("coder", session.task, None)
        initial = session.apply_executor(result, include_plus=True)
        if initial is None:
            raise RuntimeError("Initial generator did not produce new code")
        session.initial_evaluation = initial
        return True

    def _can_edit(self, session: _Session) -> bool:
        return session.budget.can_call_model() and session.budget.can_run_base(
            reserve_final=True
        )

    def _run_e1(self, session: _Session) -> str:
        if not self._bootstrap(session, single=True):
            raise RuntimeError("Budget does not permit mandatory initial generation")
        while session.base_feedback and not session.base_feedback.passed:
            if not self._can_edit(session):
                return "budget_exhausted"
            result = self.single_agent.run(
                session.task, session.code, session.base_feedback
            )
            session.apply_executor(result)
        return "base_passed"

    def _run_e2(self, session: _Session) -> str:
        if not self._bootstrap(session, single=False):
            raise RuntimeError("Budget does not permit mandatory initial generation")
        if session.base_feedback and session.base_feedback.passed:
            return "base_passed"
        for _ in range(self.config.workflow.fixed_rounds):
            for action in ("logic_reviewer", "edge_case_analyst", "repair_agent"):
                if not session.budget.can_call_model():
                    return "budget_exhausted"
                if action == "repair_agent" and not session.budget.can_run_base(
                    reserve_final=True
                ):
                    return "budget_exhausted"
                result = self.executors.run(
                    action,
                    session.task,
                    session.code,
                    session.base_feedback,
                    session.logic_review,
                    session.edge_cases,
                )
                session.apply_executor(result)
                if (
                    action == "repair_agent"
                    and session.base_feedback
                    and session.base_feedback.passed
                ):
                    return "base_passed"
        return "fixed_rounds_complete"

    def _legal_actions(self, session: _Session) -> tuple[str, ...]:
        snapshot = session.budget.snapshot()
        calls_needed = 2 if self.router_counts_model_call else 1
        actions: list[str] = []
        if snapshot.remaining_model_calls >= calls_needed:
            actions.extend(("logic_reviewer", "edge_case_analyst"))
            if snapshot.remaining_base_executions >= 2:
                actions.extend(("coder", "repair_agent"))
        actions.append(STOP_ACTION)
        return tuple(actions)

    def _run_e3(self, session: _Session) -> str:
        if self.router is None:
            raise ValueError("E3 requires a router")
        if not self._bootstrap(session, single=False):
            raise RuntimeError("Budget does not permit mandatory initial generation")
        while True:
            if self.router_counts_model_call and not session.budget.can_call_model():
                return "budget_exhausted"
            legal = self._legal_actions(session)
            if legal == (STOP_ACTION,) and not self.router_counts_model_call:
                return "budget_exhausted"
            state = RoutingState(
                task_id=session.task.task_id,
                task_prompt=session.task.prompt,
                entry_point=session.task.entry_point,
                current_code=session.code.source if session.code else "",
                code_version=session.code.version if session.code else 0,
                base_feedback=session.base_feedback,
                logic_review=session.logic_review,
                edge_cases=session.edge_cases,
                action_history=tuple(session.action_history),
                budget=session.budget.snapshot(),
                legal_actions=legal,
            )
            decision = self.router.select(state)
            if decision.action_id not in legal:
                decision = RouterDecision(
                    action_id=STOP_ACTION,
                    usage=decision.usage,
                    latency_ms=decision.latency_ms,
                    raw_output=decision.raw_output,
                    valid=False,
                )
            session.record_router(
                decision, counts_model_call=self.router_counts_model_call
            )
            if decision.action_id == STOP_ACTION:
                return "router_stop"
            if not session.budget.can_call_model():
                return "budget_exhausted"
            result = self.executors.run(
                decision.action_id,
                session.task,
                session.code,
                session.base_feedback,
                session.logic_review,
                session.edge_cases,
            )
            session.apply_executor(result)
