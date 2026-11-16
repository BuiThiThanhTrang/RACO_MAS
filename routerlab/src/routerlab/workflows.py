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
    ROUTER_ACTIONS,
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

    def evaluate_current(self, *, include_plus: bool) -> EvaluationSnapshot:
        if self.code is None:
            raise RuntimeError("Cannot evaluate before code exists")
        feedback, snapshot = self.evaluator.evaluate(
            self.task, self.code, include_plus=include_plus
        )
        self.budget.record_base()
        self.base_feedback = feedback
        return snapshot

    def apply_executor(
        self,
        result: ExecutorResult,
    ) -> EvaluationSnapshot | None:
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
                # Reviews refer to a particular code version and become stale after edits.
                self.logic_review = ()
                self.edge_cases = ()
                # During the episode, only the original MBPP Base Tests are
                # executable feedback. EvalPlus Plus Tests remain private until
                # the single final evaluation.
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
                    **dict(decision.metadata),
                },
            )
        )

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
            )
        # Submit the current version. The harness does not rescue or override the
        # router by rolling back to a public-Base-best artifact. Regressions and
        # recoveries caused by routing are therefore observable in final quality.
        # Plus Tests are evaluated exactly once, after routing has terminated.
        # Do not reuse the Base-only initial snapshot even when the submitted
        # code version has not changed.
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
        if not session.budget.can_execute_step():
            return False
        if single:
            result = self.single_agent.run(session.task, None, None)
        else:
            result = self.executors.run("coder", session.task, None)
        initial = session.apply_executor(result)
        if initial is None:
            raise RuntimeError("Initial generator did not produce new code")
        return True

    def _can_edit(self, session: _Session) -> bool:
        return session.budget.can_execute_step() and session.budget.can_run_base(
            reserve_final=True
        )

    def _run_e1(self, session: _Session) -> str:
        if not self._bootstrap(session, single=True):
            raise RuntimeError("Budget does not permit mandatory initial generation")
        while session.base_feedback and not session.base_feedback.passed:
            if not self._can_edit(session):
                return "step_budget_exhausted"
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
                if not session.budget.can_execute_step():
                    return "step_budget_exhausted"
                if action == "repair_agent" and not session.budget.can_run_base(
                    reserve_final=True
                ):
                    return "base_test_budget_exhausted"
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
            state = RoutingState(
                task_id=session.task.task_id,
                task_prompt=session.task.prompt,
                entry_point=session.task.entry_point,
                current_code=session.code.source if session.code else None,
                code_version=session.code.version if session.code else 0,
                base_feedback=session.base_feedback,
                logic_review=session.logic_review,
                edge_cases=session.edge_cases,
                action_history=tuple(session.action_history),
                budget=session.budget.snapshot(),
                action_space=ROUTER_ACTIONS,
            )
            decision = self.router.select(state)
            budget_violation = (
                decision.action_id != STOP_ACTION
                and session.budget.remaining_steps == 0
            )
            wall_time_exhausted = (
                decision.action_id != STOP_ACTION
                and not session.budget.within_wall_time()
            )
            executable = (
                decision.action_id == STOP_ACTION
                or (not budget_violation and not wall_time_exhausted)
            )
            non_execution_reason = None
            if budget_violation:
                non_execution_reason = "no_remaining_steps"
            elif wall_time_exhausted:
                non_execution_reason = "wall_time_exhausted_after_router"
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
                return "executor_selected_with_zero_steps"
            if wall_time_exhausted:
                return "wall_time_exhausted_after_router"
            result = self.executors.run(
                decision.action_id,
                session.task,
                session.code,
                session.base_feedback,
                session.logic_review,
                session.edge_cases,
            )
            session.apply_executor(result)
