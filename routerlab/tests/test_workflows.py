from __future__ import annotations

import unittest

from routerlab.config import (
    BudgetConfig,
    DatasetConfig,
    ModelConfig,
    RunConfig,
    WorkflowConfig,
)
from routerlab.contracts import (
    BaseTestFeedback,
    EvaluationSnapshot,
    RouterDecision,
    TaskSpec,
)
from routerlab.evaluator import CodeEvaluator
from routerlab.executors import ExecutorPool, SingleAgent
from routerlab.model import ReplayModelClient
from routerlab.routers import HeuristicRouter
from routerlab.workflows import WorkflowRunner


class FakeEvaluator(CodeEvaluator):
    def __init__(self):
        self.calls: list[tuple[int, bool]] = []

    def evaluate(self, task, code, *, include_plus):
        self.calls.append((code.version, include_plus))
        passed = "return 1" in code.source
        feedback = BaseTestFeedback(
            code_version=code.version,
            status="pass" if passed else "fail",
            passed=passed,
            passed_tests=int(passed),
            total_tests=1,
            failure_count=int(not passed),
            summary="public only",
        )
        return feedback, EvaluationSnapshot(
            task_id=task.task_id,
            code_version=code.version,
            base_passed=passed,
            plus_passed=passed if include_plus else None,
            base_passed_tests=int(passed),
            base_total_tests=1,
            plus_passed_tests=int(passed) if include_plus else None,
            plus_total_tests=1 if include_plus else None,
        )


def make_config(
    method: str, router: str | None = None, *, max_steps: int = 10
) -> RunConfig:
    model = ModelConfig(backend="replay", checkpoint="fake")
    return RunConfig(
        run_name="test",
        output_dir="runs",
        dataset=DatasetConfig(limit=1),
        expert_model=model,
        budget=BudgetConfig(
            max_steps=max_steps,
            max_base_executions=6,
            max_wall_time_seconds=10,
        ),
        workflow=WorkflowConfig(method=method, fixed_rounds=1, router=router),
    )


class StopRouter:
    def select(self, state):
        return RouterDecision("STOP")


class QueueRouter:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.states = []

    def select(self, state):
        self.states.append(state)
        return RouterDecision(next(self.actions))


class WorkflowTests(unittest.TestCase):
    task = TaskSpec("Mbpp/1", "Write f", "f")

    def test_e1_automatically_tests_each_changed_code(self):
        client = ReplayModelClient(["def f():\n    return 0", "def f():\n    return 1"])
        evaluator = FakeEvaluator()
        runner = WorkflowRunner(
            make_config("e1"),
            evaluator,
            ExecutorPool(client),
            SingleAgent(client),
        )
        result = runner.run(self.task)
        self.assertTrue(result.final_success)
        self.assertEqual(result.termination_reason, "base_passed")
        self.assertEqual(evaluator.calls, [(1, False), (2, False), (2, True)])
        self.assertEqual(result.total_base_executions, 3)

    def test_unchanged_code_does_not_reexecute_base_tests(self):
        client = ReplayModelClient(
            [
                "def f():\n    return 0",
                "def f():\n    return 0",
                "def f():\n    return 1",
            ]
        )
        evaluator = FakeEvaluator()
        runner = WorkflowRunner(
            make_config("e1"),
            evaluator,
            ExecutorPool(client),
            SingleAgent(client),
        )
        result = runner.run(self.task)
        self.assertTrue(result.final_success)
        self.assertEqual(evaluator.calls, [(1, False), (2, False), (2, True)])

    def test_e2_runs_the_fixed_expert_sequence(self):
        client = ReplayModelClient(
            [
                "def f():\n    return 0",
                '{"findings": ["wrong result"]}',
                '{"findings": ["check boundary"]}',
                "def f():\n    return 1",
            ]
        )
        evaluator = FakeEvaluator()
        runner = WorkflowRunner(
            make_config("e2"),
            evaluator,
            ExecutorPool(client),
            SingleAgent(client),
        )
        result = runner.run(self.task)
        self.assertEqual(
            [step.action_id for step in result.steps],
            ["coder", "logic_reviewer", "edge_case_analyst", "repair_agent"],
        )
        self.assertEqual(result.termination_reason, "base_passed")

    def test_e3_heuristic_routes_review_review_repair(self):
        client = ReplayModelClient(
            [
                "def f():\n    return 0",
                '{"findings": ["wrong result"]}',
                '{"findings": ["check boundary"]}',
                "def f():\n    return 1",
            ]
        )
        evaluator = FakeEvaluator()
        runner = WorkflowRunner(
            make_config("e3", "heuristic"),
            evaluator,
            ExecutorPool(client),
            SingleAgent(client),
            HeuristicRouter(),
        )
        result = runner.run(self.task)
        executor_actions = [
            step.action_id for step in result.steps if not step.router_decision
        ]
        self.assertEqual(
            executor_actions,
            ["coder", "logic_reviewer", "edge_case_analyst", "repair_agent"],
        )
        self.assertTrue(result.final_success)
        self.assertEqual(result.termination_reason, "router_stop")

    def test_router_stop_is_respected(self):
        client = ReplayModelClient([])
        evaluator = FakeEvaluator()
        runner = WorkflowRunner(
            make_config("e3", "heuristic"),
            evaluator,
            ExecutorPool(client),
            SingleAgent(client),
            StopRouter(),
        )
        result = runner.run(self.task)
        self.assertEqual(result.termination_reason, "router_stop_no_submission")
        self.assertIsNone(result.final_code)
        self.assertEqual(evaluator.calls, [])

    def test_router_can_analyze_before_first_code(self):
        client = ReplayModelClient(
            [
                '{"findings": ["empty input"]}',
                '{"findings": ["preserve invariant"]}',
                "def f():\n    return 1",
            ]
        )
        router = QueueRouter(
            ["edge_case_analyst", "logic_reviewer", "coder", "STOP"]
        )
        evaluator = FakeEvaluator()
        runner = WorkflowRunner(
            make_config("e3", "heuristic"),
            evaluator,
            ExecutorPool(client),
            SingleAgent(client),
            router,
        )
        result = runner.run(self.task)
        self.assertEqual(
            [step.action_id for step in result.steps if not step.router_decision],
            ["edge_case_analyst", "logic_reviewer", "coder"],
        )
        self.assertIsNone(router.states[0].current_code)
        self.assertEqual(router.states[0].code_version, 0)
        self.assertEqual(evaluator.calls, [(1, False), (1, True)])
        self.assertTrue(result.final_success)

    def test_e3_can_review_after_base_tests_pass(self):
        client = ReplayModelClient(
            [
                "def f():\n    return 1",
                '{"findings": ["check behavior beyond the examples"]}',
            ]
        )
        router = QueueRouter(["coder", "logic_reviewer", "STOP"])
        evaluator = FakeEvaluator()
        runner = WorkflowRunner(
            make_config("e3", "heuristic"),
            evaluator,
            ExecutorPool(client),
            SingleAgent(client),
            router,
        )

        result = runner.run(self.task)

        self.assertEqual(
            [step.action_id for step in result.steps if not step.router_decision],
            ["coder", "logic_reviewer"],
        )
        self.assertEqual(result.termination_reason, "router_stop")
        self.assertTrue(result.final_success)

    def test_repair_without_code_is_counted_as_no_op(self):
        client = ReplayModelClient(
            ["def f():\n    return 999", "def f():\n    return 1"]
        )
        router = QueueRouter(["repair_agent", "coder", "STOP"])
        evaluator = FakeEvaluator()
        runner = WorkflowRunner(
            make_config("e3", "heuristic"),
            evaluator,
            ExecutorPool(client),
            SingleAgent(client),
            router,
        )
        result = runner.run(self.task)
        executor_steps = [step for step in result.steps if not step.router_decision]
        self.assertTrue(executor_steps[0].metadata["no_op"])
        self.assertEqual(executor_steps[0].code_version_after, 0)
        self.assertEqual(evaluator.calls, [(1, False), (1, True)])
        self.assertTrue(result.final_success)

    def test_executor_selected_with_zero_steps_is_a_budget_violation(self):
        client = ReplayModelClient(["def f():\n    return 1"])
        router = QueueRouter(["coder", "edge_case_analyst"])
        evaluator = FakeEvaluator()
        runner = WorkflowRunner(
            make_config("e3", "heuristic", max_steps=1),
            evaluator,
            ExecutorPool(client),
            SingleAgent(client),
            router,
        )

        result = runner.run(self.task)

        self.assertEqual(result.total_steps, 1)
        self.assertEqual(result.termination_reason, "executor_selected_with_zero_steps")
        self.assertEqual(router.states[-1].budget.remaining_steps, 0)
        executor_steps = [step for step in result.steps if not step.router_decision]
        self.assertEqual([step.action_id for step in executor_steps], ["coder"])
        final_router_step = result.steps[-1]
        self.assertTrue(final_router_step.metadata["budget_violation"])
        self.assertFalse(final_router_step.metadata["executable"])
        self.assertEqual(
            final_router_step.metadata["non_execution_reason"], "no_remaining_steps"
        )

    def test_stop_at_zero_steps_is_free_and_valid(self):
        client = ReplayModelClient(["def f():\n    return 1"])
        router = QueueRouter(["coder", "STOP"])
        evaluator = FakeEvaluator()
        runner = WorkflowRunner(
            make_config("e3", "heuristic", max_steps=1),
            evaluator,
            ExecutorPool(client),
            SingleAgent(client),
            router,
        )

        result = runner.run(self.task)

        self.assertEqual(result.total_steps, 1)
        self.assertEqual(result.termination_reason, "router_stop")
        self.assertEqual(router.states[-1].budget.remaining_steps, 0)
        self.assertFalse(result.steps[-1].metadata["budget_violation"])
        self.assertTrue(result.steps[-1].metadata["executable"])

    def test_latest_code_is_submitted_even_when_base_score_regresses(self):
        client = ReplayModelClient(
            ["def f():\n    return 1", "def f():\n    return 0"]
        )
        router = QueueRouter(["coder", "coder", "STOP"])
        evaluator = FakeEvaluator()
        runner = WorkflowRunner(
            make_config("e3", "heuristic", max_steps=2),
            evaluator,
            ExecutorPool(client),
            SingleAgent(client),
            router,
        )

        result = runner.run(self.task)

        self.assertEqual(result.final_code.version, 2)
        self.assertIn("return 0", result.final_code.source)
        self.assertFalse(result.final_success)
        self.assertEqual(evaluator.calls, [(1, False), (2, False), (2, True)])


if __name__ == "__main__":
    unittest.main()
