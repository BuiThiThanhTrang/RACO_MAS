from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from routerlab.config import (
    BudgetConfig,
    DatasetConfig,
    ModelConfig,
    ProtocolConfig,
    RunConfig,
    WorkflowConfig,
    load_config,
)
from routerlab.contracts import (
    BaseTestFeedback,
    CodeArtifact,
    EvaluationSnapshot,
    RouterDecision,
    TaskSpec,
)
from routerlab.evaluator import CodeEvaluator
from routerlab.executors import ExecutorPool, SingleAgent
from routerlab.metrics import aggregate_metrics
from routerlab.model import ReplayModelClient
from routerlab.protocol_v6 import ArtifactLedger, python_presentation_fingerprint
from routerlab.runlog import RunLogger
from routerlab.workflows import WorkflowRunner


class FakeEvaluator(CodeEvaluator):
    def __init__(self) -> None:
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
            summary="sanitized public diagnostic",
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


class QueueRouter:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.states = []

    def select(self, state):
        self.states.append(state)
        return RouterDecision(next(self.actions))


def v6_config(method="e3", *, max_steps=5) -> RunConfig:
    model = ModelConfig(backend="replay", checkpoint="fake")
    return RunConfig(
        run_name="test-v6",
        output_dir="runs",
        dataset=DatasetConfig(limit=1),
        expert_model=model,
        budget=BudgetConfig(
            max_steps=max_steps,
            max_base_executions=8,
            max_wall_time_seconds=10,
        ),
        workflow=WorkflowConfig(method=method, fixed_rounds=1, router="heuristic"),
        protocol=ProtocolConfig(
            version="v6", feedback_policy="sanitized_diagnostic_v2"
        ),
    )


class ProtocolV6Tests(unittest.TestCase):
    task = TaskSpec("Mbpp/1", "Write f", "f")

    def test_v6_configs_load_and_v5_remains_default(self):
        root = Path(__file__).parents[1]
        loaded = load_config(root / "configs" / "mbpp_e1_modal_v6.yaml")
        legacy = load_config(root / "configs" / "mbpp_e1.yaml")
        self.assertEqual(loaded.protocol.version, "v6")
        self.assertEqual(legacy.protocol.version, "v5")

    def test_presentation_fingerprint_ignores_formatting_and_comments(self):
        left = "def f():\n    # note\n    return 1\n"
        right = "def f( ):\n\n  return 1\n"
        self.assertEqual(
            python_presentation_fingerprint(left)[0],
            python_presentation_fingerprint(right)[0],
        )

    def test_candidate_scoped_artifacts_become_stale(self):
        ledger = ArtifactLedger()
        first = CodeArtifact("def f(): return 0", 1)
        second = CodeArtifact("def f(): return 1", 2)
        record = ledger.add(
            "logic_review",
            [{"claim": "wrong result"}],
            producer_action="logic_reviewer",
            producer_step=1,
            code=first,
            event_index=0,
        )
        task_record = ledger.add(
            "edge_case_analysis",
            [{"claim": "empty input"}],
            producer_action="edge_case_analyst",
            producer_step=2,
            code=None,
            event_index=1,
            scope="task_specification",
        )
        self.assertTrue(record.is_fresh(first))
        self.assertFalse(record.is_fresh(second))
        self.assertTrue(task_record.is_fresh(second))

    def test_repair_before_candidate_is_rejected_without_model_call(self):
        client = ReplayModelClient(["def f():\n    return 1"])
        evaluator = FakeEvaluator()
        router = QueueRouter(["repair_agent", "coder", "STOP"])
        result = WorkflowRunner(
            v6_config(),
            evaluator,
            ExecutorPool(client),
            SingleAgent(client),
            router,
        ).run(self.task)

        executor_steps = [step for step in result.steps if not step.router_decision]
        self.assertTrue(executor_steps[0].metadata["action_rejected"])
        self.assertEqual(result.total_steps, 2)
        self.assertEqual(result.total_model_calls, 1)
        self.assertTrue(result.final_success)

    def test_duplicate_candidate_does_not_rerun_base(self):
        client = ReplayModelClient(
            [
                "def f():\n    return 0",
                "def f():\n    return 0",
                "def f():\n    return 1",
            ]
        )
        evaluator = FakeEvaluator()
        router = QueueRouter(["coder", "coder", "coder", "STOP"])
        result = WorkflowRunner(
            v6_config(max_steps=3),
            evaluator,
            ExecutorPool(client),
            SingleAgent(client),
            router,
        ).run(self.task)

        executor_steps = [step for step in result.steps if not step.router_decision]
        self.assertTrue(executor_steps[1].metadata["duplicate_candidate"])
        self.assertFalse(executor_steps[1].base_test_ran)
        self.assertEqual(evaluator.calls, [(1, False), (2, False), (2, True)])
        self.assertTrue(result.final_success)

    def test_single_agent_retries_after_invalid_initial_candidate(self):
        client = ReplayModelClient(
            ["def wrong():\n    return 0", "def f():\n    return 1"]
        )
        evaluator = FakeEvaluator()
        result = WorkflowRunner(
            v6_config(method="e1", max_steps=2),
            evaluator,
            ExecutorPool(client),
            SingleAgent(client),
        ).run(self.task)

        self.assertTrue(result.final_success)
        self.assertEqual(result.total_steps, 2)
        self.assertEqual(evaluator.calls, [(1, False), (1, True)])
        self.assertEqual(
            result.steps[0].metadata["status"], "contract_failure"
        )

    def test_nested_state_and_request_audit_are_recorded(self):
        client = ReplayModelClient(["def f():\n    return 1"])
        evaluator = FakeEvaluator()
        router = QueueRouter(["coder", "STOP"])
        result = WorkflowRunner(
            v6_config(max_steps=1),
            evaluator,
            ExecutorPool(client),
            SingleAgent(client),
            router,
        ).run(self.task)

        state = router.states[0].to_router_payload()
        self.assertIn("current_state", state)
        self.assertIn("agent_profiles", state)
        self.assertNotIn("plus_passed", json.dumps(state).lower())
        requests = result.protocol_data["requests"]
        self.assertEqual(len(requests), 1)
        self.assertIn("messages", requests[0])
        self.assertIn("token_count_method", requests[0])

    def test_v6_logging_is_auditable_and_keeps_plus_private(self):
        client = ReplayModelClient(["def f():\n    return 1"])
        evaluator = FakeEvaluator()
        router = QueueRouter(["coder", "STOP"])
        result = WorkflowRunner(
            v6_config(max_steps=1),
            evaluator,
            ExecutorPool(client),
            SingleAgent(client),
            router,
        ).run(self.task)

        with tempfile.TemporaryDirectory() as directory:
            RunLogger(directory).log_episode(result)
            episode_dir = Path(directory, "episodes", "Mbpp_1")
            self.assertTrue((episode_dir / "state_snapshots.jsonl").exists())
            self.assertTrue((episode_dir / "trajectory.jsonl").exists())
            self.assertTrue((episode_dir / "requests.jsonl").exists())
            self.assertTrue((episode_dir / "artifacts").is_dir())
            public = (episode_dir / "result.json").read_text(encoding="utf-8")
            private = (episode_dir / "private_final_eval.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("plus_passed", public)
            self.assertIn("plus_passed", private)

    def test_end_to_end_metrics_include_base_recovery_and_progress(self):
        client = ReplayModelClient(
            ["def f():\n    return 0", "def f():\n    return 1"]
        )
        evaluator = FakeEvaluator()
        router = QueueRouter(["coder", "repair_agent", "STOP"])
        result = WorkflowRunner(
            v6_config(max_steps=2),
            evaluator,
            ExecutorPool(client),
            SingleAgent(client),
            router,
        ).run(self.task)
        metrics = aggregate_metrics([result])
        self.assertEqual(metrics["quality"]["base_recovery_opportunities"], 1)
        self.assertEqual(metrics["quality"]["base_recovery_rate"], 1.0)
        self.assertGreater(metrics["routing"]["semantic_progress_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
