from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from routerlab.contracts import (
    CodeArtifact,
    EpisodeResult,
    EvaluationSnapshot,
    StepRecord,
)
from routerlab.metrics import aggregate_metrics
from routerlab.runlog import RunLogger


def episode(initial: bool, final: bool) -> EpisodeResult:
    return EpisodeResult(
        task_id="Mbpp/1",
        method="e3",
        initial_evaluation=EvaluationSnapshot("Mbpp/1", 1, initial, initial),
        final_evaluation=EvaluationSnapshot("Mbpp/1", 2, final, final),
        final_code=CodeArtifact("def f(): return 1", 2),
        steps=(StepRecord(0, "repair_agent", 1, 2),),
        termination_reason="base_passed",
        total_model_calls=2,
        total_input_tokens=10,
        total_output_tokens=5,
        total_base_executions=3,
        total_cost_usd=0.0,
        wall_time_ms=10,
    )


def no_submission_episode() -> EpisodeResult:
    return EpisodeResult(
        task_id="Mbpp/2",
        method="e3",
        initial_evaluation=None,
        final_evaluation=None,
        final_code=None,
        steps=(
            StepRecord(
                0,
                "STOP",
                0,
                0,
                router_decision=True,
                metadata={"valid": True, "counts_model_call": True},
            ),
        ),
        termination_reason="router_stop_no_submission",
        total_model_calls=1,
        total_input_tokens=10,
        total_output_tokens=0,
        total_base_executions=0,
        total_cost_usd=0.0,
        wall_time_ms=2,
    )


class MetricsAndLoggingTests(unittest.TestCase):
    def test_recovery_metric(self):
        metrics = aggregate_metrics([episode(False, True)])
        self.assertEqual(metrics["quality"]["repair_recovery_rate"], 1.0)
        self.assertEqual(metrics["quality"]["absolute_plus_improvement"], 1.0)

    def test_no_submission_is_a_failed_routing_outcome(self):
        metrics = aggregate_metrics([no_submission_episode()])
        self.assertEqual(metrics["quality"]["final_plus_pass_rate"], 0.0)
        self.assertEqual(metrics["routing"]["no_submission_rate"], 1.0)
        self.assertEqual(metrics["routing"]["premature_stop_rate"], 1.0)

    def test_public_trace_does_not_contain_plus_results(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = RunLogger(directory)
            logger.log_episode(episode(False, True))
            public = Path(directory, "public_trace.jsonl").read_text(encoding="utf-8")
            private = Path(directory, "private_eval.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("plus_passed", public)
            self.assertIn("plus_passed", private)
            json.loads(public)
            json.loads(private)

    def test_no_submission_logging_has_no_fake_solution(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = RunLogger(directory)
            logger.log_episode(no_submission_episode())
            solution = json.loads(
                Path(directory, "solutions.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(solution["status"], "no_submission")
            self.assertIsNone(solution["solution"])


if __name__ == "__main__":
    unittest.main()
