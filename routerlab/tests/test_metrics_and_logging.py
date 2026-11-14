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


class MetricsAndLoggingTests(unittest.TestCase):
    def test_recovery_metric(self):
        metrics = aggregate_metrics([episode(False, True)])
        self.assertEqual(metrics["quality"]["repair_recovery_rate"], 1.0)
        self.assertEqual(metrics["quality"]["absolute_plus_improvement"], 1.0)

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


if __name__ == "__main__":
    unittest.main()
