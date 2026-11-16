from __future__ import annotations

import os
import unittest
from unittest.mock import patch

def check(code: str, expected: int) -> tuple[str, list[bool]]:
    from routerlab.windows_evalplus import windows_untrusted_check

    status, details = windows_untrusted_check(
        dataset="mbpp",
        code=code,
        inputs=[(1,)],
        entry_point="candidate",
        expected=[expected],
        atol=0,
        ref_time=[0.001],
        min_time_limit=0.1,
        gt_time_limit_factor=1.0,
    )
    return status, details.tolist()


@unittest.skipUnless(os.name == "nt", "Windows compatibility test")
class WindowsEvalPlusTests(unittest.TestCase):
    def test_windows_checker_passes_correct_code(self) -> None:
        self.assertEqual(
            check("def candidate(x):\n    return x + 1", 2), ("pass", [True])
        )

    def test_windows_checker_rejects_wrong_code(self) -> None:
        self.assertEqual(
            check("def candidate(x):\n    return x", 2), ("fail", [False])
        )

    def test_windows_checker_interrupts_infinite_loop(self) -> None:
        status, details = check("def candidate(x):\n    while True:\n        pass", 2)
        self.assertIn(status, {"fail", "timeout"})
        self.assertIn(details, ([], [False]))

    def test_evalplus_evaluator_uses_windows_backend_end_to_end(self) -> None:
        from routerlab.contracts import CodeArtifact, TaskSpec
        from routerlab.evaluator import EvalPlusEvaluator

        problem = {
            "task_id": "Mbpp/test",
            "prompt": "def candidate(x):\n",
            "entry_point": "candidate",
            "base_input": [(1,)],
            "plus_input": [(2,)],
            "atol": 0,
        }
        expected = {
            "Mbpp/test": {
                "base": [2],
                "base_time": [0.001],
                "plus": [3],
                "plus_time": [0.001],
            }
        }
        with patch("evalplus.evaluate.get_groundtruth", return_value=expected):
            evaluator = EvalPlusEvaluator({"Mbpp/test": problem}, "test-hash")

        feedback, snapshot = evaluator.evaluate(
            TaskSpec("Mbpp/test", "Return x plus one", "candidate"),
            CodeArtifact("    return x + 1", version=1),
            include_plus=True,
        )
        self.assertEqual(evaluator.backend, "windows-process")
        self.assertTrue(feedback.passed)
        self.assertTrue(snapshot.plus_passed)


if __name__ == "__main__":
    unittest.main()
