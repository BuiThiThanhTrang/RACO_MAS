from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .contracts import (
    BaseTestFeedback,
    CodeArtifact,
    EvaluationSnapshot,
    TaskSpec,
)


class CodeEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self, task: TaskSpec, code: CodeArtifact, *, include_plus: bool
    ) -> tuple[BaseTestFeedback, EvaluationSnapshot]:
        raise NotImplementedError


class EvalPlusEvaluator(CodeEvaluator):
    """Runs the official EvalPlus checker while keeping test data private."""

    def __init__(
        self,
        problems: dict[str, dict[str, Any]],
        dataset_hash: str,
        *,
        fast_check: bool = False,
    ) -> None:
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
        from evalplus.evaluate import get_groundtruth

        self.problems = problems
        self.expected = get_groundtruth(
            problems, dataset_hash, MBPP_OUTPUT_NOT_NONE_TASKS
        )
        self.fast_check = fast_check

    def evaluate(
        self, task: TaskSpec, code: CodeArtifact, *, include_plus: bool
    ) -> tuple[BaseTestFeedback, EvaluationSnapshot]:
        from evalplus.eval import PASS
        from evalplus.evaluate import check_correctness

        problem = self.problems[task.task_id]
        # EvalPlus defines generated code as a completion appended to its prompt.
        solution = str(problem["prompt"]) + "\n" + code.source
        result = check_correctness(
            dataset="mbpp",
            completion_id=0,
            problem=problem,
            solution=solution,
            expected_output=self.expected[task.task_id],
            base_only=not include_plus,
            fast_check=self.fast_check,
            identifier=f"{task.task_id}:v{code.version}",
        )
        base_status, base_details = result["base"]
        base_details = list(base_details)
        base_total = len(problem["base_input"])
        base_passed_count = sum(bool(value) for value in base_details)
        # fast_check may stop at the first failure, therefore omitted tests are failures.
        base_feedback = BaseTestFeedback(
            code_version=code.version,
            status=str(base_status),
            passed=base_status == PASS,
            passed_tests=base_passed_count,
            total_tests=base_total,
            failure_count=base_total - base_passed_count,
            summary=f"Base Tests: {base_status}; {base_passed_count}/{base_total} passed",
        )
        plus_passed = None
        plus_passed_count = None
        plus_total = None
        if include_plus:
            plus_status, plus_details = result["plus"]
            plus_details = list(plus_details)
            plus_total = len(problem["plus_input"])
            plus_passed_count = sum(bool(value) for value in plus_details)
            plus_passed = plus_status == PASS
        snapshot = EvaluationSnapshot(
            task_id=task.task_id,
            code_version=code.version,
            base_passed=base_feedback.passed,
            plus_passed=plus_passed,
            base_passed_tests=base_passed_count,
            base_total_tests=base_total,
            plus_passed_tests=plus_passed_count,
            plus_total_tests=plus_total,
        )
        return base_feedback, snapshot
