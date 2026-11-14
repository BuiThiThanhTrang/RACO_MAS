from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


STOP_ACTION = "STOP"
EXECUTOR_IDS = (
    "coder",
    "logic_reviewer",
    "edge_case_analyst",
    "repair_agent",
)


def _non_negative(value: int | float, name: str) -> None:
    if value < 0 or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    prompt: str
    entry_point: str

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.prompt.strip() or not self.entry_point.strip():
            raise ValueError("task_id, prompt and entry_point must not be empty")


@dataclass(frozen=True)
class CodeArtifact:
    source: str
    version: int
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.version <= 0:
            raise ValueError("code version must be positive")
        digest = hashlib.sha256(self.source.encode("utf-8")).hexdigest()
        object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        _non_negative(self.input_tokens, "input_tokens")
        _non_negative(self.output_tokens, "output_tokens")
        _non_negative(self.cost_usd, "cost_usd")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ModelResponse:
    text: str
    usage: ModelUsage = field(default_factory=ModelUsage)
    latency_ms: float = 0.0
    model_id: str = ""

    def __post_init__(self) -> None:
        _non_negative(self.latency_ms, "latency_ms")


@dataclass(frozen=True)
class BaseTestFeedback:
    code_version: int
    status: str
    passed: bool
    passed_tests: int | None = None
    total_tests: int | None = None
    failure_count: int | None = None
    summary: str = ""

    def __post_init__(self) -> None:
        if self.code_version <= 0:
            raise ValueError("code_version must be positive")
        if not self.status.strip():
            raise ValueError("status must not be empty")
        for name in ("passed_tests", "total_tests", "failure_count"):
            value = getattr(self, name)
            if value is not None:
                _non_negative(value, name)


@dataclass(frozen=True)
class EvaluationSnapshot:
    task_id: str
    code_version: int
    base_passed: bool
    plus_passed: bool | None
    base_passed_tests: int | None = None
    base_total_tests: int | None = None
    plus_passed_tests: int | None = None
    plus_total_tests: int | None = None


@dataclass(frozen=True)
class BudgetSnapshot:
    remaining_model_calls: int
    remaining_tokens: int
    remaining_base_executions: int
    remaining_wall_time_ms: float

    def __post_init__(self) -> None:
        for name in (
            "remaining_model_calls",
            "remaining_tokens",
            "remaining_base_executions",
            "remaining_wall_time_ms",
        ):
            _non_negative(getattr(self, name), name)


@dataclass(frozen=True)
class RoutingState:
    """Public state visible to a router. Plus-test data is intentionally absent."""

    task_id: str
    task_prompt: str
    entry_point: str
    current_code: str
    code_version: int
    base_feedback: BaseTestFeedback | None
    logic_review: tuple[str, ...]
    edge_cases: tuple[str, ...]
    action_history: tuple[str, ...]
    budget: BudgetSnapshot
    legal_actions: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.legal_actions)) != len(self.legal_actions):
            raise ValueError("legal_actions must be unique")
        unknown = set(self.legal_actions) - (set(EXECUTOR_IDS) | {STOP_ACTION})
        if unknown:
            raise ValueError(f"Unknown legal actions: {sorted(unknown)}")
        if not self.legal_actions:
            raise ValueError("legal_actions must not be empty")

    def to_router_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_prompt": self.task_prompt,
            "entry_point": self.entry_point,
            "current_code": self.current_code,
            "code_version": self.code_version,
            "base_feedback": asdict(self.base_feedback) if self.base_feedback else None,
            "logic_review": list(self.logic_review),
            "edge_cases": list(self.edge_cases),
            "action_history": list(self.action_history),
            "budget": asdict(self.budget),
            "legal_actions": list(self.legal_actions),
        }


@dataclass(frozen=True)
class RouterDecision:
    action_id: str
    usage: ModelUsage = field(default_factory=ModelUsage)
    latency_ms: float = 0.0
    raw_output: str = ""
    valid: bool = True

    def __post_init__(self) -> None:
        if self.action_id not in set(EXECUTOR_IDS) | {STOP_ACTION}:
            raise ValueError(f"Unknown router action: {self.action_id!r}")
        _non_negative(self.latency_ms, "latency_ms")


@dataclass(frozen=True)
class ExecutorResult:
    executor_id: str
    raw_output: str
    usage: ModelUsage
    latency_ms: float
    code: str | None = None
    findings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.executor_id not in set(EXECUTOR_IDS) | {"single_agent"}:
            raise ValueError(f"Unknown executor: {self.executor_id!r}")
        _non_negative(self.latency_ms, "latency_ms")


@dataclass(frozen=True)
class StepRecord:
    step_index: int
    action_id: str
    code_version_before: int
    code_version_after: int
    model_usage: ModelUsage = field(default_factory=ModelUsage)
    latency_ms: float = 0.0
    base_test_ran: bool = False
    router_decision: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeResult:
    task_id: str
    method: str
    initial_evaluation: EvaluationSnapshot | None
    final_evaluation: EvaluationSnapshot | None
    final_code: CodeArtifact
    steps: tuple[StepRecord, ...]
    termination_reason: str
    total_model_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_base_executions: int
    total_cost_usd: float
    wall_time_ms: float

    @property
    def initial_success(self) -> bool:
        return bool(self.initial_evaluation and self.initial_evaluation.plus_passed)

    @property
    def final_success(self) -> bool:
        return bool(self.final_evaluation and self.final_evaluation.plus_passed)
