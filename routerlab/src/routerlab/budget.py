from __future__ import annotations

import time

from .config import BudgetConfig
from .contracts import BudgetSnapshot, ModelUsage


class BudgetTracker:
    def __init__(self, config: BudgetConfig) -> None:
        self.config = config
        self.started = time.perf_counter()
        self.model_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.base_executions = 0
        self.cost_usd = 0.0

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def can_call_model(self) -> bool:
        return (
            self.model_calls < self.config.max_model_calls
            and self.total_tokens < self.config.max_total_tokens
            and self.elapsed_ms < self.config.max_wall_time_seconds * 1000.0
        )

    def can_run_base(self, *, reserve_final: bool = False) -> bool:
        reserve = 1 if reserve_final else 0
        return (
            self.base_executions + reserve < self.config.max_base_executions
            and self.elapsed_ms < self.config.max_wall_time_seconds * 1000.0
        )

    def record_model(self, usage: ModelUsage) -> None:
        self.model_calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cost_usd += usage.cost_usd

    def record_base(self) -> None:
        self.base_executions += 1

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            remaining_model_calls=max(0, self.config.max_model_calls - self.model_calls),
            remaining_tokens=max(0, self.config.max_total_tokens - self.total_tokens),
            remaining_base_executions=max(
                0, self.config.max_base_executions - self.base_executions
            ),
            remaining_wall_time_ms=max(
                0.0,
                self.config.max_wall_time_seconds * 1000.0 - self.elapsed_ms,
            ),
        )
