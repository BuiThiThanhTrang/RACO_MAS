from __future__ import annotations

from dataclasses import dataclass

from .contracts import EXECUTOR_IDS


@dataclass(frozen=True)
class ExecutorSpec:
    executor_id: str
    role_name: str
    role_goal: str
    output_kind: str


def build_executor_pool() -> tuple[ExecutorSpec, ...]:
    pool = (
        ExecutorSpec(
            "coder",
            "Coder",
            "Generate a complete Python solution that follows the required signature.",
            "code",
        ),
        ExecutorSpec(
            "logic_reviewer",
            "Logic Reviewer",
            "Find algorithmic, logical, contract, and complexity problems.",
            "findings",
        ),
        ExecutorSpec(
            "edge_case_analyst",
            "Edge-case Analyst",
            "Identify supported boundary and structural cases without inventing requirements.",
            "findings",
        ),
        ExecutorSpec(
            "repair_agent",
            "Repair Agent",
            "Repair the current code using public feedback while preserving correct behavior.",
            "code",
        ),
    )
    if tuple(item.executor_id for item in pool) != EXECUTOR_IDS:
        raise RuntimeError("executor pool does not match the routing contract")
    return pool


def executor_map() -> dict[str, ExecutorSpec]:
    return {item.executor_id: item for item in build_executor_pool()}
