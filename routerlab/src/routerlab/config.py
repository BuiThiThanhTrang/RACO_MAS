from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    backend: str
    checkpoint: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 1024
    context_limit: int = 8192
    device: str = "auto"
    base_url: str | None = None
    api_key_env: str | None = None


@dataclass(frozen=True)
class BudgetConfig:
    max_model_calls: int = 8
    max_total_tokens: int = 12000
    max_base_executions: int = 5
    max_wall_time_seconds: float = 600.0


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "mbpp"
    version: str = "default"
    manifest: str | None = None
    limit: int | None = 50
    seed: int = 42


@dataclass(frozen=True)
class WorkflowConfig:
    method: str
    fixed_rounds: int = 2
    router: str | None = None


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    output_dir: str
    dataset: DatasetConfig
    expert_model: ModelConfig
    budget: BudgetConfig
    workflow: WorkflowConfig
    router_model: ModelConfig | None = None
    prompt_version: str = "v1"


def _construct(cls, payload: dict[str, Any] | None):
    return cls(**(payload or {}))


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    workflow = _construct(WorkflowConfig, data.get("workflow"))
    if workflow.method not in {"e1", "e2", "e3"}:
        raise ValueError("workflow.method must be e1, e2 or e3")
    if workflow.method == "e3" and workflow.router not in {
        "heuristic",
        "model",
        "laya",
    }:
        raise ValueError("E3 requires workflow.router=heuristic, model or laya")
    router_model = (
        _construct(ModelConfig, data.get("router_model"))
        if data.get("router_model")
        else None
    )
    if workflow.router in {"model", "laya"} and router_model is None:
        raise ValueError(f"{workflow.router} router requires router_model configuration")
    return RunConfig(
        run_name=str(data["run_name"]),
        output_dir=str(data.get("output_dir", "runs")),
        dataset=_construct(DatasetConfig, data.get("dataset")),
        expert_model=_construct(ModelConfig, data.get("expert_model")),
        router_model=router_model,
        budget=_construct(BudgetConfig, data.get("budget")),
        workflow=workflow,
        prompt_version=str(data.get("prompt_version", "v1")),
    )
