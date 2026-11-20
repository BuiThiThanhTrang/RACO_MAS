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
    dtype: str = "auto"
    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None


@dataclass(frozen=True)
class DecisionModelConfig:
    """Configuration for a registered typed-decision model.

    Known model/provider pairs obtain their adapter, checkpoint, endpoint and API
    key environment variable from the registry. Every field after provider is an
    optional override for self-hosted or pinned deployments.
    """

    name: str
    provider: str
    checkpoint: str | None = None
    revision: str | None = None
    adapter: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None
    device: str = "auto"
    context_limit: int = 8192
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class BudgetConfig:
    max_steps: int = 7
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
class ProtocolConfig:
    version: str = "v5"
    feedback_policy: str = "counts_only_v1"
    framing_reserve_tokens: int = 256
    router_code_preview_chars: int = 4000
    recent_event_limit: int = 8


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    output_dir: str
    dataset: DatasetConfig
    expert_model: ModelConfig
    budget: BudgetConfig
    workflow: WorkflowConfig
    router_model: ModelConfig | DecisionModelConfig | None = None
    prompt_version: str = "v1"
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)


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
        "decision_model",
    }:
        raise ValueError(
            "E3 requires workflow.router=heuristic, model, laya or decision_model"
        )
    router_payload = data.get("router_model")
    if workflow.router == "decision_model" and router_payload:
        router_model = _construct(DecisionModelConfig, router_payload)
    else:
        router_model = (
            _construct(ModelConfig, router_payload) if router_payload else None
        )
    if workflow.router in {"model", "laya", "decision_model"} and router_model is None:
        raise ValueError(f"{workflow.router} router requires router_model configuration")
    protocol = _construct(ProtocolConfig, data.get("protocol"))
    if protocol.version not in {"v5", "v6"}:
        raise ValueError("protocol.version must be v5 or v6")
    if protocol.feedback_policy not in {
        "counts_only_v1",
        "sanitized_diagnostic_v2",
    }:
        raise ValueError(
            "protocol.feedback_policy must be counts_only_v1 or "
            "sanitized_diagnostic_v2"
        )
    if protocol.framing_reserve_tokens < 0:
        raise ValueError("protocol.framing_reserve_tokens must be non-negative")
    if protocol.router_code_preview_chars < 256:
        raise ValueError("protocol.router_code_preview_chars must be at least 256")
    if protocol.recent_event_limit < 1:
        raise ValueError("protocol.recent_event_limit must be positive")
    return RunConfig(
        run_name=str(data["run_name"]),
        output_dir=str(data.get("output_dir", "runs")),
        dataset=_construct(DatasetConfig, data.get("dataset")),
        expert_model=_construct(ModelConfig, data.get("expert_model")),
        router_model=router_model,
        budget=_construct(BudgetConfig, data.get("budget")),
        workflow=workflow,
        prompt_version=str(data.get("prompt_version", "v1")),
        protocol=protocol,
    )
