from __future__ import annotations

import json
import time

from .capabilities import router_capability_payload
from .config import DecisionModelConfig, ModelConfig
from .contracts import (
    RouterDecision,
    RoutingState,
    STOP_ACTION,
)
from .decision_models import DecisionAdapter, build_decision_adapter
from .model import ModelClient
from .parsing import parse_router_action
from .prompts import load_prompt


class HeuristicRouter:
    """A deterministic, non-learned baseline over the same action interface."""

    def select(self, state: RoutingState) -> RouterDecision:
        history = state.action_history
        if state.base_feedback and state.base_feedback.passed:
            action = STOP_ACTION
        elif not state.current_code:
            action = "coder"
        elif "logic_reviewer" not in history[-3:]:
            action = "logic_reviewer"
        elif "edge_case_analyst" not in history[-3:]:
            action = "edge_case_analyst"
        else:
            action = "repair_agent"
        return RouterDecision(action_id=action)


class ModelRouter:
    def __init__(self, client: ModelClient) -> None:
        self.client = client

    def select(self, state: RoutingState) -> RouterDecision:
        payload = state.to_router_payload()
        payload["action_capabilities"] = router_capability_payload()
        response = self.client.generate(
            [
                {"role": "system", "content": load_prompt("router")},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ]
        )
        action, valid = parse_router_action(response.text, state.action_space)
        return RouterDecision(
            action_id=action,
            usage=response.usage,
            latency_ms=response.latency_ms,
            raw_output=response.text,
            valid=valid,
        )


class DecisionModelRouter:
    """Provider-independent router over the canonical typed-choice contract."""

    def __init__(
        self,
        config: DecisionModelConfig,
        *,
        adapter: DecisionAdapter | None = None,
        **adapter_kwargs,
    ) -> None:
        self.config = config
        if adapter is None:
            self.model, adapter = build_decision_adapter(config, **adapter_kwargs)
        else:
            from .decision_models import resolve_decision_model

            self.model = resolve_decision_model(config)
        self.adapter = adapter

    def select(self, state: RoutingState) -> RouterDecision:
        started = time.perf_counter()
        result = self.adapter.decide(state.to_router_payload())
        latency_ms = (time.perf_counter() - started) * 1000.0
        action = (result.choice or "").strip()
        valid = action in state.action_space and not result.metadata.get("refusal", False)
        if not valid:
            action = STOP_ACTION
        return RouterDecision(
            action_id=action,
            usage=result.usage,
            latency_ms=latency_ms,
            raw_output=json.dumps(result.raw, ensure_ascii=False, default=str),
            valid=valid,
            metadata={
                "router_type": "decision_model",
                "decision_model": self.model.name,
                "provider": self.model.provider,
                "adapter": self.model.adapter,
                "checkpoint": self.model.checkpoint,
                "revision": self.model.revision,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
                **dict(result.metadata),
            },
        )


class LayaDecisionRouter(DecisionModelRouter):
    """Backward-compatible wrapper for legacy workflow.router=laya configs."""

    def __init__(self, config: ModelConfig, agent=None) -> None:
        decision_config = DecisionModelConfig(
            name="laya",
            provider="local",
            checkpoint=config.checkpoint,
            revision=None,
            device=config.device,
            context_limit=config.context_limit,
        )
        super().__init__(decision_config, agent=agent)
