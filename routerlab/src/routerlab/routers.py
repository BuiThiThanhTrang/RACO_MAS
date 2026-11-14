from __future__ import annotations

import json
import time

from .config import ModelConfig
from .contracts import (
    ModelUsage,
    ROUTER_ACTIONS,
    RouterDecision,
    RoutingState,
    STOP_ACTION,
)
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
        response = self.client.generate(
            [
                {"role": "system", "content": load_prompt("router")},
                {
                    "role": "user",
                    "content": json.dumps(state.to_router_payload(), ensure_ascii=False),
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


class LayaDecisionRouter:
    """Adapter for the open-weight Laya typed decision model."""

    QUESTIONS = {
        "next_action": {
            "type": "choice",
            "instructions": (
                "Which action should the multi-agent code workflow execute next to "
                "maximize the chance of solving the task within the remaining budget?"
            ),
            "criteria": {
                "coder": (
                    "Generate a complete implementation, or replace the current code "
                    "with a new implementation."
                ),
                "logic_reviewer": (
                    "Audit correctness obligations and logical defects without writing code."
                ),
                "edge_case_analyst": (
                    "Analyze boundary cases and the input space without writing code."
                ),
                "repair_agent": (
                    "Repair existing code using feedback; this is a no-op when no code exists."
                ),
                STOP_ACTION: (
                    "End the episode and submit current code; with no code, end with failure."
                ),
            },
        }
    }

    def __init__(self, config: ModelConfig, agent=None) -> None:
        self.config = config
        if agent is None:
            import laya

            device = None if config.device == "auto" else config.device
            agent = laya.load(config.checkpoint, device=device)
        self.agent = agent

    def select(self, state: RoutingState) -> RouterDecision:
        started = time.perf_counter()
        result = self.agent.predict(
            state.to_router_payload(),
            self.QUESTIONS,
            max_len=self.config.context_limit,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        answer = result.get("answers", {}).get("next_action", {})
        action = str(answer.get("choice", "")).strip()
        valid = action in ROUTER_ACTIONS
        if not valid:
            action = STOP_ACTION
        usage = result.get("usage", {}) or {}
        return RouterDecision(
            action_id=action,
            usage=ModelUsage(
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
            ),
            latency_ms=latency_ms,
            raw_output=json.dumps(result, ensure_ascii=False, default=str),
            valid=valid,
            metadata={
                "router_type": "laya",
                "checkpoint": self.config.checkpoint,
                "confidence": answer.get(
                    "answer_confidence", answer.get("confidence")
                ),
                "probabilities": answer.get("probabilities"),
                "input_truncated": usage.get("truncated"),
                "state_tokens": usage.get("state_tokens"),
                "state_tokens_dropped": usage.get("state_tokens_dropped"),
            },
        )
