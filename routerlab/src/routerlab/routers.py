from __future__ import annotations

import json

from .contracts import RouterDecision, RoutingState, STOP_ACTION
from .model import ModelClient
from .parsing import parse_router_action
from .prompts import load_prompt


class HeuristicRouter:
    """A deterministic, non-learned baseline over the same action interface."""

    def select(self, state: RoutingState) -> RouterDecision:
        legal = state.legal_actions
        history = state.action_history
        if state.base_feedback and state.base_feedback.passed:
            action = STOP_ACTION
        elif not state.current_code and "coder" in legal:
            action = "coder"
        elif "logic_reviewer" in legal and "logic_reviewer" not in history[-3:]:
            action = "logic_reviewer"
        elif "edge_case_analyst" in legal and "edge_case_analyst" not in history[-3:]:
            action = "edge_case_analyst"
        elif "repair_agent" in legal:
            action = "repair_agent"
        else:
            action = STOP_ACTION
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
        action, valid = parse_router_action(response.text, state.legal_actions)
        return RouterDecision(
            action_id=action,
            usage=response.usage,
            latency_ms=response.latency_ms,
            raw_output=response.text,
            valid=valid,
        )
