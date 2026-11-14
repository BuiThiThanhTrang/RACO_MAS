from __future__ import annotations

import unittest

from routerlab.config import ModelConfig
from routerlab.contracts import BudgetSnapshot, ROUTER_ACTIONS, RoutingState
from routerlab.routers import LayaDecisionRouter


class FakeLayaAgent:
    def __init__(self):
        self.calls = []

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        return {
            "answers": {
                "next_action": {
                    "choice": "edge_case_analyst",
                    "answer_confidence": 0.73,
                    "probabilities": {
                        "coder": 0.1,
                        "logic_reviewer": 0.1,
                        "edge_case_analyst": 0.73,
                        "repair_agent": 0.02,
                        "STOP": 0.05,
                    },
                }
            },
            "usage": {"input_tokens": 123, "output_tokens": 0},
        }


class LayaRouterTests(unittest.TestCase):
    def test_laya_maps_typed_choice_to_router_decision(self):
        agent = FakeLayaAgent()
        config = ModelConfig(
            backend="laya",
            checkpoint="convaiinnovations/laya",
            context_limit=512,
        )
        router = LayaDecisionRouter(config, agent=agent)
        state = RoutingState(
            task_id="Mbpp/1",
            task_prompt="Write f",
            entry_point="f",
            current_code=None,
            code_version=0,
            base_feedback=None,
            logic_review=(),
            edge_cases=(),
            action_history=(),
            budget=BudgetSnapshot(9, 14000, 20, 600000),
        )
        decision = router.select(state)
        self.assertEqual(decision.action_id, "edge_case_analyst")
        self.assertTrue(decision.valid)
        self.assertEqual(decision.usage.input_tokens, 123)
        self.assertEqual(decision.metadata["confidence"], 0.73)
        self.assertEqual(tuple(agent.calls[0][0]["action_space"]), ROUTER_ACTIONS)
        self.assertEqual(agent.calls[0][2]["max_len"], 512)


if __name__ == "__main__":
    unittest.main()
