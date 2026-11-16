from __future__ import annotations

import unittest

from routerlab.contracts import (
    BudgetSnapshot,
    EXECUTOR_IDS,
    ROUTER_ACTIONS,
    RoutingState,
    STOP_ACTION,
)
from routerlab.pool import build_executor_pool


class ContractTests(unittest.TestCase):
    def test_pool_and_router_actions_are_exactly_expected(self):
        self.assertEqual(
            tuple(item.executor_id for item in build_executor_pool()), EXECUTOR_IDS
        )
        self.assertNotIn("RUN_BASE_TESTS", EXECUTOR_IDS)
        self.assertEqual(STOP_ACTION, "STOP")

    def test_router_payload_has_no_plus_test_fields(self):
        state = RoutingState(
            task_id="Mbpp/1",
            task_prompt="prompt",
            entry_point="f",
            current_code=None,
            code_version=0,
            base_feedback=None,
            logic_review=(),
            edge_cases=(),
            action_history=(),
            budget=BudgetSnapshot(2, 2, 1000),
        )
        payload = state.to_router_payload()
        self.assertFalse(any("plus" in key.lower() for key in payload))
        self.assertEqual(tuple(payload["action_space"]), ROUTER_ACTIONS)
        self.assertEqual(payload["budget"]["remaining_steps"], 2)
        self.assertEqual(payload["budget"]["non_stop_action_cost"], 1)
        self.assertEqual(payload["budget"]["stop_cost"], 0)


if __name__ == "__main__":
    unittest.main()
