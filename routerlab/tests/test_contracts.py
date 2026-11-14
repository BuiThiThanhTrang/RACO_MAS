from __future__ import annotations

import unittest

from routerlab.contracts import (
    BudgetSnapshot,
    EXECUTOR_IDS,
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
            current_code="def f(): return 1",
            code_version=1,
            base_feedback=None,
            logic_review=(),
            edge_cases=(),
            action_history=(),
            budget=BudgetSnapshot(2, 100, 2, 1000),
            legal_actions=("coder", STOP_ACTION),
        )
        payload = state.to_router_payload()
        self.assertFalse(any("plus" in key.lower() for key in payload))


if __name__ == "__main__":
    unittest.main()
