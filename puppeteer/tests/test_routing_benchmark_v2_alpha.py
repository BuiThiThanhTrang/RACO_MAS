import json
import unittest
from collections import Counter
from pathlib import Path

from routing_benchmark.routers import OracleRouter
from routing_benchmark.v2.environment import RoutingEnvironmentV2
from routing_benchmark.v2.evaluator import evaluate_track_a_v2, evaluate_track_b_v2
from routing_benchmark.v2.generator import (
    build_alpha_bundle,
    generate_decision_rows,
)
from routing_benchmark.v2.loader import load_benchmark_v2
from routing_benchmark.v2.validation import (
    validate_alpha,
    validate_deterministic_generation,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "RoutingBench" / "v2_alpha"


class RoutingBenchmarkV2AlphaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_benchmark_v2(DATA_DIR)
        cls.rows = generate_decision_rows(cls.bundle)

    def test_alpha_has_eight_families_and_two_instances_each(self):
        counts = Counter(item.family for item in self.bundle.scenarios)
        self.assertEqual(len(self.bundle.scenarios), 16)
        self.assertEqual(len(counts), 8)
        self.assertEqual(set(counts.values()), {2})

    def test_alpha_validator_and_generation_are_deterministic(self):
        report = validate_alpha(self.bundle)
        self.assertTrue(report["valid"], report["errors"])
        self.assertTrue(validate_deterministic_generation(42))

    def test_rows_cover_v2_alpha_state_types(self):
        self.assertTrue(any(len(row["valid_actions"]) > 1 for row in self.rows))
        self.assertTrue(
            any(
                len(row["valid_actions"]) > len(row["efficient_actions"])
                for row in self.rows
            )
        )
        self.assertTrue(
            any(row["state_provenance"] == "post_failure" for row in self.rows)
        )
        self.assertTrue(
            any(row["state_provenance"] == "post_invalid" for row in self.rows)
        )
        self.assertTrue(any(row["counterfactual_group_id"] for row in self.rows))

    def test_public_router_input_has_no_model_quality_fields(self):
        forbidden = ("backbone", "capability_score", "empirical_accuracy", "uncertainty")
        for scenario in self.bundle.scenarios:
            public = json.dumps(
                RoutingEnvironmentV2(scenario, self.bundle.agents).observation(),
                sort_keys=True,
            ).casefold()
            for term in forbidden:
                self.assertNotIn(term, public)

    def test_budget_counterfactual_changes_valid_not_applicable_actions(self):
        high, low = [
            item
            for item in self.bundle.scenarios
            if item.family == "budget_sensitive"
        ]
        high_env = RoutingEnvironmentV2(high, self.bundle.agents)
        low_env = RoutingEnvironmentV2(low, self.bundle.agents)
        self.assertEqual(
            set(high_env.applicable_actions()), {"agent_07", "agent_08"}
        )
        self.assertEqual(set(high_env.valid_actions()), {"agent_07", "agent_08"})
        self.assertEqual(
            set(low_env.applicable_actions()), {"agent_07", "agent_08"}
        )
        self.assertEqual(low_env.valid_actions(), ("agent_08",))

    def test_failure_recovery_changes_required_role(self):
        scenario = next(
            item
            for item in self.bundle.scenarios
            if item.scenario_id == "v2a-failure-recovery-01"
        )
        env = RoutingEnvironmentV2(scenario, self.bundle.agents)
        self.assertEqual(env.valid_actions(), ("agent_03",))
        event = env.step("agent_03")
        self.assertTrue(event.valid)
        self.assertEqual(event.status, "RETRYABLE_FAILURE")
        self.assertEqual(env.valid_actions(), ("agent_12",))

    def test_oracle_is_perfect_and_has_zero_regret(self):
        track_a = evaluate_track_a_v2(self.bundle, OracleRouter())
        self.assertEqual(track_a["metrics"]["valid_next_action_accuracy"], 1.0)
        self.assertEqual(track_a["metrics"]["efficient_action_rate"], 1.0)
        track_b = evaluate_track_b_v2(self.bundle, OracleRouter())
        self.assertEqual(track_b["metrics"]["goal_completion_rate"], 1.0)
        self.assertEqual(track_b["metrics"]["valid_action_rate"], 1.0)
        self.assertEqual(track_b["metrics"]["efficient_action_rate"], 1.0)
        self.assertEqual(track_b["metrics"]["no_progress_action_rate"], 0.0)
        self.assertEqual(track_b["metrics"]["average_step_regret_on_success"], 0.0)
        self.assertEqual(track_b["metrics"]["average_cost_regret_on_success"], 0.0)


if __name__ == "__main__":
    unittest.main()
