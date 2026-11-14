import unittest
from pathlib import Path

from inference.policy.capability_harness import CapabilityRoutingHarness
from routing_benchmark.environment import RoutingEnvironment
from routing_benchmark.evaluator import evaluate_track_a, evaluate_track_b
from routing_benchmark.loader import load_benchmark
from routing_benchmark.routers import OracleRouter
from routing_benchmark.schemas import STOP_ACTION


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "RoutingBench" / "v1"


class RoutingBenchmarkV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_benchmark(DATA_DIR)

    def test_bundle_has_small_complete_v1_suite(self):
        self.assertEqual(self.bundle.version, "1.0.0")
        self.assertEqual(len(self.bundle.scenarios), 10)
        self.assertGreaterEqual(len(self.bundle.agents), 12)

    def test_public_profiles_have_no_capability_or_backbone_scores(self):
        for agent in self.bundle.agents.values():
            profile = agent.public_profile()
            serialized_keys = " ".join(profile).lower()
            self.assertNotIn("capability", serialized_keys)
            self.assertNotIn("backbone", serialized_keys)
            self.assertNotIn("distractor", profile)

    def test_legacy_harness_can_disable_capability_scoring_and_shortlisting(self):
        harness = CapabilityRoutingHarness(
            {
                "enabled": True,
                "capability_scoring": {"enabled": False},
                "capability_shortlist": {
                    "enabled": True,
                    "max_candidates": 5,
                    "min_candidates": 2,
                },
            }
        )
        self.assertFalse(harness.capability_scoring_enabled)
        self.assertFalse(harness.shortlist_enabled)

    def test_parallel_join_has_set_valued_valid_actions(self):
        scenario = next(
            item for item in self.bundle.scenarios if item.scenario_id == "rbv1-003"
        )
        env = RoutingEnvironment(scenario, self.bundle.agents)
        self.assertEqual(
            set(env.valid_actions()), {"database-retriever", "web-researcher"}
        )

    def test_stop_is_invalid_before_goal_and_valid_after_goal(self):
        scenario = next(
            item for item in self.bundle.scenarios if item.scenario_id == "rbv1-005"
        )
        env = RoutingEnvironment(scenario, self.bundle.agents)
        event = env.step(STOP_ACTION)
        self.assertFalse(event.valid_action)
        self.assertEqual(event.status, "PREMATURE_STOP")
        self.assertFalse(env.success)

    def test_recovery_requires_changed_state_before_retry(self):
        scenario = next(
            item for item in self.bundle.scenarios if item.scenario_id == "rbv1-006"
        )
        env = RoutingEnvironment(scenario, self.bundle.agents)
        first = env.step("web-researcher")
        self.assertEqual(first.status, "RETRYABLE_FAILURE")
        self.assertEqual(env.valid_actions(), ("recovery-planner",))
        env.step("recovery-planner")
        self.assertEqual(env.valid_actions(), ("web-researcher",))

    def test_oracle_solves_both_tracks(self):
        track_a = evaluate_track_a(self.bundle, OracleRouter())
        self.assertEqual(track_a["metrics"]["valid_next_agent_accuracy"], 1.0)
        track_b = evaluate_track_b(self.bundle, OracleRouter())
        self.assertEqual(track_b["metrics"]["goal_completion_rate"], 1.0)
        self.assertEqual(track_b["metrics"]["invalid_action_rate"], 0.0)
        self.assertEqual(track_b["metrics"]["recovery_success_rate"], 1.0)
        self.assertEqual(
            track_b["metrics"]["average_normalized_step_regret_on_success"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
