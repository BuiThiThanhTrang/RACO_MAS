import json
import unittest
from collections import Counter
from pathlib import Path

from routing_benchmark.routers import DecisionModelRouter, OracleRouter
from routing_benchmark.v2.environment import RoutingEnvironmentV2
from routing_benchmark.v2.evaluator import evaluate_track_a_v2, evaluate_track_b_v2
from routing_benchmark.v2.generator import generate_decision_rows
from routing_benchmark.v2.loader import load_benchmark_v2
from routing_benchmark.v2.validation_beta import validate_beta


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "RoutingBench" / "v2_beta"


class RoutingBenchmarkV2BetaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_benchmark_v2(DATA_DIR)
        cls.rows = generate_decision_rows(cls.bundle)

    def test_release_shape_and_grouped_splits(self):
        self.assertEqual(len(self.bundle.scenarios), 120)
        family_counts = Counter(item.family for item in self.bundle.scenarios)
        self.assertEqual(len(family_counts), 8)
        self.assertEqual(set(family_counts.values()), {15})
        self.assertEqual(
            Counter(item.split for item in self.bundle.scenarios),
            {"development": 24, "validation": 32, "test": 64},
        )
        groups = {}
        for item in self.bundle.scenarios:
            groups.setdefault(item.split_group_id, set()).add(item.split)
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))

    def test_beta_validator_and_observability_consistency(self):
        report = validate_beta(self.bundle)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["observability_collision_count"], 0)

    def test_public_input_uses_compact_memory_without_oracle_labels(self):
        forbidden = (
            "capability_score",
            "backbone",
            "valid_actions",
            "admissible_actions",
            "progress_actions",
            "oracle_efficient_actions",
            "source_scope",
            "full_history",
        )
        for scenario in self.bundle.scenarios[:16]:
            observation = RoutingEnvironmentV2(
                scenario, self.bundle.agents
            ).observation()
            serialized = json.dumps(observation, sort_keys=True).casefold()
            for term in forbidden:
                self.assertNotIn(term, serialized)
            self.assertIn("execution_memory", observation)
            self.assertIn("recent_events", observation)

    def test_unknown_source_accepts_both_attempts_then_uses_feedback(self):
        scenario = next(
            item
            for item in self.bundle.scenarios
            if item.scenario_id == "v2b-sequential-dependency-13"
        )
        env = RoutingEnvironmentV2(scenario, self.bundle.agents)
        env.step("agent_01")
        env.step("agent_02")
        env.step("agent_05")
        self.assertEqual(set(env.admissible_actions()), {"agent_02", "agent_03"})
        event = env.step("agent_03")
        self.assertTrue(event.admissible)
        self.assertEqual(event.status, "EXECUTED_NO_PROGRESS")
        self.assertEqual(env.admissible_actions(), ("agent_02",))

    def test_routing_rejection_is_not_actor_failure(self):
        scenario = next(
            item
            for item in self.bundle.scenarios
            if item.scenario_id == "v2b-sequential-dependency-01"
        )
        env = RoutingEnvironmentV2(scenario, self.bundle.agents)
        event = env.step("agent_15")
        self.assertEqual(event.status, "ACTION_REJECTED")
        self.assertEqual(env.state.routing_rejection_count, 1)
        self.assertEqual(env.state.actor_failure_count, 0)
        observation = env.observation()
        self.assertEqual(
            observation["execution_memory"]["rejected_actions"][0]["agent_id"],
            "agent_15",
        )

    def test_semantic_loop_ignores_counter_only_changes(self):
        scenario = next(
            item
            for item in self.bundle.scenarios
            if item.scenario_id == "v2b-sequential-dependency-01"
        )
        env = RoutingEnvironmentV2(scenario, self.bundle.agents)
        first = env.step("agent_15")
        second = env.step("agent_15")
        third = env.step("agent_15")
        self.assertFalse(first.stagnant_repeat)
        self.assertTrue(second.stagnant_repeat)
        self.assertTrue(third.stagnant_repeat)
        self.assertTrue(env.done)
        self.assertEqual(env.terminal_reason, "LOOP_LIMIT")

    def test_structured_rejection_guard_uses_only_public_recent_events(self):
        observation = {
            "recent_events": [
                {
                    "agent_id": "agent_06",
                    "status": "ACTION_REJECTED",
                    "semantic_delta": "none",
                }
            ]
        }
        selected, blocked = DecisionModelRouter._observable_rejection_guard(
            observation, ("agent_04", "agent_06", "STOP")
        )
        self.assertEqual(selected, ("agent_04", "STOP"))
        self.assertEqual(blocked, ("agent_06",))

        changed_observation = {
            "recent_events": [
                observation["recent_events"][0],
                {
                    "agent_id": "agent_12",
                    "status": "EXECUTED_SUCCESS",
                    "semantic_delta": "state_changed",
                },
            ]
        }
        selected, blocked = DecisionModelRouter._observable_rejection_guard(
            changed_observation, ("agent_04", "agent_06", "STOP")
        )
        self.assertEqual(selected, ("agent_04", "agent_06", "STOP"))
        self.assertEqual(blocked, ())

    def test_naive_and_structured_router_profiles_are_distinct(self):
        observation = {
            "agent_profiles": [
                {
                    "agent_id": "agent_02",
                    "role_name": "Corpus Evidence Analyst",
                    "role_goal": "Resolve evidence from supplied text.",
                    "use_when": ["a corpus dependency is explicit"],
                    "avoid_when": ["the unresolved source is live web"],
                    "supported_inputs": ["text corpus"],
                    "expected_outputs": ["evidence packet"],
                }
            ]
        }
        naive = DecisionModelRouter._criteria(
            observation, ("agent_02", "STOP"), "naive"
        )
        structured = DecisionModelRouter._criteria(
            observation, ("agent_02", "STOP"), "structured"
        )
        self.assertNotIn("Use when", naive["agent_02"])
        self.assertIn("Use when", structured["agent_02"])
        self.assertNotIn(
            "ACTION_REJECTED", DecisionModelRouter._instructions("naive", True)
        )
        self.assertIn(
            "pre-execution",
            DecisionModelRouter._instructions("structured", True),
        )

    def test_beta_quota_shape(self):
        total = len(self.rows)
        singleton = sum(
            not row["stop_ready"] and len(row["admissible_actions"]) == 1
            for row in self.rows
        )
        multi = sum(
            not row["stop_ready"] and len(row["admissible_actions"]) > 1
            for row in self.rows
        )
        stop = sum(row["stop_ready"] for row in self.rows)
        self.assertGreaterEqual(total, 750)
        self.assertLessEqual(total, 1100)
        self.assertGreaterEqual(singleton / total, 0.35)
        self.assertLessEqual(singleton / total, 0.45)
        self.assertGreaterEqual(multi / total, 0.35)
        self.assertGreaterEqual(stop / total, 0.10)
        self.assertLessEqual(stop / total, 0.15)

    def test_oracle_completes_beta(self):
        track_a = evaluate_track_a_v2(self.bundle, OracleRouter())
        self.assertEqual(
            track_a["metrics"]["admissible_next_action_accuracy"], 1.0
        )
        self.assertEqual(
            track_a["metrics"]["macro_template_admissible_accuracy"], 1.0
        )
        self.assertTrue(track_a["metrics"]["by_template"])
        track_b = evaluate_track_b_v2(self.bundle, OracleRouter())
        self.assertEqual(track_b["metrics"]["goal_completion_rate"], 1.0)
        self.assertEqual(track_b["metrics"]["admissible_action_rate"], 1.0)
        self.assertEqual(track_b["metrics"]["loop_termination_rate"], 0.0)
        self.assertEqual(
            track_b["metrics"]["macro_template_completion_rate"], 1.0
        )
        self.assertTrue(track_b["metrics"]["by_template"])


if __name__ == "__main__":
    unittest.main()
