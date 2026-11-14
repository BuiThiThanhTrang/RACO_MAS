import json
import unittest
from pathlib import Path

from routing_benchmark.routers import OracleRouter
from routing_benchmark.v3.evaluator import (
    LiveActorResult,
    ModelLiveActorExecutor,
    _canonical_output_id,
    evaluate_track_a_v3,
    evaluate_track_b_live_v3,
)
from routing_benchmark.v3.loader import BenchmarkBundleV3, load_benchmark_v3
from routing_benchmark.v3.validation import FORBIDDEN_PUBLIC_TERMS, validate_v3


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "RoutingBench" / "v3_alpha"


class FakeLiveActor:
    model = "fake-live-actor"

    def __init__(self):
        self.calls = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        self.observations = []

    def execute(self, *, task, observation, role_profile):
        del task
        self.observations.append(json.loads(json.dumps(observation)))
        self.calls += 1
        self.total_tokens += 7
        self.total_latency_ms += 1.0
        expected = tuple(role_profile.get("expected_outputs") or ("workflow output",))
        return LiveActorResult(
            status="SUCCESS",
            contribution=(
                "PRIVATE_RAW_OUTPUT: a substantive live actor contribution "
                "satisfying the visible role contract."
            ),
            produced_outputs=(expected[0],),
            tokens=7,
            latency_ms=1.0,
            model=self.model,
        )


class RecordingOracleRouter(OracleRouter):
    def __init__(self):
        self.observations = []

    def select(self, observation, candidate_actions, oracle_valid_actions=None):
        self.observations.append(json.loads(json.dumps(observation)))
        return super().select(observation, candidate_actions, oracle_valid_actions)


class RoutingBenchmarkV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_benchmark_v3(DATA_DIR)

    def test_release_shape_and_quota(self):
        report = validate_v3(self.bundle)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(len(self.bundle.scenarios), 64)
        self.assertEqual(len(self.bundle.decision_rows), 584)
        self.assertEqual(len({item.family for item in self.bundle.scenarios}), 8)
        quota = report["quota_report"]
        self.assertLessEqual(quota["singleton_valid_rate"], 0.40)
        self.assertGreaterEqual(quota["multi_valid_rate"], 0.50)
        self.assertEqual(quota["terminal_count"], 64)

    def test_no_preference_utility_or_private_router_features(self):
        for row in self.bundle.decision_rows:
            self.assertNotIn("preferred_actions", row)
            self.assertNotIn("action_utilities", row)
            public = {
                key: row.get(key)
                for key in (
                    "task",
                    "task_context",
                    "observed_state",
                    "execution_memory",
                    "recent_events",
                    "budget",
                    "agent_pool",
                    "candidate_actions",
                )
            }
            serialized = json.dumps(public, sort_keys=True).casefold()
            for term in FORBIDDEN_PUBLIC_TERMS:
                self.assertNotIn(term, serialized)

    def test_every_action_has_traceable_precondition_rationale(self):
        for row in self.bundle.decision_rows:
            candidates = set(row["candidate_actions"])
            valid = set(row["valid_actions"])
            self.assertEqual(set(row["action_rationales"]), candidates)
            for action, rationale in row["action_rationales"].items():
                self.assertEqual(
                    rationale["status"], "valid" if action in valid else "invalid"
                )
                self.assertEqual(
                    rationale["rationale_source"],
                    "formal_transition_preconditions",
                )

    def test_oracle_track_a_is_exact_for_set_valued_labels(self):
        result = evaluate_track_a_v3(self.bundle, OracleRouter())
        self.assertEqual(result["metrics"]["valid_action_accuracy"], 1.0)
        self.assertIsNone(result["metrics"]["combined_weighted_score"])

    def test_track_b_calls_live_actor_and_does_not_simulate_output(self):
        scenario = next(
            item
            for item in self.bundle.scenarios
            if item.family == "sequential_dependency"
        )
        scenario_ids = {scenario.scenario_id}
        provenance = {
            key: item
            for key, item in self.bundle.provenance.items()
            if item.scenario_id in scenario_ids
        }
        subset = BenchmarkBundleV3(
            metadata=dict(self.bundle.metadata),
            agents=self.bundle.agents,
            scenarios=(scenario,),
            provenance=provenance,
            decision_rows=tuple(
                row
                for row in self.bundle.decision_rows
                if row["scenario_id"] in scenario_ids
            ),
            root=self.bundle.root,
        )
        actor = FakeLiveActor()
        router = RecordingOracleRouter()
        result = evaluate_track_b_live_v3(subset, router, actor)
        self.assertGreater(actor.calls, 0)
        self.assertEqual(result["metrics"]["execution_mode"], "live_llm_actor")
        self.assertFalse(result["metrics"]["simulated_actor_outputs"])
        self.assertEqual(result["metrics"]["goal_completion_rate"], 1.0)
        self.assertEqual(result["metrics"]["actor_contract_failure_count"], 0)
        self.assertEqual(result["metrics"]["actor_no_progress_count"], 0)

        router_payload = json.dumps(router.observations, ensure_ascii=False)
        self.assertNotIn("PRIVATE_RAW_OUTPUT", router_payload)
        self.assertNotIn("prior_live_contributions", router_payload)
        self.assertNotIn("candidate_answer", router_payload)
        self.assertTrue(
            any(
                observation.get("recent_events")
                and observation.get("execution_memory", {}).get(
                    "completed_contributions"
                )
                for observation in router.observations[1:]
            )
        )

        self.assertGreater(len(actor.observations), 1)
        later_actor_payload = json.dumps(actor.observations[1], ensure_ascii=False)
        self.assertIn("PRIVATE_RAW_OUTPUT", later_actor_payload)
        self.assertIn("prior_live_contributions", later_actor_payload)
        event_payload = json.dumps(result["episodes"][0]["events"], ensure_ascii=False)
        self.assertIn("PRIVATE_RAW_OUTPUT", event_payload)

    def test_contract_output_ids_normalize_human_labels(self):
        self.assertEqual(_canonical_output_id("candidate answer"), "candidate_answer")
        self.assertEqual(_canonical_output_id("Candidate-Answer"), "candidate_answer")
        self.assertEqual(
            _canonical_output_id(" committed   final artifact "),
            "committed_final_artifact",
        )

    def test_live_actor_retries_contract_mismatch(self):
        class FakeQueryManager:
            def __init__(self):
                self.calls = 0
                self.schemas = []

            def query_structured(self, model, messages, schema, schema_name):
                del model, messages, schema_name
                self.schemas.append(schema)
                self.calls += 1
                if self.calls == 1:
                    return (
                        {
                            "status": "SUCCESS",
                            "contribution": "Produced evidence instead of the requested answer artifact.",
                            "produced_outputs": ["grounded_evidence_packet"],
                            "candidate_answer": "F",
                        },
                        11,
                    )
                return (
                    {
                        "status": "SUCCESS",
                        "contribution": "Produced a supported final candidate answer for the task.",
                        "produced_outputs": ["candidate_answer"],
                        "candidate_answer": "F",
                    },
                    13,
                )

        executor = ModelLiveActorExecutor.__new__(ModelLiveActorExecutor)
        executor.query_manager = FakeQueryManager()
        executor.model = "fake-model"
        executor.max_attempts = 2
        executor.calls = 0
        executor.total_tokens = 0
        executor.total_latency_ms = 0.0

        result = executor.execute(
            task="Solve the question.",
            observation={},
            role_profile={
                "role_name": "Quantitative Reasoner",
                "role_goal": "Solve a quantitative dependency.",
                "supported_inputs": ["question"],
                "expected_outputs": ["candidate answer", "formal derivation"],
            },
        )

        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.tokens, 24)
        self.assertEqual(result.produced_outputs, ("candidate_answer",))
        self.assertEqual(executor.calls, 1)
        expected_enum = executor.query_manager.schemas[0]["properties"][
            "produced_outputs"
        ]["items"]["enum"]
        self.assertEqual(expected_enum, ["candidate_answer", "formal_derivation"])


if __name__ == "__main__":
    unittest.main()
