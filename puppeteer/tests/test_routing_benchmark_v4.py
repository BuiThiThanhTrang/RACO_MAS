from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from routing_benchmark.routers import OracleRouter, RandomRouter, RoutingDecision
from routing_benchmark.v2.environment import RoutingEnvironmentV2
from routing_benchmark.v3.evaluator import LiveActorResult
from routing_benchmark.v4.evaluator import (
    _view_observation,
    evaluate_track_a_v4,
    evaluate_track_b_v4,
)
from routing_benchmark.v4.generator import (
    VARIANTS,
    _safe_actions,
    build_v4_bundle,
    generate_decision_rows_v4,
    write_v4_release,
)
from routing_benchmark.v4.loader import BenchmarkBundleV4, load_benchmark_v4
from routing_benchmark.v4.post_fault import generate_post_fault_track
from routing_benchmark.v4.post_fault_evaluator import evaluate_track_c_v4
from routing_benchmark.v4.schemas import V4_SCHEMA_VERSION
from routing_benchmark.v4.validation import validate_v4


class RoutingBenchmarkV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bundle, provenance, fixtures, annotations, task_inputs = build_v4_bundle()
        rows = generate_decision_rows_v4(
            bundle, provenance, fixtures, annotations
        )
        (
            post_fault_inputs,
            post_fault_runtime,
            post_fault_labels,
            post_fault_pairs,
        ) = generate_post_fault_track(
            scenarios=bundle.scenarios,
            agents=bundle.agents,
            fixtures=fixtures,
            annotations=annotations,
            benchmark_version=V4_SCHEMA_VERSION,
        )
        cls.bundle = BenchmarkBundleV4(
            metadata={**bundle.metadata, "decision_state_count": len(rows)},
            agents=bundle.agents,
            scenarios=bundle.scenarios,
            provenance=provenance,
            decision_rows=tuple(rows),
            executor_outputs=tuple(fixtures),
            recovery_annotations=annotations,
            task_inputs=task_inputs,
            post_fault_inputs=tuple(post_fault_inputs),
            post_fault_runtime=tuple(post_fault_runtime),
            post_fault_labels=tuple(post_fault_labels),
            post_fault_pairs=tuple(post_fault_pairs),
        )

    def test_balanced_domains_variants_and_primary_configs(self):
        self.assertEqual(len(self.bundle.scenarios), 30)
        self.assertEqual(
            self.bundle.metadata["primary_configurations"],
            ["naive_state", "structured_state", "structured_history"],
        )
        for domain in {item.family for item in self.bundle.scenarios}:
            variants = {
                self.bundle.recovery_annotations[item.scenario_id]["variant"]
                for item in self.bundle.scenarios
                if item.family == domain
            }
            self.assertEqual(variants, set(VARIANTS))

    def test_raw_history_is_development_only_and_state_view_has_no_history(self):
        for row in self.bundle.decision_rows:
            views = row["memory_views"]
            self.assertEqual(views["state"]["recent_events"], [])
            self.assertEqual(views["state"]["execution_memory"], {})
            self.assertEqual("raw" in views, row["split"] == "development")

    def test_counterfactuals_share_pool_and_initial_safe_actions(self):
        for domain in {item.family for item in self.bundle.scenarios}:
            scenarios = [
                item for item in self.bundle.scenarios if item.family == domain
            ]
            pools = {item.agent_pool for item in scenarios}
            self.assertEqual(len(pools), 1)
            initial_actions = set()
            for scenario in scenarios:
                env = RoutingEnvironmentV2(scenario, self.bundle.agents)
                annotation = self.bundle.recovery_annotations[scenario.scenario_id]
                initial_actions.add(_safe_actions(env, annotation))
            self.assertEqual(len(initial_actions), 1)
            self.assertEqual(len(next(iter(initial_actions))), 2)

    def test_router_visible_id_hides_fault_variant(self):
        initial_rows = [
            item
            for item in self.bundle.decision_rows
            if item["state_provenance"] == "initial"
        ]
        by_domain = {}
        for row in initial_rows:
            observation = _view_observation(row=row, memory_view="state")
            by_domain.setdefault(row["workflow_family"], set()).add(
                observation["scenario_id"]
            )
            self.assertNotIn("invalid", observation["scenario_id"])
            self.assertNotIn("corruption", observation["scenario_id"])
            self.assertNotIn("failure", observation["scenario_id"])
            self.assertNotIn("conflicting", observation["scenario_id"])
        self.assertTrue(all(len(values) == 1 for values in by_domain.values()))

    def test_applicable_no_progress_is_not_action_rejected(self):
        scenario = next(
            item
            for item in self.bundle.scenarios
            if item.scenario_id == "v4-transactional_service-clean"
        )
        annotation = self.bundle.recovery_annotations[scenario.scenario_id]
        env = RoutingEnvironmentV2(scenario, self.bundle.agents)
        self.assertIn("agent_01", env.applicable_actions())
        self.assertNotIn("agent_01", _safe_actions(env, annotation))
        event = env.step("agent_01")
        self.assertEqual(event.status, "EXECUTED_INEFFICIENT")
        self.assertEqual(env.state.actor_failure_count, 0)
        self.assertEqual(env.state.routing_rejection_count, 0)

    def test_consumer_role_card_matches_downstream_operation(self):
        profile = self.bundle.agents["agent_29"]
        self.assertEqual(profile.role_name, "Deterministic Workflow Operator")
        self.assertIn("downstream artifact", profile.expected_outputs)

    def test_structured_data_alternate_has_truthful_role_contract(self):
        profile = self.bundle.agents["agent_41"]
        self.assertEqual(profile.role_name, "Structured Transformation Executor")
        self.assertIn("transformation contract", profile.supported_inputs)
        scenarios = [
            item
            for item in self.bundle.scenarios
            if item.family == "structured_data_processing"
        ]
        self.assertTrue(all("agent_41" in item.agent_pool for item in scenarios))
        for scenario in scenarios:
            env = RoutingEnvironmentV2(scenario, self.bundle.agents)
            annotation = self.bundle.recovery_annotations[scenario.scenario_id]
            self.assertIn("agent_41", _safe_actions(env, annotation))

    def test_recovery_cards_exclude_planning_only_inefficiency(self):
        for agent_id in ("agent_12", "agent_32"):
            profile = self.bundle.agents[agent_id]
            self.assertTrue(
                any("planning" in item for item in profile.avoid_when)
            )

    def test_unsafe_transition_is_executable_but_not_valid(self):
        scenario = next(
            item
            for item in self.bundle.scenarios
            if item.scenario_id == "v4-transactional_service-silent_corruption"
        )
        annotation = self.bundle.recovery_annotations[scenario.scenario_id]
        env = RoutingEnvironmentV2(scenario, self.bundle.agents)
        env.step(scenario.transitions[0].action)
        applicable = set(env.applicable_actions())
        safe = set(_safe_actions(env, annotation))
        unsafe_actions = {
            transition.action
            for transition in scenario.transitions
            if transition.transition_id in annotation["unsafe_transition_ids"]
            and transition.matches(env.state)
        }
        self.assertTrue(unsafe_actions)
        self.assertTrue(unsafe_actions.issubset(applicable))
        self.assertFalse(unsafe_actions.intersection(safe))

    def test_oracle_reaches_full_routing_and_recovery_success(self):
        track_a = evaluate_track_a_v4(
            self.bundle, OracleRouter(), memory_view="state"
        )
        self.assertEqual(track_a["metrics"]["valid_action_accuracy"], 1.0)
        track_b = evaluate_track_b_v4(
            self.bundle, OracleRouter(), memory_view="state"
        )
        self.assertEqual(track_b["metrics"]["goal_completion_rate"], 1.0)

        class FaultCarrierOracle(OracleRouter):
            primary_by_family = {
                "transactional_service": "agent_04",
                "structured_data_processing": "agent_11",
                "cross_app_workflow": "agent_10",
                "code_debugging": "agent_21",
                "personal_device_tool": "agent_29",
                "scientific_procedure": "agent_34",
            }

            def select(self, observation, candidate_actions, oracle_valid_actions=None):
                state = observation.get("state") or {}
                primary = self.primary_by_family[observation["workflow_family"]]
                if int(state.get("step_index", 0)) == 0:
                    if primary in tuple(oracle_valid_actions or ()):
                        return RoutingDecision(primary)
                # Exercise the scheduled fault once, then prevent the broad
                # valid-action oracle from selecting the same known-failed
                # carrier again after recovery.
                if int(state.get("actor_failure_count", 0)) > 0:
                    oracle_valid_actions = tuple(
                        action
                        for action in tuple(oracle_valid_actions or ())
                        if action != primary
                    )
                return super().select(
                    observation, candidate_actions, oracle_valid_actions
                )

        recovery = evaluate_track_b_v4(
            self.bundle, FaultCarrierOracle(), memory_view="state"
        )
        self.assertGreater(recovery["metrics"]["fault_injected_episode_count"], 0)
        self.assertEqual(recovery["metrics"]["recovery_success_rate"], 1.0)
        self.assertEqual(recovery["metrics"]["unsafe_consumption_rate"], 0.0)
        self.assertEqual(recovery["metrics"]["cascade_containment_rate"], 1.0)

    def test_track_c_has_balanced_paired_post_fault_checkpoints(self):
        self.assertEqual(len(self.bundle.post_fault_runtime), 48)
        self.assertEqual(len(self.bundle.post_fault_pairs), 24)
        labels = {
            item["checkpoint_id"]: item for item in self.bundle.post_fault_labels
        }
        split_counts = {}
        for runtime in self.bundle.post_fault_runtime:
            split_counts[runtime["split"]] = split_counts.get(runtime["split"], 0) + 1
        self.assertEqual(
            split_counts,
            {"development": 16, "validation": 16, "test": 16},
        )
        runtime = {
            item["checkpoint_id"]: item for item in self.bundle.post_fault_runtime
        }
        scenario_index = {
            item.scenario_id: item for item in self.bundle.scenarios
        }
        for pair in self.bundle.post_fault_pairs:
            fault_id = pair["fault_checkpoint_id"]
            control_id = pair["control_checkpoint_id"]
            self.assertEqual(labels[fault_id]["condition"], "fault")
            self.assertEqual(labels[control_id]["condition"], "control")
            self.assertEqual(
                runtime[fault_id]["state"]["steps_used"],
                runtime[control_id]["state"]["steps_used"],
            )
            fault_scenario = scenario_index[
                runtime[fault_id]["internal_scenario_id"]
            ]
            control_scenario = scenario_index[
                runtime[control_id]["internal_scenario_id"]
            ]
            self.assertEqual(fault_scenario.task, control_scenario.task)
            self.assertEqual(fault_scenario.agent_pool, control_scenario.agent_pool)

    def test_track_c_public_inputs_hide_private_fault_labels(self):
        forbidden = {
            "condition",
            "pair_id",
            "fault_class",
            "valid_actions",
            "oracle_min_remaining_calls",
            "internal_scenario_id",
        }

        def keys(value):
            result = set()
            if isinstance(value, dict):
                for key, child in value.items():
                    result.add(key)
                    result.update(keys(child))
            elif isinstance(value, list):
                for child in value:
                    result.update(keys(child))
            return result

        for row in self.bundle.post_fault_inputs:
            self.assertTrue(row["checkpoint_id"].startswith("pfc-"))
            self.assertFalse(forbidden.intersection(keys(row)))
            expected = (
                {"state", "structured", "raw"}
                if row["split"] == "development"
                else {"state", "structured"}
            )
            self.assertEqual(set(row["observations"]), expected)

    def test_track_c_oracle_recovers_and_random_is_worse(self):
        oracle = evaluate_track_c_v4(
            self.bundle, OracleRouter(), memory_view="state"
        )
        self.assertEqual(
            oracle["metrics"]["post_fault_goal_completion_rate"], 1.0
        )
        self.assertEqual(
            oracle["metrics"]["clean_continuation_success_rate"], 1.0
        )
        self.assertEqual(
            oracle["metrics"]["first_action_valid_accuracy"], 1.0
        )
        self.assertEqual(oracle["metrics"]["unsafe_consumption_rate"], 0.0)
        random = evaluate_track_c_v4(
            self.bundle, RandomRouter(42), memory_view="state"
        )
        self.assertLess(
            random["metrics"]["post_checkpoint_valid_transition_rate"], 1.0
        )

    def test_random_router_is_below_oracle_and_exposes_propagation_metrics(self):
        result = evaluate_track_a_v4(
            self.bundle, RandomRouter(42), memory_view="state"
        )
        self.assertLess(result["metrics"]["valid_action_accuracy"], 1.0)
        result_b = evaluate_track_b_v4(
            self.bundle, RandomRouter(42), memory_view="state"
        )
        self.assertIn("mean_failure_amplification", result_b["metrics"])
        self.assertIn("fault_injection_coverage", result_b["metrics"])
        self.assertIn("pre_execution_rejection_rate", result_b["metrics"])
        self.assertIn("inefficient_execution_rate", result_b["metrics"])
        self.assertIn(
            "recovery_after_executor_failure_rate", result_b["metrics"]
        )

    def test_live_track_b_executes_actor_and_keeps_controlled_faults(self):
        class SuccessfulActor:
            model = "gemini-2.5-flash"
            calls = 0
            total_tokens = 0
            total_latency_ms = 0.0

            def execute(self, *, task, observation, role_profile):
                del task, observation
                self.calls += 1
                return LiveActorResult(
                    status="SUCCESS",
                    contribution="Produced the requested workflow artifact and observable postcondition evidence.",
                    produced_outputs=tuple(role_profile.get("expected_outputs") or ()),
                    model=self.model,
                )

        scenario = next(
            item
            for item in self.bundle.scenarios
            if item.scenario_id == "v4-transactional_service-explicit_invalid"
        )
        transition_ids = {item.transition_id for item in scenario.transitions}
        subset = BenchmarkBundleV4(
            metadata={**self.bundle.metadata, "workflow_count": 1},
            agents=self.bundle.agents,
            scenarios=(scenario,),
            provenance={
                key: item
                for key, item in self.bundle.provenance.items()
                if item.scenario_id == scenario.scenario_id
            },
            decision_rows=tuple(
                item
                for item in self.bundle.decision_rows
                if item["scenario_id"] == scenario.scenario_id
            ),
            executor_outputs=tuple(
                item
                for item in self.bundle.executor_outputs
                if item["transition_id"] in transition_ids
            ),
            recovery_annotations={
                scenario.scenario_id: self.bundle.recovery_annotations[
                    scenario.scenario_id
                ]
            },
            task_inputs={
                scenario.scenario_id: self.bundle.task_inputs[scenario.scenario_id]
            },
        )
        actor = SuccessfulActor()
        result = evaluate_track_b_v4(
            subset, OracleRouter(), memory_view="state", actor=actor
        )
        self.assertEqual(
            result["metrics"]["execution_mode"],
            "live_llm_actor_with_controlled_fault_injection",
        )
        self.assertGreater(result["metrics"]["live_actor_call_count"], 0)
        self.assertGreater(actor.calls, 0)

    def test_release_round_trip_and_validation(self):
        project_dir = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=project_dir) as directory:
            root = Path(directory)
            write_v4_release(root)
            loaded = load_benchmark_v4(root)
            report = validate_v4(loaded)
            self.assertTrue((root / "public_inputs.jsonl").exists())
            self.assertTrue((root / "private_labels.jsonl").exists())
            self.assertTrue((root / "post_fault_inputs.jsonl").exists())
            self.assertTrue((root / "post_fault_labels.jsonl").exists())
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["workflow_count"], 30)
        self.assertTrue(report["targets"]["has_unsafe_cascade_states"])
        self.assertTrue(
            report["targets"]["public_private_packaging_separated"]
        )
        self.assertTrue(report["targets"]["post_fault_checkpoint_count_48"])
        self.assertTrue(
            report["targets"]["post_fault_replay_and_oracle_valid"]
        )


if __name__ == "__main__":
    unittest.main()
