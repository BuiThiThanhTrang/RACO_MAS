from __future__ import annotations

import unittest
import json
from dataclasses import replace
from pathlib import Path

import yaml

from routing_benchmark.v3.evaluator import _score_end_task
from routing_benchmark.v2.environment import RoutingEnvironmentV2
from routing_benchmark.v2.generator import build_alpha_bundle
from routing_benchmark.v2.schemas import STOP_ACTION
from routing_benchmark.v3beta.generator import (
    RELEASE_REVISION,
    SOURCE_REGISTRY,
    agent_catalog_v3_beta,
)
from routing_benchmark.v3beta.schemas import (
    V3_BETA_SCHEMA_VERSION,
    v3_beta_schema_definitions,
)
from scripts.run_routing_benchmark_v3 import _resolved_run_config


class RoutingBenchmarkV3BetaTests(unittest.TestCase):
    def _terminal_budget_environment(self):
        bundle = build_alpha_bundle()
        _, scenario = [
            item for item in bundle.scenarios if item.family == "budget_sensitive"
        ]
        scenario = replace(
            scenario,
            schema_version=V3_BETA_SCHEMA_VERSION,
            max_steps=1,
            max_cost=1.0,
        )
        return RoutingEnvironmentV2(scenario, bundle.agents)

    def test_stop_has_reserved_terminal_decision_outside_execution_budget(self):
        env = self._terminal_budget_environment()
        finalizer = env.valid_actions()[0]
        final_event = env.step(finalizer)
        self.assertTrue(env.goal_reached)
        self.assertFalse(env.done)
        self.assertEqual(env.remaining_steps, 0)
        self.assertEqual(env.applicable_actions(), (STOP_ACTION,))

        stop_event = env.step(STOP_ACTION)
        self.assertTrue(env.success)
        self.assertEqual(env.terminal_reason, "GOAL_COMPLETE")
        self.assertEqual(env.state.steps_used, 1)
        self.assertEqual(stop_event.step, final_event.step + 1)
        self.assertEqual(stop_event.cost, 0.0)

    def test_non_stop_action_on_reserved_terminal_decision_is_router_error(self):
        env = self._terminal_budget_environment()
        env.step(env.valid_actions()[0])
        event = env.step(env.scenario.agent_pool[0])
        self.assertEqual(event.status, "ACTION_REJECTED")
        self.assertEqual(env.terminal_reason, "TERMINAL_ACTION_ERROR")
        self.assertEqual(env.state.steps_used, 1)
        self.assertEqual(event.cost, 0.0)

    def test_luna_decisions_configs_use_openrouter_typed_endpoint(self):
        config_dir = Path(__file__).resolve().parents[1] / "config" / "routingbench_v3"
        for name in (
            "track_a_openai_decisions_naive.yaml",
            "track_a_openai_decisions_structured.yaml",
            "beta_openai_decisions_naive.yaml",
            "beta_openai_decisions_structured.yaml",
        ):
            config = yaml.safe_load((config_dir / name).read_text(encoding="utf-8"))
            router = config["router"]
            self.assertEqual(router["provider"], "openrouter")
            self.assertEqual(router["model"], "openai/gpt-6-luna-decisions")
            self.assertEqual(router["api_key_env"], "OPENROUTER_API_KEY")

    def test_beta_luna_track_b_uses_gemini_actor(self):
        config_dir = Path(__file__).resolve().parents[1] / "config" / "routingbench_v3"
        for name in (
            "beta_openai_decisions_naive.yaml",
            "beta_openai_decisions_structured.yaml",
        ):
            config = yaml.safe_load((config_dir / name).read_text(encoding="utf-8"))
            self.assertEqual(config["execution"]["actor_model"], "gemini-2.5-flash")

    def test_resolved_config_materializes_cli_overrides(self):
        resolved = _resolved_run_config(
            {
                "benchmark": {"split": "development", "track": "a"},
                "router": {"model": "old-router"},
                "execution": {"actor_model": "old-actor"},
            },
            split="validation",
            track="b",
            profile="structured",
            router_model="openai/gpt-6-luna-decisions",
            actor_model="gemini-2.5-flash",
            source="musique",
            real_tasks_only=True,
            limit=4,
        )
        self.assertEqual(resolved["benchmark"]["split"], "validation")
        self.assertEqual(resolved["benchmark"]["track"], "b")
        self.assertTrue(resolved["benchmark"]["real_tasks_only"])
        self.assertEqual(resolved["router"]["orchestration_profile"], "structured")
        self.assertEqual(resolved["execution"]["actor_model"], "gemini-2.5-flash")

    def test_beta_schema_keeps_set_valued_labels_and_forbids_utilities(self):
        decision = v3_beta_schema_definitions()["decision.schema.json"]
        self.assertEqual(
            decision["properties"]["schema_version"]["const"],
            V3_BETA_SCHEMA_VERSION,
        )
        forbidden = decision["not"]["anyOf"]
        self.assertIn({"required": ["preferred_actions"]}, forbidden)
        self.assertIn({"required": ["action_utilities"]}, forbidden)

    def test_gold_scoring_is_deterministic_and_reports_f1_separately(self):
        exact = _score_end_task(
            "International Tennis Federation (ITF)",
            {
                "metric": "normalized_exact_match",
                "answer": "International Tennis Federation",
                "aliases": ["ITF"],
            },
        )
        self.assertFalse(exact["correct"])
        self.assertGreater(exact["token_f1"], 0.8)
        choice = _score_end_task(
            "Final answer: (C)",
            {"metric": "exact_choice", "answer": "C", "aliases": []},
        )
        self.assertTrue(choice["correct"])

    def test_gaia_is_local_only_and_not_redistributed(self):
        self.assertFalse(SOURCE_REGISTRY["GAIA"]["included"])
        self.assertFalse(SOURCE_REGISTRY["GAIA"]["redistributable"])
        self.assertTrue(SOURCE_REGISTRY["GAIA"]["local_only"])

    def test_beta_pool_adds_domain_overlap_without_backbone_metadata(self):
        catalog = agent_catalog_v3_beta()
        self.assertEqual(len(catalog), 39)
        for agent_id in ("agent_33", "agent_34", "agent_35", "agent_36", "agent_37", "agent_38", "agent_39"):
            profile = catalog[agent_id].to_dict()
            self.assertNotIn("backbone", profile)
            self.assertEqual(profile["action_cost"], 1.0)

    def test_general_solver_card_requires_observable_evidence_inputs(self):
        profile = agent_catalog_v3_beta()["agent_14"].to_dict()
        self.assertNotIn("planning", profile["role_goal"].lower())
        self.assertNotIn("lower-cost", " ".join(profile["avoid_when"]))
        self.assertIn("grounded evidence packet", profile["supported_inputs"])
        self.assertIn("resolved intermediate relation", profile["expected_outputs"])
        self.assertIn(
            "the task still requires initial planning", profile["avoid_when"]
        )

    def test_role_card_release_revision_and_sample_oracle_are_stable(self):
        root = Path(__file__).resolve().parents[1] / "data" / "RoutingBench" / "v3_beta"
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["release_revision"], RELEASE_REVISION)
        with (root / "decision_states.jsonl").open(encoding="utf-8") as handle:
            sample = next(
                json.loads(line)
                for line in handle
                if '"decision_id": "v3b-musique-4hop-03:state-000"' in line
            )
        self.assertEqual(
            set(sample["valid_actions"]), {"agent_01", "agent_02", "agent_25"}
        )
        card = next(
            item for item in sample["agent_pool"] if item["agent_id"] == "agent_14"
        )
        self.assertIn("the task still requires initial planning", card["avoid_when"])


if __name__ == "__main__":
    unittest.main()
