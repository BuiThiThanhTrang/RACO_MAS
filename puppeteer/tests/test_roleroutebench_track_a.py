from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from routing_benchmark.roleroutebench.track_a.canonicalize import canonical_json, write_jsonl
from routing_benchmark.roleroutebench.track_a.sampling import sample_states
from routing_benchmark.roleroutebench.track_a.schema import derived_action_sets, validate_prelabel
from review_app.track_a_presentation import (
    artifact_items,
    candidate_actions,
    contract_badges,
    current_facts,
    discover_labeling_dirs,
    load_observation,
    role_cards,
    role_pool_is_complete,
    visible_sections,
)


class TrackASchemaTest(unittest.TestCase):
    def test_valid_prelabel_requires_all_actions_in_order(self) -> None:
        candidates = ["Inspector", "Executor", "STOP"]
        payload = {
            "actions": [
                {"action": "Inspector", "classification": "valid_progress", "rationale": "No inspection exists."},
                {"action": "Executor", "classification": "invalid_precondition", "rationale": "No candidate exists."},
                {"action": "STOP", "classification": "invalid_precondition", "rationale": "Task is not terminal."},
            ],
            "terminal_valid": False,
            "state_phase": "initial",
            "confidence": "high",
            "summary": "Inspection is needed.",
        }
        self.assertEqual(validate_prelabel(payload, candidates), [])
        self.assertEqual(derived_action_sets(payload)["valid_actions"], ["Inspector"])
        payload["actions"].reverse()
        self.assertIn("actions must exactly match candidate_actions in source order", validate_prelabel(payload, candidates))

    def test_terminal_flag_and_stop_must_agree(self) -> None:
        candidates = ["STOP"]
        payload = {
            "actions": [{"action": "STOP", "classification": "valid_stop", "rationale": "Done."}],
            "terminal_valid": False,
            "state_phase": "terminal",
            "confidence": "high",
            "summary": "Done.",
        }
        self.assertTrue(validate_prelabel(payload, candidates))


class TrackASamplingTest(unittest.TestCase):
    def test_seeded_sampling_is_stable_and_non_destructive(self) -> None:
        rows = [
            {
                "decision_id": f"d-{index}",
                "domain": "test",
                "router_input_json": canonical_json({"index": index}),
                "candidate_actions_json": canonical_json(["STOP"]),
            }
            for index in range(20)
        ]
        test_tmp = Path(__file__).resolve().parents[1] / ".test-tmp"
        test_tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_tmp) as temp:
            root = Path(temp)
            source = root / "states.jsonl"
            write_jsonl(source, rows)
            first = sample_states(source, root / "a", count=7, seed=42)
            second = sample_states(source, root / "b", count=7, seed=42)
            self.assertEqual(first["decision_ids"], second["decision_ids"])
            self.assertEqual(len(first["decision_ids"]), len(set(first["decision_ids"])))
            self.assertEqual(len(source.read_text(encoding="utf-8").splitlines()), 20)


class TrackAReviewPresentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1] / "data" / "RoleRouteBench" / "track_a_4d_live_v1"
        cls.states = [
            json.loads(line)
            for line in (cls.root / "track_a_states.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_all_router_visible_role_pools_are_complete_and_unmasked(self) -> None:
        self.assertEqual(len(self.states), 568)
        for state in self.states:
            observation = load_observation(state)
            actions = candidate_actions(state, observation)
            self.assertTrue(role_pool_is_complete(state, observation), state["decision_id"])
            self.assertEqual(actions[-1], "STOP")
            self.assertEqual(actions[:-1], [card["agent_id"] for card in role_cards(observation)])

    def test_every_top_level_router_field_remains_in_exact_audit_view(self) -> None:
        for state in self.states:
            observation = load_observation(state)
            sections = visible_sections(observation)
            self.assertEqual(sections["audit_json"], observation)
            self.assertEqual(set(sections["audit_json"]), set(observation))

    def test_each_domain_has_brief_facts_contract_badges_and_evidence(self) -> None:
        by_domain = {}
        for state in self.states:
            by_domain.setdefault(state["domain"], state)
        self.assertEqual(
            set(by_domain),
            {"database_migration_integrity", "code_debugging", "formal_proof", "structured_data_processing"},
        )
        for domain, state in by_domain.items():
            observation = load_observation(state)
            facts = current_facts(domain, observation)
            self.assertTrue(facts, domain)
            self.assertTrue(all(fact.get("path", "").startswith("current_state.") for fact in facts), domain)
            self.assertTrue(contract_badges(domain, observation), domain)
            self.assertIsInstance(artifact_items(observation), list)

    def test_all_labeling_cohorts_are_discovered(self) -> None:
        names = [path.name for path in discover_labeling_dirs(self.root, "all")]
        self.assertEqual(
            names,
            ["labeling", "labeling_openrouter_100_v1", "labeling_openrouter_remaining_v1"],
        )


if __name__ == "__main__":
    unittest.main()

