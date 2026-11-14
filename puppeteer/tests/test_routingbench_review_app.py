from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from review_app.annotation_store import AnnotationStore
from review_app.data_loader import (
    PUBLIC_FORBIDDEN_KEYS,
    RoutingBenchReviewDataset,
    public_leakage_keys,
)
from review_app.presentation import pretty_json, wrapped_json_html
from review_app.workflow_graph import workflow_to_dot


PROJECT_DIR = Path(__file__).resolve().parents[1]
V4_DATASET = PROJECT_DIR / "data" / "RoutingBench" / "v4_beta"


class ReviewDataLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = RoutingBenchReviewDataset(V4_DATASET)

    def test_v4_beta_review_units_and_lazy_public_payloads(self):
        self.assertEqual(self.dataset.version, "4.0-beta")
        self.assertEqual(len(self.dataset.base_workflows), 124)
        self.assertEqual(len(self.dataset.post_fault_pairs), 496)
        scenarios = self.dataset.scenarios_for_base("transactional_service-01")
        self.assertEqual(len(scenarios), 5)
        decisions = self.dataset.decisions_for_scenario(str(scenarios[0]["scenario_id"]))
        self.assertTrue(decisions)
        public = decisions[0]["public"]["observations"]["structured"]
        self.assertEqual(public_leakage_keys(public), [])

    def test_track_c_pair_matches_and_public_payload_is_leakage_safe(self):
        pair_id = self.dataset.track_c_pair_ids()[0]
        report = self.dataset.pair_match_report(pair_id)
        self.assertTrue(all(report.values()), report)
        bundle = self.dataset.track_c_bundle(pair_id)
        for side in ("fault", "control"):
            payload = bundle[side]["public"]["observations"]["structured"]
            self.assertEqual(public_leakage_keys(payload), [])

    def test_raw_view_contains_executor_outputs_for_executed_states(self):
        scenario_id = "v4-code_debugging-01-clean"
        decisions = self.dataset.decisions_for_scenario(scenario_id)
        raw_payloads = [
            item["public"]["observations"]["raw"]
            for item in decisions
            if "raw" in item["public"].get("observations", {})
        ]
        outputs = [
            output
            for payload in raw_payloads
            for output in payload.get("raw_executor_outputs", [])
        ]
        self.assertTrue(outputs)
        self.assertIn("raw_output", outputs[0])
        self.assertIn("agent_id", outputs[0])

    def test_leakage_detector_recurses(self):
        payload = {"state": {"safe": True}, "memory": [{"valid_actions": ["x"]}]}
        self.assertIn("valid_actions", public_leakage_keys(payload))
        self.assertIn("valid_actions", PUBLIC_FORBIDDEN_KEYS)


class PresentationTests(unittest.TestCase):
    def test_wrapped_json_is_safe_and_uses_wrapping_class(self):
        value = {"long": "alpha beta <script>alert('x')</script>"}
        rendered = wrapped_json_html(value)
        self.assertIn('class="rb-json-wrap"', rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("<script>", rendered)
        self.assertIn('"long"', pretty_json(value))


class AnnotationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT_DIR)
        self.root = Path(self.temporary.name)
        self.store = AnnotationStore(self.root / "reviews.sqlite")
        self.items = ["public state is sufficient", "provenance is correct"]

    def tearDown(self):
        self.temporary.cleanup()

    def _save(
        self,
        reviewer: str,
        labels: dict[str, str],
        *,
        notes: dict[str, str] | None = None,
        status: str = "submitted",
    ):
        return self.store.save_review(
            dataset_version="4.0-beta",
            reviewer_id=reviewer,
            object_kind="base_workflow",
            object_id="transactional_service-01",
            checklist_items=self.items,
            checklist_labels=labels,
            checklist_notes=notes or {},
            status=status,
        )

    def test_draft_submission_validation_and_progress(self):
        first = self._save("reviewer_a", {self.items[0]: "approved"}, status="draft")
        self.assertEqual(first["revision"], 1)
        with self.assertRaises(ValueError):
            self._save("reviewer_a", {self.items[0]: "approved"})
        with self.assertRaises(ValueError):
            self._save(
                "reviewer_a",
                {self.items[0]: "approved", self.items[1]: "needs_revision"},
            )
        second = self._save(
            "reviewer_a",
            {self.items[0]: "approved", self.items[1]: "needs_revision"},
            notes={self.items[1]: "The source license needs verification."},
        )
        self.assertEqual(second["revision"], 2)
        progress = self.store.progress("4.0-beta", "reviewer_a", "base_workflow", 1)
        self.assertEqual(progress["submitted"], 1)
        self.assertEqual(progress["flagged"], 1)

    def test_disagreement_adjudication_and_export(self):
        self._save(
            "reviewer_a",
            {self.items[0]: "approved", self.items[1]: "approved"},
        )
        self._save(
            "reviewer_b",
            {self.items[0]: "needs_revision", self.items[1]: "approved"},
            notes={self.items[0]: "The public state omits a required signal."},
        )
        disagreements = self.store.disagreements(
            "4.0-beta", "reviewer_a", "reviewer_b"
        )
        self.assertEqual(len(disagreements), 1)
        report = self.store.agreement_report(
            "4.0-beta", "reviewer_a", "reviewer_b"
        )
        self.assertEqual(report["compared"], 1)
        self.assertEqual(report["exact_agreement"], 0.0)

        self.store.save_adjudication(
            dataset_version="4.0-beta",
            object_kind="base_workflow",
            object_id="transactional_service-01",
            checklist_items=self.items,
            reviewer_a="reviewer_a",
            reviewer_b="reviewer_b",
            reviewer_a_labels=disagreements[0]["reviewer_a_labels"],
            reviewer_b_labels=disagreements[0]["reviewer_b_labels"],
            final_labels={self.items[0]: "needs_revision", self.items[1]: "approved"},
            rationale="The observable signal is required by the transition contract.",
            adjudicator_id="reviewer_c",
        )
        self.assertEqual(
            self.store.adjudicated_ids("4.0-beta"),
            {("base_workflow", "transactional_service-01")},
        )
        counts = self.store.export(self.root / "export", "4.0-beta")
        self.assertEqual(counts["base_workflow_reviews"], 2)
        rows = [
            json.loads(line)
            for line in (self.root / "export" / "review_adjudications.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(rows[0]["final_labels"][self.items[0]], "needs_revision")


class WorkflowGraphTests(unittest.TestCase):
    def test_graph_connects_producer_consumer_and_goal(self):
        scenario = {
            "initial_facts": ["unplanned"],
            "initial_artifacts": [],
            "goal_facts": ["final_committed"],
            "goal_artifacts": ["final_artifact"],
            "transitions": [
                {
                    "transition_id": "plan",
                    "action": "agent_01",
                    "requires_all": ["unplanned"],
                    "add_facts": ["plan_ready"],
                    "add_artifacts": [],
                    "requires_artifacts": [],
                    "requires_any": [],
                    "status": "EXECUTED_SUCCESS",
                    "public_effect": "Plan ready",
                },
                {
                    "transition_id": "finish",
                    "action": "agent_08",
                    "requires_all": ["plan_ready"],
                    "add_facts": ["final_committed"],
                    "add_artifacts": ["final_artifact"],
                    "requires_artifacts": [],
                    "requires_any": [],
                    "status": "EXECUTED_SUCCESS",
                    "public_effect": "Final committed",
                },
            ],
        }
        dot = workflow_to_dot(
            scenario, {"agent_01": "Planner", "agent_08": "Finalizer"}
        )
        self.assertIn("start -> t0", dot)
        self.assertIn("t0 -> t1", dot)
        self.assertIn("t1 -> goal", dot)
        self.assertIn("Planner", dot)


if __name__ == "__main__":
    unittest.main()
