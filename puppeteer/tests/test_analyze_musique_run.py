from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_musique_run import build_report, summarize_records


class AnalyzeMusiqueRunTests(unittest.TestCase):
    def test_aggregates_quality_and_collaboration_metrics(self):
        records = [
            {
                "correct": True,
                "semantic_correct": True,
                "answer_em": 1.0,
                "answer_f1": 1.0,
                "support_precision": 1.0,
                "support_recall": 0.5,
                "support_f1": 2 / 3,
                "paper_compatible_support_precision": 1.0,
                "paper_compatible_support_recall": 0.5,
                "paper_compatible_support_f1": 2 / 3,
                "collaboration": {
                    "milestone_achievement_rate": 0.75,
                    "useful_handoff_rate": 0.5,
                    "collaboration_effectiveness": 0.375,
                    "recovery_success_rate": None,
                    "redundant_transition_rate": 0.0,
                    "useful_call_ratio": 0.75,
                    "total_calls": 4,
                    "handoff_count": 3,
                    "useful_handoff_count": 2,
                    "contribution_by_role": {"Evidence Retriever": 2},
                },
                "router_cs": {
                    "routing_planning_score": 4,
                    "state_handoff_communication_score": 3,
                    "collaboration_score_100": 70.0,
                    "judge_tokens": 20,
                },
            },
            {
                "correct": False,
                "semantic_correct": True,
                "answer_em": 0.0,
                "answer_f1": 0.5,
                "support_precision": None,
                "support_recall": None,
                "support_f1": None,
                "paper_compatible_support_precision": 0.0,
                "paper_compatible_support_recall": 0.0,
                "paper_compatible_support_f1": 0.0,
                "collaboration": {
                    "milestone_achievement_rate": 0.25,
                    "useful_handoff_rate": None,
                    "collaboration_effectiveness": None,
                    "recovery_success_rate": 1.0,
                    "redundant_transition_rate": 0.5,
                    "useful_call_ratio": 0.5,
                    "total_calls": 2,
                    "handoff_count": 0,
                    "useful_handoff_count": 0,
                    "contribution_by_role": {"Evidence Retriever": 1},
                },
            },
        ]

        summary = summarize_records(records)

        self.assertEqual(summary["tasks"], 2)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["semantic_correct"], 2)
        self.assertEqual(summary["semantic_accuracy"], 1.0)
        self.assertEqual(summary["answer_em"], 0.5)
        self.assertEqual(summary["answer_f1"], 0.75)
        self.assertEqual(summary["support_f1"], 2 / 3)
        self.assertEqual(summary["paper_compatible_support_f1"], 1 / 3)
        collaboration = summary["collaboration"]
        self.assertEqual(collaboration["milestone_achievement_rate"], 0.5)
        self.assertEqual(collaboration["useful_handoff_rate"], 0.5)
        self.assertEqual(collaboration["mean_calls_per_task"], 3.0)
        self.assertEqual(collaboration["total_handoffs"], 3)
        self.assertEqual(
            collaboration["contribution_by_role"]["Evidence Retriever"], 3
        )
        self.assertEqual(summary["router_cs"]["scored_tasks"], 1)
        self.assertEqual(summary["router_cs"]["collaboration_score_100"], 70.0)
        self.assertEqual(summary["router_cs"]["judge_tokens"], 20)

    def test_optional_cs_sidecar_is_namespaced_without_changing_musique_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            results = run_dir / "results"
            analysis = run_dir / "analysis"
            results.mkdir()
            analysis.mkdir()
            record = {
                "correct": True,
                "answer_em": 1.0,
                "answer_f1": 1.0,
                "support_precision": 1.0,
                "support_recall": 1.0,
                "support_f1": 1.0,
                "hop_count": 2,
                "collaboration": {},
            }
            (results / "MuSiQue_validation.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            sidecar = {
                "scored_tasks": 1,
                "collaboration_score_100": 70.0,
            }
            (analysis / "multiagentbench_cs_summary.json").write_text(
                json.dumps(sidecar), encoding="utf-8"
            )

            report = build_report(run_dir)

        self.assertEqual(report["overall"]["answer_em"], 1.0)
        self.assertEqual(report["overall"]["support_f1"], 1.0)
        self.assertEqual(report["multiagentbench_cs"], sidecar)

    def test_run_directory_prefers_canonical_results_over_rejudged_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            results = run_dir / "results"
            results.mkdir()
            base = {
                "correct": True,
                "answer_em": 1.0,
                "answer_f1": 1.0,
                "hop_count": 2,
                "collaboration": {},
            }
            (results / "MuSiQue_validation_all_hops.jsonl").write_text(
                json.dumps(base) + "\n", encoding="utf-8"
            )
            (results / "MuSiQue_validation_all_hops.model.rejudged.jsonl").write_text(
                json.dumps({**base, "correct": False}) + "\n", encoding="utf-8"
            )

            report = build_report(run_dir)

        self.assertEqual(report["overall"]["tasks"], 1)
        self.assertEqual(report["overall"]["correct"], 1)

    def test_backfills_paper_support_metric_from_audit_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            results = run_dir / "results"
            trace = run_dir / "audit" / "task" / "attempt-0001"
            results.mkdir()
            trace.mkdir(parents=True)
            snapshot = {
                "aggregation": {"selected_candidate_index": 0},
                "paths": [
                    {
                        "path_uid": "path-1",
                        "steps": [
                            {
                                "agent": "Answer Integrator",
                                "success": "Success",
                                "result": {
                                    "final_answer": "Miquette Giraudy",
                                    "supporting_paragraph_ids": [5, 10],
                                },
                            }
                        ],
                    }
                ],
            }
            evaluation = {
                "answer": "Miquette Giraudy",
                "answer_aliases": [],
                "gold_decomposition": [],
                "supporting_paragraph_indices": [5, 10],
            }
            (trace / "candidates.json").write_text(
                json.dumps(snapshot), encoding="utf-8"
            )
            (trace / "evaluation.json").write_text(
                json.dumps(evaluation), encoding="utf-8"
            )
            record = {
                "correct": True,
                "answer_em": 1.0,
                "answer_f1": 1.0,
                "hop_count": 2,
                "collaboration": {},
                "trace_path": str(trace),
            }
            result_path = results / "MuSiQue_validation_all_hops.jsonl"
            result_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            report = build_report(run_dir)

        self.assertEqual(report["overall"]["paper_compatible_support_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
