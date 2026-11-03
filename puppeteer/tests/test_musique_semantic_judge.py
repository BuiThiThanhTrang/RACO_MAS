from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from inference.reasoning.reasoning import GraphReasoning
from role_aware.audit_trace import AuditTrace
from role_aware.musique_semantic_judge import (
    build_semantic_judge_input,
    judge_semantic_equivalence,
    validate_semantic_verdict,
)


class MusiqueSemanticJudgeTests(unittest.TestCase):
    def test_input_contains_only_committed_answer_question_and_accepted_answers(self):
        value = build_semantic_judge_input(
            "International Tennis Federation (ITF)",
            ["International Tennis Federation", "ITF"],
            question="What is the competition named after?",
        )
        self.assertEqual(value["prediction"], "International Tennis Federation (ITF)")
        self.assertEqual(value["accepted_answers"], ["International Tennis Federation", "ITF"])
        self.assertNotIn("paths", value)
        self.assertNotIn("candidate_context", value)

    def test_equivalent_verdict_is_validated(self):
        result = validate_semantic_verdict(
            {
                "verdict": "EQUIVALENT",
                "confidence": 0.99,
                "reason": "The prediction adds the accepted acronym.",
            }
        )
        self.assertTrue(result["equivalent"])

    def test_judge_supports_structured_fake_query(self):
        def fake_query(*args, **kwargs):
            return (
                {
                    "verdict": "EQUIVALENT",
                    "confidence": 0.98,
                    "reason": "Full name plus its acronym.",
                },
                31,
            )

        result, tokens = judge_semantic_equivalence(
            "International Tennis Federation (ITF)",
            ["International Tennis Federation", "ITF"],
            question="What is the competition named after?",
            model="fake",
            query_func=fake_query,
        )
        self.assertTrue(result["equivalent"])
        self.assertEqual(result["verdict"], "EQUIVALENT")
        self.assertEqual(tokens, 31)

    def test_empty_prediction_fails_without_model_call(self):
        called = False

        def fake_query(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("query should not be called")

        result, tokens = judge_semantic_equivalence(
            "",
            ["gold"],
            model="fake",
            query_func=fake_query,
        )
        self.assertFalse(result["equivalent"])
        self.assertEqual(tokens, 0)
        self.assertFalse(called)

    def test_reasoning_combined_post_task_judge_is_cached(self):
        reasoning = GraphReasoning.__new__(GraphReasoning)
        reasoning.runtime_config = {
            "musique": {
                "semantic_outcome": {
                    "enabled": True,
                    "model": "fake",
                    "use_for": ["reporting", "route_experience"],
                }
            }
        }
        reasoning.task = {
            "Question": "Question: What is the competition named after?",
        }
        reasoning.audit = AuditTrace(Path("."), "run", "task", "attempt", enabled=False)
        reasoning._musique_semantic_cache = {}
        fake_result = (
            {
                "prompt_version": "router_cs_semantic_v2",
                "routing_planning_score": 4,
                "state_handoff_communication_score": 3,
                "collaboration_score_raw": 3.5,
                "collaboration_score_100": 70.0,
                "planning_rationale": "Adaptive sequence.",
                "communication_rationale": "Usable handoff.",
                "failure_tags": [],
                "critical_decision_ids": [],
                "answer_verdict": "EQUIVALENT",
                "answer_equivalent": True,
                "semantic_correct": True,
                "answer_confidence": 0.99,
                "answer_rationale": "Full name plus acronym.",
                "judge_model": "fake",
                "judge_tokens": 21,
            },
            21,
        )
        snapshot = {
            "task_id": "task",
            "question": reasoning.task["Question"],
            "paths": [],
            "candidate_hash": "candidate-hash",
        }
        with patch(
            "inference.reasoning.reasoning.judge_router_cs_semantic",
            return_value=fake_result,
        ) as judge:
            first = reasoning._judge_musique_post_task(
                snapshot,
                "International Tennis Federation (ITF)",
                ["International Tennis Federation", "ITF"],
                official_success=False,
            )
            second = reasoning._judge_musique_post_task(
                snapshot,
                "International Tennis Federation (ITF)",
                ["International Tennis Federation", "ITF"],
                official_success=False,
            )

        self.assertTrue(first["effective_equivalent"])
        self.assertEqual(second["collaboration_score_100"], 70.0)
        judge.assert_called_once()


if __name__ == "__main__":
    unittest.main()
