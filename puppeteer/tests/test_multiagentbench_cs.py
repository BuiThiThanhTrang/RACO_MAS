from __future__ import annotations

import unittest

from role_aware.multiagentbench_cs import (
    build_router_cs_trace,
    build_router_cs_semantic_trace,
    judge_router_cs,
    judge_router_cs_semantic,
    validate_router_cs_payload,
    validate_router_cs_semantic_payload,
)


class MultiAgentBenchStyleCSTests(unittest.TestCase):
    def test_trace_contains_routing_and_cross_role_handoffs_but_not_gold(self):
        candidates = {
            "task_id": "task-1",
            "question": "Question with candidate paragraphs",
            "answer": "hidden gold",
            "correct": True,
            "paths": [
                {
                    "path_uid": "path-1",
                    "prediction": "hidden prediction",
                    "steps": [
                        {
                            "agent": "Evidence Retriever",
                            "action": {"action": "reasoning"},
                            "success": "Success",
                            "result": {"step_data": "P2 supplies entity X."},
                        },
                        {
                            "agent": "Bridge Entity Reasoner",
                            "action": {"action": "reasoning"},
                            "success": "Success",
                            "result": {"step_data": "Using X, P8 supplies Y."},
                        },
                    ],
                }
            ],
        }
        events = [
            {
                "event_type": "routing_decision",
                "decision_id": "decision-1",
                "selected": ["musique-evidence-retriever"],
                "routing_state": {"candidate_answer_exists": False},
            }
        ]

        trace = build_router_cs_trace(candidates, events)

        self.assertEqual(trace["handoff_count"], 1)
        self.assertEqual(
            trace["cross_role_handoffs"][0]["to_role"],
            "Bridge Entity Reasoner",
        )
        rendered = str(trace)
        self.assertNotIn("hidden gold", rendered)
        self.assertNotIn("hidden prediction", rendered)

    def test_no_handoff_forces_strict_zero_communication_score(self):
        score = validate_router_cs_payload(
            {
                "routing_planning_score": 4,
                "state_handoff_communication_score": 5,
                "planning_rationale": "Good route",
                "communication_rationale": "No actual handoff",
                "failure_tags": [],
                "critical_decision_ids": [],
            },
            has_handoffs=False,
        )
        self.assertEqual(score["state_handoff_communication_score"], 0)
        self.assertEqual(score["collaboration_score_raw"], 2.0)
        self.assertEqual(score["collaboration_score_100"], 40.0)

    def test_judge_is_offline_injectable_and_uses_two_component_average(self):
        captured = {}

        def fake_query(model, messages, schema, **kwargs):
            captured.update(
                model=model,
                messages=messages,
                schema=schema,
                kwargs=kwargs,
            )
            return (
                {
                    "routing_planning_score": 4,
                    "state_handoff_communication_score": 3,
                    "planning_rationale": "Adaptive sequence",
                    "communication_rationale": "Usable handoff",
                    "failure_tags": [],
                    "critical_decision_ids": ["decision-2"],
                },
                21,
            )

        score, tokens = judge_router_cs(
            {"task_id": "task-1", "handoff_count": 1},
            model="judge-test",
            query_func=fake_query,
        )

        self.assertEqual(tokens, 21)
        self.assertEqual(score["collaboration_score_raw"], 3.5)
        self.assertEqual(score["collaboration_score_100"], 70.0)
        self.assertEqual(captured["kwargs"]["schema_name"], "router_collaboration_score")

    def test_combined_trace_separates_collaboration_from_committed_answer(self):
        candidates = {
            "task_id": "task-1",
            "question": "Question with candidate paragraphs",
            "paths": [],
        }
        trace = build_router_cs_semantic_trace(
            candidates,
            [],
            prediction="International Tennis Federation (ITF)",
            accepted_answers=["International Tennis Federation", "ITF"],
            question="Which organization?",
        )

        self.assertNotIn("accepted_answers", trace["collaboration_trace"])
        self.assertEqual(
            trace["answer_evaluation"]["committed_prediction"],
            "International Tennis Federation (ITF)",
        )
        self.assertEqual(
            trace["answer_evaluation"]["accepted_answers"],
            ["International Tennis Federation", "ITF"],
        )

    def test_combined_payload_keeps_official_success_on_uncertain(self):
        result = validate_router_cs_semantic_payload(
            {
                "routing_planning_score": 3,
                "state_handoff_communication_score": 2,
                "planning_rationale": "Reasonable route",
                "communication_rationale": "Partial handoff",
                "failure_tags": [],
                "critical_decision_ids": [],
                "answer_verdict": "UNCERTAIN",
                "answer_confidence": 0.4,
                "answer_rationale": "The qualifier is ambiguous.",
            },
            has_handoffs=True,
            official_success=True,
        )

        self.assertTrue(result["semantic_correct"])
        self.assertEqual(result["answer_verdict"], "UNCERTAIN")

    def test_combined_judge_returns_cs_and_semantic_in_one_query(self):
        calls = 0

        def fake_query(model, messages, schema, **kwargs):
            nonlocal calls
            calls += 1
            return (
                {
                    "routing_planning_score": 4,
                    "state_handoff_communication_score": 3,
                    "planning_rationale": "Adaptive sequence",
                    "communication_rationale": "Usable handoff",
                    "failure_tags": [],
                    "critical_decision_ids": ["decision-2"],
                    "answer_verdict": "EQUIVALENT",
                    "answer_confidence": 0.98,
                    "answer_rationale": "Full name plus accepted acronym.",
                },
                33,
            )

        score, tokens = judge_router_cs_semantic(
            {"collaboration_trace": {"handoff_count": 1}},
            model="judge-test",
            query_func=fake_query,
        )

        self.assertEqual(calls, 1)
        self.assertEqual(tokens, 33)
        self.assertEqual(score["collaboration_score_100"], 70.0)
        self.assertTrue(score["semantic_correct"])
        self.assertEqual(score["answer_verdict"], "EQUIVALENT")


if __name__ == "__main__":
    unittest.main()
