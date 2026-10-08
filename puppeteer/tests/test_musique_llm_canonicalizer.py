from __future__ import annotations

import unittest

from role_aware.musique_llm_canonicalizer import (
    build_llm_canonicalization_input,
    canonicalize_with_llm,
    extract_runtime_answer_with_llm,
    original_question,
    validate_llm_extraction,
    validate_runtime_extraction,
)


class MusiqueLLMCanonicalizerTests(unittest.TestCase):
    @staticmethod
    def snapshot():
        return {
            "question": (
                "Instructions.\n\nQuestion: Who is the spouse of the Green performer?"
                "\n\nCandidate paragraphs:\n[P1] Evidence"
            ),
            "paths": [
                {
                    "stop_reason": "policy_stop",
                    "steps": [
                        {
                            "agent": "Evidence Verifier",
                            "result": {
                                "step_data": (
                                    "FINAL ANSWER: Miquette Giraudy is the partner "
                                    "of Steve Hillage."
                                )
                            },
                        }
                    ],
                },
                {
                    "stop_reason": "depth_limit",
                    "steps": [
                        {
                            "agent": "Answer Integrator",
                            "result": {
                                "step_data": (
                                    '{"status":"NEEDS_EVIDENCE",'
                                    '"final_answer":null,"missing_subquestion":"x"}'
                                )
                            },
                        }
                    ],
                },
            ],
        }

    def test_input_uses_original_question_and_only_path_final_outputs(self):
        value = build_llm_canonicalization_input(self.snapshot())
        self.assertEqual(
            value["original_question"], "Who is the spouse of the Green performer?"
        )
        self.assertEqual(len(value["path_final_outputs"]), 2)
        self.assertNotIn("candidate_context", value)

    def test_null_and_nonfinal_integrator_are_rejected(self):
        judge_input = build_llm_canonicalization_input(self.snapshot())
        with self.assertRaises(ValueError):
            validate_llm_extraction(
                {
                    "status": "ANSWER",
                    "source_path_index": 2,
                    "answer": "null,",
                    "answer_quote": "null,",
                },
                judge_input,
            )

    def test_llm_can_select_short_exact_span_without_generating(self):
        def fake_query(*args, **kwargs):
            return (
                {
                    "status": "ANSWER",
                    "source_path_index": 1,
                    "answer": "Miquette Giraudy",
                    "answer_quote": "Miquette Giraudy",
                },
                19,
            )

        result, tokens = canonicalize_with_llm(
            self.snapshot(), model="fake", query_func=fake_query
        )
        self.assertEqual(result["answer"], "Miquette Giraudy")
        self.assertEqual(result["source_path_index"], 1)
        self.assertEqual(tokens, 19)

    def test_generated_answer_not_in_source_is_rejected(self):
        judge_input = build_llm_canonicalization_input(self.snapshot())
        with self.assertRaises(ValueError):
            validate_llm_extraction(
                {
                    "status": "ANSWER",
                    "source_path_index": 1,
                    "answer": "Invented Person",
                    "answer_quote": "Invented Person",
                },
                judge_input,
            )

    def test_original_question_falls_back_for_plain_input(self):
        self.assertEqual(original_question("Plain question"), "Plain question")

    def test_runtime_extractor_is_gold_blind_and_quote_preserving(self):
        captured = {}

        def fake_query(model, messages, schema, **kwargs):
            captured["messages"] = messages
            return (
                {
                    "status": "FINAL",
                    "answer": "International Tennis Federation (ITF)",
                    "answer_quote": "International Tennis Federation (ITF)",
                },
                23,
            )

        result, tokens = extract_runtime_answer_with_llm(
            "I conclude: International Tennis Federation (ITF)",
            question="What is the competition named after?",
            role="Answer Integrator",
            model="fake",
            query_func=fake_query,
        )

        self.assertEqual(result["status"], "FINAL")
        self.assertEqual(result["answer"], "International Tennis Federation (ITF)")
        self.assertEqual(tokens, 23)
        user_payload = captured["messages"][1]["content"]
        self.assertNotIn("accepted_answers", user_payload)
        self.assertNotIn("gold_answer", user_payload)

    def test_runtime_extractor_rejects_generated_span(self):
        with self.assertRaises(ValueError):
            validate_runtime_extraction(
                {
                    "status": "FINAL",
                    "answer": "International Tennis Federation",
                    "answer_quote": "International Tennis Federation",
                },
                "The output only says ITF.",
            )


if __name__ == "__main__":
    unittest.main()
