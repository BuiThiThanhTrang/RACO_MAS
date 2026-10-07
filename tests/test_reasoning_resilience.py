"""Regression tests for malformed or overlong agent responses."""

import ast
import re
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def isolated_function(relative_path, function_name, namespace, class_name=None):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    nodes = tree.body
    if class_name is not None:
        class_node = next(
            node
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        nodes = class_node.body
    node = next(
        node
        for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    node.decorator_list = []
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / relative_path), "exec"),
        namespace,
    )
    return namespace[function_name]


class ReasoningResilienceTests(unittest.TestCase):
    def test_chat_request_forwards_qwen_chat_template_options(self):
        request = isolated_function(
            "puppeteer/model/model_utils.py",
            "chat_completion_request",
            {
                "Dict": dict,
                "APIConfig": SimpleNamespace(SLOW_FLAG=False, TRUNCATE_FACTOR=0),
                "model_log_and_print": lambda *args: None,
            },
        )
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                usage=SimpleNamespace(
                    completion_tokens=1,
                    prompt_tokens=1,
                    total_tokens=2,
                ),
                choices=[SimpleNamespace(message=SimpleNamespace(content="A"))],
            )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        request(
            messages=[{"role": "user", "content": "answer"}],
            model="Qwen/Qwen3.5-9B",
            new_client=client,
            model_config_dict={
                "max_tokens": 256,
                "temperature": 0.1,
                "top_p": 1.0,
                "n": 1,
                "stream": False,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "logit_bias": {},
                "extra_body": extra_body,
            },
        )

        self.assertEqual(captured["extra_body"], extra_body)

    def test_query_stores_response_text_instead_of_tuple(self):
        query = isolated_function(
            "puppeteer/agent/reasoning_agent.py",
            "_query",
            {},
            class_name="Reasoning_Agent",
        )
        agent = SimpleNamespace(
            dialog_history=[{"role": "system", "content": "system"}],
            query_func=lambda messages, max_tokens=None: ("FINAL ANSWER: I", 17),
        )

        response, tokens = query(agent, "solve", max_tokens=256)

        self.assertEqual(response, "FINAL ANSWER: I")
        self.assertEqual(tokens, 17)
        self.assertEqual(
            agent.dialog_history[-1],
            {"role": "assistant", "content": "FINAL ANSWER: I"},
        )

    def test_mmlu_choice_parser_accepts_supported_formats_only(self):
        extract = isolated_function(
            "puppeteer/agent/reasoning_agent.py",
            "_extract_mmlu_choice",
            {"re": re},
        )

        self.assertEqual(extract("FINAL ANSWER: [i]"), "I")
        self.assertEqual(extract(" option c "), "C")
        self.assertEqual(extract("J"), "J")
        self.assertEqual(extract("I\n\nHowever, I will re-check."), "I")
        self.assertEqual(extract("Discussing variables A and B"), "")

    def test_mmlu_majority_vote_ignores_empty_candidates(self):
        majority_vote = isolated_function(
            "puppeteer/inference/reasoning/reasoning.py",
            "majority_vote",
            {
                "List": list,
                "BenchmarkEvaluator": SimpleNamespace(
                    extract_choice_answer=lambda answer: answer
                ),
                "main_logger": SimpleNamespace(info=lambda *args: None),
            },
            class_name="GraphReasoning",
        )
        reasoning = SimpleNamespace(task={"type": "MMLU-Pro"})

        self.assertEqual(majority_vote(reasoning, ["B", "", "", "B"]), "B")
        self.assertEqual(majority_vote(reasoning, ["", "", "I", "B"]), "B")
        self.assertEqual(majority_vote(reasoning, ["", ""]), "")

    def test_empty_candidate_list_does_not_call_aggregator_or_crash(self):
        aggregate = isolated_function(
            "puppeteer/inference/reasoning/reasoning.py",
            "aggregate_answers",
            {"main_logger": SimpleNamespace(warning=lambda *args: None)},
            class_name="GraphReasoning",
        )
        reasoning = SimpleNamespace(task={"type": "MMLU-Pro"})

        result = aggregate(
            reasoning,
            global_info=SimpleNamespace(),
            answers=[],
            query_func=lambda **kwargs: self.fail("aggregator should not be called"),
        )

        self.assertEqual(result, "")

    def test_single_mmlu_candidate_does_not_call_aggregator(self):
        aggregate = isolated_function(
            "puppeteer/inference/reasoning/reasoning.py",
            "aggregate_answers",
            {"main_logger": SimpleNamespace(info=lambda *args: None)},
            class_name="GraphReasoning",
        )
        reasoning = SimpleNamespace(task={"type": "MMLU-Pro"})

        result = aggregate(
            reasoning,
            global_info=SimpleNamespace(),
            answers=["I"],
            query_func=lambda **kwargs: self.fail("aggregator should not be called"),
        )

        self.assertEqual(result, "I")


if __name__ == "__main__":
    unittest.main()
