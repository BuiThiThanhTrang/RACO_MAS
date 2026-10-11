"""Regression tests for malformed or overlong agent responses."""

import ast
import json
import os
import re
from pathlib import Path
import string
import tempfile
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
    def test_initial_policy_masks_terminator_and_renormalizes(self):
        import torch

        init_forward = isolated_function(
            "puppeteer/inference/policy/REINFORCE_continuous.py",
            "init_forward",
            {
                "torch": torch,
                "logger": SimpleNamespace(info=lambda *args: None),
            },
            class_name="ContinuousREINFORCE",
        )
        captured = {}

        def select_agents(probabilities):
            captured["probabilities"] = probabilities.detach().clone()
            return torch.tensor([[0]])

        policy = SimpleNamespace(
            current_task=None,
            update_executed_trajectories=lambda: None,
            get_state_representation=lambda global_info: (
                torch.tensor([[1.0]]),
                0.0,
            ),
            policy_network=lambda state: torch.tensor(
                [[0.2, 0.3, 0.5]], requires_grad=True
            ),
            end_action=torch.tensor(2),
            action_probs_history=[],
            reward_from_rm=[],
            entropy_history=[],
            select_agents_by_threshold=select_agents,
            current_trajectories=[],
            current_trajectory_idx=0,
            append_to_trajectory=lambda *args: None,
        )

        selected = init_forward(
            policy,
            SimpleNamespace(task={"type": "MMLU-Pro"}),
        )

        self.assertEqual(selected.tolist(), [0])
        self.assertAlmostEqual(
            captured["probabilities"][0, 0].item(), 0.4, places=6
        )
        self.assertAlmostEqual(
            captured["probabilities"][0, 1].item(), 0.6, places=6
        )
        self.assertEqual(captured["probabilities"][0, 2].item(), 0.0)
        self.assertAlmostEqual(
            captured["probabilities"].sum().item(), 1.0, places=6
        )

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

    def test_query_rejects_an_empty_provider_response(self):
        query = isolated_function(
            "puppeteer/agent/reasoning_agent.py",
            "_query",
            {},
            class_name="Reasoning_Agent",
        )
        agent = SimpleNamespace(
            dialog_history=[{"role": "system", "content": "system"}],
            query_func=lambda messages, max_tokens=None: ("   ", 17),
        )

        with self.assertRaisesRegex(RuntimeError, "empty response"):
            query(agent, "solve", max_tokens=256)
        self.assertEqual(
            agent.dialog_history[-1],
            {"role": "user", "content": "solve"},
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

    def test_choice_evaluator_rejects_arbitrary_or_empty_text(self):
        extract = isolated_function(
            "puppeteer/tasks/evaluator.py",
            "extract_choice_answer",
            {"re": re},
            class_name="BenchmarkEvaluator",
        )

        self.assertEqual(extract("FINAL ANSWER: [i]"), "I")
        self.assertEqual(extract("I\n\nHowever, I will re-check."), "I")
        self.assertEqual(extract("The best option is (C)."), "C")
        self.assertEqual(extract(""), "")
        self.assertEqual(extract("Discussing variables A and B"), "")

    def test_mmlu_writer_refuses_an_empty_prediction(self):
        require_choice = isolated_function(
            "puppeteer/tasks/mmlu_pro.py",
            "require_mmlu_choice",
            {"string": __import__("string")},
        )
        evaluator = SimpleNamespace(
            extract_choice_answer=lambda response: response.strip().upper()
        )

        self.assertEqual(require_choice(evaluator, " i ", 123), "I")
        with self.assertRaisesRegex(RuntimeError, "result and checkpoint were not written"):
            require_choice(evaluator, "", 123)

    def test_mmlu_run_does_not_persist_or_checkpoint_an_empty_prediction(self):
        require_choice = isolated_function(
            "puppeteer/tasks/mmlu_pro.py",
            "require_mmlu_choice",
            {"string": string},
        )

        class OneRowDataset:
            def iterrows(self):
                yield 0, {"question_id": 123}

            def __len__(self):
                return 1

        run = isolated_function(
            "puppeteer/tasks/mmlu_pro.py",
            "run",
            {
                "json": json,
                "os": os,
                "load_dataset": lambda *args, **kwargs: OneRowDataset(),
                "format_question": lambda row: {
                    "id": row["question_id"],
                    "Answer": "A",
                },
                "require_mmlu_choice": require_choice,
            },
        )
        runner = SimpleNamespace(
            run_reasoning=lambda task: "",
            save_resume_checkpoint=lambda **kwargs: self.fail(
                "an invalid result must not advance the checkpoint"
            ),
        )
        evaluator = SimpleNamespace(
            extract_choice_answer=lambda response: "",
            check_mmlu=lambda prediction, answer: False,
        )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "result and checkpoint were not written"):
                run(
                    runner=runner,
                    evaluator=evaluator,
                    results_dir=directory,
                    mode="train",
                    data_limit=1,
                )

            result = Path(directory) / "MMLU-Pro_train.jsonl"
            self.assertTrue(result.exists())
            self.assertEqual(result.read_text(encoding="utf-8"), "")

    def test_empty_candidate_list_does_not_call_aggregator_or_crash(self):
        aggregate = isolated_function(
            "puppeteer/inference/reasoning/reasoning.py",
            "aggregate_answers",
            {
                "BenchmarkEvaluator": SimpleNamespace(
                    extract_choice_answer=lambda answer: answer
                ),
                "main_logger": SimpleNamespace(warning=lambda *args: None),
            },
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
            {
                "BenchmarkEvaluator": SimpleNamespace(
                    extract_choice_answer=lambda answer: answer
                ),
                "main_logger": SimpleNamespace(info=lambda *args: None),
            },
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

    def test_multiple_mmlu_candidates_use_direct_majority_vote(self):
        aggregate = isolated_function(
            "puppeteer/inference/reasoning/reasoning.py",
            "aggregate_answers",
            {
                "BenchmarkEvaluator": SimpleNamespace(
                    extract_choice_answer=lambda answer: answer
                ),
                "main_logger": SimpleNamespace(
                    info=lambda *args: None,
                    warning=lambda *args: None,
                ),
            },
            class_name="GraphReasoning",
        )
        reasoning = SimpleNamespace(task={"type": "MMLU-Pro"})

        result = aggregate(
            reasoning,
            global_info=SimpleNamespace(),
            answers=["A", "A"],
            query_func=lambda **kwargs: self.fail(
                "MMLU aggregation should not call a provider"
            ),
        )

        self.assertEqual(result, "A")

    def test_mmlu_majority_vote_prefers_later_candidate_on_tie(self):
        aggregate = isolated_function(
            "puppeteer/inference/reasoning/reasoning.py",
            "aggregate_answers",
            {
                "BenchmarkEvaluator": SimpleNamespace(
                    extract_choice_answer=lambda answer: answer
                ),
                "main_logger": SimpleNamespace(
                    info=lambda *args: None,
                    warning=lambda *args: None,
                ),
            },
            class_name="GraphReasoning",
        )
        reasoning = SimpleNamespace(task={"type": "MMLU-Pro"})

        result = aggregate(
            reasoning,
            global_info=SimpleNamespace(),
            answers=["I", "B"],
            query_func=lambda **kwargs: self.fail(
                "MMLU aggregation should not call a provider"
            ),
        )

        self.assertEqual(result, "B")


if __name__ == "__main__":
    unittest.main()
