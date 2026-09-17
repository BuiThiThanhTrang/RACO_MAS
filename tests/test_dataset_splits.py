"""Offline regression coverage for dataset splits and generated artifacts."""
import ast
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "puppeteer"))
from tasks import creative_writing, gsm_hard, mmlu_pro, splits, srdd

LOADERS = {
    "mmlu_pro": mmlu_pro.load_dataset,
    "gsm_hard": gsm_hard.load_dataset,
    "srdd": srdd.load_dataset,
    "cw": creative_writing.load_dataset,
}


class DatasetSplitTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.data = Path(temp.name)
        self.manifests = self.data / "splits"
        self.manifests.mkdir()
        rows = [dict(question_id=i, question=f"question {i}", options=["a", "b"],
                     answer="A", category="math", input=f"input {i}", target=i,
                     Description=f"description {i}", concepts=[f"concept{i}"])
                for i in range(10)]
        frame = pd.DataFrame(rows)
        frame.to_parquet(self.data / "questions.parquet", index=False)
        frame.to_csv(self.data / "requirements.csv", index=False)
        (self.data / "concepts.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        for dataset in LOADERS:
            source = {"srdd": "requirements.csv", "cw": "concepts.jsonl"}.get(
                dataset, "questions.parquet")
            for seed, selected in [(42, {"train": [5, 1], "dev": [7, 2], "final": [8, 0]}),
                                   (77, {"train": [4, 3], "dev": [6, 9], "final": [1, 5]})]:
                manifest = {"dataset": dataset, "seed": seed, "source": source,
                            "sizes": {k: len(v) for k, v in selected.items()},
                            "splits": selected}
                (self.manifests / f"{dataset}_seed{seed}.json").write_text(
                    json.dumps(manifest), encoding="utf-8")
        for name, value in [("DATA_DIR", self.data), ("SPLITS_DIR", self.manifests)]:
            patched = patch.object(splits, name, value)
            patched.start()
            self.addCleanup(patched.stop)

    def ids(self, data):
        return [row["question_id"] for row in data] if isinstance(data, list) else data.question_id.tolist()

    def test_all_modes_use_exact_manifest_rows_in_order(self):
        for dataset, loader in LOADERS.items():
            observed = {}
            for mode, expected in [("train", [5, 1]), ("validation", [7, 2]), ("test", [8, 0])]:
                with self.subTest(dataset=dataset, mode=mode):
                    observed[mode] = self.ids(loader(mode))
                    self.assertEqual(observed[mode], expected)
                    self.assertEqual(self.ids(loader(mode)), expected)
            self.assertTrue(set(observed["train"]).isdisjoint(observed["validation"]))
            self.assertTrue(set(observed["train"]).isdisjoint(observed["test"]))

    def test_limit_is_applied_after_split_selection(self):
        for dataset, loader in LOADERS.items():
            with self.subTest(dataset=dataset):
                self.assertEqual(self.ids(loader("validation", data_limit=1)), [7])
                self.assertEqual(self.ids(loader("test", data_limit=100)), [8, 0])
                self.assertEqual(self.ids(loader("test", data_limit=0)), [])

    def test_seed_selects_corresponding_manifest_for_every_dataset(self):
        for dataset, loader in LOADERS.items():
            with self.subTest(dataset=dataset):
                self.assertEqual(self.ids(loader("validation", seed=77)), [6, 9])
                with self.assertRaises(FileNotFoundError):
                    loader("validation", seed=99)

    def test_missing_split_never_falls_back_to_source_dataset(self):
        for dataset, loader in LOADERS.items():
            with self.subTest(dataset=dataset):
                path = self.manifests / f"{dataset}_seed42.json"
                manifest = json.loads(path.read_text())
                del manifest["splits"]["final"]
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    loader("test")

    def test_invalid_indices_are_rejected(self):
        path = self.manifests / "mmlu_pro_seed42.json"
        manifest = json.loads(path.read_text())
        for indices, error in [([7, 7], ValueError), ([7, 10], IndexError), ([-1, 2], IndexError)]:
            with self.subTest(indices=indices):
                manifest["splits"]["dev"] = indices
                path.write_text(json.dumps(manifest))
                with self.assertRaises(error):
                    mmlu_pro.load_dataset("validation")

    def test_mmlu_resume_offset_is_within_selected_split(self):
        self.assertEqual(self.ids(mmlu_pro.load_dataset("train", data_start=1, data_limit=1)), [1])
        result = self.data / "result.jsonl"
        result.write_text(json.dumps({"id": 5}) + "\n")
        mmlu_pro.validate_resume(result, "train", 1, 42)
        result.write_text(json.dumps({"id": 0}) + "\n")
        with self.assertRaises(ValueError):
            mmlu_pro.validate_resume(result, "train", 1, 42)


def production_functions(path, names, namespace, class_name=None):
    """Load production functions without initializing unrelated API/model clients."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = tree.body
    if class_name:
        body = next(node for node in body if isinstance(node, ast.ClassDef)
                    and node.name == class_name).body
    nodes = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / path), "exec"), namespace)


class ArtifactOutputTests(unittest.TestCase):
    def test_srdd_and_cw_save_artifacts_for_the_evaluator(self):
        namespace = {"os": os, "ast": ast, "json": json, "re": re, "Optional": Optional,
                     "main_logger": logging.getLogger("artifact-test")}
        production_functions("puppeteer/utils/file_utils.py",
            {"read_code", "write_file", "write_code", "write_text", "code_is_valid", "extract_code_from_text"}, namespace)
        production_functions("puppeteer/agent/reasoning_agent.py", {"_reasoning_operation"}, namespace,
                             class_name="Reasoning_Agent")
        production_functions("puppeteer/inference/reasoning/reasoning.py", {"aggregate_answers"}, namespace,
                             class_name="GraphReasoning")
        cases = [
            (srdd.format_question({"Description": "print a greeting"}, 0), "code", ".py",
             "```python\nmessage = 'Hello'\nprint(message)\n```", "print(message)"),
            (creative_writing.format_question({"concepts": ["cat", "mat"]}, 0), "text", ".txt",
             "REASONING RESULT: use both concepts.\nFINAL ANSWER: The cat sits on the mat.",
             "The cat sits on the mat."),
        ]
        for task, req, extension, response, expected in cases:
            with self.subTest(task=task["type"]), tempfile.TemporaryDirectory() as folder:
                self.assertEqual(task["req"], req)
                answers = []
                info = SimpleNamespace(task=task, code_path="", logger=namespace["main_logger"],
                                       add_answer=answers.append)
                agent = SimpleNamespace(workspace_path=folder, role="TestAgent", system_prompt="test",
                                        _query=lambda _: (response, 10))
                previous_cwd = Path.cwd()
                try:
                    os.chdir(ROOT / "puppeteer")
                    with contextlib.redirect_stdout(io.StringIO()):
                        namespace["_reasoning_operation"](agent, {"action": "reasoning", "parameter": ""}, info)
                finally:
                    os.chdir(previous_cwd)
                artifact = Path(info.code_path)
                self.assertEqual(artifact.suffix, extension)
                self.assertIn(expected, artifact.read_text(encoding="utf-8"))
                self.assertEqual(json.loads(answers[-1])["code_path"], str(artifact))
                selected = namespace["aggregate_answers"](
                    SimpleNamespace(task=task), info, answers, query_func=lambda **_: self.fail("Unexpected API call"))
                self.assertEqual(selected, str(artifact))


if __name__ == "__main__":
    unittest.main()