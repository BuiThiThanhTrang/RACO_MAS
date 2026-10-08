import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from agent.register.persona_loader import load_teammate_specs
from config.runtime import load_experiment_config
from role_aware.aggregation import (
    aggregate_gaia_candidates,
    select_gaia_path_answer,
)
from scripts.reaggregate_gaia_run import path_last_answers, replay_record
from tasks import gaia
from tasks.evaluator import BenchmarkEvaluator


PROJECT_DIR = Path(__file__).resolve().parents[1]


class GaiaBenchmarkTests(unittest.TestCase):
    def test_official_quasi_exact_match_behavior(self):
        self.assertTrue(BenchmarkEvaluator.check_gaia("$1,234", "1234"))
        self.assertTrue(
            BenchmarkEvaluator.check_gaia(
                "FINAL ANSWER: Sea gull!", "seagull"
            )
        )
        self.assertTrue(
            BenchmarkEvaluator.check_gaia("New York; 3", "new york,3")
        )
        self.assertFalse(BenchmarkEvaluator.check_gaia("3, New York", "new york,3"))

    def test_gaia_aggregation_votes_on_normalized_open_answers(self):
        result = aggregate_gaia_candidates(
            ["Sea gull", "seagull!", "another answer"],
            mode="majority",
            seed=42,
            task_id="fixture",
        )
        self.assertEqual(
            BenchmarkEvaluator.normalize_gaia_string(result["prediction"]),
            "seagull",
        )
        self.assertFalse(result["tie"])

    def test_gaia_path_uses_last_nonempty_answer(self):
        self.assertEqual(
            select_gaia_path_answer(["AN", "CUB", " CUB ", ""]),
            " CUB ",
        )

    def test_reaggregation_recovers_last_answers_without_actor_calls(self):
        snapshot = {
            "question": "fixture",
            "paths": [
                {"before_aggregation": "FINAL ANSWER: CUB", "steps": []},
                {"before_aggregation": " CUB", "steps": []},
            ],
        }
        record = {
            "task_id": "fixture",
            "model_answer": "AN",
            "answer": "CUB",
            "correct": False,
        }
        self.assertEqual(path_last_answers(snapshot), ["FINAL ANSWER: CUB", " CUB"])
        replayed = replay_record(record, snapshot)
        self.assertEqual(replayed["model_answer"], "CUB")
        self.assertTrue(replayed["correct"])
        self.assertTrue(replayed["reaggregation"]["changed"])

    def test_loader_preserves_official_split_and_attachment_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "GAIA"
            folder = root / "2023" / "validation"
            folder.mkdir(parents=True)
            attachment = folder / "fixture.txt"
            attachment.write_text("decisive evidence", encoding="utf-8")
            frame = pd.DataFrame(
                [
                    {
                        "task_id": "task-1",
                        "Question": "Read the attachment.",
                        "Level": 1,
                        "Final answer": "evidence",
                        "file_name": "fixture.txt",
                        "file_path": "2023/validation/fixture.txt",
                        "Annotator Metadata": None,
                    }
                ]
            )
            frame.to_parquet(folder / "metadata.level1.parquet", index=False)

            loaded = gaia.load_dataset(
                "dev", level=1, data_root=root, data_limit=1
            )
            task = gaia.format_question(loaded.iloc[0], data_root=root)

        self.assertEqual(task["type"], "GAIA")
        self.assertEqual(task["id"], "task-1")
        self.assertEqual(task["Answer"], "evidence")
        self.assertEqual(
            task["file_name"], "GAIA/2023/validation/fixture.txt"
        )
        self.assertTrue(task["has_attachment"])
        self.assertIn("attachment", task["Question"].lower())

    def test_gaia_pool_and_configs_are_static_and_tool_enabled(self):
        pool = PROJECT_DIR / "personas" / "role_aware" / "gaia_qwen35_9b_pool.jsonl"
        specs = load_teammate_specs(pool)
        self.assertEqual(len(specs), 11)
        self.assertEqual({spec.backbone for spec in specs}, {"qwen-3.5-9b"})
        tools = {tool for spec in specs for tool in spec.role_card.tools}
        self.assertEqual(
            tools,
            {
                "read_file",
                "inspect_media",
                "inspect_spreadsheet",
                "search_web",
                "access_website",
                "run_python",
            },
        )

        expected_depths = {
            "decision_gaia_naive_jev.yaml": 4,
            "frozen_gaia_naive_sol.yaml": 4,
            "decision_gaia_evolving_jev.yaml": 5,
            "decision_gaia_naive_clef.yaml": 5,
            "decision_gaia_evolving_clef.yaml": 5,
        }
        for name, expected_depth in expected_depths.items():
            config = load_experiment_config(
                PROJECT_DIR / "config" / "experiments" / name
            )
            self.assertEqual(config.dataset.name, "GAIA")
            self.assertEqual(config.dataset.mode, "validation")
            self.assertEqual(config.dataset.level, 1)
            self.assertEqual(config.policy["policy_mode"], "frozen")
            self.assertEqual(config.global_config["graph"], {
                "max_width": 2,
                "max_depth": expected_depth,
            })
            self.assertIn("search_web", config.tools.allowed)
            self.assertIn("inspect_media", config.tools.allowed)
            self.assertIn("inspect_spreadsheet", config.tools.allowed)
            self.assertEqual(
                config.policy["routing_guard"]["profile"], "gaia_stage_v2"
            )

    def test_answer_prompt_defines_gaia_contracts(self):
        prompt = json.loads(
            (PROJECT_DIR / "prompts" / "general" / "answer_prompt.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("GAIA_answer", prompt)
        self.assertIn("GAIA_aggregation", prompt)

    def test_downloader_and_compose_docs_do_not_embed_secrets(self):
        source = (
            PROJECT_DIR / "scripts" / "download_gaia.py"
        ).read_text(encoding="utf-8")
        self.assertIn('os.getenv("HF_TOKEN")', source)
        self.assertNotIn("hf_", source)


if __name__ == "__main__":
    unittest.main()
