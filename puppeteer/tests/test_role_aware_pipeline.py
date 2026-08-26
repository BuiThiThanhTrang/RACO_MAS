import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from agent.register.persona_loader import load_teammate_specs
from agent.register.register import AgentRegister
from config.runtime import load_experiment_config
from inference.graph.agent_graph import AgentGraph
from inference.policy.role_aware_reinforce import CandidateScoringPolicyNetwork, RoleAwareREINFORCE
from role_aware.evidence import PathTerminalEvidence, ProfileEvidenceAccumulator
from role_aware.profile_store import ProfileStore
from tasks import creative_writing, gsm_hard, mmlu_pro, srdd
from tasks.evaluator import BenchmarkEvaluator
from tools.code_interpreter import PythonInterpreter
from utils.file_utils import prepare_python_code


PROJECT = Path(__file__).resolve().parents[1]
S0 = PROJECT / "personas" / "role_aware" / "s0_pool.jsonl"
S3 = PROJECT / "personas" / "role_aware" / "s3_pool.jsonl"
MANIFESTS = PROJECT / "data" / "splits"


class FakeRunner:
    def run_reasoning(self, task):
        return task.get("Answer", "artifact.txt")


class FakeEvaluator:
    def check_gsm8k(self, prediction, answer):
        return prediction == answer

    def check_mmlu(self, prediction, answer):
        return prediction == answer

    def check_srdd(self, prediction, question):
        metrics = {
            "executability": torch.tensor(1.0),
            "completeness": torch.tensor(1.0),
            "consistency": torch.tensor(0.8),
        }
        return torch.tensor(0.8), metrics

    def check_commongen(self, concepts, prediction):
        metrics = {
            "coverage": torch.tensor(1.0),
            "grammar": torch.tensor(0.75),
            "relevance": torch.tensor(0.75),
            "consistency": torch.tensor(0.75),
        }
        return torch.tensor(0.75), metrics

    srdd_binary_success = staticmethod(BenchmarkEvaluator.srdd_binary_success)
    commongen_binary_success = staticmethod(BenchmarkEvaluator.commongen_binary_success)
    metrics_to_json = staticmethod(BenchmarkEvaluator.metrics_to_json)


class RoleAwarePipelineTests(unittest.TestCase):
    def test_s0_has_exactly_two_teammates_for_each_of_eleven_roles(self):
        specs = load_teammate_specs(S0)
        counts = {}
        for spec in specs:
            counts[spec.role_card.role_name] = counts.get(spec.role_card.role_name, 0) + 1
            self.assertNotIn("terminate", spec.role_card.allowed_actions)
        self.assertEqual(len(counts), 11)
        self.assertEqual(set(counts.values()), {2})
        self.assertNotIn("Stop Controller", counts)

    def test_s3_is_fully_unseen_mistral_pool(self):
        specs = load_teammate_specs(S3)
        self.assertTrue(all("mistral" in spec.backbone.lower() for spec in specs))

    def test_closed_book_mask_disables_python_teammates(self):
        specs = load_teammate_specs(S0)
        registry = AgentRegister()
        registry.load(specs)
        profiles = ProfileStore()
        profiles.initialize(specs)
        closed = AgentGraph(registry, profiles, allowed_tools=())
        opened = AgentGraph(registry, profiles, allowed_tools=("run_python",))
        for agent, available in zip(registry.ordered_agents, closed.availability_mask):
            self.assertEqual(available, agent.role_card.role_name != "Python Tool Agent")
        self.assertTrue(all(opened.availability_mask))

    def test_profile_batch_is_order_invariant_and_task_relevant(self):
        specs = load_teammate_specs(S0)
        planner = next(spec for spec in specs if spec.role_card.role_name == "Planner / Decomposer")

        def apply(order):
            store = ProfileStore(alpha=0.2)
            store.initialize(specs)
            accumulator = ProfileEvidenceAccumulator(store, specs)
            for path_id, reward in order:
                accumulator.record(PathTerminalEvidence(
                    task_id="x", task_type="GSM-Hard", path_id=path_id,
                    teammate_ids=(planner.teammate_id, planner.teammate_id),
                    reward=reward, role_adherence={planner.teammate_id: True},
                ))
            accumulator.flush_task("x")
            return store.get(planner.teammate_id)

        first = apply([(0, 1.0), (1, 0.0)])
        second = apply([(1, 0.0), (0, 1.0)])
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.global_reliability_count, 2)
        relevant = {"planning", "general_reasoning", "integration"}
        for name, count in zip(first.capability_names, first.observation_count):
            self.assertEqual(count, 2 if name in relevant else 0)

    def test_candidate_scorer_supports_variable_pool_sizes(self):
        torch.manual_seed(42)
        network = CandidateScoringPolicyNetwork(16, 48)
        state = torch.zeros(1, 16)
        self.assertEqual(network(state, torch.zeros(22, 48)).shape, (1, 23))
        self.assertEqual(network(state, torch.zeros(11, 48)).shape, (1, 12))
        self.assertEqual(network(state, torch.zeros(33, 48)).shape, (1, 34))

    def test_split_manifests_are_disjoint_and_have_locked_sizes(self):
        expected = {
            "gsm_hard": {"train": 200, "dev": 100, "reference": 100, "probe": 10, "final": 909},
            "mmlu_pro": {"train": 200, "dev": 140, "reference": 280, "probe": 10, "final": 2000},
            "srdd": {"train": 200, "dev": 100, "reference": 100, "probe": 10, "final": 790},
            "cw": {"train": 80, "dev": 20, "reference": 40, "probe": 10, "final": 50},
        }
        for dataset, sizes in expected.items():
            manifest = json.loads((MANIFESTS / f"{dataset}_seed42.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["sizes"], sizes)
            flat = [item for split in manifest["splits"].values() for item in split]
            self.assertEqual(len(flat), len(set(flat)))

    def test_reference_budget_is_fifty_per_teammate(self):
        config = load_experiment_config(PROJECT / "config" / "experiments" / "role_aware_gsm.yaml")
        self.assertEqual(config.dataset.reference_items_per_teammate, 50)

    def test_task_schema_sets_required_artifact_type(self):
        self.assertEqual(srdd.format_question({"Description": "x"}, 0)["req"], "code")
        self.assertEqual(creative_writing.format_question({"concepts": ["a"]}, 0)["req"], "text")

    def test_two_items_do_not_grow_registry(self):
        specs = load_teammate_specs(S0)
        registry = AgentRegister()
        registry.load(specs)
        count = registry.agent_num
        for _ in range(2):
            registry.reset_episode_state()
            self.assertEqual(registry.agent_num, count)

    def test_four_task_smoke_runs_create_jsonl_artifacts(self):
        runner = FakeRunner()
        evaluator = FakeEvaluator()
        with tempfile.TemporaryDirectory() as directory:
            gsm_hard.run(runner, evaluator, directory, "probe", 1, seed=42)
            mmlu_pro.run(runner, evaluator, directory, "probe", 1, seed=42)
            srdd.run(runner, evaluator, directory, "probe", 1, seed=42)
            creative_writing.run(runner, evaluator, directory, "probe", 1, seed=42)
            artifacts = list(Path(directory).glob("*.jsonl"))
            self.assertEqual(len(artifacts), 4)
            for artifact in artifacts:
                records = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(records), 1)


    def test_python_tool_protocol_rejects_invalid_and_preserves_multiline_code(self):
        response = "```python\nvalue = 8580402\nprint(value)\n```"
        code, error = prepare_python_code(response)
        self.assertEqual((code, error), ("value = 8580402\nprint(value)", ""))

        prepared, error = prepare_python_code(
            "```python\ndef solution():\n    return 42\n```"
        )
        self.assertEqual(error, "")
        self.assertTrue(prepared.endswith("print(solution())"))

        json_code, _ = prepare_python_code(
            json.dumps({"action": "run_python", "parameter": "print(42)"})
        )
        self.assertEqual(json_code, "")

        invalid, error = prepare_python_code("x = 1 y = 2")
        self.assertEqual(invalid, "")
        self.assertIn("invalid syntax", error.lower())

    def test_python_interpreter_uses_current_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            flag, output = PythonInterpreter("run_python").execute(
                work_path=directory,
                code="print(6 * 7)",
                file_path="",
                timeout_detected=True,
            )
        self.assertTrue(flag, output)
        self.assertEqual(output.strip(), "42")
    def test_python_interpreter_handles_unicode_source_and_output(self):
        code = (
            "# Unicode transition: input → output\n"
            "message = 'Kết quả → 42'\n"
            "print(message)"
        )
        with tempfile.TemporaryDirectory() as directory:
            flag, output = PythonInterpreter("run_python").execute(
                work_path=directory,
                code=code,
                file_path="",
                timeout_detected=True,
            )
        self.assertTrue(flag, output)

    def test_dynamic_routing_threshold_uses_available_agent_count(self):
        policy = object.__new__(RoleAwareREINFORCE)
        policy.training = False
        policy.agent_hash_list = [f"agent-{index}" for index in range(4)]
        policy.agent_graph = SimpleNamespace(availability_mask=[True, False, True, True])
        policy.max_width = 4
        policy.device = torch.device("cpu")
        selected, _ = policy._choose(
            torch.tensor([[1 / 3, 0.0, 1 / 3, 1 / 3, 0.0]]),
            allow_stop=False,
        )
        self.assertEqual(set(selected), {0, 2, 3})

    def test_dynamic_routing_threshold_respects_width_cap(self):
        policy = object.__new__(RoleAwareREINFORCE)
        policy.training = False
        policy.agent_hash_list = [f"agent-{index}" for index in range(4)]
        policy.agent_graph = SimpleNamespace(availability_mask=[True] * 4)
        policy.max_width = 3
        policy.device = torch.device("cpu")
        selected, _ = policy._choose(
            torch.tensor([[0.25, 0.25, 0.25, 0.25, 0.0]]),
            allow_stop=False,
        )
        self.assertEqual(len(selected), 3)

    def test_role_aware_configs_have_no_manual_threshold_or_sample_size(self):
        config_dir = PROJECT / "config" / "experiments"
        for config_path in config_dir.glob("role_aware_*.yaml"):
            text = config_path.read_text(encoding="utf-8")
            self.assertNotIn("sample_size:", text, config_path.name)
            self.assertNotIn("\n    threshold:", text, config_path.name)
            self.assertNotIn("initial_stop_masked:", text, config_path.name)

if __name__ == "__main__":
    unittest.main()
