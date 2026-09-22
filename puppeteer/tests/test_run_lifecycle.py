import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from agent.register.persona_loader import load_teammate_specs
from config.runtime import load_experiment_config
from inference.policy.role_aware_reinforce import RoleAwareREINFORCE
from main import (
    _resolve_policy_config,
    _resolve_profile_initialization,
)
from role_aware.checkpointing import (
    RunCheckpointManager,
    pool_fingerprint,
    restore_rng_state,
)
from role_aware.profile_store import ProfileStore
from tasks.reference_profile import run_probe_profiles
from tasks.runner import BenchmarkRunner


PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENT = PROJECT / "config" / "experiments" / "role_aware_gsm.yaml"
S0 = PROJECT / "personas" / "role_aware" / "s0_pool.jsonl"
S1 = PROJECT / "personas" / "role_aware" / "s1_pool.jsonl"


class DummyPolicy:
    def __init__(self):
        self.weight = torch.tensor([1.0])

    def training_state_dict(self):
        return {
            "model_state_dict": {"weight": self.weight.clone()},
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "state_dim": 1024,
            "candidate_dim": 48,
            "roles": (),
            "global_step": 7,
            "config": {},
        }


class DummyRunnerPolicy:
    def __init__(self, agent_graph, action_graph, config, runtime_config):
        self.global_step = 0
        self.loaded_optimizer = None
        self.optimizer_updates_enabled = bool(
            config.get("training", {}).get("training", False)
        )

    def training_state_dict(self):
        return {
            "model_state_dict": {"weight": torch.tensor([3.0])},
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "state_dim": 1024,
            "candidate_dim": 48,
            "roles": (),
            "global_step": self.global_step,
            "config": {},
        }

    def load_training_state_dict(self, checkpoint, load_optimizer=True):
        self.loaded_optimizer = bool(load_optimizer)
        self.global_step = int(checkpoint["global_step"])
        self.loaded_weight = checkpoint["model_state_dict"]["weight"].clone()


class RunLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.specs = load_teammate_specs(S0)

    def _profiles(self):
        store = ProfileStore(alpha=0.2)
        store.initialize(self.specs)
        return store

    def _manager(self, directory):
        return RunCheckpointManager(
            directory,
            {
                "dataset_name": "gsm-hard",
                "seed": 42,
                "pool_fingerprint": pool_fingerprint(self.specs),
                "split_manifest_hash": "split-hash",
            },
            interval_items=20,
            snapshot_interval_items=20,
            keep_snapshots=3,
        )

    def _runner_policy_config(self, policy_mode, dataset_mode):
        experiment = load_experiment_config(EXPERIMENT)
        policy = copy.deepcopy(dict(experiment.policy))
        policy["dataset_name"] = "gsm-hard"
        policy["dataset_mode"] = dataset_mode
        policy["policy_mode"] = policy_mode
        policy["seed"] = 42
        policy.setdefault("training", {})["training"] = (
            policy_mode == "train" and dataset_mode == "train"
        )
        policy["training"]["loading"] = False
        return policy

    def test_mode_matrix_disables_non_train_and_requires_evolved_checkpoint(self):
        experiment = load_experiment_config(EXPERIMENT)
        train = _resolve_policy_config(
            experiment, "gsm-hard", "train", "train", None, 42
        )
        initialized = _resolve_policy_config(
            experiment, "gsm-hard", "probe", "initialized", None, 42
        )
        self.assertTrue(train["training"]["training"])
        self.assertFalse(initialized["training"]["training"])
        with self.assertRaisesRegex(ValueError, "only valid"):
            _resolve_policy_config(
                experiment, "gsm-hard", "probe", "train", None, 42
            )
        with self.assertRaisesRegex(ValueError, "explicit checkpoint"):
            _resolve_policy_config(
                experiment, "gsm-hard", "probe", "evolved", None, 42
            )

    @patch("tasks.runner.RoleAwareREINFORCE", DummyRunnerPolicy)
    def test_fresh_train_accepts_explicit_prior_profiles(self):
        experiment = load_experiment_config(EXPERIMENT)
        runtime_config = {
            "graph": {"max_width": 3, "max_depth": 4},
            "chat_context": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            runner = BenchmarkRunner(
                S0,
                runtime_config,
                policy_config=self._runner_policy_config("train", "train"),
                tool_policy=experiment.tools,
                profile_config=experiment.profiles,
                checkpoint_config=experiment.checkpoint,
                run_dir=Path(directory) / "train-priors",
                dataset_name="gsm-hard",
                dataset_mode="train",
                seed=42,
                profile_source="priors",
            )

            self.assertEqual(runner.profile_source, "priors")
            self.assertTrue(runner.profile_updates_enabled)
            self.assertFalse(runner.resume_training)
        with self.assertRaisesRegex(ValueError, "cannot load a checkpoint"):
            _resolve_policy_config(
                experiment,
                "gsm-hard",
                "dev",
                "initialized",
                str(EXPERIMENT),
                42,
            )

    def test_reference_profile_build_never_overwrites_probe_artifact(self):
        experiment = load_experiment_config(EXPERIMENT)
        probe_source, probe_path = _resolve_profile_initialization(
            experiment, build_source="probe"
        )
        reference_source, reference_path = _resolve_profile_initialization(
            experiment, build_source="reference"
        )
        self.assertEqual(probe_source, "probe")
        self.assertEqual(reference_source, "reference")
        self.assertEqual(probe_path, experiment.profiles.initialization.path)
        self.assertNotEqual(reference_path, probe_path)
        self.assertTrue(reference_path.endswith("_reference_profiles.json"))
        with self.assertRaisesRegex(ValueError, "explicit --profile_path"):
            _resolve_profile_initialization(
                experiment, source_override="reference"
            )
        source, path = _resolve_profile_initialization(
            experiment, source_override="priors"
        )
        self.assertEqual((source, path), ("priors", None))
        source, path = _resolve_profile_initialization(
            experiment, source_override="checkpoint"
        )
        self.assertEqual((source, path), ("checkpoint", None))
        with self.assertRaisesRegex(ValueError, "profile source checkpoint"):
            _resolve_profile_initialization(
                experiment,
                source_override="checkpoint",
                path_override="profile.json",
            )

    def test_checkpoint_role_schema_is_validated_before_loading_weights(self):
        class TrackingNetwork:
            state_dim = 1024
            candidate_dim = 48
            loaded = False

            def load_state_dict(self, state, strict=True):
                self.loaded = True

        policy = object.__new__(RoleAwareREINFORCE)
        policy.policy_network = TrackingNetwork()
        policy.global_step = 0
        checkpoint = {
            "state_dim": 1024,
            "candidate_dim": 48,
            "roles": ("wrong-role",),
            "model_state_dict": {},
            "global_step": 7,
        }
        with self.assertRaisesRegex(ValueError, "role schema"):
            policy.load_training_state_dict(checkpoint, load_optimizer=False)
        self.assertFalse(policy.policy_network.loaded)
        self.assertEqual(policy.global_step, 0)

    def test_non_train_update_never_steps_or_increments_global_step(self):
        policy = object.__new__(RoleAwareREINFORCE)
        policy.trajectories = [[
            {
                "log_prob": torch.tensor([0.2], requires_grad=True),
                "reward": 1.0,
                "finalized": True,
            }
        ]]
        policy.entropies = []
        policy.decisions = []
        policy.gamma = 0.99
        policy.entropy_coef = 0.01
        policy.optimizer_updates_enabled = False
        policy.global_step = 4
        policy.device = "cpu"
        metrics = policy.update()
        self.assertFalse(metrics["optimizer_stepped"])
        self.assertEqual(policy.global_step, 4)

    def test_checkpoint_is_written_each_twenty_items_and_keeps_three_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self._manager(directory)
            policy = DummyPolicy()
            profiles = self._profiles()
            progress = {"completed_items": 19}
            self.assertIsNone(manager.save(policy, profiles, progress))
            self.assertFalse(manager.latest_path.exists())
            for completed in (20, 40, 60, 80):
                progress["completed_items"] = completed
                manager.save(policy, profiles, progress)
            self.assertTrue(manager.latest_path.is_file())
            snapshots = sorted(manager.checkpoint_dir.glob("checkpoint_item_*.pt"))
            self.assertEqual(
                [path.name for path in snapshots],
                [
                    "checkpoint_item_0040.pt",
                    "checkpoint_item_0060.pt",
                    "checkpoint_item_0080.pt",
                ],
            )
            progress["completed_items"] = 81
            manager.save(policy, profiles, progress, force=True)
            self.assertFalse(
                (manager.checkpoint_dir / "checkpoint_item_0081.pt").exists()
            )

    def test_fresh_train_saves_immutable_initial_checkpoint_before_first_item(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self._manager(directory)
            policy = DummyPolicy()
            policy.optimizer_updates_enabled = True
            runner = object.__new__(BenchmarkRunner)
            runner.resume_training = False
            runner._initial_checkpoint_saved = False
            runner.policy = policy
            runner.profile_store = self._profiles()
            runner.checkpoint_manager = manager
            runner.progress = {
                "split": "train",
                "initial_data_start": None,
                "requested_data_limit": None,
                "next_split_offset": None,
                "completed_items": 0,
                "last_task_id": None,
                "result_record_count": 0,
                "result_file": None,
            }
            result_path = Path(directory) / "gsm-hard_train_200.jsonl"

            self.assertEqual(runner.resolve_data_window(0, 200), (0, 200))
            self.assertEqual(runner.register_result_path(result_path), "w")
            self.assertTrue(manager.initial_path.is_file())
            self.assertFalse(manager.latest_path.exists())

            payload = manager.load(manager.initial_path)
            self.assertEqual(payload["metadata"]["checkpoint_kind"], "initial")
            self.assertEqual(payload["progress"]["completed_items"], 0)
            self.assertEqual(payload["progress"]["initial_data_start"], 0)
            self.assertEqual(payload["progress"]["requested_data_limit"], 200)
            self.assertEqual(payload["progress"]["next_split_offset"], 0)
            self.assertEqual(
                payload["progress"]["result_file"], result_path.name
            )
            initial_bytes = manager.initial_path.read_bytes()

            runner.progress["completed_items"] = 20
            manager.save(policy, runner.profile_store, runner.progress)
            self.assertTrue(manager.latest_path.is_file())
            self.assertEqual(manager.initial_path.read_bytes(), initial_bytes)
            with self.assertRaises(FileExistsError):
                manager.save_initial(policy, runner.profile_store, runner.progress | {
                    "completed_items": 0
                })

    def test_checkpoint_restores_rng_and_contains_full_run_state(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self._manager(directory)
            random.seed(11)
            np.random.seed(11)
            torch.manual_seed(11)
            manager.save(
                DummyPolicy(),
                self._profiles(),
                {
                    "completed_items": 5,
                    "next_split_offset": 5,
                    "result_record_count": 5,
                },
                force=True,
            )
            expected = (
                random.random(),
                float(np.random.random()),
                float(torch.rand(1).item()),
            )
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            payload = manager.load(manager.latest_path)
            restore_rng_state(payload["rng"])
            actual = (
                random.random(),
                float(np.random.random()),
                float(torch.rand(1).item()),
            )
            self.assertEqual(expected, actual)
            self.assertIn("policy", payload)
            self.assertIn("profiles", payload)
            self.assertIn("progress", payload)
            self.assertEqual(payload["policy"]["global_step"], 7)

    def test_policy_transfer_allows_new_pool_but_exact_restore_rejects_it(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self._manager(directory)
            source.save(
                DummyPolicy(),
                self._profiles(),
                {"completed_items": 20},
                force=True,
            )
            target_metadata = dict(source.metadata)
            target_metadata["pool_fingerprint"] = "new-pool"
            target = RunCheckpointManager(directory, target_metadata)

            with self.assertRaisesRegex(ValueError, "pool_fingerprint"):
                target.load(source.latest_path)
            payload = target.load(
                source.latest_path, validation_scope="policy_transfer"
            )
            self.assertEqual(payload["policy"]["global_step"], 7)
            with self.assertRaisesRegex(ValueError, "validation_scope"):
                target.load(source.latest_path, validation_scope="unsupported")


    def test_completed_probe_profile_loads_before_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            profile_path = directory / "probe_profiles.json"
            store = self._profiles()
            teammate_id = self.specs[0].teammate_id
            store.get(teammate_id).global_reliability_mean = 0.91
            store.save(profile_path)
            fingerprint = pool_fingerprint(self.specs)
            manifest = {
                "schema_version": "1.0",
                "complete": True,
                "task": "gsm-hard",
                "seed": 42,
                "items_per_teammate": 10,
                "pool_fingerprint": fingerprint,
                "split_manifest_hash": "split-hash",
                "teammate_ids": [spec.teammate_id for spec in self.specs],
            }
            profile_path.with_suffix(".manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            runner = object.__new__(BenchmarkRunner)
            runner.dataset_name = "gsm-hard"
            runner.seed = 42
            runner.pool_fingerprint = fingerprint
            runner.split_manifest_hash = "split-hash"
            runner.probe_items_per_teammate = 10
            runner.reference_items_per_teammate = 50
            runner.registry = SimpleNamespace(agent_config=self.specs)
            runner.profile_store = self._profiles()
            runner._load_initial_profiles(profile_path, "probe")
            self.assertEqual(
                runner.profile_store.get(teammate_id).global_reliability_mean,
                0.91,
            )
            manifest["complete"] = False
            profile_path.with_suffix(".manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                runner._load_initial_profiles(profile_path, "probe")

    def test_probe_builder_runs_every_teammate_on_all_ten_items(self):
        class FakeTaskModule:
            @staticmethod
            def load_dataset(mode, count, seed=42, data_start=0):
                self.assertEqual(mode, "probe")
                return pd.DataFrame([{"value": index} for index in range(count)])

            @staticmethod
            def format_question(row, index):
                return {"id": index, "type": "GSM-Hard", "Question": str(row["value"])}

        class FakeProfileStore:
            @staticmethod
            def save(path):
                Path(path).write_text("{}", encoding="utf-8")

        calls = []
        agents = [SimpleNamespace(hash="a"), SimpleNamespace(hash="b")]
        runner = SimpleNamespace(
            registry=SimpleNamespace(
                ordered_agents=agents,
                agent_config=[
                    SimpleNamespace(teammate_id="a"),
                    SimpleNamespace(teammate_id="b"),
                ],
            ),
            graph=SimpleNamespace(availability_mask=[True, True]),
            pool_fingerprint="pool",
            split_manifest_hash="split",
            profile_store=FakeProfileStore(),
            run_reference_item=lambda task, teammate_id: calls.append(
                (teammate_id, task["id"])
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "probe_profiles.json"
            manifest = run_probe_profiles(
                runner,
                "gsm-hard",
                FakeTaskModule,
                directory,
                count=10,
                seed=42,
                profile_path=profile_path,
            )
            self.assertEqual(len(calls), 20)
            self.assertEqual(
                [task_id for teammate, task_id in calls if teammate == "a"],
                list(range(10)),
            )
            self.assertTrue(manifest["complete"])
            self.assertTrue(profile_path.with_suffix(".manifest.json").is_file())
            self.assertTrue(profile_path.with_suffix(".assignment.json").is_file())

    @patch("tasks.runner.RoleAwareREINFORCE", DummyRunnerPolicy)
    def test_evolved_policy_transfer_uses_new_pool_external_profiles(self):
        experiment = load_experiment_config(EXPERIMENT)
        runtime_config = {
            "graph": {"max_width": 3, "max_depth": 4},
            "chat_context": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_runner = BenchmarkRunner(
                S0,
                runtime_config,
                policy_config=self._runner_policy_config("initialized", "dev"),
                tool_policy=experiment.tools,
                profile_config=experiment.profiles,
                checkpoint_config=experiment.checkpoint,
                run_dir=directory / "source",
                dataset_name="gsm-hard",
                dataset_mode="dev",
                seed=42,
                profile_source="priors",
            )
            source_id = self.specs[0].teammate_id
            source_runner.profile_store.get(
                source_id
            ).global_reliability_mean = 0.11
            source_runner.policy.global_step = 23
            checkpoint_path = source_runner.checkpoint_manager.save(
                source_runner.policy,
                source_runner.profile_store,
                {"completed_items": 20},
                force=True,
            )

            s1_specs = load_teammate_specs(S1)
            s1_store = ProfileStore(alpha=experiment.profiles.alpha)
            s1_store.initialize(s1_specs)
            s1_id = s1_specs[0].teammate_id
            s1_store.get(s1_id).global_reliability_mean = 0.87
            profile_path = directory / "gsm-hard_s1_probe_profiles.json"
            s1_store.save(profile_path)
            profile_manifest = {
                "schema_version": "1.0",
                "complete": True,
                "task": "gsm-hard",
                "seed": 42,
                "items_per_teammate": 10,
                "pool_fingerprint": pool_fingerprint(s1_specs),
                "split_manifest_hash": source_runner.split_manifest_hash,
                "teammate_ids": [spec.teammate_id for spec in s1_specs],
            }
            profile_path.with_suffix(".manifest.json").write_text(
                json.dumps(profile_manifest), encoding="utf-8"
            )

            target_runner = BenchmarkRunner(
                S1,
                runtime_config,
                policy_config=self._runner_policy_config("evolved", "final"),
                tool_policy=experiment.tools,
                profile_config=experiment.profiles,
                checkpoint_config=experiment.checkpoint,
                run_dir=directory / "target",
                dataset_name="gsm-hard",
                dataset_mode="final",
                seed=42,
                checkpoint_path=checkpoint_path,
                profile_path=profile_path,
                profile_source="probe",
            )

            self.assertFalse(target_runner.checkpoint_profile_restore)
            self.assertFalse(target_runner.resume_training)
            self.assertFalse(target_runner.profile_updates_enabled)
            self.assertFalse(target_runner.policy.loaded_optimizer)
            self.assertEqual(target_runner.policy.global_step, 23)
            self.assertTrue(
                torch.equal(
                    target_runner.policy.loaded_weight,
                    torch.tensor([3.0]),
                )
            )
            self.assertEqual(
                set(target_runner.profile_store.to_dict()),
                {spec.teammate_id for spec in s1_specs},
            )
            self.assertNotIn(source_id, target_runner.profile_store.to_dict())
            self.assertEqual(
                target_runner.profile_store.get(s1_id).global_reliability_mean,
                0.87,
            )
            self.assertEqual(target_runner.progress["completed_items"], 0)

    def test_resume_window_and_artifact_count_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.jsonl"
            result_path.write_text(
                "".join(json.dumps({"id": index}) + "\n" for index in range(25)),
                encoding="utf-8",
            )
            runner = object.__new__(BenchmarkRunner)
            runner.resume_training = True
            runner.progress = {
                "initial_data_start": 0,
                "requested_data_limit": 40,
                "next_split_offset": 20,
                "completed_items": 20,
                "result_record_count": 20,
                "result_file": result_path.name,
            }
            self.assertEqual(runner.resolve_data_window(0, 40), (20, 20))
            self.assertEqual(runner.register_result_path(result_path), "a")
            committed = [
                line for line in result_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(committed), 20)
            backup = Path(directory) / "result.uncommitted_after_0020.jsonl"
            self.assertTrue(backup.is_file())
            self.assertEqual(
                len(backup.read_text(encoding="utf-8").splitlines()), 5
            )
            self.assertEqual(runner.resolve_data_window(0, 41), (20, 21))
            self.assertEqual(runner.progress["requested_data_limit"], 41)
            with self.assertRaisesRegex(ValueError, "match or increase"):
                runner.resolve_data_window(0, 40)

    def test_evolved_evaluation_never_resumes_training_dataset_progress(self):
        runner = object.__new__(BenchmarkRunner)
        runner.resume_training = False
        runner.progress = {
            "split": "probe",
            "initial_data_start": None,
            "requested_data_limit": None,
            "next_split_offset": None,
            "completed_items": 0,
            "last_task_id": None,
            "result_record_count": 0,
            "result_file": None,
        }

        self.assertEqual(runner.resolve_data_window(3, 10), (3, 10))
        self.assertEqual(runner.progress["initial_data_start"], 3)
        self.assertEqual(runner.progress["requested_data_limit"], 10)
        self.assertEqual(runner.progress["next_split_offset"], 3)
        self.assertEqual(runner.progress["completed_items"], 0)


if __name__ == "__main__":
    unittest.main()
