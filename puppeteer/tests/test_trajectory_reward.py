import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from inference.policy.role_aware_reinforce import RoleAwareREINFORCE
from role_aware.reward_scorer import AuxiliaryRewardScorer
from role_aware.trajectory_reward import build_trajectory_reward_input


class RecordingScorer:
    def __init__(self, score=0.5, error=None):
        self.score = score
        self.error = error
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.score


def fake_action(result="working"):
    return SimpleNamespace(
        action={"action": "reasoning", "parameter": "solve carefully"},
        result={"step_data": result, "answer": "candidate"},
        success="Success",
        agent_role="QuantitativeReasoner",
        agent_model="SECRET_BACKBONE",
        teammate_id="SECRET_TEAMMATE",
        provider_profile="SECRET_PROVIDER",
        cost=123,
        tokens=456,
    )


def fake_global_info(workpath):
    return SimpleNamespace(
        task={
            "Question": "What is 2 + 2?",
            "Answer": "SECRET_GOLD_ANSWER",
            "model": "SECRET_TASK_MODEL",
        },
        workflow=SimpleNamespace(workflow=[fake_action()]),
        workpath=str(workpath),
        total_cost=1000,
        total_tokens=100,
    )


def bare_policy(scorer, *, active=True):
    policy = object.__new__(RoleAwareREINFORCE)
    policy.trajectories = [[
        {
            "log_prob": torch.tensor([0.0]),
            "action": "teammate",
            "reward": -0.1,
            "finalized": False,
        }
    ]]
    policy.step_penalty = 0.1
    policy.token_cost_weight = 0.1
    policy.cost_normalization = 100000.0
    policy.reward_weight = 0.1
    policy.reward_centered = True
    policy.reward_failure_policy = "task_reward_only"
    policy.trajectory_reward_active = active
    policy.trajectory_reward_config = {
        "model": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        "max_chars": 12000,
        "max_step_chars": 3000,
        "trace_file": "trajectory_rewards.jsonl",
    }
    policy.reward_scorer = scorer
    policy.rewards_history = []
    policy.reward_breakdowns = []
    return policy


class TrajectoryRewardTests(unittest.TestCase):
    def test_serializer_excludes_gold_and_model_identity(self):
        reward_input = build_trajectory_reward_input(
            task={
                "Question": "Solve the task",
                "Answer": "SECRET_GOLD_ANSWER",
            },
            workflow=SimpleNamespace(workflow=[fake_action()]),
            candidate_output="final answer",
        )
        serialized = json.dumps(reward_input.messages, ensure_ascii=False)
        self.assertIn("Solve the task", serialized)
        self.assertIn("QuantitativeReasoner", serialized)
        self.assertIn("final answer", serialized)
        for secret in (
            "SECRET_GOLD_ANSWER",
            "SECRET_BACKBONE",
            "SECRET_TEAMMATE",
            "SECRET_PROVIDER",
        ):
            self.assertNotIn(secret, serialized)

    def test_serializer_reads_only_artifacts_inside_path_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inside = root / "answer.txt"
            inside.write_text("GENERATED_ARTIFACT", encoding="utf-8")
            included = build_trajectory_reward_input(
                {"Question": "write"}, [], "answer.txt", artifact_root=root
            )
            self.assertEqual(included.candidate_output, "GENERATED_ARTIFACT")

            outside = root.parent / "outside-secret.txt"
            outside.write_text("OUTSIDE_SECRET", encoding="utf-8")
            try:
                excluded = build_trajectory_reward_input(
                    {"Question": "write"}, [], str(outside), artifact_root=root
                )
                self.assertNotIn("OUTSIDE_SECRET", excluded.candidate_output)
            finally:
                outside.unlink(missing_ok=True)

    def test_truncation_is_deterministic_and_keeps_final_output(self):
        workflow = SimpleNamespace(
            workflow=[
                fake_action(result=f"step-{index}-" + "x" * 400)
                for index in range(10)
            ]
        )
        first = build_trajectory_reward_input(
            {"Question": "Q" * 1000},
            workflow,
            "FINAL_OUTPUT_MUST_REMAIN",
            max_chars=800,
            max_step_chars=200,
        )
        second = build_trajectory_reward_input(
            {"Question": "Q" * 1000},
            workflow,
            "FINAL_OUTPUT_MUST_REMAIN",
            max_chars=800,
            max_step_chars=200,
        )
        self.assertEqual(first.messages, second.messages)
        self.assertEqual(first.digest, second.digest)
        self.assertIn("FINAL_OUTPUT_MUST_REMAIN", first.candidate_output)
        self.assertIn("Step 10", first.trajectory_text)

    def test_state_representation_never_calls_reward_scorer(self):
        policy = object.__new__(RoleAwareREINFORCE)
        policy.device = "cpu"
        policy.state_representation = lambda messages: (torch.ones(1, 4), 0.0)
        policy.reward_scorer = RecordingScorer(error=AssertionError("must not run"))
        info = SimpleNamespace(
            task={"Question": "router input"},
            workflow=SimpleNamespace(workflow=[], language_state="None"),
        )
        state = policy.get_state_representation(info)
        self.assertEqual(tuple(state.shape), (1, 4))
        self.assertEqual(policy.reward_scorer.calls, [])

    def test_skywork_is_called_once_after_path_completion(self):
        scorer = RecordingScorer(score=1.0)
        with tempfile.TemporaryDirectory() as directory:
            policy = bare_policy(scorer)
            policy.finalize_task(
                {
                    "path_id": 0,
                    "reward": 1.0,
                    "candidate_output": "4",
                    "metrics": {},
                },
                fake_global_info(directory),
            )
            self.assertEqual(len(scorer.calls), 1)
            breakdown = policy.reward_breakdowns[0]
            self.assertEqual(breakdown["skywork_status"], "ok")
            self.assertAlmostEqual(breakdown["skywork_raw"], 1.0)
            self.assertAlmostEqual(breakdown["skywork_weighted"], 0.1)
            self.assertAlmostEqual(breakdown["token_cost_penalty"], 0.001)
            self.assertAlmostEqual(breakdown["terminal_reward"], 1.099)
            self.assertAlmostEqual(breakdown["path_reward"], 0.999)
            trace = Path(directory) / "trajectory_rewards.jsonl"
            self.assertTrue(trace.is_file())
            self.assertEqual(len(trace.read_text(encoding="utf-8").splitlines()), 1)

    def test_centered_score_has_positive_neutral_and_negative_effect(self):
        expected = {1.0: 0.1, 0.5: 0.0, 0.0: -0.1}
        for score, weighted in expected.items():
            with self.subTest(score=score), tempfile.TemporaryDirectory() as directory:
                policy = bare_policy(RecordingScorer(score=score))
                policy.finalize_task(
                    {"path_id": 0, "reward": 0.0, "candidate_output": "x"},
                    fake_global_info(directory),
                )
                self.assertAlmostEqual(
                    policy.reward_breakdowns[0]["skywork_weighted"], weighted
                )

    def test_scorer_failure_falls_back_to_task_reward_only(self):
        scorer = RecordingScorer(error=RuntimeError("server unavailable"))
        with tempfile.TemporaryDirectory() as directory:
            policy = bare_policy(scorer)
            policy.finalize_task(
                {"path_id": 0, "reward": 1.0, "candidate_output": "4"},
                fake_global_info(directory),
            )
            breakdown = policy.reward_breakdowns[0]
            self.assertEqual(breakdown["skywork_status"], "failed")
            self.assertEqual(breakdown["skywork_weighted"], 0.0)
            self.assertAlmostEqual(breakdown["terminal_reward"], 0.999)

    def test_disabled_reward_never_calls_scorer(self):
        scorer = RecordingScorer(error=AssertionError("must not run"))
        with tempfile.TemporaryDirectory() as directory:
            policy = bare_policy(scorer, active=False)
            policy.finalize_task(
                {"path_id": 0, "reward": 1.0, "candidate_output": "4"},
                fake_global_info(directory),
            )
            self.assertEqual(scorer.calls, [])
            self.assertEqual(
                policy.reward_breakdowns[0]["skywork_status"], "disabled"
            )

    def test_probe_mode_does_not_construct_scorer(self):
        graph = SimpleNamespace(num=0, hash_nodes=[], role_nodes=[])
        analyzer = SimpleNamespace(dim=4)
        config = {
            "device": {"type": "cpu"},
            "dataset_mode": "probe",
            "policy_mode": "train",
            "training": {"training": True},
            "paths": {},
        }
        runtime = {
            "task_analyzer": {},
            "trajectory_reward": {
                "enabled": True,
                "apply_during": "train",
                "backend": "http",
            },
        }
        with (
            patch(
                "inference.policy.role_aware_reinforce.TaskAnalyzerRepresentation",
                return_value=analyzer,
            ),
            patch(
                "inference.policy.role_aware_reinforce.AuxiliaryRewardScorer"
            ) as scorer_class,
        ):
            policy = RoleAwareREINFORCE(graph, None, config, runtime)
        self.assertFalse(policy.trajectory_reward_active)
        scorer_class.assert_not_called()

    def test_http_backend_uses_reward_endpoint_contract(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"score": 0.75}
        scorer = AuxiliaryRewardScorer(
            {
                "backend": "http",
                "model": "Skywork/test",
                "base_url": "http://reward.local/v1",
                "api_key": "test-key",
                "max_retries": 1,
            }
        )
        messages = [{"role": "user", "content": "task"}]
        with patch("requests.post", return_value=response) as post:
            self.assertEqual(scorer(messages), 0.75)
        _, kwargs = post.call_args
        self.assertEqual(post.call_args.args[0], "http://reward.local/v1/reward")
        self.assertEqual(kwargs["json"], {"model": "Skywork/test", "messages": messages})
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")

    def test_all_terminal_transitions_carry_candidate_output(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "inference"
            / "reasoning"
            / "reasoning.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(source.count("'candidate_output': aggregated_answer"), 4)


if __name__ == "__main__":
    unittest.main()
