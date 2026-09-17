import ast
import copy
import math
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import requests

from puppeteer.reward_api.client import (
    RewardAPIError,
    RewardClient,
    compact_messages,
    validate_response,
)
from deploy.reward_model.schemas import validate_request, InputTooLong, make_response
from deploy.reward_model.artifacts import LOCK


def response_data():
    return {"schema_version": 1, "reward": -3.0, "last_hidden_state": [0.25] * 8192,
            "input_tokens": 20, "model_revision": "abc", "quantization": "Q5_K_M"}


def http_response(status=200, data=None):
    result = Mock(status_code=status)
    result.json.return_value = response_data() if data is None else data
    return result


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"MODAL_KEY": "test-key", "MODAL_SECRET": "test-secret", "REWARD_MODEL_URL": "https://example.modal.run"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.session, self.sleep = Mock(), Mock()
        self.client = RewardClient({}, session=self.session, sleep=self.sleep)

    def test_valid_response_and_transport(self):
        response = http_response()
        self.session.post.return_value = response
        messages = [{"role": "system", "content": "initial state"}]
        original = copy.deepcopy(messages)
        state, reward = self.client.score(messages)
        self.assertEqual((len(state), reward), (8192, -3.0))
        self.assertEqual(messages, original)
        call = self.session.post.call_args.kwargs
        self.assertEqual(call["json"]["messages"], messages)
        self.assertEqual(call["timeout"], (10.0, 180.0))
        self.assertFalse(call["allow_redirects"])
        response.close.assert_called_once()

    def test_transient_errors_retry_then_succeed(self):
        self.session.post.side_effect = [requests.Timeout(), http_response(503), http_response()]
        self.client.score([])
        self.assertEqual(self.session.post.call_count, 3)
        self.assertEqual(self.sleep.call_count, 2)

    def test_permanent_status_never_retried(self):
        for status in (301, 400, 401, 403, 404, 422):
            with self.subTest(status=status):
                self.session.reset_mock()
                self.session.post.return_value = http_response(status)
                with self.assertRaises(RewardAPIError):
                    self.client.score([])
                self.assertEqual(self.session.post.call_count, 1)

    def test_retry_exhaustion_redacts_transport_error(self):
        self.session.post.side_effect = requests.ConnectionError("test-secret")
        with self.assertRaises(RewardAPIError) as error:
            self.client.score([])
        self.assertNotIn("test-secret", str(error.exception))
        self.assertEqual(self.session.post.call_count, 3)

    def test_json_and_output_errors_not_retried(self):
        response = http_response()
        response.json.side_effect = ValueError("bad JSON")
        self.session.post.return_value = response
        with self.assertRaises(RewardAPIError):
            self.client.score([])
        self.assertEqual(self.session.post.call_count, 1)

    def test_invalid_response_contract(self):
        variants = [None, {}, {**response_data(), "schema_version": True},
            {**response_data(), "reward": float("nan")}, {**response_data(), "reward": True},
            {**response_data(), "last_hidden_state": [0.0]},
            {**response_data(), "last_hidden_state": [float("inf")] * 8192},
            {**response_data(), "last_hidden_state": [[0.0]] * 8192},
            {**response_data(), "input_tokens": 4097}]
        for data in variants:
            with self.subTest(data_type=type(data)):
                with self.assertRaises(RewardAPIError):
                    validate_response(data)

    def test_invalid_output_is_not_retried(self):
        self.session.post.return_value = http_response(data={**response_data(), "reward": float("nan")})
        with self.assertRaises(RewardAPIError):
            self.client.score([])
        self.assertEqual(self.session.post.call_count, 1)

    def test_bad_config(self):
        with patch.dict(os.environ, {"REWARD_MODEL_URL": "http://example.com"}):
            with self.assertRaises(ValueError):
                RewardClient({})
        for config in ({"max_attempts": 0}, {"max_attempts": 1.5}, {"read_timeout_seconds": float("nan")}):
            with self.assertRaises(ValueError):
                RewardClient(config)

    def test_credentials_required(self):
        with patch.dict(os.environ, {"MODAL_SECRET": ""}):
            with self.assertRaises(ValueError):
                RewardClient({})



class ContextCompactionTests(unittest.TestCase):
    def test_compaction_preserves_roles_and_does_not_mutate_history(self):
        messages = [
            {"role": "system", "content": "question-" + "q" * 2500},
            {"role": "assistant", "content": "start-" + "a" * 9000 + "-end"},
            {"role": "user", "content": "latest-" + "z" * 1000},
        ]
        original = copy.deepcopy(messages)

        compacted = compact_messages(messages, 8000)

        self.assertEqual(messages, original)
        self.assertEqual(
            [message["role"] for message in compacted],
            [message["role"] for message in messages],
        )
        self.assertLessEqual(
            sum(len(message["content"]) for message in compacted), 8000
        )
        self.assertTrue(compacted[1]["content"].startswith("start-"))
        self.assertTrue(compacted[1]["content"].endswith("-end"))
        self.assertEqual(compacted[2], messages[2])

    def test_http_422_includes_server_detail_and_sent_size(self):
        session = Mock()
        response = http_response(
            status=422,
            data={"detail": "Conversation must contain 1 to 4096 tokens"},
        )
        session.post.return_value = response
        with patch.dict(
            os.environ,
            {
                "MODAL_KEY": "test-key",
                "MODAL_SECRET": "test-secret",
                "REWARD_MODEL_URL": "https://example.modal.run",
            },
        ):
            client = RewardClient(
                {"max_input_chars": 8000}, session=session, sleep=Mock()
            )

        with self.assertRaises(RewardAPIError) as error:
            client.score([{"role": "user", "content": "x" * 9000}])

        message = str(error.exception)
        self.assertIn("4096 tokens", message)
        self.assertIn("sent 8000 content characters", message)
        sent = session.post.call_args.kwargs["json"]["messages"]
        self.assertEqual(sum(len(item["content"]) for item in sent), 8000)

class ContractTests(unittest.TestCase):
    def test_initial_system_message_allowed(self):
        payload = {"schema_version": 1, "messages": [{"role": "system", "content": "Task"}]}
        self.assertEqual(validate_request(payload), payload["messages"])

    def test_invalid_request(self):
        for messages in ([], "text", [{"role": "tool", "content": "x"}], [{"role": "user", "content": 5}]):
            with self.assertRaises(ValueError):
                validate_request({"schema_version": 1, "messages": messages})
        with self.assertRaises(InputTooLong):
            validate_request({"schema_version": 1, "messages": [{"role": "user", "content": "x" * 65537}]})

    def test_server_response_matches_client(self):
        state, reward = validate_response(make_response(-1, [0.1] * 8192, 12, LOCK))
        self.assertEqual(len(state), 8192)
        self.assertEqual(reward, -1)


class AdapterTests(unittest.TestCase):
    def test_remote_wrapper_and_policy_forward_without_transformers(self):
        import torch
        # Execute the production class without unrelated OpenAI/Chroma module startup.
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "puppeteer"))
        self.addCleanup(lambda: sys.path.remove(str(root / "puppeteer")))
        source = ast.parse((root / "puppeteer/model/embedding.py").read_text())
        node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "RewardModelTokenRepresentation")
        scope = {"GLOBAL_CONFIG": {"reward_model": {"backend": "remote"}}, "torch": torch, "List": list}
        fake_client = Mock()
        fake_client.score.return_value = ([0.1] * 8192, -2.0)
        with patch.dict(sys.modules, {"transformers": None}), patch("reward_api.representation.RewardClient", return_value=fake_client):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "embedding.py", "exec"), scope)
            adapter = scope["RewardModelTokenRepresentation"](device="cpu")
            state, reward = adapter([{"role": "system", "content": "task"}])
        self.assertEqual(state.shape, (1, 8192))
        self.assertEqual(state.dtype, torch.float32)
        self.assertEqual(state.device.type, "cpu")
        self.assertFalse(state.requires_grad)
        self.assertEqual(reward, -2.0)
        policy_source = ast.parse((root / "puppeteer/inference/policy/REINFORCE_continuous.py").read_text())
        mlp = next(n for n in policy_source.body if isinstance(n, ast.ClassDef) and n.name == "MLP_PolicyNetwork")
        mlp.decorator_list = []
        scope = {"torch": torch, "nn": torch.nn}
        exec(compile(ast.Module(body=[mlp], type_ignores=[]), "policy.py", "exec"), scope)
        policy = scope["MLP_PolicyNetwork"](8192, 3)
        probabilities = policy(state)
        self.assertEqual(probabilities.shape, (1, 3))
        self.assertAlmostEqual(probabilities.sum().item(), 1, places=5)
        (-probabilities[0, 0].log()).backward()
        self.assertIsNotNone(policy.fc1.weight.grad)
        from deploy.reward_model.compare_policy import probabilities as offline_probabilities
        self.assertTrue(torch.allclose(probabilities, offline_probabilities(state.tolist(), policy.state_dict())))


if __name__ == "__main__":
    unittest.main()
