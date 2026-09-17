import copy
import inspect
import json
import tempfile
import unittest
from json import JSONDecodeError
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import torch

from agent.register.persona_loader import load_teammate_specs
from agent.register.register import AgentRegister
from config.runtime import load_experiment_config
from inference.graph.agent_graph import AgentGraph
from inference.policy.REINFORCE_continuous import ContinuousREINFORCE, MLP_PolicyNetwork
from model.model_utils import (
    build_context_window,
    chat_completion_request,
    estimate_messages_tokens,
)
from role_aware.profile_store import ProfileStore
from role_aware.validation import assert_public_view_has_no_internal_fields


PROJECT_DIR = Path(__file__).resolve().parents[1]
PERSONAS = PROJECT_DIR / "personas" / "role_aware" / "gsm_pool.jsonl"
EXPERIMENT = PROJECT_DIR / "config" / "experiments" / "role_aware_gsm.yaml"


class RoleAwareFoundationTests(unittest.TestCase):
    def setUp(self):
        self.specs = load_teammate_specs(PERSONAS)

    def test_registry_is_loaded_once_and_episode_reset_does_not_grow_pool(self):
        registry = AgentRegister()
        registry.load(self.specs)
        initial_ids = tuple(registry.agent_identifiers)

        registry.reset_episode_state()
        registry.reset_episode_state()

        self.assertEqual(registry.agent_num, len(self.specs))
        self.assertEqual(tuple(registry.agent_identifiers), initial_ids)
        with self.assertRaises(RuntimeError):
            registry.load(self.specs)

    def test_public_router_view_excludes_internal_identity(self):
        registry = AgentRegister()
        registry.load(self.specs)
        profiles = ProfileStore(alpha=0.2)
        profiles.initialize(self.specs)
        graph = AgentGraph(registry=registry, profile_store=profiles)

        for view in profiles.public_views(self.specs):
            assert_public_view_has_no_internal_fields(view)
        prompt = graph.agent_prompt
        for spec in self.specs:
            self.assertNotIn(spec.teammate_id, prompt)
            self.assertNotIn(spec.backbone, prompt)
            if spec.provider_profile:
                self.assertNotIn(spec.provider_profile, prompt)

    def test_policy_network_is_not_a_singleton_and_seed_is_repeatable(self):
        torch.manual_seed(17)
        first = MLP_PolicyNetwork(4, 3)
        torch.manual_seed(17)
        second = MLP_PolicyNetwork(4, 3)

        self.assertIsNot(first, second)
        for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(first_parameter, second_parameter))

    def test_policy_requires_explicit_config(self):
        signature = inspect.signature(ContinuousREINFORCE.__init__)
        self.assertIn("config", signature.parameters)
        self.assertNotIn("config_path", signature.parameters)

    def test_experiment_snapshot_does_not_mutate_source(self):
        source_before = EXPERIMENT.read_bytes()
        experiment = load_experiment_config(EXPERIMENT)
        with tempfile.TemporaryDirectory() as directory:
            isolated = replace(experiment, output_dir=directory, run_id="test-run")
            snapshot = isolated.write_snapshot()
            self.assertTrue(snapshot.is_file())
            self.assertIn("personas_path", snapshot.read_text(encoding="utf-8"))
        self.assertEqual(EXPERIMENT.read_bytes(), source_before)

    def test_context_window_keeps_full_dialog_and_is_deterministic(self):
        messages = [
            {"role": "system", "content": "system " * 150},
            {"role": "user", "content": "old-user " * 2000},
            {"role": "assistant", "content": "old-answer " * 2000},
            {"role": "user", "content": "LATEST REQUEST"},
        ]
        original = copy.deepcopy(messages)
        first, first_stats = build_context_window(
            messages,
            max_input_tokens=1000,
            preserve_latest_messages=2,
            chars_per_token=4.0,
        )
        second, second_stats = build_context_window(
            messages,
            max_input_tokens=1000,
            preserve_latest_messages=2,
            chars_per_token=4.0,
        )

        self.assertEqual(messages, original)
        self.assertEqual(first, second)
        self.assertEqual(first_stats, second_stats)
        self.assertEqual(first[0]["role"], "system")
        self.assertEqual(first[-1]["content"], "LATEST REQUEST")
        self.assertLessEqual(estimate_messages_tokens(first, 4.0), 1000)
        self.assertGreater(first_stats["dropped_messages"], 0)
        self.assertEqual([message["role"] for message in first], ["system", "user"])

    def test_context_window_keeps_provider_safe_role_order(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "x" * 6000},
            {"role": "user", "content": "middle question"},
            {"role": "assistant", "content": "y" * 6000},
            {"role": "user", "content": "latest question"},
        ]

        request, stats = build_context_window(
            messages,
            max_input_tokens=1000,
            preserve_latest_messages=4,
            chars_per_token=4.0,
        )

        roles = [message["role"] for message in request]
        self.assertEqual(roles[0], "system")
        self.assertEqual(roles[1], "user")
        self.assertEqual(roles[-1], "user")
        self.assertNotIn(
            ("assistant", "assistant"), list(zip(roles, roles[1:]))
        )
        self.assertNotIn(("user", "user"), list(zip(roles, roles[1:])))
        self.assertLessEqual(stats["request_estimated_tokens"], 1000)

    def test_chat_retry_shrinks_only_request_copy(self):
        calls = []

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(copy.deepcopy(kwargs))
                if len(calls) < 3:
                    raise RuntimeError("context rejected")
                return SimpleNamespace(
                    usage=SimpleNamespace(
                        completion_tokens=1, prompt_tokens=2, total_tokens=3
                    ),
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content="OK"))
                    ],
                )

        client = SimpleNamespace(
            base_url="https://router.huggingface.co/v1",
            chat=SimpleNamespace(completions=FakeCompletions()),
        )
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "x" * 20000},
        ]
        original = copy.deepcopy(messages)
        with (
            patch("model.model_utils.CHAT_MAX_RETRY_TIMES", 3),
            patch("model.model_utils.CHAT_CONTEXT_MAX_INPUT_TOKENS", 1200),
            patch("model.model_utils.CHAT_CONTEXT_MIN_INPUT_TOKENS", 300),
            patch("model.model_utils.CHAT_CONTEXT_RETRY_SHRINK_FACTOR", 0.5),
            patch("model.model_utils.time.sleep"),
        ):
            response, tokens = chat_completion_request(
                messages,
                "meta-llama/Llama-3.1-8B-Instruct:novita",
                client,
                {"temperature": 0.1, "max_tokens": 4096},
            )

        request_sizes = [
            estimate_messages_tokens(call["messages"]) for call in calls
        ]
        self.assertEqual(messages, original)
        self.assertEqual(tokens, 3)
        self.assertEqual(response.choices[0].message.content, "OK")
        self.assertEqual(len(calls), 3)
        self.assertLessEqual(request_sizes[0], 1200)
        self.assertLessEqual(request_sizes[1], 600)
        self.assertLessEqual(request_sizes[2], 300)
        self.assertGreater(request_sizes[0], request_sizes[1])
        self.assertGreater(request_sizes[1], request_sizes[2])

    def test_chat_recovers_valid_completion_with_trailing_json(self):
        payload = {
            "id": "chatcmpl-fixture",
            "object": "chat.completion",
            "created": 1,
            "model": "fixture",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
        }
        valid_json = json.dumps(payload)
        body = valid_json + json.dumps({"provider_debug": True})

        class RawResponse:
            http_response = httpx.Response(200, text=body)

            @staticmethod
            def parse():
                raise JSONDecodeError("Extra data", body, len(valid_json))

        raw_api = SimpleNamespace(create=lambda **kwargs: RawResponse())
        client = SimpleNamespace(
            base_url="https://router.huggingface.co/v1",
            chat=SimpleNamespace(
                completions=SimpleNamespace(with_raw_response=raw_api)
            ),
        )

        response, tokens = chat_completion_request(
            [{"role": "user", "content": "Question"}], "fixture", client
        )

        self.assertEqual(response.choices[0].message.content, "OK")
        self.assertEqual(tokens, 3)

    def test_chat_does_not_recover_truncated_json(self):
        body = '{"id":"chatcmpl-fixture","choices":['

        class RawResponse:
            http_response = httpx.Response(200, text=body)

            @staticmethod
            def parse():
                raise JSONDecodeError("Expecting value", body, len(body))

        raw_api = SimpleNamespace(create=lambda **kwargs: RawResponse())
        client = SimpleNamespace(
            base_url="https://router.huggingface.co/v1",
            chat=SimpleNamespace(
                completions=SimpleNamespace(with_raw_response=raw_api)
            ),
        )

        with (
            patch("model.model_utils.CHAT_MAX_RETRY_TIMES", 1),
            self.assertRaises(JSONDecodeError),
        ):
            chat_completion_request(
                [{"role": "user", "content": "Question"}], "fixture", client
            )

    def test_entrypoint_does_not_write_shared_policy_config(self):
        source = (PROJECT_DIR / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("config/policy.json", source)
        self.assertIn("experiment.write_snapshot()", source)


if __name__ == "__main__":
    unittest.main()
