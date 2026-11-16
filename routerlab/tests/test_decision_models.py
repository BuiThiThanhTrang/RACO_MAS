from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from routerlab.config import DecisionModelConfig, load_config
from routerlab.contracts import BudgetSnapshot, ROUTER_ACTIONS, RoutingState
from routerlab.capabilities import ROUTER_CAPABILITY_CARDS
from routerlab.decision_models import (
    ACTION_DESCRIPTIONS,
    LayaAdapter,
    OpenAIDecisionsAdapter,
    QUESTION_INSTRUCTIONS,
    SystemOneHTTPAdapter,
    resolve_decision_model,
)
from routerlab.model import ReplayModelClient
from routerlab.routers import DecisionModelRouter, ModelRouter


def state() -> RoutingState:
    return RoutingState(
        task_id="Mbpp/1",
        task_prompt="Write f",
        entry_point="f",
        current_code=None,
        code_version=0,
        base_feedback=None,
        logic_review=(),
        edge_cases=(),
        action_history=(),
        budget=BudgetSnapshot(9, 20, 600000),
    )


class DecisionModelTests(unittest.TestCase):
    def test_registry_resolves_supported_model_provider_pairs(self):
        cases = {
            ("laya", "local"): "laya",
            ("jev", "typesafe"): "systemone_http",
            ("jev", "openrouter"): "systemone_http",
            ("jev", "huggingface"): "hf_decision",
            ("openai_decisions", "openai"): "openai_decisions",
            ("openai_decisions", "openrouter"): "systemone_http",
            ("drex", "local"): "systemone_http",
            ("drex", "modal"): "systemone_http",
        }
        for (name, provider), adapter in cases.items():
            with self.subTest(name=name, provider=provider):
                resolved = resolve_decision_model(
                    DecisionModelConfig(name=name, provider=provider)
                )
                self.assertEqual(resolved.adapter, adapter)

    def test_laya_uses_typed_decisions_checkpoint(self):
        resolved = resolve_decision_model(
            DecisionModelConfig(name="laya", provider="local")
        )
        self.assertEqual(
            resolved.checkpoint, "convaiinnovations/laya-typed-decisions"
        )

    def test_laya_revision_is_forwarded_to_loader(self):
        revision = "989e28febe5886389496fe13a8036c62b2a20ae0"
        resolved = resolve_decision_model(
            DecisionModelConfig(
                name="laya",
                provider="local",
                revision=revision,
                device="cpu",
            )
        )

        with patch("routerlab.decision_models._load_laya", return_value=object()) as load:
            LayaAdapter(resolved)

        load.assert_called_once_with(
            "convaiinnovations/laya-typed-decisions",
            device="cpu",
            revision=revision,
        )

    def test_systemone_adapter_uses_canonical_state_and_actions(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update(
                url=url, payload=payload, headers=headers, timeout=timeout
            )
            return {
                "model": "jev-1.13.0",
                "answers": {
                    "next_action": {
                        "type": "choice",
                        "choice": "coder",
                        "probabilities": {"coder": 0.8, "STOP": 0.2},
                        "confidence": 0.7,
                    }
                },
                "usage": {
                    "input_tokens": 99,
                    "output_tokens": 0,
                    "cost": 0.000123,
                },
            }

        config = DecisionModelConfig(name="jev", provider="openrouter")
        resolved = resolve_decision_model(config)
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "secret"}):
            adapter = SystemOneHTTPAdapter(resolved, transport=transport)
            decision = adapter.decide(state().to_router_payload())

        self.assertEqual(decision.choice, "coder")
        self.assertEqual(decision.usage.input_tokens, 99)
        self.assertEqual(decision.usage.cost_usd, 0.000123)
        self.assertEqual(captured["payload"]["state"]["task_id"], "Mbpp/1")
        criteria = captured["payload"]["questions"]["next_action"]["criteria"]
        self.assertEqual(tuple(criteria), ROUTER_ACTIONS)
        self.assertEqual(criteria, ROUTER_CAPABILITY_CARDS)
        self.assertEqual(captured["payload"]["model"], "typesafe/jev-1.13")
        self.assertEqual(
            captured["url"], "https://openrouter.ai/api/alpha/decisions"
        )
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")

    def test_router_contract_treats_sparse_base_pass_as_incomplete_evidence(self):
        self.assertIn("incomplete evidence", QUESTION_INSTRUCTIONS)
        self.assertIn("not proof", QUESTION_INSTRUCTIONS)
        self.assertIn("whether Base Tests pass or fail", ACTION_DESCRIPTIONS["logic_reviewer"])
        self.assertIn("whether Base Tests pass or fail", ACTION_DESCRIPTIONS["edge_case_analyst"])
        self.assertNotIn("not proof", ACTION_DESCRIPTIONS["STOP"])

    def test_systemone_adapter_fails_before_run_when_api_key_is_missing(self):
        resolved = resolve_decision_model(
            DecisionModelConfig(name="jev", provider="openrouter")
        )
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                SystemOneHTTPAdapter(resolved)

    def test_openai_adapter_converts_choice_response(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update(payload=payload, headers=headers)
            return {
                "model": "gpt-6-luna",
                "answers": [
                    {
                        "type": "choice",
                        "name": "next_action",
                        "choice": "logic_reviewer",
                        "probabilities": [
                            {"value": "logic_reviewer", "probability": 0.75},
                            {"value": "coder", "probability": 0.25},
                        ],
                        "confidence": 0.6,
                    }
                ],
                "usage": {"input_tokens": 88, "output_tokens": 0},
            }

        resolved = resolve_decision_model(
            DecisionModelConfig(name="openai_decisions", provider="openai")
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "secret"}):
            adapter = OpenAIDecisionsAdapter(resolved, transport=transport)
            decision = adapter.decide(state().to_router_payload())

        self.assertEqual(decision.choice, "logic_reviewer")
        self.assertEqual(decision.probabilities["logic_reviewer"], 0.75)
        choices = captured["payload"]["questions"][0]["choices"]
        self.assertEqual(tuple(item["value"] for item in choices), ROUTER_ACTIONS)

    def test_luna_decisions_resolves_to_openrouter_decisions_api(self):
        resolved = resolve_decision_model(
            DecisionModelConfig(name="openai_decisions", provider="openrouter")
        )

        self.assertEqual(resolved.adapter, "systemone_http")
        self.assertEqual(resolved.checkpoint, "openai/gpt-6-luna-decisions")
        self.assertEqual(
            resolved.base_url, "https://openrouter.ai/api/alpha/decisions"
        )
        self.assertEqual(resolved.api_key_env, "OPENROUTER_API_KEY")

    def test_common_router_normalizes_invalid_choice_to_stop(self):
        class InvalidAdapter:
            def decide(self, payload):
                from routerlab.contracts import ModelUsage
                from routerlab.decision_models import AdapterDecision

                return AdapterDecision(
                    choice="unknown",
                    probabilities=None,
                    confidence=None,
                    usage=ModelUsage(input_tokens=1),
                    raw={"choice": "unknown"},
                )

        config = DecisionModelConfig(name="laya", provider="local")
        router = DecisionModelRouter(config, adapter=InvalidAdapter())
        decision = router.select(state())
        self.assertEqual(decision.action_id, "STOP")
        self.assertFalse(decision.valid)

    def test_generative_router_receives_the_same_capability_cards(self):
        captured = {}

        def respond(messages):
            captured["messages"] = messages
            return '{"action": "coder"}'

        router = ModelRouter(ReplayModelClient(respond))
        decision = router.select(state())
        payload = json.loads(captured["messages"][1]["content"])

        self.assertEqual(decision.action_id, "coder")
        self.assertEqual(payload["action_capabilities"], ROUTER_CAPABILITY_CARDS)
        self.assertEqual(tuple(payload["action_capabilities"]), ROUTER_ACTIONS)

    def test_config_loads_common_decision_model_mode(self):
        payload = """
run_name: test
output_dir: runs
dataset: {name: mbpp}
expert_model: {backend: replay, checkpoint: fake}
router_model: {name: jev, provider: openrouter}
budget: {}
workflow: {method: e3, router: decision_model}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(payload, encoding="utf-8")
            config = load_config(path)
        self.assertIsInstance(config.router_model, DecisionModelConfig)
        self.assertEqual(config.router_model.name, "jev")


if __name__ == "__main__":
    unittest.main()
