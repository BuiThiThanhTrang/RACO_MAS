import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.agent_info.global_info import GlobalInfo
from agent.register.persona_loader import load_teammate_specs
from agent.register.register import AgentRegister
from config.runtime import load_experiment_config
from inference.graph.action_graph import ActionGraph
from inference.graph.agent_graph import AgentGraph
from inference.policy.frozen_llm_planner import FrozenLLMPlannerPolicy
from model.query_manager import ModelQueryManager, StructuredResponseError
from inference.reasoning.reasoning import GraphReasoning
from model.model_config import model_registry
from model.model_utils import _request_payload
from role_aware.audit_trace import AuditTrace
from role_aware.route_experience import RouteExperienceEvent, RouteExperienceStore
from role_aware.routing_profile_store import StaticRoutingProfileStore
from role_aware.validation import validate_frozen_planner_pool


PROJECT_DIR = Path(__file__).resolve().parents[1]
MMLU_POOL = PROJECT_DIR / "personas" / "role_aware" / "mmlu_pro_qwen35_9b_pool.jsonl"
SRDD_POOL = PROJECT_DIR / "personas" / "role_aware" / "srdd_qwen35_9b_pool.jsonl"


class FrozenPlannerTests(unittest.TestCase):
    def _graph(self):
        specs = load_teammate_specs(MMLU_POOL)
        registry = AgentRegister()
        registry.load(specs)
        profiles = StaticRoutingProfileStore()
        profiles.initialize(specs)
        return AgentGraph(registry, profiles, allowed_tools=()), registry, profiles

    def test_new_configs_use_only_two_experience_modes(self):
        expected = {
            "frozen_mmlu_pro_naive.yaml": "none",
            "frozen_mmlu_pro_evolving.yaml": "route_outcome",
            "frozen_srdd_naive.yaml": "none",
            "frozen_srdd_evolving.yaml": "route_outcome",
        }
        for name, mode in expected.items():
            config = load_experiment_config(PROJECT_DIR / "config" / "experiments" / name)
            self.assertEqual(config.policy["type"], "frozen_llm_planner")
            self.assertEqual(
                config.policy["planner"]["model"], "gpt-6-luna-openrouter"
            )
            self.assertEqual(config.profiles.mode, "static_role_cards")
            self.assertFalse(config.profiles.update_enabled)
            self.assertFalse(config.profiles.include_uncertainty)
            self.assertEqual(config.experience.mode, mode)
            self.assertEqual(config.global_config["graph"], {"max_width": 3, "max_depth": 2})

    def test_luna_planner_uses_openrouter_structured_outputs(self):
        config = model_registry.get_model_config("gpt-6-luna-openrouter")
        self.assertIsNotNone(config)
        self.assertEqual(config.api_model_name, "openai/gpt-6-luna")
        self.assertEqual(config.provider, "openai_compatible")
        self.assertEqual(config.api_profile, "openrouter")
        self.assertTrue(config.structured_outputs)
        self.assertTrue(config.provider_require_parameters)
        self.assertEqual(config.reasoning_effort, "medium")
        self.assertEqual(config.reasoning_parameter, "reasoning")
        self.assertEqual(
            config.request_parameter_allowlist,
            ["reasoning", "max_tokens", "response_format"],
        )

    def test_luna_structured_request_requires_compatible_provider(self):
        manager = ModelQueryManager.__new__(ModelQueryManager)
        manager.registry = model_registry
        manager.clients = {"gpt-6-luna-openrouter": object()}
        manager.client_errors = {}
        schema = {
            "type": "object",
            "properties": {"stop": {"type": "boolean"}},
            "required": ["stop"],
            "additionalProperties": False,
        }
        with patch.object(
            sys.modules["model.query_manager"],
            "chat_completion_request",
            return_value=('{"stop": true}', 7),
        ) as request:
            payload, tokens = manager.query_structured(
                "gpt-6-luna-openrouter",
                [{"role": "user", "content": "route this task"}],
                schema,
            )
        self.assertEqual(payload, {"stop": True})
        self.assertEqual(tokens, 7)
        kwargs = request.call_args.kwargs
        self.assertEqual(kwargs["model"], "openai/gpt-6-luna")
        self.assertEqual(
            kwargs["model_config_dict"]["extra_body"],
            {
                "reasoning": {"effort": "medium"},
                "provider": {"require_parameters": True},
            },
        )
        self.assertNotIn("reasoning_effort", kwargs["model_config_dict"])
        self.assertEqual(
            kwargs["model_config_dict"]["response_format"]["type"],
            "json_schema",
        )
        request_payload = _request_payload(
            [{"role": "user", "content": "route this task"}],
            kwargs["model"],
            kwargs["model_config_dict"],
            SimpleNamespace(base_url="https://openrouter.ai/api/v1"),
            4096,
        )
        self.assertEqual(
            set(request_payload),
            {"model", "messages", "max_tokens", "stream", "response_format", "extra_body"},
        )
        for unsupported in (
            "temperature",
            "top_p",
            "n",
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "reasoning_effort",
        ):
            self.assertNotIn(unsupported, request_payload)

    def test_qwen_actor_uses_non_thinking_mode(self):
        config = model_registry.get_model_config("qwen-3.5-9b")
        request_config = {
            "temperature": config.temperature,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "logit_bias": {},
            "max_tokens": config.max_tokens,
        }
        ModelQueryManager._apply_model_request_options(config, request_config)
        payload = _request_payload(
            [{"role": "user", "content": "answer directly"}],
            config.api_model_name,
            request_config,
            SimpleNamespace(base_url="https://router.huggingface.co/v1"),
            4096,
        )
        self.assertEqual(
            payload["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertEqual(payload["temperature"], 0.1)

    def test_pools_are_homogeneous_and_have_no_stop_persona(self):
        for path, expected_count in ((MMLU_POOL, 7), (SRDD_POOL, 6)):
            specs = validate_frozen_planner_pool(load_teammate_specs(path))
            self.assertEqual(len(specs), expected_count)
            self.assertEqual(len({spec.backbone for spec in specs}), 1)
            self.assertNotIn("terminate", {spec.role_card.allowed_actions[0] for spec in specs})

    def test_static_planner_view_contains_no_identity_or_uncertainty(self):
        graph, registry, _ = self._graph()
        serialized = json.dumps(
            list(graph.public_agent_views()), sort_keys=True
        )
        for spec in registry.agent_config:
            self.assertNotIn(spec.teammate_id, serialized)
            self.assertNotIn(spec.backbone, serialized)
            self.assertNotIn(spec.provider_profile, serialized)
        self.assertNotIn("uncertainty", serialized)
        self.assertNotIn("observation_count", serialized)

    def test_planner_returns_structured_assignments(self):
        graph, registry, _ = self._graph()
        policy = FrozenLLMPlannerPolicy(
            graph,
            ActionGraph(allowed_tools=()),
            {"planner": {"model": "gpt-6.1-sol", "max_repair_attempts": 0}},
            {},
        )
        info = GlobalInfo(-1, ".", {"type": "MMLU-Pro", "Question": "Q"})
        info.path_uid = "root"
        info.remaining_depth = 2
        payload = {
            "stop": False,
            "task_signature": ["formal_quantitative"],
            "state_gaps": ["independent check"],
            "selections": [
                {"candidate_id": 0, "subtask": "derive", "expected_contribution": "solution"},
                {"candidate_id": 6, "subtask": "verify", "expected_contribution": "check"},
            ],
        }
        with patch(
            "inference.policy.frozen_llm_planner.query_manager.query_structured",
            return_value=(payload, 11),
        ):
            proposal = policy.propose(info, 2)
        self.assertEqual(
            proposal["actions"],
            [registry.agent_config[0].teammate_id, registry.agent_config[6].teammate_id],
        )
        self.assertEqual(proposal["assignments"][0]["subtask"], "derive")
        self.assertEqual(proposal["task_signature"], ["formal_quantitative"])

    def test_invalid_planner_output_uses_bounded_fallback(self):
        graph, registry, _ = self._graph()
        policy = FrozenLLMPlannerPolicy(
            graph,
            ActionGraph(allowed_tools=()),
            {"planner": {"model": "gpt-6.1-sol", "max_repair_attempts": 0}},
            {},
        )
        info = GlobalInfo(-1, ".", {"type": "MMLU-Pro", "Question": "Q"})
        info.path_uid = "root"
        with patch(
            "inference.policy.frozen_llm_planner.query_manager.query_structured",
            side_effect=StructuredResponseError("invalid"),
        ):
            proposal = policy.propose(info, 3)
        self.assertTrue(proposal["fallback"])
        self.assertEqual(len(proposal["actions"]), 1)
        self.assertEqual(proposal["actions"][0], registry.agent_config[0].teammate_id)

    def test_root_planner_can_choose_fewer_paths_than_max_width(self):
        graph, registry, _ = self._graph()
        policy = FrozenLLMPlannerPolicy(
            graph,
            ActionGraph(allowed_tools=()),
            {"planner": {"model": "gpt-6-luna-openrouter", "max_repair_attempts": 0}},
            {},
        )
        info = GlobalInfo(-1, ".", {"type": "MMLU-Pro", "Question": "Q"})
        info.path_uid = "root"
        info.remaining_depth = 2
        payload = {
            "stop": False,
            "task_signature": ["straightforward_domain_question"],
            "state_gaps": [],
            "selections": [
                {
                    "candidate_id": 5,
                    "subtask": "solve directly",
                    "expected_contribution": "one sufficient candidate",
                }
            ],
        }
        with patch(
            "inference.policy.frozen_llm_planner.query_manager.query_structured",
            return_value=(payload, 5),
        ):
            proposal = policy.propose(info, 3)
        self.assertEqual(proposal["actions"], [registry.agent_config[5].teammate_id])
        planner_input = policy._planner_input(info, 3)
        self.assertTrue(planner_input["constraints"]["width_is_upper_bound"])
        self.assertEqual(planner_input["constraints"]["maximum_selections"], 3)

    def test_provider_failure_does_not_silently_become_naive_fallback(self):
        graph, _, _ = self._graph()
        policy = FrozenLLMPlannerPolicy(
            graph,
            ActionGraph(allowed_tools=()),
            {"planner": {"model": "gpt-6.1-sol", "max_repair_attempts": 0}},
            {},
        )
        info = GlobalInfo(-1, ".", {"type": "MMLU-Pro", "Question": "Q"})
        info.path_uid = "root"
        with (
            patch(
                "inference.policy.frozen_llm_planner.query_manager.query_structured",
                side_effect=RuntimeError("provider unavailable"),
            ),
            self.assertRaises(RuntimeError),
        ):
            policy.propose(info, 3)

    def test_route_experience_is_task_level_and_committed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experience.json"
            store = RouteExperienceStore("route_outcome", path)
            event = RouteExperienceEvent(
                task_id="task-1",
                dataset="MMLU-Pro",
                task_signature=("formal_quantitative",),
                routes=(("Quantitative & Formal Reasoner", "Adversarial Verifier"),),
                success=True,
            )
            self.assertTrue(store.observe(event))
            record = store.snapshot(("formal_quantitative",))[0]
            self.assertEqual(record["attempts"], 1)
            self.assertEqual(record["successes"], 1)
            self.assertNotIn("uncertainty", record)
            with self.assertRaises(ValueError):
                store.observe(event)
            restored = RouteExperienceStore("route_outcome", path)
            self.assertEqual(restored.snapshot()[0]["success_rate"], 1.0)

    def test_naive_experience_store_never_mutates(self):
        store = RouteExperienceStore("none")
        event = RouteExperienceEvent("task", "MMLU-Pro", (), (("Role",),), False)
        self.assertFalse(store.observe(event))
        self.assertEqual(store.snapshot(), [])

    def test_w3d2_executes_three_two_actor_paths_and_hands_off_assignments(self):
        class SilentLogs:
            def __init__(self, *args, folder_path=None, **kwargs):
                self.folder_path = str(folder_path)
                self.logger = logging.getLogger("frozen-planner-test")
                self.logger.addHandler(logging.NullHandler())

            def create_logger(self, *args, **kwargs):
                return None

            def get_logger(self, *args):
                return self.logger

        graph, registry, _ = self._graph()
        prompts = []
        for agent in registry.ordered_agents:
            def query(messages, system_prompt=None):
                prompts.append(json.dumps(messages, ensure_ascii=False))
                return "FINAL ANSWER: A", 3
            agent.query_func = query
        policy = FrozenLLMPlannerPolicy(
            graph,
            ActionGraph(allowed_tools=()),
            {"planner": {"model": "gpt-6.1-sol", "max_repair_attempts": 0}},
            {},
        )
        root = {
            "stop": False,
            "task_signature": ["mixed_domain"],
            "state_gaps": [],
            "selections": [
                {"candidate_id": i, "subtask": f"first-{i}",
                 "expected_contribution": "independent candidate"}
                for i in range(3)
            ],
        }
        second = lambda i: {
            "stop": False,
            "task_signature": ["mixed_domain"],
            "state_gaps": ["verification"],
            "selections": [{"candidate_id": 6, "subtask": f"verify-{i}",
                            "expected_contribution": "checked answer"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            trace = AuditTrace(directory, "test", "q", "attempt", enabled=False)
            with (
                patch("inference.reasoning.reasoning.LogManager", SilentLogs),
                patch(
                    "inference.policy.frozen_llm_planner.query_manager.query_structured",
                    side_effect=[(root, 1), (second(0), 1), (second(1), 1), (second(2), 1)],
                ),
            ):
                reasoning = GraphReasoning(
                    {"id": "q", "type": "MMLU-Pro", "Question": "Question", "Answer": "A",
                     "choices": "ABCDEFGHIJ"},
                    graph,
                    policy,
                    ActionGraph(allowed_tools=()),
                    3,
                    2,
                    {"aggregation": {"mode": "majority"}, "audit_split": "test"},
                    registry,
                    audit=trace,
                )
                reasoning.start(None)
                answer, _ = reasoning.n_step(2)
        self.assertEqual(answer, "A")
        self.assertEqual(len(reasoning.reasoning_paths), 3)
        self.assertTrue(all(path.completed_steps == 2 for path in reasoning.reasoning_paths))
        self.assertTrue(any("Current orchestrator assignment" in prompt for prompt in prompts))
        self.assertTrue(all(len(path.agent_sequence) == 2 for path in reasoning.reasoning_paths))


if __name__ == "__main__":
    unittest.main()
