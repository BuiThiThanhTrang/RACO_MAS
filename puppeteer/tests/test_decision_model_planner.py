import json
import logging
import tempfile
import unittest
from pathlib import Path

import httpx

from agent.agent_info.global_info import GlobalInfo
from agent.register.persona_loader import load_teammate_specs
from agent.register.register import AgentRegister
from config.runtime import load_experiment_config
from inference.graph.action_graph import ActionGraph
from inference.graph.agent_graph import AgentGraph
from inference.policy.decision_model_client import (
    DecisionJudgment,
    DecisionModelTransportError,
    DecisionResult,
    SystemOneDecisionClient,
)
from inference.policy.decision_planner import (
    ORCHESTRATOR_STOP,
    DecisionPlannerPolicy,
)
from inference.policy.route_bundle import RouteBundleBuilder
from inference.reasoning.reasoning import GraphReasoning
from role_aware.audit_trace import AuditTrace
from role_aware.routing_profile_store import StaticRoutingProfileStore
from role_aware.validation import validate_decision_planner_pool


PROJECT_DIR = Path(__file__).resolve().parents[1]
MMLU_POOL = PROJECT_DIR / "personas" / "role_aware" / "mmlu_pro_qwen35_9b_pool.jsonl"
SRDD_POOL = PROJECT_DIR / "personas" / "role_aware" / "srdd_qwen35_9b_pool.jsonl"
MMLU_HETEROGENEOUS_POOL = (
    PROJECT_DIR / "personas" / "role_aware" / "mmlu_pro_heterogeneous_pool.jsonl"
)
PUPPETEER_BASELINE_NO_TRAIN_POOL = (
    PROJECT_DIR
    / "personas"
    / "role_aware"
    / "puppeteer_baseline_roles_no_train_pool.jsonl"
)


class FakeDecisionClient:
    provider = "fake"

    def __init__(self, choices):
        self.choices = iter(choices)
        self.calls = []

    def decide(self, state, questions, question_name):
        self.calls.append((state, questions, question_name))
        choice = next(self.choices)
        return DecisionResult(
            choice=choice,
            confidence=0.99,
            probabilities={choice: 0.99},
            input_tokens=17,
            model="fake-decision-model",
            model_version="test",
            latency_ms=1.5,
        )


class TransportFailureClient:
    provider = "fake"

    def decide(self, state, questions, question_name):
        raise DecisionModelTransportError("provider unavailable")


class DecisionPlannerTests(unittest.TestCase):
    def _graph(self, pool=MMLU_POOL, allowed_tools=()):
        specs = load_teammate_specs(pool)
        registry = AgentRegister()
        registry.load(specs)
        profiles = StaticRoutingProfileStore()
        profiles.initialize(specs)
        return AgentGraph(registry, profiles, allowed_tools=allowed_tools), registry

    @staticmethod
    def _policy(graph, client, **decision_overrides):
        decision_config = {
            "provider": "systemone",
            "model": "test-model",
            "base_url": "http://unused.invalid",
            "max_options": 255,
        }
        decision_config.update(decision_overrides)
        return DecisionPlannerPolicy(
            graph,
            ActionGraph(allowed_tools=()),
            {
                "decision": decision_config
            },
            {},
            decision_client=client,
        )

    def test_bundle_counts_match_w3_role_pools(self):
        mmlu_graph, _ = self._graph()
        srdd_graph, _ = self._graph(SRDD_POOL, allowed_tools=("run_python",))
        builder = RouteBundleBuilder(max_options=255)
        self.assertEqual(len(builder.root_options(mmlu_graph.public_agent_views(), 3)), 63)
        self.assertEqual(len(builder.root_options(srdd_graph.public_agent_views(), 3)), 41)

    def test_search_bing_enables_baseline_web_researcher_within_jev_limit(self):
        graph, _ = self._graph(
            PUPPETEER_BASELINE_NO_TRAIN_POOL,
            allowed_tools=("search_bing",),
        )
        available_views = [
            view for view in graph.public_agent_views() if view["available"]
        ]
        available_roles = {view["role_name"] for view in available_views}
        options = RouteBundleBuilder(max_options=255).root_options(available_views, 3)

        self.assertIn("Web Researcher", available_roles)
        self.assertEqual(len(available_views), 9)
        self.assertEqual(len(options), 129)

    def test_sequential_root_selection_supports_all_thirteen_agents(self):
        allowed_tools = (
            "read_file",
            "search_arxiv",
            "search_bing",
            "access_website",
            "run_python",
        )
        graph, registry = self._graph(
            PUPPETEER_BASELINE_NO_TRAIN_POOL,
            allowed_tools=allowed_tools,
        )
        client = FakeDecisionClient(
            ["candidate__0", "candidate__2", "finish_bundle"]
        )
        policy = self._policy(
            graph,
            client,
            root_selection_mode="sequential",
        )
        info = GlobalInfo(-1, ".", {"type": "MMLU-Pro", "Question": "Question"})
        info.path_uid = "root"

        proposal = policy.propose(info, 3)

        self.assertEqual(len(graph.public_agent_views()), 14)
        self.assertEqual(
            sum(bool(view["available"]) for view in graph.public_agent_views()),
            13,
        )
        self.assertEqual(
            proposal["actions"],
            [registry.agent_config[0].teammate_id, registry.agent_config[2].teammate_id],
        )
        self.assertEqual(proposal["planner_tokens"], 51)
        self.assertEqual(
            [question_name for _, _, question_name in client.calls],
            ["root_agent_1", "root_agent_2", "root_agent_3"],
        )
        option_counts = [
            len(questions[question_name]["criteria"])
            for _, questions, question_name in client.calls
        ]
        self.assertEqual(option_counts, [13, 13, 12])
        first_criteria = client.calls[0][1]["root_agent_1"]["criteria"]
        second_criteria = client.calls[1][1]["root_agent_2"]["criteria"]
        third_criteria = client.calls[2][1]["root_agent_3"]["criteria"]
        self.assertNotIn("finish_bundle", first_criteria)
        self.assertIn("finish_bundle", second_criteria)
        self.assertIn("finish_bundle", third_criteria)
        self.assertNotIn("candidate__0", second_criteria)
        self.assertNotIn("candidate__2", third_criteria)

    def test_sequential_root_can_use_full_width(self):
        allowed_tools = (
            "read_file",
            "search_arxiv",
            "search_bing",
            "access_website",
            "run_python",
        )
        graph, registry = self._graph(
            PUPPETEER_BASELINE_NO_TRAIN_POOL,
            allowed_tools=allowed_tools,
        )
        client = FakeDecisionClient(
            ["candidate__0", "candidate__1", "candidate__2"]
        )
        policy = self._policy(
            graph,
            client,
            root_selection_mode="sequential",
        )
        info = GlobalInfo(-1, ".", {"type": "MMLU-Pro", "Question": "Question"})
        info.path_uid = "root"

        proposal = policy.propose(info, 3)

        self.assertEqual(
            proposal["actions"],
            [
                registry.agent_config[0].teammate_id,
                registry.agent_config[1].teammate_id,
                registry.agent_config[2].teammate_id,
            ],
        )
        self.assertEqual(len(client.calls), 3)

    def test_decision_planner_accepts_heterogeneous_actor_pool(self):
        specs = validate_decision_planner_pool(
            load_teammate_specs(MMLU_HETEROGENEOUS_POOL)
        )
        self.assertEqual(len(specs), 7)
        self.assertEqual(
            {spec.backbone for spec in specs},
            {
                "qwen-3.5-9b",
                "llama-3.1-8b",
                "gemma-3-12b-it",
                "mistral-nemo-12b",
            },
        )
        self.assertNotIn(
            "terminate", {spec.role_card.allowed_actions[0] for spec in specs}
        )

    def test_baseline_derived_no_train_pool_preserves_roles_and_backbones(self):
        specs = validate_decision_planner_pool(
            load_teammate_specs(PUPPETEER_BASELINE_NO_TRAIN_POOL)
        )
        expected = [
            ("File Analyst", "qwen-3.5-4b"),
            ("Academic Researcher", "qwen-2.5-14b"),
            ("Web Researcher", "gemma-3-12b-it"),
            ("Website Reader", "qwen-2.5-7b"),
            ("Python Tool Agent", "mistralai/ministral-3b-2512"),
            ("Planner / Decomposer", "mistral-nemo-12b"),
            ("General Reasoner", "qwen-2.5-14b"),
            ("Critic / Verifier", "gemma-3-12b-it"),
            ("Reflector", "mistral-nemo-12b"),
            ("Problem Decomposer", "qwen-2.5-7b"),
            ("Summarizer", "qwen-3.5-4b"),
            ("Integrator / Concluder", "mistralai/ministral-3b-2512"),
            ("Modifier / Repair", "qwen-2.5-14b"),
            ("Stop Controller", "gemma-3-12b-it"),
        ]
        self.assertEqual(
            [(spec.role_card.role_name, spec.backbone) for spec in specs], expected
        )
        self.assertFalse(specs[-1].available)
        graph, _ = self._graph(PUPPETEER_BASELINE_NO_TRAIN_POOL)
        options = RouteBundleBuilder(max_options=255).root_options(
            graph.public_agent_views(), 3
        )
        self.assertEqual(len(options), 92)

    def test_baseline_pool_has_explicit_stage_aware_routing_profiles(self):
        records = [
            json.loads(line)
            for line in PUPPETEER_BASELINE_NO_TRAIN_POOL.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        self.assertTrue(records)
        self.assertTrue(all("routing_profile" in record for record in records))
        profiles = {
            record["role_card"]["role_name"]: record["routing_profile"]
            for record in records
        }

        self.assertIn(
            "at least two substantive candidate outputs or evidence items are available",
            profiles["Integrator / Concluder"]["use_when"],
        )
        self.assertIn(
            "no candidate answer exists yet",
            profiles["Critic / Verifier"]["avoid_when"],
        )
        self.assertIn(
            "a prior candidate exists and a concrete defect has already been identified",
            profiles["Modifier / Repair"]["use_when"],
        )
        self.assertIn(
            "no specific URL is available",
            profiles["Website Reader"]["avoid_when"],
        )

    def test_route_choices_expose_stage_constraints_to_decision_model(self):
        graph, _ = self._graph(
            PUPPETEER_BASELINE_NO_TRAIN_POOL,
            allowed_tools=(
                "read_file",
                "search_arxiv",
                "search_bing",
                "access_website",
                "run_python",
            ),
        )
        views = graph.public_agent_views()
        options = RouteBundleBuilder(max_options=255).sequential_root_options(
            views,
            selected_ids=(),
            allow_finish=False,
        )
        descriptions = {
            option.candidate_ids[0]: option.description for option in options
        }
        role_ids = {view["role_name"]: view["candidate_id"] for view in views}

        critic_description = descriptions[role_ids["Critic / Verifier"]]
        website_reader_description = descriptions[role_ids["Website Reader"]]
        self.assertIn("avoid when: no candidate answer exists yet", critic_description)
        self.assertIn(
            "avoid when: no specific URL is available", website_reader_description
        )
        self.assertIn("handoff requirements:", critic_description)

    def test_root_bundle_controls_adaptive_width_and_templates_assignments(self):
        graph, registry = self._graph()
        client = FakeDecisionClient(["route__0_2"])
        policy = self._policy(graph, client)
        info = GlobalInfo(
            -1,
            ".",
            {
                "type": "MMLU-Pro",
                "category": "physics",
                "Question": "Question",
            },
        )
        info.path_uid = "root"
        info.remaining_depth = 2
        proposal = policy.propose(info, 3)
        self.assertEqual(
            proposal["actions"],
            [registry.agent_config[0].teammate_id, registry.agent_config[2].teammate_id],
        )
        self.assertEqual(len(proposal["assignments"]), 2)
        self.assertIn("task_type:mmlu_pro", proposal["task_signature"])
        self.assertIn("category:physics", proposal["task_signature"])
        self.assertEqual(proposal["decision_model"]["confidence"], 0.99)
        state, questions, name = client.calls[0]
        self.assertEqual(name, "route_bundle")
        self.assertTrue(state["constraints"]["probabilities_are_audit_only"])
        self.assertEqual(len(questions[name]["criteria"]), 63)

    def test_existing_path_can_stop(self):
        graph, _ = self._graph()
        policy = self._policy(graph, FakeDecisionClient(["stop"]))
        info = GlobalInfo(0, ".", {"type": "MMLU-Pro", "Question": "Question"})
        info.path_uid = "path-1"
        info.remaining_depth = 1
        proposal = policy.propose(info, 2)
        self.assertEqual(proposal["actions"], [ORCHESTRATOR_STOP])
        self.assertEqual(proposal["assignments"], [None])

    def test_unknown_root_choice_falls_back_to_one_path(self):
        graph, registry = self._graph()
        policy = self._policy(graph, FakeDecisionClient(["not_an_option"]))
        info = GlobalInfo(-1, ".", {"type": "MMLU-Pro", "Question": "Question"})
        info.path_uid = "root"
        proposal = policy.propose(info, 3)
        self.assertTrue(proposal["fallback"])
        self.assertEqual(proposal["actions"], [registry.agent_config[0].teammate_id])

    def test_transport_failure_is_not_hidden_by_fallback(self):
        graph, _ = self._graph()
        policy = self._policy(graph, TransportFailureClient())
        info = GlobalInfo(-1, ".", {"type": "MMLU-Pro", "Question": "Question"})
        info.path_uid = "root"
        with self.assertRaises(DecisionModelTransportError):
            policy.propose(info, 3)

    def test_systemone_client_normalizes_typed_choice(self):
        captured = {}

        def handler(request):
            captured["request"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "model": "jev-1.13",
                    "model_version": "jev-1.13-test",
                    "answers": {
                        "route_bundle": {
                            "choice": "route__1_3",
                            "confidence": 0.8,
                            "probabilities": {
                                "route__1_3": 0.8,
                                "route__0": 0.2,
                            },
                        }
                    },
                    "usage": {"input_tokens": 31, "output_tokens": 0},
                },
            )

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        client = SystemOneDecisionClient(
            {
                "provider": "systemone",
                "model": "jev-1.13",
                "base_url": "https://example.test",
                "max_retries": 0,
            },
            http_client=http_client,
        )
        result = client.decide(
            {"task": "Q"},
            {
                "route_bundle": {
                    "type": "choice",
                    "criteria": {"route__1_3": "roles 1 and 3"},
                }
            },
            "route_bundle",
        )
        self.assertEqual(result.choice, "route__1_3")
        self.assertEqual(result.input_tokens, 31)
        self.assertEqual(result.model_version, "jev-1.13-test")
        self.assertEqual(captured["request"]["model"], "jev-1.13")

    def test_systemone_client_returns_all_typed_judgments(self):
        def handler(request):
            return httpx.Response(
                200,
                json={
                    "model": "jev-1.13",
                    "answers": {
                        "evidence_is_sufficient": {
                            "noul": "no",
                            "confidence": 0.9,
                        },
                        "progress_since_previous_step": {"score": 3},
                        "dominant_next_need": {"choice": "resolve_bridge"},
                    },
                    "usage": {"input_tokens": 29},
                },
            )

        client = SystemOneDecisionClient(
            {
                "provider": "systemone",
                "model": "jev-1.13",
                "base_url": "https://example.test",
                "max_retries": 0,
            },
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        judgment = client.judge(
            {"task": "Q"},
            {
                "evidence_is_sufficient": {"type": "noul"},
                "progress_since_previous_step": {"type": "score"},
                "dominant_next_need": {"type": "choice"},
            },
        )

        self.assertIsInstance(judgment, DecisionJudgment)
        self.assertEqual(judgment.input_tokens, 29)
        self.assertEqual(
            judgment.answers["dominant_next_need"]["choice"],
            "resolve_bridge",
        )

    def test_openrouter_client_uses_decisions_endpoint_and_headers(self):
        captured = {}

        def handler(request):
            captured["url"] = str(request.url)
            captured["authorization"] = request.headers["Authorization"]
            captured["referer"] = request.headers.get("HTTP-Referer")
            captured["title"] = request.headers.get("X-OpenRouter-Title")
            captured["request"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "model": "typesafe/jev-1.13-20260917",
                    "provider": "TypeSafe",
                    "answers": {
                        "next_action": {
                            "type": "choice",
                            "choice": "stop",
                            "confidence": 0.9,
                            "probabilities": {"stop": 0.9, "candidate__0": 0.1},
                        }
                    },
                    "usage": {"input_tokens": 23, "output_tokens": 4},
                },
            )

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        from unittest.mock import patch

        with patch.dict(
            "os.environ",
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_SITE_URL": "https://example.test",
                "OPENROUTER_APP_NAME": "RACO MAS",
            },
        ):
            client = SystemOneDecisionClient(
                {
                    "provider": "openrouter",
                    "model": "typesafe/jev-1.13",
                    "max_retries": 0,
                },
                http_client=http_client,
            )
            result = client.decide(
                {"task": "Q"},
                {
                    "next_action": {
                        "type": "choice",
                        "criteria": {"stop": "Stop now"},
                    }
                },
                "next_action",
            )

        self.assertEqual(captured["url"], "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(captured["authorization"], "Bearer test-key")
        self.assertEqual(captured["referer"], "https://example.test")
        self.assertEqual(captured["title"], "RACO MAS")
        self.assertEqual(captured["request"]["model"], "typesafe/jev-1.13")
        self.assertEqual(result.choice, "stop")
        self.assertEqual(result.input_tokens, 23)

    def test_openrouter_retries_529_with_exponential_backoff(self):
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) < 3:
                return httpx.Response(
                    529,
                    json={"error": {"code": 529, "message": "Provider returned error"}},
                )
            return httpx.Response(
                200,
                json={
                    "model": "cloudflare/clef",
                    "answers": {
                        "next_action": {
                            "type": "choice",
                            "choice": "stop",
                            "confidence": 1.0,
                            "probabilities": {"stop": 1.0},
                        }
                    },
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            )

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        from unittest.mock import call, patch

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), patch(
            "inference.policy.decision_model_client.time.sleep"
        ) as sleep:
            client = SystemOneDecisionClient(
                {
                    "provider": "openrouter",
                    "model": "cloudflare/clef",
                    "max_retries": 2,
                    "retry_backoff_seconds": 1,
                },
                http_client=http_client,
            )
            result = client.decide(
                {"task": "Q"},
                {
                    "next_action": {
                        "type": "choice",
                        "criteria": {"stop": "Stop now"},
                    }
                },
                "next_action",
            )

        self.assertEqual(result.choice, "stop")
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleep.call_args_list, [call(1.0), call(2.0)])

    def test_w3d2_pipeline_executes_decision_routes(self):
        class SilentLogs:
            def __init__(self, *args, folder_path=None, **kwargs):
                self.folder_path = str(folder_path)
                self.logger = logging.getLogger("decision-planner-test")
                self.logger.addHandler(logging.NullHandler())

            def create_logger(self, *args, **kwargs):
                return None

            def get_logger(self, *args):
                return self.logger

        graph, registry = self._graph()
        prompts = []
        for agent in registry.ordered_agents:
            def query(messages, system_prompt=None):
                prompts.append(json.dumps(messages, ensure_ascii=False))
                return "FINAL ANSWER: A", 3

            agent.query_func = query
        client = FakeDecisionClient(
            ["route__0_1_2", "candidate__6", "candidate__6", "candidate__6"]
        )
        policy = self._policy(graph, client)
        with tempfile.TemporaryDirectory() as directory:
            trace = AuditTrace(directory, "test", "q", "attempt", enabled=False)
            from unittest.mock import patch

            with patch("inference.reasoning.reasoning.LogManager", SilentLogs):
                reasoning = GraphReasoning(
                    {
                        "id": "q",
                        "type": "MMLU-Pro",
                        "category": "physics",
                        "Question": "Question",
                        "Answer": "A",
                        "choices": "ABCDEFGHIJ",
                    },
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
        self.assertTrue(
            all(path.completed_steps == 2 for path in reasoning.reasoning_paths)
        )
        self.assertTrue(
            any("Current orchestrator assignment" in prompt for prompt in prompts)
        )
        self.assertEqual(len(client.calls), 4)

    def test_all_decision_configs_are_static_w3d2(self):
        names = [
            f"decision_{dataset}_{experience}_{provider}.yaml"
            for dataset in ("mmlu_pro", "srdd")
            for experience in ("naive", "evolving")
            for provider in ("jev", "clef")
        ]
        for name in names:
            config = load_experiment_config(
                PROJECT_DIR / "config" / "experiments" / name
            )
            self.assertEqual(config.policy["type"], "decision_model_planner")
            self.assertEqual(config.policy["decision"]["provider"], "openrouter")
            self.assertEqual(
                config.policy["decision"]["api_key_env"], "OPENROUTER_API_KEY"
            )
            expected_model = (
                "typesafe/jev-1.13" if name.endswith("_jev.yaml") else "cloudflare/clef"
            )
            self.assertEqual(config.policy["decision"]["model"], expected_model)
            self.assertEqual(config.profiles.mode, "static_role_cards")
            self.assertFalse(config.profiles.update_enabled)
            self.assertFalse(config.profiles.include_uncertainty)
            self.assertEqual(
                config.experience.mode,
                "route_outcome" if "evolving" in name else "none",
            )
            self.assertEqual(
                config.global_config["graph"],
                {"max_width": 3, "max_depth": 2},
            )

        search_config = load_experiment_config(
            PROJECT_DIR
            / "config"
            / "experiments"
            / "decision_mmlu_pro_naive_jev.yaml"
        )
        self.assertEqual(
            search_config.tools.allowed,
            (
                "read_file",
                "search_arxiv",
                "search_bing",
                "access_website",
                "run_python",
            ),
        )
        self.assertEqual(
            search_config.policy["decision"]["root_selection_mode"],
            "sequential",
        )


if __name__ == "__main__":
    unittest.main()
