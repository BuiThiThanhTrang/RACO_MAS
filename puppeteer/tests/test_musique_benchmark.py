import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from agent.agent_info.global_info import GlobalInfo
from agent.agent_info.workflow import Action
from agent.register.persona_loader import load_teammate_specs
from agent.register.register import AgentRegister
from config.runtime import load_experiment_config
from inference.graph.action_graph import ActionGraph
from inference.graph.agent_graph import AgentGraph
from inference.policy.decision_model_client import DecisionJudgment, DecisionResult
from inference.policy.decision_planner import DecisionPlannerPolicy
from inference.policy.frozen_llm_planner import FrozenLLMPlannerPolicy
from inference.policy.stage_routing import StageRoutingGuard
from inference.reasoning.reasoning import GraphReasoning
from role_aware.aggregation import aggregate_musique_candidates
from role_aware.collaboration_metrics import evaluate_musique_collaboration
from role_aware.musique_answers import (
    canonicalize_musique_candidate,
    parse_musique_output,
)
from role_aware.routing_profile_store import StaticRoutingProfileStore
from tasks import musique
from tasks.evaluator import BenchmarkEvaluator


PROJECT_DIR = Path(__file__).resolve().parents[1]
SMALL_POOL = PROJECT_DIR / "personas" / "role_aware" / "musique_qwen35_9b_pool.jsonl"


def fixture_row():
    return {
        "id": "2hop__1_2",
        "paragraphs": [
            {
                "idx": 10,
                "title": "Green",
                "paragraph_text": "Green was performed by Steve Hillage.",
                "is_supporting": True,
            },
            {
                "idx": 5,
                "title": "Miquette Giraudy",
                "paragraph_text": "Miquette Giraudy is Steve Hillage's partner.",
                "is_supporting": True,
            },
            {
                "idx": 0,
                "title": "Distractor",
                "paragraph_text": "Unrelated text.",
                "is_supporting": False,
            },
        ],
        "question": "Who is the spouse of the Green performer?",
        "question_decomposition": [
            {
                "id": 1,
                "question": "Green >> performer",
                "answer": "Steve Hillage",
                "paragraph_support_idx": 10,
            },
            {
                "id": 2,
                "question": "#1 >> spouse",
                "answer": "Miquette Giraudy",
                "paragraph_support_idx": 5,
            },
        ],
        "answer": "Miquette Giraudy",
        "answer_aliases": [],
        "answerable": True,
    }


class MusiqueBenchmarkTests(unittest.TestCase):
    def test_loader_filters_hops_and_formats_context_without_gold_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = [fixture_row(), {**fixture_row(), "id": "unanswerable", "answerable": False}]
            pd.DataFrame(rows).to_parquet(root / "validation.parquet", index=False)
            loaded = musique.load_dataset(
                "dev", data_root=root, hop_count=2, data_limit=10
            )
            task = musique.format_question(loaded.iloc[0])

        self.assertEqual(len(loaded), 1)
        self.assertEqual(task["type"], "MuSiQue")
        self.assertEqual(task["hop_count"], 2)
        self.assertIn("[P10] Green", task["Question"])
        self.assertNotIn("is_supporting", task["Question"])
        self.assertEqual(task["supporting_paragraph_indices"], [10, 5])

    def test_gold_fields_are_not_exposed_to_agents_or_planner(self):
        task = musique.format_question(pd.Series(fixture_row()))
        reasoning = GraphReasoning.__new__(GraphReasoning)
        reasoning.task = task
        reasoning.workspace_path = "."
        reasoning.env = None
        reasoning.env_name = None
        reasoning.runtime_config = {"audit_split": "validation"}
        public = reasoning._new_info(0).task
        self.assertNotIn("Answer", public)
        self.assertNotIn("answer_aliases", public)
        self.assertNotIn("gold_decomposition", public)
        self.assertNotIn("supporting_paragraph_indices", public)

    def test_official_style_answer_and_support_scores(self):
        scores = BenchmarkEvaluator.musique_answer_scores(
            "FINAL ANSWER: The Miquette Giraudy!", ["Miquette Giraudy"]
        )
        self.assertEqual(scores["answer_em"], 1.0)
        self.assertEqual(scores["answer_f1"], 1.0)
        support = BenchmarkEvaluator.musique_support_scores({5, 10}, {5, 10})
        self.assertEqual(support["support_f1"], 1.0)
        self.assertEqual(
            BenchmarkEvaluator.extract_musique_support_ids(
                "Supporting paragraphs: P10, P5"
            ),
            {5, 10},
        )

    def test_collaboration_metrics_reward_state_advancing_handoffs(self):
        decomposition = fixture_row()["question_decomposition"]
        paths = [
            {
                "path_uid": "path-1",
                "steps": [
                    {
                        "agent": "Evidence Retriever",
                        "success": "Success",
                        "result": {
                            "step_data": "P10 states the performer is Steve Hillage.",
                            "answer": "",
                        },
                    },
                    {
                        "agent": "Bridge Entity Reasoner",
                        "success": "Success",
                        "result": {
                            "step_data": "P5 states the spouse is Miquette Giraudy.",
                            "answer": "Miquette Giraudy",
                        },
                    },
                    {
                        "agent": "Evidence Verifier",
                        "success": "Success",
                        "result": {
                            "step_data": "Verified against P10 and P5.",
                            "answer": "FINAL ANSWER: Miquette Giraudy",
                        },
                    },
                ],
            }
        ]
        metrics = evaluate_musique_collaboration(
            paths, decomposition, [10, 5], "Miquette Giraudy"
        )
        self.assertEqual(metrics["milestone_achievement_rate"], 1.0)
        self.assertEqual(metrics["useful_handoff_rate"], 1.0)
        self.assertEqual(metrics["collaboration_effectiveness"], 1.0)
        self.assertEqual(metrics["support_f1"], 1.0)
        self.assertEqual(metrics["contribution_by_role"]["Evidence Verifier"], 2)

    def test_paper_compatible_support_uses_only_selected_terminal_output(self):
        decomposition = fixture_row()["question_decomposition"]
        paths = [
            {
                "path_uid": "path-unselected",
                "steps": [
                    {
                        "agent": "Evidence Retriever",
                        "success": "Success",
                        "result": {"step_data": "P5 and P10 are relevant."},
                    }
                ],
            },
            {
                "path_uid": "path-selected",
                "steps": [
                    {
                        "agent": "Evidence Retriever",
                        "success": "Success",
                        "result": {"step_data": "P99 is a distractor."},
                    },
                    {
                        "agent": "Answer Integrator",
                        "success": "Success",
                        "result": {
                            "step_data": (
                                '{"status":"FINAL",'
                                '"supporting_paragraph_ids":[10,5],'
                                '"final_answer":"Miquette Giraudy"}'
                            )
                        },
                    },
                ],
            },
        ]

        metrics = evaluate_musique_collaboration(
            paths,
            decomposition,
            [10, 5],
            "Miquette Giraudy",
            selected_candidate_index=1,
        )

        self.assertEqual(metrics["paper_compatible_support_f1"], 1.0)
        self.assertEqual(
            metrics["paper_compatible_supporting_paragraphs"], [5, 10]
        )
        self.assertEqual(
            metrics["paper_compatible_support_source"]["path_uid"],
            "path-selected",
        )

    def test_paper_compatible_support_does_not_borrow_earlier_evidence(self):
        decomposition = fixture_row()["question_decomposition"]
        paths = [
            {
                "path_uid": "path-selected",
                "steps": [
                    {
                        "agent": "Evidence Retriever",
                        "success": "Success",
                        "result": {"step_data": "P5 and P10 are relevant."},
                    },
                    {
                        "agent": "Answer Integrator",
                        "success": "Success",
                        "result": {
                            "final_answer": "Miquette Giraudy",
                            "step_data": '{"status":"FINAL"}',
                        },
                    },
                ],
            }
        ]

        metrics = evaluate_musique_collaboration(
            paths,
            decomposition,
            [10, 5],
            "Miquette Giraudy",
            selected_candidate_index=0,
        )

        self.assertEqual(metrics["support_f1"], 1.0)
        self.assertEqual(metrics["paper_compatible_support_f1"], 0.0)
        self.assertEqual(metrics["paper_compatible_supporting_paragraphs"], [])

    def test_pool_and_paired_configs_change_only_actor_backbone(self):
        small = load_teammate_specs(SMALL_POOL)
        large = load_teammate_specs(
            PROJECT_DIR / "personas" / "role_aware" / "musique_gemini25_flash_pool.jsonl"
        )
        self.assertEqual(len(small), 6)
        self.assertEqual(len(large), 6)
        self.assertEqual({spec.backbone for spec in small}, {"qwen-3.5-9b"})
        self.assertEqual({spec.backbone for spec in large}, {"gemini-2.5-flash"})
        self.assertEqual(
            [spec.role_card.role_name for spec in small],
            [spec.role_card.role_name for spec in large],
        )

        qwen = load_experiment_config(
            PROJECT_DIR / "config" / "experiments" / "decision_musique_naive_jev_qwen.yaml"
        )
        gemini = load_experiment_config(
            PROJECT_DIR / "config" / "experiments" / "decision_musique_naive_jev_gemini.yaml"
        )
        self.assertEqual(qwen.dataset.name, "MuSiQue")
        self.assertEqual(qwen.global_config, gemini.global_config)
        self.assertEqual(qwen.policy, gemini.policy)
        self.assertEqual(qwen.global_config["graph"], {"max_width": 2, "max_depth": 5})

    def test_dynamic_jev_and_sol_configs_share_guard_and_hide_oracle_fields(self):
        jev = load_experiment_config(
            PROJECT_DIR
            / "config"
            / "experiments"
            / "decision_musique_dynamic_jev_gemini.yaml"
        )
        sol = load_experiment_config(
            PROJECT_DIR
            / "config"
            / "experiments"
            / "frozen_musique_dynamic_sol_gemini.yaml"
        )
        self.assertEqual(jev.policy["type"], "decision_model_planner")
        self.assertEqual(sol.policy["type"], "frozen_llm_planner")
        self.assertEqual(
            jev.policy["routing_guard"], sol.policy["routing_guard"]
        )
        self.assertEqual(
            jev.policy["routing_input"]["exclude_task_fields"],
            ["hop_count", "category"],
        )
        self.assertEqual(
            sol.policy["routing_input"]["exclude_task_fields"],
            ["hop_count", "category"],
        )
        self.assertEqual(
            jev.global_config["graph"], {"max_width": 2, "max_depth": 5}
        )
        self.assertEqual(jev.global_config["graph"], sol.global_config["graph"])

    def test_dynamic_sol_input_uses_same_filtered_task_and_feasible_pool(self):
        specs = load_teammate_specs(SMALL_POOL)
        registry = AgentRegister()
        registry.load(specs)
        profiles = StaticRoutingProfileStore()
        profiles.initialize(specs)
        graph = AgentGraph(registry, profiles, allowed_tools=())
        policy = FrozenLLMPlannerPolicy(
            graph,
            ActionGraph(allowed_tools=()),
            {
                "routing_input": {
                    "exclude_task_fields": ["hop_count", "category"]
                },
                "routing_guard": {
                    "enabled": True,
                    "profile": "musique_dynamic_v2",
                    "max_role_calls_per_path": 3,
                    "prevent_role_repeat_same_state": True,
                },
                "planner": {"model": "gpt-6-sol-openrouter"},
            },
            {},
        )
        info = GlobalInfo(
            -1,
            ".",
            {
                "type": "MuSiQue",
                "Question": "Question and paragraphs",
                "hop_count": 4,
                "category": "composition",
            },
        )
        planner_input = policy._planner_input(info, 2)

        self.assertNotIn("hop_count", planner_input["task"])
        self.assertNotIn("category", planner_input["task"])
        self.assertEqual(planner_input["routing_state"]["profile"], "musique_dynamic_v2")
        roles = {
            view["role_name"] for view in planner_input["candidate_profiles"]
        }
        self.assertNotIn("Evidence Verifier", roles)
        self.assertNotIn("Answer Integrator", roles)
        self.assertIn("Bridge Entity Reasoner", roles)
        self.assertIn("diagnose the semantic state", policy._system_prompt())

    def test_candidate_only_canonicalization_supports_structured_outputs(self):
        evidence_only = (
            '{"supporting_paragraph_ids":["P2"],'
            '"evidence_quotes":["quoted evidence"]}'
        )
        self.assertEqual(canonicalize_musique_candidate(evidence_only), "")

        verifier = parse_musique_output(
            "```json\n"
            '{"verdict":"correct","corrected_answer":"Eddie Argos (P2, P8)"}'
            "\n```"
        )
        self.assertFalse(verifier.has_final)
        self.assertEqual(verifier.candidate_answer, "Eddie Argos")

        integrator = parse_musique_output(
            '{"supporting_paragraph_ids":["P2","P8"],'
            '"final_answer":"Eddie Argos (P2, P8)"}'
        )
        self.assertTrue(integrator.has_final)
        self.assertEqual(integrator.final_answer, "Eddie Argos")
        self.assertEqual(
            canonicalize_musique_candidate("FINAL ANSWER: Eddie Argos [P2, P8]"),
            "Eddie Argos",
        )
        self.assertEqual(
            canonicalize_musique_candidate("FINAL ANSWER: It is Eddie Argos."),
            "Eddie Argos.",
        )

        verbose = parse_musique_output(
            "FINAL ANSWER: The company that makes Nirbhay, the Defence Research "
            "and Development Organisation, was established from the 1950s to the 1970s."
        )
        self.assertFalse(verbose.has_final)
        self.assertTrue(verbose.has_candidate)

        aggregation = aggregate_musique_candidates(
            [evidence_only, '{"corrected_answer":"Eddie Argos (P2, P8)"}']
        )
        self.assertEqual(aggregation["prediction"], "Eddie Argos")
        self.assertEqual(aggregation["selected_candidate_index"], 1)
        fenced_intermediate = parse_musique_output(
            'FINAL ANSWER: ```json\n{"intermediate_answer":"George"}\n```'
        )
        self.assertEqual(fenced_intermediate.final_answer, "")
        self.assertEqual(fenced_intermediate.candidate_answer, "")
        self.assertEqual(
            canonicalize_musique_candidate(
                'FINAL ANSWER: ```json\n{"intermediate_answer":"George"}\n```'
            ),
            "",
        )

    def test_verifier_without_final_answer_forces_answer_integrator(self):
        specs = load_teammate_specs(SMALL_POOL)
        registry = AgentRegister()
        registry.load(specs)
        profiles = StaticRoutingProfileStore()
        profiles.initialize(specs)
        graph = AgentGraph(registry, profiles, allowed_tools=())
        guard = StageRoutingGuard(
            {
                "enabled": True,
                "profile": "musique_stage_v1",
                "max_role_calls_per_path": 2,
                "prevent_role_repeat_same_state": True,
            }
        )
        by_id = {
            int(view["candidate_id"]): view["role_name"]
            for view in graph.public_agent_views()
        }
        task = {
            "type": "MuSiQue",
            "Question": "Question and candidate paragraphs",
            "hop_count": 2,
            "paragraph_ids": [2, 8],
        }
        path = GlobalInfo(0, ".", task)
        path.workflow.add_action(
            Action(
                {"action": "reasoning", "parameter": ""},
                {
                    "step_data": (
                        '{"candidate_answer":"Eddie Argos",'
                        '"supporting_paragraph_ids":["P2","P8"]}'
                    ),
                    "candidate_answer": "Eddie Argos",
                    "answer": "Eddie Argos",
                },
                "Success",
                "Comparison & Composition Reasoner",
                "qwen-3.5-9b",
            )
        )
        path.workflow.add_action(
            Action(
                {"action": "critique", "parameter": ""},
                {
                    "step_data": (
                        '{"verdict":"correct",'
                        '"corrected_answer":"Eddie Argos"}'
                    ),
                    "corrected_answer": "Eddie Argos",
                    "answer": "Eddie Argos",
                },
                "Success",
                "Evidence Verifier",
                "qwen-3.5-9b",
            )
        )
        snapshot = guard.evaluate(path, graph.public_agent_views())
        roles = {by_id[index] for index in snapshot.eligible_candidate_ids}
        self.assertFalse(snapshot.state["final_answer_exists"])
        self.assertTrue(snapshot.state["integrator_required"])
        self.assertEqual(roles, {"Answer Integrator"})

        path.workflow.add_action(
            Action(
                {"action": "conclude", "parameter": ""},
                {
                    "step_data": '{"final_answer":"Eddie Argos (P2, P8)"}',
                    "final_answer": "Eddie Argos",
                    "answer": "Eddie Argos",
                },
                "Success",
                "Answer Integrator",
                "qwen-3.5-9b",
            )
        )
        terminal = guard.evaluate(path, graph.public_agent_views())
        self.assertTrue(terminal.state["final_answer_exists"])
        self.assertEqual(terminal.eligible_candidate_ids, ())

    def test_runtime_llm_fallback_updates_router_state_and_is_cached(self):
        specs = load_teammate_specs(SMALL_POOL)
        registry = AgentRegister()
        registry.load(specs)
        profiles = StaticRoutingProfileStore()
        profiles.initialize(specs)
        graph = AgentGraph(registry, profiles, allowed_tools=())
        guard = StageRoutingGuard(
            {
                "enabled": True,
                "profile": "musique_dynamic_v2",
                "max_role_calls_per_path": 3,
                "answer_extraction": {
                    "enabled": True,
                    "model": "fake",
                    "roles": ["Evidence Verifier"],
                },
            }
        )
        task = {
            "type": "MuSiQue",
            "Question": "Question: What is the organization?",
            "paragraph_ids": [1],
        }
        path = GlobalInfo(0, ".", task)
        path.workflow.add_action(
            Action(
                {"action": "critique", "parameter": ""},
                {
                    "step_data": (
                        "The evidence is consistent. The organization is "
                        "International Tennis Federation (ITF)."
                    )
                },
                "Success",
                "Evidence Verifier",
                "qwen-3.5-9b",
            )
        )
        fake_result = (
            {
                "status": "FINAL",
                "answer": "International Tennis Federation (ITF)",
                "answer_quote": "International Tennis Federation (ITF)",
                "model": "fake",
                "tokens": 12,
            },
            12,
        )
        with patch(
            "inference.policy.stage_routing.extract_runtime_answer_with_llm",
            return_value=fake_result,
        ) as extractor:
            first = guard.evaluate(path, graph.public_agent_views())
            second = guard.evaluate(path, graph.public_agent_views())

        self.assertTrue(first.state["final_answer_exists"])
        self.assertEqual(
            first.state["final_answers"],
            ["International Tennis Federation (ITF)"],
        )
        self.assertEqual(second.state["final_answers"], first.state["final_answers"])
        extractor.assert_called_once()

    def test_decision_planner_does_not_offer_stop_before_integrator(self):
        specs = load_teammate_specs(SMALL_POOL)
        registry = AgentRegister()
        registry.load(specs)
        profiles = StaticRoutingProfileStore()
        profiles.initialize(specs)
        graph = AgentGraph(registry, profiles, allowed_tools=())
        integrator_id = next(
            int(view["candidate_id"])
            for view in graph.public_agent_views()
            if view["role_name"] == "Answer Integrator"
        )

        class Client:
            provider = "fake"

            def __init__(self):
                self.calls = []

            def decide(self, state, questions, question_name):
                self.calls.append((state, questions, question_name))
                choice = f"candidate__{integrator_id}"
                return DecisionResult(
                    choice=choice,
                    confidence=1.0,
                    probabilities={choice: 1.0},
                    input_tokens=1,
                    model="fake",
                    model_version="test",
                    latency_ms=0.0,
                )

        client = Client()
        policy = DecisionPlannerPolicy(
            graph,
            ActionGraph(allowed_tools=()),
            {
                "routing_guard": {
                    "enabled": True,
                    "profile": "musique_stage_v1",
                    "max_role_calls_per_path": 2,
                    "prevent_role_repeat_same_state": True,
                },
                "decision": {
                    "provider": "fake",
                    "model": "fake",
                    "max_options": 255,
                },
            },
            {},
            decision_client=client,
        )
        info = GlobalInfo(
            0,
            ".",
            {
                "type": "MuSiQue",
                "Question": "Question and candidate paragraphs",
                "hop_count": 2,
                "paragraph_ids": [2, 8],
            },
        )
        info.path_uid = "path-test"
        info.workflow.add_action(
            Action(
                {"action": "reasoning", "parameter": ""},
                {
                    "step_data": '{"candidate_answer":"Eddie Argos"}',
                    "candidate_answer": "Eddie Argos",
                },
                "Success",
                "Comparison & Composition Reasoner",
                "qwen-3.5-9b",
            )
        )
        info.workflow.add_action(
            Action(
                {"action": "critique", "parameter": ""},
                {
                    "step_data": (
                        '{"verdict":"correct",'
                        '"corrected_answer":"Eddie Argos"}'
                    ),
                    "corrected_answer": "Eddie Argos",
                },
                "Success",
                "Evidence Verifier",
                "qwen-3.5-9b",
            )
        )

        proposal = policy.propose(info, 1)

        state, questions, question_name = client.calls[0]
        criteria = questions[question_name]["criteria"]
        self.assertFalse(state["constraints"]["stop_allowed"])
        self.assertNotIn("stop", criteria)
        self.assertEqual(list(criteria), [f"candidate__{integrator_id}"])
        self.assertEqual(
            proposal["actions"], [registry.agent_config[integrator_id].teammate_id]
        )

    def test_musique_guard_masks_roles_by_state(self):
        specs = load_teammate_specs(SMALL_POOL)
        registry = AgentRegister()
        registry.load(specs)
        profiles = StaticRoutingProfileStore()
        profiles.initialize(specs)
        graph = AgentGraph(registry, profiles, allowed_tools=())
        guard = StageRoutingGuard(
            {
                "enabled": True,
                "profile": "musique_stage_v1",
                "max_role_calls_per_path": 2,
                "prevent_role_repeat_same_state": True,
            }
        )
        by_id = {
            int(view["candidate_id"]): view["role_name"]
            for view in graph.public_agent_views()
        }
        task = {
            "type": "MuSiQue",
            "Question": "Question and candidate paragraphs",
            "hop_count": 2,
            "paragraph_ids": [0, 5, 10],
        }
        root = GlobalInfo(-1, ".", task)
        snapshot = guard.evaluate(root, graph.public_agent_views())
        root_roles = {by_id[index] for index in snapshot.eligible_candidate_ids}
        self.assertEqual(
            root_roles,
            {
                "Task Decomposer",
                "Evidence Retriever",
                "Comparison & Composition Reasoner",
            },
        )

        path = GlobalInfo(0, ".", task)
        path.workflow.add_action(
            Action(
                {"action": "reasoning", "parameter": ""},
                {"step_data": "P10 says Steve Hillage is the performer.", "answer": ""},
                "Success",
                "Evidence Retriever",
                "qwen-3.5-9b",
            )
        )
        snapshot = guard.evaluate(path, graph.public_agent_views())
        path_roles = {by_id[index] for index in snapshot.eligible_candidate_ids}
        self.assertIn("Bridge Entity Reasoner", path_roles)
        self.assertNotIn("Evidence Verifier", path_roles)

    def test_dynamic_guard_masks_only_feasibility_and_no_progress(self):
        specs = load_teammate_specs(SMALL_POOL)
        registry = AgentRegister()
        registry.load(specs)
        profiles = StaticRoutingProfileStore()
        profiles.initialize(specs)
        graph = AgentGraph(registry, profiles, allowed_tools=())
        guard = StageRoutingGuard(
            {
                "enabled": True,
                "profile": "musique_dynamic_v2",
                "max_role_calls_per_path": 3,
                "prevent_role_repeat_same_state": True,
            }
        )
        by_id = {
            int(view["candidate_id"]): view["role_name"]
            for view in graph.public_agent_views()
        }
        task = {
            "type": "MuSiQue",
            "Question": "Question and candidate paragraphs",
            "hop_count": 4,
            "category": "composition",
            "paragraph_ids": [2, 8],
        }
        root = guard.evaluate(GlobalInfo(-1, ".", task), graph.public_agent_views())
        root_roles = {by_id[index] for index in root.eligible_candidate_ids}
        self.assertEqual(
            root_roles,
            {
                "Task Decomposer",
                "Evidence Retriever",
                "Bridge Entity Reasoner",
                "Comparison & Composition Reasoner",
            },
        )
        self.assertNotIn("hop_count", root.state)

        path = GlobalInfo(0, ".", task)
        path.workflow.add_action(
            Action(
                {"action": "reasoning", "parameter": ""},
                {"step_data": "P2 identifies Voyager 2.", "answer": ""},
                "Success",
                "Evidence Retriever",
                "qwen-3.5-9b",
            )
        )
        progressed = guard.evaluate(path, graph.public_agent_views())
        progressed_roles = {
            by_id[index] for index in progressed.eligible_candidate_ids
        }
        self.assertIn("Evidence Retriever", progressed_roles)
        self.assertIn("Task Decomposer", progressed_roles)
        self.assertNotIn("Evidence Verifier", progressed_roles)

        path.workflow.add_action(
            Action(
                {"action": "reasoning", "parameter": ""},
                {"step_data": "P2 again identifies Voyager 2.", "answer": ""},
                "Success",
                "Evidence Retriever",
                "qwen-3.5-9b",
            )
        )
        stalled = guard.evaluate(path, graph.public_agent_views())
        stalled_roles = {by_id[index] for index in stalled.eligible_candidate_ids}
        self.assertNotIn("Evidence Retriever", stalled_roles)
        self.assertIn("Bridge Entity Reasoner", stalled_roles)

    def test_dynamic_jev_uses_semantic_diagnosis_before_role_choice(self):
        specs = load_teammate_specs(SMALL_POOL)
        registry = AgentRegister()
        registry.load(specs)
        profiles = StaticRoutingProfileStore()
        profiles.initialize(specs)
        graph = AgentGraph(registry, profiles, allowed_tools=())
        bridge_id = next(
            int(view["candidate_id"])
            for view in graph.public_agent_views()
            if view["role_name"] == "Bridge Entity Reasoner"
        )

        class Client:
            provider = "fake"

            def __init__(self):
                self.calls = []

            def judge(self, state, questions):
                self.calls.append(("judge", state, questions))
                return DecisionJudgment(
                    answers={
                        "answer_addresses_original_question": {"noul": "no"},
                        "evidence_is_sufficient": {"noul": "no"},
                        "planned_dependencies_resolved": {"noul": "no"},
                        "candidate_conflict_present": {"noul": "no"},
                        "progress_since_previous_step": {"score": 3},
                        "dominant_next_need": {"choice": "resolve_bridge"},
                    },
                    input_tokens=13,
                    model="fake",
                    model_version="test",
                    latency_ms=1.0,
                )

            def decide(self, state, questions, question_name):
                self.calls.append(("decide", state, questions))
                choice = f"candidate__{bridge_id}"
                return DecisionResult(
                    choice=choice,
                    confidence=1.0,
                    probabilities={choice: 1.0},
                    input_tokens=17,
                    model="fake",
                    model_version="test",
                    latency_ms=1.0,
                )

        client = Client()
        policy = DecisionPlannerPolicy(
            graph,
            ActionGraph(allowed_tools=()),
            {
                "routing_input": {
                    "exclude_task_fields": ["hop_count", "category"]
                },
                "routing_guard": {
                    "enabled": True,
                    "profile": "musique_dynamic_v2",
                    "max_role_calls_per_path": 3,
                    "prevent_role_repeat_same_state": True,
                },
                "decision": {
                    "provider": "fake",
                    "model": "fake",
                    "semantic_diagnosis": {"enabled": True},
                },
            },
            {},
            decision_client=client,
        )
        info = GlobalInfo(
            0,
            ".",
            {
                "type": "MuSiQue",
                "Question": "Who is related to the entity in P2?",
                "hop_count": 4,
                "category": "composition",
            },
        )
        info.path_uid = "path-dynamic"
        info.workflow.add_action(
            Action(
                {"action": "reasoning", "parameter": ""},
                {"step_data": "P2 identifies the bridge entity.", "answer": ""},
                "Success",
                "Evidence Retriever",
                "qwen-3.5-9b",
            )
        )

        proposal = policy.propose(info, 1)

        self.assertEqual([call[0] for call in client.calls], ["judge", "decide"])
        route_state = client.calls[1][1]
        self.assertNotIn("hop_count", route_state["task"])
        self.assertNotIn("category", route_state["task"])
        self.assertEqual(
            route_state["semantic_diagnostics"]["dominant_next_need"]["choice"],
            "resolve_bridge",
        )
        self.assertEqual(proposal["planner_tokens"], 30)
        self.assertEqual(
            proposal["actions"], [registry.agent_config[bridge_id].teammate_id]
        )


if __name__ == "__main__":
    unittest.main()
