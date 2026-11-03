import unittest
from pathlib import Path
from unittest.mock import patch

from agent.agent_info.global_info import GlobalInfo
from agent.agent_info.workflow import Action
from agent.register.persona_loader import load_teammate_specs
from agent.register.register import AgentRegister
from config.runtime import load_experiment_config
from inference.graph.action_graph import ActionGraph
from inference.graph.agent_graph import AgentGraph
from inference.policy.decision_model_client import DecisionResult
from inference.policy.decision_planner import DecisionPlannerPolicy
from inference.policy.frozen_llm_planner import FrozenLLMPlannerPolicy
from inference.policy.stage_routing import StageRoutingGuard
from model.model_config import model_registry
from role_aware.routing_profile_store import StaticRoutingProfileStore


PROJECT_DIR = Path(__file__).resolve().parents[1]
GAIA_POOL = PROJECT_DIR / "personas" / "role_aware" / "gaia_qwen35_9b_pool.jsonl"


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
            confidence=1.0,
            probabilities={choice: 1.0},
            input_tokens=1,
            model="fake",
            model_version="test",
            latency_ms=1.0,
        )


class GaiaStageRoutingTests(unittest.TestCase):
    def setUp(self):
        specs = load_teammate_specs(GAIA_POOL)
        self.registry = AgentRegister()
        self.registry.load(specs)
        profiles = StaticRoutingProfileStore()
        profiles.initialize(specs)
        self.graph = AgentGraph(
            self.registry,
            profiles,
            allowed_tools=(
                "read_file",
                "inspect_media",
                "inspect_spreadsheet",
                "search_web",
                "access_website",
                "run_python",
            ),
        )
        self.guard = StageRoutingGuard(
            {
                "enabled": True,
                "profile": "gaia_stage_v2",
                "max_role_calls_per_path": 3,
                "prevent_role_repeat_same_state": True,
            }
        )
        self.role_ids = {
            view["role_name"]: int(view["candidate_id"])
            for view in self.graph.public_agent_views()
        }

    @staticmethod
    def _info(question="Who wrote the cited article?", *, path_id=-1, **task_fields):
        task = {
            "type": "GAIA",
            "Question": question,
            "level": 1,
            "has_attachment": False,
            **task_fields,
        }
        info = GlobalInfo(path_id, ".", task)
        info.path_uid = "root" if path_id == -1 else "path-test"
        info.remaining_depth = 4
        return info

    @staticmethod
    def _append_action(info, role, action_name, result, success="Success"):
        info.workflow.add_action(
            Action(
                {"action": action_name, "parameter": "test"},
                result,
                success,
                role,
                "qwen-3.5-9b",
            )
        )

    def _eligible_roles(self, info):
        snapshot = self.guard.evaluate(info, self.graph.public_agent_views())
        by_id = {
            int(view["candidate_id"]): view["role_name"]
            for view in self.graph.public_agent_views()
        }
        return {by_id[index] for index in snapshot.eligible_candidate_ids}, snapshot

    def test_level_one_root_masks_stage_invalid_roles(self):
        roles, snapshot = self._eligible_roles(self._info())
        self.assertEqual(roles, {"Web Researcher", "General Evidence Solver"})
        self.assertFalse(snapshot.state["has_url"])
        reasons = {
            item["role_name"]: item["reason"] for item in snapshot.masked_candidates
        }
        self.assertEqual(reasons["Website Reader"], "requires_known_url")
        self.assertEqual(
            reasons["Evidence Verifier"], "role_not_valid_at_root"
        )

    def test_explicit_url_enables_website_reader_at_root(self):
        roles, snapshot = self._eligible_roles(
            self._info("Read https://example.com/report and name its author.")
        )
        self.assertIn("Website Reader", roles)
        self.assertTrue(snapshot.state["question_has_url"])

    def test_search_output_enables_reader_and_allows_new_search_state(self):
        info = self._info(path_id=0)
        self._append_action(
            info,
            "Web Researcher",
            "search_web",
            {
                "step_data": "Result: [Source](https://example.com/report)",
                "answer": "",
            },
        )
        roles, snapshot = self._eligible_roles(info)
        self.assertIn("Web Researcher", roles)
        self.assertIn("Website Reader", roles)
        self.assertIn("General Evidence Solver", roles)
        self.assertTrue(snapshot.state["web_search_completed"])
        self.assertEqual(snapshot.state["evidence_count"], 1)
        self.assertTrue(snapshot.state["role_state_changed"]["Web Researcher"])

    def test_failed_search_blocks_same_role_on_unchanged_state(self):
        info = self._info(path_id=0)
        self._append_action(
            info,
            "Web Researcher",
            "search_web",
            {"step_data": "Tool error: provider unavailable", "answer": ""},
            success="Failure",
        )
        roles, snapshot = self._eligible_roles(info)
        self.assertNotIn("Web Researcher", roles)
        reasons = {
            item["role_name"]: item["reason"]
            for item in snapshot.masked_candidates
        }
        self.assertEqual(
            reasons["Web Researcher"], "role_already_called_for_current_state"
        )
        self.assertFalse(snapshot.state["role_state_changed"]["Web Researcher"])
        self.assertIn("Recovery Strategist", roles)

    def test_refusal_is_not_a_substantive_state_change(self):
        info = self._info(path_id=0)
        self._append_action(
            info,
            "General Evidence Solver",
            "reasoning",
            {
                "step_data": "I cannot provide the answer because I do not have access to the file.",
                "answer": "",
            },
        )
        roles, snapshot = self._eligible_roles(info)
        self.assertNotIn("General Evidence Solver", roles)
        self.assertIn("Recovery Strategist", roles)
        self.assertEqual(snapshot.state["evidence_count"], 0)
        self.assertEqual(snapshot.state["last_error_type"], "non_substantive_output")

    def test_consecutive_different_searches_each_create_a_new_state(self):
        info = self._info(path_id=0)
        self._append_action(
            info,
            "Web Researcher",
            "search_web",
            {"step_data": "First evidence https://example.com/a", "answer": ""},
        )
        info.workflow.workflow[-1].action["parameter"] = "first query"
        roles, first_snapshot = self._eligible_roles(info)
        self.assertIn("Web Researcher", roles)

        self._append_action(
            info,
            "Web Researcher",
            "search_web",
            {"step_data": "Second evidence https://example.com/b", "answer": ""},
        )
        info.workflow.workflow[-1].action["parameter"] = "different query"
        roles, second_snapshot = self._eligible_roles(info)
        self.assertIn("Web Researcher", roles)
        self.assertNotEqual(
            first_snapshot.state["state_fingerprint"],
            second_snapshot.state["state_fingerprint"],
        )
        self.assertEqual(
            second_snapshot.state["used_search_queries"],
            ["first query", "different query"],
        )

    def test_audio_attachment_routes_to_media_not_generic_file_or_python(self):
        roles, snapshot = self._eligible_roles(
            self._info(
                "List the ingredients spoken in the recording.",
                has_attachment=True,
                file_name="GAIA/2023/validation/sample.mp3",
            )
        )
        self.assertIn("Media Analyst", roles)
        self.assertNotIn("File Analyst", roles)
        self.assertNotIn("Python Data Analyst", roles)
        self.assertEqual(snapshot.state["attachment_extension"], ".mp3")

    def test_complete_media_evidence_routes_to_solver_without_repeating_media(self):
        info = self._info(
            "Read the board and give the winning move.",
            path_id=0,
            has_attachment=True,
            file_name="GAIA/2023/validation/board.png",
        )
        self._append_action(
            info,
            "Media Analyst",
            "inspect_media",
            {
                "step_data": (
                    'Structured visual evidence: {"complete": true, '
                    '"observations": ["black rook d8"], '
                    '"candidate_answer": "Rd5"}'
                ),
                "answer": "",
            },
        )
        roles, snapshot = self._eligible_roles(info)
        self.assertTrue(snapshot.state["media_inspection_completed"])
        self.assertFalse(snapshot.state["candidate_answer_exists"])
        self.assertNotIn("Media Analyst", roles)
        self.assertNotIn("Evidence Verifier", roles)
        self.assertIn("General Evidence Solver", roles)
        reasons = {
            item["role_name"]: item["reason"] for item in snapshot.masked_candidates
        }
        self.assertEqual(reasons["Media Analyst"], "media_already_inspected")

    def test_frozen_planner_disallows_stop_after_media_retrieval_without_candidate(self):
        info = self._info(
            "Read the board and give the winning move.",
            path_id=0,
            has_attachment=True,
            file_name="GAIA/2023/validation/board.png",
        )
        self._append_action(
            info,
            "Media Analyst",
            "inspect_media",
            {
                "step_data": (
                    'Structured visual evidence: {"complete": true, '
                    '"observations": ["black rook d8"]}'
                ),
                "answer": "",
            },
        )
        policy = FrozenLLMPlannerPolicy(
            self.graph,
            ActionGraph(allowed_tools=()),
            {
                "routing_guard": {
                    "enabled": True,
                    "profile": "gaia_stage_v2",
                },
                "planner": {"model": "gpt-6-sol-openrouter"},
            },
            {},
        )
        snapshot = policy.routing_guard.evaluate(
            info, self.graph.public_agent_views()
        )
        planner_input = policy._planner_input(info, 1, snapshot)

        self.assertFalse(planner_input["constraints"]["stop_allowed"])

    def test_xlsx_attachment_routes_to_spreadsheet_and_python(self):
        roles, snapshot = self._eligible_roles(
            self._info(
                "Which spreadsheet cell has the greatest value?",
                has_attachment=True,
                file_name="GAIA/2023/validation/sample.xlsx",
            )
        )
        self.assertIn("Spreadsheet Analyst", roles)
        self.assertIn("Python Data Analyst", roles)
        self.assertNotIn("File Analyst", roles)
        self.assertEqual(snapshot.state["attachment_extension"], ".xlsx")

    def test_candidate_routes_only_to_verification_or_integration(self):
        info = self._info(path_id=0)
        self._append_action(
            info,
            "Web Researcher",
            "search_web",
            {
                "step_data": "Evidence at https://example.com/report",
                "answer": "",
            },
        )
        self._append_action(
            info,
            "General Evidence Solver",
            "reasoning",
            {"step_data": "Supported conclusion", "answer": "CUB"},
        )
        info.answers.append("CUB")
        roles, snapshot = self._eligible_roles(info)
        self.assertEqual(roles, {"Evidence Verifier", "Answer Integrator"})
        self.assertTrue(snapshot.state["candidate_answer_exists"])

    def test_verified_answer_forces_deterministic_stop(self):
        info = self._info(path_id=0)
        self._append_action(
            info,
            "General Evidence Solver",
            "reasoning",
            {"step_data": "Candidate", "answer": "CUB"},
        )
        self._append_action(
            info,
            "Evidence Verifier",
            "critique",
            {"step_data": "Verified", "answer": "CUB"},
        )
        info.answers.extend(["CUB", "CUB"])
        roles, snapshot = self._eligible_roles(info)
        self.assertEqual(roles, set())
        self.assertTrue(
            all(
                item["reason"] in {"terminal_answer_ready", "role_call_limit_reached"}
                for item in snapshot.masked_candidates
            )
        )

    def test_http_404_is_exposed_as_failure_and_enables_recovery(self):
        info = self._info(
            "Read https://example.com/missing and identify the title.", path_id=0
        )
        self._append_action(
            info,
            "Website Reader",
            "access_website",
            {"step_data": "## Error 404\nPage Not Found", "answer": ""},
        )
        roles, snapshot = self._eligible_roles(info)
        self.assertEqual(snapshot.state["last_action_status"], "failure")
        self.assertEqual(snapshot.state["last_error_type"], "http_404")
        self.assertIn("Recovery Strategist", roles)
        self.assertNotIn("Website Reader", roles)

    def test_recovery_output_cannot_create_a_candidate_answer(self):
        info = self._info(path_id=0)
        self._append_action(
            info,
            "Web Researcher",
            "search_web",
            {"step_data": "Tool error: provider unavailable", "answer": ""},
            success="Failure",
        )
        self._append_action(
            info,
            "Recovery Strategist",
            "reflect",
            {
                "step_data": "Try a direct evidence-based solver next.",
                "answer": "UNSUPPORTED GUESS",
            },
        )
        info.answers.append("UNSUPPORTED GUESS")
        roles, snapshot = self._eligible_roles(info)
        self.assertFalse(snapshot.state["candidate_answer_exists"])
        self.assertNotIn("Evidence Verifier", roles)
        self.assertIn("General Evidence Solver", roles)

    def test_jev_only_receives_root_eligible_candidates(self):
        client = FakeDecisionClient(
            [f"candidate__{self.role_ids['General Evidence Solver']}", "finish_bundle"]
        )
        policy = DecisionPlannerPolicy(
            self.graph,
            ActionGraph(allowed_tools=()),
            {
                "routing_guard": {"enabled": True},
                "decision": {
                    "provider": "systemone",
                    "model": "test",
                    "root_selection_mode": "sequential",
                },
            },
            {},
            decision_client=client,
        )
        proposal = policy.propose(self._info(), 2)
        criteria = client.calls[0][1]["root_agent_1"]["criteria"]
        self.assertEqual(
            set(criteria),
            {
                f"candidate__{self.role_ids['Web Researcher']}",
                f"candidate__{self.role_ids['General Evidence Solver']}",
            },
        )
        self.assertEqual(len(proposal["actions"]), 1)

    def test_sol_rejects_masked_candidate_and_falls_back_within_mask(self):
        policy = FrozenLLMPlannerPolicy(
            self.graph,
            ActionGraph(allowed_tools=()),
            {
                "routing_guard": {"enabled": True},
                "planner": {
                    "model": "gpt-6-sol-openrouter",
                    "max_repair_attempts": 0,
                },
            },
            {},
        )
        payload = {
            "stop": False,
            "task_signature": ["gaia_level_1"],
            "state_gaps": [],
            "selections": [
                {
                    "candidate_id": self.role_ids["Website Reader"],
                    "subtask": "open a page",
                    "expected_contribution": "page evidence",
                }
            ],
        }
        with patch(
            "inference.policy.frozen_llm_planner.query_manager.query_structured",
            return_value=(payload, 1),
        ):
            proposal = policy.propose(self._info(), 2)
        selected_id = self.registry.get_agent_from_idx(proposal["actions"][0]).hash
        self.assertNotEqual(selected_id, "gaia-website-reader")
        self.assertTrue(proposal["fallback"])

    def test_paired_configs_share_pool_budget_and_aggregation(self):
        jev = load_experiment_config(
            PROJECT_DIR / "config" / "experiments" / "decision_gaia_naive_jev.yaml"
        )
        sol = load_experiment_config(
            PROJECT_DIR / "config" / "experiments" / "frozen_gaia_naive_sol.yaml"
        )
        self.assertEqual(jev.personas_path, sol.personas_path)
        self.assertEqual(jev.tools.allowed, sol.tools.allowed)
        self.assertEqual(jev.global_config, sol.global_config)
        self.assertEqual(jev.experience.mode, "none")
        self.assertEqual(sol.experience.mode, "none")
        self.assertTrue(jev.policy["routing_guard"]["enabled"])
        self.assertTrue(sol.policy["routing_guard"]["enabled"])
        self.assertEqual(
            jev.policy["decision"]["root_selection_mode"], "flat_bundle"
        )
        self.assertEqual(jev.global_config["graph"], {"max_width": 2, "max_depth": 4})

    def test_sol_openrouter_registry_uses_low_effort_structured_output(self):
        config = model_registry.get_model_config("gpt-6-sol-openrouter")
        self.assertEqual(config.api_model_name, "openai/gpt-6-sol")
        self.assertEqual(config.api_profile, "openrouter")
        self.assertTrue(config.structured_outputs)
        self.assertEqual(config.reasoning_effort, "low")
        self.assertEqual(config.reasoning_parameter, "reasoning")


if __name__ == "__main__":
    unittest.main()
