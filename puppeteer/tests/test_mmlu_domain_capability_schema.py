import unittest
from pathlib import Path

from agent.register.persona_loader import load_teammate_specs
from agent.register.register import AgentRegister
from inference.graph.agent_graph import AgentGraph
from role_aware.capability_scopes import ROLE_CAPABILITY_SCOPES, TASK_CAPABILITY_SCOPES
from role_aware.profile_store import ProfileStore
from role_aware.schemas import CAPABILITY_DIMENSIONS, CapabilityProfile, RoleCard


PROJECT = Path(__file__).resolve().parents[1]
POOL = PROJECT / "personas" / "role_aware" / "mmlu_pro_domain_specialist_domain_capability_pool.jsonl"


class MMLUDomainCapabilitySchemaTests(unittest.TestCase):
    def test_new_pool_has_complete_domain_profiles_and_stop_controller(self):
        specs = load_teammate_specs(POOL)
        self.assertEqual(len(specs), 8)
        for spec in specs:
            self.assertEqual(set(spec.role_card.capability_prior), set(CAPABILITY_DIMENSIONS))

        registry = AgentRegister()
        registry.load(specs)
        profiles = ProfileStore()
        profiles.initialize(specs)
        graph = AgentGraph(registry, profiles, allowed_tools=("run_python",))
        self.assertEqual(graph.terminator_agent_index, 7)

        stop = specs[graph.terminator_agent_index].role_card.capability_prior
        self.assertGreater(stop["stop_decision"], stop["formal_quantitative"])
        self.assertGreater(stop["stop_decision"], stop["computing_engineering"])

    def test_specialist_scopes_are_mmlu_subsets(self):
        allowed = set(TASK_CAPABILITY_SCOPES["MMLU-Pro"])
        for role in (
            "Quantitative & Formal Reasoner",
            "Natural & Life Science Specialist",
            "Computing & Engineering Specialist",
            "Social, Legal & Business Specialist",
            "Humanities & Behavioral Specialist",
            "Generalist Independent Solver",
            "Adversarial Verifier",
            "Stop Controller",
        ):
            self.assertTrue(set(ROLE_CAPABILITY_SCOPES[role]) <= allowed)

    def test_legacy_persona_prior_is_mapped_but_old_profile_artifact_is_rejected(self):
        card = RoleCard.from_dict({
            "role_name": "General Reasoner",
            "role_goal": "Solve a task.",
            "core_functions": ["Solve a task."],
            "allowed_actions": ["reasoning"],
            "forbidden_actions": [],
            "expected_input": {"type": "task_context"},
            "expected_output": {"type": "action_result"},
            "tools": [],
            "capability_prior": {
                "planning": 0.7,
                "domain_reasoning": 0.8,
                "verification": 0.9,
                "tool_use": 0.95,
            },
        })
        self.assertEqual(card.capability_prior["task_planning"], 0.7)
        self.assertEqual(card.capability_prior["general_reasoning"], 0.8)
        self.assertEqual(card.capability_prior["evidence_verification"], 0.9)
        self.assertNotIn("tool_use", card.capability_prior)
        with self.assertRaisesRegex(ValueError, "Incompatible capability profile schema"):
            CapabilityProfile.from_dict({
                "capability_names": ["planning", "general_reasoning"],
                "mean": [0.5, 0.5],
                "uncertainty": [1.0, 1.0],
                "observation_count": [0, 0],
            })


if __name__ == "__main__":
    unittest.main()
