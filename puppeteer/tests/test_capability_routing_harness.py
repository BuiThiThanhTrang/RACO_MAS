import unittest

from inference.policy.capability_harness import CapabilityRoutingHarness
from inference.policy.route_bundle import RouteBundleBuilder
from inference.policy.stage_routing import RoutingSnapshot


def _view(candidate_id, role, expected_type, affinity=None):
    return {
        "candidate_id": candidate_id,
        "role_name": role,
        "role_goal": f"Perform {role}",
        "core_functions": [role],
        "allowed_actions": ["reasoning"],
        "expected_input": {"type": expected_type, "fields": []},
        "routing_profile": {
            "capability_affinity": affinity or {},
            "use_when": [],
            "avoid_when": [],
            "expected_contribution": [],
            "output_contract": {"type": "result", "fields": ["answer"]},
            "handoff_requirements": [],
        },
        "tools": [],
        "available": True,
    }


class CapabilityRoutingHarnessTests(unittest.TestCase):
    def _snapshot(self, count, **state):
        base = {
            "enabled": True,
            "profile": "musique_dynamic_v2",
            "stage": "path",
            "last_action_status": "success",
            "candidate_answer_exists": False,
            "final_answer_exists": False,
            "paragraph_reference_count": 0,
        }
        base.update(state)
        return RoutingSnapshot(base, tuple(range(count)), ())

    def test_scope_gate_removes_explicit_mismatch_but_keeps_generic_role(self):
        views = [
            _view(0, "MuSiQue Retriever", "musique_question"),
            _view(1, "MMLU Specialist", "mmlu_multiple_choice"),
            _view(2, "Generic Verifier", "candidate_answer"),
        ]
        harness = CapabilityRoutingHarness(
            {
                "enabled": True,
                "capability_shortlist": {"enabled": False},
            }
        )
        result = harness.apply(
            task_type="MuSiQue",
            root=False,
            capacity=2,
            views=views,
            snapshot=self._snapshot(len(views)),
        )
        self.assertEqual(result.snapshot.eligible_candidate_ids, (0, 2))
        self.assertEqual(
            result.snapshot.masked_candidates[0]["reason"],
            "explicit_benchmark_scope_mismatch",
        )

    def test_need_aware_shortlist_prefers_declared_verification_capability(self):
        views = [
            _view(0, "Retriever", "generic", {"retrieval": "high"}),
            _view(1, "Verifier", "generic", {"verification": "high"}),
            _view(2, "Planner", "generic", {"planning": "high"}),
        ]
        harness = CapabilityRoutingHarness(
            {
                "enabled": True,
                "dataset_scope_gate": False,
                "capability_shortlist": {
                    "enabled": True,
                    "max_candidates": 1,
                    "min_candidates": 1,
                },
            }
        )
        result = harness.apply(
            task_type="MuSiQue",
            root=False,
            capacity=2,
            views=views,
            snapshot=self._snapshot(len(views), candidate_answer_exists=True),
            semantic_diagnostics={
                "dominant_next_need": {"choice": "verify_candidate"}
            },
        )
        self.assertEqual(result.snapshot.eligible_candidate_ids, (1,))

    def test_terminal_gate_requires_final_answer_evidence_and_healthy_action(self):
        harness = CapabilityRoutingHarness({"enabled": True})
        state = {
            "enabled": True,
            "profile": "musique_dynamic_v2",
            "final_answer_exists": True,
            "paragraph_reference_count": 2,
            "last_action_status": "success",
        }
        self.assertTrue(harness.terminal_ready(state))
        self.assertFalse(harness.terminal_ready(state | {"paragraph_reference_count": 0}))
        self.assertFalse(harness.terminal_ready(state | {"last_action_status": "failure"}))

    def test_branching_is_conditional_and_root_starts_with_one_path(self):
        harness = CapabilityRoutingHarness(
            {
                "enabled": True,
                "adaptive_branching": {
                    "enabled": True,
                    "initial_paths": 1,
                    "max_branch_options": 3,
                },
            }
        )
        state = {"path_stalled": False, "repeat_without_progress": False}
        self.assertEqual(
            harness.route_capacity(root=True, capacity=3, routing_state=state), 1
        )
        self.assertEqual(
            harness.route_capacity(root=False, capacity=3, routing_state=state), 1
        )
        diagnostics = {"candidate_conflict_present": {"noul": 0.9}}
        self.assertEqual(
            harness.route_capacity(
                root=False,
                capacity=3,
                routing_state=state,
                semantic_diagnostics=diagnostics,
            ),
            2,
        )

    def test_next_options_only_expose_branch_bundles_when_enabled(self):
        views = [
            _view(0, "A", "generic"),
            _view(1, "B", "generic"),
            _view(2, "C", "generic"),
        ]
        builder = RouteBundleBuilder(max_options=20)
        normal = builder.next_options(views, capacity=2, allow_branch=False)
        branched = builder.next_options(
            views,
            capacity=2,
            allow_branch=True,
            max_branch_options=2,
        )
        self.assertFalse(any(option.option_id.startswith("branch__") for option in normal))
        self.assertEqual(
            sum(option.option_id.startswith("branch__") for option in branched), 2
        )

    def test_blackboard_is_typed_and_bounds_raw_history(self):
        harness = CapabilityRoutingHarness(
            {"enabled": True, "max_previous_outputs": 2}
        )
        board = harness.blackboard(
            task={"type": "MuSiQue", "Question": "Who?"},
            routing_state={
                "enabled": True,
                "profile": "musique_dynamic_v2",
                "referenced_paragraph_ids": [1, 3],
                "paragraph_reference_count": 2,
                "candidate_answers": ["A"],
                "final_answers": [],
            },
            previous_outputs=[{"id": 1}, {"id": 2}, {"id": 3}],
            remaining_depth=3,
            remaining_width=2,
        )
        self.assertEqual(board["objective"], "Who?")
        self.assertEqual(board["evidence"]["resource_ids"], [1, 3])
        self.assertEqual(
            board["execution"]["recent_outputs"], [{"id": 2}, {"id": 3}]
        )


if __name__ == "__main__":
    unittest.main()

