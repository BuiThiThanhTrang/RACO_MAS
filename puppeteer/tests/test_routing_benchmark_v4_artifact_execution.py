from __future__ import annotations

import tempfile
from pathlib import Path

from routing_benchmark.v4.artifacts import (
    PROJECT_AUTHORED_DOMAINS,
    build_artifact_pack,
    validate_artifact_pack,
)
from routing_benchmark.v4.execution.registry import adapter_for_manifest
from routing_benchmark.v4.execution.actor import DeterministicToolActor
from routing_benchmark.v4.execution.branching import (
    BRANCH_SCHEMA_VERSION,
    branch_metrics,
    build_branch_checkpoint,
)
from routing_benchmark.v4.execution.schema import (
    COUNTERFACTUAL_SCHEMA_VERSION,
    validate_counterfactual_record,
)
from routing_benchmark.v4.execution.snapshot_store import SnapshotStore
from routing_benchmark.v4.execution.trajectory_runner import EmpiricalTrajectoryRunner
from routing_benchmark.routers import OracleRouter, RoutingDecision


TEST_TEMP_ROOT = Path(__file__).resolve().parent


def _pack(root: Path, domain: str):
    manifest = build_artifact_pack(
        output_root=root / "packs",
        base_workflow_id=f"{domain}-01",
        domain=domain,
        task=f"Concrete {domain} task.",
    )
    path = root / "packs" / domain / f"{domain}-01"
    return manifest, path


def test_every_project_authored_domain_builds_and_executes_end_to_end():
    with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
        root = Path(temporary)
        for domain in PROJECT_AUTHORED_DOMAINS:
            manifest, pack = _pack(root / domain, domain)
            assert validate_artifact_pack(pack)["valid"]
            adapter = adapter_for_manifest(manifest)
            env = adapter.materialize(pack, root / domain / "workspace")
            adapter.configure_fault(env, "clean")
            assert adapter.execute_tool(env, "produce_candidate", {}).status == "success"
            assert adapter.execute_tool(env, "validate_candidate", {}).status == "success"
            assert adapter.execute_tool(env, "execute_downstream", {}).status == "success"
            assert adapter.execute_tool(env, "finalize", {}).status == "success"
            assert adapter.terminal_status(env)["terminal"]
            assert not (env.workspace_dir / "private").exists()


def test_snapshot_restore_isolated_and_content_equivalent():
    with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
        root = Path(temporary)
        manifest, pack = _pack(root, "structured_data_processing")
        adapter = adapter_for_manifest(manifest)
        env = adapter.materialize(pack, root / "workspace")
        adapter.configure_fault(env, "clean")
        store = SnapshotStore(root / "snapshots")
        snapshot = store.create(env, public_state_fingerprint="before", budget={"calls_remaining": 4})
        first = store.restore(snapshot, root / "branch-a", manifest=manifest, pack_dir=pack)
        second = store.restore(snapshot, root / "branch-b", manifest=manifest, pack_dir=pack)
        assert store.equivalent_pre_state(first, second)
        assert store.restored_matches_snapshot(snapshot, first)
        adapter.execute_tool(first, "produce_candidate", {})
        assert not store.equivalent_pre_state(first, second)
        assert not (second.workspace_dir / "_runtime/candidate.json").exists()


def test_fault_variant_is_hidden_until_executor_produces_an_outcome():
    with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
        root = Path(temporary)
        manifest, pack = _pack(root, "personal_device_tool")
        adapter = adapter_for_manifest(manifest)
        clean = adapter.materialize(pack, root / "clean")
        fault = adapter.materialize(pack, root / "fault")
        adapter.configure_fault(clean, "clean")
        adapter.configure_fault(fault, "retryable_failure")
        assert adapter.public_observation(clean) == adapter.public_observation(fault)
        result = adapter.execute_tool(fault, "produce_candidate", {})
        assert result.status == "tool_failure"
        assert adapter.public_observation(fault)["issues"]
        assert adapter.execute_tool(fault, "diagnose_failure", {}).status == "success"
        assert adapter.execute_tool(fault, "repair_candidate", {}).status == "success"


def test_counterfactual_schema_requires_explicit_prestate_equivalence():
    validate_counterfactual_record(
        {
            "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
            "counterfactual_group_id": "cfg-test",
            "snapshot_id": "snap-test",
            "pre_state_fingerprint": "before",
            "branch_type": "action",
            "factual_action": "agent_a",
            "branch_action": "agent_b",
            "pre_state_equivalent": True,
            "execution_status": "success",
            "post_state_fingerprint": "after",
        }
    )


def test_human_action_override_resumes_router_from_saved_snapshot():
    class SequenceRouter:
        def __init__(self, actions):
            self.actions = list(actions)
            self.index = 0

        def reset(self, seed=None):
            del seed
            self.index = 0

        def select(self, observation, candidates, valid):
            del observation, candidates, valid
            action = self.actions[min(self.index, len(self.actions) - 1)]
            self.index += 1
            return RoutingDecision(action)

    with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
        root = Path(temporary)
        manifest, pack = _pack(root, "structured_data_processing")
        profiles = [
            {"agent_id": "Producer", "role_name": "Candidate Producer", "role_goal": "Produce the candidate."},
            {"agent_id": "Validator", "role_name": "Candidate Validator", "role_goal": "Validate the candidate."},
            {"agent_id": "Operator", "role_name": "Workflow Operator", "role_goal": "Run the downstream operation."},
            {"agent_id": "Finalizer", "role_name": "Finalizer", "role_goal": "Commit the final artifact."},
        ]
        row = {
            "decision_id": "branch-test",
            "scenario_id": "branch-test",
            "template_id": "branch-test",
            "workflow_family": "structured_data_processing",
            "task": manifest.task,
            "task_context": {},
            "agent_pool": profiles,
            "candidate_actions": ["Producer", "Validator", "Operator", "Finalizer", "STOP"],
            "valid_actions": ["Producer"],
        }
        adapter = adapter_for_manifest(manifest)
        parent_store = SnapshotStore(root / "parent" / "snapshots")
        parent_runner = EmpiricalTrajectoryRunner(
            router=SequenceRouter(["Producer", "STOP"]),
            actor=DeterministicToolActor(),
            snapshot_store=parent_store,
            memory_view="structured",
            max_calls=4,
        )
        parent = parent_runner.run(
            row=row,
            manifest_path=pack,
            adapter=adapter,
            workspace=root / "parent" / "workspace",
        )
        assert not parent["terminal_status"]["terminal"]
        checkpoint = build_branch_checkpoint(parent, fork_step=1)
        intervention = {
            "schema_version": BRANCH_SCHEMA_VERSION,
            "source_run_id": "parent",
            "parent_episode_id": manifest.base_workflow_id,
            "fork_step": 1,
            "forced_action": "Validator",
            "intervention_source": "human",
            "rationale": "A candidate exists and requires validation.",
        }
        branch_runner = EmpiricalTrajectoryRunner(
            router=OracleRouter(),
            actor=DeterministicToolActor(),
            snapshot_store=SnapshotStore(root / "branch" / "snapshots"),
            memory_view="structured",
            max_calls=4,
        )
        branch = branch_runner.resume_from_checkpoint(
            row=row,
            checkpoint=checkpoint,
            intervention=intervention,
            adapter=adapter,
            source_snapshot_store=parent_store,
            branch_workspace=root / "branch" / "workspace",
        )
        assert branch["terminal_status"]["terminal"], branch
        assert branch["steps"][1]["selected_action"] == "Validator"
        assert branch["steps"][1]["decision_source"] == "human_override"
        assert branch["steps"][2]["decision_source"] == "router"
        assert branch["branch_metadata"]["original_action"] == "STOP"
        metrics = branch_metrics(
            parent_episode=parent, branch_episode=branch, fork_step=1
        )
        assert metrics["branch_vs_factual_completion_delta"] == 1
        parent_outputs = list((root / "parent" / "outputs").rglob("output-*.json"))
        branch_outputs = list((root / "branch" / "outputs").rglob("output-*.json"))
        assert parent_outputs and branch_outputs
        assert not set(parent_outputs) & set(branch_outputs)


def test_patch_author_semantic_family_owns_repair_tool_even_when_goal_mentions_diagnosis():
    from routing_benchmark.v4.execution.adapters.external_code_debugging import (
        ExternalCodeDebuggingAdapter,
    )

    tools = ExternalCodeDebuggingAdapter().tool_spec(
        None,
        "Patch Author",
        {
            "role_name": "Patch Author",
            "role_goal": "Repair the diagnosed defect with a minimal unified diff.",
            "semantic_family": "repair",
        },
    )
    assert [item.name for item in tools] == [
        "inspect_inputs", "inspect_repository_file", "search_repository", "repair_candidate"
    ]
