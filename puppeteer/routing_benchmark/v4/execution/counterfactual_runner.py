from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from routing_benchmark.v4.artifacts.manifest import load_manifest
from routing_benchmark.v4.artifacts.schema import canonical_hash

from .base import EnvironmentSnapshot


def run_action_counterfactuals(
    *,
    selection: Mapping[str, Any],
    adapter,
    actor,
    snapshot_store,
    profiles: Mapping[str, Mapping[str, Any]],
    output_root: str | Path,
    branch_actions: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    manifest_path = Path(str(selection["manifest_path"]))
    manifest = load_manifest(manifest_path)
    pack_dir = manifest_path if manifest_path.is_dir() else manifest_path.parent
    snapshot_dir = Path(str(selection["snapshot_dir"]))
    snapshot = EnvironmentSnapshot(
        snapshot_id=str(selection["snapshot_id"]),
        source_workspace=snapshot_dir,
        snapshot_dir=snapshot_dir,
        tree_hash="",
        public_state_fingerprint=str(selection["pre_state_fingerprint"]),
        budget=dict((selection.get("pre_action_observation") or {}).get("budget") or {}),
    )
    group_id = "cfg-" + canonical_hash(
        [selection.get("base_workflow_id"), selection["snapshot_id"], selection["factual_action"], selection.get("selection_reasons")]
    )
    records: list[dict[str, Any]] = []
    for index, action in enumerate(branch_actions or selection.get("branch_actions") or ()):
        if action not in profiles:
            continue
        branch_dir = Path(output_root) / group_id / f"branch-{index:03d}-{action}"
        env = snapshot_store.restore(
            snapshot,
            branch_dir,
            manifest=manifest,
            pack_dir=pack_dir,
        )
        before_equivalence = (
            str(selection["pre_state_fingerprint"]) == snapshot.public_state_fingerprint
            and snapshot_store.restored_matches_snapshot(snapshot, env)
        )
        precondition = adapter.public_preconditions(env, action, profiles[action])
        if precondition.executable:
            result = actor.execute(
                env=env,
                adapter=adapter,
                action=action,
                role_profile=profiles[action],
                task=manifest.task,
            )
            status = result.get("transport_status", "success")
        else:
            result = {"raw_output": "", "tool_events": [], "artifact_diff": {"added": [], "modified": [], "removed": []}}
            status = "pre_execution_rejection"
        records.append(
            {
                "schema_version": "4.2-counterfactual-rc1",
                "counterfactual_group_id": group_id,
                "snapshot_id": snapshot.snapshot_id,
                "pre_state_fingerprint": snapshot.public_state_fingerprint,
                "branch_type": "action",
                "factual_action": selection["factual_action"],
                "branch_action": action,
                "selection_reasons": list(selection.get("selection_reasons") or ()),
                "pre_state_equivalent": before_equivalence,
                "execution_status": status,
                "executor_result": result,
                "post_environment": adapter.public_observation(env),
                "post_state_fingerprint": canonical_hash(adapter.public_observation(env), 20),
                "human_review_status": "pending",
            }
        )
    return records
