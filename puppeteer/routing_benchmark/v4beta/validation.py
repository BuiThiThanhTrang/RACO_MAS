from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from routing_benchmark.v2.environment import RoutingEnvironmentV2
from routing_benchmark.v4.generator import VARIANTS, _safe_actions
from routing_benchmark.v4.loader import BenchmarkBundleV4, load_benchmark_v4
from routing_benchmark.v4.post_fault import restore_checkpoint_environment

from .schemas import V4_BETA_SCHEMA_VERSION


FORBIDDEN_PUBLIC_KEYS = {
    "valid_actions",
    "action_rationales",
    "preferred_actions",
    "action_utilities",
    "capability_scores",
    "fault_class",
    "fault_origin",
    "failure_origin",
    "failure_attribution",
    "fault_variant",
    "expected_recovery_action",
    "oracle_min_remaining_calls",
    "unsafe_transition_ids",
    "detector_actions_available",
    "requires_detection_agent",
}
ALLOWED_EXECUTION_STATUSES = {
    "EXECUTED_SUCCESS",
    "EXECUTED_INEFFICIENT",
    "EXECUTED_INVALID_OUTPUT",
    "EXECUTED_NO_PROGRESS",
    "EXECUTED_RETRYABLE_FAILURE",
    "EXECUTED_HARD_FAILURE",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _keys(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            result.add(str(key))
            result.update(_keys(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_keys(child))
    return result


def _normalized_task(task: str) -> str:
    return " ".join(task.casefold().split())


def _role_contract_errors(bundle: BenchmarkBundleV4) -> list[str]:
    errors: list[str] = []
    hidden_terms = {"fault_class", "valid_actions", "preferred_actions", "capability_score", "backbone"}
    for profile in bundle.agents.values():
        public = profile.public_profile()
        if not public.get("supported_inputs") or not public.get("expected_outputs"):
            errors.append(f"Role card lacks inputs or outputs: {profile.agent_id}")
        text = json.dumps(public, ensure_ascii=False).casefold()
        leaked = [term for term in hidden_terms if term in text]
        if leaked:
            errors.append(f"Role card leaks private/ranking terms {leaked}: {profile.agent_id}")
    for scenario in bundle.scenarios:
        for transition in scenario.transitions:
            if transition.action not in scenario.agent_pool:
                errors.append(f"Transition action absent from pool: {transition.transition_id}")
            if transition.status not in ALLOWED_EXECUTION_STATUSES:
                errors.append(f"Unsupported execution status {transition.status}: {transition.transition_id}")
            profile = bundle.agents[transition.action]
            if transition.add_artifacts and not profile.expected_outputs:
                errors.append(f"Artifact-producing transition lacks declared role outputs: {transition.transition_id}")
            if transition.requires_artifacts and not profile.supported_inputs:
                errors.append(f"Artifact-consuming transition lacks declared role inputs: {transition.transition_id}")
    return errors


def validate_v4_beta(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    bundle = load_benchmark_v4(root)
    errors: list[str] = []
    warnings: list[str] = []
    if bundle.metadata.get("version") != V4_BETA_SCHEMA_VERSION:
        errors.append("metadata.version is not 4.0-beta")
    if len(bundle.scenarios) != 620:
        errors.append(f"Expected 620 scenarios, found {len(bundle.scenarios)}")
    base_rows = _read_jsonl(root / "base_workflows.jsonl")
    if len(base_rows) != 124:
        errors.append(f"Expected 124 base workflows, found {len(base_rows)}")
    base_counts = Counter(item["task_domain"] for item in base_rows)
    expected_counts = {
        **{domain: 13 for domain in bundle.metadata.get("task_domains", [])[:8]},
        **{domain: 10 for domain in bundle.metadata.get("unseen_domain_holdouts", [])},
    }
    if dict(base_counts) != expected_counts:
        errors.append(f"Unexpected per-domain base quotas: {dict(base_counts)}")
    split_counts = Counter(item["split"] for item in base_rows)
    if split_counts != Counter({"development": 24, "validation": 40, "test": 60}):
        errors.append(f"Unexpected base split quotas: {dict(split_counts)}")
    holdouts = set(bundle.metadata.get("unseen_domain_holdouts") or ())
    leaked_holdouts = [item["base_workflow_id"] for item in base_rows if item["task_domain"] in holdouts and item["split"] != "test"]
    if leaked_holdouts:
        errors.append(f"Unseen-domain holdouts leak outside test: {leaked_holdouts}")

    source_group_splits: dict[str, set[str]] = defaultdict(set)
    task_hash_splits: dict[str, set[str]] = defaultdict(set)
    for item in base_rows:
        source_group_splits[str(item["source_task_group"])].add(str(item["split"]))
        task_hash_splits[hashlib.sha256(_normalized_task(str(item["task"])).encode()).hexdigest()].add(str(item["split"]))
    if any(len(values) > 1 for values in source_group_splits.values()):
        errors.append("A source_task_group crosses dataset splits")
    if any(len(values) > 1 for values in task_hash_splits.values()):
        errors.append("A normalized task duplicate crosses dataset splits")

    annotations = bundle.recovery_annotations
    grouped: dict[str, list[Any]] = defaultdict(list)
    for scenario in bundle.scenarios:
        if scenario.schema_version != V4_BETA_SCHEMA_VERSION:
            errors.append(f"Incorrect scenario schema version: {scenario.scenario_id}")
        annotation = annotations[scenario.scenario_id]
        grouped[str(annotation["base_workflow_id"])].append(scenario)
    for base_id, scenarios in grouped.items():
        if len(scenarios) != len(VARIANTS):
            errors.append(f"Base workflow does not have five variants: {base_id}")
            continue
        invariant = {
            (
                item.task,
                tuple(item.agent_pool),
                tuple(sorted(item.initial_facts)),
                tuple(sorted(item.initial_artifacts)),
                item.max_steps,
                item.max_cost,
                tuple(sorted(item.goal_facts)),
                tuple(sorted(item.goal_artifacts)),
            )
            for item in scenarios
        }
        if len(invariant) != 1:
            errors.append(f"Counterfactual invariants differ: {base_id}")
        initial_sets = set()
        for scenario in scenarios:
            env = RoutingEnvironmentV2(scenario, bundle.agents)
            initial_sets.add(tuple(sorted(_safe_actions(env, annotations[scenario.scenario_id]))))
        if len(initial_sets) != 1:
            errors.append(f"Initial safe actions differ across variants: {base_id}")

    errors.extend(_role_contract_errors(bundle))
    for row in bundle.decision_rows:
        if row.get("schema_version") != V4_BETA_SCHEMA_VERSION:
            errors.append(f"Incorrect decision schema: {row.get('decision_id')}")
        if not set(row.get("valid_actions") or ()).issubset(set(row.get("candidate_actions") or ())):
            errors.append(f"Valid action outside candidate set: {row.get('decision_id')}")
    public_rows = _read_jsonl(root / "public_inputs.jsonl")
    for row in public_rows:
        leaked = FORBIDDEN_PUBLIC_KEYS.intersection(_keys(row))
        if leaked:
            errors.append(f"Public row leaks private fields {sorted(leaked)}: {row.get('decision_id')}")
        if row.get("workflow_family") != "dynamic_workflow":
            errors.append(f"Public row exposes analysis domain/pattern label: {row.get('decision_id')}")
        for observation in (row.get("observations") or {}).values():
            if observation.get("workflow_family") != "dynamic_workflow":
                errors.append(f"Public observation exposes analysis domain label: {row.get('decision_id')}")

    labels = {str(item["checkpoint_id"]): item for item in bundle.post_fault_labels}
    runtime = {str(item["checkpoint_id"]): item for item in bundle.post_fault_runtime}
    if len(bundle.post_fault_pairs) != 496 or len(bundle.post_fault_runtime) != 992:
        errors.append(
            f"Expected 496 Track C pairs/992 episodes, found {len(bundle.post_fault_pairs)}/{len(bundle.post_fault_runtime)}"
        )
    pair_ids = [str(item["pair_id"]) for item in bundle.post_fault_pairs]
    checkpoint_ids = [str(item["checkpoint_id"]) for item in bundle.post_fault_labels]
    if len(set(pair_ids)) != len(pair_ids):
        errors.append("Duplicate Track C pair IDs")
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        errors.append("Duplicate Track C checkpoint IDs")
    scenarios = {item.scenario_id: item for item in bundle.scenarios}
    for pair in bundle.post_fault_pairs:
        fault_id = str(pair["fault_checkpoint_id"])
        control_id = str(pair["control_checkpoint_id"])
        if fault_id not in labels or control_id not in labels:
            errors.append(f"Track C pair references missing label: {pair['pair_id']}")
            continue
        fault = labels[fault_id]
        control = labels[control_id]
        fault_state = runtime[fault_id].get("state") or {}
        control_state = runtime[control_id].get("state") or {}
        fault_scenario = scenarios[str(fault["internal_scenario_id"])]
        control_scenario = scenarios[str(control["internal_scenario_id"])]
        matched = (
            fault_state.get("steps_used") == control_state.get("steps_used")
            and fault_state.get("cost_used") == control_state.get("cost_used")
            and fault_scenario.max_steps - int(fault_state.get("steps_used", 0))
            == control_scenario.max_steps - int(control_state.get("steps_used", 0))
            and fault_scenario.max_cost - float(fault_state.get("cost_used", 0.0))
            == control_scenario.max_cost - float(control_state.get("cost_used", 0.0))
            and tuple(fault_scenario.agent_pool) == tuple(control_scenario.agent_pool)
            and fault_scenario.task == control_scenario.task
        )
        if not matched:
            errors.append(f"Unmatched Track C prefix/budget: {pair['pair_id']}")
        for checkpoint_id in (fault_id, control_id):
            restored = restore_checkpoint_environment(runtime[checkpoint_id], scenarios, bundle.agents)
            if restored.state.fingerprint() != labels[checkpoint_id]["checkpoint_state_fingerprint"]:
                errors.append(f"Track C checkpoint replay mismatch: {checkpoint_id}")

    for item in bundle.provenance.values():
        source = PROJECT_DIR / item.source_path
        if not source.exists():
            errors.append(f"Missing provenance source: {item.source_path}")
        elif hashlib.sha256(source.read_bytes()).hexdigest() != item.source_sha256:
            errors.append(f"Provenance hash mismatch: {item.provenance_id}")

    review_rows = _read_jsonl(root / "human_review_queue.jsonl")
    review_pending = sum(
        item.get("reviewer_1_status") != "approved"
        or item.get("reviewer_2_status") != "approved"
        or item.get("adjudication_status") not in {"not_required", "resolved"}
        for item in review_rows
    )
    if review_pending:
        warnings.append(
            f"Two-reviewer human review pending for {review_pending} base workflows"
        )
    report = {
        "benchmark_version": V4_BETA_SCHEMA_VERSION,
        "valid": not errors,
        "automated_release_gates_passed": not errors,
        "human_review_complete": review_pending == 0,
        "release_ready": not errors and review_pending == 0,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings[:25],
        "human_review_pending_count": review_pending,
        "counts": {
            "base_workflows": len(base_rows),
            "scenarios": len(bundle.scenarios),
            "decision_states": len(bundle.decision_rows),
            "post_fault_pairs": len(bundle.post_fault_pairs),
            "post_fault_episodes": len(bundle.post_fault_runtime),
        },
    }
    return report


PROJECT_DIR = Path(__file__).resolve().parents[2]
