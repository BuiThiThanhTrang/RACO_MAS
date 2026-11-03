from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from routing_benchmark.v2.environment import RoutingEnvironmentV2

from .generator import PROJECT_DIR, VARIANTS, _safe_actions
from .loader import BenchmarkBundleV4
from .post_fault import (
    _shortest_safe_continuation,
    checkpoint_safe_actions,
    restore_checkpoint_environment,
)
from .public import build_public_observation
from .schemas import V4_SCHEMA_VERSION


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
    "fault_carrier",
    "expected_recovery_action",
    "inefficient_transition_ids",
    "internal_scenario_id",
    "clean_scenario_id",
    "condition",
    "pair_id",
    "oracle_min_remaining_calls",
    "checkpoint_state_fingerprint",
    "unsafe_transition_ids",
    "detector_actions_available",
    "requires_detection_agent",
}


def _public_keys(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            result.add(str(key))
            result.update(_public_keys(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_public_keys(child))
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _safe_path_exists(bundle: BenchmarkBundleV4, scenario_id: str) -> bool:
    scenario = next(item for item in bundle.scenarios if item.scenario_id == scenario_id)
    annotation = bundle.recovery_annotations[scenario_id]
    initial = RoutingEnvironmentV2(scenario, bundle.agents)
    queue = deque([initial])
    seen = {initial.state.fingerprint()}
    while queue:
        env = queue.popleft()
        if env.goal_reached:
            return True
        for action in _safe_actions(env, annotation):
            if action == "STOP":
                continue
            child = env.clone()
            child.step(action)
            if child.done and not child.goal_reached:
                continue
            fingerprint = child.state.fingerprint()
            if fingerprint not in seen:
                seen.add(fingerprint)
                queue.append(child)
    return False


def validate_v4(bundle: BenchmarkBundleV4) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if bundle.metadata.get("version") != V4_SCHEMA_VERSION:
        errors.append(f"Expected version {V4_SCHEMA_VERSION}")
    for field in ("preferred_actions", "action_utilities", "capability_scores"):
        if bundle.metadata.get(field) is not False:
            errors.append(f"Metadata must disable {field}")

    scenario_ids = {item.scenario_id for item in bundle.scenarios}
    if scenario_ids != set(bundle.recovery_annotations):
        errors.append("Every and only scenario must have a recovery annotation")
    if scenario_ids != set(bundle.task_inputs):
        errors.append("Every and only scenario must have a task input")
    if scenario_ids != {item.scenario_id for item in bundle.provenance.values()}:
        errors.append("Scenario and provenance IDs do not match")

    fixture_ids = {str(item["transition_id"]) for item in bundle.executor_outputs}
    expected_transition_ids = {
        transition.transition_id
        for scenario in bundle.scenarios
        for transition in scenario.transitions
    }
    if fixture_ids != expected_transition_ids:
        missing = sorted(expected_transition_ids - fixture_ids)
        extra = sorted(fixture_ids - expected_transition_ids)
        errors.append(f"Executor fixture coverage mismatch; missing={missing[:3]} extra={extra[:3]}")

    pair_splits: dict[str, set[str]] = defaultdict(set)
    pools_by_family: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    initial_actions_by_family: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for scenario in bundle.scenarios:
        annotation = bundle.recovery_annotations[scenario.scenario_id]
        pair_splits[str(annotation["pair_id"])].add(scenario.split)
        pools_by_family[scenario.family].add(tuple(scenario.agent_pool))
        initial_env = RoutingEnvironmentV2(scenario, bundle.agents)
        initial_actions_by_family[scenario.family].add(
            tuple(_safe_actions(initial_env, annotation))
        )
        clean_id = str(annotation["clean_scenario_id"])
        if clean_id not in scenario_ids:
            errors.append(f"Missing clean counterpart: {scenario.scenario_id}")
        unsafe = set(map(str, annotation.get("unsafe_transition_ids") or ()))
        inefficient = set(
            map(str, annotation.get("inefficient_transition_ids") or ())
        )
        if not unsafe.issubset({item.transition_id for item in scenario.transitions}):
            errors.append(f"Unknown unsafe transition: {scenario.scenario_id}")
        if not inefficient.issubset(
            {item.transition_id for item in scenario.transitions}
        ):
            errors.append(f"Unknown inefficient transition: {scenario.scenario_id}")
        if not _safe_path_exists(bundle, scenario.scenario_id):
            errors.append(f"No safe path to goal: {scenario.scenario_id}")
        provenance = next(
            item for item in bundle.provenance.values() if item.scenario_id == scenario.scenario_id
        )
        source = PROJECT_DIR / provenance.source_path
        if not source.exists():
            errors.append(f"Missing provenance source: {source}")
        elif hashlib.sha256(source.read_bytes()).hexdigest() != provenance.source_sha256:
            errors.append(f"Provenance hash mismatch: {provenance.provenance_id}")
        if provenance.human_review_status == "pending":
            warnings.append(f"Human review pending: {scenario.scenario_id}")

    crossing = [key for key, values in pair_splits.items() if len(values) != 1]
    if crossing:
        errors.append(f"Counterfactual pairs cross splits: {crossing}")
    unequal_pools = [key for key, values in pools_by_family.items() if len(values) != 1]
    if unequal_pools:
        errors.append(f"Counterfactual agent pools differ: {unequal_pools}")
    unequal_initial = [
        key for key, values in initial_actions_by_family.items() if len(values) != 1
    ]
    if unequal_initial:
        errors.append(f"Counterfactual initial safe actions differ: {unequal_initial}")

    label_index = {
        str(item["decision_id"]): item
        for annotation in bundle.recovery_annotations.values()
        for item in annotation.get("decision_labels") or ()
    }
    decision_ids: set[str] = set()
    valid_sets_by_public_observation: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in bundle.decision_rows:
        decision_id = str(row["decision_id"])
        if decision_id in decision_ids:
            errors.append(f"Duplicate decision ID: {decision_id}")
        decision_ids.add(decision_id)
        if row.get("schema_version") != V4_SCHEMA_VERSION:
            errors.append(f"Incorrect schema version: {decision_id}")
        router_scenario_id = str(row.get("router_scenario_id", ""))
        if not router_scenario_id:
            errors.append(f"Missing router_scenario_id: {decision_id}")
        elif any(variant in router_scenario_id for variant in VARIANTS):
            errors.append(f"Router-visible scenario ID leaks variant: {decision_id}")
        candidates = set(map(str, row.get("candidate_actions") or ()))
        valid = set(map(str, row.get("valid_actions") or ()))
        if not valid or not valid.issubset(candidates):
            errors.append(f"Invalid action set: {decision_id}")
        label = label_index.get(decision_id)
        if label is None:
            errors.append(f"Missing recovery decision label: {decision_id}")
        elif valid.intersection(label.get("unsafe_actions") or ()):
            errors.append(f"Unsafe action appears in valid set: {decision_id}")
        views = row.get("memory_views") or {}
        if set(views) not in ({"state", "structured"}, {"state", "structured", "raw"}):
            errors.append(f"Invalid memory view set: {decision_id}")
        if views.get("state", {}).get("recent_events"):
            errors.append(f"State-only view leaks recent events: {decision_id}")
        if views.get("state", {}).get("execution_memory"):
            errors.append(f"State-only view leaks persistent memory: {decision_id}")
        if row.get("split") == "development" and "raw" not in views:
            errors.append(f"Development row lacks raw diagnostic view: {decision_id}")
        if row.get("split") != "development" and "raw" in views:
            errors.append(f"Raw history leaks outside development: {decision_id}")
        public = {
            "task": row.get("task"),
            "task_context": row.get("task_context"),
            "memory_views": views,
            "budget": row.get("budget"),
            "agent_pool": row.get("agent_pool"),
            "candidate_actions": row.get("candidate_actions"),
        }
        leaked = FORBIDDEN_PUBLIC_KEYS.intersection(_public_keys(public))
        if leaked:
            errors.append(f"Public view leaks private keys {sorted(leaked)}: {decision_id}")
        for memory_view in views:
            observation = build_public_observation(row, memory_view)
            signature = hashlib.sha256(
                json.dumps(
                    observation, ensure_ascii=False, sort_keys=True
                ).encode("utf-8")
            ).hexdigest()
            valid_sets_by_public_observation[signature].add(
                tuple(sorted(valid))
            )

    inconsistent_observations = [
        key
        for key, values in valid_sets_by_public_observation.items()
        if len(values) != 1
    ]
    if inconsistent_observations:
        errors.append(
            "Identical public observations have different valid action sets: "
            f"{len(inconsistent_observations)} signatures"
        )

    public_packaging_valid = True
    post_fault_packaging_valid = True
    if bundle.root is not None:
        public_path = bundle.root / "public_inputs.jsonl"
        private_path = bundle.root / "private_labels.jsonl"
        if not public_path.exists() or not private_path.exists():
            errors.append("Release must contain public_inputs.jsonl and private_labels.jsonl")
            public_packaging_valid = False
        else:
            public_rows = _read_jsonl(public_path)
            private_rows = _read_jsonl(private_path)
            if len(public_rows) != len(bundle.decision_rows):
                errors.append("Public input count does not match decision state count")
                public_packaging_valid = False
            if len(private_rows) != len(bundle.decision_rows):
                errors.append("Private label count does not match decision state count")
                public_packaging_valid = False
            for item in public_rows:
                leaked = FORBIDDEN_PUBLIC_KEYS.intersection(_public_keys(item))
                if leaked:
                    errors.append(
                        "Packaged public input leaks private keys "
                        f"{sorted(leaked)}: {item.get('decision_id')}"
                    )
                    public_packaging_valid = False
                public_decision_id = str(item.get("decision_id", ""))
                if any(variant in public_decision_id for variant in VARIANTS):
                    errors.append(
                        "Packaged public decision ID leaks a fault variant: "
                        f"{public_decision_id}"
                    )
                    public_packaging_valid = False
                for observation in (item.get("observations") or {}).values():
                    scenario_id = str(observation.get("scenario_id", ""))
                    if any(variant in scenario_id for variant in VARIANTS):
                        errors.append(
                            "Packaged router scenario ID leaks a fault variant: "
                            f"{item.get('decision_id')}"
                        )
                        public_packaging_valid = False

        post_fault_paths = {
            name: bundle.root / name
            for name in (
                "post_fault_inputs.jsonl",
                "post_fault_runtime.jsonl",
                "post_fault_labels.jsonl",
                "post_fault_pair_manifest.jsonl",
            )
        }
        missing_post_fault = [
            name for name, path in post_fault_paths.items() if not path.exists()
        ]
        if missing_post_fault:
            errors.append(
                f"Release lacks Track C artifacts: {missing_post_fault}"
            )
            post_fault_packaging_valid = False

    post_fault_inputs = {
        str(item["checkpoint_id"]): item for item in bundle.post_fault_inputs
    }
    post_fault_runtime = {
        str(item["checkpoint_id"]): item for item in bundle.post_fault_runtime
    }
    post_fault_labels = {
        str(item["checkpoint_id"]): item for item in bundle.post_fault_labels
    }
    checkpoint_id_sets = {
        tuple(sorted(post_fault_inputs)),
        tuple(sorted(post_fault_runtime)),
        tuple(sorted(post_fault_labels)),
    }
    if len(checkpoint_id_sets) != 1:
        errors.append("Track C public, runtime, and label checkpoint IDs differ")
        post_fault_packaging_valid = False

    post_fault_observation_labels: dict[
        str, set[tuple[str, ...]]
    ] = defaultdict(set)
    scenario_index = {item.scenario_id: item for item in bundle.scenarios}
    for checkpoint_id, public_row in post_fault_inputs.items():
        leaked = FORBIDDEN_PUBLIC_KEYS.intersection(_public_keys(public_row))
        if leaked:
            errors.append(
                "Track C public input leaks private keys "
                f"{sorted(leaked)}: {checkpoint_id}"
            )
            post_fault_packaging_valid = False
        if not checkpoint_id.startswith("pfc-") or any(
            variant in checkpoint_id for variant in VARIANTS
        ):
            errors.append(f"Track C checkpoint ID is not opaque: {checkpoint_id}")
            post_fault_packaging_valid = False
        views = public_row.get("observations") or {}
        expected_views = (
            {"state", "structured", "raw"}
            if public_row.get("split") == "development"
            else {"state", "structured"}
        )
        if set(views) != expected_views:
            errors.append(f"Track C memory view mismatch: {checkpoint_id}")
        label = post_fault_labels.get(checkpoint_id)
        runtime = post_fault_runtime.get(checkpoint_id)
        if label is None or runtime is None:
            continue
        try:
            env = restore_checkpoint_environment(
                runtime, scenario_index, bundle.agents
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"Cannot restore Track C checkpoint {checkpoint_id}: {exc}")
            continue
        if env.state.fingerprint() != label.get("checkpoint_state_fingerprint"):
            errors.append(f"Track C state fingerprint mismatch: {checkpoint_id}")
        valid = tuple(checkpoint_safe_actions(env, label))
        if not valid or tuple(label.get("valid_actions") or ()) != valid:
            errors.append(f"Track C valid action mismatch: {checkpoint_id}")
        minimum = _shortest_safe_continuation(env, label)
        if minimum is None or minimum != label.get("oracle_min_remaining_calls"):
            errors.append(f"Track C oracle continuation mismatch: {checkpoint_id}")
        for observation in views.values():
            signature = hashlib.sha256(
                json.dumps(
                    observation, ensure_ascii=False, sort_keys=True
                ).encode("utf-8")
            ).hexdigest()
            post_fault_observation_labels[signature].add(tuple(sorted(valid)))

    inconsistent_post_fault = [
        key
        for key, values in post_fault_observation_labels.items()
        if len(values) != 1
    ]
    if inconsistent_post_fault:
        errors.append(
            "Identical Track C observations have different valid action sets: "
            f"{len(inconsistent_post_fault)} signatures"
        )

    pair_conditions = Counter()
    pair_budget_equal = True
    pair_contract_equal = True
    for pair in bundle.post_fault_pairs:
        pair_id = str(pair["pair_id"])
        fault_id = str(pair["fault_checkpoint_id"])
        control_id = str(pair["control_checkpoint_id"])
        fault_label = post_fault_labels.get(fault_id)
        control_label = post_fault_labels.get(control_id)
        fault_runtime = post_fault_runtime.get(fault_id)
        control_runtime = post_fault_runtime.get(control_id)
        if not all((fault_label, control_label, fault_runtime, control_runtime)):
            errors.append(f"Incomplete Track C pair: {pair_id}")
            continue
        pair_conditions.update(
            (str(fault_label["condition"]), str(control_label["condition"]))
        )
        fault_state = fault_runtime["state"]
        control_state = control_runtime["state"]
        if (
            fault_state.get("steps_used") != control_state.get("steps_used")
            or fault_state.get("cost_used") != control_state.get("cost_used")
        ):
            pair_budget_equal = False
            errors.append(f"Track C pair budget differs: {pair_id}")
        fault_scenario = scenario_index[str(fault_runtime["internal_scenario_id"])]
        control_scenario = scenario_index[
            str(control_runtime["internal_scenario_id"])
        ]
        if (
            fault_scenario.task != control_scenario.task
            or fault_scenario.agent_pool != control_scenario.agent_pool
        ):
            pair_contract_equal = False
            errors.append(f"Track C pair task/pool differs: {pair_id}")

    domain_counts = Counter(item.family for item in bundle.scenarios)
    variant_counts = Counter(
        item["variant"] for item in bundle.recovery_annotations.values()
    )
    targets = {
        "six_domains": len(domain_counts) == 6,
        "five_variants_per_domain": set(domain_counts.values()) == {5},
        "all_variants_balanced": set(variant_counts) == set(VARIANTS)
        and set(variant_counts.values()) == {6},
        "has_recovery_states": any(
            item.get("state_provenance")
            in {"verification", "recovery", "repair", "contaminated"}
            for item in bundle.decision_rows
        ),
        "has_unsafe_cascade_states": any(
            item.get("unsafe_actions") for item in label_index.values()
        ),
        "raw_history_development_only": all(
            (("raw" in (item.get("memory_views") or {})) == (item.get("split") == "development"))
            for item in bundle.decision_rows
        ),
        "counterfactual_agent_pools_equal": all(
            len(values) == 1 for values in pools_by_family.values()
        ),
        "counterfactual_initial_actions_equal": all(
            len(values) == 1 for values in initial_actions_by_family.values()
        ),
        "identical_public_observations_have_equal_valid_actions": not inconsistent_observations,
        "public_private_packaging_separated": public_packaging_valid,
        "post_fault_checkpoint_count_48": len(post_fault_runtime) == 48,
        "post_fault_pair_count_24": len(bundle.post_fault_pairs) == 24,
        "post_fault_conditions_balanced": pair_conditions
        == Counter({"fault": 24, "control": 24}),
        "post_fault_split_balance": Counter(
            item.get("split") for item in bundle.post_fault_runtime
        )
        == Counter({"development": 16, "validation": 16, "test": 16}),
        "post_fault_replay_and_oracle_valid": not any(
            "Track C" in item and (
                "restore" in item
                or "fingerprint" in item
                or "valid action" in item
                or "oracle continuation" in item
            )
            for item in errors
        ),
        "post_fault_pair_budget_equal": pair_budget_equal,
        "post_fault_pair_contract_equal": pair_contract_equal,
        "post_fault_identical_observations_have_equal_valid_actions": not inconsistent_post_fault,
        "post_fault_public_private_packaging_separated": post_fault_packaging_valid,
    }
    if not all(targets.values()):
        errors.append(
            "V4 quota targets failed: "
            + str([key for key, value in targets.items() if not value])
        )
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "workflow_count": len(bundle.scenarios),
        "decision_state_count": len(bundle.decision_rows),
        "post_fault_checkpoint_count": len(bundle.post_fault_runtime),
        "post_fault_pair_count": len(bundle.post_fault_pairs),
        "domain_counts": dict(sorted(domain_counts.items())),
        "variant_counts": dict(sorted(variant_counts.items())),
        "human_review_pending_count": sum(
            item.human_review_status == "pending" for item in bundle.provenance.values()
        ),
        "targets": targets,
    }
