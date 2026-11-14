from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from routing_benchmark.v2.loader import validate_bundle
from routing_benchmark.v3.loader import BenchmarkBundleV3
from routing_benchmark.v3.schemas import STOP_ACTION

from .generator import PROJECT_DIR
from .schemas import V3_BETA_SCHEMA_VERSION


FORBIDDEN_LABELS = ("preferred_actions", "action_utilities", "capability_scores")


def _source_ids(path: Path, dataset: str) -> set[str]:
    suffix = path.suffix.casefold()
    if suffix == ".jsonl":
        values = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    for key in ("scenario_id", "id", "task_id", "question_id"):
                        if item.get(key) is not None:
                            values.add(str(item[key]))
        return values
    if suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path)
        preferred = {
            "MuSiQue": "id",
            "MMLU-Pro": "question_id",
            "GAIA": "task_id",
        }.get(dataset)
        columns = [preferred] if preferred in frame.columns else [
            key for key in ("id", "task_id", "question_id", "scenario_id") if key in frame.columns
        ]
        return {
            str(value)
            for key in columns
            for value in frame[key].tolist()
            if value is not None
        }
    if suffix == ".csv":
        import pandas as pd

        frame = pd.read_csv(path)
        key = next((item for item in ("id", "task_id", "Name") if item in frame.columns), None)
        return set() if key is None else {str(value) for value in frame[key].tolist()}
    raise ValueError(f"Unsupported provenance source format: {path}")


def _public_signature(row: dict[str, Any]) -> str:
    public = {
        key: row.get(key)
        for key in (
            "task",
            "task_context",
            "observed_state",
            "execution_memory",
            "recent_events",
            "budget",
            "agent_pool",
            "candidate_actions",
        )
    }
    return json.dumps(public, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_v3_beta(bundle: BenchmarkBundleV3) -> dict[str, Any]:
    validate_bundle(bundle.as_v2_bundle())
    errors: list[str] = []
    warnings: list[str] = []
    if bundle.metadata.get("version") != V3_BETA_SCHEMA_VERSION:
        errors.append(
            f"Expected {V3_BETA_SCHEMA_VERSION}, got {bundle.metadata.get('version')}"
        )
    for field in ("preferred_actions", "action_utilities"):
        if bundle.metadata.get(field) is not False:
            errors.append(f"V3-beta metadata must explicitly disable {field}")
    if bundle.metadata.get("terminal_stop_budget_policy") != (
        "reserved_zero_cost_terminal_decision"
    ):
        errors.append("V3-beta must reserve a zero-cost terminal STOP decision")

    scenario_ids = {item.scenario_id for item in bundle.scenarios}
    provenance_scenarios = {item.scenario_id for item in bundle.provenance.values()}
    if scenario_ids != provenance_scenarios:
        errors.append("Scenario and provenance IDs do not match exactly")

    source_cache: dict[tuple[Path, str], tuple[str, set[str]]] = {}
    for item in bundle.provenance.values():
        source_path = PROJECT_DIR / item.source_path
        if not source_path.exists():
            errors.append(f"Missing provenance source: {source_path}")
            continue
        cache_key = (source_path, item.source_dataset)
        if cache_key not in source_cache:
            source_cache[cache_key] = (
                hashlib.sha256(source_path.read_bytes()).hexdigest(),
                _source_ids(source_path, item.source_dataset),
            )
        digest, source_ids = source_cache[cache_key]
        if digest != item.source_sha256:
            errors.append(f"Source hash mismatch: {item.provenance_id}")
        if item.source_item_id not in source_ids:
            errors.append(
                f"Source item is absent: {item.provenance_id}/{item.source_item_id}"
            )
        if item.human_review_status == "pending":
            warnings.append(f"Human review pending: {item.provenance_id}")

    real_ids = {
        item.scenario_id
        for item in bundle.scenarios
        if "real_task" in item.difficulty_tags
    }
    if set(bundle.task_inputs) != real_ids:
        errors.append("Every and only real-task scenario must have a task-input sidecar")
    if set(bundle.evaluation) != real_ids:
        errors.append("Every and only real-task scenario must have an evaluation sidecar")
    for scenario_id, item in bundle.evaluation.items():
        provenance = next(
            value for value in bundle.provenance.values() if value.scenario_id == scenario_id
        )
        if str(item.get("source_item_id")) != provenance.source_item_id:
            errors.append(f"Evaluation/source provenance mismatch: {scenario_id}")

    labels_by_signature: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    decision_ids: set[str] = set()
    for row in bundle.decision_rows:
        decision_id = str(row.get("decision_id"))
        if decision_id in decision_ids:
            errors.append(f"Duplicate decision ID: {decision_id}")
        decision_ids.add(decision_id)
        if row.get("schema_version") != V3_BETA_SCHEMA_VERSION:
            errors.append(f"Incorrect decision schema version: {decision_id}")
        for field in FORBIDDEN_LABELS:
            if field in row:
                errors.append(f"Forbidden label {field!r}: {decision_id}")
        candidates = set(map(str, row.get("candidate_actions") or ()))
        valid = set(map(str, row.get("valid_actions") or ()))
        if row.get("budget", {}).get("reserved_terminal_stop") is not True:
            errors.append(f"Reserved terminal STOP policy missing: {decision_id}")
        if not valid or not valid.issubset(candidates):
            errors.append(f"Invalid valid-action set: {decision_id}")
        rationales = row.get("action_rationales") or {}
        if set(rationales) != candidates:
            errors.append(f"Rationale coverage mismatch: {decision_id}")
        for action in candidates:
            rationale = rationales.get(action) or {}
            expected = "valid" if action in valid else "invalid"
            if rationale.get("status") != expected:
                errors.append(f"Rationale label mismatch: {decision_id}/{action}")
            if expected == "invalid" and not rationale.get("violated_preconditions"):
                errors.append(f"Invalid action lacks an explanation: {decision_id}/{action}")
        if bool(row.get("terminal_valid")) != (valid == {STOP_ACTION}):
            errors.append(f"Terminal label mismatch: {decision_id}")
        labels_by_signature[_public_signature(row)].add(tuple(sorted(valid)))

    collisions = [key for key, labels in labels_by_signature.items() if len(labels) > 1]
    if collisions:
        errors.append(
            f"Found {len(collisions)} observationally identical states with conflicting labels"
        )
    split_groups: dict[str, set[str]] = defaultdict(set)
    for item in bundle.scenarios:
        split_groups[item.split_group_id].add(item.split)
    crossing = [key for key, values in split_groups.items() if len(values) > 1]
    if crossing:
        errors.append(f"Split groups cross partitions: {crossing[:5]}")

    domain_splits: dict[str, set[str]] = defaultdict(set)
    for item in bundle.scenarios:
        if "source:mmlu-pro" not in item.difficulty_tags:
            continue
        domain = next(
            tag.split(":", 1)[1]
            for tag in item.difficulty_tags
            if tag.startswith("domain:")
        )
        domain_splits[domain].add(item.split)
    leaking_domains = [key for key, values in domain_splits.items() if len(values) > 1]
    if leaking_domains:
        errors.append(f"MMLU domain holdout leakage: {leaking_domains}")

    nonterminal = [item for item in bundle.decision_rows if not item["terminal_valid"]]
    multi = [item for item in nonterminal if len(item["valid_actions"]) >= 2]
    targets = {
        "at_least_100_workflows": len(bundle.scenarios) >= 100,
        "at_least_50_real_tasks": len(real_ids) >= 50,
        "multi_nonterminal_at_least_50_percent": (
            len(multi) / len(nonterminal) >= 0.50 if nonterminal else False
        ),
        "real_tasks_have_private_gold": set(bundle.evaluation) == real_ids,
        "real_tasks_have_actor_inputs": set(bundle.task_inputs) == real_ids,
    }
    if not all(targets.values()):
        errors.append(
            f"V3-beta quota targets failed: {[key for key, value in targets.items() if not value]}"
        )
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "workflow_count": len(bundle.scenarios),
        "real_task_workflow_count": len(real_ids),
        "decision_state_count": len(bundle.decision_rows),
        "family_counts": dict(sorted(Counter(item.family for item in bundle.scenarios).items())),
        "split_counts": dict(sorted(Counter(item.split for item in bundle.scenarios).items())),
        "provenance_count": len(bundle.provenance),
        "human_review_pending_count": sum(
            item.human_review_status == "pending" for item in bundle.provenance.values()
        ),
        "observability_collision_count": len(collisions),
        "domain_split_map": {key: sorted(value) for key, value in sorted(domain_splits.items())},
        "targets": targets,
    }
