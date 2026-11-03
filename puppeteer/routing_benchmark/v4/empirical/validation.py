from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl
from .schema import (
    EMPIRICAL_SCHEMA_VERSION,
    canonical_hash,
    validate_action_annotation,
    validate_empirical_output,
    validate_outcome,
)


FORBIDDEN_ROUTER_KEYS = {
    "valid_actions",
    "proposed_valid_actions",
    "human_valid_actions",
    "action_annotations",
    "preferred_actions",
    "action_utilities",
    "capability_scores",
    "fault_class",
    "failure_origin",
    "semantic_correctness",
    "judge_provenance",
    "executor_profile_id",
    "executor_model",
}


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_nested_keys(child))
    return keys


def validate_empirical_release(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    required = (
        "metadata.json",
        "empirical_outputs.jsonl",
        "outcome_interpretations.jsonl",
        "empirical_decision_states.jsonl",
        "action_preannotations.jsonl",
        "public_inputs.jsonl",
        "private_labels.jsonl",
        "human_review_queue.jsonl",
    )
    for name in required:
        if not (root / name).exists():
            errors.append(f"missing_file:{name}")
    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("version") != EMPIRICAL_SCHEMA_VERSION:
        errors.append("metadata_version_mismatch")
    outputs = read_jsonl(root / "empirical_outputs.jsonl")
    outcomes = read_jsonl(root / "outcome_interpretations.jsonl")
    decisions = read_jsonl(root / "empirical_decision_states.jsonl")
    annotations = read_jsonl(root / "action_preannotations.jsonl")
    public_rows = read_jsonl(root / "public_inputs.jsonl")
    private_rows = read_jsonl(root / "private_labels.jsonl")
    reviews = read_jsonl(root / "human_review_queue.jsonl")
    for record in outputs:
        try:
            validate_empirical_output(record)
        except ValueError as exc:
            errors.append(f"output:{record.get('output_id')}:{exc}")
    for record in outcomes:
        try:
            validate_outcome(record)
        except ValueError as exc:
            errors.append(f"outcome:{record.get('output_id')}:{exc}")
    for record in annotations:
        try:
            validate_action_annotation(record)
        except ValueError as exc:
            errors.append(f"action_annotation:{record.get('decision_id')}:{exc}")
    for row in public_rows:
        leaked = FORBIDDEN_ROUTER_KEYS.intersection(_nested_keys(row))
        if leaked:
            errors.append(
                f"public_input_leak:{row.get('decision_id')}:{','.join(sorted(leaked))}"
            )
        has_raw = "raw_empirical" in (row.get("observations") or {})
        if has_raw != (row.get("split") == "development"):
            errors.append(f"raw_empirical_split_policy:{row.get('decision_id')}")

    ids = {
        "outcomes": {str(item["output_id"]) for item in outcomes},
        "decisions": {str(item["decision_id"]) for item in decisions},
        "annotations": {str(item["decision_id"]) for item in annotations},
        "public": {str(item["decision_id"]) for item in public_rows},
        "private": {str(item["decision_id"]) for item in private_rows},
        "reviews": {str(item["decision_id"]) for item in reviews},
    }
    output_ids = {str(item["output_id"]) for item in outputs}
    if ids["outcomes"] != output_ids:
        warnings.append("not_all_empirical_outputs_have_interpretations")
    expected_decisions = ids["decisions"]
    for name in ("annotations", "public", "private", "reviews"):
        if ids[name] != expected_decisions:
            errors.append(f"decision_id_mismatch:{name}")

    # Observational-equivalence invariant: identical public input produces the
    # same human-final/proposed valid action set.
    labels = {str(item["decision_id"]): item for item in private_rows}
    by_observation: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in public_rows:
        decision_id = str(row["decision_id"])
        label = labels.get(decision_id, {})
        valid = label.get("human_valid_actions") or label.get("proposed_valid_actions") or ()
        by_observation[canonical_hash(row.get("observations"))].add(tuple(sorted(map(str, valid))))
    inconsistent = [key for key, values in by_observation.items() if len(values) > 1]
    if inconsistent:
        errors.append(f"observational_equivalence_violation:{len(inconsistent)}")

    pending = sum(
        item.get("reviewer_1_status") != "approved"
        or item.get("reviewer_2_status") != "approved"
        or item.get("adjudication_status") not in {"not_required", "resolved"}
        for item in reviews
    )
    if pending:
        warnings.append(f"human_review_pending:{pending}")
    return {
        "benchmark_version": EMPIRICAL_SCHEMA_VERSION,
        "valid": not errors,
        "release_ready": not errors and pending == 0,
        "human_review_complete": pending == 0,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "empirical_outputs": len(outputs),
            "outcomes": len(outcomes),
            "decision_states": len(decisions),
            "review_pending": pending,
        },
    }
