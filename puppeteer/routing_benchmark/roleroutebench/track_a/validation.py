from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .canonicalize import canonical_json, read_json, read_jsonl, sha256_text, write_json
from .schema import validate_prelabel
from .source_manifest import PROJECT


def validate_release(root: Path) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    manifest = read_json(root / "source_manifest.json")
    raw = read_jsonl(root / "raw_states.jsonl")
    primary = read_jsonl(root / "track_a_states.jsonl")
    audit = read_jsonl(root / "audit_only_states.jsonl")
    provenance = read_jsonl(root / "track_a_provenance.jsonl")
    expected = manifest["expected_counts"]
    actual = {"raw": len(raw), "primary": len(primary), "audit_only": len(audit)}
    for key, value in actual.items():
        if value != expected[key]:
            errors.append(f"{key} count: {value} != {expected[key]}")
    if len(provenance) != len(raw):
        errors.append("provenance count differs from raw count")
    if {row["decision_id"] for row in raw} != {row["decision_id"] for row in provenance}:
        errors.append("provenance decision IDs differ from raw states")
    provenance_by_id = {row["decision_id"]: row for row in provenance}
    source_cache: dict[str, dict[str, dict[str, Any]]] = {}
    hashes: list[str] = []
    for row in raw:
        try:
            observation = json.loads(row["router_input_json"])
            candidates = json.loads(row["candidate_actions_json"])
        except Exception as exc:
            errors.append(f"invalid JSON in {row.get('decision_id')}: {exc}")
            continue
        digest = sha256_text(canonical_json(observation))
        hashes.append(digest)
        if digest != row["public_state_sha256"]:
            errors.append(f"state hash mismatch: {row['decision_id']}")
        if candidates != observation.get("constraints", {}).get("allowed_actions"):
            errors.append(f"candidate mismatch: {row['decision_id']}")
        source_rel = provenance_by_id.get(row["decision_id"], {}).get("source_state_file")
        if source_rel:
            if source_rel not in source_cache:
                source_cache[source_rel] = {
                    item["decision_id"]: item for item in read_jsonl(PROJECT / source_rel)
                }
            source = source_cache[source_rel].get(row["decision_id"])
            if source is None or canonical_json(source.get("observation")) != row["router_input_json"]:
                errors.append(f"router payload differs from exact source observation: {row['decision_id']}")
        forbidden = {"valid_actions", "selected_action", "action_as_run", "final_result"}
        if forbidden.intersection(observation):
            errors.append(f"private/outcome field in router payload: {row['decision_id']}")
    if len(set(row["public_state_sha256"] for row in primary)) != len(primary):
        errors.append("duplicate primary public-state fingerprint")
    domain_counts = dict(Counter(row["domain"] for row in primary))
    if domain_counts != manifest["expected_primary_by_domain"]:
        errors.append(f"domain counts mismatch: {domain_counts}")
    for item in manifest["input_files"]:
        if not (PROJECT / item["path"]).exists():
            errors.append(f"missing source file: {item['path']}")
    cohort_roots = [root / "labeling"] + sorted(
        path for path in root.glob("labeling_*") if (path / "sample_manifest.json").exists()
    )
    all_sample_ids: set[str] = set()
    all_attempt_ids: set[str] = set()
    all_prelabels: list[dict[str, Any]] = []
    cohort_counts: dict[str, dict[str, int]] = {}
    for cohort_root in cohort_roots:
        cohort_name = cohort_root.name
        sample_manifest = read_json(cohort_root / "sample_manifest.json")
        expected_sample_count = int(sample_manifest["sample_count"])
        sample_states = read_jsonl(cohort_root / "sample_states.jsonl")
        prelabels = read_jsonl(cohort_root / "llm_prelabels.jsonl")
        attempts = read_jsonl(cohort_root / "llm_attempts.jsonl")
        sample_ids = {row["decision_id"] for row in sample_states}
        prelabel_ids = [row["decision_id"] for row in prelabels]
        attempt_ids = {row["decision_id"] for row in attempts}
        if len(sample_states) != expected_sample_count or len(sample_ids) != expected_sample_count:
            errors.append(
                f"{cohort_name}: expected {expected_sample_count} unique sampled states, "
                f"got {len(sample_states)}/{len(sample_ids)}"
            )
        overlap = all_sample_ids.intersection(sample_ids)
        if overlap:
            errors.append(f"{cohort_name}: sample overlaps prior cohort: {sorted(overlap)[:5]}")
        if len(prelabel_ids) != len(set(prelabel_ids)):
            errors.append(f"{cohort_name}: duplicate successful prelabel records")
        if prelabels and set(prelabel_ids) != sample_ids:
            errors.append(f"{cohort_name}: successful prelabels do not exactly cover sample")
        if not attempt_ids.issubset(sample_ids):
            errors.append(f"{cohort_name}: LLM attempt exists outside sample")
        all_sample_ids.update(sample_ids)
        all_attempt_ids.update(attempt_ids)
        all_prelabels.extend(prelabels)
        cohort_counts[cohort_name] = {
            "sample_states": len(sample_states),
            "prelabels": len(prelabels),
            "llm_attempts": len(attempts),
        }
    primary_ids = {row["decision_id"] for row in primary}
    expected_without_attempt = len(primary_ids) - len(all_sample_ids)
    if len(primary_ids - all_attempt_ids) != expected_without_attempt:
        errors.append(
            f"expected {expected_without_attempt} primary states without LLM attempts, "
            f"got {len(primary_ids - all_attempt_ids)}"
        )
    state_by_id = {row["decision_id"]: row for row in primary}
    for label in all_prelabels:
        state = state_by_id.get(label["decision_id"])
        if state is None:
            errors.append(f"prelabel without primary state: {label['decision_id']}")
            continue
        if label["public_state_sha256"] != state["public_state_sha256"]:
            errors.append(f"prelabel state hash mismatch: {label['decision_id']}")
        candidates = json.loads(state["candidate_actions_json"])
        errors.extend(
            f"{label['decision_id']}: {error}"
            for error in validate_prelabel(label["classification"], candidates)
        )
    report = {
        "valid": not errors,
        "errors": errors,
        "counts": {
            **actual,
            "sample_states": len(all_sample_ids),
            "prelabels": len(all_prelabels),
            "llm_attempts": sum(item["llm_attempts"] for item in cohort_counts.values()),
            "primary_without_llm_attempt": len(primary_ids - all_attempt_ids),
        },
        "cohorts": cohort_counts,
        "primary_by_domain": domain_counts,
    }
    write_json(root / "validation_report.json", report)
    return report

