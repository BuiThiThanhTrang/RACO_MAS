from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ELIGIBLE_FAMILIES = {
    "multi_hop_evidence",
    "domain_reasoning_verification",
    "sequential_dependency",
    "conditional_branch",
    "conflict_resolution",
    "failure_recovery",
    "parallel_join",
    "terminal_control",
    "tool_routing",
    "budget_sensitive",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def audit_v3_sources(v3_root: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Classify V3 rows before they are considered as V4-beta task seeds.

    V3 role cards, transitions and labels are deliberately never copied.  The
    audit only exposes a task seed, workflow pattern and provenance pointer.
    """

    root = Path(v3_root)
    scenarios = _read_jsonl(root / "scenarios.jsonl")
    provenance = {
        str(item["scenario_id"]): item
        for item in _read_jsonl(root / "provenance.jsonl")
    }
    task_inputs = {
        str(item["scenario_id"]): dict(item.get("task_input") or {})
        for item in _read_jsonl(root / "task_inputs.jsonl")
    }
    seen_task_hashes: set[str] = set()
    manifest: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_id = str(scenario["scenario_id"])
        family = str(scenario.get("family", ""))
        task = " ".join(str(scenario.get("task", "")).split())
        task_hash = hashlib.sha256(task.casefold().encode("utf-8")).hexdigest()
        reasons: list[str] = []
        status = "eligible"
        if family not in ELIGIBLE_FAMILIES:
            status = "reject"
            reasons.append("family_not_in_v4beta_scope")
        if not task:
            status = "reject"
            reasons.append("missing_task_text")
        if task_hash in seen_task_hashes:
            status = "reject"
            reasons.append("duplicate_normalized_task")
        seen_task_hashes.add(task_hash)
        if family not in {"multi_hop_evidence", "domain_reasoning_verification"} and status == "eligible":
            status = "needs_rewrite"
            reasons.append("synthetic_v3_task_is_pattern_reference_only")
        source = provenance.get(scenario_id, {})
        manifest.append(
            {
                "v3_scenario_id": scenario_id,
                "v3_family": family,
                "audit_status": status,
                "audit_reasons": reasons,
                "task_seed": task,
                "task_input": task_inputs.get(scenario_id, {}),
                "source_provenance_id": source.get("provenance_id"),
                "source_dataset": source.get("source_dataset"),
                "source_item_id": source.get("source_item_id"),
                "source_license": source.get("license"),
                "task_sha256": task_hash,
            }
        )
    report = {
        "source_root": str(root),
        "source_scenario_count": len(scenarios),
        "status_counts": dict(sorted(Counter(item["audit_status"] for item in manifest).items())),
        "family_counts": dict(sorted(Counter(item["v3_family"] for item in manifest).items())),
        "policy": (
            "Only task seeds, workflow patterns and provenance may migrate. "
            "V3 role cards, transition graphs, valid-action labels and utilities are excluded."
        ),
    }
    return manifest, report


def write_audit(v3_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest, report = audit_v3_sources(v3_root)
    with (output / "v3_audit_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "v3_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def select_audited_rows(
    manifest: Iterable[dict[str, Any]], family: str, count: int
) -> list[dict[str, Any]]:
    rows = [
        item
        for item in manifest
        if item["v3_family"] == family and item["audit_status"] == "eligible"
    ]
    rows.sort(key=lambda item: (item["task_sha256"], item["v3_scenario_id"]))
    if len(rows) < count:
        raise ValueError(f"Need {count} eligible {family} rows, found {len(rows)}")
    return rows[:count]
