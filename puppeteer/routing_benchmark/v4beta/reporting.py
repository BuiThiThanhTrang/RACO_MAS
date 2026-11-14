from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | int | None]:
    if total <= 0:
        return {"successes": successes, "count": total, "rate": None, "ci95_low": None, "ci95_high": None}
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return {
        "successes": successes,
        "count": total,
        "rate": p,
        "ci95_low": max(0.0, centre - margin),
        "ci95_high": min(1.0, centre + margin),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _groups(
    rows: Iterable[dict[str, Any]], key: Callable[[dict[str, Any]], str], outcome: Callable[[dict[str, Any]], bool]
) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(bool(outcome(row)))
    return {
        name: wilson_interval(sum(values), len(values))
        for name, values in sorted(grouped.items())
    }


def analyze_runs(data_dir: str | Path, run_dirs: Iterable[str | Path]) -> dict[str, Any]:
    data_dir = Path(data_dir)
    scenarios = _read_jsonl(data_dir / "scenarios.jsonl")
    annotations = {
        str(item["scenario_id"]): item
        for item in _read_jsonl(data_dir / "recovery_annotations.jsonl")
    }
    scenario_meta = {
        str(item["scenario_id"]): {
            "task_domain": item["family"],
            "workflow_pattern": annotations[str(item["scenario_id"])]["workflow_pattern"],
            "fault_variant": annotations[str(item["scenario_id"])]["variant"],
        }
        for item in scenarios
    }
    checkpoint_meta = {
        str(item["checkpoint_id"]): {
            "task_domain": item.get("task_domain", item.get("workflow_family", "unknown")),
            "workflow_pattern": item.get("workflow_pattern", "unknown"),
            "fault_variant": item.get("fault_class", "unknown"),
        }
        for item in _read_jsonl(data_dir / "post_fault_labels.jsonl")
    }
    report: dict[str, Any] = {
        "benchmark_version": "4.0-beta",
        "interval": "Wilson 95%",
        "composite_score": None,
        "runs": {},
    }
    for run_path in map(Path, run_dirs):
        summary_path = run_path / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        run_report: dict[str, Any] = {"summary": summary, "tracks": {}}
        for track, filename, outcome_field in (
            ("A", "track_a_records.jsonl", "valid"),
            ("B", "track_b_episodes.jsonl", "success"),
            ("C", "track_c_episodes.jsonl", "success"),
        ):
            rows = _read_jsonl(run_path / filename)
            if not rows:
                continue
            def meta(row: dict[str, Any], field: str) -> str:
                scenario_id = str(row.get("scenario_id") or row.get("internal_scenario_id") or "")
                checkpoint_id = str(row.get("checkpoint_id") or "")
                if field == "task_domain":
                    return str(
                        row.get("task_domain")
                        or row.get("workflow_family")
                        or scenario_meta.get(scenario_id, {}).get(field)
                        or checkpoint_meta.get(checkpoint_id, {}).get(field, "unknown")
                    )
                return str(
                    row.get(field)
                    or scenario_meta.get(scenario_id, {}).get(field)
                    or checkpoint_meta.get(checkpoint_id, {}).get(field, "unknown")
                )
            outcome = lambda row, field=outcome_field: bool(row.get(field))
            micro = wilson_interval(sum(outcome(row) for row in rows), len(rows))
            by_domain = _groups(rows, lambda row: meta(row, "task_domain"), outcome)
            macro_values = [value["rate"] for value in by_domain.values() if value["rate"] is not None]
            run_report["tracks"][track] = {
                "micro": micro,
                "macro_domain_rate": sum(macro_values) / len(macro_values) if macro_values else None,
                "by_task_domain": by_domain,
                "by_workflow_pattern": _groups(rows, lambda row: meta(row, "workflow_pattern"), outcome),
                "by_fault_variant": _groups(rows, lambda row: meta(row, "fault_variant"), outcome),
                "by_failure_attribution": _groups(
                    rows,
                    lambda row: str(row.get("failure_attribution", "not_reported")),
                    outcome,
                ),
            }
        report["runs"][run_path.name] = run_report
    return report
