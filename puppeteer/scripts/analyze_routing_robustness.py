from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _result_file(run_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in (run_dir / "results").glob("*.jsonl")
        if ".rejudged." not in path.name
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one primary result JSONL in {run_dir / 'results'}, "
            f"found {[path.name for path in candidates]}"
        )
    return candidates[0]


def _rows(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with _result_file(run_dir).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id") or row.get("id") or "")
            if task_id:
                rows[task_id] = row
    return rows


def _collaboration(row: dict[str, Any]) -> dict[str, Any]:
    direct = row.get("collaboration")
    if isinstance(direct, dict):
        return direct
    nested = (row.get("final_metrics") or {}).get("collaboration")
    return nested if isinstance(nested, dict) else {}


def _events_for(row: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    trace = Path(str(row.get("trace_path") or ""))
    candidates = [trace / "events.jsonl"] if str(trace) else []
    if not candidates or not candidates[0].exists():
        candidates = list((run_dir / "audit").glob("task-*/attempt-*/events.jsonl"))
    task_id = str(row.get("task_id") or row.get("id") or "")
    for path in candidates:
        events = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        if any(str(event.get("task_id") or "") == task_id for event in events):
            return events
    return []


def _task_metrics(
    row: dict[str, Any], run_dir: Path, noise_roles: set[str]
) -> dict[str, float]:
    collaboration = _collaboration(row)
    events = _events_for(row, run_dir) if noise_roles else []
    action_roles = [
        str(event.get("role") or "")
        for event in events
        if event.get("event_type") == "action_started"
    ]
    noise_calls = sum(role.casefold() in noise_roles for role in action_roles)
    total_calls = float(collaboration.get("total_calls") or len(action_roles) or 0)
    semantic = row.get("semantic_correct")
    if semantic is None:
        semantic = (row.get("semantic_outcome") or {}).get("correct")
    return {
        "exact_accuracy": float(bool(row.get("correct", False))),
        "semantic_accuracy": (
            float(bool(semantic)) if semantic is not None else float("nan")
        ),
        "answer_f1": float(row.get("answer_f1") or 0.0),
        "support_f1": float(
            row.get("paper_compatible_support_f1")
            or row.get("support_f1")
            or 0.0
        ),
        "collaboration_effectiveness": float(
            collaboration.get("collaboration_effectiveness") or 0.0
        ),
        "useful_call_ratio": float(collaboration.get("useful_call_ratio") or 0.0),
        "redundant_transition_rate": float(
            collaboration.get("redundant_transition_rate") or 0.0
        ),
        "recovery_success_rate": float(
            collaboration.get("recovery_success_rate") or 0.0
        ),
        "total_calls": total_calls,
        "noise_calls": float(noise_calls),
        "noise_selected": float(noise_calls > 0),
        "forked": float(len(collaboration.get("path_reports") or ()) > 1),
    }


def _aggregate(items: list[dict[str, float]]) -> dict[str, float | None]:
    if not items:
        return {}
    result: dict[str, float | None] = {}
    for key in items[0]:
        values = [item[key] for item in items]
        values = [value for value in values if value == value]
        result[key] = _mean(values)
    result["task_count"] = float(len(items))
    total_calls = sum(item["total_calls"] for item in items)
    result["noise_call_rate"] = (
        sum(item["noise_calls"] for item in items) / total_calls
        if total_calls
        else 0.0
    )
    return result


def compare_runs(
    clean_dir: Path, noisy_dir: Path, noise_roles: set[str]
) -> dict[str, Any]:
    clean_rows = _rows(clean_dir)
    noisy_rows = _rows(noisy_dir)
    task_ids = sorted(set(clean_rows) & set(noisy_rows))
    if not task_ids:
        raise ValueError("The two runs have no shared task IDs")
    clean_items = [
        _task_metrics(clean_rows[task_id], clean_dir, set()) for task_id in task_ids
    ]
    noisy_items = [
        _task_metrics(noisy_rows[task_id], noisy_dir, noise_roles)
        for task_id in task_ids
    ]
    clean = _aggregate(clean_items)
    noisy = _aggregate(noisy_items)
    delta = {
        key: (
            None
            if clean.get(key) is None or noisy.get(key) is None
            else float(noisy[key]) - float(clean[key])
        )
        for key in sorted(set(clean) | set(noisy))
        if key != "task_count"
    }
    clean_calls = float(clean.get("total_calls") or 0.0)
    noisy_calls = float(noisy.get("total_calls") or 0.0)
    return {
        "clean_run": str(clean_dir),
        "noisy_run": str(noisy_dir),
        "shared_task_count": len(task_ids),
        "noise_roles": sorted(noise_roles),
        "clean": clean,
        "noisy": noisy,
        "noisy_minus_clean": delta,
        "call_inflation_ratio": (
            noisy_calls / clean_calls if clean_calls else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare clean-pool and noisy-pool routing robustness."
    )
    parser.add_argument("clean_run", type=Path)
    parser.add_argument("noisy_run", type=Path)
    parser.add_argument(
        "--noise-role",
        action="append",
        default=[],
        help="Role name intentionally added as noise; repeat this option as needed.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare_runs(
        args.clean_run,
        args.noisy_run,
        {role.casefold() for role in args.noise_role},
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
