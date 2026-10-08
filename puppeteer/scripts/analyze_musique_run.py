"""Aggregate answer, evidence, and collaboration metrics for MuSiQue runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from role_aware.collaboration_metrics import evaluate_musique_snapshot


COLLABORATION_RATES = (
    "milestone_achievement_rate",
    "useful_handoff_rate",
    "collaboration_effectiveness",
    "recovery_success_rate",
    "redundant_transition_rate",
    "useful_call_ratio",
)


def _mean(values: Iterable[float | int | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _resolve_result_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    candidates = sorted((source / "results").glob("MuSiQue_*.jsonl"))
    if not candidates:
        candidates = sorted(source.glob("MuSiQue_*.jsonl"))
    canonical = [path for path in candidates if ".rejudged." not in path.name]
    if canonical:
        candidates = canonical
    if not candidates:
        raise FileNotFoundError(
            f"No MuSiQue result JSONL found in {source} or {source / 'results'}"
        )
    return candidates


def _load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if record.get("paper_compatible_support_f1") is None:
                trace_value = str(record.get("trace_path") or "").strip()
                trace_path = Path(trace_value) if trace_value else None
                candidates_path = trace_path / "candidates.json" if trace_path else None
                evaluation_path = trace_path / "evaluation.json" if trace_path else None
                if (
                    candidates_path is not None
                    and evaluation_path is not None
                    and candidates_path.is_file()
                    and evaluation_path.is_file()
                ):
                    snapshot = json.loads(candidates_path.read_text(encoding="utf-8"))
                    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
                    collaboration = evaluate_musique_snapshot(snapshot, evaluation)
                    for name in (
                        "paper_compatible_support_precision",
                        "paper_compatible_support_recall",
                        "paper_compatible_support_f1",
                        "paper_compatible_supporting_paragraphs",
                        "paper_compatible_support_source",
                    ):
                        record[name] = collaboration.get(name)
            record["_result_file"] = str(path)
            records.append(record)
    if not records:
        raise ValueError("MuSiQue result files contain no records")
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    role_contributions: Counter[str] = Counter()
    for record in records:
        role_contributions.update(
            (record.get("collaboration") or {}).get("contribution_by_role") or {}
        )

    collaboration = {
        name: _mean(
            (record.get("collaboration") or {}).get(name) for record in records
        )
        for name in COLLABORATION_RATES
    }
    collaboration.update(
        {
            "mean_calls_per_task": _mean(
                (record.get("collaboration") or {}).get("total_calls")
                for record in records
            ),
            "total_handoffs": sum(
                int((record.get("collaboration") or {}).get("handoff_count") or 0)
                for record in records
            ),
            "total_useful_handoffs": sum(
                int(
                    (record.get("collaboration") or {}).get(
                        "useful_handoff_count"
                    )
                    or 0
                )
                for record in records
            ),
            "contribution_by_role": dict(role_contributions.most_common()),
        }
    )
    semantic_records = [
        record for record in records if record.get("semantic_correct") is not None
    ]
    router_cs_records = [
        record.get("router_cs")
        for record in records
        if isinstance(record.get("router_cs"), dict)
    ]
    return {
        "tasks": len(records),
        "correct": sum(bool(record.get("correct")) for record in records),
        "semantic_correct": sum(
            bool(record.get("semantic_correct")) for record in semantic_records
        ),
        "semantic_accuracy": (
            _mean(record.get("semantic_correct") for record in semantic_records)
            if semantic_records
            else None
        ),
        "answer_em": _mean(record.get("answer_em") for record in records),
        "answer_f1": _mean(record.get("answer_f1") for record in records),
        "support_precision": _mean(
            record.get("support_precision") for record in records
        ),
        "support_recall": _mean(record.get("support_recall") for record in records),
        "support_f1": _mean(record.get("support_f1") for record in records),
        "paper_compatible_support_precision": _mean(
            record.get("paper_compatible_support_precision") for record in records
        ),
        "paper_compatible_support_recall": _mean(
            record.get("paper_compatible_support_recall") for record in records
        ),
        "paper_compatible_support_f1": _mean(
            record.get("paper_compatible_support_f1") for record in records
        ),
        "collaboration": collaboration,
        "router_cs": {
            "scored_tasks": len(router_cs_records),
            "routing_planning_score": _mean(
                record.get("routing_planning_score") for record in router_cs_records
            ),
            "state_handoff_communication_score": _mean(
                record.get("state_handoff_communication_score")
                for record in router_cs_records
            ),
            "collaboration_score_100": _mean(
                record.get("collaboration_score_100") for record in router_cs_records
            ),
            "judge_tokens": sum(
                int(record.get("judge_tokens") or 0) for record in router_cs_records
            ),
        },
    }


def build_report(source: Path) -> dict[str, Any]:
    result_files = _resolve_result_files(source)
    records = _load_records(result_files)
    by_hop: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_hop[int(record.get("hop_count") or 0)].append(record)
    report = {
        "source": str(source.resolve()),
        "result_files": [str(path.resolve()) for path in result_files],
        "overall": summarize_records(records),
        "by_hop_count": {
            str(hop): summarize_records(group)
            for hop, group in sorted(by_hop.items())
        },
    }
    run_dir = source if source.is_dir() else source.parent.parent
    cs_summary = run_dir / "analysis" / "multiagentbench_cs_summary.json"
    if cs_summary.exists():
        report["multiagentbench_cs"] = json.loads(
            cs_summary.read_text(encoding="utf-8")
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="Run directory, results directory, or MuSiQue result JSONL",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = build_report(args.source)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
