"""Evaluate router collaboration and semantic correctness from saved audit logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from role_aware.multiagentbench_cs import (
    COMBINED_PROMPT_VERSION,
    build_router_cs_semantic_trace,
    judge_router_cs_semantic,
    load_jsonl,
)
from role_aware.musique_llm_canonicalizer import original_question


def _attempts(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "audit").glob("task-*/attempt-*"))


def _load_existing(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    for record in load_jsonl(path):
        key = (str(record.get("task_id")), str(record.get("attempt_id")))
        existing[key] = record
    return existing


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [record for record in records if "collaboration_score_raw" in record]
    if not scored:
        return {"scored_tasks": 0}
    mean = lambda key: sum(float(row[key]) for row in scored) / len(scored)
    semantic = [record for record in scored if "semantic_correct" in record]
    return {
        "scored_tasks": len(scored),
        "routing_planning_score": mean("routing_planning_score"),
        "state_handoff_communication_score": mean(
            "state_handoff_communication_score"
        ),
        "collaboration_score_raw": mean("collaboration_score_raw"),
        "collaboration_score_100": mean("collaboration_score_100"),
        "judge_tokens": sum(int(row.get("judge_tokens") or 0) for row in scored),
        "semantic_scored_tasks": len(semantic),
        "semantic_accuracy": (
            sum(bool(row.get("semantic_correct")) for row in semantic) / len(semantic)
            if semantic
            else None
        ),
        "semantic_uncertain_rate": (
            sum(row.get("answer_verdict") == "UNCERTAIN" for row in semantic)
            / len(semantic)
            if semantic
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="MuSiQue run directory")
    parser.add_argument("--model", default="gpt-6-luna-openrouter")
    parser.add_argument("--reasoning_effort", default="low")
    parser.add_argument("--max_repair_attempts", type=int, default=1)
    parser.add_argument("--data_start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    run_dir = args.source.resolve()
    output_dir = run_dir / "analysis"
    output_path = output_dir / "multiagentbench_cs.jsonl"
    summary_path = output_dir / "multiagentbench_cs_summary.json"
    existing = {} if args.overwrite else _load_existing(output_path)
    attempts = _attempts(run_dir)[args.data_start :]
    if args.limit is not None:
        attempts = attempts[: args.limit]

    records = dict(existing)
    pending = []
    for attempt in attempts:
        candidates_path = attempt / "candidates.json"
        events_path = attempt / "events.jsonl"
        if not candidates_path.exists() or not events_path.exists():
            continue
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
        key = (str(candidates.get("task_id")), str(candidates.get("attempt_id")))
        if (
            key in existing
            and existing[key].get("prompt_version") == COMBINED_PROMPT_VERSION
        ):
            continue
        evaluation_path = attempt / "evaluation.json"
        if not evaluation_path.exists():
            continue
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        prediction = evaluation.get("prediction")
        accepted_answers = [
            evaluation.get("gold"),
            *(evaluation.get("answer_aliases") or ()),
        ]
        trace = build_router_cs_semantic_trace(
            candidates,
            load_jsonl(events_path),
            prediction=prediction,
            accepted_answers=accepted_answers,
            question=original_question(candidates.get("question") or ""),
        )
        official_success = bool(
            (evaluation.get("final_outcome") or {}).get("success", False)
        )
        pending.append((attempt, candidates, trace, official_success))

    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "cached": len(existing),
                    "pending": len(pending),
                    "sample_trace": pending[0][2] if pending else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    for attempt, candidates, trace, official_success in pending:
        score, _ = judge_router_cs_semantic(
            trace,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_repair_attempts=args.max_repair_attempts,
            official_success=official_success,
        )
        key = (str(candidates.get("task_id")), str(candidates.get("attempt_id")))
        records[key] = {
            "run_id": candidates.get("run_id"),
            "task_id": candidates.get("task_id"),
            "attempt_id": candidates.get("attempt_id"),
            "trace_path": str(attempt),
            "handoff_count": trace["collaboration_trace"]["handoff_count"],
            "committed_prediction": trace["answer_evaluation"][
                "committed_prediction"
            ],
            "official_correct": official_success,
            **score,
        }

    ordered_records = sorted(
        records.values(),
        key=lambda row: (str(row.get("task_id")), str(row.get("attempt_id"))),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in ordered_records
        ),
        encoding="utf-8",
    )
    summary = {
        "source": str(run_dir),
        "sidecar": str(output_path),
        "judge_model": args.model,
        **_summary(ordered_records),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
