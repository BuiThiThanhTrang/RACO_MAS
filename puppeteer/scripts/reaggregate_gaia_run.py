from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from role_aware.aggregation import (
    aggregate_gaia_candidates,
    select_gaia_path_answer,
)
from tasks.evaluator import BenchmarkEvaluator


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def _answer_from_path(path_record):
    before = path_record.get("before_aggregation")
    if before is not None and str(before).strip():
        return before

    step_answers = []
    for step in path_record.get("steps", []):
        result = step.get("result") or {}
        step_answers.append(result.get("answer"))
    return select_gaia_path_answer(step_answers)


def path_last_answers(snapshot):
    answers = []
    for path_record in snapshot.get("paths", []):
        answer = _answer_from_path(path_record)
        if answer is not None and str(answer).strip():
            answers.append(answer)
    return answers


def _find_result_file(run_dir: Path):
    candidates = [
        path
        for path in (run_dir / "results").glob("GAIA_*.jsonl")
        if "submission" not in path.name and "reaggregated" not in path.name
    ]
    if len(candidates) != 1:
        raise SystemExit(
            "Expected exactly one GAIA result JSONL. Use --result-file when the "
            f"run has {len(candidates)} candidates."
        )
    return candidates[0]


def _resolve_trace_path(run_dir: Path, record):
    trace_path = record.get("trace_path")
    if not trace_path:
        raise ValueError(f"Result {record.get('task_id')} has no trace_path")
    path = Path(trace_path)
    if not path.is_absolute():
        path = run_dir / path
    return path


def replay_record(record, snapshot, seed=42, tie_policy="keep-original"):
    path_answers = path_last_answers(snapshot)
    aggregation = aggregate_gaia_candidates(
        path_answers,
        mode="majority",
        seed=seed,
        task_id=str(record.get("task_id", "")),
        question=snapshot.get("question", ""),
    )
    original = BenchmarkEvaluator.extract_gaia_answer(record.get("model_answer", ""))
    prediction = aggregation["prediction"]
    tie_resolution = "majority"
    if aggregation["tie"] and tie_policy == "keep-original":
        prediction = original
        tie_resolution = "kept_original"
    elif aggregation["tie"]:
        tie_resolution = "seeded"

    updated = dict(record)
    updated["original_model_answer"] = original
    updated["model_answer"] = BenchmarkEvaluator.extract_gaia_answer(prediction)
    updated["reaggregation"] = {
        "path_policy": "last_nonempty_answer",
        "tie_policy": tie_policy,
        "tie_resolution": tie_resolution,
        "path_answers": [
            BenchmarkEvaluator.extract_gaia_answer(answer) for answer in path_answers
        ],
        "normalized": aggregation["normalized"],
        "votes": aggregation["votes"],
        "changed": updated["model_answer"] != original,
    }
    gold = record.get("answer")
    if gold is not None:
        correct = BenchmarkEvaluator.check_gaia(updated["model_answer"], gold)
        updated["correct"] = correct
        updated["final_success"] = correct
        updated["final_reward"] = 1.0 if correct else -1.0
    return updated


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Replay GAIA aggregation from immutable audit logs without rerunning "
            "the router, actors, or tools. The original result file is never changed."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tie-policy",
        choices=["keep-original", "seeded"],
        default="keep-original",
        help=(
            "keep-original avoids changing unresolved cross-path ties; seeded "
            "uses the deterministic seed-based fallback."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    result_file = (args.result_file or _find_result_file(run_dir)).resolve()
    output = args.output or result_file.with_name(
        f"{result_file.stem}.last_path_reaggregated.jsonl"
    )
    output = output.resolve()
    if output == result_file:
        raise SystemExit("Refusing to overwrite the original result file.")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}. Pass --overwrite to replace it.")

    records = list(_read_jsonl(result_file))
    replayed = []
    for record in records:
        trace_path = _resolve_trace_path(run_dir, record)
        snapshot_path = trace_path / "candidates.json"
        if not snapshot_path.is_file():
            raise SystemExit(f"Missing aggregation audit: {snapshot_path}")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        replayed.append(
            replay_record(record, snapshot, seed=args.seed, tie_policy=args.tie_policy)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for record in replayed:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    evaluable = [record for record in replayed if record.get("answer") is not None]
    original_correct = sum(
        BenchmarkEvaluator.check_gaia(record["original_model_answer"], record["answer"])
        for record in evaluable
    )
    replay_correct = sum(bool(record.get("correct")) for record in evaluable)
    changed = sum(record["reaggregation"]["changed"] for record in replayed)
    ties = Counter(
        record["reaggregation"]["tie_resolution"] for record in replayed
    )
    summary = {
        "source": str(result_file),
        "output": str(output),
        "records": len(replayed),
        "changed": changed,
        "evaluable": len(evaluable),
        "original_correct": original_correct,
        "reaggregated_correct": replay_correct,
        "original_accuracy": original_correct / len(evaluable) if evaluable else None,
        "reaggregated_accuracy": replay_correct / len(evaluable) if evaluable else None,
        "tie_resolutions": dict(ties),
        "post_hoc": True,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
