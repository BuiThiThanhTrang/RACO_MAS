"""Rejudge MuSiQue path outputs with a gold-blind LLM canonicalizer.

The original run and result JSONL are never modified. Gold labels are read only
after the LLM has committed its answer and are used solely for the summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from role_aware.musique_llm_canonicalizer import canonicalize_with_llm
from tasks.evaluator import BenchmarkEvaluator


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _find_result_file(run_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in (run_dir / "results").glob("MuSiQue_*.jsonl")
        if "rejudged" not in path.name
    )
    if len(candidates) != 1:
        raise SystemExit(
            "Expected exactly one original MuSiQue result JSONL; pass "
            f"--result_file when {len(candidates)} files are present."
        )
    return candidates[0]


def _output_paths(result_file: Path, model: str) -> tuple[Path, Path]:
    model_slug = "".join(
        character if character.isalnum() else "_" for character in model
    ).strip("_")
    output = result_file.with_name(f"{result_file.stem}.{model_slug}.rejudged.jsonl")
    return output, output.with_suffix(".summary.json")


def _trace_path(run_dir: Path, record: dict[str, Any]) -> Path:
    path = Path(str(record.get("trace_path") or ""))
    if not path.is_absolute():
        path = run_dir / path
    return path


def _evaluate(record: dict[str, Any], prediction: str) -> dict[str, Any]:
    # Gold is deliberately accessed only after the model answer is committed.
    aliases = [record.get("answer", ""), *(record.get("answer_aliases") or [])]
    return BenchmarkEvaluator.musique_answer_scores(prediction, aliases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--result_file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="gpt-6-luna-openrouter")
    parser.add_argument("--reasoning_effort", default="low")
    parser.add_argument("--max_repair_attempts", type=int, default=1)
    parser.add_argument("--data_start", type=int, default=0)
    parser.add_argument("--data_limit", type=int, default=None)
    parser.add_argument(
        "--include_candidate_context",
        action="store_true",
        help="Also send candidate paragraphs so Luna can select between conflicting paths.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted output and skip task IDs already written.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    result_file = (args.result_file or _find_result_file(run_dir)).resolve()
    default_output, default_summary = _output_paths(result_file, args.model)
    output = (args.output or default_output).resolve()
    summary_path = (
        output.with_suffix(".summary.json")
        if args.output is not None
        else default_summary.resolve()
    )
    if output == result_file:
        raise SystemExit("Refusing to overwrite the original result file")
    if args.overwrite and args.resume:
        raise SystemExit("Use only one of --overwrite or --resume")
    if output.exists() and not (args.overwrite or args.resume):
        raise SystemExit(
            f"Output already exists: {output}; pass --resume or --overwrite"
        )

    records = _read_jsonl(result_file)
    selected = records[args.data_start :]
    if args.data_limit is not None:
        selected = selected[: args.data_limit]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "source": str(result_file),
                    "output": str(output),
                    "records": len(selected),
                    "model": args.model,
                    "include_candidate_context": args.include_candidate_context,
                },
                indent=2,
            )
        )
        return

    existing = _read_jsonl(output) if output.exists() and args.resume else []
    existing_by_task = {str(row.get("task_id")): row for row in existing}
    rejudged_by_task = dict(existing_by_task)
    total_tokens = sum(
        int((row.get("llm_canonicalization") or {}).get("tokens") or 0)
        for row in existing
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume or not output.exists():
        output.write_text("", encoding="utf-8")
    for record in selected:
        task_id = str(record.get("task_id"))
        if task_id in existing_by_task:
            continue
        snapshot_path = _trace_path(run_dir, record) / "candidates.json"
        if not snapshot_path.is_file():
            raise SystemExit(f"Missing candidate snapshot: {snapshot_path}")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        llm_result, tokens = canonicalize_with_llm(
            snapshot,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            include_candidate_context=args.include_candidate_context,
            max_repair_attempts=args.max_repair_attempts,
        )
        prediction = llm_result["answer"]
        total_tokens += tokens
        scores = _evaluate(record, prediction)
        updated = dict(record)
        final_metrics = dict(record.get("final_metrics") or {})
        final_metrics.update(
            {"answer_em": scores["answer_em"], "answer_f1": scores["answer_f1"]}
        )
        updated.update(
            {
                "original_model_answer": record.get("model_answer", ""),
                "model_answer": prediction,
                "correct": scores["answer_em"] == 1.0,
                "answer_em": scores["answer_em"],
                "answer_f1": scores["answer_f1"],
                "final_success": scores["answer_em"] == 1.0,
                "final_reward": 1.0 if scores["answer_em"] == 1.0 else -1.0,
                "final_metrics": final_metrics,
                "llm_canonicalization": llm_result,
                "post_hoc": True,
            }
        )
        rejudged_by_task[task_id] = updated
        with output.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(updated, ensure_ascii=False) + "\n")

    rejudged = [
        rejudged_by_task[str(record.get("task_id"))]
        for record in selected
        if str(record.get("task_id")) in rejudged_by_task
    ]
    original_correct = sum(bool(row.get("correct")) for row in selected)
    rejudged_correct = sum(bool(row.get("correct")) for row in rejudged)
    changed = sum(
        str(row.get("original_model_answer") or "").strip()
        != str(row.get("model_answer") or "").strip()
        for row in rejudged
    )
    summary = {
        "source": str(result_file),
        "output": str(output),
        "records": len(rejudged),
        "requested_records": len(selected),
        "resumed_records": len(existing_by_task),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "include_candidate_context": args.include_candidate_context,
        "changed": changed,
        "original_correct": original_correct,
        "rejudged_correct": rejudged_correct,
        "original_accuracy": original_correct / len(selected) if selected else None,
        "rejudged_accuracy": rejudged_correct / len(rejudged) if rejudged else None,
        "tokens": total_tokens,
        "post_hoc": True,
        "gold_visible_to_llm": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
