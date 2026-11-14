from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from tasks.progress import (
    complete_item,
    finalize_run,
    register_result_path,
    resolve_data_window,
)


DEFAULT_DATA_ROOT = Path("data") / "MuSiQue"
SPLIT_ALIASES = {
    "train": "train",
    "dev": "validation",
    "validation": "validation",
    "test": "test",
    "final": "test",
}


def _official_split(mode: str) -> str:
    try:
        return SPLIT_ALIASES[str(mode).lower()]
    except KeyError as error:
        raise ValueError(
            "MuSiQue supports train, validation/dev, and test/final splits."
        ) from error


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        value = [value]
    return [dict(item) for item in value]


def _values(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        return [value]
    return list(value)


def load_dataset(
    mode,
    data_limit=None,
    seed=42,
    data_start=0,
    hop_count=None,
    data_root=DEFAULT_DATA_ROOT,
    task_ids=None,
):
    del seed  # MuSiQue uses official immutable splits.
    if hop_count is not None and int(hop_count) not in {2, 3, 4}:
        raise ValueError("MuSiQue hop_count must be 2, 3, 4, or omitted")
    split = _official_split(mode)
    source = Path(data_root) / f"{split}.parquet"
    if not source.is_file():
        raise FileNotFoundError(
            f"MuSiQue split not found: {source}. Run "
            f"python -m scripts.download_musique --split {split}."
        )
    data = pd.read_parquet(source)
    required = {
        "id",
        "question",
        "paragraphs",
        "question_decomposition",
        "answer",
        "answer_aliases",
        "answerable",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"MuSiQue data is missing columns: {missing}")
    data = data[data["answerable"].astype(bool)].copy()
    data["hop_count"] = data["question_decomposition"].map(
        lambda value: len(_records(value))
    )
    if hop_count is not None:
        data = data[data["hop_count"] == int(hop_count)]
    if task_ids is not None:
        ordered_ids = [str(value) for value in task_ids]
        indexed = {
            str(row["id"]): row
            for _, row in data.iterrows()
        }
        missing_ids = [task_id for task_id in ordered_ids if task_id not in indexed]
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise ValueError(
                f"Retry source contains {len(missing_ids)} task IDs not present "
                f"in MuSiQue {split}: {preview}"
            )
        data = pd.DataFrame([indexed[task_id] for task_id in ordered_ids])
    data = data.iloc[int(data_start):]
    if data_limit is not None:
        data = data.iloc[: int(data_limit)]
    return data.reset_index(drop=True)


def _resolve_retry_result_path(source: str | os.PathLike[str]) -> Path:
    candidate = Path(source)
    if not candidate.exists():
        candidate = Path("runs") / candidate
    if candidate.is_file():
        if candidate.suffix.casefold() != ".jsonl":
            raise ValueError("--retry_failed_from must resolve to a JSONL result")
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"Retry source not found: {source}")
    results_dir = candidate / "results" if (candidate / "results").is_dir() else candidate
    matches = sorted(results_dir.glob("MuSiQue_*.jsonl"))
    if len(matches) != 1:
        raise ValueError(
            f"Retry source must contain exactly one MuSiQue JSONL; found "
            f"{len(matches)} in {results_dir}"
        )
    return matches[0]


def load_failed_retry_records(source: str | os.PathLike[str]) -> list[dict[str, Any]]:
    result_path = _resolve_retry_result_path(source)
    failed: list[dict[str, Any]] = []
    seen: set[str] = set()
    with result_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {result_path}:{line_number}"
                ) from error
            if record.get("correct") is not False:
                continue
            task_id = str(record.get("id") or record.get("task_id") or "").strip()
            if not task_id or task_id in seen:
                continue
            seen.add(task_id)
            failed.append(
                {
                    "id": task_id,
                    "offset": record.get("offset"),
                    "source_result": str(result_path.resolve()),
                }
            )
    if not failed:
        raise ValueError(f"Retry source has no records with correct=false: {result_path}")
    return failed


def format_question(row, idx=None):
    paragraphs = sorted(_records(row["paragraphs"]), key=lambda item: int(item["idx"]))
    decomposition = _records(row["question_decomposition"])
    context = "\n\n".join(
        f"[P{int(item['idx'])}] {str(item['title']).strip()}\n"
        f"{str(item['paragraph_text']).strip()}"
        for item in paragraphs
    )
    question = str(row["question"]).strip()
    prompt = (
        "Answer the multi-hop question using only the candidate paragraphs below. "
        "Preserve paragraph IDs in intermediate evidence. Do not assume that every "
        "paragraph is relevant. When a complete candidate is available, end with "
        "FINAL ANSWER: <short answer>.\n\n"
        f"Question: {question}\n\nCandidate paragraphs:\n{context}"
    )
    answer = str(row["answer"]).strip()
    aliases = [str(value).strip() for value in _values(row["answer_aliases"])]
    support_indices = [
        int(item["paragraph_support_idx"])
        for item in decomposition
        if item.get("paragraph_support_idx") is not None
    ]
    task_id = str(row.get("id") or idx)
    hop_count = int(row.get("hop_count") or len(decomposition))
    return {
        "type": "MuSiQue",
        "Question": prompt,
        "id": task_id,
        "category": f"{hop_count}_hop",
        "hop_count": hop_count,
        "paragraph_ids": [int(item["idx"]) for item in paragraphs],
        # Evaluation-only fields are stripped before GlobalInfo is constructed.
        "Answer": answer,
        "answer_aliases": aliases,
        "gold_decomposition": decomposition,
        "supporting_paragraph_indices": support_indices,
    }


def run(
    runner,
    evaluator,
    results_dir,
    mode,
    data_limit=None,
    data_start=0,
    seed=42,
    hop_count=None,
    retry_failed_from=None,
):
    effective_start, effective_limit = resolve_data_window(
        runner, data_start, data_limit
    )
    retry_records = (
        load_failed_retry_records(retry_failed_from)
        if retry_failed_from is not None
        else []
    )
    retry_ids = [record["id"] for record in retry_records]
    retry_offsets = {record["id"]: record.get("offset") for record in retry_records}
    dataset = load_dataset(
        mode,
        effective_limit,
        seed=seed,
        data_start=effective_start,
        hop_count=hop_count,
        task_ids=retry_ids if retry_records else None,
    )
    split = _official_split(mode)
    hop_suffix = "all_hops" if hop_count is None else f"{int(hop_count)}hop"
    result_path = os.path.join(
        results_dir, f"MuSiQue_{split}_{hop_suffix}.jsonl"
    )
    file_mode = register_result_path(runner, result_path)
    with open(result_path, file_mode, encoding="utf-8") as destination:
        for index, (_, row) in enumerate(tqdm(dataset.iterrows(), total=len(dataset))):
            default_offset = effective_start + index
            source_offset = retry_offsets.get(str(row.get("id")))
            absolute_offset = (
                int(source_offset)
                if source_offset is not None
                else default_offset
            )
            task = format_question(row, absolute_offset)
            prediction = runner.run_reasoning(task)
            metadata = dict(getattr(runner, "last_result_metadata", {}) or {})
            metrics = dict(metadata.get("final_metrics") or {})
            answer_scores = evaluator.musique_answer_scores(
                prediction, [task["Answer"], *task["answer_aliases"]]
            )
            success = metadata.get("final_success")
            if success is None:
                success = bool(answer_scores["answer_em"])
            semantic_success = metadata.get("semantic_success")
            if semantic_success is None:
                semantic_success = bool(success)
            record = {
                "id": task["id"],
                "model_answer": evaluator.extract_musique_answer(prediction),
                "answer": task["Answer"],
                "answer_aliases": task["answer_aliases"],
                "hop_count": task["hop_count"],
                "correct": bool(success),
                "official_correct": bool(answer_scores["answer_em"]),
                "semantic_correct": bool(semantic_success),
                "answer_em": float(answer_scores["answer_em"]),
                "answer_f1": float(answer_scores["answer_f1"]),
                "support_precision": metrics.get("support_precision"),
                "support_recall": metrics.get("support_recall"),
                "support_f1": metrics.get("support_f1"),
                "paper_compatible_support_precision": metrics.get(
                    "paper_compatible_support_precision"
                ),
                "paper_compatible_support_recall": metrics.get(
                    "paper_compatible_support_recall"
                ),
                "paper_compatible_support_f1": metrics.get(
                    "paper_compatible_support_f1"
                ),
                "paper_compatible_supporting_paragraphs": metrics.get(
                    "paper_compatible_supporting_paragraphs"
                ),
                "paper_compatible_support_source": metrics.get(
                    "paper_compatible_support_source"
                ),
                "collaboration": metrics.get("collaboration", {}),
                "router_cs": metrics.get("router_cs"),
                "offset": absolute_offset,
                **(
                    {
                        "retry_failed_only": True,
                        "retry_source": retry_records[0]["source_result"],
                    }
                    if retry_records
                    else {}
                ),
                **metadata,
            }
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
            complete_item(
                runner,
                task["id"],
                effective_start + index + 1,
                result_path,
            )
    finalize_run(runner)
