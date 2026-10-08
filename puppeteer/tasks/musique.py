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
    data = data.iloc[int(data_start):]
    if data_limit is not None:
        data = data.iloc[: int(data_limit)]
    return data.reset_index(drop=True)


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
):
    effective_start, effective_limit = resolve_data_window(
        runner, data_start, data_limit
    )
    dataset = load_dataset(
        mode,
        effective_limit,
        seed=seed,
        data_start=effective_start,
        hop_count=hop_count,
    )
    split = _official_split(mode)
    hop_suffix = "all_hops" if hop_count is None else f"{int(hop_count)}hop"
    result_path = os.path.join(
        results_dir, f"MuSiQue_{split}_{hop_suffix}.jsonl"
    )
    file_mode = register_result_path(runner, result_path)
    with open(result_path, file_mode, encoding="utf-8") as destination:
        for index, (_, row) in enumerate(tqdm(dataset.iterrows(), total=len(dataset))):
            absolute_offset = effective_start + index
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
                **metadata,
            }
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
            complete_item(
                runner,
                task["id"],
                absolute_offset + 1,
                result_path,
            )
    finalize_run(runner)
