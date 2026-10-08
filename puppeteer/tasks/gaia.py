from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tasks.progress import (
    complete_item,
    finalize_run,
    register_result_path,
    resolve_data_window,
)


DEFAULT_DATA_ROOT = Path("data") / "GAIA"
SPLIT_ALIASES = {
    "dev": "validation",
    "validation": "validation",
    "final": "test",
    "test": "test",
}


def _official_split(mode: str) -> str:
    try:
        return SPLIT_ALIASES[str(mode).lower()]
    except KeyError as error:
        raise ValueError(
            "GAIA supports validation/dev for local scoring and test/final for "
            "leaderboard submissions."
        ) from error


def _metadata_path(data_root: str | Path, split: str, level: int | None) -> Path:
    filename = "metadata.parquet" if level is None else f"metadata.level{level}.parquet"
    return Path(data_root) / "2023" / split / filename


def _optional_text(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def load_dataset(
    mode,
    data_limit=None,
    seed=42,
    data_start=0,
    level=None,
    data_root=DEFAULT_DATA_ROOT,
):
    del seed  # GAIA uses its official immutable validation/test splits.
    if level is not None and int(level) not in {1, 2, 3}:
        raise ValueError("GAIA level must be 1, 2, 3, or omitted for all levels")
    split = _official_split(mode)
    metadata_path = _metadata_path(data_root, split, level)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"GAIA metadata not found: {metadata_path}. Set HF_TOKEN after "
            "accepting the gated dataset terms, then run "
            "python -m scripts.download_gaia --split all."
        )
    data = pd.read_parquet(metadata_path)
    required = {"task_id", "Question", "Level", "Final answer", "file_path"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"GAIA metadata is missing columns: {missing}")
    data = data.iloc[int(data_start):]
    if data_limit is not None:
        data = data.iloc[: int(data_limit)]
    return data.reset_index(drop=True)


def _relative_attachment(file_path: str | None, data_root: str | Path) -> str | None:
    if not file_path:
        return None
    root = Path(data_root).resolve()
    attachment = (root / file_path).resolve()
    if not attachment.is_relative_to(root):
        raise ValueError(f"GAIA attachment escapes dataset root: {file_path}")
    if not attachment.is_file():
        raise FileNotFoundError(f"GAIA attachment not found: {attachment}")
    # Agent tools resolve file names relative to config.file_path.root_file_path
    # (./data), not relative to the GAIA snapshot directory.
    return (Path("GAIA") / Path(file_path)).as_posix()


def format_question(row, idx=None, data_root=DEFAULT_DATA_ROOT):
    task_id = _optional_text(row.get("task_id")) or str(idx)
    question = str(row["Question"]).strip()
    official_file_path = _optional_text(row.get("file_path"))
    file_name = _relative_attachment(official_file_path, data_root)
    if file_name:
        question += (
            "\n\nAn attachment is provided with this task. Route a file-capable "
            "agent when its contents are needed; the attachment path is available "
            "to the read_file and run_python tools."
        )
    answer = _optional_text(row.get("Final answer"))
    level = int(row.get("Level"))
    task = {
        "type": "GAIA",
        "Question": question,
        "id": task_id,
        "level": level,
        "category": f"level_{level}",
        "has_attachment": bool(file_name),
        "file_name": file_name,
        "official_file_path": official_file_path,
    }
    if answer is not None:
        task["Answer"] = answer
    return task


def run(
    runner,
    evaluator,
    results_dir,
    mode,
    data_limit=None,
    data_start=0,
    seed=42,
    level=None,
):
    effective_start, effective_limit = resolve_data_window(
        runner, data_start, data_limit
    )
    dataset = load_dataset(
        mode,
        effective_limit,
        seed=seed,
        data_start=effective_start,
        level=level,
    )
    split = _official_split(mode)
    level_suffix = "all" if level is None else f"level{level}"
    result_path = os.path.join(
        results_dir, f"GAIA_2023_{split}_{level_suffix}.jsonl"
    )
    submission_path = os.path.join(
        results_dir, f"GAIA_2023_{split}_{level_suffix}_submission.jsonl"
    )
    file_mode = register_result_path(runner, result_path)
    submission = open(submission_path, file_mode, encoding="utf-8") if split == "test" else None
    try:
        with open(result_path, file_mode, encoding="utf-8") as destination:
            for index, (_, row) in enumerate(
                tqdm(dataset.iterrows(), total=len(dataset))
            ):
                absolute_offset = effective_start + index
                task = format_question(row, absolute_offset)
                prediction = runner.run_reasoning(task)
                metadata = dict(getattr(runner, "last_result_metadata", {}) or {})
                gold = task.get("Answer")
                success = metadata.get("final_success")
                if success is None and gold is not None:
                    success = evaluator.check_gaia(prediction, gold)
                record = {
                    "task_id": task["id"],
                    "model_answer": evaluator.extract_gaia_answer(prediction),
                    "level": task["level"],
                    "has_attachment": task["has_attachment"],
                    "correct": success,
                    "offset": absolute_offset,
                    **metadata,
                }
                if gold is not None:
                    record["answer"] = gold
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                destination.flush()
                os.fsync(destination.fileno())
                if submission is not None:
                    submission.write(
                        json.dumps(
                            {
                                "task_id": task["id"],
                                "model_answer": record["model_answer"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    submission.flush()
                    os.fsync(submission.fileno())
                complete_item(
                    runner, task["id"], absolute_offset + 1, result_path
                )
    finally:
        if submission is not None:
            submission.close()
    finalize_run(runner)
