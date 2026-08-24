import json
import os

import pandas as pd
from tqdm import tqdm

from tasks.splits import load_split_indices
from tasks.progress import (
    complete_item,
    finalize_run,
    read_jsonl,
    register_result_path,
    resolve_data_window,
)

def load_dataset(mode, data_limit=None, seed=42, data_start=0):
    path = os.path.join("data", "GSM-Hard", "test.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"GSM-Hard dataset not found: {path}")
    data = pd.read_parquet(path).reset_index().rename(columns={"index": "original_index"})
    indices = load_split_indices("gsm_hard", mode, seed=seed)
    data = data.iloc[indices].reset_index(drop=True)
    if data_start < 0:
        raise ValueError("data_start must be >= 0")
    data = data.iloc[data_start:]
    if data_limit is not None:
        data = data.iloc[:data_limit]
    return data.reset_index(drop=True)

def format_question(row, idx):
    return {
        "type": "GSM-Hard",
        "Question": "Solve this math problem and return the final numerical answer:\n" + row["input"],
        "Answer": row["target"],
        "id": int(row.get("original_index", idx)),
        "original_index": int(row.get("original_index", idx)),
    }

def run(runner, evaluator, results_dir, mode, data_limit=None, data_start=0, seed=42, result_suffix=None):
    initial_data_start = data_start
    effective_start, effective_limit = resolve_data_window(
        runner, data_start, data_limit
    )
    dataset = load_dataset(
        mode, effective_limit, seed=seed, data_start=effective_start
    )
    if result_suffix is None:
        result_suffix = f"{mode}_start{initial_data_start}_seed{seed}"
    result_path = os.path.join(results_dir, f"gsm-hard_{result_suffix}.jsonl")
    file_mode = register_result_path(runner, result_path)
    with open(result_path, file_mode, encoding="utf-8") as fd:
        for idx, (_, row) in enumerate(tqdm(dataset.iterrows(), total=len(dataset))):
            absolute_offset = effective_start + idx
            task = format_question(row, absolute_offset)
            prediction = runner.run_reasoning(task)
            success = evaluator.check_gsm8k(prediction, task["Answer"])
            fd.write(json.dumps({
                "id": task["id"], "original_index": task["original_index"],
                "batch_index": absolute_offset, "split": mode, "seed": seed,
                "pred": prediction, "answer": task["Answer"], "correct": success,
            }, ensure_ascii=False) + "\n")
            fd.flush()
            os.fsync(fd.fileno())
            complete_item(runner, task["id"], absolute_offset + 1, result_path)
    finalize_run(runner)
    records = read_jsonl(result_path)
    total = len(records)
    correct = sum(int(record.get("correct", False)) for record in records)
    summary = {"task": "gsm-hard", "split": mode, "total": total, "correct": correct,
               "accuracy": correct / total if total else 0.0, "result_path": result_path}
    with open(os.path.join(results_dir, f"summary_{result_suffix}.json"), "w", encoding="utf-8") as fd:
        json.dump(summary, fd, ensure_ascii=False, indent=2)
