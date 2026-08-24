import json
import os

import pandas as pd
from tqdm import tqdm

from tasks.splits import load_split_indices
from tasks.progress import (
    complete_item,
    finalize_run,
    register_result_path,
    resolve_data_window,
)

def load_dataset(mode, data_limit=None, seed=42, data_start=0):
    data = pd.read_csv("./data/SRDD/SRDD.csv")
    data = data.iloc[load_split_indices("srdd", mode, seed=seed)].reset_index(drop=True)
    data = data.iloc[data_start:]
    if data_limit is not None:
        data = data.iloc[:data_limit]
    return data.reset_index(drop=True)

def format_question(row, idx):
    return {
        "type": "SRDD", "req": "code",
        "Question": "Develop executable Python software following this description:\n" + row["Description"],
        "id": idx,
    }

def run(runner, evaluator, results_dir, mode, data_limit=None, data_start=0, seed=42):
    effective_start, effective_limit = resolve_data_window(
        runner, data_start, data_limit
    )
    dataset = load_dataset(
        mode, effective_limit, seed=seed, data_start=effective_start
    )
    result_path = os.path.join(results_dir, f"srdd_{mode}.jsonl")
    file_mode = register_result_path(runner, result_path)
    with open(result_path, file_mode, encoding="utf-8") as fd:
        for idx, (_, row) in enumerate(tqdm(dataset.iterrows(), total=len(dataset))):
            absolute_offset = effective_start + idx
            task = format_question(row, absolute_offset)
            prediction = runner.run_reasoning(task)
            reward, metrics = evaluator.check_srdd(prediction, task["Question"])
            success = evaluator.srdd_binary_success(metrics)
            fd.write(json.dumps({"id": task["id"], "pred": prediction, "reward": float(reward), "success": success, "metrics": evaluator.metrics_to_json(metrics)}, ensure_ascii=False) + "\n")
            fd.flush()
            os.fsync(fd.fileno())
            complete_item(runner, task["id"], absolute_offset + 1, result_path)
    finalize_run(runner)
