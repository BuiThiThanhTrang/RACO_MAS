import json
import os

from tqdm import tqdm

from tasks.splits import load_split_indices
from tasks.progress import (
    complete_item,
    finalize_run,
    register_result_path,
    resolve_data_window,
)

def load_dataset(mode, data_limit=None, seed=42, data_start=0):
    path = "./data/CW/creative_writing.jsonl"
    with open(path, "r", encoding="utf-8") as source:
        all_data = [json.loads(line) for line in source if line.strip()]
    data = [all_data[index] for index in load_split_indices("cw", mode, seed=seed)]
    data = data[data_start:]
    return data[:data_limit] if data_limit is not None else data

def format_question(item, idx):
    return {
        "type": "CW", "req": "text",
        "Question": "Concepts: " + ", ".join(item["concepts"]) + "\nGenerate a sentence including every concept, with correct grammar and coherent commonsense.",
        "id": idx, "concepts": item["concepts"],
    }

def run(runner, evaluator, results_dir, mode, data_limit=None, data_start=0, seed=42):
    effective_start, effective_limit = resolve_data_window(
        runner, data_start, data_limit
    )
    dataset = load_dataset(
        mode, effective_limit, seed=seed, data_start=effective_start
    )
    result_path = os.path.join(results_dir, f"cw_{mode}.jsonl")
    file_mode = register_result_path(runner, result_path)
    with open(result_path, file_mode, encoding="utf-8") as fd:
        for idx, item in enumerate(tqdm(dataset)):
            absolute_offset = effective_start + idx
            task = format_question(item, absolute_offset)
            prediction = runner.run_reasoning(task)
            reward, metrics = evaluator.check_commongen(task["concepts"], prediction)
            success = evaluator.commongen_binary_success(metrics)
            fd.write(json.dumps({"id": task["id"], "pred": prediction, "reward": float(reward), "success": success, "metrics": evaluator.metrics_to_json(metrics)}, ensure_ascii=False) + "\n")
            fd.flush()
            os.fsync(fd.fileno())
            complete_item(runner, task["id"], absolute_offset + 1, result_path)
    finalize_run(runner)
