import os
import json
import time
import pandas as pd
from tqdm import tqdm
from tasks.base.base_task import BaseTask

def load_dataset(mode, data_limit=None, seed=42, data_start=0):
    mode_path = os.path.join("data", "GSM-Hard", f"{mode}.parquet")
    test_path = os.path.join("data", "GSM-Hard", "test.parquet")
    path = mode_path if os.path.exists(mode_path) else test_path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GSM-Hard dataset not found. Expected {mode_path} or {test_path}. "
            "Run this command from the puppeteer folder or use puppeteer/main.py from the repo root."
        )
    data = pd.read_parquet(path)
    data = data.reset_index().rename(columns={"index": "original_index"})
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    data["shuffled_index"] = data.index
    if data_start < 0:
        raise ValueError("data_start must be >= 0")
    if data_limit:
        return data.iloc[data_start:data_start + data_limit].reset_index(drop=True)
    return data.iloc[data_start:].reset_index(drop=True)

def format_question(row, idx):
    return {
        "type": "GSM-Hard",
        "Question": "You need to write python program to solve math problems:\n" + row["input"],
        "Answer": row["target"],
        "id": int(row.get("shuffled_index", idx)),
        "original_index": int(row.get("original_index", idx)),
    }

def run(runner, evaluator, results_dir, mode, data_limit=None, data_start=0, seed=42, result_suffix=None):
    dataset = load_dataset(mode, data_limit, seed=seed, data_start=data_start)
    if result_suffix is None:
        limit_label = "all" if data_limit is None else str(data_limit)
        result_suffix = f"start{data_start}_limit{limit_label}_seed{seed}"
    result_path = os.path.join(results_dir, f"gsm-hard_{result_suffix}.jsonl")
    acc = 0

    with open(result_path, "w", encoding="utf-8") as fd:
        for idx, row in enumerate(tqdm(dataset.iterrows(), total=len(dataset))):
            task = format_question(row[1], idx)
            final_ans = runner.run_reasoning(task)
            flag = evaluator.check_gsm8k(final_ans, task["Answer"])
            if flag: acc += 1
            record = {
            "id": task["id"],
            "original_index": task["original_index"],
            "batch_index": idx,
            "data_start": data_start,
            "seed": seed,
            "pred": final_ans,
            "answer": task["Answer"],
            "correct": flag
            }
            fd.write(json.dumps(record, ensure_ascii=False) + "\n")
            fd.flush()
            time.sleep(20) # Prevent API rate limiting
    
    total = len(dataset)
    accuracy = acc / total if total else 0
    summary = {
        "task": "gsm-hard",
        "mode": mode,
        "total": total,
        "correct": acc,
        "accuracy": accuracy,
        "data_start": data_start,
        "data_limit": data_limit,
        "seed": seed,
        "result_path": result_path,
    }
    summary_path = os.path.join(results_dir, f"summary_{result_suffix}.json")
    with open(summary_path, "w", encoding="utf-8") as fd:
        json.dump(summary, fd, ensure_ascii=False, indent=2)
    print(f"GSM-Hard accuracy: {accuracy:.4f} ({acc}/{total})")
    print(f"Summary written to: {summary_path}")
