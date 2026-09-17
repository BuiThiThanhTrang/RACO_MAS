import os
from tqdm import tqdm
import json

from tasks.splits import load_split_frame


def load_dataset(mode, data_limit=None, seed=42):
    data = load_split_frame("srdd", mode, seed=seed)
    if data_limit is not None:
        data = data.iloc[:data_limit]
    return data

def format_question(row, idx):
    return {
        "type": "SRDD",
        "req": "code",
        "Question": "Develop a pythonic software following description:\n" + row["Description"],
        "id": idx
    }

def run(runner, evaluator, results_dir, mode, data_limit=None, seed=42):
    dataset = load_dataset(mode, data_limit, seed=seed)
    result_path = os.path.join(results_dir, "srdd.jsonl")

    with open(result_path, "w", encoding="utf-8") as fd:
        for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
            task = format_question(row, idx)
            final_ans = runner.run_reasoning(task)

            record = {
            "id": task["id"],
            "pred": final_ans
            }
            fd.write(json.dumps(record, ensure_ascii=False) + "\n")