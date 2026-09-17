import os
import json
from tqdm import tqdm

from tasks.splits import load_split_frame


def load_dataset(mode, data_limit=None, seed=42):
    data = load_split_frame("gsm_hard", mode, seed=seed)
    if data_limit is not None:
        data = data.iloc[:data_limit]
    return data

def format_question(row, idx):
    return {
        "type": "GSM-Hard",
        "Question": "You need to write python program to solve math problems:\n" + row["input"],
        "Answer": row["target"],
        "id": idx
    }

def run(runner, evaluator, results_dir, mode, data_limit=None, seed=42):
    dataset = load_dataset(mode, data_limit, seed=seed)
    result_path = os.path.join(results_dir, "gsm-hard.jsonl")
    acc = 0

    with open(result_path, "w", encoding="utf-8") as fd:
        for idx, row in enumerate(tqdm(dataset.iterrows(), total=len(dataset))):
            task = format_question(row[1], idx)
            final_ans = runner.run_reasoning(task)
            flag = evaluator.check_gsm8k(final_ans, task["Answer"])
            if flag: acc += 1
            record = {
            "id": task["id"],
            "pred": final_ans,
            "correct": flag
            }
            fd.write(json.dumps(record, ensure_ascii=False) + "\n")