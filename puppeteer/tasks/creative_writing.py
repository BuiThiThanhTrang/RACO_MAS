import os
import json
from tqdm import tqdm

from tasks.splits import load_split_frame


def load_dataset(mode, data_limit=None, seed=42):
    data = load_split_frame("cw", mode, seed=seed)
    if data_limit is not None:
        data = data.iloc[:data_limit]
    return data.to_dict("records")

def format_question(q, idx):
    question = "Concepts: " + ", ".join(q["concepts"]) + \
               "\nGenerate a sentence including all key concepts, grammatically correct and coherent."
    return {
        "type": "CW",
        "req": "text",
        "Question": question,
        "id": idx,
        "concepts": q["concepts"]
    }

def run(runner, evaluator, results_dir, mode, data_limit=None, seed=42):
    dataset = load_dataset(mode, data_limit, seed=seed)
    result_path = os.path.join(results_dir, "cw.jsonl")

    with open(result_path, "w", encoding="utf-8") as fd:
        for idx, q in enumerate(tqdm(dataset)):
            task = format_question(q, idx)
            final_ans = runner.run_reasoning(task)

            record = {
            "id": task["id"],
            "pred": final_ans
            }
            fd.write(json.dumps(record, ensure_ascii=False) + "\n")
