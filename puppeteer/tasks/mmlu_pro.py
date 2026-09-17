import json
import os
import string
from pathlib import Path

from tqdm import tqdm

from tasks.splits import load_split_frame


def load_dataset(mode, data_limit=None, seed=42, data_start=0):
    if data_start < 0:
        raise ValueError("data_start must be non-negative")

    data = load_split_frame("mmlu_pro", mode, seed=seed)

    data = data.iloc[data_start:]
    if data_limit is not None:
        data = data.iloc[:data_limit]
    return data.reset_index(drop=True)


def format_question(task):
    options = [
        f"{letter}: {option}"
        for letter, option in zip(string.ascii_uppercase, task["options"])
    ]
    prompt = (
        "The following are multiple choice questions (with answers) "
        f"about {task['category']}."
    )
    question = prompt + "\n" + task["question"] + "\n" + " ".join(options)
    return {
        "type": "MMLU-Pro",
        "Question": question,
        "Answer": task["answer"],
        "id": task["question_id"],
    }


def _result_ids(result_path):
    path = Path(result_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot resume because result file is missing: {path}"
        )

    ids = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
                ids.append(record["id"])
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    f"Invalid result record at line {line_number}: {path}"
                ) from error
    return ids


def validate_resume(result_path, mode, data_start, seed):
    existing_ids = _result_ids(result_path)
    if len(existing_ids) != data_start:
        raise ValueError(
            f"Resume mismatch: --data_start is {data_start}, but "
            f"{result_path} contains {len(existing_ids)} completed rows"
        )

    expected = load_dataset(
        mode, data_limit=data_start, seed=seed, data_start=0
    )
    expected_ids = expected["question_id"].tolist()
    if existing_ids != expected_ids:
        raise ValueError(
            "Resume mismatch: result IDs are not the expected prefix "
            f"of the {mode} split for seed {seed}"
        )


def run(
    runner,
    evaluator,
    results_dir,
    mode,
    data_limit=None,
    seed=42,
    data_start=0,
):
    result_path = os.path.join(results_dir, f"MMLU-Pro_{mode}.jsonl")
    if data_start:
        validate_resume(result_path, mode, data_start, seed)

    dataset = load_dataset(
        mode, data_limit, seed=seed, data_start=data_start
    )
    file_mode = "a" if data_start else "w"
    acc = 0

    with open(result_path, file_mode, encoding="utf-8") as fd:
        for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
            task = format_question(row)
            final_ans = runner.run_reasoning(task)
            flag = evaluator.check_mmlu(final_ans, task["Answer"])
            if flag:
                acc += 1
            record = {
                "id": task["id"],
                "pred": final_ans,
                "correct": flag,
            }
            fd.write(json.dumps(record, ensure_ascii=False) + "\n")
            fd.flush()
