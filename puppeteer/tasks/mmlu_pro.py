import json
import os
import string
from pathlib import Path

from tqdm import tqdm

from tasks.splits import load_split_frame


def require_mmlu_choice(evaluator, response, task_id):
    """Normalize an answer and prevent invalid predictions from being persisted."""
    choice = evaluator.extract_choice_answer(response)
    if (
        not isinstance(choice, str)
        or len(choice) != 1
        or choice not in string.ascii_uppercase[:10]
    ):
        raise RuntimeError(
            "MMLU-Pro produced no valid A-J answer for question "
            f"{task_id!r}; result and checkpoint were not written"
        )
    return choice


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


def validate_resume(
    result_path,
    mode,
    data_start,
    seed,
    checkpoint_completed_rows=None,
    checkpoint_last_result_id=None,
):
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
    if (
        checkpoint_completed_rows is not None
        and checkpoint_completed_rows != data_start
    ):
        raise ValueError(
            "Resume mismatch: checkpoint metadata reports "
            f"{checkpoint_completed_rows} completed rows, but --data_start is "
            f"{data_start}"
        )
    if (
        checkpoint_last_result_id is not None
        and existing_ids[-1] != checkpoint_last_result_id
    ):
        raise ValueError(
            "Resume mismatch: checkpoint metadata reports last result ID "
            f"{checkpoint_last_result_id!r}, but the result file ends with "
            f"{existing_ids[-1]!r}"
        )


def run(
    runner,
    evaluator,
    results_dir,
    mode,
    data_limit=None,
    seed=42,
    data_start=0,
    checkpoint_every=1,
    checkpoint_completed_rows=None,
    checkpoint_last_result_id=None,
):
    result_path = os.path.join(results_dir, f"MMLU-Pro_{mode}.jsonl")
    if data_start:
        validate_resume(
            result_path,
            mode,
            data_start,
            seed,
            checkpoint_completed_rows=checkpoint_completed_rows,
            checkpoint_last_result_id=checkpoint_last_result_id,
        )

    dataset = load_dataset(
        mode, data_limit, seed=seed, data_start=data_start
    )
    file_mode = "a" if data_start else "w"
    acc = 0

    with open(result_path, file_mode, encoding="utf-8") as fd:
        for row_offset, (_, row) in enumerate(dataset.iterrows(), start=1):
            task = format_question(row)
            raw_final_ans = runner.run_reasoning(task)
            final_ans = require_mmlu_choice(
                evaluator,
                raw_final_ans,
                task["id"],
            )
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
            os.fsync(fd.fileno())

            completed_rows = data_start + row_offset
            if (
                completed_rows % checkpoint_every == 0
                or row_offset == len(dataset)
            ):
                runner.save_resume_checkpoint(
                    completed_rows=completed_rows,
                    result_last_id=task["id"],
                )
