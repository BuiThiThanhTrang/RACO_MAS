import argparse
import ast
import glob
import json
import math
import os
import re
from datetime import datetime


FLOAT_TOLERANCE = 1e-3


def read_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_number(text):
    if text is None:
        return None
    if not isinstance(text, str):
        try:
            return float(text)
        except (TypeError, ValueError):
            return None
    matches = re.findall(r"-?\d+\.\d+|-?\d+", text.replace(",", ""))
    return float(matches[0]) if matches else None


def check_gsm_answer(predicted, gold):
    pred_num = extract_number(predicted)
    gold_num = extract_number(gold)
    if pred_num is None or gold_num is None:
        return False
    if not (math.isfinite(pred_num) and math.isfinite(gold_num)):
        return False

    if abs(pred_num - gold_num) < FLOAT_TOLERANCE:
        return True
    if round(pred_num) == round(gold_num):
        return True
    if abs(int(pred_num)) > 100 and abs(int(gold_num)) > 100:
        return int(pred_num) == int(gold_num)
    return False


def parse_timestamp(folder_name):
    try:
        return datetime.strptime(folder_name, "%Y-%m-%d-%H-%M-%S").isoformat()
    except ValueError:
        return None


def extract_task(meta_text):
    task_matches = re.findall(r"'task': (\{.*?\}), 'workflow':", meta_text, flags=re.DOTALL)
    for raw_task in reversed(task_matches):
        try:
            task = ast.literal_eval(raw_task)
        except (SyntaxError, ValueError):
            continue
        if isinstance(task, dict):
            return task
    return {}


def extract_final_answer(meta_text):
    matches = re.findall(r"\[Final Answer\]:\s*(.*)", meta_text)
    if matches:
        return matches[-1].strip()
    return None


def extract_path_answers(meta_text):
    answers = []
    pattern = r"\[Aggregated Answer From Path (\d+)\]:\s*(.*)"
    for path_id, answer in re.findall(pattern, meta_text):
        answers.append({"path_id": int(path_id), "answer": answer.strip()})
    return answers


def parse_log_folder(folder_path):
    folder_name = os.path.basename(folder_path)
    meta_text = read_text(os.path.join(folder_path, "meta.log"))
    task_source_text = meta_text
    for log_path in sorted(glob.glob(os.path.join(folder_path, "path*.log"))):
        task_source_text += "\n" + read_text(log_path)
    task_source_text += "\n" + read_text(os.path.join(folder_path, "train.log"))

    task = extract_task(task_source_text)
    final_answer = extract_final_answer(meta_text)
    path_answers = extract_path_answers(meta_text)
    gold_answer = task.get("Answer")
    correct = check_gsm_answer(final_answer, gold_answer) if final_answer is not None else False

    parse_errors = []
    if not meta_text:
        parse_errors.append("missing_meta_log")
    if not task:
        parse_errors.append("missing_task")
    if gold_answer is None:
        parse_errors.append("missing_gold_answer")
    if final_answer is None:
        parse_errors.append("missing_final_answer")

    return {
        "folder": folder_path,
        "run_id": folder_name,
        "run_timestamp": parse_timestamp(folder_name),
        "id": task.get("id"),
        "original_index": task.get("original_index"),
        "question": task.get("Question"),
        "gold_answer": gold_answer,
        "final_answer": final_answer,
        "predicted_number": extract_number(final_answer),
        "gold_number": extract_number(gold_answer),
        "correct": correct,
        "path_answers": path_answers,
        "parse_errors": parse_errors,
    }


def iter_log_folders(logs_root):
    for name in sorted(os.listdir(logs_root)):
        path = os.path.join(logs_root, name)
        if os.path.isdir(path):
            yield path


def main():
    parser = argparse.ArgumentParser(
        description="Compute GSM-Hard accuracy directly from timestamped log folders."
    )
    parser.add_argument(
        "--logs-root",
        default=os.path.join("puppeteer", "logs", "GSM-Hard"),
        help="Folder containing GSM-Hard timestamp run folders.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("puppeteer", "results", "gsm-hard_logs_accuracy.jsonl"),
        help="Output jsonl path for parsed per-run records.",
    )
    parser.add_argument(
        "--summary",
        default=os.path.join("puppeteer", "results", "gsm-hard_logs_accuracy_summary.json"),
        help="Output json summary path.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.logs_root):
        raise ValueError(f"Logs root does not exist: {args.logs_root}")

    records = [parse_log_folder(path) for path in iter_log_folders(args.logs_root)]
    parsed_records = [record for record in records if not record["parse_errors"]]
    total_runs = len(parsed_records)
    correct = sum(1 for record in parsed_records if record["correct"])
    accuracy = correct / total_runs if total_runs else 0.0
    unparsed = [record for record in records if record["parse_errors"]]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "task": "gsm-hard",
        "logs_root": args.logs_root,
        "total_folders": len(records),
        "parsed_runs": total_runs,
        "correct": correct,
        "accuracy": accuracy,
        "unparsed_folders": [
            {
                "run_id": record["run_id"],
                "folder": record["folder"],
                "parse_errors": record["parse_errors"],
            }
            for record in unparsed
        ],
        "output": args.output,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Parsed runs: {total_runs}/{len(records)}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total_runs})")
    if unparsed:
        print(f"Unparsed folders: {len(unparsed)}")
        for record in unparsed[:10]:
            print(f"- {record['run_id']}: {', '.join(record['parse_errors'])}")
    print(f"Records: {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
