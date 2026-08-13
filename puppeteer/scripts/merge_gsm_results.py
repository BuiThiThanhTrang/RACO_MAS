import argparse
import glob
import json
import os


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Merge GSM-Hard batch result jsonl files.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input jsonl files or glob patterns, e.g. results/gsm-hard_test_initialized/gsm-hard_start*.jsonl",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Merged output jsonl path.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional merged summary json path.",
    )
    args = parser.parse_args()

    input_paths = []
    for item in args.inputs:
        matches = glob.glob(item)
        input_paths.extend(matches if matches else [item])
    input_paths = sorted(dict.fromkeys(input_paths))

    if not input_paths:
        raise ValueError("No input files found.")

    merged_by_id = {}
    duplicates = []
    for path in input_paths:
        for record in iter_jsonl(path):
            record_id = record.get("id", record.get("original_index"))
            if record_id is None:
                raise ValueError(f"Record without id in {path}: {record}")
            record_id = int(record_id)
            if record_id in merged_by_id:
                duplicates.append(record_id)
            merged_by_id[record_id] = {
                **record,
                "source_file": path,
            }

    merged_records = [merged_by_id[key] for key in sorted(merged_by_id)]
    total = len(merged_records)
    correct = sum(1 for record in merged_records if bool(record.get("correct")))
    accuracy = correct / total if total else 0.0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in merged_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = args.summary
    if summary_path is None:
        summary_path = os.path.splitext(args.output)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "gsm-hard",
                "input_files": input_paths,
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "duplicate_ids_overwritten": sorted(set(duplicates)),
                "output": args.output,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Merged {total} records from {len(input_paths)} file(s).")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    if duplicates:
        print(f"Warning: overwritten duplicate ids: {sorted(set(duplicates))}")
    print(f"Merged results: {args.output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
