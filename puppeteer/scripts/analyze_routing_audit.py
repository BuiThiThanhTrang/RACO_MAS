"""Validate audit traces and summarize stored MMLU-Pro outcomes without model calls."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from role_aware.audit_analysis import collect_tasks, load_events, summarize, validate_events
from role_aware.audit_trace import write_json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Run/audit directories")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validations = []
    for source in args.inputs:
        for path in sorted(Path(source).rglob("events.jsonl")):
            validations.append(dict(file=str(path), **validate_events(load_events(path))))
    tasks = collect_tasks(args.inputs)
    result = dict(summary=summarize(tasks), validations=validations,
                  split_counts={s: sum(t["split"] == s for t in tasks) for s in {t["split"] for t in tasks}},
                  note="Training/dev diagnostics are not final generalization estimates.")
    write_json(args.output, result)
    print(json.dumps(dict(tasks=len(tasks), traces=len(validations),
                         errors=sum(len(v["errors"]) for v in validations), output=args.output)))
    return 1 if any(v["errors"] for v in validations) else 0

if __name__ == "__main__":
    raise SystemExit(main())
