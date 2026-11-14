from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from routing_benchmark.roleroutebench.track_a.canonicalize import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT / "data" / "RoleRouteBench" / "track_a_4d_live_v1")
    args = parser.parse_args()
    source = args.root / "reviews" / "human_reviews.jsonl"
    latest = {}
    for row in read_jsonl(source):
        latest[row["decision_id"]] = row
    labels = [row for row in latest.values() if row.get("label_status") == "human_verified"]
    labels.sort(key=lambda row: row["decision_id"])
    target = args.root / "track_a_labels.jsonl"
    write_jsonl(target, labels)
    print(json.dumps({"review_records": len(read_jsonl(source)), "human_verified_labels": len(labels), "output": str(target)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

