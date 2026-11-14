from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from routing_benchmark.roleroutebench.track_a.reporting import build_human_review_report, build_prelabel_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT / "data" / "RoleRouteBench" / "track_a_4d_live_v1")
    parser.add_argument("--labeling-dir", default="labeling")
    args = parser.parse_args()
    report = {"prelabels": build_prelabel_report(args.root, args.labeling_dir)}
    if (args.root / "reviews" / "human_reviews.jsonl").exists():
        report["human_reviews"] = build_human_review_report(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

