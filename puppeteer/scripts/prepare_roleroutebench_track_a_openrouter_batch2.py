from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from routing_benchmark.roleroutebench.track_a.canonicalize import read_jsonl
from routing_benchmark.roleroutebench.track_a.sampling import sample_states


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the disjoint second 100-state Track A sample")
    parser.add_argument("--root", type=Path, default=PROJECT / "data" / "RoleRouteBench" / "track_a_4d_live_v1")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--output-dir", default="labeling_openrouter_100_v1")
    args = parser.parse_args()
    first = read_jsonl(args.root / "labeling" / "sample_states.jsonl")
    excluded = {row["decision_id"] for row in first}
    manifest = sample_states(
        args.root / "track_a_states.jsonl",
        args.root / args.output_dir,
        count=args.count,
        seed=args.seed,
        exclude_decision_ids=excluded,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
