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
    parser = argparse.ArgumentParser(
        description="Prepare one Track A cohort containing every state not sampled by an existing cohort"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT / "data" / "RoleRouteBench" / "track_a_4d_live_v1",
    )
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--output-dir", default="labeling_openrouter_remaining_v1")
    args = parser.parse_args()

    source_rows = read_jsonl(args.root / "track_a_states.jsonl")
    source_ids = {row["decision_id"] for row in source_rows}
    excluded: set[str] = set()
    for cohort in sorted(args.root.glob("labeling*")):
        if cohort.name == args.output_dir or not (cohort / "sample_manifest.json").exists():
            continue
        excluded.update(row["decision_id"] for row in read_jsonl(cohort / "sample_states.jsonl"))

    remaining_count = len(source_ids - excluded)
    if remaining_count <= 0:
        raise RuntimeError("no unsampled Track A states remain")
    manifest = sample_states(
        args.root / "track_a_states.jsonl",
        args.root / args.output_dir,
        count=remaining_count,
        seed=args.seed,
        exclude_decision_ids=excluded,
    )
    manifest["cohort_purpose"] = "all_remaining_primary_states"
    manifest["prior_cohort_count"] = len(excluded)
    (args.root / args.output_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
