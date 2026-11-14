from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from routing_benchmark.roleroutebench.track_a.collector import build_track_a
from routing_benchmark.roleroutebench.track_a.sampling import sample_states
from routing_benchmark.roleroutebench.track_a.source_manifest import DOMAIN_SPECS


def main() -> int:
    parser = argparse.ArgumentParser(description="Build live-state RoleRouteBench Track A")
    parser.add_argument("--output", type=Path, default=PROJECT / "data" / "RoleRouteBench" / "track_a_4d_live_v1")
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--revised-sources", type=Path, help="JSON mapping revised domains to packet_root and ordered runs")
    args = parser.parse_args()
    specs = None
    if args.revised_sources:
        if args.output.exists():
            parser.error("--revised-sources requires a new --output directory")
        revised = json.loads(args.revised_sources.read_text(encoding="utf-8"))
        expected_sizes = {"formal_proof": 30, "structured_data_processing": 20}
        if set(revised) != set(expected_sizes):
            parser.error("--revised-sources must specify formal_proof and structured_data_processing")
        specs = dict(DOMAIN_SPECS)
        for domain, size in expected_sizes.items():
            packet_root = (PROJECT / revised[domain]["packet_root"]).resolve()
            if len(list(packet_root.glob("*/manifest.json"))) != size:
                parser.error(f"{domain} requires {size} packet manifests")
            runs = revised[domain]["runs"]
            if not runs or set(runs).intersection(DOMAIN_SPECS[domain]["runs"]):
                parser.error(f"{domain} requires new source runs")
            specs[domain] = {"packet_root": packet_root, "runs": runs}
    report = build_track_a(args.output, specs)
    sample = sample_states(
        args.output / "track_a_states.jsonl",
        args.output / "labeling",
        count=args.sample_count,
        seed=args.seed,
    )
    print(json.dumps({"build": report, "sample": sample}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

