from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from routing_benchmark.v4beta.reporting import analyze_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate RoutingBench V4-beta results by domain, pattern and fault.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--data_dir", type=Path, default=PROJECT_DIR / "data/RoutingBench/v4_beta")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze_runs(args.data_dir, args.run_dirs)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
