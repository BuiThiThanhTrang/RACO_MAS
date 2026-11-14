from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from routing_benchmark.v4beta.validation import validate_v4_beta


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an existing RoutingBench V4-beta package.")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_DIR / "data/RoutingBench/v4_beta")
    args = parser.parse_args()
    report = validate_v4_beta(args.data_dir)
    (args.data_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
