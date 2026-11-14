from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from routing_benchmark.v4beta.generator import write_v4_beta_release
from routing_benchmark.v4beta.validation import validate_v4_beta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate RoutingBench V4-beta.")
    parser.add_argument("--v3_root", type=Path, default=PROJECT_DIR / "data/RoutingBench/v3_beta")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_DIR / "data/RoutingBench/v4_beta")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = write_v4_beta_release(args.output_dir, args.v3_root, args.seed)
    validation = validate_v4_beta(args.output_dir)
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({**result, "validation": validation}, ensure_ascii=False, indent=2))
    if not validation["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
