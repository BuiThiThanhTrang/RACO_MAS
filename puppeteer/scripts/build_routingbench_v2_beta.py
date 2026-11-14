from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from routing_benchmark.v2.generator_beta import build_beta_bundle, write_beta_release
from routing_benchmark.v2.validation_beta import (
    validate_beta,
    validate_beta_determinism,
)


DEFAULT_OUTPUT = PROJECT_DIR / "data" / "RoutingBench" / "v2_beta"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RoutingBench V2 beta.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = write_beta_release(args.output_dir, args.seed)
    validation = validate_beta(build_beta_bundle(args.seed))
    validation["deterministic_generation"] = validate_beta_determinism(args.seed)
    validation_path = args.output_dir / "validation_report.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not validation["valid"] or not validation["deterministic_generation"]:
        raise SystemExit("RoutingBench V2 beta validation failed")
    print(f"RoutingBench V2 beta written to: {args.output_dir}")


if __name__ == "__main__":
    main()

