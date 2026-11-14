from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from routing_benchmark.v2.generator import build_alpha_bundle, write_alpha_release
from routing_benchmark.v2.validation import (
    validate_alpha,
    validate_deterministic_generation,
)


DEFAULT_OUTPUT = PROJECT_DIR / "data" / "RoutingBench" / "v2_alpha"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RoutingBench V2 alpha.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = write_alpha_release(args.output_dir, args.seed)
    validation = validate_alpha(build_alpha_bundle(args.seed))
    validation["deterministic_generation"] = validate_deterministic_generation(
        args.seed
    )
    validation_path = args.output_dir / "validation_report.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not validation["valid"] or not validation["deterministic_generation"]:
        raise SystemExit("RoutingBench V2 alpha validation failed")
    print(f"RoutingBench V2 alpha written to: {args.output_dir}")


if __name__ == "__main__":
    main()
