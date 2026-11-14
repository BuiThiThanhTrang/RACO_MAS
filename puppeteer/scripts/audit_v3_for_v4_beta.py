from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from routing_benchmark.v4beta.audit import write_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RoutingBench V3 task seeds for V4-beta migration.")
    parser.add_argument("--v3_root", type=Path, default=PROJECT_DIR / "data/RoutingBench/v3_beta")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_DIR / "data/RoutingBench/v4_beta_audit")
    args = parser.parse_args()
    print(json.dumps(write_audit(args.v3_root, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
