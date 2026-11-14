"""Run V4-beta with the same audited evaluator used by V4-alpha.2."""

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_routing_benchmark_v4 import main


if __name__ == "__main__":
    main()
