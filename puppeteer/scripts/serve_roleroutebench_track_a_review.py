from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT / "data" / "RoleRouteBench" / "track_a_4d_live_v1")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument(
        "--labeling-dir",
        default="all",
        help="A labeling directory name or 'all' to review every disjoint cohort.",
    )
    args = parser.parse_args()
    app = PROJECT / "review_app" / "roleroutebench_track_a.py"
    return subprocess.call(
        [
            "streamlit", "run", str(app), "--server.port", str(args.port), "--",
            "--root", str(args.root.resolve()), "--labeling-dir", args.labeling_dir,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

