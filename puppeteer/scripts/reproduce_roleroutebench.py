#!/usr/bin/env python3
"""Download and certify the frozen RoleRouteBench beta.2 release.

This is a convenience wrapper around the frozen downloader and certification
tools. It does not modify the runtime, adapters, packets, or visibility rules.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASET_REPO = "TrangBui/RoleRouteBench"
DATASET_REVISION = "cdc67bad1458f6b939f759e3f764f2e93612f424"


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/RoleRouteBench/v4_beta_4d_82_20261102"),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    python = sys.executable
    run(
        [
            python,
            "scripts/download_routingbench_v4.py",
            "--repo-id",
            DATASET_REPO,
            "--revision",
            DATASET_REVISION,
            "--output",
            str(output),
        ],
        project_root,
    )
    run(
        [python, "-m", "unittest", "tests.test_routingbench_v4_private_isolation", "-v"],
        project_root,
    )
    run(
        [
            python,
            "scripts/certify_routingbench_v4_release.py",
            "--release-root",
            str(output),
            "--runtime-manifest",
            "reproducibility/runtime_core.sha256.json",
            "--report",
            "reproducibility/certification_report.json",
        ],
        project_root,
    )
    print("RoleRouteBench beta.2 certification completed successfully.")


if __name__ == "__main__":
    main()
