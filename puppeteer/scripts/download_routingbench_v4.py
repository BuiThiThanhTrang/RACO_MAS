#!/usr/bin/env python3
"""Download the pinned Hugging Face release into the runner's local layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time


DEFAULT_REPO = "TrangBui/RoutingBench-V4"
DEFAULT_REVISION = "cdc67bad1458f6b939f759e3f764f2e93612f424"
DOMAIN_ORDER = (
    "database_migration_integrity",
    "code_debugging",
    "formal_proof",
    "structured_data_processing",
)


def _decode_json(value: str | None) -> list:
    return json.loads(value or "[]")


def _role_cards(snapshot: Path) -> dict[str, list[dict]]:
    import pyarrow.parquet as parquet

    table = parquet.read_table(snapshot / "data/track_b_agents/full.parquet")
    grouped = {domain: [] for domain in DOMAIN_ORDER}
    for row in table.to_pylist():
        domain = row["domain"]
        card = {"agent_id": row["agent_id"]}
        # Code-debugging role cards were frozen with the extended semantic
        # contract.  The other three domains use the compact five-field card.
        if domain == "code_debugging":
            card["role_name"] = row["role_name"]
        card["role_goal"] = row["role_goal"]
        if domain == "code_debugging":
            card["semantic_family"] = row["semantic_family"]
            card["tools"] = _decode_json(row["tools_json"])
        card["use_when"] = _decode_json(row["use_when_json"])
        card["avoid_when"] = _decode_json(row["avoid_when_json"])
        if domain == "code_debugging":
            card["supported_inputs"] = _decode_json(row["supported_inputs_json"])
        card["expected_outputs"] = _decode_json(row["expected_outputs_json"])
        grouped[domain].append(card)
    return grouped


def _obtain_snapshot(staging: Path, args: argparse.Namespace) -> Path:
    snapshot = staging / "snapshot"
    if args.local_source:
        shutil.copytree(args.local_source, snapshot)
        return snapshot
    from huggingface_hub import snapshot_download

    patterns = [
        "track_b_packets/**",
        "data/track_b_agents/full.parquet",
        "VERSION",
        "release_manifest.json",
        "checksums.sha256",
        "LICENSE_DATA",
        "README.md",
    ]
    for attempt in range(1, args.download_attempts + 1):
        try:
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.revision,
                local_dir=snapshot,
                allow_patterns=patterns,
                max_workers=args.max_workers,
            )
            break
        except Exception:
            if attempt >= args.download_attempts:
                raise
            time.sleep(min(2 ** attempt, 10))
    return snapshot


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--download-attempts", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "data/RoutingBench/v4_beta_4d_82_20261102",
    )
    parser.add_argument("--local-source", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.download_attempts < 1 or args.max_workers < 1:
        raise SystemExit("--download-attempts and --max-workers must be positive")

    with tempfile.TemporaryDirectory(
        prefix="routingbench-v4-download-",
        dir=output.parent,
        ignore_cleanup_errors=True,
    ) as temp:
        staging = Path(temp)
        snapshot = _obtain_snapshot(staging, args)
        release = staging / "release"
        release.mkdir()
        shutil.copytree(snapshot / "track_b_packets", release / "packs")

        role_root = release / "agent_pools"
        role_root.mkdir()
        roles = _role_cards(snapshot)
        for domain in DOMAIN_ORDER:
            (role_root / f"{domain}.json").write_bytes(
                (json.dumps(roles[domain], ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            )

        packet_count = len(list((release / "packs").glob("*/*/manifest.json")))
        if packet_count != 82:
            raise RuntimeError(f"Expected 82 Track B packets, found {packet_count}")
        metadata = {
            "source_repo": args.repo_id,
            "source_revision": args.revision,
            "packet_count": packet_count,
            "layout": "routingbench_shared_environment_v1",
        }
        (release / "download_manifest.json").write_bytes(
            (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        )
        release.replace(output)

    print(json.dumps({"output": str(output), "packet_count": 82, "revision": args.revision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
