from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_DIR / "data" / "RoutingBench" / "v2_beta"
DEFAULT_OUTPUT = PROJECT_DIR / "hf_release" / "RoutingBench-V2"
SPLITS = ("development", "validation", "test")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def _write_parquet(
    path: Path,
    rows: Iterable[dict[str, Any]],
    schema: pa.Schema | None = None,
) -> int:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write empty split: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(materialized, schema=schema)
    pq.write_table(table, path, compression="zstd")
    return len(materialized)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(root: Path) -> None:
    checksum_path = root / "checksums.sha256"
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        relative = path.relative_to(root).as_posix()
        lines.append(f"{_sha256(path)}  {relative}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def package(source: Path, output: Path) -> dict[str, Any]:
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    scenarios = _read_jsonl(source / "scenarios.jsonl")
    decisions = _read_jsonl(source / "decision_states.jsonl")
    agents = _read_jsonl(source / "agents.jsonl")

    # A configuration must expose one Arrow schema across all of its splits.
    # Several benchmark fields are sparse maps keyed by agent/action identifiers;
    # inferring each split independently would produce incompatible struct fields.
    decision_schema = pa.Table.from_pylist(decisions).schema
    interactive_schema = pa.Table.from_pylist(scenarios).schema
    agent_schema = pa.Table.from_pylist(agents).schema

    counts: dict[str, Any] = {"decision": {}, "interactive": {}}
    for split in SPLITS:
        counts["decision"][split] = _write_parquet(
            output / "data" / "decision" / f"{split}.parquet",
            (row for row in decisions if row.get("split") == split),
            decision_schema,
        )
        counts["interactive"][split] = _write_parquet(
            output / "data" / "interactive" / f"{split}.parquet",
            (row for row in scenarios if row.get("split") == split),
            interactive_schema,
        )
    counts["agent_catalog"] = {
        "train": _write_parquet(
            output / "data" / "agent_catalog" / "train.parquet",
            agents,
            agent_schema,
        )
    }

    output.joinpath("specs").mkdir(parents=True, exist_ok=True)
    for path in (source / "specs").glob("*.json"):
        shutil.copy2(path, output / "specs" / path.name)
    for name in (
        "metadata.json",
        "quota_report.json",
        "validation_report.json",
        "split_manifest.json",
    ):
        shutil.copy2(source / name, output / name)

    manifest = {
        "release": "2.0.0-beta.1",
        "source_dataset_version": metadata["version"],
        "source_seed": metadata["seed"],
        "formats": ["parquet"],
        "configs": counts,
        "oracle_labels_public": True,
        "sealed_test": False,
    }
    (output / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_checksums(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package RoutingBench V2-beta for Hugging Face."
    )
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = package(args.source_dir.resolve(), args.output_dir.resolve())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Hugging Face release written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
