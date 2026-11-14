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
DEFAULT_SOURCE = PROJECT_DIR / "data" / "RoutingBench" / "v3_beta"
DEFAULT_OUTPUT = PROJECT_DIR / "hf_release" / "RoutingBench-V3"
README_TEMPLATE = PROJECT_DIR / "docs" / "hf_routingbench_v3_beta_README.md"
SPLITS = ("development", "validation", "test")
BASELINE_RUNS = (
    "routingbench_v3beta_luna_naive_terminal_v1_dev",
    "routingbench_v3beta_luna_structured_terminal_v1_dev",
    "routingbench_v3beta_luna_naive_gemini_terminal_v1_validation",
    "routingbench_v3beta_luna_structured_gemini_terminal_v1_validation",
)
RELEASE = "3.0.0-beta.2"
JSON_COLUMNS = {
    "decision": (
        "task_context",
        "observed_state",
        "execution_memory",
        "recent_events",
        "budget",
        "agent_pool",
        "action_rationales",
    ),
    "interactive": (
        "fact_descriptions",
        "artifact_descriptions",
        "transitions",
        "public_context",
    ),
    "actor_inputs": ("task_input",),
}


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
        lines.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _with_split(
    rows: list[dict[str, Any]], scenario_splits: dict[str, str]
) -> list[dict[str, Any]]:
    materialized = []
    for row in rows:
        scenario_id = str(row["scenario_id"])
        materialized.append({**row, "split": scenario_splits[scenario_id]})
    return materialized


def _encode_nested_columns(
    rows: list[dict[str, Any]], columns: Iterable[str]
) -> list[dict[str, Any]]:
    """Encode dynamic nested objects as explicit UTF-8 JSON for Hub compatibility."""
    encoded = []
    for row in rows:
        item = dict(row)
        for column in columns:
            if column not in item:
                continue
            item[f"{column}_json"] = json.dumps(
                item.pop(column),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        encoded.append(item)
    return encoded


def _contains_null_type(data_type: pa.DataType) -> bool:
    if pa.types.is_null(data_type):
        return True
    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
        return _contains_null_type(data_type.value_type)
    if pa.types.is_fixed_size_list(data_type):
        return _contains_null_type(data_type.value_type)
    if pa.types.is_struct(data_type):
        return any(_contains_null_type(field.type) for field in data_type)
    if pa.types.is_map(data_type):
        return _contains_null_type(data_type.key_type) or _contains_null_type(
            data_type.item_type
        )
    return False


def _validate_parquet_schemas(output: Path) -> None:
    for path in output.glob("data/**/*.parquet"):
        schema = pq.read_schema(path)
        null_fields = [
            field.name for field in schema if _contains_null_type(field.type)
        ]
        if null_fields:
            raise ValueError(
                f"Hub-incompatible null Arrow type in {path}: {null_fields}"
            )


def _copy_json_files(source: Path, output: Path) -> None:
    output.joinpath("specs").mkdir(parents=True, exist_ok=True)
    for path in (source / "specs").glob("*.json"):
        shutil.copy2(path, output / "specs" / path.name)
    for name in (
        "metadata.json",
        "quota_report.json",
        "validation_report.json",
        "split_manifest.json",
        "source_registry.json",
    ):
        shutil.copy2(source / name, output / name)


def _write_release_docs(output: Path) -> None:
    shutil.copy2(README_TEMPLATE, output / "README.md")
    (output / "VERSION").write_text(f"{RELEASE}\n", encoding="utf-8")
    (output / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        "## 3.0.0-beta.2\n\n"
        "- Encodes dynamic nested objects as documented JSON strings so every "
        "Parquet schema is compatible with the Hugging Face Dataset Viewer.\n"
        "- Adds a packaging-time guard against Arrow null element types.\n\n"
        "## 3.0.0-beta.1\n\n"
        "- First public V3-beta research preview.\n"
        "- Adds licensed MuSiQue and MMLU-Pro task-derived workflows.\n"
        "- Adds live-actor Track B and private-to-actor/evaluation sidecars.\n"
        "- Reserves a zero-cost terminal STOP decision outside actor-call budget.\n"
        "- Human review remains pending for all workflows.\n",
        encoding="utf-8",
    )
    (output / "LICENSE_DATA").write_text(
        "RoutingBench V3-beta is a mixed-license dataset.\n\n"
        "Project-authored workflow specifications, annotations, schemas, and "
        "documentation are released under CC BY 4.0.\n\n"
        "MuSiQue-derived content retains the upstream CC BY 4.0 license.\n"
        "MMLU-Pro-derived content retains the upstream MIT license.\n\n"
        "Users must preserve source attribution and comply with the applicable "
        "license for each record. See source_registry.json and provenance config.\n",
        encoding="utf-8",
    )
    (output / "CITATION.cff").write_text(
        "cff-version: 1.2.0\n"
        "message: If you use this dataset, please cite this dataset release.\n"
        "title: RoutingBench V3-beta\n"
        "type: dataset\n"
        f"version: {RELEASE}\n"
        "authors:\n"
        "  - family-names: Bui\n"
        "    given-names: Trang\n"
        "date-released: 2026-10-12\n"
        "repository-code: https://github.com/BuiThiThanhTrang/RACO_MAS\n",
        encoding="utf-8",
    )


def _collect_baselines(output: Path, runs_dir: Path) -> list[dict[str, Any]]:
    records = []
    baseline_dir = output / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for run_id in BASELINE_RUNS:
        run_dir = runs_dir / run_id
        summary_path = run_dir / "summary.json"
        config_path = run_dir / "resolved_config.json"
        if not summary_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"Required baseline run is missing: {run_id}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        records.append(
            {
                "run_id": run_id,
                "summary": summary,
                "resolved_config": config,
            }
        )
        shutil.copy2(summary_path, baseline_dir / f"{run_id}.summary.json")
        shutil.copy2(config_path, baseline_dir / f"{run_id}.resolved_config.json")
    (output / "baseline_results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return records


def package(source: Path, output: Path, runs_dir: Path) -> dict[str, Any]:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (source / "validation_report.json").read_text(encoding="utf-8")
    )
    if not validation.get("valid") or validation.get("errors"):
        raise ValueError("Refusing to package a dataset that failed formal validation")
    if metadata.get("terminal_stop_budget_policy") != (
        "reserved_zero_cost_terminal_decision"
    ):
        raise ValueError("Unexpected terminal STOP budget policy")

    scenarios = _read_jsonl(source / "scenarios.jsonl")
    decisions = _read_jsonl(source / "decision_states.jsonl")
    agents = _read_jsonl(source / "agents.jsonl")
    task_inputs = _read_jsonl(source / "task_inputs.jsonl")
    evaluations = _read_jsonl(source / "evaluation.jsonl")
    provenance = _read_jsonl(source / "provenance.jsonl")
    reviews = _read_jsonl(source / "human_review_queue.jsonl")
    scenario_splits = {str(row["scenario_id"]): str(row["split"]) for row in scenarios}

    configs = {
        "decision": _encode_nested_columns(decisions, JSON_COLUMNS["decision"]),
        "interactive": _encode_nested_columns(
            scenarios, JSON_COLUMNS["interactive"]
        ),
        "actor_inputs": _encode_nested_columns(
            _with_split(task_inputs, scenario_splits), JSON_COLUMNS["actor_inputs"]
        ),
        "evaluation": _with_split(evaluations, scenario_splits),
        "provenance": _with_split(provenance, scenario_splits),
        "human_review": _with_split(reviews, scenario_splits),
    }
    counts: dict[str, Any] = {}
    for config_name, rows in configs.items():
        schema = pa.Table.from_pylist(rows).schema
        counts[config_name] = {}
        for split in SPLITS:
            counts[config_name][split] = _write_parquet(
                output / "data" / config_name / f"{split}.parquet",
                (row for row in rows if row.get("split") == split),
                schema,
            )
    counts["agent_catalog"] = {
        "train": _write_parquet(
            output / "data" / "agent_catalog" / "train.parquet",
            agents,
            pa.Table.from_pylist(agents).schema,
        )
    }
    _validate_parquet_schemas(output)

    _copy_json_files(source, output)
    _write_release_docs(output)
    baseline_records = _collect_baselines(output, runs_dir)
    manifest = {
        "release": RELEASE,
        "source_dataset_version": metadata["version"],
        "source_seed": metadata["seed"],
        "formats": ["parquet"],
        "nested_fields_encoding": "canonical_json_utf8_with_json_suffix",
        "configs": counts,
        "oracle_labels_public": True,
        "sealed_test": False,
        "human_review_status": "pending",
        "human_review_pending_count": validation["human_review_pending_count"],
        "terminal_stop_budget_policy": metadata["terminal_stop_budget_policy"],
        "baseline_run_ids": [item["run_id"] for item in baseline_records],
    }
    (output / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_checksums(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package RoutingBench V3-beta for Hugging Face."
    )
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs_dir", type=Path, default=PROJECT_DIR / "runs")
    args = parser.parse_args()
    manifest = package(
        args.source_dir.resolve(), args.output_dir.resolve(), args.runs_dir.resolve()
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Hugging Face release written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
