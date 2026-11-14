from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from routing_benchmark.v3.generator import _checksums


def _rewrite_jsonl(path: Path, transform) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            target.write(json.dumps(transform(json.loads(line)), ensure_ascii=False) + "\n")
    temporary.replace(path)


def refresh(root: str | Path) -> dict[str, int]:
    """Refresh derived hashes and public/private packaging after source edits.

    This never changes tasks, graphs, valid actions or evaluation labels.
    """

    root = Path(root)
    generator_path = PROJECT_DIR / "routing_benchmark/v4beta/generator.py"
    generator_hash = hashlib.sha256(generator_path.read_bytes()).hexdigest()
    authored_count = 0

    def provenance(row):
        nonlocal authored_count
        if str(row.get("source_type", "")).startswith("project_authored"):
            authored_count += 1
            row["source_sha256"] = generator_hash
            row["source_version"] = f"sha256:{generator_hash}"
        return row

    _rewrite_jsonl(root / "provenance.jsonl", provenance)

    public_count = 0

    def public(row):
        nonlocal public_count
        public_count += 1
        row["workflow_family"] = "dynamic_workflow"
        for observation in (row.get("observations") or {}).values():
            observation["workflow_family"] = "dynamic_workflow"
        return row

    _rewrite_jsonl(root / "public_inputs.jsonl", public)

    labels = {
        str(item["checkpoint_id"]): item
        for item in _read_jsonl(root / "post_fault_labels.jsonl")
    }
    annotations = {
        str(item["scenario_id"]): item
        for item in _read_jsonl(root / "recovery_annotations.jsonl")
    }

    def post_fault(row):
        scenario_id = str(row.get("internal_scenario_id") or "")
        if not scenario_id and row.get("fault_checkpoint_id"):
            scenario_id = str(
                labels.get(str(row["fault_checkpoint_id"]), {}).get(
                    "internal_scenario_id", ""
                )
            )
        annotation = annotations.get(scenario_id, {})
        row["task_domain"] = annotation.get(
            "task_domain", row.get("workflow_family")
        )
        row["workflow_pattern"] = annotation.get("workflow_pattern")
        return row

    for name in (
        "post_fault_inputs.jsonl",
        "post_fault_runtime.jsonl",
        "post_fault_labels.jsonl",
        "post_fault_pair_manifest.jsonl",
    ):
        _rewrite_jsonl(root / name, post_fault)

    generated = [
        path
        for path in root.iterdir()
        if path.is_file()
        and path.name not in {"checksums.json", "validation_report.json"}
    ]
    (root / "checksums.json").write_text(
        json.dumps(_checksums(generated), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"authored_provenance_rows": authored_count, "public_rows": public_count}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh derived V4-beta package metadata without changing labels.")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_DIR / "data/RoutingBench/v4_beta")
    args = parser.parse_args()
    print(json.dumps(refresh(args.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
