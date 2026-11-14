from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

from .canonicalize import read_json, read_jsonl, sha256_text, write_json, write_jsonl


def sample_states(
    states_path: Path,
    output_dir: Path,
    *,
    count: int,
    seed: int,
    exclude_decision_ids: set[str] | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(states_path)
    excluded = set(exclude_decision_ids or set())
    candidates = [row for row in rows if row["decision_id"] not in excluded]
    if count < 1 or count > len(candidates):
        raise ValueError(f"sample count must be in [1, {len(candidates)}]")
    selected = random.Random(seed).sample(candidates, count)
    selected.sort(key=lambda row: row["decision_id"])
    decision_ids = [row["decision_id"] for row in selected]
    manifest = {
        "sample_manifest_version": "roleroutebench.track_a.sample.v1",
        "source_states": str(states_path.resolve()),
        "source_count": len(rows),
        "candidate_count_after_exclusion": len(candidates),
        "sample_count": count,
        "seed": seed,
        "sampling_method": "python_random_sample_without_replacement_then_sort_by_decision_id",
        "excluded_decision_ids_count": len(excluded),
        "excluded_decision_ids_sha256": sha256_text("\n".join(sorted(excluded))) if excluded else None,
        "decision_ids": decision_ids,
        "decision_ids_sha256": sha256_text("\n".join(decision_ids)),
        "domain_counts": dict(Counter(row["domain"] for row in selected)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_path = output_dir / "sample_manifest.json"
    if existing_path.exists() and read_json(existing_path) != manifest:
        raise ValueError(f"refusing to overwrite a different sample manifest: {existing_path}")
    write_json(existing_path, manifest)
    write_jsonl(output_dir / "sample_states.jsonl", selected)
    return manifest

