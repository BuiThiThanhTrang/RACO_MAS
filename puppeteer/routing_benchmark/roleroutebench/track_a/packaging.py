from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonicalize import read_jsonl, write_json


def _labeling_cohorts(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.glob("labeling*")
        if path.is_dir() and (path / "sample_manifest.json").exists()
    )


def package_parquet(root: Path, output_dir: Path) -> dict[str, Any]:
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    configs: dict[str, list[dict[str, Any]]] = {
        "track_a_states": read_jsonl(root / "track_a_states.jsonl"),
        "track_a_provenance": read_jsonl(root / "track_a_provenance.jsonl"),
        "track_a_prelabels": [],
        "track_a_reviews": read_jsonl(root / "reviews" / "human_reviews.jsonl"),
    }
    for cohort in _labeling_cohorts(root):
        for row in read_jsonl(cohort / "llm_prelabels.jsonl"):
            configs["track_a_prelabels"].append({"labeling_cohort": cohort.name, **row})

    counts: dict[str, int] = {}
    for name, rows in configs.items():
        counts[name] = len(rows)
        if not rows:
            continue
        normalized = []
        for row in rows:
            normalized.append(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
        target = output_dir / name / "full.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(normalized).to_parquet(target, index=False)
    report = {"configs": counts, "output_dir": str(output_dir.resolve())}
    write_json(output_dir / "package_report.json", report)
    return report

