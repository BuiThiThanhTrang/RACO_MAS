from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from review_app.annotation_store import AnnotationStore
from review_app.data_loader import RoutingBenchReviewDataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export RoutingBench human-review SQLite data to JSONL."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_DIR / "data" / "RoutingBench" / "v4_beta",
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    db_path = (
        args.db or data_dir / "reviews" / "routingbench_v4_reviews.sqlite"
    ).resolve()
    output_dir = (args.output_dir or data_dir / "reviews" / "exports").resolve()
    dataset = RoutingBenchReviewDataset(data_dir)
    store = AnnotationStore(db_path)
    counts = store.export(output_dir, dataset.version)
    print(
        json.dumps(
            {
                "dataset_version": dataset.version,
                "database": str(db_path),
                "output_dir": str(output_dir),
                "counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

