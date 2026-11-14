from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from review_app.annotation_store import AnnotationStore
from review_app.data_loader import RoutingBenchReviewDataset


def _expected(dataset: RoutingBenchReviewDataset) -> dict[str, dict[str, list[str]]]:
    return {
        "base_workflow": {
            object_id: list(row.get("checklist", []))
            for object_id, row in dataset.review_queue.items()
        },
        "track_c_pair": {
            object_id: list(row.get("checklist", []))
            for object_id, row in dataset.post_fault_review_queue.items()
        },
    }


def validate(
    dataset: RoutingBenchReviewDataset,
    store: AnnotationStore,
    reviewer_a: str,
    reviewer_b: str,
) -> dict[str, Any]:
    expected = _expected(dataset)
    errors: list[str] = []
    warnings: list[str] = []
    reviewer_counts: dict[str, Any] = {}

    with store.connection() as connection:
        for reviewer in (reviewer_a, reviewer_b):
            rows = connection.execute(
                """
                SELECT object_kind, object_id, checklist_labels_json,
                       checklist_notes_json, status
                FROM v4_reviews
                WHERE dataset_version=? AND reviewer_id=?
                """,
                (dataset.version, reviewer),
            ).fetchall()
            submitted = {
                (str(row["object_kind"]), str(row["object_id"]))
                for row in rows
                if row["status"] == "submitted"
            }
            counts: dict[str, int] = {}
            flagged = 0
            for kind, objects in expected.items():
                completed = {object_id for row_kind, object_id in submitted if row_kind == kind}
                missing = sorted(set(objects) - completed)
                if missing:
                    errors.append(
                        f"{reviewer} has {len(missing)} unsubmitted {kind} objects"
                    )
                counts[kind] = len(completed)

            for row in rows:
                if row["status"] != "submitted":
                    continue
                kind = str(row["object_kind"])
                object_id = str(row["object_id"])
                if kind not in expected or object_id not in expected[kind]:
                    errors.append(f"{reviewer}:{kind}:{object_id} is not in the review queue")
                    continue
                labels = json.loads(row["checklist_labels_json"])
                notes = json.loads(row["checklist_notes_json"])
                items = set(expected[kind][object_id])
                if set(labels) != items:
                    errors.append(
                        f"{reviewer}:{kind}:{object_id} does not label every checklist item"
                    )
                non_approved = [item for item, label in labels.items() if label != "approved"]
                flagged += int(bool(non_approved))
                for item in non_approved:
                    if not str(notes.get(item, "")).strip():
                        errors.append(
                            f"{reviewer}:{kind}:{object_id} has a non-approved item without a note"
                        )
                if non_approved:
                    errors.append(
                        f"{reviewer}:{kind}:{object_id} still requires correction and re-review"
                    )
            reviewer_counts[reviewer] = {**counts, "flagged_objects": flagged}

    disagreements = store.disagreements(dataset.version, reviewer_a, reviewer_b)
    adjudicated = store.adjudicated_ids(dataset.version)
    unresolved = [
        f"{row['object_kind']}:{row['object_id']}"
        for row in disagreements
        if (row["object_kind"], row["object_id"]) not in adjudicated
    ]
    if unresolved:
        errors.append(f"{len(unresolved)} reviewer disagreements remain unadjudicated")

    report = store.agreement_report(dataset.version, reviewer_a, reviewer_b)
    expected_total = sum(len(objects) for objects in expected.values())
    if report["compared"] != expected_total:
        warnings.append(
            "Agreement metrics cover fewer objects than the full queue until both reviews are complete"
        )
    return {
        "valid": not errors,
        "dataset_version": dataset.version,
        "expected": {kind: len(objects) for kind, objects in expected.items()},
        "reviewers": reviewer_counts,
        "agreement": report,
        "adjudicated_objects": len(adjudicated),
        "unresolved_disagreements": unresolved,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate completeness of RoutingBench V4 human review."
    )
    parser.add_argument("--reviewer_a", required=True)
    parser.add_argument("--reviewer_b", required=True)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_DIR / "data" / "RoutingBench" / "v4_beta",
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    db_path = (
        args.db or data_dir / "reviews" / "routingbench_v4_reviews.sqlite"
    ).resolve()
    dataset = RoutingBenchReviewDataset(data_dir)
    store = AnnotationStore(db_path)
    report = validate(dataset, store, args.reviewer_a, args.reviewer_b)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.report.resolve().write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
