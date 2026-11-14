from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


REVIEW_LABELS = frozenset({"approved", "needs_revision", "reject"})
REVIEW_STATUSES = frozenset({"draft", "submitted"})
OBJECT_KINDS = frozenset({"base_workflow", "track_c_pair"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str | None, default: Any) -> Any:
    return json.loads(value) if value else default


class AnnotationStore:
    """SQLite working store for two-reviewer RoutingBench V4 annotation."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS v4_reviews (
                    dataset_version TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    object_kind TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    checklist_labels_json TEXT NOT NULL,
                    checklist_notes_json TEXT NOT NULL,
                    summary_notes TEXT NOT NULL,
                    status TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    reviewed_at TEXT NOT NULL,
                    PRIMARY KEY (dataset_version, reviewer_id, object_kind, object_id)
                );

                CREATE TABLE IF NOT EXISTS v4_adjudications (
                    dataset_version TEXT NOT NULL,
                    object_kind TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    reviewer_a TEXT NOT NULL,
                    reviewer_b TEXT NOT NULL,
                    reviewer_a_labels_json TEXT NOT NULL,
                    reviewer_b_labels_json TEXT NOT NULL,
                    final_labels_json TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    adjudicator_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    adjudicated_at TEXT NOT NULL,
                    PRIMARY KEY (dataset_version, object_kind, object_id)
                );

                CREATE TABLE IF NOT EXISTS v4_review_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    object_kind TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_v4_reviews_object
                ON v4_reviews(dataset_version, object_kind, object_id);
                CREATE INDEX IF NOT EXISTS idx_v4_review_events_object
                ON v4_review_events(dataset_version, object_kind, object_id);
                """
            )

    @staticmethod
    def _validate_reviewer_id(reviewer_id: str) -> str:
        reviewer_id = reviewer_id.strip()
        if not reviewer_id:
            raise ValueError("Reviewer ID is required")
        if len(reviewer_id) > 80:
            raise ValueError("Reviewer ID must be at most 80 characters")
        return reviewer_id

    @staticmethod
    def _validate_kind(object_kind: str) -> str:
        if object_kind not in OBJECT_KINDS:
            raise ValueError(f"Unsupported review object: {object_kind}")
        return object_kind

    def save_review(
        self,
        *,
        dataset_version: str,
        reviewer_id: str,
        object_kind: str,
        object_id: str,
        checklist_items: list[str],
        checklist_labels: dict[str, str],
        checklist_notes: dict[str, str] | None = None,
        summary_notes: str = "",
        status: str = "draft",
    ) -> dict[str, Any]:
        reviewer_id = self._validate_reviewer_id(reviewer_id)
        object_kind = self._validate_kind(object_kind)
        if status not in REVIEW_STATUSES:
            raise ValueError(f"Unsupported review status: {status}")

        items = list(dict.fromkeys(str(item) for item in checklist_items))
        labels = {str(key): str(value) for key, value in checklist_labels.items()}
        notes = {
            str(key): str(value).strip()
            for key, value in (checklist_notes or {}).items()
            if str(value).strip()
        }
        unknown_items = (set(labels) | set(notes)) - set(items)
        if unknown_items:
            raise ValueError(f"Unknown checklist items: {sorted(unknown_items)}")
        invalid_labels = set(labels.values()) - REVIEW_LABELS
        if invalid_labels:
            raise ValueError(f"Unsupported checklist labels: {sorted(invalid_labels)}")
        if status == "submitted":
            missing = set(items) - set(labels)
            if missing:
                raise ValueError(f"Submitted review is missing {len(missing)} checklist labels")
            missing_notes = [
                item
                for item in items
                if labels.get(item) in {"needs_revision", "reject"}
                and not notes.get(item, "").strip()
            ]
            if missing_notes:
                raise ValueError(
                    "Every needs_revision or reject label requires an item note"
                )

        payload = {
            "checklist_labels": labels,
            "checklist_notes": notes,
            "summary_notes": summary_notes.strip(),
            "status": status,
        }
        timestamp = _now()
        with self.connection() as connection:
            current = connection.execute(
                """
                SELECT revision FROM v4_reviews
                WHERE dataset_version=? AND reviewer_id=?
                  AND object_kind=? AND object_id=?
                """,
                (dataset_version, reviewer_id, object_kind, object_id),
            ).fetchone()
            revision = int(current["revision"]) + 1 if current else 1
            connection.execute(
                """
                INSERT INTO v4_reviews (
                    dataset_version, reviewer_id, object_kind, object_id,
                    checklist_labels_json, checklist_notes_json, summary_notes,
                    status, revision, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_version, reviewer_id, object_kind, object_id)
                DO UPDATE SET checklist_labels_json=excluded.checklist_labels_json,
                    checklist_notes_json=excluded.checklist_notes_json,
                    summary_notes=excluded.summary_notes,
                    status=excluded.status, revision=excluded.revision,
                    reviewed_at=excluded.reviewed_at
                """,
                (
                    dataset_version,
                    reviewer_id,
                    object_kind,
                    object_id,
                    _json(labels),
                    _json(notes),
                    summary_notes.strip(),
                    status,
                    revision,
                    timestamp,
                ),
            )
            connection.execute(
                """
                INSERT INTO v4_review_events (
                    event_type, dataset_version, actor_id, object_kind,
                    object_id, revision, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "review_saved",
                    dataset_version,
                    reviewer_id,
                    object_kind,
                    object_id,
                    revision,
                    _json(payload),
                    timestamp,
                ),
            )
        return {**payload, "revision": revision, "reviewed_at": timestamp}

    def get_review(
        self,
        dataset_version: str,
        reviewer_id: str,
        object_kind: str,
        object_id: str,
    ) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM v4_reviews
                WHERE dataset_version=? AND reviewer_id=?
                  AND object_kind=? AND object_id=?
                """,
                (dataset_version, reviewer_id, object_kind, object_id),
            ).fetchone()
        return self._review_row(row) if row else None

    @staticmethod
    def _review_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["checklist_labels"] = _loads(item.pop("checklist_labels_json"), {})
        item["checklist_notes"] = _loads(item.pop("checklist_notes_json"), {})
        return item

    def reviewer_ids(self, dataset_version: str) -> list[str]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT reviewer_id FROM v4_reviews
                WHERE dataset_version=? ORDER BY reviewer_id
                """,
                (dataset_version,),
            ).fetchall()
        return [str(row["reviewer_id"]) for row in rows]

    def progress(
        self,
        dataset_version: str,
        reviewer_id: str,
        object_kind: str,
        total: int,
    ) -> dict[str, int | float]:
        self._validate_kind(object_kind)
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS saved,
                       SUM(CASE WHEN status='submitted' THEN 1 ELSE 0 END) AS submitted,
                       SUM(CASE WHEN status='submitted' AND (
                           checklist_labels_json LIKE '%needs_revision%' OR
                           checklist_labels_json LIKE '%reject%'
                       ) THEN 1 ELSE 0 END) AS flagged
                FROM v4_reviews
                WHERE dataset_version=? AND reviewer_id=? AND object_kind=?
                """,
                (dataset_version, reviewer_id, object_kind),
            ).fetchone()
        submitted = int(row["submitted"] or 0)
        return {
            "saved": int(row["saved"] or 0),
            "submitted": submitted,
            "flagged": int(row["flagged"] or 0),
            "total": total,
            "percent": (100.0 * submitted / total) if total else 0.0,
        }

    def disagreements(
        self,
        dataset_version: str,
        reviewer_a: str,
        reviewer_b: str,
        object_kind: str | None = None,
    ) -> list[dict[str, Any]]:
        params: list[Any] = [dataset_version, reviewer_a, reviewer_b]
        kind_clause = ""
        if object_kind:
            self._validate_kind(object_kind)
            kind_clause = " AND a.object_kind=?"
            params.append(object_kind)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT a.object_kind, a.object_id,
                       a.checklist_labels_json AS labels_a,
                       b.checklist_labels_json AS labels_b,
                       a.checklist_notes_json AS notes_a,
                       b.checklist_notes_json AS notes_b
                FROM v4_reviews a
                JOIN v4_reviews b
                  ON a.dataset_version=b.dataset_version
                 AND a.object_kind=b.object_kind
                 AND a.object_id=b.object_id
                WHERE a.dataset_version=?
                  AND a.reviewer_id=? AND b.reviewer_id=?
                  AND a.status='submitted' AND b.status='submitted'
                  {kind_clause}
                ORDER BY a.object_kind, a.object_id
                """,
                params,
            ).fetchall()
        result = []
        for row in rows:
            labels_a = _loads(row["labels_a"], {})
            labels_b = _loads(row["labels_b"], {})
            if labels_a != labels_b:
                result.append(
                    {
                        "object_kind": str(row["object_kind"]),
                        "object_id": str(row["object_id"]),
                        "reviewer_a_labels": labels_a,
                        "reviewer_b_labels": labels_b,
                        "reviewer_a_notes": _loads(row["notes_a"], {}),
                        "reviewer_b_notes": _loads(row["notes_b"], {}),
                    }
                )
        return result

    def agreement_report(
        self, dataset_version: str, reviewer_a: str, reviewer_b: str
    ) -> dict[str, Any]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT a.object_kind,
                       a.checklist_labels_json AS labels_a,
                       b.checklist_labels_json AS labels_b
                FROM v4_reviews a
                JOIN v4_reviews b
                  ON a.dataset_version=b.dataset_version
                 AND a.object_kind=b.object_kind
                 AND a.object_id=b.object_id
                WHERE a.dataset_version=?
                  AND a.reviewer_id=? AND b.reviewer_id=?
                  AND a.status='submitted' AND b.status='submitted'
                """,
                (dataset_version, reviewer_a, reviewer_b),
            ).fetchall()
        totals = {kind: 0 for kind in OBJECT_KINDS}
        exact = {kind: 0 for kind in OBJECT_KINDS}
        for row in rows:
            kind = str(row["object_kind"])
            totals[kind] += 1
            exact[kind] += int(_loads(row["labels_a"], {}) == _loads(row["labels_b"], {}))
        compared = len(rows)
        exact_total = sum(exact.values())
        return {
            "dataset_version": dataset_version,
            "reviewer_a": reviewer_a,
            "reviewer_b": reviewer_b,
            "compared": compared,
            "exact_agreement": exact_total / compared if compared else None,
            "by_kind": {
                kind: {
                    "compared": totals[kind],
                    "exact": exact[kind],
                    "rate": exact[kind] / totals[kind] if totals[kind] else None,
                }
                for kind in sorted(OBJECT_KINDS)
            },
            "disagreement_count": len(
                self.disagreements(dataset_version, reviewer_a, reviewer_b)
            ),
        }

    def save_adjudication(
        self,
        *,
        dataset_version: str,
        object_kind: str,
        object_id: str,
        checklist_items: list[str],
        reviewer_a: str,
        reviewer_b: str,
        reviewer_a_labels: dict[str, str],
        reviewer_b_labels: dict[str, str],
        final_labels: dict[str, str],
        rationale: str,
        adjudicator_id: str,
    ) -> dict[str, Any]:
        adjudicator_id = self._validate_reviewer_id(adjudicator_id)
        object_kind = self._validate_kind(object_kind)
        items = set(checklist_items)
        if set(final_labels) != items:
            raise ValueError("Adjudication must resolve every checklist item")
        invalid = set(final_labels.values()) - REVIEW_LABELS
        if invalid:
            raise ValueError(f"Unsupported final labels: {sorted(invalid)}")
        if not rationale.strip():
            raise ValueError("Adjudication rationale is required")

        timestamp = _now()
        with self.connection() as connection:
            current = connection.execute(
                """
                SELECT revision FROM v4_adjudications
                WHERE dataset_version=? AND object_kind=? AND object_id=?
                """,
                (dataset_version, object_kind, object_id),
            ).fetchone()
            revision = int(current["revision"]) + 1 if current else 1
            connection.execute(
                """
                INSERT INTO v4_adjudications (
                    dataset_version, object_kind, object_id, reviewer_a,
                    reviewer_b, reviewer_a_labels_json, reviewer_b_labels_json,
                    final_labels_json, rationale, adjudicator_id, revision,
                    adjudicated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_version, object_kind, object_id)
                DO UPDATE SET reviewer_a=excluded.reviewer_a,
                    reviewer_b=excluded.reviewer_b,
                    reviewer_a_labels_json=excluded.reviewer_a_labels_json,
                    reviewer_b_labels_json=excluded.reviewer_b_labels_json,
                    final_labels_json=excluded.final_labels_json,
                    rationale=excluded.rationale,
                    adjudicator_id=excluded.adjudicator_id,
                    revision=excluded.revision,
                    adjudicated_at=excluded.adjudicated_at
                """,
                (
                    dataset_version,
                    object_kind,
                    object_id,
                    reviewer_a,
                    reviewer_b,
                    _json(reviewer_a_labels),
                    _json(reviewer_b_labels),
                    _json(final_labels),
                    rationale.strip(),
                    adjudicator_id,
                    revision,
                    timestamp,
                ),
            )
            payload = {
                "reviewer_a": reviewer_a,
                "reviewer_b": reviewer_b,
                "final_labels": final_labels,
                "rationale": rationale.strip(),
            }
            connection.execute(
                """
                INSERT INTO v4_review_events (
                    event_type, dataset_version, actor_id, object_kind,
                    object_id, revision, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "adjudication_saved",
                    dataset_version,
                    adjudicator_id,
                    object_kind,
                    object_id,
                    revision,
                    _json(payload),
                    timestamp,
                ),
            )
        return {
            "object_kind": object_kind,
            "object_id": object_id,
            "final_labels": final_labels,
            "revision": revision,
            "adjudicated_at": timestamp,
        }

    def adjudicated_ids(self, dataset_version: str) -> set[tuple[str, str]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT object_kind, object_id FROM v4_adjudications
                WHERE dataset_version=?
                """,
                (dataset_version,),
            ).fetchall()
        return {(str(row["object_kind"]), str(row["object_id"])) for row in rows}

    def export(self, output_dir: str | Path, dataset_version: str) -> dict[str, int]:
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        specs = {
            "base_workflow_reviews": (
                "v4_reviews",
                "base_workflow_reviews.jsonl",
                "dataset_version=? AND object_kind='base_workflow' ORDER BY reviewer_id, object_id",
            ),
            "track_c_pair_reviews": (
                "v4_reviews",
                "track_c_pair_reviews.jsonl",
                "dataset_version=? AND object_kind='track_c_pair' ORDER BY reviewer_id, object_id",
            ),
            "adjudications": (
                "v4_adjudications",
                "review_adjudications.jsonl",
                "dataset_version=? ORDER BY object_kind, object_id",
            ),
            "events": (
                "v4_review_events",
                "review_events.jsonl",
                "dataset_version=? ORDER BY event_id",
            ),
        }
        counts: dict[str, int] = {}
        with self.connection() as connection:
            for name, (table, filename, where) in specs.items():
                rows = connection.execute(
                    f"SELECT * FROM {table} WHERE {where}", (dataset_version,)
                ).fetchall()
                decoded = [self._decode_export_row(table, dict(row)) for row in rows]
                with (output / filename).open("w", encoding="utf-8") as handle:
                    for item in decoded:
                        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[name] = len(decoded)
        return counts

    @staticmethod
    def _decode_export_row(table: str, item: dict[str, Any]) -> dict[str, Any]:
        columns = {
            "v4_reviews": ("checklist_labels_json", "checklist_notes_json"),
            "v4_adjudications": (
                "reviewer_a_labels_json",
                "reviewer_b_labels_json",
                "final_labels_json",
            ),
            "v4_review_events": ("payload_json",),
        }
        for column in columns.get(table, ()):
            item[column.removesuffix("_json")] = _loads(item.pop(column), None)
        return item
