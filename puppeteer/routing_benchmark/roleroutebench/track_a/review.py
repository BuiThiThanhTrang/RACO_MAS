from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonicalize import append_jsonl, read_jsonl, sha256_text
from .schema import derived_action_sets, validate_prelabel


def make_review_record(
    state: dict[str, Any],
    prelabel: dict[str, Any],
    reviewed_classification: dict[str, Any],
    *,
    reviewer_id: str,
    decision: str,
    correction_category: str | None = None,
) -> dict[str, Any]:
    candidates = __import__("json").loads(state["candidate_actions_json"])
    errors = validate_prelabel(reviewed_classification, candidates)
    if errors:
        raise ValueError("; ".join(errors))
    if decision not in {"accepted", "corrected", "rejected_for_adjudication"}:
        raise ValueError(f"invalid review decision: {decision}")
    sets = derived_action_sets(reviewed_classification)
    payload = {
        "decision_id": state["decision_id"],
        "public_state_sha256": state["public_state_sha256"],
        "prelabel_prompt_version": prelabel["prompt_version"],
        "reviewer_id": reviewer_id,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "correction_category": correction_category,
        "classification": reviewed_classification,
        **sets,
        "terminal_valid": reviewed_classification["terminal_valid"],
        "state_phase": reviewed_classification["state_phase"],
        "label_status": "human_verified" if decision != "rejected_for_adjudication" else "pending_adjudication",
    }
    payload["review_record_sha256"] = sha256_text(
        __import__("json").dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return payload


def append_review(path: Path, record: dict[str, Any]) -> None:
    append_jsonl(path, record)


def latest_reviews(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        latest[row["decision_id"]] = row
    return latest

