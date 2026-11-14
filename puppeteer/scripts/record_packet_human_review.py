"""Record an explicit packet-level human review without claiming state-label review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet_root", type=Path)
    parser.add_argument("--review-date", required=True)
    parser.add_argument("--reviewer", default="user")
    parser.add_argument(
        "--scope",
        default="track_b_packet_publication",
        help="Packet-review scope. This never marks routing-decision labels as reviewed.",
    )
    parser.add_argument(
        "--approval-basis",
        default="Explicit project-owner confirmation after human packet review.",
    )
    args = parser.parse_args()

    changed: list[str] = []
    already_reviewed: list[str] = []
    for manifest_path in sorted(args.packet_root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        packet_id = manifest["base_workflow_id"]
        if manifest.get("human_review_status") == "human_verified":
            already_reviewed.append(packet_id)
            continue

        review = {
            "schema_version": "routingbench.human_review.v1",
            "packet_id": packet_id,
            "domain": manifest["domain"],
            "decision": "approved",
            "human_review_status": "human_verified",
            "packet_review_status": "accepted",
            "reviewer": args.reviewer,
            "technical_reviewer": "Codex",
            "review_date": args.review_date,
            "approval_basis": args.approval_basis,
            "scope": args.scope,
            "routing_decision_label_reviewed": False,
        }
        review_path = manifest_path.parent / "private" / "human_review.json"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(review_path, review)
        review_hash = hashlib.sha256(review_path.read_bytes()).hexdigest()

        private_files = manifest.setdefault("private_files", [])
        private_files[:] = [item for item in private_files if item.get("path") != "private/human_review.json"]
        private_files.append(
            {
                "artifact_id": "private-human_review.json",
                "path": "private/human_review.json",
                "media_type": "application/json",
                "visibility": "private",
                "sha256": review_hash,
            }
        )
        manifest["human_review_status"] = "human_verified"
        manifest["packet_review_status"] = "accepted"
        manifest.setdefault("metadata", {})["human_review"] = {
            "reviewer": args.reviewer,
            "review_date": args.review_date,
            "decision": "approved",
            "record_path": "private/human_review.json",
            "scope": args.scope,
            "routing_decision_label_reviewed": False,
        }
        dump_json(manifest_path, manifest)
        changed.append(packet_id)

    report = {
        "schema_version": "routingbench.packet_review_report.v1",
        "review_date": args.review_date,
        "reviewer": args.reviewer,
        "scope": args.scope,
        "routing_decision_label_reviewed": False,
        "changed_count": len(changed),
        "already_reviewed_count": len(already_reviewed),
        "total_human_verified": len(changed) + len(already_reviewed),
        "changed_packet_ids": changed,
        "already_reviewed_packet_ids": already_reviewed,
    }
    report_path = args.packet_root / f"human_review_report_{args.review_date.replace('-', '')}.json"
    dump_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
