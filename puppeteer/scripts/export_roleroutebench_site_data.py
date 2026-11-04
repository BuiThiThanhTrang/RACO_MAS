#!/usr/bin/env python3
"""Export the public RoleRouteBench Track B tables for the static explorer.

This script intentionally reads only public/analysis tables. It does not read
packet ``private/`` directories or evaluator-only ``track_b_results`` fields
beyond the already published aggregate outcome columns used by the explorer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DOMAIN_LABELS = {
    "database_migration_integrity": "Database migration & integrity",
    "code_debugging": "Code debugging",
    "formal_proof": "Formal proof",
    "structured_data_processing": "Structured data processing",
}


def _records(root: Path, config: str) -> list[dict]:
    frame = pd.read_parquet(root / "data" / config / "full.parquet")
    return frame.where(pd.notnull(frame), None).to_dict(orient="records")


def _json(value: object, fallback: object) -> object:
    if not isinstance(value, str) or not value:
        return fallback
    return json.loads(value)


def build_payload(dataset_root: Path) -> dict:
    tasks = {row["packet_id"]: row for row in _records(dataset_root, "track_b_tasks")}
    trajectories = {
        row["packet_id"]: row for row in _records(dataset_root, "track_b_trajectories")
    }
    results = {row["packet_id"]: row for row in _records(dataset_root, "track_b_results")}
    runs = _records(dataset_root, "track_b_runs")
    agents = _records(dataset_root, "track_b_agents")

    packets: list[dict] = []
    for packet_id, task in sorted(tasks.items()):
        trajectory = trajectories[packet_id]
        result = results[packet_id]
        decisions = _json(trajectory["decision_sequence_json"], [])
        events = {
            event["decision_id"]: event
            for event in _json(trajectory["event_sequence_json"], [])
        }
        steps = []
        for index, decision in enumerate(decisions):
            event = events.get(decision.get("decision_id"), {})
            probabilities = decision.get("probabilities") or {}
            top_choices = sorted(
                (
                    {"role": role, "probability": probability}
                    for role, probability in probabilities.items()
                ),
                key=lambda item: item["probability"],
                reverse=True,
            )[:3]
            steps.append(
                {
                    "index": index + 1,
                    "decision_id": decision.get("decision_id"),
                    "role": decision.get("agent_id"),
                    "budget_before": decision.get("budget_before"),
                    "confidence": decision.get("confidence"),
                    "dispatch_status": decision.get("dispatch_status"),
                    "event_status": event.get("status", "not_executed"),
                    "semantic_delta": event.get("semantic_delta", "not_executed"),
                    "state_before": event.get("state_before_hash"),
                    "state_after": event.get("state_after_hash"),
                    "top_choices": top_choices,
                }
            )

        packets.append(
            {
                "packet_id": packet_id,
                "domain": task["domain"],
                "domain_label": DOMAIN_LABELS.get(task["domain"], task["domain"]),
                "task": task["task"].strip(),
                "source_type": task["source_type"],
                "license": task["license"],
                "packet_review_status": task["packet_review_status"],
                "execution_readiness": task["execution_readiness"],
                "run_id": trajectory["run_id"],
                "router_model": trajectory["router_model"],
                "executor_model": trajectory["executor_model"],
                "max_calls": int(trajectory["max_calls"]),
                "trajectory_passed": bool(result["trajectory_passed"]),
                "independent_end_task_passed": bool(
                    result["independent_end_task_passed"]
                ),
                "terminal_reason": result["terminal_reason"],
                "router_stop_selected": bool(result["router_stop_selected"]),
                "primary_routing_decisions": int(result["primary_routing_decisions"]),
                "raw_routing_decisions": int(result["router_decision_count"]),
                "suppressed_by_budget": int(result["suppressed_by_budget"]),
                "steps": steps,
            }
        )

    role_cards: dict[str, list[dict]] = {}
    for agent in agents:
        role_cards.setdefault(agent["domain"], []).append(
            {
                "role": agent["role_name"],
                "goal": agent["role_goal"],
                "use_when": _json(agent["use_when_json"], []),
                "avoid_when": _json(agent["avoid_when_json"], []),
            }
        )

    return {
        "metadata": {
            "name": "RoleRouteBench",
            "version": "4.0.0-beta.2",
            "doi": "10.57967/hf/11430",
            "doi_revision": "bd936a4b61500e374837702ca08ac631950b2f96",
            "dataset_url": "https://huggingface.co/datasets/TrangBui/RoleRouteBench",
            "router": "typesafe/jev-1.13",
            "packet_count": len(packets),
            "domain_count": len(runs),
        },
        "runs": [
            {
                "domain": row["domain"],
                "domain_label": DOMAIN_LABELS.get(row["domain"], row["domain"]),
                "packet_count": int(row["packet_count"]),
                "trajectory_passed": int(row["trajectory_passed"]),
                "independent_end_task_passed": int(row["independent_end_task_passed"]),
                "primary_routing_decisions": int(row["primary_routing_decisions"]),
                "raw_routing_decisions": int(row["router_decisions"]),
                "suppressed_by_budget": int(row["suppressed_by_budget"]),
            }
            for row in runs
        ],
        "role_cards": role_cards,
        "packets": packets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(args.dataset_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"exported {len(payload['packets'])} packets to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
