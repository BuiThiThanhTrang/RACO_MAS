"""Build the four-domain live Track B update for the RoutingBench-V4 Hub repo."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "RoutingBench"
RUNS = PROJECT / "runs"

DOMAIN_SPECS = {
    "database_migration_integrity": {
        "packet_root": DATA / "v4_domain_packets_rc1" / "packs" / "database_migration_integrity",
        "runs": ["database_shared_jev_16k_full_20261031"],
        "canonical_runs": ["database_shared_jev_16k_full_20261031"],
        "review_status": "human_verified",
    },
    "code_debugging": {
        "packet_root": DATA / "v4_code_debugging_rc1" / "packs" / "code_debugging",
        "runs": ["code_shared_jev_gemini_v3_20261101"],
        "canonical_runs": ["code_shared_jev_gemini_v3_20261101"],
        "review_status": "human_verified",
    },
    "formal_proof": {
        "packet_root": DATA / "v4_formal_proof_30_20261101" / "packs" / "formal_proof",
        "runs": ["formal_shared_jev_prover_v4_20261031", "formal_minif2f_20_jev_live_20261101"],
        "canonical_runs": ["formal_shared_jev_prover_v4_20261031", "formal_minif2f_20_jev_live_20261101"],
        "review_status": "human_verified",
    },
    "structured_data_processing": {
        "packet_root": DATA / "v4_five_domains_100_20261101" / "packs" / "structured_data_processing",
        "runs": [
            "structured_data_10_jev_qwen35_9b_live_20261102_b",
            "structured_data_250_20_jev_qwen35_9b_live_20261102_c",
            "structured_data_final10_jev_qwen35_9b_live_20261102_b",
            "structured_data_formula2_jev_qwen35_9b_live_20261102_c",
        ],
        "canonical_runs": [
            "structured_data_10_jev_qwen35_9b_live_20261102_b",
            "structured_data_250_20_jev_qwen35_9b_live_20261102_c",
            "structured_data_final10_jev_qwen35_9b_live_20261102_b",
            "structured_data_formula2_jev_qwen35_9b_live_20261102_c",
        ],
        "review_status": "pending",
    },
}

REMOVE_CONFIG_DIRS = {
    "interactive_evaluator",
    "actor_inputs",
    "executor_outputs",
    "recovery_annotations",
    "track_c_state",
    "track_c_structured",
    "track_c_raw_development",
    "track_c_labels",
    "track_c_runtime",
    "track_c_pairs",
    "track_b_tasks",
    "track_b_agents",
    "track_b_trajectories",
    "track_b_results",
    "track_b_runs",
    "track_b_provenance",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def jdump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_episode_sources(spec: dict) -> dict[str, str]:
    selected: dict[str, str] = {}
    for run_id in spec["canonical_runs"]:
        summary = load_json(RUNS / run_id / "summary.json")
        for episode in summary["episodes"]:
            selected[episode["packet_id"]] = run_id
    return selected


def compact_decisions(path: Path) -> list[dict]:
    rows = []
    for item in load_jsonl(path):
        meta = item.get("metadata", {})
        rows.append(
            {
                "decision_id": item.get("decision_id"),
                "agent_id": item.get("agent_id"),
                "budget_before": item.get("budget_before"),
                "dispatch_status": item.get("dispatch_status"),
                "confidence": meta.get("confidence"),
                "probabilities": meta.get("probabilities"),
                "input_tokens": meta.get("input_tokens"),
                "latency_ms": meta.get("latency_ms"),
                "model": meta.get("model"),
                "orchestration_profile": meta.get("orchestration_profile"),
                "prompt_version": meta.get("prompt_version"),
                "role_card_mode": meta.get("role_card_mode"),
                "harness_version": meta.get("harness_version"),
                "observable_rejection_guard_enabled": meta.get("observable_rejection_guard_enabled"),
                "guard_blocked_actions": meta.get("guard_blocked_actions", []),
            }
        )
    return rows


def compact_events(path: Path) -> list[dict]:
    keys = (
        "kind",
        "decision_id",
        "agent_id",
        "status",
        "origin",
        "semantic_delta",
        "state_before_hash",
        "state_after_hash",
    )
    return [{key: item.get(key) for key in keys} for item in load_jsonl(path)]


def task_text(packet_dir: Path, manifest: dict) -> str:
    task_path = packet_dir / "public" / "task.md"
    return task_path.read_text(encoding="utf-8") if task_path.exists() else manifest.get("task", "")


def public_manifest(manifest: dict) -> dict:
    keep = {
        "base_workflow_id",
        "domain",
        "task",
        "adapter_id",
        "public_files",
        "fixture_files",
        "source_type",
        "generator_version",
        "seed",
        "license",
        "human_review_status",
        "packet_review_status",
        "execution_readiness",
        "artifact_schema_version",
        "metadata",
    }
    projected = {key: value for key, value in manifest.items() if key in keep}
    projected["private_files"] = []
    return projected


def build(output: Path) -> dict:
    output = output.resolve()
    if output.name != "RoutingBench-V4" or not (output / "data").is_dir():
        raise ValueError(f"Refusing to modify unexpected output directory: {output}")

    for name in REMOVE_CONFIG_DIRS:
        target = output / "data" / name
        if target.exists():
            shutil.rmtree(target)
    packet_target = output / "track_b_packets"
    if packet_target.exists():
        shutil.rmtree(packet_target)
    packet_target.mkdir()

    tasks: list[dict] = []
    agents: list[dict] = []
    trajectories: list[dict] = []
    results: list[dict] = []
    provenance: list[dict] = []
    run_rows: list[dict] = []

    for domain, spec in DOMAIN_SPECS.items():
        episode_sources = canonical_episode_sources(spec)
        packet_dirs = sorted(p.parent for p in spec["packet_root"].glob("*/manifest.json"))
        if set(episode_sources) != {p.name for p in packet_dirs}:
            missing = sorted({p.name for p in packet_dirs} - set(episode_sources))
            extra = sorted(set(episode_sources) - {p.name for p in packet_dirs})
            raise ValueError(f"Canonical run mismatch for {domain}: missing={missing}, extra={extra}")

        role_seen: set[str] = set()
        for run_id in spec["runs"]:
            for card in load_json(RUNS / run_id / "role_cards.json"):
                agent_id = card["agent_id"]
                if agent_id in role_seen:
                    continue
                role_seen.add(agent_id)
                agents.append(
                    {
                        "domain": domain,
                        "agent_id": agent_id,
                        "role_name": card.get("role_name", agent_id),
                        "role_goal": card.get("role_goal", ""),
                        "semantic_family": card.get("semantic_family"),
                        "tools_json": jdump(card.get("tools", [])),
                        "use_when_json": jdump(card.get("use_when", [])),
                        "avoid_when_json": jdump(card.get("avoid_when", [])),
                        "supported_inputs_json": jdump(card.get("supported_inputs", [])),
                        "expected_outputs_json": jdump(card.get("expected_outputs", [])),
                    }
                )

        eval_maps: dict[str, dict[str, dict]] = {}
        for run_id in spec["runs"]:
            report_path = RUNS / run_id / "evaluation_report.json"
            report = load_json(report_path) if report_path.exists() else {"episodes": []}
            eval_maps[run_id] = {item["packet_id"]: item for item in report.get("episodes", [])}

        for packet_dir in packet_dirs:
            manifest = load_json(packet_dir / "manifest.json")
            packet_id = packet_dir.name
            run_id = episode_sources[packet_id]
            run_dir = RUNS / run_id
            episode_dir = run_dir / "episodes" / packet_id
            result = load_json(episode_dir / "result.json")
            evaluation = eval_maps.get(run_id, {}).get(packet_id, {})
            config = load_json(run_dir / "resolved_config.json")

            copied = packet_target / domain / packet_id
            shutil.copytree(packet_dir, copied)
            public_path = f"track_b_packets/{domain}/{packet_id}/public"
            evaluator_path = f"track_b_packets/{domain}/{packet_id}/private"
            source_review = manifest.get("human_review_status", "pending")
            review = spec["review_status"] if spec["review_status"] == "human_verified" else source_review
            readiness = manifest.get("execution_readiness") or "live_trajectory_completed"
            packet_status = manifest.get("packet_review_status") or (
                "accepted" if review == "human_verified" else "pending"
            )
            tasks.append(
                {
                    "packet_id": packet_id,
                    "domain": domain,
                    "task": task_text(packet_dir, manifest),
                    "adapter_id": manifest.get("adapter_id"),
                    "source_type": manifest.get("source_type"),
                    "license": manifest.get("license", "NOASSERTION"),
                    "human_review_status": review,
                    "packet_review_status": packet_status,
                    "execution_readiness": readiness,
                    "public_packet_path": public_path,
                    "evaluator_packet_path": evaluator_path,
                    "public_manifest_json": jdump(public_manifest(manifest)),
                    "split": "full",
                }
            )
            meta = manifest.get("metadata", {})
            provenance.append(
                {
                    "packet_id": packet_id,
                    "domain": domain,
                    "source_type": manifest.get("source_type"),
                    "source_repository": meta.get("source_repository"),
                    "source_revision": meta.get("source_revision"),
                    "source_split": meta.get("source_split"),
                    "source_record_id": meta.get("source_record_id"),
                    "license": manifest.get("license", "NOASSERTION"),
                    "generator_version": manifest.get("generator_version"),
                    "manifest_sha256": hashlib.sha256((packet_dir / "manifest.json").read_bytes()).hexdigest(),
                    "split": "full",
                }
            )
            decisions = compact_decisions(episode_dir / "decisions.jsonl")
            events = compact_events(episode_dir / "trajectory.jsonl")
            trajectories.append(
                {
                    "packet_id": packet_id,
                    "domain": domain,
                    "run_id": run_id,
                    "router_model": config.get("router", {}).get("model"),
                    "executor_model": config.get("executor", {}).get("model"),
                    "max_calls": config.get("max_calls"),
                    "seed": config.get("seed"),
                    "decision_sequence_json": jdump(decisions),
                    "event_sequence_json": jdump(events),
                    "raw_outputs_included": False,
                    "split": "full",
                }
            )
            audit = evaluation.get("end_task_audit", {})
            metrics = evaluation.get("metrics", {})
            results.append(
                {
                    "packet_id": packet_id,
                    "domain": domain,
                    "run_id": run_id,
                    "trajectory_passed": bool(result.get("passed")),
                    "independent_end_task_passed": audit.get("passed"),
                    "terminal_contract_passed": bool(result.get("terminal_contract_passed")),
                    "terminal_reason": result.get("terminal_reason"),
                    "router_stop_selected": bool(result.get("router_stop_selected")),
                    "primary_routing_decisions": result.get("primary_routing_decisions"),
                    "router_decision_count": result.get("router_decision_count"),
                    "suppressed_by_budget": result.get("suppressed_by_budget"),
                    "dispatched": result.get("dispatched"),
                    "executor_calls": result.get("executor_calls"),
                    "metrics_json": jdump(metrics),
                    "split": "full",
                }
            )

        selected_results = [row for row in results if row["domain"] == domain]
        run_configs = [load_json(RUNS / run_id / "resolved_config.json") for run_id in spec["runs"]]
        run_rows.append(
            {
                "domain": domain,
                "packet_count": len(selected_results),
                "router_model": "typesafe/jev-1.13",
                "executor_models_json": jdump(sorted({c.get("executor", {}).get("model") for c in run_configs})),
                "run_ids_json": jdump(spec["runs"]),
                "trajectory_passed": sum(bool(x["trajectory_passed"]) for x in selected_results),
                "independent_end_task_passed": sum(x["independent_end_task_passed"] is True for x in selected_results),
                "primary_routing_decisions": sum(int(x["primary_routing_decisions"] or 0) for x in selected_results),
                "router_decisions": sum(int(x["router_decision_count"] or 0) for x in selected_results),
                "suppressed_by_budget": sum(int(x["suppressed_by_budget"] or 0) for x in selected_results),
                "packet_human_review_status": spec["review_status"],
                "state_label_review_status": "pending",
                "metric_notes": (
                    "Structured-data headline metrics use the tiered final audit: accepted candidate 11/20; "
                    "public execution 6/20; public validation plus STOP 2/20; trusted hard pass 1/20; "
                    "trusted cases 3/59."
                    if domain == "structured_data_processing"
                    else "Trajectory pass and independent end-task pass are reported separately."
                ),
                "split": "full",
            }
        )

    expected = {"database_migration_integrity": 12, "code_debugging": 20, "formal_proof": 30, "structured_data_processing": 20}
    actual = {domain: sum(row["domain"] == domain for row in tasks) for domain in expected}
    if actual != expected:
        raise ValueError(f"Unexpected packet counts: {actual}")

    tables = {
        "track_b_tasks": tasks,
        "track_b_agents": agents,
        "track_b_trajectories": trajectories,
        "track_b_results": results,
        "track_b_runs": run_rows,
        "track_b_provenance": provenance,
    }
    for name, rows in tables.items():
        target = output / "data" / name
        target.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(target / "full.parquet", index=False)

    report = {
        "schema_version": "routingbench.hf_track_b_package.v1",
        "release": "4.0.0-beta.2",
        "track_a_policy": "unchanged from 4.0.0-beta.1",
        "track_b_split": "full",
        "track_c_policy": "temporarily removed",
        "counts": {name: len(rows) for name, rows in tables.items()},
        "domain_packet_counts": actual,
        "human_review": {
            "database_migration_integrity": "human_verified",
            "code_debugging": "human_verified",
            "formal_proof": "human_verified",
            "structured_data_processing": "pending",
            "routing_decision_labels": "pending_all_domains",
        },
        "private_evaluator_artifacts_published": True,
    }
    write_json(output / "track_b_package_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
