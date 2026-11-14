from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .canonicalize import canonical_json, read_jsonl, sha256_file, sha256_text, write_json, write_jsonl
from .source_manifest import DOMAIN_SPECS, PROJECT, RUNS, build_source_manifest


def _rows_for_packet(domain: str, packet_id: str, run_id: str, packet_root: Path) -> tuple[list[dict], list[dict]]:
    episode = RUNS / run_id / "episodes" / packet_id
    states = read_jsonl(episode / "decision_states.jsonl")
    decisions = read_jsonl(episode / "decisions.jsonl")
    by_id = {row["decision_id"]: row for row in decisions}
    if set(by_id) != {row["decision_id"] for row in states}:
        raise ValueError(f"state/decision mismatch: {packet_id}")
    packet_manifest = packet_root / packet_id / "manifest.json"
    packet_manifest_sha = sha256_file(packet_manifest)
    public_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        decision = by_id[state["decision_id"]]
        observation = state["observation"]
        candidates = state["candidate_actions"]
        allowed = observation.get("constraints", {}).get("allowed_actions")
        if candidates != allowed:
            raise ValueError(f"candidate mismatch: {state['decision_id']}")
        dispatch_status = decision.get("dispatch_status")
        primary = dispatch_status != "suppressed_by_budget"
        router_input_json = canonical_json(observation)
        state_hash = sha256_text(router_input_json)
        public_rows.append(
            {
                "sample_id": f"rra-{state_hash[:24]}",
                "decision_id": state["decision_id"],
                "domain": domain,
                "packet_id": packet_id,
                "router_input_json": router_input_json,
                "candidate_actions_json": canonical_json(candidates),
                "budget_remaining": observation.get("budget", {}).get("calls_remaining"),
                "primary_eligible": primary,
                "exclusion_reason": None if primary else "suppressed_by_budget",
                "public_state_sha256": state_hash,
                "split": "full",
            }
        )
        provenance_rows.append(
            {
                "decision_id": state["decision_id"],
                "domain": domain,
                "packet_id": packet_id,
                "source_run_id": run_id,
                "source_decision_index": index,
                "source_dispatch_status": dispatch_status,
                "source_action_as_run": decision.get("agent_id"),
                "source_packet_manifest_sha256": packet_manifest_sha,
                "public_state_sha256": state_hash,
                "source_state_file": str((episode / "decision_states.jsonl").relative_to(PROJECT)),
                "source_decision_file": str((episode / "decisions.jsonl").relative_to(PROJECT)),
            }
        )
    return public_rows, provenance_rows


def build_track_a(output_dir: Path, domain_specs=None) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest = build_source_manifest(domain_specs)
    specs = DOMAIN_SPECS if domain_specs is None else domain_specs
    raw_rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for domain, mapping in manifest["packet_to_run"].items():
        for packet_id, run_id in mapping.items():
            rows, prov = _rows_for_packet(domain, packet_id, run_id, specs[domain]["packet_root"])
            raw_rows.extend(rows)
            provenance.extend(prov)
    primary = [row for row in raw_rows if row["primary_eligible"]]
    audit = [row for row in raw_rows if not row["primary_eligible"]]
    fingerprints = [row["public_state_sha256"] for row in primary]
    if len(set(fingerprints)) != len(fingerprints):
        duplicates = [key for key, count in Counter(fingerprints).items() if count > 1]
        raise ValueError(f"duplicate primary public states: {duplicates[:5]}")
    counts = {
        "episodes": sum(len(mapping) for mapping in manifest["packet_to_run"].values()),
        "raw": len(raw_rows),
        "primary": len(primary),
        "audit_only": len(audit),
        "primary_by_domain": dict(Counter(row["domain"] for row in primary)),
    }
    if domain_specs is not None:
        manifest["expected_counts"] = {key: value for key, value in counts.items() if key != "primary_by_domain"}
        manifest["expected_primary_by_domain"] = counts["primary_by_domain"]
    expected = manifest["expected_counts"]
    for key in ("episodes", "raw", "primary", "audit_only"):
        if counts[key] != expected[key]:
            raise ValueError(f"unexpected {key}: {counts[key]} != {expected[key]}")
    if counts["primary_by_domain"] != manifest["expected_primary_by_domain"]:
        raise ValueError(f"unexpected domain counts: {counts['primary_by_domain']}")
    write_json(output_dir / "source_manifest.json", manifest)
    write_jsonl(output_dir / "raw_states.jsonl", raw_rows)
    write_jsonl(output_dir / "track_a_states.jsonl", primary)
    write_jsonl(output_dir / "audit_only_states.jsonl", audit)
    write_jsonl(output_dir / "track_a_provenance.jsonl", provenance)
    report = {"status": "valid", "counts": counts, "output_dir": str(output_dir)}
    write_json(output_dir / "build_report.json", report)
    return report


def decode_router_input(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["router_input_json"])

