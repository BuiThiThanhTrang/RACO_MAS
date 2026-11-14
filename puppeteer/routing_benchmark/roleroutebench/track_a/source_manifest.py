from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonicalize import read_json, sha256_file, write_json


PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data" / "RoutingBench"
RUNS = PROJECT / "runs"

PUBLICATION_REVISION = "b70803c339a19c767fe9f0718c5218f51d4ed130"
TRACK_B_DATA_REVISION = "cdc67bad1458f6b939f759e3f764f2e93612f424"
RUNTIME_SHA256 = "779cfc4d9e913c26b8145f7435a983cd200771465707f54360fdec15ece2f16e"

DOMAIN_SPECS = {
    "database_migration_integrity": {
        "packet_root": DATA / "v4_domain_packets_rc1" / "packs" / "database_migration_integrity",
        "runs": ["database_shared_jev_16k_full_20261031"],
    },
    "code_debugging": {
        "packet_root": DATA / "v4_code_debugging_rc1" / "packs" / "code_debugging",
        "runs": ["code_shared_jev_gemini_v3_20261101"],
    },
    "formal_proof": {
        "packet_root": DATA / "v4_formal_proof_30_20261101" / "packs" / "formal_proof",
        "runs": ["formal_shared_jev_prover_v4_20261031", "formal_minif2f_20_jev_live_20261101"],
    },
    "structured_data_processing": {
        "packet_root": DATA / "v4_five_domains_100_20261101" / "packs" / "structured_data_processing",
        "runs": [
            "structured_data_10_jev_qwen35_9b_live_20261102_b",
            "structured_data_250_20_jev_qwen35_9b_live_20261102_c",
            "structured_data_final10_jev_qwen35_9b_live_20261102_b",
            "structured_data_formula2_jev_qwen35_9b_live_20261102_c",
        ],
    },
}


def canonical_packet_sources(domain_specs=DOMAIN_SPECS) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for domain, spec in domain_specs.items():
        selected: dict[str, str] = {}
        for run_id in spec["runs"]:
            summary = read_json(RUNS / run_id / "summary.json")
            for episode in summary["episodes"]:
                packet_id = episode["packet_id"]
                # Match the Track B publication script exactly: later runs in
                # the frozen ordered list supersede earlier attempts for the
                # same packet ID.
                selected[packet_id] = run_id
        packet_ids = {path.parent.name for path in spec["packet_root"].glob("*/manifest.json")}
        if set(selected) != packet_ids:
            raise ValueError(
                f"canonical source mismatch for {domain}: "
                f"missing={sorted(packet_ids - set(selected))}, extra={sorted(set(selected) - packet_ids)}"
            )
        result[domain] = dict(sorted(selected.items()))
    return result


def build_source_manifest(domain_specs=None) -> dict[str, Any]:
    custom_sources = domain_specs is not None
    if domain_specs is None:
        domain_specs = DOMAIN_SPECS
    sources = canonical_packet_sources(domain_specs)
    files: list[dict[str, str]] = []
    role_pool_hashes: dict[str, str] = {}
    for domain, mapping in sources.items():
        run_ids = sorted(set(mapping.values()))
        role_hashes: list[str] = []
        for run_id in run_ids:
            role_path = RUNS / run_id / "role_cards.json"
            digest = sha256_file(role_path)
            role_hashes.append(digest)
            files.append({"path": str(role_path.relative_to(PROJECT)), "sha256": digest})
        role_pool_hashes[domain] = "+".join(role_hashes)
        for packet_id, run_id in mapping.items():
            episode = RUNS / run_id / "episodes" / packet_id
            packet_manifest = domain_specs[domain]["packet_root"] / packet_id / "manifest.json"
            for path in (episode / "decision_states.jsonl", episode / "decisions.jsonl", packet_manifest):
                files.append({"path": str(path.relative_to(PROJECT)), "sha256": sha256_file(path)})
    if custom_sources:
        return {
            "manifest_version": "roleroutebench.track_a.source.v2",
            "packet_to_run": sources,
            "role_pool_hashes": role_pool_hashes,
            "input_files": sorted(files, key=lambda item: item["path"]),
        }
    return {
        "manifest_version": "roleroutebench.track_a.source.v1",
        "publication_revision": PUBLICATION_REVISION,
        "track_b_data_revision": TRACK_B_DATA_REVISION,
        "runtime_sha256": RUNTIME_SHA256,
        "expected_counts": {"episodes": 82, "raw": 613, "primary": 568, "audit_only": 45},
        "expected_primary_by_domain": {
            "database_migration_integrity": 84,
            "code_debugging": 143,
            "formal_proof": 185,
            "structured_data_processing": 156,
        },
        "packet_to_run": sources,
        "role_pool_hashes": role_pool_hashes,
        "input_files": sorted(files, key=lambda item: item["path"]),
    }


def write_source_manifest(path: Path) -> dict[str, Any]:
    manifest = build_source_manifest()
    write_json(path, manifest)
    return manifest

