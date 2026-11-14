"""Materialize or verify one canonical, multi-domain packet release.

The assembler copies reviewed packet bytes without rewriting manifests.  A
release is write-once: rebuilding requires a new release ID and output path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from routing_benchmark.v4.artifacts.manifest import (  # noqa: E402
    load_manifest,
    validate_artifact_pack,
)
from routing_benchmark.v4.domain_packets.roles import role_cards_for  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def load_roles(value: str) -> list[dict[str, Any]]:
    if value.startswith("builtin:"):
        return role_cards_for(value.split(":", 1)[1])
    roles = read_json(project_path(value))
    if not isinstance(roles, list) or not roles:
        raise ValueError(f"Role-card source must contain a non-empty list: {value}")
    return roles


def payload_files(release_root: Path) -> list[Path]:
    excluded = {"release_manifest.json"}
    return sorted(
        path
        for path in release_root.rglob("*")
        if path.is_file() and path.name not in excluded
    )


def verify_release(release_root: Path) -> dict[str, Any]:
    manifest_path = release_root / "release_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    release = read_json(manifest_path)
    errors: list[str] = []
    expected_paths = {entry["path"] for entry in release["files"]}
    actual_paths = {
        path.relative_to(release_root).as_posix()
        for path in payload_files(release_root)
    }
    if expected_paths != actual_paths:
        for value in sorted(expected_paths - actual_paths):
            errors.append(f"missing_release_file:{value}")
        for value in sorted(actual_paths - expected_paths):
            errors.append(f"unexpected_release_file:{value}")
    for entry in release["files"]:
        path = release_root / entry["path"]
        if path.is_file() and sha256(path) != entry["sha256"]:
            errors.append(f"release_checksum_mismatch:{entry['path']}")

    counts: dict[str, int] = {}
    adapters: dict[str, set[str]] = {}
    for domain, domain_spec in release["domains"].items():
        manifests = sorted((release_root / "packs" / domain).glob("*/manifest.json"))
        counts[domain] = len(manifests)
        adapters[domain] = set()
        if len(manifests) != domain_spec["packets"]:
            errors.append(
                f"packet_count:{domain}:{len(manifests)}!={domain_spec['packets']}"
            )
        for manifest_path in manifests:
            check = validate_artifact_pack(manifest_path.parent)
            if not check["valid"]:
                errors.extend(
                    f"invalid_pack:{manifest_path.parent.name}:{item}"
                    for item in check["errors"]
                )
            packet = load_manifest(manifest_path)
            adapters[domain].add(packet.adapter_id)
            if packet.domain != domain:
                errors.append(
                    f"domain_mismatch:{manifest_path.parent.name}:{packet.domain}"
                )
        if adapters[domain] != {domain_spec["adapter_id"]}:
            errors.append(
                f"adapter_mismatch:{domain}:{sorted(adapters[domain])}"
            )

    report = {
        "valid": not errors,
        "release_id": release["release_id"],
        "packet_count": sum(counts.values()),
        "packets_by_domain": counts,
        "adapters_by_domain": {
            domain: sorted(values) for domain, values in adapters.items()
        },
        "payload_files": len(actual_paths),
        "errors": errors,
    }
    if errors:
        raise ValueError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build(spec_path: Path) -> Path:
    spec = read_json(spec_path)
    release_root = project_path(spec["output_root"])
    if release_root.exists():
        raise FileExistsError(
            f"Release already exists: {release_root}. Use --verify or a new release ID."
        )
    release_root.mkdir(parents=True)

    source_lock: dict[str, Any] = {
        "release_id": spec["release_id"],
        "policy": "Source packet bytes are copied without manifest rewriting.",
        "domains": {},
    }
    release_domains: dict[str, Any] = {}
    validation_rows: list[dict[str, Any]] = []

    for domain, domain_spec in spec["domains"].items():
        source_root = project_path(domain_spec["source_root"])
        source_manifests = sorted((source_root / "packs" / domain).glob("*/manifest.json"))
        expected = int(domain_spec["expected_packets"])
        if len(source_manifests) != expected:
            raise ValueError(f"{domain}: expected {expected} packets, found {len(source_manifests)}")

        destination = release_root / "packs" / domain
        destination.mkdir(parents=True)
        packet_ids: list[str] = []
        source_entries: list[dict[str, str]] = []
        for manifest_path in source_manifests:
            check = validate_artifact_pack(manifest_path.parent)
            if not check["valid"]:
                raise ValueError(json.dumps(check, ensure_ascii=False, indent=2))
            manifest = load_manifest(manifest_path)
            if manifest.domain != domain or manifest.adapter_id != domain_spec["adapter_id"]:
                raise ValueError(
                    f"Contract mismatch in {manifest_path}: {manifest.domain}/{manifest.adapter_id}"
                )
            shutil.copytree(manifest_path.parent, destination / manifest_path.parent.name)
            packet_ids.append(manifest.base_workflow_id)
            source_entries.append(
                {
                    "packet_id": manifest.base_workflow_id,
                    "manifest": manifest_path.relative_to(PROJECT).as_posix(),
                    "manifest_sha256": sha256(manifest_path),
                }
            )
            validation_rows.append(check)

        role_path = release_root / "agent_pools" / f"{domain}.json"
        write_json(role_path, load_roles(domain_spec["role_cards"]))
        source_lock["domains"][domain] = {
            "source_root": domain_spec["source_root"],
            "packets": source_entries,
            "role_cards_source": domain_spec["role_cards"],
            "run_config": domain_spec["run_config"],
        }
        release_domains[domain] = {
            "packets": expected,
            "packet_ids": packet_ids,
            "adapter_id": domain_spec["adapter_id"],
            "agent_pool": role_path.relative_to(release_root).as_posix(),
            "source_root": domain_spec["source_root"],
            "run_config": domain_spec["run_config"],
        }

    write_json(release_root / "source_lock.json", source_lock)
    write_json(
        release_root / "routing_contract.json",
        {
            "release_id": spec["release_id"],
            "purpose": "Evaluate a pluggable decision model routing bounded specialist roles over multi-step tasks.",
            "common_observation_policy": "The router sees only adapter-produced public state and prior public action outcomes.",
            "executor_policy": "Each selected role receives its frozen role card and only adapter-authorized tools.",
            "termination_policy": "STOP is evaluated against the adapter terminal contract; budget-suppressed decisions are recorded separately.",
            "mutation_policy": "All executor mutations and selected recovery actions remain observable in trajectory evidence.",
            "performance_policy": "The environment supplies information and deterministic checks; it does not prescribe an optimal workflow.",
        },
    )
    write_json(
        release_root / "validation_report.json",
        {
            "valid": True,
            "checks": len(validation_rows),
            "packets_by_domain": {
                domain: value["packets"] for domain, value in release_domains.items()
            },
            "packet_validation": validation_rows,
        },
    )
    files = [
        {
            "path": path.relative_to(release_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in payload_files(release_root)
    ]
    write_json(
        release_root / "release_manifest.json",
        {
            "release_id": spec["release_id"],
            "description": spec["description"],
            "write_once": True,
            "packet_count": sum(value["packets"] for value in release_domains.values()),
            "domains": release_domains,
            "files": files,
        },
    )
    verify_release(release_root)
    return release_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if bool(args.spec) == bool(args.verify):
        parser.error("Provide exactly one of --spec or --verify")
    if args.verify:
        report = verify_release(args.verify.resolve())
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    release = build(args.spec.resolve())
    print(json.dumps(verify_release(release), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
