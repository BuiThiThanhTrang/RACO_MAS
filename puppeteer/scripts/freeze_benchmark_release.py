"""Freeze or verify a canonical multi-domain benchmark release.

The archive contains packet bytes, shared runtime source/configuration and
bounded historical evidence. External runtimes, credentials and episode
workspaces are deliberately excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sqlite3
import sys
import zipfile
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.assemble_benchmark_release import verify_release  # noqa: E402


EPISODE_FILES = (
    "trajectory.jsonl",
    "decisions.jsonl",
    "decision_states.jsonl",
    "result.json",
    "runtime.json",
)
RUN_ROOT_FILES = {
    "attempt_status.json",
    "difficulty_summary.json",
    "evaluation_report.json",
    "execution_health.json",
    "execution_validation_report.json",
    "freeze_verification.json",
    "human_state_review_queue.jsonl",
    "resolved_config.json",
    "role_cards.json",
    "report.json",
    "source_hashes.json",
    "summary.json",
    "tests.log",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def add_tree(files: dict[str, Path], root: Path, archive_root: str) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            files[f"{archive_root}/{path.relative_to(root).as_posix()}"] = path


def collect_run(files: dict[str, Path], run_name: str) -> dict[str, Any]:
    root = PROJECT / "runs" / run_name
    if not root.is_dir():
        raise FileNotFoundError(root)
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name in RUN_ROOT_FILES:
            files[f"evidence/runs/{run_name}/{path.name}"] = path
    episodes = root / "episodes"
    episode_count = 0
    if episodes.is_dir():
        for episode in sorted(path for path in episodes.iterdir() if path.is_dir()):
            episode_count += 1
            for name in EPISODE_FILES:
                path = episode / name
                if path.is_file():
                    files[f"evidence/runs/{run_name}/episodes/{episode.name}/{name}"] = path
            evaluation = episode / "evaluation"
            if evaluation.is_dir():
                for path in sorted(evaluation.glob("*.json")):
                    files[
                        f"evidence/runs/{run_name}/episodes/{episode.name}/evaluation/{path.name}"
                    ] = path
    summary: dict[str, Any] = {"episodes": episode_count}
    report_path = root / "evaluation_report.json"
    if report_path.is_file():
        report = read_json(report_path)
        for key in (
            "completed",
            "passed",
            "end_task_passed",
            "router_decisions",
            "primary_routing_decisions",
        ):
            if key in report:
                summary[key] = report[key]
    summary["provenance"] = (
        "Historical evidence retains its own resolved_config/source_hashes and "
        "is not relabeled as a rerun on the canonical release."
    )
    return summary


def verify(folder: Path) -> dict[str, Any]:
    manifest = read_json(folder / "manifest.json")
    archive = folder / manifest["archive"]
    errors: list[str] = []
    if not archive.is_file():
        errors.append("missing_archive")
    elif sha256(archive) != manifest["archive_sha256"]:
        errors.append("archive_checksum_mismatch")
    if not errors:
        with zipfile.ZipFile(archive) as handle:
            names = set(handle.namelist())
            expected = {entry["path"] for entry in manifest["files"]}
            if names != expected:
                errors.append("archive_entry_set_mismatch")
            for entry in manifest["files"]:
                if entry["path"] in names and sha256_bytes(handle.read(entry["path"])) != entry["sha256"]:
                    errors.append(f"archive_entry_checksum:{entry['path']}")
    report = {
        "valid": not errors,
        "benchmark_id": manifest["benchmark_id"],
        "packet_release": manifest["packet_release"],
        "packet_count": manifest["packet_count"],
        "archive_files": len(manifest["files"]),
        "archive_sha256": manifest["archive_sha256"],
        "errors": errors,
    }
    write_json(folder / "verification_report.json", report)
    if errors:
        raise ValueError(json.dumps(report, indent=2))
    return report


def build(spec_path: Path) -> Path:
    spec = read_json(spec_path)
    release_root = project_path(spec["output_root"])
    release_check = verify_release(release_root)
    target = project_path(spec["benchmark_output"])
    if target.exists():
        raise FileExistsError(f"Benchmark freeze already exists: {target}")
    target.mkdir(parents=True)

    files: dict[str, Path] = {}
    add_tree(files, release_root, f"packet_release/{release_root.name}")
    add_tree(files, PROJECT / "routing_benchmark" / "v4", "project/routing_benchmark/v4")

    config_paths = {
        spec_path,
        *(project_path(value["run_config"]) for value in spec["domains"].values()),
    }
    for value in spec["domains"].values():
        role_source = value["role_cards"]
        if not role_source.startswith("builtin:"):
            config_paths.add(project_path(role_source))
    for path in sorted(config_paths):
        files[f"project/{path.relative_to(PROJECT).as_posix()}"] = path

    source_paths = [
        PROJECT / "routing_benchmark" / "routers.py",
        PROJECT / "inference" / "policy" / "decision_model_client.py",
        PROJECT / "scripts" / "assemble_benchmark_release.py",
        PROJECT / "scripts" / "freeze_benchmark_release.py",
        PROJECT / "scripts" / "run_shared_environment.py",
        PROJECT / "scripts" / "evaluate_shared_environment.py",
        PROJECT / "scripts" / "validate_shared_environment.py",
        PROJECT / "scripts" / "preflight_shared_domains.py",
        PROJECT / "scripts" / "preflight_spreadsheetbench_packets.py",
        PROJECT / "tests" / "test_shared_environment.py",
        PROJECT / "tests" / "test_routing_benchmark_v4_artifact_execution.py",
        PROJECT / "docs" / "domain_extension_packet_adapter_artifact_standard_vi.md",
        PROJECT / "docs" / "benchmark_domain_packet_completion_playbook_vi.md",
        PROJECT / "docs" / "structured_data_packet_completion_guide_vi.md",
        PROJECT / "docs" / "four_domain_release_migration_vi.md",
    ]
    for path in source_paths:
        if path.is_file():
            files[f"project/{path.relative_to(PROJECT).as_posix()}"] = path

    evidence = {
        run_name: collect_run(files, run_name)
        for run_name in spec.get("evidence_runs", [])
    }
    database_reference = PROJECT / "benchmarks" / "database_v1_20261031"
    for name in ("database_v1.zip", "manifest.json"):
        path = database_reference / name
        if path.is_file():
            files[f"references/database_v1_20261031/{name}"] = path

    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version
    runtime_inventory = {
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "installed_packages": dict(sorted(packages.items())),
        "external_runtime_note": (
            "Repository checkouts, Python venvs, Lean/Mathlib caches, provider services "
            "and credentials are not bundled; provision and preflight after restore."
        ),
    }
    write_json(target / "runtime_inventory.json", runtime_inventory)
    files["release/runtime_inventory.json"] = target / "runtime_inventory.json"

    archive = target / "four_domain_v1.zip"
    entries: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for name, path in sorted(files.items()):
            raw = path.read_bytes()
            handle.writestr(name, raw)
            entries.append(
                {"path": name, "bytes": len(raw), "sha256": sha256_bytes(raw)}
            )
    manifest = {
        "benchmark_id": target.name,
        "packet_release": spec["release_id"],
        "packet_count": release_check["packet_count"],
        "domains": release_check["packets_by_domain"],
        "scope": (
            "Write-once packet/source/config/evidence freeze. Historical evidence is "
            "preserved with original provenance and is not a new cross-router experiment."
        ),
        "archive": archive.name,
        "archive_sha256": sha256(archive),
        "historical_evidence": evidence,
        "files": entries,
    }
    write_json(target / "manifest.json", manifest)
    (target / "README.md").write_text(
        "# Four-domain RoutingBench v4-beta freeze — 2026-11-02\n\n"
        "82 packets: database 12, code debugging 20, formal proof 30 and "
        "structured data processing 20.\n\n"
        "This is a write-once packet/source/config/evidence freeze, not a runtime "
        "container and not a claim that historical runs used this newly assembled "
        "path. Restore external runtimes and run preflight before new evaluation.\n",
        encoding="utf-8",
        newline="\n",
    )
    verify(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if bool(args.spec) == bool(args.verify):
        parser.error("Provide exactly one of --spec or --verify")
    report = verify(args.verify.resolve()) if args.verify else verify(build(args.spec.resolve()))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
