#!/usr/bin/env python3
"""Certify a frozen RoutingBench v4 release without changing its runtime.

The checks in this module are deliberately out-of-band: they instantiate the
existing adapters, inspect the resulting public model contexts, and verify a
pre-recorded checksum boundary.  They never alter routing or execution policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from routing_benchmark.v4.artifacts.manifest import load_manifest
from routing_benchmark.v4.artifacts.schema import sha256_file
from routing_benchmark.v4.execution.registry import live_adapter_for_manifest


FORBIDDEN_MODEL_MARKERS = (
    "private/",
    "gold.patch",
    "reference.lean",
    "answer.xlsx",
    "evaluation_target.json",
)
EXPECTED_DOMAIN_COUNTS = {
    "database_migration_integrity": 12,
    "code_debugging": 20,
    "formal_proof": 30,
    "structured_data_processing": 20,
}


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).casefold()


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value.casefold()
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key).casefold()
            yield from _strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _strings(item)


def _leaks(value: Any) -> list[str]:
    values = tuple(_strings(value))
    leaked = [marker for marker in FORBIDDEN_MODEL_MARKERS if any(marker in text for text in values)]
    # Match a real Windows path segment, but not the escaped closing quote in
    # a JSON preview such as `"evaluator_oracle_visibility": "private"`.
    if any(re.search(r"(?:^|[\\/])private[\\/](?![\"'])", text) for text in values):
        leaked.append("private-path-segment")
    return leaked


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_files(project: Path, manifest: dict[str, Any]) -> list[Path]:
    selected: set[Path] = set()
    for pattern in manifest["path_globs"]:
        selected.update(path for path in project.glob(pattern) if path.is_file())
    for relative in manifest.get("exclude", []):
        selected.discard(project / relative)
    return sorted(selected, key=lambda path: path.relative_to(project).as_posix())


def runtime_aggregate(project: Path, files: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(project).as_posix().encode("utf-8")
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(file_hash)
        digest.update(b"\n")
    return digest.hexdigest()


def verify_runtime_manifest(project: Path, path: Path) -> dict[str, Any]:
    manifest = _load_json(path)
    files = _runtime_files(project, manifest)
    aggregate = runtime_aggregate(project, files)
    errors: list[str] = []
    if len(files) != manifest["file_count"]:
        errors.append(f"runtime_file_count:{len(files)}!={manifest['file_count']}")
    if aggregate != manifest["aggregate_sha256"]:
        errors.append(f"runtime_aggregate:{aggregate}!={manifest['aggregate_sha256']}")
    return {
        "valid": not errors,
        "file_count": len(files),
        "aggregate_sha256": aggregate,
        "errors": errors,
    }


def certify_packets(release_root: Path) -> dict[str, Any]:
    packs = release_root / "packs"
    role_root = release_root / "agent_pools"
    errors: list[str] = []
    counts: dict[str, int] = {}
    checked_contexts = 0

    manifest_paths = sorted(packs.glob("*/*/manifest.json"))
    temp_parent = PROJECT / ".test-tmp"
    temp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="routingbench-v4-cert-", dir=temp_parent) as temp:
        temp_root = Path(temp)
        for index, manifest_path in enumerate(manifest_paths):
            manifest = load_manifest(manifest_path)
            counts[manifest.domain] = counts.get(manifest.domain, 0) + 1
            for artifact in (*manifest.public_files, *manifest.private_files, *manifest.fixture_files):
                artifact_path = (manifest_path.parent / artifact.path).resolve()
                try:
                    artifact_path.relative_to(manifest_path.parent.resolve())
                except ValueError:
                    errors.append(f"{manifest.base_workflow_id}:path_escape:{artifact.path}")
                    continue
                if not artifact_path.is_file():
                    errors.append(f"{manifest.base_workflow_id}:missing_artifact:{artifact.path}")
                elif sha256_file(artifact_path) != artifact.sha256:
                    errors.append(f"{manifest.base_workflow_id}:artifact_checksum:{artifact.path}")
            roles_path = role_root / f"{manifest.domain}.json"
            if not roles_path.is_file():
                errors.append(f"{manifest.base_workflow_id}:missing_role_cards")
                continue
            roles = _load_json(roles_path)
            adapter = live_adapter_for_manifest(manifest)
            workspace = temp_root / f"{index:03d}-{manifest.base_workflow_id}"
            try:
                env = adapter.materialize(manifest_path.parent, workspace)
                private_paths = [
                    path.relative_to(workspace).as_posix()
                    for path in workspace.rglob("*")
                    if path.is_file() and "private" in [part.casefold() for part in path.relative_to(workspace).parts]
                ]
                if private_paths:
                    errors.append(f"{manifest.base_workflow_id}:private_workspace:{private_paths}")

                observation = adapter.public_observation(env)
                leaked = _leaks(observation)
                if leaked:
                    errors.append(f"{manifest.base_workflow_id}:router_context:{leaked}")
                checked_contexts += 1

                for role in roles:
                    action = role["agent_id"]
                    if hasattr(adapter, "executor_context"):
                        context = adapter.executor_context(env, action, role)
                    else:
                        context = {"state": observation}
                    leaked = _leaks(context)
                    if leaked:
                        errors.append(f"{manifest.base_workflow_id}:{action}:executor_context:{leaked}")
                    checked_contexts += 1
            except Exception as error:  # certificate must report every packet
                errors.append(f"{manifest.base_workflow_id}:{type(error).__name__}:{error}")
            finally:
                shutil.rmtree(workspace, ignore_errors=True)

    if counts != EXPECTED_DOMAIN_COUNTS:
        errors.append(f"domain_counts:{counts}!={EXPECTED_DOMAIN_COUNTS}")

    return {
        "valid": not errors,
        "packet_count": len(manifest_paths),
        "domain_counts": counts,
        "model_contexts_checked": checked_contexts,
        "errors": errors,
    }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                yield {"__parse_error__": f"{path}:{line_number}:{error}"}


def certify_runs(run_dirs: Iterable[Path]) -> dict[str, Any]:
    run_dirs = list(run_dirs)
    errors: list[str] = []
    packets: set[str] = set()
    decision_states = 0
    executor_requests = 0
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            errors.append(f"missing_run_dir:{run_dir}")
            continue
        for path in run_dir.rglob("decision_states.jsonl"):
            for record in _iter_jsonl(path):
                if "__parse_error__" in record:
                    errors.append(record["__parse_error__"])
                    continue
                decision_states += 1
                packets.add(str(record.get("packet_id") or path.parent.name))
                if record.get("router_visible_private_reference") is not False:
                    errors.append(f"{path}:router_visible_private_reference_not_false")
                leaked = _leaks(record.get("observation", {}))
                if leaked:
                    errors.append(f"{path}:router_observation:{leaked}")
        for path in run_dir.rglob("trajectory.jsonl"):
            for record in _iter_jsonl(path):
                result = record.get("result") or {}
                for request in result.get("requests") or []:
                    executor_requests += 1
                    leaked = _leaks(request.get("messages", []))
                    if leaked:
                        errors.append(f"{path}:executor_request:{leaked}")
        for path in run_dir.rglob("*"):
            if path.is_file() and "workspace" in [p.casefold() for p in path.parts]:
                relative_parts = [p.casefold() for p in path.relative_to(run_dir).parts]
                if "private" in relative_parts:
                    errors.append(f"{path}:private_file_in_workspace")
    return {
        "valid": not errors,
        "run_count": len(run_dirs),
        "unique_packet_count": len(packets),
        "decision_state_count": decision_states,
        "executor_request_count": executor_requests,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-root",
        type=Path,
        default=PROJECT / "data/RoutingBench/v4_beta_4d_82_20261102",
    )
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        default=PROJECT / "reproducibility/runtime_core.sha256.json",
    )
    parser.add_argument("--run-dir", action="append", type=Path, default=[])
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT / "reproducibility/certification_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = list(args.run_dir)
    report = {
        "schema_version": "routingbench-v4-release-certificate-v1",
        "certification_is_out_of_band": True,
        "runtime": verify_runtime_manifest(PROJECT, args.runtime_manifest),
        "packets": certify_packets(args.release_root),
        "runs": certify_runs(run_dirs),
    }
    report["valid"] = all(section["valid"] for section in (report["runtime"], report["packets"], report["runs"]))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
