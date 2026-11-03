from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


ARTIFACT_SCHEMA_VERSION = "4.2-artifact-rc1"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any, length: int = 24) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


@dataclass(frozen=True)
class ArtifactFile:
    artifact_id: str
    path: str
    media_type: str
    sha256: str
    visibility: str


@dataclass(frozen=True)
class ArtifactManifest:
    base_workflow_id: str
    domain: str
    task: str
    adapter_id: str
    public_files: tuple[ArtifactFile, ...]
    private_files: tuple[ArtifactFile, ...]
    fixture_files: tuple[ArtifactFile, ...] = ()
    source_type: str = "project_authored_programmatic"
    generator_version: str = "routingbench-v4-artifacts-v1"
    seed: int = 0
    license: str = "project-authored"
    human_review_status: str = "pending"
    packet_review_status: str = "pending"
    execution_readiness: str = "pending_runtime_verification"
    artifact_schema_version: str = ARTIFACT_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactManifest":
        return cls(
            artifact_schema_version=str(value["artifact_schema_version"]),
            base_workflow_id=str(value["base_workflow_id"]),
            domain=str(value["domain"]),
            task=str(value["task"]),
            adapter_id=str(value["adapter_id"]),
            public_files=tuple(ArtifactFile(**item) for item in value["public_files"]),
            private_files=tuple(ArtifactFile(**item) for item in value["private_files"]),
            fixture_files=tuple(ArtifactFile(**item) for item in value.get("fixture_files", ())),
            source_type=str(value.get("source_type", "project_authored_programmatic")),
            generator_version=str(value.get("generator_version", "unknown")),
            seed=int(value.get("seed", 0)),
            license=str(value.get("license", "project-authored")),
            human_review_status=str(value.get("human_review_status", "pending")),
            packet_review_status=str(value.get("packet_review_status", "pending")),
            execution_readiness=str(value.get("execution_readiness", "pending_runtime_verification")),
            metadata=dict(value.get("metadata") or {}),
        )


def validate_manifest_shape(manifest: ArtifactManifest) -> None:
    if manifest.artifact_schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected artifact schema version: {manifest.artifact_schema_version}"
        )
    if not manifest.base_workflow_id or not manifest.domain or not manifest.adapter_id:
        raise ValueError("Artifact manifest identifiers must be non-empty")
    if manifest.packet_review_status not in {"pending", "recommended_accept", "accepted", "needs_revision", "rejected"}:
        raise ValueError(f"Invalid packet review status: {manifest.packet_review_status}")
    if manifest.execution_readiness not in {"pending_runtime_verification", "runtime_verified", "blocked"}:
        raise ValueError(f"Invalid execution readiness: {manifest.execution_readiness}")
    if not manifest.public_files:
        raise ValueError("Artifact pack must contain at least one public file")
    paths: set[str] = set()
    ids: set[str] = set()
    for item in (*manifest.public_files, *manifest.private_files, *manifest.fixture_files):
        relative = Path(item.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Artifact path escapes the pack: {item.path}")
        if item.path in paths:
            raise ValueError(f"Duplicate artifact path: {item.path}")
        if item.artifact_id in ids:
            raise ValueError(f"Duplicate artifact id: {item.artifact_id}")
        if item.visibility not in {"public", "private", "fixture"}:
            raise ValueError(f"Invalid visibility: {item.visibility}")
        paths.add(item.path)
        ids.add(item.artifact_id)
