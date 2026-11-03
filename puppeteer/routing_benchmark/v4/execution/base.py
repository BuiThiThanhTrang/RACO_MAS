from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from routing_benchmark.v4.artifacts.schema import ArtifactManifest


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
        }


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    status: str
    output: Mapping[str, Any]
    observable_message: str
    cost: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "output": dict(self.output),
            "observable_message": self.observable_message,
            "cost": self.cost,
        }


@dataclass(frozen=True)
class PreconditionResult:
    executable: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnvironmentHandle:
    manifest: ArtifactManifest
    pack_dir: Path
    workspace_dir: Path


@dataclass(frozen=True)
class EnvironmentSnapshot:
    snapshot_id: str
    source_workspace: Path
    snapshot_dir: Path
    tree_hash: str
    public_state_fingerprint: str
    budget: Mapping[str, Any]


@dataclass(frozen=True)
class ArtifactDiff:
    added: tuple[Mapping[str, Any], ...] = ()
    modified: tuple[Mapping[str, Any], ...] = ()
    removed: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": [dict(item) for item in self.added],
            "modified": [dict(item) for item in self.modified],
            "removed": [dict(item) for item in self.removed],
        }


class ExecutionAdapter(Protocol):
    adapter_id: str

    def materialize(self, manifest_path: str | Path, workspace: str | Path) -> EnvironmentHandle: ...

    def public_observation(self, env: EnvironmentHandle) -> dict[str, Any]: ...

    def public_preconditions(
        self,
        env: EnvironmentHandle,
        action_id: str,
        role_profile: Mapping[str, Any] | None = None,
    ) -> PreconditionResult: ...

    def tool_spec(
        self,
        env: EnvironmentHandle,
        action_id: str,
        role_profile: Mapping[str, Any] | None = None,
    ) -> Sequence[ToolDefinition]: ...

    def execute_tool(
        self,
        env: EnvironmentHandle,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> ToolResult: ...

    def artifact_diff(
        self, before: Mapping[str, str], after: Mapping[str, str]
    ) -> ArtifactDiff: ...

    def validate_postcondition(self, env: EnvironmentHandle) -> Mapping[str, Any]: ...

    def terminal_status(self, env: EnvironmentHandle) -> Mapping[str, Any]: ...
