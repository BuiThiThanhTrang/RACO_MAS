from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from routing_benchmark.v4.artifacts.schema import canonical_hash, sha256_file

from .base import EnvironmentHandle, EnvironmentSnapshot


def _copytree(source: Path, destination: Path) -> None:
    """Copy deep repository trees without hitting the legacy Windows MAX_PATH."""

    source_arg: str | Path = source
    destination_arg: str | Path = destination
    if os.name == "nt":
        source_arg = "\\\\?\\" + str(source.resolve())
        destination_arg = "\\\\?\\" + str(destination.resolve())
    shutil.copytree(source_arg, destination_arg)


def tree_hashes(root: str | Path) -> dict[str, str]:
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class SnapshotStore:
    """Content-addressed workspace snapshots used to isolate action branches."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        env: EnvironmentHandle,
        *,
        public_state_fingerprint: str,
        budget: Mapping[str, Any],
    ) -> EnvironmentSnapshot:
        hashes = tree_hashes(env.workspace_dir)
        snapshot_id = f"snap-{canonical_hash([env.manifest.base_workflow_id, env.manifest.task, hashes, public_state_fingerprint, budget])}"
        snapshot_dir = self.root / snapshot_id
        if not snapshot_dir.exists():
            _copytree(env.workspace_dir, snapshot_dir)
            metadata = {
                "snapshot_id": snapshot_id,
                "tree_hash": canonical_hash(hashes, length=64),
                "public_state_fingerprint": public_state_fingerprint,
                "budget": dict(budget),
            }
            (snapshot_dir / ".snapshot.json").write_text(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        return EnvironmentSnapshot(
            snapshot_id=snapshot_id,
            source_workspace=env.workspace_dir,
            snapshot_dir=snapshot_dir,
            tree_hash=canonical_hash(hashes, length=64),
            public_state_fingerprint=public_state_fingerprint,
            budget=dict(budget),
        )

    def restore(
        self,
        snapshot: EnvironmentSnapshot,
        destination: str | Path,
        *,
        manifest,
        pack_dir: str | Path,
    ) -> EnvironmentHandle:
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite branch workspace: {destination}")
        _copytree(snapshot.snapshot_dir, destination)
        marker = destination / ".snapshot.json"
        if marker.exists():
            marker.unlink()
        return EnvironmentHandle(
            manifest=manifest,
            pack_dir=Path(pack_dir),
            workspace_dir=destination,
        )

    @staticmethod
    def equivalent_pre_state(
        first: EnvironmentHandle, second: EnvironmentHandle
    ) -> bool:
        return tree_hashes(first.workspace_dir) == tree_hashes(second.workspace_dir)

    @staticmethod
    def restored_matches_snapshot(
        snapshot: EnvironmentSnapshot, restored: EnvironmentHandle
    ) -> bool:
        expected = {
            key: value
            for key, value in tree_hashes(snapshot.snapshot_dir).items()
            if key != ".snapshot.json"
        }
        return expected == tree_hashes(restored.workspace_dir)

    @staticmethod
    def metadata(snapshot: EnvironmentSnapshot) -> dict[str, Any]:
        marker = snapshot.snapshot_dir / ".snapshot.json"
        if not marker.is_file():
            return {}
        value = json.loads(marker.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, Mapping) else {}
