from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = "1.0"
TASK_MANIFEST_NAMES = {
    "gsm-hard": "gsm_hard",
    "MMLU-Pro": "mmlu_pro",
    "SRDD": "srdd",
    "CW": "cw",
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pool_fingerprint(specs: Iterable[Any]) -> str:
    records = [
        spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
        for spec in specs
    ]
    return canonical_hash(records)


def split_manifest_path(task_name: str, seed: int) -> Path:
    try:
        stem = TASK_MANIFEST_NAMES[task_name]
    except KeyError as error:
        raise ValueError(f"Unknown task for split manifest: {task_name!r}") from error
    return Path("data") / "splits" / f"{stem}_seed{seed}.json"


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


class RunCheckpointManager:
    def __init__(
        self,
        run_dir: str | Path,
        metadata: Mapping[str, Any],
        interval_items: int = 20,
        snapshot_interval_items: int = 20,
        keep_snapshots: int = 3,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        self.interval_items = int(interval_items)
        self.snapshot_interval_items = int(snapshot_interval_items)
        self.keep_snapshots = int(keep_snapshots)
        if self.interval_items <= 0 or self.snapshot_interval_items <= 0:
            raise ValueError("Checkpoint intervals must be positive")
        if self.keep_snapshots <= 0:
            raise ValueError("keep_snapshots must be positive")

    @property
    def latest_path(self) -> Path:
        return self.checkpoint_dir / "latest.pt"

    @property
    def initial_path(self) -> Path:
        return self.checkpoint_dir / "checkpoint_initial.pt"

    @staticmethod
    def _atomic_save(payload: Mapping[str, Any], target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        try:
            with temporary.open("wb") as destination:
                torch.save(dict(payload), destination)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def _payload(self, policy, profile_store, progress) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "metadata": {
                **self.metadata,
                "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
            "policy": policy.training_state_dict(),
            "profiles": profile_store.state_dict(),
            "rng": capture_rng_state(),
            "progress": dict(progress),
        }

    def save_initial(self, policy, profile_store, progress) -> Path:
        completed = int(progress.get("completed_items", 0))
        if completed != 0:
            raise ValueError("Initial checkpoint requires completed_items == 0")
        if self.initial_path.exists():
            raise FileExistsError(
                f"Initial checkpoint already exists: {self.initial_path}"
            )
        payload = self._payload(policy, profile_store, progress)
        payload["metadata"]["checkpoint_kind"] = "initial"
        self._atomic_save(payload, self.initial_path)
        return self.initial_path

    def should_save(self, completed_items: int) -> bool:
        return completed_items > 0 and completed_items % self.interval_items == 0

    def save(self, policy, profile_store, progress, force: bool = False) -> Path | None:
        completed = int(progress.get("completed_items", 0))
        if not force and not self.should_save(completed):
            return None
        payload = self._payload(policy, profile_store, progress)
        self._atomic_save(payload, self.latest_path)
        if completed > 0 and completed % self.snapshot_interval_items == 0:
            snapshot = self.checkpoint_dir / f"checkpoint_item_{completed:04d}.pt"
            self._atomic_save(payload, snapshot)
            self._prune_snapshots()
        return self.latest_path

    def _prune_snapshots(self) -> None:
        snapshots = sorted(
            self.checkpoint_dir.glob("checkpoint_item_*.pt"),
            key=lambda path: path.name,
            reverse=True,
        )
        for stale in snapshots[self.keep_snapshots :]:
            stale.unlink()

    def load(
        self, path: str | Path, validation_scope: str = "exact"
    ) -> dict[str, Any]:
        if validation_scope not in {"exact", "policy_transfer"}:
            raise ValueError(
                "Checkpoint validation_scope must be exact or policy_transfer"
            )
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Run checkpoint not found: {target}")
        payload = torch.load(target, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Unsupported run checkpoint schema")
        for key in ("metadata", "policy", "profiles", "rng", "progress"):
            if key not in payload:
                raise ValueError(f"Run checkpoint is missing {key!r}")
        actual = payload["metadata"]
        exact_keys = (
            "dataset_name",
            "seed",
            "split_manifest_hash",
            "config_hash",
        )
        if validation_scope == "exact":
            exact_keys = (*exact_keys, "pool_fingerprint")
        for key in exact_keys:
            if actual.get(key) != self.metadata.get(key):
                raise ValueError(f"Run checkpoint metadata mismatch for {key}")
        if (
            "chat_context_hash" in actual
            and actual.get("chat_context_hash")
            != self.metadata.get("chat_context_hash")
        ):
            raise ValueError(
                "Run checkpoint metadata mismatch for chat_context_hash"
            )
        return payload
