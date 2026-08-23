from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from role_aware.capability_scopes import relevant_capabilities
from role_aware.profile_store import ProfileStore
from role_aware.schemas import TeammateSpec


@dataclass(frozen=True)
class PathTerminalEvidence:
    task_id: str
    task_type: str
    path_id: int
    teammate_ids: tuple[str, ...]
    reward: float
    role_adherence: Mapping[str, bool]

    def __post_init__(self) -> None:
        if self.reward not in (0, 1, 0.0, 1.0):
            raise ValueError("Capability evidence reward must be binary")


class ProfileEvidenceAccumulator:
    """Apply deduplicated, order-invariant path-terminal evidence."""

    def __init__(self, store: ProfileStore, specs: Iterable[TeammateSpec]) -> None:
        self.store = store
        self.specs = {spec.teammate_id: spec for spec in specs}
        self._pending: dict[str, list[PathTerminalEvidence]] = {}
        self._seen_paths: set[tuple[str, int, str]] = set()

    def record(self, evidence: PathTerminalEvidence) -> None:
        for teammate_id in tuple(dict.fromkeys(evidence.teammate_ids)):
            if teammate_id not in self.specs:
                raise KeyError(f"Unknown teammate in evidence: {teammate_id!r}")
            key = (evidence.task_id, evidence.path_id, teammate_id)
            if key in self._seen_paths:
                continue
            self._seen_paths.add(key)
            self._pending.setdefault(evidence.task_id, []).append(
                PathTerminalEvidence(
                    task_id=evidence.task_id,
                    task_type=evidence.task_type,
                    path_id=evidence.path_id,
                    teammate_ids=(teammate_id,),
                    reward=float(evidence.reward),
                    role_adherence={
                        teammate_id: evidence.role_adherence.get(teammate_id, True)
                    },
                )
            )

    def flush_task(self, task_id: str) -> None:
        evidence_items = self._pending.pop(task_id, [])
        grouped: dict[str, list[PathTerminalEvidence]] = {}
        for item in evidence_items:
            grouped.setdefault(item.teammate_ids[0], []).append(item)

        for teammate_id, items in grouped.items():
            rewards = [float(item.reward) for item in items]
            spec = self.specs[teammate_id]
            self.store.update_terminal_batch(
                teammate_id=teammate_id,
                capabilities=relevant_capabilities(
                    spec.role_card.role_name, items[0].task_type
                ),
                mean_reward=sum(rewards) / len(rewards),
                observation_count=len(items),
                role_adhered=all(
                    item.role_adherence.get(teammate_id, True) for item in items
                ),
            )

        self._seen_paths = {key for key in self._seen_paths if key[0] != task_id}
