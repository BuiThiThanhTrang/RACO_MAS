from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class RouteExperienceEvent:
    task_id: str
    dataset: str
    task_signature: tuple[str, ...]
    routes: tuple[tuple[str, ...], ...]
    success: bool

    def to_dict(self) -> dict:
        return asdict(self)


class RouteExperienceStore:
    """Task-level route outcomes, kept separate from teammate skill profiles."""

    def __init__(self, mode: str = "none", path: str | Path | None = None) -> None:
        if mode not in {"none", "route_outcome"}:
            raise ValueError("experience.mode must be none or route_outcome")
        self.mode = mode
        self.path = Path(path) if path else None
        self._records: dict[str, dict] = {}
        self._completed_task_ids: set[str] = set()
        if self.mode == "route_outcome" and self.path and self.path.is_file():
            self.load(self.path)

    @staticmethod
    def _key(dataset: str, signature: Iterable[str], routes: Iterable[Iterable[str]]) -> str:
        payload = {
            "dataset": str(dataset),
            "task_signature": sorted({str(value) for value in signature}),
            "routes": [list(map(str, route)) for route in routes],
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def observe(self, event: RouteExperienceEvent) -> bool:
        if self.mode == "none":
            return False
        if event.task_id in self._completed_task_ids:
            raise ValueError(f"Route experience already committed for task {event.task_id!r}")
        key = self._key(event.dataset, event.task_signature, event.routes)
        record = self._records.setdefault(
            key,
            {
                "dataset": event.dataset,
                "task_signature": list(event.task_signature),
                "routes": [list(route) for route in event.routes],
                "attempts": 0,
                "successes": 0,
                "success_rate": 0.0,
                "last_task_id": None,
            },
        )
        record["attempts"] += 1
        record["successes"] += int(event.success)
        record["success_rate"] = record["successes"] / record["attempts"]
        record["last_task_id"] = event.task_id
        self._completed_task_ids.add(event.task_id)
        self.save()
        return True

    def snapshot(self, task_signature: Iterable[str] = (), limit: int = 20) -> list[dict]:
        if self.mode == "none":
            return []
        signature = set(map(str, task_signature))
        records = list(self._records.values())
        records.sort(
            key=lambda item: (
                -len(signature & set(item.get("task_signature", ()))),
                -int(item.get("attempts", 0)),
                json.dumps(item.get("routes", ()), sort_keys=True),
            )
        )
        return [dict(record) for record in records[: max(0, int(limit))]]

    def state_dict(self) -> dict:
        return {
            "schema_version": "1.0",
            "mode": self.mode,
            "records": self._records,
            "completed_task_ids": sorted(self._completed_task_ids),
        }

    def load_state_dict(self, payload: Mapping) -> None:
        if payload.get("mode", self.mode) != self.mode:
            raise ValueError("Route experience mode mismatch")
        self._records = {
            str(key): dict(value) for key, value in (payload.get("records") or {}).items()
        }
        self._completed_task_ids = set(map(str, payload.get("completed_task_ids") or ()))

    def load(self, path: str | Path) -> None:
        self.load_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self) -> None:
        if self.mode == "none" or self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as destination:
                json.dump(self.state_dict(), destination, ensure_ascii=False, indent=2)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

