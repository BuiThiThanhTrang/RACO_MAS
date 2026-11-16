from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .config import DatasetConfig
from .contracts import TaskSpec


@dataclass(frozen=True)
class DatasetSelection:
    tasks: tuple[TaskSpec, ...]
    dataset_hash: str
    task_ids: tuple[str, ...]


class EvalPlusMBPPDataset:
    """Load canonical MBPP+ prompts while keeping evaluator fields private.

    EvalPlus v0.2.0 already embeds one example assertion in ``prompt``. Do not
    append ``assertion``, ``base_input`` or ``plus_input`` here: the remaining
    Base Tests belong to execution feedback and Plus Tests to final evaluation.
    """

    def __init__(self, config: DatasetConfig) -> None:
        if config.name.lower() != "mbpp":
            raise ValueError("Only MBPP+ is implemented in this experiment package")
        from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash

        self.problems = get_mbpp_plus(version=config.version)
        self.dataset_hash = get_mbpp_plus_hash(version=config.version)
        self.config = config

    def select(self) -> DatasetSelection:
        ids = self._selected_ids()
        tasks = tuple(
            TaskSpec(
                task_id=task_id,
                prompt=str(self.problems[task_id]["prompt"]),
                entry_point=str(self.problems[task_id]["entry_point"]),
            )
            for task_id in ids
        )
        return DatasetSelection(tasks, self.dataset_hash, tuple(ids))

    def _selected_ids(self) -> list[str]:
        if self.config.manifest:
            payload = json.loads(Path(self.config.manifest).read_text(encoding="utf-8"))
            ids = payload["task_ids"] if isinstance(payload, dict) else payload
            unknown = set(ids) - set(self.problems)
            if unknown:
                raise ValueError(f"Manifest contains unknown task ids: {sorted(unknown)}")
            return [str(task_id) for task_id in ids]
        ids = sorted(self.problems)
        random.Random(self.config.seed).shuffle(ids)
        return ids[: self.config.limit] if self.config.limit is not None else ids


def write_manifest(path: str | Path, selection: DatasetSelection) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "dataset": "mbpp+",
                "dataset_hash": selection.dataset_hash,
                "task_ids": list(selection.task_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
