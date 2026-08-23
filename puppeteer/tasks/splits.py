from __future__ import annotations

import json
from pathlib import Path

DEFAULT_MANIFEST_DIR = Path("data/splits")
SPLIT_ALIASES = {"test": "final", "validation": "dev"}

def canonical_split(name: str) -> str:
    return SPLIT_ALIASES.get(name, name)

def load_split_indices(
    task_name: str,
    split: str,
    seed: int = 42,
    manifest_dir: str | Path = DEFAULT_MANIFEST_DIR,
) -> list[int]:
    normalized = task_name.lower().replace("-", "_")
    manifest_path = Path(manifest_dir) / f"{normalized}_seed{seed}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}. "
            "Run python -m scripts.build_dataset_splits from puppeteer/."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested = canonical_split(split)
    try:
        return [int(index) for index in manifest["splits"][requested]]
    except KeyError as exc:
        raise ValueError(
            f"Unknown split {split!r}; available: {sorted(manifest['splits'])}"
        ) from exc
