import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SPLITS_DIR = DATA_DIR / "splits"
MODE_SPLITS = {"train": "train", "validation": "dev", "test": "final"}


def load_split_frame(dataset, mode, seed=42):
    """Read the manifest source and select its rows in the saved split order."""
    import pandas as pd

    if mode not in MODE_SPLITS:
        raise ValueError(f"Unsupported dataset mode: {mode!r}")
    manifest = load_split_manifest(dataset, seed=seed)
    source_path = DATA_DIR / manifest["source"]
    if source_path.suffix == ".parquet":
        data = pd.read_parquet(source_path)
    elif source_path.suffix == ".csv":
        data = pd.read_csv(source_path)
    elif source_path.suffix == ".jsonl":
        with source_path.open(encoding="utf-8") as source:
            data = pd.DataFrame(json.loads(line) for line in source if line.strip())
    else:
        raise ValueError(f"Unsupported split source format: {source_path}")
    indices, _ = load_split_indices(
        dataset, MODE_SPLITS[mode], seed=seed, source_size=len(data)
    )
    return data.iloc[indices]



def load_split_manifest(dataset, seed=42):
    normalized_dataset = dataset.lower().replace("-", "_")
    manifest_path = SPLITS_DIR / f"{normalized_dataset}_seed{seed}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}. "
            "Copy the matching split file into data/splits."
        )

    with manifest_path.open("r", encoding="utf-8") as source:
        manifest = json.load(source)

    if manifest.get("dataset") != normalized_dataset:
        raise ValueError(
            f"Split manifest dataset mismatch: expected {normalized_dataset!r}, "
            f"got {manifest.get('dataset')!r}"
        )
    if manifest.get("seed") != seed:
        raise ValueError(
            f"Split manifest seed mismatch: expected {seed}, "
            f"got {manifest.get('seed')!r}"
        )
    if not isinstance(manifest.get("splits"), dict):
        raise ValueError(f"Split manifest has no valid 'splits' object: {manifest_path}")
    return manifest


def load_split_indices(dataset, split, seed=42, source_size=None):
    manifest = load_split_manifest(dataset, seed=seed)
    if split not in manifest["splits"]:
        available = ", ".join(sorted(manifest["splits"]))
        raise ValueError(
            f"Split {split!r} is not defined for {dataset}; available: {available}"
        )

    indices = manifest["splits"][split]
    if not isinstance(indices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in indices
    ):
        raise ValueError(f"Split {split!r} must contain integer row indices")
    if len(indices) != len(set(indices)):
        raise ValueError(f"Split {split!r} contains duplicate row indices")

    expected_size = (manifest.get("sizes") or {}).get(split)
    if expected_size is not None and len(indices) != expected_size:
        raise ValueError(
            f"Split {split!r} size mismatch: expected {expected_size}, "
            f"got {len(indices)}"
        )
    if source_size is not None and any(
        index < 0 or index >= source_size for index in indices
    ):
        raise IndexError(
            f"Split {split!r} contains an index outside source size {source_size}"
        )
    return indices, manifest
