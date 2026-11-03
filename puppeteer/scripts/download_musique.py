"""Download official-format MuSiQue parquet shards from a Hugging Face mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


DATASET = "awinml/musique"
CONFIG = "default"
DEFAULT_ROOT = Path("data") / "MuSiQue"
SPLITS = ("train", "validation", "test")
REQUIRED_COLUMNS = {
    "id",
    "paragraphs",
    "question",
    "question_decomposition",
    "answer",
    "answer_aliases",
    "answerable",
}


def _request_json(url: str) -> dict:
    request = urllib.request.Request(url)
    token = os.getenv("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _parquet_urls() -> dict[str, str]:
    dataset = urllib.parse.quote(DATASET, safe="")
    payload = _request_json(
        f"https://datasets-server.huggingface.co/parquet?dataset={dataset}"
    )
    urls = {}
    for item in payload.get("parquet_files", []):
        if item.get("config") == CONFIG and item.get("split") in SPLITS:
            split = str(item["split"])
            if split in urls:
                raise RuntimeError(
                    f"Expected one parquet shard for {split}, found multiple"
                )
            urls[split] = str(item["url"])
    missing = sorted(set(SPLITS) - set(urls))
    if missing:
        raise RuntimeError(f"Dataset Viewer did not return splits: {missing}")
    return urls


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    request = urllib.request.Request(url)
    token = os.getenv("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as destination:
                shutil.copyfileobj(response, destination)
        frame = pd.read_parquet(temporary)
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"Downloaded MuSiQue shard is missing columns: {missing}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=[*SPLITS, "all"],
        default="validation",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    urls = _parquet_urls()
    requested = SPLITS if args.split == "all" else (args.split,)
    manifest = {
        "dataset": DATASET,
        "config": CONFIG,
        "format": "official MuSiQue JSON schema converted to parquet",
        "splits": {},
    }
    for split in requested:
        target = args.output_dir / f"{split}.parquet"
        if args.force or not target.is_file():
            print(f"Downloading MuSiQue {split} -> {target}")
            _download(urls[split], target)
        frame = pd.read_parquet(target, columns=["id"])
        manifest["splits"][split] = {
            "path": str(target),
            "source_url": urls[split],
            "rows": len(frame),
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

