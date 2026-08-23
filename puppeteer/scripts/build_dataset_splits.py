from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "splits"
SEED = 42

SIZES = {
    "gsm_hard": {"train": 200, "dev": 100, "reference": 100, "probe": 10, "final": 909},
    "mmlu_pro": {"train": 200, "dev": 140, "reference": 280, "probe": 10, "final": 2000},
    "srdd": {"train": 200, "dev": 100, "reference": 100, "probe": 10, "final": 790},
    "cw": {"train": 80, "dev": 20, "reference": 40, "probe": 10, "final": 50},
}

def sequential_splits(total: int, sizes: dict[str, int], seed: int) -> dict[str, list[int]]:
    if sum(sizes.values()) != total:
        raise ValueError(f"Requested {sum(sizes.values())} items from dataset of {total}")
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    result = {}
    offset = 0
    for name, size in sizes.items():
        result[name] = indices[offset:offset + size]
        offset += size
    return result

def stratified_splits(labels: list[str], sizes: dict[str, int], seed: int) -> dict[str, list[int]]:
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(str(label), []).append(index)
    for values in groups.values():
        rng.shuffle(values)

    result = {}
    for split_name, size in sizes.items():
        remaining_total = sum(len(values) for values in groups.values())
        if size > remaining_total:
            raise ValueError("MMLU-Pro split request exceeds remaining test items")
        quotas = {
            label: size * len(values) / remaining_total
            for label, values in groups.items()
        }
        counts = {label: min(len(groups[label]), int(quota)) for label, quota in quotas.items()}
        remainder = size - sum(counts.values())
        order = sorted(
            groups,
            key=lambda label: (quotas[label] - int(quotas[label]), len(groups[label]), label),
            reverse=True,
        )
        while remainder:
            progressed = False
            for label in order:
                if counts[label] < len(groups[label]):
                    counts[label] += 1
                    remainder -= 1
                    progressed = True
                    if remainder == 0:
                        break
            if not progressed:
                raise RuntimeError("Unable to allocate stratified split")
        chosen = []
        for label in sorted(groups):
            take = counts[label]
            chosen.extend(groups[label][:take])
            del groups[label][:take]
        rng.shuffle(chosen)
        result[split_name] = chosen
    return result

def write_manifest(name: str, source: str, total: int, splits: dict[str, list[int]], extra=None) -> None:
    flat = [index for values in splits.values() for index in values]
    if len(flat) != len(set(flat)):
        raise ValueError(f"Overlapping split indices for {name}")
    manifest = {
        "schema_version": "1.0",
        "dataset": name,
        "seed": SEED,
        "source": source,
        "source_total": total,
        "sizes": {key: len(value) for key, value in splits.items()},
        "splits": splits,
    }
    if extra:
        manifest.update(extra)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"{name}_seed{SEED}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def main() -> None:
    gsm = pd.read_parquet(DATA_DIR / "GSM-Hard" / "test.parquet")
    write_manifest("gsm_hard", "GSM-Hard/test.parquet", len(gsm), sequential_splits(len(gsm), SIZES["gsm_hard"], SEED))

    mmlu = pd.read_parquet(DATA_DIR / "MMLU-Pro" / "test.parquet")
    mmlu_splits = stratified_splits(mmlu["category"].astype(str).tolist(), SIZES["mmlu_pro"], SEED)
    write_manifest(
        "mmlu_pro",
        "MMLU-Pro/test.parquet",
        len(mmlu),
        mmlu_splits,
        extra={"official_validation_preserved": True, "unused_test_items": len(mmlu) - sum(SIZES["mmlu_pro"].values())},
    )

    srdd = pd.read_csv(DATA_DIR / "SRDD" / "SRDD.csv")
    write_manifest("srdd", "SRDD/SRDD.csv", len(srdd), sequential_splits(len(srdd), SIZES["srdd"], SEED))

    cw_path = DATA_DIR / "CW" / "creative_writing.jsonl"
    cw_count = sum(1 for line in cw_path.read_text(encoding="utf-8").splitlines() if line.strip())
    write_manifest("cw", "CW/creative_writing.jsonl", cw_count, sequential_splits(cw_count, SIZES["cw"], SEED))

if __name__ == "__main__":
    main()
