"""Create the role-aware, model-card-prior variant of the legacy Puppeteer pool.

The legacy pool is deliberately left untouched: it is the baseline.  This script
uses the existing deterministic ``mimas_pool.jsonl`` migration for the fourteen
role cards, replaces its generic capability priors, and writes a separate pool
with the same agents, actions, backbones, and decoding settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DIMENSIONS = (
    "planning",
    "general_reasoning",
    "quantitative_reasoning",
    "domain_reasoning",
    "software_engineering",
    "commonsense_generation",
    "verification",
    "repair",
    "integration",
    "tool_use",
)

# Ordinal engineering priors, not calibrated probabilities.  They summarize
# comparable evidence from the official cards listed in the accompanying doc.
# ``verification`` and ``repair`` have no directly comparable published metric,
# so each receives the neutral backbone value 0.70.
BACKBONE_BASES: dict[str, dict[str, float]] = {
    "qwen-2.5-7b": dict(zip(DIMENSIONS, (0.69, 0.75, 0.80, 0.73, 0.81, 0.75, 0.70, 0.70, 0.74, 0.75))),
    "qwen-2.5-14b": dict(zip(DIMENSIONS, (0.76, 0.82, 0.84, 0.80, 0.84, 0.78, 0.70, 0.70, 0.81, 0.78))),
    "llama-3.1-8b": dict(zip(DIMENSIONS, (0.68, 0.70, 0.80, 0.70, 0.78, 0.76, 0.70, 0.70, 0.78, 0.80))),
    "llama-3.2-3b": dict(zip(DIMENSIONS, (0.62, 0.64, 0.74, 0.64, 0.62, 0.70, 0.70, 0.70, 0.68, 0.67))),
    "mistralai/ministral-3b-2512": dict(zip(DIMENSIONS, (0.68, 0.70, 0.77, 0.70, 0.73, 0.72, 0.70, 0.70, 0.72, 0.74))),
    "mistral-nemo-12b": dict(zip(DIMENSIONS, (0.72, 0.72, 0.67, 0.72, 0.70, 0.76, 0.70, 0.70, 0.75, 0.80))),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def blended_prior(base: dict[str, float], role: dict[str, float], role_mean: dict[str, float]) -> dict[str, float]:
    return {
        dimension: round(min(0.99, max(0.01, base[dimension] + 0.60 * (role[dimension] - role_mean[dimension]))), 6)
        for dimension in DIMENSIONS
    }


def generate(source: Path, output: Path) -> list[dict[str, Any]]:
    records = load_jsonl(source)
    if not records:
        raise ValueError(f"No role-aware records found in {source}")

    role_mean = {
        dimension: sum(record["role_card"]["capability_prior"][dimension] for record in records) / len(records)
        for dimension in DIMENSIONS
    }

    for role_slot, record in enumerate(records):
        backbone = record["backbone"]
        if backbone not in BACKBONE_BASES:
            raise ValueError(f"Missing model-card base prior for {backbone!r}")
        role_card = record["role_card"]
        role_card["capability_prior"] = blended_prior(
            BACKBONE_BASES[backbone], role_card["capability_prior"], role_mean
        )
        metadata = record.setdefault("metadata", {})
        metadata.update(
            {
                "role_slot": role_slot,
                "prior_profile_version": "puppeteer-model-card-role-blend-v1",
                "prior_profile_source": "official-model-cards-and-published-benchmarks",
                "derived_from": "puppeteer_pool.jsonl",
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return records


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "personas" / "role_aware" / "mimas_pool.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "personas" / "role_aware" / "puppeteer_model_card_role_aware_pool.jsonl",
    )
    args = parser.parse_args()
    records = generate(args.source, args.output)
    print(json.dumps({"output": str(args.output), "agents": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
