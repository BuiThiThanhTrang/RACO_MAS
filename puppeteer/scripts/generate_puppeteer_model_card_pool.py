"""Create the role-aware, model-card-prior variant of the legacy Puppeteer pool.

The legacy pool is deliberately left untouched: it is the baseline.  This script
uses the existing deterministic ``mimas_pool.jsonl`` migration for the fourteen
role cards, replaces its generic capability priors, and writes a separate pool
with the same agents, actions, and decoding settings.  The optional replacement
map below creates the Gemma 3/Qwen ablation pool requested for this experiment.
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
    # Qwen 3.5 4B: MMLU-Pro 79.1, HMMT 74.0, LiveCodeBench 55.8,
    # IFEval 89.8, BFCL 50.3, and TAU2-Bench 79.9 in the official card.
    # These are translated into ordinal priors, never copied as probabilities.
    "qwen-3.5-4b": dict(zip(DIMENSIONS, (0.70, 0.84, 0.79, 0.82, 0.70, 0.76, 0.70, 0.70, 0.80, 0.73))),
    # Gemma 3 12B has published 12B-family results for general, mathematical,
    # coding, and long-context tasks.  Tool use remains conservative because
    # the card does not provide a directly comparable function-calling score.
    "gemma-3-12b-it": dict(zip(DIMENSIONS, (0.76, 0.80, 0.80, 0.80, 0.77, 0.79, 0.70, 0.70, 0.81, 0.74))),
    "llama-3.1-8b": dict(zip(DIMENSIONS, (0.68, 0.70, 0.80, 0.70, 0.78, 0.76, 0.70, 0.70, 0.78, 0.80))),
    "llama-3.2-3b": dict(zip(DIMENSIONS, (0.62, 0.64, 0.74, 0.64, 0.62, 0.70, 0.70, 0.70, 0.68, 0.67))),
    "mistralai/ministral-3b-2512": dict(zip(DIMENSIONS, (0.68, 0.70, 0.77, 0.70, 0.73, 0.72, 0.70, 0.70, 0.72, 0.74))),
    "mistral-nemo-12b": dict(zip(DIMENSIONS, (0.72, 0.72, 0.67, 0.72, 0.70, 0.76, 0.70, 0.70, 0.75, 0.80))),
}

# Applied when generating the model-card pool.  Keep the legacy source pool
# unchanged so historical baseline checkpoints remain reproducible.
BACKBONE_REPLACEMENTS = {
    "qwen-2.5-7b": "qwen-3.5-4b",
    "llama-3.1-8b": "gemma-3-12b-it",
    "llama-3.2-3b": "qwen-2.5-7b",
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
        source_backbone = record["backbone"]
        backbone = BACKBONE_REPLACEMENTS.get(source_backbone, source_backbone)
        if backbone not in BACKBONE_BASES:
            raise ValueError(f"Missing model-card base prior for {backbone!r}")
        record["backbone"] = backbone
        role_card = record["role_card"]
        role_card["capability_prior"] = blended_prior(
            BACKBONE_BASES[backbone], role_card["capability_prior"], role_mean
        )
        metadata = record.setdefault("metadata", {})
        metadata.update(
            {
                "role_slot": role_slot,
                "prior_profile_version": "puppeteer-model-card-role-blend-v3-qwen35",
                "prior_profile_source": "official-model-cards-and-published-benchmarks",
                "derived_from": "puppeteer_pool.jsonl",
                "backbone_replacement": (
                    f"{source_backbone}->{backbone}" if source_backbone != backbone else "unchanged"
                ),
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
