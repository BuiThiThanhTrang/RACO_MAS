"""Build matched persona pools for a role-versus-profile routing experiment.

The source S0 pool couples each role to a fixed pair of backbones. This script
crosses every role with all four S0 backbones. The profile-aware condition uses
ordinal model-card priors, not raw benchmark accuracy or probe evidence.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "personas" / "role_aware" / "s0_pool.jsonl"
OUTPUT_DIR = SOURCE.parent

BACKBONES = (
    "qwen-3.5-9b",
    "qwen-3.5-4b",
    "gemma-3-12b-it",
    "llama-3.1-8b",
)

# Ordinal model-card priors. Values are deliberately neutral for capabilities
# with no direct reported measurement (verification and repair).
#
# Qwen 3.5: MMLU-Pro 82.5/79.1, HMMT 83.2/74.0, LiveCodeBench 65.6/55.8,
# IFEval 91.5/89.8, BFCL 66.1/50.3, TAU2 79.1/79.9.
# Gemma 3 12B (PT evaluation): MMLU 74.5, MMLU-Pro CoT 45.3, GSM8K 71.0,
# MATH 43.3, MBPP 60.4, HumanEval 45.7, plus HellaSwag/BoolQ/PIQA.
# Llama 3.1 8B Instruct: MMLU-Pro CoT 48.3, CommonSenseQA 75.0,
# HumanEval 72.6, MBPP 72.8, GSM8K 84.5, MATH 51.9, API-Bank 82.6,
# BFCL 76.1, and IFEval 80.4.
BACKBONE_CAPABILITY_BASES = {
    "qwen-3.5-9b": {
        "planning": 0.72, "general_reasoning": 0.88,
        "quantitative_reasoning": 0.86, "domain_reasoning": 0.86,
        "software_engineering": 0.78, "commonsense_generation": 0.79,
        "verification": 0.70, "repair": 0.70, "integration": 0.85,
        "tool_use": 0.80,
    },
    "qwen-3.5-4b": {
        "planning": 0.70, "general_reasoning": 0.84,
        "quantitative_reasoning": 0.79, "domain_reasoning": 0.82,
        "software_engineering": 0.70, "commonsense_generation": 0.76,
        "verification": 0.70, "repair": 0.70, "integration": 0.80,
        "tool_use": 0.73,
    },
    "gemma-3-12b-it": {
        "planning": 0.65, "general_reasoning": 0.73,
        "quantitative_reasoning": 0.66, "domain_reasoning": 0.74,
        "software_engineering": 0.62, "commonsense_generation": 0.82,
        "verification": 0.70, "repair": 0.70, "integration": 0.72,
        "tool_use": 0.60,
    },
    "llama-3.1-8b": {
        "planning": 0.68, "general_reasoning": 0.70,
        "quantitative_reasoning": 0.80, "domain_reasoning": 0.70,
        "software_engineering": 0.78, "commonsense_generation": 0.76,
        "verification": 0.70, "repair": 0.70, "integration": 0.78,
        "tool_use": 0.80,
    },
}
ROLE_SPECIALIZATION_WEIGHT = 0.60


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _role_templates(source: list[dict]) -> list[dict]:
    by_slot: dict[int, list[dict]] = {}
    for record in source:
        by_slot.setdefault(int(record["metadata"]["role_slot"]), []).append(record)

    templates: list[dict] = []
    for role_slot, records in sorted(by_slot.items()):
        if len(records) != 2:
            raise ValueError(
                f"Expected exactly two source teammates for role {role_slot}, "
                f"got {len(records)}"
            )
        template = deepcopy(records[0])
        first = records[0]["role_card"]["capability_prior"]
        second = records[1]["role_card"]["capability_prior"]
        template["role_card"]["capability_prior"] = {
            key: round((float(first[key]) + float(second[key])) / 2.0, 6)
            for key in first
        }
        templates.append(template)
    return templates


def _record(
    template: dict,
    role_baseline: dict[str, float],
    role_slot: int,
    teammate_slot: int,
    backbone: str,
    *,
    profile_aware: bool,
) -> dict:
    record = deepcopy(template)
    record["teammate_id"] = f"s0x-r{role_slot:02d}-t{teammate_slot:02d}"
    record["backbone"] = backbone
    record["metadata"].update(
        {
            "scenario": "s0_role_backbone_crossed",
            "role_slot": role_slot,
            "teammate_slot": teammate_slot,
            "prior_profile_version": "model-card-role-blend-v2",
            "profile_experiment": (
                "profile_aware" if profile_aware else "role_only_control"
            ),
        }
    )
    if profile_aware:
        role_prior = record["role_card"]["capability_prior"]
        model_prior = BACKBONE_CAPABILITY_BASES[backbone]
        record["role_card"]["capability_prior"] = {
            key: round(
                min(
                    0.99,
                    max(
                        0.01,
                        model_prior[key]
                        + ROLE_SPECIALIZATION_WEIGHT
                        * (float(value) - role_baseline[key]),
                    ),
                ),
                6,
            )
            for key, value in role_prior.items()
        }
    return record


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    templates = _role_templates(_read_jsonl(SOURCE))
    role_baseline = {
        key: sum(
            float(template["role_card"]["capability_prior"][key])
            for template in templates
        )
        / len(templates)
        for key in templates[0]["role_card"]["capability_prior"]
    }
    role_only: list[dict] = []
    profile_aware: list[dict] = []
    for role_slot, template in enumerate(templates):
        for teammate_slot, backbone in enumerate(BACKBONES):
            role_only.append(
                _record(
                    template, role_baseline, role_slot, teammate_slot, backbone,
                    profile_aware=False,
                )
            )
            profile_aware.append(
                _record(
                    template, role_baseline, role_slot, teammate_slot, backbone,
                    profile_aware=True,
                )
            )

    outputs = {
        "role_only": OUTPUT_DIR / "s0_role_backbone_role_only_control_pool.jsonl",
        "profile_aware": OUTPUT_DIR / "s0_role_backbone_profile_aware_pool.jsonl",
    }
    _write_jsonl(outputs["role_only"], role_only)
    _write_jsonl(outputs["profile_aware"], profile_aware)
    print(json.dumps({
        key: {"path": str(path), "teammates": len(role_only)}
        for key, path in outputs.items()
    }))


if __name__ == "__main__":
    main()
