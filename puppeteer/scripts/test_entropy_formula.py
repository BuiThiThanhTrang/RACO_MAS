"""Verify the exact routing entropy formula used by RoleAwareREINFORCE.

The policy first masks unavailable actions, renormalizes the remaining mass, then
computes torch.distributions.Categorical(probs).entropy().  This script uses 20
deterministic agent distributions: ten nearly uniform but unequal, and ten
clearly differentiated distributions.

Run from the puppeteer directory:
    & $Python scripts/test_entropy_formula.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EntropyCase:
    name: str
    group: str
    weights: tuple[float, ...]
    available: tuple[bool, ...] | None = None


def normalized(weights: tuple[float, ...]) -> tuple[float, ...]:
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def routing_distribution(
    raw_weights: tuple[float, ...],
    available: tuple[bool, ...] | None,
) -> torch.Tensor:
    """Match RoleAwareREINFORCE._distribution after the network softmax."""
    probs = torch.tensor(raw_weights, dtype=torch.float64).unsqueeze(0)
    mask_values = available or (True,) * len(raw_weights)
    mask = torch.tensor(mask_values, dtype=torch.bool).unsqueeze(0)
    probs = probs.masked_fill(~mask, 0.0)
    return probs / probs.sum(dim=-1, keepdim=True)


def build_cases() -> list[EntropyCase]:
    near = [
        EntropyCase("near_01_linear_0.1pct", "near-uniform", normalized(
            (1.000, 1.001, 1.002, 1.003, 1.004, 1.005, 1.006, 1.007, 1.008, 1.009)
        )),
        EntropyCase("near_02_linear_0.5pct", "near-uniform", normalized(
            (1.000, 1.005, 1.010, 1.015, 1.020, 1.025, 1.030, 1.035, 1.040, 1.045)
        )),
        EntropyCase("near_03_one_slightly_higher", "near-uniform", normalized(
            (1.010, 1.000, 0.999, 1.001, 1.000, 0.999, 1.001, 1.000, 0.999, 1.000)
        )),
        EntropyCase("near_04_two_slightly_higher", "near-uniform", normalized(
            (1.006, 1.005, 1.000, 0.998, 1.001, 0.999, 1.002, 0.997, 1.003, 0.996)
        )),
        EntropyCase("near_05_alternating", "near-uniform", normalized(
            (1.004, 0.996, 1.003, 0.997, 1.002, 0.998, 1.001, 0.999, 1.000, 1.000)
        )),
        EntropyCase("near_06_small_wave", "near-uniform", normalized(
            (1.000, 1.004, 1.006, 1.003, 0.999, 0.996, 0.995, 0.997, 1.001, 1.005)
        )),
        EntropyCase("near_07_tiny_monotonic", "near-uniform", normalized(
            (1.0000, 1.0005, 1.0010, 1.0015, 1.0020, 1.0025, 1.0030, 1.0035, 1.0040, 1.0045)
        )),
        EntropyCase("near_08_symmetric", "near-uniform", normalized(
            (1.000, 1.002, 1.004, 1.006, 1.008, 1.008, 1.006, 1.004, 1.002, 1.000)
        )),
        EntropyCase("near_09_mask_two_unavailable", "near-uniform", normalized(
            (1.002, 1.001, 1.000, 0.999, 0.998, 1.003, 1.004, 0.997, 0.60, 0.40)
        ), (True, True, True, True, True, True, True, True, False, False)),
        EntropyCase("near_10_mask_stop_like_action", "near-uniform", normalized(
            (1.000, 1.003, 0.999, 1.002, 0.998, 1.001, 1.004, 0.997, 1.000, 0.996, 0.70)
        ), (True, True, True, True, True, True, True, True, True, True, False)),
    ]

    clear = [
        EntropyCase("clear_11_one_agent_70pct", "separated", (0.70, 0.10, 0.05, 0.04, 0.03, 0.03, 0.02, 0.015, 0.010, 0.005)),
        EntropyCase("clear_12_one_agent_90pct", "separated", (0.90, 0.03, 0.02, 0.015, 0.010, 0.008, 0.006, 0.005, 0.004, 0.002)),
        EntropyCase("clear_13_top_two", "separated", (0.48, 0.32, 0.06, 0.04, 0.03, 0.025, 0.020, 0.010, 0.009, 0.006)),
        EntropyCase("clear_14_three_specialists", "separated", (0.40, 0.28, 0.18, 0.04, 0.03, 0.025, 0.020, 0.010, 0.008, 0.007)),
        EntropyCase("clear_15_long_tail", "separated", normalized(
            (1.00, 0.55, 0.30, 0.18, 0.10, 0.06, 0.04, 0.025, 0.015, 0.01)
        )),
        EntropyCase("clear_16_two_equal_dominant", "separated", (0.44, 0.44, 0.03, 0.025, 0.020, 0.015, 0.010, 0.008, 0.005, 0.003)),
        EntropyCase("clear_17_single_expert_55pct", "separated", (0.55, 0.12, 0.09, 0.07, 0.05, 0.04, 0.03, 0.025, 0.015, 0.01)),
        EntropyCase("clear_18_four_ranked", "separated", (0.143, 0.071, 0.077, 0.1, 0.07, 0.15, 0.03, 0.09, 0.08, 0.189)),
        EntropyCase("clear_19_masked_high_raw_action", "separated", normalized(
            (0.50, 0.20, 0.10, 0.08, 0.06, 0.03, 0.02, 0.01, 0.005, 0.004, 0.80)
        ), (True, True, True, True, True, True, True, True, True, True, False)),
        EntropyCase("clear_20_near_deterministic", "separated", (0.965, 0.012, 0.007, 0.005, 0.003, 0.0025, 0.002, 0.0015, 0.001, 0.001)),
    ]
    return near + clear


def evaluate(case: EntropyCase) -> dict[str, float | int | str]:
    probs = routing_distribution(case.weights, case.available).squeeze(0)
    active = probs[probs > 0]
    entropy_from_policy = torch.distributions.Categorical(probs).entropy().item()
    entropy_manual = (-(active * torch.log(active)).sum()).item()
    max_entropy = math.log(len(active))
    normalized_entropy = entropy_from_policy / max_entropy
    ordered = torch.sort(active, descending=True).values
    top_margin = (ordered[0] - ordered[1]).item() if len(ordered) > 1 else ordered[0].item()

    assert math.isclose(entropy_from_policy, entropy_manual, abs_tol=1e-12)
    assert 0.0 <= normalized_entropy <= 1.0 + 1e-12

    return {
        "name": case.name,
        "group": case.group,
        "active": len(active),
        "entropy": entropy_from_policy,
        "normalized_entropy": normalized_entropy,
        "top1": ordered[0].item(),
        "top1_top2_margin": top_margin,
    }


def main() -> None:
    results = [evaluate(case) for case in build_cases()]
    assert len(results) == 20

    near = [row for row in results if row["group"] == "near-uniform"]
    separated = [row for row in results if row["group"] == "separated"]
    assert min(float(row["normalized_entropy"]) for row in near) > 0.999
    assert max(float(row["normalized_entropy"]) for row in separated) < 0.99

    print("case                           group          active  entropy   normalized_H  top1     top1-top2")
    print("-" * 104)
    for row in results:
        print(
            f"{row['name']:<30} {row['group']:<14} {row['active']:>6}  "
            f"{row['entropy']:>7.5f}   {row['normalized_entropy']:>10.6f}  "
            f"{row['top1']:>7.4f}  {row['top1_top2_margin']:>9.6f}"
        )

    mean_near = sum(float(row["normalized_entropy"]) for row in near) / len(near)
    mean_separated = sum(float(row["normalized_entropy"]) for row in separated) / len(separated)
    print("\nAssertions passed: manual Shannon entropy matches Categorical.entropy() in all 20 cases.")
    print(f"Mean normalized entropy — near-uniform: {mean_near:.6f}")
    print(f"Mean normalized entropy — separated:    {mean_separated:.6f}")


if __name__ == "__main__":
    main()
