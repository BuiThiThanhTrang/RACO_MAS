"""Check the actual RoleAwareREINFORCE update direction on synthetic logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.policy.role_aware_reinforce import RoleAwareREINFORCE


class LogitPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.0]))


def run_update(reward: float, learning_rate: float, entropy_coef: float) -> dict:
    network = LogitPolicy()
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
    before_distribution = torch.distributions.Categorical(logits=network.logits)
    before_probability = float(before_distribution.probs[0].detach())
    before_entropy = float(before_distribution.entropy().detach())

    policy = object.__new__(RoleAwareREINFORCE)
    policy.policy_network = network
    policy.optimizer = optimizer
    policy.device = torch.device("cpu")
    policy.optimizer_updates_enabled = True
    policy.routing_mode = "legacy_threshold"
    policy.estimator_version = "legacy_surrogate_v1"
    policy.gamma = 1.0
    policy.entropy_coef = entropy_coef
    policy.audit = None
    policy._path_lookup = {"path-0001": 0}
    policy.rewards_history = []
    policy.reward_breakdowns = []
    policy.global_step = 0
    policy.decisions = []
    policy.trajectories = [
        [
            {
                "log_prob": before_distribution.log_prob(torch.tensor(0)).reshape(1),
                "reward": reward,
                "finalized": True,
            }
        ]
    ]
    policy.entropies = [before_distribution.entropy()]
    metrics = RoleAwareREINFORCE.update(policy)

    after_distribution = torch.distributions.Categorical(logits=network.logits)
    after_probability = float(after_distribution.probs[0].detach())
    after_entropy = float(after_distribution.entropy().detach())
    return {
        "reward": reward,
        "entropy_coef": entropy_coef,
        "selected_probability_before": before_probability,
        "selected_probability_after": after_probability,
        "selected_probability_delta": after_probability - before_probability,
        "entropy_before": before_entropy,
        "entropy_after": after_entropy,
        "entropy_delta": after_entropy - before_entropy,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    positive = run_update(1.0, args.learning_rate, args.entropy_coef)
    negative = run_update(-1.0, args.learning_rate, args.entropy_coef)
    entropy_only = run_update(0.0, args.learning_rate, args.entropy_coef)
    checks = {
        "positive_reward_increases_selected_probability": positive[
            "selected_probability_delta"
        ]
        > 0,
        "negative_reward_decreases_selected_probability": negative[
            "selected_probability_delta"
        ]
        < 0,
        "entropy_bonus_does_not_reduce_entropy_when_reward_is_zero": entropy_only[
            "entropy_delta"
        ]
        >= 0,
        "all_gradients_finite_and_nonzero": all(
            result["metrics"]["gradient_norm"] is not None
            and result["metrics"]["gradient_norm"] > 0
            for result in (positive, negative, entropy_only)
            if result["reward"] != 0 or args.entropy_coef != 0
        ),
    }
    result = {
        "learning_rate": args.learning_rate,
        "positive_reward": positive,
        "negative_reward": negative,
        "entropy_only": entropy_only,
        "checks": checks,
        "passed": all(checks.values()),
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
