from __future__ import annotations

import random
from typing import Any, Mapping, Sequence


def select_counterfactual_steps(
    episodes: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    random_audit_rate: float = 0.1,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for episode in episodes:
        for step in episode.get("steps") or ():
            valid = list(step.get("valid_actions") or ())
            selected_action = str(step.get("selected_action", ""))
            reasons: list[str] = []
            if len(valid) > 1:
                reasons.append("multi_valid_action_audit")
            if selected_action == "STOP" or "STOP" in valid:
                reasons.append("terminal_decision_audit")
            stage = str(((step.get("pre_action_observation") or {}).get("environment") or {}).get("stage", ""))
            if stage in {"candidate_invalid", "diagnosed"}:
                reasons.append("recovery_state_audit")
            if rng.random() < random_audit_rate:
                reasons.append("seeded_random_audit")
            alternates = [action for action in valid if action != selected_action and action != "STOP"]
            if reasons and alternates:
                selected.append(
                    {
                        "scenario_id": episode.get("scenario_id"),
                        "base_workflow_id": episode.get("base_workflow_id"),
                        "domain": episode.get("domain"),
                        "manifest_path": episode.get("manifest_path"),
                        "snapshot_id": step.get("snapshot_id"),
                        "snapshot_dir": step.get("snapshot_dir"),
                        "pre_state_fingerprint": step.get("pre_state_fingerprint"),
                        "factual_action": selected_action,
                        "branch_actions": alternates,
                        "candidate_actions": step.get("candidate_actions", []),
                        "selection_reasons": sorted(set(reasons)),
                        "pre_action_observation": step.get("pre_action_observation", {}),
                    }
                )
    return selected
