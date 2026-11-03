from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from routing_benchmark.v2.schemas import STOP_ACTION


class EmpiricalReplayBank:
    """Frozen one-step empirical transitions keyed by observable state and action."""

    def __init__(self, edges: Iterable[Mapping[str, Any]]) -> None:
        self._edges: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            key = (str(edge["pre_state_fingerprint"]), str(edge["selected_action"]))
            self._edges[key].append(dict(edge))
        for values in self._edges.values():
            values.sort(key=lambda item: (int(item.get("replicate_id", 0)), str(item.get("output_id"))))

    def actions(self, pre_state_fingerprint: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                action
                for state, action in self._edges
                if state == pre_state_fingerprint
            )
        )

    def resolve(
        self,
        pre_state_fingerprint: str,
        action: str,
        replicate_id: int = 0,
    ) -> dict[str, Any] | None:
        values = self._edges.get((str(pre_state_fingerprint), str(action)), [])
        for value in values:
            if int(value.get("replicate_id", 0)) == int(replicate_id):
                return dict(value)
        return dict(values[0]) if values else None


def evaluate_empirical_track_a(
    decisions: Iterable[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    router,
    *,
    memory_view: str = "structured",
    seed: int = 42,
) -> dict[str, Any]:
    from .pipeline import public_observation_for

    records: list[dict[str, Any]] = []
    router.reset(seed)
    for decision in decisions:
        decision_id = str(decision["decision_id"])
        label = labels[decision_id]
        # Human-final labels take precedence. Pending releases use proposed labels
        # only for pilot analysis and expose that status in every output record.
        valid = tuple(
            label.get("human_valid_actions")
            or label.get("proposed_valid_actions")
            or ()
        )
        candidate_actions = tuple(map(str, decision.get("candidate_actions") or ()))
        selected = router.select(
            public_observation_for(decision, memory_view),
            candidate_actions,
            valid,
        )
        terminal_valid = STOP_ACTION in valid
        records.append(
            {
                "decision_id": decision_id,
                "selected_action": selected.action,
                "valid_actions": list(valid),
                "valid": selected.action in valid,
                "terminal_valid": terminal_valid,
                "premature_stop": selected.action == STOP_ACTION and not terminal_valid,
                "label_status": label.get("review_status", "pending"),
                "router_metadata": dict(selected.metadata),
            }
        )
    count = len(records)
    nonterminal = [item for item in records if not item["terminal_valid"]]
    return {
        "metrics": {
            "track": "A-empirical",
            "decision_count": count,
            "valid_action_accuracy": (
                sum(bool(item["valid"]) for item in records) / count if count else None
            ),
            "premature_stop_rate": (
                sum(bool(item["premature_stop"]) for item in nonterminal) / len(nonterminal)
                if nonterminal
                else None
            ),
            "human_review_complete": all(
                item["label_status"] in {"accepted", "adjudicated"} for item in records
            ),
        },
        "records": records,
    }
