from __future__ import annotations

from typing import Iterable

from role_aware.schemas import PlannerCandidateView, TeammateSpec


class StaticRoutingProfileStore:
    """Immutable planner-facing role/profile catalogue."""

    def __init__(self) -> None:
        self._specs: tuple[TeammateSpec, ...] = ()

    def initialize(self, specs: Iterable[TeammateSpec]) -> None:
        ordered = tuple(specs)
        if len({spec.teammate_id for spec in ordered}) != len(ordered):
            raise ValueError("Static routing profiles require unique teammate IDs")
        self._specs = ordered

    def public_views(
        self, specs: Iterable[TeammateSpec] | None = None
    ) -> tuple[PlannerCandidateView, ...]:
        ordered = tuple(specs) if specs is not None else self._specs
        return tuple(
            PlannerCandidateView(
                candidate_id=index,
                role_name=spec.role_card.role_name,
                role_goal=spec.role_card.role_goal,
                core_functions=spec.role_card.core_functions,
                allowed_actions=spec.role_card.allowed_actions,
                expected_input=spec.role_card.expected_input,
                routing_profile=spec.routing_profile,
                tools=spec.role_card.tools,
                available=spec.available,
            )
            for index, spec in enumerate(ordered)
        )

    def state_dict(self) -> dict:
        return {
            spec.teammate_id: {
                "role_card": spec.role_card.to_dict(),
                "routing_profile": spec.routing_profile.to_dict(),
            }
            for spec in self._specs
        }

    def reset(self, teammate_ids=None) -> None:
        return None

