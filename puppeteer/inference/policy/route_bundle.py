from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class RouteOption:
    option_id: str
    candidate_ids: tuple[int, ...]
    description: str
    stop: bool = False


def _candidate_summary(view: Mapping[str, Any]) -> str:
    profile = view.get("routing_profile") or {}
    use_when = "; ".join(map(str, profile.get("use_when") or ()))
    avoid_when = "; ".join(map(str, profile.get("avoid_when") or ()))
    contribution = "; ".join(
        map(str, profile.get("expected_contribution") or ())
    )
    handoff = "; ".join(map(str, profile.get("handoff_requirements") or ()))
    parts = [
        f"candidate {view['candidate_id']}: {view.get('role_name', 'unnamed role')}",
        str(view.get("role_goal", "")).strip(),
    ]
    if use_when:
        parts.append(f"use when: {use_when}")
    if avoid_when:
        parts.append(f"avoid when: {avoid_when}")
    if contribution:
        parts.append(f"expected contribution: {contribution}")
    if handoff:
        parts.append(f"handoff requirements: {handoff}")
    return ". ".join(part for part in parts if part)


class RouteBundleBuilder:
    """Build a finite action space without exposing teammate identities."""

    def __init__(self, max_options: int = 255) -> None:
        self.max_options = int(max_options)
        if self.max_options < 1:
            raise ValueError("max_options must be positive")

    @staticmethod
    def _available(views: Iterable[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
        return tuple(view for view in views if bool(view.get("available", False)))

    def root_options(
        self, views: Iterable[Mapping[str, Any]], capacity: int
    ) -> tuple[RouteOption, ...]:
        available = self._available(views)
        by_id = {int(view["candidate_id"]): view for view in available}
        identifiers = tuple(sorted(by_id))
        max_size = min(max(0, int(capacity)), len(identifiers))
        options = []
        for size in range(1, max_size + 1):
            for candidate_ids in combinations(identifiers, size):
                summaries = [_candidate_summary(by_id[index]) for index in candidate_ids]
                options.append(
                    RouteOption(
                        option_id="route__" + "_".join(map(str, candidate_ids)),
                        candidate_ids=tuple(candidate_ids),
                        description=(
                            f"Open {size} independent reasoning path(s) using: "
                            + " | ".join(summaries)
                            + ". Select this bundle only when every path adds distinct value."
                        ),
                    )
                )
        if not options:
            raise ValueError("No available route bundle can be constructed")
        if len(options) > self.max_options:
            raise ValueError(
                f"Route action space has {len(options)} options, above limit "
                f"{self.max_options}"
            )
        return tuple(options)

    def sequential_root_options(
        self,
        views: Iterable[Mapping[str, Any]],
        selected_ids: Iterable[int],
        allow_finish: bool,
    ) -> tuple[RouteOption, ...]:
        """Build one bounded step of greedy root-path selection.

        Unlike ``root_options``, this never enumerates combinations. Every
        unselected available teammate is represented once, plus an optional
        finish action after at least one root path has been selected.
        """
        selected = {int(candidate_id) for candidate_id in selected_ids}
        options = []
        if allow_finish:
            options.append(
                RouteOption(
                    option_id="finish_bundle",
                    candidate_ids=(),
                    description=(
                        "Finish root routing with the roles already selected. "
                        "Choose this when another independent path would be redundant "
                        "or would not add enough value."
                    ),
                    stop=True,
                )
            )
        for view in self._available(views):
            candidate_id = int(view["candidate_id"])
            if candidate_id in selected:
                continue
            options.append(
                RouteOption(
                    option_id=f"candidate__{candidate_id}",
                    candidate_ids=(candidate_id,),
                    description=_candidate_summary(view),
                )
            )
        if not options:
            raise ValueError("No sequential root action can be constructed")
        if len(options) > self.max_options:
            raise ValueError(
                f"Sequential root action space has {len(options)} options, above limit "
                f"{self.max_options}"
            )
        return tuple(options)

    def next_options(
        self, views: Iterable[Mapping[str, Any]], allow_stop: bool = True
    ) -> tuple[RouteOption, ...]:
        options = []
        if allow_stop:
            options.append(
                RouteOption(
                    option_id="stop",
                    candidate_ids=(),
                    description=(
                        "Stop this path because the current output already contains a "
                        "usable final answer or complete artifact."
                    ),
                    stop=True,
                )
            )
        for view in self._available(views):
            candidate_id = int(view["candidate_id"])
            options.append(
                RouteOption(
                    option_id=f"candidate__{candidate_id}",
                    candidate_ids=(candidate_id,),
                    description=_candidate_summary(view),
                )
            )
        if not options:
            raise ValueError("No next action can be constructed")
        if len(options) > self.max_options:
            raise ValueError(
                f"Next-action space has {len(options)} options, above limit "
                f"{self.max_options}"
            )
        return tuple(options)
