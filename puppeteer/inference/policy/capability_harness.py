from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from inference.policy.stage_routing import RoutingSnapshot


_BENCHMARK_NAMES = ("musique", "gaia", "mmlu", "srdd")

_NEED_TERMS: dict[str, tuple[str, ...]] = {
    "revise_plan": ("planning", "decompos", "subquestion", "dependency"),
    "retrieve_evidence": (
        "retriev",
        "evidence",
        "research",
        "search",
        "file",
        "paragraph",
    ),
    "resolve_bridge": ("bridge", "intermediate", "domain_reasoning", "relation"),
    "compose_evidence": ("integration", "compos", "general_reasoning", "combine"),
    "verify_candidate": ("verification", "verif", "critique", "repair", "check"),
    "integrate_answer": ("integration", "final", "conclud", "format"),
    "recover_failure": ("recovery", "repair", "modifier", "reflect", "fallback"),
    "ready_to_stop": (),
}


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("_", " ").split())


def _flatten_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(
            f"{_flatten_text(key)} {_flatten_text(item)}"
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_text(item) for item in value)
    return _normalized(value)


def _answer_value(answer: Any, key: str) -> Any:
    if not isinstance(answer, Mapping):
        return None
    return answer.get(key)


def _is_yes(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) >= 0.5
    return _normalized(value) in {"yes", "true", "1", "y"}


def _is_no(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) < 0.5
    return _normalized(value) in {"no", "false", "0", "n"}


@dataclass(frozen=True)
class HarnessResult:
    snapshot: RoutingSnapshot
    dominant_need: str
    branch_allowed: bool
    route_capacity: int


class CapabilityRoutingHarness:
    """Identity-safe, opt-in routing constraints for frozen planners.

    The harness never chooses an agent. It removes candidates that are explicitly
    out of scope and can conservatively shorten the list according to a diagnosed
    capability need. Generic/unscoped roles remain portable across benchmarks.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.typed_blackboard = bool(self.config.get("typed_blackboard", True))
        self.dataset_scope_gate = bool(
            self.config.get("dataset_scope_gate", True)
        )
        shortlist = dict(self.config.get("capability_shortlist") or {})
        self.shortlist_enabled = bool(shortlist.get("enabled", True))
        self.max_candidates = int(shortlist.get("max_candidates", 6))
        self.min_candidates = int(shortlist.get("min_candidates", 2))
        branching = dict(self.config.get("adaptive_branching") or {})
        self.branching_enabled = bool(branching.get("enabled", False))
        self.initial_paths = int(branching.get("initial_paths", 1))
        self.max_branch_options = int(branching.get("max_branch_options", 6))
        terminal = dict(self.config.get("terminal_gate") or {})
        self.terminal_gate_enabled = bool(terminal.get("enabled", True))
        self.require_musique_evidence = bool(
            terminal.get("require_evidence_for_musique", True)
        )
        self.max_previous_outputs = int(
            self.config.get("max_previous_outputs", 6)
        )
        if self.max_candidates < 1 or self.min_candidates < 1:
            raise ValueError("routing_harness candidate limits must be positive")
        if self.min_candidates > self.max_candidates:
            raise ValueError(
                "routing_harness min_candidates cannot exceed max_candidates"
            )
        if self.initial_paths < 1 or self.max_branch_options < 0:
            raise ValueError("routing_harness branching limits are invalid")
        if self.max_previous_outputs < 1:
            raise ValueError("routing_harness.max_previous_outputs must be positive")

    @staticmethod
    def dominant_need(
        routing_state: Mapping[str, Any],
        semantic_diagnostics: Mapping[str, Any] | None = None,
    ) -> str:
        diagnostics = semantic_diagnostics or {}
        diagnosed = _answer_value(diagnostics.get("dominant_next_need"), "choice")
        if diagnosed in _NEED_TERMS:
            return str(diagnosed)
        if routing_state.get("last_action_status") == "failure" or routing_state.get(
            "path_stalled"
        ):
            return "recover_failure"
        if routing_state.get("final_answer_exists"):
            return "ready_to_stop"
        if routing_state.get("integrator_required"):
            return "integrate_answer"
        if routing_state.get("candidate_answer_exists"):
            return "verify_candidate"
        if routing_state.get("intermediate_answer_count", 0):
            return "compose_evidence"
        if routing_state.get("paragraph_reference_count", 0) or routing_state.get(
            "evidence_count", 0
        ):
            return "resolve_bridge"
        if routing_state.get("stage") == "root":
            return "revise_plan"
        return "retrieve_evidence"

    @staticmethod
    def _candidate_scopes(view: Mapping[str, Any]) -> set[str]:
        text = _flatten_text(
            {
                "expected_input": view.get("expected_input"),
                "role_goal": view.get("role_goal"),
                "routing_profile": view.get("routing_profile"),
                "core_functions": view.get("core_functions"),
            }
        )
        return {name for name in _BENCHMARK_NAMES if name in text}

    @staticmethod
    def _task_scope(task_type: Any) -> str:
        text = _normalized(task_type)
        if "mmlu" in text:
            return "mmlu"
        return next((name for name in _BENCHMARK_NAMES if name in text), "")

    @staticmethod
    def _capability_score(view: Mapping[str, Any], need: str) -> int:
        terms = _NEED_TERMS.get(need, ())
        if not terms:
            return 0
        profile = view.get("routing_profile") or {}
        affinity = profile.get("capability_affinity") or {}
        text = _flatten_text(
            {
                "role": view.get("role_name"),
                "goal": view.get("role_goal"),
                "functions": view.get("core_functions"),
                "profile": profile,
                "tools": view.get("tools"),
            }
        )
        score = sum(1 for term in terms if _normalized(term) in text)
        for name, level in affinity.items():
            normalized_name = _normalized(name)
            if any(_normalized(term) in normalized_name for term in terms):
                score += {"high": 4, "medium": 2, "low": 1}.get(
                    _normalized(level), 1
                )
        return score

    def terminal_ready(self, routing_state: Mapping[str, Any]) -> bool:
        if not routing_state.get("enabled", False):
            return True
        profile = _normalized(routing_state.get("profile"))
        if not self.terminal_gate_enabled:
            return bool(
                routing_state.get(
                    "final_answer_exists"
                    if profile.startswith("musique")
                    else "candidate_answer_exists",
                    False,
                )
            )
        healthy = routing_state.get("last_action_status") != "failure"
        if profile.startswith("musique"):
            evidence_ready = (
                int(routing_state.get("paragraph_reference_count", 0)) > 0
                if self.require_musique_evidence
                else True
            )
            return bool(
                routing_state.get("final_answer_exists")
                and evidence_ready
                and healthy
            )
        if profile.startswith("gaia"):
            return bool(routing_state.get("candidate_answer_exists") and healthy)
        return healthy

    def should_branch(
        self,
        routing_state: Mapping[str, Any],
        semantic_diagnostics: Mapping[str, Any] | None,
        capacity: int,
    ) -> bool:
        if not self.enabled or not self.branching_enabled or capacity < 2:
            return False
        diagnostics = semantic_diagnostics or {}
        conflict = _is_yes(
            _answer_value(diagnostics.get("candidate_conflict_present"), "noul")
        )
        progress = _answer_value(
            diagnostics.get("progress_since_previous_step"), "score"
        )
        low_progress = isinstance(progress, (int, float)) and float(progress) <= 1.0
        return bool(
            conflict
            or routing_state.get("path_stalled")
            or routing_state.get("repeat_without_progress")
            or low_progress
        )

    def route_capacity(
        self,
        *,
        root: bool,
        capacity: int,
        routing_state: Mapping[str, Any],
        semantic_diagnostics: Mapping[str, Any] | None = None,
    ) -> int:
        if not self.enabled or not self.branching_enabled:
            return capacity
        if root:
            return min(capacity, self.initial_paths)
        return min(
            capacity,
            2 if self.should_branch(routing_state, semantic_diagnostics, capacity) else 1,
        )

    def apply(
        self,
        *,
        task_type: Any,
        root: bool,
        capacity: int,
        views: Iterable[Mapping[str, Any]],
        snapshot: RoutingSnapshot,
        semantic_diagnostics: Mapping[str, Any] | None = None,
    ) -> HarnessResult:
        if not self.enabled:
            return HarnessResult(snapshot, "", False, capacity)
        eligible_ids = set(snapshot.eligible_candidate_ids)
        candidates = [
            dict(view)
            for view in views
            if int(view["candidate_id"]) in eligible_ids
            and bool(view.get("available", False))
        ]
        masks = list(snapshot.masked_candidates)
        task_scope = self._task_scope(task_type)
        kept: list[dict[str, Any]] = []
        for view in candidates:
            scopes = self._candidate_scopes(view)
            if (
                self.dataset_scope_gate
                and task_scope
                and scopes
                and task_scope not in scopes
            ):
                masks.append(
                    {
                        "candidate_id": int(view["candidate_id"]),
                        "role_name": str(view.get("role_name") or ""),
                        "reason": "explicit_benchmark_scope_mismatch",
                    }
                )
            else:
                kept.append(view)

        need = self.dominant_need(snapshot.state, semantic_diagnostics)
        if need == "ready_to_stop" and not self.terminal_ready(snapshot.state):
            profile = _normalized(snapshot.state.get("profile"))
            if profile.startswith("musique") and not snapshot.state.get(
                "paragraph_reference_count", 0
            ):
                need = "retrieve_evidence"
            elif snapshot.state.get("candidate_answer_exists"):
                need = "verify_candidate"
            else:
                need = "retrieve_evidence"
        scores = {
            int(view["candidate_id"]): self._capability_score(view, need)
            for view in kept
        }
        if self.shortlist_enabled and len(kept) > self.max_candidates:
            ranked = sorted(
                kept,
                key=lambda view: (
                    -scores[int(view["candidate_id"])],
                    int(view["candidate_id"]),
                ),
            )
            positive = [
                view for view in ranked if scores[int(view["candidate_id"])] > 0
            ]
            selected = positive[: self.max_candidates]
            if len(selected) < self.min_candidates:
                selected_ids = {int(view["candidate_id"]) for view in selected}
                selected.extend(
                    view
                    for view in ranked
                    if int(view["candidate_id"]) not in selected_ids
                )
            selected = selected[: self.max_candidates]
            selected_ids = {int(view["candidate_id"]) for view in selected}
            for view in kept:
                if int(view["candidate_id"]) not in selected_ids:
                    masks.append(
                        {
                            "candidate_id": int(view["candidate_id"]),
                            "role_name": str(view.get("role_name") or ""),
                            "reason": f"lower_affinity_for:{need}",
                        }
                    )
            kept = selected

        if not kept and candidates:
            # Never make the route space empty because of an incomplete role card.
            kept = [candidates[0]]
            masks = [
                item
                for item in masks
                if int(item.get("candidate_id", -1))
                != int(candidates[0]["candidate_id"])
            ]

        state = dict(snapshot.state)
        terminal_ready = self.terminal_ready(state)
        branch_allowed = self.should_branch(
            state, semantic_diagnostics, capacity
        )
        route_capacity = self.route_capacity(
            root=root,
            capacity=capacity,
            routing_state=state,
            semantic_diagnostics=semantic_diagnostics,
        )
        state["terminal_ready"] = terminal_ready
        state["routing_harness"] = {
            "enabled": True,
            "dominant_need": need,
            "capability_scores": scores,
            "branch_allowed": branch_allowed,
            "route_capacity": route_capacity,
            "task_scope": task_scope,
        }
        kept_ids = tuple(int(view["candidate_id"]) for view in kept)
        state["eligible_candidate_ids"] = list(kept_ids)
        state["masked_candidates"] = masks
        resolved = RoutingSnapshot(state, kept_ids, tuple(masks))
        return HarnessResult(resolved, need, branch_allowed, route_capacity)

    def blackboard(
        self,
        *,
        task: Mapping[str, Any],
        routing_state: Mapping[str, Any],
        previous_outputs: Iterable[Mapping[str, Any]],
        remaining_depth: Any,
        remaining_width: int,
        dominant_need: str = "",
    ) -> dict[str, Any]:
        outputs = list(previous_outputs)[-self.max_previous_outputs :]
        question = next(
            (
                task.get(key)
                for key in ("Question", "question", "prompt", "task")
                if task.get(key) is not None
            ),
            "",
        )
        return {
            "objective": question,
            "task_type": task.get("type"),
            "evidence": {
                "count": int(
                    routing_state.get(
                        "paragraph_reference_count",
                        routing_state.get("evidence_count", 0),
                    )
                    or 0
                ),
                "resource_ids": list(
                    routing_state.get("referenced_paragraph_ids") or ()
                ),
            },
            "answers": {
                "intermediate_count": int(
                    routing_state.get("intermediate_answer_count", 0) or 0
                ),
                "candidates": list(routing_state.get("candidate_answers") or ()),
                "finals": list(routing_state.get("final_answers") or ()),
                "candidate_exists": bool(
                    routing_state.get("candidate_answer_exists", False)
                ),
                "final_exists": bool(
                    routing_state.get("final_answer_exists", False)
                ),
            },
            "execution": {
                "last_role": routing_state.get("last_role"),
                "last_status": routing_state.get("last_action_status"),
                "last_error": routing_state.get("last_error_type"),
                "path_stalled": bool(routing_state.get("path_stalled", False)),
                "role_history": list(routing_state.get("role_history") or ()),
                "recent_outputs": outputs,
            },
            "budget": {
                "remaining_depth": remaining_depth,
                "remaining_width": remaining_width,
            },
            "diagnosis": {
                "dominant_next_need": dominant_need
                or self.dominant_need(routing_state),
                "terminal_ready": self.terminal_ready(routing_state),
            },
        }
